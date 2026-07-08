"""Eval GPU worker - duel API, vLLM generation, model resolution, scoring, artifacts."""

from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import parse_qsl, urlencode, urlparse
from uuid import UUID, uuid4

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, WebSocket
from loguru import logger
from starlette.websockets import WebSocketDisconnect

from albedo.evaluation import EvalRequest
from albedo.judges import (
    CHALLENGER_WIN_MARGIN,
    JUDGE_MODELS,
    aggregate_scores,
    challenger_beats_king,
)
from albedo.remote.common import (
    Run,
    RunStore,
    WorkerBusy,
    apply_canonical_model_config,
    bearer_auth,
    canonical_generation_config,
    canonical_max_model_len,
    download_heartbeat,
    ready_payload,
    require_startup_token,
)
from albedo.sampling import (
    EvalSample,
    load_manifest_file,
    load_swe_zero_samples,
    multi_source_manifest_sample_ids,
)
from albedo.settings import RemoteEvalSettings, get_settings

# ── vLLM duel generation (merged from generation.py) ─────────────────────────

QWEN3_IM_END_TOKEN_ID = 248046  # <|im_end|> for Qwen3.6-35B-A3B (was 151645 for Qwen3-4B genesis)


@dataclass(frozen=True)
class GenerationResult:
    # One model's output (or error) for a single sample.
    sample_id: str
    text: str
    error: str | None = None


class VllmProcessGenerator:
    # Runs vLLM in a spawned subprocess pinned to a GPU group; the parent stays CUDA-free.
    def __init__(
        self,
        *,
        model: str,
        gpu_ids: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None = None,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        gpu_memory_utilization: float = 0.95,
        kv_cache_dtype: str = "auto",
    ):
        # Captures the full vLLM launch configuration for one duel side.
        self.model = model
        self.gpu_ids = gpu_ids
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_dtype = kv_cache_dtype

    def generate(self, samples: list[EvalSample]) -> list[GenerationResult]:
        # Spawns the vLLM worker process and collects one result per sample (errors fan out to all).
        if not samples:
            return []

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_vllm_worker,
            kwargs={
                "model": self.model,
                "gpu_ids": self.gpu_ids,
                "prompts": [sample.prompt for sample in samples],
                "sample_ids": [sample.sample_id for sample in samples],
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_model_len": self.max_model_len,
                "enforce_eager": self.enforce_eager,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "kv_cache_dtype": self.kv_cache_dtype,
                "queue": result_queue,
            },
        )
        process.start()
        payload = None
        while process.is_alive():
            try:
                payload = result_queue.get(timeout=1)
                break
            except queue_module.Empty:
                continue
        if payload is None:
            try:
                payload = result_queue.get_nowait()
            except queue_module.Empty:
                payload = {
                    "error": f"vLLM process exited {process.exitcode} without result payload"
                }
        process.join()

        if process.exitcode != 0:
            error = payload.get("error") or f"vLLM process exited {process.exitcode}"
            return [
                GenerationResult(sample_id=sample.sample_id, text="", error=error)
                for sample in samples
            ]
        if payload.get("error"):
            return [
                GenerationResult(sample_id=sample.sample_id, text="", error=payload["error"])
                for sample in samples
            ]
        return [GenerationResult(**item) for item in payload["results"]]


def _vllm_worker(
    *,
    model: str,
    gpu_ids: list[str],
    prompts: list[str],
    sample_ids: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_model_len: int | None,
    enforce_eager: bool,
    gpu_memory_utilization: float = 0.95,
    kv_cache_dtype: str = "auto",
    queue=None,
) -> None:
    # Subprocess body: pin GPUs, boot vLLM, generate every prompt, ship results back over the queue.
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

        from vllm import LLM, SamplingParams

        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": len(gpu_ids),
            "trust_remote_code": True,
            # Match the benchmark eval server's `--generation-config vllm`:
            # do not let vLLM auto-import Hugging Face generation_config.json.
            "generation_config": "vllm",
            "reasoning_parser": "qwen3",
            "gpu_memory_utilization": gpu_memory_utilization,
            "kv_cache_dtype": kv_cache_dtype,
            # Text-only eval: cap multimodal inputs to 0 so vLLM skips vision-encoder
            # profiling, which hangs for the multimodal Qwen3.6 genesis architecture.
            "limit_mm_per_prompt": {"image": 0, "video": 0},
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
        llm = LLM(**llm_kwargs)
        params_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": [QWEN3_IM_END_TOKEN_ID],
        }
        if top_k is not None:
            params_kwargs["top_k"] = top_k
        params = SamplingParams(**params_kwargs)
        outputs = llm.generate(prompts, params)
        results = []
        for sample_id, output in zip(sample_ids, outputs, strict=True):
            text = output.outputs[0].text if output.outputs else ""
            results.append({"sample_id": sample_id, "text": text, "error": None})
        queue.put({"results": results})
    except Exception as exc:
        logger.exception(f"[remote-gen] vLLM worker failed model={model} gpu_ids={gpu_ids}: {exc}")
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def cleanup_stale_vllm_resources() -> None:
    # Kills orphaned vLLM children from a prior crash and clears stale /dev/shm IPC files (EAGAIN).
    subprocess.run(["pkill", "-9", "-f", "vllm.v1.engine.core"], check=False)
    subprocess.run(["pkill", "-9", "-f", "vllm.v1.executor.multiproc"], check=False)
    # Also kill any spawn_main processes left hanging from a prior NCCL-crash stuck subprocess.
    subprocess.run(["pkill", "-9", "-f", "multiprocessing.spawn.spawn_main"], check=False)
    for path in glob.glob("/dev/shm/psm_*") + glob.glob("/dev/shm/sem.mp-*"):
        try:
            os.unlink(path)
        except OSError as exc:
            logger.debug(f"[remote-gen] best-effort /dev/shm cleanup failed path={path}: {exc}")


# ── Model artifact resolution (merged from models_resolve.py) ────────────────

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


_DEFAULT_OCI_REGISTRY = "registry.hippius.com"


@dataclass(frozen=True)
class ResolvedModel:
    # A model ref materialized to a local directory, with provenance for the event stream.
    original_ref: str
    local_path: str
    source: str
    cache_hit: bool
    file_count: int
    total_size_bytes: int

    def as_event(self, *, side: str) -> dict[str, object]:
        # Event payload fragment describing this resolution.
        return {
            "side": side,
            "source": self.source,
            "original_ref": self.original_ref,
            "local_path": self.local_path,
            "cache_hit": self.cache_hit,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
        }


class ModelArtifactResolver:
    # Resolves local paths, file://, s3:// and OCI refs into verified local model directories.
    def __init__(self, settings: RemoteEvalSettings):
        # Keeps the settings and the cache root for downloaded artifacts.
        self.settings = settings
        self.cache_root = Path(settings.model_cache_dir)

    def resolve(self, model_ref: str) -> ResolvedModel:
        # Dispatches on the ref scheme; passthrough when resolution is disabled or unrecognized.
        if not self.settings.resolve_model_artifacts:
            return ResolvedModel(
                original_ref=model_ref,
                local_path=model_ref,
                source="disabled",
                cache_hit=True,
                file_count=0,
                total_size_bytes=0,
            )

        local_path = Path(model_ref)
        if local_path.exists():
            return self._resolved_local(model_ref, local_path, source="local")
        if model_ref.startswith("file://"):
            path = Path(urlparse(model_ref).path)
            if not path.exists():
                raise FileNotFoundError(f"model file URI does not exist: {model_ref}")
            return self._resolved_local(model_ref, path, source="local")
        if model_ref.startswith("s3://"):
            return self._resolve_s3(model_ref)
        parsed_oci = parse_oci_ref(model_ref)
        if parsed_oci:
            registry, repository, digest = parsed_oci
            return self._resolve_oci(
                registry=registry, repository=repository, digest=digest, original_ref=model_ref
            )
        return ResolvedModel(
            original_ref=model_ref,
            local_path=model_ref,
            source="passthrough",
            cache_hit=True,
            file_count=0,
            total_size_bytes=0,
        )

    def _resolved_local(self, original_ref: str, path: Path, *, source: str) -> ResolvedModel:
        # Applies the canonical config to an already-local model and reports its stats.
        self._apply_canonical_config(path)
        file_count, total_size_bytes = _tree_stats(path)
        return ResolvedModel(
            original_ref=original_ref,
            local_path=str(path),
            source=source,
            cache_hit=True,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
        )

    def _resolve_s3(self, model_ref: str) -> ResolvedModel:
        # Mirrors an s3:// prefix into the cache dir, guarded by a done-marker.
        bucket, prefix = split_s3_uri(model_ref)
        cache_dir = self.cache_root / "s3" / bucket / prefix.strip("/")
        done_marker = cache_dir / ".albedo-model-cache.json"
        if done_marker.exists():
            if _has_loadable_model_files(cache_dir):
                self._apply_canonical_config(cache_dir)
                file_count, total_size_bytes = _tree_stats(cache_dir)
                return ResolvedModel(
                    model_ref, str(cache_dir), "s3", True, file_count, total_size_bytes
                )
            logger.warning(
                f"model_cache_invalid source=s3 path={cache_dir} reason=missing_model_files"
            )
            shutil.rmtree(cache_dir, ignore_errors=True)

        import boto3

        cache_dir.mkdir(parents=True, exist_ok=True)
        session_kwargs: dict[str, str] = {}
        if self.settings.s3_access_key_id:
            session_kwargs["aws_access_key_id"] = self.settings.s3_access_key_id
        if self.settings.s3_secret_access_key:
            session_kwargs["aws_secret_access_key"] = self.settings.s3_secret_access_key
        if self.settings.s3_session_token:
            session_kwargs["aws_session_token"] = self.settings.s3_session_token
        if self.settings.s3_region:
            session_kwargs["region_name"] = self.settings.s3_region
        client_kwargs: dict[str, str] = {}
        if self.settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = self.settings.s3_endpoint_url
        client = boto3.session.Session(**session_kwargs).client("s3", **client_kwargs)

        paginator = client.get_paginator("list_objects_v2")
        found = False
        with download_heartbeat(model_ref):
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if key.endswith("/"):
                        continue
                    found = True
                    rel = key[len(prefix) :].lstrip("/") if prefix else key
                    destination = cache_dir / rel
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, key, str(destination))
        if not found:
            raise FileNotFoundError(f"no model objects found under {model_ref}")
        _require_loadable_model_files(cache_dir, source="s3")
        done_marker.write_text(
            json.dumps({"source": model_ref}, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._apply_canonical_config(cache_dir)
        file_count, total_size_bytes = _tree_stats(cache_dir)
        return ResolvedModel(model_ref, str(cache_dir), "s3", False, file_count, total_size_bytes)

    def _resolve_oci(
        self, *, registry: str, repository: str, digest: str, original_ref: str
    ) -> ResolvedModel:
        # Pulls an OCI artifact layer-by-layer with sha256 verification and resumable staging.
        cache_dir = (
            self.cache_root
            / "oci"
            / registry
            / repository.replace("/", "__")
            / digest.removeprefix("sha256:")
        )
        done_marker = cache_dir / ".albedo-model-cache.json"
        if done_marker.exists():
            if _has_loadable_model_files(cache_dir):
                self._apply_canonical_config(cache_dir)
                file_count, total_size_bytes = _tree_stats(cache_dir)
                return ResolvedModel(
                    original_ref, str(cache_dir), "oci", True, file_count, total_size_bytes
                )
            logger.warning(
                f"model_cache_invalid source=oci path={cache_dir} reason=missing_model_files"
            )
            shutil.rmtree(cache_dir, ignore_errors=True)

        temp_dir = cache_dir.with_suffix(".partial")
        # Resume an interrupted download: keep shards already fetched into .partial instead of
        # wiping and re-downloading everything on each retry. A shard's final filename only
        # appears after it is fully streamed and digest-verified, so anything present is complete.
        temp_dir.mkdir(parents=True, exist_ok=True)

        with httpx.Client(timeout=None, follow_redirects=True) as client:
            manifest_url = f"https://{registry}/v2/{repository}/manifests/{digest}"
            headers = {
                "Accept": (
                    "application/vnd.oci.image.manifest.v1+json, "
                    "application/vnd.docker.distribution.manifest.v2+json"
                )
            }
            response = client.get(manifest_url, headers=headers)
            if response.status_code == 401:
                token = _bearer_token(client, response, repository)
                response = client.get(
                    manifest_url, headers={**headers, "Authorization": f"Bearer {token}"}
                )
            response.raise_for_status()
            _verify_digest(response.content, digest, label="manifest")
            manifest = response.json()
            token = None
            if response.request.headers.get("Authorization", "").startswith("Bearer "):
                token = response.request.headers["Authorization"].removeprefix("Bearer ")
            pending: list[tuple[str, str]] = []
            for index, layer in enumerate(manifest.get("layers", [])):
                layer_digest = layer.get("digest")
                if not isinstance(layer_digest, str) or not _DIGEST_RE.match(layer_digest):
                    raise ValueError(f"OCI layer {index} is missing a sha256 digest")
                name = _layer_filename(layer, index)
                destination = temp_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    logger.info(f"model_download_skip ref={name} (already cached)")
                    continue
                pending.append((layer_digest, name))

            token_lock = threading.Lock()

            def _download_layer(layer_digest: str, name: str) -> None:
                # Streams one shard, refreshing the shared bearer token once on a 401.
                nonlocal token
                destination = temp_dir / name
                blob_url = f"https://{registry}/v2/{repository}/blobs/{layer_digest}"
                with token_lock:
                    current = token
                blob_headers = {"Authorization": f"Bearer {current}"} if current else {}
                auth_response = _stream_blob_to_file(
                    client, blob_url, blob_headers, destination, layer_digest, label=name
                )
                if auth_response is not None:
                    with token_lock:
                        # Another worker may have refreshed the token while we streamed.
                        if token == current:
                            token = _bearer_token(client, auth_response, repository)
                        current = token
                    _stream_blob_to_file(
                        client,
                        blob_url,
                        {"Authorization": f"Bearer {current}"},
                        destination,
                        layer_digest,
                        label=name,
                    )

            if pending:
                max_workers = max(1, min(self.settings.model_download_concurrency, len(pending)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_download_layer, layer_digest, name)
                        for layer_digest, name in pending
                    ]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
        _require_loadable_model_files(temp_dir, source="oci")
        done_marker_payload = {
            "source": original_ref,
            "registry": registry,
            "repository": repository,
            "digest": digest,
        }
        (temp_dir / ".albedo-model-cache.json").write_text(
            json.dumps(done_marker_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.replace(cache_dir)
        self._apply_canonical_config(cache_dir)
        file_count, total_size_bytes = _tree_stats(cache_dir)
        return ResolvedModel(
            original_ref, str(cache_dir), "oci", False, file_count, total_size_bytes
        )

    def _apply_canonical_config(self, model_dir: Path) -> None:
        # Overwrites the model's config files with the canonical genesis pin when enabled.
        if not self.settings.use_canonical_model_config:
            return
        if not model_dir.is_dir():
            return
        apply_canonical_model_config(model_dir)


def _stream_blob_to_file(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    destination: Path,
    expected_digest: str,
    *,
    label: str,
) -> httpx.Response | None:
    # Streams one blob to a temp file with sha256 verification; returns the response on a 401.
    temp_destination = destination.with_suffix(destination.suffix + ".download")
    digest = hashlib.sha256()
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == 401:
            return response
        response.raise_for_status()
        with download_heartbeat(label), temp_destination.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                digest.update(chunk)
                handle.write(chunk)
    actual = "sha256:" + digest.hexdigest()
    if actual != expected_digest:
        temp_destination.unlink(missing_ok=True)
        raise ValueError(f"{label} digest mismatch: expected {expected_digest}, got {actual}")
    temp_destination.replace(destination)
    return None


def parse_oci_ref(model_ref: str) -> tuple[str, str, str] | None:
    # Parses "[registry/]repo@sha256:..." into (registry, repository, digest); None if not OCI.
    ref = model_ref.removeprefix("oci://").removeprefix("docker://")
    if "@sha256:" not in ref:
        return None
    left, digest_tail = ref.rsplit("@", 1)
    digest = digest_tail
    if not _DIGEST_RE.match(digest):
        return None
    registry, sep, repository = left.partition("/")
    if not sep or not repository:
        return None
    if "." not in registry:
        return _DEFAULT_OCI_REGISTRY, left, digest
    return registry, repository, digest


def split_s3_uri(uri: str) -> tuple[str, str]:
    # Splits s3://bucket/prefix into its parts.
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _bearer_token(client: httpx.Client, response: httpx.Response, repository: str) -> str:
    # Exchanges a registry 401 bearer challenge for a pull token.
    challenge = response.headers.get("www-authenticate", "")
    if not challenge.lower().startswith("bearer "):
        raise RuntimeError("registry returned 401 without a bearer challenge")
    params = _parse_auth_challenge(challenge[len("Bearer ") :])
    realm = params.get("realm")
    if not realm:
        raise RuntimeError("registry bearer challenge did not include a realm")
    query = dict(parse_qsl(urlparse(realm).query))
    if params.get("service"):
        query["service"] = params["service"]
    query.setdefault("scope", f"repository:{repository}:pull")
    token_url = realm.split("?", 1)[0] + "?" + urlencode(query)
    token_response = client.get(token_url)
    token_response.raise_for_status()
    payload = token_response.json()
    token = payload.get("token") or payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("registry token endpoint did not return a token")
    return token


def _parse_auth_challenge(raw: str) -> dict[str, str]:
    # Parses key="value" pairs out of a WWW-Authenticate challenge.
    values: dict[str, str] = {}
    for part in re.finditer(r'(\w+)="([^"]*)"', raw):
        values[part.group(1)] = part.group(2)
    return values


def _layer_filename(layer: dict[str, Any], index: int) -> str:
    # Path-safe filename for a layer: its title annotation when safe, else the digest.
    annotations = layer.get("annotations") if isinstance(layer.get("annotations"), dict) else {}
    title = annotations.get("org.opencontainers.image.title")
    if (
        isinstance(title, str)
        and title
        and not title.startswith("/")
        and ".." not in Path(title).parts
    ):
        return title
    digest = str(layer.get("digest", f"layer-{index}"))
    return digest.replace(":", "-")


def _has_loadable_model_files(path: Path) -> bool:
    # True when the directory holds a config plus at least one safetensors artifact.
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file() and not (path / "params.json").is_file():
        return False
    if (path / "model.safetensors.index.json").is_file():
        return True
    return any(path.glob("*.safetensors"))


def _require_loadable_model_files(path: Path, *, source: str) -> None:
    # Deletes and raises when a download completed without loadable model files.
    if _has_loadable_model_files(path):
        return
    logger.warning(
        f"model_cache_invalid source={source} path={path} reason=download_missing_model_files"
    )
    shutil.rmtree(path, ignore_errors=True)
    raise FileNotFoundError(f"downloaded {source} model at {path} is missing loadable model files")


def _verify_digest(payload: bytes, expected: str, *, label: str) -> None:
    # Raises when the payload's sha256 does not match the pinned digest.
    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} digest mismatch: expected {expected}, got {actual}")


def _tree_stats(path: Path) -> tuple[int, int]:
    # Returns (file_count, total_size_bytes) for a file or directory tree.
    if path.is_file():
        return 1, path.stat().st_size
    file_count = 0
    total_size = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_size += item.stat().st_size
    return file_count, total_size


# ── Score-bridge scoring (merged from scoring.py) ────────────────────────────


class ScoreBridgeUnavailable(RuntimeError):
    # Raised when no backend score-bridge client is connected within the grace window.
    pass


class ScoreBridgeHub:
    # Holds the single attached backend websocket and correlates request/response frames by id.
    def __init__(self) -> None:
        # Lock-guarded socket + event loop + pending-future map.
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._websocket: WebSocket | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    @property
    def connected(self) -> bool:
        # True while a backend client socket is attached.
        with self._lock:
            return self._websocket is not None and self._loop is not None

    async def attach(self, websocket: WebSocket) -> None:
        # Accepts a backend client, replacing any previous one (its pending futures fail fast).
        await websocket.accept()
        loop = asyncio.get_running_loop()
        old: WebSocket | None = None
        with self._lock:
            old = self._websocket
            self._loop = loop
            self._websocket = websocket
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ScoreBridgeUnavailable("score bridge client replaced"))
            self._pending.clear()
        if old is not None and old is not websocket:
            try:
                await old.close(code=1012)
            except Exception as exc:  # noqa: BLE001 - best-effort close of replaced bridge
                logger.debug(
                    f"[score-bridge] best-effort close of replaced websocket failed: {exc}"
                )
        try:
            while True:
                message = await websocket.receive_json()
                await self._handle_message(message)
        except WebSocketDisconnect:
            pass
        finally:
            with self._lock:
                if self._websocket is websocket:
                    self._websocket = None
                    self._loop = None
                    for future in self._pending.values():
                        if not future.done():
                            future.set_exception(
                                ScoreBridgeUnavailable("score bridge disconnected")
                            )
                    self._pending.clear()

    def request(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
        endpoint: str = "/score-batch",
    ) -> dict[str, Any]:
        # Sends one score_request frame from a worker thread and blocks on the correlated response.
        # Wait up to 120s for the bridge to (re)connect - covers disconnects during vLLM cleanup.
        _deadline = time.monotonic() + 120.0
        while True:
            with self._lock:
                loop = self._loop
                websocket = self._websocket
            if loop is not None and websocket is not None:
                break
            remaining = _deadline - time.monotonic()
            if remaining <= 0:
                raise ScoreBridgeUnavailable("no score bridge client connected")
            time.sleep(min(2.0, remaining))
        future = asyncio.run_coroutine_threadsafe(
            self._request_on_loop(
                websocket, payload, timeout_seconds=timeout_seconds, endpoint=endpoint
            ),
            loop,
        )
        return future.result(timeout=timeout_seconds + 5.0)

    async def _request_on_loop(
        self,
        websocket: WebSocket,
        payload: dict[str, Any],
        *,
        timeout_seconds: float,
        endpoint: str,
    ) -> dict[str, Any]:
        # On the WS event loop: register the pending future, send the frame, await the response.
        request_id = str(uuid4())
        loop = asyncio.get_running_loop()
        response_future: asyncio.Future[dict[str, Any]] = loop.create_future()
        with self._lock:
            if self._websocket is not websocket:
                raise ScoreBridgeUnavailable("score bridge client changed")
            self._pending[request_id] = response_future
        try:
            await websocket.send_json(
                {
                    "type": "score_request",
                    "request_id": request_id,
                    "endpoint": endpoint,
                    "payload": payload,
                }
            )
            return await asyncio.wait_for(response_future, timeout=timeout_seconds)
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        # Resolves the pending future matching a score_response frame (error or body).
        if message.get("type") != "score_response":
            return
        request_id = str(message.get("request_id") or "")
        with self._lock:
            future = self._pending.get(request_id)
        if future is None or future.done():
            return
        error = message.get("error")
        if error:
            future.set_exception(RuntimeError(str(error)))
            return
        body = message.get("body")
        if not isinstance(body, dict):
            future.set_exception(RuntimeError("score bridge response body must be an object"))
            return
        future.set_result(body)


score_bridge_hub = ScoreBridgeHub()


class ScoreBridgeScorer:
    # Sends category-prep and score-batch payloads through the attached backend bridge.
    def __init__(self, settings: RemoteEvalSettings):
        # Keeps timeout + validity-fraction knobs.
        self.settings = settings

    def start_category_prep(self, *, request: Any, samples: list[EvalSample]) -> str | None:
        # Kicks off async category generation on the backend; returns its prep id when started.
        body = score_bridge_hub.request(
            _category_prep_payload(request, samples),
            timeout_seconds=self.settings.scoring_timeout_seconds,
            endpoint="/category-prep",
        )
        value = body.get("category_prep_id")
        return value if isinstance(value, str) and value else None

    def score(
        self,
        *,
        request: Any,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        category_prep_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # Scores every valid pair batch-by-batch and returns (records, merged summary).
        all_records: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for payload in _score_batch_payloads(
            request, samples, king_results, challenger_results, category_prep_id=category_prep_id
        ):
            body = score_bridge_hub.request(
                payload, timeout_seconds=self.settings.scoring_timeout_seconds
            )
            records = body.get("scoring_records", [])
            if not isinstance(records, list):
                raise ValueError("score bridge returned non-list scoring_records")
            all_records.extend(records)
            summary = body.get("summary", {})
            if isinstance(summary, dict):
                summaries.append(summary)
        merged = _merge_summaries(
            all_records, summaries, min_valid_fraction=self.settings.scoring_min_valid_fraction
        )
        return all_records, merged


def _category_prep_payload(request: Any, samples: list[EvalSample]) -> dict[str, Any]:
    # /category-prep body (compat name): every sample's prompt for question generation.
    return {
        "eval_run_id": str(request.eval_run_id),
        "batch_id": "category-prep",
        "total_sample_count": len(samples),
        "samples": [{"sample_id": sample.sample_id, "prompt": sample.prompt} for sample in samples],
    }


def _score_batch_payloads(
    request: Any,
    samples: list[EvalSample],
    king_results: list[GenerationResult],
    challenger_results: list[GenerationResult],
    *,
    category_prep_id: str | None = None,
) -> list[dict[str, Any]]:
    # /score-batch bodies for every valid king/challenger pair, chunked by scoring_batch_size.
    king_by_id = {result.sample_id: result for result in king_results}
    challenger_by_id = {result.sample_id: result for result in challenger_results}
    valid_samples = [
        sample
        for sample in samples
        if sample.sample_id in king_by_id
        and sample.sample_id in challenger_by_id
        and not king_by_id[sample.sample_id].error
        and not challenger_by_id[sample.sample_id].error
    ]
    payloads = []
    for batch_idx, batch in enumerate(
        _chunks(valid_samples, request.dataset.scoring_batch_size), start=1
    ):
        payloads.append(
            {
                "eval_run_id": str(request.eval_run_id),
                "batch_id": f"score-{batch_idx:04d}",
                "judge_models": list(JUDGE_MODELS[: request.scoring.judge_count]),
                "total_sample_count": len(samples),
                "category_prep_id": category_prep_id,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "previous_king_output": king_by_id[sample.sample_id].text,
                        "challenger_output": challenger_by_id[sample.sample_id].text,
                    }
                    for sample in batch
                ],
            }
        )
    return payloads


def _merge_summaries(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    min_valid_fraction: float,
) -> dict[str, Any]:
    # Re-aggregates all records into one summary, keeping the per-batch summaries for the trace.
    summary = aggregate_scores(records, min_valid_fraction=min_valid_fraction)
    if summaries:
        summary["batch_summaries"] = summaries
    return summary


def _chunks(items: list, size: int) -> list[list]:
    # Splits a list into fixed-size chunks.
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


# ── Artifact spool + upload (merged from artifacts.py) ───────────────────────


@dataclass(frozen=True)
class ArtifactUpload:
    # One uploaded (or locally spooled) artifact with addressing + integrity metadata.
    name: str
    uri: str
    bucket: str
    object_key: str
    sha256: str
    size_bytes: int
    content_type: str
    local_path: Path

    def metadata(self) -> dict[str, object]:
        # Metadata block published in the verdict event.
        return {
            "uri": self.uri,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
        }


class LocalOnlyArtifactUploader:
    # Upload-disabled mode: reports local-cache:// URIs so the verdict still carries addressing.
    def upload_run_artifacts(
        self, *, eval_run_id: UUID, artifact_prefix: str, files: dict[str, Path]
    ) -> dict[str, ArtifactUpload]:
        # Computes checksums + keys without touching S3.
        uploads: dict[str, ArtifactUpload] = {}
        bucket, prefix = split_s3_prefix(artifact_prefix)
        for name, path in files.items():
            object_key = f"{prefix}/{path.name}" if prefix else path.name
            uploads[name] = ArtifactUpload(
                name=name,
                uri=f"local-cache://{path}"
                if not bucket
                else f"local-cache://{bucket}/{object_key}",
                bucket=bucket or "",
                object_key=object_key,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                content_type=content_type_for(path),
                local_path=path,
            )
        return uploads


class S3ArtifactUploader:
    # Uploads every artifact under the request's s3:// prefix with sha256 + run metadata.
    def __init__(self, settings: RemoteEvalSettings):
        # Keeps the ALBEDO_REMOTE_S3_* credentials/endpoint.
        self.settings = settings

    def upload_run_artifacts(
        self, *, eval_run_id: UUID, artifact_prefix: str, files: dict[str, Path]
    ) -> dict[str, ArtifactUpload]:
        # Uploads each file and returns its addressing + integrity record.
        bucket, prefix = split_s3_prefix(artifact_prefix)
        if not bucket:
            raise ValueError(f"artifact_prefix must be an s3:// URI, got {artifact_prefix}")

        import boto3

        session_kwargs: dict[str, str] = {}
        if self.settings.s3_access_key_id:
            session_kwargs["aws_access_key_id"] = self.settings.s3_access_key_id
        if self.settings.s3_secret_access_key:
            session_kwargs["aws_secret_access_key"] = self.settings.s3_secret_access_key
        if self.settings.s3_session_token:
            session_kwargs["aws_session_token"] = self.settings.s3_session_token
        if self.settings.s3_region:
            session_kwargs["region_name"] = self.settings.s3_region

        client_kwargs: dict[str, str] = {}
        if self.settings.s3_endpoint_url:
            client_kwargs["endpoint_url"] = self.settings.s3_endpoint_url

        client = boto3.session.Session(**session_kwargs).client("s3", **client_kwargs)
        uploads: dict[str, ArtifactUpload] = {}
        for name, path in files.items():
            object_key = f"{prefix}/{path.name}" if prefix else path.name
            content_type = content_type_for(path)
            checksum = sha256_file(path)
            try:
                client.upload_file(
                    str(path),
                    bucket,
                    object_key,
                    ExtraArgs={
                        "ContentType": content_type,
                        "Metadata": {
                            "sha256": checksum,
                            "eval_run_id": str(eval_run_id),
                            "artifact_name": name,
                        },
                    },
                )
            except Exception as exc:
                logger.exception(
                    f"[remote-artifacts] S3 upload failed eval_run={eval_run_id} "
                    f"artifact={name} key={object_key} path={path}: {exc}"
                )
                raise
            uploads[name] = ArtifactUpload(
                name=name,
                uri=f"s3://{bucket}/{object_key}",
                bucket=bucket,
                object_key=object_key,
                sha256=checksum,
                size_bytes=path.stat().st_size,
                content_type=content_type,
                local_path=path,
            )
        return uploads


class RunArtifactSpool:
    # Per-run local spool directory the artifact files are written into before upload.
    def __init__(self, root: str | Path, eval_run_id: UUID):
        # Creates <root>/<eval_run_id>/.
        self.run_dir = Path(root) / str(eval_run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(self, filename: str, rows: list[dict[str, Any]]) -> Path:
        # Writes rows as compact sorted-key JSONL.
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        return path

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        # Writes one pretty-printed sorted-key JSON document.
        path = self.run_dir / filename
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return path

    def write_text(self, filename: str, payload: str) -> Path:
        # Writes a plain-text artifact.
        path = self.run_dir / filename
        path.write_text(payload, encoding="utf-8")
        return path

    def cleanup(self) -> None:
        # Removes the spool directory (best-effort).
        shutil.rmtree(self.run_dir, ignore_errors=True)


def build_artifact_uploader(
    settings: RemoteEvalSettings,
) -> LocalOnlyArtifactUploader | S3ArtifactUploader:
    # Chooses the S3 uploader unless uploads are disabled.
    if not settings.upload_artifacts:
        return LocalOnlyArtifactUploader()
    return S3ArtifactUploader(settings)


def split_s3_prefix(uri: str) -> tuple[str | None, str]:
    # Splits an s3:// prefix into (bucket, key prefix); (None, uri) for non-S3 prefixes.
    if not uri.startswith("s3://"):
        return None, uri.rstrip("/")
    without_scheme = uri.removeprefix("s3://").rstrip("/")
    bucket, _, prefix = without_scheme.partition("/")
    return bucket or None, prefix.strip("/")


def sha256_file(path: Path) -> str:
    # Streams the file through sha256 and returns "sha256:<hex>".
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def content_type_for(path: Path) -> str:
    # Content type by artifact extension.
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


# ── Worker API + duel orchestration ───────────────────────────────────────────


T = TypeVar("T")
_CANONICAL_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "tokenizers" / "Qwen3.6-35B-A3B"
)
# Fixed vLLM engine knobs the simplified settings no longer expose (source defaults).
_GPU_MEMORY_UTILIZATION = 0.95
_KV_CACHE_DTYPE = "auto"

app = FastAPI(title="Albedo Remote Eval API", version="0.1.0")
store = RunStore()
require_auth = bearer_auth(lambda: get_settings().remote_eval.auth_token)
_AUTH = Depends(require_auth)


@dataclass
class EvalRun(Run):
    # An eval run: adds the verdict-shaped failure event and status document.
    def fail(self, *, fault_code: str, fault_message: str, retryable: bool = True) -> None:
        # Emits a terminal failed verdict event and marks the run failed.
        self.append_event(
            {
                "type": "verdict",
                "eval_run_id": str(self.request.eval_run_id),
                "state": "failed",
                "fault_class": "REMOTE_EVAL_FAULT",
                "fault_code": fault_code,
                "fault_message": fault_message,
                "retryable": retryable,
                "artifacts": {},
            }
        )
        self.set_state("failed")

    def as_status(self) -> dict[str, Any]:
        # Returns the final verdict if present, else a lightweight status snapshot.
        verdict = self.final_event("verdict")
        if verdict:
            return verdict
        return {
            "remote_run_id": self.run_id,
            "eval_run_id": str(self.request.eval_run_id),
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class GpuTopology:
    # The fixed 4+4 GPU split the duel runs on.
    accelerator: str
    previous_king: list[str]
    challenger: list[str]
    tensor_parallel_size_per_model: int

    def as_dict(self) -> dict[str, object]:
        # Event/verdict payload fragment.
        return {
            "accelerator": self.accelerator,
            "previous_king": self.previous_king,
            "challenger": self.challenger,
            "tensor_parallel_size_per_model": self.tensor_parallel_size_per_model,
        }


# ── HTTP API (wire-identical to the source remote_api.py) ────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    # Liveness probe (unauthenticated).
    return {"status": "ok"}


@app.get("/ready", dependencies=[_AUTH])
def ready() -> dict[str, object]:
    # Readiness + identity + busy state for the backend's host selection.
    settings = get_settings().remote_eval
    active = store.list_active()
    warnings = []
    if not settings.dataset_root and not (settings.mock_auto_verdict or settings.mock_scored_duel):
        warnings.append("ALBEDO_REMOTE_DATASET_ROOT is not set")
    return ready_payload(
        host_id=settings.host_id,
        role="EVAL",
        active_runs=len(active),
        accelerator_type=settings.accelerator_type,
        gpu_count=settings.gpu_count,
        free_gpu_count=0 if active else settings.gpu_count,
        generation_backend="vllm",
        warnings=warnings,
        score_bridge_connected=score_bridge_hub.connected,
    )


@app.websocket("/score-bridge")
async def score_bridge(websocket: WebSocket) -> None:
    # Backend-initiated scoring channel; bad bearer token closes with 1008.
    settings = get_settings().remote_eval
    if settings.auth_token:
        expected = f"Bearer {settings.auth_token}"
        if websocket.headers.get("authorization") != expected:
            await websocket.close(code=1008)
            return
    await score_bridge_hub.attach(websocket)


@app.post("/eval-runs", dependencies=[_AUTH])
def start_eval_run(request: EvalRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    # Accepts one duel (idempotent on eval_run_id, 409 while busy) and executes in the background.
    settings = get_settings().remote_eval
    remote_run_id = str(request.eval_run_id)

    def _factory() -> EvalRun:
        # Builds the accepted run; mock mode short-circuits straight to a smoke verdict.
        run = EvalRun(run_id=remote_run_id, request=request, state="accepted")
        run.append_event(
            {
                "type": "eval_started",
                "remote_run_id": remote_run_id,
                "eval_run_id": remote_run_id,
                "message": "Remote eval run accepted",
            }
        )
        if settings.mock_auto_verdict:
            for event in _smoke_progress_and_verdict(
                request, challenger_won=settings.mock_challenger_won
            ):
                run.append_event(event)
            run.set_state("succeeded")
        return run

    try:
        run = store.get_or_create(remote_run_id, _factory)
    except WorkerBusy as busy:
        raise HTTPException(
            status_code=409, detail=f"eval worker busy: {busy.active_count} active run(s)"
        ) from busy
    if not settings.mock_auto_verdict:
        queued_run = store.mark_worker_started(remote_run_id)
        if queued_run:
            background_tasks.add_task(_execute_remote_run, queued_run, settings)
    return {"remote_run_id": run.run_id, "state": run.state}


@app.get("/eval-runs/{remote_run_id}", dependencies=[_AUTH])
def get_eval_run(remote_run_id: str) -> dict[str, object]:
    # Status snapshot (or the final verdict once done).
    run = store.get(remote_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="remote run not found")
    return run.as_status()


@app.get("/eval-runs/{remote_run_id}/events", dependencies=[_AUTH])
def get_eval_run_events(remote_run_id: str) -> dict[str, list[dict[str, object]]]:
    # Full event list for the backend to poll until a verdict appears.
    run = store.get(remote_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="remote run not found")
    return {"events": run.events}


def _execute_remote_run(run: EvalRun, settings: RemoteEvalSettings) -> None:
    # Background-task shim: the worker must never die silently.
    try:
        RemoteEvalWorker(settings).execute(run)
    except Exception as exc:  # noqa: BLE001 - background task must not die silently
        logger.exception(f"[remote-api] background eval run failed remote_run={run.run_id}: {exc}")


# ── Duel execution ────────────────────────────────────────────────────────────


class RemoteEvalWorker:
    # Runs one duel end-to-end: samples, models, parallel vLLM generation, scoring, artifacts.
    def __init__(self, settings: RemoteEvalSettings):
        # Builds the resolver, uploader, and score-bridge scorer for this run.
        self.settings = settings
        self._model_resolver = ModelArtifactResolver(settings)
        self._artifact_uploader: LocalOnlyArtifactUploader | S3ArtifactUploader = (
            build_artifact_uploader(settings)
        )
        self._scorer = ScoreBridgeScorer(settings)

    def execute(self, run: EvalRun) -> None:
        # Executes the duel, converting any exception into a failed verdict event.
        try:
            self._execute(run)
        except Exception as exc:
            logger.exception(
                f"[remote-worker] eval failed remote_run={run.run_id} "
                f"eval_run={run.request.eval_run_id} submission={run.request.submission_id}: {exc}"
            )
            run.fail(
                fault_code="remote_worker_failed", fault_message=f"{type(exc).__name__}: {exc}"
            )

    def _execute_scored_mock(self, run: EvalRun) -> None:
        # Fabricated generations, REAL scoring: score-bridge -> judges -> aggregation -> margin.
        request = run.request
        topology = self._topology(request)
        n = max(1, request.dataset.sample_count)
        samples = [
            EvalSample(
                sample_id=f"mock-shard-{i:05d}-of-00001.parquet:{i}:0",
                prompt=f"Mock coding task #{i}: implement the function described in the ticket.",
                messages=[{"role": "user", "content": f"Mock coding task #{i}"}],
            )
            for i in range(n)
        ]
        category_prep_id = self._start_category_prep(run, request, samples)
        run.set_state("generating")
        run.append_event(
            {
                "type": "generation_started",
                "eval_run_id": str(request.eval_run_id),
                "gpu_topology": topology.as_dict(),
                "sample_count": len(samples),
                "generation_batch_size": request.dataset.generation_batch_size,
                "mock_scored_duel": True,
            }
        )
        king_results = [
            GenerationResult(s.sample_id, f"King draft for {s.sample_id}: a plain attempt.")
            for s in samples
        ]
        challenger_results = [
            GenerationResult(
                s.sample_id,
                f"MOCK-STRONG-CHALLENGER answer for {s.sample_id}: thorough, tested fix.",
            )
            for s in samples
        ]
        self._emit_generation_batches(
            run, request, samples, king_results, challenger_results, topology
        )
        run.set_state("scoring")
        run.append_event(
            {
                "type": "scoring_started",
                "eval_run_id": str(request.eval_run_id),
                "scoring_batch_size": request.dataset.scoring_batch_size,
            }
        )
        scoring_records, scoring_summary = self._score_pairs(
            request=request,
            samples=samples,
            king_results=king_results,
            challenger_results=challenger_results,
            category_prep_id=category_prep_id,
        )
        self._emit_scoring_batches(run, request, scoring_records)
        verdict = self._build_verdict(
            request=request,
            topology=topology,
            samples=samples,
            king_results=king_results,
            challenger_results=challenger_results,
            scoring_summary=scoring_summary,
        )
        run.append_event(verdict)
        run.set_state(str(verdict["state"]))

    def _execute(self, run: EvalRun) -> None:
        # Pipeline: samples -> category prep -> models -> generate -> score -> verdict -> artifacts.
        if self.settings.mock_scored_duel:
            self._execute_scored_mock(run)
            return
        request = run.request
        topology = self._topology(request)
        samples = self._load_samples(request, tokenizer_path=str(_CANONICAL_TOKENIZER_PATH))
        category_prep_id = self._start_category_prep(run, request, samples)
        run.append_event(
            {
                "type": "model_resolution_started",
                "eval_run_id": str(request.eval_run_id),
                "models": ["challenger", "previous_king"],
            }
        )
        king_model = self._resolve_model_for_side(run, request, side="previous_king")
        challenger_model = self._resolve_model_for_side(run, request, side="challenger")
        run.append_event(
            {
                "type": "model_resolution_done",
                "eval_run_id": str(request.eval_run_id),
                "models": [
                    king_model.as_event(side="previous_king"),
                    challenger_model.as_event(side="challenger"),
                ],
            }
        )
        run.set_state("generating")
        run.append_event(
            {
                "type": "generation_started",
                "eval_run_id": str(request.eval_run_id),
                "gpu_topology": topology.as_dict(),
                "sample_count": len(samples),
                "generation_batch_size": request.dataset.generation_batch_size,
            }
        )

        king_generator = self._vllm_generator(topology.previous_king, king_model.local_path)
        challenger_generator = self._vllm_generator(
            topology.challenger, challenger_model.local_path
        )
        cleanup_stale_vllm_resources()
        with ThreadPoolExecutor(max_workers=2) as executor:
            king_future = executor.submit(king_generator.generate, samples)
            challenger_future = executor.submit(challenger_generator.generate, samples)
            king_results = king_future.result()
            challenger_results = challenger_future.result()

        self._emit_generation_batches(
            run, request, samples, king_results, challenger_results, topology
        )
        run.set_state("scoring")
        run.append_event(
            {
                "type": "scoring_started",
                "eval_run_id": str(request.eval_run_id),
                "scoring_batch_size": request.dataset.scoring_batch_size,
            }
        )
        scoring_records, scoring_summary = self._score_pairs(
            request=request,
            samples=samples,
            king_results=king_results,
            challenger_results=challenger_results,
            category_prep_id=category_prep_id,
        )
        self._emit_scoring_batches(run, request, scoring_records)
        verdict = self._build_verdict(
            request=request,
            topology=topology,
            samples=samples,
            king_results=king_results,
            challenger_results=challenger_results,
            scoring_summary=scoring_summary,
        )
        verdict = self._write_and_upload_artifacts(
            run=run,
            request=request,
            verdict=verdict,
            samples=samples,
            king_results=king_results,
            challenger_results=challenger_results,
            scoring_records=scoring_records,
        )
        run.append_event(verdict)
        run.set_state(str(verdict["state"]))

    def _start_category_prep(
        self, run: EvalRun, request: EvalRequest, samples: list[EvalSample]
    ) -> str | None:
        # Kicks off backend category generation early; failure falls back to fixed-metric scoring.
        run.append_event(
            {
                "type": "category_prep_started",
                "eval_run_id": str(request.eval_run_id),
                "sample_count": len(samples),
            }
        )
        try:
            category_prep_id = self._scorer.start_category_prep(request=request, samples=samples)
        except Exception as exc:
            logger.warning(
                f"[remote-worker] category prep failed eval_run={request.eval_run_id} "
                f"submission={request.submission_id}, falling back to sync/fixed scoring: {exc}"
            )
            run.append_event(
                {
                    "type": "category_prep_failed",
                    "eval_run_id": str(request.eval_run_id),
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback": "score_batch_synchronous_or_fixed_metrics",
                }
            )
            return None
        run.append_event(
            {
                "type": "category_prep_done",
                "eval_run_id": str(request.eval_run_id),
                "category_prep_id": category_prep_id,
                "state": "started" if category_prep_id else "skipped",
            }
        )
        return category_prep_id

    def _load_samples(
        self, request: EvalRequest, *, tokenizer_path: str | None = None
    ) -> list[EvalSample]:
        # Loads the pinned samples; re-derives deterministic sample ids when the request omits them.
        if not self.settings.dataset_root:
            raise ValueError("ALBEDO_REMOTE_DATASET_ROOT is required for SWE-ZERO parquet loading")

        sample_ids = list(request.dataset.sample_ids)
        if not sample_ids:
            manifest_path = Path(self.settings.dataset_root) / "manifest.json"
            manifest = load_manifest_file(
                manifest_path, expected_sha256=request.dataset.manifest_hash
            )
            sample_ids = multi_source_manifest_sample_ids(
                manifest,
                block_hash=request.dataset.sample_seed,
                sample_count=request.dataset.sample_count,
                max_turns_per_sample=request.dataset.max_turns_per_sample,
            )
        return load_swe_zero_samples(
            dataset_root=self.settings.dataset_root,
            sample_ids=sample_ids,
            tokenizer_path=tokenizer_path,
            enable_thinking=True,
        )

    def _resolve_model_for_side(
        self, run: EvalRun, request: EvalRequest, *, side: str
    ) -> ResolvedModel:
        # Downloads/verifies one side's model artifact and records the resolution event.
        model_ref = (
            request.previous_king.model_uri
            if side == "previous_king"
            else request.challenger.model_uri
        )
        resolved = self._model_resolver.resolve(model_ref)
        run.append_event(
            {
                "type": "model_resolved",
                "eval_run_id": str(request.eval_run_id),
                **resolved.as_event(side=side),
            }
        )
        return resolved

    def _vllm_generator(self, gpu_ids: list[str], model: str) -> VllmProcessGenerator:
        # Builds one side's vLLM generator with the canonical (or fallback) sampling config.
        sampling_config = self._effective_sampling_config()
        return VllmProcessGenerator(
            model=model,
            gpu_ids=gpu_ids,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=sampling_config["temperature"],
            top_p=sampling_config["top_p"],
            top_k=sampling_config["top_k"],
            max_model_len=self._effective_max_model_len(),
            enforce_eager=self.settings.enforce_eager,
            gpu_memory_utilization=_GPU_MEMORY_UTILIZATION,
            kv_cache_dtype=_KV_CACHE_DTYPE,
        )

    def _effective_max_model_len(self) -> int | None:
        # The canonical context length when pinning is on, else vLLM's model default.
        if self.settings.use_canonical_model_config:
            return canonical_max_model_len()
        return None

    def _effective_sampling_config(self) -> dict[str, float | int | None]:
        # The canonical generation config when pinning is on, else greedy decoding.
        if not self.settings.use_canonical_model_config:
            return {"temperature": 0.0, "top_p": 1.0, "top_k": None}
        generation_config = canonical_generation_config()
        return {
            "temperature": float(generation_config["temperature"]),
            "top_p": float(generation_config["top_p"]),
            "top_k": int(generation_config["top_k"]),
        }

    def _topology(self, request: EvalRequest) -> GpuTopology:
        # Validates the fixed 8-GPU 4+4 topology against the request.
        previous_king = _parse_gpu_ids(self.settings.previous_king_gpu_ids)
        challenger = _parse_gpu_ids(self.settings.challenger_gpu_ids)
        if request.gpu_request.min_gpus != 8 or request.gpu_request.preferred_gpus != 8:
            raise ValueError("remote eval target requires an 8-GPU request")
        if request.gpu_request.tensor_parallel_size_per_model != 4:
            raise ValueError("remote eval requires tensor_parallel_size_per_model=4")
        if len(previous_king) != request.gpu_request.previous_king_gpu_count:
            raise ValueError("previous king GPU group does not match request")
        if len(challenger) != request.gpu_request.challenger_gpu_count:
            raise ValueError("challenger GPU group does not match request")
        if len(previous_king) != 4 or len(challenger) != 4:
            raise ValueError("remote eval requires fixed 4 GPU groups for both models")
        overlap = set(previous_king) & set(challenger)
        if overlap:
            raise ValueError(f"GPU groups overlap: {sorted(overlap)}")
        return GpuTopology(
            accelerator=self.settings.accelerator_type,
            previous_king=previous_king,
            challenger=challenger,
            tensor_parallel_size_per_model=request.gpu_request.tensor_parallel_size_per_model,
        )

    def _emit_generation_batches(
        self,
        run: EvalRun,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        topology: GpuTopology,
    ) -> None:
        # Emits one generation_batch_done event per batch with per-side error counts.
        king_by_id = {result.sample_id: result for result in king_results}
        challenger_by_id = {result.sample_id: result for result in challenger_results}
        for batch_idx, batch in enumerate(
            _chunks(samples, request.dataset.generation_batch_size), start=1
        ):
            sample_ids = [sample.sample_id for sample in batch]
            run.append_event(
                {
                    "type": "generation_batch_done",
                    "eval_run_id": str(request.eval_run_id),
                    "batch_id": f"gen-{batch_idx:04d}",
                    "sample_ids": sample_ids,
                    "models": ["challenger", "previous_king"],
                    "gpu_ids": topology.previous_king + topology.challenger,
                    "king_errors": sum(
                        1 for sample_id in sample_ids if king_by_id[sample_id].error
                    ),
                    "chal_errors": sum(
                        1 for sample_id in sample_ids if challenger_by_id[sample_id].error
                    ),
                    "generated_sample_count": min(
                        batch_idx * request.dataset.generation_batch_size, len(samples)
                    ),
                    "state": "succeeded",
                }
            )

    def _emit_scoring_batches(
        self, run: EvalRun, request: EvalRequest, scoring_records: list[dict[str, object]]
    ) -> None:
        # Emits one scoring_batch_done event per batch with judge-error/category counters.
        scored_so_far = 0
        for batch_idx, batch in enumerate(
            _chunks(scoring_records, request.dataset.scoring_batch_size), start=1
        ):
            batch_scored = sum(1 for record in batch if record.get("scored"))
            scored_so_far += batch_scored
            judge_errors = sum(
                1
                for record in batch
                for result in record.get("judge_results", [])
                if isinstance(result, dict) and not result.get("parse_ok")
            )
            run.append_event(
                {
                    "type": "scoring_batch_done",
                    "eval_run_id": str(request.eval_run_id),
                    "batch_id": f"score-{batch_idx:04d}",
                    "sample_ids": [str(record["sample_id"]) for record in batch],
                    "judge_config_hash": request.scoring.judge_config_hash,
                    "judge_count": request.scoring.judge_count,
                    "allowed_scores": request.scoring.allowed_scores,
                    "scored_sample_count": scored_so_far,
                    "judge_errors": judge_errors,
                    "scoring_modes": sorted(
                        {str(record.get("scoring_mode") or "") for record in batch}
                    ),
                    "category_generation_errors": sum(
                        1 for record in batch if record.get("category_generation_error")
                    ),
                    "state": "succeeded"
                    if batch_scored == len(batch) and judge_errors == 0
                    else "failed",
                }
            )

    def _score_pairs(
        self,
        *,
        request: EvalRequest,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        category_prep_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # Scores valid pairs via the bridge, mapping empty/failed scoring into fault summaries.
        valid_pair_count = _valid_generated_pair_count(samples, king_results, challenger_results)
        if valid_pair_count == 0:
            return [], {
                "state": "failed",
                "score_challenger": None,
                "score_king": None,
                "challenger_won": None,
                "valid_turns": 0,
                "total_turns": 0,
                "judge_errors": 0,
                "scored_sample_count": 0,
                "fault_class": "REMOTE_EVAL_FAULT",
                "fault_code": "no_valid_generated_pairs",
                "fault_message": "No sample pair had both king and challenger output",
                "retryable": True,
            }
        try:
            return self._scorer.score(
                request=request,
                samples=samples,
                king_results=king_results,
                challenger_results=challenger_results,
                category_prep_id=category_prep_id,
            )
        except Exception as exc:
            logger.exception(
                f"[remote-worker] judge scoring failed eval_run={request.eval_run_id} "
                f"submission={request.submission_id} valid_pairs={valid_pair_count}: {exc}"
            )
            return [], {
                "state": "failed",
                "score_challenger": None,
                "score_king": None,
                "challenger_won": None,
                "valid_turns": 0,
                "total_turns": valid_pair_count,
                "judge_errors": valid_pair_count * request.scoring.judge_count,
                "scored_sample_count": 0,
                "fault_class": "PROVIDER_FAULT",
                "fault_code": "judge_provider_exhausted",
                "fault_message": f"Judge scoring failed: {type(exc).__name__}: {exc}",
                "retryable": True,
            }

    def _build_verdict(
        self,
        *,
        request: EvalRequest,
        topology: GpuTopology,
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        scoring_summary: dict[str, object],
    ) -> dict[str, object]:
        # Assembles the verdict event, re-checking the win margin on succeeded runs.
        king_errors = sum(1 for result in king_results if result.error)
        chal_errors = sum(1 for result in challenger_results if result.error)
        state = str(scoring_summary.get("state") or "failed")
        score_challenger = scoring_summary.get("score_challenger")
        score_king = scoring_summary.get("score_king")
        valid_turns = int(scoring_summary.get("valid_turns") or 0)
        scored_sample_count = int(scoring_summary.get("scored_sample_count") or valid_turns)
        judge_errors = int(scoring_summary.get("judge_errors") or 0)
        verdict = {
            "type": "verdict",
            "eval_run_id": str(request.eval_run_id),
            "state": state,
            "challenger_won": scoring_summary.get("challenger_won"),
            "score_challenger": score_challenger,
            "score_king": score_king,
            "judge_count": request.scoring.judge_count,
            "allowed_scores": request.scoring.allowed_scores,
            "valid_turns": valid_turns,
            "total_turns": len(samples),
            "generated_sample_count": len(samples),
            "scored_sample_count": scored_sample_count,
            "king_vllm_errors": king_errors,
            "chal_vllm_errors": chal_errors,
            "judge_errors": judge_errors,
            "required_win_margin": scoring_summary.get(
                "required_win_margin", CHALLENGER_WIN_MARGIN
            ),
            "gpu_topology": topology.as_dict(),
            "score_breakdown": {
                "by_judge": scoring_summary.get("by_judge", {}),
                "by_metric": scoring_summary.get("by_metric", {}),
            },
            "scoring_mode": scoring_summary.get("scoring_mode"),
            "artifacts": {},
            "artifact_metadata": {},
            "fault_class": scoring_summary.get("fault_class") if state != "succeeded" else None,
            "fault_code": scoring_summary.get("fault_code") if state != "succeeded" else None,
            "fault_message": scoring_summary.get("fault_message") if state != "succeeded" else None,
            "retryable": scoring_summary.get("retryable") if state != "succeeded" else None,
        }
        if state == "succeeded" and score_challenger is not None and score_king is not None:
            verdict["challenger_won"] = challenger_beats_king(
                float(score_challenger), float(score_king)
            )
        return verdict

    def _write_and_upload_artifacts(
        self,
        *,
        run: EvalRun,
        request: EvalRequest,
        verdict: dict[str, object],
        samples: list[EvalSample],
        king_results: list[GenerationResult],
        challenger_results: list[GenerationResult],
        scoring_records: list[dict[str, object]],
    ) -> dict[str, object]:
        # Writes the 7 trace files, uploads them, then uploads the enriched verdict.json itself.
        spool = RunArtifactSpool(self.settings.artifact_spool_dir, request.eval_run_id)
        king_by_id = {result.sample_id: result for result in king_results}
        challenger_by_id = {result.sample_id: result for result in challenger_results}
        generated_rows = []
        transcript_rows = []
        for sample in samples:
            king = king_by_id.get(sample.sample_id)
            challenger = challenger_by_id.get(sample.sample_id)
            generated_row = {
                "eval_run_id": str(request.eval_run_id),
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "previous_king_output": king.text if king else "",
                "challenger_output": challenger.text if challenger else "",
                "king_error": king.error if king else "missing_generation",
                "chal_error": challenger.error if challenger else "missing_generation",
            }
            generated_rows.append(generated_row)
            transcript_rows.append({**generated_row, "target": sample.target})
        judge_rows = []
        for record in scoring_records:
            for result in record.get("judge_results", []):
                if isinstance(result, dict):
                    judge_rows.append(
                        {
                            "eval_run_id": str(request.eval_run_id),
                            "sample_id": record.get("sample_id"),
                            "order": record.get("order"),
                            "sample_score": record.get("sample_score"),
                            **result,
                        }
                    )
        progress_rows = [
            {"sequence": idx, **event} for idx, event in enumerate(run.events, start=1)
        ]
        remote_log_text = _remote_log_summary(run, verdict)

        files = {
            "request": spool.write_json("request.json", request.model_dump(mode="json")),
            "progress": spool.write_jsonl("progress.jsonl", progress_rows),
            "generated_samples": spool.write_jsonl("generated-samples.jsonl", generated_rows),
            "transcript": spool.write_jsonl("duel-transcript.jsonl", transcript_rows),
            "scoring_results": spool.write_jsonl("scoring-results.jsonl", scoring_records),
            "judge_results": spool.write_jsonl("judge-results.jsonl", judge_rows),
            "remote_logs": spool.write_text("remote-logs.txt", remote_log_text),
        }
        uploads = self._artifact_uploader.upload_run_artifacts(
            eval_run_id=request.eval_run_id,
            artifact_prefix=request.artifact_prefix,
            files=files,
        )
        enriched = {**verdict}
        enriched["artifacts"] = {name: upload.uri for name, upload in sorted(uploads.items())}
        enriched["artifact_metadata"] = {
            name: upload.metadata() for name, upload in sorted(uploads.items())
        }
        verdict_path = spool.write_json("verdict.json", enriched)
        verdict_upload = self._artifact_uploader.upload_run_artifacts(
            eval_run_id=request.eval_run_id,
            artifact_prefix=request.artifact_prefix,
            files={"verdict": verdict_path},
        )["verdict"]
        enriched["artifacts"] = {**enriched["artifacts"], "verdict": verdict_upload.uri}
        enriched["artifact_metadata"] = {
            **enriched["artifact_metadata"],
            "verdict": verdict_upload.metadata(),
        }
        if self.settings.cleanup_local_artifacts:
            spool.cleanup()
        return enriched


# ── Helpers ───────────────────────────────────────────────────────────────────


def _smoke_progress_and_verdict(
    request: EvalRequest, *, challenger_won: bool
) -> list[dict[str, Any]]:
    # Mock-mode event stream for control-plane/idempotency tests (no GPU work).
    score_challenger = 0.58 if challenger_won else 0.42
    score_king = 1 - score_challenger
    return [
        {
            "type": "generation_started",
            "eval_run_id": str(request.eval_run_id),
            "message": "Smoke generation event emitted by control-plane test mode",
        },
        {
            "type": "scoring_started",
            "eval_run_id": str(request.eval_run_id),
            "message": "Smoke scoring event emitted by control-plane test mode",
        },
        {
            "type": "verdict",
            "eval_run_id": str(request.eval_run_id),
            "state": "succeeded",
            "challenger_won": challenger_won,
            "score_challenger": score_challenger,
            "score_king": score_king,
            "judge_count": request.scoring.judge_count,
            "allowed_scores": request.scoring.allowed_scores,
            "valid_turns": request.dataset.sample_count,
            "total_turns": request.dataset.sample_count,
            "king_vllm_errors": 0,
            "chal_vllm_errors": 0,
            "judge_errors": 0,
            "gpu_topology": {
                "accelerator": request.gpu_request.accelerator,
                "previous_king": ["0", "1", "2", "3"],
                "challenger": ["4", "5", "6", "7"],
                "tensor_parallel_size_per_model": (
                    request.gpu_request.tensor_parallel_size_per_model
                ),
            },
            "artifacts": {},
        },
    ]


def _remote_log_summary(run: EvalRun, verdict: dict[str, object]) -> str:
    # Compact plain-text run summary shipped as remote-logs.txt.
    lines = [
        f"remote_run_id={run.run_id}",
        f"eval_run_id={run.request.eval_run_id}",
        f"state={verdict.get('state')}",
        f"events={len(run.events)}",
        f"king_vllm_errors={verdict.get('king_vllm_errors', 0)}",
        f"chal_vllm_errors={verdict.get('chal_vllm_errors', 0)}",
        f"judge_errors={verdict.get('judge_errors', 0)}",
    ]
    fault_code = verdict.get("fault_code")
    if fault_code:
        lines.append(f"fault_code={fault_code}")
        lines.append(f"fault_message={verdict.get('fault_message', '')}")
    return "\n".join(lines) + "\n"


def _valid_generated_pair_count(
    samples: list[EvalSample],
    king_results: list[GenerationResult],
    challenger_results: list[GenerationResult],
) -> int:
    # Counts samples where both sides produced error-free output.
    king_by_id = {result.sample_id: result for result in king_results}
    challenger_by_id = {result.sample_id: result for result in challenger_results}
    return sum(
        1
        for sample in samples
        if sample.sample_id in king_by_id
        and sample.sample_id in challenger_by_id
        and not king_by_id[sample.sample_id].error
        and not challenger_by_id[sample.sample_id].error
    )


def _parse_gpu_ids(raw: str) -> list[str]:
    # Splits a comma-separated GPU id list and rejects duplicates.
    gpu_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU group contains duplicate IDs")
    return gpu_ids


def _chunks(items: list[T], size: int) -> list[list[T]]:
    # Splits a list into fixed-size chunks.
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def run_server() -> None:
    # Serves the eval worker API; refuses to start with an empty token unless mock mode is on.
    import uvicorn

    settings = get_settings().remote_eval
    require_startup_token(
        settings.auth_token,
        mock=settings.mock_auto_verdict or settings.mock_scored_duel,
        service="gpu-eval",
    )
    uvicorn.run("albedo.remote.eval_worker:app", host="0.0.0.0", port=settings.api_port)
