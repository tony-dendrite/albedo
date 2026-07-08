"""Stateless sanity GPU worker - downloads a challenger, generates via vLLM, runs heuristics."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from albedo.remote.common import (
    GENESIS_PREPROCESSOR_CONFIG,
    GENESIS_VIDEO_PREPROCESSOR_CONFIG,
    Run,
    RunStore,
    WorkerBusy,
    bearer_auth,
    canonical_max_model_len,
    download_heartbeat,
    ready_payload,
    require_startup_token,
)
from albedo.sampling import format_messages
from albedo.sanity_gate import (
    check_code_present,
    check_collapsed,
    check_one,
    check_uniform_length,
)
from albedo.settings import SanityRemoteSettings, get_settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FORBIDDEN_CONFIG_KEYS = frozenset({"auto_map", "quantization_config"})
_QWEN3_IM_END_TOKEN_ID = 248046
# Generation/engine knobs the simplified settings no longer expose (source defaults).
_GEN_TEMPERATURE = 0.7
_GEN_TOP_P = 0.8
_GEN_TOP_K = 20
_GEN_MIN_P = 0.0
_GEN_READ_TIMEOUT_S = 900.0
_VLLM_LIMIT_MM = '{"image": 0, "video": 0}'
_PROCESSOR_FILES = {
    "preprocessor_config.json": GENESIS_PREPROCESSOR_CONFIG,
    "video_preprocessor_config.json": GENESIS_VIDEO_PREPROCESSOR_CONFIG,
}

app = FastAPI(title="Albedo Sanity Remote Worker", version="0.1.0")
store = RunStore()
require_auth = bearer_auth(lambda: get_settings().sanity_remote.auth_token)
_AUTH = Depends(require_auth)


class SanityRunRequest(BaseModel):
    # A pre-eval generation job: the backend supplies the model and the pre-sampled prompts.
    run_id: str
    model_uri: str
    digest: str
    prompts: list[str]
    prompt_messages: list[list[dict[str, str]]] | None = None
    gen_max_tokens: int = 32768
    min_tokens: int = 5
    max_repetition: float = 0.85
    min_vocab_ratio: float = 0.05


@dataclass
class SanityRun(Run):
    # A sanity run: adds the terminal result events the dispatcher polls for.
    def succeed(self, *, responses: list[str], heuristics: list[dict[str, Any]]) -> None:
        # Emits the terminal success result carrying responses + heuristic verdicts.
        self.append_event(
            {
                "type": "result",
                "run_id": self.run_id,
                "state": "succeeded",
                "responses": responses,
                "heuristics": heuristics,
            }
        )
        self.set_state("succeeded")

    def fail(self, *, fault_code: str, fault_message: str, retryable: bool = True) -> None:
        # Emits a terminal failure result (retryable=infra, else a miner/model fault).
        self.append_event(
            {
                "type": "result",
                "run_id": self.run_id,
                "state": "failed",
                "fault_code": fault_code,
                "fault_message": fault_message,
                "retryable": retryable,
            }
        )
        self.set_state("failed")

    def as_status(self) -> dict[str, Any]:
        # Returns the final result if done, else a lightweight status snapshot.
        result = self.final_event("result")
        if result:
            return result
        return {
            "run_id": self.run_id,
            "digest": self.request.digest,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# ── HTTP API (wire-identical to the source sanity_remote/api.py) ──────────────


@app.get("/health")
def health() -> dict[str, str]:
    # Liveness probe (unauthenticated).
    return {"status": "ok"}


@app.get("/ready", dependencies=[_AUTH])
def ready() -> dict[str, object]:
    # Readiness + host identity + busy state for the dispatcher's host selection.
    settings = get_settings().sanity_remote
    return ready_payload(
        host_id=settings.host_id, role="PRE_EVAL", active_runs=len(store.list_active())
    )


@app.post("/sanity-runs", dependencies=[_AUTH])
async def start_run(request: SanityRunRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    # Accepts a generation job (idempotent on run_id, 409 while busy) and runs it in the background.
    settings = get_settings().sanity_remote

    def _factory() -> SanityRun:
        # Builds the accepted run with its acceptance event.
        run = SanityRun(run_id=request.run_id, request=request, state="accepted")
        run.append_event({"type": "accepted", "run_id": request.run_id, "digest": request.digest})
        return run

    try:
        run = store.get_or_create(request.run_id, _factory)
    except WorkerBusy as busy:
        raise HTTPException(
            status_code=409, detail=f"sanity worker busy: {busy.active_count} active run(s)"
        ) from busy
    queued = store.mark_worker_started(run.run_id)
    if queued is not None:
        logger.info(f"[sanity-remote] queuing run={run.run_id} digest={request.digest[:16]}")
        background_tasks.add_task(generate, queued, settings)
    else:
        logger.info(f"[sanity-remote] duplicate run_id={run.run_id} state={run.state}")
    return {"run_id": run.run_id, "state": run.state}


@app.get("/sanity-runs/{run_id}", dependencies=[_AUTH])
def get_run(run_id: str) -> dict[str, object]:
    # Status snapshot (or the final result once done).
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run.as_status()


@app.get("/sanity-runs/{run_id}/events", dependencies=[_AUTH])
def get_run_events(run_id: str) -> dict[str, list[dict[str, object]]]:
    # Full event list for the dispatcher to poll until a result appears.
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return {"events": run.events}


# ── Model materialization ─────────────────────────────────────────────────────


class WorkerFault(Exception):
    # Carries a fault code + retryability for the run's failure event.
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _strip_thinking(text: str) -> str:
    # Removes <think>...</think> so heuristics judge the answer; "" when thinking never closed.
    if "<think>" not in text:
        return text
    if "</think>" not in text:
        return ""
    return _THINK_RE.sub("", text).strip()


def _strip_model_config(model_dir: str) -> None:
    # Drops keys that can redirect loading or force quantization (top level + nested text_config).
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"[sanity-remote] could not read config.json: {exc}")
        return
    stripped = {k: v for k, v in config.items() if k not in _FORBIDDEN_CONFIG_KEYS}
    removed = set(config) - set(stripped)
    if isinstance(stripped.get("text_config"), dict):
        clean_tc = {
            k: v for k, v in stripped["text_config"].items() if k not in _FORBIDDEN_CONFIG_KEYS
        }
        removed |= {f"text_config.{k}" for k in set(stripped["text_config"]) - set(clean_tc)}
        stripped["text_config"] = clean_tc
    if not removed:
        return
    config_path.write_text(json.dumps(stripped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(f"[sanity-remote] stripped forbidden keys from config.json: {removed}")


def _inject_processor_configs(model_dir: str) -> None:
    # Writes the canonical genesis image/video processor configs when a miner repo omits them
    # (Qwen3.6 declares a vision config, so vLLM refuses to boot without an image processor).
    for name, payload in _PROCESSOR_FILES.items():
        path = Path(model_dir) / name
        if path.exists():
            continue
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info(f"[sanity-remote] injected canonical {name} from genesis pin")


def _format_prompt_messages(
    tokenizer_path: str, prompt_messages: list[list[dict[str, str]]]
) -> list[str]:
    # Renders each message list through the loaded model's own chat template.
    return [
        format_messages(messages, tokenizer_path=tokenizer_path, enable_thinking=False)
        for messages in prompt_messages
    ]


def _model_ref_parts(model_uri: str, digest: str) -> tuple[str, str]:
    # Splits "[registry/]namespace/name[@digest]"; hippius_hub wants the repo WITHOUT the
    # registry host (else the manifest URL doubles the host and the registry answers 401).
    repo, sep, uri_digest = model_uri.partition("@")
    parts = repo.split("/")
    if len(parts) > 2 and "." in parts[0]:
        repo = "/".join(parts[1:])
    return repo, uri_digest if sep else digest


def _model_present(model_dir: str) -> bool:
    # A reusable on-disk copy: dir exists with config.json + a safetensors shard.
    p = Path(model_dir)
    if not p.is_dir():
        return False
    return (p / "config.json").exists() and any(p.glob("*.safetensors"))


def _hippius_cache_dir(cache_root: str, repo: str, digest: str) -> Path:
    # Per-(repo, digest) cache dir, guarded against path traversal via crafted repo names.
    safe_digest = digest.replace(":", "_")
    root = Path(cache_root).resolve()
    resolved = (root / repo / safe_digest).resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise ValueError(f"model repo {repo!r} resolves outside cache root - blocked")
    return resolved


def _download_model(repo: str, digest: str, cache_root: str) -> str:
    # Downloads the full Hippius snapshot into the cache dir (blocking; run in a thread).
    import hippius_hub

    dest = _hippius_cache_dir(cache_root, repo, digest)
    dest.mkdir(parents=True, exist_ok=True)
    with download_heartbeat(f"{repo}@{digest[:16]}"):
        hippius_hub.snapshot_download(
            repo,
            revision=digest,
            local_dir=str(dest),
            max_workers=8,
            token=get_settings().hippius_hub_token or None,
        )
    return str(dest)


# ── vLLM engine ───────────────────────────────────────────────────────────────


class VllmEngine:
    # One warm vLLM subprocess; swaps the model only when the digest changes.
    def __init__(self, settings: SanityRemoteSettings) -> None:
        # Captures settings, clears any port squatter left by a previous worker process.
        self._s = settings
        self._proc: subprocess.Popen | None = None
        self._loaded_digest = ""
        self._loaded_dir = ""
        self._lock = asyncio.Lock()
        self._kill_stale_vllm()
        self._kill_port_squatter()

    async def run_job(
        self,
        model_uri: str,
        digest: str,
        prompts: list[str],
        max_tokens: int,
        prompt_messages: list[list[dict[str, str]]] | None = None,
    ) -> list[str]:
        # Serializes one generation job: ensure the model is loaded, then generate the prompts.
        n = len(prompt_messages) if prompt_messages is not None else len(prompts)
        logger.info(
            f"[sanity-remote] run_job digest={digest[:16]} prompts={n} max_tokens={max_tokens}"
        )
        async with self._lock:
            await self._ensure_model(model_uri, digest)
            if prompt_messages is not None:
                prompts = await asyncio.to_thread(
                    _format_prompt_messages, self._loaded_dir, prompt_messages
                )
            return await self._run_prompts(digest, prompts, max_tokens)

    async def teardown(self) -> None:
        # Frees the GPUs after a run by killing vLLM and forcing a cold load next time.
        async with self._lock:
            await self._kill_vllm()
            self._loaded_digest = ""

    def forget(self) -> None:
        # Forces a reload next time; keeps _loaded_dir so a stale model stays reclaimable.
        self._loaded_digest = ""

    def _kill_stale_vllm(self) -> None:
        # This worker owns the box's GPU: kill ANY leftover vLLM server before starting a new
        # one (a booting vLLM has not bound its port yet, so the port check alone misses it -
        # two concurrent loads wedge the GPU and starve the API event loop).
        result = subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"], capture_output=True
        )
        if result.returncode == 0:
            logger.warning("[sanity-remote] killed stale vLLM server process(es) before boot")
            time.sleep(3.0)

    def _kill_port_squatter(self) -> None:
        # On startup, kill any orphaned vLLM still holding the configured port (else the new
        # digest would be marked loaded while the old server still serves the previous model).
        import socket

        try:
            with socket.socket() as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", self._s.vllm_port)) != 0:
                    return  # port is free, nothing to kill
        except Exception as exc:  # noqa: BLE001 - port probe is best-effort
            logger.debug(f"[sanity-remote] port squatter socket check failed: {exc}")
            return
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{self._s.vllm_port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for pid_str in result.stdout.split():
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                    logger.info(
                        f"[sanity-remote] killed orphan vLLM pid={pid_str} "
                        f"on port {self._s.vllm_port}"
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort kill of orphan pid
                    logger.debug(f"[sanity-remote] failed to kill orphan pid={pid_str}: {exc}")
        except Exception as exc:  # noqa: BLE001 - best-effort lsof + pid parse
            logger.debug(f"[sanity-remote] lsof/pid parse for port squatter failed: {exc}")

    async def _ensure_model(self, model_uri: str, digest: str) -> None:
        # Reuses a healthy warm model, otherwise downloads + swaps vLLM to the new digest.
        if digest == self._loaded_digest and await self._healthy():
            logger.info(f"[sanity-remote] reusing warm model {digest[:16]}")
            return
        logger.info(f"[sanity-remote] cold load - digest={digest[:16]} uri={model_uri}")
        try:
            model_dir = await asyncio.wait_for(
                self._materialize(model_uri, digest), timeout=self._s.download_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise WorkerFault(
                "download_timeout", f"download exceeded {self._s.download_timeout_s}s"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - download failures are retryable infra by default
            raise WorkerFault("download_failed", f"model download failed: {exc}") from exc

        await self._kill_vllm()
        old_dir = self._loaded_dir
        self._loaded_digest = ""
        self._loaded_dir = model_dir
        await asyncio.to_thread(_strip_model_config, model_dir)
        try:
            await self._start_vllm(model_dir, digest)
        except Exception as exc:  # noqa: BLE001 - boot failures are retryable infra
            raise WorkerFault("vllm_boot_failed", f"vLLM did not start: {exc}") from exc
        self._loaded_digest = digest
        if old_dir and old_dir != model_dir:
            await asyncio.to_thread(shutil.rmtree, old_dir, True)

    async def _materialize(self, model_uri: str, digest: str) -> str:
        # Reuses an already-downloaded copy if present; otherwise downloads from Hippius.
        repo, ref_digest = _model_ref_parts(model_uri, digest)
        dest = str(_hippius_cache_dir(self._s.model_cache_dir, repo, ref_digest))
        if _model_present(dest):
            logger.info(f"[sanity-remote] reusing on-disk model at {dest} - skipping download")
        else:
            logger.info(f"[sanity-remote] downloading {repo} digest={ref_digest[:16]} to {dest}")
            dest = await asyncio.to_thread(
                _download_model, repo, ref_digest, self._s.model_cache_dir
            )
        await asyncio.to_thread(_inject_processor_configs, dest)
        return dest

    async def _start_vllm(self, model_dir: str, model_name: str) -> None:
        # Launches a vLLM subprocess (no --trust-remote-code) and waits until it reports healthy.
        logger.info(
            f"[sanity-remote] starting vLLM port={self._s.vllm_port} model={model_name[:40]}"
        )
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_dir,
            "--served-model-name",
            model_name,
            "--port",
            str(self._s.vllm_port),
            "--gpu-memory-utilization",
            str(self._s.gpu_util),
            "--dtype",
            self._s.vllm_dtype,
            "--max-model-len",
            str(self._s.max_model_len or canonical_max_model_len()),
            "--generation-config",
            "vllm",
            "--limit-mm-per-prompt",
            _VLLM_LIMIT_MM,
            # The Qwen3.x MoE hybrids use gated-delta-net cache blocks; CUDA-graph capture
            # fails when max_num_seqs exceeds available Mamba blocks, so stay eager.
            "--enforce-eager",
        ]
        tensor_parallel = len([g for g in self._s.gpu_ids.split(",") if g.strip()])
        if tensor_parallel > 1:
            cmd += ["--tensor-parallel-size", str(tensor_parallel)]
        self._proc = subprocess.Popen(
            cmd,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": self._s.gpu_ids,
                # Disables the FlashInfer sampler; required on A6000 (CUB version mismatch),
                # harmless on 5090/B200 (falls back to PyTorch-native sampler).
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await self._wait_healthy(self._s.vllm_startup_s)
        logger.info(f"[sanity-remote] vLLM healthy on {self._s.vllm_port} model={model_name[:40]}")

    async def _healthy(self) -> bool:
        # True only if the process is alive AND its health endpoint responds 200.
        if self._proc is None or self._proc.poll() is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                return (
                    await c.get(f"http://localhost:{self._s.vllm_port}/health")
                ).status_code == 200
        except Exception as exc:  # noqa: BLE001 - any probe failure means not healthy
            logger.debug(f"[sanity-remote] health probe failed: {exc}")
            return False

    async def _wait_healthy(self, timeout: float) -> None:
        # Polls the vLLM health endpoint until 200 or the timeout expires.
        url = f"http://localhost:{self._s.vllm_port}/health"
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5.0) as c:
            while time.monotonic() < deadline:
                try:
                    if (await c.get(url)).status_code == 200:
                        return
                except Exception as exc:  # noqa: BLE001 - keep polling until the deadline
                    logger.debug(f"[sanity-remote] health poll retry: {exc}")
                await asyncio.sleep(2.0)
        raise RuntimeError(f"vLLM did not become healthy within {timeout}s")

    async def _run_prompts(self, model_name: str, prompts: list[str], max_tokens: int) -> list[str]:
        # Per prompt: HTTP-error/malformed -> "" (model fault); transport error -> raise (infra).
        # Prompts arrive pre-formatted with the Qwen chat template, so send raw completions.
        url = f"http://localhost:{self._s.vllm_port}/v1/completions"
        timeout = httpx.Timeout(connect=5.0, read=_GEN_READ_TIMEOUT_S, write=5.0, pool=5.0)

        async def _one(prompt: str, budget: int = max_tokens, retried: bool = False) -> str:
            # Sends one completion request and strips CoT thinking from the answer.
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(
                    url,
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "max_tokens": budget,
                        "temperature": _GEN_TEMPERATURE,
                        "top_p": _GEN_TOP_P,
                        "top_k": _GEN_TOP_K,
                        "min_p": _GEN_MIN_P,
                        "stop_token_ids": [_QWEN3_IM_END_TOKEN_ID],
                    },
                )
            if r.status_code == 400 and not retried:
                # Context-length overflow: vLLM's message carries the real numbers; retry once
                # with the remaining budget instead of failing the model for our own sizing.
                nums = [int(n) for n in re.findall(r"\d+", r.text)]
                if len(nums) >= 2:
                    ctx, prompt_tokens = nums[0], nums[2] if len(nums) > 2 else nums[1]
                    allowed = ctx - prompt_tokens - 32
                    if allowed > 64:
                        logger.warning(
                            f"[sanity-remote] context overflow (ctx={ctx} prompt={prompt_tokens}); "
                            f"retrying with max_tokens={allowed}"
                        )
                        return await _one(prompt, budget=allowed, retried=True)
                logger.warning(f"[sanity-remote] vLLM 400 not recoverable: {r.text[:160]}")
                return ""
            if r.status_code >= 400:
                logger.warning(
                    f"[sanity-remote] vLLM HTTP {r.status_code} for prompt - model fault"
                )
                return ""
            try:
                choice = r.json()["choices"][0]
                raw = choice["text"] or ""
                finish = choice.get("finish_reason", "unknown")
                answer = _strip_thinking(raw) or raw
                logger.info(
                    f"[sanity-remote] prompt finish={finish} thinking={'<think>' in raw} "
                    f"answer_words={len(answer.split())}"
                )
                return answer
            except (KeyError, IndexError, ValueError):
                logger.warning("[sanity-remote] malformed vLLM response body - model fault")
                return ""

        results = await asyncio.gather(*[_one(p) for p in prompts], return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                raise WorkerFault(
                    "generation_transport_error", f"vLLM request failed: {res}"
                ) from res
        return list(results)

    async def _kill_vllm(self) -> None:
        # Kills the vLLM process group; retries once if it doesn't exit within 5 seconds.
        if not self._proc:
            return
        logger.info(f"[sanity-remote] killing vLLM pid={self._proc.pid}")
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        except Exception as exc:  # noqa: BLE001 - process may already be gone
            logger.debug(f"[sanity-remote] killpg failed (process may be gone): {exc}")
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("[sanity-remote] vLLM did not exit after SIGKILL - retrying")
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception as exc:  # noqa: BLE001 - best-effort kill
                logger.debug(f"[sanity-remote] retry kill failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - best-effort reap
            logger.debug(f"[sanity-remote] reap failed: {exc}")
        self._proc = None


_ENGINE: VllmEngine | None = None


def _engine() -> VllmEngine:
    # Lazily builds the process-wide vLLM engine singleton.
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VllmEngine(get_settings().sanity_remote)
    return _ENGINE


def _heuristics(responses: list[str], req: SanityRunRequest) -> list[dict[str, Any]]:
    # Per-response heuristic verdicts; a set-level collapse signal fails all responses.
    out: list[dict[str, Any]] = []
    for i, resp in enumerate(responses):
        r = check_one(
            resp,
            min_tokens=req.min_tokens,
            max_repetition=req.max_repetition,
            min_vocab_ratio=req.min_vocab_ratio,
        )
        if not r.passed:
            logger.warning(
                f"[sanity-remote] per-response heuristic failed i={i} reason={r.reason!r}"
            )
        out.append({"passed": r.passed, "reason": r.reason})
    if any(not item["passed"] for item in out):
        return out

    set_fail = next(
        (
            c
            for c in (
                check_collapsed(responses),
                check_uniform_length(responses),
                check_code_present(responses),
            )
            if not c.passed
        ),
        None,
    )
    if set_fail is not None:
        logger.warning(f"[sanity-remote] set-level heuristic failed reason={set_fail.reason!r}")
        return [{"passed": False, "reason": set_fail.reason} for _ in responses]
    logger.info(f"[sanity-remote] all heuristics passed responses={len(responses)}")
    return out


async def generate(run: SanityRun, settings: SanityRemoteSettings) -> None:
    # Executes one run: ensure model -> generate -> heuristics -> emit result (or a fault).
    req: SanityRunRequest = run.request
    logger.info(
        f"[sanity-remote] generate start run={run.run_id} digest={req.digest[:16]} "
        f"uri={req.model_uri}"
    )
    if settings.mock_auto_result:
        run.succeed(
            responses=[f"mock response to: {p[:30]}" for p in req.prompts],
            heuristics=[{"passed": True, "reason": "mock"} for _ in req.prompts],
        )
        return

    engine = _engine()
    try:
        run.append_event({"type": "generation_started", "run_id": run.run_id})
        responses = await engine.run_job(
            req.model_uri,
            req.digest,
            req.prompts,
            req.gen_max_tokens,
            req.prompt_messages,
        )
        heuristics = _heuristics(responses, req)
        all_passed = all(h["passed"] for h in heuristics)
        logger.info(
            f"[sanity-remote] generate done run={run.run_id} responses={len(responses)} "
            f"heuristics_passed={all_passed}"
        )
        run.succeed(responses=responses, heuristics=heuristics)
    except WorkerFault as fault:
        engine.forget()
        logger.warning(
            f"[sanity-remote] worker fault code={fault.code} digest={req.digest[:16]} "
            f"retryable={fault.retryable}: {fault}"
        )
        run.fail(fault_code=fault.code, fault_message=str(fault), retryable=fault.retryable)
    except Exception as exc:  # noqa: BLE001 - never let a worker crash strand the run
        engine.forget()
        logger.exception(f"[sanity-remote] generation failed for {req.digest}")
        run.fail(fault_code="worker_error", fault_message=str(exc), retryable=True)
    finally:
        # Tear down vLLM so the GPUs free up between preevals (eval-server parity);
        # the next run cold-loads. Best-effort: never let teardown strand the run.
        try:
            await engine.teardown()
        except Exception:  # noqa: BLE001
            logger.exception("[sanity-remote] vLLM teardown after run failed (best-effort)")


def run_server() -> None:
    # Serves the sanity worker API; refuses to start with an empty token unless mock mode is on.
    import uvicorn

    settings = get_settings().sanity_remote
    require_startup_token(settings.auth_token, mock=settings.mock_auto_result, service="gpu-sanity")
    uvicorn.run(
        "albedo.remote.sanity_worker:app",
        host="0.0.0.0",
        port=settings.api_port,
        log_level="info",
    )
