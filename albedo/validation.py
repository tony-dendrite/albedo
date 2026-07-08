"""Hippius validation worker - manifest, dtype, download, index, architecture, and dedup gates."""

from __future__ import annotations

import asyncio
import fnmatch
import functools
import hashlib
import json
import os
import re
import shutil
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

import asyncpg
import httpx
from loguru import logger

from albedo import s3
from albedo.db import heartbeat_attempt, pool, record_event
from albedo.fingerprint import compute_fingerprint, find_duplicate, health, index_fingerprint
from albedo.settings import get_settings

_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
_POLL_S = 5.0
_HEARTBEAT_S = 30.0
_FP_FILE = "fingerprint.json"
_TENSORS_FILE = "tensors.json"

# Strict file allowlist (mirrors chain.toml [files]).
_REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
)
_ALLOWED_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    ".gitattributes",
    "LICENSE",
    "README.md",
    "configuration.json",
)
_ALLOWED_GLOBS = ("model-*-of-*.safetensors", "model.safetensors")
_FORBIDDEN_GLOBS = ("*.py",)
_ALLOWED_DTYPES = frozenset({"F16", "BF16"})
_INDEX_NAME = "model.safetensors.index.json"

_NOT_FOUND_MARKERS = (
    "not found",
    "404",
    "no such",
    "does not exist",
    "nosuchkey",
    "no revision",
    "not exist",
    "norepo",
)

_VALIDATED_OR_BEYOND = (
    "HIPPIUS_VALIDATED",
    "PRE_EVAL_QUEUED",
    "PRE_EVAL_RUNNING",
    "PRE_EVAL_PASSED",
    "EVAL_QUEUED",
    "EVAL_RUNNING",
    "EVAL_WIN",
    "SET_REIGN_RUNNING",
    "REIGN_SET",
    "WEIGHT_SET_RUNNING",
    "COMPLETE_LOSS",
    "COMPLETE_CORONATED",
    "TERMINAL_INVALID",
)


@dataclass
class Outcome:
    # Result of one model validation: done, or a miner/infra fault with full evidence.
    state: str
    fault_class: str | None = None
    fault_code: str | None = None
    fault_message: str = ""
    retryable: bool = False
    result_summary: dict = field(default_factory=dict)
    fault_detail: dict = field(default_factory=dict)


def _miner(code: str, msg: str, summary: dict, fault_detail: dict | None = None) -> Outcome:
    # Terminal miner fault - never retried, published to S3 as fault.json.
    return Outcome("failed", "MINER_FAULT", code, msg, False, summary, fault_detail or {})


def _infra(code: str, msg: str) -> Outcome:
    # Retryable infra fault - requeued up to hv_max_attempts.
    return Outcome("failed", "INFRA_FAULT", code, msg, True, {})


def _is_not_found(exc: Exception) -> bool:
    # True if a Hippius error means the repo/revision simply doesn't exist (miner fault).
    return any(m in str(exc).lower() for m in _NOT_FOUND_MARKERS)


# ── Checks ────────────────────────────────────────────────────────────────────


def _matches_any(name: str, globs: tuple[str, ...]) -> bool:
    # True if name matches any of the glob patterns.
    return any(fnmatch.fnmatch(name, g) for g in globs)


def _check_repo(files: list[str]) -> tuple[bool, str]:
    # Strict manifest: required files, >=1 safetensors, allowlisted extras only, none forbidden.
    present = set(files)
    missing = [f for f in _REQUIRED_FILES if f not in present]
    if not any(f.endswith(".safetensors") for f in present):
        missing.append("*.safetensors")

    forbidden = sorted(f for f in present if _matches_any(f, _FORBIDDEN_GLOBS))
    allowed_exact = set(_REQUIRED_FILES) | set(_ALLOWED_FILES)
    extras = sorted(
        f
        for f in present
        if f not in allowed_exact
        and not _matches_any(f, _ALLOWED_GLOBS)
        and not _matches_any(f, _FORBIDDEN_GLOBS)
    )

    if missing or forbidden or extras:
        parts = []
        if missing:
            parts.append(f"missing required: {missing}")
        if forbidden:
            parts.append(f"forbidden present: {forbidden[:10]}")
        if extras:
            parts.append(f"unexpected extras: {extras[:10]}")
        return False, "; ".join(parts)
    return True, ""


def _check_dtypes(shard_dtypes: dict[str, set[str]]) -> tuple[bool, str]:
    # Every tensor must be 16-bit (F16/BF16) - rejects quantized and full-precision checkpoints.
    for name in sorted(shard_dtypes):
        bad = sorted(shard_dtypes[name] - _ALLOWED_DTYPES)
        if bad:
            return (
                False,
                f"model weights must be 16-bit (F16/BF16); shard {name} has dtype(s): {bad}",
            )
    return True, ""


def _shard_tensor_keys(path: Path) -> set[str]:
    # Tensor keys declared in a safetensors file's header (header only, no tensor data).
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(header_len))
    return {k for k in header if k != "__metadata__"}


def _check_index(model_dir: str) -> tuple[bool, str]:
    # The index's weight_map is what transformers loads: no unused/missing shards or dead tensors.
    mdir = Path(model_dir)
    actual = {p.name for p in mdir.glob("*.safetensors")}
    index_path = mdir / _INDEX_NAME

    if not index_path.exists():
        # A single monolithic checkpoint needs no index; anything sharded does.
        if actual == {"model.safetensors"}:
            return True, ""
        return False, f"sharded checkpoint missing {_INDEX_NAME}"

    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("empty or non-object weight_map")
    except Exception as exc:  # noqa: BLE001 - the index is the miner's artifact
        logger.warning(f"[validation] malformed {_INDEX_NAME}: {exc}")
        return False, f"malformed {_INDEX_NAME}: {exc}"

    referenced = set(weight_map.values())
    extra = sorted(actual - referenced)
    if extra:
        return False, (
            f"repo contains {len(extra)} safetensors not used by the model "
            f"(not referenced in {_INDEX_NAME}): {extra[:10]}"
        )
    missing = sorted(referenced - actual)
    if missing:
        return False, f"{_INDEX_NAME} references missing shard(s): {missing[:10]}"

    for shard in sorted(referenced):
        index_keys = {k for k, v in weight_map.items() if v == shard}
        try:
            header_keys = _shard_tensor_keys(mdir / shard)
        except Exception as exc:  # noqa: BLE001 - unreadable shard is the miner's fault
            logger.warning(f"[validation] could not read safetensors header of {shard}: {exc}")
            return False, f"could not read safetensors header of {shard}: {exc}"
        dead = sorted(header_keys - index_keys)
        if dead:
            return False, f"shard {shard} contains tensors not referenced by the index: {dead[:10]}"
        absent = sorted(index_keys - header_keys)
        if absent:
            return False, f"{_INDEX_NAME} maps tensors not present in shard {shard}: {absent[:10]}"
    return True, ""


@functools.lru_cache(maxsize=4)
def _load_spec(path: str) -> dict[str, Any]:
    # Loads and normalizes the architecture spec JSON (architectures/expected/forbidden_keys).
    spec = json.loads(Path(path).read_text())
    spec.setdefault("architectures", None)
    spec.setdefault("expected", {})
    spec.setdefault("forbidden_keys", [])
    return spec


def _check_architecture(model_dir: str) -> tuple[bool, str]:
    # Spec-driven architecture lock: compares config.json against architecture_spec.json.
    spec_path = get_settings().validation.arch_spec or str(
        Path(__file__).with_name("architecture_spec.json")
    )
    spec = _load_spec(spec_path)
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    cfg = json.loads(cfg_path.read_text())

    for key in spec["forbidden_keys"]:
        if key in cfg:
            return False, f"config.json must not contain {key!r}"
    if spec["architectures"] is not None and cfg.get("architectures") != spec["architectures"]:
        return False, (
            "architectures mismatch: "
            f"expected {spec['architectures']!r}, got {cfg.get('architectures')!r}"
        )
    text_cfg = cfg.get("text_config") or {}
    for key, want in spec["expected"].items():
        got = cfg[key] if key in cfg else text_cfg.get(key)
        if got != want:
            return False, f"arch key {key!r} mismatch: expected {want!r}, got {got!r}"
    return True, ""


# ── Fingerprint corpus on S3 (sync - runs inside the worker thread) ───────────


def _corpus_get(key: str) -> dict:
    # Loads a JSON dict from the bucket, or {} if absent/malformed.
    c = s3.client()
    try:
        obj = c.get_object(Bucket=get_settings().s3.bucket, Key=key)
        data = json.loads(obj["Body"].read())
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 - missing corpus file -> start empty
        logger.debug(f"[validation] s3 get({key}) failed, starting empty: {exc}")
        return {}


def _corpus_put(key: str, data: dict) -> str | None:
    # Uploads a public-read JSON document; None when the put fails.
    bucket = get_settings().s3.bucket
    try:
        s3.client().put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, default=str).encode(),
            ContentType="application/json",
            ACL="public-read",
        )
        return f"s3://{bucket}/{key}"
    except Exception as exc:  # noqa: BLE001 - never wedge validation on an upload
        logger.warning(f"[validation] s3 put({key}) failed: {exc}")
        return None


def _update_fingerprint_corpus(model_uri: str, fp: dict) -> tuple[str | None, str | None]:
    # Read-modify-writes this model into the two aggregate corpus files; no-op when S3 is off.
    if s3.client() is None:
        logger.debug(f"[validation] S3 disabled; skipping corpus update for {model_uri}")
        return None, None

    fdict = _corpus_get(_FP_FILE)
    fdict[model_uri] = {
        "method": fp.get("method"),
        "layer_keys": fp.get("layer_keys"),
        "norm_vector": fp.get("norm_vector"),
    }
    f_uri = _corpus_put(_FP_FILE, fdict)

    tdict = _corpus_get(_TENSORS_FILE)
    tdict[model_uri] = {
        "layer_keys": fp.get("layer_keys"),
        "tensor_samples": fp.get("tensor_samples"),
    }
    t_uri = _corpus_put(_TENSORS_FILE, tdict)
    logger.info(f"[validation] corpus updated: {model_uri} (now {len(fdict)} fingerprints)")
    return f_uri, t_uri


_EXPECTED_CHAT_TEMPLATE_SHA256 = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"


def _check_chat_template(config_dir: str, files: list[str]) -> tuple[bool, str]:
    # Both the standalone jinja file and tokenizer_config's embedded template must hash-match
    # the canonical genesis template (blocks template-level prompt injection / drift).
    root = Path(config_dir)
    problems: list[str] = []

    if "chat_template.jinja" not in files:
        problems.append("missing required chat_template.jinja")
    else:
        try:
            got = hashlib.sha256((root / "chat_template.jinja").read_bytes()).hexdigest()
        except OSError as exc:
            return False, f"could not read chat_template.jinja: {exc}"
        if got != _EXPECTED_CHAT_TEMPLATE_SHA256:
            problems.append(f"chat_template.jinja sha256 {got} != {_EXPECTED_CHAT_TEMPLATE_SHA256}")

    try:
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
    except OSError as exc:
        return False, f"could not read tokenizer_config.json: {exc}"
    except json.JSONDecodeError as exc:
        return False, f"invalid tokenizer_config.json: {exc}"

    template = tokenizer_config.get("chat_template")
    if not isinstance(template, str) or not template:
        problems.append("tokenizer_config.json missing string chat_template")
    else:
        got = hashlib.sha256(template.encode()).hexdigest()
        if got != _EXPECTED_CHAT_TEMPLATE_SHA256:
            problems.append(
                "tokenizer_config.json chat_template sha256 "
                f"{got} != {_EXPECTED_CHAT_TEMPLATE_SHA256}"
            )

    if problems:
        return False, "; ".join(problems)
    return True, ""


# ── Per-model check flow (blocking; runs in asyncio.to_thread) ────────────────


def _prune_model_cache(max_age_hours: float = 48.0) -> None:
    # Deletes cache entries older than the window - covers dirs leaked by terminal infra faults.
    cache = Path(get_settings().validation.model_cache_dir)
    if not cache.is_dir():
        return
    cutoff = time.time() - max_age_hours * 3600
    for entry in cache.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                logger.info(f"[validation] pruned stale model cache entry {entry.name}")
        except OSError:
            continue


def process_model(model_uri: str, hotkey: str) -> Outcome:
    # Runs the full check flow: manifest -> dtype preflight -> download -> index -> arch -> dedup.
    if get_settings().validation.mock:
        return Outcome("done", result_summary={"mock": True})
    v = get_settings().validation
    repo, _, digest = model_uri.partition("@")
    parts = repo.split("/")
    if len(parts) > 2 and "." in parts[0]:
        # model_uri carries the registry host; hippius_hub + ModelRef want namespace/name only.
        repo = "/".join(parts[1:])
    ref = ModelRef(repo=repo, digest=digest)
    _prune_model_cache()

    # 1 - file manifest
    try:
        files = list_files(ref)
    except Exception as exc:  # noqa: BLE001 - classify hub errors as miner vs infra
        if _is_not_found(exc):
            logger.exception(
                f"[validation] repo/revision not found listing files repo={repo}: {exc}"
            )
            return _miner("repo_not_found", f"repo/revision not found on Hippius: {exc}", {})
        logger.exception(f"[validation] could not list repo files repo={repo}: {exc}")
        return _infra("list_files_failed", f"could not list repo files: {exc}")
    if not files:
        return _miner("empty_repo", "Hippius repo has no files", {})
    ok, msg = _check_repo(files)
    if not ok:
        return _miner("file_manifest", msg, {"files": sorted(files)[:50]})

    # 1.5 - dtype preflight: reject non-16-bit weights from shard headers only (HTTP Range)
    try:
        shard_dtypes = safetensors_dtypes(ref)
    except Exception as exc:  # noqa: BLE001 - classify hub errors as miner vs infra
        if _is_not_found(exc):
            logger.exception(
                f"[validation] repo/revision not found reading headers repo={repo}: {exc}"
            )
            return _miner("repo_not_found", f"repo/revision not found on Hippius: {exc}", {})
        logger.exception(f"[validation] could not read safetensors headers repo={repo}: {exc}")
        return _infra("preflight_failed", f"could not read safetensors headers: {exc}")
    ok, msg = _check_dtypes(shard_dtypes)
    if not ok:
        return _miner("weight_dtype", msg, {})

    # 1.75 - small config/template download + canonical chat-template gate (pre-full-download)
    try:
        config_dir = download_config(ref)
    except Exception as exc:  # noqa: BLE001 - classify hub errors as miner vs infra
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on Hippius: {exc}", {})
        logger.exception(f"[validation] config download failed repo={repo}: {exc}")
        return _infra("download_config_failed", f"model config download failed: {exc}")
    ok, msg = _check_chat_template(config_dir, files)
    if not ok:
        return _miner("chat_template", msg, {})

    # 2 - full download
    try:
        model_dir = download_full(ref)
    except Exception as exc:  # noqa: BLE001 - classify hub errors as miner vs infra
        if _is_not_found(exc):
            logger.exception(f"[validation] repo/revision not found downloading repo={repo}: {exc}")
            return _miner("repo_not_found", f"repo/revision not found on Hippius: {exc}", {})
        logger.exception(f"[validation] model download failed repo={repo}: {exc}")
        return _infra("download_failed", f"model download failed: {exc}")
    outcome = _post_download_checks(model_uri, hotkey, repo, digest, v, model_dir)
    # Free the ~70 GB download once the verdict is final; keep it for infra retries.
    if not outcome.retryable:
        shutil.rmtree(model_dir, ignore_errors=True)
    return outcome


def _post_download_checks(
    model_uri: str, hotkey: str, repo: str, digest: str, v: Any, model_dir: str
) -> Outcome:
    # Index/arch/fingerprint/dedup checks over the downloaded snapshot.
    mdir = Path(model_dir)
    if not (mdir / "config.json").exists() or not any(mdir.glob("*.safetensors")):
        return _miner(
            "incomplete_repo", "downloaded repo is missing config.json or *.safetensors", {}
        )

    # 2.5 - safetensors must match model.safetensors.index.json (no unused shards/tensors)
    ok, msg = _check_index(model_dir)
    if not ok:
        return _miner("safetensors_index", msg, {})

    # 3 - universal, spec-driven architecture
    try:
        ok, msg = _check_architecture(model_dir)
    except FileNotFoundError as exc:
        logger.exception(
            f"[validation] config.json missing during architecture check repo={repo}: {exc}"
        )
        return _miner("architecture", f"config.json missing: {exc}", {})
    except Exception as exc:  # noqa: BLE001 - unreadable config.json is an infra retry
        logger.exception(f"[validation] could not read config.json repo={repo}: {exc}")
        return _infra("architecture_read_failed", f"could not read config.json: {exc}")
    if not ok:
        return _miner("architecture", msg, {})

    # 4 - fingerprint + dedup
    try:
        fp = compute_fingerprint(model_dir)
    except Exception as exc:  # noqa: BLE001 - fingerprint failure is retryable
        logger.exception(f"[validation] could not fingerprint model repo={repo}: {exc}")
        return _infra("fingerprint_failed", f"could not fingerprint model: {exc}")

    # A norm_vector over the lucene knn_vector cap is a non-canonical arch: reject terminally
    # instead of letting the mapping error loop as a retryable infra fault.
    dim = len(fp.get("norm_vector") or [])
    if dim > v.max_knn_dim:
        return _miner(
            "fingerprint_too_large",
            f"model fingerprint has {dim} dimensions (tensors), over the "
            f"{v.max_knn_dim} max - non-canonical architecture",
            {"fingerprint_dim": dim, "max_dim": v.max_knn_dim},
        )

    fp_uri, tensors_uri = _update_fingerprint_corpus(model_uri, fp)

    try:
        dedup = find_duplicate(fp, hotkey)
    except Exception as exc:  # noqa: BLE001 - OpenSearch blip is retryable
        logger.exception(f"[validation] dedup search failed repo={repo}: {exc}")
        return _infra("opensearch_failed", f"dedup search failed: {exc}")

    if dedup["is_duplicate"]:
        reason = (
            f"duplicate of {dedup['matched_key']}: similarity "
            f"{dedup['similarity']:.6f} >= {dedup['threshold']} threshold"
        )
        summary = {
            "reason": reason,
            "similarity": dedup["similarity"],
            "threshold": dedup["threshold"],
            "duplicate_of": dedup["matched_key"],
            "duplicate_of_hotkey": dedup["matched_hotkey"],
            "candidates_checked": dedup["candidates_checked"],
        }
        return _miner("duplicate", reason, summary, fault_detail={**summary, "fingerprint": fp})

    created_at = datetime.now(timezone.utc).isoformat()
    try:
        index_fingerprint(
            model_uri,
            fp,
            hotkey=hotkey,
            repo=repo,
            digest=digest,
            model_uri=model_uri,
            created_at=created_at,
        )
    except Exception as exc:  # noqa: BLE001 - OpenSearch blip is retryable
        logger.exception(f"[validation] could not index fingerprint repo={repo}: {exc}")
        return _infra("opensearch_index_failed", f"could not index fingerprint: {exc}")

    return Outcome(
        "done",
        result_summary={
            "similarity": dedup["similarity"],
            "threshold": dedup["threshold"],
            "fingerprint_file": fp_uri,
            "tensors_file": tensors_uri,
            "n_tensors": len(fp.get("layer_keys", [])),
        },
    )


# ── DB state machine ──────────────────────────────────────────────────────────


async def _sweep_expired() -> int:
    # Returns expired HIPPIUS attempts to HIPPIUS_RETRYABLE for crash recovery.
    p = await pool()
    async with p.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                UPDATE stage_attempts sa
                SET state = 'ABANDONED', worker_id = NULL, lease_expires_at = NULL,
                    finished_at = now(), fault_class = 'INFRA_FAULT',
                    fault_code = 'hippius_attempt_lease_expired',
                    fault_message = 'Hippius validation lease expired before completion'
                FROM model_submissions ms
                WHERE ms.id = sa.submission_id
                  AND sa.stage = 'HIPPIUS'
                  AND sa.state = 'RUNNING'
                  AND sa.lease_expires_at < now()
                  AND ms.state = 'HIPPIUS_RUNNING'
                RETURNING sa.id, sa.submission_id
                """
            )
            for row in rows:
                await conn.execute(
                    """
                    UPDATE model_submissions
                    SET state = 'HIPPIUS_RETRYABLE', fault_class = 'INFRA_FAULT',
                        fault_code = 'hippius_attempt_lease_expired',
                        fault_message = 'Hippius validation lease expired before completion',
                        retry_count = retry_count + 1, updated_at = now()
                    WHERE id = $1
                    """,
                    row["submission_id"],
                )
                await record_event(
                    conn,
                    submission_id=row["submission_id"],
                    stage_attempt_id=row["id"],
                    event_type="hippius_attempt_abandoned",
                    severity="WARN",
                    message="Hippius validation lease expired before completion",
                )
    return len(rows)


async def _claim_next(lease_seconds: int) -> dict | None:
    # Claims the oldest SUBMITTED/HIPPIUS_RETRYABLE submission (block order, SKIP LOCKED).
    p = await pool()
    async with p.acquire() as conn:
        async with conn.transaction():
            submission = await conn.fetchrow(
                """
                SELECT ms.id AS submission_id, ms.chain_commit_id, ms.hotkey,
                       ms.model_uri, ms.commit_hash, ms.retry_count, ms.priority,
                       cc.block_number, cc.payload_hash
                FROM model_submissions ms
                JOIN chain_commits cc ON cc.id = ms.chain_commit_id
                WHERE ms.state IN ('SUBMITTED', 'HIPPIUS_RETRYABLE')
                ORDER BY cc.block_number ASC, ms.priority ASC, ms.created_at ASC
                FOR UPDATE OF ms SKIP LOCKED
                LIMIT 1
                """
            )
            if submission is None:
                return None

            attempt_number = await conn.fetchval(
                """
                SELECT COALESCE(MAX(attempt_number), 0) + 1
                FROM stage_attempts WHERE submission_id = $1 AND stage = 'HIPPIUS'
                """,
                submission["submission_id"],
            )
            attempt = await conn.fetchrow(
                """
                INSERT INTO stage_attempts (
                    submission_id, stage, attempt_number, state, worker_id,
                    lease_expires_at, started_at, input_snapshot
                )
                VALUES ($1, 'HIPPIUS', $2, 'RUNNING', $3,
                        now() + ($4 * interval '1 second'), now(), $5)
                RETURNING id, attempt_number
                """,
                submission["submission_id"],
                attempt_number,
                _WORKER_ID,
                lease_seconds,
                {
                    "chain_commit_id": str(submission["chain_commit_id"]),
                    "model_uri": submission["model_uri"],
                    "hotkey": submission["hotkey"],
                    "block_number": submission["block_number"],
                },
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'HIPPIUS_RUNNING', fault_class = NULL,
                    fault_code = NULL, fault_message = NULL, updated_at = now()
                WHERE id = $1
                """,
                submission["submission_id"],
            )
            await record_event(
                conn,
                submission_id=submission["submission_id"],
                stage_attempt_id=attempt["id"],
                event_type="hippius_claimed",
                message=f"Hippius validation claimed by {_WORKER_ID}",
                data={"worker_id": _WORKER_ID},
            )
            return {
                **dict(submission),
                "id": attempt["id"],
                "attempt_number": attempt["attempt_number"],
            }


async def _attempt_submission(conn: asyncpg.Connection, attempt_id: UUID) -> asyncpg.Record:
    # Locks and returns the submission row behind a HIPPIUS attempt.
    row = await conn.fetchrow(
        """
        SELECT sa.submission_id, ms.commit_hash, ms.model_hash, ms.model_uri
        FROM stage_attempts sa
        JOIN model_submissions ms ON ms.id = sa.submission_id
        WHERE sa.id = $1 AND sa.stage = 'HIPPIUS'
        FOR UPDATE OF sa, ms
        """,
        attempt_id,
    )
    if row is None:
        raise RuntimeError(f"HIPPIUS attempt not found: {attempt_id}")
    return row


def _model_manifest_uri(model_uri: str) -> str:
    # Canonical manifest URI for the MODEL_MANIFEST artifact row.
    if model_uri.startswith(("s3://", "file://", "local-cache://", "registry.hippius.com/")):
        return model_uri
    if "@sha256:" in model_uri:
        return f"registry.hippius.com/{model_uri}"
    return model_uri


def _storage_backend_for_uri(uri: str) -> str:
    # Maps a URI scheme to the artifacts.storage_backend enum.
    if uri.startswith("s3://"):
        return "s3"
    if uri.startswith(("local-cache://", "file://")):
        return "local-cache"
    return "hippius"


async def _mark_done(attempt_id: UUID, result_summary: dict) -> None:
    # Success: MODEL_MANIFEST artifact + SUCCEEDED attempt + HIPPIUS_VALIDATED submission.
    p = await pool()
    async with p.acquire() as conn:
        async with conn.transaction():
            row = await _attempt_submission(conn, attempt_id)
            model_hash = result_summary.get("model_hash") or row["model_hash"] or row["commit_hash"]
            manifest_uri = _model_manifest_uri(row["model_uri"])
            await conn.execute(
                """
                INSERT INTO artifacts (
                    submission_id, stage_attempt_id, artifact_type, storage_backend,
                    uri, sha256, content_type
                )
                SELECT $1, $2, 'MODEL_MANIFEST', $3, $4, $5,
                       'application/vnd.oci.image.manifest.v1+json'
                WHERE NOT EXISTS (
                    SELECT 1 FROM artifacts
                    WHERE submission_id = $1 AND artifact_type = 'MODEL_MANIFEST' AND uri = $4
                )
                """,
                row["submission_id"],
                attempt_id,
                _storage_backend_for_uri(manifest_uri),
                manifest_uri,
                model_hash.removeprefix("sha256:") if model_hash else None,
            )
            result_summary = {**result_summary, "model_manifest_uri": manifest_uri}
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = 'SUCCEEDED', finished_at = now(), lease_expires_at = NULL,
                    result_summary = $2
                WHERE id = $1
                """,
                attempt_id,
                result_summary,
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'HIPPIUS_VALIDATED',
                    model_hash = CASE
                        WHEN model_hash IS NOT NULL THEN model_hash
                        WHEN EXISTS (
                            SELECT 1 FROM model_submissions m2
                            WHERE m2.model_hash = $2 AND m2.id != $1
                        ) THEN NULL
                        ELSE $2
                    END,
                    updated_at = now(), fault_class = NULL, fault_code = NULL, fault_message = NULL
                WHERE id = $1
                """,
                row["submission_id"],
                model_hash,
            )
            await record_event(
                conn,
                submission_id=row["submission_id"],
                stage_attempt_id=attempt_id,
                event_type="hippius_succeeded",
                message="Hippius validation succeeded",
                data=result_summary,
            )


async def _mark_failed(
    attempt_id: UUID, *, fault_class: str, fault_code: str, fault_message: str, result_summary: dict
) -> None:
    # Terminal fault: FAILED_TERMINAL attempt + TERMINAL_INVALID submission (+ fault artifact).
    p = await pool()
    async with p.acquire() as conn:
        async with conn.transaction():
            row = await _attempt_submission(conn, attempt_id)
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = 'FAILED_TERMINAL', finished_at = now(), lease_expires_at = NULL,
                    fault_class = $2, fault_code = $3, fault_message = $4, result_summary = $5
                WHERE id = $1
                """,
                attempt_id,
                fault_class,
                fault_code,
                fault_message,
                result_summary,
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'TERMINAL_INVALID', fault_class = $2,
                    fault_code = $3, fault_message = $4, updated_at = now(), finished_at = now()
                WHERE id = $1
                """,
                row["submission_id"],
                fault_class,
                fault_code,
                fault_message,
            )
            fault_uri = result_summary.get("fault_uri")
            if fault_uri:
                await conn.execute(
                    """
                    INSERT INTO artifacts (
                        submission_id, stage_attempt_id, artifact_type, storage_backend,
                        uri, content_type
                    )
                    VALUES ($1, $2, 'FAULT_REPORT', 's3', $3, 'application/json')
                    """,
                    row["submission_id"],
                    attempt_id,
                    fault_uri,
                )
            await record_event(
                conn,
                submission_id=row["submission_id"],
                stage_attempt_id=attempt_id,
                event_type="hippius_failed_terminal",
                severity="WARN",
                message=fault_message,
                data={"fault_class": fault_class, "fault_code": fault_code, **result_summary},
            )


async def _mark_retry(
    attempt_id: UUID, *, attempt_number: int, fault_class: str, fault_code: str, fault_message: str
) -> str:
    # Infra fault: HIPPIUS_RETRYABLE if under hv_max_attempts, else terminal infra failure.
    max_attempts = get_settings().validation.hv_max_attempts
    terminal = attempt_number >= max_attempts
    attempt_state = "FAILED_TERMINAL" if terminal else "FAILED_RETRYABLE"
    visible_state = "TERMINAL_INFRA_FAILED" if terminal else "HIPPIUS_RETRYABLE"
    if terminal:
        logger.error(
            f"[validation] terminal infra failure at retry cap {attempt_number}/{max_attempts} "
            f"fault_code={fault_code}: {fault_message}"
        )
    p = await pool()
    async with p.acquire() as conn:
        async with conn.transaction():
            row = await _attempt_submission(conn, attempt_id)
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = $2, worker_id = NULL, lease_expires_at = NULL,
                    finished_at = now(), fault_class = $3, fault_code = $4, fault_message = $5
                WHERE id = $1
                """,
                attempt_id,
                attempt_state,
                fault_class,
                fault_code,
                fault_message,
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = $2, fault_class = $3, fault_code = $4,
                    fault_message = $5, retry_count = retry_count + 1, updated_at = now(),
                    finished_at = CASE WHEN $2 = 'TERMINAL_INFRA_FAILED' THEN now()
                        ELSE finished_at END
                WHERE id = $1
                """,
                row["submission_id"],
                visible_state,
                fault_class,
                fault_code,
                fault_message,
            )
            await record_event(
                conn,
                submission_id=row["submission_id"],
                stage_attempt_id=attempt_id,
                event_type="hippius_retryable_failed",
                severity="ERROR" if terminal else "WARN",
                message=fault_message,
                data={"fault_class": fault_class, "fault_code": fault_code, "terminal": terminal},
            )
    return "failed" if terminal else "queued"


async def _hotkey_validated(hotkey: str) -> bool:
    # Has this hotkey already had a model pass Hippius validation or beyond (one model per hotkey)?
    p = await pool()
    return bool(
        await p.fetchval(
            "SELECT EXISTS(SELECT 1 FROM model_submissions"
            " WHERE hotkey = $1 AND state = ANY($2::text[]))",
            hotkey,
            list(_VALIDATED_OR_BEYOND),
        )
    )


async def _hotkey_sanity_block_reason(hotkey: str) -> str | None:
    # Returns the reason if this hotkey has a prior injection/low-vocab sanity failure; else None.
    p = await pool()
    return await p.fetchval(
        """
        SELECT sr.reason
        FROM sanity_results sr
        JOIN model_submissions ms ON ms.model_uri = sr.repo
        WHERE ms.hotkey = $1
          AND sr.passed = false
          AND (sr.reason ILIKE '%injection%' OR sr.reason ILIKE '%low vocab%')
        ORDER BY sr.checked_at DESC
        LIMIT 1
        """,
        hotkey,
    )


# ── Worker loop ───────────────────────────────────────────────────────────────


async def _hotkey_duplicate_block_reason(hotkey: str) -> str | None:
    # fault_message of this hotkey's most recent duplicate rejection, or None.
    p = await pool()
    return await p.fetchval(
        """
        SELECT fault_message FROM model_submissions
        WHERE hotkey = $1 AND state = 'TERMINAL_INVALID'
          AND fault_class = 'MINER_FAULT' AND fault_code = 'duplicate'
        ORDER BY created_at DESC LIMIT 1
        """,
        hotkey,
    )


async def _heartbeat_loop(attempt_id: UUID, lease_seconds: int) -> None:
    # Extends the attempt lease every _HEARTBEAT_S while processing runs in the thread.
    while True:
        await asyncio.sleep(_HEARTBEAT_S)
        await heartbeat_attempt(attempt_id, lease_seconds)


async def _finalize(attempt: dict, outcome: Outcome) -> None:
    # Routes the Outcome to done / retryable / terminal, publishing fault.json on miner faults.
    if outcome.state == "done":
        await _mark_done(attempt["id"], outcome.result_summary)
        logger.info(f"[validation] done - {attempt['model_uri']}")
    elif outcome.retryable:
        new_state = await _mark_retry(
            attempt["id"],
            attempt_number=attempt["attempt_number"],
            fault_class=outcome.fault_class,
            fault_code=outcome.fault_code,
            fault_message=outcome.fault_message,
        )
        logger.warning(
            f"[validation] infra fault [{outcome.fault_code}] {attempt['model_uri']} -> {new_state}"
        )
    else:
        digest = attempt["model_uri"].partition("@")[2]
        fault_doc = {
            "model_uri": attempt["model_uri"],
            "hotkey": attempt["hotkey"],
            "block_number": attempt["block_number"],
            "fault_class": outcome.fault_class,
            "fault_code": outcome.fault_code,
            "fault_message": outcome.fault_message,
            **(outcome.fault_detail or {"details": outcome.result_summary}),
        }
        key = f"hippius_validation/{attempt['hotkey']}/{digest.replace(':', '_')}/fault.json"
        fault_uri = await s3.put_json(key, fault_doc)
        await _mark_failed(
            attempt["id"],
            fault_class=outcome.fault_class,
            fault_code=outcome.fault_code,
            fault_message=outcome.fault_message,
            result_summary={**outcome.result_summary, "fault_uri": fault_uri},
        )
        logger.warning(
            f"[validation] miner fault [{outcome.fault_code}] {attempt['model_uri']} - "
            f"{outcome.fault_message}"
        )


async def run_worker() -> None:
    # The loop: sweep leases -> claim -> pre-claim guards -> process in a thread -> finalize.
    v = get_settings().validation
    if not get_settings().validation.mock and not health():
        raise RuntimeError(f"OpenSearch not healthy at {get_settings().opensearch.url}")
    logger.info(f"[validation] worker started - worker={_WORKER_ID}")

    while True:
        await _sweep_expired()
        attempt = await _claim_next(v.hv_lease_seconds)
        if attempt is None:
            await asyncio.sleep(_POLL_S)
            continue

        logger.info(
            f"[validation] claim - block={attempt['block_number']} "
            f"hotkey={attempt['hotkey'][:10]} {attempt['model_uri']}"
        )

        # A hotkey whose model failed the sanity gate for injection or low vocabulary is blocked
        # for good; any later commitment is rejected here without re-validating.
        sanity_reason = await _hotkey_sanity_block_reason(attempt["hotkey"])
        if sanity_reason is not None:
            await _mark_failed(
                attempt["id"],
                fault_class="MINER_FAULT",
                fault_code="hotkey_sanity_blocked",
                fault_message=f"hotkey blocked - prior sanity failure: {sanity_reason}",
                result_summary={"hotkey": attempt["hotkey"], "sanity_reason": sanity_reason},
            )
            logger.info(
                f"[validation] skip - sanity-blocked {attempt['hotkey'][:10]} {sanity_reason}"
            )
            continue

        # A hotkey that previously submitted a duplicate model is blocked for good;
        # any later commitment is rejected here without re-validating (upstream 77acfc3).
        dup_reason = await _hotkey_duplicate_block_reason(attempt["hotkey"])
        if dup_reason is not None:
            await _mark_failed(
                attempt["id"],
                fault_class="MINER_FAULT",
                fault_code="hotkey_duplicate_blocked",
                fault_message=(
                    f"hotkey blocked from further submissions - prior duplicate: {dup_reason}"
                ),
                result_summary={"hotkey": attempt["hotkey"], "duplicate_reason": dup_reason},
            )
            logger.info(f"[validation] skip - duplicate-blocked: {attempt['hotkey'][:10]}")
            continue

        # One passed Hippius validation per hotkey; a later commit is a miner-side duplicate.
        if await _hotkey_validated(attempt["hotkey"]):
            await _mark_failed(
                attempt["id"],
                fault_class="MINER_FAULT",
                fault_code="hotkey_already_validated",
                fault_message="hotkey already has a validated model submission",
                result_summary={"hotkey": attempt["hotkey"]},
            )
            logger.info(f"[validation] skip - hotkey already validated: {attempt['hotkey'][:10]}")
            continue

        hb = asyncio.create_task(_heartbeat_loop(attempt["id"], v.hv_lease_seconds))
        try:
            outcome = await asyncio.to_thread(
                process_model, attempt["model_uri"], attempt["hotkey"]
            )
        except Exception as exc:  # noqa: BLE001 - unexpected error is an infra retry
            logger.exception(
                f"[validation] unexpected error processing {attempt['model_uri']} "
                f"hotkey={attempt['hotkey'][:10]}: {exc}"
            )
            outcome = _infra("unexpected", f"{type(exc).__name__}: {exc}")
        finally:
            hb.cancel()
        await _finalize(attempt, outcome)


# ── Hippius OCI registry access (merged from hippius.py) ─────────────────────

# Lowercase Hippius "<namespace>/<name>" id and OCI manifest digest (the subnet is Hippius-only).
_REPO_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMEOUT = 60
_HEARTBEAT_S = 10.0


@dataclass(frozen=True)
class ModelRef:
    # Immutable pointer to a specific model snapshot (repo + digest).
    repo: str
    digest: str

    def __post_init__(self) -> None:
        # Rejects malformed repo ids and digests before any registry I/O.
        if not _REPO_RE.match(self.repo):
            raise ValueError(
                f"ModelRef.repo {self.repo!r} is not a valid lowercase '<namespace>/<name>' id"
            )
        if not _DIGEST_RE.match(self.digest):
            raise ValueError(
                f"ModelRef.digest must be a Hippius 'sha256:<hex64>'; got {self.digest!r}"
            )

    @property
    def immutable_ref(self) -> str:
        # Stable string identifier: repo@digest.
        return f"{self.repo}@{self.digest}"


def _token() -> str | None:
    # Hippius hub bearer token from settings; None means anonymous.
    return get_settings().hippius_hub_token or None


def _cache_dir(ref: ModelRef) -> Path:
    # Per-(repo, digest) cache dir, guarded against path traversal via crafted repo names.
    root = Path(get_settings().validation.model_cache_dir).resolve()
    resolved = (root / ref.repo / ref.digest.replace(":", "_")).resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise ValueError(f"ModelRef.repo {ref.repo!r} resolves outside cache root - blocked")
    return resolved


@contextmanager
def _download_heartbeat(label: str) -> Iterator[None]:
    # Logs every _HEARTBEAT_S seconds that label is still downloading (snapshot_download is silent).
    stop = threading.Event()
    start = time.monotonic()

    def _beat() -> None:
        # Emits the periodic still-downloading line until the download returns.
        while not stop.wait(_HEARTBEAT_S):
            logger.info(
                f"[hippius] still downloading {label} ({time.monotonic() - start:.0f}s elapsed)"
            )

    thread = threading.Thread(target=_beat, name="hippius-dl-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def list_files(ref: ModelRef) -> list[str]:
    # Filenames present in the Hippius repo at the pinned digest.
    import hippius_hub

    return hippius_hub.list_repo_files(ref.repo, revision=ref.digest, token=_token())


def download_config(ref: ModelRef) -> str:
    # Downloads only the small JSON/template files (pre-full-download gates); returns local path.
    import hippius_hub

    dest = _cache_dir(ref)
    dest.mkdir(parents=True, exist_ok=True)
    with _download_heartbeat(f"{ref.immutable_ref} (config only)"):
        hippius_hub.snapshot_download(
            ref.repo,
            revision=ref.digest,
            local_dir=str(dest),
            max_workers=8,
            allow_patterns=["*.json", "chat_template.jinja"],
            token=_token(),
        )
    return str(dest)


def download_full(ref: ModelRef) -> str:
    # Downloads the full model snapshot into the guarded cache dir; returns the local path.
    import hippius_hub

    dest = _cache_dir(ref)
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"[hippius] downloading {ref.immutable_ref} -> {dest}")
    with _download_heartbeat(ref.immutable_ref):
        hippius_hub.snapshot_download(
            ref.repo, revision=ref.digest, local_dir=str(dest), max_workers=8, token=_token()
        )
    return str(dest)


# ── Pre-download dtype preflight (safetensors headers via HTTP Range) ─────────


def _oci_context(ref: ModelRef) -> tuple[str, str, dict[str, str], dict]:
    # (registry, oci_repo, auth_headers, manifest) - bearer handshake + manifest fetch in one place.
    from hippius_hub._oci import fetch_manifest
    from hippius_hub.auth import get_oci_bearer_token
    from hippius_hub.constants import resolve_registry
    from hippius_hub.file_download import _oci_repo_path

    registry = resolve_registry(None)
    oci_repo = _oci_repo_path(ref.repo, None)
    oci_token = get_oci_bearer_token(oci_repo, _token(), endpoint=None)
    manifest = fetch_manifest(registry, oci_repo, ref.digest, oci_token).manifest
    return registry, oci_repo, {"Authorization": f"Bearer {oci_token}"}, manifest


def _ranged(client: httpx.Client, url: str, headers: dict[str, str], start: int, end: int) -> bytes:
    # GETs bytes [start, end]; insists on 206 so a Range-ignoring registry can't stream GBs.
    rng = {**headers, "Range": f"bytes={start}-{end}"}
    with client.stream("GET", url, headers=rng) as resp:
        if resp.status_code != 206:
            raise RuntimeError(
                f"registry did not honor Range (status {resp.status_code}) for {url}"
            )
        return resp.read()


def _read_header(client: httpx.Client, url: str, headers: dict[str, str]) -> dict:
    # Reads a safetensors header: 8-byte little-endian length, then that many JSON bytes.
    hlen = int.from_bytes(_ranged(client, url, headers, 0, 7), "little")
    return json.loads(_ranged(client, url, headers, 8, 8 + hlen - 1))


def safetensors_dtypes(ref: ModelRef) -> dict[str, set[str]]:
    # {shard_filename: {dtypes}} for every *.safetensors layer, reading headers only.
    from hippius_hub._oci import iter_titled_layers

    registry, oci_repo, auth, manifest = _oci_context(ref)
    out: dict[str, set[str]] = {}
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        for title, layer in iter_titled_layers(manifest):
            if not title.endswith(".safetensors"):
                continue
            blob_url = f"{registry}/v2/{oci_repo}/blobs/{layer['digest']}"
            header = _read_header(client, blob_url, auth)
            out[title] = {info["dtype"] for k, info in header.items() if k != "__metadata__"}
    return out
