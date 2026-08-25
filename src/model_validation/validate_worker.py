from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path

import asyncpg
from loguru import logger as log

from albedo_config import get_model_validation_settings
from model_validation import db, dedup
from model_validation.opensearch import health
from model_validation.storage import (
    download_config,
    download_full,
    list_files,
    make_ref,
    safetensors_dtypes,
)
from model_validation.uploads import put_fault
from model_validation.validate import (
    check_architecture,
    check_dtypes,
    check_genesis,
    check_index,
    check_repo,
)
from model_validation.validate.chat_template import check as check_chat_template

config = get_model_validation_settings()

_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


@dataclass
class Outcome:
    state: str
    fault_class: str | None = None
    fault_code: str | None = None
    fault_message: str = ""
    retryable: bool = False
    result_summary: dict = field(default_factory=dict)
    fault_detail: dict = field(default_factory=dict)


def _miner(code: str, msg: str, summary: dict, fault_detail: dict | None = None) -> Outcome:
    return Outcome("failed", "MINER_FAULT", code, msg, False, summary, fault_detail or {})


def _infra(code: str, msg: str) -> Outcome:
    return Outcome("failed", "INFRA_FAULT", code, msg, True, {})


def _ban_suffix(fails: int, max_fails: int) -> str:
    left = max(0, max_fails - fails)
    if left > 0:
        return f" — hotkey has {left} attempt(s) left before ban"
    return " — hotkey has 0 attempts left and is now banned from further submissions"


_NOT_FOUND_MARKERS = (
    "not found",
    "404",
    "no such",
    "does not exist",
    "nosuchkey",
    "no revision",
    "not exist",
    "norepo",
    "gated",
    "restricted",
)


def _is_not_found(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _NOT_FOUND_MARKERS)


def process_model(model_uri: str, hotkey: str, coldkey: str = "") -> Outcome:
    repo, _, digest = model_uri.partition("@")
    try:
        ref = make_ref(repo, digest)
    except ValueError as exc:
        return _miner("invalid_ref", f"malformed on-chain model reference: {exc}", {})

    try:
        files = list_files(ref)
    except Exception as exc:
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on {ref.backend}: {exc}", {})
        return _infra("list_files_failed", f"could not list repo files: {exc}")
    if not files:
        return _miner("empty_repo", "model repo has no files", {})
    ok, msg = check_repo(files)
    if not ok:
        return _miner("file_manifest", msg, {"files": sorted(files)[:50]})

    try:
        shard_dtypes = safetensors_dtypes(ref)
    except Exception as exc:
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on {ref.backend}: {exc}", {})
        return _infra("preflight_failed", f"could not read safetensors headers: {exc}")
    ok, msg = check_dtypes(shard_dtypes)
    if not ok:
        return _miner("weight_dtype", msg, {})

    try:
        config_dir = download_config(ref)
    except Exception as exc:
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on {ref.backend}: {exc}", {})
        return _infra("download_config_failed", f"model config download failed: {exc}")
    ok, msg = check_chat_template(config_dir, files)
    if not ok:
        return _miner("chat_template_hash", msg, {})

    ok, msg = check_genesis(config_dir, files)
    if not ok:
        return _miner("metadata_hash", msg, {})

    try:
        ok, msg = check_architecture(config_dir)
    except FileNotFoundError as exc:
        return _miner("architecture", f"config.json missing: {exc}", {})
    except Exception as exc:
        return _infra("architecture_read_failed", f"could not read config.json: {exc}")
    if not ok:
        return _miner("architecture", msg, {})

    try:
        model_dir = download_full(ref)
    except Exception as exc:
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on {ref.backend}: {exc}", {})
        return _infra("download_failed", f"model download failed: {exc}")
    mdir = Path(model_dir)
    if not any(mdir.glob("*.safetensors")):
        return _miner("incomplete_repo", "downloaded repo is missing *.safetensors", {})

    ok, msg = check_index(model_dir, files)
    if not ok:
        return _miner("safetensors_index", msg, {})

    try:
        res = dedup.run(model_dir, model_uri, hotkey, repo, digest, coldkey)
    except Exception as exc:
        return _infra("dedup_failed", f"dedup stage failed: {type(exc).__name__}: {exc}")
    if res.infra_error:
        return _infra("dedup_failed", res.infra_error)

    summary = dedup.public_summary(res)
    if res.rejected:
        if config.DEDUP_ENFORCE:
            code = "duplicate_own" if res.verdict.reason == "OWN-COPY" else "duplicate"
            return _miner(code, dedup.public_message(res), summary, fault_detail=summary)
        log.warning("[shadow] dedup REJECT {} — {}", model_uri, dedup.public_message(res))
        summary = {"dedup": "pass"}

    return Outcome("done", result_summary=summary)


async def _heartbeat_loop(pool, attempt_id) -> None:
    while True:
        await asyncio.sleep(config.HEARTBEAT_S)
        await db.heartbeat(pool, attempt_id, config.LEASE_SECONDS)


async def _finalize(pool, attempt, outcome: Outcome) -> None:
    if outcome.state == "done":
        try:
            await db.mark_done(pool, attempt["id"], outcome.result_summary)
        except asyncpg.UniqueViolationError as exc:
            await db.mark_failed(
                pool,
                attempt["id"],
                fault_class="MINER_FAULT",
                fault_code="duplicate",
                fault_message=f"model_hash already belongs to another submission: {exc}",
                result_summary=outcome.result_summary,
            )
            log.warning("duplicate model_hash on mark_done — {}", attempt["model_uri"])
            return
        log.info("done — {}", attempt["model_uri"])
    elif outcome.retryable:
        new_state = await db.mark_retry(
            pool,
            attempt["id"],
            attempt_number=attempt["attempt_number"],
            max_attempts=config.MAX_ATTEMPTS,
            fault_class=outcome.fault_class,
            fault_code=outcome.fault_code,
            fault_message=outcome.fault_message,
        )
        log.warning("infra fault [{}] {} → {}", outcome.fault_code, attempt["model_uri"], new_state)
    else:
        if outcome.fault_code != "duplicate":
            fails = await db.hotkey_preeval_fail_count(pool, attempt["hotkey"]) + 1
            outcome.fault_message += _ban_suffix(fails, config.PREEVAL_MAX_FAILS)
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
        fault_uri = await asyncio.to_thread(put_fault, attempt["hotkey"], digest, fault_doc)
        summary = {**outcome.result_summary, "fault_uri": fault_uri}
        await db.mark_failed(
            pool,
            attempt["id"],
            fault_class=outcome.fault_class,
            fault_code=outcome.fault_code,
            fault_message=outcome.fault_message,
            result_summary=summary,
        )
        log.warning(
            "miner fault [{}] {} — {}",
            outcome.fault_code,
            attempt["model_uri"],
            outcome.fault_message,
        )


async def run() -> None:
    pool = await db.connect(config.DB_URL)
    if not health():
        raise RuntimeError(f"OpenSearch not healthy at {config.OPENSEARCH_URL}")

    log.info("model_validation started — worker={} netuid={}", _WORKER_ID, config.NETUID)
    n = await db.enqueue_from_commits(pool, config.NETUID)
    log.info("enqueued {} new commit(s)", n)

    try:
        while True:
            await db.sweep_expired(pool)
            attempt = await db.claim_next(pool, _WORKER_ID, config.LEASE_SECONDS)
            if attempt is None:
                await db.enqueue_from_commits(pool, config.NETUID)
                await asyncio.sleep(config.POLL_INTERVAL_S)
                continue

            log.info(
                "claim — block={} hotkey={} {}",
                attempt["block_number"],
                attempt["hotkey"][:10],
                attempt["model_uri"],
            )

            sanity_reason = await db.hotkey_sanity_block_reason(pool, attempt["hotkey"])
            if sanity_reason is not None:
                await db.mark_failed(
                    pool,
                    attempt["id"],
                    fault_class="MINER_FAULT",
                    fault_code="hotkey_sanity_blocked",
                    fault_message=f"hotkey blocked from further submissions — prior sanity failure: {sanity_reason}",  # noqa: E501
                    result_summary={"hotkey": attempt["hotkey"], "sanity_reason": sanity_reason},
                )
                log.info(
                    "skip — hotkey sanity-blocked ({}): {}", sanity_reason, attempt["hotkey"][:10]
                )
                continue

            dup_reason = await db.hotkey_duplicate_block_reason(pool, attempt["hotkey"])
            if dup_reason is not None:
                await db.mark_failed(
                    pool,
                    attempt["id"],
                    fault_class="MINER_FAULT",
                    fault_code="hotkey_duplicate_blocked",
                    fault_message=f"hotkey blocked from further submissions — prior duplicate: {dup_reason}",  # noqa: E501
                    result_summary={"hotkey": attempt["hotkey"], "duplicate_reason": dup_reason},
                )
                log.info("skip — hotkey duplicate-blocked: {}", attempt["hotkey"][:10])
                continue

            fails = await db.hotkey_preeval_fail_count(pool, attempt["hotkey"])
            if fails >= config.PREEVAL_MAX_FAILS:
                await db.mark_failed(
                    pool,
                    attempt["id"],
                    fault_class="MINER_FAULT",
                    fault_code="hotkey_preeval_blocked",
                    fault_message=(
                        f"hotkey is blocked — failed preeval validation {fails} times "
                        f"(limit {config.PREEVAL_MAX_FAILS})"
                    ),
                    result_summary={"hotkey": attempt["hotkey"], "preeval_fail_count": fails},
                )
                log.info(
                    "skip — hotkey preeval-blocked ({} fails): {}", fails, attempt["hotkey"][:10]
                )
                continue

            if await db.hotkey_validated(pool, attempt["hotkey"]):
                await db.mark_failed(
                    pool,
                    attempt["id"],
                    fault_class="MINER_FAULT",
                    fault_code="hotkey_already_validated",
                    fault_message="hotkey already has a validated model submission",
                    result_summary={"hotkey": attempt["hotkey"]},
                )
                log.info("skip — hotkey already validated: {}", attempt["hotkey"][:10])
                continue

            holder = None
            if attempt["commit_hash"]:
                holder = await db.model_hash_holder(
                    pool, attempt["commit_hash"], attempt["submission_id"]
                )
            if holder is not None:
                reason = (
                    f"exact duplicate: digest {attempt['commit_hash']} already submitted "
                    f"by hotkey {holder['hotkey']} ({holder['model_uri']})"
                )
                await db.mark_failed(
                    pool,
                    attempt["id"],
                    fault_class="MINER_FAULT",
                    fault_code="duplicate",
                    fault_message=reason,
                    result_summary={
                        "duplicate_of": holder["model_uri"],
                        "duplicate_of_hotkey": holder["hotkey"],
                    },
                )
                log.info(
                    "skip — exact digest duplicate of {}: {}",
                    holder["hotkey"][:10],
                    attempt["hotkey"][:10],
                )
                continue

            hb = asyncio.create_task(_heartbeat_loop(pool, attempt["id"]))
            try:
                outcome = await asyncio.to_thread(
                    process_model, attempt["model_uri"], attempt["hotkey"], attempt["coldkey"] or ""
                )
            except Exception as exc:
                outcome = _infra("unexpected", f"{type(exc).__name__}: {exc}")
            finally:
                hb.cancel()
            try:
                await _finalize(pool, attempt, outcome)
            except Exception as exc:
                log.error(
                    "finalize failed for {} — left to lease expiry: {}", attempt["model_uri"], exc
                )
    finally:
        await pool.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("model_validation stopped")


if __name__ == "__main__":
    main()
