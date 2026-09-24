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
from model_validation.dedup import bank
from model_validation.opensearch import health
from model_validation.storage import (
    download_config,
    download_full,
    list_files,
    make_ref,
    make_room,
    safetensors_headers,
)
from model_validation.uploads import put_fault
from model_validation.validate import (
    check_dtypes,
    check_genesis,
    check_index,
    check_repo,
    check_shapes,
    dtypes_from_headers,
    seed_shapes,
    shapes_from_headers,
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
        return f" — hotkey has {left} validation strike(s) left before ban"
    return " — hotkey has 0 validation strikes left and is now banned from further submissions"


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


def _shape_outcome(headers: dict[str, dict]) -> Outcome | None:
    """Candidate tensor shapes against the seed's, from the headers the dtype preflight already
    fetched.  Placed after the metadata_hash check (a byte-identical config.json already pins the
    architecture and gives the clearer message when that is what is wrong) but before
    download_full, so a rebuilt model costs one range request rather than a multi-GB download and
    five dedup attempts that all die inside canonicalize().  In shadow mode it can never change the
    outcome, failures included.
    """
    try:
        ok, msg = check_shapes(shapes_from_headers(headers), seed_shapes(dedup.ref_dir()))
    except Exception as exc:
        if not config.SHAPE_ENFORCE:
            log.warning("[shadow] tensor shape check unavailable: {}", exc)
            return None
        return _infra("seed_shapes_failed", f"could not read genesis seed tensor shapes: {exc}")
    if ok:
        return None
    if not config.SHAPE_ENFORCE:
        log.warning(
            "[shadow] tensor shape mismatch — {} — not enforced (ALBEDO_SHAPE_ENFORCE)", msg
        )
        return None
    return _miner("tensor_shape", msg, {})


def process_model(
    model_uri: str,
    hotkey: str,
    coldkey: str = "",
    protected_repos: frozenset[str] = frozenset(),
) -> Outcome:
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
        headers = safetensors_headers(ref)
    except Exception as exc:
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on {ref.backend}: {exc}", {})
        return _infra("preflight_failed", f"could not read safetensors headers: {exc}")
    ok, msg = check_dtypes(dtypes_from_headers(headers))
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

    shape_outcome = _shape_outcome(headers)
    if shape_outcome is not None:
        return shape_outcome

    try:
        make_room(ref, protected_repos)
        model_dir = download_full(ref)
    except Exception as exc:
        if _is_not_found(exc):
            return _miner("repo_not_found", f"repo/revision not found on {ref.backend}: {exc}", {})
        return _infra("download_failed", f"model download failed: {exc}")
    mdir = Path(model_dir)
    if not any(mdir.glob("*.safetensors")):
        return _miner("incomplete_repo", "downloaded repo is missing *.safetensors", {})

    ok, msg = check_index(model_dir)
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
        reason = res.verdict.reason if res.verdict else None
        if dedup.enforces(reason, coldkey):
            code = dedup.fault_code(reason)
            return _miner(code, dedup.public_message(res), summary, fault_detail=summary)
        log.warning(
            "[shadow] dedup REJECT {} — {} — not enforced "
            "(reason={} enforce={} coldkey_known={} allowed={})",
            model_uri,
            dedup.public_message(res),
            reason,
            config.DEDUP_ENFORCE,
            bool(coldkey),
            sorted(dedup.enforced_reasons()),
        )
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
        log.warning(
            "infra fault [{}] {} → {} — {}",
            outcome.fault_code,
            attempt["model_uri"],
            new_state,
            outcome.fault_message,
        )
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
    log.info("dedup bank: {} banked models in {}", bank.preflight(), bank.index_name())

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

            if await db.hotkey_reregistered(pool, attempt["hotkey"]):
                await db.mark_failed(
                    pool,
                    attempt["id"],
                    fault_class="MINER_FAULT",
                    fault_code="hotkey_reregistered",
                    fault_message=(
                        "hotkey was used before it was deregistered — register a new hotkey"
                    ),
                    result_summary={"hotkey": attempt["hotkey"]},
                )
                log.info("skip — hotkey re-registered: {}", attempt["hotkey"][:10])
                continue

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

            coldkey = attempt["coldkey"] or await db.coldkey_for(pool, attempt["hotkey"])
            if not coldkey:
                log.warning(
                    "no coldkey for hotkey {} — the miner's own models cannot be excluded from "
                    "dedup; copy verdicts will be logged, not enforced",
                    attempt["hotkey"][:10],
                )

            hb = asyncio.create_task(_heartbeat_loop(pool, attempt["id"]))
            try:
                protected = frozenset(await db.protected_pre_eval_repos(pool))
                outcome = await asyncio.to_thread(
                    process_model,
                    attempt["model_uri"],
                    attempt["hotkey"],
                    coldkey=coldkey,
                    protected_repos=protected,
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
