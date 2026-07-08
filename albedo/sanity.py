"""Sanity pre-eval stage - claim, sample, GPU worker dispatch, LLM judge gate, janitor timers."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

import httpx
from loguru import logger

from albedo import db, s3
from albedo.judges import openrouter_chat
from albedo.remote_client import RemoteRunClient
from albedo.sampling import sample_prompts
from albedo.sanity_gate import PROBE_SYSTEM, VIABILITY_SYSTEM
from albedo.settings import SanityDispatchSettings, get_settings

_DISPATCH_POLL_S = 5.0
_EVENT_POLL_S = 5.0
_JANITOR_INTERVAL_S = 60.0
_RECONCILE_LIMIT = 10
_FOLLOW_TIMEOUT_S = 50.0
# Minimum resolved votes needed to decide; capped at the panel size so a single-model config works.
_MIN_RESOLVED = 2
_RAW_LOG_CHARS = 240
_ERR_LOG_CHARS = 400
# The injection re-check runs at a higher temperature so it is a genuine independent re-sample,
# not a deterministic repeat of the first probe (at temp 0.0 the re-check always confirms - even
# a spurious flag). A real injection stays flagged under variance; a false positive can clear.
_RECHECK_TEMPERATURE = get_settings().sanity.injection_recheck_temperature


# ── Judge gate: prompt assembly + parsing ─────────────────────────────────────


def _short(value: str | None, limit: int) -> str:
    # Truncates and flattens a string for structured log fields.
    return (value or "").replace("\n", "\\n")[:limit]


def _resolve_models() -> tuple[str, ...]:
    # Reads SANITY_JUDGE_MODELS from env, falling back to the eval judge defaults.
    from albedo.judges import JUDGE_MODELS

    env = get_settings().sanity.judge_models.strip()
    if env:
        return tuple(m.strip() for m in env.split(",") if m.strip())
    return JUDGE_MODELS


def _build_injection_user(prompt: str, reply: str) -> str:
    # JSON payload for the PROBE_SYSTEM auditor: untrusted "conversation" + "candidate_reply" data.
    return json.dumps(
        {"conversation": (prompt or "").rstrip(), "candidate_reply": (reply or "").rstrip()},
        ensure_ascii=False,
    )


def _build_viability_user(prompt: str, reply: str) -> str:
    # User message for the viability reviewer: context + reply + instruction.
    return (
        "TASK CONTEXT (conversation so far):\n"
        "------\n"
        f"{(prompt or '').rstrip()}\n"
        "------\n\n"
        "CANDIDATE REPLY (the model's proposed next turn):\n"
        "------\n"
        f"{(reply or '').rstrip()}\n"
        "------"
        "\n\nReview the CANDIDATE REPLY and answer with the strict JSON from the system prompt."
    )


def _extract_obj(raw: str, key: str | None = None) -> dict | None:
    # Returns the judge's JSON verdict; when key is given, prefers the object that CONTAINS it
    # (so a trailing non-verdict object cannot shadow a real verdict).
    text = (raw or "").strip()
    if not text:
        return None
    try:
        whole = json.loads(text)
        if isinstance(whole, dict):
            return whole
    except Exception:  # noqa: BLE001 - not pure JSON; scan for an embedded object below
        pass
    decoder = json.JSONDecoder()
    found: dict | None = None
    keyed: dict | None = None
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])  # handles braces inside JSON strings
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict):
            found = obj
            if key is not None and key in obj:
                keyed = obj
        i += max(end, 1)
    result = keyed if keyed is not None else found
    if result is None:
        logger.debug(f"[sanity/gate] no JSON object in judge reply: {text[:120]!r}")
    return result


def _as_bool(value: object) -> bool | None:
    # Coerces a judge flag to True/False; None when it is not a recognizable boolean (vote ignored).
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "1", "yes", "y"):
            return True
        if token in ("false", "0", "no", "n"):
            return False
    return None


def _parse_injection(raw: str) -> tuple[bool | None, str]:
    # Returns (injection_flag, evidence); flag is None when unparseable or not a clean boolean.
    obj = _extract_obj(raw, "injection")
    if obj is None or "injection" not in obj:
        return None, ""
    flag = _as_bool(obj.get("injection"))
    if flag is None:
        return None, ""
    return flag, str(obj.get("evidence", ""))[:300]


def _parse_viability(raw: str) -> tuple[bool | None, str]:
    # Returns (viable_flag, reason); flag is None when unparseable or not a clean boolean.
    obj = _extract_obj(raw, "viable")
    if obj is None or "viable" not in obj:
        return None, ""
    flag = _as_bool(obj.get("viable"))
    if flag is None:
        return None, ""
    return flag, str(obj.get("reason", ""))[:300]


# ── Judge gate: panel + verdicts ──────────────────────────────────────────────


class LLMGate(StrEnum):
    # Outcome of the gate, surfaced in the result and the sanity_results cache.
    PASSED = "passed"
    FAILED = "failed"  # not viable (veto/consensus)
    INJECTION = "injection"  # confirmed prompt-injection (terminal miner fault)
    SKIPPED = "skipped"  # judges unavailable -> infra defer


@dataclass
class JudgeVote:
    # One judge's verdict on one probe; None flags mean the judge gave no usable answer.
    model: str
    injection: bool | None = None
    inj_evidence: str = ""
    viable: bool | None = None
    via_reason: str = ""
    error: str | None = None


@dataclass
class SampleVerdict:
    # Combined per-sample outcome after injection + viability.
    prompt_excerpt: str
    passed: bool
    reason: str
    injection: bool = False
    infra: bool = False
    rechecked: bool = False
    votes: list[JudgeVote] = field(default_factory=list)


@dataclass
class SampleInput:
    # A sampled prompt plus the challenger's generated response and its heuristic pre-filter result.
    prompt: str
    response: str
    heuristic_passed: bool = True
    heuristic_reason: str = ""


@dataclass
class GateResult:
    # Aggregate gate decision across all samples.
    passed: bool
    reason: str
    infra_fault: bool
    llm_gate: LLMGate
    decision_mode: str
    per_sample: list[SampleVerdict] = field(default_factory=list)


@dataclass(frozen=True)
class _RawJudge:
    # One judge model's raw reply (or captured error) for one probe.
    model: str
    raw: str
    error: str | None = None


async def _query_panel(
    system: str, user: str, models: tuple[str, ...], temperature: float | None = None
) -> list[_RawJudge]:
    # Sends the prompt to all judges concurrently; one response per model, errors captured.
    logger.info(f"[sanity/panel] querying {len(models)} judges: {list(models)}")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def _one(model: str) -> _RawJudge:
        # Queries one judge, converting any failure into an error vote so the panel survives.
        try:
            raw = await openrouter_chat(model, messages, temperature=temperature)
            logger.info(f"[sanity/panel] {model} ok chars={len(raw or '')}")
            return _RawJudge(model=model, raw=raw)
        except Exception as exc:  # noqa: BLE001 - a dead judge must not abort the panel
            logger.warning(f"[sanity/panel] {model} failed: {type(exc).__name__}: {exc}")
            return _RawJudge(model=model, raw="", error=f"{type(exc).__name__}: {exc}")

    results = list(await asyncio.gather(*[_one(m) for m in models]))
    logger.info(
        f"[sanity/panel] done resolved={sum(1 for r in results if not r.error)}/{len(models)}"
    )
    return results


async def _injection_probe(
    prompt: str, response: str, models: tuple[str, ...], temperature: float | None = None
) -> tuple[bool | None, list[JudgeVote]]:
    # Returns (suspected, votes); suspected is None when too few judges resolved (treat as infra).
    raws = await _query_panel(
        PROBE_SYSTEM, _build_injection_user(prompt, response), models, temperature=temperature
    )
    votes: list[JudgeVote] = []
    details: list[dict[str, object]] = []
    for r in raws:
        flag, evidence = (None, "") if r.error is not None else _parse_injection(r.raw)
        votes.append(JudgeVote(model=r.model, injection=flag, inj_evidence=evidence, error=r.error))
        details.append(
            {
                "model": r.model,
                "resolved": flag is not None,
                "injection": flag,
                "evidence": _short(evidence, _RAW_LOG_CHARS),
                "error": _short(r.error, _ERR_LOG_CHARS),
                "raw_prefix": "" if r.error is not None else _short(r.raw, _RAW_LOG_CHARS),
            }
        )
    resolved = [v for v in votes if v.injection is not None]
    if len(resolved) < min(_MIN_RESOLVED, len(models)):
        logger.warning(
            f"[sanity/gate] injection judges unavailable resolved={len(resolved)}/{len(votes)} "
            f"response_chars={len(response or '')} prompt_prefix={_short(prompt, 120)!r} "
            f"details={details}"
        )
        return None, votes
    return any(v.injection for v in resolved), votes


async def _viability_probe(
    prompt: str, response: str, consensus: bool, models: tuple[str, ...]
) -> tuple[bool | None, str, list[JudgeVote]]:
    # Returns (passed, reason, votes); passed is None when too few judges resolved (treat as infra).
    raws = await _query_panel(VIABILITY_SYSTEM, _build_viability_user(prompt, response), models)
    votes: list[JudgeVote] = []
    details: list[dict[str, object]] = []
    for r in raws:
        flag, reason = (None, "") if r.error is not None else _parse_viability(r.raw)
        votes.append(JudgeVote(model=r.model, viable=flag, via_reason=reason, error=r.error))
        details.append(
            {
                "model": r.model,
                "resolved": flag is not None,
                "viable": flag,
                "reason": _short(reason, _RAW_LOG_CHARS),
                "error": _short(r.error, _ERR_LOG_CHARS),
                "raw_prefix": "" if r.error is not None else _short(r.raw, _RAW_LOG_CHARS),
            }
        )
    resolved = [v for v in votes if v.viable is not None]
    if len(resolved) < min(_MIN_RESOLVED, len(models)):
        logger.warning(
            f"[sanity/gate] viability judges unavailable resolved={len(resolved)}/{len(votes)} "
            f"response_chars={len(response or '')} prompt_prefix={_short(prompt, 120)!r} "
            f"details={details}"
        )
        return None, "viability judges unavailable", votes
    yes = sum(1 for v in resolved if v.viable)
    # consensus = majority of resolved judges; veto (default) = unanimity, any False vetoes.
    passed = yes > len(resolved) / 2 if consensus else yes == len(resolved)
    if passed:
        return True, "", votes
    nay = next((v.via_reason for v in resolved if not v.viable), "")
    return False, f"not viable: {nay}".strip()[:200], votes


async def _judge_sample(s: SampleInput, consensus: bool, models: tuple[str, ...]) -> SampleVerdict:
    # Runs the per-sample flow: heuristics -> injection (+re-check) -> viability.
    excerpt = (s.prompt or "")[:60]
    if not s.heuristic_passed:
        return SampleVerdict(excerpt, passed=False, reason=f"heuristic: {s.heuristic_reason}")

    suspected, votes = await _injection_probe(s.prompt, s.response, models)
    if suspected is None:
        return SampleVerdict(
            excerpt, False, "injection judges unavailable", infra=True, votes=votes
        )

    rechecked = False
    if suspected:
        rechecked = True
        # Re-check at a higher temperature (genuine re-sample) so a spurious first flag can clear.
        confirmed, votes = await _injection_probe(
            s.prompt, s.response, models, temperature=_RECHECK_TEMPERATURE
        )
        if confirmed is None:
            return SampleVerdict(
                excerpt,
                False,
                "injection judges unavailable",
                infra=True,
                rechecked=True,
                votes=votes,
            )
        if confirmed:
            evidence = next((v.inj_evidence for v in votes if v.injection), "")
            return SampleVerdict(
                excerpt,
                False,
                f"injection: {evidence}".strip()[:200],
                injection=True,
                rechecked=True,
                votes=votes,
            )
        # Re-check came back clean - the first flag was a false positive; continue to viability.

    decided, reason, vvotes = await _viability_probe(s.prompt, s.response, consensus, models)
    if decided is None:
        return SampleVerdict(excerpt, False, reason, infra=True, rechecked=rechecked, votes=vvotes)
    return SampleVerdict(excerpt, passed=decided, reason=reason, rechecked=rechecked, votes=vvotes)


def _aggregate(verdicts: list[SampleVerdict], mode: str) -> GateResult:
    # Combines per-sample verdicts with priority injection > infra > viability-fail > pass.
    injected = next((v for v in verdicts if v.injection), None)
    if injected is not None:
        return GateResult(False, injected.reason, False, LLMGate.INJECTION, mode, verdicts)
    if any(v.infra for v in verdicts):
        return GateResult(False, "judges unavailable", True, LLMGate.SKIPPED, mode, verdicts)
    failed = next((v for v in verdicts if not v.passed), None)
    if failed is not None:
        return GateResult(False, failed.reason, False, LLMGate.FAILED, mode, verdicts)
    return GateResult(True, "", False, LLMGate.PASSED, mode, verdicts)


async def run_gate(samples: list[SampleInput], *, consensus: bool = False) -> GateResult:
    # Judges every sample concurrently and returns the aggregate gate decision.
    mode = "consensus" if consensus else "veto"
    models = _resolve_models()
    if not samples:
        return GateResult(False, "no samples", True, LLMGate.SKIPPED, mode)
    verdicts = list(await asyncio.gather(*[_judge_sample(s, consensus, models) for s in samples]))
    result = _aggregate(verdicts, mode)
    (logger.info if result.passed else logger.warning)(
        f"[sanity/gate] passed={result.passed} gate={result.llm_gate} mode={mode} "
        f"reason={result.reason!r}"
    )
    return result


# ── Repository: claim / verdict / lease state transitions ────────────────────


@dataclass(frozen=True)
class _Claimed:
    # A claimed pre-eval job ready to dispatch to a worker (run_id = attempt_id).
    submission_id: UUID
    attempt_id: UUID
    host_id: str
    base_url: str
    request: dict[str, Any]


@dataclass(frozen=True)
class _Active:
    # An in-flight pre-eval recovered for reconciliation; the worker run_id is the attempt id.
    submission_id: UUID
    attempt_id: UUID
    base_url: str
    repo: str
    digest: str
    prompts: list[str]


def _build_request(
    submission: dict[str, Any], attempt_id: UUID, s: SanityDispatchSettings
) -> dict[str, Any]:
    # Samples the prompts deterministically from block_hash and builds the worker request payload.
    samples = sample_prompts(
        seed=str(submission["block_hash"]),
        n=s.sample_count,
        max_turns=s.max_turns_per_sample,
        manifest_path=s.dataset_manifest_path,
        manifest_hash=s.dataset_manifest_hash,
        dataset_root=s.dataset_root,
    )
    return {
        "run_id": str(attempt_id),
        "model_uri": submission["model_uri"],
        "digest": submission["model_hash"] or "",
        "prompts": [smp.prompt for smp in samples],
        "prompt_messages": [
            smp.messages or [{"role": "user", "content": smp.prompt}] for smp in samples
        ],
        "gen_max_tokens": s.gen_max_tokens,
        "min_tokens": 5,
        "max_repetition": 0.85,
        "min_vocab_ratio": 0.05,
    }


async def _claim_next(s: SanityDispatchSettings) -> _Claimed | None:
    # Claims the oldest claimable submission (HIPPIUS_VALIDATED first, then PRE_EVAL_RETRYABLE)
    # under the "pre_eval" advisory lock, plus a READY PRE_EVAL host, all in one transaction.
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        if not await db.advisory_xact_lock(conn, "pre_eval"):
            return None
        submission = await conn.fetchrow(
            """
            SELECT ms.*, cc.block_hash
            FROM model_submissions ms
            JOIN chain_commits cc ON cc.id = ms.chain_commit_id
            WHERE ms.state IN ('HIPPIUS_VALIDATED', 'PRE_EVAL_RETRYABLE')
              AND ms.retry_count < $1
              AND cc.block_hash IS NOT NULL
            ORDER BY
              CASE WHEN ms.state = 'HIPPIUS_VALIDATED' THEN 0 ELSE 1 END ASC,
              ms.priority ASC,
              ms.retry_count ASC,
              ms.created_at ASC
            FOR UPDATE OF ms SKIP LOCKED
            LIMIT 1
            """,
            s.max_retry_count,
        )
        if not submission:
            return None
        host = await conn.fetchrow(
            """
            SELECT id, base_url
            FROM remote_gpu_hosts
            WHERE role = 'PRE_EVAL' AND state = 'READY' AND free_gpu_count >= $1
            ORDER BY free_gpu_count DESC, last_heartbeat_at DESC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """,
            s.min_free_gpus,
        )
        if not host:
            return None

        attempt_number = await db.next_attempt_number(conn, submission["id"], "PRE_EVAL")
        attempt_id = uuid4()
        request = await asyncio.to_thread(_build_request, dict(submission), attempt_id, s)
        await conn.execute(
            """
            INSERT INTO stage_attempts (
                id, submission_id, stage, attempt_number, state, worker_id,
                lease_expires_at, started_at, input_snapshot
            )
            VALUES ($1, $2, 'PRE_EVAL', $3, 'RUNNING', $4,
                    now() + ($5 || ' seconds')::interval, now(), $6)
            """,
            attempt_id,
            submission["id"],
            attempt_number,
            s.worker_id,
            str(s.lease_seconds),
            {"host_id": host["id"], **request},
        )
        await conn.execute(
            """
            UPDATE model_submissions
            SET state = 'PRE_EVAL_RUNNING', updated_at = now(),
                fault_class = NULL, fault_code = NULL, fault_message = NULL
            WHERE id = $1
            """,
            submission["id"],
        )
        await db.record_event(
            conn,
            submission_id=submission["id"],
            stage_attempt_id=attempt_id,
            event_type="pre_eval_claimed",
            message=f"Pre-eval claimed by {s.worker_id} on host {host['id']}",
            data={"host_id": host["id"]},
        )
        return _Claimed(submission["id"], attempt_id, host["id"], host["base_url"], request)


async def _record_remote_event(
    submission_id: UUID, attempt_id: UUID, event: dict[str, Any]
) -> None:
    # Persists a worker event under the attempt.
    p = await db.pool()
    async with p.acquire() as conn:
        await db.record_event(
            conn,
            submission_id=submission_id,
            stage_attempt_id=attempt_id,
            event_type=f"remote_{event.get('type', 'event')}",
            message=str(event.get("message") or event.get("type") or "remote event"),
            data=event,
        )


async def _mark_passed(
    submission_id: UUID, attempt_id: UUID, repo: str, digest: str, responses: list[str], reason: str
) -> None:
    # Records the cached result, completes the attempt, and advances to PRE_EVAL_PASSED.
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        await _write_sanity_result(conn, repo, digest, True, reason, responses)
        await conn.execute(
            """
            UPDATE stage_attempts SET state = 'SUCCEEDED', finished_at = now(), result_summary = $1
            WHERE id = $2
            """,
            {"passed": True, "reason": reason},
            attempt_id,
        )
        await conn.execute(
            "UPDATE model_submissions SET state = 'PRE_EVAL_PASSED', updated_at = now()"
            " WHERE id = $1",
            submission_id,
        )
        await db.record_event(
            conn,
            submission_id=submission_id,
            stage_attempt_id=attempt_id,
            event_type="pre_eval_passed",
            message="Pre-eval passed",
        )


async def _mark_failed(
    *,
    submission_id: UUID,
    attempt_id: UUID,
    repo: str,
    digest: str,
    fault_class: str,
    fault_code: str,
    fault_message: str,
    retryable: bool,
    responses: list[str] | None = None,
    artifact_uri: str | None = None,
) -> None:
    # Fails the attempt; retryable -> PRE_EVAL_RETRYABLE (unless retries exhausted, then
    # TERMINAL_INVALID), terminal -> TERMINAL_INVALID with a cached sanity_results row.
    s = get_settings().sanity
    attempt_state = "FAILED_RETRYABLE" if retryable else "FAILED_TERMINAL"
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        if not retryable:
            await _write_sanity_result(conn, repo, digest, False, fault_message, responses or [])
            if artifact_uri:
                await _insert_sanity_artifact(conn, submission_id, attempt_id, artifact_uri)
        await conn.execute(
            """
            UPDATE stage_attempts
            SET state = $1, finished_at = now(),
                fault_class = $2, fault_code = $3, fault_message = $4
            WHERE id = $5
            """,
            attempt_state,
            fault_class,
            fault_code,
            fault_message,
            attempt_id,
        )
        # Cap retryable failures: once retry_count reaches max, move to TERMINAL_INVALID so the
        # submission does not sit in PRE_EVAL_RETRYABLE forever unclaimed by the claim query.
        await conn.execute(
            """
            UPDATE model_submissions
            SET state = CASE
                    WHEN $1::boolean AND retry_count + 1 >= $2::int THEN 'TERMINAL_INVALID'
                    WHEN $1::boolean THEN 'PRE_EVAL_RETRYABLE'
                    ELSE 'TERMINAL_INVALID'
                END,
                fault_class = $3, fault_code = $4, fault_message = $5,
                retry_count = retry_count + 1, updated_at = now()
            WHERE id = $6
            """,
            retryable,
            s.max_retry_count,
            fault_class,
            fault_code,
            fault_message,
            submission_id,
        )
        await db.record_event(
            conn,
            submission_id=submission_id,
            stage_attempt_id=attempt_id,
            event_type="pre_eval_failed",
            severity="ERROR",
            message=fault_message,
            data={"fault_class": fault_class, "fault_code": fault_code, "retryable": retryable},
        )


async def _release_claim(submission_id: UUID, attempt_id: UUID, fault_message: str) -> None:
    # Releases a worker-busy (409) claim back to HIPPIUS_VALIDATED without consuming a retry.
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            UPDATE stage_attempts
            SET state = 'FAILED_RETRYABLE', finished_at = now(),
                fault_class = 'INFRA_FAULT', fault_code = 'worker_busy', fault_message = $1
            WHERE id = $2
            """,
            fault_message,
            attempt_id,
        )
        await conn.execute(
            """
            UPDATE model_submissions
            SET state = 'HIPPIUS_VALIDATED', updated_at = now()
            WHERE id = $1 AND state = 'PRE_EVAL_RUNNING'
            """,
            submission_id,
        )


async def _write_sanity_result(
    conn: Any, repo: str, digest: str, passed: bool, reason: str, responses: list[str]
) -> None:
    # Upserts the digest-keyed cache row (first verdict wins).
    await conn.execute(
        """
        INSERT INTO sanity_results (repo, digest, passed, reason, responses, timing, checked_at)
        VALUES ($1, $2, $3, $4, $5, $6, now())
        ON CONFLICT (digest) DO NOTHING
        """,
        repo,
        digest,
        passed,
        reason,
        responses,
        {},
    )


async def _insert_sanity_artifact(
    conn: Any, submission_id: UUID, attempt_id: UUID, uri: str
) -> None:
    # Records the uploaded fault report so the dashboard can link it (artifact_type SANITY_RESULT).
    bucket, object_key = (None, None)
    if uri.startswith("s3://"):
        bucket, _, object_key = uri[len("s3://") :].partition("/")
    await conn.execute(
        """
        INSERT INTO artifacts (
            id, submission_id, stage_attempt_id, artifact_type,
            storage_backend, uri, bucket, object_key, content_type
        )
        VALUES ($1, $2, $3, 'SANITY_RESULT', 's3', $4, $5, $6, 'application/json')
        """,
        uuid4(),
        submission_id,
        attempt_id,
        uri,
        bucket or None,
        object_key or None,
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────


def _make_client(base_url: str, s: SanityDispatchSettings) -> RemoteRunClient:
    # Builds the GPU worker client for the sanity-runs API.
    return RemoteRunClient(
        base_url=base_url, run_kind="sanity-runs", auth_token=s.remote_auth_token
    )


async def _follow_until_result(
    client: RemoteRunClient,
    *,
    submission_id: UUID,
    attempt_id: UUID,
    run_id: str,
    lease_seconds: int,
) -> dict[str, Any]:
    # Polls the worker, recording events and refreshing the lease, until a result appears.
    # Heartbeat runs once per poll tick (not just per event) so a long model download or
    # vLLM boot - which emits no events - does not let the lease expire mid-wait.
    seen = 0
    while True:
        events = await client.get_events(run_id)
        for event in events[seen:]:
            logger.info(
                f"[sanity-dispatch] worker event={event.get('type', '?')} run={run_id} "
                f"submission={str(submission_id)[:8]}"
            )
            await _record_remote_event(submission_id, attempt_id, event)
            if event.get("type") == "result":
                logger.info(
                    f"[sanity-dispatch] result received run={run_id} state={event.get('state')} "
                    f"submission={str(submission_id)[:8]}"
                )
                await db.heartbeat_attempt(attempt_id, lease_seconds)
                return event
        seen = max(seen, len(events))
        # Heartbeat on every tick so a silent download/boot period does not expire the lease.
        await db.heartbeat_attempt(attempt_id, lease_seconds)
        status = await client.get_run(run_id)
        if status.get("type") == "result" or status.get("state") in {"succeeded", "failed"}:
            if status.get("type") == "result":
                await _record_remote_event(submission_id, attempt_id, status)
            return status
        await asyncio.sleep(_EVENT_POLL_S)


async def _complete(
    *,
    submission_id: UUID,
    attempt_id: UUID,
    repo: str,
    digest: str,
    prompts: list[str],
    result: dict[str, Any],
    s: SanityDispatchSettings,
) -> None:
    # Judges the generated responses and writes the terminal verdict.
    logger.info(
        f"[sanity-dispatch] completing submission={str(submission_id)[:8]} "
        f"digest={digest[:16]} state={result.get('state')}"
    )
    if result.get("state") == "failed":
        await _mark_failed(
            submission_id=submission_id,
            attempt_id=attempt_id,
            repo=repo,
            digest=digest,
            fault_class="INFRA_FAULT",
            fault_code=result.get("fault_code", "worker_fault"),
            fault_message=result.get("fault_message", ""),
            retryable=bool(result.get("retryable", True)),
        )
        return

    responses = list(result.get("responses", []))
    heuristics = list(result.get("heuristics", []))
    samples = [
        SampleInput(
            prompt=prompts[i] if i < len(prompts) else "",
            response=responses[i],
            heuristic_passed=bool(heuristics[i].get("passed", True))
            if i < len(heuristics)
            else True,
            heuristic_reason=heuristics[i].get("reason", "") if i < len(heuristics) else "",
        )
        for i in range(len(responses))
    ]
    try:
        gate = await run_gate(samples, consensus=s.consensus)
    except Exception as exc:  # noqa: BLE001 - a judge/OpenRouter failure must fail cleanly, not escape
        logger.exception(f"[sanity-dispatch] judge gate failed submission={submission_id}: {exc}")
        await _mark_failed(
            submission_id=submission_id,
            attempt_id=attempt_id,
            repo=repo,
            digest=digest,
            fault_class="INFRA_FAULT",
            fault_code="judges_failed",
            fault_message=str(exc),
            retryable=True,
        )
        return

    if gate.infra_fault:
        await _mark_failed(
            submission_id=submission_id,
            attempt_id=attempt_id,
            repo=repo,
            digest=digest,
            fault_class="INFRA_FAULT",
            fault_code="judges_unavailable",
            fault_message=gate.reason,
            retryable=True,
        )
    elif gate.passed:
        await _mark_passed(submission_id, attempt_id, repo, digest, responses, gate.reason)
    else:
        # Terminal miner fault: publish a fault report to Hippius (reason + per-judge evidence)
        # so it can be linked from the dashboard, then record the artifact alongside the verdict.
        detail = {
            "submission_id": str(submission_id),
            "repo": repo,
            "digest": digest,
            "fault_code": str(gate.llm_gate),
            "reason": gate.reason,
            "decision_mode": gate.decision_mode,
            "gate": dataclasses.asdict(gate),
            "prompts": prompts,
            "responses": responses,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        key = f"sanity/{submission_id}/{digest.replace(':', '_')}/fault.json"
        artifact_uri = await s3.put_json(key, detail)
        await _mark_failed(
            submission_id=submission_id,
            attempt_id=attempt_id,
            repo=repo,
            digest=digest,
            fault_class="MINER_FAULT",
            fault_code=str(gate.llm_gate),
            fault_message=gate.reason,
            retryable=False,
            responses=responses,
            artifact_uri=artifact_uri,
        )


async def _dispatch_once(s: SanityDispatchSettings) -> bool:
    # Claims and runs one pre-eval end to end; returns False when nothing was claimable.
    claimed = await _claim_next(s)
    if not claimed:
        logger.debug("[sanity-dispatch] no claimable pre-eval")
        return False
    digest = claimed.request["digest"]
    logger.info(
        f"[sanity-dispatch] claimed submission={claimed.submission_id} "
        f"digest={digest[:16]} host={claimed.host_id}"
    )
    client = _make_client(claimed.base_url, s)
    try:
        await client.ready()
        start = await client.start_run(claimed.request)
        run_id = str(start.get("run_id") or claimed.attempt_id)
        await db.heartbeat_attempt(claimed.attempt_id, s.lease_seconds)
        result = await _follow_until_result(
            client,
            submission_id=claimed.submission_id,
            attempt_id=claimed.attempt_id,
            run_id=run_id,
            lease_seconds=s.lease_seconds,
        )
        await _complete(
            submission_id=claimed.submission_id,
            attempt_id=claimed.attempt_id,
            repo=claimed.request["model_uri"],
            digest=digest,
            prompts=list(claimed.request["prompts"]),
            result=result,
            s=s,
        )
        return True
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 409:
            logger.warning(
                f"[sanity-dispatch] worker busy, releasing claim "
                f"submission={claimed.submission_id} digest={digest[:16]}: {exc}"
            )
            await _release_claim(claimed.submission_id, claimed.attempt_id, str(exc))
            return True
        logger.warning(
            f"[sanity-dispatch] worker HTTP error submission={claimed.submission_id} "
            f"digest={digest[:16]}: {exc}"
        )
        await _mark_failed(
            submission_id=claimed.submission_id,
            attempt_id=claimed.attempt_id,
            repo=claimed.request["model_uri"],
            digest=digest,
            fault_class="INFRA_FAULT",
            fault_code="worker_unreachable",
            fault_message=str(exc),
            retryable=True,
        )
        return True
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        logger.warning(
            f"[sanity-dispatch] worker unreachable submission={claimed.submission_id} "
            f"digest={digest[:16]}: {exc}"
        )
        await _mark_failed(
            submission_id=claimed.submission_id,
            attempt_id=claimed.attempt_id,
            repo=claimed.request["model_uri"],
            digest=digest,
            fault_class="INFRA_FAULT",
            fault_code="worker_unreachable",
            fault_message=str(exc),
            retryable=True,
        )
        return True
    finally:
        await client.aclose()


async def run_dispatcher() -> None:
    # Continuously claims and dispatches pre-evals; keeps the loop alive across transient errors.
    s = get_settings().sanity
    while True:
        try:
            did_work = await _dispatch_once(s)
            if not did_work:
                await asyncio.sleep(_DISPATCH_POLL_S)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive across DB blips, etc.
            logger.exception(
                f"[sanity-dispatch] unhandled error in dispatch loop, "
                f"retrying in {_DISPATCH_POLL_S}s: {exc}"
            )
            await asyncio.sleep(_DISPATCH_POLL_S)


# ── Janitor: sweeper + reconciler ─────────────────────────────────────────────


async def _sweep_abandoned(s: SanityDispatchSettings) -> int:
    # Reclaims expired RUNNING pre-eval attempts (dead dispatcher/host) back to the queue.
    # When retry_count already reached the cap the submission moves to TERMINAL_INVALID instead
    # of RETRYABLE so it does not sit in a ghost state that the claim query never picks up.
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        rows = await conn.fetch(
            """
            SELECT sa.id AS attempt_id, sa.submission_id, ms.retry_count
            FROM stage_attempts sa
            JOIN model_submissions ms ON ms.id = sa.submission_id
            WHERE sa.stage = 'PRE_EVAL' AND sa.state = 'RUNNING'
              AND sa.lease_expires_at < now() AND ms.state = 'PRE_EVAL_RUNNING'
            FOR UPDATE OF sa, ms SKIP LOCKED
            """
        )
        for row in rows:
            exhausted = row["retry_count"] + 1 >= s.max_retry_count
            next_state = "TERMINAL_INVALID" if exhausted else "PRE_EVAL_RETRYABLE"
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = 'ABANDONED', finished_at = now(), fault_class = 'INFRA_FAULT',
                    fault_code = 'pre_eval_lease_expired',
                    fault_message = 'lease expired before completion'
                WHERE id = $1
                """,
                row["attempt_id"],
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = $1, fault_class = 'INFRA_FAULT',
                    fault_code = 'pre_eval_lease_expired',
                    fault_message = 'lease expired before completion',
                    retry_count = retry_count + 1, updated_at = now()
                WHERE id = $2
                """,
                next_state,
                row["submission_id"],
            )
            await db.record_event(
                conn,
                submission_id=row["submission_id"],
                stage_attempt_id=row["attempt_id"],
                event_type="pre_eval_abandoned",
                severity="WARN",
                message=f"Pre-eval lease expired before completion (-> {next_state})",
                data={"worker_id": s.worker_id, "exhausted": exhausted},
            )
        return len(rows)


async def _list_reconcilable(limit: int) -> list[_Active]:
    # Finds RUNNING pre-eval attempts (dispatcher may have crashed mid-poll) to replay.
    p = await db.pool()
    rows = await p.fetch(
        """
        SELECT sa.id AS attempt_id, sa.submission_id, sa.input_snapshot, h.base_url
        FROM stage_attempts sa
        JOIN model_submissions ms ON ms.id = sa.submission_id
        JOIN remote_gpu_hosts h ON h.id = (sa.input_snapshot->>'host_id')
        WHERE sa.stage = 'PRE_EVAL' AND sa.state = 'RUNNING' AND ms.state = 'PRE_EVAL_RUNNING'
        ORDER BY sa.started_at ASC
        LIMIT $1
        """,
        limit,
    )
    active: list[_Active] = []
    for row in rows:
        snap = row["input_snapshot"] or {}
        active.append(
            _Active(
                submission_id=row["submission_id"],
                attempt_id=row["attempt_id"],
                base_url=row["base_url"],
                repo=snap.get("model_uri", ""),
                digest=snap.get("digest", ""),
                prompts=list(snap.get("prompts", [])),
            )
        )
    return active


async def _reconcile_running(s: SanityDispatchSettings) -> int:
    # Replays in-flight pre-evals whose dispatcher may have crashed mid-poll; the follow timeout
    # keeps one stuck run from monopolizing the janitor tick.
    in_flight = await _list_reconcilable(_RECONCILE_LIMIT)
    logger.info(f"[sanity-dispatch] reconcile found={len(in_flight)}")
    if not in_flight:
        return 0
    reconciled = 0
    for active in in_flight:
        client = _make_client(active.base_url, s)
        try:
            result = await asyncio.wait_for(
                _follow_until_result(
                    client,
                    submission_id=active.submission_id,
                    attempt_id=active.attempt_id,
                    run_id=str(active.attempt_id),
                    lease_seconds=s.lease_seconds,
                ),
                timeout=_FOLLOW_TIMEOUT_S,
            )
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            logger.warning(
                f"[sanity-dispatch] reconcile skipped submission={active.submission_id} "
                f"run={active.attempt_id}: {exc}"
            )
            continue
        finally:
            await client.aclose()
        try:
            await _complete(
                submission_id=active.submission_id,
                attempt_id=active.attempt_id,
                repo=active.repo,
                digest=active.digest,
                prompts=active.prompts,
                result=result,
                s=s,
            )
        except Exception as exc:  # noqa: BLE001 - one bad completion must not abort the loop
            logger.exception(
                f"[sanity-dispatch] reconcile _complete failed "
                f"submission={active.submission_id}: {exc}"
            )
            continue
        reconciled += 1
    return reconciled


async def run_janitor() -> None:
    # Every 60s: sweep expired PRE_EVAL leases, then re-attach to in-flight runs after a crash.
    s = get_settings().sanity
    while True:
        try:
            swept = await _sweep_abandoned(s)
            if swept:
                logger.info(f"[sanity-janitor] abandoned={swept}")
            reconciled = await _reconcile_running(s)
            if reconciled:
                logger.info(f"[sanity-janitor] reconciled={reconciled}")
        except Exception as exc:  # noqa: BLE001 - the janitor must survive DB/worker blips
            logger.exception(f"[sanity-janitor] tick failed: {exc}")
        await asyncio.sleep(_JANITOR_INTERVAL_S)
