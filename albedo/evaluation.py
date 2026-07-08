"""Eval orchestration - claims EVAL_QUEUED submissions, drives the remote GPU duel to a verdict."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from albedo import db, notifications
from albedo.remote_client import RemoteRunClient
from albedo.settings import get_settings

_POLL_S = get_settings().eval.dispatch_poll_seconds
_REMOTE_TIMEOUT_S = get_settings().eval.remote_event_timeout_seconds
_JUDGE_COUNT = get_settings().eval.judge_count
_JANITOR_INTERVAL_S = 60.0
_MAX_POLL_ERRORS = 5

# States meaning a hotkey already reached eval (or beyond) - one scored eval per hotkey.
_SCORED_OR_BEYOND = [
    "EVAL_RUNNING",
    "EVAL_WIN",
    "SET_REIGN_RUNNING",
    "REIGN_SET",
    "WEIGHT_SET_RUNNING",
    "COMPLETE_LOSS",
    "COMPLETE_CORONATED",
]

# ── Wire contract with the GPU eval worker ───────────────────────────────────


class Challenger(BaseModel):
    # Challenger model reference sent to the remote worker.
    model_uri: str
    model_hash: str


class PreviousKing(BaseModel):
    # Reigning slot-1 king the challenger duels against.
    model_uri: str
    model_hash: str
    king_version: int


class DatasetConfig(BaseModel):
    # Pinned dataset + deterministic sampling parameters for the duel.
    version: str
    manifest_uri: str
    manifest_hash: str
    sample_count: int
    max_turns_per_sample: int = 10
    sample_seed: str
    sampling_algo: str
    generation_batch_size: int = 8
    scoring_batch_size: int = 8
    sample_ids: list[str] = Field(default_factory=list)


class ScoringConfig(BaseModel):
    # Judge ensemble pin for the duel.
    judge_config_hash: str
    judge_count: int = 3
    allowed_scores: list[float] = Field(default_factory=lambda: [0, 0.5, 1])


class GpuRequest(BaseModel):
    # GPU topology the duel needs (two 4-way tensor-parallel vLLMs).
    accelerator: str = "B200"
    min_gpus: int = 8
    preferred_gpus: int = 8
    previous_king_gpu_count: int = 4
    challenger_gpu_count: int = 4
    tensor_parallel_size_per_model: int = 4


class EvalRequest(BaseModel):
    # Full POST /eval-runs body.
    eval_run_id: UUID
    submission_id: UUID
    challenger: Challenger
    previous_king: PreviousKing
    dataset: DatasetConfig
    scoring: ScoringConfig
    gpu_request: GpuRequest = Field(default_factory=GpuRequest)
    artifact_prefix: str


# ── Fault classification (ported verbatim from faults.py) ────────────────────

MINER_FAULT = "MINER_FAULT"
INFRA_FAULT = "INFRA_FAULT"
REMOTE_EVAL_FAULT = "REMOTE_EVAL_FAULT"
PROVIDER_FAULT = "PROVIDER_FAULT"
UNKNOWN_FAULT = "UNKNOWN_FAULT"


@dataclass(frozen=True)
class FaultDecision:
    # Normalized fault verdict driving the retryable/terminal state split.
    fault_class: str
    fault_code: str
    fault_message: str
    retryable: bool


def classify_failure_verdict(verdict: dict[str, Any]) -> FaultDecision:
    # Missing/malformed fields default to infra/retryable so miner faults are never accidental.
    fault_class = str(verdict.get("fault_class") or UNKNOWN_FAULT)
    fault_code = str(verdict.get("fault_code") or "unknown_remote_failure")
    fault_message = str(verdict.get("fault_message") or "Remote eval failed")
    retryable = bool(verdict.get("retryable", fault_class != MINER_FAULT))

    if fault_class == MINER_FAULT:
        retryable = False
    elif fault_class in {INFRA_FAULT, REMOTE_EVAL_FAULT, PROVIDER_FAULT, UNKNOWN_FAULT}:
        retryable = True
    else:
        fault_class = UNKNOWN_FAULT
        retryable = True

    return FaultDecision(fault_class, fault_code, fault_message, retryable)


def broken_stream_fault(message: str = "Remote eval stream ended before verdict") -> FaultDecision:
    # Transport failure before a verdict arrived - always retryable.
    return FaultDecision(REMOTE_EVAL_FAULT, "remote_stream_broken", message, True)


# ── Slack failure notifications (best-effort, deduped) ───────────────────────


def _notify_failure(
    *,
    submission_id: UUID,
    eval_run_id: UUID,
    fault: FaultDecision,
    remote_run_id: str = "",
) -> None:
    # Posts an eval failure via the shared Slack helper; deduped per run/code, never raises.
    notifications.notify_error(
        notifications.EvalErrorNotification(
            component="eval-dispatcher",
            severity="error",
            message=fault.fault_message,
            eval_run_id=str(eval_run_id),
            submission_id=str(submission_id),
            fault_class=fault.fault_class,
            fault_code=fault.fault_code,
            retryable=fault.retryable,
            details={"remote_run_id": remote_run_id or ""},
        )
    )


# ── Request building ─────────────────────────────────────────────────────────


def _build_sample_ids(block_hash: str) -> list[str]:
    # Deterministic sample coordinates from the pinned manifest; [] defers sampling to the worker.
    e = get_settings().eval
    if not e.dataset_manifest_path:
        return []
    from albedo.sampling import load_manifest_file, multi_source_manifest_sample_ids

    manifest = load_manifest_file(e.dataset_manifest_path, expected_sha256=e.dataset_manifest_hash)
    return multi_source_manifest_sample_ids(
        manifest,
        block_hash=block_hash,
        sample_count=e.sample_count,
        max_turns_per_sample=e.max_turns_per_sample,
    )


def build_eval_request(submission: Any, king: Any, eval_run_id: UUID) -> EvalRequest:
    # Assembles the full duel request from the claimed submission and the active slot-1 king.
    e = get_settings().eval
    return EvalRequest(
        eval_run_id=eval_run_id,
        submission_id=submission["id"],
        challenger=Challenger(
            model_uri=submission["model_uri"], model_hash=submission["model_hash"]
        ),
        previous_king=PreviousKing(
            model_uri=king["model_uri"],
            model_hash=king["model_hash"],
            king_version=king["king_version"],
        ),
        dataset=DatasetConfig(
            version=e.dataset_version,
            manifest_uri=e.dataset_manifest_uri,
            manifest_hash=e.dataset_manifest_hash,
            sample_count=e.sample_count,
            max_turns_per_sample=e.max_turns_per_sample,
            sample_seed=submission["block_hash"],
            sampling_algo=e.sampling_algo,
            sample_ids=_build_sample_ids(submission["block_hash"]),
        ),
        scoring=ScoringConfig(judge_config_hash=e.judge_config_hash, judge_count=_JUDGE_COUNT),
        artifact_prefix=(
            f"{e.artifact_prefix.rstrip('/')}/submissions/{submission['id']}/eval/{eval_run_id}"
        ),
    )


# ── Claiming ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClaimedEval:
    # One claimed duel: durable ids plus where to send the request.
    submission_id: UUID
    attempt_id: UUID
    eval_run_id: UUID
    host_base_url: str
    request: EvalRequest


async def _claim_next_eval() -> ClaimedEval | None:
    # Strictly serial claim: advisory lock, bail on any running eval or pending reign work.
    e = get_settings().eval
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        if not await db.advisory_xact_lock(conn, "full_eval"):
            return None
        if await conn.fetchrow(
            "SELECT id FROM model_submissions WHERE state = 'EVAL_RUNNING' LIMIT 1"
        ):
            return None
        if await conn.fetchrow(
            "SELECT id FROM model_submissions"
            " WHERE state IN ('EVAL_WIN', 'SET_REIGN_RUNNING') LIMIT 1"
        ):
            return None

        while True:
            submission = await conn.fetchrow(
                """
                SELECT ms.*, cc.block_hash
                FROM model_submissions ms
                JOIN chain_commits cc ON cc.id = ms.chain_commit_id
                WHERE ms.state = 'EVAL_QUEUED'
                  AND cc.block_hash IS NOT NULL
                ORDER BY ms.priority ASC, ms.created_at ASC
                FOR UPDATE OF ms SKIP LOCKED
                LIMIT 1
                """
            )
            if not submission:
                return None
            already = await conn.fetchrow(
                "SELECT 1 FROM model_submissions"
                " WHERE hotkey = $1 AND state = ANY($2::text[]) LIMIT 1",
                submission["hotkey"],
                _SCORED_OR_BEYOND,
            )
            if not already:
                break
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'TERMINAL_INVALID', fault_class = 'MINER_FAULT',
                    fault_code = 'hotkey_already_validated',
                    fault_message = 'hotkey already has a validated model submission',
                    updated_at = now(), finished_at = now()
                WHERE id = $1
                """,
                submission["id"],
            )
            await db.record_event(
                conn,
                submission_id=submission["id"],
                stage_attempt_id=None,
                event_type="eval_skipped_hotkey_already_validated",
                severity="WARN",
                message="hotkey already has a validated model submission",
                data={"hotkey": submission["hotkey"]},
            )

        host = await conn.fetchrow(
            """
            SELECT id, base_url
            FROM remote_gpu_hosts
            WHERE role = 'EVAL' AND state = 'READY' AND free_gpu_count >= 8
            ORDER BY free_gpu_count DESC, last_heartbeat_at DESC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """
        )
        if not host:
            return None

        king = await conn.fetchrow(
            """
            SELECT kv.version AS king_version, kv.submission_id AS king_submission_id,
                   kv.model_hash, a.uri AS model_uri
            FROM reigns r
            JOIN reign_members rm ON rm.reign_id = r.id AND rm.slot = 1
            JOIN king_versions kv ON kv.id = rm.king_version_id
            JOIN artifacts a ON a.id = kv.artifact_id
            WHERE r.state = 'ACTIVE'
            ORDER BY r.version DESC
            LIMIT 1
            """
        )
        if not king:
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'EVAL_RETRYABLE', fault_class = 'INFRA_FAULT',
                    fault_code = 'missing_active_lead_king',
                    fault_message = 'No active lead king is available for eval',
                    retry_count = retry_count + 1, updated_at = now()
                WHERE id = $1
                """,
                submission["id"],
            )
            return None

        attempt_number = await db.next_attempt_number(conn, submission["id"], "EVAL")
        attempt_id = uuid4()
        eval_run_id = uuid4()
        request = build_eval_request(submission, king, eval_run_id)

        await conn.execute(
            """
            INSERT INTO stage_attempts (
                id, submission_id, stage, attempt_number, state, worker_id,
                lease_expires_at, started_at, input_snapshot
            )
            VALUES ($1, $2, 'EVAL', $3, 'RUNNING', $4,
                    now() + ($5 || ' seconds')::interval, now(), $6)
            """,
            attempt_id,
            submission["id"],
            attempt_number,
            e.worker_id,
            str(e.lease_seconds),
            request.model_dump(mode="json"),
        )
        await conn.execute(
            """
            UPDATE model_submissions
            SET state = 'EVAL_RUNNING', updated_at = now(), fault_class = NULL,
                fault_code = NULL, fault_message = NULL
            WHERE id = $1
            """,
            submission["id"],
        )
        await conn.execute(
            """
            INSERT INTO eval_runs (
                id, submission_id, stage_attempt_id, king_submission_id,
                king_model_hash, challenger_model_hash, remote_host_id, state, gpu_count,
                dataset_version, dataset_manifest_hash, dataset_sample_seed, dataset_sample_ids,
                dataset_max_turns_per_sample, dataset_sampling_algo, judge_config_hash,
                judge_count, sample_count, started_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'DISPATCHED', 8,
                    $8, $9, $10, $11, $12, $13, $14, $15, $16, now())
            """,
            eval_run_id,
            submission["id"],
            attempt_id,
            king["king_submission_id"],
            king["model_hash"],
            submission["model_hash"],
            host["id"],
            request.dataset.version,
            request.dataset.manifest_hash,
            request.dataset.sample_seed,
            request.dataset.sample_ids,
            request.dataset.max_turns_per_sample,
            request.dataset.sampling_algo,
            request.scoring.judge_config_hash,
            request.scoring.judge_count,
            request.dataset.sample_count,
        )
        await db.record_event(
            conn,
            submission_id=submission["id"],
            stage_attempt_id=attempt_id,
            event_type="eval_claimed",
            severity="INFO",
            message=f"Eval claimed by {e.worker_id} on host {host['id']}",
            data={"eval_run_id": str(eval_run_id), "remote_host_id": host["id"]},
        )

    return ClaimedEval(
        submission_id=submission["id"],
        attempt_id=attempt_id,
        eval_run_id=eval_run_id,
        host_base_url=host["base_url"],
        request=request,
    )


# ── Durable state transitions ────────────────────────────────────────────────


def _gpu_ids_from_topology(topology: object) -> list[str] | None:
    # Flattens {previous_king: [...], challenger: [...]} into eval_runs.gpu_ids.
    if not isinstance(topology, dict):
        return None
    ids: list[str] = []
    for key in ("previous_king", "challenger"):
        values = topology.get(key)
        if isinstance(values, list):
            ids.extend(str(value) for value in values)
    return ids or None


def _win_margin(verdict: dict[str, Any]) -> float | None:
    # challenger - king score delta, None until both scores exist.
    challenger, king = verdict.get("score_challenger"), verdict.get("score_king")
    if challenger is None or king is None:
        return None
    return float(challenger) - float(king)


async def _set_remote_run_id(eval_run_id: UUID, remote_run_id: str) -> None:
    # Persists the worker-assigned run id so a crashed dispatcher can reconcile.
    p = await db.pool()
    await p.execute(
        "UPDATE eval_runs SET remote_run_id = $1 WHERE id = $2", remote_run_id, eval_run_id
    )


async def _record_remote_event(
    submission_id: UUID, attempt_id: UUID, event: dict[str, Any]
) -> None:
    # Records one remote event exactly once and maps progress onto eval_runs state.
    event_type = f"remote_{event.get('type', 'event')}"
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        exists = await conn.fetchrow(
            """
            SELECT 1 FROM events
            WHERE stage_attempt_id = $1 AND event_type = $2 AND data = $3
            LIMIT 1
            """,
            attempt_id,
            event_type,
            event,
        )
        if exists:
            return
        eval_run_id = event.get("eval_run_id")
        kind = event.get("type")
        if eval_run_id and kind == "generation_started":
            await conn.execute(
                """
                UPDATE eval_runs
                SET state = 'GENERATING', gpu_topology = $1, gpu_ids = $2,
                    sample_count = COALESCE($3, sample_count)
                WHERE id = $4
                """,
                event.get("gpu_topology", {}),
                _gpu_ids_from_topology(event.get("gpu_topology")),
                event.get("sample_count"),
                UUID(str(eval_run_id)),
            )
        elif eval_run_id and kind == "generation_batch_done":
            await conn.execute(
                """
                UPDATE eval_runs
                SET generated_sample_count = GREATEST(generated_sample_count, COALESCE($1, 0))
                WHERE id = $2
                """,
                event.get("generated_sample_count"),
                UUID(str(eval_run_id)),
            )
        elif eval_run_id and kind == "scoring_started":
            await conn.execute(
                "UPDATE eval_runs SET state = 'SCORING' WHERE id = $1", UUID(str(eval_run_id))
            )
        elif eval_run_id and kind == "scoring_batch_done":
            await conn.execute(
                """
                UPDATE eval_runs
                SET scored_sample_count = GREATEST(scored_sample_count, COALESCE($1, 0))
                WHERE id = $2
                """,
                event.get("scored_sample_count"),
                UUID(str(eval_run_id)),
            )
        elif eval_run_id and kind == "verdict":
            await conn.execute(
                "UPDATE eval_runs SET state = 'VERDICT_READY'"
                " WHERE id = $1 AND state <> 'SUCCEEDED'",
                UUID(str(eval_run_id)),
            )
        await db.record_event(
            conn,
            submission_id=submission_id,
            stage_attempt_id=attempt_id,
            event_type=event_type,
            severity="INFO",
            message=str(event.get("message") or event.get("type") or "Remote eval event"),
            data=event,
        )


_VERDICT_ARTIFACT_TYPES = {
    "transcript": "EVAL_TRANSCRIPT",
    "generated_samples": "GENERATED_SAMPLES",
    "scoring_results": "SCORING_RESULTS",
    "judge_results": "JUDGE_RESULTS",
    "verdict": "EVAL_VERDICT",
    "remote_logs": "REMOTE_LOGS",
    "progress": "REMOTE_PROGRESS",
}


async def _insert_artifacts(
    conn: Any, submission_id: UUID, attempt_id: UUID, verdict: dict[str, Any]
) -> None:
    # Persists every known artifact URI from the verdict into the artifacts ledger.
    artifacts = verdict.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    metadata_map = verdict.get("artifact_metadata")
    metadata_map = metadata_map if isinstance(metadata_map, dict) else {}
    for name, uri in sorted(artifacts.items()):
        artifact_type = _VERDICT_ARTIFACT_TYPES.get(name)
        if not uri or not artifact_type:
            continue
        meta = metadata_map.get(name)
        meta = meta if isinstance(meta, dict) else {}
        bucket = object_key = None
        if uri.startswith("s3://"):
            backend = "s3"
            bucket, _, object_key = uri.removeprefix("s3://").partition("/")
            bucket, object_key = bucket or None, object_key or None
        elif uri.startswith(("local-cache://", "file://")):
            backend = "local-cache"
        else:
            backend = "hippius"
        size_bytes = meta.get("size_bytes")
        await conn.execute(
            """
            INSERT INTO artifacts (
                id, submission_id, stage_attempt_id, artifact_type, storage_backend,
                uri, bucket, object_key, sha256, size_bytes, content_type
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            uuid4(),
            submission_id,
            attempt_id,
            artifact_type,
            backend,
            uri,
            bucket or (meta.get("bucket") or None),
            object_key or (meta.get("object_key") or None),
            meta.get("sha256") or None,
            size_bytes if isinstance(size_bytes, int) else None,
            meta.get("content_type") or None,
        )


async def _mark_eval_succeeded(
    submission_id: UUID, attempt_id: UUID, eval_run_id: UUID, verdict: dict[str, Any]
) -> None:
    # Persists scores/artifacts and advances to EVAL_WIN or COMPLETE_LOSS.
    next_state = "EVAL_WIN" if verdict.get("challenger_won") else "COMPLETE_LOSS"
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            UPDATE eval_runs
            SET state = 'SUCCEEDED',
                generated_sample_count = COALESCE($1, generated_sample_count),
                scored_sample_count = COALESCE($2, scored_sample_count),
                score_challenger = $3::float8, score_king = $4::float8,
                win_margin = $5::float8, challenger_won = $6,
                valid_turns = $7, total_turns = $8,
                king_vllm_errors = $9, chal_vllm_errors = $10, judge_errors = $11,
                gpu_topology = $12, finished_at = now()
            WHERE id = $13
            """,
            verdict.get("generated_sample_count"),
            verdict.get("scored_sample_count"),
            verdict.get("score_challenger"),
            verdict.get("score_king"),
            _win_margin(verdict),
            verdict.get("challenger_won"),
            verdict.get("valid_turns"),
            verdict.get("total_turns"),
            verdict.get("king_vllm_errors", 0),
            verdict.get("chal_vllm_errors", 0),
            verdict.get("judge_errors", 0),
            verdict.get("gpu_topology", {}),
            eval_run_id,
        )
        await conn.execute(
            """
            UPDATE stage_attempts
            SET state = 'SUCCEEDED', finished_at = now(), result_summary = $1
            WHERE id = $2
            """,
            verdict,
            attempt_id,
        )
        await conn.execute(
            """
            UPDATE model_submissions
            SET state = $1, updated_at = now(),
                finished_at = CASE WHEN $1 = 'COMPLETE_LOSS' THEN now() ELSE finished_at END
            WHERE id = $2
            """,
            next_state,
            submission_id,
        )
        await _insert_artifacts(conn, submission_id, attempt_id, verdict)
        await db.record_event(
            conn,
            submission_id=submission_id,
            stage_attempt_id=attempt_id,
            event_type="eval_succeeded",
            severity="INFO",
            message=f"Eval completed with state {next_state}",
            data=verdict,
        )


async def _mark_eval_failed(
    submission_id: UUID, attempt_id: UUID, eval_run_id: UUID, fault: FaultDecision
) -> None:
    # Fails run/attempt/submission together: retryable -> EVAL_RETRYABLE, else TERMINAL_INVALID.
    attempt_state = "FAILED_RETRYABLE" if fault.retryable else "FAILED_TERMINAL"
    submission_state = "EVAL_RETRYABLE" if fault.retryable else "TERMINAL_INVALID"
    logger.warning(
        f"[eval] eval failed submission={submission_id} eval_run={eval_run_id} "
        f"fault_class={fault.fault_class} fault_code={fault.fault_code} "
        f"retryable={fault.retryable}: {fault.fault_message}"
    )
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            UPDATE eval_runs
            SET state = $1, fault_class = $2, fault_code = $3, fault_message = $4,
                finished_at = now()
            WHERE id = $5
            """,
            attempt_state,
            fault.fault_class,
            fault.fault_code,
            fault.fault_message,
            eval_run_id,
        )
        await conn.execute(
            """
            UPDATE stage_attempts
            SET state = $1, finished_at = now(), fault_class = $2, fault_code = $3,
                fault_message = $4
            WHERE id = $5
            """,
            attempt_state,
            fault.fault_class,
            fault.fault_code,
            fault.fault_message,
            attempt_id,
        )
        await conn.execute(
            """
            UPDATE model_submissions
            SET state = $1, fault_class = $2, fault_code = $3, fault_message = $4,
                retry_count = retry_count + 1, updated_at = now()
            WHERE id = $5
            """,
            submission_state,
            fault.fault_class,
            fault.fault_code,
            fault.fault_message,
            submission_id,
        )
        await db.record_event(
            conn,
            submission_id=submission_id,
            stage_attempt_id=attempt_id,
            event_type="eval_failed",
            severity="ERROR",
            message=fault.fault_message,
            data={
                "fault_class": fault.fault_class,
                "fault_code": fault.fault_code,
                "retryable": fault.retryable,
            },
        )


# ── Follow loop + dispatch ───────────────────────────────────────────────────


def _client(base_url: str) -> RemoteRunClient:
    # Fresh client per run so a broken pool never leaks across duels.
    return RemoteRunClient(
        base_url=base_url,
        run_kind="eval-runs",
        auth_token=get_settings().eval.remote_auth_token,
        timeout_seconds=_REMOTE_TIMEOUT_S,
    )


async def _follow_until_verdict(
    client: RemoteRunClient, *, submission_id: UUID, attempt_id: UUID, remote_run_id: str
) -> dict[str, Any]:
    # Polls events + run state until a verdict; heartbeats the lease; 5 straight errors fail.
    e = get_settings().eval
    seen_event_count = 0
    consecutive_errors = 0
    while True:
        try:
            events = await client.get_events(remote_run_id)
            consecutive_errors = 0
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            consecutive_errors += 1
            if consecutive_errors >= _MAX_POLL_ERRORS:
                raise
            logger.warning(
                f"[eval] transient poll error ({consecutive_errors}/{_MAX_POLL_ERRORS}), "
                f"retrying: {exc}"
            )
            await asyncio.sleep(min(10 * consecutive_errors, 60))
            continue

        for event in events[seen_event_count:]:
            await _record_remote_event(submission_id, attempt_id, event)
            await db.heartbeat_attempt(attempt_id, e.lease_seconds)
            if event.get("type") == "verdict":
                return event
        seen_event_count = max(seen_event_count, len(events))

        try:
            remote_state = await client.get_run(remote_run_id)
            consecutive_errors = 0
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            consecutive_errors += 1
            if consecutive_errors >= _MAX_POLL_ERRORS:
                raise
            logger.warning(
                f"[eval] transient state error ({consecutive_errors}/{_MAX_POLL_ERRORS}), "
                f"retrying: {exc}"
            )
            await asyncio.sleep(min(10 * consecutive_errors, 60))
            continue

        if remote_state.get("type") == "verdict" or remote_state.get("state") in {
            "succeeded",
            "failed",
        }:
            if len(events) == seen_event_count and remote_state.get("type") == "verdict":
                await _record_remote_event(submission_id, attempt_id, remote_state)
            return remote_state
        # Renew the lease even when no new events arrive (generation runs 30+ min silently).
        await db.heartbeat_attempt(attempt_id, e.lease_seconds)
        await asyncio.sleep(e.remote_event_poll_seconds)


async def _complete_eval(
    submission_id: UUID, attempt_id: UUID, eval_run_id: UUID, verdict: dict[str, Any]
) -> None:
    # Routes a final remote document to the success or classified-failure path.
    if verdict.get("state") == "succeeded":
        await _mark_eval_succeeded(submission_id, attempt_id, eval_run_id, verdict)
        return
    fault = classify_failure_verdict(verdict)
    await _mark_eval_failed(submission_id, attempt_id, eval_run_id, fault)
    await asyncio.to_thread(
        _notify_failure,
        submission_id=submission_id,
        eval_run_id=eval_run_id,
        fault=fault,
        remote_run_id=str(verdict.get("eval_run_id") or eval_run_id),
    )


async def _dispatch_once() -> bool:
    # Claims at most one duel, POSTs it to the GPU worker, and follows it to completion.
    claimed = await _claim_next_eval()
    if not claimed:
        return False

    e = get_settings().eval
    remote_run_id = str(claimed.eval_run_id)
    client = _client(claimed.host_base_url)
    try:
        await client.ready()
        start_response = await client.start_run(claimed.request.model_dump(mode="json"))
        remote_run_id = str(start_response.get("remote_run_id") or claimed.eval_run_id)
        await _set_remote_run_id(claimed.eval_run_id, remote_run_id)
        await db.heartbeat_attempt(claimed.attempt_id, e.lease_seconds)
        verdict = await _follow_until_verdict(
            client,
            submission_id=claimed.submission_id,
            attempt_id=claimed.attempt_id,
            remote_run_id=remote_run_id,
        )
        await _complete_eval(
            claimed.submission_id, claimed.attempt_id, claimed.eval_run_id, verdict
        )
        return True
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        fault = broken_stream_fault(str(exc))
        await _mark_eval_failed(
            claimed.submission_id, claimed.attempt_id, claimed.eval_run_id, fault
        )
        await asyncio.to_thread(
            _notify_failure,
            submission_id=claimed.submission_id,
            eval_run_id=claimed.eval_run_id,
            fault=fault,
            remote_run_id=remote_run_id,
        )
        return True
    finally:
        await client.aclose()


async def run_dispatcher() -> None:
    # Main dispatch loop; refuses to start unauthenticated unless the GPU mock is on.
    s = get_settings()
    if not s.eval.remote_auth_token and not s.remote_eval.mock_auto_verdict:
        raise RuntimeError("ALBEDO_EVAL_REMOTE_AUTH_TOKEN is empty and mock mode is off")
    logger.info(f"[eval] dispatcher started - worker_id={s.eval.worker_id} poll={_POLL_S}s")
    while True:
        try:
            did_work = await _dispatch_once()
            if not did_work:
                await asyncio.sleep(_POLL_S)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive across unexpected errors
            logger.exception(f"[eval] unhandled dispatch error, retrying in {_POLL_S}s: {exc}")
            await asyncio.sleep(_POLL_S)


# ── Janitor: requeue + sweep + reconcile (old cron apps as in-process timers) ─


async def _queue_pre_eval_passed(limit: int = 100) -> int:
    # PRE_EVAL_PASSED -> EVAL_QUEUED so the dispatcher can claim them.
    e = get_settings().eval
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        rows = await conn.fetch(
            """
            SELECT id FROM model_submissions ms
            WHERE state = 'PRE_EVAL_PASSED'
            ORDER BY priority ASC, updated_at ASC
            LIMIT $1
            FOR UPDATE OF ms SKIP LOCKED
            """,
            limit,
        )
        for row in rows:
            await conn.execute(
                "UPDATE model_submissions SET state = 'EVAL_QUEUED', updated_at = now()"
                " WHERE id = $1",
                row["id"],
            )
            await db.record_event(
                conn,
                submission_id=row["id"],
                stage_attempt_id=None,
                event_type="eval_queued_from_pre_eval",
                severity="INFO",
                message=f"Queued for eval by {e.worker_id} after passing pre-eval",
                data={"worker_id": e.worker_id},
            )
        return len(rows)


async def _requeue_retryable(limit: int = 100) -> int:
    # EVAL_RETRYABLE -> EVAL_QUEUED under the retry cap; at/over the cap -> TERMINAL_INVALID.
    e = get_settings().eval
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        capped_rows = await conn.fetch(
            """
            SELECT id, retry_count, fault_class, fault_code, fault_message
            FROM model_submissions ms
            WHERE state = 'EVAL_RETRYABLE'
              AND retry_count >= $1
              AND NOT EXISTS (
                  SELECT 1 FROM stage_attempts sa
                  WHERE sa.submission_id = ms.id
                    AND sa.stage = 'EVAL'
                    AND sa.state IN ('CLAIMED', 'RUNNING')
              )
            FOR UPDATE OF ms SKIP LOCKED
            """,
            e.max_retry_count,
        )
        for row in capped_rows:
            logger.error(
                f"[eval] terminal retry-cap submission={row['id']} attempts={row['retry_count']} "
                f"last_fault_class={row['fault_class']} last_fault_code={row['fault_code']}: "
                f"{row['fault_message']}"
            )
            message = (
                f"Eval retry cap reached: retry_count={row['retry_count']} "
                f"max_retry_count={e.max_retry_count}; "
                f"last_fault={row['fault_code']}: {row['fault_message']}"
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'TERMINAL_INVALID',
                    fault_class = COALESCE(fault_class, 'INFRA_FAULT'),
                    fault_code = 'eval_retry_limit_exceeded',
                    fault_message = $1, finished_at = now(), updated_at = now()
                WHERE id = $2
                """,
                message,
                row["id"],
            )
            await db.record_event(
                conn,
                submission_id=row["id"],
                stage_attempt_id=None,
                event_type="eval_retry_limit_exceeded",
                severity="ERROR",
                message=message,
                data={
                    "worker_id": e.worker_id,
                    "retry_count": row["retry_count"],
                    "max_retry_count": e.max_retry_count,
                    "previous_fault_class": row["fault_class"],
                    "previous_fault_code": row["fault_code"],
                    "previous_fault_message": row["fault_message"],
                },
            )

        rows = await conn.fetch(
            """
            SELECT id, fault_class, fault_code, fault_message
            FROM model_submissions ms
            WHERE state = 'EVAL_RETRYABLE'
              AND retry_count < $1
              AND NOT EXISTS (
                  SELECT 1 FROM stage_attempts sa
                  WHERE sa.submission_id = ms.id
                    AND sa.stage = 'EVAL'
                    AND sa.state IN ('CLAIMED', 'RUNNING')
              )
            ORDER BY priority ASC, updated_at ASC
            LIMIT $2
            FOR UPDATE OF ms SKIP LOCKED
            """,
            e.max_retry_count,
            limit,
        )
        for row in rows:
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'EVAL_QUEUED', fault_class = NULL, fault_code = NULL,
                    fault_message = NULL, updated_at = now()
                WHERE id = $1
                """,
                row["id"],
            )
            await db.record_event(
                conn,
                submission_id=row["id"],
                stage_attempt_id=None,
                event_type="eval_retry_requeued",
                severity="INFO",
                message=f"Eval retry requeued by {e.worker_id}",
                data={
                    "worker_id": e.worker_id,
                    "max_retry_count": e.max_retry_count,
                    "previous_fault_class": row["fault_class"],
                    "previous_fault_code": row["fault_code"],
                    "previous_fault_message": row["fault_message"],
                },
            )
        return len(rows)


async def _sweep_abandoned() -> int:
    # Expired RUNNING eval leases -> ABANDONED attempt, retryable run + submission.
    e = get_settings().eval
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        rows = await conn.fetch(
            """
            SELECT sa.id AS attempt_id, sa.submission_id, er.id AS eval_run_id
            FROM stage_attempts sa
            JOIN model_submissions ms ON ms.id = sa.submission_id
            LEFT JOIN eval_runs er ON er.stage_attempt_id = sa.id
            WHERE sa.stage = 'EVAL'
              AND sa.state = 'RUNNING'
              AND sa.lease_expires_at < now()
              AND ms.state = 'EVAL_RUNNING'
            FOR UPDATE OF sa, ms SKIP LOCKED
            """
        )
        for row in rows:
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = 'ABANDONED', finished_at = now(),
                    fault_class = 'REMOTE_EVAL_FAULT',
                    fault_code = 'eval_attempt_lease_expired',
                    fault_message = 'Eval attempt lease expired before completion'
                WHERE id = $1
                """,
                row["attempt_id"],
            )
            if row["eval_run_id"]:
                await conn.execute(
                    """
                    UPDATE eval_runs
                    SET state = 'FAILED_RETRYABLE', finished_at = now(),
                        fault_class = 'REMOTE_EVAL_FAULT',
                        fault_code = 'eval_attempt_lease_expired',
                        fault_message = 'Eval attempt lease expired before completion'
                    WHERE id = $1
                    """,
                    row["eval_run_id"],
                )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'EVAL_RETRYABLE',
                    fault_class = 'REMOTE_EVAL_FAULT',
                    fault_code = 'eval_attempt_lease_expired',
                    fault_message = 'Eval attempt lease expired before completion',
                    retry_count = retry_count + 1, updated_at = now()
                WHERE id = $1
                """,
                row["submission_id"],
            )
            await db.record_event(
                conn,
                submission_id=row["submission_id"],
                stage_attempt_id=row["attempt_id"],
                event_type="eval_attempt_abandoned",
                severity="WARN",
                message="Eval attempt lease expired before completion",
                data={"worker_id": e.worker_id, "eval_run_id": str(row["eval_run_id"] or "")},
            )
        return len(rows)


async def _reconcile_running(limit: int = 10) -> int:
    # Re-attaches to in-flight remote runs after a dispatcher crash and follows to verdict.
    p = await db.pool()
    rows = await p.fetch(
        """
        SELECT er.id AS eval_run_id, er.remote_run_id, er.submission_id,
               er.stage_attempt_id AS attempt_id, h.base_url
        FROM eval_runs er
        JOIN stage_attempts sa ON sa.id = er.stage_attempt_id
        JOIN model_submissions ms ON ms.id = er.submission_id
        JOIN remote_gpu_hosts h ON h.id = er.remote_host_id
        WHERE er.remote_run_id IS NOT NULL
          AND er.state IN ('DISPATCHED', 'GENERATING', 'SCORING', 'VERDICT_READY')
          AND sa.state = 'RUNNING'
          AND ms.state = 'EVAL_RUNNING'
        ORDER BY er.started_at ASC
        LIMIT $1
        """,
        limit,
    )
    reconciled = 0
    for row in rows:
        client = _client(row["base_url"])
        try:
            verdict = await _follow_until_verdict(
                client,
                submission_id=row["submission_id"],
                attempt_id=row["attempt_id"],
                remote_run_id=row["remote_run_id"],
            )
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            logger.warning(
                f"[eval] reconcile skipped submission={row['submission_id']} "
                f"eval_run={row['eval_run_id']} remote_run={row['remote_run_id']}: {exc}"
            )
            continue
        finally:
            await client.aclose()

        if verdict.get("type") == "verdict" or verdict.get("state") in {"succeeded", "failed"}:
            await _complete_eval(
                row["submission_id"], row["attempt_id"], row["eval_run_id"], verdict
            )
            reconciled += 1
    return reconciled


async def run_janitor() -> None:
    # 60s timers folding the old requeuer/sweeper/reconciler cron apps into one loop.
    logger.info(f"[eval] janitor started - interval={_JANITOR_INTERVAL_S}s")
    while True:
        try:
            queued = await _queue_pre_eval_passed()
            requeued = await _requeue_retryable()
            swept = await _sweep_abandoned()
            reconciled = await _reconcile_running()
            if queued or requeued or swept or reconciled:
                logger.info(
                    f"[eval] janitor tick: queued={queued} requeued={requeued} "
                    f"swept={swept} reconciled={reconciled}"
                )
        except Exception:  # noqa: BLE001 - keep the janitor alive across DB blips
            logger.exception("[eval] janitor tick failed")
        await asyncio.sleep(_JANITOR_INTERVAL_S)
