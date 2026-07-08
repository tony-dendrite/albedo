"""Dashboard publisher - builds dashboard.json + state.json from Postgres and pushes to Hippius."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg
from loguru import logger

from albedo import db, s3
from albedo.settings import get_settings

_HISTORY_LIMIT = get_settings().monitor.dashboard_history_limit
# Only surface the current reference-model family (in-place cutover leaves 4B rows in the DB).
_MODEL_FILTER = get_settings().monitor.dashboard_model_filter
_DATA_DIR = Path(get_settings().monitor.dashboard_data_dir)

# Mirrors the judge ensemble (website/js/config.js expects these names).
JUDGE_MODELS = ["z-ai/glm-5.1", "qwen/qwen3.5-397b-a17b", "deepseek/deepseek-v3.2"]

# Artifact types the website knows how to render.
DASHBOARD_ARTIFACT_TYPES = [
    "EVAL_VERDICT",
    "GENERATED_SAMPLES",
    "SCORING_RESULTS",
    "JUDGE_RESULTS",
    "EVAL_TRANSCRIPT",
    "REMOTE_PROGRESS",
    "REMOTE_LOGS",
    "SANITY_RESULT",
]

QUEUE_STATES = [
    "PRE_EVAL_QUEUED",
    "PRE_EVAL_RUNNING",
    "PRE_EVAL_PASSED",
    "EVAL_QUEUED",
    "EVAL_RUNNING",
]
FAIL_STATES = ["TERMINAL_INVALID", "TERMINAL_INFRA_FAILED"]
ACTIVE_EVAL_STATES = ["QUEUED", "DISPATCHED", "GENERATING", "SCORING", "VERDICT_READY"]

# state.json: per-stage running/queued buckets. Handoff states (HIPPIUS_VALIDATED,
# PRE_EVAL_PASSED) are "queued for the next stage" - exactly what the next dispatcher claims.
STAGE_BUCKETS: dict[str, dict[str, tuple[str, ...]]] = {
    "hippius_validate": {
        "queued": ("SUBMITTED", "HIPPIUS_RETRYABLE"),
        "running": ("HIPPIUS_RUNNING",),
    },
    "pre_eval": {
        "queued": ("HIPPIUS_VALIDATED", "PRE_EVAL_QUEUED", "PRE_EVAL_RETRYABLE"),
        "running": ("PRE_EVAL_RUNNING",),
    },
    "eval": {
        "queued": ("PRE_EVAL_PASSED", "EVAL_QUEUED", "EVAL_RETRYABLE"),
        "running": ("EVAL_RUNNING",),
    },
}


def _num(value: Any) -> float | None:
    # NUMERIC -> float for JSON, preserving None.
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return value


def _json_default(value: Any) -> Any:
    # Serializer for the DB types json.dumps cannot handle.
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")


def _public_url(uri: str | None, base: str) -> str | None:
    # s3:// -> public gateway URL; http(s) passes through; local URIs are not fetchable.
    if not uri:
        return None
    if uri.startswith("s3://"):
        return f"{base.rstrip('/')}/{uri[len('s3://') :]}"
    if uri.startswith(("http://", "https://")):
        return uri
    return None


def _eval_artifact_tail(uri: str | None) -> tuple[str, str] | None:
    # Extracts (eval_run_id, artifact_name) from .../eval/<run_id>/<name> URIs.
    if not uri or "/eval/" not in uri:
        return None
    tail = uri.split("/eval/", 1)[1]
    if "/" not in tail:
        return None
    eval_run_id, name = tail.split("/", 1)
    return (eval_run_id, name) if eval_run_id and name else None


# ── dashboard.json ───────────────────────────────────────────────────────────


async def _king_version_map(conn: asyncpg.Connection) -> dict[int, int]:
    # Raw king_versions.version -> display regnal number within the current model family.
    rows = await conn.fetch(
        """
        SELECT kv.version,
               ROW_NUMBER() OVER (ORDER BY kv.version ASC) - 1 AS regnal
        FROM king_versions kv
        JOIN model_submissions ms ON ms.id = kv.submission_id
        WHERE ms.model_uri LIKE $1
        """,
        f"%{_MODEL_FILTER}%",
    )
    return {int(row["version"]): int(row["regnal"]) for row in rows}


async def _reign(conn: asyncpg.Connection, version_map: dict[int, int]) -> dict[str, Any]:
    # The ACTIVE reign's members with their crowning scores.
    rows = await conn.fetch(
        """
        SELECT rm.slot, rm.uid, rm.hotkey, rm.weight_bps, rm.model_hash,
               kv.version AS king_version, ms.model_uri,
               er.score_challenger, er.score_king, er.id AS eval_run_id
        FROM reigns r
        JOIN reign_members rm ON rm.reign_id = r.id
        JOIN king_versions kv ON kv.id = rm.king_version_id
        JOIN model_submissions ms ON ms.id = rm.submission_id
        LEFT JOIN eval_runs er ON er.id = kv.eval_run_id
        WHERE r.state = 'ACTIVE'
        ORDER BY rm.slot ASC
        """
    )
    members = [
        {
            "king_version": version_map.get(row["king_version"]),
            "model_uri": row["model_uri"],
            "model_hash": row["model_hash"],
            "hotkey": row["hotkey"],
            "uid": row["uid"],
            "weight_bps": row["weight_bps"],
            "score_challenger": _num(row["score_challenger"]),
            "score_king": _num(row["score_king"]),
            "eval_run_id": str(row["eval_run_id"]) if row["eval_run_id"] else None,
        }
        for row in rows
    ]
    return {"members": members}


async def _artifacts_for(
    conn: asyncpg.Connection, submission_ids: list, base: str
) -> dict[str, dict[str, str]]:
    # Public artifact URLs keyed by eval_run_id (when parseable) and submission_id.
    if not submission_ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT submission_id, artifact_type, uri
        FROM artifacts
        WHERE submission_id = ANY($1::uuid[]) AND artifact_type = ANY($2::text[])
        """,
        submission_ids,
        DASHBOARD_ARTIFACT_TYPES,
    )
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        url = _public_url(row["uri"], base)
        if not url:
            continue
        keys = [str(row["submission_id"])]
        parsed = _eval_artifact_tail(row["uri"])
        if parsed:
            keys.insert(0, parsed[0])
        for key in keys:
            out.setdefault(key, {}).setdefault(row["artifact_type"], url)
    return out


async def _eval_runs(
    conn: asyncpg.Connection, base: str, version_map: dict[int, int]
) -> list[dict[str, Any]]:
    # Completed duel history with score breakdowns and artifact links.
    rows = await conn.fetch(
        """
        SELECT er.id AS eval_run_id, er.submission_id,
               er.challenger_won, er.score_challenger, er.score_king, er.win_margin,
               er.valid_turns, er.total_turns, er.chal_vllm_errors, er.king_vllm_errors,
               er.finished_at,
               ms.uid, ms.hotkey, ms.model_uri,
               sa.result_summary,
               ckv.version AS crowned_king_version,
               kms.uid AS king_uid, kms.hotkey AS king_hotkey, kms.model_uri AS king_model_uri,
               kkv.version AS king_king_version
        FROM eval_runs er
        JOIN model_submissions ms ON ms.id = er.submission_id
        LEFT JOIN stage_attempts sa ON sa.id = er.stage_attempt_id
        LEFT JOIN king_versions ckv ON ckv.eval_run_id = er.id
        LEFT JOIN model_submissions kms ON kms.id = er.king_submission_id
        LEFT JOIN LATERAL (
            SELECT version FROM king_versions
            WHERE submission_id = er.king_submission_id
            ORDER BY version DESC LIMIT 1
        ) kkv ON true
        WHERE er.state = 'SUCCEEDED'
          AND ms.model_uri LIKE $1
        ORDER BY er.finished_at DESC NULLS LAST
        LIMIT $2
        """,
        f"%{_MODEL_FILTER}%",
        _HISTORY_LIMIT,
    )
    artifacts = await _artifacts_for(conn, [row["submission_id"] for row in rows], base)

    runs: list[dict[str, Any]] = []
    for row in rows:
        verdict = row["result_summary"] if isinstance(row["result_summary"], dict) else {}
        breakdown = verdict.get("score_breakdown")
        breakdown = breakdown if isinstance(breakdown, dict) else {}
        runs.append(
            {
                "eval_run_id": str(row["eval_run_id"]),
                "challenger_won": row["challenger_won"],
                "coronated": row["crowned_king_version"] is not None,
                "king_version": version_map.get(row["crowned_king_version"]),
                "score_challenger": _num(row["score_challenger"]),
                "score_king": _num(row["score_king"]),
                "win_margin": _num(row["win_margin"]),
                "finished_at": row["finished_at"],
                "model_uri": row["model_uri"],
                "hotkey": row["hotkey"],
                "uid": row["uid"],
                "total_turns": row["total_turns"],
                "valid_turns": row["valid_turns"],
                "chal_vllm_errors": row["chal_vllm_errors"],
                "king_vllm_errors": row["king_vllm_errors"],
                "scored_sample_count": verdict.get("scored_sample_count"),
                "judge_errors": verdict.get("judge_errors"),
                "required_win_margin": _num(verdict.get("required_win_margin")),
                "scoring_mode": verdict.get("scoring_mode"),
                "score_breakdown": {
                    "by_judge": breakdown.get("by_judge", {}),
                    "by_metric": breakdown.get("by_metric", {}),
                    "by_category": breakdown.get("by_category", {}),
                },
                "king": {
                    "king_version": version_map.get(row["king_king_version"]),
                    "model_uri": row["king_model_uri"],
                    "uid": row["king_uid"],
                    "hotkey": row["king_hotkey"],
                },
                "artifacts": artifacts.get(
                    str(row["eval_run_id"]), artifacts.get(str(row["submission_id"]), {})
                ),
            }
        )
    return runs


async def _current_eval(conn: asyncpg.Connection) -> dict[str, Any] | None:
    # The in-flight duel, if any.
    row = await conn.fetchrow(
        """
        SELECT er.id AS eval_run_id, er.state, er.sample_count, er.generated_sample_count,
               er.started_at, ms.id AS submission_id, ms.model_uri, ms.hotkey, ms.uid
        FROM eval_runs er
        JOIN model_submissions ms ON ms.id = er.submission_id
        WHERE er.state = ANY($1::text[])
          AND ms.model_uri LIKE $2
        ORDER BY er.started_at DESC NULLS LAST
        LIMIT 1
        """,
        ACTIVE_EVAL_STATES,
        f"%{_MODEL_FILTER}%",
    )
    if not row:
        return None
    return {
        "eval_run_id": str(row["eval_run_id"]),
        "submission_id": str(row["submission_id"]),
        "state": row["state"],
        "sample_count": row["sample_count"],
        "generated_sample_count": row["generated_sample_count"],
        "started_at": row["started_at"],
        "model_uri": row["model_uri"],
        "hotkey": row["hotkey"],
        "uid": row["uid"],
    }


async def _queue(
    conn: asyncpg.Connection, exclude_submission_id: str | None
) -> list[dict[str, Any]]:
    # Submissions waiting between pre-eval and eval, minus the one currently duelling.
    rows = await conn.fetch(
        """
        SELECT id AS submission_id, state, model_uri, hotkey, uid, created_at
        FROM model_submissions
        WHERE state = ANY($1::text[])
          AND model_uri LIKE $2
        ORDER BY priority ASC, created_at ASC
        """,
        QUEUE_STATES,
        f"%{_MODEL_FILTER}%",
    )
    return [
        {
            "submission_id": str(row["submission_id"]),
            "state": row["state"],
            "model_uri": row["model_uri"],
            "hotkey": row["hotkey"],
            "uid": row["uid"],
            "created_at": row["created_at"],
        }
        for row in rows
        if str(row["submission_id"]) != exclude_submission_id
    ]


async def _fails(conn: asyncpg.Connection, base: str) -> list[dict[str, Any]]:
    # Recent terminal failures with their fault details and artifact links.
    rows = await conn.fetch(
        """
        SELECT ms.id AS submission_id, ms.state, ms.model_uri, ms.hotkey, ms.uid,
               ms.fault_class, ms.fault_code, ms.fault_message, ms.model_hash, ms.updated_at,
               (SELECT er.id FROM eval_runs er
                WHERE er.submission_id = ms.id
                ORDER BY er.started_at DESC NULLS LAST LIMIT 1) AS eval_run_id
        FROM model_submissions ms
        WHERE ms.state = ANY($1::text[])
          AND ms.model_uri LIKE $2
        ORDER BY ms.updated_at DESC
        LIMIT $3
        """,
        FAIL_STATES,
        f"%{_MODEL_FILTER}%",
        _HISTORY_LIMIT,
    )
    artifacts = await _artifacts_for(conn, [row["submission_id"] for row in rows], base)
    return [
        {
            "submission_id": str(row["submission_id"]),
            "eval_run_id": str(row["eval_run_id"]) if row["eval_run_id"] else None,
            "model_uri": row["model_uri"],
            "hotkey": row["hotkey"],
            "uid": row["uid"],
            "state": row["state"],
            "fault_class": row["fault_class"],
            "fault_code": row["fault_code"],
            "fault_message": row["fault_message"],
            "model_hash": row["model_hash"],
            "updated_at": row["updated_at"],
            "artifacts": artifacts.get(str(row["submission_id"]), {}),
        }
        for row in rows
    ]


async def _stats(conn: asyncpg.Connection) -> dict[str, Any]:
    # Distinct models evaluated: a model re-evaluated several times counts once.
    n = await conn.fetchval(
        """
        SELECT count(DISTINCT er.submission_id)
        FROM eval_runs er
        JOIN model_submissions ms ON ms.id = er.submission_id
        WHERE er.state = 'SUCCEEDED'
          AND ms.model_uri LIKE $1
        """,
        f"%{_MODEL_FILTER}%",
    )
    return {"evaluated": int(n or 0)}


async def build_dashboard(conn: asyncpg.Connection) -> dict[str, Any]:
    # Full dashboard.json document (reign, history, queue, fails, live eval).
    m = get_settings().monitor
    base = m.dashboard_artifact_base_url
    current = await _current_eval(conn)
    version_map = await _king_version_map(conn)
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "chain": {"netuid": m.dashboard_netuid, "judge_models": list(JUDGE_MODELS)},
        "stats": await _stats(conn),
        "reign": await _reign(conn, version_map),
        "current_eval": current,
        "queue": await _queue(conn, current["submission_id"] if current else None),
        "eval_runs": await _eval_runs(conn, base, version_map),
        "fails": await _fails(conn, base),
    }


# ── state.json ───────────────────────────────────────────────────────────────


async def build_state(conn: asyncpg.Connection) -> dict[str, Any]:
    # Live pipeline status: running/queued submissions bucketed per stage.
    tracked = sorted(
        {s for stage in STAGE_BUCKETS.values() for bucket in stage.values() for s in bucket}
    )
    rows = await conn.fetch(
        """
        SELECT id AS submission_id, state, uid, hotkey, model_uri, updated_at
        FROM model_submissions
        WHERE state = ANY($1::text[])
          AND model_uri LIKE $2
        ORDER BY updated_at DESC
        """,
        tracked,
        f"%{_MODEL_FILTER}%",
    )
    stages: dict[str, dict[str, list]] = {
        name: {"running": [], "queued": []} for name in STAGE_BUCKETS
    }
    for row in rows:
        item = {
            "submission_id": str(row["submission_id"]),
            "uid": row["uid"],
            "hotkey": row["hotkey"],
            "model_uri": row["model_uri"],
            "state": row["state"],
            "updated_at": row["updated_at"],
        }
        for stage_name, buckets in STAGE_BUCKETS.items():
            for bucket, states in buckets.items():
                if row["state"] in states:
                    stages[stage_name][bucket].append(item)
    counts = {
        name: {b: len(items) for b, items in buckets.items()} for name, buckets in stages.items()
    }
    return {"updated_at": datetime.now(UTC).isoformat(), "counts": counts, "stages": stages}


# ── publish loop ─────────────────────────────────────────────────────────────


def _upload(key: str, body: bytes) -> bool:
    # Best-effort no-cache public upload to the Hippius bucket; False keeps the file local-only.
    if get_settings().monitor.mock:
        # Hard guard for flow tests: never publish dashboards, even with real S3 creds in .env.
        logger.info(f"[monitor] ALBEDO_MONITOR_MOCK=true - kept local {key} (not uploaded)")
        return False
    client = s3.client()
    if client is None:
        logger.warning(f"[monitor] ALBEDO_S3_* unset; kept local {key} (not uploaded)")
        return False
    try:
        client.put_object(
            Bucket=get_settings().s3.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            CacheControl="no-cache, must-revalidate",
            ACL="public-read",
        )
        return True
    except Exception as exc:  # noqa: BLE001 - never wedge the loop on an upload
        logger.error(f"[monitor] upload failed for {key}: {exc}")
        return False


async def _signature(conn: asyncpg.Connection) -> tuple:
    # Cheap change-detection poll so a tick only rebuilds when the DB moved.
    row = await conn.fetchrow(
        """
        SELECT (SELECT max(updated_at) FROM model_submissions) AS ms_max,
               (SELECT count(*)        FROM model_submissions) AS ms_count,
               (SELECT max(finished_at) FROM eval_runs)        AS er_max,
               (SELECT max(version)     FROM reigns)           AS reign_max
        """
    )
    return (row["ms_max"], row["ms_count"], row["er_max"], row["reign_max"])


async def _generate() -> None:
    # Builds both documents, writes them locally, and uploads them to Hippius.
    p = await db.pool()
    async with p.acquire() as conn:
        dashboard = await build_dashboard(conn)
        state = await build_state(conn)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploads: dict[str, bool] = {}
    for name, data in (("dashboard.json", dashboard), ("state.json", state)):
        body = json.dumps(data, default=_json_default, indent=2).encode("utf-8")
        (_DATA_DIR / name).write_bytes(body)
        uploads[name] = await asyncio.to_thread(_upload, f"data/{name}", body)

    members = dashboard["reign"]["members"]
    king_version = max(
        (m["king_version"] for m in members if m.get("king_version") is not None), default=None
    )
    current = dashboard["current_eval"]
    logger.info(
        f"[monitor] published update: evaluated={dashboard['stats']['evaluated']} "
        f"reign_king=v{king_version} eval_runs={len(dashboard['eval_runs'])} "
        f"queued={len(dashboard['queue'])} "
        f"current_eval={current['state'] if current else 'idle'} "
        f"fails={len(dashboard['fails'])} "
        f"upload={'ok' if all(uploads.values()) else 'FAILED'}"
    )


async def run_publisher() -> None:
    # Signature-polled publish loop; only rebuilds and uploads when the DB changed.
    interval = get_settings().monitor.monitor_interval_s
    logger.info(f"[monitor] publisher started - interval={interval}s filter={_MODEL_FILTER}")
    last_sig: tuple | None = None
    while True:
        try:
            p = await db.pool()
            async with p.acquire() as conn:
                sig = await _signature(conn)
            if sig != last_sig:
                await _generate()
                last_sig = sig
        except Exception as exc:  # noqa: BLE001 - keep the loop alive; the next tick retries
            logger.error(f"[monitor] tick failed: {exc}")
        await asyncio.sleep(interval)
