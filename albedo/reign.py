"""Reign promotion - crowns EVAL_WIN challengers into slot 1 of a new ACTIVE reign."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from loguru import logger

from albedo import db
from albedo.settings import get_settings

_WORKER_ID = get_settings().reign.worker_id
_POLL_S = get_settings().reign.poll_seconds


@dataclass(frozen=True)
class ReignMember:
    # One king occupying (or about to occupy) a reign slot.
    king_version_id: UUID
    submission_id: UUID
    hotkey: str
    uid: int
    model_hash: str
    previous_slot: int | None = None


@dataclass(frozen=True)
class PlannedReignMember:
    # A member placed into the new reign with its slot and weight share.
    member: ReignMember
    slot: int
    weight_bps: int
    is_challenger: bool = False


# ── Pure reign math (identical to the source worker) ─────────────────────────


def build_reign_plan(
    active_members: list[ReignMember], challenger: ReignMember
) -> list[PlannedReignMember]:
    # Challenger takes slot 1; existing kings shift down, deduped by hotkey/hash, capped at 5.
    active = sorted(active_members, key=lambda member: member.previous_slot or 99)
    filtered = [
        member
        for member in active
        if member.hotkey != challenger.hotkey
        and member.model_hash != challenger.model_hash
        and member.submission_id != challenger.submission_id
    ]
    ordered = [challenger, *filtered[:4]]
    weight_bps = weight_bps_for_member_count(len(ordered))
    return [
        PlannedReignMember(
            member=member,
            slot=index + 1,
            weight_bps=weight_bps[index],
            is_challenger=member.king_version_id == challenger.king_version_id,
        )
        for index, member in enumerate(ordered)
    ]


def weight_bps_for_member_count(member_count: int) -> list[int]:
    # Even 10000-bps split with the remainder spread across the lowest slots.
    if member_count < 0 or member_count > 5:
        raise ValueError("member_count must be between 0 and 5")
    if member_count == 0:
        return []
    base = 10000 // member_count
    remainder = 10000 % member_count
    return [base + (1 if index < remainder else 0) for index in range(member_count)]


def weight_epoch_payload(
    planned_members: list[PlannedReignMember],
) -> tuple[list[int], list[Decimal]]:
    # (uids, weights) for the coronation weight epoch; an empty reign burns to uid 0.
    if not planned_members:
        return [0], [Decimal("1")]
    return (
        [member.member.uid for member in planned_members],
        [Decimal(member.weight_bps) / Decimal(10000) for member in planned_members],
    )


def _retired_members(
    active_members: list[ReignMember], planned_members: list[PlannedReignMember]
) -> list[ReignMember]:
    # Members of the old reign that did not make it into the new one.
    planned_ids = {member.member.king_version_id for member in planned_members}
    return [member for member in active_members if member.king_version_id not in planned_ids]


def _weight_policy(reign_version: int, planned_members: list[PlannedReignMember]) -> dict[str, Any]:
    # Structured description of the split, stored on the weight epoch for auditability.
    return {
        "policy": "five_king_genesis_split_v1",
        "reign_version": reign_version,
        "genesis_rule": "new_challenger_slot_1; shift_existing_down; split_weight_evenly",
        "max_slots": 5,
        "member_count": len(planned_members),
        "empty_reign_burn_uid": 0,
        "slot_weight_bps": {str(member.slot): member.weight_bps for member in planned_members},
    }


def _weight_hash(
    *,
    netuid: int,
    reign_version: int,
    planned_members: list[PlannedReignMember],
    weight_policy: dict[str, Any],
) -> str:
    # Deterministic dedupe key for the (netuid, membership, policy) triple.
    payload = {
        "netuid": netuid,
        "reign_version": reign_version,
        "members": _planned_members_json(planned_members),
        "weight_policy": weight_policy,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _planned_members_json(planned_members: list[PlannedReignMember]) -> list[dict[str, Any]]:
    # JSON-safe member list for events, result summaries, and the weight hash.
    return [
        {
            "slot": member.slot,
            "king_version_id": str(member.member.king_version_id),
            "submission_id": str(member.member.submission_id),
            "hotkey": member.member.hotkey,
            "uid": member.member.uid,
            "model_hash": member.member.model_hash,
            "weight_bps": member.weight_bps,
            "is_challenger": member.is_challenger,
        }
        for member in planned_members
    ]


# ── DB helpers ────────────────────────────────────────────────────────────────


async def _next_version(conn: asyncpg.Connection, table_name: str) -> int:
    # Next monotonic version for reigns/king_versions (callers pass literal table names).
    return int(await conn.fetchval(f"SELECT COALESCE(MAX(version), 0) + 1 FROM {table_name}"))


async def _mark_retryable(
    conn: asyncpg.Connection,
    *,
    submission_id: UUID,
    attempt_id: UUID,
    fault_code: str,
    fault_message: str,
) -> None:
    # Fails the attempt and parks the submission in SET_REIGN_RETRYABLE with backoff.
    await conn.execute(
        """
        UPDATE stage_attempts
        SET state = 'FAILED_RETRYABLE', finished_at = now(), lease_expires_at = NULL,
            fault_class = 'INFRA_FAULT', fault_code = $1, fault_message = $2
        WHERE id = $3
        """,
        fault_code,
        fault_message,
        attempt_id,
    )
    await conn.execute(
        """
        UPDATE model_submissions
        SET state = 'SET_REIGN_RETRYABLE', fault_class = 'INFRA_FAULT',
            fault_code = $1, fault_message = $2,
            retry_count = retry_count + 1, updated_at = now()
        WHERE id = $3
        """,
        fault_code,
        fault_message,
        submission_id,
    )
    await db.record_event(
        conn,
        submission_id=submission_id,
        stage_attempt_id=attempt_id,
        event_type="set_reign_failed_retryable",
        severity="ERROR",
        message=fault_message,
        data={"fault_class": "INFRA_FAULT", "fault_code": fault_code},
    )


# ── Promotion ─────────────────────────────────────────────────────────────────


async def _promote_next_winner(*, lease_seconds: int) -> bool:
    # Claims the oldest EVAL_WIN (or backed-off retryable), validates it, writes the new reign.
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        if not await db.advisory_xact_lock(conn, "reign_promotion"):
            return False

        submission = await conn.fetchrow(
            """
            SELECT id, netuid, uid, hotkey, model_hash, model_uri, priority, created_at
            FROM model_submissions
            WHERE (
                state = 'EVAL_WIN'
                OR (
                    state = 'SET_REIGN_RETRYABLE'
                    AND updated_at <= now()
                        - (LEAST(GREATEST(retry_count, 1), 60) * interval '60 seconds')
                )
            )
            ORDER BY priority ASC, created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """
        )
        if not submission:
            return False

        attempt_id = uuid4()
        attempt_number = await db.next_attempt_number(conn, submission["id"], "SET_REIGN")
        await conn.execute(
            """
            INSERT INTO stage_attempts (
                id, submission_id, stage, attempt_number, state, worker_id,
                lease_expires_at, started_at, input_snapshot
            )
            VALUES ($1, $2, 'SET_REIGN', $3, 'RUNNING', $4,
                    now() + ($5 || ' seconds')::interval, now(), $6)
            """,
            attempt_id,
            submission["id"],
            attempt_number,
            _WORKER_ID,
            str(lease_seconds),
            {
                "submission_id": str(submission["id"]),
                "model_hash": submission["model_hash"],
                "hotkey": submission["hotkey"],
                "uid": submission["uid"],
            },
        )
        await conn.execute(
            """
            UPDATE model_submissions
            SET state = 'SET_REIGN_RUNNING', fault_class = NULL, fault_code = NULL,
                fault_message = NULL, updated_at = now()
            WHERE id = $1
            """,
            submission["id"],
        )
        await db.record_event(
            conn,
            submission_id=submission["id"],
            stage_attempt_id=attempt_id,
            event_type="set_reign_claimed",
            severity="INFO",
            message=f"Set reign claimed by {_WORKER_ID}",
            data={"worker_id": _WORKER_ID},
        )

        eval_run = await conn.fetchrow(
            """
            SELECT id, king_submission_id, king_model_hash, challenger_model_hash,
                   challenger_won, state
            FROM eval_runs
            WHERE submission_id = $1
              AND state = 'SUCCEEDED'
              AND challenger_won IS TRUE
            ORDER BY finished_at DESC NULLS LAST, started_at DESC NULLS LAST
            LIMIT 1
            """,
            submission["id"],
        )
        if not eval_run:
            logger.warning(
                f"[set-reign] bailing (retryable): no winning eval_run for "
                f"submission={submission['id']} hotkey={submission['hotkey']}"
            )
            await _mark_retryable(
                conn,
                submission_id=submission["id"],
                attempt_id=attempt_id,
                fault_code="missing_winning_eval_run",
                fault_message="No successful winning eval run is available for reign promotion",
            )
            return True

        if eval_run["challenger_model_hash"] != submission["model_hash"]:
            logger.warning(
                f"[set-reign] bailing (retryable): challenger model hash mismatch for "
                f"submission={submission['id']} eval_run={eval_run['id']} "
                f"eval_hash={eval_run['challenger_model_hash']} "
                f"submission_hash={submission['model_hash']}"
            )
            await _mark_retryable(
                conn,
                submission_id=submission["id"],
                attempt_id=attempt_id,
                fault_code="challenger_model_hash_mismatch",
                fault_message=(
                    "Winning eval challenger model hash does not match submission model hash"
                ),
            )
            return True

        active_reign = await conn.fetchrow(
            """
            SELECT id, version
            FROM reigns
            WHERE state = 'ACTIVE'
            ORDER BY version DESC
            LIMIT 1
            FOR UPDATE
            """
        )
        if not active_reign:
            logger.warning(
                f"[set-reign] bailing (retryable): no ACTIVE reign for winner "
                f"submission={submission['id']} hotkey={submission['hotkey']}"
            )
            await _mark_retryable(
                conn,
                submission_id=submission["id"],
                attempt_id=attempt_id,
                fault_code="missing_active_reign",
                fault_message="No active reign exists for winner promotion",
            )
            return True

        active_rows = await conn.fetch(
            """
            SELECT rm.slot, rm.king_version_id, rm.submission_id, rm.hotkey,
                   rm.uid, rm.model_hash
            FROM reign_members rm
            JOIN king_versions kv ON kv.id = rm.king_version_id
            WHERE rm.reign_id = $1
            ORDER BY rm.slot ASC
            FOR UPDATE OF rm, kv
            """,
            active_reign["id"],
        )
        active_members = [
            ReignMember(
                previous_slot=row["slot"],
                king_version_id=row["king_version_id"],
                submission_id=row["submission_id"],
                hotkey=row["hotkey"],
                uid=row["uid"],
                model_hash=row["model_hash"],
            )
            for row in active_rows
        ]

        lead = next((member for member in active_members if member.previous_slot == 1), None)
        if not lead:
            logger.warning(
                f"[set-reign] bailing (retryable): active reign {active_reign['id']} "
                f"has no slot 1 lead king for submission={submission['id']}"
            )
            await _mark_retryable(
                conn,
                submission_id=submission["id"],
                attempt_id=attempt_id,
                fault_code="missing_active_lead_king",
                fault_message="Active reign has no slot 1 lead king",
            )
            return True

        if (
            eval_run["king_submission_id"] != lead.submission_id
            or eval_run["king_model_hash"] != lead.model_hash
        ):
            logger.warning(
                f"[set-reign] bailing (retryable): stale winning eval for "
                f"submission={submission['id']} eval_run={eval_run['id']}; did not beat "
                f"current lead king (lead_submission={lead.submission_id})"
            )
            await _mark_retryable(
                conn,
                submission_id=submission["id"],
                attempt_id=attempt_id,
                fault_code="stale_winning_eval",
                fault_message="Winning eval did not beat the current active lead king",
            )
            return True

        artifact = await conn.fetchrow(
            """
            SELECT id
            FROM artifacts
            WHERE submission_id = $1
              AND artifact_type = 'MODEL_MANIFEST'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            submission["id"],
        )
        if not artifact:
            logger.warning(
                f"[set-reign] bailing (retryable): no MODEL_MANIFEST artifact for "
                f"submission={submission['id']} hotkey={submission['hotkey']}"
            )
            await _mark_retryable(
                conn,
                submission_id=submission["id"],
                attempt_id=attempt_id,
                fault_code="missing_challenger_model_artifact",
                fault_message="Winning submission has no MODEL_MANIFEST artifact for king version",
            )
            return True

        king_version_id = uuid4()
        king_version = await _next_version(conn, "king_versions")
        reign_id = uuid4()
        reign_version = await _next_version(conn, "reigns")
        challenger_member = ReignMember(
            king_version_id=king_version_id,
            submission_id=submission["id"],
            hotkey=submission["hotkey"],
            uid=submission["uid"],
            model_hash=submission["model_hash"],
        )
        planned_members = build_reign_plan(active_members, challenger_member)
        challenger_plan = next(member for member in planned_members if member.is_challenger)
        retired = _retired_members(active_members, planned_members)

        await conn.execute(
            """
            INSERT INTO king_versions (
                id, submission_id, model_hash, artifact_id, eval_run_id,
                version, entered_reign_id, entered_slot, activated_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            king_version_id,
            submission["id"],
            submission["model_hash"],
            artifact["id"],
            eval_run["id"],
            king_version,
            reign_id,
            challenger_plan.slot,
            _WORKER_ID,
        )
        await conn.execute(
            "UPDATE reigns SET state = 'SUPERSEDED' WHERE id = $1", active_reign["id"]
        )
        await conn.execute(
            """
            INSERT INTO reigns (
                id, version, reason, trigger_eval_run_id, trigger_submission_id,
                previous_reign_id, state, activated_at
            )
            VALUES ($1, $2, 'CORONATION', $3, $4, $5, 'ACTIVE', now())
            """,
            reign_id,
            reign_version,
            eval_run["id"],
            submission["id"],
            active_reign["id"],
        )
        for planned in planned_members:
            await conn.execute(
                """
                INSERT INTO reign_members (
                    id, reign_id, slot, king_version_id, submission_id,
                    hotkey, uid, model_hash, weight_bps
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                uuid4(),
                reign_id,
                planned.slot,
                planned.member.king_version_id,
                planned.member.submission_id,
                planned.member.hotkey,
                planned.member.uid,
                planned.member.model_hash,
                planned.weight_bps,
            )

        if retired:
            await conn.execute(
                """
                UPDATE king_versions
                SET retired_at = now(),
                    retire_reason = CASE
                        WHEN submission_id = $1 THEN 'REPLACED_DUPLICATE'
                        ELSE 'SHIFTED_OUT'
                    END
                WHERE id = ANY($2::uuid[])
                  AND retired_at IS NULL
                """,
                submission["id"],
                [member.king_version_id for member in retired],
            )

        weight_policy = _weight_policy(reign_version, planned_members)
        weight_hash = _weight_hash(
            netuid=submission["netuid"],
            reign_version=reign_version,
            planned_members=planned_members,
            weight_policy=weight_policy,
        )
        weight_uids, weight_values = weight_epoch_payload(planned_members)
        await conn.execute(
            """
            INSERT INTO weight_epochs (
                id, netuid, reason, reign_id, state, uids, weights, weight_policy, weight_hash
            )
            VALUES ($1, $2, 'CORONATION', $3, 'PENDING', $4, $5, $6, $7)
            ON CONFLICT (netuid, weight_hash) DO NOTHING
            """,
            uuid4(),
            submission["netuid"],
            reign_id,
            weight_uids,
            weight_values,
            weight_policy,
            weight_hash,
        )
        result_summary = {
            "reign_id": str(reign_id),
            "reign_version": reign_version,
            "king_version_id": str(king_version_id),
            "king_version": king_version,
            "members": _planned_members_json(planned_members),
            "retired_king_version_ids": [str(member.king_version_id) for member in retired],
            "weight_hash": weight_hash,
        }
        await conn.execute(
            """
            UPDATE stage_attempts
            SET state = 'SUCCEEDED', finished_at = now(), lease_expires_at = NULL,
                result_summary = $1
            WHERE id = $2
            """,
            result_summary,
            attempt_id,
        )
        await conn.execute(
            """
            UPDATE model_submissions
            SET state = 'REIGN_SET', fault_class = NULL, fault_code = NULL,
                fault_message = NULL, updated_at = now()
            WHERE id = $1
            """,
            submission["id"],
        )
        await db.record_event(
            conn,
            submission_id=submission["id"],
            stage_attempt_id=attempt_id,
            event_type="reign_set",
            severity="INFO",
            message=f"Created active reign version {reign_version}",
            data=result_summary,
        )
        return True


async def run_worker() -> None:
    # Promotion loop: one winner per pass, sleeping when idle; never dies on a bad tick.
    lease_seconds = get_settings().reign.lease_seconds or get_settings().eval.lease_seconds
    logger.info(f"[set-reign] worker started - worker_id={_WORKER_ID} poll={_POLL_S}s")
    while True:
        try:
            did_work = await _promote_next_winner(lease_seconds=lease_seconds)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive across DB blips
            logger.exception(f"[set-reign] iteration failed, continuing: {exc}")
            did_work = False
        if not did_work:
            await asyncio.sleep(_POLL_S)
