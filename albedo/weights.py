"""Weight setter - submits reign weights on-chain, burning deregistered slots to the burn uid."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID, uuid4

import asyncpg
from loguru import logger

from albedo import db
from albedo.settings import get_settings


@dataclass(frozen=True)
class WeightMember:
    # One reign member's expected on-chain identity and weight share.
    slot: int
    uid: int
    hotkey: str
    weight_bps: int


@dataclass(frozen=True)
class WeightPayload:
    # Final uids/weights vector plus the audit policy actually submitted.
    uids: list[int]
    weights: list[float]
    policy: dict[str, Any]


@dataclass(frozen=True)
class ClaimedWeightEpoch:
    # A claimed epoch with the ledger ids needed to mark success/failure.
    epoch_id: UUID
    transaction_id: UUID
    stage_attempt_id: UUID | None
    reign_id: UUID | None
    trigger_submission_id: UUID | None
    netuid: int
    weight_policy: dict[str, Any]


@dataclass(frozen=True)
class SetWeightsResult:
    # Normalized outcome of a set_weights extrinsic.
    success: bool
    message: str = ""
    extrinsic_hash: str | None = None


class ChainClient(Protocol):
    # Minimal chain surface the setter needs; satisfied by the mock and Bittensor clients.
    @property
    def block(self) -> int: ...

    def hotkey_by_uid(self, netuid: int) -> dict[int, str]: ...

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> SetWeightsResult: ...


_MOCK_EPOCH = time.monotonic()


class MockChainClient:
    # Satisfies ChainClient without touching Bittensor - used when ALBEDO_WEIGHT_MOCK=true.

    @property
    def block(self) -> int:
        # Advances ~1 block/s of wall time so rate-limit windows pass in mock flow tests
        # (a frozen height would rate-limit every epoch after the first forever).
        return 1_000_000 + int(time.monotonic() - _MOCK_EPOCH)

    def hotkey_by_uid(self, netuid: int) -> dict[int, str]:
        # Empty metagraph: every slot reads as deregistered and burns.
        return {}

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> SetWeightsResult:
        # Always succeeds with a fake extrinsic hash.
        return SetWeightsResult(success=True, extrinsic_hash="0x" + "ab" * 32)


class BittensorChainClient:
    # Real chain client; every method is blocking and must be called via asyncio.to_thread.
    def __init__(self, *, coldkey: str, hotkey: str, network: str, wallet_path: str = ""):
        import bittensor as bt

        wallet_kwargs = {"name": coldkey, "hotkey": hotkey}
        if wallet_path:
            wallet_kwargs["path"] = wallet_path
        self.wallet = bt.Wallet(**wallet_kwargs)
        self.subtensor = bt.Subtensor(network=network)

    @property
    def block(self) -> int:
        # Current chain height (0 when the subtensor is unreachable).
        return int(getattr(self.subtensor, "block", 0) or 0)

    def hotkey_by_uid(self, netuid: int) -> dict[int, str]:
        # Live uid -> hotkey map from the metagraph, tolerant of both neuron shapes.
        metagraph = self.subtensor.metagraph(netuid)
        neurons = getattr(metagraph, "neurons", None)
        if neurons is not None:
            return {int(neuron.uid): str(neuron.hotkey) for neuron in neurons}
        hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
        return {uid: str(hotkey) for uid, hotkey in enumerate(hotkeys)}

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> SetWeightsResult:
        # Submits the extrinsic and normalizes tuple/object return shapes.
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
        if isinstance(result, tuple):
            success = bool(result[0])
            message = str(result[1] or "") if len(result) > 1 else ""
            return SetWeightsResult(success=success, message=message)
        success = bool(getattr(result, "success", result))
        message = str(getattr(result, "message", "") or "")
        extrinsic_hash = getattr(result, "extrinsic_hash", None) or getattr(result, "hash", None)
        return SetWeightsResult(
            success=success,
            message=message,
            extrinsic_hash=str(extrinsic_hash) if extrinsic_hash else None,
        )


# ── Pure payload math (identical to the source service) ─────────────────────


def build_weight_payload(
    members: list[WeightMember],
    *,
    uid_hotkeys: dict[int, str],
    burn_uid: int,
    base_policy: dict[str, Any] | None = None,
) -> WeightPayload:
    # Maps reign members to uid weights; deregistered slots burn their share to burn_uid.
    policy = dict(base_policy or {})
    policy["burn_uid"] = burn_uid
    policy["deregistered_slots"] = []
    policy["submitted_members"] = []

    if not members:
        policy["empty_reign_burned"] = True
        return WeightPayload(uids=[burn_uid], weights=[1.0], policy=policy)

    by_uid: dict[int, Decimal] = {}
    burned_weight = Decimal("0")
    for member in sorted(members, key=lambda item: item.slot):
        weight = Decimal(member.weight_bps) / Decimal(10000)
        if uid_hotkeys.get(member.uid) == member.hotkey:
            by_uid[member.uid] = by_uid.get(member.uid, Decimal("0")) + weight
            policy["submitted_members"].append(
                {
                    "slot": member.slot,
                    "uid": member.uid,
                    "hotkey": member.hotkey,
                    "weight_bps": member.weight_bps,
                }
            )
        else:
            burned_weight += weight
            policy["deregistered_slots"].append(
                {
                    "slot": member.slot,
                    "expected_uid": member.uid,
                    "expected_hotkey": member.hotkey,
                    "current_hotkey": uid_hotkeys.get(member.uid),
                    "weight_bps": member.weight_bps,
                }
            )

    if burned_weight:
        by_uid[burn_uid] = by_uid.get(burn_uid, Decimal("0")) + burned_weight

    if not by_uid:
        return WeightPayload(uids=[burn_uid], weights=[1.0], policy=policy)

    ordered = sorted(by_uid.items(), key=lambda item: (item[0] != burn_uid, item[0]))
    return WeightPayload(
        uids=[uid for uid, _weight in ordered],
        weights=[float(weight) for _uid, weight in ordered],
        policy=policy,
    )


def validate_weight_payload(payload: WeightPayload) -> None:
    # Rejects empty, mismatched, non-positive, or non-normalized payloads before submission.
    if not payload.uids:
        raise ValueError("weight payload has no uids")
    if len(payload.uids) != len(payload.weights):
        raise ValueError("weight payload uids and weights have different lengths")
    if any(weight <= 0 for weight in payload.weights):
        raise ValueError("weight payload includes a non-positive weight")
    total = sum(payload.weights)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weight payload must sum to 1.0, got {total}")


def periodic_refresh_weight_hash(
    *, netuid: int, reign_id: UUID | None, current_block: int, rate_limit_blocks: int
) -> str:
    # One refresh per rate-limit window per reign - the hash dedupes re-inserts.
    refresh_window = current_block // max(rate_limit_blocks, 1)
    payload = {
        "netuid": netuid,
        "reign_id": str(reign_id) if reign_id else None,
        "reason": "PERIODIC_REFRESH",
        "refresh_window": refresh_window,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


# ── DB access ────────────────────────────────────────────────────────────────


async def _create_periodic_refresh_epoch(
    conn: asyncpg.Connection,
    *,
    netuid: int,
    current_block: int,
    rate_limit_blocks: int,
    burn_uid: int,
) -> asyncpg.Record | None:
    # Synthesizes a PERIODIC_REFRESH epoch from the active reign (or a burn epoch when empty).
    active_reign = await conn.fetchrow(
        "SELECT id, version FROM reigns WHERE state = 'ACTIVE' ORDER BY version DESC LIMIT 1"
    )
    reign_id = active_reign["id"] if active_reign else None
    reign_version = int(active_reign["version"]) if active_reign else None

    members = []
    if reign_id:
        members = await conn.fetch(
            """
            SELECT slot, uid, hotkey, weight_bps
            FROM reign_members
            WHERE reign_id = $1
            ORDER BY slot ASC
            """,
            reign_id,
        )

    if members:
        uids = [int(member["uid"]) for member in members]
        weights = [Decimal(int(member["weight_bps"])) / Decimal(10000) for member in members]
        slot_weight_bps = {str(member["slot"]): int(member["weight_bps"]) for member in members}
    else:
        uids = [burn_uid]
        weights = [Decimal(1)]
        slot_weight_bps = {}

    refresh_window = current_block // max(rate_limit_blocks, 1)
    policy = {
        "policy": "periodic_refresh_v1",
        "burn_uid": burn_uid,
        "current_block": current_block,
        "rate_limit_blocks": rate_limit_blocks,
        "refresh_window": refresh_window,
        "reign_version": reign_version,
        "member_count": len(members),
        "slot_weight_bps": slot_weight_bps,
        "empty_reign_burned": not bool(members),
    }
    weight_hash = periodic_refresh_weight_hash(
        netuid=netuid,
        reign_id=reign_id,
        current_block=current_block,
        rate_limit_blocks=rate_limit_blocks,
    )
    inserted = await conn.fetchrow(
        """
        INSERT INTO weight_epochs (
            id, netuid, reason, reign_id, state, uids, weights, weight_policy, weight_hash
        )
        VALUES ($1, $2, 'PERIODIC_REFRESH', $3, 'PENDING', $4, $5, $6, $7)
        ON CONFLICT (netuid, weight_hash) DO NOTHING
        RETURNING id, netuid, reason, reign_id, uids, weights, weight_policy, weight_hash,
                  NULL::uuid AS trigger_submission_id
        """,
        uuid4(),
        netuid,
        reign_id,
        uids,
        weights,
        policy,
        weight_hash,
    )
    if inserted:
        return inserted
    return await conn.fetchrow(
        """
        SELECT we.id, we.netuid, we.reason, we.reign_id, we.uids, we.weights,
               we.weight_policy, we.weight_hash, NULL::uuid AS trigger_submission_id
        FROM weight_epochs we
        WHERE we.netuid = $1 AND we.weight_hash = $2
        LIMIT 1
        """,
        netuid,
        weight_hash,
    )


async def _claim_next_epoch(*, current_block: int) -> ClaimedWeightEpoch | None:
    # Rate-limited claim: newest PENDING/backed-off epoch wins, older ones are superseded.
    w = get_settings().weights
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        if not await db.advisory_xact_lock(conn, "weight_setter"):
            return None

        recent_marker = await conn.fetchrow(
            """
            SELECT wt.block_number
            FROM weight_transactions wt
            JOIN weight_epochs we ON we.id = wt.weight_epoch_id
            WHERE we.netuid = $1
              AND (wt.state = 'SUCCESS' OR wt.fault_code = 'weight_set_rate_limited')
              AND wt.block_number IS NOT NULL
            ORDER BY wt.block_number DESC
            LIMIT 1
            """,
            w.netuid,
        )
        if recent_marker and current_block - int(recent_marker["block_number"]) < w.set_rate_blocks:
            return None

        epoch = await conn.fetchrow(
            """
            SELECT we.id, we.netuid, we.reason, we.reign_id, we.uids, we.weights,
                   we.weight_policy, we.weight_hash, we.created_at,
                   r.trigger_submission_id
            FROM weight_epochs we
            LEFT JOIN reigns r ON r.id = we.reign_id
            WHERE we.netuid = $1
              AND (
                we.state = 'PENDING'
                OR (
                    we.state = 'FAILED_RETRYABLE'
                    AND we.updated_at <= now() - interval '60 seconds'
                )
              )
            ORDER BY we.created_at DESC, we.id DESC
            FOR UPDATE OF we SKIP LOCKED
            LIMIT 1
            """,
            w.netuid,
        )
        if epoch:
            await conn.execute(
                """
                UPDATE weight_epochs
                SET state = 'FAILED_TERMINAL', updated_at = now(),
                    last_fault_class = NULL,
                    last_fault_code = 'superseded_by_newer_weight_epoch'
                WHERE netuid = $1
                  AND id <> $2
                  AND created_at <= $3
                  AND state IN ('PENDING', 'FAILED_RETRYABLE')
                """,
                w.netuid,
                epoch["id"],
                epoch["created_at"],
            )
        if not epoch:
            epoch = await _create_periodic_refresh_epoch(
                conn,
                netuid=w.netuid,
                current_block=current_block,
                rate_limit_blocks=w.set_rate_blocks,
                burn_uid=w.burn_uid,
            )
        if not epoch:
            return None

        stage_attempt_id = None
        if epoch["trigger_submission_id"]:
            stage_attempt_id = uuid4()
            attempt_number = await db.next_attempt_number(
                conn, epoch["trigger_submission_id"], "WEIGHT_SET"
            )
            await conn.execute(
                """
                INSERT INTO stage_attempts (
                    id, submission_id, stage, attempt_number, state,
                    worker_id, started_at, input_snapshot
                )
                VALUES ($1, $2, 'WEIGHT_SET', $3, 'RUNNING', $4, now(), $5)
                """,
                stage_attempt_id,
                epoch["trigger_submission_id"],
                attempt_number,
                w.worker_id,
                {"weight_epoch_id": str(epoch["id"]), "weight_hash": epoch["weight_hash"]},
            )
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'WEIGHT_SET_RUNNING', fault_class = NULL, fault_code = NULL,
                    fault_message = NULL, updated_at = now()
                WHERE id = $1
                  AND state IN ('REIGN_SET', 'WEIGHT_SET_RETRYABLE')
                """,
                epoch["trigger_submission_id"],
            )

        transaction_id = uuid4()
        await conn.execute(
            """
            INSERT INTO weight_transactions (
                id, weight_epoch_id, stage_attempt_id, wallet_hotkey,
                subtensor_url, state, block_number
            )
            VALUES ($1, $2, $3, $4, $5, 'CREATED', $6)
            """,
            transaction_id,
            epoch["id"],
            stage_attempt_id,
            w.hotkey,
            w.network,
            current_block,
        )
        await conn.execute(
            """
            UPDATE weight_epochs
            SET state = 'RUNNING', attempt_count = attempt_count + 1, updated_at = now(),
                last_fault_class = NULL, last_fault_code = NULL
            WHERE id = $1
            """,
            epoch["id"],
        )
        if epoch["trigger_submission_id"]:
            await db.record_event(
                conn,
                submission_id=epoch["trigger_submission_id"],
                stage_attempt_id=stage_attempt_id,
                event_type="weight_set_claimed",
                severity="INFO",
                message=f"Weight epoch claimed by {w.worker_id}",
                data={
                    "weight_epoch_id": str(epoch["id"]),
                    "weight_transaction_id": str(transaction_id),
                    "current_block": current_block,
                },
            )

        return ClaimedWeightEpoch(
            epoch_id=epoch["id"],
            transaction_id=transaction_id,
            stage_attempt_id=stage_attempt_id,
            reign_id=epoch["reign_id"],
            trigger_submission_id=epoch["trigger_submission_id"],
            netuid=epoch["netuid"],
            weight_policy=epoch["weight_policy"] or {},
        )


async def _active_reign_members(reign_id: UUID | None) -> list[WeightMember]:
    # Members of the claimed epoch's reign, slot order; [] for reignless refreshes.
    if reign_id is None:
        return []
    p = await db.pool()
    rows = await p.fetch(
        """
        SELECT slot, uid, hotkey, weight_bps
        FROM reign_members
        WHERE reign_id = $1
        ORDER BY slot ASC
        """,
        reign_id,
    )
    return [
        WeightMember(
            slot=int(row["slot"]),
            uid=int(row["uid"]),
            hotkey=row["hotkey"],
            weight_bps=int(row["weight_bps"]),
        )
        for row in rows
    ]


async def _mark_success(
    claimed: ClaimedWeightEpoch,
    payload: WeightPayload,
    result: SetWeightsResult,
    block_number: int,
) -> None:
    # Records the extrinsic and coronates the triggering submission.
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            UPDATE weight_transactions
            SET state = 'SUCCESS', extrinsic_hash = $1, block_number = $2, updated_at = now()
            WHERE id = $3
            """,
            result.extrinsic_hash,
            block_number,
            claimed.transaction_id,
        )
        await conn.execute(
            """
            UPDATE weight_epochs
            SET state = 'SUCCESS', uids = $1, weights = $2, weight_policy = $3,
                updated_at = now(), succeeded_at = now(),
                last_fault_class = NULL, last_fault_code = NULL
            WHERE id = $4
            """,
            payload.uids,
            [Decimal(str(weight)) for weight in payload.weights],
            payload.policy,
            claimed.epoch_id,
        )
        if claimed.stage_attempt_id:
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = 'SUCCEEDED', finished_at = now(), result_summary = $1
                WHERE id = $2
                """,
                {
                    "weight_epoch_id": str(claimed.epoch_id),
                    "weight_transaction_id": str(claimed.transaction_id),
                    "uids": payload.uids,
                    "weights": payload.weights,
                    "block_number": block_number,
                    "extrinsic_hash": result.extrinsic_hash,
                },
                claimed.stage_attempt_id,
            )
        if claimed.trigger_submission_id:
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'COMPLETE_CORONATED', fault_class = NULL, fault_code = NULL,
                    fault_message = NULL, updated_at = now(), finished_at = now()
                WHERE id = $1
                """,
                claimed.trigger_submission_id,
            )
            await db.record_event(
                conn,
                submission_id=claimed.trigger_submission_id,
                stage_attempt_id=claimed.stage_attempt_id,
                event_type="weight_set_succeeded",
                severity="INFO",
                message="Weights submitted successfully",
                data={
                    "weight_epoch_id": str(claimed.epoch_id),
                    "weight_transaction_id": str(claimed.transaction_id),
                    "uids": payload.uids,
                    "weights": payload.weights,
                    "block_number": block_number,
                    "extrinsic_hash": result.extrinsic_hash,
                },
            )


async def _mark_failed(
    claimed: ClaimedWeightEpoch, fault_code: str, fault_message: str, block_number: int
) -> None:
    # CHAIN_FAULT retryable failure across transaction, epoch, attempt, and submission.
    p = await db.pool()
    async with p.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            UPDATE weight_transactions
            SET state = 'FAILED_RETRYABLE', block_number = $1, fault_class = 'CHAIN_FAULT',
                fault_code = $2, fault_message = $3, updated_at = now()
            WHERE id = $4
            """,
            block_number,
            fault_code,
            fault_message,
            claimed.transaction_id,
        )
        await conn.execute(
            """
            UPDATE weight_epochs
            SET state = 'FAILED_RETRYABLE', last_fault_class = 'CHAIN_FAULT',
                last_fault_code = $1, updated_at = now()
            WHERE id = $2
            """,
            fault_code,
            claimed.epoch_id,
        )
        if claimed.stage_attempt_id:
            await conn.execute(
                """
                UPDATE stage_attempts
                SET state = 'FAILED_RETRYABLE', finished_at = now(), fault_class = 'CHAIN_FAULT',
                    fault_code = $1, fault_message = $2
                WHERE id = $3
                """,
                fault_code,
                fault_message,
                claimed.stage_attempt_id,
            )
        if claimed.trigger_submission_id:
            await conn.execute(
                """
                UPDATE model_submissions
                SET state = 'WEIGHT_SET_RETRYABLE', fault_class = 'CHAIN_FAULT',
                    fault_code = $1, fault_message = $2, updated_at = now()
                WHERE id = $3
                """,
                fault_code,
                fault_message,
                claimed.trigger_submission_id,
            )
            await db.record_event(
                conn,
                submission_id=claimed.trigger_submission_id,
                stage_attempt_id=claimed.stage_attempt_id,
                event_type="weight_set_failed_retryable",
                severity="ERROR",
                message=fault_message,
                data={
                    "weight_epoch_id": str(claimed.epoch_id),
                    "weight_transaction_id": str(claimed.transaction_id),
                    "fault_class": "CHAIN_FAULT",
                    "fault_code": fault_code,
                    "block_number": block_number,
                },
            )


# ── Worker loop ──────────────────────────────────────────────────────────────


async def _run_once(chain: ChainClient) -> bool:
    # One pass: claim an epoch, build+validate the payload, submit, record the outcome.
    w = get_settings().weights
    current_block = await asyncio.to_thread(lambda: chain.block)
    claimed = await _claim_next_epoch(current_block=current_block)
    if not claimed:
        return False

    try:
        members = await _active_reign_members(claimed.reign_id)
        uid_hotkeys = await asyncio.to_thread(chain.hotkey_by_uid, claimed.netuid)
        payload = build_weight_payload(
            members,
            uid_hotkeys=uid_hotkeys,
            burn_uid=w.burn_uid,
            base_policy=claimed.weight_policy,
        )
        validate_weight_payload(payload)
        result = await asyncio.to_thread(
            chain.set_weights, netuid=claimed.netuid, uids=payload.uids, weights=payload.weights
        )
    except Exception as exc:  # noqa: BLE001 - any submit failure marks the epoch retryable
        logger.exception(f"[weight-setter] payload build failed: {exc}")
        await _mark_failed(
            claimed, "weight_set_exception", f"{type(exc).__name__}: {exc}", current_block
        )
        return True

    if result.success:
        await _mark_success(claimed, payload, result, current_block)
    else:
        fault_code = "weight_set_rate_limited" if not result.message else "weight_set_rejected"
        fault_message = result.message or "set_weights returned false without a message"
        await _mark_failed(claimed, fault_code, fault_message, current_block)
    return True


async def run_worker() -> None:
    # Weight-setting loop; real Bittensor client unless ALBEDO_WEIGHT_MOCK=true.
    w = get_settings().weights
    if w.mock:
        chain: ChainClient = MockChainClient()
        logger.info("[weight-setter] using MockChainClient (ALBEDO_WEIGHT_MOCK=true)")
    else:
        if not w.coldkey or not w.hotkey:
            raise RuntimeError("set ALBEDO_WEIGHT_COLDKEY and ALBEDO_WEIGHT_HOTKEY")
        chain = await asyncio.to_thread(
            lambda: BittensorChainClient(
                coldkey=w.coldkey, hotkey=w.hotkey, network=w.network, wallet_path=w.wallet_path
            )
        )
    logger.info(
        f"[weight-setter] started - netuid={w.netuid} set_rate_blocks={w.set_rate_blocks} "
        f"burn_uid={w.burn_uid} poll={w.poll_seconds}s"
    )
    while True:
        try:
            did_work = await _run_once(chain)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive across chain/DB blips
            logger.exception(f"[weight-setter] iteration failed, continuing: {exc}")
            did_work = False
        if not did_work:
            await asyncio.sleep(w.poll_seconds)
