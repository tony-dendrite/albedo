"""Chain ingest - v7 reveal scan, chain_guard hotkey ledger, and submission creation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import UUID

import asyncpg
from loguru import logger

from albedo.db import pool, record_event
from albedo.s3 import put_json
from albedo.settings import get_settings

_BLOCK_HASH_CACHE: dict[int, str] = {}


@dataclass(frozen=True)
class Commit:
    # One well-formed v7 reveal discovered on chain, ready for Postgres.
    netuid: int
    block_number: int
    block_hash: str | None
    extrinsic_hash: str | None
    uid: int
    hotkey: str
    commit_payload: dict[str, Any]
    model_uri: str
    payload_hash: str


# ── Chain reading ────────────────────────────────────────────────────────────


def _connect(network: str) -> Any:
    # Opens the (blocking) subtensor connection; callers wrap in to_thread.
    import bittensor as bt

    logger.info(f"[chain] connecting to bittensor network={network}")
    return bt.Subtensor(network=network)


def _iter_revealed(subtensor: Any, netuid: int) -> Iterator[tuple[str, int, str]]:
    # Yields (hotkey, block, payload) from Commitments.RevealedCommitments, decoding the
    # SCALE compact-length prefix from either hex or latin-1-wrapped raw bytes.
    qm = subtensor.query_map(module="Commitments", name="RevealedCommitments", params=[netuid])
    for k, v in qm:
        hotkey = str(getattr(k, "value", k))
        data = getattr(v, "value", v)
        try:
            for text, block in data:
                raw = (
                    bytes.fromhex(text[2:])
                    if text.startswith(("0x", "0X"))
                    else text.encode("latin-1")
                )
                if not raw:
                    raise ValueError("empty commitment payload")
                mode = raw[0] & 0b11
                offset = 1 if mode == 0 else 2 if mode == 1 else 4
                yield hotkey, int(block), raw[offset:].decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001 - one bad commitment must not stop the scan
            logger.debug(f"[chain] failed to decode revealed commitment for {hotkey}: {exc}")


def _uid_map(subtensor: Any, netuid: int) -> dict[str, int]:
    # hotkey -> uid via the metagraph; empty on failure so the scan just skips those commits.
    try:
        meta = subtensor.metagraph(netuid)
        return {str(n.hotkey): int(n.uid) for n in meta.neurons}
    except Exception as exc:  # noqa: BLE001 - a metagraph blip only delays ingest one tick
        logger.warning(f"[chain] metagraph({netuid}) failed: {exc}")
        return {}


def _block_hash(subtensor: Any, block: int) -> str | None:
    # Cached block-number -> block-hash lookup; None when the RPC fails.
    if block not in _BLOCK_HASH_CACHE:
        try:
            _BLOCK_HASH_CACHE[block] = str(subtensor.get_block_hash(block))
        except Exception as exc:  # noqa: BLE001 - hash is informational, keep the commit
            logger.debug(f"[chain] get_block_hash({block}) failed: {exc}")
            return None
    return _BLOCK_HASH_CACHE[block]


def _parse_v7(data: str, chain_hotkey: str) -> dict[str, Any] | None:
    # Strict `v7|<repo>|sha256:<digest>` parse; None for anything malformed.
    parts = data.split("|")
    if len(parts) != 3 or parts[0] != "v7":
        return None
    _, repo, digest = parts
    if "/" not in repo or not digest.startswith("sha256:"):
        return None
    return {
        "version": "v7",
        "repo": repo,
        "digest": digest,
        "author_hotkey": chain_hotkey,
        "spoofed": False,
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    # sha256 of the canonical (sorted, compact) JSON - the dedup key on (netuid, hotkey).
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _scan_commitments(subtensor: Any, netuid: int, start_block: int) -> list[Commit]:
    # Full revealed-commitments scan narrowed to v7 commits at/after start_block with a known uid.
    uids = _uid_map(subtensor, netuid)
    commits: list[Commit] = []
    n_total = n_skipped = 0
    for hotkey, block, data in _iter_revealed(subtensor, netuid):
        n_total += 1
        payload = _parse_v7(data, hotkey)
        if block < start_block or payload is None:
            n_skipped += 1
            continue
        uid = uids.get(hotkey)
        if uid is None:
            logger.warning(f"[chain] no uid for hotkey={hotkey}; skipping")
            n_skipped += 1
            continue
        commits.append(
            Commit(
                netuid=netuid,
                block_number=block,
                block_hash=_block_hash(subtensor, block),
                extrinsic_hash=None,
                uid=uid,
                hotkey=hotkey,
                commit_payload=payload,
                model_uri=f"{payload['repo']}@{payload['digest']}",
                payload_hash=_payload_hash(payload),
            )
        )
    logger.info(f"[chain] scan: total={n_total} v7_commits={len(commits)} skipped={n_skipped}")
    return commits


def _scan_all_raw(subtensor: Any, netuid: int) -> list[tuple[str, int, str]]:
    # Every revealed commitment, all versions unparsed - feeds the chain_guard legacy backfill.
    return list(_iter_revealed(subtensor, netuid))


# ── chain_guard ledger ───────────────────────────────────────────────────────


async def _record_legacy(rows: list[tuple[str, int, str]], ignore_to_block: int) -> int:
    # Seeds used_hotkeys with every hotkey that committed at/before the cutoff; idempotent.
    legacy = [(hk, block, raw) for hk, block, raw in rows if block <= ignore_to_block]
    if not legacy:
        return 0
    inserted = 0
    p = await pool()
    async with p.acquire() as conn, conn.transaction():
        for hotkey, block, raw in legacy:
            status = await conn.execute(
                """
                INSERT INTO used_hotkeys (hotkey, block_number, raw_payload, source)
                VALUES ($1, $2, $3, 'backfill')
                ON CONFLICT (hotkey) DO NOTHING
                """,
                hotkey,
                block,
                raw,
            )
            if status.endswith("1"):
                inserted += 1
    return inserted


async def _is_used(conn: asyncpg.Connection, hotkey: str) -> bool:
    # True if the hotkey is already in the ledger (legacy or burned by a prior eval).
    return bool(await conn.fetchval("SELECT 1 FROM used_hotkeys WHERE hotkey = $1", hotkey))


# ── Postgres ingest ──────────────────────────────────────────────────────────


async def _insert_new_commits(commits: list[Commit]) -> int:
    # Upserts miners + chain_commits and creates submissions; returns count of new commits.
    if not commits:
        return 0
    inserted = 0
    p = await pool()
    async with p.acquire() as conn, conn.transaction():
        for c in commits:
            miner_id = await conn.fetchval(
                """
                INSERT INTO miners (hotkey, uid, netuid, updated_at)
                VALUES ($1, $2, $3, now())
                ON CONFLICT (hotkey) DO UPDATE SET
                    uid = EXCLUDED.uid, netuid = EXCLUDED.netuid, updated_at = now()
                RETURNING id
                """,
                c.hotkey,
                c.uid,
                c.netuid,
            )
            row = await conn.fetchrow(
                """
                INSERT INTO chain_commits
                    (netuid, block_number, block_hash, extrinsic_hash, uid, hotkey,
                     commit_payload, model_uri, payload_hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (netuid, hotkey, payload_hash) DO UPDATE SET
                    block_number = EXCLUDED.block_number,
                    block_hash = EXCLUDED.block_hash,
                    extrinsic_hash =
                        COALESCE(chain_commits.extrinsic_hash, EXCLUDED.extrinsic_hash),
                    uid = EXCLUDED.uid,
                    commit_payload = EXCLUDED.commit_payload,
                    model_uri = EXCLUDED.model_uri
                RETURNING id, submission_id, (xmax = 0) AS inserted
                """,
                c.netuid,
                c.block_number,
                c.block_hash,
                c.extrinsic_hash,
                c.uid,
                c.hotkey,
                c.commit_payload,
                c.model_uri,
                c.payload_hash,
            )
            if row["inserted"]:
                inserted += 1
            if row["submission_id"] is not None:
                continue
            idempotency_key = f"chain:{c.netuid}:{c.hotkey}:{c.payload_hash}"
            if await _is_used(conn, c.hotkey):
                await _reject_reused_commit(conn, c, miner_id, row["id"], idempotency_key)
                continue
            submission_id = await conn.fetchval(
                """
                INSERT INTO model_submissions (
                    miner_id, chain_commit_id, netuid, uid, hotkey, model_uri,
                    commit_hash, state, idempotency_key
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'SUBMITTED', $8)
                ON CONFLICT (idempotency_key) DO UPDATE SET
                    miner_id = EXCLUDED.miner_id,
                    chain_commit_id = EXCLUDED.chain_commit_id,
                    uid = EXCLUDED.uid,
                    model_uri = EXCLUDED.model_uri,
                    updated_at = now()
                RETURNING id
                """,
                miner_id,
                row["id"],
                c.netuid,
                c.uid,
                c.hotkey,
                c.model_uri,
                c.commit_payload.get("digest"),
                idempotency_key,
            )
            await conn.execute(
                "UPDATE chain_commits SET submission_id = $1"
                " WHERE id = $2 AND submission_id IS NULL",
                submission_id,
                row["id"],
            )
            await record_event(
                conn,
                submission_id=submission_id,
                stage_attempt_id=None,
                event_type="chain_commit_discovered",
                severity="INFO",
                message="Chain reader discovered model commit and created submission",
                data={
                    "chain_commit_id": str(row["id"]),
                    "netuid": c.netuid,
                    "block_number": c.block_number,
                    "block_hash": c.block_hash,
                    "hotkey": c.hotkey,
                    "uid": c.uid,
                    "model_uri": c.model_uri,
                    "payload_hash": c.payload_hash,
                },
            )
    return inserted


async def _reject_reused_commit(
    conn: asyncpg.Connection,
    c: Commit,
    miner_id: UUID,
    chain_commit_id: UUID,
    idempotency_key: str,
) -> None:
    # Records a reused-hotkey commit as TERMINAL_INVALID and publishes the detection to S3.
    prior = await conn.fetchrow(
        "SELECT submission_id, source, block_number FROM used_hotkeys WHERE hotkey = $1",
        c.hotkey,
    )
    submission_id = await conn.fetchval(
        """
        INSERT INTO model_submissions (
            miner_id, chain_commit_id, netuid, uid, hotkey, model_uri, commit_hash,
            state, fault_class, fault_code, fault_message, idempotency_key, finished_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7,
                'TERMINAL_INVALID', 'MINER_FAULT', 'hotkey_reused', $8, $9, now())
        ON CONFLICT (idempotency_key) DO UPDATE SET
            chain_commit_id = EXCLUDED.chain_commit_id,
            model_uri = EXCLUDED.model_uri,
            updated_at = now()
        RETURNING id
        """,
        miner_id,
        chain_commit_id,
        c.netuid,
        c.uid,
        c.hotkey,
        c.model_uri,
        c.commit_payload.get("digest"),
        "hotkey already used - reuse blocked by chain_guard",
        idempotency_key,
    )
    await conn.execute(
        "UPDATE chain_commits SET submission_id = $1 WHERE id = $2 AND submission_id IS NULL",
        submission_id,
        chain_commit_id,
    )
    detail = {
        "chain_commit_id": str(chain_commit_id),
        "submission_id": str(submission_id),
        "netuid": c.netuid,
        "block_number": c.block_number,
        "hotkey": c.hotkey,
        "uid": c.uid,
        "model_uri": c.model_uri,
        "payload_hash": c.payload_hash,
        "prior_submission_id": str(prior["submission_id"])
        if prior and prior["submission_id"]
        else None,
        "prior_source": prior["source"] if prior else None,
        "prior_block_number": prior["block_number"] if prior else None,
    }
    await record_event(
        conn,
        submission_id=submission_id,
        stage_attempt_id=None,
        event_type="hotkey_rejected_reused",
        severity="WARN",
        message="chain_guard rejected commit: hotkey already used",
        data=detail,
    )
    uri = await put_json(f"chain_guard/{c.hotkey}/{c.block_number}/detection.json", detail)
    if uri:
        await conn.execute(
            """
            INSERT INTO artifacts (submission_id, artifact_type, storage_backend, uri, content_type)
            VALUES ($1, 'GUARD_DETECTION', 's3', $2, 'application/json')
            """,
            submission_id,
            uri,
        )


# ── Entry point ──────────────────────────────────────────────────────────────


async def run_ingest() -> None:
    # Startup guard backfill, then a block-paced scan-and-diff loop that never dies.
    cfg = get_settings().chain
    if cfg.mock:
        # Hard offline guard for flow tests: never touch bittensor, never ingest real commits.
        logger.warning("[chain] CHAIN_MOCK=true - ingest disabled, no bittensor connection")
        while True:
            await asyncio.sleep(3600)
    subtensor = await asyncio.to_thread(_connect, cfg.network)
    logger.info(
        f"[chain] ingest started - netuid={cfg.netuid} network={cfg.network} "
        f"start_block={cfg.start_block} ignore_commits_to_block={cfg.ignore_commits_to_block}"
    )

    if cfg.ignore_commits_to_block > 0:
        try:
            raw = await asyncio.to_thread(_scan_all_raw, subtensor, cfg.netuid)
            seeded = await _record_legacy(raw, cfg.ignore_commits_to_block)
            logger.info(
                f"[chain] guard backfill done: scanned={len(raw)} seeded_blocked_hotkeys={seeded}"
            )
        except Exception:  # noqa: BLE001 - a backfill failure must not silently leave hotkeys unblocked
            logger.exception("[chain] guard backfill failed, hotkeys may be unblocked")
    else:
        logger.info("[chain] guard backfill skipped (ignore_commits_to_block unset/0)")

    last_block: int | None = None
    while True:
        try:
            cur = await asyncio.to_thread(subtensor.get_current_block)
            if cur != last_block:
                commits = await asyncio.to_thread(
                    _scan_commitments, subtensor, cfg.netuid, cfg.start_block
                )
                n_new = await _insert_new_commits(commits)
                logger.info(f"[chain] block={cur} scanned={len(commits)} new={n_new}")
                last_block = cur
        except Exception as exc:  # noqa: BLE001 - keep the loop alive across RPC/DB blips
            logger.opt(exception=True).warning(f"[chain] tick failed ({exc}) - retrying")
        await asyncio.sleep(cfg.poll_interval_s)
