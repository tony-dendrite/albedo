"""Async Postgres layer for the used_hotkeys guard ledger."""
from __future__ import annotations

import asyncio
import json

import asyncpg
from loguru import logger as log

from chain_guard import uploads as guard_s3
from chain_guard.swap import SwapEvent, describe


async def is_used(conn: asyncpg.Connection, hotkey: str) -> bool:
    """True if this hotkey is in the ledger (legacy or already burned by eval)."""
    return bool(await conn.fetchval("SELECT 1 FROM used_hotkeys WHERE hotkey = $1", hotkey))


async def record_legacy(pool: asyncpg.Pool, rows: list[tuple[str, int, str]], ignore_to_block: int) -> int:
    """Seed the ledger with every (hotkey, block, raw_payload) committed at/before ``ignore_to_block``.

    Idempotent: a hotkey is recorded at most once. Returns the number of rows newly inserted.
    """
    legacy = [(hk, block, raw) for hk, block, raw in rows if block <= ignore_to_block]
    if not legacy:
        return 0
    log.debug(f"[chain-guard] record_legacy starting: candidates={len(legacy)} ignore_to_block={ignore_to_block}")
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
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
    log.debug(f"[chain-guard] record_legacy done: inserted={inserted} of candidates={len(legacy)}")
    return inserted


async def load_uid_state(pool: asyncpg.Pool) -> dict[int, tuple[str, int | None]]:
    """uid -> (hotkey, registration_block) of the most recently seen miner per uid."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (uid) uid, hotkey, registration_block
            FROM miners
            WHERE uid IS NOT NULL
            ORDER BY uid, updated_at DESC
            """
        )
    return {row["uid"]: (row["hotkey"], row["registration_block"]) for row in rows}


async def record_swaps(pool: asyncpg.Pool, swaps: list[SwapEvent], netuid: int, block_number: int) -> int:
    """Ledger each swap's NEW hotkey (source='swap') so its commits are rejected before eval,
    log a dashboard-visible ERROR event, and publish an S3 detection report.

    Idempotent: a hotkey already in the ledger is skipped. Returns newly ledgered count.
    """
    inserted = 0
    for swap in swaps:
        detail = swap.detail()
        async with pool.acquire() as conn:
            status = await conn.execute(
                """
                INSERT INTO used_hotkeys (hotkey, netuid, block_number, raw_payload, source)
                VALUES ($1, $2, $3, $4, 'swap')
                ON CONFLICT (hotkey) DO NOTHING
                """,
                swap.new_hotkey,
                netuid,
                block_number,
                json.dumps(detail),
            )
            if not status.endswith("1"):
                continue
            inserted += 1
            await conn.execute(
                """
                INSERT INTO events (event_type, severity, message, data)
                VALUES ('hotkey_swap_detected', 'ERROR', $1, $2::jsonb)
                """,
                describe(detail),
                json.dumps({**detail, "netuid": netuid, "detected_at_block": block_number}),
            )
        log.warning("[chain-guard] {}", describe(detail))
        await asyncio.to_thread(guard_s3.put_detection, swap.new_hotkey, block_number, detail)
    return inserted


async def refresh_registration_blocks(pool: asyncpg.Pool, snapshot: list[tuple[int, str, int]]) -> int:
    """Persist BlockAtRegistration on the miners rows of the CURRENT metagraph occupants.

    Old rows (previous occupants) are never touched — their stored registration_block is
    exactly what find_swaps compares against. Returns the number of rows updated.
    """
    uids = [uid for uid, _hk, _rb in snapshot]
    hotkeys = [hk for _uid, hk, _rb in snapshot]
    reg_blocks = [rb for _uid, _hk, rb in snapshot]
    async with pool.acquire() as conn:
        status = await conn.execute(
            """
            UPDATE miners m
            SET registration_block = s.reg_block
            FROM unnest($1::int[], $2::text[], $3::bigint[]) AS s(uid, hotkey, reg_block)
            WHERE m.hotkey = s.hotkey
              AND m.uid = s.uid
              AND m.registration_block IS DISTINCT FROM s.reg_block
            """,
            uids,
            hotkeys,
            reg_blocks,
        )
    return int(status.rsplit(" ", 1)[-1] or 0)
