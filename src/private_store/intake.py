"""Apply on-chain private-store signals to private_registrations rows.

Called by chain_reader every tick. RevealedCommitments re-yields every signal
on every scan, so everything here is idempotent and a signal that cannot be
applied yet (e.g. ready before credentials) is simply retried next tick.
"""

from __future__ import annotations

from functools import lru_cache

import asyncpg
from loguru import logger as log

from chain_guard import db as guard
from chain_reader.chain import PrivateSignal
from private_store.contracts import (
    parse_activation_pubkey,
    parse_ready_signal,
    registration_id,
)
from private_store.settings import PrivateStoreSettings


@lru_cache(maxsize=1)
def _settings() -> PrivateStoreSettings:
    return PrivateStoreSettings()


async def handle_signals(pool: asyncpg.Pool, signals: list[PrivateSignal]) -> int:
    if not signals:
        return 0
    applied = 0
    for signal in signals:
        try:
            handler = _apply_activate if signal.kind == "activate" else _apply_ready
            applied += await handler(pool, signal)
        except Exception as exc:
            log.warning(
                "[private-store] {} signal from {} rejected: {}", signal.kind, signal.hotkey, exc
            )
    return applied


async def _apply_activate(pool: asyncpg.Pool, signal: PrivateSignal) -> int:
    if signal.uid is None:
        return 0  # not in the metagraph (yet); retried next tick
    settings = _settings()
    submission_pubkey = parse_activation_pubkey(signal.payload)
    rid = registration_id(
        netuid=signal.netuid, hotkey=signal.hotkey, chain_generation=settings.chain_generation
    )
    async with pool.acquire() as conn:
        if await guard.is_used(conn, signal.hotkey):
            return 0
        inserted = await conn.fetchval(
            """
            INSERT INTO private_registrations
                (netuid, uid, hotkey, registration_id, activation_block, submission_pubkey)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (registration_id) DO NOTHING
            RETURNING id
            """,
            signal.netuid,
            signal.uid,
            signal.hotkey,
            rid,
            signal.block_number,
            submission_pubkey.hex(),
        )
    if inserted is not None:
        log.info("[private-store] activated registration {} hotkey {}", rid, signal.hotkey)
    return int(inserted is not None)


async def _apply_ready(pool: asyncpg.Pool, signal: PrivateSignal) -> int:
    settings = _settings()
    manifest_sha256 = parse_ready_signal(signal.payload)
    rid = registration_id(
        netuid=signal.netuid, hotkey=signal.hotkey, chain_generation=settings.chain_generation
    )
    async with pool.acquire() as conn:
        updated = await conn.fetchval(
            """
            UPDATE private_registrations
            SET state = 'READY', manifest_sha256 = $2, ready_block = $3,
                ready_block_hash = $4, updated_at = now()
            WHERE registration_id = $1 AND state = 'CREDENTIALED'
              AND $3 > activation_block
            RETURNING id
            """,
            rid,
            manifest_sha256,
            signal.block_number,
            signal.block_hash,
        )
    if updated is not None:
        log.info("[private-store] ready for registration {} manifest {}", rid, manifest_sha256)
    return int(updated is not None)
