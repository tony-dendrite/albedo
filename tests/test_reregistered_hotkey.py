"""A deregistered hotkey may not come back: miners have to register a new hotkey.

Runs against a real Postgres (the SQL is the thing under test); set ALBEDO_TEST_DATABASE_URL
to a superuser URL, e.g. `docker run -d -e POSTGRES_PASSWORD=t -p 55432:5432 postgres:16-alpine`
and `postgresql://postgres:t@127.0.0.1:55432/postgres`.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from pathlib import Path

import asyncpg
import pytest
from nacl.signing import SigningKey

from chain_guard import db as guard_db
from chain_guard import swap as guard_swap
from chain_guard import uploads as guard_uploads
from chain_reader import db as chain_db
from chain_reader.chain import Commit
from model_validation import db as mv_db
from private_store.crypto import encode_ss58_public_key

_ADMIN_URL = os.environ.get("ALBEDO_TEST_DATABASE_URL")
_SCHEMA = (Path(__file__).resolve().parents[1] / "schema.sql").read_text()

HOTKEY = encode_ss58_public_key(bytes(SigningKey(b"r" * 32).verify_key))
OTHER = encode_ss58_public_key(bytes(SigningKey(b"w" * 32).verify_key))
OLD_REG, NEW_REG = 1_000, 2_000


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def pool_factory(monkeypatch):
    if not _ADMIN_URL:
        pytest.skip("ALBEDO_TEST_DATABASE_URL is not set")
    monkeypatch.setattr(guard_uploads, "put_detection", lambda *a, **k: None)
    name = f"t_{secrets.token_hex(6)}"
    base, _, _ = _ADMIN_URL.rpartition("/")

    async def setup():
        admin = await asyncpg.connect(_ADMIN_URL)
        await admin.execute(f"CREATE DATABASE {name}")
        await admin.close()
        conn = await asyncpg.connect(f"{base}/{name}")
        await conn.execute(_SCHEMA)
        await conn.close()

    _run(setup())
    yield lambda: asyncpg.create_pool(f"{base}/{name}", min_size=1, max_size=2)

    async def teardown():
        admin = await asyncpg.connect(_ADMIN_URL)
        await admin.execute(f"DROP DATABASE {name} WITH (FORCE)")
        await admin.close()

    _run(teardown())


async def _miner(conn, hotkey: str, *, uid: int, registration_block: int | None) -> None:
    await conn.execute(
        "INSERT INTO miners (hotkey, uid, netuid, registration_block) VALUES ($1, $2, 97, $3)",
        hotkey,
        uid,
        registration_block,
    )


async def _submission(conn, hotkey: str, *, block: int, state: str, fault_code=None):
    commit_id = await conn.fetchval(
        """
        INSERT INTO chain_commits (netuid, block_number, block_hash, uid, hotkey, model_uri,
                                   payload_hash)
        VALUES (97, $1, '0x', 44, $2, 'hf://m', $3) RETURNING id
        """,
        block,
        hotkey,
        secrets.token_hex(8),
    )
    await conn.execute(
        """
        INSERT INTO model_submissions (chain_commit_id, netuid, uid, hotkey, model_uri, state,
                                       fault_class, fault_code, idempotency_key)
        VALUES ($1, 97, 44, $2, $3, $4, $5, $6, $7)
        """,
        commit_id,
        hotkey,
        f"hf://m-{block}",
        state,
        "MINER_FAULT" if fault_code else None,
        fault_code,
        str(uuid.uuid4()),
    )


def test_a_hotkey_with_a_submission_from_an_earlier_registration_is_rejected(pool_factory):
    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, HOTKEY, uid=44, registration_block=NEW_REG)
            await _submission(conn, HOTKEY, block=1_500, state="COMPLETE_LOSS")
            await _submission(conn, HOTKEY, block=2_100, state="SUBMITTED")
        result = await mv_db.hotkey_reregistered(pool, HOTKEY)
        await pool.close()
        return result

    assert _run(go()) is True


def test_a_hotkey_submitting_again_within_its_registration_is_not_rejected(pool_factory):
    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, HOTKEY, uid=44, registration_block=OLD_REG)
            await _submission(conn, HOTKEY, block=1_500, state="COMPLETE_LOSS")
            await _submission(conn, HOTKEY, block=1_600, state="SUBMITTED")
            await _miner(conn, OTHER, uid=45, registration_block=None)  # never refreshed
            await _submission(conn, OTHER, block=1_700, state="SUBMITTED")
        result = (
            await mv_db.hotkey_reregistered(pool, HOTKEY),
            await mv_db.hotkey_reregistered(pool, OTHER),
        )
        await pool.close()
        return result

    assert _run(go()) == (False, False)


def test_the_rejection_is_not_a_strike(pool_factory):
    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, HOTKEY, uid=44, registration_block=NEW_REG)
            await _submission(
                conn,
                HOTKEY,
                block=2_100,
                state="TERMINAL_INVALID",
                fault_code="hotkey_reregistered",
            )
        fails = await mv_db.hotkey_preeval_fail_count(pool, HOTKEY)
        await pool.close()
        return fails

    assert _run(go()) == 0


def test_refresh_moves_the_registration_block_even_while_the_uid_is_stale(pool_factory):
    """The miners row keeps its old uid until the hotkey commits again, so the refresh must
    match on hotkey alone, or the check above would see the old registration block."""

    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, HOTKEY, uid=142, registration_block=OLD_REG)
        await guard_db.refresh_registration_blocks(pool, [(44, HOTKEY, NEW_REG)])
        block = await pool.fetchval(
            "SELECT registration_block FROM miners WHERE hotkey = $1", HOTKEY
        )
        await pool.close()
        return block

    assert _run(go()) == NEW_REG


async def _tick_guard(pool, snapshot, owned: set[str]) -> list:
    """The chain reader's order: detect swaps on the prior state, confirm, ledger, refresh."""
    candidates = guard_swap.find_swaps(await guard_db.load_uid_state(pool), snapshot)
    confirmed = [s for s in candidates if s.old_hotkey not in owned]  # confirm_swaps
    await guard_db.record_swaps(pool, confirmed, 97, 9_999)
    await guard_db.refresh_registration_blocks(pool, snapshot)
    return candidates


def test_a_hotkey_swap_is_still_banned(pool_factory):
    """swap_hotkey keeps BlockAtRegistration; the guard must ban the new hotkey even when it
    already has a stale miners row."""

    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, HOTKEY, uid=7, registration_block=OLD_REG)
            await _miner(conn, OTHER, uid=142, registration_block=500)
        candidates = await _tick_guard(pool, [(7, OTHER, OLD_REG)], owned=set())
        commit = Commit(97, 1_600, "0x", None, 7, OTHER, {}, "hf://w@r", "p-w")
        await chain_db.insert_new_commits(pool, [commit])
        fault = await pool.fetchval(
            "SELECT fault_code FROM model_submissions WHERE hotkey = $1", OTHER
        )
        await pool.close()
        return candidates, fault

    candidates, fault = _run(go())
    assert [(s.uid, s.old_hotkey, s.new_hotkey) for s in candidates] == [(7, HOTKEY, OTHER)]
    assert fault == "hotkey_swap"


def test_reregistering_elsewhere_in_the_block_someone_takes_the_old_uid_is_not_a_swap(
    pool_factory,
):
    """The refresh gives a stale-uid row its new registration block, so its old uid can look
    swapped; the ownership check (the hotkey is still registered) must reject it."""

    async def go():
        pool = await pool_factory()
        async with pool.acquire() as conn:
            await _miner(conn, HOTKEY, uid=7, registration_block=OLD_REG)
        snapshot = [(44, HOTKEY, NEW_REG), (7, OTHER, NEW_REG)]
        first = await _tick_guard(pool, snapshot, owned={HOTKEY})
        second = await _tick_guard(pool, snapshot, owned={HOTKEY})
        banned = await pool.fetchval("SELECT count(*) FROM used_hotkeys")
        await pool.close()
        return first, second, banned

    first, second, banned = _run(go())
    assert first == []
    assert [(s.uid, s.old_hotkey) for s in second] == [(7, HOTKEY)]  # candidate only
    assert banned == 0
