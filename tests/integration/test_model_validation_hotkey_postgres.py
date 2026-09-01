from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from model_validation import db as mv_db

pytestmark = pytest.mark.integration

SCHEMA = Path(__file__).resolve().parents[2] / "schema.sql"


def _database_url() -> str:
    url = os.environ.get("ALBEDO_TEST_DATABASE_URL")
    if not url:
        pytest.skip("ALBEDO_TEST_DATABASE_URL is not set")
    return url


async def _reset(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        has = await conn.fetchval("SELECT to_regclass('public.model_submissions')")
        if has is None:
            await conn.execute(SCHEMA.read_text(encoding="utf-8"))
        await conn.execute(
            "TRUNCATE model_submissions, chain_commits, miners RESTART IDENTITY CASCADE"
        )


async def _seed(
    pool: asyncpg.Pool, *, state: str, commit_block: int, registration_block: int | None
) -> None:
    async with pool.acquire() as conn:
        miner_id = await conn.fetchval(
            """
            INSERT INTO miners (hotkey, uid, netuid, registration_block)
            VALUES ('miner-hotkey', 7, 1, $1)
            ON CONFLICT (hotkey) DO UPDATE SET registration_block = EXCLUDED.registration_block
            RETURNING id
            """,
            registration_block,
        )
        commit_id = await conn.fetchval(
            """
            INSERT INTO chain_commits (
                netuid, block_number, block_hash, uid, hotkey,
                commit_payload, model_uri, payload_hash
            )
            VALUES (1, $1, '0xold', 7, 'miner-hotkey', '{}'::jsonb, 's3://models/old', $2)
            RETURNING id
            """,
            commit_block,
            f"payload-{uuid4().hex[:8]}",
        )
        await conn.execute(
            """
            INSERT INTO model_submissions (
                miner_id, chain_commit_id, netuid, uid, hotkey, model_uri,
                state, idempotency_key
            )
            VALUES ($1, $2, 1, 7, 'miner-hotkey', 's3://models/old', $3, $4)
            """,
            miner_id,
            commit_id,
            state,
            f"idem-{uuid4().hex[:8]}",
        )


def _check(
    *, commit_block: int, registration_block: int | None, state: str = "HIPPIUS_VALIDATED"
) -> bool:
    url = _database_url()

    async def scenario() -> bool:
        pool = await asyncpg.create_pool(dsn=url, min_size=1, max_size=2)
        try:
            await _reset(pool)
            await _seed(
                pool,
                state=state,
                commit_block=commit_block,
                registration_block=registration_block,
            )
            return await mv_db.hotkey_validated(pool, "miner-hotkey")
        finally:
            await pool.close()

    return asyncio.run(scenario())


def test_hotkey_validated_blocks_when_registration_block_unknown():
    assert _check(commit_block=101, registration_block=None) is True


def test_hotkey_validated_blocks_within_current_registration():
    assert _check(commit_block=101, registration_block=50) is True


def test_hotkey_validated_allows_after_reregistration():
    assert _check(commit_block=101, registration_block=200) is False


def test_hotkey_validated_ignores_terminal_invalid_submissions():
    assert _check(commit_block=101, registration_block=50, state="TERMINAL_INVALID") is False
