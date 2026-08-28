"""Real-Postgres tests for the destructive reaper/sweep SQL.

These prove the highest-harm property against the actual WHERE/ORDER/OFFSET
(not a mocked fetch): the loser reaper only ever targets COMPLETE_LOSS models
beyond the keep window, and NEVER a winner or a model still in eval.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg
import pytest

from private_store import controller

pytestmark = pytest.mark.integration

SCHEMA = Path(__file__).resolve().parents[2] / "schema.sql"


def _database_url() -> str:
    url = os.environ.get("ALBEDO_TEST_DATABASE_URL")
    if not url:
        pytest.skip("ALBEDO_TEST_DATABASE_URL is not set")
    return url


class _RecordingUploads:
    """Stands in for R2: records which prefixes would be wiped."""

    def __init__(self) -> None:
        self.wiped: list[str] = []

    def cleanup_model_prefix(self, prefix: str) -> int:
        self.wiped.append(prefix)
        return 1


def _deps(keep: int) -> controller.Deps:
    uploads = _RecordingUploads()
    settings = type("S", (), {"keep_recent_losers": keep, "upload_window_seconds": 86_400.0})()
    return controller.Deps(
        settings=settings, gateway=None, mailbox=None, uploads=uploads, cipher=None
    )


async def _reset(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        with SCHEMA.open() as handle:
            has = await conn.fetchval("SELECT to_regclass('public.private_registrations')")
            if has is None:
                await conn.execute(handle.read())
        await conn.execute(
            "TRUNCATE private_registrations, model_submissions, chain_commits, miners"
            " RESTART IDENTITY CASCADE"
        )


async def _seed_submission(conn: asyncpg.Connection, hotkey: str, state: str) -> str:
    miner_id = await conn.fetchval(
        "INSERT INTO miners (hotkey, uid, netuid, updated_at) VALUES ($1, 1, 97, now())"
        " ON CONFLICT (hotkey) DO UPDATE SET updated_at = now() RETURNING id",
        hotkey,
    )
    uri = f"s3://b/models/registrations/{hotkey}"
    commit_id = await conn.fetchval(
        "INSERT INTO chain_commits (netuid, block_number, block_hash, uid, hotkey, model_uri,"
        " payload_hash) VALUES (97, 1, '0x00', 1, $1, $2, $3) RETURNING id",
        hotkey,
        uri,
        f"ph:{hotkey}",
    )
    return await conn.fetchval(
        "INSERT INTO model_submissions (miner_id, chain_commit_id, netuid, uid, hotkey, model_uri,"
        " state, idempotency_key, finished_at) VALUES ($1, $2, 97, 1, $3, $4, $5, $6, now())"
        " RETURNING id",
        miner_id,
        commit_id,
        hotkey,
        uri,
        state,
        f"idem:{hotkey}",
    )


async def _seed_registration(
    conn: asyncpg.Connection, hotkey: str, rid: str, submission_id, state: str
) -> None:
    await conn.execute(
        "INSERT INTO private_registrations (netuid, uid, hotkey, registration_id,"
        " activation_block, submission_pubkey, state, submission_id)"
        " VALUES (97, 1, $1, $2, 1, $3, $4, $5)",
        hotkey,
        rid,
        "00" * 32,
        state,
        submission_id,
    )


def test_reaper_spares_winners_and_in_eval_models():
    async def run() -> None:
        pool = await asyncpg.create_pool(dsn=_database_url(), min_size=1, max_size=2)
        try:
            await _reset(pool)
            async with pool.acquire() as conn:
                # a winner, a model still running eval, and 25 losers
                win_sub = await _seed_submission(conn, "hk-winner", "COMPLETE_CORONATED")
                await _seed_registration(conn, "hk-winner", "a" * 64, win_sub, "SUBMITTED")
                run_sub = await _seed_submission(conn, "hk-running", "EVAL_RUNNING")
                await _seed_registration(conn, "hk-running", "b" * 64, run_sub, "SUBMITTED")
                loser_rids = []
                for i in range(25):
                    rid = f"{i:064x}"
                    loser_rids.append(rid)
                    sub = await _seed_submission(conn, f"hk-loser-{i}", "COMPLETE_LOSS")
                    await _seed_registration(conn, f"hk-loser-{i}", rid, sub, "SUBMITTED")

            deps = _deps(keep=20)
            reaped = await controller.reap_losers(pool, deps)

            # exactly the 5 oldest losers beyond the 20 kept — never the winner or the running one
            assert reaped == 5
            assert len(deps.uploads.wiped) == 5
            assert all("models/registrations/" in p for p in deps.uploads.wiped)
            for rid in ("a" * 64, "b" * 64):
                assert f"models/registrations/{rid}/" not in deps.uploads.wiped

            async with pool.acquire() as conn:
                reaped_states = await conn.fetch(
                    "SELECT registration_id FROM private_registrations WHERE state = 'REAPED'"
                )
                assert len(reaped_states) == 5
                winner = await conn.fetchval(
                    "SELECT state FROM private_registrations WHERE registration_id = $1", "a" * 64
                )
                running = await conn.fetchval(
                    "SELECT state FROM private_registrations WHERE registration_id = $1", "b" * 64
                )
                assert winner == "SUBMITTED" and running == "SUBMITTED"
        finally:
            await pool.close()

    asyncio.run(run())


def test_reaper_noop_when_losers_within_keep_window():
    async def run() -> None:
        pool = await asyncpg.create_pool(dsn=_database_url(), min_size=1, max_size=2)
        try:
            await _reset(pool)
            async with pool.acquire() as conn:
                for i in range(5):  # fewer than keep=20
                    sub = await _seed_submission(conn, f"hk-l-{i}", "COMPLETE_LOSS")
                    await _seed_registration(conn, f"hk-l-{i}", f"{i:064x}", sub, "SUBMITTED")
            deps = _deps(keep=20)
            assert await controller.reap_losers(pool, deps) == 0
            assert deps.uploads.wiped == []
        finally:
            await pool.close()

    asyncio.run(run())
