"""Each stage gets its own retry budget: passing a stage resets the submission's retry count.

Runs against a real Postgres; set ALBEDO_TEST_DATABASE_URL to a superuser URL, e.g.
`docker run -d -e POSTGRES_PASSWORD=t -p 55432:5432 postgres:16-alpine` and
`postgresql://postgres:t@127.0.0.1:55432/postgres`.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from pathlib import Path

import asyncpg
import pytest

from model_validation import db as mv_db
from sanity_service.db import PreEvalRepository

_ADMIN_URL = os.environ.get("ALBEDO_TEST_DATABASE_URL")
_SCHEMA = (Path(__file__).resolve().parents[1] / "schema.sql").read_text()


@pytest.fixture
def db_url():
    if not _ADMIN_URL:
        pytest.skip("ALBEDO_TEST_DATABASE_URL is not set")
    name = f"t_{secrets.token_hex(6)}"
    base, _, _ = _ADMIN_URL.rpartition("/")

    async def run(sql: str, url: str) -> None:
        conn = await asyncpg.connect(url)
        await conn.execute(sql)
        await conn.close()

    asyncio.run(run(f"CREATE DATABASE {name}", _ADMIN_URL))
    asyncio.run(run(_SCHEMA, f"{base}/{name}"))
    yield f"{base}/{name}"
    asyncio.run(run(f"DROP DATABASE {name} WITH (FORCE)", _ADMIN_URL))


async def _submission(url: str, *, state: str, stage: str, retry_count: int) -> tuple:
    conn = await asyncpg.connect(url)
    commit = await conn.fetchval(
        """
        INSERT INTO chain_commits (netuid, block_number, block_hash, uid, hotkey, model_uri,
                                   payload_hash)
        VALUES (97, 100, '0x', 62, 'hk', 'hf://m@abc', $1) RETURNING id
        """,
        secrets.token_hex(8),
    )
    submission = await conn.fetchval(
        """
        INSERT INTO model_submissions (chain_commit_id, netuid, uid, hotkey, model_uri, state,
                                       retry_count, idempotency_key)
        VALUES ($1, 97, 62, 'hk', 'hf://m@abc', $2, $3, $4) RETURNING id
        """,
        commit,
        state,
        retry_count,
        str(uuid.uuid4()),
    )
    attempt = await conn.fetchval(
        """
        INSERT INTO stage_attempts (submission_id, stage, attempt_number, state, started_at)
        VALUES ($1, $2, $3, 'RUNNING', now()) RETURNING id
        """,
        submission,
        stage,
        retry_count + 1,
    )
    await conn.close()
    return submission, attempt


async def _retry_count(url: str, submission) -> int:
    conn = await asyncpg.connect(url)
    value = await conn.fetchval(
        "SELECT retry_count FROM model_submissions WHERE id = $1", submission
    )
    await conn.close()
    return value


def test_passing_validation_resets_the_retry_count(db_url):
    """Four validation infra retries used to leave pre-eval a single attempt (uid 62)."""

    async def go():
        submission, attempt = await _submission(
            db_url, state="HIPPIUS_RUNNING", stage="HIPPIUS", retry_count=4
        )
        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=1)
        await mv_db.mark_done(pool, attempt, {"model_hash": "sha256:" + "a" * 64})
        await pool.close()
        return await _retry_count(db_url, submission)

    assert asyncio.run(go()) == 0


def test_passing_pre_eval_resets_the_retry_count(db_url):
    submission, attempt = asyncio.run(
        _submission(db_url, state="PRE_EVAL_RUNNING", stage="PRE_EVAL", retry_count=2)
    )
    PreEvalRepository(db_url).mark_pre_eval_passed(
        submission_id=submission,
        attempt_id=attempt,
        repo="hf://m",
        digest="sha256:" + "b" * 64,
        responses=[],
        reason="ok",
        timing={},
    )
    assert asyncio.run(_retry_count(db_url, submission)) == 0
