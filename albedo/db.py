"""One asyncpg pool plus the shared state-machine helpers every loop uses."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from albedo.settings import get_settings

_pool: asyncpg.Pool | None = None


async def pool() -> asyncpg.Pool:
    # Lazily creates the process-wide pool; jsonb codec set up so dicts pass straight through.
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=get_settings().db.dsn, min_size=1, max_size=8, init=_init_conn
        )
    return _pool


async def _init_conn(conn: asyncpg.Connection) -> None:
    # Registers jsonb <-> dict codecs on every pooled connection.
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def close_pool() -> None:
    # Clean shutdown hook for the backend supervisor.
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def advisory_xact_lock(conn: asyncpg.Connection, name: str) -> bool:
    # True if this transaction got the named advisory lock (non-blocking); serializes claim paths.
    return bool(await conn.fetchval("SELECT pg_try_advisory_xact_lock(hashtext($1))", name))


async def next_attempt_number(conn: asyncpg.Connection, submission_id: UUID, stage: str) -> int:
    # Next 1-based attempt number for a submission/stage.
    return int(
        await conn.fetchval(
            """
            SELECT COALESCE(MAX(attempt_number), 0) + 1
            FROM stage_attempts WHERE submission_id = $1 AND stage = $2
            """,
            submission_id,
            stage,
        )
    )


async def record_event(
    conn: asyncpg.Connection,
    *,
    submission_id: UUID | None,
    stage_attempt_id: UUID | None,
    event_type: str,
    severity: str = "INFO",
    message: str = "",
    data: dict[str, Any] | None = None,
) -> None:
    # Appends one audit event inside the caller's transaction.
    await conn.execute(
        """
        INSERT INTO events
            (id, submission_id, stage_attempt_id, event_type, severity, message, data)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        uuid4(),
        submission_id,
        stage_attempt_id,
        event_type,
        severity,
        message,
        data or {},
    )


async def heartbeat_attempt(attempt_id: UUID, lease_seconds: int) -> None:
    # Extends an in-flight stage attempt's lease; called on every dispatcher poll tick.
    p = await pool()
    await p.execute(
        """
        UPDATE stage_attempts
        SET lease_expires_at = now() + ($2 || ' seconds')::interval
        WHERE id = $1 AND state = 'RUNNING'
        """,
        attempt_id,
        str(lease_seconds),
    )
