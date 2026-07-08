"""Applies the idempotent schema.sql to the configured database."""

from __future__ import annotations

from pathlib import Path

import asyncpg
from loguru import logger

from albedo.settings import get_settings


async def apply_schema(schema_path: str) -> None:
    # schema.sql is CREATE-IF-NOT-EXISTS only, so re-applying is always safe.
    sql = Path(schema_path).read_text()
    conn = await asyncpg.connect(dsn=get_settings().db.dsn)
    try:
        await conn.execute(sql)
        logger.info(f"[migrate] applied {schema_path}")
    finally:
        await conn.close()
