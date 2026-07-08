"""GPU host heartbeater - polls each remote_gpu_hosts /ready and keeps state/last_health fresh."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from albedo.db import pool
from albedo.settings import get_settings

_INTERVAL_S = 30.0
_OFFLINE_AFTER = 3
_failures: dict[str, int] = {}


def _auth_token(role: str) -> str:
    # EVAL hosts use the eval dispatcher token, PRE_EVAL hosts the sanity dispatcher token.
    s = get_settings()
    return s.eval.remote_auth_token if role == "EVAL" else s.sanity.remote_auth_token


async def _check_host(client: httpx.AsyncClient, host: Any) -> None:
    # One /ready probe: READY + health on 200, OFFLINE after 3 consecutive failures.
    p = await pool()
    try:
        resp = await client.get(
            f"{host['base_url'].rstrip('/')}/ready",
            headers={"Authorization": f"Bearer {_auth_token(host['role'])}"},
        )
        resp.raise_for_status()
        health = resp.json() if isinstance(resp.json(), dict) else {}
        _failures[host["id"]] = 0
        await p.execute(
            """
            UPDATE remote_gpu_hosts
            SET state = 'READY',
                free_gpu_count = COALESCE($2, free_gpu_count),
                last_heartbeat_at = now(),
                last_health = $3
            WHERE id = $1
            """,
            host["id"],
            health.get("free_gpu_count"),
            health,
        )
    except Exception as exc:  # noqa: BLE001 - an unreachable host is expected, not fatal
        n = _failures.get(host["id"], 0) + 1
        _failures[host["id"]] = n
        logger.warning(f"[hosts] {host['id']} /ready failed ({exc}) - consecutive={n}")
        if n >= _OFFLINE_AFTER:
            await p.execute(
                "UPDATE remote_gpu_hosts SET state = 'OFFLINE' WHERE id = $1", host["id"]
            )


async def run_heartbeater() -> None:
    # Every 30s, probes every registered GPU host; never lets one bad tick kill the loop.
    logger.info(
        f"[hosts] heartbeater started - interval={_INTERVAL_S}s offline_after={_OFFLINE_AFTER}"
    )
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                p = await pool()
                hosts = await p.fetch("SELECT id, role, base_url FROM remote_gpu_hosts")
                for host in hosts:
                    await _check_host(client, host)
            except Exception:  # noqa: BLE001 - keep the loop alive across DB blips
                logger.exception("[hosts] heartbeat tick failed")
            await asyncio.sleep(_INTERVAL_S)
