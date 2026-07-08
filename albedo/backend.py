"""Backend supervisor - every CPU-side loop runs as a task in this one process."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from albedo import db

_RESTART_DELAY_S = 5.0


async def _supervise(name: str, fn: Callable[[], Awaitable[None]]) -> None:
    # Restarts a loop forever on crash so one bad tick never takes the process down.
    while True:
        try:
            await fn()
            logger.warning(f"[backend] task {name} returned; restarting in {_RESTART_DELAY_S}s")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - keep every backend loop alive across blips
            logger.exception(f"[backend] task {name} crashed; restarting in {_RESTART_DELAY_S}s")
        await asyncio.sleep(_RESTART_DELAY_S)


async def run_backend() -> None:
    # Boots the pool then runs all orchestration loops until cancelled.
    from albedo import chain, evaluation, hosts, judges, monitor, reign, sanity
    from albedo import validation as validation_mod
    from albedo import weights as weights_mod

    await db.pool()
    tasks = {
        "chain-ingest": chain.run_ingest,
        "validation": validation_mod.run_worker,
        "sanity-dispatch": sanity.run_dispatcher,
        "sanity-janitor": sanity.run_janitor,
        "eval-dispatch": evaluation.run_dispatcher,
        "eval-janitor": evaluation.run_janitor,
        "score-bridge": judges.run_bridge_client,
        "set-reign": reign.run_worker,
        "weight-setter": weights_mod.run_worker,
        "gpu-hosts": hosts.run_heartbeater,
        "monitor": monitor.run_publisher,
    }
    logger.info(f"[backend] starting {len(tasks)} loops: {', '.join(tasks)}")
    try:
        async with asyncio.TaskGroup() as tg:
            for name, fn in tasks.items():
                tg.create_task(_supervise(name, fn), name=name)
    finally:
        await db.close_pool()
