"""Get-or-produce store for simulated observations.

Introduced for the eval side in a2344a0 (judge_api.ObservationSimulationService); shared so
pre-eval answers repeated reads the same way. A real shell answers the same read identically
until state changes, so both sides key on (sample, state fingerprint, command) and serve the
stored answer verbatim. The sides differ only in where the state fingerprint comes from: eval
derives it from the repo-context service, pre-eval from the trajectory's mutating commands.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

MEMO_MAX_ENTRIES = 8192


class ObservationMemo:
    def __init__(self, max_entries: int = MEMO_MAX_ENTRIES) -> None:
        self._memo: dict[str, str] = {}
        self._inflight: dict[str, asyncio.Task[str]] = {}
        self._max_entries = max_entries

    def remember(self, key: str, observation: str) -> None:
        self._memo[key] = observation
        while len(self._memo) > self._max_entries:
            self._memo.pop(next(iter(self._memo)))

    async def observe(self, key: str, produce: Callable[[], Awaitable[str]]) -> str:
        stored = self._memo.get(key)
        if stored is not None:
            return stored
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(produce())
            self._inflight[key] = task
            task.add_done_callback(lambda _: self._inflight.pop(key, None))
        observation = await asyncio.shield(task)
        self.remember(key, observation)
        return observation
