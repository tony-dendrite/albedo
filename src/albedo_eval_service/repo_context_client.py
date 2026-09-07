"""HTTP client for the repo-context service.

Lives beside judge_llm_client.py rather than in shared/ because it takes JudgeSettings and shared/
is deliberately config-free. Both the eval judge API and the pre-eval dispatcher construct one, so
grounding is resolved the same way on both paths.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import httpx
from loguru import logger

from albedo_config import JudgeSettings


class Grounding(NamedTuple):
    context: str | None
    exact_output: str | None
    exact_returncode: int | None
    state: str


class RepoContextClient:
    def __init__(self, settings: JudgeSettings):
        self._client = httpx.AsyncClient(
            base_url=settings.repo_context_url.rstrip("/"),
            timeout=settings.repo_context_timeout_seconds,
        )
        self._last_warning = 0.0

    async def context_for(
        self, sample_id: str, assistant_output: str, messages: list[dict[str, str]] | None = None
    ) -> Grounding:
        try:
            response = await self._client.post(
                "/repo-context",
                json={
                    "sample_id": sample_id,
                    "assistant_output": assistant_output,
                    "messages": messages or [],
                },
            )
            response.raise_for_status()
            body = response.json()
            context = body.get("context")
            exact = body.get("exact_output")
            returncode = body.get("exact_returncode")
            state = body.get("state")
            return Grounding(
                context if isinstance(context, str) and context else None,
                exact if isinstance(exact, str) else None,
                returncode if isinstance(returncode, int) else None,
                state if isinstance(state, str) else "",
            )
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_warning > 60.0:
                self._last_warning = now
                logger.warning(
                    "repo_context_unavailable sample_id={} error={}",
                    sample_id,
                    f"{type(exc).__name__}: {exc}",
                )
            return Grounding(None, None, None, "")

    async def aclose(self) -> None:
        await self._client.aclose()
