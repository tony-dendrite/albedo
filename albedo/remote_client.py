"""Generic dispatcher-side HTTP client for the GPU run APIs (eval and sanity share one protocol)."""

from __future__ import annotations

from typing import Any

import httpx


class RemoteRunClient:
    # Talks to a GPU worker over its SSH tunnel: /ready, POST /{kind}, GET /{kind}/{id}[/events].
    def __init__(
        self,
        *,
        base_url: str,
        run_kind: str,
        auth_token: str = "",
        timeout_seconds: float = 30.0,
    ):
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        self._kind = run_kind.strip("/")
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds
        )

    async def aclose(self) -> None:
        # Releases the underlying connection pool.
        await self._client.aclose()

    async def ready(self) -> dict[str, Any]:
        # Worker liveness + capacity snapshot.
        response = await self._client.get("/ready")
        response.raise_for_status()
        return response.json()

    async def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Submits a run; the worker answers 409 when busy (surfaced as httpx.HTTPStatusError).
        response = await self._client.post(f"/{self._kind}", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_run(self, run_id: str) -> dict[str, Any]:
        # Current run state document.
        response = await self._client.get(f"/{self._kind}/{run_id}")
        response.raise_for_status()
        return response.json()

    async def get_events(self, run_id: str) -> list[dict[str, Any]]:
        # Full event list; callers slice by their own seen-count cursor.
        response = await self._client.get(f"/{self._kind}/{run_id}/events")
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return payload
        return payload.get("events", [])
