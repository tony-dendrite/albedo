"""ObjectStore: single entry point composing R2Store (private) and HippiusStore (dashboard)."""
from __future__ import annotations

from albedo.storage.hippius import HippiusStore
from albedo.storage.r2 import R2Store


class ObjectStore:
    """Unified storage façade: R2 for private state, Hippius for dashboard."""

    def __init__(self) -> None:
        self._r2      = R2Store()
        self._hippius = HippiusStore()

    def get(self, key: str) -> dict | None:
        return self._r2.get(key)

    def put(self, key: str, data: dict) -> bool:
        return self._r2.put(key, data)

    def delete(self, key: str) -> None:
        self._r2.delete(key)

    def put_dashboard(self, key: str, data: dict, **kw) -> bool:
        return self._hippius.put(key, data, **kw)

    def put_dashboard_raw(
        self, key: str, body: bytes, content_type: str, **kw
    ) -> bool:
        return self._hippius.put_raw(key, body, content_type, **kw)
