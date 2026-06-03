"""albedo.models.compat — arch/lock-key compatibility shared by miner and validator."""
from __future__ import annotations

from typing import Any

from albedo.config import ALL_LOCK_KEYS


def config_lock_violation(
    king_cfg: dict[str, Any],
    challenger_cfg: dict[str, Any],
) -> str | None:
    """Return a rejection reason if the challenger breaks arch/lock compat, else None."""
    if king_cfg.get("architectures") != challenger_cfg.get("architectures"):
        return (
            f"architectures mismatch: king={king_cfg.get('architectures')!r} "
            f"challenger={challenger_cfg.get('architectures')!r}"
        )
    for key in ALL_LOCK_KEYS:
        if king_cfg.get(key) != challenger_cfg.get(key):
            return (
                f"lock key mismatch for {key!r}: king={king_cfg.get(key)!r} "
                f"challenger={challenger_cfg.get(key)!r}"
            )
    return None
