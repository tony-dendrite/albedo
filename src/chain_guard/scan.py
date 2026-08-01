from __future__ import annotations

from typing import Any

from chain_reader.chain import _iter_revealed


def scan_all_raw(subtensor: Any, netuid: int) -> list[tuple[str, int, str]]:
    return list(_iter_revealed(subtensor, netuid))
