from __future__ import annotations

from typing import Any, NamedTuple

from config_validation.models.ref import ModelRef

_VERSION = "v7"
_SEP = "|"


def decode_raw(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        if raw.startswith("0x"):
            try:
                return bytes.fromhex(raw[2:]).decode("utf-8", errors="replace")
            except ValueError:
                return raw
        return raw
    return str(raw)


class Reveal(NamedTuple):
    ref: ModelRef


def parse_reveal(data: str) -> Reveal:
    if not data.startswith(_VERSION + _SEP):
        raise ValueError(f"not a {_VERSION} reveal: {data[:16]!r}")

    parts = data.split(_SEP)
    if len(parts) != 3:
        raise ValueError(f"unexpected v7 part count {len(parts)}: {data[:32]!r}")

    _, repo, digest = parts
    ref = ModelRef(repo=repo, digest=digest)
    return Reveal(ref=ref)
