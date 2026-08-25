from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from model_validation.dedup.layout import WS_VERSION

MIN_SECRET_BYTES = 32


def load_secret(value: str, path: str) -> bytes:
    if value:
        raw = value.encode()
    else:
        if not path:
            raise RuntimeError("set ALBEDO_DEDUP_SECRET or ALBEDO_DEDUP_SECRET_FILE")
        p = Path(path)
        if not p.is_file():
            raise RuntimeError(f"dedup secret file not found: {path}")
        if p.stat().st_mode & 0o077:
            raise RuntimeError(f"dedup secret file {path} must not be group/other readable")
        raw = p.read_bytes().strip()
    if len(raw) < MIN_SECRET_BYTES:
        raise RuntimeError(f"dedup secret must be at least {MIN_SECRET_BYTES} bytes")
    return raw


def key_id(secret: bytes, text_set_sha: str = "") -> str:
    return hashlib.sha256(secret + text_set_sha.encode()).hexdigest()[:8]


def seed_for(secret: bytes, name: str, which: str) -> int:
    msg = f"{WS_VERSION}:{which}:{name}".encode()
    return int.from_bytes(hmac.new(secret, msg, hashlib.sha256).digest()[:8], "little")
