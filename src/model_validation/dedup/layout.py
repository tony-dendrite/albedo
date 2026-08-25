from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

WS_VERSION = "ws-canon-v2"
LM_PREFIX = "model.language_model."
HEAD = "lm_head.weight"
EMBED = "embed_tokens.weight"

ROW_HIDDEN = (
    "q_proj",
    "k_proj",
    "v_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "mlp.gate.weight",
)
COL_HIDDEN = ("o_proj", "out_proj", "shared_expert.down_proj")
RES_ROWS = ("o_proj", "out_proj", "shared_expert.down_proj", "experts.down_proj")
TYPES = ("experts", "shared_expert", "linear_attn", "self_attn", "embed_tokens", "lm_head")
BODY_TYPES = ("self_attn", "linear_attn", "shared_expert")
EMBED_TYPES = ("embed_tokens", "lm_head")


def canon(name: str) -> str | None:
    if name == HEAD:
        return name
    return name[len(LM_PREFIX) :] if name.startswith(LM_PREFIX) else None


def ttype(c: str) -> str:
    for k in TYPES:
        if k in c:
            return k
    return "other"


def read_header(shard: Path) -> dict:
    with shard.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header


def tensor_names(model_dir: str) -> dict[str, tuple[str, str]]:
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {}
        for shard in sorted(root.glob("*.safetensors")):
            for key in read_header(shard):
                weight_map[key] = shard.name
    out: dict[str, tuple[str, str]] = {}
    for key, fname in weight_map.items():
        c = canon(key)
        if c is not None:
            out[c] = (key, fname)
    return out


def arch_key(shapes: dict[str, tuple[int, ...]]) -> str:
    lines = sorted(f"{n}:{','.join(str(d) for d in s)}" for n, s in shapes.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]
