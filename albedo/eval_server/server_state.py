"""albedo.eval_server.server_state — Module-level singleton shared by all endpoints."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from albedo.eval_server.vllm import VLLMProcess

_KING_GPUS = os.environ.get("ALBEDO_KING_GPUS", "7")
_CHAL_GPUS = os.environ.get("ALBEDO_CHAL_GPUS", "6")

KING_PORT = 8001
CHAL_PORT = 8002


@dataclass
class EvalState:
    king_proc:       VLLMProcess
    chal_proc:       VLLMProcess
    eval_lock:       asyncio.Lock = field(default_factory=asyncio.Lock)
    current_eval_id: str | None   = None
    # key (repo@digest) -> v2 fingerprint dict (layer_keys, norm_vector, tensor_samples)
    fingerprints:    dict[str, dict] = field(default_factory=dict)


STATE: EvalState = EvalState(
    king_proc=VLLMProcess(role="king",       gpus=_KING_GPUS, port=KING_PORT),
    chal_proc=VLLMProcess(role="challenger", gpus=_CHAL_GPUS, port=CHAL_PORT),
)
