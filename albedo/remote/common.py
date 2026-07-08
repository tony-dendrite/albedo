"""Shared GPU-worker plumbing - in-memory run store, bearer auth, /ready payloads."""

from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Annotated, Any, Callable

from fastapi import Header, HTTPException
from loguru import logger

_TERMINAL_STATES = {"succeeded", "failed"}


class WorkerBusy(Exception):
    # Raised when a new run arrives while a different run is still active.
    def __init__(self, active_count: int) -> None:
        super().__init__(f"worker busy: {active_count} active run(s)")
        self.active_count = active_count


@dataclass
class Run:
    # One in-memory run: request, state, append-only events; lost on worker death, backend requeues.
    run_id: str
    request: Any
    state: str
    events: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    worker_started: bool = False

    def set_state(self, state: str) -> None:
        # Updates the state and bumps the timestamp.
        self.state = state
        self.updated_at = datetime.now(UTC)

    def append_event(self, event: dict[str, Any]) -> None:
        # Records a progress/result event for the backend to poll.
        self.events.append(event)
        self.updated_at = datetime.now(UTC)

    def final_event(self, event_type: str) -> dict[str, Any] | None:
        # Returns the latest terminal event of the given type, if any.
        for event in reversed(self.events):
            if event.get("type") == event_type:
                return event
        return None


class RunStore:
    # Thread-safe run_id -> Run map; idempotent on run_id, one active run at a time.
    def __init__(self) -> None:
        # Empty store guarded by a re-entrant lock.
        self._runs: dict[str, Run] = {}
        self._lock = RLock()

    def get_or_create(self, run_id: str, factory: Callable[[], Run]) -> Run:
        # Returns the run for run_id, else registers factory(); WorkerBusy if another is active.
        with self._lock:
            existing = self._runs.get(run_id)
            if existing:
                return existing
            active = [run for run in self._runs.values() if run.state not in _TERMINAL_STATES]
            if active:
                raise WorkerBusy(len(active))
            run = factory()
            self._runs[run.run_id] = run
            return run

    def mark_worker_started(self, run_id: str) -> Run | None:
        # Moves an accepted run to queued exactly once; None if already started or terminal.
        with self._lock:
            run = self._runs.get(run_id)
            if not run or run.worker_started or run.state in _TERMINAL_STATES:
                return None
            run.worker_started = True
            run.set_state("queued")
            return run

    def get(self, run_id: str) -> Run | None:
        # Returns the run for this id, or None.
        with self._lock:
            return self._runs.get(run_id)

    def list_active(self) -> list[Run]:
        # Returns runs that have not reached a terminal state.
        with self._lock:
            return [run for run in self._runs.values() if run.state not in _TERMINAL_STATES]


def bearer_auth(token_getter: Callable[[], str]) -> Callable[..., None]:
    # Builds a FastAPI dependency enforcing "Authorization: Bearer <token>"; open when empty (mock).
    def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        # Rejects requests whose bearer token does not match the configured one.
        token = token_getter()
        if not token:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid remote auth token")

    return require_auth


def require_startup_token(token: str, *, mock: bool, service: str) -> None:
    # Hardening: a worker with an empty auth token refuses to start unless its mock flag is on.
    if token or mock:
        return
    logger.error(f"[{service}] refusing to start: auth token is empty and mock mode is off")
    sys.exit(1)


def ready_payload(*, host_id: str, role: str, active_runs: int, **extra: Any) -> dict[str, Any]:
    # /ready body shared by both workers: identity, role, busy state, plus worker-specific extras.
    return {"ready": True, "host_id": host_id, "role": role, "active_runs": active_runs, **extra}


# ── Canonical genesis model config (merged from canonical_config.py) ─────────

GENESIS_MODEL_CONFIG_REF = (
    "registry.hippius.com/teutonic/qwen3.6-35b-a3b-genesis@"
    "sha256:efd5b8d0a1c1f472be56ff919419cdd0561bdecd9013d5c2a96dd0e23e89c165"
)


GENESIS_MODEL_CONFIG: dict[str, Any] = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5_moe",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 248044,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "model_type": "qwen3_5_moe_text",
        "moe_intermediate_size": 512,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "output_router_logits": False,
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "router_aux_loss_coef": 0.001,
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 248320,
    },
    "tie_word_embeddings": False,
    "transformers_version": "4.57.1",
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5_moe",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053,
}


GENESIS_GENERATION_CONFIG: dict[str, Any] = {
    "bos_token_id": 248044,
    "do_sample": True,
    "eos_token_id": [248046, 248044],
    "pad_token_id": 248044,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
}


GENESIS_PREPROCESSOR_CONFIG: dict[str, Any] = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}


GENESIS_VIDEO_PREPROCESSOR_CONFIG: dict[str, Any] = {
    "size": {"longest_edge": 25165824, "shortest_edge": 4096},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "video_processor_type": "Qwen3VLVideoProcessor",
}


GENESIS_ARCH_SPEC: dict[str, Any] = {
    "architectures": GENESIS_MODEL_CONFIG["architectures"],
    "expected": GENESIS_MODEL_CONFIG,
    "forbidden_keys": ["auto_map", "quantization_config"],
}


def canonical_model_config() -> dict[str, Any]:
    # Returns a fresh copy of the pinned genesis Hugging Face model config.
    return deepcopy(GENESIS_MODEL_CONFIG)


def canonical_generation_config() -> dict[str, Any]:
    # Returns a fresh copy of the pinned genesis Hugging Face generation config.
    return deepcopy(GENESIS_GENERATION_CONFIG)


def canonical_max_model_len() -> int:
    # max_position_embeddings sits under text_config for the MoE config; top level wins when flat.
    config = GENESIS_MODEL_CONFIG
    if "max_position_embeddings" in config:
        return int(config["max_position_embeddings"])
    return int(config["text_config"]["max_position_embeddings"])


def apply_canonical_model_config(model_dir: Path) -> bool:
    # Replaces model-supplied configs with the genesis pin (keeps extra keys, drops forbidden ones).
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False

    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"model config is not valid JSON: {config_path}") from exc
    if not isinstance(existing, dict):
        raise ValueError(f"model config must be a JSON object: {config_path}")

    forbidden = set(GENESIS_ARCH_SPEC["forbidden_keys"])
    merged = {key: value for key, value in existing.items() if key not in forbidden}
    merged.update(canonical_model_config())
    config_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    generation_config_path = model_dir / "generation_config.json"
    generation_config_path.write_text(
        json.dumps(canonical_generation_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Pin the canonical image+video processor configs so vLLM can construct the
    # multimodal model. Text-only eval never uses them, but vLLM/HF refuse to load
    # the multimodal Qwen3.6 architecture without an image+video processor present.
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps(GENESIS_PREPROCESSOR_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (model_dir / "video_preprocessor_config.json").write_text(
        json.dumps(GENESIS_VIDEO_PREPROCESSOR_CONFIG, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return True


# ── SWE-ZERO dataset loading + prompt formatting (merged from dataset.py) ────

_IM_START = "<|im_start|>"


_IM_END = "<|im_end|>"


# ── Download heartbeat (shared by both workers) ──────────────────────────────

_HEARTBEAT_INTERVAL_S = 10.0


@contextmanager
def download_heartbeat(label: str):
    # Emits a periodic "still downloading" log line while a blocking fetch runs (daemon thread).
    stop = threading.Event()
    start = time.monotonic()

    def _beat() -> None:
        # Logs elapsed time every interval until the download finishes.
        while not stop.wait(_HEARTBEAT_INTERVAL_S):
            logger.info(
                f"model_download_progress ref={label} elapsed_s={time.monotonic() - start:.0f}"
            )

    thread = threading.Thread(target=_beat, name="model-dl-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
