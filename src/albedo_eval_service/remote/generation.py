from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from loguru import logger

from .dataset import EvalSample, _load_tokenizer
from .prompt_remote import QWEN3_IM_END_TOKEN_ID


@dataclass(frozen=True)
class GenerationResult:
    sample_id: str
    text: str
    error: str | None = None
    turns: list[dict[str, Any]] | None = None
    truncated: bool = False
    # feedback shown to the model for each dropped bad attempt at this turn (bench-style)
    retry_feedbacks: tuple[str, ...] = ()


class Generator(Protocol):
    def generate(self, samples: list[EvalSample]) -> list[GenerationResult]: ...


def format_scored_trajectory(turns: list[dict[str, Any]]) -> str:
    target_count = sum(
        1 for turn in turns if turn.get("role") == "assistant" and turn.get("score_target")
    )
    target_label = (
        "CANDIDATE OUTPUT"
        if target_count == 1
        else f"CANDIDATE OUTPUT 1 through CANDIDATE OUTPUT {target_count}"
    )
    assistant_index = 0
    parts = [
        "FULL CANDIDATE TRAJECTORY",
        f"Score ONLY {target_label}. The ENVIRONMENT OBSERVATION is context only.",
    ]
    for turn in turns:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").rstrip()
        if role == "assistant" and turn.get("score_target"):
            assistant_index += 1
            label = f"CANDIDATE OUTPUT {assistant_index}"
        elif role == "user" and turn.get("environment_observation"):
            label = "ENVIRONMENT OBSERVATION (context only, do not score)"
        else:
            label = (
                f"CONTEXT {role.upper()} (do not score)" if role else "CONTEXT TURN (do not score)"
            )
        parts.append(f"\n{label}:\n------\n{content}\n------")
    return "\n".join(parts).strip()


_CONTEXT_SAFETY_MARGIN_TOKENS = 64
_SERVED_MODEL_NAME = "candidate"


class VllmServerGenerator:
    """One `vllm serve` per model. Requests are independent, so every trajectory runs at its
    own pace while the engine batches whatever is in flight."""

    def __init__(
        self,
        *,
        model: str,
        gpu_ids: list[str],
        port: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None = None,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        compile_cache_dir: str = "",
        gpu_memory_utilization: float = 0.95,
        kv_cache_dtype: str = "auto",
        max_num_seqs: int = 256,
        startup_timeout_seconds: float = 1800.0,
        result_timeout_seconds: float = 900.0,
    ):
        self.model = model
        self.gpu_ids = gpu_ids
        self.port = port
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.compile_cache_dir = compile_cache_dir
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_dtype = kv_cache_dtype
        self.max_num_seqs = max_num_seqs
        self.startup_timeout_seconds = startup_timeout_seconds
        self.result_timeout_seconds = result_timeout_seconds
        self._process: subprocess.Popen | None = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._client = httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=httpx.Timeout(result_timeout_seconds)
        )

    def generate(self, samples: list[EvalSample]) -> list[GenerationResult]:
        if not samples:
            return []
        self._start()
        return [self._complete(sample) for sample in samples]

    def close(self) -> None:
        if self._process is None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self._process.pid, signal.SIGKILL)
            self._process.wait(timeout=10)
        except ProcessLookupError:
            pass
        self._process = None

    def _complete(self, sample: EvalSample) -> GenerationResult:
        if self.max_model_len and (
            len(self._tokenizer(sample.prompt).input_ids)
            >= self.max_model_len - _CONTEXT_SAFETY_MARGIN_TOKENS
        ):
            return GenerationResult(sample.sample_id, "", truncated=True)
        body = {
            "model": _SERVED_MODEL_NAME,
            "prompt": sample.prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop_token_ids": [QWEN3_IM_END_TOKEN_ID],
        }
        if self.top_k is not None:
            body["top_k"] = self.top_k
        try:
            response = self._client.post("/v1/completions", json=body)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.exception(
                f"[remote-gen] vLLM request failed model={self.model} "
                f"sample={sample.sample_id}: {exc}"
            )
            return GenerationResult(sample.sample_id, "", f"{type(exc).__name__}: {exc}")
        choice = payload["choices"][0]
        completion_tokens = int((payload.get("usage") or {}).get("completion_tokens") or 0)
        return GenerationResult(
            sample_id=sample.sample_id,
            text=choice.get("text") or "",
            truncated=choice.get("finish_reason") == "length"
            and completion_tokens >= self.max_new_tokens,
        )

    def _start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            self._process = subprocess.Popen(
                self._command(),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(self.gpu_ids)},
                start_new_session=True,
            )
            deadline = time.monotonic() + max(1.0, self.startup_timeout_seconds)
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server exited {self._process.returncode} during startup"
                    )
                try:
                    if self._client.get("/health", timeout=5.0).status_code == 200:
                        self._tokenizer = self._tokenizer or _load_tokenizer(self.model)
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(3)
            self.close()
            raise TimeoutError(f"vLLM server not ready after {self.startup_timeout_seconds:g}s")

    def _command(self) -> list[str]:
        command = [
            str(Path(sys.executable).with_name("vllm")),
            "serve",
            self.model,
            "--served-model-name",
            _SERVED_MODEL_NAME,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(len(self.gpu_ids)),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--trust-remote-code",
            "--generation-config",
            "vllm",
            "--enable-prefix-caching",
            "--limit-mm-per-prompt",
            json.dumps({"image": 0, "video": 0}),
        ]
        if self.max_model_len is not None:
            command += ["--max-model-len", str(self.max_model_len)]
        if self.enforce_eager:
            command.append("--enforce-eager")
        if self.compile_cache_dir:
            command += ["--compilation-config", json.dumps({"cache_dir": self.compile_cache_dir})]
        return command
