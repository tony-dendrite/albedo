"""albedo.judge.client — Async Chutes LLM-as-judge client."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx

import random
import time

from albedo.config import (
    JUDGE_API_KEY_ENV,
    JUDGE_BASE_URL_ENV,
    JUDGE_CALL_TIMEOUT_S,
    JUDGE_MAX_TOKENS,
    JUDGE_METRIC_KEYS,
    JUDGE_RETRY_BACKOFF,
    JUDGE_RETRY_MAX,
    JUDGE_SCORE_MAX_TOKENS,
    JUDGE_TEMPERATURE,
    JUDGE_THINKING_MODELS,
    JUDGE_THINKING_TOKENS,
    JUDGE_429_MAX_WAIT_S,
)
from albedo.judge.rubric import PAIRWISE_RUBRIC_SYSTEM, PROBE_SYSTEM, build_pairwise_user
from albedo.judge.verdict import MetricVerdict, parse_metric_verdict

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://llm.chutes.ai"


class DeadlineExceeded(RuntimeError):
    """Raised when a judge call exhausts its total time budget.

    Caught by pairwise_judge() and converted to a parse_failure MetricVerdict so
    the duel turn continues with score=0.0 rather than the whole turn being aborted.
    """


def _is_thinking(model: str) -> bool:
    return model in JUDGE_THINKING_MODELS


def _merge_thinking(choices: list[dict]) -> str:
    """Concatenate reasoning_content + content from the first choice."""
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    reasoning = msg.get("reasoning_content") or ""
    content   = msg.get("content") or ""
    if reasoning:
        return f"{reasoning}\n{content}"
    return content


class ChutesJudge:
    """Async Chutes LLM-as-judge client; one instance shared across all judge models."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key:  str | None = None,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get(JUDGE_BASE_URL_ENV, "")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key = api_key or os.environ.get(JUDGE_API_KEY_ENV, "")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        # Separate timeouts per model class: thinking models (Qwen3-235B, Kimi-K2.6)
        # regularly need 300–600 s for reasoning traces; regular models are fast.
        self._client       = httpx.AsyncClient(
            base_url=self._base_url, headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=150.0, write=30.0, pool=10.0),
        )
        self._think_client = httpx.AsyncClient(
            base_url=self._base_url, headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
        )

    async def _chat(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        deadline: float,
    ) -> list[dict]:
        """POST /v1/chat/completions with retry.

        Args:
            deadline: monotonic timestamp; extended whenever a 429 wait is taken
                      so rate-limit sleeps never consume the caller's time budget.

        Retry strategy:
          - 429 (rate-limited): retry INDEFINITELY — each sleep extends the
            deadline so the turn is never abandoned due to rate limiting alone.
            Backoff is exponential, capped at JUDGE_429_MAX_WAIT_S per attempt,
            respecting Retry-After headers.
          - 5xx / network: up to JUDGE_RETRY_MAX attempts, 2× backoff.
          - 4xx (not 429): raised immediately — permanent client error.
          - DeadlineExceeded only fires when the server is genuinely unreachable
            (non-429 budget exhausted); 429 waits never count toward the deadline.
        """
        client  = self._think_client if _is_thinking(model) else self._client
        payload: dict[str, Any] = {
            "model":       model,
            "messages":    messages,
            "temperature": JUDGE_TEMPERATURE,
            "max_tokens":  max_tokens,
        }

        # Small random jitter before the first attempt so simultaneous king and
        # challenger calls to the same judge don't land at exactly the same time,
        # reducing double rate-limit pressure.
        jitter = random.uniform(0.0, 0.5)
        if time.monotonic() + jitter < deadline:
            await asyncio.sleep(jitter)

        max_non429 = JUDGE_RETRY_MAX
        n_429      = 0
        n_non429   = 0
        last_exc: Exception | None = None

        while True:
            # Check deadline before each attempt (only applies to non-429 budget).
            if time.monotonic() >= deadline:
                raise DeadlineExceeded(
                    f"Judge {model} deadline exceeded "
                    f"({n_429} rate-limited, {n_non429} server errors)"
                )

            try:
                resp = await client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                return resp.json().get("choices", [])

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code

                if status == 429:
                    n_429 += 1
                    last_exc = exc

                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            raw_wait = max(float(retry_after), 1.0)
                        except ValueError:
                            raw_wait = JUDGE_RETRY_BACKOFF * (4 ** min(n_429 - 1, 4))
                    else:
                        raw_wait = JUDGE_RETRY_BACKOFF * (4 ** min(n_429 - 1, 4))

                    wait = min(raw_wait, JUDGE_429_MAX_WAIT_S)
                    deadline += wait  # extend so 429 sleeps are free — never give up

                    log.warning(
                        "Judge %s 429 (#%d) — waiting %.1fs; deadline extended",
                        model, n_429, wait,
                    )
                    await asyncio.sleep(wait)
                    continue  # retry; 429s are never counted against the error budget

                if status < 500:
                    raise  # permanent 4xx — don't retry

                last_exc = exc

            except DeadlineExceeded:
                raise
            except Exception as exc:
                last_exc = exc

            n_non429 += 1
            if n_non429 >= max_non429:
                break
            backoff = min(
                JUDGE_RETRY_BACKOFF * (2 ** (n_non429 - 1)),
                max(0.1, deadline - time.monotonic() - 0.1),
            )
            log.warning(
                "Judge %s error (%d/%d): %s — retry in %.1fs",
                model, n_non429, max_non429, last_exc, backoff,
            )
            await asyncio.sleep(backoff)

        raise RuntimeError(
            f"Judge {model} failed after {max_non429} non-429 attempts"
        ) from last_exc

    def _build_pairwise_messages(
        self,
        context_messages: list[dict],
        king_reply: str,
        chal_reply: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": PAIRWISE_RUBRIC_SYSTEM},
            {"role": "user",   "content": build_pairwise_user(
                context_messages, king_reply, chal_reply)},
        ]

    def _build_probe_messages(
        self,
        messages: list[dict],
        reply: str,
    ) -> list[dict]:
        combined = json.dumps(
            {
                "conversation": messages,
                "candidate_reply": reply,
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": PROBE_SYSTEM},
            {"role": "user",   "content": combined},
        ]

    async def pairwise_judge(
        self,
        context_messages: list[dict] | None = None,
        king_reply: str = "",
        chal_reply: str = "",
        *,
        model: str,
        messages_prefix: list[dict] | None = None,
        messages_prompt: list[dict] | None = None,
        hotkey: str = "",
        seed: bytes = b"",
    ) -> MetricVerdict:
        """Score one turn head-to-head (MODEL 1 = king, MODEL 2 = challenger).

        Accepts either context_messages or messages_prefix + messages_prompt
        (the form used by turn.py). One judge call returns per-metric scores
        in challenger perspective.

        Returns a parse_failure MetricVerdict (all-zero, judge_mean=0.0) when the
        call exceeds JUDGE_CALL_TIMEOUT_S — the duel continues rather than stalling.
        """
        if context_messages is None:
            context_messages = list(messages_prefix or []) + list(messages_prompt or [])

        thinking   = _is_thinking(model)
        max_tokens = JUDGE_THINKING_TOKENS if thinking else JUDGE_SCORE_MAX_TOKENS
        deadline   = time.monotonic() + JUDGE_CALL_TIMEOUT_S

        messages = self._build_pairwise_messages(context_messages, king_reply, chal_reply)
        try:
            choices = await self._chat(
                messages, model=model, max_tokens=max_tokens, deadline=deadline
            )
            raw = _merge_thinking(choices)
        except DeadlineExceeded:
            log.warning(
                "Judge %s score deadline exceeded (%.0fs budget) — returning parse_failure",
                model, JUDGE_CALL_TIMEOUT_S,
            )
            return MetricVerdict(
                metric_scores={k: 0.0 for k in JUDGE_METRIC_KEYS},
                judge_mean=0.0, raw="", parse_ok=False, model=model,
            )

        verdict       = parse_metric_verdict(raw)
        verdict.model = model
        return verdict

    async def probe(
        self,
        messages: list[dict],
        reply: str,
        *,
        model: str,
    ) -> tuple[bool, str]:
        """Injection probe; returns (is_injected, evidence).

        Uses PROBE_SYSTEM to check whether the reply contains injection patterns.
        Retries indefinitely on transient errors — callers must always receive a
        definitive answer (True/False) and must never silently skip the probe.

        Schema: {"injection": true|false, "evidence": "<snippet or 'none'>"}
        """
        thinking   = _is_thinking(model)
        max_tokens = JUDGE_THINKING_TOKENS if thinking else JUDGE_MAX_TOKENS

        probe_messages = self._build_probe_messages(messages, reply)
        probe_re       = re.compile(
            r'\{[^{}]*"injection"\s*:\s*(?:true|false)[^{}]*\}', re.DOTALL
        )

        attempt = 0
        while True:
            # Probe uses a fresh deadline per attempt — probes must always complete.
            deadline = time.monotonic() + JUDGE_CALL_TIMEOUT_S
            try:
                choices = await self._chat(
                    probe_messages, model=model, max_tokens=max_tokens, deadline=deadline
                )
                raw = _merge_thinking(choices)
            except Exception as exc:
                attempt += 1
                wait = min(JUDGE_RETRY_BACKOFF * (2 ** min(attempt - 1, 6)), 120.0)
                log.warning(
                    "probe: judge %s error (attempt %d) — retrying in %.0fs: %s",
                    model, attempt, wait, exc,
                )
                await asyncio.sleep(wait)
                continue

            def _extract(data: dict) -> tuple[bool, str]:
                is_injected = bool(data.get("injection", False))
                evidence = str(data.get("evidence") or data.get("reason") or "").strip()
                if evidence.lower() == "none":
                    evidence = ""
                return is_injected, evidence

            matches = probe_re.findall(raw)
            if matches:
                try:
                    return _extract(json.loads(matches[-1]))
                except json.JSONDecodeError:
                    pass

            stripped = raw.strip()
            if stripped.startswith("{"):
                try:
                    return _extract(json.loads(stripped))
                except json.JSONDecodeError:
                    pass

            # Judge responded but output was unparseable — retry to get a valid JSON.
            attempt += 1
            wait = min(JUDGE_RETRY_BACKOFF * (2 ** min(attempt - 1, 4)), 60.0)
            log.warning(
                "probe: judge %s returned unparseable output (attempt %d) "
                "— retrying in %.0fs: %r",
                model, attempt, wait, raw[:200],
            )
            await asyncio.sleep(wait)

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._think_client.aclose()

    async def __aenter__(self) -> "ChutesJudge":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
