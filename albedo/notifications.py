"""Slack webhook alerts for eval/scoring faults - best-effort, deduped, secret-redacting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from albedo.settings import get_settings


@dataclass(frozen=True)
class EvalErrorNotification:
    # One alertable fault event; only non-empty fields are rendered into the Slack line.
    component: str
    severity: str
    message: str
    eval_run_id: str | None = None
    submission_id: str | None = None
    batch_id: str | None = None
    fault_class: str | None = None
    fault_code: str | None = None
    provider_route: str | None = None
    scoring_mode: str | None = None
    retryable: bool | None = None
    details: dict[str, Any] | None = None


_SENT: dict[tuple[str, str, str], float] = {}


def notify_error(event: EvalErrorNotification) -> None:
    # Best-effort Slack post; no-ops without a webhook and dedupes repeats inside the window.
    slack = get_settings().slack
    if not slack.webhook_url:
        return
    if _is_duplicate(event):
        return
    message = _format_message(event)
    payload = {
        "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": message}}],
        "username": slack.username,
        "icon_url": slack.icon_url,
    }
    try:
        response = httpx.post(
            slack.webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=slack.timeout_seconds,
        )
        if response.status_code != 200:
            logger.warning(
                f"Slack error notification failed: status={response.status_code} "
                f"body={response.text[:500]}"
            )
        else:
            logger.info(f"Slack error notification sent for eval_run_id={event.eval_run_id or ''}")
    except Exception as exc:  # noqa: BLE001 - alerting must never take down the caller
        logger.warning(f"Slack error notification failed: {exc}")


def _is_duplicate(event: EvalErrorNotification) -> bool:
    # Suppresses repeats of the same (run, component, fault) within the dedupe window.
    key = (event.eval_run_id or "", event.component, event.fault_code or event.message[:80])
    now = time.monotonic()
    previous = _SENT.get(key)
    if previous is not None and now - previous < get_settings().slack.dedupe_seconds:
        return True
    _SENT[key] = now
    return False


def _format_message(event: EvalErrorNotification) -> str:
    # Renders the alert as one prefixed line of key=value fields.
    label = get_settings().slack.env_label
    prefix = f"[{label}] " if label else ""
    fields = [
        f"component={event.component}",
        f"severity={event.severity}",
    ]
    for name in (
        "eval_run_id",
        "submission_id",
        "batch_id",
        "fault_class",
        "fault_code",
        "provider_route",
        "scoring_mode",
    ):
        value = getattr(event, name)
        if value:
            fields.append(f"{name}={value}")
    if event.retryable is not None:
        fields.append(f"retryable={event.retryable}")
    safe_details = _redact_details(event.details or {})
    if safe_details:
        fields.append(f"details={safe_details}")
    return f"{prefix}Albedo eval/scoring alert: {event.message}\n" + " | ".join(fields)


def _redact_details(details: dict[str, Any]) -> dict[str, Any]:
    # Masks any detail whose key smells like a secret, prompt, or raw model output.
    blocked = ("secret", "token", "key", "authorization", "prompt", "output", "raw", "response")
    redacted: dict[str, Any] = {}
    for key, value in details.items():
        lower = str(key).lower()
        if any(part in lower for part in blocked):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted
