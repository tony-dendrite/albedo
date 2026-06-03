"""albedo.eval_server.sink — Upload typed eval artifacts to S3 after each duel.

Directory layout per eval:
  evals/{YYYY-MM-DD}/{counter}/
    responses_champion.jsonl    — one line per turn (king's reply + metadata)
    responses_challenger.jsonl  — one line per turn (challenger's reply + metadata)
    judge_raw.jsonl             — one line per judge call (prompt + raw response)
    scores.json                 — final verdict + aggregated scores
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_BUCKET      = os.environ.get("ALBEDO_EVALS_S3_BUCKET", "")
_ENDPOINT    = os.environ.get("ALBEDO_EVALS_S3_ENDPOINT", "")
_ACCESS_KEY  = os.environ.get("ALBEDO_EVALS_S3_ACCESS_KEY", "")
_SECRET_KEY  = os.environ.get("ALBEDO_EVALS_S3_SECRET_KEY", "")
_PREFIX      = os.environ.get("ALBEDO_EVALS_S3_PREFIX", "evals")
_PUBLIC_BASE = os.environ.get("ALBEDO_EVALS_PUBLIC_BASE", "")

_ENABLED = bool(_BUCKET and _ENDPOINT and _ACCESS_KEY and _SECRET_KEY)


def _jsonl(data: dict) -> bytes:
    return json.dumps(data, separators=(",", ":")).encode() + b"\n"


class DatasetSink:
    """Buffers typed eval artifacts and uploads them as 4 files to S3."""

    def __init__(
        self,
        *,
        eval_id:           str,
        eval_counter:      int | None = None,
        challenger_hotkey: str = "",
        king_hotkey:       str = "",
    ) -> None:
        self._eval_id           = eval_id
        # Derive counter from eval-NNNNNN format if not supplied explicitly.
        if eval_counter is None:
            try:
                eval_counter = int(eval_id.split("-")[-1])
            except (ValueError, IndexError):
                eval_counter = 0
        self._eval_counter      = eval_counter
        self._challenger_hotkey = challenger_hotkey
        self._king_hotkey       = king_hotkey
        self._date              = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        self._cham_lines:      list[bytes] = []
        self._chal_lines:      list[bytes] = []
        self._judge_raw_lines: list[bytes] = []
        self._scores:          dict | None  = None

    # ------------------------------------------------------------------ #
    # Write helpers                                                        #
    # ------------------------------------------------------------------ #

    def _base_meta(self) -> dict:
        return {
            "date":             self._date,
            "eval_id":          self._eval_id,
            "challenger_hotkey": self._challenger_hotkey,
            "king_hotkey":      self._king_hotkey,
        }

    def write_champion(self, data: dict) -> None:
        """Buffer one JSONL line for responses_champion.jsonl."""
        self._cham_lines.append(_jsonl({**self._base_meta(), **data}))

    def write_challenger(self, data: dict) -> None:
        """Buffer one JSONL line for responses_challenger.jsonl."""
        self._chal_lines.append(_jsonl({**self._base_meta(), **data}))

    def write_judge_raw(self, data: dict) -> None:
        """Buffer one JSONL line for judge_raw.jsonl."""
        self._judge_raw_lines.append(_jsonl({**self._base_meta(), **data}))

    def set_scores(self, verdict: dict) -> None:
        """Store the final verdict dict for scores.json."""
        self._scores = {**self._base_meta(), **verdict}

    # ------------------------------------------------------------------ #
    # Legacy compatibility — old callers used write_line()                 #
    # ------------------------------------------------------------------ #

    def write_line(self, data: dict) -> None:
        """Deprecated: write a raw line; routed to challenger buffer."""
        self._chal_lines.append(_jsonl(data))

    # ------------------------------------------------------------------ #
    # Upload                                                               #
    # ------------------------------------------------------------------ #

    def _dir_prefix(self) -> str:
        prefix = _PREFIX.rstrip("/")
        return f"{prefix}/{self._date}/{self._eval_counter:03d}"

    def _dir_url(self) -> str | None:
        if not _PUBLIC_BASE:
            return None
        base = _PUBLIC_BASE.rstrip("/")
        return f"{base}/{self._dir_prefix()}"

    async def flush(self) -> dict:
        """Upload all 4 artifact files; returns {"enabled", "uploaded", "dir_url"}."""
        if not _ENABLED:
            log.debug("DatasetSink: S3 not configured — skipping upload for %r", self._eval_id)
            return {"enabled": False, "uploaded": False, "dir_url": None}

        has_data = (
            self._cham_lines
            or self._chal_lines
            or self._judge_raw_lines
            or self._scores is not None
        )
        if not has_data:
            log.debug("DatasetSink: no data buffered for %r — skipping upload", self._eval_id)
            return {"enabled": True, "uploaded": False, "dir_url": None}

        return await asyncio.to_thread(self._upload_sync)

    def _upload_sync(self) -> dict:
        """Blocking S3 upload — called via asyncio.to_thread."""
        try:
            import boto3
            from botocore.client import Config as BotoConfig
        except ImportError as exc:
            log.error("boto3 is not installed — cannot upload eval artifacts: %s", exc)
            return {"enabled": True, "uploaded": False, "dir_url": None}

        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=_ENDPOINT,
                aws_access_key_id=_ACCESS_KEY,
                aws_secret_access_key=_SECRET_KEY,
                config=BotoConfig(signature_version="s3v4"),
            )
        except Exception as exc:
            log.error("DatasetSink: failed to create S3 client: %s", exc)
            return {"enabled": True, "uploaded": False, "dir_url": None}

        prefix  = self._dir_prefix()
        success = True

        files: list[tuple[str, bytes, str]] = [
            (
                f"{prefix}/responses_champion.jsonl",
                b"".join(self._cham_lines),
                "application/x-ndjson",
            ),
            (
                f"{prefix}/responses_challenger.jsonl",
                b"".join(self._chal_lines),
                "application/x-ndjson",
            ),
            (
                f"{prefix}/judge_raw.jsonl",
                b"".join(self._judge_raw_lines),
                "application/x-ndjson",
            ),
            (
                f"{prefix}/scores.json",
                json.dumps(self._scores or {}, separators=(",", ":")).encode(),
                "application/json",
            ),
        ]

        for key, body, content_type in files:
            if not body and content_type == "application/x-ndjson":
                continue  # skip empty JSONL files
            try:
                s3.put_object(
                    Bucket=_BUCKET,
                    Key=key,
                    Body=body,
                    ContentType=content_type,
                )
                log.debug("DatasetSink: uploaded %d bytes to s3://%s/%s", len(body), _BUCKET, key)
            except Exception:
                log.error("DatasetSink: S3 upload failed for %r", key, exc_info=True)
                success = False

        dir_url = self._dir_url()
        log.info(
            "DatasetSink: eval %r uploaded to %s (success=%s)",
            self._eval_id, prefix, success,
        )
        return {"enabled": True, "uploaded": success, "dir_url": dir_url}
