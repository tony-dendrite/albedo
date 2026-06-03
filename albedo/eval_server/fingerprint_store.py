"""albedo.eval_server.fingerprint_store — persist near-duplicate fingerprint state to S3.

Reuses the eval-trace S3 credentials (ALBEDO_EVALS_S3_*). Best-effort: when the store
is unconfigured the server degrades to in-memory only — the king is still re-fingerprinted
on /set_king and challengers within a run are still compared, but dedup state is not
retained across eval-server restarts.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

_BUCKET     = os.environ.get("ALBEDO_EVALS_S3_BUCKET", "")
_ENDPOINT   = os.environ.get("ALBEDO_EVALS_S3_ENDPOINT", "")
_ACCESS_KEY = os.environ.get("ALBEDO_EVALS_S3_ACCESS_KEY", "")
_SECRET_KEY = os.environ.get("ALBEDO_EVALS_S3_SECRET_KEY", "")
_PREFIX     = os.environ.get("ALBEDO_EVALS_S3_PREFIX", "evals")

_ENABLED = bool(_BUCKET and _ENDPOINT and _ACCESS_KEY and _SECRET_KEY)
_KEY = f"{_PREFIX.rstrip('/')}/state/model_fingerprints.json"


def _client():
    import boto3
    from botocore.client import Config as BotoConfig
    return boto3.client(
        "s3", endpoint_url=_ENDPOINT,
        aws_access_key_id=_ACCESS_KEY, aws_secret_access_key=_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4"),
    )


def load_fingerprints() -> dict[str, dict]:
    """Load persisted fingerprint state from S3; returns {} if unconfigured or absent."""
    if not _ENABLED:
        return {}
    try:
        obj = _client().get_object(Bucket=_BUCKET, Key=_KEY)
        data = json.loads(obj["Body"].read())
        return data if isinstance(data, dict) else {}
    except Exception:
        log.info("fingerprint_store: no existing state at s3://%s/%s (starting empty)", _BUCKET, _KEY)
        return {}


def save_fingerprints(state: dict[str, dict]) -> None:
    """Persist fingerprint state to S3 (best-effort; logs and continues on failure)."""
    if not _ENABLED:
        return
    try:
        body = json.dumps(state, separators=(",", ":")).encode()
        _client().put_object(Bucket=_BUCKET, Key=_KEY, Body=body, ContentType="application/json")
        log.debug("fingerprint_store: saved %d fingerprints to s3://%s/%s", len(state), _BUCKET, _KEY)
    except Exception:
        log.warning("fingerprint_store: save failed", exc_info=True)
