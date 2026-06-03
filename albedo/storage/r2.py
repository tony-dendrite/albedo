"""Cloudflare R2 client for validator-private state JSON. Never raises; logs on failure."""
from __future__ import annotations

import json
import logging
import os

import boto3
from botocore.config import Config

log = logging.getLogger(__name__)

_BOTO_CFG = Config(
    connect_timeout=15,
    read_timeout=45,
    retries={"mode": "adaptive", "max_attempts": 3},
)


class R2Store:
    """Low-level R2 get/put/delete for JSON objects."""

    def __init__(self) -> None:
        self._bucket = os.environ.get("ALBEDO_R2_BUCKET", "")
        self._client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("ALBEDO_R2_ENDPOINT", ""),
            aws_access_key_id=os.environ.get("ALBEDO_R2_ACCESS_KEY", ""),
            aws_secret_access_key=os.environ.get("ALBEDO_R2_SECRET_KEY", ""),
            config=_BOTO_CFG,
        )

    def get(self, key: str) -> dict | None:
        """Return parsed JSON for *key*, or None on missing/error."""
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return json.loads(resp["Body"].read())
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception as exc:  # network, auth, malformed JSON, …
            log.warning("R2 get(%r) failed: %s", key, exc)
            return None

    def put(self, key: str, data: dict) -> bool:
        """Serialise *data* to JSON and upload to *key*. Returns True on success."""
        try:
            body = json.dumps(data, default=str).encode()
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
            return True
        except Exception as exc:
            log.warning("R2 put(%r) failed: %s", key, exc)
            return False

    def delete(self, key: str) -> None:
        """Delete *key*; treats missing key as success."""
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            log.warning("R2 delete(%r) failed: %s", key, exc)
