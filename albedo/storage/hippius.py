"""Hippius public S3 client for dashboard JSON.

Never raises; returns False on failure. A 60-second cooldown after any failure
prevents a Hippius outage from wedging the eval loop.
"""
from __future__ import annotations

import json
import logging
import os
import time

import boto3
from botocore.config import Config

log = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://s3.hippius.com"
_REGION           = "decentralized"
_FAILURE_COOLDOWN = 60.0          # seconds
_DEFAULT_CC       = "no-cache, must-revalidate"

_BOTO_CFG = Config(
    connect_timeout=15,
    read_timeout=45,
    retries={"mode": "adaptive", "max_attempts": 3},
)

_PUBLIC_POLICY_TEMPLATE = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Sid":       "PublicReadGetObject",
        "Effect":    "Allow",
        "Principal": "*",
        "Action":    ["s3:GetObject"],
        "Resource":  ["arn:aws:s3:::{bucket}/*"],
    }],
})


class HippiusStore:
    """Public-read S3 client for Hippius dashboard uploads."""

    def __init__(self) -> None:
        self._bucket = os.environ.get("ALBEDO_DS_BUCKET", "")
        self._client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("ALBEDO_DS_ENDPOINT", _DEFAULT_ENDPOINT),
            aws_access_key_id=os.environ.get("ALBEDO_DS_ACCESS_KEY", ""),
            aws_secret_access_key=os.environ.get("ALBEDO_DS_SECRET_KEY", ""),
            region_name=_REGION,
            config=_BOTO_CFG,
        )
        self._cooldown_until: float = 0.0
        self._ensure_public_bucket()

    def put(self, key: str, data: dict, *, cache_control: str = _DEFAULT_CC) -> bool:
        """Serialise *data* to JSON and upload to *key*."""
        body = json.dumps(data, default=str).encode()
        return self.put_raw(key, body, "application/json", cache_control=cache_control)

    def put_raw(
        self,
        key: str,
        body: bytes,
        content_type: str,
        *,
        cache_control: str = _DEFAULT_CC,
    ) -> bool:
        """Upload raw bytes to *key*; returns False if cooldown active or on failure."""
        if time.monotonic() < self._cooldown_until:
            log.debug("Hippius cooldown active; skipping put(%r)", key)
            return False
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                CacheControl=cache_control,
            )
            return True
        except Exception as exc:
            log.warning("Hippius put_raw(%r) failed: %s", key, exc)
            self._cooldown_until = time.monotonic() + _FAILURE_COOLDOWN
            return False

    def _ensure_public_bucket(self) -> None:
        """Apply a public-read bucket policy so dashboard files are accessible."""
        try:
            policy = _PUBLIC_POLICY_TEMPLATE.replace("{bucket}", self._bucket)
            self._client.put_bucket_policy(Bucket=self._bucket, Policy=policy)
        except Exception as exc:
            log.warning("Hippius _ensure_public_bucket failed: %s", exc)
