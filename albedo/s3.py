"""One Hippius S3 client - fault reports, guard detections, fingerprint corpus, sanity results."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import boto3
from loguru import logger

from albedo.settings import get_settings

_client = None


def client():
    # Lazy boto3 client for the Hippius S3 endpoint; None when S3 is unconfigured.
    global _client
    s3 = get_settings().s3
    if not s3.enabled:
        return None
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=s3.endpoint,
            aws_access_key_id=s3.access_key,
            aws_secret_access_key=s3.secret_key,
            region_name="decentralized",
        )
    return _client


async def put_json(key: str, payload: dict[str, Any]) -> str | None:
    # Uploads a public-read JSON document; returns the object URI or None when S3 is off/failing.
    c = client()
    if c is None:
        logger.debug(f"[s3] skipped put of {key} - S3 not configured")
        return None
    s3 = get_settings().s3
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    try:
        await asyncio.to_thread(
            c.put_object,
            Bucket=s3.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ACL="public-read",
        )
        return f"s3://{s3.bucket}/{key}"
    except Exception:  # noqa: BLE001 - S3 publish is best-effort, never fails the pipeline
        logger.exception(f"[s3] put_json failed for {key}")
        return None
