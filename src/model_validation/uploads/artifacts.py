from __future__ import annotations

import functools
import json

from loguru import logger as log

from albedo_config import get_model_validation_settings

config = get_model_validation_settings()

ENABLED = bool(config.S3_BUCKET and config.S3_ACCESS_KEY and config.S3_SECRET_KEY)


@functools.lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY,
        region_name="auto",
        config=Config(
            connect_timeout=15, read_timeout=60, retries={"mode": "adaptive", "max_attempts": 3}
        ),
    )


def _safe_digest(digest: str) -> str:
    return digest.replace(":", "_")


def _put(key: str, data: dict) -> str | None:
    if not ENABLED:
        log.debug("S3 disabled (ALBEDO_S3_* unset); skipping put({})", key)
        return None
    try:
        _client().put_object(
            Bucket=config.S3_BUCKET,
            Key=key,
            Body=json.dumps(data, default=str).encode(),
            ContentType="application/json",
            ACL="public-read",
        )
        uri = f"s3://{config.S3_BUCKET}/{key}"
        log.info("uploaded artifact {}", uri)
        return uri
    except Exception as exc:
        log.warning("S3 put({}) failed: {}", key, exc)
        return None


def put_fault(hotkey: str, digest: str, detail: dict) -> str | None:
    key = f"hippius_validation/{hotkey}/{_safe_digest(digest)}/fault.json"
    return _put(key, detail)
