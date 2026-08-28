"""Private-store (Cloudflare R2) model backend.

Unlike HF/Hippius there is no server-side revision guarantee here, so
download_full re-hashes every byte against the chain-pinned model digest
before the model is allowed into the cache.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

from config_validation.models import ModelRef
from config_validation.storage._hf import _CONFIG_ONLY_PATTERNS
from config_validation.storage._paths import _cache_dir

log = logging.getLogger(__name__)

_LOCATION_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/(?P<prefix>.+[^/])$")
_VERIFIED_SUFFIX = ".albedo-verified"


def _location(ref: ModelRef) -> tuple[str, str]:
    match = _LOCATION_RE.match(ref.repo)
    if match is None:
        raise ValueError(f"not an s3 model ref: {ref.repo!r}")
    return match["bucket"], match["prefix"] + "/"


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )


def _keys(ref: ModelRef) -> list[str]:
    bucket, prefix = _location(ref)
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = _client().list_objects_v2(**kwargs)
        for item in page.get("Contents", []):
            rel = item["Key"][len(prefix) :]
            if rel and not rel.endswith("/"):
                keys.append(rel)
        if not page.get("IsTruncated"):
            return keys
        token = page.get("NextContinuationToken")


def list_files(ref: ModelRef) -> list[str]:
    # the signed manifest is store infrastructure, not part of the model
    return [name for name in _keys(ref) if name != "manifest.json"]


def revision_resolves(ref: ModelRef) -> tuple[bool, str]:
    files = list_files(ref)
    if not files:
        return False, "no objects under the private model prefix"
    return True, f"prefix resolved ({len(files)} files)"


def _safe_target(dest: Path, name: str) -> Path:
    """Resolve an object's relative name under dest, rejecting path escapes."""
    root = dest.resolve()
    target = (dest / name).resolve()
    if target != root and not str(target).startswith(str(root) + os.sep):
        raise ValueError(f"object key escapes the model directory: {name!r}")
    return target


def _fetch(ref: ModelRef, names: list[str]) -> Path:
    from boto3.s3.transfer import TransferConfig

    bucket, prefix = _location(ref)
    dest = _cache_dir(ref)
    dest.mkdir(parents=True, exist_ok=True)
    config = TransferConfig(max_concurrency=16)
    for name in names:
        target = _safe_target(dest, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        _client().download_file(bucket, prefix + name, str(target), Config=config)
    return dest


def _marker(dest: Path) -> Path:
    # lives beside the model dir: an extra file inside would change its digest
    return dest.parent / (dest.name + _VERIFIED_SUFFIX)


def download_config(ref: ModelRef) -> str:
    names = [name for name in list_files(ref) if name in _CONFIG_ONLY_PATTERNS]
    log.info("s3: downloading %s config files for %s", len(names), ref.immutable_ref)
    return str(_fetch(ref, names))


def download_full(ref: ModelRef) -> str:
    from private_store.digests import verify_snapshot

    dest = _cache_dir(ref)
    if _marker(dest).exists() and any(dest.glob("*.safetensors")):
        log.info("s3: reusing verified model at %s", dest)
        return str(dest)
    shutil.rmtree(dest, ignore_errors=True)  # unverified partials start over
    names = list_files(ref)
    log.info("s3: downloading %s files for %s → %s", len(names), ref.immutable_ref, dest)
    _fetch(ref, names)
    verify_snapshot(dest, ref.digest.removeprefix("sha256:"))
    _marker(dest).write_text("")
    return str(dest)


def safetensors_dtypes(ref: ModelRef) -> dict[str, set[str]]:
    bucket, prefix = _location(ref)
    out: dict[str, set[str]] = {}
    for name in list_files(ref):
        if not name.endswith(".safetensors"):
            continue

        def ranged(start: int, end: int) -> bytes:
            response = _client().get_object(
                Bucket=bucket, Key=prefix + name, Range=f"bytes={start}-{end}"
            )
            return response["Body"].read()

        header_len = int.from_bytes(ranged(0, 7), "little")
        header = json.loads(ranged(8, 8 + header_len - 1))
        out[name] = {info["dtype"] for key, info in header.items() if key != "__metadata__"}
    return out
