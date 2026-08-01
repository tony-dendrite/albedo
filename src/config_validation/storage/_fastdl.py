"""Fast, verifiable shard downloads from the HF hub.

huggingface_hub >= 1.0 removed ``hf_transfer`` support: every file is fetched over a
single HTTP stream through the xet CAS bridge, which throttles and wedges long-lived
connections (streams decay to zero bytes while keep-alive chunks defeat the read
timeout, or drop mid-file). Worse, each retry writes to a fresh uuid-suffixed
``*.incomplete`` opened at byte 0, so a killed attempt re-pays whole shards and
orphans multi-GB temp files that nothing cleans.

This module fetches ``*.safetensors`` payloads with the still-installable
``hf_transfer`` engine directly (parallel short-lived ranged requests), verifies each
file's sha256 against its LFS metadata before install, installs atomically, and
writes the hub-compatible sidecar so hub and this module both trust — and skip —
completed files on any later attempt. Stale temp files from dead attempts are removed
up front, and only files that fail verification are re-fetched. Small files
(configs/tokenizer/index) stay on ``snapshot_download``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Env-tunable per host. 8 connections x 10MB ranged chunks mirrors what hub itself
# used before dropping hf_transfer.
CONNECTIONS = int(os.environ.get("ALBEDO_FASTDL_CONNECTIONS", "8"))
CHUNK_BYTES = int(os.environ.get("ALBEDO_FASTDL_CHUNK_MB", "10")) * 1024 * 1024
FILE_RETRIES = int(os.environ.get("ALBEDO_FASTDL_FILE_RETRIES", "3"))
# Grows linearly per attempt. Instant retries burn the whole budget inside transient
# server-side throttling windows (hf CDN 403 "no permits available" lasts seconds).
RETRY_BACKOFF_S = float(os.environ.get("ALBEDO_FASTDL_RETRY_BACKOFF_S", "10"))

_TMP_SUFFIX = ".fastdl"


@dataclass(frozen=True)
class ShardSpec:
    name: str
    size: int
    sha256: str


def available() -> bool:
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        return False
    return True


def plan_shards(repo: str, revision: str, token: str | None) -> list[ShardSpec]:
    """One metadata call -> (name, size, sha256) for every safetensors payload."""
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
    shards: list[ShardSpec] = []
    for sib in info.siblings or []:
        if not sib.rfilename.endswith(".safetensors"):
            continue
        if sib.lfs is None:
            # Payload files are always LFS; without it there is no digest to verify.
            raise RuntimeError(f"{repo}@{revision}: {sib.rfilename} has no LFS metadata")
        shards.append(ShardSpec(sib.rfilename, sib.lfs.size, sib.lfs.sha256))
    return shards


def _sidecar_dir(dest: Path) -> Path:
    return dest / ".cache" / "huggingface" / "download"


def _sidecar_ok(dest: Path, spec: ShardSpec, revision: str) -> bool:
    path = _sidecar_dir(dest) / f"{spec.name}.metadata"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[0] == revision and lines[1] == spec.sha256
    except (OSError, IndexError):
        return False


def _write_sidecar(dest: Path, spec: ShardSpec, revision: str) -> None:
    # Same 3-line format hub writes (commit, etag, timestamp). Written only after the
    # verified file is in place, with timestamp > file mtime, so hub trusts it too.
    side = _sidecar_dir(dest)
    side.mkdir(parents=True, exist_ok=True)
    (side / f"{spec.name}.metadata").write_text(
        f"{revision}\n{spec.sha256}\n{time.time()}\n", encoding="utf-8"
    )


def _complete(dest: Path, spec: ShardSpec, revision: str) -> bool:
    try:
        if (dest / spec.name).stat().st_size != spec.size:
            return False
    except OSError:
        return False
    return _sidecar_ok(dest, spec, revision)


def _clean_stale(dest: Path) -> None:
    """Drop temp corpses from dead attempts — hub's uuid ``*.incomplete`` never resume."""
    side = _sidecar_dir(dest)
    if side.is_dir():
        for stale in side.glob("*.incomplete"):
            stale.unlink(missing_ok=True)
    for stale in dest.glob(f"*{_TMP_SUFFIX}"):
        stale.unlink(missing_ok=True)


def _sha256(path: Path, heartbeat: Path | None = None) -> str:
    """Hash ``path``; optionally append a disk block to ``heartbeat`` every ~256MB.

    Verification writes no payload bytes, so a byte-growth stall watchdog watching the
    dest dir would see silence for the whole hash of a multi-GB shard and could
    false-kill a healthy attempt. The heartbeat file keeps allocated blocks growing
    (st_blocks only advances per block, hence 4KiB writes) at ~16KiB per verified GB.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for index, block in enumerate(iter(lambda: handle.read(1 << 22), b"")):
            digest.update(block)
            if heartbeat is not None and index % 64 == 0:
                with heartbeat.open("ab") as pulse:
                    pulse.write(b"\0" * 4096)
    return digest.hexdigest()


def _resolve_signed_url(url: str, headers: dict[str, str]) -> str:
    import urllib.error
    import urllib.request

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=30):
            return url  # served directly, no redirect to follow
    except urllib.error.HTTPError as err:
        location = err.headers.get("Location") if 300 <= err.code < 400 else None
        if location is None:
            raise
        return location


def _fetch_one(url: str, tmp: Path, spec: ShardSpec, headers: dict[str, str]) -> None:
    import hf_transfer

    tmp.unlink(missing_ok=True)
    hf_transfer.download(
        url,
        str(tmp),
        CONNECTIONS,
        CHUNK_BYTES,
        parallel_failures=4,
        max_retries=8,
        headers=headers,
    )
    size = tmp.stat().st_size
    if size != spec.size:
        raise RuntimeError(f"{spec.name}: downloaded size {size} != {spec.size}")


def fetch_shards(repo: str, revision: str, dest: Path, token: str | None) -> None:
    """Fetch every missing/broken safetensors payload of ``repo@revision`` into ``dest``.

    ``revision`` must be the pinned commit sha (it is, on both albedo HF paths).
    Raises after ``FILE_RETRIES`` failed attempts on any single file.
    """
    shards = plan_shards(repo, revision, token)
    _clean_stale(dest)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    todo = [spec for spec in shards if not _complete(dest, spec, revision)]
    log.info("fastdl: fetching %d/%d shards for %s@%s", len(todo), len(shards), repo, revision)
    for spec in todo:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{spec.name}"
        tmp = dest / (spec.name + _TMP_SUFFIX)
        last_err: Exception | None = None
        for attempt in range(1, FILE_RETRIES + 1):
            try:
                # Re-resolve per attempt: signed URLs expire and throttling windows pass.
                _fetch_one(_resolve_signed_url(url, headers), tmp, spec, headers)
                actual = _sha256(tmp, heartbeat=dest / f"verify-progress{_TMP_SUFFIX}")
                if actual != spec.sha256:
                    raise RuntimeError(f"{spec.name}: sha256 {actual} != {spec.sha256}")
                os.replace(tmp, dest / spec.name)
                _write_sidecar(dest, spec, revision)
                last_err = None
                break
            except Exception as err:  # noqa: BLE001 — every failure mode retries the file
                tmp.unlink(missing_ok=True)
                last_err = err
                log.warning(
                    "fastdl: %s attempt %d/%d failed: %s", spec.name, attempt, FILE_RETRIES, err
                )
                if attempt < FILE_RETRIES:
                    time.sleep(RETRY_BACKOFF_S * attempt)
        if last_err is not None:
            raise RuntimeError(
                f"fastdl: {spec.name} failed after {FILE_RETRIES} attempts"
            ) from last_err
    (dest / f"verify-progress{_TMP_SUFFIX}").unlink(missing_ok=True)
