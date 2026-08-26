from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

from loguru import logger as log

from albedo_config.chain_spec import MODEL_CACHE_DIR
from config_validation.models import ModelRef
from config_validation.storage import cache_dir as _cache_dir
from config_validation.storage import download_config as _download_config
from config_validation.storage import download_full as _download_full
from config_validation.storage import list_files as _list_files

MAX_CACHED_MODELS = int(os.environ.get("ALBEDO_MODEL_CACHE_MAX", "4"))


def make_room(ref: ModelRef, protected_repos: Iterable[str] = ()) -> None:
    """Evict newest cached models until `ref` fits within MAX_CACHED_MODELS."""
    root = Path(MODEL_CACHE_DIR).resolve()
    keep = _cache_dir(ref)
    guarded = set(protected_repos)
    models: list[tuple[float, Path]] = []
    for digest_dir in root.glob("*/*/*/*"):  # <backend>/<owner>/<name>/<digest>
        if not digest_dir.is_dir() or digest_dir == keep:
            continue
        if "/".join(digest_dir.parts[-3:-1]) in guarded:
            continue
        with suppress(OSError):
            models.append((digest_dir.stat().st_mtime, digest_dir))
    models.sort()
    while len(models) > max(MAX_CACHED_MODELS - 1, 0):
        _mtime, victim = models.pop()
        log.info("cache: evicting {} to make room for {}", victim, ref.immutable_ref)
        shutil.rmtree(victim, ignore_errors=True)
        for parent in (victim.parent, victim.parent.parent):
            with suppress(OSError):
                parent.rmdir()


def make_ref(repo: str, digest: str) -> ModelRef:
    return ModelRef(repo=repo, digest=digest)


def cache_dir(ref: ModelRef) -> Path:
    return _cache_dir(ref)


def list_files(ref: ModelRef) -> list[str]:
    return _list_files(ref)


def download_config(ref: ModelRef) -> str:
    return _download_config(ref)


def download_full(ref: ModelRef) -> str:
    return _download_full(ref)
