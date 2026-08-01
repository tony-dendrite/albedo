from __future__ import annotations

import logging
import os
import sys

from config_validation.models import ModelRef
from config_validation.storage import _supervise
from config_validation.storage._paths import _cache_dir

log = logging.getLogger(__name__)

_TOKEN_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
_CONFIG_ONLY_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
]
_MODEL_ONLY_PATTERNS = ["*.safetensors", "model.safetensors.index.json"]


def _token() -> str | None:
    for env in _TOKEN_ENVS:
        tok = os.environ.get(env)
        if tok:
            return tok
    return None


def _download_child() -> None:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    from config_validation.storage import _fastdl

    repo, revision, local_dir, max_workers = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    if _fastdl.available():
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=["model.safetensors.index.json"],
            token=_token(),
        )
        _fastdl.fetch_shards(repo, revision, Path(local_dir), _token())
        return
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=local_dir,
        max_workers=max(1, int(max_workers)),
        allow_patterns=_MODEL_ONLY_PATTERNS,
        token=_token(),
    )


def _download(ref: ModelRef, *, config_only: bool, max_workers: int) -> str:
    dest = _cache_dir(ref)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("hf: downloading %s (config_only=%s) → %s", ref.immutable_ref, config_only, dest)
    if config_only or not _supervise.OUT_OF_PROCESS:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=ref.repo,
            revision=ref.digest,
            local_dir=str(dest),
            max_workers=max_workers,
            allow_patterns=_CONFIG_ONLY_PATTERNS if config_only else _MODEL_ONLY_PATTERNS,
            token=_token(),
        )
        return str(dest)
    _supervise.supervise_download(
        child_call="from config_validation.storage._hf import _download_child; _download_child()",
        args=[ref.repo, ref.digest, str(dest), str(max_workers)],
        watch_dir=dest,
        label=ref.immutable_ref,
    )
    return str(dest)


def download_config(ref: ModelRef) -> str:
    return _download(ref, config_only=True, max_workers=8)


def download_full(ref: ModelRef) -> str:
    return _download(ref, config_only=False, max_workers=8)


def list_files(ref: ModelRef) -> list[str]:
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id=ref.repo, revision=ref.digest, token=_token()))


def revision_resolves(ref: ModelRef) -> tuple[bool, str]:
    try:
        files = list_files(ref)
    except Exception as exc:
        log.error(f"revision {ref.digest} did not resolve on HuggingFace repo={ref.repo}: {exc}")
        return False, f"revision {ref.digest} did not resolve on HuggingFace: {exc}"
    if not files:
        return False, f"revision {ref.digest} resolved but the repo is empty"
    return True, f"revision resolved ({len(files)} files)"
