from __future__ import annotations

from config_validation.models import BACKEND_HF, BACKEND_HIPPIUS, BACKEND_S3, ModelRef
from config_validation.storage import _hf, _hippius, _s3
from config_validation.storage._paths import cache_dir

_IMPL = {BACKEND_HF: _hf, BACKEND_HIPPIUS: _hippius, BACKEND_S3: _s3}


def _impl(ref: ModelRef):
    try:
        return _IMPL[ref.backend]
    except KeyError:
        raise ValueError(f"unknown model backend {ref.backend!r}") from None


def download_config(ref: ModelRef) -> str:
    return _impl(ref).download_config(ref)


def download_full(ref: ModelRef) -> str:
    return _impl(ref).download_full(ref)


def list_files(ref: ModelRef) -> list[str]:
    return _impl(ref).list_files(ref)


def revision_resolves(ref: ModelRef) -> tuple[bool, str]:
    return _impl(ref).revision_resolves(ref)


__all__ = ["cache_dir", "download_config", "download_full", "list_files", "revision_resolves"]
