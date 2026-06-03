"""albedo.validator.admission — Config validation gate before eval dispatch."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from albedo.config import REPO_PATTERN
from albedo.models import (
    ModelRef, config_lock_violation, list_remote_files, materialize_model,
)

log = logging.getLogger(__name__)

_SAFETENSORS_SUFFIX = ".safetensors"
_PY_SUFFIX = ".py"


def _load_config_json(local_dir: str) -> dict[str, Any]:
    cfg_path = Path(local_dir) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    with cfg_path.open() as fh:
        return json.load(fh)


def validate_challenger_config(
    model_repo: str,
    challenger_digest: str,
    king_repo: str,
    king_digest: str,
) -> str | None:
    """Return a rejection reason string, or None if the challenger passes all checks.

    Checks: repo pattern, sha256 digest prefix, config.json downloadable,
    architectures/ALL_LOCK_KEYS match king, no auto_map, no .py files,
    at least one .safetensors. King config check is skipped (with warning)
    if the hub is unavailable, to avoid blocking valid challengers.
    """
    if not re.match(REPO_PATTERN, model_repo):
        return f"repo name does not match required pattern: {model_repo!r}"

    if not challenger_digest.startswith("sha256:"):
        return f"digest must start with 'sha256:': {challenger_digest!r}"

    try:
        challenger_ref = ModelRef(repo=model_repo, digest=challenger_digest)
        challenger_dir = materialize_model(challenger_ref, config_only=True)
        challenger_cfg = _load_config_json(challenger_dir)
    except Exception as exc:
        return f"could not download challenger config.json: {exc}"

    # Load king config best-effort; skip lock checks if unavailable
    king_cfg: dict[str, Any] | None = None
    try:
        king_ref = ModelRef(repo=king_repo, digest=king_digest)
        king_dir = materialize_model(king_ref, config_only=True)
        king_cfg = _load_config_json(king_dir)
    except Exception as exc:
        log.warning(
            "validate_challenger_config: king config unavailable, skipping lock check: %s", exc
        )

    if king_cfg is not None:
        reason = config_lock_violation(king_cfg, challenger_cfg)
        if reason:
            return reason

    if "auto_map" in challenger_cfg:
        return "config.json must not contain 'auto_map'"

    if "quantization_config" in challenger_cfg:
        return "config.json must not contain 'quantization_config' — quantized models not allowed"

    try:
        remote_files = list_remote_files(challenger_ref)
    except Exception as exc:
        return f"could not list remote files: {exc}"

    py_files = [f for f in remote_files if f.endswith(_PY_SUFFIX)]
    if py_files:
        return f"repo contains .py files: {py_files[:5]}"

    safetensor_files = [f for f in remote_files if f.endswith(_SAFETENSORS_SUFFIX)]
    if not safetensor_files:
        return "repo contains no .safetensors files"

    return None
