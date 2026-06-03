"""albedo.models.upload — push a local model directory to Hippius Hub."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from albedo.models.ref import ModelRef

log = logging.getLogger(__name__)

_HUB_TOKEN_ENV = "HIPPIUS_HUB_TOKEN"


def _hub() -> "hippius_hub.HippiusHub":  # type: ignore[name-defined]  # noqa: F821
    """Lazy-import and construct a HippiusHub client."""
    try:
        import hippius_hub  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hippius_hub is not installed; run: pip install hippius-hub"
        ) from exc
    token = os.environ.get(_HUB_TOKEN_ENV)
    return hippius_hub.HippiusHub(token=token)


def upload_model_folder(
    local_dir: str,
    *,
    repo: str,
    revision: str = "main",
    commit_message: str = "",
) -> ModelRef:
    """Upload local_dir to Hippius Hub and return a pinned ModelRef.

    Raises FileNotFoundError if local_dir is missing, ValueError if the Hub
    returns an unexpected digest format.
    """
    src = Path(local_dir)
    if not src.exists():
        raise FileNotFoundError(f"upload_model_folder: {local_dir!r} does not exist")
    if not src.is_dir():
        raise ValueError(f"upload_model_folder: {local_dir!r} is not a directory")

    msg = commit_message or f"upload via albedo.models.upload_model_folder"
    log.info("upload_model_folder: %s → hippius:%s@%s", src, repo, revision)

    hub = _hub()
    result = hub.upload_folder(
        local_dir=str(src),
        repo=repo,
        revision=revision,
        commit_message=msg,
    )

    # Hub returns {"digest": "sha256:<hex64>"} on success.
    digest: str = result.get("digest", "") if isinstance(result, dict) else str(result)
    if not digest.startswith("sha256:"):
        raise ValueError(
            f"Hub upload returned unexpected digest: {digest!r}. "
            "Expected 'sha256:<hex64>'."
        )

    ref = ModelRef(repo=repo.lower(), digest=digest)
    log.info("upload_model_folder: pinned as %s", ref.immutable_ref)
    return ref
