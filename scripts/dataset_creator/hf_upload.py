"""Ship chunks to a HF dataset repo, one named split per reference model.

Chunks upload to data/<model>/ only — the repo README/dataset card is managed
by hand, never touched here. Upload status lives in the SQLite ledger, so
failures are retried on the next pipeline tick.
"""

from __future__ import annotations

import os

from config import Config
from state import State


def upload(cfg: Config, state: State, repo: str) -> int:
    """Upload all pending chunks."""
    from huggingface_hub import HfApi
    token = os.environ.get("DATASET_CREATOR_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("DATASET_CREATOR_HF_TOKEN (or HF_TOKEN) env var not set")
    api = HfApi(token=token)

    n = 0
    for rel, model in state.pending_uploads():
        local = cfg.out_dir / model / rel.rsplit("/", 1)[-1]
        api.upload_file(path_or_fileobj=local, path_in_repo=rel,
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"add {local.name}")
        state.mark_uploaded(rel)
        n += 1
        print(f"  uploaded {rel}")
    return n
