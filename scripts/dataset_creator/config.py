from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE file; existing keys win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env_file(_REPO_ROOT / ".env")          # deployment values
_load_env_file(_REPO_ROOT / ".env.example")  # fallback for anything unset


def _env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(
        _env("DATASET_CREATOR_DATA_DIR", "data")).expanduser())
    chunk_size: int = 100
    # eval machine: primary artifact source, polled for new run dirs
    eval_ssh_host: str | None = field(default_factory=lambda: _env("DATASET_CREATOR_EVAL_SSH_HOST"))
    eval_artifacts_dir: str | None = field(default_factory=lambda: _env(
        "DATASET_CREATOR_EVAL_ARTIFACTS_DIR"))
    poll_seconds: int = field(default_factory=lambda: int(_env("DATASET_CREATOR_POLL_SECONDS", "300")))
    stall_hours: float = field(default_factory=lambda: float(_env("DATASET_CREATOR_STALL_HOURS", "2")))
    # eval DB access: direct DSN (on infra) or ssh + docker exec (workstation)
    db_url: str | None = field(default_factory=lambda: _env("DATASET_CREATOR_DB_URL"))
    ssh_host: str | None = field(default_factory=lambda: _env("DATASET_CREATOR_DB_SSH_HOST"))
    remote_env_file: str | None = field(default_factory=lambda: _env("DATASET_CREATOR_DB_ENV_FILE"))
    pg_container: str | None = field(default_factory=lambda: _env("DATASET_CREATOR_DB_CONTAINER"))
    # anonymous public read of the artifact bucket
    s3_public: str | None = field(default_factory=lambda: _env("DATASET_CREATOR_S3_BASE"))
    hf_repo: str | None = field(default_factory=lambda: _env("DATASET_CREATOR_HF_REPO"))

    @property
    def db_path(self) -> Path:
        """SQLite ledger: processed runs, cut chunks, upload status, meta."""
        return self.data_dir / "state.db"

    @property
    def pending_dir(self) -> Path:
        """JSON buffers of samples waiting to fill the next chunk (deleted after use)."""
        return self.data_dir / "pending"

    @property
    def out_dir(self) -> Path:
        """Per-model exact-`chunk_size` chunks ready for HF upload."""
        return self.data_dir / "hf_out"

    def run_dir(self, run_id: str) -> Path:
        return self.data_dir / run_id
