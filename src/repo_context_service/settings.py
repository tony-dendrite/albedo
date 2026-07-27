from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RepoContextSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_REPO_CONTEXT_",
        extra="ignore",
    )

    api_host: str = "127.0.0.1"
    api_port: int = 8093
    # Explicit snapshot download directory — REQUIRED; the service refuses to start without it.
    cache_dir: str = ""
    dataset_manifest_path: str = ""
    dataset_manifest_hash: str = ""
    # Parquet shards root (= ALBEDO_REMOTE_DATASET_ROOT) — needed for the trajectory fallback.
    dataset_root: str = ""
    github_token: str = ""
    max_paths: int = 2000
    max_files: int = 12
    max_file_chars: int = 30000
    max_context_chars: int = 120000
    max_snapshot_mb: int = 500
    # Total cap on the snapshot cache; when reached the snapshots dir is cleared (0 = unlimited).
    max_cache_gb: float = 60.0
    max_trajectory_pairs: int = 8


@lru_cache
def get_repo_context_settings() -> RepoContextSettings:
    return RepoContextSettings()
