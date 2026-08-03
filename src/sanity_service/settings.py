
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from sanity_service.config import DB_URL as _DEFAULT_DB_URL


class SanitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SANITY_DISPATCH_", extra="ignore")

    database_url: str = _DEFAULT_DB_URL
    worker_id: str = "sanity-dispatcher"
    remote_auth_token: str = ""
    consensus: bool = False

    dataset_manifest_path: str = ""
    dataset_manifest_hash: str = "e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40"
    dataset_root: str = ""
    sample_count: int = 3
    trajectory_assistant_turns: int = 8
    gen_max_tokens: int = 16384

    skip_viability: bool = False

    lease_seconds: int = 600
    dispatch_poll_seconds: float = 5.0
    remote_event_timeout_seconds: float = 30.0
    remote_event_poll_seconds: float = 5.0
    min_free_gpus: int = 1
    max_retry_count: int = 5


@lru_cache
def get_settings() -> SanitySettings:
    return SanitySettings()
