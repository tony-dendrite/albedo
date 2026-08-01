from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .sampling import SAMPLING_ALGO


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ALBEDO_EVAL_",
        extra="ignore",
    )

    database_url: str = Field(..., description="Postgres DSN")
    worker_id: str = "eval-dispatcher"
    remote_auth_token: str = ""

    dataset_version: str = "mini-coder+open-swe+smith-rs+hero-v1"
    dataset_manifest_uri: str
    dataset_manifest_hash: str = "e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40"
    dataset_manifest_path: str | None = None
    sample_count: int = 100
    sampling_algo: str = SAMPLING_ALGO
    judge_config_hash: str
    judge_count: int = 3

    artifact_bucket: str = "albedo-artifacts"
    artifact_prefix: str = "s3://albedo-artifacts"

    lease_seconds: int = 1800
    dispatch_poll_seconds: float = 5.0
    remote_event_timeout_seconds: float = 30.0
    remote_event_poll_seconds: float = 5.0
    max_retry_count: int = 3
    prefetch_next_challenger: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
