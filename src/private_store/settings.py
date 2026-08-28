from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from private_store.r2_credentials import MAX_TEMPORARY_CREDENTIAL_TTL_SECONDS
from private_store.storage import MAX_MINER_UPLOAD_BYTES


class PrivateStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="R2_",
        extra="ignore",
        populate_by_name=True,
    )

    account_id: str = Field(..., description="Cloudflare account id")
    api_token: str = Field(..., description="Account API token that mints/revokes child tokens")
    access_key_id: str = Field(..., description="Master S3 key id (verification, mailbox, cleanup)")
    secret_access_key: str = Field(..., description="Master S3 secret")
    endpoint: str = Field(..., description="https://<account>.r2.cloudflarestorage.com")
    private_models_bucket_name: str = Field(..., description="Private bucket miners upload into")
    mailbox_bucket_name: str = Field(
        ..., description="Public bucket for encrypted credential envelopes"
    )
    mailbox_public_base_url: str = Field(
        "", description="Public r2.dev base URL of the mailbox bucket"
    )
    mailbox_signing_key: str = Field(
        ..., description="32-byte hex Ed25519 seed signing mailbox envelopes"
    )
    jurisdiction: str = "default"
    chain_generation: str = "albedo-mainnet-1"
    credential_ttl_seconds: int = MAX_TEMPORARY_CREDENTIAL_TTL_SECONDS
    max_upload_bytes: int = MAX_MINER_UPLOAD_BYTES
    upload_window_seconds: float = 86_400.0
    keep_recent_losers: int = 20
    poll_interval_s: float = 10.0
    sweep_interval_s: float = 120.0

    def mailbox_signing_key_bytes(self) -> bytes:
        key = bytes.fromhex(self.mailbox_signing_key)
        if len(key) != 32:
            raise ValueError("R2_MAILBOX_SIGNING_KEY must be 32 bytes of hex")
        return key
