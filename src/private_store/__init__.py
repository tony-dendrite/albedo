"""Private R2 model submissions: credentials, commitments, manifest verification.

Ported from the proven implementation in the teutonic subnet. Miners upload
models into a prefix-jailed private bucket with validator-issued temporary
credentials; upload access is revoked before any verification runs.
"""

from private_store.cloudflare import CloudflareR2TokenGateway, ParentToken
from private_store.contracts import (
    Manifest,
    ManifestFile,
    activation_signal_payload,
    mailbox_object_key,
    model_prefix,
    parse_activation_pubkey,
    parse_ready_signal,
    ready_signal_payload,
    registration_id,
)
from private_store.crypto import MailboxCipher, verify_ed25519, verify_hotkey_signature
from private_store.digests import (
    ArtifactIntegrityError,
    model_digest_from_inventory,
    sha256_file,
    snapshot_digest,
    verify_snapshot,
)
from private_store.r2_credentials import (
    MAX_TEMPORARY_CREDENTIAL_TTL_SECONDS,
    TemporaryCredentials,
    create_local_temporary_credentials,
)
from private_store.settings import PrivateStoreSettings
from private_store.storage import MailboxStore, R2UploadController, UploadQuotaExceeded

__all__ = [
    "ArtifactIntegrityError",
    "CloudflareR2TokenGateway",
    "MailboxCipher",
    "MailboxStore",
    "Manifest",
    "ManifestFile",
    "MAX_TEMPORARY_CREDENTIAL_TTL_SECONDS",
    "ParentToken",
    "PrivateStoreSettings",
    "R2UploadController",
    "TemporaryCredentials",
    "UploadQuotaExceeded",
    "activation_signal_payload",
    "create_local_temporary_credentials",
    "mailbox_object_key",
    "model_digest_from_inventory",
    "model_prefix",
    "parse_activation_pubkey",
    "parse_ready_signal",
    "ready_signal_payload",
    "registration_id",
    "sha256_file",
    "snapshot_digest",
    "verify_ed25519",
    "verify_hotkey_signature",
    "verify_snapshot",
]
