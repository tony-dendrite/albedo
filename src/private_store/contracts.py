from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from private_store.digests import ArtifactIntegrityError, model_digest_from_inventory

PROTOCOL_VERSION = 1
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")
_ACTIVATION = re.compile(r"^r2activate:v1:(?P<pubkey>[A-Za-z0-9_-]{43})$")
_READY_SIGNAL = re.compile(r"^r2ready:v1:(?P<manifest_sha256>[0-9a-f]{64})$")
CANONICAL_MODEL_PREFIX = re.compile(
    r"^models/(?:registrations/(?P<first>[0-9a-f]{64})"
    r"|attempts/(?P<retry>[0-9a-f]{64})/a(?P<attempt>[2-9]|[1-9][0-9]+))/$"
)


def _require_hash(value: str, field: str) -> str:
    if not _HEX_SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_ss58(value: str, field: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} is not a canonical SS58-like identifier")
    return value


def _require_chain_generation(value: str) -> str:
    if not value or "|" in value or len(value) > 128:
        raise ValueError("chain_generation must be non-empty and delimiter-safe")
    return value


def registration_id(*, netuid: int, hotkey: str, chain_generation: str) -> str:
    """Derive one registration identity from public chain data.

    Hotkeys are one-shot on albedo, so the hotkey alone identifies the
    occupancy — no uid or registration block needed.
    """
    if netuid < 0:
        raise ValueError("netuid must be non-negative")
    _require_ss58(hotkey, "hotkey")
    _require_chain_generation(chain_generation)
    body = json.dumps(
        {"chain_generation": chain_generation, "hotkey": hotkey, "netuid": netuid},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"albedo-registration-v1\0" + body).hexdigest()


def model_prefix(registration_id: str, attempt: int = 1) -> str:
    """Return the R2 key prefix owning one upload attempt.

    Attempt 1 lives at `models/registrations/<rid>/` and retries under a sibling
    root, so no attempt's prefix contains another's: clearing one never touches the
    others, and each attempt's objects are inventoried on their own.
    """
    _require_hash(registration_id, "registration_id")
    if attempt < 1:
        raise ValueError("upload attempt must start at one")
    if attempt == 1:
        return f"models/registrations/{registration_id}/"
    return f"models/attempts/{registration_id}/a{attempt}/"


def parse_model_prefix(prefix: str) -> tuple[str, int]:
    """Recover (registration_id, attempt) from a prefix; the inverse of model_prefix."""
    match = CANONICAL_MODEL_PREFIX.fullmatch(prefix)
    if match is None:
        raise ValueError(f"not a canonical model prefix: {prefix!r}")
    attempt = match["attempt"]
    return (match["first"] or match["retry"]), int(attempt) if attempt else 1


def activation_signal_payload(submission_pubkey: bytes) -> str:
    if len(submission_pubkey) != 32:
        raise ValueError("submission public key must contain 32 bytes (Ed25519)")
    encoded = base64.urlsafe_b64encode(submission_pubkey).rstrip(b"=").decode("ascii")
    return f"r2activate:v1:{encoded}"


def parse_activation_pubkey(payload: str) -> bytes:
    """Return the 32-byte Ed25519 submission public key from an r2activate:v1 commit."""
    match = _ACTIVATION.fullmatch(payload)
    if match is None:
        raise ValueError("activation signal is not a valid r2activate:v1 commitment")
    pubkey = base64.urlsafe_b64decode(match.group("pubkey") + "=")
    if len(pubkey) != 32:
        raise ValueError("activation signal must carry a 32-byte Ed25519 public key")
    return pubkey


def ready_signal_payload(manifest_sha256: str) -> str:
    _require_hash(manifest_sha256, "manifest_sha256")
    return f"r2ready:v1:{manifest_sha256}"


def parse_ready_signal(payload: str) -> str:
    """Return the manifest SHA-256 from an r2ready:v1 commitment."""
    match = _READY_SIGNAL.fullmatch(payload)
    if match is None:
        raise ValueError("ready signal is not a valid r2ready:v1 commitment")
    return match.group("manifest_sha256")


def mailbox_object_key(registration_id: str, generation: int) -> str:
    """Return the immutable public mailbox key for one credential generation."""
    _require_hash(registration_id, "registration_id")
    if generation < 1:
        raise ValueError("credential generation must start at one")
    return f"mailbox/v1/{registration_id}/generations/{generation:020d}.bin"


@dataclass(frozen=True, slots=True, order=True)
class ManifestFile:
    path: str
    size: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: object) -> "ManifestFile":
        if not isinstance(value, Mapping) or set(value) != {"path", "size", "sha256"}:
            raise ValueError("manifest file entries require exactly path, size, and sha256")
        path = value["path"]
        size = value["size"]
        sha256 = value["sha256"]
        if not isinstance(path, str) or not path or path == "manifest.json":
            raise ValueError("manifest file path is invalid")
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != path:
            raise ValueError(f"unsafe manifest path: {path!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"manifest size is invalid for {path!r}")
        if not isinstance(sha256, str) or not _HEX_SHA256.fullmatch(sha256):
            raise ValueError(f"manifest SHA-256 is invalid for {path!r}")
        return cls(path=path, size=size, sha256=sha256)


@dataclass(frozen=True, slots=True)
class Manifest:
    registration_id: str
    hotkey: str
    model_name: str
    files: tuple[ManifestFile, ...]
    model_digest: str
    signature: str
    protocol_version: int = 1
    signature_scheme: str = "ed25519"

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Manifest":
        try:
            value = json.loads(raw)
        except Exception as exc:
            raise ValueError("manifest is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("manifest must be an object")
        required = {
            "protocol_version",
            "signature_scheme",
            "registration_id",
            "hotkey",
            "model_name",
            "files",
            "model_digest",
            "signature",
        }
        if set(value) != required:
            raise ValueError(
                f"manifest fields differ from v1 contract: {sorted(set(value) - required)}"
            )
        if value["protocol_version"] != 1 or value["signature_scheme"] != "ed25519":
            raise ValueError("manifest requires protocol v1 with Ed25519")
        if not isinstance(value["registration_id"], str) or not _HEX_SHA256.fullmatch(
            value["registration_id"]
        ):
            raise ValueError("manifest registration_id is invalid")
        if not isinstance(value["hotkey"], str) or not value["hotkey"]:
            raise ValueError("manifest hotkey is required")
        if not isinstance(value["model_name"], str) or not value["model_name"].strip():
            raise ValueError("manifest model_name is required")
        if not isinstance(value["files"], list):
            raise ValueError("manifest files must be an array")
        files = tuple(ManifestFile.from_mapping(item) for item in value["files"])
        if not files or len({item.path for item in files}) != len(files):
            raise ValueError("manifest must contain a non-empty unique file inventory")
        if not isinstance(value["model_digest"], str) or not _HEX_SHA256.fullmatch(
            value["model_digest"]
        ):
            raise ValueError("manifest model_digest is invalid")
        try:
            observed = model_digest_from_inventory(
                [(item.path, item.size, item.sha256) for item in files]
            )
        except ArtifactIntegrityError as exc:
            raise ValueError(str(exc)) from exc
        if observed != value["model_digest"]:
            raise ValueError("manifest model_digest does not match its file inventory")
        if not isinstance(value["signature"], str) or not value["signature"]:
            raise ValueError("manifest signature is required")
        return cls(
            registration_id=value["registration_id"],
            hotkey=value["hotkey"],
            model_name=value["model_name"].strip(),
            files=files,
            model_digest=value["model_digest"],
            signature=value["signature"],
        )

    def signing_payload(self) -> bytes:
        value: dict[str, Any] = {
            "files": [
                {"path": item.path, "sha256": item.sha256, "size": item.size} for item in self.files
            ],
            "hotkey": self.hotkey,
            "model_digest": self.model_digest,
            "model_name": self.model_name,
            "protocol_version": self.protocol_version,
            "registration_id": self.registration_id,
            "signature_scheme": self.signature_scheme,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def as_dict(self) -> dict[str, Any]:
        return {**json.loads(self.signing_payload()), "signature": self.signature}

    def as_bytes(self) -> bytes:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.as_bytes()).hexdigest()
