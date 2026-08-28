"""Miner side of private R2 submissions: activate, upload, ready.

Any hotkey type (sr25519 or Ed25519) works: the hotkey only authors the on-chain
commits (via set_reveal_commitment, the same channel chain_reader reads). A
one-time Ed25519 "submission key" — generated locally, its pubkey committed at
activate — is what credentials are sealed to and what signs the manifest.

    activate     commit r2activate with the submission pubkey — asks for creds
    upload       poll the sealed mailbox, upload the model + signed manifest
    ready        commit r2ready — freezes the upload and triggers verification

`submit-private` runs activate -> upload -> ready end to end.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from loguru import logger

from private_store.contracts import (
    Manifest,
    ManifestFile,
    activation_signal_payload,
    mailbox_object_key,
    model_digest_from_inventory,
    ready_signal_payload,
    registration_id,
)
from private_store.crypto import MailboxCipher, encode_signature

CHAIN_GENERATION = os.environ.get("R2_CHAIN_GENERATION", "albedo-mainnet-1")
_DEFAULT_MAILBOX_BASE_URL = "https://pub-713d89b84ef44529925592f6cc947b1a.r2.dev"
MAILBOX_BASE_URL = os.environ.get("R2_MAILBOX_PUBLIC_BASE_URL", _DEFAULT_MAILBOX_BASE_URL)
_UPLOAD_AUTH = "upload-auth.json"
_SUBMISSION_KEY = "submission-key"


def _state_dir(hotkey_ss58: str) -> Path:
    path = (Path.cwd() / ".albedo-miner" / hotkey_ss58).resolve()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def submission_key(hotkey_ss58: str):
    """Load, or create-and-persist, this hotkey's one-time Ed25519 submission key.

    The seed lives at .albedo-miner/<hotkey>/submission-key (0600). activate
    commits its public half; keep it — losing it means the sealed credentials
    can't be decrypted and the hotkey must start over with a fresh submission.
    """
    from nacl.signing import SigningKey

    path = _state_dir(hotkey_ss58) / _SUBMISSION_KEY
    if path.exists():
        return SigningKey(bytes.fromhex(path.read_text().strip()))
    seed = os.urandom(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(seed.hex())
    return SigningKey(seed)


def _reveal_commit(wallet, netuid: int, network: str, payload: str, *, assume_yes: bool) -> None:
    import bittensor as bt

    if not assume_yes:
        print(f"about to commit on netuid {netuid} ({network}): {payload}")
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted — nothing committed")
    st = bt.Subtensor(network=network)
    result = st.set_reveal_commitment(
        wallet=wallet, netuid=netuid, data=payload, blocks_until_reveal=1
    )
    if not getattr(result, "success", True):
        raise SystemExit(f"commit failed: {getattr(result, 'message', result)}")
    logger.info("committed: {}", payload)


def activate(wallet, *, netuid: int, network: str, assume_yes: bool = False) -> str:
    ss58 = wallet.hotkey.ss58_address
    key = submission_key(ss58)  # load-or-create the one-time submission key
    payload = activation_signal_payload(bytes(key.verify_key))
    _reveal_commit(wallet, netuid, network, payload, assume_yes=assume_yes)
    return registration_id(netuid=netuid, hotkey=ss58, chain_generation=CHAIN_GENERATION)


def fetch_credentials(
    *, hotkey_ss58: str, registration_id: str, signing_key, timeout_s: float = 900.0
) -> dict:
    """Poll the public mailbox until the validator publishes sealed credentials."""
    if not MAILBOX_BASE_URL:
        raise SystemExit("set R2_MAILBOX_PUBLIC_BASE_URL to the validator's mailbox URL")
    key = mailbox_object_key(registration_id, 1)
    deadline = time.time() + timeout_s
    attempt = 0
    logger.info("waiting for the validator to publish upload credentials…")
    while True:
        try:
            # r2.dev 403s the default urllib UA and edge-caches 404s
            request = urllib.request.Request(
                f"{MAILBOX_BASE_URL}/{key}?attempt={attempt}",
                headers={"User-Agent": "albedo-miner/1.0"},
            )
            with urllib.request.urlopen(request, timeout=15) as resp:
                ciphertext = resp.read()
            envelope = MailboxCipher.decrypt_for_miner(ciphertext, signing_key)
            if envelope["hotkey"] != hotkey_ss58 or envelope["allowed_prefix"].split("/")[2] != (
                registration_id
            ):
                raise SystemExit("mailbox envelope does not match this registration")
            state = _state_dir(hotkey_ss58) / _UPLOAD_AUTH
            fd = os.open(state, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(envelope, handle)
            logger.info("credentials received; prefix {}", envelope["allowed_prefix"])
            return envelope
        except SystemExit:
            raise
        except Exception:
            if time.time() > deadline:
                raise SystemExit("timed out waiting for credentials; is activate finalized?")
            time.sleep(10)
            attempt += 1


def plan_upload(local_dir: str) -> list[tuple[str, int, str, Path]]:
    """Inventory a model dir as (relpath, size, sha256, absolute path)."""
    root = Path(local_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"model path is not a directory: {local_dir}")
    plan: list[tuple[str, int, str, Path]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == "manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        plan.append((rel, path.stat().st_size, digest.hexdigest(), path))
    if not plan:
        raise SystemExit("model directory has no files to upload")
    return plan


def build_manifest(
    signing_key, *, registration_id: str, hotkey: str, model_name: str, plan: list
) -> Manifest:
    files = tuple(ManifestFile(path=rel, size=size, sha256=sha) for rel, size, sha, _ in plan)
    digest = model_digest_from_inventory([(f.path, f.size, f.sha256) for f in files])
    unsigned = Manifest(
        registration_id=registration_id,
        hotkey=hotkey,
        model_name=model_name,
        files=files,
        model_digest=digest,
        signature="pending",
    )
    signature = encode_signature(bytes(signing_key.sign(unsigned.signing_payload()).signature))
    return Manifest(
        registration_id=registration_id,
        hotkey=hotkey,
        model_name=model_name,
        files=files,
        model_digest=digest,
        signature=signature,
    )


def _s3_from_envelope(envelope: dict):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=envelope["r2_endpoint"],
        aws_access_key_id=envelope["access_key_id"],
        aws_secret_access_key=envelope["secret_access_key"],
        aws_session_token=envelope["session_token"],
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )


def upload_objects(client, envelope: dict, manifest: Manifest, plan: list) -> str:
    """Upload every model file, then manifest.json last, each with sha256 metadata."""
    from boto3.s3.transfer import TransferConfig

    bucket = envelope["private_model_bucket"]
    prefix = envelope["allowed_prefix"]
    config = TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=16)
    by_path = {rel: (sha, path) for rel, _size, sha, path in plan}
    for item in manifest.files:
        sha, path = by_path[item.path]
        logger.info("uploading {} ({} bytes)", item.path, item.size)
        client.upload_file(
            str(path),
            bucket,
            prefix + item.path,
            ExtraArgs={"Metadata": {"sha256": sha}},
            Config=config,
        )
    raw = manifest.as_bytes()
    client.put_object(
        Bucket=bucket,
        Key=prefix + "manifest.json",
        Body=raw,
        Metadata={"sha256": hashlib.sha256(raw).hexdigest()},
    )
    return manifest.manifest_sha256


def upload(
    *, hotkey_ss58: str, registration_id: str, signing_key, local_dir: str, model_name: str
) -> str:
    auth = _state_dir(hotkey_ss58) / _UPLOAD_AUTH
    if not auth.exists():
        raise SystemExit("no upload credentials on disk; run activate/fetch first")
    envelope = json.loads(auth.read_text())
    plan = plan_upload(local_dir)
    manifest = build_manifest(
        signing_key,
        registration_id=registration_id,
        hotkey=hotkey_ss58,
        model_name=model_name,
        plan=plan,
    )
    client = _s3_from_envelope(envelope)
    manifest_sha256 = upload_objects(client, envelope, manifest, plan)
    logger.info("upload complete; manifest_sha256 {}", manifest_sha256)
    return manifest_sha256


def ready(wallet, *, netuid: int, network: str, manifest_sha256: str, assume_yes: bool = False):
    _reveal_commit(
        wallet, netuid, network, ready_signal_payload(manifest_sha256), assume_yes=assume_yes
    )
    auth = _state_dir(wallet.hotkey.ss58_address) / _UPLOAD_AUTH
    auth.unlink(missing_ok=True)
    logger.info("ready committed; upload access will be revoked and the model verified")


def submit_private(
    *,
    coldkey: str,
    hotkey: str,
    netuid: int,
    network: str,
    local_dir: str,
    model_name: str,
    assume_yes: bool = False,
) -> str:
    from miner.commit import build_wallet

    wallet = build_wallet(coldkey, hotkey)
    ss58 = wallet.hotkey.ss58_address
    key = submission_key(ss58)
    rid = activate(wallet, netuid=netuid, network=network, assume_yes=assume_yes)
    fetch_credentials(hotkey_ss58=ss58, registration_id=rid, signing_key=key)
    manifest_sha256 = upload(
        hotkey_ss58=ss58,
        registration_id=rid,
        signing_key=key,
        local_dir=local_dir,
        model_name=model_name,
    )
    ready(
        wallet,
        netuid=netuid,
        network=network,
        manifest_sha256=manifest_sha256,
        assume_yes=assume_yes,
    )
    logger.info("submitted registration {}", rid)
    return rid
