"""Miner-side private submission: manifest signing, upload, mailbox fetch.

Everything here is exercised without bittensor (chain writes are the only part
that needs it); the crypto/upload path a validator later verifies is fully
covered by round-tripping through the real private_store verifier.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest
from nacl.signing import SigningKey
from test_private_store import BUCKET, GENESIS_CONFIG, FakeS3
from test_s3_model_backend import RID  # canonical registration id "a"*64

from miner import private_store as mp
from private_store.contracts import Manifest, mailbox_object_key, model_prefix
from private_store.crypto import MailboxCipher, encode_ss58_public_key
from private_store.storage import R2UploadController

MINER_KEY = SigningKey(b"m" * 32)
HOTKEY = encode_ss58_public_key(bytes(MINER_KEY.verify_key))
SUBMISSION_KEY = SigningKey(b"s" * 32)
SUBMISSION_PUBKEY = bytes(SUBMISSION_KEY.verify_key)


def _model_dir(tmp_path):
    files = {
        "config.json": GENESIS_CONFIG,
        "model-00001-of-00002.safetensors": b"T" * 128,
        "model-00002-of-00002.safetensors": b"U" * 128,
        "model.safetensors.index.json": b'{"weight_map": {}}',
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    return files


def test_plan_upload_inventories_files_and_skips_manifest(tmp_path):
    files = _model_dir(tmp_path)
    (tmp_path / "manifest.json").write_bytes(b"stale")
    plan = mp.plan_upload(str(tmp_path))
    assert {rel for rel, _s, _d, _p in plan} == set(files)  # manifest.json excluded
    for rel, size, sha, _path in plan:
        assert size == len(files[rel])
        assert sha == hashlib.sha256(files[rel]).hexdigest()


def test_plan_upload_skips_hidden_and_junk(tmp_path):
    files = _model_dir(tmp_path)
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / ".gitignore").write_bytes(b"*")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_bytes(b"[core]")
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"junk")
    plan = mp.plan_upload(str(tmp_path))
    rels = {rel for rel, _s, _d, _p in plan}
    assert rels == set(files)  # only real model files; no .cache/.git/.DS_Store/__pycache__
    assert not any(r.startswith(".") or "__pycache__" in r for r in rels)


def test_plan_upload_rejects_empty_or_missing_dirs(tmp_path):
    with pytest.raises(SystemExit, match="not a directory"):
        mp.plan_upload(str(tmp_path / "nope"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no files"):
        mp.plan_upload(str(tmp_path / "empty"))


def test_built_manifest_verifies_against_the_validator(tmp_path):
    _model_dir(tmp_path)
    plan = mp.plan_upload(str(tmp_path))
    manifest = mp.build_manifest(
        MINER_KEY, registration_id=RID, hotkey=HOTKEY, model_name="albedo-test", plan=plan
    )
    # round-trips through the validator's parser and signature check
    parsed = Manifest.from_bytes(manifest.as_bytes())
    assert parsed == manifest
    from private_store.crypto import verify_hotkey_signature

    verify_hotkey_signature(HOTKEY, parsed.signing_payload(), parsed.signature)


def _envelope(prefix: str) -> dict:
    return {
        "r2_endpoint": "https://account.r2.cloudflarestorage.com",
        "private_model_bucket": BUCKET,
        "allowed_prefix": prefix,
        "access_key_id": "tok",
        "secret_access_key": "sec",
        "session_token": "sess",
    }


def test_upload_objects_then_verify_manifest_end_to_end(tmp_path):
    _model_dir(tmp_path)
    prefix = model_prefix(RID)
    plan = mp.plan_upload(str(tmp_path))
    manifest = mp.build_manifest(
        SUBMISSION_KEY, registration_id=RID, hotkey=HOTKEY, model_name="albedo-test", plan=plan
    )
    s3 = FakeS3()
    s3.upload_file = lambda path, bucket, key, ExtraArgs, Config: s3.put(
        bucket, key, open(path, "rb").read(), sha256=ExtraArgs["Metadata"]["sha256"]
    )
    manifest_sha256 = mp.upload_objects(s3, _envelope(prefix), manifest, plan)
    assert manifest_sha256 == manifest.manifest_sha256

    # the validator accepts exactly what the miner produced
    controller = R2UploadController(
        s3,
        private_model_bucket=BUCKET,
        genesis_contract_files={"config.json": hashlib.sha256(GENESIS_CONFIG).hexdigest()},
    )
    verified = controller.verify_manifest(
        model_prefix=prefix,
        registration_id=RID,
        hotkey=HOTKEY,
        submission_pubkey=SUBMISSION_PUBKEY,
        expected_manifest_sha256=manifest_sha256,
    )
    assert verified.manifest.model_digest == manifest.model_digest


def test_fetch_credentials_polls_decrypts_and_persists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mp, "MAILBOX_BASE_URL", "https://pub.example.r2.dev")
    prefix = model_prefix(RID)
    cipher = MailboxCipher(SigningKey(b"v" * 32))
    ciphertext, _ = cipher.create_ciphertext(
        submission_pubkey=SUBMISSION_PUBKEY,
        hotkey=HOTKEY,
        netuid=97,
        registration_id=RID,
        generation=1,
        endpoint="https://account.r2.cloudflarestorage.com",
        private_model_bucket=BUCKET,
        allowed_prefix=prefix,
        access_key_id="tok",
        secret_access_key="sec",
        session_token="sess",
        expires_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        chain_generation="albedo-mainnet-1",
    )
    expected_url = f"https://pub.example.r2.dev/{mailbox_object_key(RID, 1)}"

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return ciphertext

    def _fake_urlopen(request, timeout):
        assert request.full_url.startswith(expected_url)
        assert request.headers["User-agent"] == "albedo-miner/1.0"
        return _Resp()

    monkeypatch.setattr(mp.urllib.request, "urlopen", _fake_urlopen)
    envelope = mp.fetch_credentials(
        hotkey_ss58=HOTKEY, registration_id=RID, signing_key=SUBMISSION_KEY
    )
    assert envelope["allowed_prefix"] == prefix
    saved = tmp_path / ".albedo-miner" / HOTKEY / "upload-auth.json"
    assert json.loads(saved.read_text())["access_key_id"] == "tok"
    assert oct(saved.stat().st_mode)[-3:] == "600"


def test_fetch_credentials_accepts_a_retry_envelope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mp, "MAILBOX_BASE_URL", "https://pub.example.r2.dev")
    retry_prefix = model_prefix(RID, 2)
    cipher = MailboxCipher(SigningKey(b"v" * 32))
    ciphertext, _ = cipher.create_ciphertext(
        submission_pubkey=SUBMISSION_PUBKEY,
        hotkey=HOTKEY,
        netuid=97,
        registration_id=RID,
        generation=1,
        endpoint="https://account.r2.cloudflarestorage.com",
        private_model_bucket=BUCKET,
        allowed_prefix=retry_prefix,
        access_key_id="tok",
        secret_access_key="sec",
        session_token="sess",
        expires_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        chain_generation="albedo-mainnet-1",
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return ciphertext

    monkeypatch.setattr(mp.urllib.request, "urlopen", lambda request, timeout: _Resp())
    envelope = mp.fetch_credentials(
        hotkey_ss58=HOTKEY, registration_id=RID, signing_key=SUBMISSION_KEY
    )
    assert envelope["allowed_prefix"] == f"models/attempts/{RID}/a2/"


def test_fetch_credentials_rejects_a_mismatched_envelope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mp, "MAILBOX_BASE_URL", "https://pub.example.r2.dev")
    other = "b" * 64
    cipher = MailboxCipher(SigningKey(b"v" * 32))
    ciphertext, _ = cipher.create_ciphertext(
        submission_pubkey=SUBMISSION_PUBKEY,
        hotkey=HOTKEY,
        netuid=97,
        registration_id=other,  # envelope for a different registration
        generation=1,
        endpoint="https://account.r2.cloudflarestorage.com",
        private_model_bucket=BUCKET,
        allowed_prefix=model_prefix(other),
        access_key_id="tok",
        secret_access_key="sec",
        session_token="sess",
        expires_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        chain_generation="albedo-mainnet-1",
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return ciphertext

    monkeypatch.setattr(mp.urllib.request, "urlopen", lambda request, timeout: _Resp())
    with pytest.raises(SystemExit, match="does not match this registration"):
        mp.fetch_credentials(hotkey_ss58=HOTKEY, registration_id=RID, signing_key=SUBMISSION_KEY)
