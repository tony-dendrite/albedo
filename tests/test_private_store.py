"""Unit tests for the private R2 submission core (ported from teutonic)."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from nacl.signing import SigningKey

from private_store.cloudflare import CloudflareR2TokenGateway
from private_store.contracts import (
    Manifest,
    ManifestFile,
    activation_signal_payload,
    mailbox_object_key,
    model_prefix,
    parse_activation_pubkey,
    parse_model_prefix,
    parse_ready_signal,
    ready_signal_payload,
    registration_id,
)
from private_store.crypto import (
    MailboxCipher,
    encode_signature,
    encode_ss58_public_key,
    verify_ed25519,
    verify_hotkey_signature,
)
from private_store.digests import (
    ArtifactIntegrityError,
    model_digest_from_inventory,
    snapshot_digest,
    verify_snapshot,
)
from private_store.r2_credentials import (
    MINER_PREFIX_SCOPE,
    create_local_temporary_credentials,
)
from private_store.storage import (
    GenesisContractMismatch,
    MailboxInvariantError,
    MailboxStore,
    R2UploadController,
    UploadQuotaExceeded,
)

CHAIN_GENERATION = "albedo-mainnet-1"


def _miner() -> tuple[SigningKey, str]:
    key = SigningKey(b"m" * 32)
    return key, encode_ss58_public_key(bytes(key.verify_key))


# the one-time submission key the miner commits at activate and signs with
SUBMISSION_KEY = SigningKey(b"s" * 32)
SUBMISSION_PUBKEY = bytes(SUBMISSION_KEY.verify_key)


def _registration(hotkey: str) -> str:
    return registration_id(netuid=97, hotkey=hotkey, chain_generation=CHAIN_GENERATION)


# --- contracts ---------------------------------------------------------------


def test_registration_id_is_stable_and_generation_specific():
    _, hotkey = _miner()
    one = _registration(hotkey)
    assert one == _registration(hotkey)
    other = registration_id(netuid=97, hotkey=hotkey, chain_generation="albedo-mainnet-2")
    assert one != other
    assert len(one) == 64


def test_activation_carries_the_submission_pubkey():
    payload = activation_signal_payload(SUBMISSION_PUBKEY)
    assert payload.startswith("r2activate:v1:")
    assert len(payload.encode()) <= 128  # fits the commitment budget
    assert parse_activation_pubkey(payload) == SUBMISSION_PUBKEY
    with pytest.raises(ValueError):
        activation_signal_payload(b"tooshort")
    with pytest.raises(ValueError):
        parse_activation_pubkey("r2activate:v1:short")


def test_manifest_verifies_against_the_committed_submission_key():
    _, hotkey = _miner()
    registration = _registration(hotkey)
    manifest = build_manifest(SUBMISSION_KEY, hotkey, registration, model_files())
    verify_ed25519(SUBMISSION_PUBKEY, manifest.signing_payload(), manifest.signature)
    wrong = bytes(SigningKey(b"x" * 32).verify_key)
    with pytest.raises(ValueError):
        verify_ed25519(wrong, manifest.signing_payload(), manifest.signature)


def test_ready_signal_roundtrip_and_rejection():
    manifest = "c" * 64
    payload = ready_signal_payload(manifest)
    assert len(payload.encode()) <= 128
    assert parse_ready_signal(payload) == manifest
    with pytest.raises(ValueError):
        parse_ready_signal("r2ready:v1:UPPER" + "c" * 59)
    with pytest.raises(ValueError):
        parse_ready_signal("v7|repo/name|" + "d" * 64)


def test_mailbox_generations_are_immutable_lexically_ordered_keys():
    registration = "b" * 64
    first = mailbox_object_key(registration, 1)
    second = mailbox_object_key(registration, 2)
    assert first < second
    assert first.endswith("/00000000000000000001.bin")
    with pytest.raises(ValueError):
        mailbox_object_key(registration, 0)


def test_model_prefix_requires_canonical_registration():
    assert model_prefix("f" * 64) == f"models/registrations/{'f' * 64}/"
    with pytest.raises(ValueError):
        model_prefix("not-a-digest")


def test_retry_attempts_get_their_own_prefix_outside_the_first():
    rid = "f" * 64
    first, retry = model_prefix(rid), model_prefix(rid, 2)
    assert first == f"models/registrations/{rid}/"
    assert retry == f"models/attempts/{rid}/a2/"
    assert not retry.startswith(first)
    assert first.split("/")[2] == rid and retry.split("/")[2] == rid
    with pytest.raises(ValueError):
        model_prefix(rid, 0)


def test_model_prefix_round_trips_through_parse():
    rid = "f" * 64
    for attempt in (1, 2, 3, 9, 10, 47):
        assert parse_model_prefix(model_prefix(rid, attempt)) == (rid, attempt)
    for bad in (
        "",
        "models/",
        "models/registrations/",
        f"models/registrations/{rid}",
        f"models/registrations/{rid}/a2/",  # retries never nest under the first attempt
        f"models/attempts/{rid}/a1/",  # attempt 1 has exactly one spelling
        f"models/attempts/{rid}/a0/",
        f"models/attempts/{rid}/",
        f"models/attempts/{'z' * 64}/a2/",
    ):
        with pytest.raises(ValueError):
            parse_model_prefix(bad)


# --- temporary credentials ---------------------------------------------------


def test_r2_temporary_credentials_are_jailed_to_the_registration_prefix():
    prefix = f"models/registrations/{'c' * 64}/"
    credentials = create_local_temporary_credentials(
        endpoint="https://account.r2.cloudflarestorage.com",
        account_id="a" * 32,
        parent_access_key_id="parent-id",
        parent_secret_access_key="parent-secret",
        bucket="albedo-private-models-enam",
        prefix=prefix,
        ttl_seconds=900,
        issued_at_unix=1_700_000_000,
    )
    token = base64.b64decode(credentials.session_token).decode()
    assert token.startswith("jwt/")
    claims_segment = token.removeprefix("jwt/").split(".")[1]
    claims_segment += "=" * (-len(claims_segment) % 4)
    claims = json.loads(base64.urlsafe_b64decode(claims_segment))
    assert claims["exp"] - claims["iat"] == 900
    assert claims["paths"]["prefixPaths"] == [prefix]
    assert claims["paths"]["objectPaths"] == []
    assert "actions" not in claims
    assert claims["scope"] == MINER_PREFIX_SCOPE


def test_r2_temporary_credentials_reject_unsafe_inputs():
    common = dict(
        endpoint="https://account.r2.cloudflarestorage.com",
        account_id="a" * 32,
        parent_access_key_id="parent-id",
        parent_secret_access_key="parent-secret",
        bucket="albedo-private-models-enam",
    )
    with pytest.raises(ValueError):
        create_local_temporary_credentials(**common, prefix="missing-trailing-slash")
    with pytest.raises(ValueError):
        create_local_temporary_credentials(**common, prefix="p/", ttl_seconds=0)
    with pytest.raises(ValueError):
        create_local_temporary_credentials(
            **{**common, "endpoint": "http://account.r2.cloudflarestorage.com"}, prefix="p/"
        )


# --- crypto ------------------------------------------------------------------


def _envelope_fields(registration: str) -> dict:
    return dict(
        netuid=97,
        registration_id=registration,
        generation=1,
        endpoint="https://account.r2.cloudflarestorage.com",
        private_model_bucket="albedo-private-models-enam",
        allowed_prefix=f"models/registrations/{registration}/",
        access_key_id="child-token-id",
        secret_access_key="child-secret",
        session_token="c2Vzc2lvbg==",
        expires_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        chain_generation=CHAIN_GENERATION,
    )


def test_mailbox_envelope_roundtrips_only_for_the_submission_key():
    _, hotkey = _miner()
    registration = _registration(hotkey)
    cipher = MailboxCipher(SigningKey(b"v" * 32))
    ciphertext, envelope = cipher.create_ciphertext(
        submission_pubkey=SUBMISSION_PUBKEY, hotkey=hotkey, **_envelope_fields(registration)
    )
    decrypted = MailboxCipher.decrypt_for_miner(ciphertext, SUBMISSION_KEY)
    assert decrypted["allowed_prefix"] == f"models/registrations/{registration}/"
    assert decrypted["validator_identity"] == cipher.validator_identity
    assert decrypted["secret_access_key"] == "child-secret"
    assert envelope["validator_signature"] == decrypted["validator_signature"]
    with pytest.raises(Exception):
        MailboxCipher.decrypt_for_miner(ciphertext, SigningKey(b"e" * 32))
    with pytest.raises(Exception):
        MailboxCipher.decrypt_for_miner(
            ciphertext[:-1] + bytes([ciphertext[-1] ^ 1]), SUBMISSION_KEY
        )


def test_mailbox_envelope_signature_is_verified_on_decrypt():
    _, hotkey = _miner()
    registration = _registration(hotkey)
    cipher = MailboxCipher(SigningKey(b"v" * 32))
    envelope = cipher.build_envelope(hotkey=hotkey, **_envelope_fields(registration))
    envelope["secret_access_key"] = "tampered"
    forged = cipher.encrypt_for_pubkey(envelope, SUBMISSION_PUBKEY)
    with pytest.raises(ValueError):
        MailboxCipher.decrypt_for_miner(forged, SUBMISSION_KEY)


# --- manifest ----------------------------------------------------------------


GENESIS_CONFIG = b'{"architectures": ["Qwen3"]}'


def build_manifest(
    signing_key: SigningKey,
    hotkey: str,
    registration: str,
    files: dict[str, bytes],
) -> Manifest:
    inventory = tuple(
        ManifestFile(path=path, size=len(data), sha256=hashlib.sha256(data).hexdigest())
        for path, data in sorted(files.items())
    )
    digest = model_digest_from_inventory([(f.path, f.size, f.sha256) for f in inventory])
    unsigned = Manifest(
        registration_id=registration,
        hotkey=hotkey,
        model_name="albedo-qwen3.6-35b-test",
        files=inventory,
        model_digest=digest,
        signature="pending",
    )
    signature = encode_signature(signing_key.sign(unsigned.signing_payload()).signature)
    return dataclasses.replace(unsigned, signature=signature)


def model_files() -> dict[str, bytes]:
    return {
        "config.json": GENESIS_CONFIG,
        "model-00001-of-00002.safetensors": b"tensor-bytes-1",
        "model-00002-of-00002.safetensors": b"tensor-bytes-2",
        "model.safetensors.index.json": b'{"weight_map": {}}',
    }


def test_manifest_roundtrip_signature_and_hash_are_stable():
    key, hotkey = _miner()
    registration = _registration(hotkey)
    manifest = build_manifest(key, hotkey, registration, model_files())
    parsed = Manifest.from_bytes(manifest.as_bytes())
    assert parsed == manifest
    assert parsed.manifest_sha256 == manifest.manifest_sha256
    verify_hotkey_signature(hotkey, parsed.signing_payload(), parsed.signature)


def test_manifest_rejects_tampered_inventory_and_unknown_fields():
    key, hotkey = _miner()
    registration = _registration(hotkey)
    manifest = build_manifest(key, hotkey, registration, model_files())
    tampered = json.loads(manifest.as_bytes())
    tampered["files"][1]["size"] += 1
    with pytest.raises(ValueError, match="model_digest does not match"):
        Manifest.from_bytes(json.dumps(tampered).encode())
    extra = {**json.loads(manifest.as_bytes()), "note": "hi"}
    with pytest.raises(ValueError, match="differ from v1 contract"):
        Manifest.from_bytes(json.dumps(extra).encode())
    with pytest.raises(ValueError, match="unsafe manifest path"):
        ManifestFile.from_mapping({"path": "../escape", "size": 1, "sha256": "0" * 64})


# --- upload controller against a fake S3 -------------------------------------


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read(self, size: int | None = None) -> bytes:
        if size is None:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        pass


class FakeS3:
    _EPOCH = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.multipart: list[dict[str, str]] = []
        self.deleted: list[str] = []
        self.modified: dict[tuple[str, str], datetime] = {}

    def _stamp(self, bucket: str, key: str) -> datetime:
        return self.modified.get((bucket, key), self._EPOCH)

    def put(self, bucket: str, key: str, data: bytes, *, sha256: str | None = None) -> None:
        metadata = {"sha256": sha256 or hashlib.sha256(data).hexdigest()}
        self.objects[(bucket, key)] = (data, metadata)

    def put_object(self, Bucket: str, Key: str, Body: bytes, Metadata: dict, **_: object):
        self.objects[(Bucket, Key)] = (Body, dict(Metadata))
        return {}

    def get_object(self, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
                "GetObject",
            )
        data, metadata = self.objects[(Bucket, Key)]
        return {
            "Body": FakeBody(data),
            "Metadata": metadata,
            "ETag": f'"{hashlib.md5(data).hexdigest()}"',
        }

    def head_object(self, Bucket: str, Key: str):
        data, metadata = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(data),
            "Metadata": metadata,
            "ETag": f'"{hashlib.md5(data).hexdigest()}"',
        }

    def list_objects_v2(self, Bucket: str, Prefix: str, **_: object):
        contents = [
            {
                "Key": key,
                "Size": len(data),
                "ETag": f'"{hashlib.md5(data).hexdigest()}"',
                "LastModified": self._stamp(bucket, key),
            }
            for (bucket, key), (data, _meta) in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def list_multipart_uploads(self, Bucket: str, Prefix: str, **_: object):
        uploads = [item for item in self.multipart if item["Key"].startswith(Prefix)]
        return {"Uploads": uploads, "IsTruncated": False}

    def abort_multipart_upload(self, Bucket: str, Key: str, UploadId: str):
        self.multipart = [item for item in self.multipart if item["UploadId"] != UploadId]

    def delete_objects(self, Bucket: str, Delete: dict):
        for entry in Delete["Objects"]:
            self.objects.pop((Bucket, entry["Key"]), None)
            self.deleted.append(entry["Key"])
        return {}


BUCKET = "albedo-private-models-enam"


def seeded_controller(**overrides):
    key, hotkey = _miner()
    registration = _registration(hotkey)
    prefix = model_prefix(registration)
    files = model_files()
    manifest = build_manifest(SUBMISSION_KEY, hotkey, registration, files)
    s3 = FakeS3()
    for path, data in files.items():
        s3.put(BUCKET, f"{prefix}{path}", data)
    s3.put(BUCKET, f"{prefix}manifest.json", manifest.as_bytes())
    controller = R2UploadController(
        s3,
        private_model_bucket=BUCKET,
        genesis_contract_files={"config.json": hashlib.sha256(GENESIS_CONFIG).hexdigest()},
        **overrides,
    )
    return s3, controller, manifest, registration, hotkey, prefix


def test_verify_manifest_accepts_a_faithful_upload():
    _, controller, manifest, registration, hotkey, prefix = seeded_controller()
    verified = controller.verify_manifest(
        model_prefix=prefix,
        registration_id=registration,
        hotkey=hotkey,
        submission_pubkey=SUBMISSION_PUBKEY,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    assert verified.manifest.model_digest == manifest.model_digest
    assert verified.manifest_sha256 == manifest.manifest_sha256


def test_verify_manifest_rejects_wrong_chain_pin_and_foreign_identity():
    _, controller, manifest, registration, hotkey, prefix = seeded_controller()
    with pytest.raises(ArtifactIntegrityError, match="does not match the finalized ready"):
        controller.verify_manifest(
            model_prefix=prefix,
            registration_id=registration,
            hotkey=hotkey,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(ArtifactIntegrityError, match="not canonical"):
        controller.verify_manifest(
            model_prefix="models/registrations/wrong/",
            registration_id=registration,
            hotkey=hotkey,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=manifest.manifest_sha256,
        )
    with pytest.raises(ArtifactIntegrityError, match="does not own the model prefix"):
        controller.verify_manifest(
            model_prefix=prefix,
            registration_id=registration,
            hotkey="5" + "B" * 47,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_verify_manifest_rejects_undeclared_or_missing_objects():
    s3, controller, manifest, registration, hotkey, prefix = seeded_controller()
    s3.put(BUCKET, f"{prefix}extra.bin", b"sneaky")
    with pytest.raises(ArtifactIntegrityError, match="undeclared"):
        controller.verify_manifest(
            model_prefix=prefix,
            registration_id=registration,
            hotkey=hotkey,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=manifest.manifest_sha256,
        )
    s3.objects.pop((BUCKET, f"{prefix}extra.bin"))
    s3.objects.pop((BUCKET, f"{prefix}model-00002-of-00002.safetensors"))
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        controller.verify_manifest(
            model_prefix=prefix,
            registration_id=registration,
            hotkey=hotkey,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_verify_manifest_rejects_bad_object_metadata_and_genesis_drift():
    s3, controller, manifest, registration, hotkey, prefix = seeded_controller()
    shard = f"{prefix}model-00001-of-00002.safetensors"
    data, _ = s3.objects[(BUCKET, shard)]
    s3.put(BUCKET, shard, data, sha256="f" * 64)
    with pytest.raises(ArtifactIntegrityError, match="metadata is missing or incorrect"):
        controller.verify_manifest(
            model_prefix=prefix,
            registration_id=registration,
            hotkey=hotkey,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=manifest.manifest_sha256,
        )

    _, hotkey2 = _miner()
    files = model_files()
    files["config.json"] = b'{"architectures": ["NotQwen"]}'
    drifted = build_manifest(SUBMISSION_KEY, hotkey2, _registration(hotkey2), files)
    s3b = FakeS3()
    prefix2 = model_prefix(drifted.registration_id)
    for path, blob in files.items():
        s3b.put(BUCKET, f"{prefix2}{path}", blob)
    s3b.put(BUCKET, f"{prefix2}manifest.json", drifted.as_bytes())
    controller2 = R2UploadController(
        s3b,
        private_model_bucket=BUCKET,
        genesis_contract_files={"config.json": hashlib.sha256(GENESIS_CONFIG).hexdigest()},
    )
    with pytest.raises(GenesisContractMismatch):
        controller2.verify_manifest(
            model_prefix=prefix2,
            registration_id=drifted.registration_id,
            hotkey=hotkey2,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=drifted.manifest_sha256,
        )


def test_verify_manifest_enforces_quota_and_aborts_leftover_multipart():
    s3, controller, manifest, registration, hotkey, prefix = seeded_controller(max_upload_bytes=10)
    with pytest.raises(UploadQuotaExceeded):
        controller.verify_manifest(
            model_prefix=prefix,
            registration_id=registration,
            hotkey=hotkey,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=manifest.manifest_sha256,
        )
    # a leftover incomplete multipart (killed upload) must NOT fail an otherwise
    # complete submission: it is invisible to the object inventory either way.
    s3b, controller_b, manifest_b, registration_b, hotkey_b, prefix_b = seeded_controller()
    s3b.multipart.append({"Key": f"{prefix_b}model-inflight", "UploadId": "u1"})
    verified = controller_b.verify_manifest(
        model_prefix=prefix_b,
        registration_id=registration_b,
        hotkey=hotkey_b,
        submission_pubkey=SUBMISSION_PUBKEY,
        expected_manifest_sha256=manifest_b.manifest_sha256,
    )
    assert verified.manifest.registration_id == registration_b
    assert s3b.multipart == [{"Key": f"{prefix_b}model-inflight", "UploadId": "u1"}]


def test_cleanup_model_prefix_removes_everything_once():
    _, controller, _manifest, _registration_id, _hotkey, prefix = seeded_controller()
    assert controller.cleanup_model_prefix(prefix) == 5
    assert controller.cleanup_model_prefix(prefix) == 0


def test_cleanup_model_prefix_also_aborts_multipart_uploads():
    # a verify-FAILED prefix can hold an unfinished multipart with no completed objects;
    # cleanup must abort it so the miner cannot leak R2 storage.
    s3, controller, _manifest, _registration_id, _hotkey, prefix = seeded_controller()
    s3.multipart.append({"Key": f"{prefix}model-inflight", "UploadId": "u1"})
    assert controller.cleanup_model_prefix(prefix) == 6  # 5 objects + 1 aborted multipart
    assert s3.multipart == []
    assert controller.cleanup_model_prefix(prefix) == 0


def test_cleanup_refuses_non_registration_prefixes():
    # the one mass-delete must never fire on "" (whole bucket) or a stray prefix
    _, controller, _m, _r, _h, prefix = seeded_controller()
    bad_prefixes = [
        "",
        "models/",
        "models/registrations/",
        f"models/registrations/{'z' * 64}/",
        prefix[:-1],
        f"models/attempts/{'a' * 64}/a1/",  # not canonical: attempt 1 is the bare prefix
        f"models/attempts/{'a' * 64}/",
    ]
    for bad in bad_prefixes:
        with pytest.raises(ValueError, match="refusing to bulk-delete"):
            controller.cleanup_model_prefix(bad)


def test_cleanup_accepts_a_retry_prefix():
    s3, controller, _m, registration_id, _h, _prefix = seeded_controller()
    retry = model_prefix(registration_id, 2)
    s3.put(BUCKET, f"{retry}model.safetensors", b"retry-bytes")
    assert controller.cleanup_model_prefix(retry) == 1


def test_retained_uploads_inventories_every_attempt():
    s3, controller, _m, registration_id, _h, prefix = seeded_controller()
    retry = model_prefix(registration_id, 2)
    s3.put(BUCKET, f"{retry}model.safetensors", b"retry-bytes")
    s3.put(BUCKET, "models/registrations/not-a-registration", b"stray")
    held = controller.retained_uploads()
    assert set(held) == {prefix, retry}
    assert all(value is not None for value in held.values())
    # the seeded objects are untouched by the rejected deletes
    assert controller.cleanup_model_prefix(prefix) == 5


# --- adversarial: upload-window abuse ----------------------------------------


def test_verify_manifest_rejects_too_many_objects():
    from private_store.storage import MAX_MINER_UPLOAD_OBJECTS

    s3, controller, manifest, registration, hotkey, prefix = seeded_controller()
    for i in range(MAX_MINER_UPLOAD_OBJECTS + 1):
        s3.put(BUCKET, f"{prefix}junk-{i}.bin", b"x")
    with pytest.raises(UploadQuotaExceeded, match="objects"):
        controller.verify_manifest(
            model_prefix=prefix,
            registration_id=registration,
            hotkey=hotkey,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_verify_manifest_rejects_a_giant_manifest():
    from private_store.storage import MAX_MANIFEST_BYTES

    s3, controller, manifest, registration, hotkey, prefix = seeded_controller()
    s3.put(BUCKET, f"{prefix}manifest.json", b"x" * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ArtifactIntegrityError, match="implausibly large"):
        controller.verify_manifest(
            model_prefix=prefix,
            registration_id=registration,
            hotkey=hotkey,
            submission_pubkey=SUBMISSION_PUBKEY,
            expected_manifest_sha256=manifest.manifest_sha256,
        )


def test_quota_breach_early_exits_on_bytes_and_objects():
    s3 = FakeS3()
    prefix = "models/registrations/" + "a" * 64 + "/"
    controller = R2UploadController(
        s3,
        private_model_bucket=BUCKET,
        genesis_contract_files={"config.json": hashlib.sha256(GENESIS_CONFIG).hexdigest()},
        max_upload_bytes=1000,
    )
    assert controller.quota_breach(prefix) is None  # empty prefix is fine
    s3.put(BUCKET, f"{prefix}a", b"x" * 400)
    s3.put(BUCKET, f"{prefix}b", b"x" * 400)
    assert controller.quota_breach(prefix) is None  # 800 < 1000
    s3.put(BUCKET, f"{prefix}c", b"x" * 400)
    assert "bytes" in controller.quota_breach(prefix)  # 1200 > 1000


def test_quota_breach_counts_objects_even_when_small():
    from private_store.storage import MAX_MINER_UPLOAD_OBJECTS

    s3 = FakeS3()
    prefix = "models/registrations/" + "a" * 64 + "/"
    controller = R2UploadController(
        s3,
        private_model_bucket=BUCKET,
        genesis_contract_files={"config.json": hashlib.sha256(GENESIS_CONFIG).hexdigest()},
    )
    for i in range(MAX_MINER_UPLOAD_OBJECTS + 5):
        s3.put(BUCKET, f"{prefix}tiny-{i}", b"x")
    assert "objects" in controller.quota_breach(prefix)


# --- mailbox store -----------------------------------------------------------


def test_mailbox_store_publish_is_idempotent_but_never_rewrites():
    s3 = FakeS3()
    store = MailboxStore(s3, bucket="albedo-mailbox-enam")
    key = mailbox_object_key("a" * 64, 1)
    store.publish(key, b"ciphertext")
    store.publish(key, b"ciphertext")
    with pytest.raises(MailboxInvariantError):
        store.publish(key, b"different")
    assert store.delete([key]) == 1
    with pytest.raises(MailboxInvariantError):
        store.delete(["models/registrations/whoops"])


# --- digests -----------------------------------------------------------------


def test_snapshot_digest_matches_inventory_and_detects_drift(tmp_path: Path):
    files = model_files()
    for path, data in files.items():
        (tmp_path / path).write_bytes(data)
    (tmp_path / "manifest.json").write_bytes(b"excluded")
    inventory = [
        (path, len(data), hashlib.sha256(data).hexdigest()) for path, data in files.items()
    ]
    expected = model_digest_from_inventory(inventory)
    assert snapshot_digest(tmp_path) == expected
    assert verify_snapshot(tmp_path, expected) == expected
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"flipped")
    with pytest.raises(ArtifactIntegrityError, match="digest mismatch"):
        verify_snapshot(tmp_path, expected)


def test_snapshot_digest_rejects_symlinks(tmp_path: Path):
    (tmp_path / "a.safetensors").write_bytes(b"data")
    (tmp_path / "link").symlink_to(tmp_path / "a.safetensors")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        snapshot_digest(tmp_path)


# --- cloudflare gateway ------------------------------------------------------


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class FakeCloudflare:
    def __init__(self) -> None:
        self.tokens: dict[str, str] = {"stale-id": "albedo-registration-x"}
        self.revoked: list[str] = []
        self._counter = 0

    def request(self, method: str, url: str, headers: dict, **kwargs):
        assert headers["Authorization"].startswith("Bearer ")
        if url.endswith("/tokens/permission_groups"):
            result = [{"id": "perm-1", "name": "Workers R2 Storage Bucket Item Write"}]
        elif method == "GET" and url.endswith("/tokens"):
            result = [{"id": token_id, "name": name} for token_id, name in self.tokens.items()]
        elif method == "POST" and url.endswith("/tokens"):
            self._counter += 1
            token_id = f"token-{self._counter}"
            self.tokens[token_id] = kwargs["json"]["name"]
            resource = next(iter(kwargs["json"]["policies"][0]["resources"]))
            assert resource == f"com.cloudflare.edge.r2.bucket.{'a' * 32}_default_{BUCKET}"
            result = {"id": token_id, "value": f"secret-{self._counter}"}
        elif method == "DELETE":
            token_id = url.rsplit("/", 1)[1]
            self.tokens.pop(token_id, None)
            self.revoked.append(token_id)
            result = {"id": token_id}
        else:
            raise AssertionError(f"unexpected request {method} {url}")
        return FakeHttpResponse({"success": True, "result": result, "errors": []})


def test_token_gateway_recreates_orphans_and_derives_s3_secret():
    http = FakeCloudflare()
    gateway = CloudflareR2TokenGateway(
        http,
        account_id="a" * 32,
        management_token="cf-token",
        bucket=BUCKET,
    )
    parent = gateway.create_parent_token("albedo-registration-x")
    assert http.revoked == ["stale-id"]
    assert parent.access_key_id == parent.token_id
    assert parent.secret_access_key == hashlib.sha256(b"secret-1").hexdigest()
    gateway.revoke_parent_token(parent.token_id)
    assert parent.token_id in http.revoked
