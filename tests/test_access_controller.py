"""Intake and access-controller lifecycle tests (fake chain, S3, Cloudflare, DB)."""

from __future__ import annotations

import asyncio
import hashlib
import uuid

import pytest
from nacl.signing import SigningKey
from test_private_store import (
    BUCKET,
    GENESIS_CONFIG,
    FakeS3,
    build_manifest,
    model_files,
)

from chain_reader.chain import PrivateSignal
from private_store import controller, intake
from private_store.cloudflare import ParentToken
from private_store.contracts import (
    activation_signal_payload,
    mailbox_object_key,
    model_prefix,
    ready_signal_payload,
    registration_id,
)
from private_store.crypto import MailboxCipher, encode_ss58_public_key
from private_store.settings import PrivateStoreSettings
from private_store.storage import MailboxStore, R2UploadController

SETTINGS = PrivateStoreSettings(
    _env_file=None,
    account_id="a" * 32,
    api_token="cf-token",
    access_key_id="master-key",
    secret_access_key="master-secret",
    endpoint="https://account.r2.cloudflarestorage.com",
    private_models_bucket_name=BUCKET,
    mailbox_bucket_name="albedo-mailbox-enam",
    mailbox_signing_key="ab" * 32,
)

MINER_KEY = SigningKey(b"m" * 32)
HOTKEY = encode_ss58_public_key(bytes(MINER_KEY.verify_key))
RID = registration_id(netuid=97, hotkey=HOTKEY, chain_generation=SETTINGS.chain_generation)
PREFIX = model_prefix(RID)
SUBMISSION_KEY = SigningKey(b"s" * 32)
SUBMISSION_PUBKEY = bytes(SUBMISSION_KEY.verify_key)


class _Ctx:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, registration: dict | None = None, used: bool = False, age_s: float = 0.0):
        self.registration = registration
        self.used = used
        self.age_s = age_s
        self.submission_id = None
        self.reapable: list[dict] = []  # loser rows the DB's ORDER BY/OFFSET would return
        self.last_offset = None

    def transaction(self):
        return _Ctx(self)

    async def fetchval(self, sql, *args):
        if "used_hotkeys" in sql:
            return 1 if self.used else None
        if "INSERT INTO private_registrations" in sql:
            if self.registration is not None:
                return None
            self.registration = {
                "id": 1,
                "netuid": args[0],
                "uid": args[1],
                "hotkey": args[2],
                "registration_id": args[3],
                "activation_block": args[4],
                "submission_pubkey": args[5],
                "state": "ACTIVATED",
                "ready_block": None,
                "manifest_sha256": None,
                "model_digest": None,
                "parent_token_id": None,
                "submission_id": None,
                "fault_message": None,
            }
            return 1
        if "SET state = 'READY'" in sql:
            row = self.registration
            if row and row["state"] == "CREDENTIALED" and row["registration_id"] == args[0]:
                row.update(state="READY", manifest_sha256=args[1], ready_block=args[2])
                return row["id"]
            return None
        if "SELECT submission_id FROM chain_commits" in sql:
            return self.submission_id
        raise AssertionError(f"unexpected fetchval: {sql}")

    async def fetchrow(self, sql, *args):
        assert "FROM private_registrations" in sql
        row = self.registration
        if row and row["state"] in ("ACTIVATED", "READY", "REVOKED"):
            return dict(row)
        return None

    async def fetch(self, sql, *args):
        if "COMPLETE_LOSS" in sql:
            assert "OFFSET" in sql and "DESC" in sql  # keep N most recent
            self.last_offset = args[0]
            return self.reapable
        assert "state = 'CREDENTIALED'" in sql
        row = self.registration
        if not (row and row["state"] == "CREDENTIALED"):
            return []
        return [{**row, "age_s": self.age_s}]

    async def execute(self, sql, *args):
        if "SET state = 'CREDENTIALED'" in sql:
            self.registration.update(state="CREDENTIALED", parent_token_id=args[1])
        elif "SET state = 'REVOKED'" in sql:
            self.registration["state"] = "REVOKED"
        elif "SET state = 'FAILED'" in sql:
            self.registration.update(state="FAILED", fault_message=args[1])
        elif "SET state = 'SUBMITTED'" in sql:
            self.registration.update(state="SUBMITTED", model_digest=args[1], submission_id=args[2])
        elif "SET state = 'REAPED'" in sql:
            self.reaped = getattr(self, "reaped", [])
            self.reaped.append(args[0])
        elif "SET updated_at" in sql:
            pass
        else:
            raise AssertionError(f"unexpected execute: {sql}")


class FakePool:
    def __init__(self, conn: FakeConn):
        self.conn = conn

    def acquire(self):
        return _Ctx(self.conn)


class FakeGateway:
    def __init__(self, fail: bool = False):
        self.minted: list[str] = []
        self.revoked: list[str] = []
        self.fail = fail

    def create_parent_token(self, name: str) -> ParentToken:
        if self.fail:
            raise RuntimeError("cloudflare down")
        self.minted.append(name)
        return ParentToken("tok-1", "tok-1", "s" * 64)

    def revoke_parent_token(self, token_id: str) -> None:
        self.revoked.append(token_id)


def make_deps(s3: FakeS3, gateway: FakeGateway | None = None) -> controller.Deps:
    return controller.Deps(
        settings=SETTINGS,
        gateway=gateway or FakeGateway(),
        mailbox=MailboxStore(s3, bucket=SETTINGS.mailbox_bucket_name),
        uploads=R2UploadController(
            s3,
            private_model_bucket=BUCKET,
            genesis_contract_files={"config.json": hashlib.sha256(GENESIS_CONFIG).hexdigest()},
        ),
        cipher=MailboxCipher(SigningKey(b"v" * 32)),
    )


@pytest.fixture(autouse=True)
def _pin_settings(monkeypatch):
    monkeypatch.setattr(intake, "_settings", lambda: SETTINGS)


def _activate_signal(payload: str | None = None, uid: int | None = 5) -> PrivateSignal:
    if payload is None:
        payload = activation_signal_payload(SUBMISSION_PUBKEY)
    return PrivateSignal("activate", 97, 100, uid, HOTKEY, payload)


# --- intake ------------------------------------------------------------------


def _apply(conn_or_pool, signals) -> int:
    pool = conn_or_pool if isinstance(conn_or_pool, FakePool) else FakePool(conn_or_pool)
    return asyncio.run(intake.handle_signals(pool, signals))


def test_intake_activates_a_signed_registration():
    conn = FakeConn()
    assert _apply(conn, [_activate_signal()]) == 1
    assert conn.registration["registration_id"] == RID
    assert conn.registration["state"] == "ACTIVATED"
    # re-scan of the same signal is a no-op
    assert _apply(conn, [_activate_signal()]) == 0


def test_intake_rejects_malformed_used_hotkeys_and_unregistered():
    malformed = _activate_signal(payload="r2activate:v1:not-a-valid-pubkey")
    assert _apply(FakeConn(), [malformed]) == 0
    assert _apply(FakeConn(used=True), [_activate_signal()]) == 0
    assert _apply(FakeConn(), [_activate_signal(uid=None)]) == 0


def test_intake_applies_ready_only_after_credentials():
    ready = PrivateSignal("ready", 97, 200, 5, HOTKEY, ready_signal_payload("c" * 64))
    conn = FakeConn()
    assert _apply(conn, [ready]) == 0  # unknown registration
    conn.registration = {"id": 1, "registration_id": RID, "state": "ACTIVATED"}
    assert _apply(conn, [ready]) == 0  # not credentialed yet
    conn.registration["state"] = "CREDENTIALED"
    assert _apply(conn, [ready]) == 1
    assert conn.registration["manifest_sha256"] == "c" * 64
    assert conn.registration["ready_block"] == 200


# --- controller lifecycle ----------------------------------------------------


def _seeded_row() -> dict:
    return {
        "id": 1,
        "netuid": 97,
        "uid": 5,
        "hotkey": HOTKEY,
        "registration_id": RID,
        "activation_block": 100,
        "submission_pubkey": SUBMISSION_PUBKEY.hex(),
        "state": "ACTIVATED",
        "ready_block": None,
        "manifest_sha256": None,
        "model_digest": None,
        "parent_token_id": None,
        "submission_id": None,
        "fault_message": None,
    }


def test_controller_runs_the_full_lifecycle(monkeypatch):
    s3 = FakeS3()
    files = model_files()
    manifest = build_manifest(SUBMISSION_KEY, HOTKEY, RID, files)
    for path, data in files.items():
        s3.put(BUCKET, f"{PREFIX}{path}", data)
    s3.put(BUCKET, f"{PREFIX}manifest.json", manifest.as_bytes())

    conn = FakeConn(_seeded_row())
    pool = FakePool(conn)
    deps = make_deps(s3)

    # ACTIVATED -> CREDENTIALED: token minted, sealed envelope in the mailbox
    assert asyncio.run(controller.tick(pool, deps))
    assert conn.registration["state"] == "CREDENTIALED"
    assert conn.registration["parent_token_id"] == "tok-1"
    mail_key = mailbox_object_key(RID, 1)
    ciphertext, _meta = s3.objects[(SETTINGS.mailbox_bucket_name, mail_key)]
    envelope = MailboxCipher.decrypt_for_miner(ciphertext, SUBMISSION_KEY)
    assert envelope["allowed_prefix"] == PREFIX
    assert envelope["access_key_id"] == "tok-1"

    # ready signal arrives on chain
    conn.registration.update(
        state="READY", manifest_sha256=manifest.manifest_sha256, ready_block=200
    )

    # READY -> REVOKED: token dead, mailbox wiped before anything is verified
    assert asyncio.run(controller.tick(pool, deps))
    assert conn.registration["state"] == "REVOKED"
    assert deps.gateway.revoked == ["tok-1"]
    assert (SETTINGS.mailbox_bucket_name, mail_key) not in s3.objects

    # REVOKED -> SUBMITTED: verified prefix handed off as a model submission
    recorded = []

    async def _fake_insert(pool_arg, commits):
        recorded.extend(commits)
        return len(commits)

    monkeypatch.setattr(controller.chain_db, "insert_new_commits", _fake_insert)
    conn.submission_id = uuid.uuid4()
    assert asyncio.run(controller.tick(pool, deps))
    assert conn.registration["state"] == "SUBMITTED"
    assert conn.registration["submission_id"] == conn.submission_id
    (commit,) = recorded
    assert commit.model_uri == (
        f"s3://{BUCKET}/models/registrations/{RID}@sha256:{manifest.model_digest}"
    )
    assert commit.commit_payload["digest"] == f"sha256:{manifest.model_digest}"
    assert commit.uid == 5 and commit.hotkey == HOTKEY and commit.block_number == 200

    # nothing left to do
    assert not asyncio.run(controller.tick(pool, deps))


def test_controller_fails_and_cleans_up_a_bad_upload():
    s3 = FakeS3()
    s3.put(BUCKET, f"{PREFIX}model.safetensors", b"whatever")
    row = _seeded_row()
    row.update(state="REVOKED", manifest_sha256="0" * 64, parent_token_id="tok-1")
    conn = FakeConn(row)
    assert asyncio.run(controller.tick(FakePool(conn), make_deps(s3)))
    assert conn.registration["state"] == "FAILED"
    assert "manifest.json" in conn.registration["fault_message"]
    assert not [k for k in s3.objects if k[1].startswith(PREFIX)]


def test_controller_retries_transient_failures_without_burning_state():
    conn = FakeConn(_seeded_row())
    deps = make_deps(FakeS3(), gateway=FakeGateway(fail=True))
    assert asyncio.run(controller.tick(FakePool(conn), deps))
    assert conn.registration["state"] == "ACTIVATED"


def test_sweep_kills_a_petabyte_uploader_in_the_credentialed_window():
    s3 = FakeS3()
    # miner with live creds fills the prefix but never sends ready
    for i in range(6):
        s3.put(BUCKET, f"{PREFIX}shard-{i}", b"x" * 400)
    row = _seeded_row()
    row.update(state="CREDENTIALED", parent_token_id="tok-1")
    conn = FakeConn(row)  # age_s=0 → within window, so only quota can fail it
    deps = make_deps(s3)
    deps.uploads.max_upload_bytes = 1000  # 2400 bytes uploaded > cap

    breached = asyncio.run(controller.sweep_credentialed(FakePool(conn), deps))
    assert breached == 1
    assert conn.registration["state"] == "FAILED"
    assert "quota abused" in conn.registration["fault_message"]
    assert deps.gateway.revoked == ["tok-1"]
    assert not [k for k in s3.objects if k[1].startswith(PREFIX)]


def test_sweep_abandons_a_miner_who_uploads_but_never_sends_ready():
    s3 = FakeS3()
    s3.put(BUCKET, f"{PREFIX}model.safetensors", b"x" * 100)  # a fine upload, just no ready
    row = _seeded_row()
    row.update(state="CREDENTIALED", parent_token_id="tok-1")
    deps = make_deps(s3)
    conn = FakeConn(row, age_s=deps.settings.upload_window_seconds + 1)  # window elapsed

    assert asyncio.run(controller.sweep_credentialed(FakePool(conn), deps)) == 1
    assert conn.registration["state"] == "FAILED"
    assert "upload window expired" in conn.registration["fault_message"]
    assert deps.gateway.revoked == ["tok-1"]
    assert not [k for k in s3.objects if k[1].startswith(PREFIX)]


def test_sweep_leaves_honest_in_window_uploaders_alone():
    s3 = FakeS3()
    s3.put(BUCKET, f"{PREFIX}model.safetensors", b"x" * 100)
    row = _seeded_row()
    row.update(state="CREDENTIALED", parent_token_id="tok-1")
    conn = FakeConn(row, age_s=60)  # fresh, under quota
    deps = make_deps(s3)
    assert asyncio.run(controller.sweep_credentialed(FakePool(conn), deps)) == 0
    assert conn.registration["state"] == "CREDENTIALED"
    assert deps.gateway.revoked == []


def test_reap_losers_wipes_prefix_and_marks_reaped():
    s3 = FakeS3()
    old_loser = "c" * 64  # a loss beyond the keep window
    s3.put(BUCKET, f"{model_prefix(old_loser)}model.safetensors", b"x" * 500)
    conn = FakeConn()
    conn.reapable = [{"id": 7, "registration_id": old_loser}]
    deps = make_deps(s3)

    reaped = asyncio.run(controller.reap_losers(FakePool(conn), deps))
    assert reaped == 1
    assert conn.last_offset == deps.settings.keep_recent_losers  # asks the DB to keep N
    assert conn.reaped == [7]  # row kept as REAPED (audit trail)
    assert not [k for k in s3.objects if k[1].startswith(model_prefix(old_loser))]


def test_reap_losers_noop_when_within_the_keep_window():
    conn = FakeConn()
    conn.reapable = []  # OFFSET keep returned nothing → all losses still within window
    assert asyncio.run(controller.reap_losers(FakePool(conn), make_deps(FakeS3()))) == 0
