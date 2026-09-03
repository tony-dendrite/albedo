"""Intake and access-controller lifecycle tests (fake chain, S3, Cloudflare, DB)."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta, timezone

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
        self.sanity_blocked = False
        self.protected: list[dict] = []  # rows the reaper's guard query returns

    def transaction(self):
        return _Ctx(self)

    async def fetchval(self, sql, *args):
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
                "attempt_count": 1,  # schema default
                "ready_block": None,
                "ready_block_hash": None,
                "manifest_sha256": None,
                "model_digest": None,
                "parent_token_id": None,
                "model_prefix": None,
                "submission_id": None,
                "fault_message": None,
            }
            return 1
        if "attempt_count = attempt_count + 1" in sql and "submission_pubkey = $2" in sql:  # re-key
            row = self.registration
            if not (
                row
                and row["state"] in ("ACTIVATED", "CREDENTIALED")
                and row["submission_pubkey"] != args[1]
                and args[2] > row["activation_block"]  # only a newer activate re-keys
                and row["attempt_count"] < args[3]
            ):
                return None
            row.update(
                submission_pubkey=args[1],
                attempt_count=row["attempt_count"] + 1,
                state="ACTIVATED",
                activation_block=args[2],
                credential_expires_at=None,
                model_prefix=None,
            )
            return row["attempt_count"]
        if "used_hotkeys" in sql:
            return 1 if self.used else None
        if "sanity_results" in sql:
            return 1 if self.sanity_blocked else None
        if "SET state = 'READY'" in sql:
            row = self.registration
            if (
                row
                and row["state"] == "CREDENTIALED"
                and row["registration_id"] == args[0]
                and args[2] > row.get("activation_block", 0)
            ):
                row.update(
                    state="READY",
                    manifest_sha256=args[1],
                    ready_block=args[2],
                    ready_block_hash=args[3],
                )
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
        if "LEFT JOIN model_submissions" in sql:  # _protected_prefixes query
            return self.protected
        if "TERMINAL_INVALID" in sql:  # resubmit_terminal query
            row = self.registration
            if (
                row
                and row["state"] == "SUBMITTED"
                and row.get("submission_state") == "TERMINAL_INVALID"
                and row["attempt_count"] < args[0]
            ):
                return [row]
            return []
        assert "state = 'CREDENTIALED'" in sql
        row = self.registration
        if not (row and row["state"] == "CREDENTIALED"):
            return []
        return [{**row, "age_s": self.age_s}]

    async def execute(self, sql, *args):
        if "SET state = 'CREDENTIALED'" in sql:
            self.registration.update(
                state="CREDENTIALED", parent_token_id=args[1], model_prefix=args[3]
            )
        elif "SET state = 'REVOKED'" in sql:
            self.registration["state"] = "REVOKED"
        elif "SET state = 'FAILED'" in sql:
            self.registration.update(state="FAILED", fault_message=args[1])
        elif "SET state = 'SUBMITTED'" in sql:
            self.registration.update(state="SUBMITTED", model_digest=args[1], submission_id=args[2])
        elif "SET state = 'ACTIVATED'" in sql:
            self.registration.update(
                model_prefix=None,
                state="ACTIVATED",
                attempt_count=self.registration["attempt_count"] + 1,
                submission_id=None,
                ready_block=None,
                manifest_sha256=None,
            )
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


def make_deps(
    s3: FakeS3, gateway: FakeGateway | None = None, *, retention_hours: float | None = None
) -> controller.Deps:
    settings = SETTINGS
    if retention_hours is not None:
        settings = SETTINGS.model_copy(update={"retention_hours": retention_hours})
    return controller.Deps(
        settings=settings,
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


def _activate_signal(
    payload: str | None = None, uid: int | None = 5, block: int = 100
) -> PrivateSignal:
    if payload is None:
        payload = activation_signal_payload(SUBMISSION_PUBKEY)
    return PrivateSignal("activate", 97, block, uid, HOTKEY, payload)


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


def test_intake_rekeys_a_credentialed_registration_as_a_new_attempt():
    s3 = FakeS3()
    s3.put(BUCKET, f"{PREFIX}model-00001.safetensors", b"x" * 500)  # attempt 1's upload
    row = _seeded_row()
    row.update(state="CREDENTIALED", parent_token_id="tok-old")
    conn = FakeConn(row)
    new_key = bytes(SigningKey(b"n" * 32).verify_key)

    assert _apply(conn, [_activate_signal(activation_signal_payload(new_key), block=150)]) == 1
    assert conn.registration["state"] == "ACTIVATED"
    assert conn.registration["attempt_count"] == 2
    assert conn.registration["submission_pubkey"] == new_key.hex()
    assert conn.registration["activation_block"] == 150  # forward only
    # re-scan of that same signal is a no-op
    assert _apply(conn, [_activate_signal(activation_signal_payload(new_key), block=150)]) == 0

    deps = make_deps(s3)
    deps.uploads.max_upload_bytes = 600  # attempt 1's bytes alone would breach a shared quota
    assert asyncio.run(controller.tick(FakePool(conn), deps))
    assert conn.registration["state"] == "CREDENTIALED"
    assert "tok-old" in deps.gateway.revoked  # the previous session's token is dead
    assert conn.registration["model_prefix"] == model_prefix(RID, 2)  # own prefix
    assert [k for k in s3.objects if k[1].startswith(PREFIX)]  # attempt 1 untouched
    assert deps.uploads.quota_breach(model_prefix(RID, 2)) is None  # counted separately


def test_intake_ignores_replayed_older_activates():
    row = _seeded_row()
    row.update(state="CREDENTIALED", parent_token_id="tok-1", activation_block=500)
    conn = FakeConn(row)
    old_key = bytes(SigningKey(b"o" * 32).verify_key)
    assert _apply(conn, [_activate_signal(activation_signal_payload(old_key), block=100)]) == 0
    assert conn.registration["attempt_count"] == 1
    assert conn.registration["submission_pubkey"] == SUBMISSION_PUBKEY.hex()
    assert conn.registration["activation_block"] == 500


def test_intake_ignores_a_new_key_once_ready_done_or_out_of_attempts():
    new_key = bytes(SigningKey(b"k" * 32).verify_key)
    for state in ("READY", "REVOKED", "SUBMITTED", "FAILED"):
        row = _seeded_row()
        row.update(state=state)
        conn = FakeConn(row)
        assert _apply(conn, [_activate_signal(activation_signal_payload(new_key))]) == 0
        assert conn.registration["state"] == state
    capped = _seeded_row()
    capped.update(state="CREDENTIALED", attempt_count=SETTINGS.max_attempts)
    conn = FakeConn(capped)
    assert _apply(conn, [_activate_signal(activation_signal_payload(new_key))]) == 0
    assert conn.registration["attempt_count"] == SETTINGS.max_attempts


def test_intake_rejects_malformed_used_hotkeys_and_unregistered():
    malformed = _activate_signal(payload="r2activate:v1:not-a-valid-pubkey")
    assert _apply(FakeConn(), [malformed]) == 0
    assert _apply(FakeConn(used=True), [_activate_signal()]) == 0
    assert _apply(FakeConn(), [_activate_signal(uid=None)]) == 0


def test_intake_applies_ready_only_after_credentials():
    ready = PrivateSignal("ready", 97, 200, 5, HOTKEY, ready_signal_payload("c" * 64), "0xdead200")
    conn = FakeConn()
    assert _apply(conn, [ready]) == 0  # unknown registration
    conn.registration = {
        "id": 1,
        "registration_id": RID,
        "state": "ACTIVATED",
        "activation_block": 100,
    }
    assert _apply(conn, [ready]) == 0  # not credentialed yet
    conn.registration["state"] = "CREDENTIALED"
    assert _apply(conn, [ready]) == 1
    assert conn.registration["manifest_sha256"] == "c" * 64
    assert conn.registration["ready_block"] == 200
    assert conn.registration["ready_block_hash"] == "0xdead200"


def test_intake_ignores_ready_older_than_activation():
    # after a reset bumps activation_block past the old ready block, a stale r2ready
    # re-yielded from the chain must NOT re-fire the old submission.
    conn = FakeConn()
    conn.registration = {
        "id": 1,
        "registration_id": RID,
        "state": "CREDENTIALED",
        "activation_block": 500,
    }
    stale = PrivateSignal("ready", 97, 200, 5, HOTKEY, ready_signal_payload("c" * 64), "0xstale200")
    assert _apply(conn, [stale]) == 0  # ready block 200 <= activation 500 -> ignored
    assert conn.registration["state"] == "CREDENTIALED"  # unchanged
    # a fresh ready committed after the re-activation still applies
    fresh = PrivateSignal("ready", 97, 600, 5, HOTKEY, ready_signal_payload("d" * 64), "0xfresh600")
    assert _apply(conn, [fresh]) == 1
    assert conn.registration["state"] == "READY"
    assert conn.registration["ready_block"] == 600


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
        "attempt_count": 1,  # decides which prefix this attempt owns
        "model_prefix": None,  # set by _credential; NULL rows fall back to attempt 1's prefix
        "ready_block": None,
        "ready_block_hash": None,
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
        state="READY",
        manifest_sha256=manifest.manifest_sha256,
        ready_block=200,
        ready_block_hash="0xabc200",
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
    assert commit.block_hash == "0xabc200"  # same seed logic as public commits

    # nothing left to do
    assert not asyncio.run(controller.tick(pool, deps))


def test_controller_fails_a_bad_upload_but_keeps_its_bytes():
    s3 = FakeS3()
    s3.put(BUCKET, f"{PREFIX}model.safetensors", b"whatever")
    row = _seeded_row()
    row.update(state="REVOKED", manifest_sha256="0" * 64, parent_token_id="tok-1")
    conn = FakeConn(row)
    assert asyncio.run(controller.tick(FakePool(conn), make_deps(s3)))
    assert conn.registration["state"] == "FAILED"
    assert "manifest.json" in conn.registration["fault_message"]
    # only the capacity reaper deletes; a verification failure keeps what was uploaded
    assert [k for k in s3.objects if k[1].startswith(PREFIX)]


def test_a_retry_uploads_beside_the_previous_attempt_and_both_survive(monkeypatch):
    """A granted retry must not cost the miner the model it already uploaded.

    Attempt 1 is verified and evaluated, comes back TERMINAL_INVALID, and a retry is
    granted. Attempt 2 uploads to its own prefix and verifies, while attempt 1's bytes
    remain for the capacity reaper to decide on later.
    """
    s3 = FakeS3()
    files = model_files()
    first = build_manifest(SUBMISSION_KEY, HOTKEY, RID, files)
    for path, data in files.items():
        s3.put(BUCKET, f"{PREFIX}{path}", data)
    s3.put(BUCKET, f"{PREFIX}manifest.json", first.as_bytes())

    # attempt 1 was verified and then failed evaluation
    row = _seeded_row()
    row.update(
        state="SUBMITTED",
        submission_state="TERMINAL_INVALID",
        submission_id=uuid.uuid4(),
        fault_code=None,
    )
    conn = FakeConn(row)
    pool = FakePool(conn)
    deps = make_deps(s3)

    assert asyncio.run(controller.resubmit_terminal(pool, deps)) == 1
    assert conn.registration["attempt_count"] == 2
    assert [k for k in s3.objects if k[1].startswith(PREFIX)]  # attempt 1 retained

    # attempt 2 gets credentials for a prefix of its own
    retry_prefix = model_prefix(RID, 2)
    assert asyncio.run(controller.tick(pool, deps))
    assert conn.registration["state"] == "CREDENTIALED"
    ciphertext, _meta = s3.objects[(SETTINGS.mailbox_bucket_name, mailbox_object_key(RID, 1))]
    envelope = MailboxCipher.decrypt_for_miner(ciphertext, SUBMISSION_KEY)
    assert envelope["allowed_prefix"] == retry_prefix
    assert envelope["allowed_prefix"].split("/")[2] == RID  # what the miner CLI checks

    # the miner uploads a different model to the retry prefix
    second_files = {**files, "model.safetensors": b"second attempt weights"}
    second = build_manifest(SUBMISSION_KEY, HOTKEY, RID, second_files)
    for path, data in second_files.items():
        s3.put(BUCKET, f"{retry_prefix}{path}", data)
    s3.put(BUCKET, f"{retry_prefix}manifest.json", second.as_bytes())
    conn.registration.update(
        state="READY",
        manifest_sha256=second.manifest_sha256,
        ready_block=300,
        ready_block_hash="0xabc300",
    )
    assert asyncio.run(controller.tick(pool, deps))  # -> REVOKED

    recorded = []

    async def _fake_insert(pool_arg, commits):
        recorded.extend(commits)
        return len(commits)

    monkeypatch.setattr(controller.chain_db, "insert_new_commits", _fake_insert)
    conn.submission_id = uuid.uuid4()
    assert asyncio.run(controller.tick(pool, deps))
    assert conn.registration["state"] == "SUBMITTED"

    (commit,) = recorded
    assert commit.model_uri == (
        f"s3://{BUCKET}/models/attempts/{RID}/a2@sha256:{second.model_digest}"
    )
    # both attempts hold their own objects
    assert [k for k in s3.objects if k[1].startswith(PREFIX)]
    assert [k for k in s3.objects if k[1].startswith(retry_prefix)]
    assert first.model_digest != second.model_digest


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
    # access is revoked, but the bytes stay for the capacity reaper to judge
    assert [k for k in s3.objects if k[1].startswith(PREFIX)]


def test_sweep_abandons_a_miner_who_uploads_but_never_sends_ready():
    s3 = FakeS3()
    s3.put(BUCKET, f"{PREFIX}model.safetensors", b"x" * 100)  # a fine upload, just no ready
    s3.multipart.append({"Key": f"{PREFIX}model-inflight", "UploadId": "u1"})
    row = _seeded_row()
    row.update(state="CREDENTIALED", parent_token_id="tok-1")
    deps = make_deps(s3)
    conn = FakeConn(row, age_s=deps.settings.upload_window_seconds + 1)  # window elapsed

    assert asyncio.run(controller.sweep_credentialed(FakePool(conn), deps)) == 1
    assert conn.registration["state"] == "FAILED"
    assert "upload window expired" in conn.registration["fault_message"]
    assert deps.gateway.revoked == ["tok-1"]
    assert [k for k in s3.objects if k[1].startswith(PREFIX)]  # completed bytes retained
    # the unfinished multipart is aborted: it holds no listed object, so the capacity reaper
    # could never see it and it would otherwise occupy storage forever
    assert not s3.multipart


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


def _aged(s3: FakeS3, prefix: str, hours: float) -> None:
    key = f"{prefix}model.safetensors"
    s3.put(BUCKET, key, b"x" * 500)
    s3.modified[(BUCKET, key)] = datetime.now(timezone.utc) - timedelta(hours=hours)


def _held(s3: FakeS3, prefix: str) -> bool:
    return bool([k for k in s3.objects if k[1].startswith(prefix)])


def test_reaper_deletes_uploads_older_than_the_retention_window():
    s3 = FakeS3()
    stale, older, fresh = "a" * 64, "b" * 64, "c" * 64
    for rid, hours in ((stale, 100), (older, 80), (fresh, 1)):
        _aged(s3, model_prefix(rid), hours)
    assert asyncio.run(controller.reap_expired(FakePool(FakeConn()), make_deps(s3))) == 2
    assert not _held(s3, model_prefix(stale)) and not _held(s3, model_prefix(older))
    assert _held(s3, model_prefix(fresh))


def test_reaper_noop_when_everything_is_within_the_window():
    s3 = FakeS3()
    _aged(s3, model_prefix("c" * 64), 71)
    assert asyncio.run(controller.reap_expired(FakePool(FakeConn()), make_deps(s3))) == 0
    assert _held(s3, model_prefix("c" * 64))


def test_reaper_spares_uploads_a_live_registration_still_needs():
    """Old but still queued, evaluating, or a winner: the guard keeps it; an old loser goes."""
    s3 = FakeS3()
    live, loser = "a" * 64, "b" * 64
    _aged(s3, model_prefix(live), 100)
    _aged(s3, model_prefix(loser), 100)
    conn = FakeConn()
    conn.protected = [{"registration_id": live, "attempt_count": 1, "model_prefix": None}]
    assert asyncio.run(controller.reap_expired(FakePool(conn), make_deps(s3))) == 1
    assert _held(s3, model_prefix(live))
    assert not _held(s3, model_prefix(loser))


def test_reaper_evicts_an_old_earlier_attempt_but_keeps_the_fresh_retry():
    s3 = FakeS3()
    rid, other = "a" * 64, "b" * 64
    _aged(s3, model_prefix(rid), 100)  # attempt 1 lost long ago
    _aged(s3, model_prefix(rid, 2), 1)  # the retry is fresh
    _aged(s3, model_prefix(other), 1)
    assert asyncio.run(controller.reap_expired(FakePool(FakeConn()), make_deps(s3))) == 1
    assert not _held(s3, model_prefix(rid))
    assert _held(s3, model_prefix(rid, 2)) and _held(s3, model_prefix(other))


def test_retention_window_is_configurable():
    s3 = FakeS3()
    _aged(s3, model_prefix("d" * 64), 10)
    deps = make_deps(s3, retention_hours=5)
    assert asyncio.run(controller.reap_expired(FakePool(FakeConn()), deps)) == 1


def _submitted_row(
    *, attempt: int = 1, sub_state: str = "TERMINAL_INVALID", fault_code: str | None = None
) -> dict:
    return {
        "id": 1,
        "registration_id": RID,
        "hotkey": HOTKEY,
        "state": "SUBMITTED",
        "attempt_count": attempt,
        "model_prefix": None,
        "submission_state": sub_state,  # models the JOIN to model_submissions.state
        "fault_code": fault_code,
        "activation_block": 100,
        "ready_block": 200,
        "parent_token_id": None,
    }


def test_resubmit_terminal_grants_retry_on_invalid():
    s3 = FakeS3()
    s3.put(BUCKET, f"{PREFIX}model.safetensors", b"x" * 500)  # bytes from the failed attempt
    conn = FakeConn(_submitted_row(attempt=1))
    assert asyncio.run(controller.resubmit_terminal(FakePool(conn), make_deps(s3))) == 1
    assert conn.registration["state"] == "ACTIVATED"  # back for another attempt
    assert conn.registration["attempt_count"] == 2
    assert conn.registration["submission_id"] is None
    # the previous attempt is retained; the next one uploads to its own prefix
    assert [k for k in s3.objects if k[1].startswith(PREFIX)]
    assert model_prefix(RID, 2) != PREFIX


def test_resubmit_terminal_stops_at_max_attempts():
    conn = FakeConn(_submitted_row(attempt=SETTINGS.max_attempts))  # out of attempts
    assert asyncio.run(controller.resubmit_terminal(FakePool(conn), make_deps(FakeS3()))) == 0
    assert conn.registration["state"] == "SUBMITTED"  # not resurrected


def test_resubmit_terminal_skips_sanity_blocked_cheaters():
    conn = FakeConn(_submitted_row(attempt=1))
    conn.sanity_blocked = True  # prompt-injection / low-vocab strike
    assert asyncio.run(controller.resubmit_terminal(FakePool(conn), make_deps(FakeS3()))) == 0
    assert conn.registration["state"] == "SUBMITTED"


def test_resubmit_terminal_leaves_lost_duels_to_reaper():
    conn = FakeConn(_submitted_row(attempt=1, sub_state="COMPLETE_LOSS"))
    assert asyncio.run(controller.resubmit_terminal(FakePool(conn), make_deps(FakeS3()))) == 0
    assert conn.registration["state"] == "SUBMITTED"  # fair loss — reaped, not retried


def test_resubmit_terminal_skips_already_blocked_hotkeys():
    conn = FakeConn(_submitted_row(attempt=1, fault_code="hotkey_preeval_blocked"))
    assert asyncio.run(controller.resubmit_terminal(FakePool(conn), make_deps(FakeS3()))) == 0
    assert conn.registration["state"] == "SUBMITTED"  # blocked hotkey — no churn
