"""Access controller: drives private_registrations through its lifecycle.

ACTIVATED -> CREDENTIALED  mint parent token, publish sealed credentials
READY     -> REVOKED       kill upload access, wipe the mailbox
REVOKED   -> SUBMITTED     verify the frozen prefix, hand off a model submission
          -> FAILED        miner-fault verification failure (uploaded bytes retained)

The linear state machine is the revoke-before-verify gate: verification only
ever runs on rows whose parent token was successfully deleted.

Uploaded bytes are retained on every terminal path. reap_expired is the only
place that deletes a model, once it is older than the retention window and no
live registration still needs it; each attempt owns a distinct prefix, so a retry
never disturbs an earlier upload.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Collection, Mapping

import asyncpg
import boto3
import httpx
from botocore.config import Config
from loguru import logger as log
from nacl.signing import SigningKey

from albedo_config import get_chain_reader_settings
from chain_reader import db as chain_db
from chain_reader.chain import Commit, _payload_hash
from model_validation.validate.genesis_files import GENESIS_SHA256
from private_store.cloudflare import CloudflareR2TokenGateway
from private_store.contracts import mailbox_object_key, model_prefix
from private_store.crypto import MailboxCipher
from private_store.digests import ArtifactIntegrityError
from private_store.r2_credentials import create_local_temporary_credentials
from private_store.settings import PrivateStoreSettings
from private_store.storage import MailboxStore, R2UploadController


@dataclass
class Deps:
    settings: PrivateStoreSettings
    gateway: CloudflareR2TokenGateway
    mailbox: MailboxStore
    uploads: R2UploadController
    cipher: MailboxCipher


def build_deps(settings: PrivateStoreSettings) -> Deps:
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.endpoint,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )
    return Deps(
        settings=settings,
        gateway=CloudflareR2TokenGateway(
            httpx.Client(timeout=30),
            account_id=settings.account_id,
            management_token=settings.api_token,
            bucket=settings.private_models_bucket_name,
            jurisdiction=settings.jurisdiction,
        ),
        mailbox=MailboxStore(s3, bucket=settings.mailbox_bucket_name),
        uploads=R2UploadController(
            s3,
            private_model_bucket=settings.private_models_bucket_name,
            genesis_contract_files=GENESIS_SHA256,
            max_upload_bytes=settings.max_upload_bytes,
        ),
        cipher=MailboxCipher(SigningKey(settings.mailbox_signing_key_bytes())),
    )


def _prefix_of(row: Mapping[str, Any]) -> str:
    return row["model_prefix"] or model_prefix(row["registration_id"])


async def _credential(conn: asyncpg.Connection, row: asyncpg.Record, deps: Deps) -> None:
    settings = deps.settings
    rid = row["registration_id"]
    prefix = model_prefix(rid, row["attempt_count"])
    if row["parent_token_id"]:
        await asyncio.to_thread(deps.gateway.revoke_parent_token, row["parent_token_id"])
    parent = await asyncio.to_thread(deps.gateway.create_parent_token, f"albedo-registration-{rid}")
    temp = create_local_temporary_credentials(
        endpoint=settings.endpoint,
        account_id=settings.account_id,
        parent_access_key_id=parent.access_key_id,
        parent_secret_access_key=parent.secret_access_key,
        bucket=settings.private_models_bucket_name,
        prefix=prefix,
        ttl_seconds=settings.credential_ttl_seconds,
    )
    ciphertext, _ = deps.cipher.create_ciphertext(
        submission_pubkey=bytes.fromhex(row["submission_pubkey"]),
        hotkey=row["hotkey"],
        netuid=row["netuid"],
        registration_id=rid,
        generation=1,
        endpoint=settings.endpoint,
        private_model_bucket=settings.private_models_bucket_name,
        allowed_prefix=prefix,
        access_key_id=temp.access_key_id,
        secret_access_key=temp.secret_access_key,
        session_token=temp.session_token,
        expires_at=datetime.fromtimestamp(temp.expires_at_unix, tz=timezone.utc),
        chain_generation=settings.chain_generation,
    )
    key = mailbox_object_key(rid, 1)
    # a crash after publish re-mints the parent, so clear any stale envelope
    await asyncio.to_thread(deps.mailbox.delete, [key])
    await asyncio.to_thread(deps.mailbox.publish, key, ciphertext)
    await conn.execute(
        """
        UPDATE private_registrations
        SET state = 'CREDENTIALED', parent_token_id = $2,
            credential_expires_at = to_timestamp($3), model_prefix = $4, updated_at = now()
        WHERE id = $1
        """,
        row["id"],
        parent.token_id,
        temp.expires_at_unix,
        prefix,
    )
    log.info("[access-controller] credentials published for {}", rid)


async def _revoke(conn: asyncpg.Connection, row: asyncpg.Record, deps: Deps) -> None:
    rid = row["registration_id"]
    if row["parent_token_id"]:
        await asyncio.to_thread(deps.gateway.revoke_parent_token, row["parent_token_id"])
    await asyncio.to_thread(deps.mailbox.delete, [mailbox_object_key(rid, 1)])
    await conn.execute(
        "UPDATE private_registrations SET state = 'REVOKED', updated_at = now() WHERE id = $1",
        row["id"],
    )
    log.info("[access-controller] upload access revoked for {}", rid)


async def _verify(
    conn: asyncpg.Connection, row: asyncpg.Record, deps: Deps, pool: asyncpg.Pool
) -> None:
    settings = deps.settings
    rid = row["registration_id"]
    prefix = _prefix_of(row)
    try:
        verified = await asyncio.to_thread(
            deps.uploads.verify_manifest,
            model_prefix=prefix,
            registration_id=rid,
            hotkey=row["hotkey"],
            submission_pubkey=bytes.fromhex(row["submission_pubkey"]),
            expected_manifest_sha256=row["manifest_sha256"],
        )
    except ArtifactIntegrityError as exc:
        await asyncio.to_thread(deps.uploads.abort_multipart_uploads, prefix)
        await conn.execute(
            """
            UPDATE private_registrations
            SET state = 'FAILED', fault_message = $2, updated_at = now()
            WHERE id = $1
            """,
            row["id"],
            str(exc),
        )
        log.warning("[access-controller] verification FAILED for {}: {}", rid, exc)
        return
    digest = verified.manifest.model_digest
    payload = {
        "version": "r2",
        "registration_id": rid,
        "digest": f"sha256:{digest}",
        "manifest_sha256": row["manifest_sha256"],
        "model_name": verified.manifest.model_name,
    }
    commit = Commit(
        netuid=row["netuid"],
        block_number=row["ready_block"],
        block_hash=row["ready_block_hash"],
        extrinsic_hash=None,
        uid=row["uid"],
        hotkey=row["hotkey"],
        commit_payload=payload,
        model_uri=(
            f"s3://{settings.private_models_bucket_name}/{prefix.rstrip('/')}@sha256:{digest}"
        ),
        payload_hash=_payload_hash(payload),
    )
    await chain_db.insert_new_commits(pool, [commit])
    submission_id = await conn.fetchval(
        "SELECT submission_id FROM chain_commits"
        " WHERE netuid = $1 AND hotkey = $2 AND payload_hash = $3",
        commit.netuid,
        commit.hotkey,
        commit.payload_hash,
    )
    await conn.execute(
        """
        UPDATE private_registrations
        SET state = 'SUBMITTED', model_digest = $2, submission_id = $3, updated_at = now()
        WHERE id = $1
        """,
        row["id"],
        digest,
        submission_id,
    )
    log.info(
        "[access-controller] verified {} -> submission {} model_uri {}",
        rid,
        submission_id,
        commit.model_uri,
    )


async def _close_upload_window(
    pool: asyncpg.Pool, deps: Deps, row: asyncpg.Record, reason: str
) -> None:
    """Take upload access away from a registration that will not be evaluated.

    Every completed object stays until the reaper needs the room. This ends write
    access — revoke the parent token, remove the sealed envelope — aborts unfinished
    multiparts (invisible to the reaper's inventory, so they would leak forever), and
    records why, which is also what stops the row being swept again.
    """
    if row["parent_token_id"]:
        await asyncio.to_thread(deps.gateway.revoke_parent_token, row["parent_token_id"])
    await asyncio.to_thread(deps.mailbox.delete, [mailbox_object_key(row["registration_id"], 1)])
    await asyncio.to_thread(deps.uploads.abort_multipart_uploads, _prefix_of(row))
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE private_registrations SET state = 'FAILED', fault_message = $2,"
            " updated_at = now() WHERE id = $1",
            row["id"],
            reason,
        )
    log.warning(
        "[access-controller] upload window closed for {} — {}", row["registration_id"], reason
    )


async def sweep_credentialed(pool: asyncpg.Pool, deps: Deps) -> int:
    """Fail credentialed registrations that never submit or abuse the upload window.

    Two reasons a miner holding live credentials is dropped without evaluation:
    they never committed ready within upload_window_seconds, or their prefix
    crossed the byte/object quota (petabytes, millions of objects).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, registration_id, parent_token_id, attempt_count, model_prefix,"
            " EXTRACT(EPOCH FROM (now() - updated_at)) AS age_s"
            " FROM private_registrations WHERE state = 'CREDENTIALED'"
        )
    swept = 0
    for row in rows:
        if row["age_s"] > deps.settings.upload_window_seconds:
            reason = f"upload window expired ({int(row['age_s'])}s with no ready signal)"
        else:
            breach = await asyncio.to_thread(deps.uploads.quota_breach, _prefix_of(row))
            reason = f"upload quota abused: {breach}" if breach else None
        if reason is None:
            continue
        await _close_upload_window(pool, deps, row, reason)
        swept += 1
    return swept


_REAPABLE_SUBMISSION_STATES = ("COMPLETE_LOSS", "TERMINAL_INVALID", "TERMINAL_INFRA_FAILED")


def plan_reap(
    retained: Mapping[str, Any], *, cutoff: datetime, protected: Collection[str]
) -> list[str]:
    """Prefixes whose newest object predates `cutoff`, skipping any a live registration needs."""
    return sorted(p for p, newest in retained.items() if newest < cutoff and p not in protected)


async def _protected_prefixes(pool: asyncpg.Pool) -> set[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT pr.registration_id, pr.attempt_count, pr.model_prefix
            FROM private_registrations pr
            LEFT JOIN model_submissions ms ON ms.id = pr.submission_id
            WHERE pr.state <> 'FAILED'
              AND COALESCE(ms.state, '') <> ALL($1::text[])
            """,
            list(_REAPABLE_SUBMISSION_STATES),
        )
    return {_prefix_of(row) for row in rows}


async def reap_expired(pool: asyncpg.Pool, deps: Deps) -> int:
    """Delete private uploads older than the retention window — the only place bytes go.

    Age is the newest object's LastModified, i.e. when the upload finished. Losers,
    failed attempts, abandoned uploads and orphans with no row all go once older than
    `retention_hours`; anything a live registration still needs is skipped. Deleting a
    prefix drops it from the next inventory, so nothing is recorded to avoid reaping twice.
    """
    hours = deps.settings.retention_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    retained = await asyncio.to_thread(deps.uploads.retained_uploads)
    victims = plan_reap(retained, cutoff=cutoff, protected=await _protected_prefixes(pool))
    if not victims:
        return 0
    log.warning(
        "[access-controller] {} of {} uploads older than {}h — reaping",
        len(victims),
        len(retained),
        hours,
    )
    reaped = 0
    for prefix in victims:
        try:
            removed = await asyncio.to_thread(deps.uploads.cleanup_model_prefix, prefix)
        except Exception as exc:
            log.opt(exception=True).warning(
                "[access-controller] reap failed for {} — will retry: {}", prefix, exc
            )
            continue
        reaped += 1
        log.warning("[access-controller] reaped {} ({} objects)", prefix, removed)
    return reaped


async def _sanity_blocked(pool: asyncpg.Pool, hotkey: str) -> bool:
    async with pool.acquire() as conn:
        return bool(
            await conn.fetchval(
                """
                SELECT 1 FROM sanity_results sr
                JOIN model_submissions ms ON ms.model_uri = sr.repo
                WHERE ms.hotkey = $1 AND sr.passed = false
                  AND (sr.reason ILIKE '%injection%' OR sr.reason ILIKE '%low vocab%')
                LIMIT 1
                """,
                hotkey,
            )
        )


async def _reset_for_retry(pool: asyncpg.Pool, deps: Deps, row: asyncpg.Record) -> None:
    """Grant another attempt. The previous attempt's bytes are kept.

    Bumping attempt_count moves the next upload to its own prefix, so nothing from
    this attempt can be mistaken for part of the next one.
    """
    rid = row["registration_id"]
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE private_registrations
            SET state = 'ACTIVATED',
                activation_block = GREATEST(
                    activation_block,
                    COALESCE((SELECT max(block_number) FROM chain_commits), activation_block)),
                attempt_count = attempt_count + 1,
                parent_token_id = NULL, credential_expires_at = NULL, model_prefix = NULL,
                ready_block = NULL, ready_block_hash = NULL, manifest_sha256 = NULL,
                model_digest = NULL, submission_id = NULL, fault_message = NULL,
                updated_at = now()
            WHERE id = $1
            """,
            row["id"],
        )
    log.info(
        "[access-controller] retry granted for {} — starting attempt {}",
        rid,
        row["attempt_count"] + 1,
    )


_BLOCKED_FAULT_CODES = frozenset(
    {
        "hotkey_sanity_blocked",
        "hotkey_preeval_blocked",
        "hotkey_already_validated",
        "hotkey_duplicate_blocked",
    }
)


async def resubmit_terminal(pool: asyncpg.Pool, deps: Deps) -> int:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT pr.id, pr.registration_id, pr.hotkey, pr.attempt_count, ms.fault_code
            FROM private_registrations pr
            JOIN model_submissions ms ON ms.id = pr.submission_id
            WHERE pr.state = 'SUBMITTED' AND ms.state = 'TERMINAL_INVALID'
              AND pr.attempt_count < $1
            """,
            deps.settings.max_attempts,
        )
    reset = 0
    for row in rows:
        if row["fault_code"] in _BLOCKED_FAULT_CODES:
            continue  # hotkey already blocked — a retry would just fail again
        if await _sanity_blocked(pool, row["hotkey"]):
            continue  # prompt-injection / low-vocab strike
        await _reset_for_retry(pool, deps, row)
        reset += 1
    return reset


async def tick(pool: asyncpg.Pool, deps: Deps) -> bool:
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT * FROM private_registrations
                WHERE state IN ('ACTIVATED', 'READY', 'REVOKED')
                ORDER BY updated_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """
            )
            if row is None:
                return False
            try:
                if row["state"] == "ACTIVATED":
                    await _credential(conn, row, deps)
                elif row["state"] == "READY":
                    await _revoke(conn, row, deps)
                else:
                    await _verify(conn, row, deps, pool)
            except Exception as exc:
                log.opt(exception=True).warning(
                    "[access-controller] {} step failed for {} — will retry: {}",
                    row["state"],
                    row["registration_id"],
                    exc,
                )
                await conn.execute(
                    "UPDATE private_registrations SET updated_at = now() WHERE id = $1",
                    row["id"],
                )
            return True


async def run() -> None:
    settings = PrivateStoreSettings()
    reader = get_chain_reader_settings()
    deps = build_deps(settings)
    pool = await asyncpg.create_pool(dsn=reader.DB_URL, min_size=1, max_size=2)
    log.info(
        "access-controller started — bucket={} mailbox={} validator_identity={}",
        settings.private_models_bucket_name,
        settings.mailbox_bucket_name,
        deps.cipher.validator_identity,
    )
    loop = asyncio.get_event_loop()
    next_sweep = 0.0
    try:
        while True:
            if loop.time() >= next_sweep:
                await sweep_credentialed(pool, deps)
                await resubmit_terminal(pool, deps)
                await reap_expired(pool, deps)
                next_sweep = loop.time() + settings.sweep_interval_s
            if not await tick(pool, deps):
                await asyncio.sleep(settings.poll_interval_s)
    finally:
        await pool.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("access-controller stopped")


if __name__ == "__main__":
    main()
