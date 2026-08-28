"""Access controller: drives private_registrations through its lifecycle.

ACTIVATED -> CREDENTIALED  mint parent token, publish sealed credentials
READY     -> REVOKED       kill upload access, wipe the mailbox
REVOKED   -> SUBMITTED     verify the frozen prefix, hand off a model submission
          -> FAILED        miner-fault verification failure (prefix cleaned up)

The linear state machine is the revoke-before-verify gate: verification only
ever runs on rows whose parent token was successfully deleted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

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


async def _credential(conn: asyncpg.Connection, row: asyncpg.Record, deps: Deps) -> None:
    settings = deps.settings
    rid = row["registration_id"]
    prefix = model_prefix(rid)
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
            credential_expires_at = to_timestamp($3), updated_at = now()
        WHERE id = $1
        """,
        row["id"],
        parent.token_id,
        temp.expires_at_unix,
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
    prefix = model_prefix(rid)
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
        removed = await asyncio.to_thread(deps.uploads.cleanup_model_prefix, prefix)
        await conn.execute(
            """
            UPDATE private_registrations
            SET state = 'FAILED', fault_message = $2, updated_at = now()
            WHERE id = $1
            """,
            row["id"],
            str(exc),
        )
        log.warning(
            "[access-controller] verification FAILED for {} ({} objects removed): {}",
            rid,
            removed,
            exc,
        )
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
        block_hash=None,
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


async def _abandon(pool: asyncpg.Pool, deps: Deps, row: asyncpg.Record, reason: str) -> None:
    """Revoke a credentialed registration's upload access, wipe its prefix, FAIL it."""
    prefix = model_prefix(row["registration_id"])
    if row["parent_token_id"]:
        await asyncio.to_thread(deps.gateway.revoke_parent_token, row["parent_token_id"])
    await asyncio.to_thread(deps.mailbox.delete, [mailbox_object_key(row["registration_id"], 1)])
    await asyncio.to_thread(deps.uploads.cleanup_model_prefix, prefix)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE private_registrations SET state = 'FAILED', fault_message = $2,"
            " updated_at = now() WHERE id = $1",
            row["id"],
            reason,
        )
    log.warning("[access-controller] abandoned {} — {}", row["registration_id"], reason)


async def sweep_credentialed(pool: asyncpg.Pool, deps: Deps) -> int:
    """Fail credentialed registrations that never submit or abuse the upload window.

    Two reasons a miner holding live credentials is dropped without evaluation:
    they never committed ready within upload_window_seconds, or their prefix
    crossed the byte/object quota (petabytes, millions of objects).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, registration_id, parent_token_id,"
            " EXTRACT(EPOCH FROM (now() - updated_at)) AS age_s"
            " FROM private_registrations WHERE state = 'CREDENTIALED'"
        )
    swept = 0
    for row in rows:
        if row["age_s"] > deps.settings.upload_window_seconds:
            reason = f"upload window expired ({int(row['age_s'])}s with no ready signal)"
        else:
            prefix = model_prefix(row["registration_id"])
            breach = await asyncio.to_thread(deps.uploads.quota_breach, prefix)
            reason = f"upload quota abused: {breach}" if breach else None
        if reason is None:
            continue
        await _abandon(pool, deps, row, reason)
        swept += 1
    return swept


async def reap_losers(pool: asyncpg.Pool, deps: Deps) -> int:
    """Delete the private bytes of losing models beyond the N most recent.

    A model that lost its duel is no longer needed, but we keep the most recent
    losses on disk for dispute/re-eval; everything older is reaped. Winners are
    reaped separately, only after king_hf_uploader mirrors them to public HF.
    The row is kept (as REAPED) for the audit trail — only the bytes go.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT pr.id, pr.registration_id
            FROM private_registrations pr
            JOIN model_submissions ms ON ms.id = pr.submission_id
            WHERE pr.state = 'SUBMITTED' AND ms.state = 'COMPLETE_LOSS'
            ORDER BY COALESCE(ms.finished_at, ms.updated_at) DESC
            OFFSET $1
            """,
            deps.settings.keep_recent_losers,
        )
    for row in rows:
        await asyncio.to_thread(
            deps.uploads.cleanup_model_prefix, model_prefix(row["registration_id"])
        )
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE private_registrations SET state = 'REAPED', updated_at = now()"
                " WHERE id = $1",
                row["id"],
            )
        log.info(
            "[access-controller] reaped losing model {} (beyond {} most recent)",
            row["registration_id"],
            deps.settings.keep_recent_losers,
        )
    return len(rows)


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
                await reap_losers(pool, deps)
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
