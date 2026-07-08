#!/usr/bin/env python3
"""Ops entrypoint: seed-genesis, preflight, arch-spec, inject-submission."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import asyncpg
import httpx
from loguru import logger

from albedo.remote.common import canonical_model_config
from albedo.settings import get_settings

# ── seed-genesis: genesis king + GPU host registration (idempotent) ──────────

NETUID = 97


UID = 0


COLDKEY = "5EUXD91ADceyH7nRWXCqG1wbaCEhsqosT4rjGhwaZDRR4ib6"


HOTKEY = "5EvHrbHz8rT8DrWazxFhzfMsmscFtPE3qhRDeY4ggKZrBcxZ"


REPO = "teutonic/qwen3.6-35b-a3b-genesis"


MODEL_HASH = "sha256:efd5b8d0a1c1f472be56ff919419cdd0561bdecd9013d5c2a96dd0e23e89c165"


MODEL_URI = f"registry.hippius.com/{REPO}@{MODEL_HASH}"


BURN_UID = 0


def _gpu_hosts() -> list[dict]:
    # Both GPU workers, addressed through the backend-side SSH tunnels.
    s = get_settings()
    return [
        {
            "id": s.remote_eval.host_id,
            "role": "EVAL",
            "base_url": s.eval.remote_base_url,
            "tunnel_name": "albedo-eval-tunnel",
            "state": "READY",
            "gpu_count": s.remote_eval.gpu_count,
            "free_gpu_count": s.remote_eval.gpu_count,
            "accelerator_type": s.remote_eval.accelerator_type or "B200",
            "capabilities": {"generation_backend": "vllm", "score_bridge_connected": True},
            "last_health": {"ready": True, "active_runs": 0},
        },
        {
            "id": s.sanity_remote.host_id,
            "role": "PRE_EVAL",
            "base_url": s.sanity.remote_base_url,
            "tunnel_name": "albedo-sanity-tunnel",
            "state": "READY",
            "gpu_count": 1,
            "free_gpu_count": 1,
            "accelerator_type": "",
            "capabilities": {},
            "last_health": {"ready": True},
        },
    ]


def _weight_hash(reign_version: int) -> str:
    # Deterministic hash of the genesis weight payload (dedupe key for weight_epochs).
    payload = {
        "netuid": NETUID,
        "reign_version": reign_version,
        "uids": [UID],
        "weights": ["1"],
        "policy": {
            "policy": "genesis_bootstrap_v1",
            "burn_uid": BURN_UID,
            "member_count": 1,
            "slot_weight_bps": {"1": 10000},
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


async def _upsert_hosts(conn: asyncpg.Connection) -> None:
    # Registers (or re-points) both GPU workers; the backend heartbeater takes over from here.
    for host in _gpu_hosts():
        await conn.execute(
            """
            INSERT INTO remote_gpu_hosts (
                id, role, base_url, tunnel_name, state, gpu_count, free_gpu_count,
                accelerator_type, capabilities, last_heartbeat_at, last_health
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now(), $10)
            ON CONFLICT (id) DO UPDATE
            SET role = EXCLUDED.role, base_url = EXCLUDED.base_url,
                tunnel_name = EXCLUDED.tunnel_name, state = EXCLUDED.state,
                gpu_count = EXCLUDED.gpu_count, free_gpu_count = EXCLUDED.free_gpu_count,
                accelerator_type = EXCLUDED.accelerator_type,
                capabilities = EXCLUDED.capabilities,
                last_heartbeat_at = now(), last_health = EXCLUDED.last_health
            """,
            host["id"],
            host["role"],
            host["base_url"],
            host["tunnel_name"],
            host["state"],
            host["gpu_count"],
            host["free_gpu_count"],
            host["accelerator_type"],
            host["capabilities"],
            host["last_health"],
        )
        logger.info(f"[seed] gpu host {host['id']} ({host['role']}) -> {host['base_url']}")


async def seed_genesis() -> None:
    # Idempotent bootstrap: refuses to touch a database that already has a different active reign.
    reveal_payload = f"v7|{REPO}|{MODEL_HASH}"
    payload_hash = hashlib.sha256(reveal_payload.encode()).hexdigest()
    idempotency_key = f"genesis:{NETUID}:{HOTKEY}:{payload_hash}"

    conn = await asyncpg.connect(dsn=get_settings().db.dsn)
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    try:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('genesis_bootstrap'))")
            await _upsert_hosts(conn)

            active = await conn.fetchrow(
                """
                SELECT r.id, r.version, rm.hotkey, rm.uid, rm.model_hash
                FROM reigns r
                LEFT JOIN reign_members rm ON rm.reign_id = r.id AND rm.slot = 1
                WHERE r.state = 'ACTIVE' ORDER BY r.version DESC LIMIT 1
                """
            )
            if active and (
                active["hotkey"] != HOTKEY
                or active["uid"] != UID
                or active["model_hash"] != MODEL_HASH
            ):
                raise SystemExit("refusing genesis seed: a different active reign already exists")

            miner_id = await conn.fetchval(
                """
                INSERT INTO miners (id, hotkey, coldkey, uid, netuid, updated_at)
                VALUES ($1, $2, $3, $4, $5, now())
                ON CONFLICT (hotkey) DO UPDATE SET coldkey = EXCLUDED.coldkey,
                    uid = EXCLUDED.uid, netuid = EXCLUDED.netuid, updated_at = now()
                RETURNING id
                """,
                uuid4(),
                HOTKEY,
                COLDKEY,
                UID,
                NETUID,
            )

            commit_id = await conn.fetchval(
                "SELECT id FROM chain_commits WHERE netuid = $1 AND hotkey = $2 "
                "AND payload_hash = $3",
                NETUID,
                HOTKEY,
                payload_hash,
            )
            if commit_id is None:
                commit_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO chain_commits (id, netuid, block_number, block_hash, uid,
                        hotkey, commit_payload, model_uri, payload_hash)
                    VALUES ($1, $2, 0, $3, $4, $5, $6, $7, $8)
                    """,
                    commit_id,
                    NETUID,
                    f"genesis-bootstrap:{payload_hash}",
                    UID,
                    HOTKEY,
                    {"version": "v7", "repo": REPO, "digest": MODEL_HASH, "bootstrap": "genesis"},
                    MODEL_URI,
                    payload_hash,
                )

            submission_id = await conn.fetchval(
                "SELECT id FROM model_submissions WHERE idempotency_key = $1 OR model_hash = $2 "
                "ORDER BY created_at ASC LIMIT 1",
                idempotency_key,
                MODEL_HASH,
            )
            if submission_id is None:
                submission_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO model_submissions (id, miner_id, chain_commit_id, netuid, uid,
                        hotkey, model_uri, commit_hash, model_hash, state, idempotency_key,
                        finished_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'COMPLETE_CORONATED', $10, now())
                    """,
                    submission_id,
                    miner_id,
                    commit_id,
                    NETUID,
                    UID,
                    HOTKEY,
                    MODEL_URI,
                    MODEL_HASH,
                    MODEL_HASH,
                    idempotency_key,
                )
            await conn.execute(
                "UPDATE chain_commits SET submission_id = $1 WHERE id = $2",
                submission_id,
                commit_id,
            )

            artifact_id = await conn.fetchval(
                "SELECT id FROM artifacts WHERE submission_id = $1 "
                "AND artifact_type = 'MODEL_MANIFEST' AND uri = $2 LIMIT 1",
                submission_id,
                MODEL_URI,
            )
            if artifact_id is None:
                artifact_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO artifacts (id, submission_id, artifact_type, storage_backend,
                        uri, sha256, content_type)
                    VALUES ($1, $2, 'MODEL_MANIFEST', 'hippius', $3, $4,
                            'application/vnd.oci.image.manifest.v1+json')
                    """,
                    artifact_id,
                    submission_id,
                    MODEL_URI,
                    MODEL_HASH.removeprefix("sha256:"),
                )

            if active:
                reign_id, reign_version = active["id"], int(active["version"])
            else:
                reign_id, reign_version, king_version_id = uuid4(), 1, uuid4()
                await conn.execute(
                    "INSERT INTO reigns (id, version, reason, trigger_submission_id, state, "
                    "activated_at) VALUES ($1, $2, 'GENESIS', $3, 'ACTIVE', now())",
                    reign_id,
                    reign_version,
                    submission_id,
                )
                await conn.execute(
                    """
                    INSERT INTO king_versions (id, submission_id, model_hash, artifact_id,
                        eval_run_id, version, entered_reign_id, entered_slot, activated_by)
                    VALUES ($1, $2, $3, $4, NULL, 1, $5, 1, 'scripts/ops.py seed-genesis')
                    """,
                    king_version_id,
                    submission_id,
                    MODEL_HASH,
                    artifact_id,
                    reign_id,
                )
                await conn.execute(
                    """
                    INSERT INTO reign_members (id, reign_id, slot, king_version_id, submission_id,
                        hotkey, uid, model_hash, weight_bps)
                    VALUES ($1, $2, 1, $3, $4, $5, $6, $7, 10000)
                    """,
                    uuid4(),
                    reign_id,
                    king_version_id,
                    submission_id,
                    HOTKEY,
                    UID,
                    MODEL_HASH,
                )

            await conn.execute(
                """
                INSERT INTO weight_epochs (id, netuid, reason, reign_id, state, uids, weights,
                    weight_policy, weight_hash)
                VALUES ($1, $2, 'SERVICE_REPLAY', $3, 'PENDING', $4, $5, $6, $7)
                ON CONFLICT (netuid, weight_hash) DO NOTHING
                """,
                uuid4(),
                NETUID,
                reign_id,
                [UID],
                [1.0],
                {
                    "policy": "genesis_bootstrap_v1",
                    "burn_uid": BURN_UID,
                    "member_count": 1,
                    "slot_weight_bps": {"1": 10000},
                    "source": "scripts/ops.py seed-genesis",
                },
                _weight_hash(reign_version),
            )
            await conn.execute(
                """
                INSERT INTO events (id, submission_id, event_type, severity, message, data)
                VALUES ($1, $2, 'genesis_king_bootstrapped', 'INFO',
                        'Genesis king bootstrap completed', $3)
                """,
                uuid4(),
                submission_id,
                {"model_uri": MODEL_URI, "reign_id": str(reign_id), "reign_version": reign_version},
            )
        logger.info(
            f"[seed] genesis ok - reign v{reign_version} ({reign_id}) submission={submission_id}"
        )
    finally:
        await conn.close()


# ── preflight: verify every connection before starting the stack ─────────────

_PASS, _FAIL, _SKIP = "PASS", "FAIL", "skip"


async def _check_db() -> tuple[str, str]:
    # Postgres reachable and the schema applied.
    import asyncpg

    try:
        conn = await asyncpg.connect(dsn=get_settings().db.dsn, timeout=10)
        try:
            n = await conn.fetchval("SELECT count(*) FROM remote_gpu_hosts")
            return _PASS, f"connected; {n} gpu host(s) registered"
        finally:
            await conn.close()
    except Exception as exc:  # noqa: BLE001 - preflight reports, never raises
        return _FAIL, str(exc)


async def _check_gpu(role: str, base_url: str, token: str) -> tuple[str, str]:
    # GPU worker /ready through its tunnel.
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{base_url.rstrip('/')}/ready", headers=headers)
            r.raise_for_status()
            return _PASS, f"{role} ready: {json.dumps(r.json())[:120]}"
    except Exception as exc:  # noqa: BLE001
        return _FAIL, f"{role} at {base_url}: {exc}"


async def _check_score_bridge() -> tuple[str, str]:
    # WS handshake against the eval worker's /score-bridge endpoint.
    import websockets

    sb = get_settings().score_bridge
    headers = {"Authorization": f"Bearer {sb.remote_auth_token}"} if sb.remote_auth_token else {}
    try:
        async with websockets.connect(
            sb.remote_ws_url, additional_headers=headers, open_timeout=10
        ):
            return _PASS, f"handshake ok: {sb.remote_ws_url}"
    except Exception as exc:  # noqa: BLE001
        return _FAIL, f"{sb.remote_ws_url}: {exc}"


async def _check_openrouter() -> tuple[str, str]:
    # One real (tiny) completion per judge model through the shared client.
    j = get_settings().judge
    if not j.openrouter_api_key:
        return _SKIP, "no ALBEDO_JUDGE_OPENROUTER_API_KEY"
    from albedo.judges import JUDGE_MODELS, openrouter_chat

    results = []
    for model in JUDGE_MODELS:
        try:
            reply = await openrouter_chat(
                model, [{"role": "user", "content": "Reply with exactly: ok"}], max_tokens=8
            )
            results.append(f"{model}: {reply.strip()[:20]!r}")
        except Exception as exc:  # noqa: BLE001
            return _FAIL, f"{model}: {exc}"
    return _PASS, "; ".join(results)


async def _check_opensearch() -> tuple[str, str]:
    # Cluster health when dedup is configured.
    os_cfg = get_settings().opensearch
    if not os_cfg.url:
        return _SKIP, "no ALBEDO_OPENSEARCH_URL (validation must be mocked)"
    try:
        auth = (os_cfg.user, os_cfg.password) if os_cfg.user else None
        async with httpx.AsyncClient(timeout=10, auth=auth, verify=False) as client:
            r = await client.get(f"{os_cfg.url.rstrip('/')}/_cluster/health")
            r.raise_for_status()
            return _PASS, f"status={r.json().get('status')}"
    except Exception as exc:  # noqa: BLE001
        return _FAIL, str(exc)


async def _check_s3() -> tuple[str, str]:
    # Bucket reachable when fault/corpus publishing is on.
    from albedo import s3

    if not get_settings().s3.enabled:
        return _SKIP, "ALBEDO_S3_* unset (publishing off - correct for test envs)"
    try:
        client = s3.client()
        await asyncio.to_thread(client.head_bucket, Bucket=get_settings().s3.bucket)
        return _PASS, f"bucket {get_settings().s3.bucket} reachable"
    except Exception as exc:  # noqa: BLE001
        return _FAIL, str(exc)


async def preflight() -> int:
    # Runs every check concurrently and prints a PASS/FAIL/skip table; exit 1 on any FAIL.
    s = get_settings()
    checks = {
        "postgres": _check_db(),
        "sanity-gpu": _check_gpu("sanity", s.sanity.remote_base_url, s.sanity.remote_auth_token),
        "eval-gpu": _check_gpu("eval", s.eval.remote_base_url, s.eval.remote_auth_token),
        "score-bridge-ws": _check_score_bridge(),
        "openrouter-judges": _check_openrouter(),
        "opensearch": _check_opensearch(),
        "s3": _check_s3(),
    }
    results = dict(zip(checks, await asyncio.gather(*checks.values())))
    failed = False
    for name, (status, detail) in results.items():
        failed |= status == _FAIL
        logger.log(
            "ERROR" if status == _FAIL else "INFO", f"[preflight] {status:4s} {name}: {detail}"
        )
    mocks = (
        f"chain={s.chain.mock} weights={s.weights.mock} "
        f"monitor={s.monitor.mock} validation={s.validation.mock}"
    )
    logger.info(f"[preflight] mock flags: {mocks}")
    return 1 if failed else 0


# ── arch-spec: regenerate albedo/architecture_spec.json from the genesis ─────

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "albedo" / "architecture_spec.json"


_LOCK_KEYS = (
    "vocab_size",
    "model_type",
    "max_position_embeddings",
    "tie_word_embeddings",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
)


_FORBIDDEN = ("auto_map", "quantization_config")


def arch_spec(args) -> int:
    # Lock-key values are read from the canonical genesis config (top-level first,
    # falling back to text_config for the multimodal MoE layout).
    out = Path(args.out) if args.out else _DEFAULT_OUT
    cfg = canonical_model_config()
    text_cfg = cfg.get("text_config") or {}
    expected = {}
    for key in _LOCK_KEYS:
        if key in cfg:
            expected[key] = cfg[key]
        elif key in text_cfg:
            expected[key] = text_cfg[key]
            print(f"  note: '{key}' sourced from text_config")
        else:
            raise SystemExit(f"lock key '{key}' missing from the canonical genesis config")
    spec = {
        "_comment": "Generated from the genesis seed config by scripts/generate_arch_spec.py.",
        "architectures": cfg.get("architectures"),
        "expected": expected,
        "forbidden_keys": list(_FORBIDDEN),
    }
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote arch spec -> {out}")
    return 0


# ── inject-submission: fake challenger commit for offline flow tests ─────────


async def inject_submission(args) -> None:
    # Mirrors what chain.py writes for a fresh reveal, without touching the chain.
    tag = args.tag
    hotkey = f"5FlowTestChallenger{tag.zfill(30)}"
    if args.model_uri:
        # Real challenger (GPU tests): registry.hippius.com/<repo>@sha256:<digest>.
        base, _, digest = args.model_uri.partition("@")
        repo = base.removeprefix("registry.hippius.com/")
        if not digest.startswith("sha256:") or "/" not in repo:
            raise SystemExit("model-uri must look like registry.hippius.com/<repo>@sha256:<hex>")
        model_uri = args.model_uri
    else:
        # Fabricated repo (CPU mock flow test - never downloaded).
        repo = f"flowtest/challenger-{tag}"
        digest = "sha256:" + hashlib.sha256(f"flowtest-model-{tag}".encode()).hexdigest()
        model_uri = f"registry.hippius.com/{repo}@{digest}"
    payload_hash = hashlib.sha256(f"v7|{repo}|{digest}".encode()).hexdigest()

    conn = await asyncpg.connect(dsn=get_settings().db.dsn)
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    try:
        async with conn.transaction():
            miner_id = await conn.fetchval(
                """
                INSERT INTO miners (id, hotkey, coldkey, uid, netuid, updated_at)
                VALUES ($1, $2, $2, $3, 97, now())
                ON CONFLICT (hotkey) DO UPDATE SET updated_at = now() RETURNING id
                """,
                uuid4(),
                hotkey,
                100 + int(tag),
            )
            commit_id = uuid4()
            await conn.execute(
                """
                INSERT INTO chain_commits (id, netuid, block_number, block_hash, uid, hotkey,
                    commit_payload, model_uri, payload_hash)
                VALUES ($1, 97, 9000000, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (netuid, hotkey, payload_hash) DO NOTHING
                """,
                commit_id,
                f"0xflowtest{tag}",
                100 + int(tag),
                hotkey,
                {"version": "v7", "repo": repo, "digest": digest},
                model_uri,
                payload_hash,
            )
            submission_id = uuid4()
            await conn.execute(
                """
                INSERT INTO model_submissions (id, miner_id, chain_commit_id, netuid, uid, hotkey,
                    model_uri, commit_hash, model_hash, state, idempotency_key)
                VALUES ($1, $2, $3, 97, $4, $5, $6, $7, $7, 'SUBMITTED', $8)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                submission_id,
                miner_id,
                commit_id,
                100 + int(tag),
                hotkey,
                model_uri,
                digest,
                f"flowtest:97:{hotkey}:{payload_hash}",
            )
            await conn.execute(
                "UPDATE chain_commits SET submission_id = $1 WHERE id = $2",
                submission_id,
                commit_id,
            )
    finally:
        await conn.close()
    print(f"injected submission {submission_id} hotkey={hotkey}")


# ── register-host: connect any extra GPU worker beyond the two defaults ──────


async def register_host(args) -> None:
    # Upserts one remote_gpu_hosts row; the heartbeater and dispatchers take it from there.
    conn = await asyncpg.connect(dsn=get_settings().db.dsn)
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    try:
        await conn.execute(
            """
            INSERT INTO remote_gpu_hosts (id, role, base_url, tunnel_name, state, gpu_count,
                free_gpu_count, accelerator_type, capabilities, last_heartbeat_at, last_health)
            VALUES ($1, $2, $3, $4, $5, $6, $6, $7, '{}'::jsonb, now(), '{}'::jsonb)
            ON CONFLICT (id) DO UPDATE
            SET role = EXCLUDED.role, base_url = EXCLUDED.base_url,
                tunnel_name = EXCLUDED.tunnel_name, state = EXCLUDED.state,
                gpu_count = EXCLUDED.gpu_count, free_gpu_count = EXCLUDED.free_gpu_count,
                accelerator_type = EXCLUDED.accelerator_type, last_heartbeat_at = now()
            """,
            args.id,
            args.role,
            args.base_url,
            args.tunnel or None,
            "DRAINING" if args.drain else "READY",
            args.gpus,
            args.accelerator,
        )
    finally:
        await conn.close()
    state = "DRAINING" if args.drain else "READY"
    logger.info(f"[ops] host {args.id} ({args.role}) -> {args.base_url} state={state}")


def main() -> None:
    # ops.py {seed-genesis | preflight | arch-spec | inject-submission | register-host}.
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed-genesis", help="seed genesis king + register the two default GPU hosts")
    sub.add_parser("preflight", help="check every connection; exit 1 on failure")
    arch = sub.add_parser("arch-spec", help="regenerate architecture_spec.json")
    arch.add_argument("--out", default="")
    inj = sub.add_parser("inject-submission", help="insert a fake SUBMITTED challenger")
    inj.add_argument("--tag", default="1")
    inj.add_argument(
        "--model-uri",
        default="",
        help="real Hippius repo for GPU tests, e.g. registry.hippius.com/<repo>@sha256:<hex>",
    )
    reg = sub.add_parser("register-host", help="connect an extra/replacement GPU worker")
    reg.add_argument("--id", required=True, help="unique host id, e.g. gpu-eval-b200-2")
    reg.add_argument("--role", required=True, choices=["EVAL", "PRE_EVAL"])
    reg.add_argument(
        "--base-url", required=True, help="tunnel-local URL, e.g. http://127.0.0.1:18091"
    )
    reg.add_argument("--gpus", type=int, default=8, help="gpu count (EVAL needs >=8)")
    reg.add_argument("--accelerator", default="", help="e.g. B200, RTX 5090")
    reg.add_argument("--tunnel", default="", help="pm2 tunnel app name (informational)")
    reg.add_argument("--drain", action="store_true", help="register as DRAINING (no new work)")
    args = parser.parse_args()
    if args.cmd == "seed-genesis":
        asyncio.run(seed_genesis())
    elif args.cmd == "preflight":
        sys.exit(asyncio.run(preflight()))
    elif args.cmd == "arch-spec":
        sys.exit(arch_spec(args))
    elif args.cmd == "register-host":
        asyncio.run(register_host(args))
    else:
        asyncio.run(inject_submission(args))


if __name__ == "__main__":
    main()
