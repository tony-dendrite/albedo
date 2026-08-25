#!/usr/bin/env python3
"""Bootstrap the dedup bank from a JSON manifest (+ optionally every accepted submission in the DB).

    uv run python scripts/dedup_seed_bank.py --manifest scripts/dedup_bank_manifest.example.json
    uv run python scripts/dedup_seed_bank.py --manifest bank.json --from-db
    uv run python scripts/dedup_seed_bank.py --manifest bank.json --dry-run

Manifest:
    {
      "root": true,                                   # index the chain.toml seed as the bank root
      "king_namespace": "dendriteholdings",           # optional; default below
      "kings": ["CIV", {"roman": "CXV", "revision": "<sha>", "hotkey": "...", "coldkey": "..."}],
      "models": [{"url": "https://huggingface.co/<ns>/<repo>[/tree/<sha>]",
                  "hotkey": "", "coldkey": ""},
                 {"repo": "<ns>/<repo>", "revision": "<sha>"}]
    }
Kings map to the dendrite HF mirror  <king_namespace>/<king_prefix>-<ROMAN>.  A missing revision
is pinned to the repo's current head at run time (logged). Downloads go through
model_validation.storage (the same path the worker uses). Models are processed one at a time:
download -> fingerprint against the seed -> store with the verdict the live gate would give (bank
on PASS, audit on REJECT) -> delete the download -> next (pass --keep to keep downloads).
Order: root -> DB accepted submissions (oldest block first) -> manifest kings -> manifest models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger as log  # noqa: E402

from albedo_config import get_model_validation_settings  # noqa: E402
from albedo_config.chain_spec import SEED_DIGEST, SEED_REPO  # noqa: E402
from model_validation import db, dedup  # noqa: E402
from model_validation.dedup import bank  # noqa: E402
from model_validation.dedup.gate import device, ref_dir  # noqa: E402
from model_validation.dedup.secret import load_secret  # noqa: E402
from model_validation.dedup.sketch import fingerprint  # noqa: E402
from model_validation.storage import cache_dir, download_full, make_ref  # noqa: E402

config = get_model_validation_settings()

KING_NAMESPACE = os.environ.get("ALBEDO_DEDUP_KING_NAMESPACE", "dendriteholdings")
KING_PREFIX = os.environ.get("ALBEDO_DEDUP_KING_PREFIX", "albedo-qwen3.6-35b-king")
_HF_URL = re.compile(r"^https?://huggingface\.co/([^/\s]+)/([^/\s]+)(?:/tree/([0-9a-f]{40}))?/?$")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def king_repo(roman: str, namespace: str = KING_NAMESPACE, prefix: str = KING_PREFIX) -> str:
    return f"{namespace}/{prefix}-{roman.strip().upper()}"


def parse_entry(entry: dict | str) -> dict:
    if isinstance(entry, str):
        entry = {"url": entry} if entry.startswith("http") else {"repo": entry}
    repo, rev = entry.get("repo", ""), entry.get("revision", "")
    if url := entry.get("url", ""):
        m = _HF_URL.match(url.strip())
        if not m:
            raise ValueError(f"not a HF model url: {url}")
        repo = f"{m.group(1)}/{m.group(2)}"
        rev = rev or (m.group(3) or "")
    if "@" in repo:
        repo, _, rev = repo.partition("@")
    if not repo or repo.count("/") != 1:
        raise ValueError(f"bad repo in manifest entry: {entry}")
    return dict(
        repo=repo, revision=rev, hotkey=entry.get("hotkey", ""), coldkey=entry.get("coldkey", "")
    )


def parse_king(
    entry: dict | str, namespace: str = KING_NAMESPACE, prefix: str = KING_PREFIX
) -> dict:
    if isinstance(entry, str):
        entry = {"roman": entry}
    return dict(
        repo=king_repo(entry["roman"], namespace, prefix),
        revision=entry.get("revision", ""),
        hotkey=entry.get("hotkey", ""),
        coldkey=entry.get("coldkey", ""),
    )


def resolve_revision(repo: str, rev: str) -> str:
    if rev:
        if not _SHA.match(rev):
            raise ValueError(f"{repo}: revision must be a 40-hex commit sha, got {rev!r}")
        return rev
    from huggingface_hub import HfApi

    sha = HfApi().model_info(repo).sha
    log.info("{}: pinned to current head {}", repo, sha)
    return sha


def index_root() -> None:
    uri = f"{SEED_REPO}@{SEED_DIGEST}"
    if bank.has_doc(uri):
        log.info("root already indexed: {}", uri)
        return
    secret = load_secret(config.DEDUP_SECRET, config.DEDUP_SECRET_FILE)
    rd = ref_dir()
    doc = fingerprint(rd, rd, secret, device(), model_uri=uri)
    bank.put_doc(doc, status=bank.STATUS_BANK, repo=SEED_REPO, digest=SEED_DIGEST, is_root=True)
    log.info("root indexed: {} ({} tensors, {}s)", uri, doc["n_tensors"], doc["secs"])


def index_model(item: dict, keep: bool) -> str:
    repo, rev = item["repo"], item["revision"]
    model_uri = f"{repo}@{rev}"
    if bank.has_doc(model_uri):
        log.info("{} — already indexed, skipping", model_uri)
        return "skipped"
    ref = make_ref(repo, rev)
    try:
        model_dir = download_full(ref)
        res = dedup.run(model_dir, model_uri, item["hotkey"], repo, rev, item["coldkey"])
    finally:
        if not keep:
            shutil.rmtree(cache_dir(ref), ignore_errors=True)
    if res.infra_error:
        log.error("{} — infra: {}", model_uri, res.infra_error)
        return "error"
    v = res.verdict
    log.info("{} — {} {} {}", model_uri, v.status, v.reason or "", v.message)
    return v.status


async def accepted_from_db() -> list[dict]:
    pool = await db.connect(config.DB_URL)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ms.model_uri, ms.hotkey, m.coldkey
                FROM model_submissions ms
                JOIN chain_commits cc ON cc.id = ms.chain_commit_id
                LEFT JOIN miners m ON m.hotkey = ms.hotkey
                WHERE ms.netuid = $1 AND ms.state = ANY($2::text[])
                ORDER BY cc.block_number ASC, ms.created_at ASC
                """,
                config.NETUID,
                list(db._VALIDATED_OR_BEYOND),
            )
    finally:
        await pool.close()
    out = []
    for r in rows:
        repo, _, rev = r["model_uri"].partition("@")
        out.append(dict(repo=repo, revision=rev, hotkey=r["hotkey"], coldkey=r["coldkey"] or ""))
    return out


def load_manifest(path: str) -> tuple[bool, list[dict], list[dict]]:
    m = json.loads(Path(path).read_text())
    ns = m.get("king_namespace", KING_NAMESPACE)
    prefix = m.get("king_prefix", KING_PREFIX)
    kings = [parse_king(k, ns, prefix) for k in m.get("kings", [])]
    models = [parse_entry(e) for e in m.get("models", [])]
    return bool(m.get("root", True)), kings, models


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True, help="JSON manifest (see module docstring)")
    ap.add_argument("--from-db", action="store_true", help="also index every accepted submission")
    ap.add_argument(
        "--keep", action="store_true", help="keep downloads (default: delete after indexing)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="resolve and print the plan, download nothing"
    )
    args = ap.parse_args()

    root, kings, models = load_manifest(args.manifest)
    todo: list[dict] = []
    if args.from_db:
        todo += asyncio.run(accepted_from_db())
    todo += kings + models
    for item in todo:
        if not item["revision"]:
            item["revision"] = resolve_revision(item["repo"], "")

    if args.dry_run:
        print(f"root: {root} ({SEED_REPO}@{SEED_DIGEST})")
        for item in todo:
            print(
                f"{item['repo']}@{item['revision']}  "
                f"hotkey={item['hotkey'] or '-'} coldkey={item['coldkey'] or '-'}"
            )
        print(f"{len(todo)} models")
        return

    bank.ensure_index()
    if root:
        index_root()
    tally: dict[str, int] = {}
    for item in todo:
        try:
            status = index_model(item, args.keep)
        except Exception as exc:
            log.error("{}@{} — failed: {}", item["repo"], item["revision"], exc)
            status = "error"
        tally[status] = tally.get(status, 0) + 1
    log.info("done: {}", tally)
    print("DEDUP_SEED_BANK_DONE", tally)


if __name__ == "__main__":
    main()
