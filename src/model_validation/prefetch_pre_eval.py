from __future__ import annotations

import asyncio
import json
from pathlib import Path

from loguru import logger as log

from albedo_config import get_model_validation_settings
from model_validation import db
from model_validation.storage import cache_dir, download_full, make_ref, make_room

POLL_SECONDS = 60


def model_present(dest: Path) -> bool:
    """Mirror of the sanity worker's completeness check: every committed shard on disk."""
    if not dest.is_dir() or not any(dest.glob("*.safetensors")):
        return False
    index = dest / "model.safetensors.index.json"
    if not index.exists():
        return (dest / "model.safetensors").exists()
    try:
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
    except (ValueError, OSError):
        return False
    return bool(shards) and all((dest / shard).exists() for shard in shards)


async def prefetch_once(pool) -> str:
    row = await pool.fetchrow(
        """
        SELECT model_uri FROM model_submissions
        WHERE state IN ('HIPPIUS_VALIDATED', 'PRE_EVAL_RETRYABLE')
        ORDER BY priority ASC, created_at ASC
        LIMIT 1
        """
    )
    if row is None:
        return "queue empty"
    model_uri = str(row["model_uri"])
    repo, _, digest = model_uri.partition("@")
    try:
        ref = make_ref(repo, digest)
    except ValueError:
        return f"malformed ref {model_uri}"
    if model_present(cache_dir(ref)):
        return f"warm {model_uri}"
    protected = await db.protected_pre_eval_repos(pool)
    protected.add(repo)
    make_room(ref, protected)
    log.info("prefetching {}", model_uri)
    await asyncio.to_thread(download_full, ref)
    return f"prefetched {model_uri}"


async def run() -> None:
    config = get_model_validation_settings()
    pool = await db.connect(config.DB_URL)
    log.info("pre-eval prefetcher started")
    last = ""
    while True:
        try:
            outcome = await prefetch_once(pool)
            if outcome != last:
                log.info("prefetch: {}", outcome)
                last = outcome
        except Exception as exc:
            log.warning("prefetch pass failed: {}", exc)
            last = ""
        await asyncio.sleep(POLL_SECONDS)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
