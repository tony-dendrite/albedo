"""albedo.validator.reset — ALBEDO_RESET one-shot competition reset.

Gated behind the ALBEDO_RESET env flag (default off, set in ecosystem.config.js). When
triggered on validator startup it wipes the 4B model / dedup / dashboard / seen state,
reseeds the genesis king, and rebuilds the eval queue from on-chain *3-part v4* commits
in chronological (commit-block) order — oldest commit first (eval-000001).

A reset_marker.json guard makes the flag safe against an accidental re-wipe on a
crash-restart: with ALBEDO_RESET=1 the reset runs once; to deliberately re-run, use
ALBEDO_RESET=force.
"""
from __future__ import annotations

import logging
import time

from albedo.config import SEED_DIGEST, SEED_REPO
from albedo.validator.chain import _decode_raw

log = logging.getLogger(__name__)

_RESET_MARKER      = "reset_marker.json"
_OLD_COMPLETED_KEY = "state/completed_repos.json"   # 1.7B-era key, still present in R2


def _commitment_blocks(subtensor, netuid: int) -> dict[str, int]:
    """hotkey -> commit block, from the Commitments.CommitmentOf storage map."""
    out: dict[str, int] = {}
    try:
        for k, v in subtensor.query_map("Commitments", "CommitmentOf", [netuid]):
            hk  = getattr(k, "value", k)
            val = getattr(v, "value", v)
            blk = val.get("block") if isinstance(val, dict) else None
            if blk is not None:
                out[str(hk)] = int(blk)
    except Exception as exc:
        log.warning("reset: CommitmentOf query failed (%s) — blocks default to 0", exc)
    return out


def _recover_excluded(state) -> tuple[set[str], set[str]]:
    """1.7B exclusion set (repo and repo@digest) recovered from the old R2 key."""
    excl_repo: set[str] = set()
    excl_full: set[str] = set()
    try:
        old = state._store.get(_OLD_COMPLETED_KEY) or {}
        for entry in old.get("repos", []):
            excl_full.add(entry)
            excl_repo.add(entry.split("@", 1)[0])
        log.info("reset: loaded %d 1.7B-era repos for exclusion", len(excl_full))
    except Exception as exc:
        log.warning("reset: could not load 1.7B exclusion set (%s) — none excluded", exc)
    return excl_repo, excl_full


def _scan_v4(subtensor, netuid, excl_repo, excl_full):
    """Return ([(block, hotkey, repo, digest)] oldest-first, set(all on-chain hotkeys))."""
    raw    = subtensor.get_all_commitments(netuid=netuid)
    blocks = _commitment_blocks(subtensor, netuid)
    rows: list[tuple[int, str, str, str]] = []
    all_hotkeys: set[str] = set()

    items = raw.items() if isinstance(raw, dict) else []
    for hk, val in items:
        hk = str(hk)
        all_hotkeys.add(hk)
        data = _decode_raw(val)
        if not data.startswith("v4|"):
            continue
        parts = data.split("|")
        if len(parts) != 3:           # 3-part only — drop 4-part and malformed
            continue
        _, repo, digest = parts
        if repo in excl_repo or f"{repo}@{digest}" in excl_full:
            log.info("reset: excluding 1.7B-evaluated repo %s", repo)
            continue
        rows.append((blocks.get(hk, 0), hk, repo, digest))

    rows.sort(key=lambda r: r[0])     # oldest commit block first
    return rows, all_hotkeys


def _clear_eval_fingerprints(eval_url: str | None) -> None:
    """Tell the eval server to drop its near-duplicate fingerprint state so the replay
    re-fingerprints from scratch (else every re-queued model false-matches itself)."""
    if not eval_url:
        log.warning("reset: no eval_url — skipping eval-server fingerprint clear")
        return
    try:
        import httpx
        r = httpx.post(eval_url.rstrip("/") + "/reset_fingerprints", timeout=60.0)
        r.raise_for_status()
        log.warning("reset: cleared eval-server fingerprints: %s", r.json())
    except Exception as exc:
        log.warning("reset: could not clear eval-server fingerprints (%s)", exc)


def run_reset(state, subtensor, *, netuid: int, eval_url: str | None = None, force: bool = False) -> int:
    """Clear state + dashboard + eval-server dup state, reseed genesis, rebuild v4 queue."""
    try:
        marker = state._store.get(_RESET_MARKER)
    except Exception:
        marker = None
    if marker and not force:
        log.warning(
            "reset: marker present (%s) — SKIPPING. Set ALBEDO_RESET=0, or ALBEDO_RESET=force to re-run.",
            marker,
        )
        return 0

    excl_repo, excl_full = _recover_excluded(state)
    rows, all_hotkeys = _scan_v4(subtensor, netuid, excl_repo, excl_full)
    log.warning("reset: clearing state and queueing %d v4 commits (oldest first)", len(rows))

    # Clear the eval server's duplicate/fingerprint state so the oldest-first replay
    # re-fingerprints cleanly (first commit fingerprinted first → later dups flagged).
    _clear_eval_fingerprints(eval_url)

    # 1. Clear all model / dedup / seen / dashboard state.
    state.king            = None
    state.king_chain      = []
    state.history         = []
    state.completed_repos = set()
    state.queue           = []
    state.counter         = 0
    state.retry_counts    = {}
    state.recovered_ids   = set()
    state.seen            = set()
    state.current_eval    = None
    state.stats           = {"queued": 0, "accepted": 0, "rejected": 0, "failed": 0}

    # 2. Reseed genesis king (so duels have an opponent).
    try:
        state.set_king(hotkey="", model_repo=SEED_REPO, model_digest=SEED_DIGEST,
                       block=0, challenge_id="genesis", dethrone_judges=[], crown_judges=[])
    except Exception as exc:
        log.error("reset: genesis reseed failed: %s", exc)

    # 3. Rebuild the queue, oldest commit first → eval-000001..
    for blk, hk, repo, digest in rows:
        eval_id = state.enqueue(
            {"hotkey": hk, "model_repo": repo, "model_digest": digest, "block": blk},
            force=True,
        )
        log.info("reset: queued %s  block=%d  %s", eval_id, blk, repo)

    # 4. seen = all current on-chain hotkeys so the live scan won't re-add/reorder them;
    #    legacy/excluded ones stay out of evaluation, future new miners still get appended.
    state.seen = set(all_hotkeys)

    # 5. Persist + write a fresh dashboard.
    state.flush()
    state.flush_dashboard(force=True)

    # 6. Marker guard against accidental re-wipe on crash-restart.
    try:
        state._store.put(_RESET_MARKER, {
            "queued": len(rows), "genesis_digest": SEED_DIGEST, "ts": time.time(),
        })
    except Exception as exc:
        log.warning("reset: marker write failed: %s", exc)

    log.warning(
        "reset: COMPLETE — king=genesis, queued=%d. Set ALBEDO_RESET=0 and restart to avoid re-reset.",
        len(rows),
    )
    return 0
