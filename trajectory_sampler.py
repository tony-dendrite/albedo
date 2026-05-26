"""Deterministic trajectory sampling from a pinned SWE-ZERO parquet shard.

The seed argument fully pins which rows and which turns within each row
will be evaluated. Validators across the network running on the same
`(block_hash, challenger_hotkey)` will pick the exact same fixture, so a
miner cannot pre-game which (instance_id, turn_idx) pairs they'll face.

The pinned parquet shard is named in `chain.toml [dataset].shard` and
prefetched to local disk by `scripts/prefetch_dataset.py`. We refuse to
sample if its sha256 doesn't match `chain.toml [dataset].shard_sha256`.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import chain_config

log = logging.getLogger("albedo.sampler")


@dataclass(frozen=True)
class Sample:
    """One (trajectory prefix, turn-to-evaluate) pair.

    `messages_prefix` is what gets sent to BOTH contestant models. The
    original assistant turn at `turn_idx` is held aside as
    `original_reply` so we can show diffs in the dashboard if we ever want
    to (judging itself only sees the candidate replies, not the original).
    """
    instance_id: str
    repo: str
    sample_idx: int       # row index inside the shard, deterministic
    turn_idx: int         # index of the assistant message in `messages`
    messages_prefix: list[dict]
    original_reply: str


def _verify_shard(shard_path: str | Path) -> Path:
    p = Path(shard_path)
    if not p.exists():
        raise FileNotFoundError(
            f"dataset shard not found at {p}; run scripts/prefetch_dataset.py first"
        )
    pinned = (chain_config.DATASET_SHARD_SHA256 or "").strip().lower()
    if not pinned or pinned == "0" * 64:
        log.warning("chain.toml [dataset].shard_sha256 is unset; skipping integrity check")
        return p

    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    got = h.hexdigest()
    if got != pinned:
        raise RuntimeError(
            f"dataset shard sha256 mismatch:\n  pinned={pinned}\n  on-disk={got}\n"
            f"shard={p}\nrefetch with scripts/prefetch_dataset.py."
        )
    return p


_LOADED_TABLE: pq.Table | None = None
_LOADED_PATH: Path | None = None


def load_shard(shard_path: str | Path) -> pq.Table:
    """Memory-map the parquet shard (idempotent per process)."""
    global _LOADED_TABLE, _LOADED_PATH
    p = _verify_shard(shard_path)
    if _LOADED_TABLE is not None and _LOADED_PATH == p:
        return _LOADED_TABLE
    log.info("loading parquet shard %s", p)
    table = pq.read_table(p, memory_map=True)
    _LOADED_TABLE = table
    _LOADED_PATH = p
    log.info("loaded shard: %d rows", table.num_rows)
    return table


def _row(table: pq.Table, idx: int) -> dict:
    """Read a single row as a Python dict. Parquet → arrow → Python is
    relatively cheap because we only pull the fields we need."""
    row = {col: table.column(col)[idx].as_py() for col in table.column_names}
    return row


def _seed_to_rng(seed: bytes) -> np.random.Generator:
    """blake2b(seed) → 32-byte entropy → PCG64DXSM. blake2b mixes inputs
    well enough that adjacent block hashes produce well-separated RNG
    streams."""
    digest = hashlib.blake2b(seed, digest_size=32).digest()
    entropy = np.frombuffer(digest, dtype=np.uint64).tolist()
    seq = np.random.SeedSequence(entropy=entropy)
    return np.random.Generator(np.random.PCG64DXSM(seq))


def _assistant_turn_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def sample(
    seed: bytes,
    *,
    n_samples: int | None = None,
    max_turns_per_sample: int | None = None,
    shard_path: str | Path,
) -> list[Sample]:
    """Pick `n_samples` rows, then up to `max_turns_per_sample` assistant
    turns from each. Total samples returned = sum-of-per-row picks (it can
    be < n_samples * max_turns if rows are short).

    Deterministic in `seed`. Same seed + same shard => same Sample list.
    """
    n_samples = n_samples if n_samples is not None else chain_config.DUEL_N_SAMPLES
    max_turns = (
        max_turns_per_sample
        if max_turns_per_sample is not None
        else chain_config.DUEL_MAX_TURNS_PER_SAMPLE
    )

    table = load_shard(shard_path)
    total_rows = table.num_rows
    rng = _seed_to_rng(seed)

    # We want n distinct rows with >=2 assistant turns. Over-sample by 4x
    # and skip degenerate ones; if the shard is somehow under-supplied,
    # truncate at whatever we got.
    candidates = rng.choice(
        total_rows, size=min(total_rows, max(n_samples * 4, n_samples + 16)),
        replace=False,
    )
    picked: list[Sample] = []
    for cidx in candidates:
        if len(picked) >= n_samples:
            break
        row = _row(table, int(cidx))
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            continue
        asst = _assistant_turn_indices(messages)
        if len(asst) < 1:
            continue
        k = min(len(asst), max_turns)
        turn_idxs = rng.choice(asst, size=k, replace=False).tolist()
        turn_idxs.sort()
        for ti in turn_idxs:
            picked.append(
                Sample(
                    instance_id=str(row.get("instance_id") or ""),
                    repo=str(row.get("repo") or ""),
                    sample_idx=int(cidx),
                    turn_idx=int(ti),
                    messages_prefix=messages[:ti],
                    original_reply=str(messages[ti].get("content") or ""),
                )
            )

    if not picked:
        raise RuntimeError(
            f"sampler produced 0 turns from shard rows={total_rows} "
            f"(n_samples={n_samples}, max_turns={max_turns}). "
            "Is the shard the expected schema (messages: list[{role,content}])?"
        )
    log.info(
        "sampled %d (sample, turn) pairs from %d rows (seed=%s)",
        len(picked),
        n_samples,
        hashlib.blake2b(seed, digest_size=8).hexdigest(),
    )
    return picked
