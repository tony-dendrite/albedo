"""albedo.validator.chain — Bittensor chain I/O for reveal commitments."""
from __future__ import annotations

import logging
from typing import Any

from albedo.models import ModelRef, parse_reveal_v4

log = logging.getLogger(__name__)


def _decode_raw(raw: Any) -> str:
    """Normalise a raw commitment value to a UTF-8 string."""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        if raw.startswith("0x"):
            try:
                return bytes.fromhex(raw[2:]).decode("utf-8", errors="replace")
            except Exception:
                return raw
        return raw
    return str(raw)


def _iter_commitments(raw: Any):
    """Yield (chain_hotkey, reveal_block, data_str) for every commitment.

    Handles two formats returned by different bittensor SDK versions:
    - dict[hotkey → data_str]  (bittensor ≥10.x)
    - list of (hotkey, [(block, data), ...]) pairs  (older SDKs)
    """
    if isinstance(raw, dict):
        for hotkey, value in raw.items():
            yield str(hotkey), None, _decode_raw(value)
        return

    for pair in raw:
        try:
            hotkey = str(pair[0])
            entries = [(int(item[0]), _decode_raw(item[1])) for item in pair[1]]
            if not entries:
                continue
            block, data = sorted(entries, key=lambda t: t[0], reverse=True)[0]
            yield hotkey, block, data
        except Exception as exc:
            log.debug("chain scan: failed to decode pair: %s", exc)


def scan_reveals(
    subtensor: Any,
    netuid: int,
    completed_repos: set[str],
    seen: set[str],
    *,
    king_hotkeys: set[str] = frozenset(),
    rejected_out: list[dict] | None = None,
) -> list[dict]:
    """Query on-chain commitments and return new v4 entries.

    Drops non-v4 formats, non-sha256 digests, already-seen hotkeys, already-completed
    repos, and the current king/king-chain hotkeys (king_hotkeys). Spoofed reveals
    (payload hotkey != chain hotkey) are appended to rejected_out.

    Accepts both 4-part reveals (v4|repo|digest|hotkey) and 3-part reveals
    (v4|repo|digest) — in the 3-part case the chain hotkey is treated as the author.
    """
    results: list[dict] = []
    n_total = n_seen = n_king = n_completed = n_non_v4 = n_invalid = n_spoofed = 0

    log.info("chain scan: querying on-chain commitments for netuid=%d", netuid)

    try:
        raw_commitments = subtensor.get_all_commitments(netuid=netuid)
    except Exception as exc:
        log.warning("chain scan: get_all_commitments failed: %s", exc)
        return results

    for chain_hotkey, reveal_block, data in _iter_commitments(raw_commitments):
        n_total += 1

        if chain_hotkey in seen:
            log.debug("chain scan: skip (already seen) hotkey=%s", chain_hotkey)
            n_seen += 1
            continue

        if chain_hotkey in king_hotkeys:
            log.debug("chain scan: skip (king/chain) hotkey=%s", chain_hotkey)
            n_king += 1
            continue

        if not data.startswith("v4|"):
            log.debug("chain scan: skip (non-v4) hotkey=%s", chain_hotkey)
            n_non_v4 += 1
            continue

        parts = data.split("|")
        try:
            if len(parts) == 4:
                # Full format: v4|repo|digest|author_hotkey
                ref, author_hotkey = parse_reveal_v4(data)
                if author_hotkey != chain_hotkey:
                    log.warning(
                        "chain scan: SPOOFED reveal — chain=%s author=%s repo=%s",
                        chain_hotkey, author_hotkey, ref.repo,
                    )
                    if rejected_out is not None:
                        rejected_out.append({
                            "hotkey":        chain_hotkey,
                            "author_hotkey": author_hotkey,
                            "block":         reveal_block,
                            "model_repo":    ref.repo,
                            "model_digest":  ref.digest,
                        })
                    n_spoofed += 1
                    continue
            elif len(parts) == 3:
                # Short format: v4|repo|digest — chain enforces hotkey, no spoof possible
                _, repo, digest = parts
                ref = ModelRef(repo=repo, digest=digest)
                author_hotkey = chain_hotkey
            else:
                raise ValueError(f"unexpected part count {len(parts)}")
        except ValueError as exc:
            log.debug("chain scan: skip (parse error) hotkey=%s: %s", chain_hotkey, exc)
            n_invalid += 1
            continue

        if ref.repo in completed_repos:
            log.debug("chain scan: skip (already evaluated) repo=%s hotkey=%s", ref.repo, chain_hotkey)
            n_completed += 1
            continue

        entry: dict = {
            "hotkey":       chain_hotkey,
            "block":        reveal_block,
            "model_repo":   ref.repo,
            "model_digest": ref.digest,
        }
        results.append(entry)
        log.info(
            "chain scan: NEW COMMIT — hotkey=%s  repo=%s  block=%s",
            chain_hotkey, ref.repo, reveal_block,
        )

    log.info(
        "chain scan: done — total=%d  new=%d  seen=%d  king=%d  completed=%d  non_v4=%d  spoofed=%d  invalid=%d",
        n_total, len(results), n_seen, n_king, n_completed, n_non_v4, n_spoofed, n_invalid,
    )
    return results
