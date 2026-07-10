"""Hotkey-swap detection.

``swap_hotkey`` moves a hotkey to a uid WITHOUT re-registering, so
``BlockAtRegistration[netuid, uid]`` keeps the previous occupant's value. A normal
dereg + re-registration (even landing on the same uid) bumps it to the registration
block. Rule per uid, comparing the last persisted (hotkey, registration_block)
against the live metagraph:

    hotkey changed + registration block unchanged -> swap (flag)
    hotkey changed + registration block bumped    -> re-registration, normal churn
    hotkey unchanged                              -> ok

Empirically confirmed on SN97 (2026-07-10): 16 swaps, every old hotkey spent/blocked
first — swap_hotkey is used as a free identity reset that keeps uid/stake.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwapEvent:
    uid: int
    old_hotkey: str
    new_hotkey: str
    registration_block: int

    def detail(self) -> dict:
        return {
            "uid": self.uid,
            "old_hotkey": self.old_hotkey,
            "new_hotkey": self.new_hotkey,
            "registration_block": self.registration_block,
        }


def find_swaps(
    prior: dict[int, tuple[str, int | None]],
    current: list[tuple[int, str, int]],
) -> list[SwapEvent]:
    """``prior``: uid -> (hotkey, registration_block) as last persisted in ``miners``.
    ``current``: (uid, hotkey, registration_block) from the live metagraph.

    A uid with no prior state, or whose prior registration_block was never recorded
    (NULL — pre-backfill), cannot be judged and is skipped.
    """
    swaps: list[SwapEvent] = []
    for uid, hotkey, reg_block in current:
        prev = prior.get(uid)
        if prev is None:
            continue
        prev_hotkey, prev_reg_block = prev
        if prev_hotkey == hotkey or prev_reg_block is None:
            continue
        if reg_block == prev_reg_block:
            swaps.append(SwapEvent(uid, prev_hotkey, hotkey, reg_block))
    return swaps


def describe(detail: dict) -> str:
    """One-line report used for the dashboard fault_message and detection events."""
    return (
        f"hotkey swap detected: uid {detail['uid']} "
        f"BEFORE hotkey={detail['old_hotkey']} BlockAtRegistration={detail['registration_block']} | "
        f"AFTER hotkey={detail['new_hotkey']} BlockAtRegistration={detail['registration_block']} "
        f"<- identical, no re-registration -> swap_hotkey"
    )
