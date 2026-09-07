"""What both question writers need: the shape of a sample, and the checks applied to any of them.

Anything only one writer uses lives with that writer - the parse-and-cap hygiene the behaviour
questions go through is in `behavior/questions.py`, and the milestone questions have their own in
`reference/prompt_ladder.py`.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ...shared.edit_detection import (
    edited_in_turn,
    trajectory_made_edit,  # noqa: F401  re-exported for judge_api and the tests
)

# shared only in the sense that more than one module reads them: the fence stripper is how
# `prompt_milestones` separates an agent's prose from the blocks it wrote, and the candidate-block
# pattern is how `judge_core` finds the scored turns inside a rendered document
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_CANDIDATE_BLOCK_RE = re.compile(r"CANDIDATE OUTPUT(?: \d+)?:\n------\n(.*?)\n------", re.DOTALL)


def sample_phase(messages: list[dict[str, str]] | None) -> str:
    """Trim phase, inferred from the prefix the sampler produced: the cut lands near the start
    (cold), just before the first edit (pre_edit), or just after it (at_edit)."""
    turns = [m.get("content") or "" for m in messages or [] if m.get("role") == "assistant"]
    if any(_edited_in_turn(t) for t in turns):
        return "at_edit"
    return "cold" if len(turns) <= 2 else "pre_edit"


HORIZON_STRATA = (12, 16)


def assign_horizons(samples) -> dict[str, int]:
    by_phase: dict[str, list[str]] = {}
    for sample in samples:
        by_phase.setdefault(sample_phase(sample.messages), []).append(sample.sample_id)
    horizons: dict[str, int] = {}
    for ids in by_phase.values():
        for index, sample_id in enumerate(sorted(ids)):
            horizons[sample_id] = HORIZON_STRATA[index % len(HORIZON_STRATA)]
    return horizons


_edited_in_turn = edited_in_turn

_UNFOLDED_AVOID_RE = re.compile(
    r"^\s*(?:does[^?]{0,60}\bavoid|is[^?]{0,60}\bfree of|does[^?]{0,60}\brefrain)", re.IGNORECASE
)
_ACTION_VERB_RE = re.compile(
    r"\b(edit|modif|submit|propagat|verif|appl|patch|chang|fix)\w*\b", re.IGNORECASE
)


def _discard_recorder(
    discards: list[dict[str, str]] | None,
    *,
    stage: str,
    origin: str = "content",
    counters: dict[str, int] | None = None,
) -> Callable[..., None]:
    def _drop(reason: str, text: str, detail: str = "") -> None:
        if counters is not None:
            counters[reason] = counters.get(reason, 0) + 1
        if discards is None:
            return
        entry = {"stage": stage, "reason": reason, "text": text, "origin": origin}
        if detail:
            entry["detail"] = detail
        discards.append(entry)

    return _drop


def enforce_question_labels(
    questions: list[dict[str, str]],
    *,
    discards: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Drop questions phrased so that doing nothing earns them, then number what survives."""
    kept: list[dict[str, str]] = []
    drops = {"unfolded_avoid": 0}
    _drop = _discard_recorder(discards, stage="enforce_question_labels", counters=drops)

    for question in questions:
        text = question.get("text", "")
        if _UNFOLDED_AVOID_RE.search(text) and not _ACTION_VERB_RE.search(text):
            _drop("unfolded_avoid", text)
            continue
        kept.append(question)
    for position, question in enumerate(kept, start=1):
        question["id"] = f"q_{position:02d}"
    return kept, drops
