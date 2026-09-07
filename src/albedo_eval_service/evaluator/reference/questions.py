from __future__ import annotations

import re
import unicodedata
from typing import Any


def format_reference_trajectory(turns: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    step = 0
    for turn in turns:
        if turn.get("score_target"):
            step += 1
            parts.append(f"REFERENCE STEP {step}:\n{turn['content']}")
        elif turn.get("environment_observation"):
            parts.append(f"ENVIRONMENT OBSERVATION:\n{turn['content']}")
    return "\n\n".join(parts)


# Wording that tells the judge material it must not have: that a reference exists, that the
# question came out of a milestone vector, or that other agents worked this task. The judge sees
# one candidate and one question, and a question that mentions any of this invites it to score the
# candidate against something it cannot see.
_LEAK_RE = re.compile(
    r"\bthe reference\b|\bmilestones?\b|\bvector of change\b|\bthe (?:other|strong) agents?\b"
    r"|\bthe (?:runs|accounts)\b",
    re.IGNORECASE,
)


def filter_reference_leaks(
    questions: list[dict[str, str]], *, discards: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    kept = []
    for q in questions:
        if _LEAK_RE.search(q["text"]):
            if discards is not None:
                discards.append(
                    {
                        "stage": "filter_reference_leaks",
                        "reason": "reference_leak",
                        "text": q["text"],
                        "origin": "content",
                    }
                )
        else:
            kept.append(q)
    return kept


_SMART = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_SPAN_TRIM = re.compile(r"^[\s'\"`]+|[\s'\"`]*(?:\.{3}|…)?[\s'\"`]*$")
# The prefix `cat -n`, `nl -ba`, `grep -n` and `grep -rn` put in front of a line. A model quoting
# code it read that way leaves the numbering out, which is correct of it, and its faithfully copied
# span then no longer occurs in the observation it came from. Narrow on purpose: the
# digits-then-single-space form is left alone because ordinary prose such as "1274 characters
# elided" matches it.
_LINE_PREFIX = re.compile(r"^\s*\d{1,6}[\t:]|^[\w./-]+\.\w+:\d{1,6}:\s?", re.M)


def normalise_span(text: str) -> str:
    """Fold the differences a model introduces when copying, and nothing else.

    Whitespace runs collapse because the reference is rendered with indentation and models
    reflow what they quote; smart quotes and dashes revert because models typographically
    "improve" it. Line-number prefixes go for the same reason: a model that read a file through
    `nl -ba` quotes the code and not the numbering. Case is preserved: code is case-sensitive, and
    folding it would let a near-miss match.

    The prefix is stripped BEFORE the whitespace collapse and cannot be moved after it: once
    `     6\ttype X` has become `6 type X` the line boundaries the anchors need are gone, and no
    regex can then tell a line number from a number in the code. Both the span and the text it is
    searched in pass through here, so the fold applies to both sides and the comparison stays
    symmetric.
    """
    folded = _LINE_PREFIX.sub("", unicodedata.normalize("NFKC", text).translate(_SMART))
    return " ".join(folded.split())
