from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

from ..shared.questions import _FENCE_RE, is_measurement_bound_question
from .prompt_reference_split import (
    REFERENCE_SCORED_WINDOW_BLOCK,
    REFERENCE_SKELETON,
    SPECIALIST_BLOCK,
    ReferenceSpecialist,
)

_WORKFLOW_HEAD_RE = re.compile(
    r"(## Recommended Workflow|<PROBLEM_SOLVING_WORKFLOW>|Follow these steps to resolve the issue:"
    r"|Phase 1\. READING)",
    re.IGNORECASE,
)

_OBSERVATION_SUCCESS_MARKERS = {
    "returncode": "<returncode>N</returncode> around <output>; failure = non-zero returncode or "
    "error text in the output",
    "swe_agent": "plain OBSERVATION text; failure is only visible as error text in it",
    "openhands": "[Command finished with exit code N] trailers; failure = non-zero exit code or "
    "error text",
}


def _workflow_text(task: str) -> str:
    m = _WORKFLOW_HEAD_RE.search(task or "")
    if not m:
        return "The task declares no numbered workflow."
    tail = task[m.start() :]
    stop = re.search(r"\n## (?!Recommended)|</PROBLEM_SOLVING_WORKFLOW>|\n<(?!/)[A-Z_]+>", tail)
    return tail[: stop.end() if stop else 1500][:1500]


def _reference_measurements(reference: str) -> str:
    steps = re.split(r"^REFERENCE STEP \d+:$", reference, flags=re.M)[1:]
    bodies = [s.split("\nENVIRONMENT OBSERVATION:\n")[0].strip() for s in steps] or [""]
    words = [len(b.split()) for b in bodies]
    chars = [len(b) for b in bodies]
    prose_words = [len(_FENCE_RE.sub("", b).split()) for b in bodies]
    return (
        "REFERENCE MEASUREMENTS (programmatic):\n"
        f"- total REFERENCE STEP words: {sum(words)}\n"
        f"- longest single REFERENCE STEP: {max(words)} words\n"
        f"- total REFERENCE STEP characters: {sum(chars)}\n"
        f"- longest single REFERENCE STEP: {max(chars)} characters\n"
        f"- total REFERENCE STEP prose words (outside fenced code): {sum(prose_words)}\n"
        f"- average REFERENCE STEP words: {round(sum(words) / len(bodies))}\n"
        f"- average REFERENCE STEP characters: {round(sum(chars) / len(bodies))}\n"
        f"- REFERENCE STEP count: {len(words)}"
    )


_ECONOMY_DUPLICATE_MULTIPLIER = 2.0


def _reference_raw_measurements(reference: str) -> dict[str, int]:
    """Same computation as _reference_measurements(), as raw numbers instead of formatted text -
    used by duplicate_economy_bounds() to recompute a bound at a different multiplier without
    reparsing the LLM's own generated question text."""
    steps = re.split(r"^REFERENCE STEP \d+:$", reference, flags=re.M)[1:]
    bodies = [s.split("\nENVIRONMENT OBSERVATION:\n")[0].strip() for s in steps] or [""]
    words = [len(b.split()) for b in bodies]
    chars = [len(b) for b in bodies]
    prose_words = [len(_FENCE_RE.sub("", b).split()) for b in bodies]
    return {
        "total_words": sum(words),
        "max_words": max(words),
        "total_chars": sum(chars),
        "max_chars": max(chars),
        "total_prose_words": sum(prose_words),
        "avg_words": round(sum(words) / len(bodies)),
        "avg_chars": round(sum(chars) / len(bodies)),
    }


_ECONOMY_TEMPLATE_METRICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "total_words",
        re.compile(r"^Is total CANDIDATE OUTPUT at most (\d[\d,]*) words\?$", re.IGNORECASE),
    ),
    (
        "max_words",
        re.compile(r"^Is the longest single output at most (\d[\d,]*) words\?$", re.IGNORECASE),
    ),
    (
        "total_chars",
        re.compile(r"^Is total CANDIDATE OUTPUT at most (\d[\d,]*) characters\?$", re.IGNORECASE),
    ),
    (
        "max_chars",
        re.compile(
            r"^Is the longest single output at most (\d[\d,]*) characters\?$", re.IGNORECASE
        ),
    ),
    (
        "total_prose_words",
        re.compile(
            r"^Is CANDIDATE OUTPUT prose, apart from code blocks, at most (\d[\d,]*) words\?$",
            re.IGNORECASE,
        ),
    ),
    (
        "avg_words",
        re.compile(
            r"^Is the average CANDIDATE OUTPUT per turn at most (\d[\d,]*) words\?$", re.IGNORECASE
        ),
    ),
    (
        "avg_chars",
        re.compile(
            r"^Is the average CANDIDATE OUTPUT per turn at most (\d[\d,]*) characters\?$",
            re.IGNORECASE,
        ),
    ),
)


def duplicate_economy_bounds(
    questions: list[dict[str, str]],
    reference: str,
    *,
    multiplier: float = _ECONOMY_DUPLICATE_MULTIPLIER,
) -> list[dict[str, str]]:
    raw = _reference_raw_measurements(reference)
    duplicates: list[dict[str, str]] = []
    for question in questions:
        text = (question.get("text") or "").strip()
        if not is_measurement_bound_question(text):
            continue
        for metric, pattern in _ECONOMY_TEMPLATE_METRICS:
            match = pattern.match(text)
            if not match:
                continue
            new_bound = math.ceil(raw[metric] * multiplier / 10) * 10
            start, end = match.span(1)
            duplicate = dict(question)
            duplicate["text"] = f"{text[:start]}{new_bound}{text[end:]}"
            duplicates.append(duplicate)
            break
    return duplicates


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


def filter_reference_leaks(
    questions: list[dict[str, str]], *, discards: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    if discards is None:
        return [q for q in questions if "the reference" not in q["text"].casefold()]
    kept = []
    for q in questions:
        if "the reference" in q["text"].casefold():
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


SPAN_MIN_CHARS = 12
SPAN_MAX_HITS = 3
SPAN_DISTINCT_CHARS = 24

_SMART = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_SPAN_TRIM = re.compile(r"^[\s'\"`]+|[\s'\"`]*(?:\.{3}|…)?[\s'\"`]*$")


def normalise_span(text: str) -> str:
    """Fold the differences a model introduces when copying, and nothing else.

    Whitespace runs collapse because the reference is rendered with indentation and models
    reflow what they quote; smart quotes and dashes revert because models typographically
    "improve" it. Case is preserved: code is case-sensitive, and folding it would let a
    near-miss match.
    """
    return " ".join(unicodedata.normalize("NFKC", text).translate(_SMART).split())


def span_of(question: dict[str, str]) -> str:
    """The quoted part of an evidence field, without its class prefix or decoration."""
    _, sep, span = (question.get("evidence") or "").partition(": ")
    return normalise_span(_SPAN_TRIM.sub("", span)) if sep else ""


def span_verdict(question: dict[str, str], reference: str, task: str) -> tuple[bool, str]:
    """Whether a question's evidence really cites the reference, and why not when it does not."""
    span = span_of(question)
    if not span:
        return False, "no_span"
    if len(span) < SPAN_MIN_CHARS:
        return False, "span_too_short"
    if span in normalise_span(task):
        return False, "span_from_task"
    hits = normalise_span(reference).count(span)
    if hits == 0:
        return False, "span_not_in_reference"
    if hits == 1 or (hits <= SPAN_MAX_HITS and len(span) >= SPAN_DISTINCT_CHARS):
        return True, "ok"
    return False, "span_not_distinctive"


def verify_spans(
    questions: list[dict[str, str]],
    reference: str,
    task: str,
    *,
    enforce: bool,
    discards: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Check every question's evidence against the reference it claims to quote.

    With enforce=False nothing is dropped and only the tally is returned, which is how the
    thresholds above get calibrated on real batches before they start deleting questions.
    """
    kept: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for question in questions:
        ok, reason = span_verdict(question, reference, task)
        counts[reason] = counts.get(reason, 0) + 1
        if ok or not enforce:
            kept.append(question)
            continue
        if discards is not None:
            discards.append(
                {
                    "stage": "verify_spans",
                    "reason": reason,
                    "text": question.get("text", ""),
                    "origin": "content",
                    "detail": f"evidence span not usable: {span_of(question)[:120]!r}",
                }
            )
    return kept, counts


# ---------------------------------------------------------------- split rubric assembly


def build_reference_specialist_messages(
    *,
    specialist: ReferenceSpecialist,
    task: str,
    reference: str,
    fmt: str,
    prefix_turns: int,
    candidate_turns: int,
) -> list[dict[str, str]]:
    """One specialist's call, laid out so the costly part of the prompt can be cached.

    The skeleton, the task and the reference are identical across every specialist, and the
    class block is appended last, so all calls share a long common prefix.
    """
    window = REFERENCE_SCORED_WINDOW_BLOCK.format(
        workflow_text=_workflow_text(task),
        prefix_turns=prefix_turns,
        candidate_turns=candidate_turns,
        observation_format=fmt,
        success_marker=_OBSERVATION_SUCCESS_MARKERS.get(
            fmt, _OBSERVATION_SUCCESS_MARKERS["returncode"]
        ),
    )
    user = (
        "TASK — the system prompt the agent operates under, and the conversation so far. Read for "
        "comprehension; it is not a source of facts about the solution:\n"
        f"------\n{task.rstrip()}\n------\n\n"
        "REFERENCE TRAJECTORY — one strong agent's continuation from the same point under the same "
        "turn limit. It evidences what is achievable; its route is not a standard:\n"
        f"------\n{reference.rstrip()}\n------\n\n"
        f"{window}\n\n"
        + SPECIALIST_BLOCK.format(
            name=specialist.name,
            tag=specialist.tag,
            extract=specialist.extract,
            subject=specialist.subject,
            predicate=specialist.predicate,
            exclude=specialist.exclude,
            lo=specialist.lo,
            hi=specialist.hi,
        )
    )
    return [
        {"role": "system", "content": REFERENCE_SKELETON},
        {"role": "user", "content": user},
    ]


def reference_specialist_schema(specialist: ReferenceSpecialist) -> dict[str, Any]:
    """Force one specialist's output shape: its own tag, and no more than its own ceiling."""
    fact = {
        "type": "object",
        "properties": {
            "statement": {"type": "string"},
            "span": {"type": "string"},
            "established": {"type": "boolean"},
            "in_prefix": {"type": "boolean"},
        },
        "required": ["statement", "span", "established", "in_prefix"],
        "additionalProperties": False,
    }
    question = {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
            "evidence": {"type": "string"},
            "text": {"type": "string"},
            "example_bad": {"type": "string"},
            "tag": {"type": "string", "enum": [specialist.tag]},
        },
        "required": ["step", "evidence", "text", "example_bad", "tag"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "facts": {"type": "array", "items": fact},
            # every class may legitimately find nothing, so the floor is zero here and the
            # checklist-wide floor is applied after the specialists are merged
            "questions": {
                "type": "array",
                "minItems": 0,
                "maxItems": specialist.hi,
                "items": question,
            },
        },
        "required": ["facts", "questions"],
        "additionalProperties": False,
    }
