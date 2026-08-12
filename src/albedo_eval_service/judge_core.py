from __future__ import annotations

import json
import math
import os as _os
import re
from collections.abc import Callable
from difflib import SequenceMatcher
from statistics import mean
from typing import Any

from .evaluator.prompt_rubric import (
    BEHAVIOR_COMMON,
    BEHAVIOR_PHASES,
    NO_TESTS_NOTE,
    REFERENCE_QUESTION_SYSTEM,
    REFERENCE_QUESTION_USER,
    REFERENCE_SCORED_WINDOW_BLOCK,
    RUBRIC_ECONOMY_CAP,
    RUBRIC_LENGTH_BOUNDS,
    RUBRIC_MAX_QUESTIONS,
    RUBRIC_MIN_QUESTIONS,
    RUBRIC_NEGATIVE_CAP,
    RUBRIC_REFERENCE_TARGET,
)
from .judge.prompt_judge import JUDGE_SYSTEM, JUDGE_USER
from .shared.observation_format import (
    THINK_CLOSE_RE,
    THINK_OPEN_RE,
    THINK_PAIR_RE,
    THINK_TAG_RE,
    mask_fenced_spans,
    unmask_fenced_spans,
)
from .simulator.prompt_simulator import COMPLETE_MARKER

CHALLENGER_WIN_MARGIN = 0.03
QUESTION_FLOOR_FRACTION = 0.22
GENERIC_HYGIENE_QUESTION_LIMIT = 3
NEGATIVE_QUESTION_LIMIT = 8


def question_floor(n: int) -> int:
    return max(1, round(n * QUESTION_FLOOR_FRACTION))


VALID_TAGS = ("explore", "verification", "action", "continuity", "economy")


def build_behavior_messages(
    *,
    phase: str,
    index: int,
    task: str,
    prefix_tail: str,
    k: int,
    tests_seen: bool,
) -> list[dict[str, str]]:
    behavior = BEHAVIOR_PHASES[phase]
    part_text = behavior.parts[index].text
    if phase == "at_edit" and index == 0 and not tests_seen:
        part_text += NO_TESTS_NOTE
    gold = "\n".join(f"  GOOD: {t}" for t in behavior.gold)
    system = (
        f"You write EXACTLY {k} yes/no checklist questions testing ONE behaviour of a "
        f"coding-agent trajectory.\n\n{behavior.head}\n\nTHE BEHAVIOUR:\n{part_text}\n\n"
        f"CANONICAL FORM — match this style:\n{gold}\n\nRULES:\n"
        f"- INSTANTIATE: at least 3 of the questions must name a concrete file, symbol, or code "
        f"fragment copied verbatim from the sample context below (never invented, never guessed "
        f"from conventions); questions with no such target keep the generic form.\n"
        f"- One property, one question: two questions a trajectory could only pass or fail "
        f"together are duplicates — replace one with a different property of this behaviour.\n"
        f"- Questions address the trajectory only; never reference these instructions.\n\n"
        + BEHAVIOR_COMMON
    )
    user = (
        f"SAMPLE CONTEXT:\n{(task or '')[:4000]}\n\nLAST CONTEXT TURNS:\n{prefix_tail}"
        f"\n\nWrite the {k} questions."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_BEHAVIOR_BANNED_RE = re.compile(
    r"reproduc|\brepro\b|failure[^?]{0,40}before[^?]{0,20}(edit|fix|chang)"
    r"|(run|execut)[^?]{0,25}after (each|every) edit",
    re.IGNORECASE,
)
_BEHAVIOR_REPEAT_FORM_RE = re.compile(r"avoid|re-?issu|retr|repeat|already received", re.IGNORECASE)
_BEHAVIOR_FILE_RE = re.compile(r"[\w/-]+\.(?:py|rs|go|js|ts|c|h|java|php|rb)\b")


def filter_behavior_questions(
    questions: list[dict[str, str]], context: str, *, discards: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """Deterministic backstop for prompt drift: drop banned-theme questions (repetition-form
    mentions of a repro command are fine) and questions naming files absent from the context."""
    kept = []
    for question in questions:
        text = question.get("text", "")
        if _BEHAVIOR_BANNED_RE.search(text) and not _BEHAVIOR_REPEAT_FORM_RE.search(text):
            if discards is not None:
                discards.append(
                    {
                        "stage": "filter_behavior_questions",
                        "reason": "behavior_banned_theme",
                        "text": text,
                        "origin": "behavior",
                    }
                )
            continue
        if any(f not in context for f in _BEHAVIOR_FILE_RE.findall(text)):
            if discards is not None:
                discards.append(
                    {
                        "stage": "filter_behavior_questions",
                        "reason": "behavior_file_not_in_context",
                        "text": text,
                        "origin": "behavior",
                    }
                )
            continue
        kept.append(question)
    return kept


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)


def measure(text: str) -> dict[str, int]:
    blocks = _FENCE_RE.findall(text)
    fence = text.find("```")
    prose = (text[:fence] if fence >= 0 else text).strip()
    return {
        "total_words": len(text.split()),
        "total_chars": len(text),
        "prose_words": len(prose.split()),
        "code_blocks": len(blocks),
        "code_lines": max((len(b.strip("\n").splitlines()) for b in blocks), default=0),
        "code_chars": sum(len(b) for b in blocks),
    }


_CANDIDATE_BLOCK_RE = re.compile(r"CANDIDATE OUTPUT(?: \d+)?:\n------\n(.*?)\n------", re.DOTALL)


def candidate_output_measure(text: str) -> dict[str, int]:
    blocks = _CANDIDATE_BLOCK_RE.findall(text)
    if not blocks:
        blocks = [text]
    words = [len(b.split()) for b in blocks]
    chars = [len(b) for b in blocks]
    prose_words = [len(_FENCE_RE.sub("", b).split()) for b in blocks]
    return {
        "blocks": len(_CANDIDATE_BLOCK_RE.findall(text)),
        "total_words": sum(words),
        "max_words": max(words) if words else 0,
        "total_chars": sum(chars),
        "max_chars": max(chars) if chars else 0,
        "total_prose_words": sum(prose_words),
        "avg_words": round(sum(words) / len(blocks)) if blocks else 0,
        "avg_chars": round(sum(chars) / len(blocks)) if blocks else 0,
    }


def measurements_block(text: str) -> str:
    m = measure(text)
    c = candidate_output_measure(text)
    return (
        "MEASUREMENTS (computed programmatically from the trajectory above — authoritative for "
        "every size or length question):\n"
        f"- CANDIDATE OUTPUT words, all scored blocks combined (use THIS for any size bound on "
        f"the candidate's own outputs): {c['total_words']}\n"
        f"- longest single CANDIDATE OUTPUT: {c['max_words']} words "
        f"({c['blocks']} scored blocks)\n"
        f"- CANDIDATE OUTPUT characters, all scored blocks combined: {c['total_chars']}\n"
        f"- longest single CANDIDATE OUTPUT: {c['max_chars']} characters\n"
        f"- CANDIDATE OUTPUT prose words, all scored blocks combined, excluding fenced code: "
        f"{c['total_prose_words']}\n"
        f"- average CANDIDATE OUTPUT words per turn: {c['avg_words']} ({c['blocks']} scored blocks)\n"  # noqa: E501
        f"- average CANDIDATE OUTPUT characters per turn: {c['avg_chars']}\n"
        f"- total words of the whole document incl. context/observations: {m['total_words']}\n"
        f"- total characters: {m['total_chars']}\n"
        f"- fenced code blocks: {m['code_blocks']} (longest: {m['code_lines']} lines, "
        f"{m['code_chars']} code characters total)\n\n"
    )


_TESTS_VISIBLE_RE = re.compile(
    r"(^|[\s'\"/(=])tests?/|test_[a-z0-9_]+\.(py|go|rs|js|ts)|_test\.(go|rs|py)|pytest|cargo test",
    re.IGNORECASE | re.MULTILINE,
)


def tests_visible(messages: list[dict[str, str]] | None) -> bool:
    return any(_TESTS_VISIBLE_RE.search(m.get("content") or "") for m in messages or [])


def sample_phase(messages: list[dict[str, str]] | None) -> str:
    """Trim phase, inferred from the prefix the sampler produced: the cut lands near the start
    (cold), just before the first edit (pre_edit), or just after it (at_edit)."""
    turns = [m.get("content") or "" for m in messages or [] if m.get("role") == "assistant"]
    if any(_edited_in_turn(t) for t in turns):
        return "at_edit"
    return "cold" if len(turns) <= 2 else "pre_edit"


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

# rubric tags carry no requires label; map them so the gate and label caps keep working
RUBRIC_TAG_REQUIRES = {
    "action": "action",
    "continuity": "action",
    "verification": "action",
    "explore": "read",
    "economy": "neutral",
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


def reference_question_schema() -> dict[str, Any]:
    step = {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
            "text": {"type": "string"},
            "already_done_in_conversation": {"type": "boolean"},
            "demonstrated_by_reference": {"type": "boolean"},
            "verdict": {"type": "string", "enum": ["demonstrated", "not_demonstrated"]},
        },
        "required": [
            "step",
            "text",
            "already_done_in_conversation",
            "demonstrated_by_reference",
            "verdict",
        ],
        "additionalProperties": False,
    }
    question = {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
            "evidence": {"type": "string"},
            "text": {"type": "string"},
            "example_bad": {"type": "string"},
            "tag": {"type": "string", "enum": list(VALID_TAGS)},
        },
        "required": ["step", "evidence", "text", "example_bad", "tag"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "ledger": {
                "type": "object",
                "properties": {
                    "steps": {"type": "array", "items": step},
                    "frontier_step": {"type": "integer"},
                    "reference_finished": {"type": "boolean"},
                    "focus": {"type": "string"},
                },
                "required": ["steps", "frontier_step", "reference_finished", "focus"],
                "additionalProperties": False,
            },
            "questions": {
                # the prompt itself allows 8-14 when the reference only diagnosed
                "type": "array",
                "minItems": 8,
                "maxItems": RUBRIC_MAX_QUESTIONS,
                "items": question,
            },
        },
        "required": ["ledger", "questions"],
        "additionalProperties": False,
    }


def build_reference_question_messages(
    *,
    task: str,
    reference: str,
    fmt: str,
    prefix_turns: int,
    candidate_turns: int,
) -> list[dict[str, str]]:
    system = REFERENCE_QUESTION_SYSTEM.format(
        target=RUBRIC_REFERENCE_TARGET,
        min_n=RUBRIC_MIN_QUESTIONS,
        max_n=RUBRIC_MAX_QUESTIONS,
        negative_cap=RUBRIC_NEGATIVE_CAP,
        economy_cap=RUBRIC_ECONOMY_CAP,
        bound_n=RUBRIC_LENGTH_BOUNDS,
    )
    window = REFERENCE_SCORED_WINDOW_BLOCK.format(
        workflow_text=_workflow_text(task),
        prefix_turns=prefix_turns,
        candidate_turns=candidate_turns,
        observation_format=fmt,
        success_marker=_OBSERVATION_SUCCESS_MARKERS.get(
            fmt, _OBSERVATION_SUCCESS_MARKERS["returncode"]
        ),
    )
    user = REFERENCE_QUESTION_USER.format(
        task=task.rstrip(),
        reference=reference.rstrip(),
        min_n=RUBRIC_MIN_QUESTIONS,
        max_n=RUBRIC_MAX_QUESTIONS,
        reference_measurements=_reference_measurements(reference),
        scored_window=window,
        bound_n=RUBRIC_LENGTH_BOUNDS,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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


_EDIT_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)
_EDIT_COMMAND_RE = re.compile(
    r"sed\s+-i"
    # a write counts only to a repo path: `2>/dev/null`, `2>&1`, /tmp scratch files
    # (reproduction scripts), `->`/`=>` arrows and `x > 5` comparisons are not edits —
    # the redirect target must look like a path (contain . or /)
    r"|(?<![-=0-9&])>>?\s*(?!/dev/|/tmp/)(?=[\w.~/-]*[./])[\w./~-]"
    r"|tee\s+(?!/dev/|/tmp/)[\w./~-]|cat\s*>>?\s*(?!/dev/|/tmp/)[\w./~-]"
    r"|str_replace|git\s+apply|patch\s+-p|applypatch|>{7}\s*REPLACE"
    r"|cp\s+[\w./-]+\s+(?!/dev/|/tmp/)[\w./-]+|mv\s+[\w./-]+\s+(?!/dev/|/tmp/)[\w./-]+",
)


def _edited_in_turn(text: str) -> bool:
    """Whether a turn's shell commands change the repository.

    Only bash blocks are scanned. Running the pattern over the whole turn made a `</think>` tag or
    a `>` in prose read as a redirect, which marked every candidate as having edited and so left
    apply_measurement_gate permanently inert.
    """
    return any(_EDIT_COMMAND_RE.search(cmd) for cmd in _EDIT_BLOCK_RE.findall(text or ""))


_SUBMIT_RE = re.compile(re.escape(COMPLETE_MARKER))
_UNFOLDED_AVOID_RE = re.compile(
    r"^\s*(?:does[^?]{0,60}\bavoid|is[^?]{0,60}\bfree of|does[^?]{0,60}\brefrain)", re.IGNORECASE
)
_ACTION_VERB_RE = re.compile(
    r"\b(edit|modif|submit|propagat|verif|appl|patch|chang|fix)\w*\b", re.IGNORECASE
)
_COMPLETED_WORK_RE = re.compile(
    r"\b(submit|final output|after the edit|the edit .{0,30}(applied|made|verified))\b",
    re.IGNORECASE,
)

READ_CAPS_BY_PHASE = {"cold": 10, "pre_edit": 7, "at_edit": 5}
READ_ONLY_QUESTION_CAP = 5


def candidate_turn_texts_from_merged(text: str) -> list[str]:
    return _CANDIDATE_BLOCK_RE.findall(text or "")


def trajectory_made_edit(turn_texts: list[str]) -> bool:
    return any(_edited_in_turn(text) for text in turn_texts)


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
    phase: str,
    reference_made_edit: bool,
    discards: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    kept: list[dict[str, str]] = []
    drops = {"read_cap": 0, "unfolded_avoid": 0, "no_edit_dead_weight": 0}
    read_kept = 0
    read_only_question_cap = READ_CAPS_BY_PHASE.get(phase, READ_ONLY_QUESTION_CAP)
    _drop = _discard_recorder(discards, stage="enforce_question_labels", counters=drops)

    for question in questions:
        text = question.get("text", "")
        requires = question.get("requires") or "neutral"
        if requires not in ("action", "read", "neutral"):
            requires = "neutral"
        is_size = is_measurement_bound_question(text)
        if not is_size and _UNFOLDED_AVOID_RE.search(text) and not _ACTION_VERB_RE.search(text):
            _drop("unfolded_avoid", text)
            continue
        if not reference_made_edit and requires == "action" and _COMPLETED_WORK_RE.search(text):
            _drop("no_edit_dead_weight", text)
            continue
        if requires == "read" and not is_size:
            if read_kept >= read_only_question_cap:
                _drop("read_cap", text)
                continue
            read_kept += 1
        question["requires"] = requires
        kept.append(question)
    for position, question in enumerate(kept, start=1):
        question["id"] = f"q_{position:02d}"
    return kept, drops


def apply_measurement_gate(
    answers: dict[str, str | None],
    questions: list[dict[str, str]],
    *,
    candidate_turn_texts: list[str],
    reference_made_edit: bool,
) -> dict[str, str | None]:
    made_edit = trajectory_made_edit(candidate_turn_texts)
    if made_edit:
        return answers
    final_is_read = bool(candidate_turn_texts) and not (
        _edited_in_turn(candidate_turn_texts[-1])
        or _SUBMIT_RE.search(candidate_turn_texts[-1] or "")
    )
    gated = dict(answers)
    for question in questions:
        qid = question.get("id")
        if qid not in gated:
            continue
        text = question.get("text", "")
        if (
            question.get("requires") == "neutral"
            and not is_measurement_bound_question(text)
            and _NEGATIVE_QUESTION_RE.search(text)
            and _ACTION_VERB_RE.search(text)
        ):
            gated.pop(qid)
            continue
        if reference_made_edit and final_is_read and (question.get("requires") == "action"):
            gated[qid] = "0"
    return gated


NO_VISIBLE_OUTPUT = "[no visible output]"
_COMMAND_FENCE_RE = re.compile(r"```(?:bash|sh|shell)[ \t]*\n", re.IGNORECASE)


def _drop_unclosed_reasoning(text: str, fences: list[str]) -> str:
    match = THINK_OPEN_RE.search(text)
    if not match:
        return text
    tail = text[match.end() :]
    if _COMMAND_FENCE_RE.search(unmask_fenced_spans(tail, fences)):
        return text[: match.start()] + tail
    return text[: match.start()]


def strip_leaked_reasoning(text: str) -> str:
    """Drop model reasoning that leaked into a scored candidate turn.

    vLLM's qwen3 reasoning parser leaves an orphaned closing tag (and occasionally the reasoning
    itself) in the generated text. A `THOUGHT:` before the tag is the mini-coder-rs corpus's own
    style and is real content, so only the stray tag is removed in that case.
    """
    masked, fences = mask_fenced_spans(text or "")
    cleaned = THINK_PAIR_RE.sub("", masked)
    cleaned = _drop_unclosed_reasoning(cleaned, fences)
    if THINK_CLOSE_RE.search(cleaned):
        head, tail = THINK_CLOSE_RE.split(cleaned, 1)
        cleaned = head + tail if "THOUGHT:" in head else tail
    cleaned = THINK_TAG_RE.sub("", cleaned)
    return unmask_fenced_spans(cleaned, fences).strip()


def strip_candidate_reasoning(trajectory: str) -> str:
    """Apply strip_leaked_reasoning inside CANDIDATE OUTPUT blocks only.

    Context turns and environment observations are left byte-for-byte intact: the mini-coder-rs
    corpus legitimately carries `</think>` in its own assistant turns.
    """

    def _rewrite(match: re.Match[str]) -> str:
        block, inner = match.group(0), match.group(1)
        start, end = match.start(1) - match.start(0), match.end(1) - match.start(0)
        stripped = strip_leaked_reasoning(inner)
        if not stripped:
            stripped = NO_VISIBLE_OUTPUT if inner.strip() else inner
        return block[:start] + stripped + block[end:]

    return _CANDIDATE_BLOCK_RE.sub(_rewrite, trajectory or "")


def build_judge_messages(*, response: str, questions: list[dict[str, str]]) -> list[dict[str, str]]:
    shown = [
        {
            "id": q["id"],
            "tag": q.get("tag", ""),
            "text": q["text"],
            "example_bad": q.get("example_bad", ""),
        }
        for q in questions
    ]
    cleaned = strip_candidate_reasoning(strip_reply_injection(response)).rstrip()
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                response=cleaned,
                measurements=measurements_block(cleaned),
                questions_json=json.dumps(shown, ensure_ascii=False, indent=1),
            ),
        },
    ]


def behavior_question_schema(n: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": question_floor(n),
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "example_bad": {"type": "string"},
                        "requires": {
                            "type": "string",
                            "enum": ["action", "read", "neutral"],
                        },
                    },
                    "required": ["text", "example_bad", "requires"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


def answer_schema(question_ids: list[str]) -> dict[str, Any]:
    count = len(question_ids)
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": question_ids},
                        "explanation": {"type": "string"},
                        "answer": {"type": "integer", "enum": [1, 0]},
                    },
                    "required": ["id", "explanation", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\S\s]*?)\s*```", re.IGNORECASE)


def extract_json(raw: str, prefer_keys: tuple[str, ...] = ()) -> Any | None:
    if not raw:
        return None
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for match in _JSON_FENCE_RE.finditer(raw):
        try:
            obj = json.loads(match.group(1).strip())
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            candidates.append(obj)
    for index, char in enumerate(raw):
        if char not in "{[":
            continue
        if index > 0 and raw[index - 1] == "`":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[index:])
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            candidates.append(obj)
    if prefer_keys:
        keyed = [c for c in candidates if isinstance(c, dict) and set(prefer_keys) & set(c)]
        if keyed:
            return keyed[-1]
    return candidates[0] if candidates else None


_QUESTION_STOPWORDS = frozenset(
    "does the response a an as its it is are do of to in on for with or and that this by be "
    "not no any all one when if such e g rather than instead avoid avoids using use uses run "
    "runs running contain contains include includes reference references".split()
)
_DUP_JACCARD = float(_os.environ.get("ALBEDO_EXP_DUP_JACCARD", "0.75"))
_DUP_CONTAINMENT = float(_os.environ.get("ALBEDO_EXP_DUP_CONTAINMENT", "0.90"))
_DUP_CHAR_RATIO = float(_os.environ.get("ALBEDO_EXP_DUP_CHAR_RATIO", "0.87"))
_DUP_CHAR_MIN_LEN = 20
_TEMPLATE_KEY_TOKENS = 5
_TEMPLATE_MAX_PER_KEY = int(_os.environ.get("ALBEDO_EXP_TEMPLATE_MAX_PER_KEY", "4"))
_CONDITIONAL_RE = re.compile(r"^\s*if\b", re.IGNORECASE)
_UNCONDITIONAL_SUBMIT_RE = re.compile(
    r"\b(?:submit|submits|submitting|finali[sz]e|finali[sz]es|finish(?:es)?)\b",
    re.IGNORECASE,
)
_BOUNDED_SUBMIT_RE = re.compile(
    r"\b(?:after success|stop after success|submit after success|after verification|"
    r"after (?:the )?(?:observations?|verification|tests?|checks?|diff|requirement)[^?]{0,80}"
    r"(?:show|shows|pass|passes|succeed|succeeds|verified|satisfied|solved)|"
    r"(?:once|when)[^?]{0,100}(?:success|verified|satisfied|solved|passes|succeeds))\b",
    re.IGNORECASE,
)
_NEGATIVE_QUESTION_RE = re.compile(
    r"\b(avoid|avoids|avoided|not|never|does\s+not|do\s+not|without|refrains?)\b",
    re.IGNORECASE,
)
_GENERIC_HYGIENE_RE = re.compile(
    r"\b("
    r"word|words|characters?|shorter|exceeding|less than|"
    r"under (?:roughly|about|\d)|below (?:about|\d)|within (?:about|\d)|"
    r"thought (?:at most|fit|free|section)|sentences?|paragraph|"
    r"self-corrections?|raw chain-of-thought|scratch work|"
    r"quoting|restating|re-printing|"
    r"bash block at most|code block|destructive operations?|rm -rf|"
    r"read-only inspection|verification step|quotes?|backslashes|regex patterns?|"
    r"plan-action match"
    r")\b",
    re.IGNORECASE,
)


def _question_signature(text: str) -> frozenset[str]:
    words = re.sub(r"[^a-z0-9_./-]+", " ", text.casefold()).split()
    return frozenset(
        w.rstrip("s") if len(w) > 3 else w for w in words if w not in _QUESTION_STOPWORDS
    )


def _template_key(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw in text.split():
        core = raw.strip("?,.!:;\"'()")
        if not core:
            continue
        is_target = (
            "`" in raw
            or any(ch.isdigit() for ch in core)
            or any(ch in "._/=<>[]{}\\" for ch in core)
            or any(ch.isupper() for ch in core[1:])
        )
        normalized = "§" if is_target else core.casefold()
        if normalized == "§" and tokens and tokens[-1] == "§":
            continue
        tokens.append(normalized)
    return tuple(tokens[:_TEMPLATE_KEY_TOKENS])


def _near_duplicate(sig_a: frozenset[str], sig_b: frozenset[str], text_a: str, text_b: str) -> bool:
    if sig_a and sig_b:
        inter = len(sig_a & sig_b)
        if inter / len(sig_a | sig_b) >= _DUP_JACCARD:
            return True
        if inter / min(len(sig_a), len(sig_b)) >= _DUP_CONTAINMENT:
            return True
    if min(len(text_a), len(text_b)) >= _DUP_CHAR_MIN_LEN:
        ratio = SequenceMatcher(None, text_a.casefold(), text_b.casefold()).ratio()
        if ratio >= _DUP_CHAR_RATIO:
            return True
    return False


def _is_generic_hygiene_question(text: str) -> bool:
    return bool(_GENERIC_HYGIENE_RE.search(text))


_MEASUREMENT_BOUND_RE = re.compile(
    r"\b\d{2,6}\b[^?]{0,40}\b(?:words?|characters?|sentences?|lines)\b"
    r"|\b(?:words?|characters?|sentences?|lines)\b[^?]{0,40}\b\d{2,6}\b",
    re.IGNORECASE,
)
MEASUREMENT_QUESTION_LIMIT = 8


def is_measurement_bound_question(text: str) -> bool:
    return bool(_MEASUREMENT_BOUND_RE.search(text))


def _is_negative_question(text: str) -> bool:
    return bool(_NEGATIVE_QUESTION_RE.search(text))


def is_unbounded_submit_question(text: str) -> bool:
    return (
        bool(_UNCONDITIONAL_SUBMIT_RE.search(text))
        and not _is_negative_question(text)
        and not bool(_BOUNDED_SUBMIT_RE.search(text))
    )


def parse_questions(
    raw: str,
    n: int,
    *,
    discards: list[dict[str, str]] | None = None,
    origin: str = "content",
) -> tuple[list[dict[str, str]], bool]:
    obj = extract_json(raw, prefer_keys=("questions",))
    items = obj.get("questions") if isinstance(obj, dict) else obj
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    kept_signatures: list[frozenset[str]] = []
    template_counts: dict[tuple[str, ...], int] = {}
    generic_hygiene_count = 0
    negative_count = 0
    measurement_count = 0

    _drop = _discard_recorder(discards, stage="parse_questions", origin=origin)

    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                _drop("malformed_item", repr(item)[:200])
                continue
            if not str(item.get("text", "")).strip():
                _drop("empty_text", str(item.get("text", "")).strip())
                continue
            text = str(item["text"]).strip()
            if _CONDITIONAL_RE.match(text):
                _drop("conditional_form", text)
                continue
            key = " ".join(text.casefold().split())
            if key in seen:
                _drop("exact_duplicate", text)
                continue
            seen.add(key)
            if is_measurement_bound_question(text):
                if measurement_count >= MEASUREMENT_QUESTION_LIMIT:
                    _drop("measurement_cap", text)
                    continue
                measurement_count += 1
                kept_signatures.append(_question_signature(text))
                out.append(
                    {
                        "text": text,
                        "example_bad": str(item.get("example_bad", "")).strip(),
                        "requires": str(item.get("requires", "neutral")),
                        "tag": t
                        if (t := str(item.get("tag", "")).strip().lower()) in VALID_TAGS
                        else "",
                    }
                )
                continue
            is_negative = _is_negative_question(text)
            if is_negative and negative_count >= NEGATIVE_QUESTION_LIMIT:
                _drop("negative_cap", text)
                continue
            is_generic_hygiene = _is_generic_hygiene_question(text)
            if is_generic_hygiene and generic_hygiene_count >= GENERIC_HYGIENE_QUESTION_LIMIT:
                _drop("generic_hygiene_cap", text)
                continue
            template = _template_key(text)
            if (
                len(template) == _TEMPLATE_KEY_TOKENS
                and template_counts.get(template, 0) >= _TEMPLATE_MAX_PER_KEY
            ):
                _drop("template_cap", text)
                continue
            signature = _question_signature(text)
            near_dup_of = None
            for prev_sig, prev in zip(kept_signatures, out):
                if _near_duplicate(signature, prev_sig, text, prev["text"]):
                    near_dup_of = prev["text"]
                    break
            if near_dup_of is not None:
                _drop(
                    "near_duplicate",
                    text,
                    detail=f"near-duplicate of kept question: {near_dup_of!r}",
                )
                continue
            template_counts[template] = template_counts.get(template, 0) + 1
            if is_negative:
                negative_count += 1
            if is_generic_hygiene:
                generic_hygiene_count += 1
            kept_signatures.append(signature)
            out.append(
                {
                    "text": text,
                    "example_bad": str(item.get("example_bad", "")).strip(),
                    "requires": str(item.get("requires", "neutral")),
                    "tag": t
                    if (t := str(item.get("tag", "")).strip().lower()) in VALID_TAGS
                    else "",
                    **({"evidence": e} if (e := str(item.get("evidence", "")).strip()) else {}),
                    **({"step": item["step"]} if isinstance(item.get("step"), int) else {}),
                }
            )
    if len(out) > n:
        for extra in out[n:]:
            _drop(
                "truncated_excess",
                extra["text"],
                detail=f"{len(out)} well-formed candidates parsed, only the first {n} requested are kept",  # noqa: E501
            )
        out = out[:n]
    for position, question in enumerate(out, start=1):
        question["id"] = f"q_{position:02d}"
    return out, len(out) >= question_floor(n)


_ANSWER_TO_BIT: dict[str, float] = {"1": 1.0, "0": 0.0}


def parse_answers(
    raw: str, question_ids: list[str]
) -> tuple[dict[str, str | None], dict[str, str], bool]:
    obj = extract_json(raw, prefer_keys=("answers",))
    items = obj.get("answers") if isinstance(obj, dict) else obj
    answers: dict[str, str | None] = {qid: None for qid in question_ids}
    explanations: dict[str, str] = {qid: "" for qid in question_ids}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("id", "")).strip()
            value = str(item.get("answer", "")).strip().lower()
            if qid in answers and value in _ANSWER_TO_BIT:
                answers[qid] = value
                explanations[qid] = str(item.get("explanation", "")).strip()
    parse_ok = all(value is not None for value in answers.values())
    return answers, explanations, parse_ok


def judge_yes_rate(
    answers: dict[str, str | None], questions: list[dict[str, str]] | None = None
) -> float | None:
    bits = [_ANSWER_TO_BIT[v] for v in answers.values() if v in _ANSWER_TO_BIT]
    return round(mean(bits), 6) if bits else None


def response_score(
    per_judge_answers: dict[str, dict[str, str | None]],
    questions: list[dict[str, str]] | None = None,
) -> float | None:
    rates = [
        r
        for r in (judge_yes_rate(a, questions) for a in per_judge_answers.values())
        if r is not None
    ]
    return round(mean(rates), 6) if rates else None


def challenger_beats_king(score_challenger: float, score_king: float) -> bool:
    return (score_challenger - score_king) >= CHALLENGER_WIN_MARGIN


def aggregate_scores(
    records: list[dict[str, Any]], *, min_valid_fraction: float = 0.8
) -> dict[str, Any]:
    total = len(records)
    valid = [r for r in records if r.get("scored")]
    valid_count = len(valid)
    judge_errors = sum(
        1
        for record in records
        for result in record.get("judge_results", [])
        if not result.get("parse_ok")
    )

    if total == 0 or valid_count / total < min_valid_fraction:
        return {
            "state": "failed",
            "score_challenger": None,
            "score_king": None,
            "challenger_won": None,
            "valid_turns": valid_count,
            "total_turns": total,
            "judge_errors": judge_errors,
            "scored_sample_count": valid_count,
            "fault_class": "PROVIDER_FAULT",
            "fault_code": "scoring_invalid",
            "fault_message": f"Only {valid_count}/{total} samples valid (< {min_valid_fraction:.0%})",  # noqa: E501
            "retryable": True,
        }

    challenger_mean = round(mean(r["challenger_score"] for r in valid), 6)
    king_mean = round(mean(r["king_score"] for r in valid), 6)

    by_judge: dict[str, float] = {}
    for judge_model in sorted(
        {
            result["judge_model"]
            for record in valid
            for result in record.get("judge_results", [])
            if result.get("side") == "challenger" and result.get("parse_ok")
        }
    ):
        rates = [
            float(result["yes_rate"])
            for record in valid
            for result in record.get("judge_results", [])
            if result.get("side") == "challenger"
            and result.get("judge_model") == judge_model
            and result.get("yes_rate") is not None
        ]
        if rates:
            by_judge[judge_model] = round(mean(rates), 6)

    return {
        "state": "succeeded",
        "score_challenger": challenger_mean,
        "score_king": king_mean,
        "challenger_won": challenger_beats_king(challenger_mean, king_mean),
        "required_win_margin": CHALLENGER_WIN_MARGIN,
        "valid_turns": valid_count,
        "total_turns": total,
        "judge_errors": judge_errors,
        "scored_sample_count": valid_count,
        "by_judge": by_judge,
        "by_metric": {},
        "scoring_mode": "binary",
        "fault_class": None,
        "fault_code": None,
        "fault_message": None,
        "retryable": None,
    }


_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\{\s*"verdict"\s*:\s*"[^"]*"[^}]*\}', re.IGNORECASE),
    re.compile(r'\{\s*"injection"\s*:\s*(true|false)[^}]*\}', re.IGNORECASE),
)
_DELIMITER_INJECTION_RE = re.compile(
    r'\s*-{3,}[\s\S]*?(?:"verdict"\s*:|GRADING\s+INSTRUCTION)[\s\S]*$',
    re.DOTALL | re.IGNORECASE,
)
_VERDICT_LABELS = frozenset({"accept", "weak_pass", "reject"})


def strip_reply_injection(reply: str) -> str:
    cleaned = _DELIMITER_INJECTION_RE.sub("", reply or "")
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    if _scan_verdict_json(cleaned):
        return ""
    return cleaned.strip()


def _scan_verdict_json(text: str) -> bool:
    decoder = json.JSONDecoder()
    start = 0
    while True:
        index = text.find("{", start)
        if index == -1:
            return False
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            start = index + 1
            continue
        if isinstance(obj, dict):
            verdict = obj.get("verdict", "")
            if isinstance(verdict, str) and verdict.lower() in _VERDICT_LABELS:
                return True
            if any(str(key).lower() == "injection" for key in obj):
                return True
        start = index + 1
