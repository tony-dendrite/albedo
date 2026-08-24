from __future__ import annotations

import posixpath
import random
import re
import shlex
from typing import Any

from loguru import logger

from albedo_eval_service.shared.json_extract import extract_json
from albedo_eval_service.shared.submit_protocol import first_bash_command

CHAIN_MICROTASK_PROMPT: str = """You design a tiny warm-up coding request for an agent working in the \
repository shown below (excerpts from its recent context).

Pick ONE real existing file and function visible in the excerpts and write a small, concrete, \
self-contained edit request that:
- is doable with 1-3 small edits by someone who has the repo,
- names the exact file and function,
- requires a NEW change: never ask for anything the excerpts already show as done or applied,
- is verifiable by reading the code afterwards, not by running it or its tests.

Return STRICT JSON only: {{"file": "<path>", "function": "<name>", "request": "<1-2 sentence instruction>", "message": "<the complete in-world message to the agent, in your own natural varied wording: it interrupts before the main issue, gives the request, and ends by telling them to submit when done by running exactly: {clause}>"}}

EDITS ALREADY APPLIED IN THIS SESSION — asking for any of these again makes the request \
impossible to satisfy, because the code already reads that way:
{applied}

REPO CONTEXT EXCERPTS:
{context}"""

_GROUNDING_RULES: str = """
HARD RULES — a message that breaks any of them is unusable:
- Name ONLY files, directories, functions and symbols that literally appear in the transcript below. \
Never invent a path, a module, or a test file. If you cannot ground a request in something visible, ask \
about the change the agent just made instead.
- The agent is allowed to modify non-test source files only. NEVER ask for a new or updated test, and \
never point at a test file as the thing to fix.
- Write only the requester's message. No THOUGHT: line, no bash block, no shell command, no markdown \
headers, no preamble."""

CHAIN_FOLLOWUP_PROMPT: str = """You simulate the in-world requester (a maintainer or CI bot) in a coding \
session. The agent just submitted work. Write the NEXT short in-world message: acknowledge briefly and give \
ONE new concrete follow-up request grounded in files or changes visible in the transcript (a CI failure \
in an adjacent area, a small extension of the change, a missed call site). Stay fully in-world; never \
mention being an AI, a simulation, or an evaluation. 2-4 sentences.
{rules}

TRANSCRIPT:
{tail}

AGENT'S SUBMISSION TURN:
{submission}"""

CHAIN_NUDGE_PROMPT: str = """You are the in-world requester in a coding session. The session must wrap up \
soon. Write ONE short message (1-2 sentences, your own natural wording, vary the phrasing) telling the \
agent to submit whatever they have right now, even if incomplete, by running exactly: {clause}
No preamble, no markdown, stay fully in-world."""

CHAIN_REJECTION_PROMPT: str = """You are the in-world requester (a maintainer or CI bot) in a coding \
session. The agent just submitted work, but you are NOT satisfied yet. Write ONE short rejection message \
(2-3 sentences, natural varied wording): point at something concrete and plausible in their submission or \
the surrounding files that needs handling before you can accept (an unhandled case, a missed call site, \
a needed docstring/comment), and ask them to address it and submit again. Stay fully in-world; \
never mention being an AI, a simulation, or an evaluation.
{rules}

TRANSCRIPT:
{tail}

AGENT'S SUBMISSION TURN:
{submission}"""

_EDIT_RE = re.compile(
    r"\b(sed -i|cat >|cat >>|tee |git apply|applypatch|str_replace|patch -p\d)"
    r"|<<\s*'?\"?[A-Za-z_][A-Za-z0-9_]*'?\"?\s*>"
    r"|<<\s*'?(EOF|PYEOF|PATCH|PY|SH|BASH|SCRIPT)\b"
    r"|\.write_text\(|\.writelines\(|open\([^)]*['\"][wa]\+?['\"]",
    re.I,
)
_MICRO_CONTEXT_CHARS = 24000


def micro_instruction(micro: dict[str, str] | str, submit_clause: str) -> str:
    # prefer the LLM-written varied message (anti-fingerprinting); template is the fallback
    if isinstance(micro, dict) and micro.get("message") and submit_clause in micro["message"]:
        return micro["message"]
    request = micro["request"] if isinstance(micro, dict) else micro
    return (
        f"Before continuing with the main issue, please handle this first: {request} "
        f"When you are done with it, submit by running exactly: {submit_clause}"
    )


def followup_instruction(followup: str, submit_clause: str, *, first: bool) -> str:
    followup = followup or (
        "Thanks, received. A reviewer asks you to double-check the change you just made "
        "against adjacent call sites and extend it if any were missed."
    )
    if first:
        return (
            f"{followup}\n\nAlso, please now continue with the original issue described "
            f"earlier. Submit the same way when done: {submit_clause}"
        )
    return f"{followup}\n\nSubmit the same way when done: {submit_clause}"


def _microtask_parsable(raw: str) -> bool:
    obj = extract_json(raw or "", prefer_keys=("request",))
    return isinstance(obj, dict) and bool(obj.get("request"))


def applied_edits(state: Any) -> str:
    """The edit commands the trajectory prefix already ran, so we never re-request them."""
    commands = [
        command
        for turn in getattr(state, "turns", ()) or ()
        if turn.get("role") == "assistant"
        for command in (first_bash_command(str(turn.get("content") or "")),)
        if command and _EDIT_RE.search(command)
    ]
    return "\n".join(f"- {c[:300]}" for c in commands) or "- (none yet)"


async def generate_microtask(
    client: Any, settings: Any, state: Any, clause: str = ""
) -> dict[str, str]:
    context = "\n\n".join(str(m.get("content") or "") for m in state.messages)[
        -_MICRO_CONTEXT_CHARS:
    ]

    result = await client.complete(
        model=settings.evaluator_model,
        messages=[
            {
                "role": "user",
                "content": CHAIN_MICROTASK_PROMPT.format(
                    context=context, clause=clause, applied=applied_edits(state)
                ),
            }
        ],
        temperature=0.7,
        accept=_microtask_parsable,
    )
    obj = extract_json(result.raw or "", prefer_keys=("request",))
    if not isinstance(obj, dict) or not obj.get("request"):
        raise ValueError(f"microtask generation unparsable for {state.sample_id}")
    return {k: str(obj.get(k) or "") for k in ("file", "function", "request", "message")}


META_LEAK_RE = re.compile(
    r"\bas an AI\b"
    r"|\bAI (?:assistant|language model)\b"
    r"|\blanguage model\b"
    r"|\bI(?:'m| am) (?:an? )?(?:AI|bot|simulat\w+)\b"
    r"|\bthe (?:user|agent|assistant) is (?:not )?an? AI\b"
    r"|\bI (?:cannot|can'?t|don'?t) actually (?:run|execute|access|see)\b"
    r"|\b(?:this|it) is (?:just )?a (?:simulation|simulated|test scenario|roleplay)\b"
    r"|\bsimulated (?:environment|shell|session|observation)\b"
    r"|\b(?:in|for) this (?:evaluation|benchmark|exercise)\b"
    r"|\bpretend(?:ing)? to be\b"
    r"|\brole[- ]?play(?:ing)?\b"
    r"|\bI(?:'m| am) (?:only )?(?:simulating|generating|producing) (?:the |an? )?(?:output|response|observation)\b",
    re.I,
)

_ROLE_LEAK_RE = re.compile(
    r"```|^\s*THOUGHT\s*:"
    r"|\bI(?:'ve|\s+have)?\s+(?:fixed|added|implemented|applied|updated|refactored|corrected)\b",
    re.MULTILINE | re.IGNORECASE,
)
_TEST_DEMAND_RE = re.compile(
    r"\b(regression test|unit test|add (a |an )?test|new test|write (a |an )?test"
    r"|test coverage|cover(ed|age)? by (a |any )?test)\b",
    re.I,
)
_SOURCE_EXT = (
    "py|pyi|go|rs|js|jsx|ts|tsx|rb|java|kt|scala|c|cc|cpp|cxx|h|hpp|cs|php|swift|m|mm|ex|exs"
    "|erl|hs|lua|pl|pm|sh|bash|sql|proto|toml|cfg|ini|yaml|yml|json|xml|md|rst|txt"
)
_MSG_PATH_RE = re.compile(
    rf"\b(?:[\w.-]+/)+[\w.-]+\.\w{{1,6}}\b|\b[\w-]+\.(?:{_SOURCE_EXT})\b", re.I
)
_MIN_TRIGRAM_DIVERSITY = 0.6


def chain_context(state: Any) -> str:
    """Everything the simulated requester is allowed to reference: the whole visible session."""
    return "\n\n".join(str(m.get("content") or "") for m in state.messages)[-_MICRO_CONTEXT_CHARS:]


def ungrounded_reason(message: str, context: str, clause: str = "") -> str:
    """Why an in-world message is unusable, or '' when it is safe to inject."""
    text = (message or "").strip()
    context = f"{context}\n{clause}"
    if not text:
        return "empty"
    if _ROLE_LEAK_RE.search(text):
        return "written as an agent turn (shell block or THOUGHT: line)"
    if META_LEAK_RE.search(text):
        return "breaks character (mentions being an AI, a simulation or an evaluation)"
    if _TEST_DEMAND_RE.search(text):
        return "asks for test changes the task instructions forbid"
    tokens = text.split()
    if len(tokens) > 20:
        trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
        if len(set(trigrams)) / len(trigrams) < _MIN_TRIGRAM_DIVERSITY:
            return "degenerate repetitive text"
    invented = [
        path
        for path in {m.group(0) for m in _MSG_PATH_RE.finditer(text)}
        if path.rsplit("/", 1)[-1] not in context
    ]
    if invented:
        return f"references files absent from the session: {sorted(invented)[:3]}"
    return ""


async def _generate_in_world(
    client: Any,
    settings: Any,
    state: Any,
    *,
    prompt: str,
    fallback: str,
    kind: str,
) -> str:
    context = chain_context(state)
    clause = str(getattr(state, "submit_clause", "") or "")
    try:
        result = await client.complete(
            model=settings.simulation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            accept=lambda raw: not ungrounded_reason(raw, context, clause),
        )
    except Exception as exc:
        logger.warning("[sanity/chain] {} generation failed for {}: {}", kind, state.sample_id, exc)
        return fallback
    message = (result.raw or "").strip()
    reason = ungrounded_reason(message, context, clause)
    if reason:
        logger.warning(
            "[sanity/chain] {} discarded for {} ({}), using fallback",
            kind,
            state.sample_id,
            reason,
        )
        return fallback
    return message


async def generate_followup(client: Any, settings: Any, state: Any, submission: str) -> str:
    return await _generate_in_world(
        client,
        settings,
        state,
        prompt=CHAIN_FOLLOWUP_PROMPT.format(
            rules=_GROUNDING_RULES,
            tail=chain_context(state),
            submission=submission[-2000:],
        ),
        # an ungroundable follow-up is no follow-up: the caller accepts the submission
        fallback="",
        kind="followup",
    )


def segment_has_edit(state: Any, segment: str) -> bool:
    return any(
        _EDIT_RE.search(str(t.get("content") or ""))
        for t in state.turns
        if t.get("role") == "assistant" and t.get("segment") == segment
    )


def micro_target_touched(state: Any, micro: dict[str, str]) -> bool:
    needles = [
        n for n in ((micro.get("file") or "").rsplit("/", 1)[-1], micro.get("function") or "") if n
    ]
    return any(
        needle in str(t.get("content") or "")
        for t in state.turns
        if t.get("role") == "assistant" and t.get("segment") == "micro"
        for needle in needles
    )


def amputated_thinking(state: Any) -> bool:
    scored = [
        str(t.get("content") or "")
        for t in state.turns
        if t.get("role") == "assistant" and t.get("score_target")
    ]
    if not scored:
        return False
    return sum(1 for t in scored if t.lstrip().startswith("</think>")) / len(scored) > 0.5


SUBMIT_NUDGE = (
    "We need to wrap up this session soon. Please submit whatever you have right now, "
    "even if it is incomplete, by running exactly: {clause}"
)


_NAME_RE = re.compile(r"[A-Za-z_][\w.\-/]{3,}")


def _names(text: str) -> set[str]:
    return {
        t.lower()
        for t in _NAME_RE.findall(text or "")
        if any(c in t for c in "_./-") or not t.islower()
    }


def empty_submit_count(state: Any, marker: str) -> int:
    count = 0
    worked = False
    asked_to_submit_as_is = False
    asked: set[str] = set()
    for turn in state.turns:
        content = str(turn.get("content") or "")
        if turn.get("role") != "assistant":
            asked_to_submit_as_is = bool(turn.get("nudge"))
            if not turn.get("environment_observation"):
                asked = _names(content)
            continue
        command = first_bash_command(content)
        if not marker or marker not in command:
            worked = worked or bool(_EDIT_RE.search(content)) or bool(_names(command) & asked)
            continue
        worked = worked or bool(_EDIT_RE.search(content))
        count += not worked and not asked_to_submit_as_is
        worked = False
    return count


async def generate_nudge(client: Any, settings: Any, clause: str) -> str:
    try:
        result = await client.complete(
            model=settings.simulation_model,
            messages=[{"role": "user", "content": CHAIN_NUDGE_PROMPT.format(clause=clause)}],
            temperature=0.7,
        )
        nudge = (result.raw or "").strip()
        if nudge and clause in nudge:
            return nudge
    except Exception as exc:
        logger.warning("[sanity/chain] nudge generation failed: {}", exc)
    return SUBMIT_NUDGE.format(clause=clause)


REJECTION_PROB = 0.34
MAX_REJECTIONS = 2


def should_reject(state: Any, rng: random.Random | None = None) -> bool:
    if sum(1 for sub in getattr(state, "submits", ()) if sub.get("rejected")) >= MAX_REJECTIONS:
        return False
    return (rng or random).random() < REJECTION_PROB


async def generate_rejection(client: Any, settings: Any, state: Any, submission: str) -> str:
    return await _generate_in_world(
        client,
        settings,
        state,
        prompt=CHAIN_REJECTION_PROMPT.format(
            rules=_GROUNDING_RULES,
            tail=chain_context(state),
            submission=submission[-2000:],
        ),
        fallback="",
        kind="rejection",
    )


_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)```", re.DOTALL)
_SEPARATORS = {"&&", "||", ";", "|", "&"}
_REPO_PATH_RE = re.compile(r"^(?:[\w.-]+/)+[\w.-]+\.[\w]+$")


def inplace_edited_paths(text: str) -> list[str]:
    """Repo paths edited in place by `sed -i` in the bash blocks of one assistant turn.

    shlex keeps the quoted sed script as a single token, so dropping flags and the script
    leaves the file operands. The script must never be read as a path: `sed -i 's/a/Cls.a/'
    pkg/mod.py` used to match `s/a/Cls.a` because it has slashes and a dot.
    """
    paths: list[str] = []
    for block in _BASH_BLOCK_RE.findall(text or ""):
        try:
            tokens = shlex.split(block)
        except ValueError:  # unbalanced quotes — no reliable operands to extract
            continue
        for command in _split_on(tokens, _SEPARATORS):
            if not command or posixpath.basename(command[0]) != "sed":
                continue
            flags = [a for a in command[1:] if a.startswith("-")]
            operands = [a for a in command[1:] if not a.startswith("-")]
            script_is_flag_arg = any(set("ef") & set(f[1:]) for f in flags)
            if not any("i" in f[1:] for f in flags):
                continue
            paths += operands if script_is_flag_arg else operands[1:]
    return [p for p in paths if _REPO_PATH_RE.match(p)]


def _split_on(tokens: list[str], separators: set[str]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    for token in tokens:
        out.append([]) if token in separators else out[-1].append(token)
    return out


def unread_edited_files(state: Any) -> list[str]:
    """In-place-edited repo files that never appeared in an earlier turn or observation."""
    seen, unread = [], []
    for t in state.turns:
        content = str(t.get("content") or "")
        if t.get("role") == "assistant" and t.get("score_target"):
            for path in inplace_edited_paths(content):
                if not path.startswith(("/tmp", "tmp/")) and not any(
                    path.rsplit("/", 1)[-1] in prev for prev in seen
                ):
                    unread.append(path)
        seen.append(content)
    return unread


_RESERVED_TOKEN_RE = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|vision_start\|>|<\|vision_end\|>|<\|image_pad\|>"
    r"|<\|video_pad\|>|<\|endoftext\|>|</?tool_call>|<function=|</?tool_response>"
)


def malformed_structure(state: Any) -> str:
    for t in state.turns:
        if t.get("role") == "assistant" and t.get("score_target"):
            hit = _RESERVED_TOKEN_RE.search(str(t.get("content") or ""))
            if hit:
                return f"reserved template token in output: {hit.group(0)}"
    return ""
