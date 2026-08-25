from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

RETURNCODE = "returncode"
SWE_AGENT = "swe_agent"
OPENHANDS = "openhands"

TRUNCATION_SENTINEL = "MODEL_RESPONSE_TOKEN_LIMIT_EXCEEDED"
UNCLOSED_THINK_BLOCK_SENTINEL = "MODEL_UNCLOSED_THINK_BLOCK"

OPENHANDS_TRUNCATION_NOTICE = "[... Observation truncated due to length ...]"
_SCAFFOLD_TRUNCATED = re.compile(
    r"<warning>|<output_head>|<response clipped>|output of your last command was too long|"
    + re.escape(OPENHANDS_TRUNCATION_NOTICE),
    re.I,
)


def is_scaffold_truncated(raw: str) -> bool:
    return bool(_SCAFFOLD_TRUNCATED.search(raw or ""))


def classify(observation: str) -> str:
    text = (observation or "").lstrip()
    if text.startswith("<returncode>"):
        return RETURNCODE
    if text.startswith("OBSERVATION:"):
        return SWE_AGENT
    return OPENHANDS


def detect_format(sample_id: str, messages: list[dict[str, str]] | None = None) -> str:
    observation = _first_observation(messages)
    if observation is not None:
        return classify(observation)
    return RETURNCODE if "mini-coder" in sample_id.casefold() else OPENHANDS


def _first_observation(messages: list[dict[str, str]] | None) -> str | None:
    seen_assistant = False
    for message in messages or []:
        role = str(message.get("role") or "").lower()
        if role == "assistant":
            seen_assistant = True
        elif role in ("user", "tool") and seen_assistant:
            content = str(message.get("content") or "")
            if content.strip():
                return content
    return None


_ROLE_MARKER = re.compile(r"(?:^|\n)\s*(?:THOUGHT:|### (?:assistant|user|system)\b)")


def valid_output(raw: str, fmt: str) -> bool:
    text = (raw or "").strip()
    if not text:
        return False
    if fmt == RETURNCODE:
        return (
            text.startswith("<returncode>")
            and "</returncode>" in text
            and "<output>\n" in text
            and text.endswith("\n</output>")
        )
    if fmt == SWE_AGENT:
        return text.startswith("OBSERVATION:")
    return not text.startswith(("<returncode>", "Observation:", "OBSERVATION:"))


def repair_output(raw: str, fmt: str) -> str:
    text = (raw or "").strip()
    if fmt != RETURNCODE or not text.startswith("<returncode>"):
        marker = _ROLE_MARKER.search(text)
        if marker and text[: marker.start()].strip():
            return text[: marker.start()].rstrip()
        return text
    if "<output>" in text and "<output>\n" not in text:
        text = text.replace("<output>", "<output>\n", 1)
    end = text.find("</output>")
    if end != -1:
        text = text[: end + len("</output>")]
    if text.endswith("</output>") and not text.endswith("\n</output>"):
        text = text[: -len("</output>")].rstrip("\n") + "\n</output>"
    return text


def wrap(body: str, fmt: str, *, returncode: int = 0) -> str:
    if fmt == RETURNCODE:
        inner = f"{body}\n" if body else ""
        return f"<returncode>{returncode}</returncode>\n<output>\n{inner}</output>"
    if fmt == SWE_AGENT:
        return f"OBSERVATION:\n{body}" if body else "OBSERVATION:"
    return (
        f"{body}\n[The command completed with exit code {returncode}.]\n"
        f"[Command finished with exit code {returncode}]"
    )


def empty_output(fmt: str) -> str:
    return wrap("", fmt)


def truncation_notice(token_limit: int) -> str:
    return (
        f"{TRUNCATION_SENTINEL}: Model returned over {token_limit} tokens in one response, "
        "stopping conversation."
    )


def is_truncated(text: str) -> bool:
    return TRUNCATION_SENTINEL in (text or "")


def unclosed_think_block_notice() -> str:
    return (
        f"{UNCLOSED_THINK_BLOCK_SENTINEL}: Model returned an unclosed think block, "
        "stopping conversation."
    )


def has_unclosed_think_block(text: str) -> bool:
    return UNCLOSED_THINK_BLOCK_SENTINEL in (text or "")


# The benchmark harness does not end a run on a malformed turn: it drops the turn, tells the
# model what went wrong and asks again, giving up only after this many consecutive failures
# (mini-swe-agent max_consecutive_format_errors=3). Pre-eval and scoring both mirror it.
MAX_CONSECUTIVE_BAD_TURNS = 3

# both harness syntaxes: our ```bash fence and the benchmark's <mswea_bash_command>
_ACTION_RE = re.compile(
    r"```(?:bash|sh|shell)\s*\n.*?```|<([a-z_]*bash[a-z_]*)>.*?</\1>", re.IGNORECASE | re.DOTALL
)

TURN_LIMIT_FEEDBACK = (
    "Your previous response reached the output token limit before you produced a complete "
    "action, so it was cut off. Respond more concisely and provide exactly one action in the "
    "required format. If you need to think more, do so briefly."
)

TURN_FORMAT_FEEDBACK = """Format error:

<error>
{reason}
</error>

Please always provide EXACTLY ONE action in a ```bash code block, as shown in <example>.

<example>
THOUGHT: Here are some thoughts about why you want to perform the action.

```bash
ls -la
```
</example>

If you have completed your assignment, consult the first message about how to submit."""


_ACTION_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```|<([a-z_]*bash[a-z_]*)>(.*?)</\2>",
    re.IGNORECASE | re.DOTALL,
)


def action_blocks(text: str) -> list[str]:
    return [
        " ".join((m.group(1) if m.group(1) is not None else m.group(3)).split())
        for m in _ACTION_BLOCK_RE.finditer(text or "")
    ]


def unusable_turn(text: str, *, truncated: bool = False) -> str:
    """Why this turn carries no usable action, or '' when it is fine.

    Covers the four ways a turn comes back unusable: cut off at the token limit, an unclosed
    think block, nothing at all, or prose with no command in it.
    """
    if truncated or is_truncated(text):
        return "response exceeded the model response token limit"
    if has_unclosed_think_block(text):
        return "response contains an unclosed think block"
    if not (text or "").strip():
        return "empty response"
    if not _ACTION_RE.search(text):
        return "no bash command found in the response"
    return ""


_LEADING_LINE_NO = re.compile(r"^\s*\d+\s+")
_DIGIT_RUN = re.compile(r"\d+")
_MIN_LINES_FOR_DEGENERACY = 10
_TOP_LINES_COUNTED = 3
_MAX_LINE_SHARE = 0.8


def degenerate_observation(text: str) -> bool:
    lines = [
        _DIGIT_RUN.sub("#", _LEADING_LINE_NO.sub("", line)).strip()
        for line in (text or "").splitlines()
    ]
    lines = [line for line in lines if line]
    if len(lines) < _MIN_LINES_FOR_DEGENERACY:
        return False
    repeated = sum(count for _, count in Counter(lines).most_common(_TOP_LINES_COUNTED))
    return repeated / len(lines) >= _MAX_LINE_SHARE


def retry_feedback(reason: str) -> str:
    """What to tell the model so its next attempt is usable."""
    if "token limit" in reason:
        return TURN_LIMIT_FEEDBACK
    return TURN_FORMAT_FEEDBACK.format(reason=reason)


THINK_PAIR_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
_FENCED_SPAN_RE = re.compile(r"```.*?```", re.DOTALL)
_FENCE_PLACEHOLDER = "\x00fence{}\x00"


def mask_fenced_spans(text: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def _take(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return _FENCE_PLACEHOLDER.format(len(spans) - 1)

    return _FENCED_SPAN_RE.sub(_take, text or ""), spans


def unmask_fenced_spans(text: str, spans: list[str]) -> str:
    for index, span in enumerate(spans):
        text = text.replace(_FENCE_PLACEHOLDER.format(index), span)
    return text


_TRAILER_RE = re.compile(
    r"^\s*\[(?:The command (?:completed|timed out)|Current working directory|"
    r"Python interpreter|Command finished)\b.*\]\s*$"
)
_VIEW_HEADER_RE = re.compile(r"^\s*Here's the (?:result of running|files and directories)\b")


def observation_body(raw: str, fmt: str) -> str:
    text = (raw or "").strip()
    if fmt == RETURNCODE:
        match = re.search(r"<output>\n(.*)\n?</output>", text, re.DOTALL)
        return _strip_view_header(match.group(1).strip("\n") if match else "")
    if fmt == SWE_AGENT:
        if not text.startswith("OBSERVATION:"):
            return _strip_view_header(text)
        return _strip_view_header(text[len("OBSERVATION:") :].strip("\n"))
    kept = [line for line in text.splitlines() if not _TRAILER_RE.match(line)]
    return _strip_view_header("\n".join(kept).strip("\n"))


def _strip_view_header(body: str) -> str:
    lines = body.splitlines()
    return "\n".join(lines[1:]).strip("\n") if lines and _VIEW_HEADER_RE.match(lines[0]) else body


def has_content(raw: str, fmt: str) -> bool:
    return bool(observation_body(raw, fmt).strip())


NO_OUTPUT_SENTENCE = "Your command ran successfully and did not produce any output."


def canonical_empty(raw: str, fmt: str) -> str:
    body = observation_body(raw, fmt).strip()
    return empty_output(fmt) if body == NO_OUTPUT_SENTENCE else raw


def silent_observation(raw: str) -> bool:
    body = observation_body(raw, classify(raw)).strip()
    return not body or body == NO_OUTPUT_SENTENCE


_FIRST_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)
_TAGGED_BLOCK_RE = re.compile(r"```(?:bash|sh)[ \t]*\n(.*?)```", re.DOTALL)


def first_bash_block(assistant_output: str) -> str:
    match = _TAGGED_BLOCK_RE.search(assistant_output or "") or _FIRST_BLOCK_RE.search(
        assistant_output or ""
    )
    return match.group(1).strip() if match else ""


_QUOTED_SPAN_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_CHAINED_RE = re.compile(r"&&|\|\||;|\n")
_READ_HEAD_RE = re.compile(r"^\s*(?:cat|nl|head|tail|less|more|sed\s+-n)\b")
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+\S+\s*&&\s*")
_ALWAYS_PRINTS_RE = re.compile(
    r"^\s*(?:git\s+(?:log|show|status|branch)\b|ls\b|pwd\b|tree\b|which\s+\S|wc\s+\S|echo\s+\S)"
)
_WRITE_RE = re.compile(r"(?<![0-9<>])>>?[ \t]*\S|<<-?[ \t]*['\"]?\w")
_SEARCH_HEAD_RE = re.compile(
    r"^\s*(?:grep|rg|egrep|fgrep|ag|ack|find|awk|diff|comm|cut|tr|sort|uniq|xargs|"
    r"git\s+grep)\b"
)
_SILENT_RE = re.compile(
    r"^\s*(?:sed\s+-i|tee\b|touch\b|mkdir\b|rmdir\b|rm\b|mv\b|cp\b|ln\b|chmod\b|chown\b|"
    r"export\b|unset\b|cd\b|pushd\b|popd\b|true\b|:\s*$|"
    r"git\s+(?:add|rm|mv|checkout|switch|restore|apply|stash|config|init|reset)\b|"
    r"apply_patch\b|patch\s+-p)"
)
_MAY_BE_EMPTY_TAIL_RE = re.compile(
    r"\|[ \t]*(?:grep|rg|ag|ack|egrep|fgrep|awk|find|comm|diff|uniq|sort[ \t]+-u)\b"
)


def _unquoted(text: str) -> str:
    """Blank quoted spans so metacharacters inside arguments are not read as chaining:
    `sed -n '1,5p;10,12p' f.py` is one command."""
    return _QUOTED_SPAN_RE.sub(lambda m: "'" + "_" * (len(m.group(0)) - 2) + "'", text)


MUST_PRINT = "must_print"
MAY_BE_SILENT = "may_be_silent"
NOT_DERIVABLE = "not_derivable"


def _missing_module(name: str) -> str:
    return f"/opt/conda/bin/python: No module named {name}"


def _pip_unavailable(names: str) -> str:
    return (
        f"ERROR: Could not find a version that satisfies the requirement {names} "
        "(from versions: none)\n"
        f"ERROR: No matching distribution found for {names}"
    )


PYTEST_MISSING = _missing_module("pytest")
PIP_PYTEST_ABSENT = _pip_unavailable("pytest")

_STAGE = r"(?:^|[;&|(]|&&|\|\|)\s*"
_PYTEST_RUN_RE = re.compile(
    _STAGE + r"(?:py\.test|pytest)(?![\w.-])"
    r"|" + _STAGE + r"[\w./-]*python[\d.]*\s+-m\s+pytest(?![\w.-])"
)
_PIP_INSTALL_RE = re.compile(
    r"\bpip[\d.]*\s+install\b([^;&|]*)"
    r"|\bpython[\d.]*\s+-m\s+pip\s+install\b([^;&|]*)"
)
_PY_MODULE_RE = re.compile(_STAGE + r"[\w./-]*python[\d.]*\s+-m\s+([A-Za-z_][\w.]*)")
_IMPORT_RE = re.compile(r"\bimport\s+([A-Za-z_][\w.]*)")
_SCRIPT_RE = re.compile(r"\b([\w-]+)\.py\b")


_PY_HEAD_RE = re.compile(r"[\w./-]*python[\d.]*\b")


def no_output_notice(command: str = "") -> tuple[str, int]:
    """What a command we cannot run reports, shaped like the pytest refusal.

    It has to read as ordinary terminal output: a note about the session would tell the
    model it is being simulated, and would never appear in a recorded trajectory. The same
    command always gets the same failure, so retrying cannot look like progress.
    """
    text = _CD_PREFIX_RE.sub("", (command or "").strip())
    if _PY_HEAD_RE.match(text):
        for pattern in (_PY_MODULE_RE, _IMPORT_RE, _SCRIPT_RE):
            if match := pattern.search(text):
                return _missing_module(match.group(1).split(".")[0]), 1
        return _missing_module("__main__"), 1
    head = re.match(r"[\w./-]+", text)
    return f"bash: {head.group(0) if head else text}: command not found", 127


def absent_tool_output(command: str) -> tuple[str, int] | None:
    text = command or ""
    if match := _PIP_INSTALL_RE.search(text):
        wanted = [
            token
            for token in (match.group(1) or match.group(2) or "").split()
            if not token.startswith("-")
        ]
        return _pip_unavailable(" ".join(wanted) or "the requested packages"), 1
    if _PYTEST_RUN_RE.search(text):
        return PYTEST_MISSING, 1
    if match := _PY_MODULE_RE.search(text):
        return _missing_module(match.group(1)), 1
    return None


def output_expectation(command: str) -> str:
    text = _CD_PREFIX_RE.sub("", (command or "").strip())
    if not text or _WRITE_RE.search(text) or _MAY_BE_EMPTY_TAIL_RE.search(text):
        return MAY_BE_SILENT
    if _ALWAYS_PRINTS_RE.match(text):
        return MUST_PRINT
    if not _READ_HEAD_RE.match(text.split("&&")[0]):
        if _SILENT_RE.match(text) or _SEARCH_HEAD_RE.match(text):
            return MAY_BE_SILENT
        return NOT_DERIVABLE
    named = re.search(r"[\w./-]*[./][\w./-]+|\s[\w-]+\.\w{1,6}\b", text.split("&&")[0])
    return MUST_PRINT if named else MAY_BE_SILENT


def is_file_read(command: str) -> bool:
    """A read of a named file, whose contents the prompt can actually supply."""
    return bool(_READ_HEAD_RE.match(_CD_PREFIX_RE.sub("", (command or "").strip())))


def requires_output(command: str) -> bool:
    return output_expectation(command) == MUST_PRINT


_SED_RANGE_RE = re.compile(r"^sed\s+-n\s+['\"][^'\"]*['\"]\s+\S+(?:\s*\|\s*(?:cat\s+-n|nl\b.*))?$")
_SED_NUM_RANGE_RE = re.compile(r"(\d+)\s*,\s*(\d+)\s*p")
_HEAD_TAIL_TAIL_RE = re.compile(r"\|\s*(?:head|tail)\s+-n?\s*(\d+)\s*$")
_HEAD_TAIL_ONLY_RE = re.compile(r"^(?:head|tail)\s+-n?\s*(\d+)\s+\S+$")


@dataclass(frozen=True)
class CommandContract:
    max_lines: int | None = None

    def __bool__(self) -> bool:
        return self.max_lines is not None


def command_contract(command: str) -> CommandContract:
    masked = _unquoted((command or "").strip())
    if not masked:
        return CommandContract()
    capped = _HEAD_TAIL_TAIL_RE.search(masked)  # a trailing pipe caps whatever precedes it
    if capped:
        return CommandContract(max_lines=int(capped.group(1)))
    if _CHAINED_RE.search(masked):
        return CommandContract()
    only = _HEAD_TAIL_ONLY_RE.match(masked)
    if only:
        return CommandContract(max_lines=int(only.group(1)))
    if _SED_RANGE_RE.match(masked):
        ranges = _SED_NUM_RANGE_RE.findall(command)
        if ranges:
            return CommandContract(max_lines=sum(int(b) - int(a) + 1 for a, b in ranges))
    return CommandContract()


def contract_violation(raw: str, fmt: str, contract: CommandContract) -> str | None:
    if not contract or is_scaffold_truncated(raw):
        return None
    lines = observation_body(raw, fmt).splitlines()
    if lines and len(lines) > contract.max_lines:
        return f"too_many_lines:{len(lines)}>{contract.max_lines}"
    return None


_RC_OUTPUT_RE = re.compile(r"(<returncode>\d+</returncode>\n<output>\n).*\n?</output>", re.DOTALL)


def repair_to_contract(raw: str, fmt: str, contract: CommandContract) -> str:
    if not contract or is_scaffold_truncated(raw) or not has_content(raw, fmt):
        return raw
    lines = observation_body(raw, fmt).splitlines()
    kept = lines[: contract.max_lines]
    return raw if kept == lines else _replace_body(raw, fmt, kept)


def with_body(raw: str, fmt: str, body: str) -> str:
    return _replace_body(raw, fmt, body.split("\n"))


def _replace_body(raw: str, fmt: str, lines: list[str]) -> str:
    body, text = "\n".join(lines), (raw or "").strip()
    if fmt == RETURNCODE:
        match = _RC_OUTPUT_RE.search(text)
        if match is None:
            return wrap(body, fmt)
        # an empty body must not leave a stray blank line behind
        return f"{match.group(1)}{body}\n</output>" if body else f"{match.group(1)}</output>"
    if fmt == SWE_AGENT:
        return f"OBSERVATION:\n{body}" if body else "OBSERVATION:"
    rest, head, tail = text.splitlines(), [], []
    if rest and _VIEW_HEADER_RE.match(rest[0]):
        head, rest = rest[:1], rest[1:]
    while rest and _TRAILER_RE.match(rest[-1]):
        tail.insert(0, rest.pop())
    return "\n".join(head + lines + tail)
