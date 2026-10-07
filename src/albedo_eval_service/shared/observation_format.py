from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .submit_protocol import first_bash_script

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


ROLE_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:THOUGHT:|### (?:assistant|user|system)\b)")
_ROLE_MARKER = ROLE_MARKER_RE


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


_NUMBERED_VIEW_LINE = re.compile(r"^(\s*)(\d+)(\t.*)$", re.DOTALL)
# a blank line of a `cat -n` view is its number and a tab, and the tab is easily dropped
_BLANK_VIEW_LINE = re.compile(r"^(\s+)(\d+)()$")
_SED_READ_RANGE = re.compile(r"\bsed\s+-n\s+'?(\d+),(\d+)p'?(?!\S)")
_CONTIGUOUS_VIEW = re.compile(r"\bcat\s+-n\b|\bnl\s+-ba\b")
_SPARSE_VIEW = re.compile(r"\bgrep\b|\brg\b|p;|;\s*\d+,\d+p")


def renumbered_view(command: str, raw: str) -> str:
    text = (command or "").strip()
    window, numberer = _SED_READ_RANGE.search(text), _CONTIGUOUS_VIEW.search(text)
    if not (window or numberer) or _SPARSE_VIEW.search(text):
        return raw
    # every view numbers its own lines, so views of two ranges or files are not one run
    if len(_SED_READ_RANGE.findall(text)) > 1 or len(_CONTIGUOUS_VIEW.findall(text)) > 1:
        return raw
    if OPENHANDS_TRUNCATION_NOTICE in (raw or ""):
        return raw
    if window and numberer and numberer.start() > window.start():
        window = None
    lines = (raw or "").splitlines()

    def view_line(line: str) -> re.Match | None:
        return _NUMBERED_VIEW_LINE.match(line) or (numberer and _BLANK_VIEW_LINE.match(line))

    numbered = [i for i, line in enumerate(lines) if view_line(line)]
    if not numbered:
        return raw
    first = view_line(lines[numbered[0]])
    start = int(window.group(1)) if window else int(first.group(2))
    fixed = False
    for count, i in enumerate(numbered):
        pad, number, rest = view_line(lines[i]).groups()
        want = str(start + count)
        if want != number:
            lines[i] = f"{' ' * max(len(pad) + len(number) - len(want), 1)}{want}{rest}"
            fixed = True
    return "\n".join(lines) if fixed else raw


_VIEW_LINE_PREFIX = re.compile(r"^\s*\d+[:\t]\s?|^\s*\d+\s{2,}")
_STUTTER_MIN_CHARS = 12


def stuttered_lines(raw: str) -> str:
    lines = [_VIEW_LINE_PREFIX.sub("", line).rstrip() for line in (raw or "").splitlines()]
    substantial = [len(line.strip()) >= _STUTTER_MIN_CHARS for line in lines]
    run = 1
    for i in range(1, len(lines)):
        run = run + 1 if lines[i] == lines[i - 1] and substantial[i] else 1
        if run >= 3:
            return f"the line {lines[i].strip()[:60]!r} rendered {run}x consecutively"
    for i in range(len(lines) - 3):
        a, b, c, d = lines[i : i + 4]
        if a == b != c and c == d and substantial[i] and substantial[i + 2]:
            return "consecutive lines each rendered twice (A A B B)"
    return ""


def leaked_turn(raw: str) -> bool:
    """A transcript turn served as shell output: it opens with a role marker, or it is prose that
    ends in a fenced command. A file view that merely contains a fence mid-way is not one."""
    text = (raw or "").strip()
    if _ROLE_MARKER.match(text):
        return True
    body = observation_body(raw, classify(raw))
    blocks = list(_ACTION_BLOCK_RE.finditer(body))
    return bool(blocks) and not body[blocks[-1].end() :].strip()


def echoed_command(command: str, raw: str) -> bool:
    body = observation_body(raw, classify(raw)).strip()
    body = re.sub(r"^```\w*\n|\n```$", "", body).strip()
    return bool(body) and body == (command or "").strip()


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


def wrap(body: str, fmt: str, *, returncode: int = 0, middle: tuple[str, ...] = ()) -> str:
    if fmt == RETURNCODE:
        inner = f"{body}\n" if body else ""
        return f"<returncode>{returncode}</returncode>\n<output>\n{inner}</output>"
    if fmt == SWE_AGENT:
        return f"OBSERVATION:\n{body}" if body else "OBSERVATION:"
    return "\n".join(
        [
            body,
            f"[The command completed with exit code {returncode}.]",
            *middle,
            f"[Command finished with exit code {returncode}]",
        ]
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
    r"```(?:bash|sh|shell)[ \t]*\n(.*?)```|<([a-z_]*bash[a-z_]*)>(.*?)</\2>",
    re.IGNORECASE | re.DOTALL,
)


def action_scripts(text: str) -> list[str]:
    """The commands a turn's action blocks hold, as written: what the harness would run."""
    return [
        m.group(1) if m.group(1) is not None else m.group(3)
        for m in _ACTION_BLOCK_RE.finditer(text or "")
    ]


def action_blocks(text: str) -> list[str]:
    return [" ".join(script.split()) for script in action_scripts(text)]


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
    count = len(action_blocks(text))
    if count > 1:
        return f"found {count} bash commands in the response, expected exactly one"
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


_NARRATION_RE = re.compile(
    r"^(?:I'll |I will |I need to |I should |I can see |Let me |Let's |"
    r"The user (?:wants|asked|is asking|has asked)|Now (?:I'll |I will |let me ))",
    re.IGNORECASE,
)


def narrated_observation(raw: str, fmt: str) -> bool:
    first = next(
        (line.strip() for line in observation_body(raw, fmt).splitlines() if line.strip()), ""
    )
    return bool(_NARRATION_RE.match(first))


def retry_feedback(reason: str) -> str:
    """What to tell the model so its next attempt is usable."""
    if "token limit" in reason:
        return TURN_LIMIT_FEEDBACK
    return TURN_FORMAT_FEEDBACK.format(reason=reason)


ABANDONMENT_SENTINEL = "CONSECUTIVE_BAD_TURNS_LIMIT_EXCEEDED"


def abandonment_notice(reason: str) -> str:
    return (
        f"{ABANDONMENT_SENTINEL}: turn unusable after {MAX_CONSECUTIVE_BAD_TURNS} attempts "
        f"({reason}); the benchmark abandons the instance here."
    )


def is_abandoned(text: str) -> bool:
    return ABANDONMENT_SENTINEL in (text or "")


THINK_PAIR_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
_FENCED_SPAN_RE = re.compile(r"```.*?```", re.DOTALL)
_FENCE_PLACEHOLDER = "\x00fence{}\x00"


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
    masked, fences = mask_fenced_spans(text or "")
    cleaned = THINK_PAIR_RE.sub("", masked)
    cleaned = _drop_unclosed_reasoning(cleaned, fences)
    if THINK_CLOSE_RE.search(cleaned):
        head, tail = THINK_CLOSE_RE.split(cleaned, 1)
        cleaned = head + tail if "THOUGHT:" in head else tail
    cleaned = THINK_TAG_RE.sub("", cleaned)
    return unmask_fenced_spans(cleaned, fences).strip()


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


# the simulator drifts "completed" to "finished"; unstripped, an empty observation reads as content
_TRAILER_RE = re.compile(
    r"^\s*\[(?:The command (?:completed|finished|timed out)|Current working directory|"
    r"Python interpreter|Command finished)\b.*\]\s*$"
)
_VIEW_HEADER_RE = re.compile(r"^\s*Here's the (?:result of running|files and directories)\b")


def observation_body(raw: str, fmt: str) -> str:
    # only the lines around the body go: the first line's indentation is part of the output
    text = (raw or "").rstrip().lstrip("\n")
    if fmt == RETURNCODE:
        # the tags delimit the output exactly: a blank line it starts with is part of it
        match = re.search(r"<output>\n(.*)</output>", text, re.DOTALL)
        return _strip_view_header(match.group(1).removesuffix("\n") if match else "")
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


_BARE_LINE_NUMBER = re.compile(r"^\s*\d+\s*$")


_PRINTS_NOTHING = re.compile(
    r"^(?:cd|rm|mkdir|mv|cp|touch|export|chmod|true|test|go build|python3? -m py_compile|git add"
    r"|sed\s+(?:-i|--in-place))\b"
)


_HEREDOC_START = re.compile(r"<<-?[ \t]*(['\"]?)(\w+)\1")
_STAGE_SEP = re.compile(r"&&|\|\||;|\n")


def heredoc_bodies(command: str) -> list[str]:
    """The text of every heredoc body in the command.

    A heredoc body is data - source code, JSON, prose - not shell words. Callers that scan a
    command for the files it names need to know which spans came from a body, because tokenising
    one as shell words invents paths out of identifiers such as `dt.datetime.now`.
    """
    text = command or ""
    bodies: list[str] = []
    for match in _HEREDOC_START.finditer(text):
        # the body starts on the next line: the rest of the `<<WORD` line is still shell
        start = cut = text.find("\n", match.end()) + 1
        if not start:
            continue
        for line in text[cut:].splitlines(keepends=True):
            if line.strip() == match.group(2):
                break
            cut += len(line)
        bodies.append(text[start:cut])
    return bodies


def command_stages(command: str) -> list[str]:
    return [stage for _, stage in command_chain(command)]


def command_chain(command: str) -> list[tuple[str, str]]:
    """The command's top-level stages, each with the separator that joins it to the one before.

    The first stage's separator is "". A stage that is empty only because of a line break
    (`a &&\\n b`) is dropped and the `&&` / `||` before it is kept, so the pair still reads as
    the shell does.

    Separators that are data rather than syntax are ignored. A `;` or newline inside a quoted
    argument or a heredoc body belongs to that argument, not to the shell, so both are blanked
    before the split: `python -c "a; b"` is one stage, and a heredoc that spans twenty lines is
    one stage rather than twenty.
    """
    text = command or ""
    masked = list(_unquoted(_COMMENT_LINE.sub(lambda m: m.group(0).replace("'", " "), text)))
    for match in _HEREDOC_START.finditer(text):
        # `cat <<'EOF' > a.py && python3 a.py` runs a second stage on the heredoc's own line
        start = cut = text.find("\n", match.end()) + 1
        if not start:
            continue
        for line in text[cut:].splitlines(keepends=True):
            # the terminator's own newline ends the heredoc and separates it from whatever runs
            # next, so it must stay visible to the split: masking it joins `EOF` to the command
            # on the following line and hides a whole stage
            cut += len(line.rstrip("\r\n")) if line.strip() == match.group(2) else len(line)
            if line.strip() == match.group(2):
                break
        # the body belongs to the `<<WORD` stage, and a stage after it on the opener's line runs
        # once the heredoc is written: moving that stage behind the terminator keeps both whole
        if tail := _STAGE_SEP.search("".join(masked[match.end() : start - 1])):
            split = match.end() + tail.start()
            body = text[start - 1 : cut]
            return command_chain(
                f"{text[:split].rstrip()}{body}\n{text[split : start - 1]}{text[cut:]}"
            )
        # from the newline that opens the body, so the body stays part of the `<<WORD` stage
        masked[start - 1 : cut] = "_" * (cut - start + 1)
    joined = "".join(masked)
    chain, last, glue = [], 0, ""
    for separator in _STAGE_SEP.finditer(joined):
        if stage := text[last : separator.start()].strip():
            chain.append((glue, stage))
            glue = ""
        glue = glue if glue in ("&&", "||") else separator.group()
        last = separator.end()
    if stage := text[last:].strip():
        chain.append((glue, stage))
    return chain


# a compound command, subshell, substitution or line continuation spans stages, so splitting
# the list would cut it in half
_SPANS_STAGES = re.compile(
    r"^(?:for|while|until|if|then|elif|else|fi|do|done|case|esac|function|select)\b|[(){}`]|\\$"
)


def flat_chain(command: str) -> list[tuple[str, str]] | None:
    """The stages of a flat `&&` / `||` / `;` list, comments dropped, or None when a stage is a
    compound command, subshell or substitution that the split would cut in half."""
    chain = [(glue, stage) for glue, stage in command_chain(command) if not stage.startswith("#")]
    # only a stage's first line is shell: the rest is a quoted argument or a heredoc body
    heads = [_QUOTED_SPAN_RE.sub("", stage).split("\n")[0].strip() for _, stage in chain]
    return None if any(_SPANS_STAGES.search(head) for head in heads) else chain


# a redirect onto a file swallows the stage's output; a bare heredoc says nothing about who
# consumes it, and `python <<EOF` or `tee f <<EOF` both print
_REDIRECTS_TO_FILE = re.compile(r"(?<![0-9<>])>>?[ \t]*\S")


def prints_nothing_on_success(command: str) -> bool:
    """True when every stage of `command` is a write, a navigation or a redirect: all quiet.

    Redirects are looked for with quoted spans blanked, so a `>` that is part of an argument -
    a comparison inside `python -c`, a diff marker in a heredoc - is not mistaken for one. A
    heredoc on its own is not a write: `python <<EOF` feeds a script to an interpreter that then
    prints, and `tee f <<EOF` copies its input to stdout as well as to the file.
    """
    stages = command_stages(command)
    return bool(stages) and all(
        _PRINTS_NOTHING.match(stage) or _REDIRECTS_TO_FILE.search(_unquoted(stage))
        for stage in stages
    )


_GIT_WORKTREE_QUERY = re.compile(r"\bgit\s+(?:diff|status)\b")
# a diff against a revision or the index is not a report on the working tree
_GIT_NOT_WORKTREE = re.compile(r"\bgit\s+diff\s+[^|;&]*(?:--cached|--staged|HEAD|[~^]|\.\.)")
_DIFF_HEADER = re.compile(r"^diff --git ")
_DIFF_BODY = re.compile(r"^(?:index |--- |\+\+\+ |@@ |[ +\-\\])")
_CHANGE_SECTION = re.compile(r"^changes (?:not staged for commit|to be committed):", re.I | re.M)
_CHANGE_ENTRY = re.compile(r"^\s*(?:modified|deleted|renamed|new file|typechange):\s+\S", re.I)
_SHORT_ENTRY = re.compile(r"^\s?[MADRCU][MADRCU ]?\s+\S")
_STATUS_HINT = re.compile(r"^\s*\(use ")
_DELETION_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+0,0 @@", re.M)


def claims_tracked_change(command: str, raw: str) -> bool:
    """True when git output reports a tracked file as changed.

    Diff hunks only count when the command diffs the working tree (`git log -p` and `git show`
    print hunks legitimately); `modified:` entries only count inside a status change section,
    so the words appearing in ordinary file content do not trip this."""
    text, body = command or "", raw or ""
    worktree = bool(_GIT_WORKTREE_QUERY.search(text)) and not _GIT_NOT_WORKTREE.search(text)
    if worktree and any(_DIFF_HEADER.match(line) for line in body.splitlines()):
        return True
    if "git status" in text and any(_SHORT_ENTRY.match(line) for line in body.splitlines()):
        return True
    return bool(_CHANGE_SECTION.search(body)) and any(
        _CHANGE_ENTRY.match(line) for line in body.splitlines()
    )


def deleted_files(command: str, raw: str) -> list[str]:
    """Paths a working-tree `git diff` claims were deleted or emptied outright."""
    text = command or ""
    if not _GIT_WORKTREE_QUERY.search(text) or _GIT_NOT_WORKTREE.search(text):
        return []
    body = raw or ""
    if not _DELETION_HUNK.search(body):
        return []
    return [
        m.group(1) for line in body.splitlines() if (m := re.match(r"^diff --git a/(\S+)", line))
    ]


def without_tracked_changes(raw: str, fmt: str) -> str:
    """Drop the change report from git output, keeping the branch and untracked-file lines."""
    kept: list[str] = []
    dropping = False
    for line in observation_body(raw, fmt).splitlines():
        if (
            _DIFF_HEADER.match(line)
            or _CHANGE_SECTION.match(line)
            or _CHANGE_ENTRY.match(line)
            or _SHORT_ENTRY.match(line)
        ):
            dropping = True
        elif dropping and (not line.strip() or _STATUS_HINT.match(line) or _DIFF_BODY.match(line)):
            pass  # the section's hint lines and hunk bodies go with it
        else:
            dropping = False
            kept.append(line)
    return wrap("\n".join(kept).strip("\n"), fmt)


def silent_observation(raw: str) -> bool:
    body = observation_body(raw, classify(raw)).strip()
    if not body or body == NO_OUTPUT_SENTENCE:
        return True
    return all(_BARE_LINE_NUMBER.match(line) for line in body.splitlines())


_LEADING_COMMENTS = re.compile(r"\A(?:[ \t]*#[^\n]*(?:\n|\Z))+")


def strip_leading_comments(command: str) -> str:
    """Drop the comment lines a model writes above its command.

    bash ignores them, but everything here reads the command's leading token, so a `#` in front
    makes the grounding parser see a comment instead of the grep it was handed, and the output
    expectation and contract checks misread the turn with it. A block that is nothing but
    comments keeps its text: there is no command underneath to find.
    """
    stripped = _LEADING_COMMENTS.sub("", command or "").strip()
    return stripped or (command or "").strip()


def first_bash_block(assistant_output: str) -> str:
    script = first_bash_script(assistant_output)
    return strip_leading_comments(script) if script else ""


_QUOTED_SPAN_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_COMMENT_LINE = re.compile(r"(?m)^[ \t]*#[^\n]*")
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
    r"git\s+(?:add|rm|mv|checkout|switch|restore|apply|stash|config|init|reset|diff|ls-files|tag|"
    r"clean|commit)\b|"
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


_PIP_RETRY_LINE = (
    "WARNING: Retrying (Retry(total={total}, connect=None, read=None, redirect=None, "
    "status=None)) after connection broken by 'NameResolutionError(\"HTTPSConnection("
    "host='pypi.org', port=443): Failed to resolve 'pypi.org' ([Errno -2] Name or service not "
    "known)\")': /simple/{path}/"
)
_PIP_OSERROR_LINE = (
    "ERROR: Could not install packages due to an OSError: HTTPSConnectionPool(host='pypi.org', "
    "port=443): Max retries exceeded with url: /simple/{path}/ (Caused by NameResolutionError("
    "\"Failed to resolve 'pypi.org' ([Errno -2] Name or service not known)\"))"
)
_PIP_RETRY_TOTALS = (4, 3, 2, 1, 0)
_INDEX_NAME_RE = re.compile(r"^[A-Za-z][\w.-]*$")


def _index_path(names: str) -> str:
    """The index path pip would have requested for this requirement.

    A local install (`pip install -e .`) never reaches the index for itself -- it goes looking
    for the build backend first -- so that is what the failed request names.
    """
    first = re.split(r"[<>=!~\[]", names.strip(), maxsplit=1)[0].strip()
    return first if _INDEX_NAME_RE.match(first) else "setuptools"


def _pip_unavailable(names: str) -> str:
    """What pip prints when the package index is unreachable.

    Deliberately the connection failure alone. A resolver error ("Could not find a version that
    satisfies the requirement X") reads as a bad requirement, and a model that sees it retries
    with other versions, other flags and other package names instead of abandoning pip. The
    unresolvable host is the signal that no install can work in this environment.
    """
    path = _index_path(names)
    return "\n".join(
        [_PIP_RETRY_LINE.format(total=total, path=path) for total in _PIP_RETRY_TOTALS]
        + [_PIP_OSERROR_LINE.format(path=path)]
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
# a requirement specifier and nothing else: a redirection left behind by the capture stopping at
# `&`, a flag's value, or a local path must never be reported as a package name
_REQUIREMENT_RE = re.compile(r"^[A-Za-z][\w.-]*(?:\[[\w,.-]+\])?(?:[<>=!~]=?[\w.*+-]+)?$")
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


_SHELL_DIAGNOSTIC_RE = re.compile(
    r"^(?:[\w./-]*(?:ba)?sh|[\w./-]*python[\d.]*):\s.*?"
    r"(?:command not found|No such file or directory|syntax error|No module named)\b"
)
_OBSERVED_RC_RE = re.compile(r"<returncode>(-?\d+)</returncode>")
# the exit status a scaffold reports, in the returncode format or OpenHands' trailer
_REPORTED_RC_RE = re.compile(
    r"<returncode>\s*(-?\d+)\s*</returncode>|\[Command finished with exit code (-?\d+)\]"
)


def observed_returncode(raw: str | None) -> int | None:
    """The exit status an observation reports, or None when its format shows none."""
    match = _REPORTED_RC_RE.search(raw or "")
    return int(match.group(1) or match.group(2)) if match else None


_PIPE_TAIL_RE = re.compile(r"\|\s*(?:head|tail)\b[^|&;]*$")


def pipeline_returncode_override(command: str) -> int | None:
    """The exit code the shell must report for this command shape, or None to leave it be."""
    text = _CD_PREFIX_RE.sub("", (command or "").strip())
    # only the last segment of an && / ; chain decides the exit code
    segment = re.split(r"&&|;", text)[-1].strip()
    return 0 if _PIPE_TAIL_RE.search(segment) else None


_TOOL_DIAGNOSTIC_RC = {
    "ls": 2, "grep": 2, "egrep": 2, "fgrep": 2, "rg": 2, "sed": 2, "diff": 2,
    "cat": 1, "head": 1, "tail": 1, "find": 1, "rm": 1, "mv": 1, "cp": 1,
    "stat": 1, "wc": 1, "nl": 1,
}  # fmt: skip
_TOOL_DIAGNOSTIC_RE = re.compile(
    r"^([\w./-]+): .*?"
    r"(?:cannot access|No such file or directory|cannot open|cannot stat"
    r"|Is a directory|Permission denied)$"
)
_COMMAND_WORD_RE = re.compile(r"[\w./-]+")


def tool_diagnostic_returncode(command: str, body: str) -> int | None:
    """The exit code a tool must report when its last output line is a read failure.

    The diagnostic is looked for on the *last* non-empty line, not the first: `echo x && cat
    missing` puts it there, and that chain is the shape that shows up most. The named tool has
    to appear in the command as well, so a file whose own last line happens to read like a
    diagnostic is left alone.
    """
    lines = [line for line in (body or "").splitlines() if line.strip()]
    match = _TOOL_DIAGNOSTIC_RE.match(lines[-1].strip()) if lines else None
    if match is None:
        return None
    tool = match.group(1).rsplit("/", 1)[-1]
    words = {word.rsplit("/", 1)[-1] for word in _COMMAND_WORD_RE.findall(command or "")}
    return _TOOL_DIAGNOSTIC_RC.get(tool) if tool in words else None


def correct_returncode(raw: str, fmt: str, command: str) -> str:
    """Rewrite a return code the shell could not have produced. Leaves the body untouched."""
    if fmt != RETURNCODE:
        return raw
    override = pipeline_returncode_override(command)
    if override is None:
        override = tool_diagnostic_returncode(command, observation_body(raw, fmt))
    if override is None:
        return raw
    match = _OBSERVED_RC_RE.search(raw or "")
    if match is None or int(match.group(1)) == override:
        return raw
    return _OBSERVED_RC_RE.sub(f"<returncode>{override}</returncode>", raw, count=1)


def impossible_success(raw: str, fmt: str, command: str) -> bool:
    """A shell diagnostic reported alongside a success return code.

    The shell exits nonzero for every one of these — 127 for a missing command or unresolvable
    path, 2 for a syntax error, 1 for a missing module — so returncode 0 beside one is a shape no
    terminal produces. Only the first non-empty line is tested, so a command that legitimately
    prints such text (a grep hit, a log file, a commit subject) is left alone.

    A pipeline ending in head/tail is the one shape where the pair is legitimate: the
    diagnostic comes from the failed first stage while the exit code is the filter's 0.
    """
    if fmt != RETURNCODE:
        return False
    if command and pipeline_returncode_override(command) is not None:
        return False
    match = _OBSERVED_RC_RE.search(raw or "")
    if match is None or int(match.group(1)) != 0:
        return False
    first = next((line for line in observation_body(raw, fmt).splitlines() if line.strip()), "")
    return bool(_SHELL_DIAGNOSTIC_RE.match(first.strip()))


def absent_tool_output(command: str) -> tuple[str, int] | None:
    text = command or ""
    if match := _PIP_INSTALL_RE.search(text):
        tokens = [t.strip("'\"") for t in (match.group(1) or match.group(2) or "").split()]
        # real pip names the first requirement it cannot satisfy and stops
        wanted = next((t for t in tokens if _REQUIREMENT_RE.match(t)), "the requested packages")
        return _pip_unavailable(wanted), 1
    if _PYTEST_RUN_RE.search(text):
        return PYTEST_MISSING, 1
    if match := _PY_MODULE_RE.search(text):
        module = match.group(1)
        if module.split(".")[0] != "pip":
            return _missing_module(module), 1
    return None


_NAMED_FILE_RE = re.compile(r"[\w./-]*[./][\w./-]+|\s[\w-]+\.\w{1,6}\b")


def output_expectation(command: str) -> str:
    text = _CD_PREFIX_RE.sub("", (command or "").strip())
    if not text or _WRITE_RE.search(text) or _MAY_BE_EMPTY_TAIL_RE.search(text):
        return MAY_BE_SILENT
    if _ALWAYS_PRINTS_RE.match(text):
        return MUST_PRINT
    stages = command_stages(text) or [text]
    # a named read prints wherever it sits in an && chain, not only when it comes first
    if any(_READ_HEAD_RE.match(s) and _NAMED_FILE_RE.search(s) for s in stages):
        return MUST_PRINT
    if not _READ_HEAD_RE.match(stages[0]):
        if _SILENT_RE.match(text) or _SEARCH_HEAD_RE.match(text):
            return MAY_BE_SILENT
        return NOT_DERIVABLE
    return MAY_BE_SILENT


def is_file_read(command: str) -> bool:
    """A read of a named file, whose contents the prompt can actually supply — in any stage."""
    text = _CD_PREFIX_RE.sub("", (command or "").strip())
    return any(_READ_HEAD_RE.match(stage) for stage in command_stages(text) or [text])


def requires_output(command: str) -> bool:
    return output_expectation(command) == MUST_PRINT


_SED_RANGE_RE = re.compile(r"^sed\s+-n\s+['\"][^'\"]*['\"]\s+\S+(?:\s*\|\s*(?:cat\s+-n|nl\b.*))?$")
_SED_NUM_RANGE_RE = re.compile(r"(\d+)\s*,\s*(\d+)\s*p")
_HEAD_TAIL_TAIL_RE = re.compile(r"\|\s*(head|tail)\s+-n?\s*(\d+)\s*$")
_HEAD_TAIL_ONLY_RE = re.compile(r"^(head|tail)\s+-n?\s*(\d+)\s+\S+$")


@dataclass(frozen=True)
class CommandContract:
    max_lines: int | None = None
    # `tail` keeps the last lines, so a repair has to cut from the front
    from_end: bool = False

    def __bool__(self) -> bool:
        return self.max_lines is not None

    def keep(self, lines: list[str]) -> list[str]:
        return lines[-self.max_lines :] if self.from_end else lines[: self.max_lines]


def command_contract(command: str) -> CommandContract:
    # a leading `cd` prints nothing, so it does not change what the rest may print
    masked = _unquoted(_CD_PREFIX_RE.sub("", (command or "").strip()))
    # in `a && b | tail -5` the pipe caps only `b`: `a` prints on top of it, so a chain has no cap
    if not masked or _CHAINED_RE.search(masked):
        return CommandContract()
    capped = _HEAD_TAIL_TAIL_RE.search(masked) or _HEAD_TAIL_ONLY_RE.match(masked)
    if capped:
        return CommandContract(max_lines=int(capped.group(2)), from_end=capped.group(1) == "tail")
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
    kept = contract.keep(lines)
    return raw if kept == lines else _replace_body(raw, fmt, kept)


def cap_last_stage(raw: str, fmt: str, command: str, leading_output: str | None) -> str:
    """Hold the last stage of an `&&` chain to its own trailing `| head -N` / `| tail -N`.

    The shell applies such a cap to the last stage alone, so the chain's observation is the output
    of the stages before it followed by at most N lines of the last one. `leading_output` is the
    exact output of those earlier stages, as the repo-context service ran them. The cap is applied
    to the lines after it, and only when the observation opens with exactly those lines: otherwise
    nothing marks where the last stage's output begins.
    """
    contract = command_contract(command.rsplit("&&", 1)[-1])
    if (
        leading_output is None
        or not contract
        or is_scaffold_truncated(raw)
        or not has_content(raw, fmt)
    ):
        return raw
    leading = [line.rstrip() for line in leading_output.splitlines()]
    lines = observation_body(raw, fmt).splitlines()
    if [line.rstrip() for line in lines[: len(leading)]] != leading:
        return raw
    rest = lines[len(leading) :]
    kept = contract.keep(rest)
    return raw if kept == rest else _replace_body(raw, fmt, lines[: len(leading)] + kept)


_OH_TRAILER_OPEN = re.compile(
    r"^\[The command (?:completed|finished|timed out) with exit code -?\d+\.\]$"
)
_OH_TRAILER_CLOSE = re.compile(r"^\[Command finished with exit code -?\d+\]$")
_OH_BRACKET_LINE = re.compile(r"^\[.*\]$")
# a read the openhands editor answers with its view: a whole file, or a range of its lines
_OH_NUMBERED_READ = re.compile(
    r"^(?:cat\s+-n\s+(?P<file>\S+)|sed\s+-n\s+'\d+,(?:\d+|\$)p'\s+(?P<range>\S+)\s+\|\s+cat\s+-n)$"
)
_OH_MOVES_CWD = re.compile(r"(?:^|[;&|]\s*)cd\b")
_OH_CD_PREFIX = re.compile(r"^\s*cd\s+(\S+)\s*&&")
_OH_CWD_LINE = "[Current working directory: "


def _relocated(middle: tuple[str, ...], command: str) -> tuple[str, ...] | None:
    """The trailer block once `command` has run: its working-directory line rewritten to the
    absolute path a single leading `cd` entered, or None when a relative hop, or any `cd` after
    the first, lands the session in a directory only it could resolve."""
    command = (command or "").strip()
    moves = _OH_MOVES_CWD.findall(command)
    if not moves:
        return middle
    target = _OH_CD_PREFIX.match(command)
    if target is None or not target.group(1).startswith("/") or len(moves) > 1:
        return None
    cwd = f"{_OH_CWD_LINE}{target.group(1).rstrip('/')}]"
    return tuple(cwd if line.startswith(_OH_CWD_LINE) else line for line in middle)


def openhands_trailer(messages: list[dict[str, str]] | None) -> tuple[str, ...] | None:
    """The lines this session prints between its two exit-code trailers.

    Whether a session reports its working directory and Python interpreter — and with which
    values — is a property of that session, so the block is copied from its own most recent
    shell observation rather than derived. An empty tuple means the session prints neither
    line; None means no observation has shown a trailer yet.
    """
    for message in reversed(messages or []):
        if str(message.get("role") or "").lower() not in ("user", "tool"):
            continue
        lines = str(message.get("content") or "").rstrip().splitlines()
        tail: list[str] = []
        while lines and _OH_BRACKET_LINE.match(lines[-1].strip()):
            tail.insert(0, lines.pop().strip())
        if len(tail) >= 2 and _OH_TRAILER_OPEN.match(tail[0]) and _OH_TRAILER_CLOSE.match(tail[-1]):
            return tuple(tail[1:-1])
    return None


def grounded_observation(
    fmt: str,
    body: str,
    returncode: int | None,
    command: str,
    messages: list[dict[str, str]] | None,
) -> str | None:
    """The observation for a command whose exact output is already known, or None.

    Answering straight from the computed output skips a model call that could only retype it.
    Only the body is knowable from the repository though: the wrapper a scaffold prints around
    it is itself part of what the assistant is scored against, so a turn whose wrapper cannot
    be derived from this transcript falls through to the simulator rather than being handed an
    invented one.
    """
    view = _OH_NUMBERED_READ.match((command or "").strip())
    if view and fmt != RETURNCODE:
        # the editor answers a successful numbered read with its view: openhands prints it with
        # no trailer at all, swe-agent after the scaffold's own prefix
        if returncode != 0 or not body:
            return None
        path = view.group("file") or view.group("range")
        rendered = f"Here's the result of running `cat -n` on {path}:\n{body}"
        return wrap(rendered, fmt) if fmt == SWE_AGENT else rendered
    if fmt == SWE_AGENT:
        return wrap(body, fmt)
    if fmt == RETURNCODE:
        if returncode is not None:
            return wrap(body, fmt, returncode=returncode)
        # a non-empty computed output was already answered as a success before exit codes were
        # derived; an empty one says nothing without a status, so it still needs the simulator
        return wrap(body, fmt, returncode=0) if body else None
    if returncode is None:
        return None
    middle = openhands_trailer(messages)
    if middle is not None:
        middle = _relocated(middle, command)
    if middle is None:
        return None
    return wrap(body, fmt, returncode=returncode, middle=middle)


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
