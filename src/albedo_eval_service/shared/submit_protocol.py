from __future__ import annotations

import hashlib
import random
import re

CANONICAL_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

MARKERS: dict[str, str] = {
    "canonical": CANONICAL_MARKER,
    "renamed": "FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "nearmiss": "COMPLETE_TASK_AND_SUBMIT_FINAL_ANSWER",
    "unrelated": "ALBEDO_TASK_DONE_SUBMIT_NOW",
    "nonce": "",
}

TAILS: dict[str, str] = {
    "bare": "",
    "gitdiff": " && git add -A && git diff --cached",
    "patchtxt": " && cat patch.txt",
}

ANY_MARKER_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(marker) for marker in MARKERS.values() if marker)
    + r"|SUBMIT_TASK_[0-9A-F]{8}"
    + r")\b"
)

_MARKER_SLOT = "<<MARKER>>"

_BARE_TEMPLATE = """## Submission

When you've completed your work and verified your fix, submit by running this EXACT command:

```bash
echo <<MARKER>>
```

Your changes will be automatically extracted from the repository. Do NOT create patch files manually.

<CRITICAL>
- This command must be run alone, not combined with other commands.
- Run it exactly as written above, byte-for-byte.
- You CANNOT continue working on this task after submitting.
- Make sure all your source code changes are saved before submitting.
- Do NOT commit your changes — just leave them as modified files.
</CRITICAL>"""

_GITDIFF_TEMPLATE = """## Submission
When you've completed your work (reading, editing, testing), and cannot make further progress
issue exactly the following command:

```bash
echo <<MARKER>> && git add -A && git diff --cached
```

Run it exactly as written above, byte-for-byte, from the repository root.
This command will submit your work.
You cannot continue working (reading, editing, testing) in any way on this task after submitting."""

_PATCHTXT_TEMPLATE = """## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
From the repository root, run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
echo <<MARKER>> && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- The patch file must be named exactly `patch.txt` and live in the current working directory (the repository root). Always refer to it as `patch.txt` — NEVER an absolute path like `/tmp/patch.txt` or `/workspace/<repo>/patch.txt`.
- Run the submission command exactly as written above, byte-for-byte.
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>"""

_SUBMISSION_TEMPLATES: dict[str, str] = {
    "bare": _BARE_TEMPLATE,
    "gitdiff": _GITDIFF_TEMPLATE,
    "patchtxt": _PATCHTXT_TEMPLATE,
}

_SUBMISSION_HEAD_RE = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:Submission|Submitting|Submit your work|Final Submission)\b.*$",
    re.IGNORECASE,
)
_CLOSING_TAG_RE = re.compile(r"^\s*</[A-Za-z_][A-Za-z0-9_]*>\s*$")
_HEADING_RE = re.compile(r"^#{1,4}\s+\S")
_FENCED_RE = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TAGGED_RE = re.compile(r"<([a-z_]*bash[a-z_]*)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_BACKTICK_RE = re.compile(r"`([^`\n]*)`")


_BASH_FENCE_RE = re.compile(r"```(?:bash|sh|shell)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def marker_for(name: str, sample_id: str, salt: str) -> str:
    if name != "nonce":
        return MARKERS[name]
    digest = hashlib.sha256(f"{salt}:{sample_id}".encode()).hexdigest()[:8].upper()
    return f"SUBMIT_TASK_{digest}"


def command_for(marker: str, tail_name: str) -> str:
    return f"echo {marker}{TAILS[tail_name]}"


def assign_submit(sample_id: str, salt: str) -> tuple[str, str]:
    rng = random.Random(f"{salt}:{sample_id}")
    name = rng.choice(sorted(MARKERS))
    tail_name = rng.choice(sorted(TAILS))
    marker = marker_for(name, sample_id, salt)
    return marker, command_for(marker, tail_name)


def _norm(command: str) -> str:
    return " ".join((command or "").split())


def first_bash_command(text: str) -> str:
    match = _BASH_FENCE_RE.search(text or "")
    return _norm(match.group(1)) if match else ""


def is_exact_submission(text: str, command: str) -> bool:
    return bool(command) and first_bash_command(text) == _norm(command)


def keep_original_ids(sample_ids: list[str], salt: str, keep_ratio: float) -> set[str]:
    ids = sorted(sample_ids)
    rng = random.Random(f"{salt}:keep-original")
    rng.shuffle(ids)
    keep_count = int(len(ids) * keep_ratio + 0.5)
    return set(ids[:keep_count])


def has_native_submission(messages: list[dict[str, str]]) -> bool:
    boundary = _instruction_boundary(messages)
    return any(CANONICAL_MARKER in str(m.get("content") or "") for m in messages[:boundary])


_BULLET_RE = re.compile(r"^\s*(?:[-*>]+|\d+[.)])\s*")


def _command_blocks(text: str) -> list[str]:
    blocks = [match.group(1) for match in _FENCED_RE.finditer(text or "")]
    blocks += [match.group(2) for match in _TAGGED_RE.finditer(text or "")]
    return [_norm(block) for block in blocks]


def _quoted_spans(line: str, marker: str) -> list[str]:
    spans = [
        group
        for match in _BACKTICK_RE.finditer(line)
        for group in (match.group(1),)
        if marker in group
    ]
    return spans or [_BULLET_RE.sub("", line)]


def stated_command(messages: list[dict[str, str]], marker: str = CANONICAL_MARKER) -> str:
    found: set[str] = set()
    boundary = _instruction_boundary(messages)
    for message in messages[:boundary]:
        text = str(message.get("content") or "")
        for block in _command_blocks(text):
            if marker in block:
                found.add(block)
        for line in text.splitlines():
            if marker not in line:
                continue
            for candidate in _quoted_spans(line, marker):
                command = _norm(candidate)
                if command.startswith("echo ") and marker in command:
                    found.add(command)
    return found.pop() if len(found) == 1 else ""


def marker_from(command: str) -> str:
    parts = (command or "").split()
    if len(parts) >= 2 and parts[0] == "echo":
        return parts[1]
    return CANONICAL_MARKER


def _tail_name(command: str) -> str:
    if "cat patch.txt" in command:
        return "patchtxt"
    if "git diff --cached" in command:
        return "gitdiff"
    return "bare"


def _instruction_boundary(messages: list[dict[str, str]]) -> int:
    for index, message in enumerate(messages):
        if str(message.get("role") or "") == "assistant":
            return index
    return len(messages)


def _replace_clause(text: str, command: str, marker: str) -> str:
    def fenced(match: re.Match[str]) -> str:
        if CANONICAL_MARKER not in match.group(1):
            return match.group(0)
        return f"```bash\n{command}\n```"

    def tagged(match: re.Match[str]) -> str:
        if CANONICAL_MARKER not in match.group(2):
            return match.group(0)
        return f"<{match.group(1)}>{command}</{match.group(1)}>"

    def backtick(match: re.Match[str]) -> str:
        if CANONICAL_MARKER not in match.group(1):
            return match.group(0)
        return f"`{command}`"

    text = _FENCED_RE.sub(fenced, text)
    text = _TAGGED_RE.sub(tagged, text)
    text = _BACKTICK_RE.sub(backtick, text)
    return text.replace(CANONICAL_MARKER, marker)


def _region_bounds(text: str) -> tuple[int, int, list[str]] | None:
    lines = (text or "").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if _SUBMISSION_HEAD_RE.match(line)), None
    )
    if start is None:
        return None
    end = len(lines)
    while end > start and (_CLOSING_TAG_RE.match(lines[end - 1]) or not lines[end - 1].strip()):
        end -= 1
    for index in range(start + 1, end):
        line = lines[index]
        if _HEADING_RE.match(line) and not _SUBMISSION_HEAD_RE.match(line):
            end = index
            break
    return start, end, lines


def _replace_region(text: str, template: str) -> str | None:
    bounds = _region_bounds(text)
    if bounds is None:
        return None
    start, end, lines = bounds
    rebuilt = lines[:start] + template.strip("\n").splitlines() + lines[end:]
    out = "\n".join(rebuilt)
    return out + "\n" if (text or "").endswith("\n") else out


def _verify(messages: list[dict[str, str]], command: str, marker: str) -> bool:
    boundary = _instruction_boundary(messages)
    instructions = "\n".join(str(m.get("content") or "") for m in messages[:boundary])
    if command not in instructions:
        return False
    if marker != CANONICAL_MARKER and CANONICAL_MARKER in instructions:
        return False
    return True


def rewrite_messages(
    messages: list[dict[str, str]], command: str
) -> tuple[list[dict[str, str]], str]:
    marker = marker_from(command)
    template = _SUBMISSION_TEMPLATES[_tail_name(command)].replace(_MARKER_SLOT, marker)
    boundary = _instruction_boundary(messages)
    head = [dict(m) for m in messages[:boundary]]
    rest = [dict(m) for m in messages[boundary:]]
    mode = ""
    for message in head:
        replaced = _replace_region(str(message.get("content") or ""), template)
        if replaced is not None:
            message["content"] = replaced
            mode = "replaced"
            break
    if not mode and any(CANONICAL_MARKER in str(m.get("content") or "") for m in head):
        for message in head:
            message["content"] = _replace_clause(str(message.get("content") or ""), command, marker)
        mode = "replaced_command_only"
    if not mode:
        if head:
            target = max(
                (i for i, m in enumerate(head) if str(m.get("role") or "") == "user"),
                default=len(head) - 1,
            )
            head[target]["content"] = (
                str(head[target].get("content") or "").rstrip() + "\n\n" + template
            )
        else:
            head.append({"role": "user", "content": template})
        mode = "added"
    rewritten = head + rest
    if _verify(rewritten, command, marker):
        return rewritten, mode
    return [dict(m) for m in messages], "failed"
