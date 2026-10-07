from __future__ import annotations

import re
from collections.abc import Iterable

from .observation_format import action_scripts

WORK_EDIT_RE = re.compile(
    r"\b(sed -i|cat >|cat >>|tee |git apply|applypatch|str_replace|patch -p\d|cp |mv )"
    r"|<<\s*'?\"?[A-Za-z_][A-Za-z0-9_]*'?\"?\s*>"
    r"|<<\s*'?(EOF|PYEOF|PATCH|PY|SH|BASH|SCRIPT)\b"
    r"|\.write_text\(|\.writelines\(|open\([^)]*['\"][wa]\+?['\"]"
    r"|(?<=\s)>>?\s*(?!/dev/)[~$\w./-]*(?:/[~$\w.-]+|\.[A-Za-z]\w*)",
    re.I,
)

# A command changes the repository when it writes somewhere other than these: device files,
# scratch space and the home directory are not part of the checkout.
_SCRATCH = r"(?:/dev/|/tmp/|/var/tmp/|~)"
_ARGS = r"[^|;&\n]*"  # the rest of one simple command
# shell-level writes, read with here-document bodies and quoted text masked (`_shell_text`), so a
# `>` inside a python script or a grep pattern is not taken for a redirect
_SHELL_EDIT_RE = re.compile(
    rf"(?<![-=0-9<>|])>>?[ \t]*(?!{_SCRATCH})(?=[\w./~-]*[A-Za-z_])[\w./-]"
    rf"|\btee\s+(?:-a\s+|--append\s+)?(?!{_SCRATCH})[\w./-]"
    rf"|\bsed\b{_ARGS}\s(?:-[a-zA-Z]*i|--in-place)|\b(?:perl|ruby)\s+-[a-zA-Z]*i"
    rf"|\b(?:cp|mv)\s+(?:-\w+\s+)*[\w./-]+\s+(?!{_SCRATCH})[\w./-]"
    rf"|\bgit\s+apply\b(?!{_ARGS}--check)|\bpatch\b(?!{_ARGS}--dry-run){_ARGS}\s-p"
    r"|\bapply_?patch\b"
)
# writes from inside a script (python here-documents, `python -c`), read over the whole command
_SCRIPT_EDIT_RE = re.compile(
    rf"\bopen\s*\((?!\s*[rbf]?['\"]{_SCRATCH})[^\n]*?['\"](?:[wax]|r(?=[bt]?\+))[bt]?\+?['\"]"
    r"|\.write_text\s*\(|\.write_bytes\s*\(|\bfileinput\.input\([^)]*inplace"
    r"|\bshutil\.(?:copy|copyfile|copy2|move)\s*\(|\bos\.(?:replace|rename)\s*\("
)
# the renderer writes an editor `str_replace` call as a SEARCH/REPLACE block, not a command
_RENDERED_EDIT_RE = re.compile(
    rf"^Editing `(?!{_SCRATCH})[^`\n]+`:\n\n```\n<<<<<<< SEARCH\n.*^>{{7}} REPLACE[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_INTERPRETER_RE = re.compile(r"\b(?:python[\d.]*|pypy[\d.]*)\b")
# a here-document with the line that opens it (`head`) and its body, up to the closing word
_HEREDOC_RE = re.compile(
    r"^(?P<head>[^\n]*?(?<!<)<<(?!<)-?[ \t]*(['\"]?)(\w+)\2[^\n]*)\n"
    r".*?(?:\n[ \t]*\3[ \t]*(?=\n|$)|\Z)",
    re.MULTILINE | re.DOTALL,
)
_QUOTED_RE = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")

REMOVAL_RE = re.compile(r"\brm\b|\bgit\s+(?:rm|checkout|stash)\b|\bmv\b")

_FILE_TOKEN = re.compile(r"[\w./-]*[\w-]+\.[A-Za-z]\w*")


def _drop_heredocs(script: str, *, keep_run: bool) -> str:
    """The command without its here-document bodies; with `keep_run`, bodies fed to python stay,
    since that code runs (a body written to a file does not)."""

    def body(match: re.Match) -> str:
        head = match.group("head")
        return match.group(0) if keep_run and _INTERPRETER_RE.search(head) else head

    return _HEREDOC_RE.sub(body, script)


def _shell_text(script: str) -> str:
    """The command as the shell parses it: here-document bodies dropped, quoted text reduced to
    the characters a file name can hold (so `> "a.py"` still names a.py)."""
    unquoted = _drop_heredocs(script, keep_run=False)
    return _QUOTED_RE.sub(lambda m: re.sub(r"[^\w./~-]", "_", m.group(0)[1:-1]), unquoted)


def command_edits(script: str) -> bool:
    """Whether running this command changes files in the repository."""
    return bool(
        _SHELL_EDIT_RE.search(_shell_text(script))
        or _SCRIPT_EDIT_RE.search(_drop_heredocs(script, keep_run=True))
    )


def edited_in_turn(text: str) -> bool:
    """Whether a turn changes the repository: it is a rendered editor edit, or its one command
    writes into the checkout. A turn holding several commands runs none of them: the harness
    rejects it as a format error."""
    if _RENDERED_EDIT_RE.search(text or ""):
        return True
    scripts = action_scripts(text)
    return len(scripts) == 1 and command_edits(scripts[0])


def trajectory_made_edit(turn_texts: list[str]) -> bool:
    return any(edited_in_turn(text) for text in turn_texts)


def shows_work(text: str) -> bool:
    """Whether a turn shows write-ish work of any kind, scratch files included."""
    return bool(WORK_EDIT_RE.search(text or ""))


def any_shows_work(texts: Iterable[str]) -> bool:
    return any(shows_work(text) for text in texts)


def named_in_removal(texts: Iterable[str], commands: Iterable[str], path: str) -> bool:
    for text, command in zip(texts, commands, strict=False):
        if not (WORK_EDIT_RE.search(text or "") or REMOVAL_RE.search(command or "")):
            continue
        for token in _FILE_TOKEN.findall(command or ""):
            if token == path or token.endswith("/" + path) or path.endswith("/" + token):
                return True
    return False
