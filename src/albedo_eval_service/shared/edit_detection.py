from __future__ import annotations

import re
from collections.abc import Iterable

WORK_EDIT_RE = re.compile(
    r"\b(sed -i|cat >|cat >>|tee |git apply|applypatch|str_replace|patch -p\d|cp |mv )"
    r"|<<\s*'?\"?[A-Za-z_][A-Za-z0-9_]*'?\"?\s*>"
    r"|<<\s*'?(EOF|PYEOF|PATCH|PY|SH|BASH|SCRIPT)\b"
    r"|\.write_text\(|\.writelines\(|open\([^)]*['\"][wa]\+?['\"]"
    r"|(?<=\s)>>?\s*(?!/dev/)[~$\w./-]*(?:/[~$\w.-]+|\.[A-Za-z]\w*)",
    re.I,
)

EDIT_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)
REPO_EDIT_RE = re.compile(
    r"sed\s+-i"
    r"|(?<![-=0-9&])>>?\s*(?!/dev/|/tmp/)(?=[\w.~/-]*[./])[\w./~-]"
    r"|tee\s+(?!/dev/|/tmp/)[\w./~-]|cat\s*>>?\s*(?!/dev/|/tmp/)[\w./~-]"
    r"|str_replace|git\s+apply|patch\s+-p|applypatch|>{7}\s*REPLACE"
    r"|cp\s+[\w./-]+\s+(?!/dev/|/tmp/)[\w./-]+|mv\s+[\w./-]+\s+(?!/dev/|/tmp/)[\w./-]+"
    r"|open\s*\([^)]*['\"][wa]\+?['\"]"
    r"|\.write_text\s*\(|\.write_bytes\s*\(|\.writelines\s*\("
    r"|fileinput\.input\([^)]*inplace"
    r"|shutil\.(?:copy|copyfile|copy2|move)\s*\(|os\.(?:replace|rename)\s*\("
    r"|(?:perl|ruby)\s+-[a-zA-Z]*i\b",
)

REMOVAL_RE = re.compile(r"\brm\b|\bgit\s+(?:rm|checkout|stash)\b|\bmv\b")

_FILE_TOKEN = re.compile(r"[\w./-]*[\w-]+\.[A-Za-z]\w*")


def edited_in_turn(text: str) -> bool:
    """Whether a turn's shell commands change the repository (bash blocks only)."""
    return any(REPO_EDIT_RE.search(cmd) for cmd in EDIT_BLOCK_RE.findall(text or ""))


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
