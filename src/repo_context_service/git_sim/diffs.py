from __future__ import annotations

import hashlib

from .templates import (
    _FUNCNAME,
    DEFAULT_MODE,
    FUNCNAME_MAX_CHARS,
    NO_NEWLINE_MARKER,
    NULL_BLOB,
)
from .views import _in_scope, _Views


def blob_hash(text: str) -> str:
    data = text.encode("utf-8", "surrogateescape")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _text_lines(text: str) -> tuple[list[str], bool]:
    if text == "":
        return [], False
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
        return parts, False
    return parts, True


def _funcname(lines: list[str], start: int) -> str:
    for index in range(min(start, len(lines)) - 1, -1, -1):
        line = lines[index]
        if _FUNCNAME.match(line):
            return " " + line.strip()[:FUNCNAME_MAX_CHARS]
    return ""


def _hunks(
    old: list[str], new: list[str], old_open: bool, new_open: bool, context: int = 3
) -> list[str]:
    from difflib import SequenceMatcher

    out: list[str] = []
    matcher = SequenceMatcher(None, old, new, autojunk=False)
    for group in matcher.get_grouped_opcodes(context):
        first, last = group[0], group[-1]
        old_start, old_len = first[1], last[2] - first[1]
        new_start, new_len = first[3], last[4] - first[3]
        old_head = f"{old_start + 1 if old_len else old_start},{old_len}"
        new_head = f"{new_start + 1 if new_len else new_start},{new_len}"
        if new_len == 1:
            new_head = str(new_start + 1)
        out.append(f"@@ -{old_head} +{new_head} @@{_funcname(old, old_start)}")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for offset in range(i1, i2):
                    out.append(" " + old[offset])
                    if old_open and new_open and offset == len(old) - 1:
                        out.append(NO_NEWLINE_MARKER)
                continue
            if tag in ("replace", "delete"):
                for offset in range(i1, i2):
                    out.append("-" + old[offset])
                    if old_open and offset == len(old) - 1:
                        out.append(NO_NEWLINE_MARKER)
            if tag in ("replace", "insert"):
                for offset in range(j1, j2):
                    out.append("+" + new[offset])
                    if new_open and offset == len(new) - 1:
                        out.append(NO_NEWLINE_MARKER)
    return out


def _diff_file(path: str, old: str | None, new: str | None, abbrev: int) -> list[str]:
    if old == new:
        return []
    header = [f"diff --git a/{path} b/{path}"]
    old_lines, old_open = _text_lines(old or "")
    new_lines, new_open = _text_lines(new or "")
    old_hash = NULL_BLOB if old is None else blob_hash(old)
    new_hash = NULL_BLOB if new is None else blob_hash(new)
    if old is None:
        header.append(f"new file mode {DEFAULT_MODE}")
        header.append(f"index {old_hash[:abbrev]}..{new_hash[:abbrev]}")
        left, right = "/dev/null", f"b/{path}"
    elif new is None:
        header.append(f"deleted file mode {DEFAULT_MODE}")
        header.append(f"index {old_hash[:abbrev]}..{new_hash[:abbrev]}")
        left, right = f"a/{path}", "/dev/null"
    else:
        header.append(f"index {old_hash[:abbrev]}..{new_hash[:abbrev]} {DEFAULT_MODE}")
        left, right = f"a/{path}", f"b/{path}"
    body = _hunks(old_lines, new_lines, old_open, new_open)
    if not body:
        return header
    header.append(f"--- {left}")
    header.append(f"+++ {right}")
    return header + body


def _diff_pairs(
    views: _Views, cached: bool, scope: list[str]
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    pairs: list[tuple[str, list[str]]] = []
    unknown: list[str] = []
    for path in views.touched():
        if not _in_scope(path, scope):
            continue
        if cached:
            if path not in views.state.index and path not in views.state.staged_deleted:
                continue
            old = views.head(path)
            new = None if path in views.state.staged_deleted else views.state.index[path]
        else:
            if path in views.overlay.created and path not in views.state.index:
                continue
            old = views.staged(path)
            new = views.work(path)
        if old is None and new is None:
            continue
        if (old is None) != (new is None) and path in views.overlay.dirty:
            unknown.append(path)
            continue
        if new is None and path in views.overlay.dirty:
            unknown.append(path)
            continue
        if old == new:
            continue
        lines = _diff_file(path, old, new, views.abbrev)
        if lines:
            pairs.append((path, lines))
    return pairs, unknown


def _diff_head(views: _Views, scope: list[str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    pairs: list[tuple[str, list[str]]] = []
    unknown: list[str] = []
    for path in views.touched():
        if not _in_scope(path, scope):
            continue
        old = views.head(path)
        new = views.work(path)
        if path in views.overlay.created and path not in views.state.index:
            continue
        if new is None:
            unknown.append(path)
            continue
        if old == new:
            continue
        block = _diff_file(path, old, new, views.abbrev)
        if block:
            pairs.append((path, block))
    return pairs, unknown
