from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from albedo_eval_service.shared.observation_format import is_scaffold_truncated

from .command_search import _bre_to_python
from .git_sim import GitState, apply_git

_ENVELOPE = re.compile(r"</?returncode>\d*|<output>|</output>")
_OUTPUT_BLOCK = re.compile(r"<output>\n?(.*?)\n?</output>", re.S)
_TRAILER = re.compile(
    r"^\s*\[(The command (completed|timed out)|Current working directory|"
    r"Python interpreter|Command finished)\b.*\]\s*$"
)
_EMPTY_SENTENCE = "Your command ran successfully and did not produce any output."
_VIEW_HEADER = re.compile(r"^\s*Here's the result of running `[^`]+` on [^:]+:\s*$")
_NUMBERED = re.compile(r"^\s*(\d+)\t(.*)$")
_CD_PREFIX = re.compile(r"^\s*cd\s+[^\s&;|]+\s*&&\s*")
_FULL_READ = re.compile(r"^(cat|nl)\b((?!\|).)*$")
_GREP_N = re.compile(r"^(grep|rg)\b(?=.*\s-\w*n)((?!\|).)*$")
_SED_RANGE = re.compile(r"^sed\s+-n\s+'?(\d+),(\d+)p'?\s+(\S+)\s*$")
_SED_INPLACE = re.compile(
    r"^sed\s+(?:-i|--in-place)\s+"
    r"(?P<scripts>(?:-e\s+)?(?:'[^']*'|\"[^\"]*\")(?:\s+-e\s+(?:'[^']*'|\"[^\"]*\"))*)"
    r"\s+(?P<target>[^\s;&|]+)\s*$"
)
_SED_SCRIPT = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_SED_SUB_HEAD = re.compile(r"^(?:(?P<start>\d+)(?:,(?P<end>\d+))?)?s(?P<delim>[/|])")
_SED_DELETE = re.compile(r"^(?P<start>\d+)(?:,(?P<end>\d+))?d$")
_SED_PLACE = re.compile(r"^(?P<line>\d+)(?P<verb>[aic])(?P<rest>.+)$", re.S)
_SED_BACKREF = re.compile(r"(?<!\\)&|\\[1-9]")

_PATH_TOKEN = re.compile(r"[\w./~-]*[\w-]+\.[A-Za-z0-9_]+")
_WRITE_TARGETS = (
    re.compile(r"(?<![0-9])>>?\s*([^\s;&|<>]+)"),
    re.compile(r"\btee\s+(?:-a\s+)?([^\s;&|]+)"),
    re.compile(r"\bsed\s+(?:-i|--in-place)\b.*?([^\s;&|]+)\s*$", re.M),
    re.compile(r"\bpatch\s+-p\d+\s+.*?([^\s;&|]+)"),
    re.compile(r"open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]"),
)
_EDIT_VERB = re.compile(
    r"sed\s+-i|\s>>?\s*[\w./~-]+|\btee\s+\S|apply_patch|\bpatch\s+-p|open\([^)]*['\"][wa]"
)
_SANDBOX = re.compile(r"^/testbed(?:/|$)|^/workspace(?:/[^/]*__[^/]*)?(?:/|$)")
_ANY_FENCE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.S)
_SHELL_FENCE = re.compile(r"```(?:bash|sh)[ \t]*\n(.*?)```", re.S)
_NUMBERED_HIT = re.compile(r"^(\d+):(.*)$")
_HEREDOC_HEAD = re.compile(
    r"^(?:cat|tee)\s*(?:>(?!>)\s*(?P<p1>[^\s<>]+)\s*<<-?\s*'?(?P<d1>\w+)'?"
    r"|<<-?\s*'?(?P<d2>\w+)'?\s*>(?!>)\s*(?P<p2>[^\s<>]+))\s*$",
    re.M,
)


@dataclass
class Overlay:
    content: dict[str, str] = field(default_factory=dict)
    created: set[str] = field(default_factory=set)
    dirty: set[str] = field(default_factory=set)
    opaque: list[tuple[str | None, str]] = field(default_factory=list)
    git: GitState = field(default_factory=GitState)

    def state(self, block: str, referenced: set[str]) -> str:
        parts = [block]
        parts += [cmd for path, cmd in self.opaque if path is None or path in referenced]
        return hashlib.sha1("\n".join(parts).encode("utf-8", "replace")).hexdigest()

    def read(self, rel_path: str) -> str | None:
        return self.content.get(rel_path)

    def is_dirty(self, rel_path: str) -> bool:
        return rel_path in self.dirty

    def know(self, rel_path: str, text: str) -> None:
        self.content[rel_path] = text
        self.dirty.discard(rel_path)

    def forget(self, rel_path: str) -> None:
        self.content.pop(rel_path, None)
        self.dirty.add(rel_path)

    def listing(self, base: list[str]) -> list[str]:
        return sorted(set(base) | self.created) if self.created else base


def _strip_envelope(text: str, trim_edges: bool = True) -> list[str]:
    block = _OUTPUT_BLOCK.search(text or "")
    body = block.group(1) if block else _ENVELOPE.sub("", text or "")
    body = re.sub(r"^OBSERVATION:\s*", "", body, flags=re.M)
    lines = [
        line
        for line in body.split("\n")
        if not _TRAILER.match(line)
        and not _VIEW_HEADER.match(line)
        and line.strip() != _EMPTY_SENTENCE
    ]
    if trim_edges:
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
    return lines


def _fit_range(lines: list[str], expected: int) -> list[str] | None:
    while len(lines) > expected and not lines[0].strip():
        lines.pop(0)
    while len(lines) > expected and not lines[-1].strip():
        lines.pop()
    return lines if len(lines) == expected else None


def _denumber(lines: list[str]) -> list[str]:
    return [m.group(2) if (m := _NUMBERED.match(line)) else line for line in lines]


def strip_sandbox(path: str) -> str:
    cleaned = path.strip("'\"")
    stripped = _SANDBOX.sub("", cleaned)
    return stripped.lstrip("/") if stripped != cleaned else cleaned


def _resolve(token: str, listing: set[str], index: dict[str, str]) -> str | None:
    token = token.strip("'\"")
    if token in listing:
        return token
    segments = token.lstrip("/").split("/")
    for start in range(len(segments)):
        if (hit := index.get("/".join(segments[start:]))) is not None:
            return hit
    return None


def _written_paths(body: str, listing: set[str], index: dict[str, str]) -> list[str] | None:
    operands = [match.group(1) for pattern in _WRITE_TARGETS for match in pattern.finditer(body)]
    if not operands:
        return None
    return [hit for token in operands if (hit := _resolve(token, listing, index)) is not None]


def _target(command: str, listing: set[str], index: dict[str, str]) -> str | None:
    body = _CD_PREFIX.sub("", command.strip())
    for token in reversed([t for t in _PATH_TOKEN.findall(body) if "." in t]):
        if (hit := _resolve(token, listing, index)) is not None:
            return hit
    return None


def _command_of(assistant_text: str) -> str:
    match = _SHELL_FENCE.search(assistant_text or "") or _ANY_FENCE.search(assistant_text or "")
    return match.group(1).strip() if match else ""


def _splice_range(
    overlay: Overlay, path: str, observed: list[str], read_base, start_line: int
) -> None:
    current = overlay.content.get(path)
    if current is None:
        current = None if overlay.is_dirty(path) else read_base(path)
    if current is None or start_line < 1:
        return
    lines = current.split("\n")
    if start_line - 1 > len(lines):
        return
    lines[start_line - 1 : start_line - 1 + len(observed)] = observed
    overlay.content[path] = "\n".join(lines)


def build_overlay(messages, listing: list[str], suffix_index: dict[str, str], read_base) -> Overlay:
    overlay = Overlay()
    listing_set = set(listing)
    index = dict(suffix_index)
    pending: str | None = None
    for turn, message in enumerate(messages or []):
        role = str(message.get("role") or "").lower()
        text = str(message.get("content") or "")
        if role == "assistant":
            pending = text
            written = (
                _apply_heredoc_create(overlay, text, listing_set, index)
                or _apply_search_replace(overlay, text, listing_set, index, read_base)
                or _apply_sed_edit(overlay, text, listing_set, index, read_base)
            )
            _mark_opaque_edit(overlay, text, listing_set, index, skip=written)
            continue
        if role not in ("user", "tool") or pending is None:
            continue
        _learn(overlay, pending, text, listing_set, index, read_base, turn)
        pending = None
    return overlay


def _learn(
    overlay: Overlay,
    assistant_text: str,
    observation: str,
    listing: set[str],
    index: dict[str, str],
    read_base,
    turn: int = 0,
) -> None:
    command = _command_of(assistant_text)
    if not command:
        return
    described = not overlay.git.unknown
    if apply_git(overlay, command, observation, sorted(listing), read_base, turn):
        if described and overlay.git.unknown:
            overlay.opaque.append((None, command))
        return
    if is_scaffold_truncated(observation):
        return
    body = _CD_PREFIX.sub("", command.strip())
    path = _target(command, listing, index)
    if path is None:
        return
    if _FULL_READ.match(body):
        lines = _denumber(_strip_envelope(observation))
        if lines and read_base(path) is not None:
            overlay.know(path, "\n".join(lines) + "\n")
        return
    if _GREP_N.match(body):
        _verify_lines(overlay, path, observation)
        return
    if ranged := _SED_RANGE.match(body):
        start, end = int(ranged.group(1)), int(ranged.group(2))
        observed = _fit_range(_strip_envelope(observation, trim_edges=False), end - start + 1)
        if observed:
            _splice_range(overlay, path, observed, read_base, start)


def _verify_lines(overlay: Overlay, path: str, observation: str) -> None:
    current = overlay.content.get(path)
    if current is None:
        return
    lines = current.split("\n")
    for entry in _strip_envelope(observation):
        if not (match := _NUMBERED_HIT.match(entry)):
            continue
        number = int(match.group(1))
        if number > len(lines) or lines[number - 1] != match.group(2):
            overlay.forget(path)
            return


_SEARCH_REPLACE = re.compile(
    r"Editing\s+`([^`]+)`:.*?<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S
)


def _apply_search_replace(
    overlay: Overlay, text: str, listing: set[str], index: dict[str, str], read_base
) -> str | None:
    match = _SEARCH_REPLACE.search(text or "")
    if match is None:
        return None
    raw, old, new = match.group(1), match.group(2), match.group(3)
    path = _resolve(raw, listing, index) or strip_sandbox(raw)
    if not path:
        return None
    current = overlay.content.get(path)
    if current is None and not overlay.is_dirty(path):
        current = read_base(path)
    if current is None or current.count(old) != 1:
        overlay.forget(path)
        return path
    overlay.know(path, current.replace(old, new))
    return path


def _heredoc_create(command: str) -> tuple[str, str] | None:
    match = _HEREDOC_HEAD.match(command.strip())
    if match is None:
        return None
    path = match.group("p1") or match.group("p2")
    delimiter = match.group("d1") or match.group("d2")
    if not path or not delimiter:
        return None
    lines = command.strip().split("\n")[1:]
    for end, line in enumerate(lines):
        if line.strip() == delimiter:
            return strip_sandbox(path), "\n".join(lines[:end])
    return None


def _apply_heredoc_create(
    overlay: Overlay, text: str, listing: set[str], index: dict[str, str]
) -> str | None:
    created = _heredoc_create(_command_of(text))
    if created is None:
        return None
    path, body = created
    if not path:
        return None
    overlay.know(path, body + "\n")
    if path not in listing:
        overlay.created.add(path)
        listing.add(path)
        index.setdefault(path.rsplit("/", 1)[-1], path)
    return path


def _mark_opaque_edit(
    overlay: Overlay,
    text: str,
    listing: set[str],
    index: dict[str, str],
    skip: str | None = None,
) -> None:
    command = _command_of(text)
    if not command or not _EDIT_VERB.search(command):
        return
    written = _written_paths(_CD_PREFIX.sub("", command.strip()), listing, index)
    if written is None:
        paths = [path] if (path := _target(command, listing, index)) is not None else [None]
    elif not written:
        return
    else:
        paths = list(dict.fromkeys(written))
    for path in paths:
        if path == skip:
            continue
        if path is not None:
            overlay.forget(path)
        overlay.opaque.append((path, command))


def _sed_unescape(text: str) -> str:
    out, index = [], 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            out.append({"n": "\n", "t": "\t"}.get(text[index + 1], text[index + 1]))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _sed_sub_fields(script: str) -> tuple[str | None, str | None, str, str, str] | None:
    """Split `[N[,M]]s<D>pattern<D>replacement<D>[flags]`, honouring backslash escapes."""
    head = _SED_SUB_HEAD.match(script)
    if head is None:
        return None
    delim, rest = head.group("delim"), script[head.end() :]
    parts, buf, index = [], [], 0
    while index < len(rest):
        char = rest[index]
        if char == "\\" and index + 1 < len(rest):
            buf.append(rest[index : index + 2])
            index += 2
            continue
        if char == delim:
            parts.append("".join(buf))
            buf = []
            index += 1
            continue
        buf.append(char)
        index += 1
    parts.append("".join(buf))
    if len(parts) != 3:
        return None
    return head.group("start"), head.group("end"), parts[0], parts[1], parts[2]


def _sed_bounds(start: str | None, end: str | None, count: int) -> tuple[int, int] | None:
    first = int(start) if start else 1
    last = int(end) if end else (int(start) if start else count)
    return None if first < 1 or last > count or first > last else (first, last)


def _sed_apply(script: str, text: str) -> str | None:
    """Run one sed expression over `text`, or None if we do not fully understand it."""
    lines = text.split("\n")
    trailing = bool(lines) and lines[-1] == ""
    if trailing:
        lines.pop()
    tail = "\n" if trailing else ""

    if fields := _sed_sub_fields(script):
        start, end, pattern, replacement, flags = fields
        if flags not in ("", "g") or _SED_BACKREF.search(replacement):
            return None
        bounds = _sed_bounds(start, end, len(lines))
        if bounds is None:
            return None
        try:
            regex = re.compile(_bre_to_python(pattern))
        except re.error:
            return None
        body = _sed_unescape(replacement)
        for number in range(bounds[0], bounds[1] + 1):
            # a lambda so re never reinterprets backslashes in the replacement
            lines[number - 1] = regex.sub(
                lambda _: body, lines[number - 1], count=0 if flags == "g" else 1
            )
        return "\n".join(lines) + tail

    if deleted := _SED_DELETE.match(script):
        bounds = _sed_bounds(deleted.group("start"), deleted.group("end"), len(lines))
        if bounds is None:
            return None
        del lines[bounds[0] - 1 : bounds[1]]
        return "\n".join(lines) + tail

    if placed := _SED_PLACE.match(script):
        bounds = _sed_bounds(placed.group("line"), None, len(lines))
        if bounds is None:
            return None
        rest = placed.group("rest")
        raw = rest[1:] if rest.startswith("\\") else rest.lstrip(" \t")
        added = _sed_unescape(raw).split("\n")
        verb, at = placed.group("verb"), bounds[0]
        if verb == "a":
            lines[at:at] = added
        elif verb == "i":
            lines[at - 1 : at - 1] = added
        else:
            lines[at - 1 : at] = added
        return "\n".join(lines) + tail

    return None


def _sed_scripts(script: str) -> list[str]:
    return [piece.strip() for piece in script.split(";") if piece.strip()]


def _sed_apply_all(scripts: list[str], text: str) -> str | None:
    for script in scripts:
        stepped = _sed_apply(script, text)
        if stepped is None:
            pieces = _sed_scripts(script)
            if len(pieces) < 2:
                return None
            for piece in pieces:
                stepped = _sed_apply(piece, text)
                if stepped is None:
                    return None
                text = stepped
            continue
        text = stepped
    return text


def _apply_sed_edit(
    overlay: Overlay, text: str, listing: set[str], index: dict[str, str], read_base
) -> str | None:
    """Apply an in-place sed we fully understand; None leaves it to _mark_opaque_edit."""
    match = _SED_INPLACE.match(_CD_PREFIX.sub("", _command_of(text).strip()))
    if match is None:
        return None
    raw = match.group("target").strip("'\"")
    path = _resolve(raw, listing, index) or strip_sandbox(raw)
    if not path:
        return None
    current = overlay.content.get(path)
    if current is None and not overlay.is_dirty(path):
        current = read_base(path)
    if current is None:
        return None
    scripts = [found[0] or found[1] for found in _SED_SCRIPT.findall(match.group("scripts"))]
    edited = _sed_apply_all(scripts, current) if scripts else None
    if edited is None:
        return None
    overlay.know(path, edited or "\n")
    return path
