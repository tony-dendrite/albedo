from __future__ import annotations

from ..command_search import _to_repo_relative
from .models import GitMeta, GitState


def _normalize_paths(tokens: list[str], listing: set[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        rel = _to_repo_relative(token, listing)
        out.append("" if rel is None else rel)
    return out


def _in_scope(path: str, scope: list[str]) -> bool:
    if not scope:
        return True
    for entry in scope:
        if entry in ("", ".", "./"):
            return True
        if path == entry or path.startswith(entry.rstrip("/") + "/"):
            return True
    return False


class _Views:
    def __init__(self, overlay, state: GitState, read_base, listing: list[str], abbrev: int):
        self.overlay = overlay
        self.state = state
        self.read_base = read_base
        self.listing = listing
        self.listing_set = set(listing)
        self.abbrev = abbrev

    def head(self, path: str) -> str | None:
        return self.read_base(path)

    def work(self, path: str) -> str | None:
        held = self.overlay.content.get(path)
        if held is not None:
            return held
        if self.overlay.is_dirty(path):
            return None
        return self.read_base(path)

    def staged(self, path: str) -> str | None:
        if path in self.state.index:
            return self.state.index[path]
        return self.read_base(path)

    def known_work(self, path: str) -> bool:
        return path in self.overlay.content or not self.overlay.is_dirty(path)

    def touched(self) -> list[str]:
        paths = set(self.overlay.content) | set(self.overlay.dirty) | set(self.state.index)
        paths |= set(self.state.staged_deleted) | set(self.overlay.created)
        return sorted(p for p in paths if p and not p.startswith("/") and ".." not in p)

    def seen(self, path: str) -> bool:
        return (
            path in self.overlay.content
            or path in self.overlay.dirty
            or path in self.state.index
            or path in self.overlay.created
        )

    def tracked(self, path: str) -> bool:
        return self.read_base(path) is not None or path in self.state.index


def _dirty_paths(views: _Views, scope: list[str], against: str) -> list[str]:
    out: list[str] = []
    for path in views.touched():
        if not _in_scope(path, scope) or not views.tracked(path):
            continue
        reference = views.head(path) if against == "head" else views.staged(path)
        current = views.work(path)
        if current is None or reference is None or current != reference:
            out.append(path)
    return out


def _resolve_head(views: _Views, meta: GitMeta) -> str | None:
    state = views.state
    if state.head_short and state.head_subject is not None:
        return state.head_short
    if not (meta.detached and meta.sha and state.abbrev and callable(meta.history)):
        return None
    payload = meta.history(None)
    commits = (payload or {}).get("commits") or []
    if not commits or commits[0]["sha"] != meta.sha:
        return None
    state.head_short = meta.sha[: state.abbrev]
    state.head_subject = commits[0]["subject"]
    return state.head_short
