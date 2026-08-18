from __future__ import annotations

from ..command_search import split_chain
from .models import GitMeta, GitPlan, GitState
from .parse import _GIT_HEAD, _SHORT_STATUS, _STASH_RESTORE, _subcommand_of, parse_git
from .render import _head_line, _status_sets
from .templates import DEFAULT_BRANCH, GIT_LEDGER_HEADER, UNCERTAIN_STATE_LINE
from .views import _Views

_EFFECT_LABELS = {
    "staged": "staged",
    "unstaged": "unstaged",
    "restored": "restored from {from}",
    "stashed": "stashed away",
    "unstashed": "restored from the stash",
    "branch": "created branch",
    "reset": "reset",
    "soft": "index and worktree untouched",
    "stash_depth": "stash entries",
    "stash": "stash",
}


def _effect_text(effect: dict) -> str:
    parts: list[str] = []
    for key, value in effect.items():
        if key == "from":
            continue
        label = _EFFECT_LABELS.get(key, key).replace("{from}", str(effect.get("from", "index")))
        if isinstance(value, list):
            if not value:
                continue
            shown = ", ".join(value[:8]) + (f" (+{len(value) - 8} more)" if len(value) > 8 else "")
            parts.append(f"{label}: {shown}")
        elif isinstance(value, bool):
            parts.append(label)
        else:
            parts.append(f"{label}: {value}")
    return "; ".join(parts) or "no file changes"


def ledger_block(state: GitState | None) -> str:
    if state is None or not state.ledger:
        return ""
    lines = [
        f"$ {entry['command']}\n  -> {_effect_text(entry['effect'])}" for entry in state.ledger
    ]
    return "\n" + GIT_LEDGER_HEADER + "\n" + "\n".join(lines) + "\n"


_NOTES = {
    "status": (
        'git status in this checkout always starts with exactly "{head}" — never any other '
        "first line, and never a branch name this prompt does not show.",
        "{state_line}",
    ),
    "status_short": (
        'git status --short prints one "XY path" line per changed file and nothing else — no '
        "branch header, no hint lines, no summary; a clean tree prints no output at all.",
        "{state_line}",
    ),
    "diff": (
        "git diff prints a unified diff of UNSTAGED changes only — anything already staged with "
        "git add does not appear here, and with no unstaged change it prints nothing.",
        "{state_line}",
    ),
    "log": (
        'git log --oneline prints "<short sha> <subject>" per commit, newest first, starting at '
        "the checked-out commit {short}.",
        "Never invent commit hashes or subjects: reuse only shas already visible in this prompt, "
        "and keep the count the command asks for.",
    ),
    "show": (
        "git show prints the commit header (commit/Author/Date, blank line, indented subject) "
        "followed by its patch, for commit {short} unless another rev is named.",
        "Never invent a commit message or patch for a sha that is not visible in this prompt.",
    ),
    "stash": (
        'git stash prints exactly one line — "Saved working directory and index state WIP on '
        '{wip_ref}: <short sha> <subject>" — and nothing else; every other line in this '
        "observation comes from the commands chained after it.",
        "After it the worktree is clean: git status shows no changes and git diff prints nothing "
        "until it is popped. Stash entries held right now: {stash_depth}.",
    ),
    "stash_pop": (
        "git stash pop prints a git-status-style listing of the restored files and then one final "
        'line, "Dropped refs/stash@{{0}} (<40-hex sha>)"; that sha is a stash object id, unrelated '
        "to any commit sha in this prompt.",
        'It fails with return code 1 and the single line "No stash entries found." when nothing '
        "was stashed. Stash entries held right now: {stash_depth}.",
    ),
    "checkout": (
        'git checkout <paths> prints exactly one line, "Updated N path(s) from the index", '
        "while git checkout -- <paths> prints nothing at all.",
        "{state_line}",
    ),
    "add": (
        "git add prints nothing at all on success — the observation body is empty "
        "with return code 0.",
        "It only moves files into the index; the worktree is unchanged and git diff stops showing "
        "what was added.",
    ),
    "branch": (
        'git status in this checkout reports "{head}", and git branch must agree with that '
        "line — no other local branch exists.",
        "Do not invent remote branches: this checkout has no remote-tracking refs.",
    ),
}


def _state_line(views: _Views, state: GitState) -> str:
    staged, unstaged, untracked, uncertain = _status_sets(views)
    if state.unknown or uncertain:
        return UNCERTAIN_STATE_LINE
    return (
        f"Modified this session: {_fmt_list([p for _, p in unstaged])}. "
        f"Staged: {_fmt_list([p for _, p in staged])}. "
        f"Untracked: {_fmt_list(untracked)}."
    )


def _fmt_list(paths: list[str], limit: int = 6) -> str:
    if not paths:
        return "none"
    shown = ", ".join(paths[:limit])
    return shown + (f" (+{len(paths) - limit} more)" if len(paths) > limit else "")


def explain_git(
    command: str, overlay, listing: list[str], read_base, meta: GitMeta | None = None
) -> str:
    if not _GIT_HEAD.search(command or ""):
        return ""
    stages = split_chain(command) or [command]
    subs: list[str] = []
    for stage in stages:
        if not _GIT_HEAD.search(stage):
            continue
        plan = parse_git(stage)
        sub = plan.sub if isinstance(plan, GitPlan) else _subcommand_of(stage)
        if sub == "status" and _SHORT_STATUS.search(stage):
            sub = "status_short"
        elif sub == "stash" and _STASH_RESTORE.search(stage):
            sub = "stash_pop"
        if sub in _NOTES and sub not in subs:
            subs.append(sub)
    if not subs:
        return ""
    meta = meta or GitMeta()
    state = getattr(overlay, "git", None) or GitState()
    views = _Views(overlay, state, read_base, listing, state.abbrev or meta.abbrev)
    values = {
        "head": _head_line(views, meta),
        "short": state.head_short or meta.short or "the checked-out commit",
        "state_line": _state_line(views, state),
        "stash_depth": str(len(state.stash)),
        "wip_ref": (
            "(no branch)"
            if state.detached or (meta.detached and not state.branch)
            else (state.branch or meta.branch or DEFAULT_BRANCH)
        ),
    }
    return "\n".join(template.format(**values) for template in _NOTES[subs[0]]) + "\n"
