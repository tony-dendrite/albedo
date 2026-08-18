from __future__ import annotations

import re

from ..command_search import ParseFailure, _to_repo_relative
from .execute import run_git
from .models import GitMeta, GitPlan, GitResult, GitState, StashEntry
from .parse import (
    _GIT_HEAD,
    _observed_returncode,
    _split_args,
    _subcommand_of,
    mutation_stages,
    parse_git,
)
from .patches import learn_from_observed_diff
from .templates import (
    _HISTORY_CHANGING,
    _OPAQUE,
    _READ_ONLY,
    DEFAULT_ABBREV,
    HARNESS_SUBJECT,
)
from .views import _in_scope, _normalize_paths, _Views

_BRANCH_LINE = re.compile(r"^On branch (\S+)$", re.M)
_DETACHED_LINE = re.compile(r"^Not currently on any branch\.$", re.M)
_INDEX_LINE = re.compile(r"^index ([0-9a-f]{4,40})\.\.[0-9a-f]{4,40}", re.M)
_WIP_LINE = re.compile(
    r"^Saved working directory and index state WIP on ([^:]+): ([0-9a-f]{4,40}) (.*)$", re.M
)
_HEAD_NOW_LINE = re.compile(r"^HEAD is now at ([0-9a-f]{4,40}) (.*)$", re.M)
_ONELINE_ENTRY = re.compile(r"^([0-9a-f]{7,12}) \S", re.M)
_HARNESS_HEAD = re.compile(r"^([0-9a-f]{7,12}) SWE-bench$", re.M)


def learn_git_facts(state: GitState, observation: str, command: str = "") -> None:
    text = observation or ""
    if match := _HARNESS_HEAD.search(text):
        state.head_short = match.group(1)
        state.head_subject = HARNESS_SUBJECT
        state.abbrev = len(match.group(1))
    if match := _BRANCH_LINE.search(text):
        state.branch = match.group(1)
        state.detached = False
    elif _DETACHED_LINE.search(text):
        state.detached = True
    if match := _WIP_LINE.search(text):
        branch, short, subject = match.groups()
        state.abbrev = len(short)
        state.head_short = short
        state.head_subject = subject
        if branch == "(no branch)":
            state.detached = True
        else:
            state.branch = branch
    elif match := _HEAD_NOW_LINE.search(text):
        state.abbrev = len(match.group(1))
        state.head_short = match.group(1)
        state.head_subject = match.group(2)
    elif match := _INDEX_LINE.search(text):
        state.abbrev = len(match.group(1))
    elif "--oneline" in (command or "") and (match := _ONELINE_ENTRY.search(text)):
        state.abbrev = len(match.group(1))


def _scope_paths(plan: GitPlan, views: _Views) -> list[str]:
    tokens = [p for p in plan.paths if p not in ("HEAD", "--")]
    if not tokens or any(t in (".", "./", "-A", "*") for t in tokens):
        return []
    return _normalize_paths(tokens, views.listing_set)


def _stage(views: _Views, paths: list[str], only_tracked: bool) -> list[str]:
    staged: list[str] = []
    for path in paths:
        if only_tracked and not views.tracked(path):
            continue
        views.state.index[path] = views.work(path)
        staged.append(path)
    return staged


def _remove(views: _Views, path: str) -> None:
    views.overlay.content.pop(path, None)
    views.overlay.created.discard(path)
    views.overlay.dirty.discard(path)


def _restore(views: _Views, paths: list[str], source: str) -> list[str]:
    restored: list[str] = []
    for path in paths:
        if source == "head" or path not in views.state.index:
            text = views.head(path)
            exists = text is not None
        else:
            text = views.state.index[path]
            exists = True
        if not exists:
            _remove(views, path)
        elif text is None:
            views.overlay.forget(path)
        else:
            views.overlay.know(path, text)
        restored.append(path)
    return restored


def _apply_add(plan: GitPlan, views: _Views) -> dict:
    _split_args(plan)
    scope = _scope_paths(plan, views)
    only_tracked = bool({"-u", "--update"} & plan.flags)
    targets = [p for p in views.touched() if _in_scope(p, scope)]
    return {"staged": _stage(views, targets, only_tracked)}


def _apply_checkout(plan: GitPlan, views: _Views) -> dict | None:
    _split_args(plan, frozenset({"-b", "-B"}))
    if {"-b", "-B"} & set(plan.values):
        return {"branch": plan.values.get("-b") or plan.values.get("-B")}
    operands = [p for p in plan.paths if p != "--"]
    source = "index"
    if operands and operands[0] == "HEAD":
        source = "head"
        operands = operands[1:]
    if not operands:
        return None
    if "--" not in plan.flags and source == "index":
        normalized = _normalize_paths(operands, views.listing_set)
        if not all(p in views.listing_set or p in views.overlay.created for p in normalized):
            return None
    scope = _normalize_paths([p for p in operands if p not in (".", "./")], views.listing_set)
    targets = [p for p in views.touched() if _in_scope(p, scope) and views.tracked(p)]
    restored = _restore(views, targets, source)
    if source == "head":
        for path in targets:
            views.state.index.pop(path, None)
    return {"restored": restored, "from": source}


def _apply_restore(plan: GitPlan, views: _Views) -> dict | None:
    _split_args(plan, frozenset({"--source"}))
    source = plan.values.get("--source", "")
    if source and source != "HEAD":
        return None
    scope = _scope_paths(plan, views)
    targets = [p for p in views.touched() if _in_scope(p, scope) and views.tracked(p)]
    effect: dict = {}
    if "--staged" in plan.flags:
        for path in targets:
            views.state.index.pop(path, None)
        effect["unstaged"] = list(targets)
    if "--staged" not in plan.flags or {"-W", "--worktree"} & plan.flags:
        effect["restored"] = _restore(views, targets, "head" if source == "HEAD" else "index")
    return effect


def _apply_reset(plan: GitPlan, views: _Views) -> dict | None:
    _split_args(plan)
    revs = [
        p
        for p in plan.paths
        if p in ("HEAD", "HEAD~1", "ORIG_HEAD")
        or (len(p) >= 7 and re.fullmatch(r"[0-9a-f]{7,40}", p))
    ]
    if any(rev != "HEAD" for rev in revs):
        return None
    if {"--merge", "--keep"} & plan.flags:
        return None
    scope = _scope_paths(plan, views)
    targets = [p for p in views.touched() if _in_scope(p, scope)]
    if "--soft" in plan.flags:
        return {"soft": True}
    tracked = [p for p in targets if views.tracked(p)]
    unstaged = [p for p in targets if p in views.state.index]
    for path in unstaged:
        views.state.index.pop(path, None)
    if "--hard" in plan.flags:
        _restore(views, tracked, "head")
        return {"reset": "hard", "restored": tracked}
    return {"unstaged": unstaged}


def _apply_stash(plan: GitPlan, views: _Views) -> dict | None:
    _split_args(plan)
    action = plan.paths[0] if plan.paths else "push"
    state = views.state
    if action in ("list", "show"):
        return {"stash_depth": len(state.stash)}
    if action == "clear":
        state.stash.clear()
        return {"stash": "cleared"}
    if action in ("push", "save"):
        keep_untracked = not bool({"-u", "--include-untracked"} & plan.flags)
        entry = StashEntry()
        targets: list[str] = []
        for path in views.touched():
            if not views.tracked(path):
                if keep_untracked:
                    continue
                entry.created[path] = views.work(path)
            elif views.head(path) is None:
                entry.created[path] = views.work(path)
            else:
                entry.content[path] = views.work(path)
            targets.append(path)
        if not targets:
            return {"stashed": []}
        state.stash.append(entry)
        _restore(views, [p for p in targets if p in entry.content], "head")
        for path in entry.created:
            _remove(views, path)
        state.index.clear()
        return {"stashed": targets}
    if action in ("pop", "apply"):
        if not state.stash:
            return None
        entry = state.stash[-1] if action == "apply" else state.stash.pop()
        for path, text in list(entry.content.items()) + list(entry.created.items()):
            if text is None:
                views.overlay.forget(path)
            else:
                views.overlay.know(path, text)
            if path in entry.created:
                views.overlay.created.add(path)
        return {"unstashed": entry.paths()}
    if action == "drop":
        if state.stash:
            state.stash.pop()
        return {"stash": "dropped"}
    return None


_MUTATORS = {
    "add": _apply_add,
    "checkout": _apply_checkout,
    "switch": _apply_checkout,
    "restore": _apply_restore,
    "reset": _apply_reset,
    "stash": _apply_stash,
}


def apply_git(
    overlay, command: str, observation: str, listing: list[str], read_base, turn: int = 0
) -> bool:
    if not _GIT_HEAD.search(command or ""):
        return False
    state = getattr(overlay, "git", None)
    if state is None:
        return True
    stages = mutation_stages(command)
    git_stages = [s for s in stages if _GIT_HEAD.search(s)]
    if not git_stages:
        return False
    learn_git_facts(state, observation, command)
    returncode = _observed_returncode(observation)
    if returncode == 0 and re.search(r"\bgit\s+(?:--no-pager\s+)?diff\b", command):
        learn_from_observed_diff(overlay, state, command, observation, read_base)
    last_stage = stages[-1] if stages else ""
    views = _Views(overlay, state, read_base, listing, DEFAULT_ABBREV)
    for stage in git_stages:
        failed = returncode != 0 and stage == last_stage
        plan = parse_git(stage)
        if isinstance(plan, ParseFailure):
            if _subcommand_of(stage) in _READ_ONLY:
                continue
            state.poison(f"unparsed:{plan.reason}:{_subcommand_of(stage) or '?'}")
            return True
        if plan.sub in _OPAQUE:
            state.poison(f"opaque:{plan.sub}", history=plan.sub in _HISTORY_CHANGING)
            return True
        if plan.redirect and not state.unknown:
            capture = GitPlan(sub=plan.sub, args=plan.args, pipeline=plan.pipeline, raw=plan.raw)
            captured = run_git(capture, overlay, read_base, listing, GitMeta())
            if isinstance(captured, GitResult) and captured.exact:
                target = _to_repo_relative(plan.redirect, set(listing)) or plan.redirect
                overlay.know(target, captured.output + "\n" if captured.output else "")
                overlay.created.add(target)
        mutator = _MUTATORS.get(plan.sub)
        if mutator is None or state.unknown:
            continue
        if failed:
            state.poison(f"failed_rc:{plan.sub}")
            return True
        effect = mutator(plan, views)
        if effect is None:
            state.poison(f"unmodelled:{plan.sub}")
            return True
        if effect:
            state.record(turn, stage.strip(), effect)
    return True
