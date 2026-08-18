from __future__ import annotations

import re

from ..command_search import ParseFailure, SearchPlan, _apply_pipeline, _to_repo_relative
from .diffs import _diff_head, _diff_pairs
from .models import GitMeta, GitPlan, GitResult
from .parse import _split_args
from .patches import _commit_header, _filter_diff, _parse_patch, _reabbrev, _retarget_funcnames
from .templates import (
    BRANCH_HEADER,
    CLEAN_TRAILER,
    DEFAULT_BRANCH,
    DETACHED_HEADER,
    ENTRY_LABEL_WIDTH,
    EVIDENCE_LOG_LIMIT,
    HARD_RESET_LINE,
    HARNESS_SUBJECT,
    RESET_HEADER,
    STAGED_HEADER,
    STAGED_HINT,
    STASH_EMPTY_LINE,
    STASH_MISSING_LINE,
    STASH_SAVED_LINE,
    UNSTAGED_HEADER,
    UNSTAGED_HINTS,
    UNSTAGED_TRAILER,
    UNTRACKED_HEADER,
    UNTRACKED_HINT,
    UNTRACKED_ONLY_TRAILER,
    UPDATED_PATHS_LINE,
)
from .views import _dirty_paths, _in_scope, _normalize_paths, _resolve_head, _Views


def _entry(label: str, path: str) -> str:
    return "\t" + (label + ":").ljust(ENTRY_LABEL_WIDTH) + path


def _status_sets(
    views: _Views, scope: list[str] | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str], bool]:
    staged: list[tuple[str, str]] = []
    unstaged: list[tuple[str, str]] = []
    untracked: list[str] = []
    unknown = False
    for path in views.touched():
        if not _in_scope(path, scope or []):
            continue
        base = views.head(path)
        index_present = path in views.state.index
        if path in views.state.staged_deleted:
            staged.append(("deleted", path))
            continue
        if base is None and not index_present:
            if path in views.overlay.created:
                untracked.append(path)
            continue
        if index_present:
            staged_text = views.state.index[path]
            if base is None:
                staged.append(("new file", path))
            elif staged_text is None:
                staged.append(("modified", path))
                unknown = True
            elif staged_text != base:
                staged.append(("modified", path))
        current = views.work(path)
        reference = views.staged(path)
        if current is None:
            unstaged.append(("modified", path))
            unknown = True
        elif reference is None or current != reference:
            unstaged.append(("modified", path))
    return staged, unstaged, untracked, unknown


def _head_line(views: _Views, meta: GitMeta) -> str:
    if views.state.detached:
        return DETACHED_HEADER
    if views.state.branch:
        return BRANCH_HEADER.format(branch=views.state.branch)
    if meta.detached:
        return DETACHED_HEADER
    return BRANCH_HEADER.format(branch=meta.branch or DEFAULT_BRANCH)


def _render_status_long(views: _Views, meta: GitMeta, scope: list[str]) -> list[str]:
    staged, unstaged, untracked, _ = _status_sets(views, scope)
    sections: list[list[str]] = []
    if staged:
        sections.append(
            [STAGED_HEADER, STAGED_HINT] + [_entry(label, path) for label, path in staged]
        )
    if unstaged:
        sections.append(
            [UNSTAGED_HEADER, *UNSTAGED_HINTS] + [_entry(label, path) for label, path in unstaged]
        )
    if untracked:
        sections.append([UNTRACKED_HEADER, UNTRACKED_HINT] + ["\t" + path for path in untracked])
    head = _head_line(views, meta)
    if not sections:
        return [head, CLEAN_TRAILER]
    lines = [head]
    for position, section in enumerate(sections):
        if position:
            lines.append("")
        lines.extend(section)
    lines.append("")
    if not staged:
        lines.append(UNSTAGED_TRAILER if unstaged else UNTRACKED_ONLY_TRAILER)
    return lines


def _render_status_short(views: _Views, scope: list[str]) -> list[str]:
    staged, unstaged, untracked, _ = _status_sets(views, scope)
    staged_map = {path: label for label, path in staged}
    unstaged_map = {path: label for label, path in unstaged}
    rows: list[tuple[str, str]] = []
    for path in sorted(set(staged_map) | set(unstaged_map)):
        left = " "
        right = " "
        if path in staged_map:
            left = {"new file": "A", "modified": "M", "deleted": "D"}[staged_map[path]]
        if path in unstaged_map:
            right = "M"
        rows.append((left + right, path))
    for path in sorted(untracked):
        rows.append(("??", path))
    return [f"{code} {path}" for code, path in sorted(rows, key=lambda r: r[1])]


def _run_diff(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan, frozenset({"--unified", "-U"}))
    unsupported = {
        "--word-diff",
        "--color",
        "--color-words",
        "-w",
        "--ignore-all-space",
        "--binary",
        "--raw",
        "--numstat",
        "--shortstat",
        "--summary",
        "--patch-with-stat",
        "-M",
        "-C",
        "--find-renames",
        "--diff-filter",
        "--relative",
        "--no-index",
        "-G",
        "-S",
        "--stat",
        "--name-status",
    }
    if plan.flags & unsupported or set(plan.values) & unsupported:
        return ParseFailure("unsupported_form", "diff flag")
    if any(".." in token for token in plan.paths):
        return ParseFailure("unsupported_form", "diff range")
    against_head = "HEAD" in plan.paths
    operands = [p for p in plan.paths if p != "HEAD"]
    normalized_operands = _normalize_paths(operands, views.listing_set)
    if operands and any(p not in views.listing_set for p in normalized_operands):
        return ParseFailure("unsupported_form", "diff pathspec")
    scope = _normalize_paths(operands, views.listing_set)
    if against_head:
        pairs, unknown = _diff_head(views, scope)
    else:
        pairs, unknown = _diff_pairs(views, bool({"--cached", "--staged"} & plan.flags), scope)
    lines: list[str] = []
    if "--name-only" in plan.flags:
        lines = [path for path, _ in pairs]
    else:
        for _, block in pairs:
            lines.extend(block)
    if not lines and scope:
        return ParseFailure("unsupported_form", "named path shows no change we can verify")
    lines = _apply_pipeline(lines, SearchPlan(pattern="", targets=[], pipeline=plan.pipeline))
    quiet = bool({"--quiet", "--exit-code"} & plan.flags)
    returncode = 1 if quiet and lines else 0
    if quiet:
        lines = []
    return GitResult(
        output="\n".join(lines),
        returncode=returncode,
        empty=not lines,
        exact=not unknown,
        incomplete=unknown,
    )


def _run_status(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan)
    if plan.flags - {"-s", "--short", "--porcelain", "--long", "--"}:
        return ParseFailure("unsupported_form", "status flag")
    scope = _normalize_paths(plan.paths, views.listing_set) if plan.paths else []
    if not any(_in_scope(path, scope) for path in views.touched()) and not views.state.ledger:
        return ParseFailure("unsupported_form", "no observed change in status scope")
    if _status_sets(views, scope)[3]:
        return ParseFailure("unsupported_form", "file with unknown content in status scope")
    if {"-s", "--short", "--porcelain"} & plan.flags:
        lines = _render_status_short(views, scope)
    else:
        lines = _render_status_long(views, meta, scope)
    lines = _apply_pipeline(lines, SearchPlan(pattern="", targets=[], pipeline=plan.pipeline))
    return GitResult(output="\n".join(lines), empty=not lines)


def _run_add(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan)
    if plan.flags - {"-A", "--all", "-u", "--update", "-f", "--force", "-v", "--verbose", "--"}:
        return ParseFailure("unsupported_form", "add flag")
    return GitResult(output="", empty=True)


def _run_ls_files(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan)
    if plan.flags - {"--"}:
        return ParseFailure("unsupported_form", "ls-files flag")
    scope = _normalize_paths(plan.paths, views.listing_set) if plan.paths else []
    paths = [p for p in views.listing if _in_scope(p, scope)]
    extra = sorted(p for p in views.state.index if p not in views.listing_set)
    lines = sorted(set(paths) | {p for p in extra if _in_scope(p, scope)})
    lines = _apply_pipeline(lines, SearchPlan(pattern="", targets=[], pipeline=plan.pipeline))
    return GitResult(output="\n".join(lines), empty=not lines)


def _run_rev_parse(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan)
    if not meta.sha:
        return ParseFailure("unsupported_form", "no head sha")
    if "--abbrev-ref" in plan.flags:
        value = "HEAD"
    elif "--short" in plan.flags:
        value = meta.short
    elif "--show-toplevel" in plan.flags:
        return ParseFailure("unsupported_form", "toplevel path unknown")
    elif plan.paths in (["HEAD"], []):
        value = meta.sha
    else:
        return ParseFailure("unsupported_form", "rev-parse target")
    return GitResult(output=value, empty=False)


def _run_remote(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan)
    if not meta.owner or not meta.repo:
        return ParseFailure("unsupported_form", "remote unknown")
    if plan.paths and plan.paths[0] != "show":
        return ParseFailure("unsupported_form", "remote subcommand")
    url = f"https://github.com/{meta.owner}/{meta.repo}"
    if "-v" in plan.flags or "--verbose" in plan.flags:
        lines = [f"origin\t{url} (fetch)", f"origin\t{url} (push)"]
    else:
        lines = ["origin"]
    return GitResult(output="\n".join(lines), empty=False)


def _run_show(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan)
    if plan.flags - {"--stat", "--", "--no-color"}:
        return ParseFailure("unsupported_form", "show flag")
    wanted: list[str] = []
    if not plan.paths:
        if not (meta.detached and meta.sha):
            return ParseFailure("unsupported_form", "show HEAD is the harness commit")
        rev = meta.sha
    elif len(plan.paths) == 1:
        rev = plan.paths[0]
    elif "--" in plan.flags:
        rev = plan.paths[0]
        wanted = _normalize_paths(plan.paths[1:], views.listing_set)
        if not all(path in views.listing_set for path in wanted):
            return ParseFailure("unsupported_form", "show pathspec not tracked")
    else:
        return ParseFailure("unsupported_form", "show needs one rev")
    if ":" in rev:
        return ParseFailure("unsupported_form", "show <rev>:<path>")
    if not re.fullmatch(r"[0-9a-f]{6,40}", rev):
        return ParseFailure("unsupported_form", "show non-sha rev")
    if not callable(meta.commit_patch):
        return ParseFailure("unsupported_form", "no patch source")
    text = meta.commit_patch(rev)
    if not text:
        return ParseFailure("unsupported_form", "patch unavailable")
    patch = _parse_patch(text)
    if patch is None:
        return ParseFailure("unsupported_form", "patch unparsed")
    lines = _commit_header(patch)
    if "--stat" in plan.flags:
        if wanted:
            return ParseFailure("unsupported_form", "show --stat with pathspec")
        lines += ["", *patch["stat"]]
    else:
        abbrev = views.state.abbrev
        if abbrev is None:
            return ParseFailure("unsupported_form", "abbrev length unknown")
        diff = _filter_diff(patch["diff"], wanted) if wanted else patch["diff"]
        if wanted and not diff:
            return ParseFailure("unsupported_form", "commit does not touch the pathspec")
        lines += ["", *_retarget_funcnames(_reabbrev(diff, abbrev), views)]
    lines = _apply_pipeline(lines, SearchPlan(pattern="", targets=[], pipeline=plan.pipeline))
    return GitResult(output="\n".join(lines), empty=not lines)


_LOG_COUNT = re.compile(r"^-(\d+)$")
_LOG_ALLOWED = {"--oneline", "--no-decorate", "--no-merges", "--no-color", "--"}


def _run_log(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    if "--oneline" not in plan.args:
        return ParseFailure("unsupported_form", "log format")
    limit = 0
    paths: list[str] = []
    after_dashes = False
    index = 0
    while index < len(plan.args):
        token = plan.args[index]
        index += 1
        if after_dashes:
            paths.append(token)
            continue
        if token == "--":
            after_dashes = True
            continue
        if match := _LOG_COUNT.match(token):
            limit = int(match.group(1))
            continue
        if token in ("-n", "--max-count") and index < len(plan.args):
            limit = int(plan.args[index])
            index += 1
            continue
        if token.startswith("--max-count="):
            limit = int(token.split("=", 1)[1])
            continue
        if token in _LOG_ALLOWED:
            continue
        if token.startswith("-"):
            return ParseFailure("unsupported_form", f"log flag {token}")
        paths.append(token)

    if len(paths) > 1:
        return ParseFailure("unsupported_form", "log with several pathspecs")
    abbrev = views.state.abbrev
    if abbrev is None:
        return ParseFailure("unsupported_form", "abbrev length unknown")
    scoped = None
    if paths:
        scoped = _to_repo_relative(paths[0], views.listing_set)
        if not scoped or scoped not in views.listing_set:
            return ParseFailure("unsupported_form", "log pathspec not tracked")
    if not callable(meta.history):
        return ParseFailure("unsupported_form", "no history source")
    payload = meta.history(scoped)
    if not payload or not payload.get("commits"):
        return ParseFailure("unsupported_form", "history unavailable")
    commits = payload["commits"]
    if not limit and not payload.get("complete"):
        if not plan.evidence:
            return ParseFailure("unsupported_form", "history incomplete for an unlimited log")
        limit = EVIDENCE_LOG_LIMIT

    lines = [f"{entry['sha'][:abbrev]} {entry['subject']}" for entry in commits]
    if scoped is None and not (views.state.detached or meta.detached):
        head = views.state.head_short
        if not head:
            return ParseFailure("unsupported_form", "harness commit sha unknown")
        lines.insert(0, f"{head} {views.state.head_subject or HARNESS_SUBJECT}")
    if limit:
        lines = lines[:limit]
    lines = _apply_pipeline(lines, SearchPlan(pattern="", targets=[], pipeline=plan.pipeline))
    return GitResult(output="\n".join(lines), empty=not lines)


def _run_checkout(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan, frozenset({"-b", "-B"}))
    if plan.flags - {"--", "-f", "--force", "-q", "--quiet"}:
        return ParseFailure("unsupported_form", "checkout flag")
    operands = list(plan.paths)
    if not operands:
        return ParseFailure("unsupported_form", "checkout without pathspec")
    if "--" in plan.flags:
        return GitResult(output="", empty=True)
    normalized = _normalize_paths(operands, views.listing_set)
    if not all(p in views.listing_set for p in normalized):
        return ParseFailure("unsupported_form", "checkout target not a tracked path")
    count = len(normalized)
    return GitResult(
        output=UPDATED_PATHS_LINE.format(n=count, s="" if count == 1 else "s"), empty=False
    )


def _run_restore(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan, frozenset({"--source"}))
    if plan.flags - {"--", "--staged", "-S", "--worktree", "-W"}:
        return ParseFailure("unsupported_form", "restore flag")
    if not plan.paths:
        return ParseFailure("unsupported_form", "restore without pathspec")
    return GitResult(output="", empty=True)


def _run_reset(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan)
    if plan.flags - {"--", "--hard", "--soft", "--mixed", "-q", "--quiet"}:
        return ParseFailure("unsupported_form", "reset flag")
    revs = [p for p in plan.paths if p == "HEAD"]
    operands = [p for p in plan.paths if p != "HEAD"]
    if len(revs) + len(operands) != len(plan.paths):
        return ParseFailure("unsupported_form", "reset rev")
    if "--soft" in plan.flags or "-q" in plan.flags or "--quiet" in plan.flags:
        return GitResult(output="", empty=True)
    if "--hard" in plan.flags:
        if operands:
            return ParseFailure("unsupported_form", "reset --hard with paths")
        _resolve_head(views, meta)
        short = views.state.head_short
        subject = views.state.head_subject
        if not short or subject is None:
            return ParseFailure("unsupported_form", "head subject unknown")
        return GitResult(output=HARD_RESET_LINE.format(short=short, subject=subject), empty=False)
    scope = _normalize_paths(operands, views.listing_set) if operands else []
    residual = _dirty_paths(views, scope, "head")
    if not residual:
        return GitResult(output="", empty=True)
    return GitResult(
        output="\n".join([RESET_HEADER] + [f"M\t{path}" for path in residual]), empty=False
    )


def _run_stash(plan: GitPlan, views: _Views, meta: GitMeta) -> GitResult | ParseFailure:
    _split_args(plan)
    action = plan.paths[0] if plan.paths else "push"
    state = views.state
    if action in ("pop", "apply") and not state.stash:
        return GitResult(output=STASH_MISSING_LINE, returncode=1, empty=False)
    if action not in ("push", "save"):
        return ParseFailure("unsupported_form", f"git stash {action}")
    if not _dirty_paths(views, [], "head") and not state.index:
        return GitResult(output=STASH_EMPTY_LINE, empty=False)
    _resolve_head(views, meta)
    short = state.head_short
    subject = state.head_subject
    if not short or subject is None:
        return ParseFailure("unsupported_form", "head subject unknown")
    branch = "(no branch)" if state.detached else (state.branch or meta.branch or DEFAULT_BRANCH)
    return GitResult(
        output=STASH_SAVED_LINE.format(branch=branch, short=short, subject=subject), empty=False
    )


_HANDLERS = {
    "status": _run_status,
    "diff": _run_diff,
    "add": _run_add,
    "ls-files": _run_ls_files,
    "rev-parse": _run_rev_parse,
    "remote": _run_remote,
    "checkout": _run_checkout,
    "log": _run_log,
    "show": _run_show,
    "restore": _run_restore,
    "reset": _run_reset,
    "stash": _run_stash,
}
