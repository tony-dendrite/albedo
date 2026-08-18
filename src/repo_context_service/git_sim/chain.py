from __future__ import annotations

from copy import deepcopy

from ..command_search import ParseFailure, parse_search, run_search
from .execute import run_git
from .models import GitMeta, GitPlan, GitResult
from .parse import _CD_ONLY, _GIT_HEAD, chain_stages, mutation_stages, parse_git
from .session import apply_git
from .templates import GIT_EVIDENCE_HEADER


def run_git_chain(
    cmd: str,
    overlay,
    read_base,
    listing: list[str],
    meta: GitMeta | None = None,
    file_text=None,
    file_size=None,
) -> GitResult | ParseFailure:
    stages = chain_stages(cmd)
    if stages is None:
        return ParseFailure("unsupported_shell", "control operators")
    if not any(_GIT_HEAD.search(stage) for stage in stages):
        return ParseFailure("not_git", "no git stage")
    working = deepcopy(overlay)
    read_work = (lambda rel: file_text(working, rel)) if file_text else None
    size_work = (lambda rel: file_size(working, rel)) if file_size else None
    parts: list[str] = []
    returncode = 0
    for stage in stages:
        text = stage.strip()
        if _CD_ONLY.match(text):
            continue
        if _GIT_HEAD.search(text):
            plan = parse_git(text)
            if isinstance(plan, ParseFailure):
                return plan
            result = run_git(plan, working, read_base, listing, meta)
            if isinstance(result, ParseFailure):
                return result
            if not result.exact:
                return ParseFailure("unsupported_form", "git content unknown")
            if result.output:
                parts.append(result.output)
            returncode = result.returncode
            if returncode:
                break
            apply_git(working, text, "", listing, read_base)
            continue
        if read_work is None:
            return ParseFailure("unsupported_form", "chain stage not executable")
        plan = parse_search(text)
        if isinstance(plan, ParseFailure):
            return plan
        result = run_search(plan, read_work, listing, size_file=size_work)
        if isinstance(result, ParseFailure) or result.empty or result.incomplete or result.missing:
            return ParseFailure("unsupported_form", "chain stage output unknown")
        parts.append(result.output)
    return GitResult(output="\n".join(parts), returncode=returncode, empty=not parts)


_RELAXABLE = {"--all", "--graph", "--decorate", "--no-merges", "-p", "--patch", "-i", "--follow"}
_RELAXABLE_PREFIX = ("--grep", "--since", "--until", "--author", "--committer", "-S", "-G")


def _drop_filters(args: list[str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        index += 1
        if token in _RELAXABLE:
            continue
        if token.startswith(_RELAXABLE_PREFIX):
            if "=" not in token and index < len(args) and not args[index].startswith("-"):
                index += 1
            continue
        out.append(token)
    return out


def _relaxed(plan: GitPlan) -> GitPlan:
    return GitPlan(sub=plan.sub, args=_drop_filters(plan.args), raw=plan.raw)


def _before_pipe(stage: str) -> str:
    from ..command_search import _split_top_level_pipes

    return _split_top_level_pipes(stage)[0].strip()


def git_evidence(
    command: str, overlay, read_base, listing: list[str], meta: GitMeta | None = None
) -> str:
    if not _GIT_HEAD.search(command or ""):
        return ""
    meta = meta or GitMeta()
    stages = chain_stages(command) or mutation_stages(command)
    fragments: list[str] = []
    for stage in stages:
        if not _GIT_HEAD.search(stage):
            continue
        label = stage.strip()
        plan = parse_git(stage)
        partial = False
        if isinstance(plan, ParseFailure):
            head = _before_pipe(stage)
            if head == stage.strip():
                continue
            plan = parse_git(head)
            if isinstance(plan, ParseFailure):
                continue
            partial = True
        result = run_git(plan, overlay, read_base, listing, meta)
        if isinstance(result, ParseFailure) or not result.exact:
            relaxed = _relaxed(plan)
            if relaxed.args == plan.args and not plan.pipeline:
                continue
            relaxed.evidence = True
            result = run_git(relaxed, overlay, read_base, listing, meta)
            partial = True
        if partial:
            label = (
                f"{label}   (computed WITHOUT this command's pipes and filters "
                "— apply them yourself)"
            )
        if isinstance(result, ParseFailure) or not result.exact or not result.output:
            continue
        fragments.append(f"$ {label}\n{result.output}")
    if not fragments:
        return ""
    return "\n" + GIT_EVIDENCE_HEADER + "\n" + "\n\n".join(fragments) + "\n"
