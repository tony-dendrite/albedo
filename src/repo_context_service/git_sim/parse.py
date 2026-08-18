from __future__ import annotations

import re

from ..command_search import (
    ParseFailure,
    _mask_quoted,
    _parse_pipe_stage,
    _redirects_stdout,
    _split_pipeline,
    _strip_redirects,
    split_chain,
)
from .models import GitPlan
from .templates import _GLOBAL_BOOL_FLAGS, _GLOBAL_VALUE_FLAGS

_STDERR_MERGE = re.compile(r"\s2>&1")
_STDERR_DROP = re.compile(r"\s2>\s*/dev/null")
STDERR_ONLY_OUTPUT = {"checkout"}


_SINGLE_REDIRECT = re.compile(r"(?<!>)>(?!>)\s*([^\s<>|&;]+)\s*$")


def _redirect_target(text: str) -> str | None:
    match = _SINGLE_REDIRECT.search(text.strip())
    if not match or text.count(">") != 1:
        return None
    return match.group(1)


def _strip_stderr_redirects(cmd: str) -> tuple[str, bool]:
    text = _STDERR_MERGE.sub("", cmd or "")
    stripped = _STDERR_DROP.sub("", text)
    return stripped, stripped != text


def parse_git(cmd: str) -> GitPlan | ParseFailure:
    text, dropped_stderr = _strip_stderr_redirects(cmd)
    segments = _split_pipeline(text)
    if isinstance(segments, ParseFailure):
        return segments
    redirect = _redirect_target(text) if any(_redirects_stdout(s) for s in segments) else None
    if redirect is None and any(_redirects_stdout(seg) for seg in segments):
        return ParseFailure("unsupported_shell", "output redirected to a file")
    segments = [seg for seg in (_strip_redirects(s) for s in segments) if seg]
    if not segments:
        return ParseFailure("unparsed", "empty after redirects")

    head = segments[0]
    if head[0] != "git":
        return ParseFailure("not_git", head[0])

    index = 1
    while index < len(head):
        token = head[index]
        if token in _GLOBAL_VALUE_FLAGS:
            index += 2
            continue
        if token in _GLOBAL_BOOL_FLAGS:
            index += 1
            continue
        if token.startswith("-"):
            return ParseFailure("unknown_flag", token)
        break
    if index >= len(head):
        return ParseFailure("unsupported_form", "git without subcommand")

    plan = GitPlan(sub=head[index], args=head[index + 1 :], raw=cmd)
    plan.dropped_stderr = dropped_stderr
    plan.redirect = redirect
    for tokens in segments[1:]:
        stage = _parse_pipe_stage(tokens)
        if isinstance(stage, ParseFailure):
            return stage
        plan.pipeline.append(stage)
    return plan


def _split_args(plan: GitPlan, value_flags: frozenset[str] = frozenset()) -> None:
    operands: list[str] = []
    after_dashes = False
    index = 0
    while index < len(plan.args):
        token = plan.args[index]
        index += 1
        if after_dashes:
            operands.append(token)
            continue
        if token == "--":
            after_dashes = True
            plan.flags.add("--")
            continue
        if token.startswith("--"):
            name, sep, value = token.partition("=")
            if sep:
                plan.values[name] = value
            elif name in value_flags and index < len(plan.args):
                plan.values[name] = plan.args[index]
                index += 1
            else:
                plan.flags.add(name)
            continue
        if token.startswith("-") and len(token) > 1:
            if token in value_flags and index < len(plan.args):
                plan.values[token] = plan.args[index]
                index += 1
            else:
                for letter in token[1:]:
                    plan.flags.add("-" + letter)
            continue
        operands.append(token)
    plan.paths = operands


_RETURNCODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>")
_GIT_HEAD = re.compile(r"(?:^|[;&|(]\s*|&&\s*)(?:\w+=\S+\s+)*(?:sudo\s+|env\s+)?git\s")
_CD_ONLY = re.compile(r"^cd\s+[^\s&;|]+$")
_GIT_SUB = re.compile(
    r"\bgit\s+(?:(?:-C|-c|--git-dir|--work-tree)\s+\S+\s+|--no-pager\s+)*([a-z][a-z-]*)"
)


_SHORT_STATUS = re.compile(r"\bgit\s+status\b[^|&]*\s(?:-s\b|--short\b|--porcelain\b)")
_STASH_RESTORE = re.compile(r"\bgit\s+stash\s+(?:pop|apply)\b")


def _subcommand_of(stage: str) -> str:
    match = _GIT_SUB.search(stage or "")
    return match.group(1) if match else ""


_UNSPLITTABLE = re.compile(r"<<|;|\|\||`|\$\(")


_HEREDOC_START = re.compile(r"<<-?\s*'?\w+'?")


def _split_ampersand(text: str) -> list[str]:
    out, buf, quote, index = [], [], None, 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                quote = None
            buf.append(char)
        elif char in "\"'":
            quote = char
            buf.append(char)
        elif text[index : index + 2] == "&&":
            out.append("".join(buf))
            buf = []
            index += 2
            continue
        else:
            buf.append(char)
        index += 1
    out.append("".join(buf))
    return [stage.strip() for stage in out if stage.strip()]


def mutation_stages(command: str) -> list[str]:
    text = command or ""
    cut = _HEREDOC_START.search(_mask_quoted(text))
    if cut:
        text = text[: cut.start()]
    return _split_ampersand(text)


def chain_stages(cmd: str) -> list[str] | None:
    text = (cmd or "").strip()
    if not text or _UNSPLITTABLE.search(_mask_quoted(text)):
        return None
    return split_chain(text) or [text]


def _observed_returncode(observation: str) -> int:
    match = _RETURNCODE.search(observation or "")
    return int(match.group(1)) if match else 0
