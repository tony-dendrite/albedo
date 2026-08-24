from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .observation_format import action_blocks

DUP_CMD_THRESHOLD = 0.5
MAX_RUN_THRESHOLD = 4

MAX_LISTED_COMMANDS = 5
MAX_COMMAND_CHARS = 120

_CANDIDATE_RE = re.compile(
    r"^CANDIDATE OUTPUT(?: \d+)?:\n------\n(.*?)\n------"
    r"(?=\n+(?:CANDIDATE OUTPUT|ENVIRONMENT OBSERVATION|CONTEXT )[^\n]*:\n------|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class LoopingCommand:
    command: str
    count: int
    longest_run: int


@dataclass(frozen=True)
class LoopVerdict:
    looped: bool
    reasons: tuple[str, ...]
    commands: tuple[LoopingCommand, ...]
    n_cmds: int
    dup_cmd_ratio: float
    max_cmd_run: int


def candidate_turns(document: str) -> list[str]:
    turns = [match.group(1).rstrip() for match in _CANDIDATE_RE.finditer(document or "")]
    if turns:
        return turns
    return [document] if document else []


def commands_of(turns: list[str]) -> list[str]:
    cmds: list[str] = []
    for turn in turns:
        cmds += action_blocks(turn)
    return cmds


def loop_stats(turns: list[str]) -> dict:
    cmds = commands_of(turns)
    max_run = run = 1
    for prev, cur in zip(cmds, cmds[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)
    return {
        "n_cmds": len(cmds),
        "dup_cmd_ratio": 1 - len(set(cmds)) / len(cmds) if cmds else 0.0,
        "max_cmd_run": max_run if cmds else 0,
    }


def _longest_runs(cmds: list[str]) -> dict[str, int]:
    longest: dict[str, int] = {}
    run = 0
    for index, cmd in enumerate(cmds):
        run = run + 1 if index and cmd == cmds[index - 1] else 1
        if run > longest.get(cmd, 0):
            longest[cmd] = run
    return longest


def loop_verdict(turns: list[str]) -> LoopVerdict:
    cmds = commands_of(turns)
    stats = loop_stats(turns)
    counts = Counter(cmds)
    longest = _longest_runs(cmds)

    reasons: list[str] = []
    if stats["dup_cmd_ratio"] >= DUP_CMD_THRESHOLD:
        reasons.append(f"duplicate command ratio {stats['dup_cmd_ratio']:.2f}")
    if stats["max_cmd_run"] >= MAX_RUN_THRESHOLD:
        reasons.append(f"same command repeated {stats['max_cmd_run']}x consecutively")

    looping = [
        LoopingCommand(command=cmd, count=count, longest_run=longest.get(cmd, 1))
        for cmd, count in counts.items()
        if count >= 2 or longest.get(cmd, 1) >= MAX_RUN_THRESHOLD
    ]
    looping.sort(key=lambda item: (-item.longest_run, -item.count, item.command))

    return LoopVerdict(
        looped=bool(reasons),
        reasons=tuple(reasons),
        commands=tuple(looping) if reasons else (),
        **stats,
    )


def loop_verdict_for_document(document: str) -> LoopVerdict:
    return loop_verdict(candidate_turns(document))


def _render_command(entry: LoopingCommand) -> str:
    command = entry.command
    if len(command) > MAX_COMMAND_CHARS:
        command = command[: MAX_COMMAND_CHARS - 1] + "…"
    if entry.longest_run >= 2:
        return f"`{command}` {entry.count}x ({entry.longest_run} consecutive)"
    return f"`{command}` {entry.count}x"


def loop_explanation(verdict: LoopVerdict) -> str:
    listed = verdict.commands[:MAX_LISTED_COMMANDS]
    parts = [
        "Trajectory is looped, so every question is scored 0 without judging.",
        "; ".join(verdict.reasons) + ".",
    ]
    if listed:
        rendered = "; ".join(_render_command(entry) for entry in listed)
        hidden = len(verdict.commands) - len(listed)
        suffix = f"; and {hidden} more" if hidden > 0 else ""
        parts.append(f"Looping commands: {rendered}{suffix}.")
    return " ".join(parts)
