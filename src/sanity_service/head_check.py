from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from albedo_eval_service.shared.json_extract import extract_json
from albedo_eval_service.shared.patch_lint import extract_commands
from sanity_service.chain import inplace_edited_paths
from sanity_service.judge_panel import make_client, query_panel
from sanity_service.rubricisity import (
    PROOF_JUDGE_QUESTIONS,
    PROOF_JUDGE_SYSTEM,
    PROOF_JUDGE_USER,
)

HEAD_JUDGE_MIN_FAILED_SAMPLES = 2

_TRAJECTORY_STEP_CAP = 24
_INJECTED_MENTION_TURNS = 8

SCRATCH_RE = re.compile(
    r"^(repro|reproduce|test|poc|scratch|debug|check|verify|demo|bug|min|issue|example|tmp)"
    r"[\w-]*\.(py|sh|js|ts|go|php|rb|c|cc|cpp|java|rs)$",
    re.I,
)
HELPER_RE = re.compile(
    r"^(fix|modify|apply|change|edit|patch|add|remove|insert|create)[\w-]*\.(py|sh|js|go|php|rb)$",
    re.I,
)
SCRATCH_DIR_RE = re.compile(r"(?:^|/)(?:test|tmp|scratch|repro|demo|example)[\w-]*/")
NOTES_EXT_RE = re.compile(r"\.(txt|md|log|json|out|diff|patch)$", re.I)
CREATE_TARGET_RE = re.compile(
    r"(?:\bcat\s*>>?|\btee\s+(?:-a\s+)?|<<\s*'?\"?\w+'?\"?\s*>)\s*['\"]?([\w./~-]+)"
)
PY_WRITE_RE = re.compile(r"open\s*\([^)]*['\"][wa]\+?['\"]|\.write_text\s*\(")
PY_PATH_RE = re.compile(r"['\"]((?:[\w.-]+/)*[\w.-]+\.py)['\"]")
EDIT_HEAD_RE = re.compile(r"\bsed\s+-i|\bgit\s+apply|\bpatch\s+-p|\bapplypatch|str_replace")
_SED_FALLBACK_RE = re.compile(
    r"sed\s+-i\S*\s+(?:-e\s+)?(?:'[^']*'|\"[^\"]*\"|\S+)\s+((?:/|\./)?[\w./-]+\.[A-Za-z]\w*)"
)
_CD_SCRATCH_RE = re.compile(
    r"(?:mkdir\s+-p\s+|cd\s+)(?:/tmp/|\S*/(?:test|tmp|scratch|repro|demo|example)[\w-]*(?:\s|/|$))"
)


def is_scratch(path: str) -> bool:
    path = re.sub(r"^\./", "", path)
    if path.startswith(("/tmp/", "tmp/")):
        return True
    base = path.rsplit("/", 1)[-1]
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    root_level = (
        not parent
        or parent in ("/tmp", "/testbed", "/workspace")
        or bool(re.match(r"^/(?:workspace|testbed)/[^/]+$", parent))
    )
    if root_level and (SCRATCH_RE.match(base) or HELPER_RE.match(base)):
        return True
    stripped = re.sub(r"^(?:/testbed/|/workspace/[^/]+/|/workspace/)", "", path)
    return bool(SCRATCH_DIR_RE.search(stripped)) and "src/" not in stripped


def turn_source_edit(body: str, command: str) -> list[str] | None:
    head = command.split("<<", 1)[0]
    sed_paths = [p for p in inplace_edited_paths(body) if not is_scratch(p)]
    if re.search(r"\bsed\s+-i", head):
        if not sed_paths:
            sed_paths = [p for p in _SED_FALLBACK_RE.findall(head) if not is_scratch(p)]
        return sed_paths or []
    if EDIT_HEAD_RE.search(head):
        return []
    if "git diff" in head or "git log" in head or "git show" in head:
        return None
    creations = CREATE_TARGET_RE.findall(head)
    cd_scratch = bool(_CD_SCRATCH_RE.search(head))
    real = [
        t
        for t in creations
        if not t.startswith(("/dev", "/tmp", "tmp/"))
        and not is_scratch(t)
        and not NOTES_EXT_RE.search(t)
        and not (cd_scratch and not t.startswith("/"))
    ]
    if creations:
        return real or None
    if re.search(r"python3?\s+-c", head) and PY_WRITE_RE.search(command):
        paths = [p for p in PY_PATH_RE.findall(command) if not is_scratch(p)]
        return paths or []
    return None


@dataclass
class HeadVerdict:
    sample_id: str
    checked: bool
    passed: bool
    reason: str = ""
    answers: dict[str, int] | None = None


def _edit_under_review(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    injected: list[tuple[int, str]] = []
    candidate_index = 0
    for turn in turns:
        content = str(turn.get("content") or "")
        if turn.get("injected"):
            injected.append((candidate_index, content))
            continue
        if turn.get("role") != "assistant" or not turn.get("score_target"):
            continue
        candidate_index += 1
        if candidate_index > _TRAJECTORY_STEP_CAP:
            return None
        recent = "\n".join(
            text for at, text in injected if candidate_index - at <= _INJECTED_MENTION_TURNS
        )
        for command in extract_commands(content):
            targets = turn_source_edit(content, command)
            if targets is None:
                continue
            bases = [p.rsplit("/", 1)[-1] for p in targets]
            if not bases or any(base in recent for base in bases):
                continue
            return {"turn": candidate_index, "command": command[:300], "paths": targets[:3]}
    return None


def _render_trajectory(turns: list[dict[str, Any]]) -> str:
    assistant_index = 0
    sections: list[str] = []
    for turn in turns:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").rstrip()
        if role == "assistant" and turn.get("score_target"):
            if assistant_index >= _TRAJECTORY_STEP_CAP:
                break
            assistant_index += 1
            label = f"CANDIDATE OUTPUT {assistant_index}"
        elif role == "user" and turn.get("environment_observation"):
            label = "ENVIRONMENT OBSERVATION (context only, do not score)"
        else:
            label = (
                f"CONTEXT {role.upper()} (do not score)" if role else "CONTEXT TURN (do not score)"
            )
        sections.append(f"{label}:\n------\n{content}\n------")
    return "\n\n".join(sections)


def _head_user(task: str, edit: dict[str, Any], trajectory: str) -> str:
    questions = "\n".join(f"{qid}: {text}" for qid, text in PROOF_JUDGE_QUESTIONS)
    return PROOF_JUDGE_USER.format(
        task=task or "",
        turn=edit["turn"],
        paths=", ".join(edit["paths"]) or "unknown",
        command=edit["command"],
        trajectory=trajectory,
        questions=questions,
    )


def _parse_head_answers(raw: str) -> tuple[dict[str, int], dict[str, str]] | None:
    obj = extract_json(raw, prefer_keys=("answers",))
    items = obj.get("answers") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return None
    answers: dict[str, int] = {}
    notes: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and str(item.get("id", "")) in dict(PROOF_JUDGE_QUESTIONS):
            try:
                answers[str(item["id"])] = 1 if int(item.get("answer", 0)) == 1 else 0
            except (TypeError, ValueError):
                continue
            notes[str(item["id"])] = str(item.get("explanation") or "")[:200]
    if len(answers) != len(PROOF_JUDGE_QUESTIONS):
        return None
    return answers, notes


async def judge_head(
    client, task: str, edit: dict[str, Any], trajectory: str, *, sample_id: str
) -> HeadVerdict:
    results = await query_panel(
        client,
        PROOF_JUDGE_SYSTEM.format(),
        _head_user(task, edit, trajectory),
        temperature=0.0,
    )
    usable = next((r for r in results if not r.error and r.raw.strip()), None)
    if usable is None:
        return HeadVerdict(sample_id, checked=True, passed=True, reason="head judge unavailable")
    parsed = _parse_head_answers(usable.raw)
    if parsed is None:
        return HeadVerdict(sample_id, checked=True, passed=True, reason="head judge unparsable")
    answers, notes = parsed
    if answers["p_01"] == 0 and answers["p_02"] == 0:
        reason = (
            "edited source before seeing the error and churned on the edit "
            f"[never observed the error: {notes.get('p_01', '')}; churn: {notes.get('p_02', '')}]"
        )
        return HeadVerdict(sample_id, checked=True, passed=False, reason=reason, answers=answers)
    return HeadVerdict(sample_id, checked=True, passed=True, answers=answers)


async def run_head_check(states, *, client=None) -> list[HeadVerdict]:
    verdicts: list[HeadVerdict] = []
    needs_judge = []
    for state in states:
        if state.error or state.heuristic_reason:
            continue
        edit = _edit_under_review(state.turns)
        if edit is None:
            verdicts.append(
                HeadVerdict(
                    state.sample_id,
                    checked=False,
                    passed=True,
                    reason="no main-task source edit",
                )
            )
            continue
        task = next(
            (str(t.get("content") or "") for t in state.turns if t.get("role") == "user"), ""
        )
        needs_judge.append((state, edit, task))

    if not needs_judge:
        return verdicts

    own_client = client is None
    if own_client:
        client = make_client()
    judge_failed: list = []
    try:
        for state, edit, task in needs_judge:
            verdict = await judge_head(
                client, task, edit, _render_trajectory(state.turns), sample_id=state.sample_id
            )
            verdicts.append(verdict)
            if not verdict.passed:
                state.head_check_flag = verdict.reason
                judge_failed.append((state, verdict))
                logger.warning("[head-check] {} {}", state.sample_id, verdict.reason)
    finally:
        if own_client:
            await client.aclose()
    if len(judge_failed) >= HEAD_JUDGE_MIN_FAILED_SAMPLES:
        for state, verdict in judge_failed:
            if not state.heuristic_reason:
                state.heuristic_reason = f"head_check: {verdict.reason}"
    elif judge_failed:
        logger.info(
            "[head-check] {} failed sample(s) below the {}-sample threshold, not failing",
            len(judge_failed),
            HEAD_JUDGE_MIN_FAILED_SAMPLES,
        )
    return verdicts
