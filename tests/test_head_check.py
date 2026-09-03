from __future__ import annotations

import asyncio
import json
from typing import Any

from albedo_eval_service.judge_llm_client import JudgeRawResponse
from sanity_service import head_check
from sanity_service.dispatcher import _TrajectoryState
from sanity_service.head_check import (
    _TRAJECTORY_STEP_CAP,
    _edit_under_review,
    _head_user,
    _parse_head_answers,
    _render_trajectory,
    is_scratch,
    run_head_check,
    turn_source_edit,
)
from sanity_service.rubricisity import PROOF_JUDGE_QUESTIONS, PROOF_JUDGE_SYSTEM


def _cand(text: str, command: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": f"THOUGHT: {text}\n\n```bash\n{command}\n```",
        "score_target": True,
    }


def _obs(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": f"<returncode>0</returncode>\n<output>{text}</output>",
        "environment_observation": True,
    }


def _injected(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text, "injected": True}


_TASK = {
    "role": "user",
    "content": (
        "<pr_description>frobnicate() crashes on empty input in mylib/core.py</pr_description>"
    ),
}


def _main_edit_turns() -> list[dict[str, Any]]:
    return [
        _TASK,
        _injected("Before continuing, please update the docstring of helpers.py"),
        _cand("micro", "sed -i 's/old doc/new doc/' pkg/helpers.py"),
        _obs(""),
        _cand("look around", "cat mylib/core.py"),
        _obs("def frobnicate(x): return x[0]"),
        _cand("fix the bug", "sed -i 's/x\\[0\\]/x[0] if x else None/' mylib/core.py"),
        _obs(""),
    ]


def _state(sample_id: str, turns: list[dict[str, Any]]) -> _TrajectoryState:
    return _TrajectoryState(sample_id=sample_id, prompt="", messages=[], turns=turns)


def _panel_from(script: list[Any]):
    calls = iter(script)

    async def _fake(client, system, user, models=(), temperature=None):
        assert "{{" not in system
        entry = next(calls)
        if entry is None:
            return [JudgeRawResponse(model="m", provider=None, raw="", error="timeout")]
        if isinstance(entry, str):
            return [JudgeRawResponse(model="m", provider=None, raw=entry)]
        p01, p02 = entry
        payload = {
            "answers": [
                {"id": "p_01", "answer": p01, "explanation": "never ran the failing case"},
                {"id": "p_02", "answer": p02, "explanation": "re-edited the same file"},
            ]
        }
        return [JudgeRawResponse(model="m", provider=None, raw=json.dumps(payload))]

    return _fake


def _run(monkeypatch, states, script: list[Any]):
    monkeypatch.setattr(head_check, "query_panel", _panel_from(script))
    return asyncio.run(run_head_check(states, client=object()))


def test_scratch_paths_are_exempt_and_real_sources_are_not():
    assert is_scratch("repro.py")
    assert is_scratch("/testbed/reproduce_bug.py")
    assert is_scratch("/tmp/check.sh")
    assert is_scratch("tests/test_frobnicate.py")
    assert is_scratch("fix_core.py")
    assert not is_scratch("mylib/core.py")
    assert not is_scratch("src/pkg/checks.py")
    assert not is_scratch("/testbed/mylib/core.py")


def test_source_edit_detection_per_command():
    assert turn_source_edit("", "sed -i 's/a/b/' mylib/core.py") == ["mylib/core.py"]
    assert turn_source_edit("", "sed -i 's/a/b/' repro.py") == []
    assert turn_source_edit("", "cat > repro.py <<'EOF'\nprint(1)\nEOF") is None
    created = turn_source_edit("", "cat > mylib/new_mod.py <<'EOF'\nx = 1\nEOF")
    assert created == ["mylib/new_mod.py"]
    assert turn_source_edit("", "git diff > patch.txt") is None
    body_only = "cat > notes.txt <<'EOF'\nsed -i 's/a/b/' mylib/core.py\nEOF"
    assert turn_source_edit(body_only, body_only) is None
    py_write = "python3 -c \"open('mylib/core.py','w').write('x')\""
    assert turn_source_edit(py_write, py_write) == ["mylib/core.py"]
    assert turn_source_edit("", "ls mylib && cat mylib/core.py") is None


def test_edit_under_review_targets_the_first_main_task_edit():
    edit = _edit_under_review(_main_edit_turns())
    assert edit is not None
    assert edit["paths"] == ["mylib/core.py"]
    assert edit["turn"] == 3
    assert "core.py" in edit["command"]
    assert _edit_under_review([_TASK, _cand("explore", "ls mylib"), _obs("core.py")]) is None


def test_injected_instruction_targets_are_skipped_until_the_mention_expires():
    def turns_with_edit_at(candidate: int) -> list[dict[str, Any]]:
        fillers = [_cand(f"t{i}", f"echo {i}") for i in range(candidate - 1)]
        return [
            _TASK,
            _injected("Please update the header comment in pkg/helpers.py"),
            *fillers,
            _cand("late edit", "sed -i 's/a/b/' pkg/helpers.py"),
        ]

    assert _edit_under_review(turns_with_edit_at(8)) is None
    edit = _edit_under_review(turns_with_edit_at(9))
    assert edit is not None and edit["paths"] == ["pkg/helpers.py"]


def test_edits_past_the_step_cap_are_not_reviewed():
    fillers = [_cand(f"t{i}", f"echo {i}") for i in range(_TRAJECTORY_STEP_CAP)]
    late = [_TASK, *fillers, _cand("late fix", "sed -i 's/a/b/' mylib/core.py")]
    assert _edit_under_review(late) is None


def test_render_keeps_observations_and_stops_at_the_step_cap():
    text = _render_trajectory(_main_edit_turns())
    assert "CANDIDATE OUTPUT 1" in text
    assert "ENVIRONMENT OBSERVATION (context only, do not score)" in text
    assert "CONTEXT USER (do not score)" in text
    assert "def frobnicate(x): return x[0]" in text
    long = [_TASK] + [_cand(f"t{i}", f"echo {i}") for i in range(_TRAJECTORY_STEP_CAP + 6)]
    text = _render_trajectory(long)
    assert f"CANDIDATE OUTPUT {_TRAJECTORY_STEP_CAP}" in text
    assert f"CANDIDATE OUTPUT {_TRAJECTORY_STEP_CAP + 1}" not in text


def test_answer_parsing_is_strict_and_notes_are_capped():
    payload = {
        "answers": [
            {"id": "p_01", "answer": 0, "explanation": "x" * 500},
            {"id": "p_02", "answer": 1, "explanation": "ok"},
        ]
    }
    parsed = _parse_head_answers(json.dumps(payload))
    assert parsed is not None
    answers, notes = parsed
    assert answers == {"p_01": 0, "p_02": 1}
    assert len(notes["p_01"]) == 200
    assert _parse_head_answers("no json here at all") is None
    assert _parse_head_answers('{"answers": [{"id": "p_01", "answer": 1}]}') is None
    bad_value = '{"answers": [{"id": "p_01", "answer": "yes"}, {"id": "p_02", "answer": 1}]}'
    assert _parse_head_answers(bad_value) is None


def test_prompt_templates_format_cleanly():
    system = PROOF_JUDGE_SYSTEM.format()
    assert '{"answers"' in system
    assert "{{" not in system
    edit = _edit_under_review(_main_edit_turns())
    user = _head_user(str(_TASK["content"]), edit, _render_trajectory(_main_edit_turns()))
    assert "EDIT UNDER REVIEW" in user
    assert "mylib/core.py" in user
    assert f"section {edit['turn']}" in user or str(edit["turn"]) in user
    for qid, _ in PROOF_JUDGE_QUESTIONS:
        assert qid in user


def test_both_zero_veto_fails_and_a_single_zero_passes(monkeypatch):
    states = [_state(f"s{i}", _main_edit_turns()) for i in range(3)]
    verdicts = _run(monkeypatch, states, [(0, 0), (1, 0), (0, 1)])
    assert [v.passed for v in verdicts] == [False, True, True]
    assert "never observed the error: never ran the failing case" in verdicts[0].reason
    assert "churn: re-edited the same file" in verdicts[0].reason


def test_two_failed_samples_fail_the_submission_and_one_does_not(monkeypatch):
    states = [_state(f"s{i}", _main_edit_turns()) for i in range(3)]
    _run(monkeypatch, states, [(0, 0), (0, 0), (1, 1)])
    assert states[0].heuristic_reason.startswith("head_check:")
    assert states[1].heuristic_reason.startswith("head_check:")
    assert states[0].head_check_flag and states[1].head_check_flag
    assert not states[2].heuristic_reason and not states[2].head_check_flag

    states = [_state(f"s{i}", _main_edit_turns()) for i in range(3)]
    _run(monkeypatch, states, [(0, 0), (1, 1), (1, 1)])
    assert not states[0].heuristic_reason
    assert states[0].head_check_flag


def test_judge_problems_pass_open(monkeypatch):
    states = [_state("s0", _main_edit_turns()), _state("s1", _main_edit_turns())]
    verdicts = _run(monkeypatch, states, [None, "not json"])
    assert verdicts[0].passed and verdicts[0].reason == "head judge unavailable"
    assert verdicts[1].passed and verdicts[1].reason == "head judge unparsable"
    assert not states[0].heuristic_reason and not states[1].heuristic_reason


class _DummyClient:
    async def aclose(self):
        return None


def _gather_checks(monkeypatch, states, *, tail_delay: float, head_delay: float) -> list[str]:
    from sanity_service import tail_check

    head_calls: list[str] = []

    async def tail_panel(client, system, user, models=(), temperature=None):
        await asyncio.sleep(tail_delay)
        answers = [{"id": qid, "answer": 0} for qid, _ in tail_check.TAIL_JUDGE_QUESTIONS]
        return [JudgeRawResponse(model="m", provider=None, raw=json.dumps({"answers": answers}))]

    async def head_panel(client, system, user, models=(), temperature=None):
        head_calls.append(user)
        await asyncio.sleep(head_delay)
        answers = [
            {"id": "p_01", "answer": 0, "explanation": "e1"},
            {"id": "p_02", "answer": 0, "explanation": "e2"},
        ]
        return [JudgeRawResponse(model="m", provider=None, raw=json.dumps({"answers": answers}))]

    monkeypatch.setattr(tail_check, "query_panel", tail_panel)
    monkeypatch.setattr(tail_check, "make_client", lambda *a, **k: _DummyClient())
    monkeypatch.setattr(head_check, "query_panel", head_panel)
    monkeypatch.setattr(head_check, "make_client", lambda *a, **k: _DummyClient())

    async def _run_both():
        await asyncio.gather(tail_check.run_tail_check(states), run_head_check(states))

    asyncio.run(_run_both())
    return head_calls


def _long_states() -> list[_TrajectoryState]:
    def long_turns(tag: str) -> list[dict[str, Any]]:
        fillers = [_cand(f"{tag}{i}", f"cat {tag}_file_{i}.py") for i in range(16)]
        return _main_edit_turns() + fillers

    looper = [_TASK] + [_cand(f"l{i}", "sed -i 's/a/b/' mylib/core.py") for i in range(21)]
    return [_state("s0", long_turns("a")), _state("s1", long_turns("b")), _state("lp", looper)]


def test_gathered_tail_and_head_checks_do_not_race(monkeypatch):
    states = _long_states()
    head_calls = _gather_checks(monkeypatch, states, tail_delay=0.001, head_delay=0.01)
    assert states[2].heuristic_reason.startswith("tail_check: looping")
    assert len(head_calls) == 2 and not states[2].head_check_flag
    for state in states[:2]:
        assert state.heuristic_reason.startswith("tail_check:")
        assert state.head_check_flag

    states = _long_states()
    head_calls = _gather_checks(monkeypatch, states, tail_delay=0.01, head_delay=0.001)
    assert len(head_calls) == 2
    for state in states[:2]:
        assert state.heuristic_reason.startswith("tail_check:")
        assert state.head_check_flag


def test_decided_states_and_no_edit_states_are_skipped(monkeypatch):
    no_edit = _state("ne", [_TASK, _cand("explore", "ls mylib"), _obs("core.py")])
    errored = _state("er", _main_edit_turns())
    errored.error = "boom"
    decided = _state("ch", _main_edit_turns())
    decided.heuristic_reason = "chain: something"
    verdicts = _run(monkeypatch, [no_edit, errored, decided], [])
    assert len(verdicts) == 1
    assert verdicts[0].checked is False and verdicts[0].passed
    assert verdicts[0].reason == "no main-task source edit"
    assert decided.heuristic_reason == "chain: something"
