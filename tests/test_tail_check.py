from __future__ import annotations

from albedo_eval_service.simulator.prompt_simulator import simulation_system_prompt
from sanity_service.tail_check import (
    DUP_CMD_THRESHOLD,
    MAX_RUN_THRESHOLD,
    loop_stats,
    looping_reason,
)


def _turns(commands: list[str]) -> list[str]:
    return [f"THOUGHT: next step.\n\n```bash\n{command}\n```" for command in commands]


# submission 0a9a8532 sample 0, verbatim order: 15 of its 28 commands are the submit command,
# six of them consecutively at the end, while the model lands four distinct `sed -i` edits.
_SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_ANSWER"
_SUBMIT_SPAM = [
    "nl -ba conan/cli/commands/config.py | sed -n '70,115p'",
    "sed -i -e 's/^from conan.cli.formatters import default_json_formatter$/from x import y/' c.py",
    "nl -ba conan/cli/commands/config.py | sed -n '1,12p;75,82p'",
    "cd /workspaces/conan && python -m conan.cli.cli config_list",
    "cat <<'PYEOF' > /tmp/test_config_list_filter.py import fnmatch PYEOF",
    _SUBMIT,
    "sed -i '80,82c\\ subparser.add_argument('pattern', nargs=\"?\")' config.py",
    "nl -ba conan/cli/commands/config.py | sed -n '76,84p'",
    _SUBMIT,
    "cd /workspaces/conan && python -m pytest test/functional/cli/test_cli_config_commands.py",
    _SUBMIT,
    _SUBMIT,
    "sed -i \"88s/subparser.add_argument('pattern')/x/\" config.py",
    _SUBMIT,
    _SUBMIT,
    'sed -i \'89s/.*/ """Show all builtin configurations"""/\' config.py',
    _SUBMIT,
    _SUBMIT,
    "nl -ba conan/cli/commands/config.py | sed -n '74,102p'",
    _SUBMIT,
    _SUBMIT,
    'grep -n -A 12 "def show" conan/api/subapi/config.py conan/internal/model/conf.py',
    *[_SUBMIT] * 6,
]

# submission d45cce5b sample 0, verbatim order: the prescribed workflow, nothing else
_VERIFY_CYCLE = [
    "git diff",
    "sed -n '145,155p' ably/realtime_conn.go",
    'grep -rn "recoverable" ably/ --include="*.go" | head -20',
    'go test ./ably/... -run "Conn" -v 2>&1 | head -50',
    'grep -n "func recoverable" ably/realtime_conn.go',
    "git status --short && git log --oneline -3",
    "git diff --name-only",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "sed -n '145,155p' ably/realtime_conn.go",
    "git diff",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "sed -n '145,155p' ably/realtime_conn.go",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    'grep -rn "recoverable" ably/ --include="*.go" | head -20',
    'go test ./ably/... -run "Conn" -v 2>&1 | head -50',
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "git diff",
    "sed -n '145,155p' ably/realtime_conn.go",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "git status --short && git log --oneline -3",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "sed -n '145,155p' ably/realtime_conn.go",
    'grep -n "recoverable" ably/realtime_conn.go',
    'go test ./ably/... -run "Conn" -v 2>&1 | head -50',
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "git diff",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
    "sed -n '145,155p' ably/realtime_conn.go",
    "echo FINALIZE_AND_SUBMIT_TASK_OUTPUT",
]


def test_our_own_submit_demands_do_not_count_as_the_model_looping():
    stats = loop_stats(_turns(_SUBMIT_SPAM))
    assert stats["n_cmds"] == len(_SUBMIT_SPAM) - _SUBMIT_SPAM.count(_SUBMIT)
    assert stats["dup_cmd_ratio"] == 0.0
    assert stats["max_cmd_run"] == 1
    assert not looping_reason(_turns(_SUBMIT_SPAM))


def test_the_prescribed_verify_cycle_is_not_a_loop():
    assert not looping_reason(_turns(_VERIFY_CYCLE))
    assert loop_stats(_turns(_VERIFY_CYCLE))["dup_cmd_ratio"] < DUP_CMD_THRESHOLD


def test_a_stuck_model_still_fails():
    stuck = ["sed -n '2273,2310p' pandas/io/stata.py"] * 21
    reason = looping_reason(_turns(stuck))
    assert "looping" in reason
    assert loop_stats(_turns(stuck))["max_cmd_run"] == 21


def test_the_consecutive_rule_is_untouched():
    varied = [f"cat file_{i}.py" for i in range(20)]
    assert not looping_reason(_turns(varied))
    assert looping_reason(_turns(varied + ["git diff"] * MAX_RUN_THRESHOLD))


def test_the_ratio_rule_still_catches_a_heavy_non_consecutive_loop():
    alternating = ["cat a.py", "cat b.py"] * 10
    stats = loop_stats(_turns(alternating))
    assert stats["max_cmd_run"] == 1
    assert stats["dup_cmd_ratio"] >= DUP_CMD_THRESHOLD
    assert "duplicate command ratio" in looping_reason(_turns(alternating))


def test_the_simulator_is_told_not_to_pre_apply_a_pending_request():
    prompt = simulation_system_prompt("openhands")
    assert "Never pre-apply a pending request" in prompt
    assert "BEFORE that" in prompt


def test_a_submit_the_harness_would_ignore_is_still_counted():
    from albedo_eval_service.shared.submit_protocol import asked_submit as _asked_submit

    assert _asked_submit("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
    assert _asked_submit("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt")
    assert _asked_submit("echo SUBMIT_TASK_33360A0E && git add -A && git diff --cached")
    assert not _asked_submit(
        "git add -A && git diff --cached && echo FINALIZE_AND_SUBMIT_TASK_OUTPUT"
    )
    assert not _asked_submit("git diff && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")

    marker_last = ["git diff && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"] * MAX_RUN_THRESHOLD
    assert looping_reason(_turns(marker_last)), "a spammed unregisterable submit is still a loop"


def test_a_sample_that_only_ever_submits_is_left_to_the_submit_checks():
    from types import SimpleNamespace

    from sanity_service.chain import empty_submit_count

    marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    only_submits = [f"echo {marker}"] * 12
    assert loop_stats(_turns(only_submits)) == {
        "n_cmds": 0,
        "dup_cmd_ratio": 0.0,
        "max_cmd_run": 0,
    }
    assert not looping_reason(_turns(only_submits))

    state = SimpleNamespace(
        turns=[
            {"role": "assistant", "content": turn, "score_target": True}
            for turn in _turns(only_submits)
        ],
        submits=[],
        submit_marker=marker,
        submit_clause=f"echo {marker}",
    )
    assert empty_submit_count(state, marker) >= 2
