from __future__ import annotations

import asyncio

from albedo_config import JudgeSettings
from albedo_eval_service.judge_llm_client import JudgeRawResponse
from albedo_eval_service.shared.observation_format import has_content, requires_output
from albedo_eval_service.simulator.prompt_simulator import MUST_PRINT_RETRY
from sanity_service import dispatcher as D

_SILENT = "<returncode>0</returncode>\n<output>\n</output>"
_REAL = (
    "<returncode>0</returncode>\n<output>\n"
    "340\t    pub fn render(&self, state: &ProgressState) -> String {\n"
    "341\t        let mut out = String::new();\n"
    "</output>"
)
_READ = "sed -n '340,550p' src/style.rs"


class _Client:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        raw = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return JudgeRawResponse(model=kwargs["model"], provider="stub", raw=raw)

    async def aclose(self):
        return None


def _state() -> D._TrajectoryState:
    messages = [
        {"role": "user", "content": "fix the progress bar rounding"},
        {"role": "assistant", "content": "THOUGHT: looking\n\n```bash\nls\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\nsrc\n</output>"},
    ]
    return D._TrajectoryState(
        sample_id="mini-coder/shard-00000.parquet:0:1",
        prompt="fix the progress bar rounding",
        messages=list(messages),
        turns=[{"role": m["role"], "content": m["content"]} for m in messages],
    )


def _simulate(client, command: str) -> str:
    state = _state()
    return asyncio.run(
        D._simulate_observation(
            client=client,
            settings=JudgeSettings(engy_api_key=""),
            eval_run_id="run",
            state=state,
            assistant_output=f"THOUGHT: read it\n\n```bash\n{command}\n```",
        )
    )


def test_the_read_that_broke_0a9a8532_is_re_asked():
    client = _Client(_SILENT, _REAL)

    assert _simulate(client, _READ) == _REAL
    assert len(client.calls) == 2
    assert MUST_PRINT_RETRY in client.calls[1]["messages"][-1]["content"]


def test_a_command_that_may_legitimately_be_silent_is_not_re_asked():
    """`sed -i` prints nothing on success and `grep` exits 1 on no match. Re-asking those would
    invent output the shell never produced."""
    for quiet in ("sed -i 's/a/b/' src/style.rs", "grep -rn 'ProgressStyle' src/"):
        assert not requires_output(quiet), quiet
        client = _Client(_SILENT)
        assert _simulate(client, quiet) == _SILENT
        assert len(client.calls) == 1, quiet


def test_a_redirected_read_is_not_re_asked():
    """`cat f > copy.py` and `sed -n '1,10p' f > out` write to a file and print nothing. Forcing
    output here would invent text the shell never emitted."""
    for redirected in (
        "cat src/style.rs > /tmp/copy.rs",
        "sed -n '340,550p' src/style.rs > /tmp/out",
        "echo hello > /tmp/f",
        "git diff src/style.rs > patch.txt",
    ):
        assert not requires_output(redirected), redirected
        client = _Client(_SILENT)
        assert _simulate(client, redirected) == _SILENT, redirected
        assert len(client.calls) == 1, redirected


def test_the_submit_command_never_reaches_the_simulator():
    """`echo <marker>` does print, so requires_output says must-print — but a turn carrying the
    marker is routed to the submit path in _append_observations, never to _simulate_observation."""
    marker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    state = _state()
    state.submit_marker = marker
    state.turns.append(
        {"role": "assistant", "content": f"```bash\necho {marker}\n```", "score_target": True}
    )

    async def _boom(**_kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("the submit turn was handed to the simulator")

    import sanity_service.dispatcher as mod

    original = mod._simulate_observation
    mod._simulate_observation = _boom
    try:
        asyncio.run(mod._append_observations([state], "run", 1))
    finally:
        mod._simulate_observation = original
    assert state.submits, "the turn should have been recorded as a submission"


def test_a_read_that_printed_first_time_is_left_alone():
    client = _Client(_REAL)
    assert _simulate(client, _READ) == _REAL
    assert len(client.calls) == 1


def test_a_second_silence_keeps_the_original_answer():
    """If the simulator insists the read is empty, we take it rather than fabricate content."""
    client = _Client(_SILENT, _SILENT)
    assert _simulate(client, _READ) == _SILENT
    assert len(client.calls) == 3
    assert client.calls[-1]["model"] != client.calls[-2]["model"]


def test_the_retry_will_not_accept_a_degenerate_replacement():
    collapsed = (
        "<returncode>0</returncode>\n<output>\n"
        + "\n".join(f"{i}\t    // bump the counter" for i in range(340, 460))
        + "\n</output>"
    )
    client = _Client(_SILENT, collapsed)
    assert _simulate(client, _READ) == _SILENT, "a collapse must not pass as recovered output"


def test_which_corpus_commands_this_covers():
    covered = ["sed -n '340,550p' src/style.rs", "cat -n boltons/urlutils.py", "wc -l foo.py"]
    not_covered = ["grep -n 'def x' a.py", "sed -i 's/a/b/' a.py", "git diff"]
    assert all(requires_output(c) for c in covered)
    assert not any(requires_output(c) for c in not_covered)
    assert not has_content(_SILENT, "returncode")
    assert has_content(_REAL, "returncode")


def test_a_run_of_silences_drops_the_sample_as_infra():
    state = _state()
    state.submit_marker = "NEVER_MATCHES"

    async def _silent(**_kwargs):
        return _SILENT

    import sanity_service.dispatcher as mod

    async def _still_silent(*_args, **_kwargs):
        return ""

    original = mod._simulate_observation
    original_confirm = mod._confirm_silence
    mod._simulate_observation = _silent
    mod._confirm_silence = _still_silent
    try:
        for turn in range(mod.MAX_CONSECUTIVE_SILENT_OBSERVATIONS):
            assert not state.error, f"gave up after only {turn} silences"
            state.turns.append(
                {
                    "role": "assistant",
                    "content": "```bash\ngrep -rn 'ProgressStyle' src/\n```",
                    "score_target": True,
                }
            )
            asyncio.run(mod._append_observations([state], "run", turn))
    finally:
        mod._simulate_observation = original
        mod._confirm_silence = original_confirm

    assert state.consecutive_silent_observations == mod.MAX_CONSECUTIVE_SILENT_OBSERVATIONS
    assert "consecutive" in state.error and "no output" in state.error


def test_a_silence_run_broken_by_the_rescue_model_continues():
    state = _state()
    state.submit_marker = "NEVER_MATCHES"

    async def _silent(**_kwargs):
        return _SILENT

    async def _rescued(*_args, **_kwargs):
        return _REAL

    import sanity_service.dispatcher as mod

    original = mod._simulate_observation
    original_confirm = mod._confirm_silence
    mod._simulate_observation = _silent
    mod._confirm_silence = _rescued
    try:
        for turn in range(mod.MAX_CONSECUTIVE_SILENT_OBSERVATIONS + 2):
            state.turns.append(
                {
                    "role": "assistant",
                    "content": "```bash\ngrep -rn 'ProgressStyle' src/\n```",
                    "score_target": True,
                }
            )
            asyncio.run(mod._append_observations([state], "run", turn))
    finally:
        mod._simulate_observation = original
        mod._confirm_silence = original_confirm

    assert not state.error, "a silence run broken by the rescue model must not drop the sample"
    assert state.consecutive_silent_observations < mod.MAX_CONSECUTIVE_SILENT_OBSERVATIONS
    # the rescued observation replaced the silence that would have crossed the threshold
    assert any(_REAL in str(m.get("content", "")) for m in state.messages)


def test_one_real_answer_resets_the_silence_run():
    state = _state()
    state.submit_marker = "NEVER_MATCHES"
    replies = [_SILENT, _SILENT, _SILENT, _REAL, _SILENT, _SILENT, _SILENT]

    seen: list[int] = []
    import sanity_service.dispatcher as mod

    original = mod._simulate_observation

    async def _tracked(**_kwargs):
        out = replies[min(len(seen), len(replies) - 1)]
        seen.append(1)
        return out

    mod._simulate_observation = _tracked
    try:
        for turn in range(len(replies)):
            state.turns.append(
                {
                    "role": "assistant",
                    "content": "```bash\ngrep -rn 'x' src/\n```",
                    "score_target": True,
                }
            )
            asyncio.run(mod._append_observations([state], "run", turn))
    finally:
        mod._simulate_observation = original

    assert not state.error, "a run broken by a real answer must not be dropped"
    assert state.consecutive_silent_observations == 3


def test_the_simulator_is_not_taught_an_empty_form_the_real_data_never_uses():
    from albedo_eval_service.shared.observation_format import (
        NO_OUTPUT_SENTENCE,
        OPENHANDS,
        RETURNCODE,
        empty_output,
        silent_observation,
    )
    from albedo_eval_service.simulator.prompt_simulator import simulation_system_prompt

    for fmt in (OPENHANDS, RETURNCODE):
        assert NO_OUTPUT_SENTENCE not in simulation_system_prompt(fmt), fmt

    # the fallbacks already match their targets byte for byte
    assert empty_output(RETURNCODE) == "<returncode>0</returncode>\n<output>\n</output>"
    assert empty_output(OPENHANDS) == (
        "\n[The command completed with exit code 0.]\n[Command finished with exit code 0]"
    )
    # the constant stays a backstop: trajectories already carry the sentence
    assert silent_observation(NO_OUTPUT_SENTENCE)


def test_the_model_can_never_be_shown_the_invented_empty_form():
    from albedo_eval_service.shared.observation_format import (
        NO_OUTPUT_SENTENCE,
        OPENHANDS,
        canonical_empty,
        empty_output,
        valid_output,
    )

    assert valid_output(NO_OUTPUT_SENTENCE, OPENHANDS), "nothing else rejects it"
    assert canonical_empty(NO_OUTPUT_SENTENCE, OPENHANDS) == empty_output(OPENHANDS)

    client = _Client(NO_OUTPUT_SENTENCE)
    shown = _simulate(client, "sed -i 's/a/b/' src/style.rs")
    assert NO_OUTPUT_SENTENCE not in shown
    assert shown == empty_output(OPENHANDS) or shown == _SILENT


def test_real_output_is_never_rewritten():
    from albedo_eval_service.shared.observation_format import OPENHANDS, canonical_empty

    for keep in (
        "hello\n[The command completed with exit code 0.]",
        "Your command ran successfully and printed this instead.",
        "<returncode>0</returncode>\n<output>\nreal\n</output>",
    ):
        assert canonical_empty(keep, OPENHANDS) == keep, keep


def test_an_invented_sed_diagnostic_is_replaced_with_the_real_one():
    from albedo_eval_service.shared.sed_check import misdiagnosed_sed as _misdiagnosed_sed

    broken = (
        "sed -i '263,279d' a.py && sed -i '/^ def f(/,/^[[:space:]]*def /{ /x/!{ /y/a\\ z' b.py"
    )
    invented = (
        "<returncode>1</returncode>\n<output>\n"
        "sed: -e expression #1, char 12: unknown command: `N'\n</output>"
    )
    assert _misdiagnosed_sed(broken, invented) == "sed: -e expression #1, char 0: unmatched `{'"

    shown = _simulate(_Client(invented), broken)
    assert "unmatched" in shown and "unknown command" not in shown

    # already accurate, opaque, or not a sed complaint at all -> left alone
    accurate = (
        "<returncode>1</returncode>\n<output>\n"
        "sed: -e expression #1, char 0: unmatched `{'\n</output>"
    )
    assert _misdiagnosed_sed(broken, accurate) == ""
    assert _misdiagnosed_sed('sed -n "${HDR},+50p" f.go', invented) == ""
    assert _misdiagnosed_sed(broken, _SILENT) == ""


def test_an_emptied_file_view_counts_as_silence():
    from albedo_eval_service.shared.observation_format import silent_observation

    header = "Here's the result of running `cat -n` on /w/src/date.ts:"
    emptied = f"{header}\n     1\t\n[The command completed with exit code 0.]"
    assert silent_observation(emptied)
    assert silent_observation(f"{header}\n     1\t\n     2\t")

    # real content, including a file whose lines are numbers, must stay visible
    assert not silent_observation(f"{header}\n     1\timport os\n     2\tx = 1")
    assert not silent_observation(f"{header}\n     1\t42\n     2\t43")
    assert not silent_observation(f"{header}\n     1\t}}")


def test_an_observation_that_is_a_transcript_turn_is_re_asked():
    from albedo_eval_service.shared.observation_format import leaked_turn

    invented = (
        "THOUGHT: [VERIFYSYM] Let me confirm the symbol I changed.\n\n"
        "```bash\ngrep -n 'check_symlinks' a.py\n```\n\n"
        "### user\n9:def check_symlinks(filename):\n\n"
        "### assistant\nTHOUGHT: [SUBMIT] The change is correct. Submitting.\n"
    )
    assert leaked_turn(invented)
    assert not leaked_turn(
        "9:def check_symlinks(filename):\n[The command completed with exit code 0.]"
    )
    assert not leaked_turn("")

    # re-asked, and a clean second answer is used (this sample runs the returncode format)
    real = "<returncode>0</returncode>\n<output>\n9:def check_symlinks(filename):\n</output>"
    client = _Client(invented, real)
    assert _simulate(client, "grep -n 'check_symlinks' a.py") == real
    assert len(client.calls) == 2


def test_a_leak_after_real_output_keeps_the_real_output():
    from albedo_eval_service.shared.observation_format import (
        classify,
        observation_body,
        repair_output,
        silent_observation,
    )

    leaked = (
        "Compiling serde_json v1.0.0 (/testbed)\nFinished test target(s)\n"
        "### assistant\nTHOUGHT: [SUBMIT] done\n```bash\necho X\n```\n"
    )
    fixed = repair_output(leaked, classify(leaked))
    assert "### assistant" not in fixed
    assert "Compiling serde_json" in observation_body(fixed, classify(fixed))
    assert not silent_observation(fixed), "must not be turned into silence"


def test_all_attempts_leaking_falls_back_to_empty_rather_than_infra():
    invented = "### assistant\nTHOUGHT: whatever\n"
    client = _Client(invented, invented, invented)
    from albedo_eval_service.shared.observation_format import empty_output

    out = _simulate(client, "grep -n x a.py")
    assert out in (empty_output("returncode"), empty_output("openhands"))


def test_a_write_that_prints_nothing_does_not_count_as_the_shell_going_quiet():
    from albedo_eval_service.shared.observation_format import prints_nothing_on_success

    for quiet in (
        "sed -i 's/a/b/' f.py",
        "cd /testbed && sed -i 's/a/b/' f.py",
        "rm -f tmp.py",
        "mkdir -p a/b",
        "git add -A",
    ):
        assert prints_nothing_on_success(quiet), quiet
    for speaks in ("sed -n '1,5p' f.py", "grep -rn x src/", "git diff", "cd /t && cat f.py", ""):
        assert not prints_nothing_on_success(speaks), speaks

    state = _state()
    state.submit_marker = "NEVER_MATCHES"

    async def _silent(**_kwargs):
        return _SILENT

    import sanity_service.dispatcher as mod

    original = mod._simulate_observation
    mod._simulate_observation = _silent
    try:
        for turn in range(mod.MAX_CONSECUTIVE_SILENT_OBSERVATIONS + 4):
            state.turns.append(
                {
                    "role": "assistant",
                    "content": "```bash\nsed -i 's/a/b/' faker/ssn/lv_LV.py\n```",
                    "score_target": True,
                }
            )
            asyncio.run(mod._append_observations([state], "run", turn))
    finally:
        mod._simulate_observation = original

    assert state.consecutive_silent_observations == 0
    assert not state.error, "a model looping on sed -i must stay a miner fault, not become infra"


def test_a_range_read_is_renumbered_to_the_range_it_asked_for():
    from albedo_eval_service.shared.observation_format import renumbered_view

    misnumbered = (
        "Here's the result of running `cat -n` on /w/klog.go:\n"
        "     1\tcase fmt.Stringer:\n"
        "     2\t\tif v == nil {\n"
        "    11\tcase []byte:\n"
        "[The command completed with exit code 0.]"
    )
    fixed = renumbered_view("sed -n '850,860p' /w/klog.go", misnumbered)
    assert "   850\tcase fmt.Stringer:" in fixed
    assert "   851\t\tif v == nil {" in fixed
    assert "   852\tcase []byte:" in fixed, "a contiguous read cannot skip lines"
    assert "[The command completed with exit code 0.]" in fixed


def test_a_cat_n_view_is_renumbered_into_one_unbroken_run():
    from albedo_eval_service.shared.observation_format import renumbered_view

    view = (
        "Here's the result of running `cat -n` on Reference.php:\n"
        "     1\t<?php\n"
        "     2\t\n"
        "     4\tdeclare(strict_types=1);\n"
        "     4\tnamespace UcanLab;\n"
        "     3\tuse InvalidArgumentException;"
    )
    fixed = renumbered_view("cat -n Reference.php", view)
    assert [f"     {n}\t" in fixed for n in (1, 2, 3, 4, 5)] == [True] * 5
    assert "     4\tnamespace" not in fixed or "     4\tdeclare" not in fixed


def test_a_range_piped_into_cat_n_numbers_from_one():
    from albedo_eval_service.shared.observation_format import renumbered_view

    view = "     1\tdef connect(self):\n     2\t    pass"
    assert renumbered_view("sed -n '315,330p' aiohttp/connector.py | cat -n", view) == view


def test_sparse_and_truncated_views_keep_their_numbering():
    from albedo_eval_service.shared.observation_format import (
        OPENHANDS_TRUNCATION_NOTICE,
        renumbered_view,
    )

    sparse = "    85\tdef __init__\n   112\tdef save_as"
    assert renumbered_view("grep -n def x.py | cat -n", sparse) == sparse
    assert renumbered_view("nl -ba c.py | sed -n '1,12p;75,82p'", sparse) == sparse
    truncated = f"     1\ta\n{OPENHANDS_TRUNCATION_NOTICE}\n   900\tz"
    assert renumbered_view("cat -n big.py", truncated) == truncated


def test_a_phantom_modification_is_recognized_and_stripped():
    from albedo_eval_service.shared.observation_format import (
        claims_tracked_change,
        without_tracked_changes,
    )

    status = (
        "<returncode>0</returncode>\n<output>\n"
        "On branch main\n"
        "Changes not staged for commit:\n"
        '  (use "git add <file>..." to update what will be committed)\n'
        "\tmodified:   src/pydicom/dataset.py\n"
        "\nUntracked files:\n"
        "\texplore.txt\n\ttest_dataset.py\n</output>"
    )
    assert claims_tracked_change("git status", status)
    stripped = without_tracked_changes(status, "returncode")
    assert "modified:" not in stripped
    assert "explore.txt" in stripped, "untracked files are real - the model created them"
    assert "On branch main" in stripped

    diff = (
        "<returncode>0</returncode>\n<output>\n"
        "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-a\n+b\n</output>"
    )
    assert claims_tracked_change("git diff", diff)
    assert "diff --git" not in without_tracked_changes(diff, "returncode")


def test_legitimate_git_output_is_not_a_tracked_change_claim():
    from albedo_eval_service.shared.observation_format import claims_tracked_change

    hunk = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-a\n+b"
    for command in ("git log -p -1", "git show HEAD", "git diff --cached", "git diff HEAD~1"):
        assert not claims_tracked_change(command, hunk), command
    assert not claims_tracked_change("cat config.yaml", "name: x\nmodified: 2024-01-01")
    assert claims_tracked_change(
        "git log --oneline -10",
        "On branch 1.0\nchanges not staged for commit:\n\tmodified:   Reference.php",
    )


def test_a_correctly_numbered_view_is_left_byte_identical():
    from albedo_eval_service.shared.observation_format import renumbered_view

    right = (
        "Here's the result of running `cat -n` on /w/k.go:\n   850\tcase fmt.Stringer:\n   851\tx"
    )
    assert renumbered_view("sed -n '850,860p' /w/k.go", right) == right


def test_renumbering_only_touches_a_numbered_range_read():
    from albedo_eval_service.shared.observation_format import renumbered_view

    view = "Here's the result of running `cat -n` on f.py:\n     1\timport os"
    # not a range read
    assert renumbered_view("cat -n f.py", view) == view
    assert renumbered_view("grep -n os f.py", view) == view
    assert renumbered_view("", view) == view
    # a range read whose answer carries no line numbers
    assert renumbered_view("sed -n '5,9p' f.py", "plain\noutput") == "plain\noutput"


def test_a_command_echoed_back_as_its_own_output_is_recognized():
    from albedo_eval_service.shared.observation_format import classify, observation_body

    cmd = 'grep -rn --include="*.py" "parse" .'
    obs = (
        'grep -rn --include="*.py" "parse" .\n'
        "[The command completed with exit code 0.]\n[Command finished with exit code 0]"
    )
    assert observation_body(obs, classify(obs)).strip() == cmd.strip()
    # a real result does not trip it
    real = "config/__init__.py:12:def parse(args):\n[The command completed with exit code 0.]"
    assert observation_body(real, classify(real)).strip() != cmd.strip()


def test_short_format_status_phantoms_are_claims_too():
    from albedo_eval_service.shared.observation_format import (
        claims_tracked_change,
        classify,
        without_tracked_changes,
    )

    obs = (
        " M djangocms_installer/install/__init__.py\n"
        "?? patch.txt\n"
        "a9286858 Fix requirements installation with pip 10+ and restore test mock\n"
        "[The command completed with exit code 0.]\n[Command finished with exit code 0]"
    )
    assert claims_tracked_change("git status --short && git log --oneline -n 3", obs)
    stripped = without_tracked_changes(obs, classify(obs))
    assert " M djangocms" not in stripped
    assert "a9286858" in stripped, "the git log half of the output is legitimate"
    assert not claims_tracked_change("git log --oneline", "Merge pull request #142 from x")


def test_a_deletion_diff_of_a_never_removed_file_is_grounded_out():
    """e7df3710 s2 t11-t26: nine `git diff`s showed install/__init__.py deleted outright
    (@@ -1,40 +0,0 @@) — a file no command ever touched — with the micro-task's requested
    docstring inside the deleted lines. The model looped interrogating git."""
    from types import SimpleNamespace

    from albedo_eval_service.shared.observation_format import deleted_files
    from sanity_service.dispatcher import _named_in_removal

    diff = (
        "diff --git a/djangocms_installer/install/__init__.py "
        "b/djangocms_installer/install/__init__.py\n"
        "index 6e3e3f8..e69de29 100644\n"
        "--- a/djangocms_installer/install/__init__.py\n"
        "+++ b/djangocms_installer/install/__init__.py\n"
        "@@ -1,40 +0,0 @@\n-# -*- coding: utf-8 -*-\n-import os\n"
    )
    assert deleted_files("git diff", diff) == ["djangocms_installer/install/__init__.py"]
    assert deleted_files("git diff --cached", diff) == []
    assert deleted_files("git show HEAD", diff) == []
    # an ordinary modification hunk is not a deletion claim
    assert deleted_files("git diff", diff.replace("+0,0", "+1,42")) == []

    edit = "```bash\nsed -i 's/a/b/' djangocms_installer/config/__init__.py\n```"
    state = SimpleNamespace(turns=[{"role": "assistant", "content": edit}])
    assert not _named_in_removal(state, "djangocms_installer/install/__init__.py"), (
        "editing config/__init__.py must not excuse deleting install/__init__.py "
        "- basenames collide on __init__.py"
    )
    removal = "```bash\nrm djangocms_installer/install/__init__.py\n```"
    state.turns.append({"role": "assistant", "content": removal})
    assert _named_in_removal(state, "djangocms_installer/install/__init__.py")


def test_stuttered_observations_are_recognized_for_retry():
    from albedo_eval_service.shared.observation_format import stuttered_lines

    assert "3x consecutively" in stuttered_lines(
        "from pandas.core.indexes.period import Period\n" * 4
    )
    assert "A A B B" in stuttered_lines(
        "     1\tdef complete(self) -> str:\n"
        "     2\tdef complete(self) -> str:\n"
        '     3\t"""Return the completion script.\n'
        '     4\t"""Return the completion script.'
    )
    assert stuttered_lines("./tests/install.py:13:        mocked.reset_mock()\n" * 3)


def test_legitimately_repetitive_code_is_not_stuttered():
    from albedo_eval_service.shared.observation_format import stuttered_lines

    assert not stuttered_lines(
        "def union(self):\n    other : Index or array-like\n    sort : bool or None\n\n"
        "def difference(self):\n    other : Index or array-like\n    sort : bool or None"
    )
    assert not stuttered_lines(
        "class A:\n    def __init__(self, options=None):\n        self.a = 1\n\n"
        "class B:\n    def __init__(self, options=None):\n        self.b = 2"
    )
    # a single doubled pair (A A) with nothing else doubled is below the bar
    assert not stuttered_lines("use swc_common::{BytePos, Span};\n" * 2)
    assert not stuttered_lines("else:\n" * 5), "short closers are ignored"


def test_the_simulator_is_told_not_to_repeat_lines():
    from albedo_eval_service.simulator.prompt_simulator import simulation_system_prompt

    prompt = simulation_system_prompt("openhands")
    assert "Never render the same line or block twice" in prompt


def test_a_repeated_read_answers_identically_until_state_changes():
    state = _state()
    client = _Client(_REAL, _SILENT)  # a second call WOULD flip the answer

    async def once(command):
        return await D._simulate_observation(
            client=client,
            settings=JudgeSettings(engy_api_key=""),
            eval_run_id="run",
            state=state,
            assistant_output=f"THOUGHT: read\n\n```bash\n{command}\n```",
        )

    first = asyncio.run(once(_READ))
    again = asyncio.run(once(_READ))
    assert first == again == _REAL, "the repeat must be served from the memo, not re-simulated"
    assert len(client.calls) == 1

    # an edit invalidates: the same read now re-simulates and may answer differently
    state.turns.append(
        {"role": "assistant", "content": "```bash\nsed -i 's/a/b/' src/style.rs\n```"}
    )
    calls_before = len(client.calls)
    asyncio.run(once(_READ))
    assert len(client.calls) > calls_before, "an edit must invalidate the memo"


def test_script_execution_also_invalidates_the_read_memo():
    state = _state()
    client = _Client(_REAL)

    async def once():
        return await D._simulate_observation(
            client=client,
            settings=JudgeSettings(engy_api_key=""),
            eval_run_id="run",
            state=state,
            assistant_output=f"THOUGHT: read\n\n```bash\n{_READ}\n```",
        )

    asyncio.run(once())
    state.turns.append(
        {"role": "assistant", "content": "```bash\ngo test ./middleware/... -run Timeout\n```"}
    )
    calls_before = len(client.calls)
    asyncio.run(once())
    assert len(client.calls) > calls_before


def test_an_echoed_must_print_read_falls_through_to_the_retry():
    echoed = f"<returncode>0</returncode>\n<output>\n{_READ}\n</output>"
    client = _Client(echoed, _REAL)
    assert _simulate(client, _READ) == _REAL
    assert len(client.calls) == 2
    assert MUST_PRINT_RETRY in client.calls[1]["messages"][-1]["content"]


def test_an_echoed_maybe_silent_command_still_becomes_empty():
    cmd = "grep -rn 'ProgressStyle' src/"
    echoed = f"<returncode>0</returncode>\n<output>\n{cmd}\n</output>"
    client = _Client(echoed)
    assert _simulate(client, cmd) == _SILENT
    assert len(client.calls) == 1


def test_cp_and_mv_count_as_edits():
    from sanity_service.chain import _EDIT_RE

    assert _EDIT_RE.search("cp /tmp/pkg/git-machete-3.7.2/git_machete/client.py /workspace/x.py")
    assert _EDIT_RE.search("mv new_impl.py git_machete/client.py")
    assert not _EDIT_RE.search("tcp handshake and recv buffers")
    assert not _EDIT_RE.search("grep -n 'cpu' file.py")


def test_think_salvage_is_unified_with_the_bench():
    from albedo_eval_service.shared.observation_format import UNCLOSED_THINK_BLOCK_SENTINEL
    from sanity_remote.worker import _strip_thinking

    salvaged = _strip_thinking("<think>plan\n```bash\nsed -i 's/a/b/' f.py\n```")
    assert "sed -i" in salvaged and UNCLOSED_THINK_BLOCK_SENTINEL not in salvaged

    assert UNCLOSED_THINK_BLOCK_SENTINEL in _strip_thinking("<think>musing with no command")

    closed = _strip_thinking("<think>hidden</think>THOUGHT: go\n```bash\nls\n```")
    assert "ls" in closed and "hidden" not in closed

    reopened = _strip_thinking("<think>a</think>ok\n<think>more\n```bash\ngit diff\n```")
    assert "git diff" in reopened and UNCLOSED_THINK_BLOCK_SENTINEL not in reopened, (
        "a reopened think carrying an action used to be fatal; the bench runs it"
    )


def test_eval_accept_now_rejects_what_preeval_rejects():
    from albedo_eval_service.judge_api import _usable_simulation_output

    ok = "<returncode>0</returncode>\n<output>\n1\tdef f():\n2\t    pass\n</output>"
    assert _usable_simulation_output(ok, "returncode", command="cat -n f.py")

    stuttered = (
        "<returncode>0</returncode>\n<output>\n"
        + "from pandas.core.indexes.period import Period\n" * 4
        + "</output>"
    )
    assert not _usable_simulation_output(stuttered, "returncode", command="cat -n f.py")

    fabricated = (
        "<returncode>1</returncode>\n<output>\n"
        "sed: -e expression #1, char 1: unterminated address regex\n</output>"
    )
    assert not _usable_simulation_output(
        fabricated, "returncode", command="sed -i 's/a/b/' f.py"
    ), "real sed accepts this script, so the error is invented"
    assert _usable_simulation_output(fabricated, "returncode", command="git diff"), (
        "a sed error under a non-sed command is not this check's business"
    )
