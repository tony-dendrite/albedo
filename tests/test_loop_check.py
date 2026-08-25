from __future__ import annotations

from albedo_eval_service.shared.loop_check import (
    DUP_CMD_THRESHOLD,
    MAX_RUN_THRESHOLD,
    candidate_turns,
    commands_of,
    loop_explanation,
    loop_stats,
    loop_verdict,
    loop_verdict_for_document,
)


def _turn(command: str) -> str:
    return f"I will run this.\n\n```bash\n{command}\n```"


def _document(commands: list[str], context_command: str = "") -> str:
    parts = ["FULL CANDIDATE TRAJECTORY", "Score ONLY CANDIDATE OUTPUT 1."]
    if context_command:
        body = f"<pr_description>\n```bash\n{context_command}\n```\n</pr_description>"
        parts.append(f"\nCONTEXT USER (do not score):\n------\n{body}\n------")
    for index, command in enumerate(commands, start=1):
        parts.append(f"\nCANDIDATE OUTPUT {index}:\n------\n{_turn(command)}\n------")
        parts.append("\nENVIRONMENT OBSERVATION (context only, do not score):\n------\nok\n------")
    return "\n".join(parts).strip()


def test_candidate_turns_extracts_only_scored_blocks():
    document = _document(["ls -la", "cat a.py"], context_command="rm -rf /")
    turns = candidate_turns(document)
    assert len(turns) == 2
    assert commands_of(turns) == ["ls -la", "cat a.py"]


def test_context_bash_fences_never_count_as_candidate_commands():
    document = _document(["ls -la"], context_command="pytest -q")
    assert "pytest -q" not in commands_of(candidate_turns(document))


def test_candidate_turns_falls_back_to_whole_document():
    assert candidate_turns("no candidate blocks here") == ["no candidate blocks here"]
    assert candidate_turns("") == []


def test_commands_are_whitespace_normalised():
    turns = ["```bash\ngit    status\n\n```", "```bash\ngit status\n```"]
    assert commands_of(turns) == ["git status", "git status"]


def test_clean_trajectory_is_not_looped():
    document = _document(["ls", "cat a.py", "pytest -q", "git diff", "echo done"])
    verdict = loop_verdict_for_document(document)
    assert verdict.looped is False
    assert verdict.reasons == ()
    assert verdict.commands == ()


def test_consecutive_run_at_threshold_is_looped():
    turns = [_turn("git status")] * MAX_RUN_THRESHOLD
    verdict = loop_verdict(turns)
    assert verdict.looped is True
    assert verdict.max_cmd_run == MAX_RUN_THRESHOLD
    assert any("consecutively" in reason for reason in verdict.reasons)


def test_one_below_consecutive_threshold_is_not_looped_by_run():
    turns = [_turn("git status")] * (MAX_RUN_THRESHOLD - 1)
    verdict = loop_verdict(turns)
    assert verdict.max_cmd_run == MAX_RUN_THRESHOLD - 1
    assert not any("consecutively" in reason for reason in verdict.reasons)


def test_duplicate_ratio_threshold_is_looped_without_consecutive_run():
    turns = [_turn(c) for c in ["a", "b", "a", "b", "a", "b"]]
    verdict = loop_verdict(turns)
    assert verdict.max_cmd_run == 1
    assert verdict.dup_cmd_ratio >= DUP_CMD_THRESHOLD
    assert verdict.looped is True
    assert any("duplicate command ratio" in reason for reason in verdict.reasons)


def test_only_looping_commands_are_reported():
    turns = [_turn(c) for c in ["dup", "dup", "dup", "dup", "once", "twice", "twice"]]
    verdict = loop_verdict(turns)
    reported = {entry.command for entry in verdict.commands}
    assert "once" not in reported
    assert reported == {"dup", "twice"}


def test_reported_counts_and_runs_are_accurate():
    turns = [_turn(c) for c in ["x", "x", "x", "x", "y", "x"]]
    verdict = loop_verdict(turns)
    entry = next(e for e in verdict.commands if e.command == "x")
    assert entry.count == 5
    assert entry.longest_run == 4


def test_looping_commands_sorted_by_longest_run_first():
    turns = [_turn(c) for c in ["short", "short", "long", "long", "long", "long", "long"]]
    verdict = loop_verdict(turns)
    assert verdict.commands[0].command == "long"


def test_loop_stats_handles_empty_input():
    stats = loop_stats([])
    assert stats == {"n_cmds": 0, "dup_cmd_ratio": 0.0, "max_cmd_run": 0}
    assert loop_verdict([]).looped is False


def test_turns_without_commands_are_not_looped():
    verdict = loop_verdict(["just prose", "more prose"])
    assert verdict.n_cmds == 0
    assert verdict.looped is False


def test_explanation_names_the_loop_the_command_and_the_count():
    turns = [_turn("pytest -q")] * 6
    verdict = loop_verdict(turns)
    text = loop_explanation(verdict)
    assert "looped" in text
    assert "`pytest -q`" in text
    assert "6x" in text
    assert "6 consecutive" in text


def test_explanation_caps_the_listed_commands():
    """Seven repeated commands, listed five at a time. The run rule is what makes this looped —
    seven-pairs alone is a 0.5 ratio, which DUP_CMD_THRESHOLD deliberately no longer catches."""
    commands = ["a"] * MAX_RUN_THRESHOLD
    for name in ["b", "c", "d", "e", "f", "g"]:
        commands += [name, name]
    verdict = loop_verdict([_turn(c) for c in commands])
    assert verdict.looped and verdict.dup_cmd_ratio < DUP_CMD_THRESHOLD
    text = loop_explanation(verdict)
    assert len(verdict.commands) == 7
    assert "and 2 more" in text


def test_explanation_truncates_a_very_long_command():
    long_command = "echo " + "x" * 400
    verdict = loop_verdict([_turn(long_command)] * 5)
    text = loop_explanation(verdict)
    assert "…" in text
    assert len(text) < 400


def test_loop_check_sees_both_harness_syntaxes():
    from albedo_eval_service.shared.loop_check import loop_stats

    fenced = ["THOUGHT: x\n\n```bash\nls -la\n```"] * 10
    tagged = ["THOUGHT: x\n\n<mswea_bash_command>ls -la</mswea_bash_command>"] * 10

    assert loop_stats(fenced) == loop_stats(tagged)
    assert loop_stats(tagged)["n_cmds"] == 10
    assert loop_stats(tagged)["max_cmd_run"] == 10

    mixed = fenced[:5] + tagged[:5]
    assert loop_stats(mixed)["max_cmd_run"] == 10
