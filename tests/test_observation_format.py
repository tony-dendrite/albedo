from __future__ import annotations

from albedo_eval_service.shared.observation_format import (
    OPENHANDS,
    OPENHANDS_TRUNCATION_NOTICE,
    PIP_PYTEST_ABSENT,
    PYTEST_MISSING,
    RETURNCODE,
    SWE_AGENT,
    TRUNCATION_SENTINEL,
    absent_tool_output,
    classify,
    command_contract,
    contract_violation,
    detect_format,
    empty_output,
    is_scaffold_truncated,
    is_truncated,
    no_output_notice,
    observation_body,
    repair_output,
    repair_to_contract,
    truncation_notice,
    valid_output,
    wrap,
)
from albedo_eval_service.simulator.prompt_simulator import (
    FORMAT_MINI_CODER,
    FORMAT_OPENHANDS,
    FORMAT_SWE_AGENT,
    format_block,
)

RC_OBS = "<returncode>0</returncode>\n<output>\ntotal 228\ndrwxr-xr-x 12 root root\n</output>"
SWE_AGENT_OBS = "OBSERVATION:\nHere's the files and directories up to 2 levels deep in /testbed:"
OPENHANDS_BASH_OBS = (
    "\n[The command completed with exit code 0.]\n"
    "[Current working directory: /workspace/pandas-dev__pandas__1.0]\n"
    "[Command finished with exit code 0]"
)
OPENHANDS_EDITOR_OBS = "File created successfully at: /workspace/attrs__1.0/reproduce.py"


def _prefix(observation: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "the task description"},
        {"role": "assistant", "content": "```bash\nls\n```"},
        {"role": "user", "content": observation},
    ]


def test_classify_reads_the_marker_each_corpus_uses():
    assert classify(RC_OBS) == RETURNCODE
    assert classify(SWE_AGENT_OBS) == SWE_AGENT
    assert classify(OPENHANDS_BASH_OBS) == OPENHANDS
    assert classify(OPENHANDS_EDITOR_OBS) == OPENHANDS


def test_detect_format_reads_the_trajectorys_own_observation():
    assert detect_format("open-swe-traces/x:0:1", _prefix(RC_OBS)) == RETURNCODE
    assert detect_format("open-swe-traces/x:0:1", _prefix(SWE_AGENT_OBS)) == SWE_AGENT
    assert detect_format("open-swe-traces/x:0:1", _prefix(OPENHANDS_BASH_OBS)) == OPENHANDS
    assert detect_format("open-swe-traces/x:0:1", _prefix(SWE_AGENT_OBS)) != detect_format(
        "open-swe-traces/x:0:1", _prefix(OPENHANDS_BASH_OBS)
    )


def test_detect_format_ignores_the_leading_task_message():
    task_only = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Fix the bug in foo.py"},
    ]
    assert detect_format("mini-coder/x:0:1", task_only) == RETURNCODE
    assert detect_format("swe-hero/x:0:1", task_only) == OPENHANDS
    assert detect_format("mini-coder/x:0:1", None) == RETURNCODE


def test_valid_output_accepts_the_native_dialect_and_rejects_the_others():
    assert valid_output(RC_OBS, RETURNCODE) is True
    assert valid_output(SWE_AGENT_OBS, SWE_AGENT) is True
    assert valid_output(OPENHANDS_BASH_OBS, OPENHANDS) is True
    assert valid_output(OPENHANDS_EDITOR_OBS, OPENHANDS) is True

    assert valid_output(SWE_AGENT_OBS, RETURNCODE) is False
    assert valid_output(OPENHANDS_EDITOR_OBS, SWE_AGENT) is False
    assert valid_output(RC_OBS, OPENHANDS) is False
    assert valid_output("Observation: retired dialect", OPENHANDS) is False
    for fmt in (RETURNCODE, SWE_AGENT, OPENHANDS):
        assert valid_output("", fmt) is False


def test_injected_observations_are_valid_in_their_own_format():
    for fmt in (RETURNCODE, SWE_AGENT, OPENHANDS):
        assert valid_output(empty_output(fmt), fmt), fmt
        assert valid_output(wrap("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", fmt), fmt), fmt
        assert valid_output(wrap("no bash command", fmt, returncode=2), fmt), fmt
    assert empty_output(RETURNCODE) == "<returncode>0</returncode>\n<output>\n</output>"
    assert wrap("done", SWE_AGENT) == "OBSERVATION:\ndone"
    assert wrap("done", RETURNCODE, returncode=2) == (
        "<returncode>2</returncode>\n<output>\ndone\n</output>"
    )


def test_repair_output_only_touches_the_returncode_wrapper():
    squashed = "<returncode>0</returncode>\n<output>ok</output>"
    assert valid_output(squashed, RETURNCODE) is False
    assert valid_output(repair_output(squashed, RETURNCODE), RETURNCODE) is True
    assert repair_output(OPENHANDS_EDITOR_OBS, OPENHANDS) == OPENHANDS_EDITOR_OBS


def test_format_block_matches_the_format():
    assert format_block(RETURNCODE) == FORMAT_MINI_CODER
    assert format_block(SWE_AGENT) == FORMAT_SWE_AGENT
    assert format_block(OPENHANDS) == FORMAT_OPENHANDS


def test_is_scaffold_truncated_knows_every_scaffolds_marker():
    assert is_scaffold_truncated(f"head\n{OPENHANDS_TRUNCATION_NOTICE}\ntail")
    assert is_scaffold_truncated("<warning>\ntoo long\n</warning><output_head>\n")
    assert is_scaffold_truncated("The output of your last command was too long.")
    assert is_scaffold_truncated("first lines\n<response clipped>")
    assert not is_scaffold_truncated(RC_OBS)
    assert not is_scaffold_truncated("")


def _clipped_returncode(body: str) -> str:
    head, tail = body[:5000], body[-5000:]
    return wrap(
        "<warning>\nThe output of your last command was too long.\n</warning><output_head>\n"
        f"{head}\n</output_head>\n<elided_chars>\n{len(body) - 10000} characters elided\n"
        f"</elided_chars>\n<output_tail>\n{tail}\n</output_tail>",
        RETURNCODE,
    )


def test_a_scaffold_clipped_observation_is_exempt_from_its_commands_contract():
    body = "\n".join("    x" * 12 for _ in range(200))
    contract = command_contract("sed -n '1,200p' mod.py")
    clipped = _clipped_returncode(body)

    assert contract.max_lines == 200
    assert is_scaffold_truncated(clipped)
    assert contract_violation(clipped, RETURNCODE, contract) is None
    assert repair_to_contract(clipped, RETURNCODE, contract) == clipped

    plain = wrap("\n".join(f"line {i}" for i in range(40)), RETURNCODE)
    short = command_contract("head -n 20 mod.py")
    assert contract_violation(plain, RETURNCODE, short) == "too_many_lines:40>20"


def test_truncation_notice_is_detectable_and_names_the_limit():
    notice = truncation_notice(16384)
    assert TRUNCATION_SENTINEL in notice
    assert "16384" in notice
    assert is_truncated(notice)
    assert is_truncated(f"CANDIDATE OUTPUT 1:\n------\n{notice}\n------")
    assert not is_truncated("an ordinary candidate answer")
    assert not is_truncated("")


def test_a_command_we_cannot_run_fails_like_a_terminal_would():
    # never a note about the session: that would tell the model it is being simulated
    for command in ("cargo test --test x", "go build ./...", "npm run lint"):
        body, returncode = no_output_notice(command)
        assert body == f"bash: {command.split()[0]}: command not found"
        assert returncode == 127
        assert "session" not in body and "captured" not in body

    # python is present in this environment, so its failure is a missing module instead
    assert no_output_notice("python -c 'import numpy; numpy.dot(a, b)'") == (
        "/opt/conda/bin/python: No module named numpy",
        1,
    )
    assert no_output_notice("cd /testbed && python reproduce_issue.py") == (
        "/opt/conda/bin/python: No module named reproduce_issue",
        1,
    )

    # the same command always fails the same way, so retrying cannot look like progress
    assert no_output_notice("cargo build") == no_output_notice("cargo build")
    for fmt in (RETURNCODE, OPENHANDS, SWE_AGENT):
        body, returncode = no_output_notice("cargo build")
        wrapped = wrap(body, fmt, returncode=returncode)
        assert valid_output(wrapped, fmt)
        assert "exit code 0" not in wrapped
        assert observation_body(wrapped, fmt) == body


def test_pytest_is_absent_from_this_environment():
    for command in (
        "pytest",
        "py.test tests/",
        "python -m pytest -k inference -q",
        "cd /workspace/x__y__1.0 && python3 -m pytest tests/ -v --tb=short",
    ):
        assert absent_tool_output(command) == (PYTEST_MISSING, 1), command
    assert absent_tool_output("pip install pytest") == (PIP_PYTEST_ABSENT, 1)
    # any package is unavailable, not just pytest, and any module is missing
    assert absent_tool_output("cd /testbed && pip install -q boto3 moto")[0].endswith(
        "No matching distribution found for boto3 moto"
    )
    assert absent_tool_output("python -m mypy src/") == (
        "/opt/conda/bin/python: No module named mypy",
        1,
    )

    assert wrap(PYTEST_MISSING, RETURNCODE, returncode=1) == (
        "<returncode>1</returncode>\n<output>\n/opt/conda/bin/python: No module named pytest\n</output>"  # noqa: E501
    )
    for fmt in (RETURNCODE, OPENHANDS, SWE_AGENT):
        wrapped = wrap(PYTEST_MISSING, fmt, returncode=1)
        assert valid_output(wrapped, fmt)
        assert observation_body(wrapped, fmt) == PYTEST_MISSING


def test_commands_that_only_name_pytest_still_run_normally():
    for command in (
        "grep -rn pytest tests/",
        "cat pytest.ini",
        'python -c "import pytest"',
        'find . -name "pytest.ini"',
        "cd /workspace/pytest-dev__pyfakefs__1.0 && grep -n dup pyfakefs/fake_os.py",
    ):
        assert absent_tool_output(command) is None, command
    # no package index in this environment, so installing anything fails the same way
    assert absent_tool_output("pip install pytest-cov")[0].endswith(
        "No matching distribution found for pytest-cov"
    )
