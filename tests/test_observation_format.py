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
    command_stages,
    contract_violation,
    detect_format,
    empty_output,
    first_bash_block,
    grounded_observation,
    is_scaffold_truncated,
    is_truncated,
    no_output_notice,
    observation_body,
    prints_nothing_on_success,
    repair_output,
    repair_to_contract,
    truncation_notice,
    valid_output,
    with_body,
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
        "No matching distribution found for boto3"
    )
    # a redirection, a flag's value and a local path are not package names
    for command, named in (
        ("pip install pytest pytest-mock virtualenv 2>&1 | tail -5", "pytest"),
        ("pip install --index-url https://pypi.org/simple/ pytest", "pytest"),
        ('pip install "requests>=2.0"', "requests>=2.0"),
        ("pip install -e . 2>&1 | tail -10", "the requested packages"),
    ):
        assert absent_tool_output(command)[0].endswith(
            f"No matching distribution found for {named}"
        ), command
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


def test_degenerate_observation_catches_a_collapsed_file():
    from albedo_eval_service.shared.observation_format import degenerate_observation

    collapsed = (
        "<returncode>0</returncode>\n<output>\n"
        + "\n".join(f"{i:>6}\tAQRSymbol," for i in range(1, 160))
        + "\n</output>"
    )
    assert degenerate_observation(collapsed)

    # the real shapes seen in healthy genesis runs, all simulator collapse
    for line in ("visible?: boolean;", "import { Property } from 'csstype';", "`"):
        assert degenerate_observation("\n".join([line] * 200)), line


def test_degenerate_observation_leaves_real_output_alone():
    from albedo_eval_service.shared.observation_format import degenerate_observation

    real_file = "\n".join(
        [
            "from qrcode.util import (",
            "    QRData,",
            "    QRCode,",
            ")",
            "",
            "class BaseImage:",
            "    def __init__(self, border, width, box_size):",
            "        self.border = border",
            "        self.width = width",
            "        self.box_size = box_size",
            "        self.modules = None",
            "    def drawrect(self, row, col):",
            "        raise NotImplementedError",
        ]
    )
    assert not degenerate_observation(real_file)

    # short outputs are never judged: too little signal
    assert not degenerate_observation("\n".join(["}"] * 6))
    assert not degenerate_observation("")
    # a listing with repeated short tokens but real variety
    assert not degenerate_observation(
        "\n".join(
            ["}", "x = 1", "}", "y = 2", "}", "z = 3", "}", "w = 4", "}", "v = 5", "}", "u = 6"]
        )
    )


OPENHANDS_SESSION = [
    {"role": "assistant", "content": "```bash\nls\n```"},
    {
        "role": "user",
        "content": (
            "README.md\n[The command completed with exit code 0.]\n"
            "[Current working directory: /workspace/o__r__1.0]\n"
            "[Python interpreter: /usr/local/bin/python]\n"
            "[Command finished with exit code 0]"
        ),
    },
]


def test_a_known_output_answers_swe_agent_exactly_as_the_simulator_would():
    # every route through the simulator ends in with_body once an exact output exists, and for
    # this format that leaves nothing of the model's answer behind
    for body in ("src/app.py\nsrc/util.py", ""):
        assert grounded_observation(SWE_AGENT, body, None, "grep -rn x src/", None) == with_body(
            wrap("invented by the model", SWE_AGENT), SWE_AGENT, body
        )


def test_an_empty_computed_search_waits_for_an_exit_status_in_returncode_format():
    # a non-empty output was already answered as a success before exit codes were derived
    assert grounded_observation(RETURNCODE, "src/app.py", None, "grep -rn x src/", None) == wrap(
        "src/app.py", RETURNCODE
    )
    # an empty one says nothing without a status: 0 and 1 mean opposite things to the assistant
    assert grounded_observation(RETURNCODE, "", None, "grep -rn x src/", None) is None
    assert grounded_observation(RETURNCODE, "", 1, "grep -rn x src/", None) == wrap(
        "", RETURNCODE, returncode=1
    )


def test_openhands_reuses_the_trailer_block_its_own_session_prints():
    observation = grounded_observation(
        OPENHANDS, "README.md", 0, "grep -rn x src/", OPENHANDS_SESSION
    )
    assert observation == (
        "README.md\n[The command completed with exit code 0.]\n"
        "[Current working directory: /workspace/o__r__1.0]\n"
        "[Python interpreter: /usr/local/bin/python]\n"
        "[Command finished with exit code 0]"
    )
    # without a precedent to copy, the wrapper would have to be invented
    assert grounded_observation(OPENHANDS, "README.md", 0, "grep -rn x src/", None) is None


def test_openhands_declines_when_the_command_moves_the_working_directory():
    # a cd into the directory the session already reports leaves the trailer correct
    stays = grounded_observation(
        OPENHANDS,
        "README.md",
        0,
        "cd /workspace/o__r__1.0 && grep -rn x src/",
        OPENHANDS_SESSION,
    )
    assert stays is not None and "[Current working directory: /workspace/o__r__1.0]" in stays
    # any real move lands somewhere this cannot spell
    for command in ("cd /tmp/scratch && grep -rn x src/", "cd sub && grep -rn x src/"):
        assert grounded_observation(OPENHANDS, "README.md", 0, command, OPENHANDS_SESSION) is None


def test_a_numbered_read_renders_as_an_openhands_view_with_no_trailer():
    observation = grounded_observation(
        OPENHANDS, "     1\tprint(1)", 0, "cat -n /workspace/o__r__1.0/app.py", OPENHANDS_SESSION
    )
    assert observation == (
        "Here's the result of running `cat -n` on /workspace/o__r__1.0/app.py:\n     1\tprint(1)"
    )
    assert "exit code" not in observation
    # a failed or empty read is not a view, and the scaffold's wording for it is not derivable
    assert (
        grounded_observation(
            OPENHANDS, "", 1, "cat -n /workspace/o__r__1.0/gone.py", OPENHANDS_SESSION
        )
        is None
    )


def test_a_comment_above_the_command_does_not_hide_it():
    fence = "```bash\n"
    # the parser that grounds a turn reads the leading token, so a note above the command used
    # to make a plain range read look like something unparseable
    assert (
        first_bash_block(f"{fence}# let me check the line numbers\nsed -n '510,530p' p.go\n```")
        == "sed -n '510,530p' p.go"
    )
    assert first_bash_block(f"{fence}# one\n# two\ngrep -n x a.py\n```") == "grep -n x a.py"
    # a `#` that is not a leading line belongs to the command
    assert (
        first_bash_block(f'{fence}echo "# not a comment" > f\n```') == 'echo "# not a comment" > f'
    )
    heredoc = "cat > f << 'EOF'\n# inside the body\nx = 1\nEOF"
    assert first_bash_block(f"{fence}{heredoc}\n```") == heredoc
    # nothing but comments has no command underneath to uncover
    assert first_bash_block(f"{fence}# only a note\n```") == "# only a note"


def test_a_command_after_a_heredoc_terminator_is_its_own_stage():
    write_then_run = "cat <<'EOF' > /tmp/t.py\nprint(1)\nEOF\npython /tmp/t.py"
    assert command_stages(write_then_run) == [
        "cat <<'EOF' > /tmp/t.py\nprint(1)\nEOF",
        "python /tmp/t.py",
    ]
    # the run prints, so the pair is not silent on success even though the write alone would be
    assert not prints_nothing_on_success(write_then_run)
    assert prints_nothing_on_success("cat <<'EOF' > /tmp/t.py\nprint(1)\nEOF")


def test_a_heredoc_alone_is_not_a_silent_write():
    # the interpreter consumes the script and prints; only a redirect onto a file is quiet
    assert not prints_nothing_on_success("python <<'EOF'\nprint(1)\nEOF")
    assert not prints_nothing_on_success("python - <<'EOF'\nprint(1)\nEOF")
    # tee copies its input to stdout as well as to the file
    assert not prints_nothing_on_success("tee f.py <<'EOF'\nx\nEOF")
    assert prints_nothing_on_success("cat <<'EOF' > f.py\nx\nEOF")
