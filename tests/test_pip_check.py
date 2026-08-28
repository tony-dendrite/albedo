from __future__ import annotations

from albedo_eval_service.shared.pip_check import (
    PIP_ROOT_WARNING,
    fabricated_pip_error,
    pip_success_body,
)

# the exact simulated observation this check was built against (a real eval fabrication)
_SCREENSHOT_ERROR = (
    "ERROR: Could not find a version that satisfies the requirement . (from versions: none)\n"
    "ERROR: No matching distribution found for ."
)

# real bench observations these must never fire on
_BENCH_MALT = (
    "ERROR: Could not find a version that satisfies the requirement malt (from versions: none)\n"
    "ERROR: No matching distribution found for malt"
)
_BENCH_DEP_PIN = (
    "ERROR: Could not find a version that satisfies the requirement dagster-pipes==1!0+dev "
    "(from dagster) (from versions: 1.5.0, 1.5.1)"
)


def test_index_error_naming_a_path_is_fabricated():
    bench_shape = "cd /testbed && pip install -e . -q 2>&1 | tail -20"
    assert fabricated_pip_error("pip install -e .", _SCREENSHOT_ERROR)
    assert fabricated_pip_error(bench_shape, _SCREENSHOT_ERROR)
    assert fabricated_pip_error("pip install ./pkg", "No matching distribution found for ./pkg")


def test_real_bench_resolver_errors_stay():
    assert not fabricated_pip_error("pip install malt -q", _BENCH_MALT)
    # a local -e . install really can fail resolving one of its pinned dependencies
    assert not fabricated_pip_error("cd /x && pip install -e . -q", _BENCH_DEP_PIN)


def test_pip_denied_is_fabricated_the_bench_always_has_pip():
    assert fabricated_pip_error("pip install pytest -q", "bash: pip: command not found")
    assert fabricated_pip_error("pip list | grep requests", "bash: pip: command not found")
    assert fabricated_pip_error(
        "python -m pip install -e .", "/usr/bin/python: No module named pip"
    )


def test_non_pip_commands_and_clean_output_pass():
    assert not fabricated_pip_error("pytest -q", _SCREENSHOT_ERROR)
    assert not fabricated_pip_error("pip install -e . -q", PIP_ROOT_WARNING)
    assert not fabricated_pip_error("", _SCREENSHOT_ERROR)
    assert not fabricated_pip_error("pip install -e .", "")


def test_success_body_is_the_bench_root_warning_unless_stderr_is_discarded():
    assert pip_success_body("pip install -e . -q 2>&1 | tail -20") == PIP_ROOT_WARNING
    assert pip_success_body("pip install -e . -q 2>/dev/null") == ""


def test_eval_accept_gate_rejects_the_fabrication():
    from albedo_eval_service.judge_api import _usable_simulation_output
    from albedo_eval_service.shared.observation_format import RETURNCODE, wrap

    bad = wrap(_SCREENSHOT_ERROR, RETURNCODE, returncode=1)
    good = wrap(PIP_ROOT_WARNING, RETURNCODE)
    assert not _usable_simulation_output(bad, RETURNCODE, command="pip install -e .")
    assert _usable_simulation_output(good, RETURNCODE, command="pip install -e .")
