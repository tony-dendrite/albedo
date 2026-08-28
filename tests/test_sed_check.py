"""The simulator invents sed syntax errors; real sed is the arbiter.

Every command below is verbatim from a pre-eval fail trajectory. 33 of 121 simulated `sed -i`
edits in that corpus came back as shell errors, and most were fabricated — real GNU sed accepts
the script. Since `sed -i` is how these agents edit, a fabricated failure means the requested
change can never land, and the trajectory is then failed for not making it.
"""

from __future__ import annotations

import shutil

import pytest

from albedo_eval_service.shared.sed_check import (
    fabricated_sed_error,
    sed_accepts,
    sed_scripts,
)

pytestmark = pytest.mark.skipif(shutil.which("sed") is None, reason="needs the real sed binary")

# fb91b967 s2 — answered "unterminated address regex" at char 1; the expression starts with `s`
_FB91 = (
    "sed -i 's/def set(self, key, report_type, value):/def set(self, key, report_type, value):"
    "\\n        if key in self.cache:\\n            self.cache[key].value = value/' "
    "deepdiff/lfucache.py"
)
# d45cce5b s2 — answered "unterminated address regex" at char 22
_D45C = (
    "sed -i 's/        if constraint\\[1\\]:/        if constraint[1] and value is not None:/' "
    "pyasn1/type/constraint.py"
)
# 3b5a443f s0 — verbatim, and real sed really does reject it: the replacement ends with an
# unescaped `'` so the shell closes the quote early and sed never sees a closing delimiter
_GENUINE = (
    "sed -i 's/ else:\\n raise TypeError.*$/ else:\\n "
    "# Fall back to default values from the function signature\\n try:\\n "
    "sig = inspect.signature(func)\\n return None\\n "
    "except (ValueError, AttributeError):\\n raise TypeError('\\./funcy/decorators.py"
)
_SED_ERROR = (
    "<returncode>1</returncode>\n<output>\n"
    "sed: -e expression #1, char 1: unterminated address regex\n</output>"
)


def test_a_fabricated_error_is_detected_on_the_real_commands():
    for command in (_FB91, _D45C):
        assert sed_scripts(command), command
        assert all(sed_accepts(script) for script in sed_scripts(command)), command
        assert fabricated_sed_error(command, _SED_ERROR), command


def test_a_genuine_error_is_left_alone():
    """Masking every failure would hide models that really do write broken seds."""
    assert not all(sed_accepts(script) for script in sed_scripts(_GENUINE))
    assert not fabricated_sed_error(_GENUINE, _SED_ERROR)


def test_silent_unless_the_observation_actually_blames_sed():
    assert not fabricated_sed_error(_FB91, "<returncode>0</returncode>\n<output>\n</output>")
    assert not fabricated_sed_error(_FB91, "")
    assert not fabricated_sed_error("git diff", _SED_ERROR)


def test_a_script_that_would_execute_or_write_is_never_excused():
    """Miner output reaches this check, so sed's `e`, `r` and `w` commands must not run. Under
    --sandbox they fail to parse, which keeps the simulator's answer rather than overriding it."""
    for script in ("1e rm -rf /tmp/x", "w /tmp/pwn", "1r /etc/passwd"):
        assert not sed_accepts(script), script
        assert not fabricated_sed_error(f"sed -i '{script}' a.py", _SED_ERROR)


def test_scripts_are_pulled_out_of_every_shape_the_corpus_uses():
    assert sed_scripts("sed -i 's/a/b/' f.py") == ["s/a/b/"]
    assert sed_scripts("sed -i.bak 's/a/b/' f.py") == ["s/a/b/"]
    assert sed_scripts("sed -n '100,106p' f.py") == ["100,106p"]
    assert sed_scripts("sed -i -e 's/a/b/' -e 's/c/d/' f.py") == ["s/a/b/", "s/c/d/"]
    # c12aac70: several seds chained, so every script has to parse before we override
    chained = "sed -i '263,279d' a.py && sed -i 's/x/y/' b.py"
    assert sed_scripts(chained) == ["263,279d", "s/x/y/"]
    assert sed_scripts("git diff") == []
    assert sed_scripts("grep -n 'sed' f.py") == []


def test_an_unjudgeable_script_keeps_the_simulator_answer():
    command = "cd /w && HDR=$(grep -n 'x' f.go | cut -d: -f1) && sed -n \"${HDR},+50p\" f.go"
    assert not fabricated_sed_error(command, _SED_ERROR)
    # unbalanced quotes are the shell's problem, not sed's
    assert sed_scripts("sed -i 's/a/b/ f.py") == []


def test_a_real_diagnostic_is_available_for_a_genuinely_broken_script():
    from albedo_eval_service.shared.sed_check import sed_error_message

    broken = (
        "sed -i '263,279d' dspy/adapters/json_adapter.py && "
        "sed -i '/^ def format_assistant_message_content(/,/^[[:space:]]*def /"
        "{ /^[[:space:]]*def /!{ /^[[:space:]]*return /a\\ z' dspy/adapters/json_adapter.py"
    )
    assert sed_error_message(broken) == "sed: -e expression #1, char 0: unmatched `{'"


def test_no_diagnostic_is_offered_when_the_shell_would_have_substituted_first():
    from albedo_eval_service.shared.sed_check import sed_error_message

    for opaque in (
        'sed -n "${HDR},+50p" f.go',
        'sed -n "$(grep -c x f.go),100p" f.go',
        "sed -i '1,5s/x/`whoami`/' a.py",
    ):
        assert sed_error_message(opaque) == "", opaque


def test_a_dollar_anchor_is_not_mistaken_for_shell_substitution():
    from albedo_eval_service.shared.sed_check import sed_accepts, sed_error_message

    anchored = "sed -i 's/ else:\\n raise TypeError.*$/ else:\\n try:/' d.py"
    assert sed_accepts("s/ else:\\n raise TypeError.*$/ else:\\n try:/")
    assert sed_error_message(anchored) == "", "a valid anchored script has no error to report"
    # and a broken one with an anchor still gets a real diagnostic
    assert sed_error_message("sed -i 's/a$/b' a.py").startswith("sed:")


def test_a_valid_or_non_sed_command_yields_no_diagnostic():
    from albedo_eval_service.shared.sed_check import sed_error_message

    assert sed_error_message("sed -i 's/a/b/' a.py") == ""
    assert sed_error_message("git diff") == ""
    assert sed_error_message("") == ""
