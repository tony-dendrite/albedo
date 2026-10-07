"""`edited_in_turn` decides where a trajectory's first edit is: the manifest's `first_edit` and the
evaluator's phase of a sample both come from it, so they agree only if it is right."""

import pytest

from albedo_eval_service.shared.edit_detection import command_edits, edited_in_turn


def bash(command: str) -> str:
    return f"THOUGHT: t\n\n```bash\n{command}\n```"


@pytest.mark.parametrize(
    "command",
    [
        "sed -i s/a/b/ f.py",
        "sed -e 's/a/b/' -i f.py",
        "cd /testbed && sed -i '1d' a.py",
        "perl -pi -e 's/a/b/' f.py",
        "echo hi > out.txt",
        'echo x > "a b.py"',
        "cmd &> log.txt",
        "tee f.py",
        "cp a.py b.py",
        "git mv a.py b.py",
        "patch -p1 < d.diff",
        "git apply d.diff",
        "cat <<'EOF' > /testbed/a.py\nx = 1\nEOF",
        "cat >> f.py << EOF\nx > y.z\nEOF",
        "python3 - <<'PY'\nfrom pathlib import Path\nPath('/testbed/a.py').write_text('x')\nPY",
        "cat <<'EOF' | python3\nopen('a.py', 'w').write('x')\nEOF",
        "python -c \"open(os.path.join('a', 'b.py'), 'w').write('x')\"",
    ],
)
def test_a_write_into_the_checkout_is_an_edit(command):
    assert command_edits(command)


@pytest.mark.parametrize(
    "command",
    [
        "cat f.py",
        "sed -n 1,10p src/my-iter.py",
        "find . 2>/dev/null",
        "ls 2>&1",
        "pytest 1>/dev/null",
        "cmd | tee /dev/null",
        "grep -n 'x > 0.5' a.py",
        'grep -rn "sed -i" .',
        "awk '$1 > 3' f",
        "cat <<< 'a > b.py'",
        "(( a > 0 ))",
        # scratch space and home are not the repository
        "cat > /tmp/r.py <<'EOF'\nx\nEOF",
        "echo x > /tmp/a.py",
        "cp a.py /tmp/",
        'mkdir -p ~/.cfg && echo "x" > ~/.cfg/config.ini',
        "python -c \"open('/tmp/x', 'w')\"",
        # dry runs change nothing
        "git apply --check d.diff",
        "patch --dry-run -p1 < d.diff",
        # code that runs but writes nothing: its `>` is a comparison
        "python3 - <<'PY'\nif a > self.limit:\n    print(open('a.py').read())\nPY",
        'python -c "\nif score > best.value:\n    print(score)\n"',
        # code written to a file is not run
        "cat > /tmp/fix.py <<'EOF'\nimport shutil\nshutil.move(a, b)\nopen('x', 'w')\nEOF",
    ],
)
def test_a_command_that_writes_nothing_in_the_checkout_is_not_an_edit(command):
    assert not command_edits(command)


def test_a_block_in_another_language_does_not_hide_the_command_after_it():
    """The old block finder took the closing fence of a ```python block for an opening one, so
    the bash block after it was never read."""
    turn = "THOUGHT: the fix is\n\n```python\nx = 1\n```\n\nApplying it:\n\n" + bash(
        "sed -i s/a/b/ f.py"
    )
    assert edited_in_turn(turn)


def test_only_the_command_is_read_not_the_prose():
    prose = "THOUGHT: we should run sed -i on it, since x > y.py\n\n"
    assert not edited_in_turn(prose + bash("ls"))


def test_a_turn_with_several_commands_runs_none_of_them():
    assert not edited_in_turn(bash("ls") + "\n\n" + bash("sed -i s/a/b/ f.py"))


def test_a_rendered_editor_edit_is_an_edit_unless_it_is_scratch():
    """The renderer writes an editor `str_replace` as a SEARCH/REPLACE block, whose text may hold
    fences of its own."""
    edit = (
        "Editing `{}`:\n\n```\n<<<<<<< SEARCH\n```bash\nold\n```\n"
        "=======\nnew\n>>>>>>> REPLACE\n```"
    )
    assert edited_in_turn("THOUGHT: t\n\n" + edit.format("/testbed/README.md"))
    assert not edited_in_turn("THOUGHT: t\n\n" + edit.format("/tmp/x.toml"))
