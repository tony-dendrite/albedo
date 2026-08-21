from __future__ import annotations

from repo_context_service.command_search import ParseFailure
from repo_context_service.git_sim import (
    GitMeta,
    blob_hash,
    is_git_command,
    ledger_block,
    run_git_chain,
)
from repo_context_service.overlay import build_overlay

BASE = {
    "src/app.py": "def main():\n    return 1\n",
    "README.md": "# demo\n",
}
LISTING = sorted(BASE)
META = GitMeta(sha="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0", owner="acme", repo="demo")


def _read_base(rel: str) -> str | None:
    return BASE.get(rel)


def _turn(command: str, body: str = "", returncode: int = 0) -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": f"```bash\n{command}\n```"},
        {
            "role": "user",
            "content": f"<returncode>{returncode}</returncode>\n<output>\n{body}\n</output>",
        },
    ]


def _run(command: str, messages: list[dict[str, str]] | None = None, meta: GitMeta = META):
    overlay = build_overlay(messages or [], LISTING, {}, _read_base)
    return run_git_chain(command, overlay, _read_base, LISTING, meta), overlay


EDIT = _turn("cat > src/app.py <<'EOF'\ndef main():\n    return 2\nEOF")
CREATE = _turn("cat > notes.txt <<'EOF'\nhello\nEOF")


def test_git_add_is_silent_with_returncode_zero():
    result, _ = _run("cd /testbed && git add -A", EDIT)
    assert result.output == ""
    assert result.returncode == 0
    assert result.empty


def test_status_reports_the_default_branch_and_the_unstaged_edit():
    result, _ = _run("git status", EDIT + CREATE)
    assert result.output.split("\n") == [
        "On branch main",
        "Changes not staged for commit:",
        '  (use "git add <file>..." to update what will be committed)',
        '  (use "git restore <file>..." to discard changes in working directory)',
        "\tmodified:   src/app.py",
        "",
        "Untracked files:",
        '  (use "git add <file>..." to include in what will be committed)',
        "\tnotes.txt",
        "",
        'no changes added to commit (use "git add" and/or "git commit -a")',
    ]


def test_status_on_a_rebench_image_reports_a_detached_head():
    result, _ = _run("git status", EDIT, meta=GitMeta(sha=META.sha, detached=True))
    assert result.output.startswith("Not currently on any branch.\n")


def test_a_staged_tree_drops_the_summary_line_and_keeps_the_trailing_blank():
    result, _ = _run("git status", EDIT + _turn("git add -A"))
    assert result.output.endswith("\tmodified:   src/app.py\n")


def test_diff_renders_real_blob_hashes_and_git_hunk_headers():
    result, _ = _run("git diff", EDIT)
    old = blob_hash(BASE["src/app.py"])[:7]
    new = blob_hash("def main():\n    return 2\n")[:7]
    assert result.output.split("\n") == [
        "diff --git a/src/app.py b/src/app.py",
        f"index {old}..{new} 100644",
        "--- a/src/app.py",
        "+++ b/src/app.py",
        "@@ -1,2 +1,2 @@",
        " def main():",
        "-    return 1",
        "+    return 2",
    ]


def test_staging_moves_the_change_from_diff_to_diff_cached():
    staged = EDIT + _turn("git add -A")
    plain, _ = _run("git diff", staged)
    cached, _ = _run("git diff --cached", staged)
    assert plain.empty
    assert "-    return 1" in cached.output


def test_a_chain_executes_stage_by_stage_with_the_staging_applied():
    result, _ = _run("cd /testbed && git add -A && git diff --cached", EDIT)
    assert result.output.startswith("diff --git a/src/app.py b/src/app.py")
    assert result.returncode == 0


def test_checkout_of_a_pathspec_reports_the_updated_path_count():
    result, _ = _run("git checkout src/app.py", EDIT)
    assert result.output == "Updated 1 path from the index"


def test_a_double_dash_checkout_is_completely_silent():
    result, _ = _run("git checkout -- src/app.py", EDIT)
    assert result.output == ""
    assert result.returncode == 0


def test_stash_leaves_a_clean_tree_and_the_ledger_records_it():
    messages = EDIT + _turn("git add -A") + _turn("git stash", "Saved working directory")
    result, overlay = _run("git status", messages)
    assert result.output == "On branch main\nnothing to commit, working tree clean"
    assert "git stash" in ledger_block(overlay.git)
    assert "git add -A" in ledger_block(overlay.git)


def test_popping_an_empty_stash_fails_the_way_git_does():
    result, _ = _run("git stash pop", EDIT)
    assert result.output == "No stash entries found."
    assert result.returncode == 1


def test_the_abbreviation_length_is_learned_from_an_earlier_observation():
    seen = _turn("git diff", "index 1234567890..abcdef1234 100644")
    result, _ = _run("git diff", seen + EDIT)
    old = blob_hash(BASE["src/app.py"])[:10]
    new = blob_hash("def main():\n    return 2\n")[:10]
    assert result.output.split("\n")[1] == f"index {old}..{new} 100644"


def test_an_unmodelled_git_command_poisons_the_state_instead_of_guessing():
    result, _ = _run("git status", _turn("git commit -m wip", "[main abc1234] wip"))
    assert isinstance(result, ParseFailure)


def test_a_chain_stage_we_cannot_execute_refuses_the_whole_command():
    result, _ = _run("python reproduce.py && git add -A && git diff --cached", EDIT)
    assert isinstance(result, ParseFailure)


def test_a_command_without_a_git_stage_is_left_to_the_search_executor():
    result, _ = _run("grep -rn TODO src", EDIT)
    assert isinstance(result, ParseFailure)
    assert result.reason == "not_git"


def test_the_ledger_is_rendered_for_git_commands_only():
    assert is_git_command("git status")
    assert is_git_command("cd /testbed && git add app.py")
    assert not is_git_command("cat README.md")
    assert not is_git_command("grep -rn digit .")
