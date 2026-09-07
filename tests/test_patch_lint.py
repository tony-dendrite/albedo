from albedo_eval_service.shared.patch_lint import final_submit_issue

M = "SUBMIT_TASK_F31DF145"
SUBMIT = f"echo {M} && cat patch.txt"
GOOD_DIFF = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n-a = 1\n+a = 2\n b = 2"


def test_a_real_patch_copied_into_place_is_fine():
    cmds = [
        "cd /workspace/repo && git diff pypika/terms.py > /tmp/patch.txt && cat /tmp/patch.txt",
        f"cp /tmp/patch.txt /workspace/repo/patch.txt && echo {M} && cat /workspace/repo/patch.txt",
    ]
    assert final_submit_issue(cmds, M) == ""


def test_copying_source_content_into_the_patch_is_still_flagged():
    cmds = ["cp src/foo.py patch.txt", SUBMIT]
    assert final_submit_issue(cmds, M) == "patch file created by copying file content, not a diff"


def test_git_diff_redirect_and_tee_are_fine():
    assert final_submit_issue(["git diff -- a.py > patch.txt", SUBMIT], M) == ""
    assert final_submit_issue(["git diff | tee patch.txt", SUBMIT], M) == ""
    assert final_submit_issue([f"git diff -- a.py > patch.txt && {SUBMIT}"], M) == ""


def test_submitting_a_patch_that_was_never_written_is_flagged():
    cmds = ["sed -i 's/a/b/' src/foo.py", "cat -n src/foo.py", SUBMIT]
    assert final_submit_issue(cmds, M) == "submits a patch file it never created"


def test_writing_the_patch_only_after_the_final_submit_is_flagged():
    cmds = [
        "sed -i 's/a/b/' src/foo.py",
        SUBMIT,
        "git diff src/foo.py > patch.txt && cat patch.txt",
    ]
    assert final_submit_issue(cmds, M) == "submits a patch file it never created"


def test_catting_source_instead_of_a_patch_is_flagged():
    cmds = [f"echo {M} && cat pyprep/reference.py | sed -n '140,150p'"]
    assert final_submit_issue(cmds, M) == "final submit cats source content instead of a patch file"


def test_hand_written_patch_is_checked_with_git_apply():
    ok = [f"cat > patch.txt <<'EOF'\n{GOOD_DIFF}\nEOF", SUBMIT]
    assert final_submit_issue(ok, M) == ""
    corrupt = [
        "cat > patch.txt <<'EOF'\n--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n-a = 1\n+a = 2\nEOF",
        SUBMIT,
    ]
    assert final_submit_issue(corrupt, M).startswith("hand-written patch file would not apply")


def test_no_marked_submission_means_nothing_to_lint():
    assert final_submit_issue(["ls", "cat patch.txt"], M) == ""
