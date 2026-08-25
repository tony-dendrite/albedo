from __future__ import annotations

import asyncio
import json
import random
from types import SimpleNamespace

from albedo_eval_service.judge_llm_client import JudgeRawResponse
from sanity_service.chain import (
    MAX_REJECTIONS,
    applied_edits,
    empty_submit_count,
    generate_followup,
    generate_microtask,
    inplace_edited_paths,
    should_reject,
    ungrounded_reason,
    unread_edited_files,
)

MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_ANSWER"
CLAUSE = f"echo {MARKER} && git add -A && git diff --cached"


def _asked(text: str) -> dict:
    return {"role": "user", "content": text, "injected": True}


def _bash(command: str) -> str:
    return f"THOUGHT: doing the thing\n\n```bash\n{command}\n```"


def _state(turns: list[dict], **kw) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id="s0",
        turns=turns,
        messages=[{"role": t["role"], "content": t["content"]} for t in turns],
        submits=[],
        submit_marker=MARKER,
        submit_clause=CLAUSE,
        **kw,
    )


def test_sed_script_is_not_mistaken_for_an_edited_path():
    # the substitution expression has slashes and a dot, but the edited file is the operand
    command = (
        "sed -i 's/copy_permissions_url_template/CopyUrlTemplates.copy_permissions_url_template/' "
        "./gspread/client.py"
    )
    assert inplace_edited_paths(_bash(command)) == ["./gspread/client.py"]


def test_inplace_paths_cover_flag_forms_and_ignore_reads():
    assert inplace_edited_paths(_bash("sed -i -e 's/a/b/' -e 's/c/d/' lib/x.rb")) == ["lib/x.rb"]
    assert inplace_edited_paths(_bash("sed -i.bak 's/a/b/' lib/x.rb")) == ["lib/x.rb"]
    assert inplace_edited_paths(_bash("sed -i 's/x/y/' a/one.py b/two.py")) == [
        "a/one.py",
        "b/two.py",
    ]
    assert inplace_edited_paths(_bash("sed -n '1,10p' pkg/mod.py")) == []
    assert inplace_edited_paths("prose mentioning sed -i 's/a/b/' pkg/mod.py") == []


def test_unread_edited_files_accepts_a_file_that_was_read_first():
    state = _state(
        [
            {
                "role": "assistant",
                "content": _bash("cat ./gspread/client.py"),
                "score_target": True,
            },
            {"role": "user", "content": "<output>class Client:</output>"},
            {
                "role": "assistant",
                "content": _bash("sed -i 's/a/Cls.a/' ./gspread/client.py"),
                "score_target": True,
            },
        ]
    )
    assert unread_edited_files(state) == []


def test_unread_edited_files_still_flags_a_blind_edit():
    state = _state(
        [
            {
                "role": "assistant",
                "content": _bash("sed -i 's/a/b/' ./pkg/never_read.py"),
                "score_target": True,
            }
        ]
    )
    assert unread_edited_files(state) == ["./pkg/never_read.py"]


def test_submission_we_asked_for_unconditionally_is_not_empty_work():
    turns = [
        {"role": "assistant", "content": _bash(f"echo {MARKER}"), "score_target": True},
        {"role": "user", "content": "Ship whatever you have.", "injected": True, "nudge": True},
        {"role": "assistant", "content": _bash(f"echo {MARKER}"), "score_target": True},
    ]
    assert empty_submit_count(_state(turns), MARKER) == 1


def test_two_unprompted_no_op_resubmits_still_count():
    turns = [
        {"role": "assistant", "content": _bash(f"echo {MARKER}"), "score_target": True},
        {"role": "user", "content": "Please handle the missed call site.", "injected": True},
        {"role": "assistant", "content": _bash(f"echo {MARKER}"), "score_target": True},
    ]
    assert empty_submit_count(_state(turns), MARKER) == 2


def test_work_between_submissions_clears_the_count():
    turns = [
        {"role": "assistant", "content": _bash(f"echo {MARKER}"), "score_target": True},
        {"role": "user", "content": "One more thing.", "injected": True},
        {
            "role": "assistant",
            "content": _bash("sed -i 's/a/b/' pkg/mod.py"),
            "score_target": True,
        },
        {"role": "assistant", "content": _bash(f"echo {MARKER}"), "score_target": True},
    ]
    assert empty_submit_count(_state(turns), MARKER) == 1


def test_rejections_are_capped_per_chain():
    rng = random.Random(7)
    state = SimpleNamespace(submits=[])
    rejected = 0
    for _ in range(32):
        if should_reject(state, rng):
            rejected += 1
            state.submits.append({"rejected": True})
        else:
            state.submits.append({})
    assert rejected == MAX_REJECTIONS


def test_ungrounded_reason_rejects_unusable_in_world_messages():
    context = "lib/bot.js\nfunction sendMessage() {}"
    assert not ungrounded_reason("Please guard sendMessage in lib/bot.js.", context)
    assert "absent from the session" in ungrounded_reason(
        "The CI failure is in tests/test_scheduler.py — fix it.", context
    )
    assert "test changes" in ungrounded_reason(
        "Nice. Could you add a regression test for that path?", context
    )
    assert "agent turn" in ungrounded_reason("THOUGHT: submitting now\n\n```bash\nls\n```", context)
    assert "degenerate" in ungrounded_reason("the the " * 30, context)


def test_ungrounded_reason_allows_dotted_attributes_and_the_submit_clause():
    context = "class Client:\n    def copy(self): ...\n"
    assert not ungrounded_reason("Use os.path here and keep self.data intact.", context, CLAUSE)
    assert not ungrounded_reason(
        f"Keep self.data intact in copy(), then submit again with {CLAUSE} && cat patch.txt",
        context,
        f"{CLAUSE} && cat patch.txt",
    )


class _Judge:
    """Returns an ungrounded message until `good_after` attempts have been made."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    async def complete(self, *, model, messages, temperature=None, accept=None, **kw):
        while self.calls < len(self.replies):
            raw = self.replies[self.calls]
            self.calls += 1
            if accept is None or accept(raw):
                return JudgeRawResponse(model="sim", provider=None, raw=raw)
        return JudgeRawResponse(model="sim", provider=None, raw=self.replies[-1])


def test_followup_retries_past_an_ungrounded_message():
    state = _state([{"role": "user", "content": "lib/bot.js has sendMessage"}])
    judge = _Judge(
        [
            "Please add a regression test in tests/test_ghost.py.",
            "The sendMessage guard in lib/bot.js misses the nil case — please handle it.",
        ]
    )
    settings = SimpleNamespace(simulation_model="sim")
    out = asyncio.run(generate_followup(judge, settings, state, "submission"))
    assert "lib/bot.js" in out
    assert judge.calls == 2


def test_followup_is_empty_when_every_attempt_is_ungrounded():
    """No grounded request means no request: the caller accepts the submission instead of
    injecting a vague 'double-check everything', which a correct agent answers with no edit."""
    state = _state([{"role": "user", "content": "lib/bot.js has sendMessage"}])
    judge = _Judge(["Fix tests/ghost_one.py.", "Fix tests/ghost_two.py."])
    settings = SimpleNamespace(simulation_model="sim")
    out = asyncio.run(generate_followup(judge, settings, state, "submission"))
    assert out == ""


def test_applied_edits_lists_what_the_prefix_already_changed():
    state = _state(
        [
            {"role": "assistant", "content": _bash("cat pkg/mod.py")},
            {"role": "assistant", "content": _bash("sed -i '49s/a/b/' pkg/mod.py")},
        ]
    )
    listed = applied_edits(state)
    assert "sed -i '49s/a/b/' pkg/mod.py" in listed
    assert "cat pkg/mod.py" not in listed
    assert applied_edits(_state([])) == "- (none yet)"


def test_microtask_accepts_a_grounded_request():
    state = _state([{"role": "user", "content": "repo has pkg/mod.py with parse()"}])
    good = json.dumps(
        {
            "file": "pkg/mod.py",
            "function": "parse",
            "request": "guard the empty case",
            "message": f"Please guard the empty case in parse in pkg/mod.py. Run exactly: {CLAUSE}",
        }
    )
    judge = _Judge([good])
    settings = SimpleNamespace(evaluator_model="eval")
    micro = asyncio.run(generate_microtask(judge, settings, state, CLAUSE))
    assert micro["file"] == "pkg/mod.py"


def test_bad_turn_is_re_asked_and_recovery_clears_the_counter():
    """A malformed turn is dropped and re-asked; a good answer resumes the chain."""
    from sanity_service import dispatcher as D

    state = D._TrajectoryState(
        sample_id="s0", prompt="p", messages=[{"role": "user", "content": "task"}], turns=[]
    )
    bad = {
        "responses": ["MODEL_RESPONSE_TOKEN_LIMIT_EXCEEDED: cut off"],
        "heuristics": [
            {"passed": False, "reason": "response exceeded the model response token limit"}
        ],
    }
    D._apply_turn_result([state], bad)
    assert state.retry_reason and state.consecutive_bad_turns == 1
    assert not state.heuristic_reason
    assert state.turns[-1]["retry_feedback"] is True
    assert "output token limit" in state.turns[-1]["content"]
    assert not [t for t in state.turns if t["role"] == "assistant"]

    good = {
        "responses": [_bash("ls -la")],
        "heuristics": [{"passed": True, "reason": ""}],
    }
    D._apply_turn_result([state], good)
    assert state.consecutive_bad_turns == 0
    assert not state.retry_reason and not state.heuristic_reason
    assert state.turns[-1]["role"] == "assistant"


def test_three_consecutive_bad_turns_fail_the_sample():
    from sanity_service import dispatcher as D

    state = D._TrajectoryState(
        sample_id="s0", prompt="p", messages=[{"role": "user", "content": "task"}], turns=[]
    )
    bad = {"responses": [""], "heuristics": [{"passed": False, "reason": "empty response"}]}
    for _ in range(D.MAX_CONSECUTIVE_BAD_TURNS):
        D._apply_turn_result([state], bad)
    assert state.consecutive_bad_turns == D.MAX_CONSECUTIVE_BAD_TURNS
    assert state.heuristic_reason == "empty response on 3 consecutive turns"


def test_non_consecutive_bad_turns_never_fail_the_sample():
    """Isolated transient turns are what the old gate died on; they must now be survivable."""
    from sanity_service import dispatcher as D

    state = D._TrajectoryState(
        sample_id="s0", prompt="p", messages=[{"role": "user", "content": "task"}], turns=[]
    )
    bad = {"responses": [""], "heuristics": [{"passed": False, "reason": "empty response"}]}
    good = {"responses": [_bash("ls")], "heuristics": [{"passed": True, "reason": ""}]}
    for _ in range(10):
        D._apply_turn_result([state], bad)
        D._apply_turn_result([state], good)
    assert not state.heuristic_reason


def test_repeated_followup_accepts_instead_of_asking_twice():
    """Seed 2: the same request was injected verbatim twice and the second cost a strike."""
    from sanity_service import dispatcher as D

    state = D._TrajectoryState(
        sample_id="s0", prompt="p", messages=[{"role": "user", "content": "task"}], turns=[]
    )
    state.submit_clause = CLAUSE
    state.segment = "micro"
    state.turns.append({"role": "assistant", "content": _bash(CLAUSE), "score_target": True})
    same = "The index.go changes look good, but check the overflowID values."
    D._advance_segment(state, _bash(CLAUSE), same, turn_index=0)
    assert not state.stopped
    state.turns.append({"role": "assistant", "content": _bash(CLAUSE), "score_target": True})
    D._advance_segment(state, _bash(CLAUSE), same, turn_index=1)
    assert state.stopped is True


def test_requester_claiming_the_work_is_rejected():
    """Seed 2: the simulator wrote as the agent — 'I've fixed ...' — so the model submitted."""
    ctx = "index.go\nfunc split() {}\nfreeOverflowBucket"
    leak = (
        "I've fixed the persistence issue by adding a second write() call after freeing the "
        "overflow buckets in index.go. This makes split() atomic and crash-safe."
    )
    assert "agent turn" in ungrounded_reason(leak, ctx)
    ok = "The split() change looks right, but check freeOverflowBucket in index.go once more."
    assert not ungrounded_reason(ok, ctx)


def test_character_break_is_rejected_on_the_real_leaks():
    """Both genuine leaks found in the dashboard fault artifacts."""
    ctx = "pkg/mod.py\nauth.go\nfunc ScrambleOldPassword"
    real_leaks = [
        "I apologize, it appears that the user is not an AI. But the user is a big deal.",
        'Please help me with this question under the request "AI assistant API and developer '
        'API reference. The user is asked to write a response on the topic".',
        "I'm simulating the output for you here.",
        "This is just a simulation, so the file does not really exist.",
        "As an AI, I cannot actually run that command.",
    ]
    for leak in real_leaks:
        assert "breaks character" in ungrounded_reason(leak, ctx), leak


def test_character_break_does_not_fire_on_real_repo_content():
    """4144 real observations contain mock/benchmark/pretend legitimately — none may trip."""
    ctx = "x"
    legit = [
        "The mock client setup in TestNewK8sSecretAuthProvider is missing the new field.",
        "maint/benchmark/ still references the old flag.",
        "The comment says: # We gently pretend we're a Python 3 mappingproxy.",
        "db, mock, err := sqlmock.New() needs the extra column.",
    ]
    for text in legit:
        assert "breaks character" not in (ungrounded_reason(text, ctx) or "")


def test_python_heredoc_edit_counts_as_work():
    real_edit = _bash(
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "path = Path('pennylane/ops/qubit/non_parametric_ops.py')\n"
        "path.write_text(source)\n"
        "PY\n"
        f"python -m pytest -q && echo {MARKER} && cat patch.txt"
    )
    turns = [
        {"role": "assistant", "content": real_edit, "score_target": True},
        _asked("The Barrier and WireCut overrides still ignore wire_order."),
        {"role": "assistant", "content": real_edit, "score_target": True},
    ]
    assert empty_submit_count(_state(turns), MARKER) == 0


def test_edit_regex_covers_the_common_writers():
    from sanity_service.chain import _EDIT_RE

    for cmd in [
        "sed -i 's/a/b/' f.py",
        "cat > f.py",
        "tee f.py",
        "git apply p.diff",
        "patch -p1 < p.diff",
        "python - <<'PY'\npath.write_text(x)\nPY",
        "cat <<'SCRIPT' > run.sh",
        "python -c \"open('f.py','w').write(x)\"",
    ]:
        assert _EDIT_RE.search(cmd), cmd
    for cmd in ["ls -la", "grep -n foo f.py", "cat f.py", "pwd", "python -m pytest -q"]:
        assert not _EDIT_RE.search(cmd), cmd


def test_edit_regex_covers_write_redirects_but_not_lookalikes():
    from sanity_service.chain import _EDIT_RE

    for cmd in [
        'echo "# IVS" > ./moto/ivs/__init__.py',
        "printf 'x' > f.py",
        "git show HEAD:src/a.ts > src/a.ts",
        "git diff -- a.py > /workspace/patch.txt",
        "cat notes >> docs/log.md",
    ]:
        assert _EDIT_RE.search(cmd), cmd
    for cmd in [
        "pytest -q 2>/dev/null",
        "make 2>&1 | tail -5",
        "ls > /dev/null",
        "echo boom >&2",
        "awk 'length > 79' f.py",
        "if (a >= b.c):",
        "x -> self.foo",
        "<p>a.b</p>",
    ]:
        assert not _EDIT_RE.search(cmd), cmd
