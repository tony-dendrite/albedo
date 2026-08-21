import re

from albedo_eval_service.remote.dataset import EvalSample, apply_submit_protocol
from albedo_eval_service.shared.submit_protocol import (
    ANY_MARKER_RE,
    CANONICAL_MARKER,
    MARKERS,
    TAILS,
    assign_submit,
    command_for,
    first_bash_command,
    has_native_submission,
    is_exact_submission,
    keep_original_ids,
    marker_for,
    marker_from,
    rewrite_messages,
    stated_command,
)

NATIVE_COMMAND = f"echo {CANONICAL_MARKER} && git add -A && git diff --cached"
SECTION_SYSTEM = (
    "You are a coding agent.\n\n"
    "## Submission\n\n"
    "When done, run this EXACT command:\n\n"
    f"```bash\n{NATIVE_COMMAND}\n```\n\n"
    "You cannot continue after submitting.\n\n"
    "## Other rules\n\n"
    "Reply with one bash block.\n"
)
INLINE_SYSTEM = (
    "You are a coding agent in a SWE-agent style environment. "
    f"When the task is complete, run `echo {CANONICAL_MARKER}`."
)
PLAIN_SYSTEM = "You are a coding agent. Reply with exactly one fenced bash command."

NONCE = marker_for("nonce", "mini-coder:1:3", "salt-a")
NONCE_COMMAND = command_for(NONCE, "patchtxt")


def _native_messages():
    return [
        {"role": "system", "content": SECTION_SYSTEM},
        {"role": "user", "content": "Fix the bug in foo.py"},
        {"role": "assistant", "content": f"history mentioning {CANONICAL_MARKER}"},
    ]


def _plain_messages():
    return [
        {"role": "system", "content": PLAIN_SYSTEM},
        {"role": "user", "content": "Fix the bug."},
    ]


def test_assign_submit_is_deterministic_per_salt_and_sample():
    assert assign_submit("s:1:2", "salt") == assign_submit("s:1:2", "salt")
    assert assign_submit("s:1:2", "salt") != assign_submit("s:1:2", "other-salt")


def test_assign_submit_covers_all_arms_and_tails():
    arms = set()
    tails = set()
    for i in range(300):
        marker, command = assign_submit(f"s:{i}:1", "salt")
        assert command.startswith(f"echo {marker}")
        if marker.startswith("SUBMIT_TASK_"):
            assert re.fullmatch(r"SUBMIT_TASK_[0-9A-F]{8}", marker)
            arms.add("nonce")
        else:
            arms.add(marker)
        tails.add(command.removeprefix(f"echo {marker}"))
    assert len(arms) == len(MARKERS)
    assert tails == set(TAILS.values())


def test_nonce_marker_varies_by_sample_and_salt():
    assert marker_for("nonce", "a", "s") == marker_for("nonce", "a", "s")
    assert marker_for("nonce", "a", "s") != marker_for("nonce", "b", "s")
    assert marker_for("nonce", "a", "s") != marker_for("nonce", "a", "t")


def test_any_marker_re_matches_only_protocol_markers():
    for marker in [m for m in MARKERS.values() if m] + ["SUBMIT_TASK_0AF3B21C"]:
        assert ANY_MARKER_RE.search(f"run echo {marker} now")
    for text in [
        "SUBMIT_TASK_123",
        "RESUBMIT_TASK_12345678",
        "SUBMIT_TASK_123456789",
        "SUBMIT_TASK_abcdef12",
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUTX",
        "COMPLETE_TASK_AND_SUBMIT_FINAL",
    ]:
        assert not ANY_MARKER_RE.search(f"text {text} text")


def test_keep_original_ids_quota():
    ids = [f"mini-coder:{i}:3" for i in range(8)]
    kept = keep_original_ids(ids, "salt", 0.25)
    assert len(kept) == 2
    assert kept <= set(ids)
    assert kept == keep_original_ids(ids, "salt", 0.25)
    assert len(keep_original_ids(ids, "salt", 0.5)) == 4
    assert keep_original_ids([], "salt", 0.25) == set()


def test_has_native_submission_checks_instruction_turns_only():
    assert has_native_submission(_native_messages())
    assert not has_native_submission(_plain_messages())
    assert not has_native_submission(
        [
            {"role": "system", "content": PLAIN_SYSTEM},
            {"role": "assistant", "content": f"history with {CANONICAL_MARKER}"},
        ]
    )


def test_stated_command_extraction():
    assert stated_command(_native_messages()) == NATIVE_COMMAND
    assert stated_command(_plain_messages()) == ""
    ambiguous = [
        {
            "role": "system",
            "content": SECTION_SYSTEM + f"\nAlso `echo {CANONICAL_MARKER}` works.",
        }
    ]
    assert stated_command(ambiguous) == ""


def test_rewrite_replaces_native_section():
    messages = _native_messages()
    rewritten, mode = rewrite_messages(messages, NONCE_COMMAND)
    head = "\n".join(m["content"] for m in rewritten[:2])
    assert mode == "replaced"
    assert CANONICAL_MARKER not in head
    assert head.count(NONCE_COMMAND) == 1
    assert "## Other rules" in head
    assert CANONICAL_MARKER in rewritten[2]["content"]
    assert CANONICAL_MARKER in messages[0]["content"]


def test_rewrite_replaces_inline_command():
    rewritten, mode = rewrite_messages(
        [{"role": "system", "content": INLINE_SYSTEM}], NONCE_COMMAND
    )
    assert mode == "replaced_command_only"
    assert NONCE_COMMAND in rewritten[0]["content"]
    assert CANONICAL_MARKER not in rewritten[0]["content"]


def test_rewrite_appends_section_when_absent():
    rewritten, mode = rewrite_messages(_plain_messages(), NONCE_COMMAND)
    assert mode == "added"
    assert NONCE_COMMAND in rewritten[1]["content"]
    assert rewritten[0]["content"] == PLAIN_SYSTEM


def test_rewrite_fails_open_when_canonical_survives():
    messages = [
        {
            "role": "system",
            "content": f"Note: {CANONICAL_MARKER} is special.\n\n" + SECTION_SYSTEM,
        }
    ]
    rewritten, mode = rewrite_messages(messages, NONCE_COMMAND)
    assert mode == "failed"
    assert rewritten[0]["content"] == messages[0]["content"]


def test_is_exact_submission():
    assert is_exact_submission(f"done\n```bash\n{NONCE_COMMAND}\n```", NONCE_COMMAND)
    assert is_exact_submission(f"```bash\necho  {NONCE}  &&  cat patch.txt\n```", NONCE_COMMAND)
    assert not is_exact_submission(
        f"```bash\necho {NONCE} && cat /tmp/patch.txt\n```", NONCE_COMMAND
    )
    assert not is_exact_submission(f"I will run {NONCE_COMMAND} soon", NONCE_COMMAND)
    assert not is_exact_submission(f"```bash\nls\n{NONCE_COMMAND}\n```", NONCE_COMMAND)
    assert not is_exact_submission(
        f"```bash\nls -la\n```\n```bash\n{NONCE_COMMAND}\n```", NONCE_COMMAND
    )
    assert not is_exact_submission("```bash\n\n```", "")
    assert first_bash_command("```\nnot bash\n```") == ""


def test_marker_from_command():
    assert marker_from(NONCE_COMMAND) == NONCE
    assert marker_from(f"echo {CANONICAL_MARKER}") == CANONICAL_MARKER
    assert marker_from("git diff") == CANONICAL_MARKER


def _samples():
    native = [
        EvalSample(sample_id=f"mini-coder:{i}:3", prompt="p", messages=_native_messages())
        for i in range(8)
    ]
    plain = [
        EvalSample(sample_id=f"open-swe-traces:{i}:2", prompt="p", messages=_plain_messages())
        for i in range(4)
    ]
    return native + plain


def test_apply_submit_protocol_quota_and_modes():
    out = apply_submit_protocol(_samples(), salt="block-hash", keep_original_ratio=0.25)
    modes = [s.rewrite_mode for s in out]
    assert modes.count("original") == 2
    assert sum(m in ("replaced", "replaced_command_only") for m in modes) == 6
    assert modes.count("added") == 4


def test_apply_submit_protocol_original_samples_untouched():
    out = apply_submit_protocol(_samples(), salt="block-hash", keep_original_ratio=0.25)
    for sample in out:
        if sample.rewrite_mode != "original":
            continue
        assert sample.prompt == "p"
        assert sample.messages == _native_messages()
        assert sample.submit_command == NATIVE_COMMAND
        assert sample.submit_marker == CANONICAL_MARKER


def test_apply_submit_protocol_rewritten_samples_consistent():
    out = apply_submit_protocol(_samples(), salt="block-hash", keep_original_ratio=0.25)
    for sample in out:
        if sample.rewrite_mode == "original":
            continue
        head = "\n".join(m["content"] for m in sample.messages[:2])
        assert sample.submit_command in head
        assert sample.submit_command in sample.prompt
        assert ANY_MARKER_RE.fullmatch(sample.submit_marker)
        assert marker_from(sample.submit_command) == sample.submit_marker


def test_apply_submit_protocol_deterministic_and_salted():
    base = _samples()
    key = lambda out: [(s.sample_id, s.submit_command, s.rewrite_mode) for s in out]  # noqa: E731
    assert key(apply_submit_protocol(base, salt="a")) == key(apply_submit_protocol(base, salt="a"))
    assert key(apply_submit_protocol(base, salt="a")) != key(apply_submit_protocol(base, salt="b"))
