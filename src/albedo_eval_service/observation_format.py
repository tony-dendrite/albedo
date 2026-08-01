"""Observation formats.

Each corpus we sample from writes environment observations in its own shape, and the renderer
(scripts/render_trajectories.py) copies tool output through verbatim — it only normalizes the
ACTION side into bash blocks. So a trajectory keeps its native observation format all the way into
eval, and the simulator has to speak that same format or every simulated turn looks foreign next to
the real ones above it.

Format cannot be derived from the source name: `open-swe-traces` merges four upstream arms, two
SWE-agent (`OBSERVATION:`) and two OpenHands (bare output + exit-code trailer), under one name. It
is read off the trajectory itself instead — the sampler never cuts at the first assistant turn, so
every sampled prefix carries at least one real observation to key on.
"""

from __future__ import annotations

RETURNCODE = "returncode"  # mini-swe-agent: <returncode>N</returncode> + <output> block
SWE_AGENT = "swe_agent"  # "OBSERVATION:\n..."
OPENHANDS = "openhands"  # bare tool output; bash calls add an exit-code trailer

FORMAT_MINI_CODER = """OUTPUT FORMAT:
- Your reply MUST have exactly this shape, with no text before or after:
<returncode>RC</returncode>
<output>
OUTPUT
</output>
  where RC is the command's exit code and OUTPUT is exactly the stdout/stderr it would produce
  (empty if the command prints nothing)."""

FORMAT_SWE_AGENT = """OUTPUT FORMAT:
- Your reply MUST begin with the literal string "OBSERVATION:" on its own line — no text may come
  before it.
- On the lines after it write exactly the stdout/stderr the command would produce — nothing else.
- If the command would produce no output, reply with exactly "OBSERVATION:" and nothing more."""

FORMAT_OPENHANDS = """OUTPUT FORMAT:
- Your reply is the tool result itself: NO "Observation:" prefix, no "OBSERVATION:" header, no
  <returncode> wrapper, no markdown code fence.
- For a shell command, write its stdout/stderr and then close with exactly these two lines:
[The command completed with exit code RC.]
[Command finished with exit code RC]
  where RC is the exit code. A command that prints nothing has an empty first line, then those two.
- For a file view (`cat -n`, `sed -n ... | cat -n`), open with
  "Here's the result of running `cat -n` on PATH:" and then the numbered lines, with no trailer.
- For a directory listing, open with "Here's the files and directories up to 2 levels deep in
  PATH, excluding hidden items:" and then the paths, with no trailer.
- For a file write, reply exactly "File created successfully at: PATH", with no trailer."""

_BLOCKS = {
    RETURNCODE: FORMAT_MINI_CODER,
    SWE_AGENT: FORMAT_SWE_AGENT,
    OPENHANDS: FORMAT_OPENHANDS,
}


def format_block(fmt: str) -> str:
    return _BLOCKS.get(fmt, FORMAT_OPENHANDS)


def classify(observation: str) -> str:
    """Format of one observation, by the marker it opens with (OpenHands has none)."""
    text = (observation or "").lstrip()
    if text.startswith("<returncode>"):
        return RETURNCODE
    if text.startswith("OBSERVATION:"):
        return SWE_AGENT
    return OPENHANDS


def detect_format(sample_id: str, messages: list[dict[str, str]] | None = None) -> str:
    """Read the format off the trajectory's own observations, falling back to the source name when
    the transcript carries none (turn-0 cuts, which the sampler does not produce, and unit tests).
    The fallback prefers OpenHands because its check is the permissive one: guessing it cannot
    reject an otherwise good observation."""
    observation = _first_observation(messages)
    if observation is not None:
        return classify(observation)
    return RETURNCODE if "mini-coder" in sample_id.casefold() else OPENHANDS


def _first_observation(messages: list[dict[str, str]] | None) -> str | None:
    """First environment turn: a user message that answers an assistant message (the leading user
    message is the task). The earliest one is always dataset-native, never simulated."""
    seen_assistant = False
    for message in messages or []:
        role = str(message.get("role") or "").lower()
        if role == "assistant":
            seen_assistant = True
        elif role == "user" and seen_assistant:
            content = str(message.get("content") or "")
            if content.strip():
                return content
    return None


def valid_output(raw: str, fmt: str) -> bool:
    text = (raw or "").strip()
    if not text:
        return False
    if fmt == RETURNCODE:
        return (
            text.startswith("<returncode>")
            and "</returncode>" in text
            and "<output>\n" in text
            and text.endswith("\n</output>")
        )
    if fmt == SWE_AGENT:
        return text.startswith("OBSERVATION:")
    # OpenHands has no marker of its own, so the check is negative: another format's header leaking
    # in means the model answered in the wrong dialect.
    return not text.startswith(("<returncode>", "Observation:", "OBSERVATION:"))


def repair_output(raw: str, fmt: str) -> str:
    """Fix the whitespace the returncode wrapper is picky about; other formats pass through."""
    text = (raw or "").strip()
    if fmt != RETURNCODE or not text.startswith("<returncode>"):
        return text
    if "<output>" in text and "<output>\n" not in text:
        text = text.replace("<output>", "<output>\n", 1)
    if text.endswith("</output>") and not text.endswith("\n</output>"):
        text = text[: -len("</output>")].rstrip("\n") + "\n</output>"
    return text


def wrap(body: str, fmt: str, *, returncode: int = 0) -> str:
    """Render an observation the harness injects itself (task submitted, no command found, empty
    output) in the trajectory's format."""
    if fmt == RETURNCODE:
        inner = f"{body}\n" if body else ""
        return f"<returncode>{returncode}</returncode>\n<output>\n{inner}</output>"
    if fmt == SWE_AGENT:
        return f"OBSERVATION:\n{body}" if body else "OBSERVATION:"
    return (
        f"{body}\n[The command completed with exit code {returncode}.]\n"
        f"[Command finished with exit code {returncode}]"
    )


def empty_output(fmt: str) -> str:
    return wrap("", fmt)
