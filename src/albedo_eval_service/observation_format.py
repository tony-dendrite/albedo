
from __future__ import annotations

RETURNCODE = "returncode"
SWE_AGENT = "swe_agent"
OPENHANDS = "openhands"

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
    text = (observation or "").lstrip()
    if text.startswith("<returncode>"):
        return RETURNCODE
    if text.startswith("OBSERVATION:"):
        return SWE_AGENT
    return OPENHANDS


def detect_format(sample_id: str, messages: list[dict[str, str]] | None = None) -> str:
    observation = _first_observation(messages)
    if observation is not None:
        return classify(observation)
    return RETURNCODE if "mini-coder" in sample_id.casefold() else OPENHANDS


def _first_observation(messages: list[dict[str, str]] | None) -> str | None:
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
    return not text.startswith(("<returncode>", "Observation:", "OBSERVATION:"))


def repair_output(raw: str, fmt: str) -> str:
    text = (raw or "").strip()
    if fmt != RETURNCODE or not text.startswith("<returncode>"):
        return text
    if "<output>" in text and "<output>\n" not in text:
        text = text.replace("<output>", "<output>\n", 1)
    if text.endswith("</output>") and not text.endswith("\n</output>"):
        text = text[: -len("</output>")].rstrip("\n") + "\n</output>"
    return text


def wrap(body: str, fmt: str, *, returncode: int = 0) -> str:
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
