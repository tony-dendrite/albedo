from __future__ import annotations

import posixpath
import re
import shlex
import subprocess

_SED_ERROR_RE = re.compile(r"^\s*sed:\s", re.MULTILINE)
_SHELL_SUBSTITUTION = re.compile(r"\$[({A-Za-z_]|`")
_SEPARATORS = frozenset({"&&", "||", ";", "|", "&"})
_SCRIPT_FLAG = re.compile(r"^-[a-zA-Z]*[ef]")
_CHECK_TIMEOUT_SECONDS = 2.0


def _segments(tokens: list[str]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    for token in tokens:
        out.append([]) if token in _SEPARATORS else out[-1].append(token)
    return out


def sed_scripts(command: str) -> list[str]:
    """Every sed script in `command`, or [] when it runs no plain sed."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    scripts: list[str] = []
    for segment in _segments(tokens):
        if not segment or posixpath.basename(segment[0]) != "sed":
            continue
        args = segment[1:]
        explicit = [
            args[i + 1] for i, a in enumerate(args) if _SCRIPT_FLAG.match(a) and i + 1 < len(args)
        ]
        if explicit:
            scripts += explicit
            continue
        operands = [a for a in args if not a.startswith("-")]
        if operands:
            scripts.append(operands[0])
    return scripts


def sed_accepts(script: str) -> bool:
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, script passed after `--`
            ["sed", "--sandbox", "-n", "--", script],
            input=b"",
            capture_output=True,
            timeout=_CHECK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def sed_error_message(command: str) -> str:
    scripts = sed_scripts(command)
    if not scripts or any(_SHELL_SUBSTITUTION.search(script) for script in scripts):
        return ""
    for script in scripts:
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, script passed after `--`
                ["sed", "--sandbox", "-n", "--", script],
                input=b"",
                capture_output=True,
                timeout=_CHECK_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if done.returncode:
            lines = done.stderr.decode("utf-8", "replace").strip().splitlines()
            return lines[0] if lines else ""
    return ""


def fabricated_sed_error(command: str, observation: str) -> bool:
    if not _SED_ERROR_RE.search(observation or ""):
        return False
    scripts = sed_scripts(command)
    return bool(scripts) and all(sed_accepts(script) for script in scripts)


def misdiagnosed_sed(command: str, observation: str) -> str:
    if not _SED_ERROR_RE.search(observation or ""):
        return ""
    real = sed_error_message(command)
    lines = (observation or "").splitlines()
    stated = next((line.strip() for line in lines if line.strip().startswith("sed:")), "")
    return real if real and real != stated else ""
