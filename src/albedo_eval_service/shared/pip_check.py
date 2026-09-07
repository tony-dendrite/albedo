from __future__ import annotations

import re

# The literal warning the benchmark's pip prints on any install it runs as root. Kept as the
# recorded shape of a root install's output; an install in this environment cannot reach the
# package index, so `absent_tool_output` answers it with the connection failure instead.
PIP_ROOT_WARNING = (
    "WARNING: Running pip as the 'root' user can result in broken permissions and conflicting "
    "behaviour with the system package manager, possibly rendering your system unusable."
    "It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. "
    "Use the --root-user-action option if you know what you are doing and want to suppress "
    "this warning."
)

_PIP = r"(?:python3?\s+-m\s+)?pip3?"
_PIP_CMD_RE = re.compile(rf"\b{_PIP}\s+[a-z-]")
_PIP_INSTALL_RE = re.compile(rf"\b{_PIP}\s+install\b")
_PIP_MISSING_RE = re.compile(r"pip3?: command not found|No module named pip")
_RESOLVER_REQ_RE = re.compile(
    r"(?:Could not find a version that satisfies the requirement"
    r"|No matching distribution found for)\s+(\S+)"
)


def fabricated_pip_error(command: str, observation: str) -> bool:
    """An error the benchmark's pip cannot produce for this command.

    Every bench image ships pip itself — 658 uses across 1100 real trajectories, zero "command
    not found" — so pip denying its own existence is always invented. What the images lack is a
    reachable package index, which `absent_tool_output` answers directly for an install; the
    resolver check here therefore applies to the pip commands that do reach the simulator.

    A resolver error naming a filesystem path is invented in any case: pip installs `.`/`-e .`
    straight from disk without consulting an index. One naming a package stays untouched.
    """
    cmd, obs = command or "", observation or ""
    if not _PIP_CMD_RE.search(cmd):
        return False
    if _PIP_MISSING_RE.search(obs):
        return True
    if not _PIP_INSTALL_RE.search(cmd):
        return False
    return any(
        req == "." or req.startswith(("./", "../", "/", "-"))
        for req in (m.group(1) for m in _RESOLVER_REQ_RE.finditer(obs))
    )
