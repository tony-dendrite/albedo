from __future__ import annotations

import re

# The literal warning the benchmark's pip prints on every install it runs as root — the whole
# observation for the dominant `pip install ... -q` shape (500 of 658 pip commands across 1100
# bench trajectories end exactly like this, exit code 0).
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

    Every bench image ships a working, networked pip (658 uses across 1100 real trajectories:
    zero "command not found", installs succeed, resolver errors only for requirements that
    genuinely cannot resolve). Two shapes are therefore always invented: pip itself denied, and
    an index resolver error naming a filesystem path — pip installs `.`/`-e .` straight from
    disk without consulting any index. A resolver error naming a package (even during a local
    `-e .` install, for one of its pinned dependencies) stays untouched: real pip does that.
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


def pip_success_body(command: str) -> str:
    """The canonical successful-install observation the bench prints for this command."""
    return "" if "2>/dev/null" in (command or "") else PIP_ROOT_WARNING
