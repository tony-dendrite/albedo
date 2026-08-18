from __future__ import annotations

from ..command_search import ParseFailure
from .models import GitMeta, GitPlan, GitResult, GitState
from .parse import STDERR_ONLY_OUTPUT
from .render import _HANDLERS
from .templates import _HISTORY_ONLY
from .views import _Views


def run_git(
    plan: GitPlan, overlay, read_base, listing: list[str], meta: GitMeta | None = None
) -> GitResult | ParseFailure:
    meta = meta or GitMeta()
    handler = _HANDLERS.get(plan.sub)
    if handler is None:
        return ParseFailure("unsupported_form", f"git {plan.sub}")
    state = getattr(overlay, "git", None) or GitState()
    if plan.sub in _HISTORY_ONLY:
        if state.history_dirty:
            return ParseFailure("unsupported_form", "commit history diverged")
    elif state.unknown:
        return ParseFailure("unsupported_form", "git state diverged")
    views = _Views(overlay, state, read_base, listing, state.abbrev or meta.abbrev)
    result = handler(plan, views, meta)
    if isinstance(result, GitResult) and plan.dropped_stderr and plan.sub in STDERR_ONLY_OUTPUT:
        return GitResult(output="", returncode=result.returncode, empty=True)
    if isinstance(result, GitResult) and plan.redirect:
        return GitResult(output="", returncode=result.returncode, empty=True)
    return result
