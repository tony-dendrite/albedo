from __future__ import annotations

from .chain import chain_stages, git_evidence, run_git_chain
from .diffs import blob_hash
from .execute import run_git
from .models import GitMeta, GitPlan, GitResult, GitState, StashEntry
from .notes import explain_git, ledger_block
from .parse import _GIT_HEAD as _GIT_HEAD
from .parse import mutation_stages, parse_git
from .patches import learn_from_observed_diff
from .session import apply_git, learn_git_facts
from .templates import (
    BRANCH_HEADER,
    DEFAULT_ABBREV,
    DEFAULT_BRANCH,
    DETACHED_HEADER,
    GIT_EVIDENCE_HEADER,
    GIT_LEDGER_HEADER,
    HARNESS_SUBJECT,
)

__all__ = [
    "BRANCH_HEADER",
    "DEFAULT_ABBREV",
    "DEFAULT_BRANCH",
    "DETACHED_HEADER",
    "GIT_EVIDENCE_HEADER",
    "GIT_LEDGER_HEADER",
    "HARNESS_SUBJECT",
    "GitMeta",
    "GitPlan",
    "GitResult",
    "GitState",
    "StashEntry",
    "apply_git",
    "blob_hash",
    "chain_stages",
    "explain_git",
    "git_evidence",
    "learn_from_observed_diff",
    "learn_git_facts",
    "ledger_block",
    "mutation_stages",
    "parse_git",
    "run_git",
    "run_git_chain",
]
