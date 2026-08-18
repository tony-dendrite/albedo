from __future__ import annotations

import re

BRANCH_HEADER = "On branch {branch}"
DETACHED_HEADER = "Not currently on any branch."
DEFAULT_BRANCH = "main"
STAGED_HEADER = "Changes to be committed:"
STAGED_HINT = '  (use "git restore --staged <file>..." to unstage)'
UNSTAGED_HEADER = "Changes not staged for commit:"
UNSTAGED_HINTS = (
    '  (use "git add <file>..." to update what will be committed)',
    '  (use "git restore <file>..." to discard changes in working directory)',
)
UNTRACKED_HEADER = "Untracked files:"
UNTRACKED_HINT = '  (use "git add <file>..." to include in what will be committed)'
CLEAN_TRAILER = "nothing to commit, working tree clean"
UNTRACKED_ONLY_TRAILER = (
    'nothing added to commit but untracked files present (use "git add" to track)'
)
UNSTAGED_TRAILER = 'no changes added to commit (use "git add" and/or "git commit -a")'
ENTRY_LABEL_WIDTH = 12
RESET_HEADER = "Unstaged changes after reset:"
HARD_RESET_LINE = "HEAD is now at {short} {subject}"
STASH_SAVED_LINE = "Saved working directory and index state WIP on {branch}: {short} {subject}"
STASH_EMPTY_LINE = "No local changes to save"
STASH_MISSING_LINE = "No stash entries found."
UPDATED_PATHS_LINE = "Updated {n} path{s} from the index"
DEFAULT_ABBREV = 7
DEFAULT_MODE = "100644"
NULL_BLOB = "0" * 40
NO_NEWLINE_MARKER = "\\ No newline at end of file"
FUNCNAME_MAX_CHARS = 80
EVIDENCE_LOG_LIMIT = 40

_MUTATING = {"add", "checkout", "restore", "reset", "stash", "rm", "mv", "apply", "commit"}
_GLOBAL_VALUE_FLAGS = {"-C", "-c", "--git-dir", "--work-tree", "--exec-path", "--namespace"}
_GLOBAL_BOOL_FLAGS = {
    "--no-pager",
    "--paginate",
    "--no-replace-objects",
    "--bare",
    "--literal-pathspecs",
}
_FUNCNAME = re.compile(r"^[A-Za-z$_]")


COMMIT_LINE = "commit {sha}"
AUTHOR_LINE = "Author: {author}"
DATE_LINE = "Date:   {date}"

HARNESS_SUBJECT = "SWE-bench"

GIT_EVIDENCE_HEADER = """GIT FACTS ALREADY COMPUTED — real repository data for parts of this
command, each shown under the invocation that produced it. A block with no caveat is that
invocation's exact output:
reproduce it unchanged. A block marked "computed WITHOUT this command's pipes and filters" is raw
material, NOT the answer — you must still apply the command's own greps, limits and filters to it,
and the result is usually far shorter than what is shown. Either way this is the only source of
truth for shas, subjects, authors and dates: never invent one that does not appear here:
"""

GIT_LEDGER_HEADER = """GIT COMMANDS ALREADY RUN — state-changing git commands executed earlier
in this session, in order, with what each one changed. The repository is in the state they
leave it in:
"""

UNCERTAIN_STATE_LINE = (
    "The working tree also changed in ways this prompt cannot reconstruct: derive the affected "
    "files from the transcript above, and never call a file unchanged on a guess."
)

_OPAQUE = {
    "commit",
    "apply",
    "am",
    "rm",
    "mv",
    "merge",
    "rebase",
    "revert",
    "cherry-pick",
    "clean",
    "pull",
}
_HISTORY_CHANGING = {"commit", "am", "merge", "rebase", "revert", "cherry-pick", "pull"}
_HISTORY_ONLY = {"log", "show", "rev-parse", "remote"}
_READ_ONLY = {
    "log",
    "show",
    "status",
    "diff",
    "branch",
    "tag",
    "ls-files",
    "ls-tree",
    "rev-parse",
    "blame",
    "cat-file",
    "describe",
    "shortlog",
    "remote",
    "config",
    "grep",
    "reflog",
    "merge-base",
    "rev-list",
    "whatchanged",
    "diff-tree",
    "cherry",
}
