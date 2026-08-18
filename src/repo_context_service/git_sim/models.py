from __future__ import annotations

from dataclasses import dataclass, field

from .templates import DEFAULT_ABBREV, DEFAULT_BRANCH


@dataclass
class StashEntry:
    content: dict[str, str | None] = field(default_factory=dict)
    created: dict[str, str | None] = field(default_factory=dict)
    subject: str = ""

    def paths(self) -> list[str]:
        return sorted(set(self.content) | set(self.created))


@dataclass
class GitState:
    index: dict[str, str | None] = field(default_factory=dict)
    staged_deleted: set[str] = field(default_factory=set)
    stash: list[StashEntry] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)
    unknown: bool = False
    branch: str | None = None
    detached: bool = False
    abbrev: int | None = None
    head_short: str | None = None
    head_subject: str | None = None
    history_dirty: bool = False

    poison_reason: str | None = None

    def record(self, turn: int, command: str, effect: dict) -> None:
        self.ledger.append({"turn": turn, "command": command, "effect": effect})

    def poison(self, reason: str, history: bool = False) -> None:
        self.unknown = True
        self.history_dirty = self.history_dirty or history
        if self.poison_reason is None:
            self.poison_reason = reason


@dataclass
class GitMeta:
    sha: str = ""
    owner: str = ""
    repo: str = ""
    abbrev: int = DEFAULT_ABBREV
    branch: str = DEFAULT_BRANCH
    detached: bool = False
    subject: str = ""
    history: object = None
    commit_patch: object = None

    @property
    def short(self) -> str:
        return self.sha[: self.abbrev]


@dataclass
class GitPlan:
    sub: str
    args: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    values: dict[str, str] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    revs: list[str] = field(default_factory=list)
    pipeline: list = field(default_factory=list)
    raw: str = ""
    dropped_stderr: bool = False
    evidence: bool = False
    redirect: str | None = None


@dataclass
class GitResult:
    output: str
    returncode: int = 0
    empty: bool = False
    exact: bool = True
    incomplete: list[str] = field(default_factory=list)
