from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx
from loguru import logger

from albedo_config import RepoContextSettings
from albedo_eval_service.remote.dataset import (
    _content,
    _extract_turns,
    _parse_sample_id,
    _read_parquet_row,
    _role,
    _unwrap_column,
)
from albedo_eval_service.shared.dataset_manifest import load_manifest_file
from albedo_eval_service.shared.observation_format import (
    OPENHANDS_TRUNCATION_NOTICE,
    detect_format,
)
from albedo_eval_service.shared.submit_protocol import ANY_MARKER_RE

from .command_search import (
    ParseFailure,
    SearchResult,
    _number_lines,
    parse_search,
    run_search,
    split_chain,
)
from .git_sim import (
    DEFAULT_BRANCH,
    GitMeta,
    explain_git,
    git_evidence,
    is_git_command,
    ledger_block,
    run_git_chain,
)
from .overlay import Overlay, attested_paths, build_overlay, session_root, strip_sandbox

_API_BASE = "https://api.github.com"
_DONE_MARKER = ".albedo-repo-context-done"
_LISTING_NAME = ".albedo-listing.json"
_REPO_META_NAME = ".albedo-repo-meta.json"
_HISTORY_NAME = ".albedo-git-log.{key}.json"
_HISTORY_PAGE = 100
_PATCH_NAME = ".albedo-commit.{key}.patch"
_MAX_PATCH_CHARS = 400_000
_NEGATIVE_TTL_SECONDS = 24 * 3600.0
_TRANSIENT_TTL_SECONDS = 900.0
_MAX_MEMBER_BYTES = 2 * 1024 * 1024
_MAX_MEMBERS = 200_000
_MAX_MISSING_PATHS = 10
_RETRIES = 3

_DETACHED_SOURCE = re.compile(r"re[-_]?bench", re.I)
_COMMAND_BLOCK_RE = re.compile(r"```(?:bash|sh)?[ \t]*\n(.*?)```", re.DOTALL)
_TAGGED_BLOCK_RE = re.compile(r"```(?:bash|sh)[ \t]*\n(.*?)```", re.DOTALL)

_TRUNCATION_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller number "
    "of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more "
    "selective search pattern.\n"
    "If you really need to see something from the full command's output, you can redirect "
    "output to a file and then search in that file."
)
_SCAFFOLD_LIMIT = {"returncode": 10_000, "openhands": 30_000}


def _scaffold_truncate(text: str, fmt: str) -> str:
    limit = _SCAFFOLD_LIMIT.get(fmt)
    raw = text + "\n"
    if limit is None or len(raw) <= limit:
        return text
    half = limit // 2
    if fmt == "openhands":
        return f"{raw[:half]}\n{OPENHANDS_TRUNCATION_NOTICE}\n{raw[-half:]}"
    return (
        f"<warning>\n{_TRUNCATION_WARNING}\n</warning><output_head>\n{raw[:half]}\n</output_head>\n"
        f"<elided_chars>\n{len(raw) - limit} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{raw[-half:]}\n</output_tail>"
    )


_RENDER_RULE = """- Print each path the way the command itself would print it: a command given an absolute search
  root (`find /`, `find {root}/x`, `ls {root}/x`, `realpath`) prints the absolute path, a command
  given a relative one (`find .`, `ls src`) prints it relative to the working directory, and `ls`
  of a single directory prints bare entry names. NEVER answer an absolutely-rooted search with a
  bare relative path — `find /` cannot return `pkg/mod.py`, only `{root}/pkg/mod.py`.
"""

LISTING_HEADER = """REPOSITORY FILE LISTING — tracked files at the current commit relevant to the
command (paths relative to the repo root, sorted; the filesystem returns files in this order).
The repo root is the directory this session starts in and has not left unless a command in the
transcript ran `cd`.
Derive the output of exploration commands (find, ls, grep -l, ...) EXACTLY from this list,
applying the command's filters and pipe limits:
- Output matching paths in EXACTLY the order they appear in this listing — never re-sort them.
- Print each path the way the command itself would print it: `find .` prefixes results with `./`,
  `find src` prefixes them with `src/`, `ls` of a single directory prints bare entry names. A
  search rooted at an absolute directory prints absolute paths — never a bare relative path.
- Do not invent paths that are not in this list and do not omit paths that match.
- This listing is complete for the scope shown: if nothing in it matches the command's
  filters, the command's output is empty.
"""

LISTING_HEADER_ROOTED = (
    """REPOSITORY FILE LISTING — tracked files at the current commit relevant to the command.
The repository is checked out at {root} and stays there for this entire session: it is never
moved, never renamed, and the same file is never served under a second root. The paths below are
absolute, exactly as this filesystem holds them, sorted (the filesystem returns files in this
order).
Derive the output of exploration commands (find, ls, grep -l, ...) EXACTLY from this list,
applying the command's filters and pipe limits:
- Output matching paths in EXACTLY the order they appear in this listing — never re-sort them.
"""
    + _RENDER_RULE
    + """- Do not invent paths that are not in this list and do not omit paths that match.
- This listing is complete for the scope shown: if nothing in it matches the command's
  filters, the command's output is empty.
"""
)

SESSION_PATHS_HEADER = """PATHS ALREADY ESTABLISHED — earlier observations in this session showed each path below
exists. They are outside the tracked listing because the listing covers only what the repository
tracks at this commit, while the machine also holds harness scripts, build output, vendored
dependencies, downloads and files this session created. Being absent from the listing is NOT
evidence of absence: treat every path below as present, never report it missing, and derive its
content or entries from what the earlier observation showed:
"""

CONTENTS_HEADER = """FILE CONTENTS — exact current content of files referenced by the command:
"""

LINE_NUMBER_NOTE_GREP = """Each content line below is prefixed with its line number as "N:", exactly the form grep -n
emits. The prefix is an annotation, not part of the file's text: reproduce it for lines the
command matches, and strip it anywhere the command does not report line numbers.
"""

LINE_NUMBER_NOTE_COLUMN = """Each content line below is prefixed with its line number, right-aligned then a tab, exactly
the form cat -n and nl emit. The prefix is an annotation, not part of the file's text:
reproduce it for lines the command prints, and strip it anywhere line numbers are not reported.
"""

NOT_PRESENT_HEADER = """FILES NOT PRESENT — ONLY the paths listed immediately below are absent
from this machine. Every other path the command names does exist; never claim otherwise, and in
particular never infer absence from a path being missing from the listing above. Do not invent
content for a listed path — report the terminal error shown beside it, inside the observation,
following the OUTPUT FORMAT exactly. When the command produces no output at all, reply with
exactly the empty observation the OUTPUT FORMAT specifies:
"""

COMPUTED_HEADER = """COMMAND OUTPUT — this search was executed against the repository at this commit, so
the text below is the command's exact output. Reply with it verbatim inside the OUTPUT FORMAT.
Do not add, reorder, re-sort or omit lines, and do not explain it:
"""

COMPUTED_EMPTY = """COMMAND OUTPUT — this search was already executed against the repository at this
commit and matched NOTHING: zero lines of output. That is the verified result, not a gap in what
you were given — the repository was searched in full and no file matched. Do NOT list any path, do
NOT guess at plausible matches, and do NOT reason about which files "should" have matched. An empty
result is a normal outcome for a search. Reply with exactly the empty observation the OUTPUT FORMAT
specifies:
"""

COMPUTED_GIT_HEADER = """COMMAND OUTPUT — this git command was executed against the repository at this commit
with the working-tree changes made earlier in this session applied, so the text below is the
command's exact output. Reply with it verbatim inside the OUTPUT FORMAT. Do not add, reorder or
omit lines, and do not explain it:
"""

COMPUTED_GIT_EMPTY = """COMMAND OUTPUT — this git command was executed against the repository at this commit
with the working-tree changes made earlier in this session applied, and it printed NOTHING: zero
lines of output. That is the verified result, not a gap in what you were given. Reply with exactly
the empty observation the OUTPUT FORMAT specifies:
"""

GIT_SEMANTICS_HEADER = """GIT SEMANTICS — how this git command behaves in this checkout; both lines are facts about
the repository state you are simulating, not guidance to repeat in the output:
"""

CHAIN_EVIDENCE_HEADER = """CHAIN STAGES ALREADY EXECUTED — this command joins several stages with &&. The stages
below were executed against the repository at this commit, so the text under each one is that
stage's exact output. Reproduce those parts unchanged in your reply and derive the remaining
stages yourself; the stages shown are not the whole observation. Each stage was executed on its
own, so a stage shown here may follow one whose success could not be checked; keep that in mind
if an earlier stage would have stopped the chain:
"""

PRE_EDIT_SUFFIX = (
    "   (as of this commit — the session has since edited this file in a way that could "
    "not be reproduced, so the text below is the version BEFORE those edits)"
)

COMPUTED_PARTIAL = """NOTE — the following files are listed in the repository but their contents are not
available at this commit, so the output above may be missing matches from them:
"""

TRAJECTORY_HEADER = """REFERENCE EXCHANGES — real command -> observation exchanges recorded in this
repository during a reference session on the same task. Use them ONLY to derive real paths, file
contents and output style; do not invent paths or contents they contradict, and never copy task
hints or solution steps into your output:
"""


@dataclass(frozen=True)
class RepoRef:
    instance_id: str
    source: str
    owner: str
    repo: str
    pr: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class GroundingContext:
    context: str | None
    kind: str
    reason: str | None = None
    exact_output: str | None = None
    exact_returncode: int | None = None
    state: str = ""


class _NotFound(Exception):
    pass


class _SnapshotTooLarge(Exception):
    pass


def _is_permanent_github_error(exc: Exception) -> bool:
    if isinstance(exc, _NotFound):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (400, 410, 422, 451)
    return False


def parse_instance(source: str, instance_id: str) -> RepoRef | None:
    try:
        if source.startswith("mini-coder"):
            parts = instance_id.split("__")
            if len(parts) < 2:
                return None
            owner, rest = parts[0], parts[1]
            tokens = rest.split(".")
            for index in range(1, len(tokens)):
                if re.fullmatch(r"[0-9a-f]{6,40}", tokens[index]):
                    return RepoRef(
                        instance_id=instance_id,
                        source=source,
                        owner=owner,
                        repo=".".join(tokens[:index]),
                        commit=tokens[index],
                    )
            return None
        owner_repo, tail = instance_id.rsplit("-", 1)
        owner, repo = owner_repo.split("__", 1)
        if tail.isdigit():
            return RepoRef(instance_id=instance_id, source=source, owner=owner, repo=repo, pr=tail)
        if re.fullmatch(r"[0-9a-f]{40}", tail):
            return RepoRef(
                instance_id=instance_id, source=source, owner=owner, repo=repo, commit=tail
            )
        return None
    except ValueError:
        return None


def _first_command(text: str) -> str:
    match = _TAGGED_BLOCK_RE.search(text or "") or _COMMAND_BLOCK_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _name_patterns(cmd: str) -> list[str]:
    return (
        re.findall(r"-name\s+\"([^\"]+)\"", cmd)
        + re.findall(r"-name\s+'([^']+)'", cmd)
        + re.findall(
            r"-name\s+([^\s\"';|]+)",
            cmd.replace('-name "', '-nameQ"').replace("-name '", "-nameQ'"),
        )
    )


def _filter_listing(paths: list[str], cmd: str) -> tuple[list[str], bool]:
    pats = _name_patterns(cmd)
    if not (pats and cmd.startswith(("find", "ls"))):
        return paths, False
    regexes = [
        re.compile(re.escape(p).replace(r"\*", ".*").replace(r"\?", ".") + "$") for p in pats
    ]
    kept = [p for p in paths if any(r.match(p.rsplit("/", 1)[-1]) or r.match(p) for r in regexes)]
    return kept, True


@lru_cache(maxsize=16)
def _suffix_index(listing: tuple[str, ...]) -> dict[str, str]:
    owners: dict[str, str | None] = {}
    for path in listing:
        segments = path.split("/")
        for index in range(len(segments)):
            suffix = "/".join(segments[index:])
            if suffix in owners:
                if owners[suffix] != path:
                    owners[suffix] = None
            else:
                owners[suffix] = path
    return {suffix: path for suffix, path in owners.items() if path is not None}


def _resolve_path(token: str, listing_set: set[str], index: dict[str, str]) -> str | None:
    if token in listing_set:
        return token
    segments = token.lstrip("/").split("/")
    for start in range(len(segments)):
        resolved = index.get("/".join(segments[start:]))
        if resolved is not None:
            return resolved
    return None


_WRITE_TARGET = re.compile(r">>?\s*[^\s|;&()\"'<>]+")
_WRITE_DEST = re.compile(r"\b(?:cp|mv|install)\s+(?:-\S+\s+)*\S+\s+(\S+)|\btee\s+(?:-\S+\s+)*(\S+)")
_CD_ONLY = re.compile(r"^cd\s+\S+$")


def _referenced_paths(cmd: str, listing: list[str]) -> tuple[list[str], list[str]]:
    listing_set = set(listing)
    top_dirs = {p.split("/", 1)[0] for p in listing_set if "/" in p}
    index = _suffix_index(tuple(listing))
    present: list[str] = []
    missing: list[str] = []
    # a copy/move/tee destination does not exist yet, which is normal rather than an error
    dests = {m.group(1) or m.group(2) for m in _WRITE_DEST.finditer(cmd)}
    for tok in re.split(r"[\s|;&<>()\"']+", _WRITE_TARGET.sub(" ", cmd)):
        if not tok or tok.startswith("-"):
            continue
        p = tok[2:] if tok.startswith("./") else tok
        if tok.startswith("/"):
            p = strip_sandbox(tok)
        if "://" not in tok and not any(ch in tok for ch in "*?[]{}$`="):
            resolved = _resolve_path(p, listing_set, index)
            if resolved is not None:
                if resolved not in present:
                    present.append(resolved)
                continue
        if "://" in tok or any(ch in tok for ch in "*?[]{}$`="):
            continue
        norm = p.rstrip("/")
        if not norm or norm in (".", "..") or norm in missing or tok in dests:
            continue
        if any(x.startswith(norm + "/") for x in listing_set):
            continue
        if tok.startswith(("/", "..")):
            missing.append(norm)
        elif _plausible_repo_path(norm, top_dirs):
            missing.append(norm)
    return present, missing[:_MAX_MISSING_PATHS]


def _split_attested(missing: list[str], attested: set[str] | None) -> tuple[list[str], list[str]]:
    """Peel the paths the transcript already vouched for out of the not-present list.

    `missing` holds whatever the command names that the tracked listing does not, which is a
    strictly narrower question than whether the path is on the machine: the harness scripts,
    build output, vendored dependencies and downloads a trajectory was recorded against are all
    real and all untracked. Anything an earlier observation showed to exist moves to the vouched
    list instead, so the block states it is present rather than staying silent about it.
    """
    if not attested:
        return missing, []
    known = {strip_sandbox(path): path for path in attested}
    kept: list[str] = []
    vouched: list[str] = []
    for path in missing:
        hit = known.get(path) or known.get(strip_sandbox(path))
        if hit is None:
            kept.append(path)
        elif hit not in vouched:
            vouched.append(hit)
    return kept, vouched


_SEPARATOR = r"(?:^|\||&&|\|\||;)\s*"
_WANTS_LINE_NUMBERS = re.compile(
    _SEPARATOR + r"(?:grep|rg)\b[^|&;]*?\s-\w*n\w*\b"
    r"|" + _SEPARATOR + r"cat\b[^|&;]*?\s-\w*[nb]\w*\b"
    r"|" + _SEPARATOR + r"nl\b"
)


_GREP_STYLE = re.compile(_SEPARATOR + r"(?:grep|rg)\b[^|&;]*?\s-\w*n\w*\b")


def _plausible_repo_path(path: str, top_dirs: set[str]) -> bool:
    segments = path.split("/")
    if segments[0] in top_dirs:
        return True
    if len(segments) == 1 and path.startswith("."):
        return True
    last = segments[-1]
    return bool(re.fullmatch(r"\.?[\w@.-]+\.[A-Za-z][A-Za-z0-9]*", last))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


@lru_cache(maxsize=64)
def _load_listing(listing_path: str) -> tuple[str, ...]:
    return tuple(json.loads(Path(listing_path).read_text()))


@lru_cache(maxsize=4096)
def _iid_from_parquet(dataset_root: str, shard_name: str, row_idx: int) -> str | None:
    import pyarrow.parquet as pq

    path = Path(dataset_root) / shard_name
    try:
        seen = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["instance_id"]):
            if seen + batch.num_rows <= row_idx:
                seen += batch.num_rows
                continue
            value = batch.column("instance_id")[row_idx - seen].as_py()
            return str(value) if value else None
    except Exception:
        return None
    return None


class RepoContextService:
    def __init__(self, settings: RepoContextSettings):
        if not settings.cache_dir:
            raise ValueError(
                "ALBEDO_REPO_CONTEXT_CACHE_DIR is required: the snapshot download directory "
                "must be configured explicitly"
            )
        self.settings = settings
        self.cache_dir = Path(settings.cache_dir).expanduser()
        self._shas_dir = self.cache_dir / "shas"
        self._snapshots_dir = self.cache_dir / "snapshots"
        self._client = httpx.Client(
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
            headers={"User-Agent": "albedo-repo-context", "Accept": "application/vnd.github+json"},
        )
        self._github_semaphore = threading.BoundedSemaphore(8)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._shards: dict[str, tuple[str, list]] | None = None
        self._manifest_error_logged = False
        if not self._auth_headers():
            logger.warning(
                "repo_context_no_github_token: running UNAUTHENTICATED (60 req/hr) — SHA "
                "resolution will rate-limit and PR/commit-based datasets (open-swe-traces, "
                "swe-hero) will fail to ground. Set ALBEDO_REPO_CONTEXT_GITHUB_TOKEN."
            )

    def close(self) -> None:
        self._client.close()

    def context_for(
        self, sample_id: str, assistant_output: str, messages: list[dict[str, str]] | None = None
    ) -> GroundingContext:
        try:
            return self._context_for(sample_id, assistant_output, messages)
        except Exception as exc:
            logger.warning(
                "repo_context_fallback sample_id={} kind=none reason=unexpected error={}",
                sample_id,
                f"{type(exc).__name__}: {exc}",
            )
            return GroundingContext(context=None, kind="none", reason="unexpected")

    def prefetch(self, sample_ids: list[str]) -> dict[str, int]:
        instances: dict[str, str] = {}
        for sample_id in sample_ids:
            try:
                shard_name, row_idx, _ = _parse_sample_id(sample_id)
            except ValueError:
                continue
            source_iid = self._iid_for(shard_name, row_idx)
            if source_iid is not None:
                instances[source_iid[1]] = source_iid[0]

        def warm(instance_id: str, source: str) -> bool:
            try:
                ref = parse_instance(source, instance_id)
                if ref is None:
                    return False
                resolved = self._resolve_sha(ref)
                if resolved is None:
                    return False
                return self._ensure_snapshot(*resolved) is not None
            except Exception as exc:
                logger.warning(
                    "repo_context_prefetch_instance_failed instance={} error={}",
                    instance_id,
                    f"{type(exc).__name__}: {exc}",
                )
                return False

        ready = 0
        if instances:
            with ThreadPoolExecutor(max_workers=8) as pool:
                ready = sum(pool.map(lambda item: warm(*item), instances.items()))
        summary = {"samples": len(sample_ids), "instances": len(instances), "ready": ready}
        logger.info(
            "repo_context_prefetch_done samples={} instances={} ready={}",
            summary["samples"],
            summary["instances"],
            summary["ready"],
        )
        return summary

    def repo_context_for_instance(
        self,
        source: str,
        instance_id: str,
        assistant_output: str,
        messages: list[dict[str, str]] | None = None,
        fmt: str = "",
    ) -> GroundingContext:
        ref = parse_instance(source, instance_id)
        if ref is None:
            return GroundingContext(context=None, kind="none", reason="instance_unparsed")
        resolved = self._resolve_sha(ref)
        if resolved is None:
            return GroundingContext(context=None, kind="none", reason="sha_unresolved")
        owner, repo, sha = resolved
        snapshot = self._ensure_snapshot(owner, repo, sha)
        if snapshot is None:
            return GroundingContext(context=None, kind="none", reason="snapshot_unavailable")
        base = list(_load_listing(str(snapshot / _LISTING_NAME)))
        overlay = build_overlay(
            messages,
            base,
            _suffix_index(tuple(base)),
            lambda rel: self._read_snapshot_file(snapshot, rel),
        )
        listing = overlay.listing(base)
        command = _first_command(assistant_output)
        block, exact, returncode = self._build_repo_block(
            snapshot,
            listing,
            command,
            overlay,
            fmt,
            self.git_meta(snapshot, source, owner, repo, sha),
            root=session_root(messages),
            attested=attested_paths(messages),
        )
        present, missing = _referenced_paths(command, listing)
        return GroundingContext(
            context=block,
            kind="repo",
            exact_output=exact,
            exact_returncode=returncode,
            state=overlay.state(block, set(present) | set(missing)),
        )

    def _context_for(
        self, sample_id: str, assistant_output: str, messages: list[dict[str, str]] | None = None
    ) -> GroundingContext:
        try:
            shard_name, row_idx, turn_idx = _parse_sample_id(sample_id)
        except ValueError:
            return GroundingContext(context=None, kind="none", reason="bad_sample_id")
        reason = "iid_unresolved"
        source_iid = self._iid_for(shard_name, row_idx)
        if source_iid is not None:
            result = self.repo_context_for_instance(
                *source_iid, assistant_output, messages, detect_format(sample_id, messages)
            )
            if result.context is not None:
                return result
            reason = result.reason or "repo_unavailable"
        block = self._trajectory_block(shard_name, row_idx, turn_idx)
        if block is not None:
            logger.info(
                "repo_context_fallback sample_id={} kind=trajectory reason={}", sample_id, reason
            )
            return GroundingContext(
                context=block,
                kind="trajectory",
                reason=reason,
                state=hashlib.sha1(block.encode("utf-8", "replace")).hexdigest(),
            )
        logger.warning("repo_context_fallback sample_id={} kind=none reason={}", sample_id, reason)
        return GroundingContext(context=None, kind="none", reason=reason)

    def _iid_for(self, shard_name: str, row_idx: int) -> tuple[str, str] | None:
        if self.settings.dataset_manifest_path:
            resolved = self._iid_from_manifest(shard_name, row_idx)
            if resolved is not None:
                return resolved
        if self.settings.dataset_root:
            iid = _iid_from_parquet(self.settings.dataset_root, shard_name, row_idx)
            if iid:
                return (shard_name.split("/data/", 1)[0], iid)
        return None

    def _iid_from_manifest(self, shard_name: str, row_idx: int) -> tuple[str, str] | None:
        try:
            shards = self._manifest_shards()
        except Exception as exc:
            if not self._manifest_error_logged:
                self._manifest_error_logged = True
                logger.warning(
                    "repo_context_manifest_unavailable path={} error={}",
                    self.settings.dataset_manifest_path,
                    f"{type(exc).__name__}: {exc}",
                )
            return None
        entry = shards.get(shard_name)
        if entry is None:
            return None
        source, rows_meta = entry
        if not 0 <= row_idx < len(rows_meta):
            return None
        iid = rows_meta[row_idx].get("iid") if isinstance(rows_meta[row_idx], dict) else None
        return (source, str(iid)) if iid else None

    def _manifest_shards(self) -> dict[str, tuple[str, list]]:
        if self._shards is None:
            with self._manifest_lock:
                if self._shards is None:
                    manifest = load_manifest_file(
                        self.settings.dataset_manifest_path,
                        expected_sha256=self.settings.dataset_manifest_hash,
                    )
                    shards: dict[str, tuple[str, list]] = {}
                    for source in manifest.get("sources", []):
                        for shard in source.get("shards", []):
                            shards[shard["path"]] = (
                                str(source.get("name", "")),
                                shard.get("rows_meta") or [],
                            )
                    self._shards = shards
        return self._shards

    def _resolve_sha(self, ref: RepoRef) -> tuple[str, str, str] | None:
        cache_path = self._shas_dir / f"{_safe_name(ref.instance_id)}.json"
        cached = _read_json(cache_path)
        if cached is not None:
            if cached.get("sha"):
                return cached["owner"], cached["repo"], cached["sha"]
            if self._negative_fresh(cached):
                return None
        with self._key_lock(f"sha:{ref.instance_id}"):
            cached = _read_json(cache_path)
            if cached is not None and cached.get("sha"):
                return cached["owner"], cached["repo"], cached["sha"]
            owner, repo = ref.owner, ref.repo
            try:
                try:
                    sha = self._sha_from_api(owner, repo, ref)
                except _NotFound:
                    data = self._github_json(f"/repos/{owner}/{repo}")
                    owner, repo = data["full_name"].split("/", 1)
                    sha = self._sha_from_api(owner, repo, ref)
            except Exception as exc:
                permanent = _is_permanent_github_error(exc)
                logger.info(
                    "repo_context_sha_unresolved instance={} kind={} error={}",
                    ref.instance_id,
                    "permanent" if permanent else "transient",
                    f"{type(exc).__name__}: {exc}",
                )
                _write_json_atomic(
                    cache_path,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": time.time(),
                        "kind": "permanent" if permanent else "transient",
                    },
                )
                return None
            _write_json_atomic(cache_path, {"owner": owner, "repo": repo, "sha": sha})
            return owner, repo, sha

    @staticmethod
    def _negative_fresh(cached: dict) -> bool:
        ttl = _NEGATIVE_TTL_SECONDS if cached.get("kind") == "permanent" else _TRANSIENT_TTL_SECONDS
        return time.time() - float(cached.get("failed_at", 0)) < ttl

    def git_meta(self, snapshot: Path, source: str, owner: str, repo: str, sha: str) -> GitMeta:
        return GitMeta(
            sha=sha,
            owner=owner,
            repo=repo,
            branch=self._default_branch(snapshot, owner, repo),
            detached=_DETACHED_SOURCE.search(source or "") is not None,
            history=lambda path: self._commit_history(snapshot, owner, repo, sha, path),
            commit_patch=lambda rev: self._commit_patch(snapshot, owner, repo, rev),
        )

    def _commit_patch(self, snapshot: Path, owner: str, repo: str, rev: str) -> str | None:
        cache_path = snapshot / _PATCH_NAME.format(key=_hashed(rev))
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8", errors="replace")
            return text or None
        try:
            text = self._github_text(
                f"/repos/{owner}/{repo}/commits/{quote(rev, safe='')}",
                "application/vnd.github.patch",
            )
        except Exception as exc:
            logger.info(
                "repo_context_patch_unavailable repo={}/{} rev={} error={}",
                owner,
                repo,
                rev,
                f"{type(exc).__name__}: {exc}",
            )
            return None
        if len(text) > _MAX_PATCH_CHARS:
            return None
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, cache_path)
        return text or None

    def _commit_history(
        self, snapshot: Path, owner: str, repo: str, sha: str, path: str | None
    ) -> dict | None:
        key = _safe_name(path) if path else "__repo__"
        cache_path = snapshot / _HISTORY_NAME.format(key=_hashed(key))
        cached = _read_json(cache_path)
        if isinstance(cached, dict):
            return cached if cached.get("commits") is not None else None
        query = f"/repos/{owner}/{repo}/commits?sha={sha}&per_page={_HISTORY_PAGE}"
        if path:
            query += f"&path={quote(path, safe='')}"
        try:
            data = self._github_json(query)
        except Exception as exc:
            logger.info(
                "repo_context_history_unavailable repo={}/{} path={} error={}",
                owner,
                repo,
                path or "-",
                f"{type(exc).__name__}: {exc}",
            )
            return None
        if not isinstance(data, list):
            return None
        payload = {
            "commits": [
                {
                    "sha": str(entry.get("sha") or ""),
                    "subject": str((entry.get("commit") or {}).get("message") or "").split("\n")[0],
                }
                for entry in data
                if entry.get("sha")
            ],
            "complete": len(data) < _HISTORY_PAGE,
        }
        _write_json_atomic(cache_path, payload)
        return payload

    def _default_branch(self, snapshot: Path, owner: str, repo: str) -> str:
        meta_path = snapshot / _REPO_META_NAME
        cached = _read_json(meta_path)
        if isinstance(cached, dict) and cached.get("default_branch"):
            return str(cached["default_branch"])
        branch = DEFAULT_BRANCH
        try:
            data = self._github_json(f"/repos/{owner}/{repo}")
            branch = str(data.get("default_branch") or DEFAULT_BRANCH)
        except Exception as exc:
            logger.info(
                "repo_context_default_branch_unavailable repo={}/{} error={}",
                owner,
                repo,
                f"{type(exc).__name__}: {exc}",
            )
            return branch
        _write_json_atomic(meta_path, {"default_branch": branch})
        return branch

    def _sha_from_api(self, owner: str, repo: str, ref: RepoRef) -> str:
        if ref.pr is not None:
            data = self._github_json(f"/repos/{owner}/{repo}/pulls/{ref.pr}")
            return data["base"]["sha"]
        data = self._github_json(f"/repos/{owner}/{repo}/commits/{ref.commit}")
        return data["sha"]

    def _ensure_snapshot(self, owner: str, repo: str, sha: str) -> Path | None:
        key = f"{_safe_name(owner)}__{_safe_name(repo)}__{sha[:12]}"
        final = self._snapshots_dir / key
        if (final / _DONE_MARKER).exists():
            return final
        failed_path = self._snapshots_dir / f"{key}.failed.json"
        if self._failure_fresh(failed_path):
            return None
        with self._key_lock(f"snapshot:{key}"):
            if (final / _DONE_MARKER).exists():
                return final
            if self._failure_fresh(failed_path):
                return None
            self._enforce_cache_limit()
            self._snapshots_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(dir=self._snapshots_dir, prefix=f".{key}.partial-"))
            try:
                with tempfile.NamedTemporaryFile(dir=self._snapshots_dir, suffix=".tar.gz") as tar:
                    self._download_tarball(owner, repo, sha, Path(tar.name))
                    listing, extracted_bytes = self._extract_tarball(Path(tar.name), tmp_dir)
                (tmp_dir / _LISTING_NAME).write_text(json.dumps(listing))
                (tmp_dir / _DONE_MARKER).write_text(json.dumps({"bytes": extracted_bytes}))
                os.replace(tmp_dir, final)
            except Exception as exc:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                if (final / _DONE_MARKER).exists():
                    return final
                ttl_kind = "oversized" if isinstance(exc, _SnapshotTooLarge) else "transient"
                logger.info(
                    "repo_context_snapshot_failed repo={}/{} sha={} kind={} error={}",
                    owner,
                    repo,
                    sha[:12],
                    ttl_kind,
                    f"{type(exc).__name__}: {exc}",
                )
                _write_json_atomic(
                    failed_path,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": time.time(),
                        "kind": ttl_kind,
                    },
                )
                return None
            return final

    @staticmethod
    def _failure_fresh(failed_path: Path) -> bool:
        failed = _read_json(failed_path)
        if failed is None:
            return False
        ttl = _NEGATIVE_TTL_SECONDS if failed.get("kind") == "oversized" else _TRANSIENT_TTL_SECONDS
        return time.time() - float(failed.get("failed_at", 0)) < ttl

    def _download_tarball(self, owner: str, repo: str, sha: str, dest: Path) -> None:
        url = f"{_API_BASE}/repos/{owner}/{repo}/tarball/{sha}"
        max_bytes = self.settings.max_snapshot_mb * 1024 * 1024
        with self._github_semaphore:
            with self._client.stream("GET", url, headers=self._auth_headers()) as response:
                if response.status_code == 404:
                    raise _NotFound(url)
                response.raise_for_status()
                declared = int(response.headers.get("content-length") or 0)
                if declared > max_bytes:
                    raise _SnapshotTooLarge(f"tarball {declared} bytes > {max_bytes}")
                written = 0
                with dest.open("wb") as out:
                    for chunk in response.iter_bytes(1 << 20):
                        written += len(chunk)
                        if written > max_bytes:
                            raise _SnapshotTooLarge(f"tarball exceeds {max_bytes} bytes")
                        out.write(chunk)

    def _enforce_cache_limit(self) -> None:
        limit = int(self.settings.max_cache_gb * 1024**3)
        if limit <= 0:
            return
        usage = 0
        for marker in self._snapshots_dir.glob(f"*/{_DONE_MARKER}"):
            data = _read_json(marker)
            usage += int(data.get("bytes", 0)) if data else 0
        if usage < limit:
            return
        logger.warning(
            "repo_context_cache_cleared usage_bytes={} limit_bytes={} dir={}",
            usage,
            limit,
            self._snapshots_dir,
        )
        shutil.rmtree(self._snapshots_dir, ignore_errors=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        _load_listing.cache_clear()

    def _extract_tarball(self, tar_path: Path, dest_dir: Path) -> tuple[list[str], int]:
        max_extracted = self.settings.max_snapshot_mb * 1024 * 1024 * 4
        total = 0
        paths: list[str] = []
        with tarfile.open(tar_path, mode="r:gz") as tar:
            for member in tar:
                if len(paths) >= _MAX_MEMBERS:
                    raise _SnapshotTooLarge(f"tarball has more than {_MAX_MEMBERS} files")
                if not member.isreg() or member.name.startswith("/"):
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2 or any(part in ("..", "") for part in parts):
                    continue
                if parts[1] in (_LISTING_NAME, _DONE_MARKER):
                    continue
                if member.size > _MAX_MEMBER_BYTES:
                    paths.append("/".join(parts[1:]))
                    continue
                total += member.size
                if total > max_extracted:
                    raise _SnapshotTooLarge(f"extracted size exceeds {max_extracted} bytes")
                rel = "/".join(parts[1:])
                target = dest_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                paths.append(rel)
        return sorted(paths), total

    _LISTING_MIN_CHARS = 8000

    def _run_command(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        attested: set[str] | None = None,
        root: str = "",
    ) -> SearchResult | None:
        plan = parse_search(cmd)
        if isinstance(plan, ParseFailure):
            return None
        result = run_search(
            plan,
            lambda rel: self._file_text(snapshot_dir, rel, overlay),
            listing,
            size_file=lambda rel: self._file_size(snapshot_dir, rel, overlay),
            root=root,
        )
        if isinstance(result, ParseFailure):
            return None
        if result.missing and _split_attested(list(result.missing), attested)[1]:
            # the search ran against the tracked commit, which does not know about harness
            # scripts or build output: a "No such file" it derives for a path the transcript
            # already showed is a fabrication, so decline and let the grounded block answer
            return None
        return result

    def _chain_evidence(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        attested: set[str] | None = None,
        root: str = "",
    ) -> str:
        stages = split_chain(cmd)
        if stages is None:
            return ""
        fragments: list[str] = []
        for stage in stages:
            if _CD_ONLY.match(stage):
                continue
            result = self._run_command(snapshot_dir, listing, stage, overlay, attested, root)
            if result is None:
                continue
            if result.output:
                fragments.append(
                    f"$ {stage}\n{_truncate(result.output, self.settings.max_file_chars)}"
                )
        if not fragments:
            return ""
        return "\n" + CHAIN_EVIDENCE_HEADER + "\n" + "\n\n".join(fragments) + "\n"

    def _run_git_command(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        meta: GitMeta,
    ):
        result = run_git_chain(
            cmd,
            overlay,
            lambda rel: self._read_snapshot_file(snapshot_dir, rel),
            listing,
            meta,
            file_text=lambda working, rel: self._file_text(snapshot_dir, rel, working),
            file_size=lambda working, rel: self._file_size(snapshot_dir, rel, working),
        )
        if isinstance(result, ParseFailure) or not result.exact:
            return None
        return result

    def _computed_git_block(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        fmt: str,
        meta: GitMeta,
    ) -> tuple[str, str | None, int | None] | None:
        result = self._run_git_command(snapshot_dir, listing, cmd, overlay, meta)
        if result is None:
            return None
        if result.empty:
            block, exact = COMPUTED_GIT_EMPTY, ""
        else:
            exact = _scaffold_truncate(result.output, fmt)
            block = COMPUTED_GIT_HEADER + "\n" + exact + "\n"
        return _truncate(block, self.settings.max_context_chars), exact, result.returncode

    def _computed_search_block(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        fmt: str = "",
        attested: set[str] | None = None,
        root: str = "",
    ) -> tuple[str, str | None] | None:
        result = self._run_command(snapshot_dir, listing, cmd, overlay, attested, root)
        if result is None:
            return None
        if result.empty and result.incomplete:
            # nothing matched only because nothing could be read: claiming a verified
            # empty here would assert the opposite of what the files actually hold
            return None

        if result.empty:
            block, exact = COMPUTED_EMPTY, ""
        else:
            exact = _scaffold_truncate(result.output, fmt)
            block = COMPUTED_HEADER + "\n" + exact + "\n"
        if result.incomplete:
            block += (
                "\n"
                + COMPUTED_PARTIAL
                + "\n".join(f"- {p}" for p in result.incomplete[:_MAX_MISSING_PATHS])
                + "\n"
            )
            exact = None
        return _truncate(block, self.settings.max_context_chars), exact

    def _build_repo_block(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        fmt: str = "",
        meta: GitMeta | None = None,
        root: str = "",
        attested: set[str] | None = None,
    ) -> tuple[str, str | None, int | None]:
        computed = self._computed_search_block(
            snapshot_dir, listing, cmd, overlay, fmt, attested, root
        )
        if computed is not None:
            return computed[0], computed[1], None
        meta = meta or GitMeta()
        computed_git = self._computed_git_block(snapshot_dir, listing, cmd, overlay, fmt, meta)
        if computed_git is not None:
            return computed_git
        evidence = self._chain_evidence(
            snapshot_dir, listing, cmd, overlay, attested, root
        ) + self._git_hints(snapshot_dir, listing, cmd, overlay, meta)
        listing_paths, _ = _filter_listing(listing, cmd)
        present, missing = _referenced_paths(cmd, listing)
        missing, vouched = _split_attested(missing, attested)
        listing_header = LISTING_HEADER_ROOTED.format(root=root) if root else LISTING_HEADER
        show = lambda path: f"{root}/{path}" if root else f"./{path}"  # noqa: E731
        missing_text = (
            "\n"
            + NOT_PRESENT_HEADER
            + "\n".join(
                f"- {show(p) if root and not p.startswith(('/', '..')) else p}"
                f"   ->   No such file or directory"
                for p in missing
            )
            + "\n"
            if missing
            else ""
        )
        vouched_text = (
            "\n" + SESSION_PATHS_HEADER + "\n".join(f"- {p}" for p in vouched) + "\n"
            if vouched
            else ""
        )

        contents_budget = (
            self.settings.max_context_chars
            - len(listing_header)
            - len(missing_text)
            - len(vouched_text)
            - len(evidence)
            - self._LISTING_MIN_CHARS
        )
        wants_numbers = bool(_WANTS_LINE_NUMBERS.search(cmd or ""))
        grep_style = bool(_GREP_STYLE.search(cmd or ""))
        note = LINE_NUMBER_NOTE_GREP if grep_style else LINE_NUMBER_NOTE_COLUMN
        contents_header = CONTENTS_HEADER + (note if wants_numbers else "")
        contents_parts: list[str] = []
        contents_used = len(contents_header) + 2
        for path in present[: self.settings.max_files]:
            text = self._file_text(snapshot_dir, path, overlay)
            stale = False
            if text is None and overlay.is_dirty(path):
                # edited by something we cannot model: the pre-edit text still beats nothing
                text, stale = self._read_snapshot_file(snapshot_dir, path), True
            if text is None:
                continue
            body = _truncate(text, self.settings.max_file_chars)
            if wants_numbers:
                body = _number_lines(body, grep_style)
            label = f"{show(path)}{PRE_EDIT_SUFFIX if stale else ''}"
            part = f"--- {label} ---\n{body}\n"
            if contents_used + len(part) > contents_budget:
                break
            contents_parts.append(part)
            contents_used += len(part) + 1
        contents_text = (
            "\n" + contents_header + "\n" + "\n".join(contents_parts) if contents_parts else ""
        )

        listing_budget = (
            self.settings.max_context_chars
            - len(listing_header)
            - len(contents_text)
            - len(missing_text)
            - len(vouched_text)
            - len(evidence)
            - 64
        )
        if not listing_paths:
            listing_text = (
                "(no files in this repository match the command's filters — "
                "the exploration output is empty)"
            )
        else:
            lines: list[str] = []
            used = 0
            for path in listing_paths[: self.settings.max_paths]:
                line = show(path)
                if used + len(line) + 1 > listing_budget:
                    break
                lines.append(line)
                used += len(line) + 1
            over = len(listing_paths) - len(lines)
            listing_text = "\n".join(lines)
            if over > 0:
                listing_text += f"\n... (+{over} more matching files)"

        block = (
            listing_header
            + "\n"
            + listing_text
            + "\n"
            + contents_text
            + vouched_text
            + missing_text
        )
        return _truncate(evidence + block, self.settings.max_context_chars), None, None

    def _git_hints(
        self,
        snapshot_dir: Path,
        listing: list[str],
        cmd: str,
        overlay: Overlay,
        meta: GitMeta,
    ) -> str:
        read_base = lambda rel: self._read_snapshot_file(snapshot_dir, rel)  # noqa: E731
        note = explain_git(cmd, overlay, listing, read_base, meta)
        hint = "\n" + GIT_SEMANTICS_HEADER + "\n" + note if note else ""
        evidence = _truncate(
            git_evidence(cmd, overlay, read_base, listing, meta), self.settings.max_file_chars
        )
        ledger = ledger_block(overlay.git) if is_git_command(cmd) else ""
        return hint + evidence + ledger

    def _file_text(self, snapshot_dir: Path, rel_path: str, overlay: Overlay) -> str | None:
        if overlay.is_dirty(rel_path):
            return None
        return overlay.read(rel_path) or self._read_snapshot_file(snapshot_dir, rel_path)

    def _file_size(self, snapshot_dir: Path, rel_path: str, overlay: Overlay) -> int | None:
        if overlay.is_dirty(rel_path):
            return None
        held = overlay.read(rel_path)
        if held is not None:
            return len(held.encode("utf-8", "replace"))
        root = snapshot_dir.resolve()
        try:
            target = (snapshot_dir / rel_path).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                return None
            return target.stat().st_size
        except OSError:
            return None

    def _read_snapshot_file(self, snapshot_dir: Path, rel_path: str) -> str | None:
        root = snapshot_dir.resolve()
        try:
            target = (snapshot_dir / rel_path).resolve()
        except OSError:
            return None
        if not target.is_relative_to(root) or not target.is_file():
            return None
        try:
            return target.read_text(errors="replace")
        except OSError:
            return None

    def _trajectory_block(self, shard_name: str, row_idx: int, turn_idx: int) -> str | None:
        if not self.settings.dataset_root:
            return None
        try:
            row = _read_parquet_row(Path(self.settings.dataset_root) / shard_name, row_idx)
        except Exception as exc:
            logger.info(
                "repo_context_trajectory_unavailable shard={} row={} error={}",
                shard_name,
                row_idx,
                f"{type(exc).__name__}: {exc}",
            )
            return None
        normalized = {key: _unwrap_column(value) for key, value in row.items()}
        turns = _extract_turns(normalized)
        assistant_positions = [i for i, turn in enumerate(turns) if _role(turn) == "assistant"]
        pairs: list[tuple[str, str]] = []
        for assistant_index, position in enumerate(assistant_positions):
            if assistant_index < turn_idx:
                continue
            command = _first_command(_content(turns[position]))
            if not command or position + 1 >= len(turns):
                continue
            if _role(turns[position + 1]) == "assistant":
                continue
            observation = _content(turns[position + 1]).strip()
            if not observation:
                continue
            if ANY_MARKER_RE.search(command) or ANY_MARKER_RE.search(observation):
                continue
            pairs.append((command, observation))
            if len(pairs) >= self.settings.max_trajectory_pairs:
                break
        if not pairs:
            return None
        parts = [TRAJECTORY_HEADER]
        for step, (command, observation) in enumerate(pairs, 1):
            parts.append(
                f"REFERENCE STEP {step}:\n```bash\n{command}\n```\n\n"
                f"ENVIRONMENT OBSERVATION:\n{_truncate(observation, self.settings.max_file_chars)}"
            )
        return _truncate("\n\n".join(parts), self.settings.max_context_chars)

    def _auth_headers(self) -> dict[str, str]:
        token = (
            self.settings.github_token
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or ""
        )
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _github_text(self, path: str, accept: str) -> str:
        headers = {**self._auth_headers(), "Accept": accept}
        last_error: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                with self._github_semaphore:
                    response = self._client.get(_API_BASE + path, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(30.0, 1.5 * 2**attempt))
                continue
            if response.status_code == 404:
                raise _NotFound(path)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(f"github {response.status_code} for {path}")
                time.sleep(min(30.0, 1.5 * 2**attempt))
                continue
            response.raise_for_status()
            return response.text
        raise RuntimeError(f"github request failed for {path}: {last_error}")

    def _github_json(self, path: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                with self._github_semaphore:
                    response = self._client.get(_API_BASE + path, headers=self._auth_headers())
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(30.0, 1.5 * 2**attempt))
                continue
            if response.status_code == 404:
                raise _NotFound(path)
            if (
                response.status_code == 429
                or response.status_code >= 500
                or (
                    response.status_code == 403
                    and response.headers.get("x-ratelimit-remaining") == "0"
                )
            ):
                last_error = RuntimeError(f"github {response.status_code} for {path}")
                retry_after = float(response.headers.get("retry-after") or 0)
                time.sleep(min(30.0, max(retry_after, 1.5 * 2**attempt)))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"github request failed for {path}: {last_error}")

    def _key_lock(self, key: str) -> threading.Lock:
        with self._locks_mutex:
            return self._locks.setdefault(key, threading.Lock())


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _hashed(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
