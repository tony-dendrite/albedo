from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass, field
from functools import lru_cache

_GREP_BOOL_FLAGS = {
    "r": "recursive",
    "R": "recursive",
    "n": "line_numbers",
    "l": "files_only",
    "i": "ignore_case",
    "w": "word",
    "c": "count_only",
    "h": "no_filename",
    "H": "with_filename",
    "E": "extended",
    "F": "fixed",
    "s": "quiet_errors",
    "a": "binary_text",
    "v": "invert",
}
_GREP_VALUE_FLAGS = {"A": "after", "B": "before", "C": "context", "m": "max_count", "e": "pattern"}
_PIPE_STAGES = {"head", "tail", "sort", "uniq", "wc", "grep", "cat", "sed"}
_QUOTED_SPAN = re.compile(r"'[^']*'|\"[^\"]*\"")


@dataclass
class ParseFailure:
    reason: str
    detail: str = ""


@dataclass
class Stage:
    name: str
    args: list[str] = field(default_factory=list)


@dataclass
class SearchPlan:
    pattern: str
    targets: list[str]
    recursive: bool = False
    ignore_case: bool = False
    word: bool = False
    fixed: bool = False
    extended: bool = False
    invert: bool = False
    line_numbers: bool = False
    files_only: bool = False
    count_only: bool = False
    with_filename: bool | None = None
    after: int = 0
    before: int = 0
    context: int = 0
    max_count: int = 0
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    list_only: bool = False
    list_dir: bool = False
    show_hidden: bool = False
    show_dots: bool = False
    long: bool = False
    literal: str | None = None
    read_mode: str = ""
    read_tool: str = ""
    search_tool: str = ""
    read_range: tuple[int, int] | None = None
    dot_target: bool = False
    stderr_quiet: bool = False
    name_globs: list[str] = field(default_factory=list)
    path_globs: list[str] = field(default_factory=list)
    not_name_globs: list[str] = field(default_factory=list)
    not_path_globs: list[str] = field(default_factory=list)
    max_depth: int = 0
    min_depth: int = 0
    dirs_only: bool = False
    pipeline: list[Stage] = field(default_factory=list)
    absolute: bool = False
    root_prefix: str = ""


@dataclass
class SearchResult:
    output: str
    truncated: bool = False
    incomplete: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    empty: bool = False


@dataclass
class Limits:
    max_files: int = 5000
    max_bytes: int = 40 * 1024 * 1024
    max_matches: int = 5000
    max_pattern: int = 500


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


_CONTROL_OP = re.compile(r"[;&`]|\$\(|\|\|")
_STDERR_QUIET = re.compile(r"2>\s*/dev/null|2>&-")


def _mask_quoted(text: str) -> str:
    return _QUOTED_SPAN.sub(lambda m: m.group(0)[0] * len(m.group(0)), text)


def _split_top_level_pipes(text: str) -> list[str]:
    out, buf, quote = [], [], None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            buf.append(ch)
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == "|":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [s for s in out]


def _split_pipeline(cmd: str) -> list[list[str]] | ParseFailure:
    text = (cmd or "").strip()
    if not text:
        return ParseFailure("unparsed", "empty command")
    if "-exec" in text:
        text = re.sub(r"\s*\\;", " ", text)
        text = re.sub(r"\s*\{\}\s*;", " ", text)
        text = re.sub(r"\s*\+\s*(?=\||$)", " ", text)
    # 2>&1 is a redirect, not a control operator: the & would otherwise refuse the command
    # before _strip_redirects ever gets to see the token
    text = re.sub(r"\s2>&1(?=\s|$)", "", text)
    if _CONTROL_OP.search(_mask_quoted(text)):
        stripped = re.sub(r"^\s*cd\s+[^\s;&]+\s*&&\s*", "", text)
        if stripped != text and not _CONTROL_OP.search(_mask_quoted(stripped)):
            text = stripped
        else:
            return ParseFailure("unsupported_shell", "control operators")
    parts = []
    for seg in _split_top_level_pipes(text):
        try:
            parts.append(shlex.split(seg))
        except ValueError:
            try:
                lexer = shlex.shlex(seg, posix=False)
                lexer.whitespace_split = True
                parts.append([_unquote(tok) for tok in lexer])
            except ValueError as exc:
                return ParseFailure("unparsed", str(exc))
    if any(not p for p in parts):
        return ParseFailure("unparsed", "empty pipeline segment")
    return parts


_CHAIN_REFUSED = re.compile(r"<<|;|\|\||`|\$\(")


def split_chain(cmd: str) -> list[str] | None:
    text = (cmd or "").strip()
    if not text or _CHAIN_REFUSED.search(_mask_quoted(text)):
        return None
    out, buf, quote = [], [], None
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                quote = None
            buf.append(char)
        elif char in "\"'":
            quote = char
            buf.append(char)
        elif text[index : index + 2] == "&&":
            out.append("".join(buf))
            buf = []
            index += 2
            continue
        else:
            buf.append(char)
        index += 1
    out.append("".join(buf))
    stages = [s.strip() for s in out if s.strip()]
    return stages if len(stages) > 1 else None


def _parse_grep(tokens: list[str], plan: SearchPlan) -> ParseFailure | None:
    plan.search_tool = tokens[0]
    pattern: str | None = None
    operands: list[str] = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            operands.extend(tokens[index + 1 :])
            break
        if token.startswith("--"):
            name, _, value = token.partition("=")
            if name == "--include" and value:
                plan.include.append(value)
            elif name == "--exclude" and value:
                plan.exclude.append(value)
            elif name in ("--recursive",):
                plan.recursive = True
            elif name in ("--line-number",):
                plan.line_numbers = True
            elif name in ("--files-with-matches",):
                plan.files_only = True
            elif name in ("--ignore-case",):
                plan.ignore_case = True
            elif name in ("--invert-match",):
                plan.invert = True
            elif name in ("--no-messages", "--color", "--colour"):
                pass
            else:
                return ParseFailure("unknown_flag", token)
        elif token.startswith("-") and len(token) > 1:
            chars = token[1:]
            position = 0
            while position < len(chars):
                char = chars[position]
                if char in _GREP_VALUE_FLAGS:
                    raw = chars[position + 1 :]
                    if not raw:
                        index += 1
                        if index >= len(tokens):
                            return ParseFailure("unparsed", f"-{char} without value")
                        raw = tokens[index]
                    field_name = _GREP_VALUE_FLAGS[char]
                    if field_name == "pattern":
                        pattern = raw
                    else:
                        if not raw.lstrip("-").isdigit():
                            return ParseFailure("unparsed", f"-{char} {raw}")
                        setattr(plan, field_name, int(raw))
                    position = len(chars)
                    continue
                if char not in _GREP_BOOL_FLAGS:
                    return ParseFailure("unknown_flag", f"-{char}")
                attr = _GREP_BOOL_FLAGS[char]
                if attr == "recursive":
                    plan.recursive = True
                elif attr == "no_filename":
                    plan.with_filename = False
                elif attr == "with_filename":
                    plan.with_filename = True
                elif attr in ("quiet_errors", "binary_text"):
                    pass
                elif attr == "extended":
                    plan.extended = True
                else:
                    setattr(plan, attr, True)
                position += 1
        elif pattern is None:
            pattern = token
        else:
            operands.append(token)
        index += 1
    if pattern is None:
        return ParseFailure("unparsed", "no pattern")
    plan.pattern = pattern
    plan.targets.extend(operands)
    return None


_PRUNE_OR = ["-prune", "-o"]
_FIND_PREDICATES = {"-name", "-iname", "-path"}


def _rewrite_prune(tokens: list[str]) -> list[str]:
    out: list[str] = []
    pruned: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] in ("-path", "-name") and tokens[index + 2 : index + 4] == _PRUNE_OR:
            skipped = tokens[index + 1].rstrip("/")
            pruned.extend(["-not", tokens[index], skipped])
            pruned.extend(["-not", tokens[index], f"{skipped}/*"])
            index += 4
            continue
        if tokens[index] == "-print" and pruned:
            index += 1
            continue
        out.append(tokens[index])
        index += 1
    return out + pruned


def _parse_find(tokens: list[str], plan: SearchPlan) -> ParseFailure | None:
    tokens = _rewrite_prune(tokens)
    if "-o" in tokens and len({t for t in tokens if t in _FIND_PREDICATES}) > 1:
        return ParseFailure("unsupported_form", "find -o across different predicates")
    index = 1
    negate = False
    while index < len(tokens):
        token = tokens[index]
        if token in ("-not", "!"):
            negate = True
            index += 1
            continue
        if token in ("-name", "-iname"):
            index += 1
            if index >= len(tokens):
                return ParseFailure("unparsed", "-name without value")
            globs = plan.not_name_globs if negate else plan.name_globs
            globs.append(tokens[index])
        elif token == "-type":
            index += 1
            if index >= len(tokens) or tokens[index] not in ("f", "d"):
                return ParseFailure("unknown_flag", "-type")
            plan.dirs_only = tokens[index] == "d"
        elif token == "-path":
            index += 1
            if index >= len(tokens):
                return ParseFailure("unparsed", "-path without value")
            globs = plan.not_path_globs if negate else plan.path_globs
            globs.append(tokens[index])
        elif token in ("-maxdepth", "-mindepth"):
            index += 1
            if index >= len(tokens) or not tokens[index].isdigit():
                return ParseFailure("unparsed", f"{token} without a number")
            setattr(plan, token[1:].replace("depth", "_depth"), int(tokens[index]))
        elif token in ("-o", "-print", "{}", ";", "\\;", "-exec"):
            if token == "-exec":
                return ParseFailure("internal", "-exec handled by caller")
        elif token.startswith("-"):
            return ParseFailure("unknown_flag", token)
        else:
            plan.targets.append(token)
        negate = False
        index += 1
    return None


def _parse_ls(tokens: list[str], plan: SearchPlan) -> ParseFailure | None:
    for token in tokens[1:]:
        if token.startswith("--"):
            if token not in ("--all", "--almost-all"):
                return ParseFailure("unknown_flag", token)
            plan.show_hidden = True
            plan.show_dots = token == "--all"
        elif token.startswith("-") and len(token) > 1:
            for char in token[1:]:
                if char not in ("1", "a", "A", "l"):
                    return ParseFailure("unsupported_form", f"ls -{char}")
                if char == "l":
                    plan.long = True
                if char in ("a", "A"):
                    plan.show_hidden = True
                    plan.show_dots = plan.show_dots or char == "a"
        elif any(ch in token for ch in "*?["):
            return ParseFailure("unsupported_form", "ls with a glob")
        else:
            plan.targets.append(token)
    if len(plan.targets) > 1:
        return ParseFailure("unsupported_form", "ls with several operands")
    if not plan.targets:
        plan.targets.append(".")
    plan.list_dir = True
    return None


_LS_DIR_MODE = "drwxr-xr-x"
_LS_FILE_MODE = "-rw-r--r--"
_LS_OWNER = "root root"
_LS_DATE = "Jan  3 20:00"
_LS_DIR_SIZE = 4096
_LS_BLOCK = 4096


def _entry_size(read_file, size_file, path: str) -> int:
    if size_file is not None:
        size = size_file(path)
        if size is not None:
            return size
    text = read_file(path)
    return len(text.encode("utf-8", "replace")) if text is not None else 0


def _long_row(name: str, is_dir: bool, size: int) -> str:
    mode, links = (_LS_DIR_MODE, 2) if is_dir else (_LS_FILE_MODE, 1)
    return f"{mode} {links} {_LS_OWNER} {size:>8} {_LS_DATE} {name}"


def _run_list_dir(
    plan: SearchPlan, read_file, size_file, listing: list[str]
) -> SearchResult | ParseFailure:
    listing_set = set(listing)
    target = plan.targets[0]
    base = _to_repo_relative(target, listing_set)
    if base is None:
        return ParseFailure("unresolved_target", target)
    if base and base in listing_set:
        if plan.long:
            return SearchResult(
                output=_long_row(target, False, _entry_size(read_file, size_file, base))
            )
        return SearchResult(output=target)
    prefix = f"{base}/" if base else ""
    names: dict[str, bool] = {}
    for path in listing:
        if prefix and not path.startswith(prefix):
            continue
        rest = path[len(prefix) :]
        if not rest:
            continue
        head, _, tail = rest.partition("/")
        names[head] = names.get(head, False) or bool(tail)
    if not names:
        return SearchResult(
            output=f"ls: cannot access '{target}': No such file or directory",
            missing=[target],
        )
    shown = sorted(n for n in names if plan.show_hidden or not n.startswith("."))
    if plan.show_dots:
        shown = [".", ".."] + shown
    if plan.long:
        rows, blocks = [], 0
        for name in shown:
            is_dir = names.get(name, True) if name not in (".", "..") else True
            size = _LS_DIR_SIZE if is_dir else _entry_size(read_file, size_file, f"{prefix}{name}")
            blocks += -(-size // _LS_BLOCK) * (_LS_BLOCK // 1024)
            rows.append(_long_row(name, is_dir, size))
        shown = [f"total {blocks}"] + rows
    shown = _apply_pipeline(shown, plan)
    return SearchResult(output="\n".join(shown), empty=not shown)


_READ_CMDS = ("cat", "nl", "head", "tail", "sed")


def _number_lines(text: str, grep_style: bool) -> str:
    lines = text.split("\n")
    if grep_style:
        return "\n".join(f"{n}:{line}" for n, line in enumerate(lines, 1))
    return "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(lines, 1))


_SED_NUMERIC_P = re.compile(r"'?(\d+)(?:,(\d+))?p'?")


def _parse_sed_script(script: str) -> tuple[int, int] | None:
    """`Np` or `N,Mp` -> the inclusive line bounds it prints."""
    numeric = _SED_NUMERIC_P.fullmatch(script)
    if numeric is None:
        return None
    start = int(numeric.group(1))
    return start, int(numeric.group(2) or start)


_GENERATED_PATH = re.compile(
    r"(^|/)(node_modules|\.venv|venv|site-packages|dist-packages|build|dist|target|out|"
    r"__pycache__|\.pytest_cache|\.tox|\.eggs|[^/]+\.egg-info|coverage|htmlcov)(/|$)"
    r"|\.(pyc|pyo|so|o|a|class|jar|whl|log)$"
)
_MISSING_READ = {
    "cat": "cat: {target}: No such file or directory",
    "nl": "nl: {target}: No such file or directory",
    "head": "head: cannot open '{target}' for reading: No such file or directory",
    "tail": "tail: cannot open '{target}' for reading: No such file or directory",
    "sed": "sed: can't read {target}: No such file or directory",
}


def _parse_read(tokens: list[str], plan: SearchPlan) -> ParseFailure | None:
    name, args = tokens[0], tokens[1:]
    plan.read_tool = name
    if name == "sed":
        if [a for a in args if a.startswith("-")] != ["-n"]:
            return ParseFailure("unsupported_form", "sed without -n")
        script = [a for a in args if not a.startswith("-")]
        if not script:
            return ParseFailure("unparsed", "sed without script")
        parsed = _parse_sed_script(script[0])
        if parsed is None:
            return ParseFailure("unsupported_form", f"sed script {script[0]}")
        plan.read_mode = "sed"
        plan.read_range = parsed
        plan.targets.extend(script[1:])
        return None
    if name in ("head", "tail"):
        count, index = 10, 0
        while index < len(args):
            token = args[index]
            if re.fullmatch(r"-\d+", token):
                count = int(token[1:])
            elif token == "-n":
                index += 1
                if index >= len(args) or not args[index].lstrip("-+").isdigit():
                    return ParseFailure("unparsed", "head/tail -n")
                count = int(args[index].lstrip("-+"))
            elif token.startswith("-"):
                return ParseFailure("unknown_flag", token)
            else:
                plan.targets.append(token)
            index += 1
        plan.read_mode, plan.read_range = name, (count, count)
        return None
    numbered = False
    for token in args:
        if not token.startswith("-"):
            plan.targets.append(token)
        elif name == "cat" and set(token[1:]) <= {"n"}:
            numbered = True
        elif name == "nl" and token in ("-ba", "-b"):
            numbered = True
        else:
            return ParseFailure("unknown_flag", token)
    if name == "nl" and not numbered:
        return ParseFailure("unsupported_form", "nl without -ba")
    plan.read_mode = "number" if numbered else "cat"
    return None


def _parse_pipe_stage(tokens: list[str]) -> Stage | ParseFailure:
    name = tokens[0]
    if name not in _PIPE_STAGES:
        return ParseFailure("unsupported_pipe", name)
    if name == "wc":
        if tokens[1:] not in (["-l"], []):
            return ParseFailure("unsupported_pipe", " ".join(tokens))
        return Stage("wc_l")
    if name in ("head", "tail"):
        count = 10
        for token in tokens[1:]:
            if re.fullmatch(r"-\d+", token):
                count = int(token[1:])
            elif token in ("-n",):
                continue
            elif token.isdigit():
                count = int(token)
            else:
                return ParseFailure("unsupported_pipe", " ".join(tokens))
        return Stage(name, [str(count)])
    if name == "sort":
        if any(t not in ("-u",) for t in tokens[1:]):
            return ParseFailure("unsupported_pipe", " ".join(tokens))
        return Stage("sort", ["-u"] if "-u" in tokens else [])
    if name == "uniq":
        if tokens[1:]:
            return ParseFailure("unsupported_pipe", " ".join(tokens))
        return Stage("uniq")
    if name == "cat":
        if tokens[1:] == ["-n"]:
            return Stage("number")
        if not tokens[1:]:
            return Stage("passthrough")
        return ParseFailure("unsupported_pipe", " ".join(tokens))
    if name == "sed":
        script = [t for t in tokens[1:] if not t.startswith("-")]
        if [t for t in tokens[1:] if t.startswith("-")] != ["-n"] or len(script) != 1:
            return ParseFailure("unsupported_pipe", " ".join(tokens))
        parsed = _parse_sed_script(script[0])
        if parsed is None:
            return ParseFailure("unsupported_pipe", " ".join(tokens))
        return Stage("slice", [str(parsed[0]), str(parsed[1])])
    flags = ""
    pattern = None
    for token in tokens[1:]:
        if token.startswith("-"):
            if set(token[1:]) <= {"v", "i", "E", "F", "n"}:
                flags += token[1:]
            else:
                return ParseFailure("unsupported_pipe", token)
        elif pattern is None:
            pattern = token
        else:
            return ParseFailure("unsupported_pipe", " ".join(tokens))
    if pattern is None:
        return ParseFailure("unsupported_pipe", "grep without pattern")
    return Stage("filter", [pattern, flags])


def _parse_echo(tokens: list[str], plan: SearchPlan) -> ParseFailure | None:
    args = tokens[1:]
    index = 0
    while index < len(args) and args[index] == "-n":
        index += 1
    for token in args[index:]:
        if token in ("-e", "-E"):
            return ParseFailure("unsupported_form", f"echo {token}")
    plan.literal = " ".join(args[index:])
    return None


_REDIRECT_RE = re.compile(r"^\d*>>?(&\d+)?$|^\d*>>?\S+$")
_STDOUT_REDIRECT_RE = re.compile(r"^1?>>?")


def _redirects_stdout(tokens: list[str]) -> bool:
    return any(_REDIRECT_RE.match(t) and _STDOUT_REDIRECT_RE.match(t) for t in tokens)


def _strip_redirects(tokens: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for token in tokens:
        if skip:
            skip = False
            continue
        if _REDIRECT_RE.match(token):
            skip = token.rstrip(">").isdigit() or token in (">", ">>")
            continue
        out.append(token)
    return out


def parse_search(cmd: str) -> SearchPlan | ParseFailure:
    segments = _split_pipeline(cmd)
    if isinstance(segments, ParseFailure):
        return segments
    if any(_redirects_stdout(seg) for seg in segments):
        return ParseFailure("unsupported_shell", "output redirected to a file")
    segments = [_strip_redirects(seg) for seg in segments]
    segments = [seg for seg in segments if seg]
    if not segments:
        return ParseFailure("unparsed", "empty after redirects")

    head = segments[0]
    plan = SearchPlan(pattern="", targets=[])
    plan.stderr_quiet = bool(_STDERR_QUIET.search(cmd or ""))

    if head[0] in _READ_CMDS:
        failure = _parse_read(head, plan)
        if failure:
            return failure
        rest = segments[1:]
    elif head[0] == "grep" or head[0] == "rg":
        failure = _parse_grep(head, plan)
        if failure:
            return failure
        rest = segments[1:]
    elif head[0] == "ls":
        failure = _parse_ls(head, plan)
        if failure:
            return failure
        rest = segments[1:]
    elif head[0] == "echo":
        failure = _parse_echo(head, plan)
        if failure:
            return failure
        return plan
    elif head[0] == "find":
        if "-exec" in head:
            cut = head.index("-exec")
            exec_tokens = [t for t in head[cut + 1 :] if t not in ("{}", ";", "\\;", "+")]
            if not exec_tokens or exec_tokens[0] not in ("grep", "rg", "ls"):
                return ParseFailure("unsupported_form", "find -exec non-grep")
            failure = _parse_find(head[:cut], plan)
            if failure:
                return failure
            if exec_tokens[0] == "ls":
                if plan.dirs_only:
                    return ParseFailure("unsupported_form", "find -type d -exec ls")
                for token in exec_tokens[1:]:
                    if not token.startswith("-") or set(token[1:]) - set("la1"):
                        return ParseFailure("unsupported_form", f"find -exec ls {token}")
                    plan.long = plan.long or "l" in token
                plan.list_only = True
            else:
                failure = _parse_grep(exec_tokens, plan)
                if failure:
                    return failure
            rest = segments[1:]
        else:
            failure = _parse_find(head, plan)
            if failure:
                return failure
            nxt = segments[1] if len(segments) > 1 else None
            if nxt is None:
                plan.list_only = True
                rest = []
            elif nxt[0] == "xargs":
                inner = nxt[1:]
                while inner and inner[0].startswith("-"):
                    if inner[0] in ("-0", "-r", "--no-run-if-empty"):
                        inner = inner[1:]
                    elif inner[0] in ("-I", "-n"):
                        inner = inner[2:]
                    else:
                        return ParseFailure("unknown_flag", inner[0])
                if not inner or inner[0] not in ("grep", "rg"):
                    return ParseFailure("unsupported_form", "xargs non-grep")
                failure = _parse_grep(inner, plan)
                if failure:
                    return failure
                rest = segments[2:]
            else:
                plan.list_only = True
                plan.recursive = True
                rest = segments[1:]
        plan.recursive = True
    else:
        return ParseFailure("not_a_search", head[0])

    if plan.read_mode and (len(plan.targets) != 1 or any(ch in plan.targets[0] for ch in "*?[")):
        return ParseFailure("unsupported_form", "read needs one named file")
    plan.dot_target = any(t in (".", "./") or t.startswith("./") for t in plan.targets)
    if not plan.targets:
        if plan.recursive:
            plan.targets.append(".")
        else:
            return ParseFailure("unsupported_form", "no search target")

    for tokens in rest:
        stage = _parse_pipe_stage(tokens)
        if isinstance(stage, ParseFailure):
            return stage
        plan.pipeline.append(stage)

    first = plan.targets[0]
    plan.absolute = first.startswith("/")
    if plan.absolute:
        match = re.match(r"^(/[^/]+)", first)
        plan.root_prefix = match.group(1) if match else ""
    if len(plan.pattern) > 500:
        return ParseFailure("unparsed", "pattern too long")
    return plan


_SANDBOX_ROOTS = {"testbed": False, "workspace": True}


def _to_repo_relative(target: str, listing: set[str]) -> str | None:
    cleaned = target.rstrip("/")
    if cleaned in ("", ".", "./"):
        return ""
    cleaned = cleaned[2:] if cleaned.startswith("./") else cleaned
    if cleaned in listing:
        return cleaned
    segments = cleaned.lstrip("/").split("/")
    for start in range(len(segments)):
        candidate = "/".join(segments[start:])
        if not candidate:
            continue
        if candidate in listing or any(p.startswith(candidate + "/") for p in listing):
            return candidate
    if not cleaned.startswith("/"):
        return cleaned
    bare = cleaned.lstrip("/")
    for root, holds_checkout in _SANDBOX_ROOTS.items():
        if bare == root:
            return ""
        if not bare.startswith(root + "/"):
            continue
        remainder = bare[len(root) + 1 :]
        if holds_checkout:
            head, _, tail = remainder.partition("/")
            if not tail:
                return ""
            remainder = tail
        return remainder
    return None


def _directories_under(paths: list[str], prefix: str) -> list[str]:
    dirs: set[str] = {prefix.rstrip("/")}
    for path in paths:
        parts = path[len(prefix) :].split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            dirs.add(prefix + "/".join(parts[:depth]))
    return sorted(dirs)


def _within_depth(path: str, prefix: str, plan: SearchPlan) -> bool:
    tail = path[len(prefix) :]
    depth = tail.count("/") + 1 if tail else 0
    if plan.max_depth and depth > plan.max_depth:
        return False
    return not (plan.min_depth and depth < plan.min_depth)


def _candidate_files(
    plan: SearchPlan, listing: list[str]
) -> tuple[list[str], list[str]] | ParseFailure:
    listing_set = set(listing)
    files: list[str] = []
    unmatched: list[str] = []
    searched_dir = False
    if plan.absolute and plan.targets:
        first = plan.targets[0]
        stem = first.rstrip("/")
        anchored = _to_repo_relative(stem, listing_set)
        if anchored is not None and (not anchored or stem.endswith(anchored)):
            plan.root_prefix = stem[: -len(anchored)].rstrip("/") if anchored else stem
    for target in plan.targets:
        if any(ch in target for ch in "*?["):
            resolved = _to_repo_relative(target, listing_set)
            if resolved is None:
                return ParseFailure("unresolved_target", target)
            glob = resolved or target.lstrip("/")
            hits = [p for p in listing if _shell_glob(p, glob) or _shell_glob(p, glob.lstrip("./"))]
            files.extend(hits)
            if not hits:
                unmatched.append(target)
            continue
        base = _to_repo_relative(target, listing_set)
        if base is None:
            return ParseFailure("unresolved_target", target)
        if base and base in listing_set:
            files.append(base)
            continue
        prefix = (base + "/") if base else ""
        hits = [p for p in listing if not prefix or p.startswith(prefix)]
        if not hits:
            unmatched.append(target)
        searched_dir = True
        if plan.dirs_only:
            hits = _directories_under(hits, prefix)
        if plan.max_depth or plan.min_depth:
            hits = [p for p in hits if _within_depth(p, prefix, plan)]
        files.extend(hits)

    def keep(path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        if plan.not_name_globs and any(fnmatch.fnmatch(name, g) for g in plan.not_name_globs):
            return False
        if plan.not_path_globs:
            shown = _display(path, plan)
            if any(
                fnmatch.fnmatch(shown, g) or fnmatch.fnmatch(path, g.lstrip("./"))
                for g in plan.not_path_globs
            ):
                return False
        if plan.name_globs and not any(fnmatch.fnmatch(name, g) for g in plan.name_globs):
            return False
        if plan.include and not any(fnmatch.fnmatch(name, g) for g in plan.include):
            return False
        if plan.exclude and any(fnmatch.fnmatch(name, g) for g in plan.exclude):
            return False
        if plan.path_globs:
            shown = _display(path, plan)
            if not any(
                fnmatch.fnmatch(shown, g) or fnmatch.fnmatch(path, g.lstrip("./"))
                for g in plan.path_globs
            ):
                return False
        return True

    seen: set[str] = set()
    ordered = [p for p in files if keep(p) and not (p in seen or seen.add(p))]
    return ordered, unmatched, searched_dir


@lru_cache(maxsize=512)
def _glob_re(pattern: str) -> re.Pattern:
    out = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            close = pattern.find("]", i)
            if close == -1:
                out.append(re.escape(c))
            else:
                out.append(pattern[i : close + 1])
                i = close + 1
                continue
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def _shell_glob(path: str, pattern: str) -> bool:
    return bool(_glob_re(pattern).match(path))


def _display(path: str, plan: SearchPlan) -> str:
    if plan.absolute and plan.root_prefix:
        return f"{plan.root_prefix}/{path}" if path else plan.root_prefix
    if plan.dot_target:
        return f"./{path}" if path else "."
    return path or "."


_POSIX_CLASS = {
    "alpha": "a-zA-Z",
    "digit": "0-9",
    "alnum": "a-zA-Z0-9",
    "upper": "A-Z",
    "lower": "a-z",
    "space": " \\t\\n\\r\\f\\v",
    "blank": " \\t",
}
_POSIX_CLASS_RE = re.compile(r"\[:(\w+):\]")


def _bre_to_python(pattern: str) -> str:
    # python's re reads [[:space:]] as a set of literal characters, so it compiles and
    # then matches nothing; expand the class to its members before anything else
    pattern = _POSIX_CLASS_RE.sub(lambda m: _POSIX_CLASS.get(m.group(1), m.group(0)), pattern)
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            nxt = pattern[index + 1]
            if nxt in "|(){}+?":
                out.append(nxt)
            else:
                out.append(char + nxt)
            index += 2
            continue
        if char in "|(){}+?":
            out.append("\\" + char)
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _compile(plan: SearchPlan) -> re.Pattern | ParseFailure:
    if plan.fixed:
        pattern = re.escape(plan.pattern)
    elif plan.extended:
        pattern = plan.pattern
    else:
        pattern = _bre_to_python(plan.pattern)
    if plan.word:
        pattern = rf"\b(?:{pattern})\b"
    try:
        return re.compile(pattern, re.IGNORECASE if plan.ignore_case else 0)
    except re.error as exc:
        return ParseFailure("bad_pattern", str(exc))


_GREP_PATTERN_ERROR = (
    ("unterminated subpattern", r"Unmatched ( or \("),
    ("unbalanced parenthesis", r"Unmatched ) or \)"),
    ("unterminated character set", "Unmatched [, [^, [:, [., or [="),
    ("bad character range", "Invalid range end"),
)


def _pattern_error(plan: SearchPlan, detail: str) -> str | None:
    if plan.search_tool != "grep":
        return None
    for needle, message in _GREP_PATTERN_ERROR:
        if needle in detail:
            return f"grep: {message}"
    return None


def _apply_pipeline(lines: list[str], plan: SearchPlan) -> list[str]:
    for stage in plan.pipeline:
        if stage.name == "passthrough":
            continue
        if stage.name == "head":
            lines = lines[: int(stage.args[0])]
        elif stage.name == "slice":
            lines = lines[int(stage.args[0]) - 1 : int(stage.args[1])]
        elif stage.name == "tail":
            count = int(stage.args[0])
            lines = lines[-count:] if count else []
        elif stage.name == "sort":
            lines = sorted(set(lines)) if "-u" in stage.args else sorted(lines)
        elif stage.name == "uniq":
            deduped: list[str] = []
            for line in lines:
                if not deduped or deduped[-1] != line:
                    deduped.append(line)
            lines = deduped
        elif stage.name == "wc_l":
            lines = [str(len(lines))]
        elif stage.name == "number":
            lines = _number_lines("\n".join(lines), False).split("\n") if lines else []
        elif stage.name == "filter":
            needle, flags = stage.args[0], stage.args[1]
            invert = "v" in flags
            if "F" in flags:
                source = re.escape(needle)
            elif "E" in flags:
                source = needle
            else:
                source = _bre_to_python(needle)
            try:
                rx = re.compile(source, re.IGNORECASE if "i" in flags else 0)
            except re.error:
                rx = re.compile(re.escape(needle), re.IGNORECASE if "i" in flags else 0)
            kept = [(i, ln) for i, ln in enumerate(lines, 1) if bool(rx.search(ln)) != invert]
            lines = [f"{i}:{ln}" for i, ln in kept] if "n" in flags else [ln for _, ln in kept]
    return lines


def _run_read(plan: SearchPlan, read_file, listing: list[str]) -> SearchResult | ParseFailure:
    candidates = _candidate_files(plan, listing)
    if isinstance(candidates, ParseFailure):
        return candidates
    files, unmatched, searched_dir = candidates
    if unmatched and not files and not _GENERATED_PATH.search(unmatched[0]):
        template = _MISSING_READ.get(plan.read_tool, "{tool}: {target}: No such file or directory")
        return SearchResult(
            output=template.format(target=unmatched[0], tool=plan.read_tool),
            missing=list(unmatched),
        )
    if unmatched or not files:
        return ParseFailure("unsupported_form", "read target not in snapshot")
    if searched_dir:
        return ParseFailure("unsupported_form", "read target is a directory")
    text = read_file(files[0])
    if text is None:
        return ParseFailure("unsupported_form", "read target unreadable")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if plan.read_mode == "sed":
        start, end = plan.read_range
        out = lines[start - 1 : end]
    elif plan.read_mode == "head":
        out = lines[: plan.read_range[0]]
    elif plan.read_mode == "tail":
        out = lines[-plan.read_range[0] :] if plan.read_range[0] else []
    elif plan.read_mode == "number":
        out = _number_lines("\n".join(lines), False).split("\n") if lines else []
    else:
        out = lines
    out = _apply_pipeline(out, plan)
    return SearchResult(output="\n".join(out), empty=not out)


def run_search(
    plan: SearchPlan,
    read_file,
    listing: list[str],
    size_file=None,
    root: str = "",
) -> SearchResult | ParseFailure:
    limits = Limits()
    if plan.literal is not None:
        return SearchResult(output=plan.literal, empty=not plan.literal)
    if plan.read_mode:
        return _run_read(plan, read_file, listing)
    if plan.list_dir:
        return _run_list_dir(plan, read_file, size_file, listing)
    if plan.list_only:
        regex = None
    else:
        regex = _compile(plan)
        if isinstance(regex, ParseFailure):
            message = _pattern_error(plan, regex.detail)
            if message is None:
                return regex
            if plan.stderr_quiet:
                return SearchResult(output="", empty=True)
            return SearchResult(output=message)

    candidates = _candidate_files(plan, listing)
    if isinstance(candidates, ParseFailure):
        return candidates
    candidates, unmatched, searched_dir = candidates
    if plan.absolute and not plan.root_prefix:
        # a search rooted above the checkout (`find /`, `grep -r x /`) reports absolute paths,
        # so it needs the directory the checkout sits at. With it, anchor the results; without
        # it, decline rather than print `pkg/mod.py` — no absolutely-rooted search can produce a
        # bare relative path, and handing the assistant one un-anchors it from the repository.
        if not root:
            return ParseFailure("unsupported_form", "absolute search root is unknown")
        plan.root_prefix = root
    truncated = len(candidates) > limits.max_files
    candidates = candidates[: limits.max_files]

    if plan.list_only:
        shown = [_display(path, plan) for path in candidates]
        if plan.long:
            shown = [
                _long_row(name, False, _entry_size(read_file, size_file, path))
                for name, path in zip(shown, candidates)
            ]
        listed = _apply_pipeline(shown, plan)
        errors = [f"find: {t}: No such file or directory" for t in unmatched]
        return SearchResult(
            output="\n".join(errors + listed),
            truncated=truncated,
            missing=unmatched,
            empty=not listed and not errors,
        )

    show_name = plan.with_filename
    if show_name is None:
        show_name = len(candidates) > 1 or searched_dir

    out: list[str] = []
    incomplete: list[str] = []
    budget = limits.max_bytes
    for path in candidates:
        text = read_file(path)
        if text is None:
            incomplete.append(path)
            continue
        budget -= len(text)
        if budget < 0:
            truncated = True
            break
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        hits = [i for i, line in enumerate(lines) if bool(regex.search(line)) != plan.invert]
        if plan.max_count:
            hits = hits[: plan.max_count]
        if not hits:
            continue
        shown = _display(path, plan)
        if plan.files_only:
            out.append(shown)
            continue
        if plan.count_only:
            continue
        before = plan.before or plan.context
        after = plan.after or plan.context
        context_on = bool(before or after)
        hit_set = set(hits)
        wanted: set[int] = set()
        for i in hits:
            wanted.update(range(max(0, i - before), min(len(lines) - 1, i + after) + 1))
        if context_on and out:
            out.append("--")
        previous: int | None = None
        for j in sorted(wanted):
            if context_on and previous is not None and j > previous + 1:
                out.append("--")
            previous = j
            sep = ":" if j in hit_set else "-"
            prefix = f"{shown}{sep}" if show_name else ""
            body = f"{j + 1}{sep}{lines[j]}" if plan.line_numbers else lines[j]
            out.append(f"{prefix}{body}")
            if len(out) >= limits.max_matches:
                truncated = True
                break
        if len(out) >= limits.max_matches:
            break

    if plan.count_only:
        out = []
        for path in candidates:
            text = read_file(path)
            if text is None:
                continue
            counted = text.split("\n")
            if counted and counted[-1] == "":
                counted.pop()
            n = sum(1 for line in counted if bool(regex.search(line)) != plan.invert)
            out.append(f"{_display(path, plan)}:{n}" if show_name else str(n))

    out = _apply_pipeline(out, plan)
    tool = "find" if plan.name_globs or plan.path_globs else "grep"
    errors = [f"{tool}: {target}: No such file or directory" for target in unmatched]
    return SearchResult(
        output="\n".join(errors + out),
        truncated=truncated,
        incomplete=incomplete,
        missing=unmatched,
        empty=not out and not errors,
    )
