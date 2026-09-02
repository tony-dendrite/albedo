from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

DIFF_HEAD_RE = re.compile(r"^(diff --git|--- |\+\+\+ |Index: )", re.M)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
CODE_HINT_RE = re.compile(r"^\s*(import |from \w+ import |def |class |#!/)", re.M)
STDIN_PATH_RE = re.compile(r"^(--- |\+\+\+ )(a/-|b/-|-|/dev/stdin)\s*$", re.M)
_BLOCK_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n```", re.S)
_MSWEA_RE = re.compile(r"<mswea_bash_command>(.*?)</mswea_bash_command>", re.S)
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(?P<delim>\w+)['\"]?")
_PATCH_TARGET_RE = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*['\"]?(?P<target>\S*patch[\w.]*)")
_DIFF_STDIN_CMD_RE = re.compile(r"\bdiff\b[^\n|;&]*(/dev/stdin|\s-(\s|$))")


def lint_patch(text: str) -> list[str]:
    if not text.strip():
        return ["empty submission"]
    issues: list[str] = []
    if (
        STDIN_PATH_RE.search(text)
        or "diff --git a/- b/-" in text
        or re.search(r"^(--- |\+\+\+ )(a/|b/)?/dev/stdin", text, re.M)
        or "diff --git /dev/stdin" in text
    ):
        issues.append("diff file path is stdin ('-' or /dev/stdin)")
    if not DIFF_HEAD_RE.search(text):
        if CODE_HINT_RE.search(text):
            issues.append("no diff markers: raw file content submitted as the patch")
        else:
            issues.append("no diff markers: free text submitted as the patch")
        return issues
    issues += _hunk_issues(text)
    return [i for i in issues if not i.startswith("hunk longer")]


def _hunk_issues(text: str) -> list[str]:
    issues: set[str] = set()
    old = new = 0
    in_hunk = False
    saw_hunk = False
    for line in text.split("\n"):
        header = HUNK_RE.match(line)
        if header:
            if in_hunk and (old > 0 or new > 0):
                issues.add("hunk shorter than declared")
            old = int(header.group(2) if header.group(2) is not None else 1)
            new = int(header.group(4) if header.group(4) is not None else 1)
            in_hunk = True
            saw_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("\\"):
            continue
        if line.startswith(("diff ", "--- ", "+++ ", "Index:")):
            if old > 0 or new > 0:
                issues.add("hunk shorter than declared")
            in_hunk = False
            continue
        if old <= 0 and new <= 0:
            in_hunk = False
            continue
        if line.startswith("+"):
            new -= 1
        elif line.startswith("-"):
            old -= 1
        elif line.startswith(" ") or line == "":
            old -= 1
            new -= 1
        else:
            issues.add("raw line inside a hunk")
            old -= 1
            new -= 1
        if old < -2 or new < -2:
            issues.add("hunk longer than declared")
            in_hunk = False
    if in_hunk and (old > 0 or new > 0):
        issues.add("truncated mid-hunk")
    if not saw_hunk:
        issues.add("diff headers but zero hunks")
    return sorted(issues)


def _commands(assistant_text: str) -> list[str]:
    blocks = [m.group(1) for m in _BLOCK_RE.finditer(assistant_text or "")]
    blocks += [m.group(1) for m in _MSWEA_RE.finditer(assistant_text or "")]
    return blocks


def extract_commands(assistant_text: str) -> list[str]:
    return [b.strip() for b in _commands(assistant_text)]


def _heredoc_body(command: str) -> str | None:
    lines = command.split("\n")
    for i, ln in enumerate(lines):
        here = _HEREDOC_RE.search(ln)
        target = _PATCH_TARGET_RE.search(ln)
        if not here or not target:
            continue
        if re.search(r"\.(py|sh|js|rb|pl)$", target.group("target")):
            continue
        delim = here.group("delim")
        body: list[str] = []
        for line in lines[i + 1 :]:
            if line.strip() == delim:
                return "\n".join(body)
            body.append(line)
        return None
    return None


_OLD_FILE_RE = re.compile(r"^--- (?:a/)?(.+?)\t?$")
_NEW_FILE_RE = re.compile(r"^\+\+\+ (?:b/)?(.+?)\t?$")


def _preimages(text: str) -> tuple[dict[str, dict[int, str]], set[str], str]:
    files: dict[str, dict[int, str]] = {}
    no_eof_newline: set[str] = set()
    cur: dict[int, str] | None = None
    cur_path = ""
    cursor = 0
    remaining_old = 0
    last_was_old = False
    for line in text.split("\n"):
        m = _OLD_FILE_RE.match(line)
        if m:
            path = m.group(1)
            if path == "/dev/null":
                cur = None
                cur_path = ""
            else:
                if ".." in path.split("/") or path.startswith("/"):
                    return {}, set(), f"unsafe path in diff: {path}"
                cur = files.setdefault(path, {})
                cur_path = path
            remaining_old = 0
            continue
        if _NEW_FILE_RE.match(line) or line.startswith(
            (
                "diff ",
                "index ",
                "similarity ",
                "rename ",
                "old mode",
                "new mode",
                "new file mode",
                "deleted file mode",
            )
        ):
            continue
        h = HUNK_RE.match(line)
        if h:
            cursor = int(h.group(1))
            remaining_old = int(h.group(2) if h.group(2) is not None else 1)
            if cursor == 0 and remaining_old == 0:
                cursor = 1
            continue
        if line.startswith("\\"):
            if last_was_old and cur_path:
                no_eof_newline.add(cur_path)
            continue
        if remaining_old <= 0 and not line.startswith("+"):
            last_was_old = False
            continue
        if line.startswith("+"):
            last_was_old = False
            continue
        content = line[1:] if line.startswith(("-", " ")) else line
        if cur is not None:
            if cursor in cur and cur[cursor] != content:
                return {}, set(), f"hunks disagree on line {cursor} of {cur_path}"
            cur[cursor] = content
        cursor += 1
        remaining_old -= 1
        last_was_old = True
    return files, no_eof_newline, ""


def patch_applies(text: str) -> tuple[bool, str]:
    issues = lint_patch(text)
    if issues:
        return False, f"lint: {issues[0]}"
    files, no_eof_newline, err = _preimages(text)
    if err:
        return False, err
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel, known in files.items():
            fp = root / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            top = max(known) if known else 0
            body = "\n".join(known.get(i, f"__pad__{i}") for i in range(1, top + 1))
            if body and rel not in no_eof_newline:
                body += "\n"
            fp.write_text(body)
        patch_fp = root / "__submission__.patch"
        patch_fp.write_text(text if text.endswith("\n") else text + "\n")
        r = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", patch_fp.name],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            msg = (r.stderr or "").strip().splitlines()
            return False, f"git apply: {msg[-1][:140] if msg else 'failed'}"
    return True, "ok"


def _cat_path(tail: str) -> str | None:
    match = re.match(r"cat\s+(\S+)", tail)
    return match.group(1) if match else None


def _is_patchlike(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return "patch" in base or base.endswith(".diff")


def _creations(commands: list[str], path: str) -> list[tuple[str, str]]:
    base = re.escape(path.rsplit("/", 1)[-1])
    found = []
    for c in commands:
        if not re.search(rf"(^|/|\s){base}(\s|$|['\"])", c):
            continue
        if re.search(rf"git\s+diff[^|;]*>>?\s*\S*{base}", c) or re.search(
            rf"git\s+diff[^|;]*\|\s*tee\s+(-a\s+)?\S*{base}", c
        ):
            found.append(("git-diff", c))
        elif "<<" in c and re.search(rf"(>|tee\s+(-a\s+)?)\s*\S*{base}", c):
            found.append(("heredoc", c))
        elif re.search(rf"\bdiff\b[^|;]*>>?\s*\S*{base}", c):
            found.append(("diff-cmd", c))
        elif re.search(rf"\b(echo|printf)\b[^|;]*>>?\s*\S*{base}", c):
            found.append(("echo-write", c))
        elif re.search(rf"\bcp\s+\S+\s+\S*{base}", c) or re.search(
            rf"\bcat\s+\S+\s*>>?\s*\S*{base}", c
        ):
            found.append(("copy", c))
        elif re.search(rf">>?\s*\S*{base}", c) or re.search(rf"\btee\s+(-a\s+)?\S*{base}", c):
            found.append(("other-write", c))
    return found


def _fallback_prints_diff(tail: str) -> bool:
    if "||" not in tail:
        return False
    fb = tail.split("||", 1)[1]
    fb = re.sub(r'"[^"]*"|\'[^\']*\'', "", fb)
    for m in re.finditer(r"git\s+(?:-C\s+\S+\s+)?diff\b[^|&;]*", fb):
        redir = re.search(r">\s*(\S+)", m.group(0))
        if not redir:
            return True
        base = re.escape(redir.group(1).rsplit("/", 1)[-1])
        if re.search(rf"cat\s+\S*{base}", fb[m.end() :]):
            return True
    return False


def final_submit_issue(commands: list[str], marker: str) -> str:
    subs_i = [i for i, c in enumerate(commands) if marker in c]
    if not subs_i:
        return ""
    final = commands[subs_i[-1]]
    segments = final.split("&&")
    marker_i = next(i for i, s in enumerate(segments) if marker in s)
    tail = "&&".join(segments[marker_i + 1 :]).strip()
    prefix = "&&".join(segments[:marker_i]).strip()
    if not tail or tail.startswith("git "):
        return ""
    path = _cat_path(tail)
    if path is None:
        return "final submit is not a plain cat of a patch file"
    earlier = commands[: subs_i[-1]] + ([prefix] if prefix else [])
    created = _creations(earlier, path)
    if not _is_patchlike(path):
        referenced = any(path in c for c in earlier)
        if _fallback_prints_diff(tail) and not referenced and not created:
            return ""
        return "final submit cats source content instead of a patch file"
    if not created:
        if _fallback_prints_diff(tail):
            return ""
        return "submits a patch file it never created"
    kind, cmd = created[-1]
    if kind == "heredoc":
        body = _heredoc_body(cmd)
        if body is None:
            return "hand-written patch file is not a valid diff (unparseable heredoc)"
        ok, why = patch_applies(body)
        if not ok:
            return f"hand-written patch file would not apply ({why})"
        return ""
    if kind == "diff-cmd" and re.search(r"/dev/stdin|\s-\s", cmd):
        return "patch constructed by diffing against stdin"
    if kind == "copy":
        return "patch file created by copying file content, not a diff"
    return ""


def handbuilt_patch_issues(assistant_texts: list[str]) -> list[str]:
    for text in assistant_texts:
        for command in _commands(text):
            if _DIFF_STDIN_CMD_RE.search(command):
                return ["patch constructed by diffing against stdin"]
            body = _heredoc_body(command)
            if body is None:
                continue
            problems = lint_patch(body)
            if problems:
                return [f"hand-written patch file is not a valid diff ({problems[0]})"]
    return []
