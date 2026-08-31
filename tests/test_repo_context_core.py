from __future__ import annotations

import io
import json
import re
import tarfile
import time

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from albedo_config import RepoContextSettings
from repo_context_service.command_search import ParseFailure, parse_search, run_search
from repo_context_service.core import (
    _NEGATIVE_TTL_SECONDS,
    _TRANSIENT_TTL_SECONDS,
    GroundingContext,
    RepoContextService,
    _filter_listing,
    _first_command,
    _is_permanent_github_error,
    _NotFound,
    _SnapshotTooLarge,
    parse_instance,
)

FULL_SHA = "abcdef1234567890abcdef1234567890abcdef12"
SNAPSHOT_KEY = f"o__r__{FULL_SHA[:12]}"


def make_settings(tmp_path, **overrides) -> RepoContextSettings:
    values = {"cache_dir": str(tmp_path / "cache")}
    values.update(overrides)
    return RepoContextSettings(_env_file=None, **values)


def make_service(tmp_path, **overrides) -> RepoContextService:
    return RepoContextService(make_settings(tmp_path, **overrides))


def make_snapshot(service, files: dict[str, str], listing: list[str] | None = None):
    snapshot = service.cache_dir / "snapshots" / SNAPSHOT_KEY
    for rel, text in files.items():
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (snapshot / ".albedo-listing.json").write_text(json.dumps(listing or sorted(files)))
    (snapshot / ".albedo-repo-context-done").write_text("")
    return snapshot


def test_cache_dir_is_required(tmp_path):
    with pytest.raises(ValueError):
        RepoContextService(RepoContextSettings(_env_file=None, cache_dir=""))


def test_parse_instance_formats():
    swe = parse_instance("swe-zero", "azure__secrets-store-csi-driver-provider-azure-466")
    assert swe.owner == "azure"
    assert (swe.repo, swe.pr) == ("secrets-store-csi-driver-provider-azure", "466")
    mini = parse_instance("mini-coder", "seperman__deepdiff.4b8fa12__m1")
    assert (mini.owner, mini.repo, mini.commit) == ("seperman", "deepdiff", "4b8fa12")
    goat = parse_instance("mini-coder", "arp242__goatcounter.854b1dd2.lm_modify__1vcxllzm")
    assert (goat.owner, goat.repo, goat.commit) == ("arp242", "goatcounter", "854b1dd2")
    rs = parse_instance("mini-coder-rs", "arp242__goatcounter.854b1dd2.lm_modify__1vcxllzm")
    assert (rs.owner, rs.repo, rs.commit) == ("arp242", "goatcounter", "854b1dd2")
    hero = parse_instance("swe-hero", "pandas-dev__pandas-dbf8aaf4a3f3b41e5c1a402473df5da43813948f")
    assert (hero.owner, hero.repo, hero.pr) == ("pandas-dev", "pandas", None)
    assert hero.commit == "dbf8aaf4a3f3b41e5c1a402473df5da43813948f"
    ost = parse_instance("open-swe-traces", "python-attrs__attrs-770")
    assert (ost.owner, ost.repo, ost.pr) == ("python-attrs", "attrs", "770")
    assert parse_instance("swe-zero", "owner__repo-notanumber") is None
    assert parse_instance("mini-coder", "noseparator") is None
    assert parse_instance("mini-coder", "owner__repo.ZZZZZZ") is None


def test_first_command_takes_first_block_only():
    text = "THOUGHT: x\n\n```bash\nls -la\n```\n```bash\nrm -rf /\n```"
    assert _first_command(text) == "ls -la"
    assert _first_command("no block") == ""


def test_iid_lookup_from_manifest(tmp_path):
    manifest = {
        "sources": [
            {
                "name": "swe-zero",
                "shards": [
                    {
                        "path": "swe-zero/data/train-00000.parquet",
                        "rows": 2,
                        "rows_meta": [{"iid": "o__r-1", "asst": 3}, {"iid": "o2__r2-5", "asst": 4}],
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    service = make_service(tmp_path, dataset_manifest_path=str(manifest_path))
    assert service._iid_for("swe-zero/data/train-00000.parquet", 1) == ("swe-zero", "o2__r2-5")
    assert service._iid_for("swe-zero/data/train-00000.parquet", 9) is None
    assert service._iid_for("mini-coder/data/train-00000.parquet", 0) is None

    bad_hash = make_service(
        tmp_path, dataset_manifest_path=str(manifest_path), dataset_manifest_hash="0" * 64
    )
    assert bad_hash._iid_for("swe-zero/data/train-00000.parquet", 1) is None


def test_resolve_sha_renamed_repo_and_cache(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    calls = []

    def fake_github_json(path):
        calls.append(path)
        if path == "/repos/old/name/pulls/12":
            raise _NotFound(path)
        if path == "/repos/old/name":
            return {"full_name": "new/name"}
        if path == "/repos/new/name/pulls/12":
            return {"base": {"sha": FULL_SHA}}
        raise AssertionError(path)

    monkeypatch.setattr(service, "_github_json", fake_github_json)
    ref = parse_instance("swe-zero", "old__name-12")
    assert service._resolve_sha(ref) == ("new", "name", FULL_SHA)
    assert service._resolve_sha(ref) == ("new", "name", FULL_SHA)
    assert len(calls) == 3


def test_resolve_sha_negative_cache_ttl(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    calls = []

    def failing(path):
        calls.append(path)
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_github_json", failing)
    ref = parse_instance("swe-zero", "own__repo-3")
    assert service._resolve_sha(ref) is None
    assert service._resolve_sha(ref) is None
    assert len(calls) == 1

    cache_file = service.cache_dir / "shas" / "own__repo-3.json"
    stale = json.loads(cache_file.read_text())
    stale["failed_at"] = time.time() - 2 * 24 * 3600
    cache_file.write_text(json.dumps(stale))
    assert service._resolve_sha(ref) is None
    assert len(calls) == 2


def test_is_permanent_github_error_classification():
    assert _is_permanent_github_error(_NotFound("/x")) is True
    for code in (400, 410, 422, 451):
        resp = httpx.Response(code, request=httpx.Request("GET", "https://api.github.com/x"))
        err = httpx.HTTPStatusError("e", request=resp.request, response=resp)
        assert _is_permanent_github_error(err)
    # rate-limit / transport errors are NOT permanent -> only short transient caching
    assert _is_permanent_github_error(RuntimeError("github 403 for /x")) is False
    assert _is_permanent_github_error(httpx.ConnectError("boom")) is False


def test_resolve_sha_transient_failure_uses_short_ttl(tmp_path, monkeypatch):
    """A rate-limit/transport failure is cached only for the transient TTL, so a throttled or
    unauthenticated run self-heals on the next pass instead of staying dead for 24h."""
    service = make_service(tmp_path)
    calls = []

    def failing(path):
        calls.append(path)
        raise RuntimeError("github 403 for " + path)

    monkeypatch.setattr(service, "_github_json", failing)
    ref = parse_instance("swe-zero", "own__repo-3")
    assert service._resolve_sha(ref) is None
    cache_file = service.cache_dir / "shas" / "own__repo-3.json"
    assert json.loads(cache_file.read_text())["kind"] == "transient"

    # aged just past the transient TTL but far inside the 24h negative TTL -> must retry
    entry = json.loads(cache_file.read_text())
    entry["failed_at"] = time.time() - (_TRANSIENT_TTL_SECONDS + 60)
    cache_file.write_text(json.dumps(entry))
    assert service._resolve_sha(ref) is None
    assert len(calls) == 2


def test_resolve_sha_permanent_failure_uses_long_ttl(tmp_path, monkeypatch):
    """A genuine 404 (PR/commit gone, rename lookup also 404) stays cached for the full negative
    TTL and is not retried on every request."""
    service = make_service(tmp_path)
    calls = []

    def failing(path):
        calls.append(path)
        raise _NotFound(path)

    monkeypatch.setattr(service, "_github_json", failing)
    ref = parse_instance("swe-zero", "gone__repo-9")
    assert service._resolve_sha(ref) is None
    cache_file = service.cache_dir / "shas" / "gone__repo-9.json"
    assert json.loads(cache_file.read_text())["kind"] == "permanent"
    settled = len(calls)

    # aged past the transient TTL but within the 24h negative TTL -> still suppressed
    entry = json.loads(cache_file.read_text())
    entry["failed_at"] = time.time() - (_TRANSIENT_TTL_SECONDS + 3600)
    cache_file.write_text(json.dumps(entry))
    assert service._resolve_sha(ref) is None
    assert len(calls) == settled

    # aged past the full negative TTL -> retry
    entry["failed_at"] = time.time() - (_NEGATIVE_TTL_SECONDS + 60)
    cache_file.write_text(json.dumps(entry))
    assert service._resolve_sha(ref) is None
    assert len(calls) > settled


def test_resolve_sha_legacy_failure_entry_self_heals(tmp_path, monkeypatch):
    """Failure entries written before ``kind`` existed (the poisoned ones from the unauthenticated
    run) expire at the transient TTL, so a fixed/authenticated rerun re-resolves them."""
    service = make_service(tmp_path)
    cache_dir = service.cache_dir / "shas"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "own__repo-5.json").write_text(
        json.dumps({"error": "boom", "failed_at": time.time() - (_TRANSIENT_TTL_SECONDS + 60)})
    )

    calls = []

    def ok(path):
        calls.append(path)
        return {"base": {"sha": FULL_SHA}}

    monkeypatch.setattr(service, "_github_json", ok)
    ref = parse_instance("swe-zero", "own__repo-5")
    assert service._resolve_sha(ref) == ("own", "repo", FULL_SHA)
    assert calls  # retried instead of honoring the stale legacy failure


def _tar_bytes() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name, data=b"", **attrs):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            for key, value in attrs.items():
                setattr(info, key, value)
            tar.addfile(info, io.BytesIO(data) if info.isreg() else None)

        add("repo-abc/src/app.py", b"print('hi')\n")
        add("repo-abc/link.py", type=tarfile.SYMTYPE, linkname="/etc/passwd")
        add("repo-abc/../evil.txt", b"evil")
        add("/abs.txt", b"evil")
        big = tarfile.TarInfo("repo-abc/huge.bin")
        big.size = 3 * 1024 * 1024
        tar.addfile(big, io.BytesIO(b"0" * big.size))
    return buffer.getvalue()


def test_extract_tarball_sanitizes_members(tmp_path):
    service = make_service(tmp_path)
    tar_path = tmp_path / "snap.tar.gz"
    tar_path.write_bytes(_tar_bytes())
    dest = tmp_path / "out"
    dest.mkdir()
    listing, extracted_bytes = service._extract_tarball(tar_path, dest)
    assert listing == ["huge.bin", "src/app.py"]
    assert extracted_bytes == len(b"print('hi')\n")
    assert (dest / "src/app.py").read_text() == "print('hi')\n"
    assert not (tmp_path / "evil.txt").exists()
    assert not (dest / "link.py").exists()
    assert not (dest / "huge.bin").exists()


def test_repo_block_contents_and_containment(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    snapshot = make_snapshot(
        service,
        {"src/app.py": "APP CONTENT\n", "README.md": "readme\n"},
        listing=["README.md", "src/app.py", "link.py"],
    )
    (snapshot / "link.py").symlink_to("/etc/passwd")
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    command = (
        "```bash\nfind . -type f && cat src/app.py .env /etc/passwd ../../secret missing/file.py "
        "&& sed -n s/a/b/ src/app.py && grep -r foo src/\n```"
    )
    result = service.repo_context_for_instance("swe-zero", "o__r-12", command)
    assert result.kind == "repo"
    block = result.context
    assert "APP CONTENT" in block
    assert "./src/app.py" in block
    not_present = block.split("FILES NOT PRESENT")[1]
    assert "- .env" in not_present
    assert "- /etc/passwd" in not_present
    assert "- ../../secret" in not_present
    assert "- missing/file.py" in not_present
    assert "s/a/b" not in not_present
    assert "- src" not in not_present
    assert "- ." not in not_present.split("- .env")[0]
    assert service._read_snapshot_file(snapshot, "link.py") is None
    assert "root:" not in block


def test_commands_cannot_reach_other_snapshots_or_host(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"src/app.py": "APP CONTENT\n"}, listing=["src/app.py"])
    other = service.cache_dir / "snapshots" / "other__repo__000000000000"
    (other / "conf").mkdir(parents=True)
    (other / "conf/secret.py").write_text("OTHER SNAPSHOT SECRET\n")
    (service.cache_dir / "shas").mkdir(parents=True, exist_ok=True)
    (service.cache_dir / "shas" / "token.json").write_text('{"sha": "HOST SECRET"}')
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    command = (
        "```bash\ncat ../other__repo__000000000000/conf/secret.py conf/secret.py "
        "../../shas/token.json /etc/passwd\n```"
    )
    block = service.repo_context_for_instance("swe-zero", "o__r-1", command).context
    assert "OTHER SNAPSHOT SECRET" not in block
    assert "HOST SECRET" not in block
    assert "root:" not in block
    snapshot = service.cache_dir / "snapshots" / SNAPSHOT_KEY
    escape = "../other__repo__000000000000/conf/secret.py"
    assert service._read_snapshot_file(snapshot, escape) is None
    assert service._read_snapshot_file(snapshot, "../../shas/token.json") is None


def test_filter_listing_and_caps(tmp_path, monkeypatch):
    listing = [f"pkg/mod_{i}.py" for i in range(5)] + ["docs/guide.md"]
    kept, filtered = _filter_listing(listing, "find . -name '*.py'")
    assert filtered is True
    assert kept == [f"pkg/mod_{i}.py" for i in range(5)]

    service = make_service(tmp_path, max_paths=3)
    make_snapshot(service, {p: "x" for p in listing}, listing=listing)
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    result = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\nfind $(pwd) -name '*.py'\n```"
    )
    assert "... (+2 more matching files)" in result.context

    nothing = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\nfind $(pwd) -name '*.zig'\n```"
    )
    assert nothing.kind == "repo"
    assert "no files in this repository match" in nothing.context
    assert "./pkg/mod_0.py" not in nothing.context


def test_a_ranged_view_piped_through_cat_n_is_computable():
    listing = ["a.py"]
    read = {"a.py": "l1\nl2\nl3\nl4\nl5\n"}.get
    out = run_search(parse_search("sed -n '2,4p' a.py | cat -n"), read, listing).output
    assert out == "     1\tl2\n     2\tl3\n     3\tl4"
    assert isinstance(parse_search("cat a.py | cat -A"), ParseFailure)


def test_a_quoted_pattern_is_not_read_as_shell_syntax():
    assert not isinstance(parse_search('grep -n "a || b" a.py'), ParseFailure)
    assert not isinstance(parse_search("grep -n 'x; y' a.py"), ParseFailure)
    assert isinstance(parse_search("cat a.py; rm -rf /"), ParseFailure)


def test_a_long_listing_is_derived_from_the_snapshot():
    listing = ["a.py", "pkg/b.py"]
    read = {"a.py": "L1\nL2\n", "pkg/b.py": "X\n"}.get
    rows = run_search(parse_search("ls -l"), read, listing).output.split("\n")
    assert rows[0] == "total 8"
    assert re.fullmatch(r"-rw-r--r-- 1 root root\s+6 \w{3} +\d+ [\d:]+ a\.py", rows[1]), rows[1]
    assert re.fullmatch(r"drwxr-xr-x 2 root root\s+4096 \w{3} +\d+ [\d:]+ pkg", rows[2]), rows[2]
    assert run_search(parse_search("ls -l a.py"), read, listing).output.endswith(" a.py")
    assert isinstance(parse_search("ls -lh"), ParseFailure)


_FIND_TREE = [
    "docs/guide.md",
    "node_modules/x/y.py",
    "src/a.py",
    "src/util/b.py",
    "src/util/deep/c.py",
    "tests/t_one.py",
]


def _find(cmd: str) -> list[str]:
    plan = parse_search(cmd)
    assert not isinstance(plan, ParseFailure), f"{cmd} -> {plan}"
    return run_search(plan, {p: "x\n" for p in _FIND_TREE}.get, _FIND_TREE).output.split("\n")


def test_find_answers_without_needing_a_grep_downstream():
    assert _find("find . -name '*.py'") == [
        "./node_modules/x/y.py",
        "./src/a.py",
        "./src/util/b.py",
        "./src/util/deep/c.py",
        "./tests/t_one.py",
    ]
    assert _find("find . -name '*.py' | head -2") == ["./node_modules/x/y.py", "./src/a.py"]
    assert _find("find . -name '*.py' | wc -l") == ["5"]
    assert _find("find . -name '*.py' | grep util") == ["./src/util/b.py", "./src/util/deep/c.py"]


def test_find_depth_and_directory_predicates():
    assert _find("find . -type d") == [
        ".",
        "./docs",
        "./node_modules",
        "./node_modules/x",
        "./src",
        "./src/util",
        "./src/util/deep",
        "./tests",
    ]
    assert _find("find . -maxdepth 1 -type d") == [
        ".",
        "./docs",
        "./node_modules",
        "./src",
        "./tests",
    ]
    assert _find("find src -maxdepth 2 -name '*.py'") == ["src/a.py", "src/util/b.py"]
    assert _find("find . -name 'util' -type d") == ["./src/util"]


def test_find_excludes_negated_and_pruned_subtrees():
    expected = ["./src/a.py", "./src/util/b.py", "./src/util/deep/c.py", "./tests/t_one.py"]
    assert _find("find . -name '*.py' -not -path '*/node_modules/*'") == expected
    assert _find("find . -path './node_modules' -prune -o -name '*.py' -print") == expected


def test_find_or_is_honoured_only_within_one_predicate():
    assert _find("find . -name '*.py' -o -name '*.md'") == [
        "./docs/guide.md",
        "./node_modules/x/y.py",
        "./src/a.py",
        "./src/util/b.py",
        "./src/util/deep/c.py",
        "./tests/t_one.py",
    ]
    assert isinstance(parse_search("find . -path '*util*' -o -name '*.py'"), ParseFailure)


def test_find_exec_ls_renders_one_long_row_per_hit():
    assert _find("find . -name '*.py' -exec ls -la {} \\;") == [
        "-rw-r--r-- 1 root root        2 Jan  3 20:00 ./node_modules/x/y.py",
        "-rw-r--r-- 1 root root        2 Jan  3 20:00 ./src/a.py",
        "-rw-r--r-- 1 root root        2 Jan  3 20:00 ./src/util/b.py",
        "-rw-r--r-- 1 root root        2 Jan  3 20:00 ./src/util/deep/c.py",
        "-rw-r--r-- 1 root root        2 Jan  3 20:00 ./tests/t_one.py",
    ]
    assert _find("find src -type f -exec ls {} \\;") == [
        "src/a.py",
        "src/util/b.py",
        "src/util/deep/c.py",
    ]


def test_find_still_refuses_what_it_cannot_stand_behind():
    for cmd in (
        "find . -name '*.py' | awk '{print $1}'",
        "find . -name '*.py' -exec cat {} \\;",
        "find . -name '*.py' -exec ls -lh {} \\;",
        "find . -type d -exec ls -la {} \\;",
        "find $(pwd) -name '*.py'",
        "find . -newer setup.py",
    ):
        assert isinstance(parse_search(cmd), ParseFailure), cmd


def test_a_missing_read_target_reports_the_shell_error_it_would_print():
    listing = ["a.py"]
    read = {"a.py": "l1\n"}.get
    expected = {
        "cat gone.py": "cat: gone.py: No such file or directory",
        "nl -ba gone.py": "nl: gone.py: No such file or directory",
        "head -5 gone.py": "head: cannot open 'gone.py' for reading: No such file or directory",
        "tail -5 gone.py": "tail: cannot open 'gone.py' for reading: No such file or directory",
        "sed -n '1,3p' gone.py": "sed: can't read gone.py: No such file or directory",
    }
    for cmd, message in expected.items():
        result = run_search(parse_search(cmd), read, listing)
        assert not isinstance(result, ParseFailure), cmd
        assert result.output == message
        assert result.missing == ["gone.py"]
    assert run_search(parse_search("cat a.py"), read, listing).output == "l1"


def test_grep_flags_that_change_nothing_on_text_are_accepted():
    listing = ["a.py"]
    read = {"a.py": "alpha\nbeta\n"}.get
    assert run_search(parse_search("grep -an alpha a.py"), read, listing).output == "1:alpha"
    assert run_search(parse_search("grep -n alpha a.py | cat"), read, listing).output == "1:alpha"
    assert isinstance(parse_search("grep -n alpha a.py | cat -v"), ParseFailure)


def test_a_pattern_grep_itself_rejects_is_reported_not_refused():
    listing = ["a.py"]
    read = {"a.py": "alpha\n"}.get
    assert (
        run_search(parse_search('grep -n "self.get\\(" a.py'), read, listing).output
        == "grep: Unmatched ( or \\("
    )
    assert (
        run_search(parse_search('grep -rn "reverse\\|[::-1]" .'), read, listing).output
        == "grep: Invalid range end"
    )
    quiet = run_search(parse_search('grep -rn "reverse\\|[::-1]" . 2>/dev/null'), read, listing)
    assert quiet.output == "" and quiet.empty is True
    assert isinstance(
        run_search(parse_search('rg -n "self.get\\(" a.py'), read, listing), ParseFailure
    )


def test_paths_built_at_container_setup_are_not_claimed_absent():
    listing = ["a.py"]
    read = {"a.py": "l1\n"}.get
    for cmd in (
        "cat node_modules/.bin/jest",
        "head -5 build/build_config.json",
        "cat faker/providers/ssn.pyc",
        "head -20 .venv/lib/python3.11/site.py",
    ):
        assert isinstance(run_search(parse_search(cmd), read, listing), ParseFailure), cmd


def test_echo_prints_its_argument_and_refuses_escapes():
    assert run_search(parse_search("echo hello world"), {}.get, []).output == "hello world"
    assert run_search(parse_search('echo "a  b"'), {}.get, []).output == "a  b"
    assert isinstance(parse_search("echo -e 'a\\nb'"), ParseFailure)


def test_output_sent_to_a_file_is_not_reported_as_printed():
    for cmd in (
        "cat a.py > out.txt",
        "ls -l > listing.txt",
        "grep -n L a.py >> hits.txt",
        "echo hi > f.txt",
    ):
        assert isinstance(parse_search(cmd), ParseFailure), cmd
    assert not isinstance(parse_search("grep -n L a.py 2>/dev/null"), ParseFailure)


def test_renderable_chain_stages_are_attached_as_evidence(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "L1\nL2\nL3\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    mixed = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\nsed -n '1,2p' a.py && python run.py\n```"
    )
    assert "CHAIN STAGES ALREADY EXECUTED" in mixed.context
    assert "$ sed -n '1,2p' a.py\nL1\nL2" in mixed.context
    assert mixed.exact_output is None
    assert not mixed.context.lstrip().startswith("COMMAND OUTPUT —")

    nothing = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ncd /testbed && python run.py\n```"
    )
    assert "CHAIN STAGES ALREADY EXECUTED" not in nothing.context


def test_exact_output_is_offered_only_when_it_can_be_stood_behind(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"src/app.py": "APP CONTENT\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    computed = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ncat src/app.py\n```"
    )
    assert computed.exact_output == "APP CONTENT"

    listing_only = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\npython src/app.py\n```"
    )
    assert listing_only.context is not None
    assert listing_only.exact_output is None


def test_contents_survive_huge_listing(tmp_path, monkeypatch):
    listing = [f"pkg/module_{i:05}.py" for i in range(6000)]
    service = make_service(tmp_path, max_context_chars=30000)
    make_snapshot(
        service, {"src/target.py": "TARGET CONTENT " * 100}, listing=listing + ["src/target.py"]
    )
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))
    block = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\npython src/target.py\n```"
    ).context
    assert len(block) <= 30000 + len("\n... (truncated)")
    assert "TARGET CONTENT" in block
    assert "more matching files)" in block
    assert block.index("TARGET CONTENT") > block.index("more matching files")


def _write_trajectory_shard(root, turns):
    shard = root / "swe-zero" / "data" / "train-00000.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"instance_id": ["o__r-1"], "messages": [turns]})
    pq.write_table(table, shard)


def test_iid_resolved_from_parquet_without_manifest(tmp_path):
    root = tmp_path / "dataset"
    _write_trajectory_shard(root, [{"role": "user", "content": "task"}])
    service = make_service(tmp_path, dataset_root=str(root))
    assert service._iid_for("swe-zero/data/train-00000.parquet", 0) == ("swe-zero", "o__r-1")
    assert service._iid_for("swe-zero/data/train-00000.parquet", 9) is None
    assert service._iid_for("swe-zero/data/train-99999.parquet", 0) is None


def test_trajectory_fallback_when_repo_unavailable(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    _write_trajectory_shard(
        root,
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "THOUGHT: a\n\n```bash\nls src\n```"},
            {"role": "user", "content": "Observation: obs-1"},
            {"role": "assistant", "content": "```bash\ncat src/x.py\n```"},
            {"role": "user", "content": "Observation: obs-2"},
            {
                "role": "assistant",
                "content": "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                " && git add -A && git diff --cached\n```",
            },
            {"role": "user", "content": "Observation: GOLD SOLUTION DIFF"},
        ],
    )
    service = make_service(tmp_path, dataset_root=str(root))
    monkeypatch.setattr(
        service,
        "repo_context_for_instance",
        lambda *a: GroundingContext(context=None, kind="none", reason="snapshot_unavailable"),
    )
    result = service.context_for("swe-zero/data/train-00000.parquet:0:0", "```bash\npwd\n```")
    assert result.kind == "trajectory"
    assert "REFERENCE STEP 1" in result.context
    assert "obs-1" in result.context and "obs-2" in result.context
    assert "THOUGHT" not in result.context
    assert "GOLD SOLUTION DIFF" not in result.context
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in result.context

    later = service.context_for("swe-zero/data/train-00000.parquet:0:1", "```bash\npwd\n```")
    assert "obs-1" not in later.context and "obs-2" in later.context


def test_fallback_chain_to_none_and_never_raises(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    result = service.context_for("swe-zero/data/train-00000.parquet:0:0", "```bash\nls\n```")
    assert result == GroundingContext(context=None, kind="none", reason="iid_unresolved")

    assert service.context_for("not a sample id", "x").kind == "none"

    monkeypatch.setattr(
        service, "_context_for", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert service.context_for("swe-zero/data/train-00000.parquet:0:0", "x").kind == "none"


def test_no_bypass_even_with_poisoned_listing(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    host_secret = tmp_path / "host_secret.txt"
    host_secret.write_text("HOST SECRET\n")
    poisoned = [
        "../../../host_secret.txt",
        str(host_secret),
        "evil/host_secret.txt",
        "src/app.py",
    ]
    snapshot = make_snapshot(service, {"src/app.py": "APP CONTENT\n"}, listing=poisoned)
    (snapshot / "evil").symlink_to(tmp_path)
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    command = (
        f"```bash\ncat ../../../host_secret.txt evil/host_secret.txt src/app.py {host_secret}\n```"
    )
    block = service.repo_context_for_instance("swe-zero", "o__r-1", command).context
    assert "HOST SECRET" not in block
    assert "APP CONTENT" in block
    for entry in poisoned[:3]:
        assert service._read_snapshot_file(snapshot, entry) is None


def test_extract_tarball_skips_metadata_collisions(tmp_path):
    service = make_service(tmp_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in [
            ("repo-abc/.albedo-listing.json", b'["fake/entry.py"]'),
            ("repo-abc/.albedo-repo-context-done", b'{"bytes": 0}'),
            ("repo-abc/real.py", b"ok"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tar_path = tmp_path / "meta.tar.gz"
    tar_path.write_bytes(buffer.getvalue())
    dest = tmp_path / "out"
    dest.mkdir()
    listing, _ = service._extract_tarball(tar_path, dest)
    assert listing == ["real.py"]
    assert not (dest / ".albedo-listing.json").exists()


def test_cache_limit_clears_snapshots_dir(tmp_path, monkeypatch):
    service = make_service(tmp_path, max_cache_gb=1024 / 1024**3)
    old_snapshot = make_snapshot(service, {"src/app.py": "x"})
    marker = old_snapshot / ".albedo-repo-context-done"
    marker.write_text(json.dumps({"bytes": 2048}))

    def fake_download(owner, repo, sha, dest):
        dest.write_bytes(_tar_bytes())

    monkeypatch.setattr(service, "_download_tarball", fake_download)
    new_sha = "0" * 40
    fresh = service._ensure_snapshot("other", "repo", new_sha)
    assert fresh is not None and (fresh / "src/app.py").exists()
    assert not old_snapshot.exists()

    roomy = make_service(tmp_path, cache_dir=str(tmp_path / "cache2"), max_cache_gb=60.0)
    kept = make_snapshot(roomy, {"src/app.py": "x"})
    monkeypatch.setattr(roomy, "_download_tarball", fake_download)
    assert roomy._ensure_snapshot("other", "repo", new_sha) is not None
    assert kept.exists()


def test_prefetch_dedupes_and_skips_already_downloaded(tmp_path, monkeypatch):
    manifest = {
        "sources": [
            {
                "name": "swe-zero",
                "shards": [
                    {
                        "path": "swe-zero/data/train-00000.parquet",
                        "rows": 3,
                        "rows_meta": [{"iid": "o__r-1"}, {"iid": "o__r-1"}, {"iid": "o__r-2"}],
                    }
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    service = make_service(tmp_path, dataset_manifest_path=str(manifest_path))
    make_snapshot(service, {"src/app.py": "x"})

    new_sha = "1" * 40
    monkeypatch.setattr(
        service,
        "_resolve_sha",
        lambda ref: ("o", "r", FULL_SHA if ref.instance_id == "o__r-2" else new_sha),
    )
    downloads = []

    def fake_download(owner, repo, sha, dest):
        downloads.append(sha)
        dest.write_bytes(_tar_bytes())

    monkeypatch.setattr(service, "_download_tarball", fake_download)
    summary = service.prefetch(
        [
            "swe-zero/data/train-00000.parquet:0:0",
            "swe-zero/data/train-00000.parquet:1:2",
            "swe-zero/data/train-00000.parquet:2:0",
            "garbage",
        ]
    )
    assert summary == {"samples": 4, "instances": 2, "ready": 2}
    assert downloads == [new_sha]

    again = service.prefetch(
        ["swe-zero/data/train-00000.parquet:0:0", "swe-zero/data/train-00000.parquet:2:0"]
    )
    assert again["ready"] == 2
    assert downloads == [new_sha]


def test_oversized_snapshot_failure_is_cached(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    calls = []

    def too_large(owner, repo, sha, dest):
        calls.append(sha)
        raise _SnapshotTooLarge("tarball exceeds limit")

    monkeypatch.setattr(service, "_download_tarball", too_large)
    assert service._ensure_snapshot("o", "r", FULL_SHA) is None
    assert service._ensure_snapshot("o", "r", FULL_SHA) is None
    assert len(calls) == 1


def test_a_chain_still_grounds_the_stages_after_one_it_cannot_render(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "L1\nL2\nL3\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    # git first: the whole chain used to be discarded because the loop stopped here
    after_git = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ngit show HEAD --stat && sed -n '1,2p' a.py\n```"
    )
    assert "CHAIN STAGES ALREADY EXECUTED" in after_git.context
    assert "$ sed -n '1,2p' a.py\nL1\nL2" in after_git.context
    assert after_git.exact_output is None

    # and a stage between two ungroundable ones is still reported
    sandwiched = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\npwd && sed -n '3p' a.py && python run.py\n```"
    )
    assert "$ sed -n '3p' a.py\nL3" in sandwiched.context


def test_a_ranged_read_accepts_one_line_and_a_pipe_stage():
    listing = ["m.py"]
    src = "def a():\n    x = 1\n    return x\n\ndef b():\n    return 2\n"
    read = {"m.py": src}.get

    def out(cmd):
        return run_search(parse_search(cmd), read, listing).output

    assert out("sed -n '3p' m.py") == "    return x"
    assert out("sed -n '2,3p' m.py") == "    x = 1\n    return x"
    assert out("nl -ba m.py | sed -n '2,3p'") == "     2\t    x = 1\n     3\t    return x"
    assert out("cat m.py | sed -n '1,2p'") == "def a():\n    x = 1"
    # only numeric addresses are modelled: three corpus commands use a regex range,
    # so resolving pattern addresses is not worth the machinery
    assert isinstance(parse_search("sed -n '/def a/,/return x/p' m.py"), ParseFailure)
    assert isinstance(parse_search("cat m.py | sed -n '/a/,/b/p'"), ParseFailure)
    assert isinstance(parse_search("sed -n '2,3d' m.py"), ParseFailure)


def test_a_search_over_an_unreadable_file_is_never_reported_as_no_match(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    make_snapshot(service, {"a.py": "    def target():\n        pass\n"})
    monkeypatch.setattr(service, "_resolve_sha", lambda ref: ("o", "r", FULL_SHA))

    # a sed -i form we deliberately refuse to model, so the path is left dirty
    edited = [
        {"role": "assistant", "content": "```bash\nsed -i '/pass/d' a.py\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"},
    ]
    dirty = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ngrep -n target a.py\n```", edited
    )
    assert "matched NOTHING" not in dirty.context
    assert dirty.exact_output is None

    clean = service.repo_context_for_instance(
        "swe-zero", "o__r-1", "```bash\ngrep -n target a.py\n```"
    )
    assert clean.exact_output == "1:    def target():"


def test_posix_classes_and_stderr_redirects_do_not_silence_a_grep():
    listing = ["m.py"]
    read = {"m.py": "class A:\n    def go(self):\n        pass\n"}.get
    hit = run_search(parse_search("grep -n '[[:space:]]*def' m.py"), read, listing)
    assert hit.output == "2:    def go(self):"
    assert not hit.empty
    # 2>&1 is a redirect; the & in it used to be read as a control operator
    assert not isinstance(parse_search("grep -rn target m.py 2>&1"), ParseFailure)
    assert not isinstance(parse_search("grep -rn target m.py 2>&1 | head -20"), ParseFailure)
    # sending stdout elsewhere is still not something we can claim was printed
    assert isinstance(parse_search("grep -n target m.py 1>&2"), ParseFailure)
