from __future__ import annotations

import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("torch")

from model_validation import validate_worker as worker  # noqa: E402
from model_validation.dedup.gate import GateResult  # noqa: E402
from model_validation.dedup.verdict import Verdict  # noqa: E402


def _patch_pipeline(monkeypatch, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(worker, "make_ref", lambda repo, digest: object())
    monkeypatch.setattr(worker, "list_files", lambda ref: ["config.json"])
    monkeypatch.setattr(worker, "check_repo", lambda files: (True, ""))
    monkeypatch.setattr(worker, "safetensors_dtypes", lambda ref: {})
    monkeypatch.setattr(worker, "check_dtypes", lambda d: (True, ""))
    monkeypatch.setattr(worker, "download_config", lambda ref: str(tmp_path))
    monkeypatch.setattr(worker, "check_chat_template", lambda d, f: (True, ""))
    monkeypatch.setattr(worker, "check_genesis", lambda d, f: (True, ""))
    monkeypatch.setattr(worker, "check_architecture", lambda d: (True, ""))
    monkeypatch.setattr(worker, "download_full", lambda ref: str(tmp_path))
    monkeypatch.setattr(worker, "check_index", lambda d, f: (True, ""))


def _run(monkeypatch, tmp_path, result, enforce):
    _patch_pipeline(monkeypatch, tmp_path)
    calls = {}

    def fake_run(model_dir, model_uri, hotkey, repo, digest, coldkey=""):
        calls.update(hotkey=hotkey, coldkey=coldkey, repo=repo, digest=digest)
        return result

    monkeypatch.setattr(worker.dedup, "run", fake_run)
    monkeypatch.setattr(worker.config, "DEDUP_ENFORCE", enforce)
    out = worker.process_model("ns/m@" + "a" * 40, "hk", "ck")
    return out, calls


def test_pass_is_done_with_bare_summary(monkeypatch, tmp_path):
    res = GateResult(verdict=Verdict("PASS", None, "root", "TRAINED", ["TRAINED"], {"F": 0.7}))
    out, calls = _run(monkeypatch, tmp_path, res, enforce=True)
    assert out.state == "done" and out.result_summary == {"dedup": "pass"}
    assert calls == dict(hotkey="hk", coldkey="ck", repo="ns/m", digest="a" * 40)


def test_reject_enforced_is_duplicate_fault(monkeypatch, tmp_path):
    v = Verdict(
        "REJECT", "NOISE-COPY", "ns/king@b", "bulk", [], {"F": 0.05, "ancestor_hotkey": "hk2"}
    )
    out, _ = _run(monkeypatch, tmp_path, GateResult(verdict=v), enforce=True)
    assert out.state == "failed" and out.fault_code == "duplicate" and not out.retryable
    assert out.result_summary["duplicate_of"] == "ns/king@b"
    assert out.result_summary["duplicate_of_hotkey"] == "hk2"
    assert out.result_summary["metrics"]["F"] == 0.05


def test_own_copy_enforced_is_a_strike_not_a_duplicate_block(monkeypatch, tmp_path):
    v = Verdict("REJECT", "OWN-COPY", "ns/mine@old", "identical", [], {"ancestor_hotkey": "hk"})
    out, _ = _run(monkeypatch, tmp_path, GateResult(verdict=v), enforce=True)
    assert out.state == "failed" and out.fault_code == "duplicate_own"
    assert out.result_summary == {
        "dedup": "reject",
        "reason": "OWN-COPY",
        "duplicate_of": "ns/mine@old",
        "notes": [],
        "duplicate_of_hotkey": "hk",
        "own_model": True,
    }


def test_reject_in_shadow_is_done_and_silent(monkeypatch, tmp_path):
    v = Verdict("REJECT", "NOISE-COPY", "ns/king@b", "bulk", [], {"F": 0.05})
    out, _ = _run(monkeypatch, tmp_path, GateResult(verdict=v), enforce=False)
    assert out.state == "done" and out.result_summary == {"dedup": "pass"}


def test_infra_error_is_retryable(monkeypatch, tmp_path):
    out, _ = _run(monkeypatch, tmp_path, GateResult(infra_error="opensearch down"), enforce=True)
    assert out.state == "failed" and out.retryable and out.fault_code == "dedup_failed"
