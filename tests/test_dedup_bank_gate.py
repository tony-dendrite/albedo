from __future__ import annotations

import pytest

pytest.importorskip("opensearchpy")
pytest.importorskip("torch")

from model_validation.dedup import bank, gate  # noqa: E402
from model_validation.dedup.verdict import Verdict  # noqa: E402


class _Indices:
    def __init__(self):
        self.created = {}

    def exists(self, index):
        return index in self.created

    def create(self, index, body):
        self.created[index] = body


class _Client:
    def __init__(self, hits=None, count=1, docs=None):
        self.indices = _Indices()
        self.hits = hits or []
        self.count_value = count
        self.docs = docs or {}
        self.indexed = []
        self.searches = []

    def search(self, index, body):
        self.searches.append(body)
        return {"hits": {"hits": self.hits}}

    def count(self, index, body):
        return {"count": self.count_value}

    def index(self, index, id, body):
        self.indexed.append((id, body))

    def mget(self, index, body):
        return {
            "docs": [
                {"_id": i, "found": i in self.docs, "_source": self.docs.get(i)}
                for i in body["ids"]
            ]
        }


DOC = {
    "model_uri": "ns/m@" + "a" * 40,
    "arch_key": "arch",
    "ws_version": "ws-canon-v2",
    "key_id": "k1",
    "tensors_hash": "h" * 64,
    "sketch_vec": [0.0] * 1024,
    "tensors": [],
}


def _patch(monkeypatch, client):
    monkeypatch.setattr(bank, "get_client", lambda: client)
    return client


def test_ensure_index_creates_knn_mapping(monkeypatch):
    c = _patch(monkeypatch, _Client())
    name = bank.ensure_index()
    body = c.indices.created[name]
    assert body["settings"]["index"]["knn"] is True
    props = body["mappings"]["properties"]
    assert props["sketch_vec"] == {"type": "knn_vector", "dimension": 1024}
    assert props["tensors"]["enabled"] is False


def test_put_doc_sets_status_and_verdict(monkeypatch):
    c = _patch(monkeypatch, _Client())
    bank.put_doc(
        DOC, status=bank.STATUS_AUDIT, hotkey="hk", verdict={"status": "REJECT"}, is_root=False
    )
    doc_id, body = c.indexed[0]
    assert doc_id == DOC["model_uri"]
    assert (
        body["status"] == "audit"
        and body["hotkey"] == "hk"
        and body["verdict"] == {"status": "REJECT"}
    )


def test_nearest_filters_bank_scope_and_converts_l2_score(monkeypatch):
    c = _patch(
        monkeypatch, _Client(hits=[{"_score": 1.0 / (1.0 + 0.25), "_source": {"model_uri": "x"}}])
    )
    out = bank.nearest(DOC, "hk", 10)
    assert out[0][0] == "x" and abs(out[0][1] - 0.5) < 1e-9
    q = c.searches[-1]["query"]["script_score"]
    filters = q["query"]["bool"]["filter"]
    assert {"term": {"status": "bank"}} in filters
    assert {"term": {"arch_key": "arch"}} in filters and {"term": {"key_id": "k1"}} in filters
    assert {"term": {"hotkey": "hk"}} in q["query"]["bool"]["must_not"]
    assert q["script"]["params"]["space_type"] == "l2"


def test_find_exact_excludes_own_hotkey_and_audit(monkeypatch):
    c = _patch(monkeypatch, _Client(hits=[]))
    assert bank.find_exact(DOC, "hk") is None
    b = c.searches[-1]["query"]["bool"]
    assert {"term": {"status": "bank"}} in b["filter"]
    assert {"term": {"hotkey": "hk"}} in b["must_not"]


def test_public_summary_pass_reports_nothing():
    res = gate.GateResult(verdict=Verdict("PASS", None, "root", "TRAINED", ["TRAINED"], {"F": 0.7}))
    assert gate.public_summary(res) == {"dedup": "pass"}


def test_public_summary_reject_reports_values_and_ancestor():
    v = Verdict(
        "REJECT",
        "NOISE-COPY",
        "ns/king@abc",
        "delta is spectral bulk",
        ["GLOBAL-RESCALE x1.0077"],
        {
            "F": 0.05,
            "rel": 0.01,
            "rel_struct": 0.002,
            "distances": [("ns/king@abc", 0.01)],
            "by_type": {},
            "opensearch_nearest": [],
            "ancestor_hotkey": "hk-king",
        },
    )
    out = gate.public_summary(gate.GateResult(verdict=v))
    assert out["duplicate_of_hotkey"] == "hk-king"
    assert (
        out["dedup"] == "reject"
        and out["duplicate_of"] == "ns/king@abc"
        and out["reason"] == "NOISE-COPY"
    )
    assert out["metrics"]["F"] == 0.05 and "distances" in out["metrics"]
    assert "by_type" not in out["metrics"] and "opensearch_nearest" not in out["metrics"]
    assert "ancestor_hotkey" not in out["metrics"]
    assert "spectral bulk" in gate.public_message(gate.GateResult(verdict=v))


def test_public_summary_exact_match():
    v = Verdict("REJECT", "COPY", "ns/orig@abc", "identical weights (tensors_hash)")
    out = gate.public_summary(
        gate.GateResult(verdict=v, exact_of={"model_uri": "ns/orig@abc", "hotkey": "hk-orig"})
    )
    assert out["exact_weights_match"] is True and out["duplicate_of_hotkey"] == "hk-orig"
    assert "metrics" not in out


def test_own_scope_queries_filter_by_coldkey_only(monkeypatch):
    c = _patch(monkeypatch, _Client(hits=[]))
    bank.nearest_own(DOC, "ck", 3)
    b = c.searches[-1]["query"]["script_score"]["query"]["bool"]
    assert {"term": {"coldkey": "ck"}} in b["filter"] and {"term": {"status": "bank"}} in b[
        "filter"
    ]
    assert b["must_not"] == []
    bank.find_exact_own(DOC, "ck")
    b = c.searches[-1]["query"]["bool"]
    assert {"term": {"coldkey": "ck"}} in b["filter"] and b["must_not"] == []


def _enforce(monkeypatch, enabled, reasons):
    monkeypatch.setattr(gate.config, "DEDUP_ENFORCE", enabled)
    monkeypatch.setattr(gate.config, "DEDUP_ENFORCE_REASONS", reasons)


def test_reason_sets_partition_every_reason_the_gate_can_emit():
    from model_validation.dedup.verdict import ALL_REASONS, EXACT_REASONS, HEURISTIC_REASONS

    assert EXACT_REASONS == {"COPY", "OWN-COPY"}
    assert HEURISTIC_REASONS == {
        "LINEAR-COMBO",
        "NOISE-COPY",
        "NOISED-COPY",
        "SPARSE-EDIT",
        "TRIVIAL-EDIT",
    }
    assert not (EXACT_REASONS & HEURISTIC_REASONS)
    assert ALL_REASONS == EXACT_REASONS | HEURISTIC_REASONS


def test_enforced_reasons_parses_the_list(monkeypatch):
    _enforce(monkeypatch, True, "COPY,OWN-COPY")
    assert gate.enforced_reasons() == {"COPY", "OWN-COPY"}

    _enforce(monkeypatch, True, " copy , noise-copy ")  # case and whitespace tolerant
    assert gate.enforced_reasons() == {"COPY", "NOISE-COPY"}

    _enforce(monkeypatch, True, "")
    assert gate.enforced_reasons() == frozenset()

    _enforce(monkeypatch, True, "COPY,NOT-A-REASON")  # unknown names warn and drop out
    assert gate.enforced_reasons() == {"COPY"}


def test_enforces_requires_both_the_switch_and_the_list(monkeypatch):
    _enforce(monkeypatch, True, "COPY,OWN-COPY")
    assert gate.enforces("COPY") is True
    assert gate.enforces("OWN-COPY") is True
    assert gate.enforces("NOISE-COPY") is False  # heuristic, not allowlisted
    assert gate.enforces("TRIVIAL-EDIT") is False
    assert gate.enforces(None) is False
    assert gate.enforces("") is False

    _enforce(monkeypatch, False, "*")  # master switch wins
    assert gate.enforces("COPY") is False


def test_device_refuses_to_fingerprint_on_cpu(monkeypatch):
    """A CPU sketch uses a different projection basis than a GPU one — never bank one."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        gate.device()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate.config, "DEDUP_GPU", 3)
    assert gate.device() == torch.device("cuda:3")


def test_fault_code_keeps_the_permanent_block_for_exact_copies_only():
    assert gate.fault_code("COPY") == "duplicate"  # the only code that blocks a hotkey forever
    assert gate.fault_code("OWN-COPY") == "duplicate_own"
    for reason in ("LINEAR-COMBO", "NOISE-COPY", "NOISED-COPY", "SPARSE-EDIT", "TRIVIAL-EDIT"):
        assert gate.fault_code(reason) == "duplicate_heuristic"
        assert gate.fault_code(reason) != "duplicate"
