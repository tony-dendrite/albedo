from __future__ import annotations

import math

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


def test_thresholds_come_from_settings():
    th = gate.thresholds()
    assert th.linear_resid == 0.20 and th.f_noise == 0.10 and th.rel_trivial == 0.0025
    assert th.embed_ratio_noise == 0.60 and th.dens_min == 0.40 and th.kurt_max == 100.0
    assert math.isclose(th.copy_rel, 1e-5)
    assert th.alpha_z == 8.0 and th.topk_partners == 5 and th.lora_max_spikes == 40
    assert th.head_scale_max == 0.02 and th.global_scale_max == 0.005 and th.reuse_cos == 0.8
    assert th.permuted_max_ident == 0.5 and th.lora_min_f == 0.9 and th.alpha_min == 0.02


def test_nearest_and_exact_exclude_own_coldkey(monkeypatch):
    c = _patch(monkeypatch, _Client(hits=[]))
    bank.nearest(DOC, "hk", 10, "ck")
    mn = c.searches[-1]["query"]["script_score"]["query"]["bool"]["must_not"]
    assert {"term": {"hotkey": "hk"}} in mn and {"term": {"coldkey": "ck"}} in mn
    bank.find_exact(DOC, "hk", "ck")
    mn = c.searches[-1]["query"]["bool"]["must_not"]
    assert {"term": {"coldkey": "ck"}} in mn


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


def test_put_doc_stores_coldkey(monkeypatch):
    c = _patch(monkeypatch, _Client())
    bank.put_doc(DOC, status=bank.STATUS_BANK, hotkey="hk", coldkey="ck")
    assert c.indexed[0][1]["coldkey"] == "ck"


def test_public_summary_own_copy_has_no_metrics():
    v = Verdict(
        "REJECT",
        "OWN-COPY",
        "ns/mine@old",
        "identical",
        [],
        {"ancestor_hotkey": "hk", "rel_dist": 0.0},
    )
    out = gate.public_summary(gate.GateResult(verdict=v))
    assert out["own_model"] is True and out["duplicate_of_hotkey"] == "hk"
    assert "metrics" not in out
