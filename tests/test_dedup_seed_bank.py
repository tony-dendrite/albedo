from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("torch")


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "dedup_seed_bank.py"
    spec = importlib.util.spec_from_file_location("dedup_seed_bank", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SHA = "f" * 40


def test_king_maps_to_dendrite_mirror():
    m = _load()
    assert m.king_repo("cxv") == f"{m.KING_NAMESPACE}/{m.KING_PREFIX}-CXV"
    k = m.parse_king({"roman": "CIV", "revision": SHA, "hotkey": "hk", "coldkey": "ck"})
    assert k == dict(repo=m.king_repo("CIV"), revision=SHA, hotkey="hk", coldkey="ck")
    assert m.parse_king("LXXXVI")["repo"].endswith("-LXXXVI")


def test_parse_hf_url_forms():
    m = _load()
    e = m.parse_entry({"url": f"https://huggingface.co/ns/repo/tree/{SHA}"})
    assert e["repo"] == "ns/repo" and e["revision"] == SHA
    e = m.parse_entry({"url": "https://huggingface.co/ns/repo", "revision": SHA, "hotkey": "hk"})
    assert e["repo"] == "ns/repo" and e["revision"] == SHA and e["hotkey"] == "hk"
    e = m.parse_entry({"repo": f"ns/repo@{SHA}"})
    assert e["revision"] == SHA
    assert m.parse_entry("ns/repo")["repo"] == "ns/repo"
    with pytest.raises(ValueError):
        m.parse_entry({"url": "https://example.com/ns/repo"})
    with pytest.raises(ValueError):
        m.parse_entry({"repo": "no-namespace"})


def test_resolve_revision_requires_commit_sha():
    m = _load()
    assert m.resolve_revision("ns/repo", SHA) == SHA
    with pytest.raises(ValueError):
        m.resolve_revision("ns/repo", "main")


def test_load_manifest(tmp_path):
    m = _load()
    p = tmp_path / "bank.json"
    p.write_text(
        json.dumps(
            {
                "root": False,
                "kings": ["CIV", {"roman": "CXV", "revision": SHA}],
                "models": [{"url": f"https://huggingface.co/a/b/tree/{SHA}"}],
            }
        )
    )
    root, kings, models = m.load_manifest(str(p))
    assert root is False and len(kings) == 2 and models[0]["repo"] == "a/b"
    assert kings[1]["revision"] == SHA and kings[0]["revision"] == ""
