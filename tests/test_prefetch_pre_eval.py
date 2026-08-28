from __future__ import annotations

import asyncio
import json

from model_validation.prefetch_pre_eval import model_present, prefetch_once


def _shards(root, index_shards, present_shards):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(index_shards)}})
    )
    for shard in present_shards:
        (root / shard).write_text("w")


def test_model_present_requires_every_indexed_shard(tmp_path):
    complete = tmp_path / "complete"
    _shards(complete, ["a.safetensors", "b.safetensors"], ["a.safetensors", "b.safetensors"])
    assert model_present(complete)

    partial = tmp_path / "partial"
    _shards(partial, ["a.safetensors", "b.safetensors"], ["a.safetensors"])
    assert not model_present(partial)

    single = tmp_path / "single"
    single.mkdir()
    (single / "model.safetensors").write_text("w")
    assert model_present(single)

    assert not model_present(tmp_path / "missing")


class _Pool:
    def __init__(self, row):
        self._row = row

    async def fetchrow(self, _query):
        return self._row

    async def fetch(self, _query):
        return []


def test_prefetch_skips_when_queue_empty_or_model_warm(tmp_path, monkeypatch):
    assert asyncio.run(prefetch_once(_Pool(None))) == "queue empty"

    warm = tmp_path / "hf" / "own" / "repo" / ("a" * 40)
    _shards(warm, ["a.safetensors"], ["a.safetensors"])
    monkeypatch.setattr("model_validation.prefetch_pre_eval.cache_dir", lambda ref: warm)
    uri = "own/repo@" + "a" * 40
    assert asyncio.run(prefetch_once(_Pool({"model_uri": uri}))) == f"warm {uri}"


def test_prefetch_downloads_a_cold_head(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "model_validation.prefetch_pre_eval.cache_dir", lambda ref: tmp_path / "cold"
    )
    monkeypatch.setattr("model_validation.prefetch_pre_eval.make_room", lambda ref, prot: None)
    fetched = []
    monkeypatch.setattr(
        "model_validation.prefetch_pre_eval.download_full",
        lambda ref: fetched.append(ref.repo),
    )
    uri = "own/repo@" + "b" * 40
    assert asyncio.run(prefetch_once(_Pool({"model_uri": uri}))) == f"prefetched {uri}"
    assert fetched == ["own/repo"]
