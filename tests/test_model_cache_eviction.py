from __future__ import annotations

import os
import time

import model_validation.storage.download as dl
from model_validation.storage import make_ref, make_room

_DIGEST = "a" * 40


def _model(root, owner, name, digest, age_rank: int):
    d = root / "hf" / owner / name / digest
    d.mkdir(parents=True)
    (d / "model.safetensors").write_text("w")
    stamp = time.time() - 10_000 + age_rank * 100  # higher rank = newer
    os.utime(d, (stamp, stamp))
    return d


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setattr(dl, "MODEL_CACHE_DIR", str(root))
    monkeypatch.setattr("config_validation.storage._paths.MODEL_CACHE_DIR", str(root))
    monkeypatch.setattr(dl, "MAX_CACHED_MODELS", 4)
    return root


def test_newest_models_are_evicted_first_down_to_the_cap(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    dirs = [_model(root, f"owner{i}", "repo", _DIGEST, age_rank=i) for i in range(6)]

    make_room(make_ref("newguy/repo", _DIGEST))

    # 3 oldest survive (the incoming download makes 4); the 3 newest are gone, parents pruned
    assert [d.exists() for d in dirs] == [True, True, True, False, False, False]
    assert not (root / "hf" / "owner5").exists()


def test_the_preeval_model_is_never_evicted_even_when_newest(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    for i in range(4):
        _model(root, f"owner{i}", "repo", _DIGEST, age_rank=i)
    running = _model(root, "active", "repo", _DIGEST, age_rank=99)  # newest: just re-fetched

    make_room(make_ref("newguy/repo", _DIGEST), protected_repos={"active/repo"})

    assert running.exists()
    # cap still enforced among the evictable rest
    survivors = [d for d in (root / "hf").glob("*/*/*") if d.is_dir()]
    assert len(survivors) == 4  # 3 evictable + the protected one


def test_own_target_and_small_caches_are_left_alone(tmp_path, monkeypatch):
    root = _setup(tmp_path, monkeypatch)
    ref = make_ref("owner0/repo", _DIGEST)
    kept = _model(root, "owner0", "repo", _DIGEST, age_rank=50)
    other = _model(root, "owner1", "repo", _DIGEST, age_rank=1)

    make_room(ref)

    assert kept.exists() and other.exists()
