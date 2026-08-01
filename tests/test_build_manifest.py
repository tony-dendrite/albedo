import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from albedo_eval_service.sampling import multi_source_manifest_sample_ids


def _load_build_manifest():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_manifest.py"
    spec = importlib.util.spec_from_file_location("build_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STRATEGIES = ["pr_1", "lm_rewrite__a", "combine_file__b", "func_pm_remove_cond__c"]


def _conversation(asst: int, edit_at: int) -> list[dict]:
    """`asst` assistant turns, the one at `edit_at` (1-based) carrying an edit command so
    _row_meta can resolve first_edit."""
    turns = [{"role": "system", "content": "s"}]
    for i in range(1, asst + 1):
        turns.append({"role": "user", "content": "o"})
        turns.append({"role": "assistant", "content": "sed -i s/a/b/ f.py" if i == edit_at else "c"})
    return turns


def _write_shard(data_dir: Path, name: str, rows: int, *, asst: int = 12) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    # Rows cycle through the four bug families (via the instance_id grammar) and edit depths, so the
    # enriched manifest carries the {iid, asst, first_edit, family} the sampler strata need.
    table = pa.table(
        {
            # repo varies: the sampler caps how many tasks one codebase can supply (REPO_CAP).
            "instance_id": [
                f"owner__repo{i}.abc1234.{_STRATEGIES[i % len(_STRATEGIES)]}" for i in range(rows)
            ],
            "messages": [_conversation(asst, 4 + (i % 5)) for i in range(rows)],
        }
    )
    pq.write_table(table, data_dir / name)


def test_parse_sources():
    bm = _load_build_manifest()
    assert bm._parse_sources("swe-hero, mini-coder") == ["swe-hero", "mini-coder"]


def test_build_source_counts_rows_and_enriches_row_meta(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "mini-coder" / "data", "train-00000.parquet", 3)
    _write_shard(tmp_path / "mini-coder" / "data", "train-00001.parquet", 2)

    source = bm._build_source("mini-coder", tmp_path)

    assert source["name"] == "mini-coder"
    assert "weight" not in source  # the mix is STEP_TRIM x FAMILY_MIX, never a per-source weight
    assert source["total_rows"] == 5
    assert [s["path"] for s in source["shards"]] == [
        "mini-coder/data/train-00000.parquet",
        "mini-coder/data/train-00001.parquet",
    ]
    assert all(len(s["sha256"]) == 64 for s in source["shards"])
    first = source["shards"][0]
    assert len(first["rows_meta"]) == first["rows"] == 3
    assert first["rows_meta"][0] == {
        "iid": "owner__repo0.abc1234.pr_1",
        "asst": 12,
        "first_edit": 4,
        "family": "pr",
        "repo": "owner__repo0",
        "language": "python",
        "verified": None,
        "chars_at": 26,
        "chars_pre": 5,
    }
    assert [m["family"] for m in first["rows_meta"]] == ["pr", "lm", "combine"]


def test_built_manifest_is_sampler_compatible(tmp_path):
    bm = _load_build_manifest()
    _write_shard(tmp_path / "mini-coder" / "data", "train-00000.parquet", 400)
    _write_shard(tmp_path / "swe-hero" / "data", "train-00000-of-00060.parquet", 400)

    sources = [
        bm._build_source("mini-coder", tmp_path),
        bm._build_source("swe-hero", tmp_path),
    ]
    manifest = {"version": "t", "sources": sources, "total_rows": 800}

    ids = multi_source_manifest_sample_ids(manifest, block_hash="0xabc", sample_count=100)
    assert len(ids) == 100 == len(set(ids))
    # instances are pooled across sources, so both contribute; the split is driven by the
    # STEP_TRIM x FAMILY_MIX grid and pool size, never by a per-source weight
    assert sum(1 for i in ids if i.startswith("mini-coder/")) > 0
    assert sum(1 for i in ids if i.startswith("swe-hero/")) > 0
