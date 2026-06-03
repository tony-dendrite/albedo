#!/usr/bin/env python3
"""Prefetch the SWE-ZERO corpus and build the eval server's dataset manifest.

Downloads every shard matching chain.toml [dataset].shard_glob from [dataset].repo
into a local directory, writes manifest.json ({"shards":[{"name","rows"}],"total_rows"})
in the schema albedo.duel.sampler reads, and prints its sha256 to paste into
chain.toml [dataset].manifest_sha256. The eval server reads ALBEDO_DATASET_DIR.

Usage:
    python scripts/prefetch_dataset.py [--out /root/albedo/dataset] [--skip-download]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from albedo.config import DATASET_REPO, DATASET_SHARD_GLOB

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("prefetch")


def _parquet_rows(path: Path) -> int:
    """Row count from the parquet footer only (no full read)."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).metadata.num_rows


def build_manifest(out_dir: Path, shard_glob: str) -> Path:
    """Write manifest.json listing every shard (name relative to out_dir) and its row count."""
    shards = sorted(out_dir.glob(shard_glob))
    if not shards:
        sys.exit(f"no shards matching {shard_glob!r} under {out_dir}")
    entries = [{"name": s.relative_to(out_dir).as_posix(), "rows": _parquet_rows(s)} for s in shards]
    manifest = {"shards": entries, "total_rows": sum(e["rows"] for e in entries)}
    manifest_path = out_dir / "manifest.json"
    # Deterministic bytes so the sha256 is reproducible across machines.
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/albedo/dataset", help="Destination directory")
    parser.add_argument("--skip-download", action="store_true",
                        help="Only rebuild manifest from shards already on disk")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Only fetch the first N shards (for local/dev — full corpus is huge)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ModuleNotFoundError:
            sys.exit("huggingface_hub not installed: pip install huggingface-hub")
        # --max-shards: resolve the first N matching shard filenames and fetch only those.
        if args.max_shards is not None:
            import fnmatch
            all_files = HfApi().list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
            shards = sorted(f for f in all_files if fnmatch.fnmatch(f, DATASET_SHARD_GLOB))[:args.max_shards]
            if not shards:
                sys.exit(f"no shards matching {DATASET_SHARD_GLOB!r} in {DATASET_REPO}")
            patterns = shards
            log.info("downloading first %d shard(s) of %s → %s", len(shards), DATASET_REPO, out_dir)
        else:
            patterns = [DATASET_SHARD_GLOB]
            log.info("downloading %s :: %s → %s", DATASET_REPO, DATASET_SHARD_GLOB, out_dir)
        snapshot_download(repo_id=DATASET_REPO, repo_type="dataset",
                          allow_patterns=patterns, local_dir=str(out_dir))
        log.info("download complete")

    manifest_path = build_manifest(out_dir, DATASET_SHARD_GLOB)
    manifest = json.loads(manifest_path.read_text())
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    print("\n" + "=" * 72)
    print(f" dataset_dir:     {out_dir}")
    print(f" shards:          {len(manifest['shards'])}")
    print(f" total_rows:      {manifest['total_rows']}")
    print(f" manifest_sha256: {sha}")
    print("=" * 72)
    print("\nPaste into chain.toml:\n\n  [dataset]\n  manifest_sha256 = \"" + sha + "\"\n")
    print(f"Set on the eval server:\n  export ALBEDO_DATASET_DIR={out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
