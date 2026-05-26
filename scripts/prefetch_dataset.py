#!/usr/bin/env python3
"""Prefetch the pinned SWE-ZERO parquet shard for the eval server.

Downloads the single parquet file named in `chain.toml [dataset].shard`
from `[dataset].repo` to `/var/albedo/dataset/<basename>.parquet`,
verifies it loads as a parquet Table, and prints the sha256 to paste back
into `chain.toml [dataset].shard_sha256`.

Usage:
    source .venv/bin/activate
    python scripts/prefetch_dataset.py [--out /var/albedo/dataset]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chain_config
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("prefetch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/var/albedo/dataset",
                        help="Destination directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / Path(chain_config.DATASET_SHARD).name

    log.info("downloading %s :: %s", chain_config.DATASET_REPO, chain_config.DATASET_SHARD)
    local = hf_hub_download(
        repo_id=chain_config.DATASET_REPO,
        filename=chain_config.DATASET_SHARD,
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    log.info("downloaded to %s", local)

    if Path(local).resolve() != dst.resolve():
        # hf_hub_download may put it under data/<file>; normalise to a
        # single flat path so eval.py's ALBEDO_DATASET_SHARD_PATH default
        # works without computing subdir.
        log.info("normalising path: %s -> %s", local, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        Path(local).replace(dst)

    table = pq.read_table(dst, memory_map=True)
    log.info("parquet rows=%d, columns=%s", table.num_rows, table.column_names)

    h = hashlib.sha256()
    with open(dst, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    sha = h.hexdigest()

    print()
    print("=" * 72)
    print(f" shard:       {dst}")
    print(f" rows:        {table.num_rows}")
    print(f" sha256:      {sha}")
    print("=" * 72)
    print()
    print("Paste into chain.toml:")
    print()
    print("  [dataset]")
    print(f"  shard_sha256 = \"{sha}\"")
    print()
    print("Set on the eval server:")
    print(f"  export ALBEDO_DATASET_SHARD_PATH={dst}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
