#!/usr/bin/env python3
"""One-shot: upload ricdomolm/mini-coder-1.7b to Hippius Hub as the
Albedo genesis king. Prints the OCI digest to paste into
chain.toml [seed].seed_digest.

Usage:
    source .venv/bin/activate
    export HIPPIUS_HUB_TOKEN=...   # or HIPPIUS_HUB_USERNAME/PASSWORD
    python scripts/seed_genesis.py [--repo albedo/Albedo-Mini-1.7B-genesis]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure repo root on path so `model_store` and `chain_config` import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download
import chain_config
from model_store import upload_model_folder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed_genesis")


def main() -> int:
    parser = argparse.ArgumentParser(description="Albedo genesis seeder")
    parser.add_argument("--hf-model", default="ricdomolm/mini-coder-1.7b",
                        help="HF repo to download as the genesis weights")
    parser.add_argument("--repo", default=chain_config.SEED_REPO,
                        help="Hippius Hub repo id to upload to "
                             "(default: chain.toml [chain].seed_repo)")
    parser.add_argument("--workdir", default="/tmp/albedo/genesis",
                        help="Local staging directory")
    args = parser.parse_args()

    log.info("downloading %s from HF -> %s", args.hf_model, args.workdir)
    local_dir = snapshot_download(
        repo_id=args.hf_model,
        local_dir=args.workdir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*",
                        "special_tokens*", "*.model", "*.txt"],
        max_workers=8,
    )
    log.info("downloaded to %s", local_dir)

    log.info("uploading to Hippius Hub repo %s", args.repo)
    ref = upload_model_folder(
        local_dir, repo=args.repo, revision="genesis",
        commit_message=f"genesis: {args.hf_model}",
    )
    print()
    print("=" * 72)
    print(f" Hippius ref:  {ref.immutable_ref}")
    print(f" repo:         {ref.repo}")
    print(f" digest:       {ref.digest}")
    print("=" * 72)
    print()
    print("Paste into chain.toml:")
    print()
    print("  [seed]")
    print(f"  seed_digest = \"{ref.digest}\"")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
