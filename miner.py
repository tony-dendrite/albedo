#!/usr/bin/env python3
"""Albedo miner — build a challenger and submit it on-chain.

Downloads the current king, produces a challenger (default: trivial
gaussian-noise perturbation as a pipeline-test stub), uploads to Hippius
Hub, and posts a `v4` reveal commitment binding
`(challenger_repo, challenger_digest, author_hotkey)`.

The noise perturbation will not beat a mature king on the LLM-as-judge
duel — it's a structural placeholder so miners can smoke-test the
pipeline. Real dethrones come from real SFT/RL on SWE-style trajectories.
Swap `train_or_perturb` for your training step and the rest applies
unchanged.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import bittensor as bt
import httpx
import torch
from safetensors.torch import load_file, save_file

# chain_config sits at the repo root; ensure it imports regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_config  # noqa: E402
from model_store import (  # noqa: E402
    ModelRef,
    build_reveal_v4,
    materialize_model,
    upload_model_folder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("miner")

DASHBOARD_URL = os.environ.get(
    "ALBEDO_DASHBOARD_URL",
    "https://us-east-1.hippius.com/albedo/dashboard.json",
)
SEED_REPO = os.environ.get("ALBEDO_SEED_REPO", chain_config.SEED_REPO)
SEED_DIGEST = os.environ.get("ALBEDO_SEED_DIGEST", chain_config.SEED_DIGEST)
NETUID = int(os.environ.get("ALBEDO_NETUID", "0"))
NETWORK = os.environ.get("ALBEDO_NETWORK", "finney")
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "albedo")

REPO_PATTERN = re.compile(os.environ.get("ALBEDO_REPO_PATTERN", chain_config.REPO_PATTERN))

# The validator re-checks these server-side; this is a cheap belt-and-suspenders
# pre-flight so miners catch obvious arch mismatches before paying upload cost.
CONFIG_MATCH_KEYS = (
    "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "intermediate_size", "model_type",
)


def validate_local_config(king_dir: str, challenger_dir: str) -> str | None:
    king_cfg_path = Path(king_dir) / "config.json"
    chall_cfg_path = Path(challenger_dir) / "config.json"

    if not king_cfg_path.exists():
        return None
    if not chall_cfg_path.exists():
        return "challenger config.json missing"

    with open(king_cfg_path) as f:
        king_cfg = json.load(f)
    with open(chall_cfg_path) as f:
        chall_cfg = json.load(f)

    king_arch = king_cfg.get("architectures", [])
    chall_arch = chall_cfg.get("architectures", [])
    if king_arch and chall_arch and king_arch != chall_arch:
        return f"architecture mismatch: king={king_arch} challenger={chall_arch}"

    for key in CONFIG_MATCH_KEYS:
        king_val = king_cfg.get(key)
        chall_val = chall_cfg.get(key)
        if king_val is not None and chall_val is not None and king_val != chall_val:
            return f"{key} mismatch: king={king_val} challenger={chall_val}"

    for key in chain_config.EXTRA_LOCK_KEYS:
        king_val = king_cfg.get(key)
        chall_val = chall_cfg.get(key)
        if king_val is not None and chall_val is not None and king_val != chall_val:
            return f"{key} mismatch (extra_lock_keys): king={king_val} challenger={chall_val}"

    st_files = list(Path(challenger_dir).glob("*.safetensors"))
    if not st_files:
        return "no .safetensors files in challenger"
    return None


def train_or_perturb(king_dir: str, challenger_dir: str, noise: float) -> None:
    """Default: copy the king and add low-amplitude noise to every float
    tensor. This is a pipeline-test stub. Replace this function with your
    real SFT/RL training step to actually beat the king on the duel.

    Recommended starting point for real training:
      1. Load `AlienKevin/SWE-ZERO-12M-trajectories` (filter to
         exit_status='Submitted' if you want a cleaner signal).
      2. Render each `messages` list to the king's chat template.
      3. SFT with assistant-token loss masking (mask system+user tokens).
      4. Keep arch + tokenizer untouched so config validation passes.
    """
    if os.path.exists(challenger_dir):
        shutil.rmtree(challenger_dir)
    shutil.copytree(king_dir, challenger_dir)

    for st_file in sorted(Path(challenger_dir).glob("*.safetensors")):
        log.info("perturbing %s", st_file.name)
        sd = load_file(str(st_file))
        new_sd = {}
        for name, tensor in sd.items():
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                noise_t = torch.randn_like(tensor.float()) * noise
                new_sd[name] = (tensor.float() + noise_t).to(tensor.dtype)
            else:
                new_sd[name] = tensor
        save_file(new_sd, str(st_file))


def main() -> int:
    parser = argparse.ArgumentParser(description="Albedo miner")
    parser.add_argument("--hotkey", default="h0", help="Wallet hotkey name")
    parser.add_argument("--noise", type=float, default=0.001,
                        help="Noise scale for the default perturb stub")
    parser.add_argument("--suffix", default=None,
                        help="Challenger repo suffix (default: hotkey name)")
    parser.add_argument("--force", action="store_true",
                        help="Bypass soft warnings (hotkey seen, king hotkey)")
    args = parser.parse_args()

    if NETUID == 0:
        log.error("set ALBEDO_NETUID to the actual subnet netuid before mining")
        return 1

    suffix = args.suffix or args.hotkey
    namespace = (
        os.environ.get("ALBEDO_CHALLENGER_NAMESPACE")
        or chain_config.SEED_NAMESPACE
        or "teutonic"
    )
    repo_base = os.environ.get("ALBEDO_CHALLENGER_REPO_NAME", chain_config.NAME)
    challenger_repo = f"{namespace}/{repo_base}-{suffix}"

    log.info("miner starting | hotkey=%s repo=%s noise=%.4f",
             args.hotkey, challenger_repo, args.noise)

    if not REPO_PATTERN.match(challenger_repo):
        log.error("repo name %s does not match required pattern %s",
                  challenger_repo, REPO_PATTERN.pattern)
        return 1

    wallet = bt.Wallet(name=WALLET_NAME, hotkey=args.hotkey)
    _pw = os.environ.get("BT_WALLET_PASSWORD")
    if _pw:
        wallet.coldkey_file.decrypt(_pw.strip())
    subtensor = bt.Subtensor(network=NETWORK)
    my_hotkey = wallet.hotkey.ss58_address
    log.info("wallet: %s", my_hotkey)

    # Pre-flight 1: registered on subnet
    try:
        meta = subtensor.metagraph(NETUID)
        if my_hotkey not in meta.hotkeys:
            log.error("hotkey %s is NOT registered on subnet %d — register before mining",
                      my_hotkey[:16], NETUID)
            return 1
        uid = meta.hotkeys.index(my_hotkey)
        log.info("hotkey registered as uid=%d on subnet %d", uid, NETUID)
    except Exception:
        log.warning("could not query metagraph — skipping registration check")

    # Discover current king from dashboard
    king_repo = SEED_REPO
    king_digest = SEED_DIGEST
    dashboard = None
    try:
        resp = httpx.get(DASHBOARD_URL, timeout=15)
        resp.raise_for_status()
        dashboard = resp.json()
        king = dashboard["king"]
        king_repo = king["model_repo"]
        king_digest = king.get("king_digest") or king.get("model_digest")
        log.info("discovered king from dashboard: %s@%s",
                 king_repo, (king_digest or "")[:19])
    except Exception:
        log.warning("could not fetch dashboard, falling back to seed %s", SEED_REPO)
    if not king_digest:
        log.error("no king digest available; set ALBEDO_SEED_DIGEST or wait for dashboard")
        return 1
    king_ref = ModelRef(king_repo, king_digest)

    # Pre-flight 2: hotkey is not the current king
    if dashboard:
        king_hotkey = dashboard.get("king", {}).get("hotkey", "")
        if king_hotkey and my_hotkey == king_hotkey:
            log.warning("your hotkey %s IS the current king — validator will skip", my_hotkey[:16])
            if not args.force:
                log.error("aborting (use --force to override)")
                return 1
            log.warning("--force set, continuing anyway")

    # Pre-flight 3: hotkey has no prior reveal
    try:
        all_reveals = subtensor.get_all_revealed_commitments(NETUID)
        if all_reveals and my_hotkey in all_reveals:
            log.warning("hotkey %s already has an existing reveal on-chain — "
                        "validator gates 1-eval-per-hotkey", my_hotkey[:16])
            if not args.force:
                log.error("aborting (use --force to override)")
                return 1
            log.warning("--force set, continuing anyway")
    except Exception:
        log.warning("could not check existing reveals — skipping seen-hotkey check")

    king_dir = "/tmp/albedo/miner/king"
    if os.path.exists(king_dir):
        shutil.rmtree(king_dir)
    log.info("downloading king from %s", king_ref.immutable_ref)
    materialize_model(king_ref, local_dir=king_dir, max_workers=16)

    challenger_dir = f"/tmp/albedo/miner/challenger-{suffix}"
    train_or_perturb(king_dir, challenger_dir, args.noise)

    rejection = validate_local_config(king_dir, challenger_dir)
    if rejection:
        log.error("config validation failed: %s", rejection)
        return 1
    log.info("config validation passed")

    log.info("uploading to Hippius Hub repo %s", challenger_repo)
    challenger_ref = upload_model_folder(
        challenger_dir,
        repo=challenger_repo,
        revision=suffix,
        commit_message=f"Challenger from {args.hotkey} (noise={args.noise})",
    )
    log.info("uploaded to %s", challenger_ref.immutable_ref)

    payload = build_reveal_v4(challenger_ref, my_hotkey)
    log.info("submitting reveal: %s", payload)

    resp = subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=NETUID,
        data=payload,
        blocks_until_reveal=3,
        wait_for_revealed_execution=False,
    )

    if resp.success:
        log.info("reveal committed: %s", resp.message)
        log.info("the validator will pick this up once revealed (~30 seconds)")
    else:
        log.error("reveal commitment failed: %s", resp.message)
        return 1

    log.info("done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
