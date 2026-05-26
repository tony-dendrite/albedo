#!/usr/bin/env python3
"""Smoke-test the eval server end-to-end without paying for chain ops.

Posts a synthetic /eval request directly to eval.py and streams the SSE
events to stdout. Use to sanity-check that:

    1. eval.py is reachable and king vLLM is up.
    2. /set_king worked for the seed king.
    3. The trajectory shard is loadable and sampling is deterministic.
    4. Chutes judge auth is correct (no 401s).
    5. The verdict shape matches what validator.py expects.

Usage:
    export ALBEDO_EVAL_SERVER=http://127.0.0.1:9000
    python scripts/smoke_eval.py \\
        --chal-repo your-org/Albedo-Mini-1.7B-smoketest \\
        --chal-digest sha256:....

If --chal-repo is omitted we point challenger at the seed king itself.
This is a useful "null duel" — both sides identical, the verdict should
have mean_delta ≈ 0 and accepted=False.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chain_config

EVAL_URL = os.environ.get("ALBEDO_EVAL_SERVER", "http://127.0.0.1:9000")


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chal-repo", default=chain_config.SEED_REPO)
    p.add_argument("--chal-digest", default=chain_config.SEED_DIGEST)
    p.add_argument("--seed-hex", default="00" * 32,
                   help="32 bytes of seed material (default: zeros). Pin to "
                        "reproduce a fixture set.")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=2)
    args = p.parse_args()

    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0)) as client:
        h = await client.get(f"{EVAL_URL}/health")
        h.raise_for_status()
        health = h.json()
        print("health:", json.dumps(health, indent=2))
        if not health.get("king", {}).get("alive"):
            print("ERROR: king vllm is not up. POST /set_king first.", file=sys.stderr)
            return 2

        req = {
            "king": {
                "repo": chain_config.SEED_REPO,
                "digest": chain_config.SEED_DIGEST,
            },
            "challenger": {"repo": args.chal_repo, "digest": args.chal_digest},
            "seed_hex": args.seed_hex,
            "eval_id": "smoke-0001",
            "n_samples": args.n_samples,
            "max_turns": args.max_turns,
        }
        print("posting /eval:", json.dumps(req, indent=2))

        async with client.stream("POST", f"{EVAL_URL}/eval", json=req,
                                  timeout=httpx.Timeout(None, connect=30.0)) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                print("HTTP", resp.status_code, err[:500].decode(errors="ignore"),
                      file=sys.stderr)
                return 3
            cur_event = ""
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    cur_event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line.split(":", 1)[1].strip()
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                except Exception:
                    continue
                if cur_event == "progress":
                    print(f"[{cur_event}] {data.get('n_done')}/{data.get('n_total')} "
                          f"king={data.get('king_mean', 0):.3f} chal={data.get('chal_mean', 0):.3f} "
                          f"Δ={data.get('mean_delta', 0):+.3f} pf={data.get('parse_failures', 0)}")
                elif cur_event == "verdict":
                    print("[verdict]", json.dumps(data, indent=2))
                    return 0 if data.get("accepted") is not None else 1
                else:
                    print(f"[{cur_event}] {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
