#!/usr/bin/env python3
"""smoke_eval.py — quick liveness check of a running eval server.

Verifies /health (subprocesses, disk, dataset), and optionally drives /set_king
to confirm the king boots and gets fingerprinted+persisted (Part B). Run this on
the eval box after deploy.

Usage:
    python scripts/smoke_eval.py [--url http://localhost:8000]
    python scripts/smoke_eval.py --set-king ns/albedo-qwen3-4b-genesis@sha256:<hex>
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx


def _check_health(client: httpx.Client, url: str) -> dict:
    r = client.get(f"{url}/health", timeout=15).raise_for_status()
    h = r.json()
    print(f"  ok={h.get('ok')}  king_alive={h['king']['alive']}  chal_alive={h['challenger']['alive']}")
    print(f"  eval_lock_held={h.get('eval_lock_held')}  current_eval_id={h.get('current_eval_id')}")
    print(f"  disk_free={h['disk']['free_bytes'] / 1e9:.1f}GB  dataset={h['dataset']}")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval server smoke test")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--set-king", default=None, metavar="REPO@sha256:HEX",
                    help="Optionally boot a king and confirm it comes alive")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    with httpx.Client() as client:
        print(f"[1] GET {url}/health")
        try:
            h = _check_health(client, url)
        except Exception as exc:
            print(f"  FAIL: eval server unreachable: {exc}")
            return 1
        if not h.get("ok"):
            print("  FAIL: /health returned ok=false")
            return 1
        if not h["dataset"].get("exists"):
            print("  WARN: dataset manifest not found — run scripts/prefetch_dataset.py")

        if args.set_king:
            repo, _, digest = args.set_king.partition("@")
            if not digest.startswith("sha256:"):
                print("  FAIL: --set-king must be REPO@sha256:<hex>")
                return 1
            print(f"\n[2] POST {url}/set_king  {repo}@{digest[:19]}…")
            r = client.post(f"{url}/set_king", json={"king": {"repo": repo, "digest": digest}}, timeout=600)
            print(f"  status={r.status_code} body={r.json()}")
            if r.status_code != 200:
                print("  FAIL: /set_king did not return 200")
                return 1
            time.sleep(5)  # let the background fingerprint+persist run
            print("\n[3] GET /health (confirm king alive after set_king)")
            h2 = _check_health(client, url)
            if not h2["king"]["alive"]:
                print("  FAIL: king not alive after /set_king")
                return 1

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
