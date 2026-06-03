#!/usr/bin/env python3
"""
collect_traces.py — Download public Albedo duel traces and extract winning turns.

Writes a JSONL file where each line is one training example:
  {"text": "<Qwen3 chat-formatted conversation>", "delta_avg": 0.12, "eval_id": "eval-0042"}

Usage:
  python scripts/collect_traces.py --out data/traces.jsonl
  python scripts/collect_traces.py --out data/traces.jsonl --min-delta 0.1 --max-evals 20
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DASHBOARD_URL = "https://us-east-1.hippius.com/albedo/dashboard.json"
EVALS_BASE    = "https://us-east-1.hippius.com/albedo/evals"


def fetch_json(url):
    with urlopen(url, timeout=30) as r:
        return json.load(r)


def fetch_bytes(url):
    with urlopen(url, timeout=120) as r:
        return r.read()


def eval_url(completed_at, challenge_id):
    date = completed_at[:10]  # "2026-06-01T04:25:43" → "2026-06-01"
    return f"{EVALS_BASE}/{date}/{challenge_id}.jsonl.gz"


def extract_winning_turns(records, min_delta):
    """Return turns where the challenger beat the king by at least min_delta."""
    out = []
    for rec in records:
        if not rec.get("parse_ok", True):
            continue
        delta = rec.get("delta_avg", 0.0)
        if delta <= min_delta:
            continue
        out.append({
            "messages_prefix": rec["messages_prefix"],
            "messages_prompt": rec["messages_prompt"],
            "chal_reply":      rec["chal_reply"],
            "king_reply":      rec["king_reply"],   # kept for DPO use later
            "delta_avg":       delta,
            "eval_id":         rec.get("eval_id", "unknown"),
            "instance_id":     rec.get("instance_id", ""),
        })
    return out


def format_as_chat(tokenizer, turn):
    """
    Build one training string using the Qwen3 chat template.

    Structure:
      [system] + [user/assistant pairs from messages_prefix]
      + [user from messages_prompt]
      + [assistant = chal_reply]  ← the target token sequence
    """
    messages = list(turn["messages_prefix"]) + list(turn["messages_prompt"])
    messages.append({"role": "assistant", "content": turn["chal_reply"]})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",        default="data/traces.jsonl")
    ap.add_argument("--min-delta",  type=float, default=0.05,
                    help="Min delta_avg to include a turn (default 0.05 = 5%%). "
                         "Higher = cleaner signal, fewer examples.")
    ap.add_argument("--max-evals",  type=int, default=None,
                    help="Process only the N most recent evals (for quick testing)")
    ap.add_argument("--cache-dir",  default="data/cache",
                    help="Local cache directory for .jsonl.gz files")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B",
                    help="Tokenizer to use for chat template formatting")
    ap.add_argument("--raw",        action="store_true",
                    help="Write raw dicts instead of formatted text (for inspection)")
    args = ap.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Fetching dashboard …")
    try:
        dashboard = fetch_json(DASHBOARD_URL)
    except URLError as e:
        sys.exit(f"Cannot reach dashboard: {e}")

    entries = []
    for h in dashboard.get("history", []):
        cid = h.get("challenge_id") or h.get("eval_id", "")
        cat = h.get("completed_at") or h.get("crowned_at", "")
        if cid and cat and cid.startswith("eval-"):
            entries.append((cat, cid))
    entries.sort()
    if args.max_evals:
        entries = entries[-args.max_evals:]

    print(f"Found {len(entries)} eval entries")
    if not entries:
        sys.exit("No eval entries — is the dashboard reachable?")

    tokenizer = None
    if not args.raw:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer ({args.base_model}) …")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    total_turns = winning_turns = 0

    with open(args.out, "w") as fout:
        for i, (completed_at, challenge_id) in enumerate(entries):
            url        = eval_url(completed_at, challenge_id)
            cache_file = Path(args.cache_dir) / f"{challenge_id}.jsonl.gz"

            if not cache_file.exists():
                print(f"  [{i+1}/{len(entries)}] Downloading {challenge_id} …",
                      end=" ", flush=True)
                try:
                    cache_file.write_bytes(fetch_bytes(url))
                    print("ok")
                except Exception as e:
                    print(f"SKIP ({e})")
                    continue
            else:
                print(f"  [{i+1}/{len(entries)}] {challenge_id} (cached)")

            try:
                lines   = gzip.decompress(cache_file.read_bytes()).splitlines()
                records = [json.loads(l) for l in lines if l.strip()]
            except Exception as e:
                print(f"    Parse error: {e}")
                continue

            total_turns += len(records)
            winning      = extract_winning_turns(records, args.min_delta)
            winning_turns += len(winning)

            for turn in winning:
                if args.raw or tokenizer is None:
                    fout.write(json.dumps(turn) + "\n")
                else:
                    text = format_as_chat(tokenizer, turn)
                    fout.write(json.dumps({
                        "text":      text,
                        "delta_avg": turn["delta_avg"],
                        "eval_id":   turn["eval_id"],
                    }) + "\n")

    print(f"\n{'─'*50}")
    print(f"Total turns:   {total_turns}")
    print(f"Winning turns: {winning_turns}  "
          f"({winning_turns / max(total_turns, 1) * 100:.1f}% win rate)")
    print(f"Output:        {args.out}")

    if winning_turns < 200:
        print(f"\n⚠  Only {winning_turns} examples — consider lowering --min-delta")


if __name__ == "__main__":
    main()
