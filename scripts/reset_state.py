#!/usr/bin/env python3
"""
reset_state.py — Wipe all validator state from R2 to start a fresh competition.

IMPORTANT: Stop the validator before running this.
  ssh templar "pm2 stop albedo-validator"

Usage:
  export ALBEDO_R2_ENDPOINT=https://...
  export ALBEDO_R2_BUCKET=albedo-state
  export ALBEDO_R2_ACCESS_KEY=...
  export ALBEDO_R2_SECRET_KEY=...
  python scripts/reset_state.py

  # Dry run (prints what would be deleted, does not delete):
  python scripts/reset_state.py --dry-run
"""

import argparse
import os
import sys

import boto3
from botocore.config import Config

KEYS_TO_WIPE = [
    "king/current.json",
    "state/king_chain.json",
    "state/seen_hotkeys.json",
    "state/completed_repos.json",
    "state/queue.json",
    "state/validator_state.json",
    "state/dashboard_history.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be deleted without deleting anything")
    ap.add_argument("--yes", action="store_true",
                    help="Skip confirmation prompt")
    args = ap.parse_args()

    endpoint  = os.environ.get("ALBEDO_R2_ENDPOINT", "")
    bucket    = os.environ.get("ALBEDO_R2_BUCKET", "")
    access    = os.environ.get("ALBEDO_R2_ACCESS_KEY", "")
    secret    = os.environ.get("ALBEDO_R2_SECRET_KEY", "")

    if not all([endpoint, bucket, access, secret]):
        sys.exit(
            "Missing R2 credentials. Set:\n"
            "  ALBEDO_R2_ENDPOINT\n  ALBEDO_R2_BUCKET\n"
            "  ALBEDO_R2_ACCESS_KEY\n  ALBEDO_R2_SECRET_KEY"
        )

    print(f"Target bucket: {bucket} at {endpoint}")
    print(f"Keys to wipe ({len(KEYS_TO_WIPE)}):")
    for key in KEYS_TO_WIPE:
        print(f"  {key}")

    if args.dry_run:
        print("\nDry run — nothing deleted.")
        return 0

    if not args.yes:
        answer = input("\nThis will wipe all validator state. Type 'yes' to confirm: ")
        if answer.strip().lower() != "yes":
            sys.exit("Aborted.")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="auto",
        config=Config(retries={"max_attempts": 3}),
    )

    print("\nDeleting …")
    errors = []
    for key in KEYS_TO_WIPE:
        try:
            s3.delete_object(Bucket=bucket, Key=key)
            print(f"  ✓  {key}")
        except Exception as e:
            print(f"  ✗  {key}: {e}")
            errors.append(key)

    if errors:
        print(f"\n⚠  {len(errors)} key(s) failed to delete: {errors}")
        return 1

    print("\nState cleared. Start the validator to begin a fresh competition:")
    print("  ssh templar \"pm2 start albedo-validator\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
