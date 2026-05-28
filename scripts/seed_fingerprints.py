"""One-shot script: seed uploaded_models_state.json with the genesis king fingerprint.

Run once after deploying the preeval system to ensure the genesis model's
fingerprint is in the DB before any challengers are evaluated.

Usage:
    cd /path/to/albedo
    source .venv/bin/activate
    python scripts/seed_fingerprints.py

Required env vars (same as eval server):
    ALBEDO_EVALS_S3_ENDPOINT  (default: https://s3.hippius.com)
    ALBEDO_EVALS_S3_BUCKET
    ALBEDO_EVALS_S3_ACCESS_KEY
    ALBEDO_EVALS_S3_SECRET_KEY
"""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("seed_fingerprints")


def main() -> None:
    import chain_config
    import preeval
    from model_store import ModelRef, materialize_model

    endpoint = os.environ.get("ALBEDO_EVALS_S3_ENDPOINT", "https://s3.hippius.com")
    bucket   = os.environ.get("ALBEDO_EVALS_S3_BUCKET", "")
    access   = os.environ.get("ALBEDO_EVALS_S3_ACCESS_KEY", "")
    secret   = os.environ.get("ALBEDO_EVALS_S3_SECRET_KEY", "")

    if not (bucket and access and secret):
        log.error(
            "Missing S3 credentials. Set ALBEDO_EVALS_S3_BUCKET, "
            "ALBEDO_EVALS_S3_ACCESS_KEY, ALBEDO_EVALS_S3_SECRET_KEY."
        )
        sys.exit(1)

    s3 = preeval._get_or_create_s3_client(endpoint, access, secret)

    log.info("Loading existing models state from s3://%s/%s", bucket, preeval.MODELS_STATE_KEY)
    state = preeval.load_models_state(s3, bucket)
    log.info("Found %d existing entries.", len(state.get("models", {})))

    genesis_ref = ModelRef(chain_config.SEED_REPO, chain_config.SEED_DIGEST)
    ref_key = genesis_ref.immutable_ref

    if genesis_ref.digest.startswith("hf:"):
        log.info("Genesis king uses HF digest (%s) — challengers are Hippius-only, skipping seed.", ref_key)
        return

    if ref_key in state.get("models", {}):
        log.info("Genesis king fingerprint already present: %s — nothing to do.", ref_key)
        return

    log.info("Materializing genesis king: %s", ref_key)
    genesis_dir = materialize_model(genesis_ref, None, 16)
    log.info("Materialized to: %s", genesis_dir)

    log.info("Computing fingerprint (this takes ~20s on CPU)…")
    fp = preeval.compute_fingerprint(genesis_dir)
    log.info("Fingerprint computed: %d layers, sha256=%s", len(fp["layer_keys"]), fp["sha256_bytes"][:16])

    preeval.add_fingerprint_to_state(
        state,
        ref_key,
        fp,
        hotkey="",
        verdict="king",
        repo=genesis_ref.repo,
        digest=genesis_ref.digest,
    )
    preeval.save_models_state(s3, bucket, state)
    log.info("Genesis fingerprint saved. Total entries: %d", len(state["models"]))

    # ── Verify upload ─────────────────────────────────────────────────────────
    log.info("Verifying upload by re-downloading from s3://%s/%s …", bucket, preeval.MODELS_STATE_KEY)
    try:
        readback = preeval.load_models_state(s3, bucket)
        if ref_key not in readback.get("models", {}):
            log.error("VERIFY FAILED: key %s not found in downloaded state", ref_key)
            sys.exit(1)
        rb_sha = readback["models"][ref_key].get("sha256_bytes", "")
        local_sha = fp["sha256_bytes"]
        if rb_sha != local_sha:
            log.error("VERIFY FAILED: sha256 mismatch — uploaded=%s downloaded=%s",
                      local_sha[:16], rb_sha[:16])
            sys.exit(1)
        log.info("Upload verified OK. sha256=%s… entries_on_hippius=%d",
                 rb_sha[:16], len(readback.get("models", {})))
    except Exception as exc:
        log.error("Verify step failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
