"""Tensor-level model fingerprinting and near-duplicate detection.

EVAL-SERVER ONLY — import from eval.py only, never from validator.py.
Fingerprints are computed from raw safetensors weight files which are only
materialized on the eval (GPU) machine. The validator never touches weights.

Computes per-layer L2 norm vectors from safetensors weight files,
stores them in `uploaded_models_state.json` on Hippius S3, and checks
incoming challengers for near-duplicates before starting the GPU duel.

Only models with Hippius OCI digests (sha256:...) are fingerprinted.
HF-backed genesis kings (hf:...) are skipped — challengers can only
submit via Hippius, so HF hashes are never comparable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("albedo.preeval")

FINGERPRINT_METHOD = "layer_norms_v1"
MODELS_STATE_KEY = "uploaded_models_state.json"

# Module-level cached boto3 client — created lazily on first use.
_S3_CLIENT: object | None = None


def _get_or_create_s3_client(endpoint: str, access: str, secret: str) -> object:
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        from botocore.config import Config as BotoConfig

        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name="decentralized",
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=15,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
    return _S3_CLIENT


def compute_fingerprint(model_dir: Path | str) -> dict:
    from safetensors import safe_open

    from model_store import sha256_safetensors

    model_dir = Path(model_dir)
    layer_norms: dict[str, float] = {}

    # Use torch backend — numpy doesn't support bfloat16 which is standard
    # for LLM weights. Torch is always available via vLLM/transformers.
    try:
        import torch as _torch
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
        _framework = "pt"
        def _norm(t) -> float:
            return float(t.to(device=_device, dtype=_torch.float32).norm().item())
    except ImportError:
        _framework = "numpy"
        def _norm(t) -> float:
            return float(np.linalg.norm(t.astype(np.float32)))

    for sf_path in sorted(model_dir.rglob("*.safetensors")):
        with safe_open(str(sf_path), framework=_framework) as f:
            for key in sorted(f.keys()):
                layer_norms[key] = _norm(f.get_tensor(key))

    if not layer_norms:
        raise ValueError(f"No *.safetensors files found under {model_dir}")

    keys = sorted(layer_norms.keys())
    return {
        "fingerprint_method": FINGERPRINT_METHOD,
        "sha256_bytes": sha256_safetensors(model_dir),
        "layer_keys": keys,
        "norm_vector": [layer_norms[k] for k in keys],
    }


def cosine_similarity(fp_a: dict, fp_b: dict) -> float:
    """Cosine similarity between two layer-norm fingerprint vectors.

    Returns 0.0 if layer_keys differ (different architecture or layer count),
    which prevents false duplicate matches across architectures.
    """
    if fp_a.get("layer_keys") != fp_b.get("layer_keys"):
        return 0.0
    va = np.array(fp_a["norm_vector"], dtype=np.float64)
    vb = np.array(fp_b["norm_vector"], dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


_UNKNOWN_BLOCK = -1  # sentinel: commit block not recorded


def check_duplicate(
    fingerprint: dict,
    state: dict,
    threshold: float,
    skip_key: str | None = None,
    commit_block: int = _UNKNOWN_BLOCK,
) -> tuple[bool, str | None]:
    """Check if a fingerprint is too similar to any stored model.

    Priority is determined by commit_block: a stored model with a lower
    commit_block is considered the original. -1 means unknown for either side;
    unknown blocks are never used to skip a match — when in doubt, flag it.

    First does an exact byte-hash comparison (fast), then falls back to
    cosine similarity on the norm vectors.

    Returns (is_duplicate, matching_ref_key).
      - is_duplicate: True if a stored model with a prior commit block matches.
      - matching_ref_key: the immutable_ref of the matched model, or None.

    skip_key: the challenger's own immutable_ref — excluded so models already
    in the DB for retries/re-evaluation don't self-match.
    """
    challenger_sha256 = fingerprint.get("sha256_bytes", "")

    for ref_key, entry in state.get("models", {}).items():
        if skip_key and ref_key == skip_key:
            continue
        stored_block: int = entry.get("commit_block", _UNKNOWN_BLOCK)
        if stored_block is None:
            stored_block = _UNKNOWN_BLOCK
        # Skip this entry only if both blocks are known AND the challenger
        # was committed strictly before the stored model (challenger is older).
        if commit_block > 0 and stored_block > 0 and commit_block <= stored_block:
            continue
        stored_sha256 = entry.get("sha256_bytes", "")
        if challenger_sha256 and stored_sha256 and challenger_sha256 == stored_sha256:
            log.info(
                "exact-hash duplicate: challenger (block=%d) matches %s (block=%d)",
                commit_block, ref_key, stored_block,
            )
            return True, ref_key
        stored_fp = {
            "layer_keys": entry.get("layer_keys", []),
            "norm_vector": entry.get("norm_vector", []),
        }
        sim = cosine_similarity(fingerprint, stored_fp)
        if sim >= threshold:
            log.info(
                "near-duplicate: cosine_sim=%.6f >= threshold=%.4f "
                "challenger (block=%d) matches %s (block=%d)",
                sim, threshold, commit_block, ref_key, stored_block,
            )
            return True, ref_key
    return False, None


def load_models_state(
    s3_client: object,
    bucket: str,
    key: str = MODELS_STATE_KEY,
) -> dict:
    """Download and parse uploaded_models_state.json from S3.

    Returns an empty skeleton dict if the file doesn't exist yet or if
    the download fails (so the eval pipeline always continues).
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
        return json.loads(obj["Body"].read())
    except Exception as exc:
        code = ""
        try:
            code = exc.response["Error"]["Code"]  # type: ignore[attr-defined]
        except Exception:
            pass
        if code != "NoSuchKey":
            log.warning("Could not load models state from s3://%s/%s: %s", bucket, key, exc)
        return {"version": 1, "updated_at": None, "models": {}}


def save_models_state(
    s3_client: object,
    bucket: str,
    state: dict,
    key: str = MODELS_STATE_KEY,
) -> None:
    """Serialise and upload uploaded_models_state.json to S3."""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    body = json.dumps(state, ensure_ascii=False, indent=2).encode()
    s3_client.put_object(  # type: ignore[attr-defined]
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
        CacheControl="public, max-age=60",
    )
    log.info(
        "saved models state (%d entries) to s3://%s/%s",
        len(state.get("models", {})),
        bucket,
        key,
    )


def load_models_state_local(path: Path | str) -> dict:
    """Load uploaded_models_state.json from a local file path.

    Returns an empty skeleton if the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return {"version": 1, "updated_at": None, "models": {}}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_models_state_local(state: dict, path: Path | str) -> None:
    """Save uploaded_models_state.json to a local file path."""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    log.info("saved models state (%d entries) to %s", len(state.get("models", {})), p)


def add_fingerprint_to_state(
    state: dict,
    ref_key: str,
    fingerprint: dict,
    *,
    hotkey: str,
    verdict: str,
    repo: str = "",
    digest: str = "",
    commit_block: int = _UNKNOWN_BLOCK,
) -> dict:
    """Insert or overwrite a model's fingerprint entry in state.

    commit_block is the chain block at which the miner committed this model.
    It is used as the tiebreaker: the model with the lower commit_block is
    always the original; later submitters with identical weights are copies.
    Use -1 (UNKNOWN_BLOCK) when the block number is not available.

    Mutates state in place and returns it for convenience.
    """
    state.setdefault("models", {})[ref_key] = {
        "repo": repo,
        "digest": digest,
        "hotkey": hotkey,
        "commit_block": commit_block,
        "sha256_bytes": fingerprint.get("sha256_bytes", ""),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "fingerprint_method": fingerprint.get("fingerprint_method", FINGERPRINT_METHOD),
        "layer_keys": fingerprint.get("layer_keys", []),
        "norm_vector": fingerprint.get("norm_vector", []),
    }
    return state
