"""Weight fingerprinting (per-tensor norms + samples) and the OpenSearch dedup corpus."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import mmap
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from loguru import logger

from albedo.settings import get_settings

FINGERPRINT_METHOD = "layer_norms_v2_with_samples"
SAMPLE_K = 16  # deterministic value samples drawn per tensor
_UNCHANGED_COSINE = 1.0 - 1e-6  # a tensor counts as unchanged at/above this per-sample cosine
_NP_DTYPE = {"F64": "<f8", "F32": "<f4", "F16": "<f2"}  # bf16 handled specially below


# ── Fingerprint math (byte-for-byte from config_validation) ──────────────────


def _deterministic_indices(key: str, n: int, k: int) -> list[int]:
    # k stable indices into a length-n tensor, derived from its key (shard-order invariant).
    if n <= 0:
        return [0] * k
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=4 * k).digest()
    return [int.from_bytes(h[i * 4 : (i + 1) * 4], "big") % n for i in range(k)]


def _to_f32(raw: bytes, dtype: str) -> np.ndarray | None:
    # Decodes a raw safetensors tensor buffer to 1-D float32, or None for non-float tensors.
    if dtype in _NP_DTYPE:
        return np.frombuffer(raw, dtype=_NP_DTYPE[dtype]).astype(np.float32, copy=False)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (u16 << 16).view(np.float32)
    return None


def _iter_tensors(shard: Path) -> Iterator[tuple[str, np.ndarray]]:
    # Yields (key, f32_flat_array) for every float tensor in a safetensors shard (numpy, no torch).
    with open(shard, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            header_len = int.from_bytes(mm[:8], "little")
            header = json.loads(mm[8 : 8 + header_len])
            data_start = 8 + header_len
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                arr = _to_f32(mm[data_start + start : data_start + end], info["dtype"])
                if arr is not None:
                    yield key, arr
        finally:
            mm.close()


def compute_fingerprint(model_dir: str) -> dict:
    # {"method", "layer_keys", "norm_vector", "tensor_samples"}; layer_keys sorted so the
    # fingerprint is shard-order invariant.
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors files found in {model_dir!r}")

    norms: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    for shard in shards:
        for key, flat in _iter_tensors(shard):
            n = int(flat.shape[0])
            norms[key] = float(np.sqrt(np.square(flat.astype(np.float64)).sum()))
            idxs = _deterministic_indices(key, n, SAMPLE_K)
            samples[key] = [float(flat[i]) for i in idxs]

    keys = sorted(norms)
    return {
        "method": FINGERPRINT_METHOD,
        "layer_keys": keys,
        "norm_vector": [norms[k] for k in keys],
        "tensor_samples": [samples[k] for k in keys],
    }


def _vector_cosine(a: list[float], b: list[float]) -> float:
    # Cosine of two equal-length vectors; 0.0 on empty, mismatched, or zero-magnitude input.
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / mag if mag else 0.0


def similarity(fp_a: dict, fp_b: dict) -> float:
    # Fraction of tensors whose sampled values are ~unchanged; 0.0 when layer_keys differ.
    if fp_a.get("layer_keys") != fp_b.get("layer_keys"):
        return 0.0

    sa, sb = fp_a.get("tensor_samples"), fp_b.get("tensor_samples")
    if sa and sb and len(sa) == len(sb):
        unchanged = 0
        for a, b in zip(sa, sb):
            cos = _vector_cosine(a, b)
            if (not any(a) and not any(b)) or cos >= _UNCHANGED_COSINE:
                unchanged += 1
        return unchanged / len(sa)

    return _vector_cosine(fp_a.get("norm_vector", []), fp_b.get("norm_vector", []))


# ── OpenSearch dedup corpus (kNN prefilter + exact rerank) ────────────────────


@functools.lru_cache(maxsize=1)
def get_client():
    # Cached OpenSearch client for the dedup corpus.
    from opensearchpy import OpenSearch

    os_cfg = get_settings().opensearch
    auth = (os_cfg.user, os_cfg.password) if os_cfg.user else None
    return OpenSearch(
        hosts=[os_cfg.url],
        http_auth=auth,
        use_ssl=os_cfg.url.lower().startswith("https"),
        verify_certs=False,
        ssl_show_warn=False,
        timeout=30,
    )


def health() -> bool:
    # True if the cluster is reachable and green/yellow.
    try:
        status = get_client().cluster.health().get("status")
        logger.info(f"[fingerprint] opensearch health: {status}")
        return status in ("green", "yellow")
    except Exception as exc:  # noqa: BLE001 - unreachable cluster is a startup gate, not a crash
        logger.warning(f"[fingerprint] opensearch health check failed: {exc}")
        return False


def _mapping(dim: int) -> dict[str, Any]:
    # Index body: lucene HNSW cosine on norm_vector, full fingerprint stored but not indexed.
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "key": {"type": "keyword"},
                "hotkey": {"type": "keyword"},
                "repo": {"type": "keyword"},
                "digest": {"type": "keyword"},
                "model_uri": {"type": "keyword"},
                "created_at": {"type": "date"},
                "norm_vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
                },
                "fingerprint": {"type": "object", "enabled": False},
            }
        },
    }


def ensure_index(dim: int) -> str:
    # Per-dimension index (knn_vector dimension is fixed, so each tensor count = one architecture).
    name = f"{get_settings().opensearch.index}_{dim}"
    c = get_client()
    if not c.indices.exists(index=name):
        c.indices.create(index=name, body=_mapping(dim))
        logger.info(f"[fingerprint] created opensearch index {name} (knn dim={dim})")
    return name


def find_duplicate(fp: dict, hotkey: str) -> dict:
    # kNN-prefilters nearest models then reranks with exact tensor similarity; skips own hotkey.
    v = get_settings().validation
    vec = fp.get("norm_vector") or []
    index = ensure_index(len(vec))
    body = {
        "size": v.knn_candidates,
        "_source": ["key", "hotkey", "model_uri", "fingerprint"],
        "query": {"knn": {"norm_vector": {"vector": vec, "k": v.knn_candidates}}},
    }
    hits = get_client().search(index=index, body=body)["hits"]["hits"]

    best_sim, matched = 0.0, {"key": "", "hotkey": "", "model_uri": ""}
    for hit in hits:
        src = hit.get("_source", {})
        if hotkey and src.get("hotkey") == hotkey:
            continue  # a miner's own prior model is not a duplicate of itself
        sim = similarity(fp, src.get("fingerprint", {}))
        if sim > best_sim:
            best_sim = sim
            matched = {
                "key": src.get("key", ""),
                "hotkey": src.get("hotkey", ""),
                "model_uri": src.get("model_uri", ""),
            }

    is_dup = best_sim >= v.sim_threshold
    if is_dup:
        logger.warning(
            f"[fingerprint] duplicate: sim={best_sim:.6f} >= {v.sim_threshold} vs {matched['key']}"
        )
    return {
        "is_duplicate": is_dup,
        "similarity": best_sim,
        "threshold": v.sim_threshold,
        "matched_key": matched["key"] if is_dup else "",
        "matched_hotkey": matched["hotkey"] if is_dup else "",
        "matched_model_uri": matched["model_uri"] if is_dup else "",
        "candidates_checked": len(hits),
    }


def index_fingerprint(
    key: str, fp: dict, *, hotkey: str, repo: str, digest: str, model_uri: str, created_at: str
) -> None:
    # Indexes a non-duplicate model's fingerprint into the per-dimension corpus (id=key).
    vec = fp.get("norm_vector") or []
    index = ensure_index(len(vec))
    get_client().index(
        index=index,
        id=key,
        body={
            "key": key,
            "hotkey": hotkey,
            "repo": repo,
            "digest": digest,
            "model_uri": model_uri,
            "created_at": created_at,
            "norm_vector": vec,
            "fingerprint": fp,
        },
    )
