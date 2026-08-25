from __future__ import annotations

import math
from datetime import datetime, timezone

from loguru import logger as log

from albedo_config import get_model_validation_settings
from model_validation.dedup.sketch import VEC_DIM
from model_validation.opensearch.client import get_client

config = get_model_validation_settings()

STATUS_BANK = "bank"
STATUS_AUDIT = "audit"

_MAPPING = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "model_uri": {"type": "keyword"},
            "hotkey": {"type": "keyword"},
            "coldkey": {"type": "keyword"},
            "repo": {"type": "keyword"},
            "digest": {"type": "keyword"},
            "arch_key": {"type": "keyword"},
            "ws_version": {"type": "keyword"},
            "key_id": {"type": "keyword"},
            "tensors_hash": {"type": "keyword"},
            "status": {"type": "keyword"},
            "is_root": {"type": "boolean"},
            "created_at": {"type": "date"},
            "secs": {"type": "float"},
            "n_tensors": {"type": "integer"},
            "identity_frac": {"type": "object", "enabled": False},
            "verdict": {"type": "object", "enabled": False},
            "sketch_vec": {"type": "knn_vector", "dimension": VEC_DIM},
            "tensors": {"type": "object", "enabled": False},
        }
    },
}

_META = [
    "model_uri",
    "hotkey",
    "coldkey",
    "repo",
    "digest",
    "status",
    "is_root",
    "verdict",
    "created_at",
]


def index_name() -> str:
    return config.OPENSEARCH_INDEX


def ensure_index() -> str:
    name = index_name()
    c = get_client()
    if not c.indices.exists(index=name):
        c.indices.create(index=name, body=_MAPPING)
        log.info("created opensearch index {}", name)
    return name


def _scope(doc: dict) -> list[dict]:
    return [
        {"term": {"arch_key": doc["arch_key"]}},
        {"term": {"ws_version": doc["ws_version"]}},
        {"term": {"key_id": doc["key_id"]}},
    ]


def _own(hotkey: str, coldkey: str) -> list[dict]:
    terms = []
    if hotkey:
        terms.append({"term": {"hotkey": hotkey}})
    if coldkey:
        terms.append({"term": {"coldkey": coldkey}})
    return terms


def put_doc(
    doc: dict,
    *,
    status: str,
    hotkey: str = "",
    coldkey: str = "",
    repo: str = "",
    digest: str = "",
    verdict: dict | None = None,
    is_root: bool = False,
) -> None:
    body = {
        **doc,
        "hotkey": hotkey,
        "coldkey": coldkey,
        "repo": repo,
        "digest": digest,
        "status": status,
        "is_root": is_root,
        "verdict": verdict or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    get_client().index(index=ensure_index(), id=doc["model_uri"], body=body)


def _exact(filters: list[dict], must_not: list[dict]) -> dict | None:
    body = {
        "size": 1,
        "_source": _META,
        "query": {"bool": {"filter": filters, "must_not": must_not}},
    }
    hits = get_client().search(index=ensure_index(), body=body)["hits"]["hits"]
    return hits[0]["_source"] if hits else None


def find_exact(doc: dict, hotkey: str, coldkey: str = "") -> dict | None:
    return _exact(
        [{"term": {"tensors_hash": doc["tensors_hash"]}}, {"term": {"status": STATUS_BANK}}],
        _own(hotkey, coldkey),
    )


def find_exact_own(doc: dict, coldkey: str) -> dict | None:
    return _exact(
        [
            {"term": {"tensors_hash": doc["tensors_hash"]}},
            {"term": {"status": STATUS_BANK}},
            {"term": {"coldkey": coldkey}},
        ],
        [],
    )


def count_scope(doc: dict) -> int:
    body = {"query": {"bool": {"filter": [*_scope(doc), {"term": {"status": STATUS_BANK}}]}}}
    return int(get_client().count(index=ensure_index(), body=body)["count"])


def nearest(doc: dict, hotkey: str, k: int, coldkey: str = "") -> list[tuple[str, float]]:
    return _nearest(
        doc, k, [*_scope(doc), {"term": {"status": STATUS_BANK}}], _own(hotkey, coldkey)
    )


def nearest_own(doc: dict, coldkey: str, k: int) -> list[tuple[str, float]]:
    filters = [*_scope(doc), {"term": {"status": STATUS_BANK}}, {"term": {"coldkey": coldkey}}]
    return _nearest(doc, k, filters, [])


def _nearest(
    doc: dict, k: int, filters: list[dict], must_not: list[dict]
) -> list[tuple[str, float]]:
    body = {
        "size": k,
        "_source": ["model_uri"],
        "query": {
            "script_score": {
                "query": {"bool": {"filter": filters, "must_not": must_not}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "sketch_vec",
                        "query_value": doc["sketch_vec"],
                        "space_type": "l2",
                    },
                },
            }
        },
    }
    hits = get_client().search(index=ensure_index(), body=body)["hits"]["hits"]
    out = []
    for h in hits:
        score = float(h["_score"])
        dist = math.sqrt(max(1.0 / score - 1.0, 0.0)) if score > 0 else float("inf")
        out.append((h["_source"]["model_uri"], dist))
    return out


def root_id(doc: dict) -> str | None:
    body = {
        "size": 1,
        "_source": ["model_uri"],
        "query": {"bool": {"filter": [*_scope(doc), {"term": {"is_root": True}}]}},
    }
    hits = get_client().search(index=ensure_index(), body=body)["hits"]["hits"]
    return hits[0]["_source"]["model_uri"] if hits else None


def has_doc(model_uri: str) -> bool:
    return bool(get_client().exists(index=ensure_index(), id=model_uri))


def fetch(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    res = get_client().mget(index=ensure_index(), body={"ids": ids})
    return {d["_id"]: d["_source"] for d in res["docs"] if d.get("found")}
