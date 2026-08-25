from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from loguru import logger as log

from albedo_config import get_model_validation_settings
from model_validation.dedup import bank
from model_validation.dedup.secret import load_secret
from model_validation.dedup.signals import mats, rel_dist
from model_validation.dedup.sketch import fingerprint
from model_validation.dedup.verdict import Thresholds, Verdict, decide

config = get_model_validation_settings()


@dataclass
class GateResult:
    verdict: Verdict | None = None
    doc: dict | None = None
    exact_of: dict | None = None
    infra_error: str | None = None

    @property
    def rejected(self) -> bool:
        return self.exact_of is not None or bool(self.verdict and self.verdict.rejected)


def thresholds() -> Thresholds:
    return Thresholds.from_settings(config)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{config.DEDUP_GPU}")
    return torch.device("cpu")


def ref_dir() -> str:
    if config.DEDUP_REF_DIR:
        return config.DEDUP_REF_DIR
    from albedo_config.chain_spec import SEED_DIGEST, SEED_REPO
    from model_validation.storage import download_full, make_ref

    return download_full(make_ref(SEED_REPO, SEED_DIGEST))


def _own_copy(doc: dict, coldkey: str) -> Verdict | None:
    if not coldkey:
        return None
    hit = bank.find_exact_own(doc, coldkey)
    if hit:
        return Verdict(
            "REJECT",
            "OWN-COPY",
            hit["model_uri"],
            "identical weights to the miner's own accepted model (tensors_hash)",
            metrics={"ancestor_hotkey": hit.get("hotkey", "")},
        )
    near = bank.nearest_own(doc, coldkey, 3)
    if not near:
        return None
    own_docs = bank.fetch([m for m, _ in near])
    cand = mats(doc)
    th = thresholds()
    for uri, odoc in own_docs.items():
        d = rel_dist(cand, mats(odoc))
        if d < th.copy_rel:
            return Verdict(
                "REJECT",
                "OWN-COPY",
                uri,
                f"sketch identical to the miner's own accepted model (rel_dist {d:.1e})",
                metrics={"ancestor_hotkey": odoc.get("hotkey", ""), "rel_dist": d},
            )
    return None


def run(
    model_dir: str, model_uri: str, hotkey: str, repo: str, digest: str, coldkey: str = ""
) -> GateResult:
    try:
        secret = load_secret(config.DEDUP_SECRET, config.DEDUP_SECRET_FILE)
        doc = fingerprint(model_dir, ref_dir(), secret, device(), model_uri=model_uri)
    except Exception as exc:
        return GateResult(infra_error=f"dedup fingerprint failed: {type(exc).__name__}: {exc}")
    log.info(
        "dedup fingerprint {} — {} tensors, identity_frac={}, {}s",
        model_uri,
        doc["n_tensors"],
        doc["identity_frac"],
        doc["secs"],
    )
    store = dict(hotkey=hotkey, coldkey=coldkey, repo=repo, digest=digest)
    try:
        exact = bank.find_exact(doc, hotkey, coldkey)
        if exact:
            verdict = Verdict(
                "REJECT", "COPY", exact["model_uri"], "identical weights (tensors_hash)"
            )
            bank.put_doc(doc, status=bank.STATUS_AUDIT, verdict=asdict(verdict), **store)
            return GateResult(verdict=verdict, doc=doc, exact_of=exact)
        own = _own_copy(doc, coldkey)
        if own is not None:
            bank.put_doc(doc, status=bank.STATUS_AUDIT, verdict=asdict(own), **store)
            return GateResult(verdict=own, doc=doc)
        if bank.count_scope(doc) == 0:
            return GateResult(
                doc=doc,
                infra_error=(
                    f"dedup bank is empty for arch_key={doc['arch_key']} key_id={doc['key_id']} "
                    "— bootstrap the bank first"
                ),
            )
        near = bank.nearest(doc, hotkey, config.DEDUP_NEAREST_K, coldkey)
        root = bank.root_id(doc)
        ids = [m for m, _ in near]
        if root and root not in ids:
            ids.append(root)
        docs = bank.fetch(ids)
    except Exception as exc:
        return GateResult(doc=doc, infra_error=f"dedup opensearch failed: {exc}")
    if not docs:
        return GateResult(doc=doc, infra_error="dedup nearest search returned no documents")
    if not near:
        verdict = Verdict(
            "PASS", None, root or "", "no other miner's model in the bank to compare with"
        )
        bank.put_doc(doc, status=bank.STATUS_BANK, verdict=asdict(verdict), **store)
        return GateResult(verdict=verdict, doc=doc)

    verdict = decide(
        mats(doc), {m: mats(d) for m, d in docs.items()}, root, doc["identity_frac"], thresholds()
    )
    verdict.metrics["opensearch_nearest"] = near
    verdict.metrics["ancestor_hotkey"] = docs.get(verdict.ancestor, {}).get("hotkey", "")
    try:
        bank.put_doc(
            doc,
            status=bank.STATUS_BANK if verdict.status == "PASS" else bank.STATUS_AUDIT,
            verdict=asdict(verdict),
            **store,
        )
    except Exception as exc:
        return GateResult(
            verdict=verdict, doc=doc, infra_error=f"dedup opensearch index failed: {exc}"
        )
    log.info(
        "dedup verdict {} — {} {}: {}",
        model_uri,
        verdict.status,
        verdict.reason or "",
        verdict.message,
    )
    return GateResult(verdict=verdict, doc=doc)


_PUBLIC_METRICS = (
    "F",
    "rel",
    "rel_struct",
    "embed_ratio",
    "density",
    "kurtosis",
    "spikes_med",
    "touched",
    "head_scale",
    "global_scale",
    "identity_frac",
    "linear",
    "reuse",
    "distances",
)


def public_summary(res: GateResult) -> dict:
    if not res.rejected or res.verdict is None:
        return {"dedup": "pass"}
    v = res.verdict
    out: dict = {
        "dedup": "reject",
        "reason": v.reason,
        "duplicate_of": v.ancestor,
        "notes": v.notes,
    }
    if res.exact_of:
        out["duplicate_of_hotkey"] = res.exact_of.get("hotkey", "")
        out["exact_weights_match"] = True
        return out
    if v.reason == "OWN-COPY":
        out["duplicate_of_hotkey"] = v.metrics.get("ancestor_hotkey", "")
        out["own_model"] = True
        return out
    out["duplicate_of_hotkey"] = v.metrics.get("ancestor_hotkey", "")
    out["metrics"] = {k: v.metrics[k] for k in _PUBLIC_METRICS if k in v.metrics}
    return out


def public_message(res: GateResult) -> str:
    v = res.verdict
    if v is None:
        return ""
    return f"duplicate of {v.ancestor}: {v.reason} — {v.message}"
