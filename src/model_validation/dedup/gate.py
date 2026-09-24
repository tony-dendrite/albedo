from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import torch
from loguru import logger as log

from albedo_config import get_model_validation_settings
from model_validation.dedup import bank
from model_validation.dedup.secret import load_secret
from model_validation.dedup.signals import mats, rel_dist
from model_validation.dedup.sketch import fingerprint
from model_validation.dedup.verdict import (
    ALL_REASONS,
    Thresholds,
    Verdict,
    decide,
)

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


def enforced_reasons() -> frozenset[str]:
    """The reject reasons DEDUP_ENFORCE_REASONS allows to fault a miner."""
    raw = (config.DEDUP_ENFORCE_REASONS or "").strip()
    if raw.upper() in ("*", "ALL"):
        return ALL_REASONS
    wanted = {part.strip().upper() for part in raw.split(",") if part.strip()}
    if unknown := wanted - ALL_REASONS:
        log.warning(
            "DEDUP_ENFORCE_REASONS lists unknown reason(s) {} — ignored; known reasons are {}",
            sorted(unknown),
            sorted(ALL_REASONS),
        )
    return frozenset(wanted & ALL_REASONS)


def enforces(reason: str | None, coldkey: str = "") -> bool:
    """True when this reject should fault the miner rather than only be logged.

    Without a coldkey, bank.nearest() cannot exclude the miner's own accepted models and
    _own_copy() is skipped entirely, so the miner's own work can be picked as the ancestor and
    reported as someone else's. That verdict is not safe to enforce: the fault permanently blocks
    the hotkey. The reject is still logged and indexed as audit, so the fingerprint is kept and
    the model never enters the bank.
    """
    if not config.DEDUP_ENFORCE or not reason or not coldkey:
        return False
    return reason.upper() in enforced_reasons()


def fault_code(reason: str | None) -> str:
    """Every enforced reject maps to `duplicate`, the one code that permanently blocks a hotkey
    (db.hotkey_duplicate_block_reason filters on it). OWN-COPY carries its own code so a miner
    re-submitting their own accepted model is told so rather than banned as a copier."""
    if (reason or "").upper() == "OWN-COPY":
        return "duplicate_own"
    return "duplicate"


def device() -> torch.device:
    """No CPU fallback on purpose: torch's CPU and CUDA RNGs differ for the same seed, so a CPU
    sketch uses a different projection basis and, unscoped by device, would poison the bank."""
    if not torch.cuda.is_available():
        raise RuntimeError("dedup needs a CUDA device — torch reports no GPU available")
    return torch.device(f"cuda:{config.DEDUP_GPU}")


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
            "identical weights (tensors_hash)",
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
                f"sketch identical (rel_dist {d:.1e} < {th.copy_rel:.1e})",
                metrics={"ancestor_hotkey": odoc.get("hotkey", ""), "rel_dist": d},
            )
    return None


def run(
    model_dir: str, model_uri: str, hotkey: str, repo: str, digest: str, coldkey: str = ""
) -> GateResult:
    try:
        # Both before ref_dir(), which downloads the seed model when DEDUP_REF_DIR is unset.
        dev = device()
        secret = load_secret(config.DEDUP_SECRET, config.DEDUP_SECRET_FILE)
        doc = fingerprint(model_dir, ref_dir(), secret, dev, model_uri=model_uri)
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
    verdict.metrics["hotkeys_by_model"] = {m: d.get("hotkey", "") for m, d in docs.items()}
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
        "dedup verdict {} — {} {} vs {}: {}",
        model_uri,
        verdict.status,
        verdict.reason or "",
        verdict.ancestor,
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


_KING_REPO = re.compile(r"albedo-qwen3\.6-35b-king-([a-z]+)$", re.IGNORECASE)


def model_label(model_uri: str, hotkey: str = "") -> str:
    """How a fault message names a banked model: genesis, a king, or the uploader's full hotkey."""
    from albedo_config.chain_spec import SEED_REPO

    repo = (model_uri or "").removeprefix("hf://").partition("@")[0]
    king = _KING_REPO.search(repo)
    if (SEED_REPO and repo == SEED_REPO) or (king and king.group(1).lower() == "genesis"):
        return "genesis"
    if king:
        return f"ALBEDO-{king.group(1).upper()}"
    return f"hotkey {hotkey}" if hotkey else "another miner's model"


def public_message(res: GateResult) -> str:
    """The miner-facing reason: what the model duplicates, named instead of its storage path,
    with the measured value and threshold that decided it."""
    v = res.verdict
    if v is None:
        return ""
    hotkeys = dict(v.metrics.get("hotkeys_by_model") or {})
    if v.ancestor:
        hotkeys[v.ancestor] = (
            v.metrics.get("ancestor_hotkey")
            or (res.exact_of or {}).get("hotkey", "")
            or hotkeys.get(v.ancestor, "")
        )
    detail = v.message
    for uri in sorted(filter(None, hotkeys), key=len, reverse=True):
        detail = detail.replace(uri, model_label(uri, hotkeys[uri]))
    who = model_label(v.ancestor or "", hotkeys.get(v.ancestor, ""))
    if v.reason == "OWN-COPY":
        who = f"your own model from {who}"
    return f"duplicate ({v.reason}) of {who}: {detail}"
