from __future__ import annotations

import base64
import math
from functools import lru_cache

import numpy as np

from model_validation.dedup.layout import BODY_TYPES, EMBED_TYPES, HEAD, ttype

Mats = dict[str, tuple[np.ndarray, float, np.ndarray]]


def mats(doc: dict) -> Mats:
    out: Mats = {}
    for t in doc["tensors"]:
        s = np.frombuffer(base64.b64decode(t["s"]), dtype=np.float32).reshape(t["k"], t["k"])
        x = np.frombuffer(base64.b64decode(t["x"]), dtype=np.float32)
        out[t["name"]] = (s, float(t["wnorm"]), x)
    return out


def rel_dist(mc: Mats, mb: Mats) -> float:
    num = den = 0.0
    for name, (sb, _, _) in mb.items():
        if name in mc:
            d = mc[name][0] - sb
            num += float((d * d).sum())
            den += float((sb * sb).sum())
    return math.sqrt(num / den) if den else float("inf")


def delta_vec(mc: Mats, ma: Mats) -> np.ndarray:
    return np.concatenate([(mc[n][0] - ma[n][0]).ravel() for n in sorted(ma) if n in mc]).astype(
        np.float64
    )


def sample_stats(mc: Mats, ma: Mats) -> dict:
    dens, kurt = [], []
    for name, (_, _, xa) in ma.items():
        if name not in mc:
            continue
        d = mc[name][2] - xa
        if not np.any(d):
            continue
        dens.append(float((d != 0).mean()))
        z = (d - d.mean()) / (d.std() + 1e-30)
        kurt.append(float((z**4).mean() - 3))
    return dict(
        density=(float(np.median(dens)) if dens else 0.0),
        kurtosis=(float(np.median(kurt)) if kurt else 0.0),
    )


@lru_cache(maxsize=None)
def gd_factor(beta: float) -> float:
    lam = math.sqrt(
        2 * (beta + 1) + 8 * beta / ((beta + 1) + math.sqrt(beta * beta + 14 * beta + 1))
    )
    lo, hi = (1 - math.sqrt(beta)) ** 2, (1 + math.sqrt(beta)) ** 2
    t = np.linspace(lo, hi, 20001)[1:-1]
    f = np.sqrt(np.maximum((hi - t) * (t - lo), 0)) / (2 * math.pi * beta * t)
    cdf = np.cumsum(f) * (t[1] - t[0])
    cdf /= cdf[-1]
    return lam / math.sqrt(float(t[np.searchsorted(cdf, 0.5)]))


def spectral(mc: Mats, ma: Mats) -> dict:
    fac = gd_factor(1.0)
    e_tot = es = 0.0
    spikes, scales = [], []
    per_type: dict[str, list[float]] = {}
    rel_num = rel_den = 0.0
    head_scale = 1.0
    for name, (sa, _, _) in ma.items():
        if name not in mc:
            continue
        sc = mc[name][0]
        ea = float((sa * sa).sum())
        scale = float((sc * sa).sum() / ea) if ea else 1.0
        if name == HEAD:
            head_scale = scale
        d = sc - scale * sa
        e = float((d * d).sum())
        rel_num += e
        rel_den += ea
        g = per_type.setdefault(ttype(name), [0.0, 0.0, 0, 0.0])
        g[3] += ea
        if float(((sc - sa) ** 2).sum()) == 0.0:
            continue
        scales.append(scale)
        if e == 0.0:
            continue
        s = np.linalg.svd(d, compute_uv=False)
        tau = float(np.median(s)) * fac
        sp = int((s > tau).sum())
        f = float((s[s > tau] ** 2).sum() / e)
        e_tot += e
        es += f * e
        spikes.append(sp)
        g[0] += e
        g[1] += f * e
        g[2] += 1
    F = es / e_tot if e_tot else 0.0
    rel = math.sqrt(rel_num / rel_den) if rel_den else 0.0
    by_type = {
        k: dict(
            F=(v[1] / v[0] if v[0] else 0.0),
            n=v[2],
            rel=(math.sqrt(v[0] / v[3]) if v[3] else 0.0),
        )
        for k, v in per_type.items()
    }
    body = [by_type[k]["rel"] for k in BODY_TYPES if k in by_type]
    emb = [by_type[k]["rel"] for k in EMBED_TYPES if k in by_type]
    embed_ratio = float(np.mean(emb) / max(np.mean(body), 1e-12)) if body and emb else 0.0
    return dict(
        F=F,
        rel=rel,
        rel_struct=math.sqrt(F) * rel,
        touched=len(spikes),
        spikes_med=(float(np.median(spikes)) if spikes else 0.0),
        embed_ratio=embed_ratio,
        head_scale=head_scale,
        global_scale=(float(np.median(scales)) if scales else 1.0),
        by_type=by_type,
    )


def combo_fit(cand: Mats, bank: dict[str, Mats], a: str, partners: list[str]) -> dict | None:
    x = delta_vec(cand, bank[a])
    if not partners:
        return None
    D = np.stack([delta_vec(bank[b], bank[a]) for b in partners], axis=1)
    xx = float(x @ x)
    if xx == 0:
        return None
    alpha, *_ = np.linalg.lstsq(D, x, rcond=None)
    r = x - D @ alpha
    resid = math.sqrt(float(r @ r) / xx)
    sigma = math.sqrt(float(r @ r) / max(len(r) - D.shape[1], 1))
    try:
        cov = np.linalg.inv(D.T @ D)
        z = [
            abs(al) / (sigma * math.sqrt(cov[i, i])) if sigma > 0 else float("inf")
            for i, al in enumerate(alpha)
        ]
    except np.linalg.LinAlgError:
        z = [0.0] * len(alpha)
    return dict(partners=list(partners), alpha=[float(v) for v in alpha], z=z, resid=resid)


def reuse_table(cand: Mats, order: list[str], bank: dict[str, Mats], root: str | None) -> list:
    a = order[0]
    dc = delta_vec(cand, bank[a])
    nc = np.linalg.norm(dc)
    if nc == 0:
        return []
    rows = []
    for b, mb in bank.items():
        if b == a or b == root:
            continue
        others = {k: v for k, v in bank.items() if k != b}
        if not others:
            continue
        refs = []
        if root in others:
            refs.append((root, delta_vec(mb, others[root])))
        anc_b = min(others, key=lambda k: rel_dist(mb, others[k]))
        if anc_b not in (root, a):
            refs.append((anc_b, delta_vec(mb, others[anc_b])))
        for anc, db in refs:
            nb = np.linalg.norm(db)
            if nb == 0:
                continue
            rows.append((a, b, anc, float(dc @ db / (nc * nb)), float(nc / nb)))
    rows.sort(key=lambda r: -abs(r[3]))
    return rows
