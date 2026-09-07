from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from model_validation.dedup.layout import BODY_TYPES, EMBED_TYPES, HEAD, ttype

# One fingerprint entry per tensor: (sketch S = psi @ W @ omega, ||W||_F, the weights at the
# secret sample positions).  Every signal below is arithmetic on sketches, never on weights:
# the sketch is linear in W, so the difference of two sketches is the sketch of the weight
# difference, and spectra, distances and merge fits follow from the fingerprints alone.
Mats = dict[str, tuple[np.ndarray, float, np.ndarray]]

# Fraction of the saturated fit's explained energy that combo_fit's minimal partner set must keep.
COMBO_KEEP = 0.95


def mats(doc: dict) -> Mats:
    """Rehydrate the sketches and samples of one fingerprint document."""
    out: Mats = {}
    for entry in doc["tensors"]:
        k = entry["k"]
        sketch = np.frombuffer(base64.b64decode(entry["s"]), dtype=np.float32).reshape(k, k)
        samples = np.frombuffer(base64.b64decode(entry["x"]), dtype=np.float32)
        out[entry["name"]] = (sketch, float(entry["wnorm"]), samples)
    return out


def rel_dist(cand: Mats, ref: Mats) -> float:
    """Relative L2 distance between two fingerprints, over the tensors they share."""
    delta_energy = ref_energy = 0.0
    for name, (ref_sketch, _, _) in ref.items():
        if name not in cand:
            continue
        delta = cand[name][0] - ref_sketch
        delta_energy += float((delta * delta).sum())
        ref_energy += float((ref_sketch * ref_sketch).sum())
    return math.sqrt(delta_energy / ref_energy) if ref_energy else float("inf")


def delta_vec(cand: Mats, anc: Mats) -> np.ndarray:
    """Candidate-minus-ancestor sketch delta, flattened into one vector.  Name order, so deltas
    from different model pairs line up element-wise as columns of combo_fit's design matrix."""
    return np.concatenate(
        [(cand[name][0] - anc[name][0]).ravel() for name in sorted(anc) if name in cand]
    ).astype(np.float64)


def sample_stats(cand: Mats, anc: Mats) -> dict:
    """Training nudges nearly every sampled weight a little (density ~1, kurtosis ~0); a sparse
    edit moves few a lot.  Medians over tensors, so one odd tensor cannot carry it."""
    densities, kurtoses = [], []
    for name, (_, _, anc_samples) in anc.items():
        if name not in cand:
            continue
        delta = cand[name][2] - anc_samples
        if not np.any(delta):
            continue
        densities.append(float((delta != 0).mean()))
        z = (delta - delta.mean()) / (delta.std() + 1e-30)
        kurtoses.append(float((z**4).mean() - 3))
    return dict(
        density=(float(np.median(densities)) if densities else 0.0),
        kurtosis=(float(np.median(kurtoses)) if kurtoses else 0.0),
    )


@lru_cache(maxsize=None)
def gd_factor(beta: float) -> float:
    """Donoho-Gavish lambda(beta) / sqrt(Marchenko-Pastur bulk median).  Times the median observed
    singular value it gives the optimal cut between noise bulk and signal spikes."""
    lam = math.sqrt(
        2 * (beta + 1) + 8 * beta / ((beta + 1) + math.sqrt(beta * beta + 14 * beta + 1))
    )
    lo, hi = (1 - math.sqrt(beta)) ** 2, (1 + math.sqrt(beta)) ** 2
    grid = np.linspace(lo, hi, 20001)[1:-1]
    density = np.sqrt(np.maximum((hi - grid) * (grid - lo), 0)) / (2 * math.pi * beta * grid)
    cdf = np.cumsum(density) * (grid[1] - grid[0])
    cdf /= cdf[-1]
    mp_median = float(grid[np.searchsorted(cdf, 0.5)])
    return lam / math.sqrt(mp_median)


@dataclass
class TypeAcc:
    """Energy accumulator for one tensor type (embed_tokens, self_attn, experts, ...)."""

    anchor_energy: float = 0.0  # sum of ||S_anc||^2 over EVERY tensor of the type
    delta_energy: float = 0.0  # sum of ||delta||^2, only over tensors that moved
    structured_energy: float = 0.0  # sum of sigma^2 over the above-bulk singular values
    n_touched: int = 0  # how many tensors of the type moved


def spectral(cand: Mats, anc: Mats) -> dict:
    """Fit out the per-tensor scalar (so a rescale is not mistaken for work), then split the rest
    against the MP bulk: iid noise stays under the edge, a training run spikes above it."""
    threshold_factor = gd_factor(1.0)  # the sketches are square, so beta = 1
    delta_energy_total = structured_energy_total = 0.0
    spike_counts: list[int] = []
    tensor_scales: list[float] = []
    per_type: dict[str, TypeAcc] = {}
    model_delta_energy = model_anchor_energy = 0.0
    head_scale = 1.0

    for name, (anc_sketch, _, _) in anc.items():
        if name not in cand:
            continue
        cand_sketch = cand[name][0]

        # Best scalar multiple of the ancestor, and the part of the candidate it cannot explain.
        anchor_energy = float((anc_sketch * anc_sketch).sum())
        scale = float((cand_sketch * anc_sketch).sum() / anchor_energy) if anchor_energy else 1.0
        if name == HEAD:
            head_scale = scale
        delta_sketch = cand_sketch - scale * anc_sketch
        delta_energy = float((delta_sketch * delta_sketch).sum())

        model_delta_energy += delta_energy
        model_anchor_energy += anchor_energy

        acc = per_type.setdefault(ttype(name), TypeAcc())
        # Accumulated before both skips below: untouched tensors belong in the denominator, so
        # by_type[...]["rel"] reads as "the share of this type's magnitude that changed".
        acc.anchor_energy += anchor_energy

        if float(((cand_sketch - anc_sketch) ** 2).sum()) == 0.0:
            continue  # bit-identical tensor — not even a rescale
        tensor_scales.append(scale)  # recorded before the next skip: a pure rescale still counts
        if delta_energy == 0.0:
            continue  # exactly a scaled copy — no residual left to analyse

        singular_values = np.linalg.svd(delta_sketch, compute_uv=False)
        bulk_edge = float(np.median(singular_values)) * threshold_factor
        is_spike = singular_values > bulk_edge
        n_spikes = int(is_spike.sum())
        structured_energy = float((singular_values[is_spike] ** 2).sum())

        delta_energy_total += delta_energy
        structured_energy_total += structured_energy
        spike_counts.append(n_spikes)

        acc.delta_energy += delta_energy
        acc.structured_energy += structured_energy
        acc.n_touched += 1

    struct_frac = structured_energy_total / delta_energy_total if delta_energy_total else 0.0
    rel = math.sqrt(model_delta_energy / model_anchor_energy) if model_anchor_energy else 0.0
    by_type = {
        type_name: dict(
            F=(acc.structured_energy / acc.delta_energy if acc.delta_energy else 0.0),
            n=acc.n_touched,
            rel=(math.sqrt(acc.delta_energy / acc.anchor_energy) if acc.anchor_energy else 0.0),
        )
        for type_name, acc in per_type.items()
    }
    # Dense multiplicative noise moves embeddings as hard as the body; training barely touches
    # them.  Unweighted mean over types, so lm_head alone can carry half the numerator.
    body_rel = [by_type[t]["rel"] for t in BODY_TYPES if t in by_type]
    embed_rel = [by_type[t]["rel"] for t in EMBED_TYPES if t in by_type]
    embed_ratio = (
        float(np.mean(embed_rel) / max(np.mean(body_rel), 1e-12)) if body_rel and embed_rel else 0.0
    )
    return dict(
        F=struct_frac,
        rel=rel,
        rel_struct=math.sqrt(struct_frac) * rel,
        touched=len(spike_counts),
        spikes_med=(float(np.median(spike_counts)) if spike_counts else 0.0),
        embed_ratio=embed_ratio,
        head_scale=head_scale,
        global_scale=(float(np.median(tensor_scales)) if tensor_scales else 1.0),
        by_type=by_type,
    )


def combo_fit(cand: Mats, bank: dict[str, Mats], ancestor: str, partners: list[str]) -> dict | None:
    """Fit the candidate's delta as a combination of banked deltas — a merge is linear in sketch
    space.  `resid_min`/`used` refit on the fewest partners explaining COMBO_KEEP of `resid`."""
    target = delta_vec(cand, bank[ancestor])
    if not partners:
        return None
    design = np.stack([delta_vec(bank[p], bank[ancestor]) for p in partners], axis=1)
    target_energy = float(target @ target)
    if target_energy == 0:
        return None
    alpha, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ alpha
    residual_energy = float(residual @ residual)
    resid = math.sqrt(residual_energy / target_energy)
    sigma = math.sqrt(residual_energy / max(len(residual) - design.shape[1], 1))
    try:
        cov = np.linalg.inv(design.T @ design)
        z = [
            abs(coef) / (sigma * math.sqrt(cov[i, i])) if sigma > 0 else float("inf")
            for i, coef in enumerate(alpha)
        ]
    except np.linalg.LinAlgError:
        z = [0.0] * len(alpha)
    resid_min, used = resid, list(partners)
    used_alpha = [float(v) for v in alpha]
    explained = target_energy - residual_energy
    rank = sorted(range(len(alpha)), key=lambda i: -abs(alpha[i]))
    if explained > 0:
        for n in range(1, len(rank)):
            cols = rank[:n]
            sub, *_ = np.linalg.lstsq(design[:, cols], target, rcond=None)
            sub_energy = float(np.square(target - design[:, cols] @ sub).sum())
            if target_energy - sub_energy >= COMBO_KEEP * explained:
                resid_min = math.sqrt(sub_energy / target_energy)
                used = [partners[i] for i in cols]
                used_alpha = [float(v) for v in sub]
                break
    return dict(
        partners=list(partners),
        alpha=[float(v) for v in alpha],
        z=z,
        resid=resid,
        resid_min=resid_min,
        used=used,
        used_alpha=used_alpha,
    )


class ReuseRow(NamedTuple):
    """One comparison of the candidate's delta against a banked model's own delta."""

    ancestor: str  # the candidate's nearest banked model, the frame both deltas are measured in
    model: str  # the banked model whose own delta is being compared
    ref: str  # what that model's delta was taken against (the root, or its own nearest)
    cos: float  # cosine between the two deltas; |cos| ~ 1 means the same delta was reused
    scale: float  # ratio of norms, i.e. by how much the reused delta was scaled


def reuse_table(
    cand: Mats, order: list[str], bank: dict[str, Mats], root: str | None
) -> list[ReuseRow]:
    """Rank banked models by how much their own delta looks like the candidate's, best first.
    Names whose delta was reused and at what multiplier — catches a lifted adapter."""
    ancestor = order[0]
    cand_delta = delta_vec(cand, bank[ancestor])
    cand_norm = float(np.linalg.norm(cand_delta))
    if cand_norm == 0:
        return []
    rows: list[ReuseRow] = []
    for model, model_mats in bank.items():
        if model in (ancestor, root):
            continue
        others = {name: m for name, m in bank.items() if name != model}
        if not others:
            continue
        # Measure this model's own delta against the root and against its own nearest neighbour;
        # a reused delta shows up whichever of the two frames it was originally built in.
        refs = []
        if root in others:
            refs.append((root, delta_vec(model_mats, others[root])))
        model_anc = min(others, key=lambda name: rel_dist(model_mats, others[name]))
        if model_anc not in (root, ancestor):
            refs.append((model_anc, delta_vec(model_mats, others[model_anc])))
        for ref, ref_delta in refs:
            ref_norm = float(np.linalg.norm(ref_delta))
            if ref_norm == 0:
                continue
            rows.append(
                ReuseRow(
                    ancestor,
                    model,
                    ref,
                    float(cand_delta @ ref_delta / (cand_norm * ref_norm)),
                    cand_norm / ref_norm,
                )
            )
    rows.sort(key=lambda row: -abs(row.cos))
    return rows
