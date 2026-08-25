from __future__ import annotations

from dataclasses import dataclass, field

from albedo_config import get_model_validation_settings
from model_validation.dedup.signals import (
    Mats,
    combo_fit,
    rel_dist,
    reuse_table,
    sample_stats,
    spectral,
)


@dataclass(frozen=True)
class Thresholds:
    copy_rel: float
    linear_resid: float
    alpha_z: float
    alpha_min: float
    f_noise: float
    embed_ratio_noise: float
    dens_min: float
    kurt_max: float
    rel_trivial: float
    lora_max_spikes: float
    lora_min_f: float
    head_scale_max: float
    global_scale_max: float
    reuse_cos: float
    topk_partners: int
    permuted_max_ident: float

    @classmethod
    def from_settings(cls, cfg=None) -> Thresholds:
        cfg = cfg or get_model_validation_settings()
        return cls(
            copy_rel=cfg.DEDUP_COPY_REL,
            linear_resid=cfg.DEDUP_LINEAR_RESID,
            alpha_z=cfg.DEDUP_ALPHA_Z,
            alpha_min=cfg.DEDUP_ALPHA_MIN,
            f_noise=cfg.DEDUP_F_NOISE,
            embed_ratio_noise=cfg.DEDUP_EMBED_RATIO_NOISE,
            dens_min=cfg.DEDUP_DENS_MIN,
            kurt_max=cfg.DEDUP_KURT_MAX,
            rel_trivial=cfg.DEDUP_REL_TRIVIAL,
            lora_max_spikes=cfg.DEDUP_LORA_MAX_SPIKES,
            lora_min_f=cfg.DEDUP_LORA_MIN_F,
            head_scale_max=cfg.DEDUP_HEAD_SCALE_MAX,
            global_scale_max=cfg.DEDUP_GLOBAL_SCALE_MAX,
            reuse_cos=cfg.DEDUP_REUSE_COS,
            topk_partners=cfg.DEDUP_TOPK_PARTNERS,
            permuted_max_ident=cfg.DEDUP_PERMUTED_MAX_IDENT,
        )


@dataclass
class Verdict:
    status: str
    reason: str | None
    ancestor: str
    message: str
    notes: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return self.status == "REJECT"


def decide(
    cand: Mats,
    bank: dict[str, Mats],
    root: str | None,
    identity_frac: dict | None,
    th: Thresholds,
) -> Verdict:
    if not bank:
        raise ValueError("empty bank")
    dist = {b: rel_dist(cand, mb) for b, mb in bank.items()}
    order = sorted(dist, key=dist.get)
    a = order[0]
    metrics: dict = {"distances": [(b, dist[b]) for b in order]}
    notes: list[str] = []
    permuted = [k for k, v in (identity_frac or {}).items() if v < th.permuted_max_ident]
    if permuted:
        notes.append(f"PERMUTED LAYOUT in {permuted} (canonicalized)")
        metrics["identity_frac"] = identity_frac

    if dist[a] < th.copy_rel:
        return Verdict("REJECT", "COPY", a, f"sketch identical to {a}", notes, metrics)

    sp = spectral(cand, bank[a])
    ss = sample_stats(cand, bank[a])
    metrics.update({k: sp[k] for k in sp if k != "by_type"})
    metrics["by_type"] = sp["by_type"]
    metrics.update(ss)

    partners = order[1 : 1 + th.topk_partners]
    if root in bank and root != a and root not in partners:
        partners.append(root)
    cf = combo_fit(cand, bank, a, partners)
    if cf:
        metrics["linear"] = cf
    ru = reuse_table(cand, order, bank, root)
    if ru:
        metrics["reuse"] = [
            dict(anc=r[0], model=r[1], ref=r[2], cos=r[3], scale=r[4]) for r in ru[:4]
        ]

    if abs(sp["head_scale"] - 1) > th.head_scale_max:
        notes.append(f"HEAD-RESCALE x{sp['head_scale']:.4f}")
    if abs(sp["global_scale"] - 1) > th.global_scale_max:
        notes.append(f"GLOBAL-RESCALE x{sp['global_scale']:.4f}")
    if ru and abs(ru[0][3]) > th.reuse_cos:
        notes.append(
            f"DELTA-REUSE of {ru[0][1]}-{ru[0][2]} (|cos| {abs(ru[0][3]):.3f}, x{ru[0][4]:.2f})"
        )

    linear = (
        cf is not None
        and cf["resid"] < th.linear_resid
        and any(abs(al) > th.alpha_min and z > th.alpha_z for al, z in zip(cf["alpha"], cf["z"]))
    )
    if linear:
        terms = " ".join(
            f"{al:+.2f}*{b}" for b, al in zip(cf["partners"], cf["alpha"]) if abs(al) > th.alpha_min
        )
        return Verdict(
            "REJECT",
            "LINEAR-COMBO",
            a,
            f"linear combination of banked models: {a} {terms} (resid {cf['resid']:.3f})",
            notes,
            metrics,
        )
    if sp["F"] < th.f_noise:
        return Verdict(
            "REJECT",
            "NOISE-COPY",
            a,
            f"delta to {a} is spectral bulk (F_struct {sp['F']:.3f} < {th.f_noise})",
            notes,
            metrics,
        )
    if sp["embed_ratio"] > th.embed_ratio_noise:
        return Verdict(
            "REJECT",
            "NOISED-COPY",
            a,
            f"embeddings changed {sp['embed_ratio']:.2f}x as much as attention/MLP vs {a} "
            "(dense noise, not training)",
            notes,
            metrics,
        )
    if ss["density"] < th.dens_min or ss["kurtosis"] > th.kurt_max:
        return Verdict(
            "REJECT",
            "SPARSE-EDIT",
            a,
            f"delta to {a} touches {ss['density']:.1%} of sampled weights "
            f"(kurtosis {ss['kurtosis']:.0f})",
            notes,
            metrics,
        )
    if sp["rel_struct"] < th.rel_trivial:
        return Verdict(
            "REJECT",
            "TRIVIAL-EDIT",
            a,
            f"structured change vs {a} is {sp['rel_struct']:.4f} of weight norm "
            f"(< {th.rel_trivial})",
            notes,
            metrics,
        )
    kind = (
        "LORA-like"
        if sp["spikes_med"] <= th.lora_max_spikes and sp["F"] > th.lora_min_f
        else "TRAINED"
    )
    notes.append(kind)
    return Verdict(
        "PASS",
        None,
        a,
        f"{kind} delta on {a} (F_struct {sp['F']:.3f}, rel_struct {sp['rel_struct']:.4f})",
        notes,
        metrics,
    )
