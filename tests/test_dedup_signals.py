from __future__ import annotations

import numpy as np

from model_validation.dedup import signals

K = 64
NAMES = [
    "embed_tokens.weight",
    "lm_head.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.linear_attn.in_proj_qkv.weight",
    "layers.0.mlp.shared_expert.up_proj.weight",
    "layers.0.mlp.experts.gate_up_proj",
    "layers.1.self_attn.q_proj.weight",
    "layers.1.mlp.shared_expert.down_proj.weight",
]


def base_mats(seed=0):
    rng = np.random.default_rng(seed)
    return {
        n: (
            rng.standard_normal((K, K)).astype(np.float32),
            100.0,
            rng.standard_normal(2048).astype(np.float32),
        )
        for n in NAMES
    }


def perturb(m, sk_fn, x_fn):
    return {n: (sk_fn(n, s), wn, x_fn(n, x)) for n, (s, wn, x) in m.items()}


def lowrank(rng, r, scale):
    return (rng.standard_normal((K, r)) @ rng.standard_normal((r, K))).astype(np.float32) * scale


def test_rel_dist_zero_for_identical_and_scales():
    m = base_mats()
    assert signals.rel_dist(m, m) == 0.0
    d = perturb(m, lambda n, s: s * 1.1, lambda n, x: x)
    assert abs(signals.rel_dist(d, m) - 0.1) < 1e-4


def test_spectral_iid_noise_is_bulk_and_lowrank_is_structure():
    rng = np.random.default_rng(1)
    m = base_mats()
    noisy = perturb(
        m, lambda n, s: s + rng.standard_normal((K, K)).astype(np.float32) * 0.05, lambda n, x: x
    )
    sp = signals.spectral(noisy, m)
    assert sp["F"] < 0.10
    lora = perturb(m, lambda n, s: s + lowrank(rng, 8, 0.05), lambda n, x: x)
    sp = signals.spectral(lora, m)
    assert sp["F"] > 0.9
    assert 6 <= sp["spikes_med"] <= 12


def test_spectral_scalar_scale_is_factored_out():
    m = base_mats()
    rng = np.random.default_rng(2)
    body = {n: lowrank(rng, 8, 0.05) for n in NAMES}
    tuned = perturb(m, lambda n, s: s + body[n], lambda n, x: x)
    rescaled = perturb(
        tuned, lambda n, s: s * 1.756 if n == "lm_head.weight" else s, lambda n, x: x
    )
    a, b = signals.spectral(tuned, m), signals.spectral(rescaled, m)
    assert abs(b["head_scale"] - 1.756) < 0.01
    assert abs(a["head_scale"] - 1.0) < 0.01
    assert abs(a["F"] - b["F"]) < 0.02
    assert a["rel"] < b["rel"] < a["rel"] * 1.756


def test_embed_ratio_separates_training_from_dense_noise():
    rng = np.random.default_rng(3)
    m = base_mats()
    body_only = perturb(
        m,
        lambda n, s: s + (lowrank(rng, 8, 0.05) if "embed" not in n and "lm_head" not in n else 0),
        lambda n, x: x,
    )
    assert signals.spectral(body_only, m)["embed_ratio"] < 0.05
    dense = perturb(
        m,
        lambda n, s: s * (1 + rng.standard_normal((K, K)).astype(np.float32) * 0.05),
        lambda n, x: x,
    )
    assert signals.spectral(dense, m)["embed_ratio"] > 0.8


def test_sample_stats_density_and_kurtosis():
    rng = np.random.default_rng(4)
    m = base_mats()
    dense = perturb(
        m, lambda n, s: s, lambda n, x: x + rng.standard_normal(2048).astype(np.float32) * 0.01
    )
    st = signals.sample_stats(dense, m)
    assert st["density"] > 0.99 and abs(st["kurtosis"]) < 1.0

    def sparse(n, x):
        d = np.zeros(2048, dtype=np.float32)
        d[rng.choice(2048, 40, replace=False)] = 0.5
        return x + d

    st = signals.sample_stats(perturb(m, lambda n, s: s, sparse), m)
    assert st["density"] < 0.05 and st["kurtosis"] > 20


def test_combo_fit_recovers_merge_alpha():
    rng = np.random.default_rng(5)
    root = base_mats(0)
    a = perturb(root, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
    b = perturb(root, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
    merged = {n: (0.65 * a[n][0] + 0.35 * root[n][0], 100.0, a[n][2]) for n in NAMES}
    fit = signals.combo_fit(merged, {"a": a, "root": root, "b": b}, "a", ["root", "b"])
    assert abs(fit["alpha"][0] - 0.35) < 0.01
    assert abs(fit["alpha"][1]) < 0.01
    assert fit["resid"] < 0.02
    assert fit["z"][0] > 8


def test_combo_fit_minimal_set_drops_the_partners_that_only_absorb_haze():
    rng = np.random.default_rng(5)
    root = base_mats(0)
    a = perturb(root, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
    others = {
        name: perturb(root, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
        for name in ("b", "c", "d")
    }
    # A two-model merge plus a little genuinely new work, so the saturated fit hands the
    # leftovers to b, c and d as small coefficients that explain almost none of the energy.
    merged = perturb(
        {n: (0.65 * a[n][0] + 0.35 * root[n][0], 100.0, a[n][2]) for n in NAMES},
        lambda n, s: s + lowrank(rng, 16, 0.004),
        lambda n, x: x,
    )
    fit = signals.combo_fit(merged, {"a": a, "root": root, **others}, "a", ["root", "b", "c", "d"])
    assert fit["used"] == ["root"]
    assert 0 < fit["resid"] < fit["resid_min"] < 1.02 * fit["resid"]
    assert max(abs(v) for v in fit["alpha"][1:]) < 0.05


def test_combo_fit_minimal_set_keeps_every_partner_of_a_real_mix():
    rng = np.random.default_rng(7)
    root = base_mats(0)
    a = perturb(root, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
    b = perturb(root, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
    # An even three-way mix: no proper subset of the partners can explain 95% of the fit.
    merged = {n: (0.4 * a[n][0] + 0.3 * b[n][0] + 0.3 * root[n][0], 100.0, a[n][2]) for n in NAMES}
    fit = signals.combo_fit(merged, {"a": a, "root": root, "b": b}, "a", ["root", "b"])
    assert sorted(fit["used"]) == ["b", "root"]
    assert fit["resid_min"] == fit["resid"]


def test_combo_fit_independent_delta_has_large_residual():
    rng = np.random.default_rng(6)
    root = base_mats(0)
    a = perturb(root, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
    cand = perturb(a, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
    fit = signals.combo_fit(cand, {"a": a, "root": root}, "a", ["root"])
    assert fit["resid"] > 0.9


def test_reuse_table_flags_moved_adapter():
    rng = np.random.default_rng(7)
    root = base_mats(0)
    delta = {n: lowrank(rng, 16, 0.1) for n in NAMES}
    lora_on_root = {n: (root[n][0] + delta[n], 100.0, root[n][2]) for n in NAMES}
    king = perturb(root, lambda n, s: s + lowrank(rng, 16, 0.1), lambda n, x: x)
    moved = {n: (king[n][0] + delta[n], 100.0, king[n][2]) for n in NAMES}
    bank = {"root": root, "lora": lora_on_root, "king": king}
    order = sorted(bank, key=lambda k: signals.rel_dist(moved, bank[k]))
    assert order[0] == "king"
    rows = signals.reuse_table(moved, order, bank, "root")
    assert rows[0][1] == "lora" and abs(rows[0][3]) > 0.99
