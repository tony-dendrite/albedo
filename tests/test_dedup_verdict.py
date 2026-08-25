from __future__ import annotations

import numpy as np

from model_validation.dedup.verdict import Thresholds, decide

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
BODY = [n for n in NAMES if "embed" not in n and "lm_head" not in n]
TH = Thresholds.from_settings()


def rng_(seed):
    return np.random.default_rng(seed)


def lowrank(rng, r, scale):
    return (rng.standard_normal((K, r)) @ rng.standard_normal((r, K))).astype(np.float32) * scale


def model(seed):
    rng = rng_(seed)
    return {
        n: (
            rng.standard_normal((K, K)).astype(np.float32) * 10,
            1000.0,
            rng.standard_normal(2048).astype(np.float32),
        )
        for n in NAMES
    }


def train(m, seed, scale=0.5, rank=16, names=BODY):
    rng = rng_(seed)
    out = {}
    for n, (s, wn, x) in m.items():
        if n in names:
            out[n] = (
                s + lowrank(rng, rank, scale),
                wn,
                x + rng.standard_normal(2048).astype(np.float32) * 0.01,
            )
        else:
            out[n] = (s, wn, x)
    return out


ROOT = model(0)
KING_A = train(ROOT, 1)
KING_B = train(ROOT, 2)
BANK = {"root": ROOT, "king_a": KING_A, "king_b": KING_B}


def verdict(cand, identity=None):
    return decide(cand, BANK, "root", identity, TH)


def test_exact_copy_is_rejected():
    v = verdict({n: (s.copy(), wn, x.copy()) for n, (s, wn, x) in KING_A.items()})
    assert v.rejected and v.reason == "COPY" and v.ancestor == "king_a"


def test_iid_noise_on_king_is_noise_copy():
    rng = rng_(10)
    cand = {
        n: (
            s + rng.standard_normal((K, K)).astype(np.float32) * 0.05,
            wn,
            x + rng.standard_normal(2048).astype(np.float32) * 1e-3,
        )
        for n, (s, wn, x) in KING_A.items()
    }
    v = verdict(cand)
    assert v.rejected and v.reason == "NOISE-COPY" and v.ancestor == "king_a"


def test_merge_of_banked_models_is_linear_combo():
    cand = {n: (0.65 * KING_A[n][0] + 0.35 * ROOT[n][0], KING_A[n][1], KING_A[n][2]) for n in NAMES}
    v = verdict(cand)
    assert v.rejected and v.reason == "LINEAR-COMBO"
    lin = v.metrics["linear"]
    i = lin["partners"].index("root")
    assert abs(lin["alpha"][i] - 0.35) < 0.02


def test_rescaled_delta_is_linear_combo():
    cand = {
        n: (ROOT[n][0] + 0.5 * (KING_A[n][0] - ROOT[n][0]), 1000.0, KING_A[n][2]) for n in NAMES
    }
    v = verdict(cand)
    assert v.rejected and v.reason == "LINEAR-COMBO"


def test_dense_multiplicative_noise_incl_embeddings_is_noised_copy():
    rng = rng_(11)
    cand = {
        n: (
            s * (1 + rng.standard_normal((K, K)).astype(np.float32) * 0.08),
            wn,
            x * (1 + rng.standard_normal(2048).astype(np.float32) * 0.08),
        )
        for n, (s, wn, x) in KING_A.items()
    }
    v = verdict(cand)
    assert v.rejected and v.reason in ("NOISED-COPY", "NOISE-COPY")


def test_sparse_edit_is_rejected():
    rng = rng_(12)
    cand = {}
    for n, (s, wn, x) in KING_A.items():
        d = np.zeros(2048, dtype=np.float32)
        d[rng.choice(2048, 30, replace=False)] = 0.5
        cand[n] = (s + lowrank(rng, 4, 0.3) if n in BODY else s, wn, x + d)
    v = verdict(cand)
    assert v.rejected and v.reason == "SPARSE-EDIT"


def test_trivial_structured_edit_is_rejected():
    cand = train(KING_A, 13, scale=0.002, rank=4)
    v = verdict(cand)
    assert v.rejected and v.reason == "TRIVIAL-EDIT"


def test_real_training_on_king_passes_as_trained():
    cand = train(KING_A, 14, scale=0.5, rank=48)
    v = verdict(cand)
    assert v.status == "PASS" and v.ancestor == "king_a"
    assert "TRAINED" in v.notes


def test_lora_like_training_on_root_passes():
    cand = train(ROOT, 15, scale=0.5, rank=8)
    v = verdict(cand)
    assert v.status == "PASS" and v.ancestor == "root"
    assert "LORA-like" in v.notes


def test_head_rescale_is_a_note_not_a_verdict():
    cand = train(KING_A, 16, scale=0.5, rank=48)
    s, wn, x = cand["lm_head.weight"]
    cand["lm_head.weight"] = (s * 1.756, wn, x * 1.756)
    v = verdict(cand)
    assert v.status == "PASS"
    assert any(n.startswith("HEAD-RESCALE x1.75") for n in v.notes)


def test_permuted_layout_note_and_metrics_present():
    cand = train(KING_A, 17, scale=0.5, rank=48)
    v = verdict(cand, identity={"shared_expert": 0.002, "residual": 1.0})
    assert any(n.startswith("PERMUTED LAYOUT") for n in v.notes)
    for k in (
        "F",
        "rel",
        "rel_struct",
        "embed_ratio",
        "density",
        "kurtosis",
        "distances",
        "linear",
    ):
        assert k in v.metrics
