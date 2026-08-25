from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model_validation.dedup import canon, layout, sketch  # noqa: E402

SECRET = b"unit-test-secret-0123456789abcdef-0123456789"
CPU = torch.device("cpu")


def test_canon_and_ttype():
    assert (
        layout.canon("model.language_model.layers.3.self_attn.q_proj.weight")
        == "layers.3.self_attn.q_proj.weight"
    )
    assert layout.canon("lm_head.weight") == "lm_head.weight"
    assert layout.canon("model.visual.blocks.0.attn.qkv.weight") is None
    assert layout.ttype("layers.0.mlp.experts.gate_up_proj") == "experts"
    assert layout.ttype("layers.0.mlp.shared_expert.up_proj.weight") == "shared_expert"
    assert layout.ttype("embed_tokens.weight") == "embed_tokens"


def test_arch_key_depends_on_shapes_only():
    a = layout.arch_key({"x": (2, 3), "y": (4,)})
    assert a == layout.arch_key({"y": (4,), "x": (2, 3)})
    assert a != layout.arch_key({"x": (2, 4), "y": (4,)})


def test_match_recovers_row_permutation():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(32, 16, generator=g)
    perm = torch.randperm(32, generator=g)
    idx, cos, ident = canon.match(w[perm], w)
    assert torch.equal(w[perm][idx], w)
    assert cos > 0.999
    assert ident < 0.2
    idx, cos, ident = canon.match(w, w)
    assert ident == 1.0


def test_canonicalize_undoes_hidden_and_residual_permutation():
    g = torch.Generator().manual_seed(1)
    hidden, res = 24, 16
    up = torch.randn(hidden, res, generator=g)
    down = torch.randn(res, hidden, generator=g)
    p_h = torch.randperm(hidden, generator=g)
    p_r = torch.randperm(res, generator=g)
    up_p = up[p_h][:, p_r]
    down_p = down[p_r][:, p_h]
    inv_r = torch.argsort(p_r)
    w_up, info = canon.canonicalize(
        "layers.0.mlp.shared_expert.up_proj.weight", up_p.clone(), up, inv_r
    )
    assert torch.allclose(w_up, up)
    assert info["ident"] < 0.3
    w_down, info = canon.canonicalize(
        "layers.0.mlp.shared_expert.down_proj.weight", down_p.clone(), down, inv_r
    )
    assert torch.allclose(w_down, down)


def test_canonicalize_experts_3d():
    g = torch.Generator().manual_seed(2)
    e, a, b = 3, 12, 8
    w = torch.randn(e, a, b, generator=g)
    p = torch.randperm(a, generator=g)
    wp = w[:, p, :]
    out, info = canon.canonicalize(
        "layers.0.mlp.experts.gate_up_proj", wp.clone(), w, torch.arange(b)
    )
    assert torch.allclose(out, w)


def test_sketch_is_deterministic_secret_bound_and_rank_limited():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(32, 128, generator=g)
    s1, n1, x1, k1 = sketch.sketch(w, "t", SECRET)
    s2, n2, x2, k2 = sketch.sketch(w, "t", SECRET)
    s3, _, x3, _ = sketch.sketch(w, "t", b"another-secret-0123456789abcdef-0123456789")
    assert k1 == 32 and s1.shape == (32, 32) and x1.shape == (sketch.NSAMP,)
    assert np.array_equal(s1, s2) and np.array_equal(x1, x2)
    assert not np.array_equal(s1, s3) and not np.array_equal(x1, x3)
    assert abs(n1 - float(w.norm())) < 1e-4


def test_sketch_is_linear_and_scale_invariant_in_relative_terms():
    g = torch.Generator().manual_seed(4)
    w = torch.randn(64, 64, generator=g)
    s, _, _, _ = sketch.sketch(w, "t", SECRET)
    s2, _, _, _ = sketch.sketch(2 * w, "t", SECRET)
    assert np.allclose(s2, 2 * s, atol=1e-4)


def _write_model(tmp_path, tensors: dict[str, torch.Tensor], shards: int):
    from safetensors.torch import save_file

    names = sorted(tensors)
    per = max(1, -(-len(names) // shards))
    weight_map = {}
    for i in range(0, len(names), per):
        fname = f"model-{i // per + 1:05d}-of-{shards:05d}.safetensors"
        save_file({n: tensors[n].contiguous() for n in names[i : i + per]}, str(tmp_path / fname))
        for n in names[i : i + per]:
            weight_map[n] = fname
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_tensors_hash_is_content_only(tmp_path):
    g = torch.Generator().manual_seed(5)
    t = {
        f"model.language_model.layers.{i}.self_attn.q_proj.weight": torch.randn(
            8, 8, generator=g
        ).to(torch.bfloat16)
        for i in range(4)
    }
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    _write_model(a, t, 1)
    _write_model(b, t, 2)
    t2 = dict(t)
    t2["model.language_model.layers.0.self_attn.q_proj.weight"] = t2[
        "model.language_model.layers.0.self_attn.q_proj.weight"
    ].clone()
    t2["model.language_model.layers.0.self_attn.q_proj.weight"][0, 0] += 1
    _write_model(c, t2, 1)
    assert sketch.tensors_hash(str(a)) == sketch.tensors_hash(str(b))
    assert sketch.tensors_hash(str(a)) != sketch.tensors_hash(str(c))


def test_fingerprint_end_to_end_small_model(tmp_path):
    g = torch.Generator().manual_seed(6)
    res, hid, vocab = 16, 24, 40
    ref = {
        "model.language_model.embed_tokens.weight": torch.randn(vocab, res, generator=g),
        "lm_head.weight": torch.randn(vocab, res, generator=g),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(hid, res, generator=g),
        "model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(res, hid, generator=g),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(
            hid, res, generator=g
        ),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(
            res, hid, generator=g
        ),
        "model.language_model.layers.0.input_layernorm.weight": torch.ones(res),
    }
    ref_dir, cand_dir = tmp_path / "ref", tmp_path / "cand"
    ref_dir.mkdir()
    cand_dir.mkdir()
    _write_model(ref_dir, {k: v.to(torch.bfloat16) for k, v in ref.items()}, 1)
    p_h = torch.randperm(hid, generator=g)
    cand = dict(ref)
    cand["model.language_model.layers.0.mlp.shared_expert.up_proj.weight"] = ref[
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight"
    ][p_h]
    cand["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"] = ref[
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"
    ][:, p_h]
    _write_model(cand_dir, {k: v.to(torch.bfloat16) for k, v in cand.items()}, 2)

    d_ref = sketch.fingerprint(str(ref_dir), str(ref_dir), SECRET, CPU, model_uri="ref")
    d_cand = sketch.fingerprint(str(cand_dir), str(ref_dir), SECRET, CPU, model_uri="cand")
    assert d_ref["arch_key"] == d_cand["arch_key"]
    assert d_ref["key_id"] == d_cand["key_id"] and len(d_ref["key_id"]) == 8
    assert d_ref["tensors_hash"] != d_cand["tensors_hash"]
    assert d_ref["n_tensors"] == 6 and len(d_ref["sketch_vec"]) == sketch.VEC_DIM
    assert d_ref["identity_frac"]["shared_expert"] == 1.0
    assert d_cand["identity_frac"]["shared_expert"] < 0.3
    from model_validation.dedup.signals import mats, rel_dist

    assert rel_dist(mats(d_cand), mats(d_ref)) < 1e-5
