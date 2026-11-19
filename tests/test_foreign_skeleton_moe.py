"""Skeleton + generic MoE block (kyojin_mlx/foreign_skeleton_moe.py): a tiny synthetic MoE GGUF built here, loaded by the skeleton from the file alone
(expert metadata + router / expert tensors + a skeleton row), against the float64 numpy decoder of test_foreign_skeleton with a numpy MoE FFN. CPU device only."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_gguf as GF, foreign_skeleton as SK, foreign_skeleton_moe as SM, foreign_synth as SY  # noqa: E402

_spec = importlib.util.spec_from_file_location("tsk", os.path.join(os.path.dirname(__file__), "test_foreign_skeleton.py"))
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)
E_, K_, FE, FS = 6, 2, 12, 20


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def moe_numpy(T, i, x, gating="softmax", norm=False, scale=1.0, shared=False, shared_gate=False, sel_bias=False):
    g = lambda n: T[f"blk.{i}.{n}"].astype(np.float64)      # noqa: E731
    lg = x @ g("ffn_gate_inp.weight").T
    pr = softmax(lg) if gating == "softmax" else 1 / (1 + np.exp(-lg))
    sel = pr + g("exp_probs_b.bias") if sel_bias else pr
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        idx = np.argsort(-sel[t], kind="stable")[:K_]
        w = pr[t, idx]
        if gating == "softmax_weight":
            w = softmax(lg[t, idx])
        if norm:
            w = w / max(w.sum(), 6.1e-5)
        w = w * scale
        for e, we in zip(idx, w):
            gt, up = x[t] @ g("ffn_gate_exps.weight")[e].T, x[t] @ g("ffn_up_exps.weight")[e].T
            out[t] += we * (((gt / (1 + np.exp(-gt))) * up) @ g("ffn_down_exps.weight")[e].T)
    if shared:
        gt, up = x @ g("ffn_gate_shexp.weight").T, x @ g("ffn_up_shexp.weight").T
        sh = ((gt / (1 + np.exp(-gt))) * up) @ g("ffn_down_shexp.weight").T
        if shared_gate:
            sh = sh * (1 / (1 + np.exp(-(x @ g("ffn_gate_inp_shexp.weight").reshape(-1)))))[:, None]
        out = out + sh
    return out


def build(tmp_path, seed=0, gating_kv=None, norm_kv=None, scale_kv=None, shared=False, shared_gate=False, sel_bias=False, dense_first=False, extra_kv=None):
    rng = np.random.default_rng(seed)
    path0, row, T, o, hl = TS.make(tmp_path, rng)
    kv = dict(GF.GgufFile(path0).kv)
    kv = {k: v for k, v in kv.items() if not k.startswith("tokenizer.")}
    for k in [k for k in T if ".ffn_" in k]:
        if not (dense_first and k.startswith("blk.0.")):
            del T[k]
    f32 = lambda *shape, s=0.4: (rng.standard_normal(shape) * s).astype(np.float32)      # noqa: E731
    for i in range(TS.NL):
        if dense_first and i == 0:
            continue
        p = f"blk.{i}."
        T[p + "ffn_norm.weight"] = T.get(p + "ffn_norm.weight", np.ones(TS.D, np.float32))
        T[p + "ffn_gate_inp.weight"] = f32(E_, TS.D, s=0.8)
        for n, shape in (("gate", (E_, FE, TS.D)), ("up", (E_, FE, TS.D)), ("down", (E_, TS.D, FE))):
            T[p + f"ffn_{n}_exps.weight"] = f32(*shape)
        if shared:
            for n, shape in (("gate", (FS, TS.D)), ("up", (FS, TS.D)), ("down", (TS.D, FS))):
                T[p + f"ffn_{n}_shexp.weight"] = f32(*shape)
        if shared_gate:
            T[p + "ffn_gate_inp_shexp.weight"] = f32(TS.D)
        if sel_bias:
            T[p + "exp_probs_b.bias"] = f32(E_, s=0.3)
    kv.update({"synth.expert_count": E_, "synth.expert_used_count": K_, "synth.expert_feed_forward_length": FE})
    if shared:
        kv["synth.expert_shared_count"] = 1
    if dense_first:
        kv["synth.leading_dense_block_count"] = 1
    if gating_kv is not None:
        kv["synth.expert_gating_func"] = gating_kv
    if norm_kv is not None:
        kv["synth.expert_weights_norm"] = norm_kv
    if scale_kv is not None:
        kv["synth.expert_weights_scale"] = scale_kv
    kv.update(extra_kv or {})
    path = str(tmp_path / "moe.gguf")
    SY.write_gguf(path, kv, [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a)) for n, a in T.items()])
    return path, row, T, o, hl


CASES = {
    "softmax_default": dict(),
    "softmax_norm_scale_from_file": dict(norm_kv=True, scale_kv=1.7, gating_kv=1),
    "sigmoid_select_bias_shared_gated": dict(gating_kv=2, norm_kv=True, shared=True, shared_gate=True, sel_bias=True),
    "shared_ungated_dense_first_layer": dict(shared=True, dense_first=True),
}


@pytest.mark.parametrize("name", list(CASES))
def test_moe_file_loads_through_the_hook_and_matches_numpy(tmp_path, name):
    c = CASES[name]
    path, row, T, o, hl = build(tmp_path, **c)
    gating = {1: "softmax", 2: "sigmoid"}.get(c.get("gating_kv"), "softmax")

    def hook(i, x):
        if c.get("dense_first") and i == 0:
            return None
        return moe_numpy(T, i, x, gating, bool(c.get("norm_kv")), c.get("scale_kv", 1.0), c.get("shared", False), c.get("shared_gate", False), c.get("sel_bias", False))
    o = {**o, "ffn_hook": hook}
    model, rep = SK.load_cpu(path, row=row)                         # no ffn_factory given: the file says MoE
    ids = np.array([3, 7, 1, 12, 5, 9, 0, 4])
    got = np.array(model(mx.array(ids[None]))[0].astype(mx.float32))
    want = TS.ref(T, o, hl, ids)
    assert np.abs(got - want).max() < 2e-4 * max(1.0, np.abs(want).max())
    moes = [l.mlp for l in model.layers if isinstance(l.mlp, SM.MoeFFN)]
    assert len(moes) == TS.NL - (1 if c.get("dense_first") else 0)
    s = moes[0].moe_spec
    assert (s.n_expert, s.top_k, s.gating) == (E_, K_, gating)
    assert s.shared == c.get("shared", False) and s.select_bias == c.get("sel_bias", False)


def test_dense_file_is_untouched(tmp_path):
    path, row, T, o, hl = TS.make(tmp_path, np.random.default_rng(1))
    model, _ = SK.load_cpu(path, row=row)
    assert all(type(l.mlp).__module__ == "kyojin_mlx.foreign_skeleton" for l in model.layers)
    assert SM.detect("synth", GF.GgufFile(path).kv, {}, row) is None


def test_row_supplies_what_the_file_does_not_say(tmp_path):
    path, row, T, o, hl = build(tmp_path)
    model, _ = SK.load_cpu(path, row={**row, "moe": {"norm_weights": True}})
    assert model.layers[0].mlp.moe_spec.norm_weights is True


def test_refusals(tmp_path):
    path, row, *_ = build(tmp_path, extra_kv={"synth.expert_chunk_count": 2})
    with pytest.raises(SK.SkeletonError, match="expert_chunk_count"):
        SK.load_cpu(path, row=row)
    path, row, *_ = build(tmp_path, extra_kv={"synth.expert_count": 0})
    with pytest.raises(SK.SkeletonError, match="disagree"):
        SK.load_cpu(path, row=row)
