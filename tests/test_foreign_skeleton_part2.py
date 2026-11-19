"""Skeleton spec fields, part 2 (norms around the expert block, shared-expert clamp, value dim != key dim + value scale + attention sinks, block-sparse attention
indexer, expert-count checks, MoE cache with window + gate) on tiny synthetic GGUF files against float64 numpy written from the cited llama.cpp graph code
(llama.cpp master checkout; file:line in each test). CPU device only; no architecture name in the loader: the row is passed in."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_gguf as GF, foreign_skeleton as SK, foreign_skeleton_moe as SM, foreign_synth as SY  # noqa: E402

_spec = importlib.util.spec_from_file_location("tf", os.path.join(os.path.dirname(__file__), "test_foreign_skeleton_fields.py"))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
D, NL, IDS = F.D, F.NL, F.IDS
E_, K_, FE, FS = 6, 2, 10, 12
TOL = 2e-4


def silu(x):
    return x / (1 + np.exp(-x))


def moeify(tmp_path, path, T, rng, latent=0, pre=False, post=False, shared=False, sh_clamp=None, post_eps=1e-5, exp_counts=None, extra_kv=None, down_scale=()):
    """Replace the dense FFN tensors of a F.make file by a sigmoid-routed MoE (+ norms, latent, shared expert) and write a new file."""
    kv = {k: v for k, v in GF.GgufFile(path).kv.items() if not k.startswith("tokenizer.")}
    for k in [k for k in T if ".ffn_gate." in k or ".ffn_up." in k or ".ffn_down." in k]:
        del T[k]
    f32 = lambda *shape, s=0.4: (rng.standard_normal(shape) * s).astype(np.float32)      # noqa: E731
    win = latent or D
    for i in range(NL):
        p = f"blk.{i}."
        n_e = (exp_counts or [E_] * NL)[i]
        T[p + "ffn_gate_inp.weight"] = f32(n_e, D, s=0.8)
        T[p + "exp_probs_b.bias"] = f32(n_e, s=0.3)
        for n, shape in (("gate", (n_e, FE, win)), ("up", (n_e, FE, win)), ("down", (n_e, win, FE))):
            T[p + f"ffn_{n}_exps.weight"] = f32(*shape)
        if i in down_scale:
            T[p + "ffn_down_exps.scale"] = (1 + 0.5 * rng.standard_normal(n_e)).astype(np.float32)
        if latent:
            T[p + "ffn_latent_down.weight"], T[p + "ffn_latent_up.weight"] = f32(latent, D), f32(D, latent)
        if pre:
            T[p + "ffn_pre_exp_norm.weight"] = (1 + 0.3 * rng.standard_normal(win)).astype(np.float32)
        if post:
            T[p + "ffn_post_exp_norm.weight"] = (1 + 0.3 * rng.standard_normal(win)).astype(np.float32)
        if shared:
            for n, shape in (("gate", (FS, D)), ("up", (FS, D)), ("down", (D, FS))):
                T[p + f"ffn_{n}_shexp.weight"] = f32(*shape, s=1.0)
    kv.update({"synth.expert_count": max(exp_counts or [E_]), "synth.expert_used_count": K_, "synth.expert_feed_forward_length": FE, "synth.expert_gating_func": 2,
               "synth.expert_weights_norm": True, "synth.expert_weights_scale": 1.5})
    if latent:
        kv["synth.moe_latent_size"] = latent
    if shared:
        kv["synth.expert_shared_count"] = 1
    if sh_clamp:
        kv["synth.swiglu_clamp_shexp"] = [float(c) for c in sh_clamp]
    kv.update(extra_kv or {})
    out = str(tmp_path / "moe.gguf")
    SY.write_gguf(out, kv, [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a)) for n, a in T.items()])
    return out


def moe_np(T, i, x, latent=0, pre=False, post=False, pre_eps=F.EPS, post_eps=1e-5, shared=False, clamp=0.0, down_scale=False):
    """llama.cpp build_moe_ffn (llama-graph.cpp:2284-2667) with the latent wrapper (muse-spark.cpp:106-143), the post-expert norm (llama-graph.cpp:2622-2625), the shared
    expert clamp (llama-graph.cpp:2066-2080: min(silu(gate), c) * clip(up, -c, c))."""
    g = lambda n: T[f"blk.{i}.{n}"].astype(np.float64)      # noqa: E731
    probs = 1 / (1 + np.exp(-(x @ g("ffn_gate_inp.weight").T)))
    sel = probs + g("exp_probs_b.bias")
    xe = x @ g("ffn_latent_down.weight").T if latent else x
    if pre:
        xe = F.rms(xe, g("ffn_pre_exp_norm.weight"), pre_eps)
    out = np.zeros_like(xe)
    for t in range(x.shape[0]):
        idx = np.argsort(-sel[t], kind="stable")[:K_]
        w = probs[t, idx]
        w = w / max(w.sum(), 6.1e-5) * 1.5
        for e, we in zip(idx, w):
            h = silu(xe[t] @ g("ffn_gate_exps.weight")[e].T) * (xe[t] @ g("ffn_up_exps.weight")[e].T)
            y = h @ g("ffn_down_exps.weight")[e].T
            if down_scale:
                y = y * g("ffn_down_exps.scale")[e]
            if post:
                y = F.rms(y * g("ffn_post_exp_norm.weight"), None, post_eps)
            out[t] += we * y
    if latent:
        out = out @ g("ffn_latent_up.weight").T
    if shared:
        a, up = silu(x @ g("ffn_gate_shexp.weight").T), x @ g("ffn_up_shexp.weight").T
        if clamp > 1e-6:
            a, up = np.minimum(a, clamp), np.clip(up, -clamp, clamp)
        out = out + (a * up) @ g("ffn_down_shexp.weight").T
    return out


def run(tmp_path, seed=0, fkw=None, mkw=None, hook_kw=None, clamps=None):
    ds = (mkw or {}).get("down_scale", ())
    rng = np.random.default_rng(seed)
    path0, row, T, o = F.make(tmp_path, rng, **(fkw or {}))
    mkw = mkw or {}
    path = moeify(tmp_path, path0, T, rng, **mkw)
    hk = dict(latent=mkw.get("latent", 0), pre=mkw.get("pre", False), post=mkw.get("post", False), shared=mkw.get("shared", False), post_eps=mkw.get("post_eps", 1e-5))
    o = {**o, "ffn_hook": lambda i, x: moe_np(T, i, x, clamp=(clamps or [0.0] * NL)[i], down_scale=i in ds, **hk)}
    model, _ = SK.load_cpu(path, row=row)
    got = np.array(model(mx.array(IDS[None]))[0]).astype(np.float64)
    want = F.ref(T, o, IDS)
    return model, got, want, float(np.abs(got - want).max())


# ---- item 1: norms before / after the expert block ----
@pytest.mark.parametrize("pre,post", [(True, False), (False, True), (True, True)])
def test_pre_post_expert_norms(tmp_path, pre, post):
    model, got, want, diff = run(tmp_path, mkw=dict(latent=8, pre=pre, post=post, post_eps=1e-8, extra_kv={"synth.post_norm_epsilon": 1e-8}))
    print("pre=%s post=%s max abs diff vs numpy %.3g" % (pre, post, diff))
    assert diff < TOL * max(1.0, np.abs(want).max())
    s = model.layers[0].mlp.moe_spec
    assert (s.pre_exp_norm, s.post_exp_norm) == (pre, post)
    assert F.cache_consistent(model, got) < 1e-4


def test_expert_norms_change_the_output_and_describe_says_stock(tmp_path):
    _, a, _, _ = run(tmp_path, mkw=dict(latent=8, pre=True, post=True))
    _, b, _, _ = run(tmp_path, mkw=dict(latent=8))
    assert np.abs(a - b).max() > 1e-3
    model, *_ = run(tmp_path, mkw=dict(latent=8, pre=True))
    from kyojin_mlx import foreign_moe_generic as G
    assert G.describe(model.layers[0].mlp.moe_spec)["kind"] == "stock"


# ---- item 2: shared expert clamp per layer ----
def test_shared_expert_clamp_per_layer(tmp_path):
    clamps = [0.0, 0.05, 0.0, 0.1]
    model, got, want, diff = run(tmp_path, mkw=dict(shared=True, sh_clamp=clamps), clamps=clamps)
    print("shared clamp %s max abs diff vs numpy %.3g" % (clamps, diff))
    assert diff < TOL * max(1.0, np.abs(want).max())
    assert [l.mlp.moe_spec.shared_clamp for l in model.layers] == [pytest.approx(c, rel=1e-6) for c in clamps]
    _, g0, _, _ = run(tmp_path, mkw=dict(shared=True))
    assert np.abs(got - g0).max() > 1e-3                           # the clamp is active
    assert F.cache_consistent(model, got) < 1e-4


# ---- a per-expert down scale on some layers only (the M3.1 file has it on 3 layers) ----
def test_down_expert_scale_on_some_layers(tmp_path):
    model, got, want, diff = run(tmp_path, mkw=dict(down_scale=(1, 3), post=True, latent=8))
    print("down scale on layers 1, 3 max abs diff vs numpy %.3g" % diff)
    assert diff < TOL * max(1.0, np.abs(want).max())
    assert [l.mlp.moe_spec.down_scale for l in model.layers] == [False, True, False, True]
    _, g0, _, _ = run(tmp_path, mkw=dict(post=True, latent=8))
    assert np.abs(got - g0).max() > 1e-3
    assert F.cache_consistent(model, got) < 1e-4
