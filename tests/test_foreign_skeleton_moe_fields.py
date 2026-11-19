"""Generic MoE experts with the swiglu_oai activation, loaded through the skeleton (plain ops; no fused kernel), against a float64 numpy MoE written from
llama.cpp build_moe_ffn (llama-graph.cpp:2552-2560, ggml-cpu/ops.cpp:3388-3452). CPU device only."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_moe_generic as G, foreign_skeleton as SK, foreign_skeleton_moe as SM  # noqa: E402

_spec = importlib.util.spec_from_file_location("tsm", os.path.join(os.path.dirname(__file__), "test_foreign_skeleton_moe.py"))
TM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TM)
TS, E_, K_ = TM.TS, TM.E_, TM.K_


def oai(g, u, alpha=1.702, limit=7.0):
    g = np.minimum(g, limit)
    return g / (1 + np.exp(-alpha * g)) * (np.clip(u, -limit, limit) + 1.0)


def moe_oai(T, i, x, shared):
    g = lambda n: T[f"blk.{i}.{n}"].astype(np.float64)      # noqa: E731
    pr = TM.softmax(x @ g("ffn_gate_inp.weight").T)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        for e in np.argsort(-pr[t], kind="stable")[:K_]:
            h = oai(x[t] @ g("ffn_gate_exps.weight")[e].T, x[t] @ g("ffn_up_exps.weight")[e].T)
            out[t] += pr[t, e] * (h @ g("ffn_down_exps.weight")[e].T)
    if shared:
        out = out + oai(x @ g("ffn_gate_shexp.weight").T, x @ g("ffn_up_shexp.weight").T) @ g("ffn_down_shexp.weight").T
    return out


@pytest.mark.parametrize("shared", [False, True])
def test_swiglu_oai_experts_match_numpy(tmp_path, shared):
    path, row, T, o, hl = TM.build(tmp_path, shared=shared)
    row = {**row, "act": "swiglu_oai"}
    model, _ = SK.load_cpu(path, row=row)
    ids = np.array([3, 7, 1, 12, 5, 9, 0, 4])
    got = np.array(model(mx.array(ids[None]))[0].astype(mx.float32))
    # the dense part of TS.ref is silu: replace every FFN by the MoE (all layers are MoE here)
    o = {**o, "ffn_hook": lambda i, x: moe_oai(T, i, x, shared)}
    want = TS.ref(T, o, hl, ids)
    diff = np.abs(got - want).max()
    print("swiglu_oai experts shared=%s max abs diff vs numpy %.3g" % (shared, diff))
    assert diff < 2e-4 * max(1.0, np.abs(want).max())
    s = model.layers[0].mlp.moe_spec
    assert s.act == "swiglu_oai" and s.shared_act == ("swiglu_oai")


def test_describe_says_no_fused_kernel(tmp_path):
    path, row, *_ = TM.build(tmp_path)
    model, _ = SK.load_cpu(path, row={**row, "act": "swiglu_oai"})
    d = G.describe(model.layers[0].mlp.moe_spec)
    text = repr(d)
    print(d)
    assert d["kind"] == "stock" and "swiglu_oai" in text and "no kernel kind" in text


# ---- group-limited routing, latent projection, per-layer arrays, per-layer clamp, MTP blocks ----
from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY  # noqa: E402

D, NL, FE = TS.D, TS.NL, 12
IDS = np.array([3, 7, 1, 12, 5, 9, 0, 4])


def build2(tmp_path, E=(8, 8), K=(2, 2), groups=None, latent=0, clamp=None, nextn=0, gating=2, norm=True, scale=1.5, seed=3, k_scalar=False, shexp_clamp=None):
    rng = np.random.default_rng(seed)
    path0, row, T, o, hl = TS.make(tmp_path, rng)
    kv = {k: v for k, v in GF.GgufFile(path0).kv.items() if not k.startswith("tokenizer.")}
    for k in [k for k in T if ".ffn_" in k]:
        del T[k]
    f32 = lambda *shape, s=0.4: (rng.standard_normal(shape) * s).astype(np.float32)      # noqa: E731
    win = latent or D
    for i in range(NL):
        p = f"blk.{i}."
        T[p + "ffn_gate_inp.weight"] = f32(E[i], D, s=0.8)
        T[p + "exp_probs_b.bias"] = f32(E[i], s=0.3)
        for n, shape in (("gate", (E[i], FE, win)), ("up", (E[i], FE, win)), ("down", (E[i], win, FE))):
            T[p + f"ffn_{n}_exps.weight"] = f32(*shape)
        if latent:
            T[p + "ffn_latent_down.weight"] = f32(latent, D)
            T[p + "ffn_latent_up.weight"] = f32(D, latent)
    kv.update({"synth.expert_count": max(E), "synth.expert_used_count": K[0] if k_scalar else list(K), "synth.expert_feed_forward_length": FE,
               "synth.expert_gating_func": gating, "synth.expert_weights_norm": norm, "synth.expert_weights_scale": scale})
    if groups:
        kv["synth.expert_group_count"], kv["synth.expert_group_used_count"] = groups
    if latent:
        kv["synth.moe_latent_size"] = latent
    if clamp:
        kv["synth.swiglu_clamp_exp"] = [float(c) for c in clamp]
    if shexp_clamp:
        kv["synth.swiglu_clamp_shexp"] = [float(c) for c in shexp_clamp]
    if nextn:                                                      # one MTP block after the decoded layers, with tensors that are not skeleton tensors
        kv["synth.block_count"] = NL + nextn
        T[f"blk.{NL}.attn_norm.weight"] = np.ones(D, np.float32)
        T[f"blk.{NL}.nextn.eh_proj.weight"] = f32(D, 2 * D)
        kv["synth.nextn_predict_layers"] = nextn
    path = str(tmp_path / "moe2.gguf")
    SY.write_gguf(path, kv, [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a)) for n, a in T.items()])
    return path, row, T, o, hl


def moe_ref(T, i, x, K, G, used, latent, clamp, gating=2, norm=True, scale=1.5):
    g = lambda n: T[f"blk.{i}.{n}"].astype(np.float64)      # noqa: E731
    probs = 1 / (1 + np.exp(-(x @ g("ffn_gate_inp.weight").T))) if gating == 2 else TM.softmax(x @ g("ffn_gate_inp.weight").T)
    sel = probs + g("exp_probs_b.bias")
    E = probs.shape[1]
    xin = x @ g("ffn_latent_down.weight").T if latent else x
    out = np.zeros_like(xin)
    for t in range(x.shape[0]):
        s_t = sel[t].copy()
        if G > 1:                                                  # llama-graph.cpp:2346-2366: group score = sum of the 2 best, keep `used` groups
            grp = s_t.reshape(G, E // G)
            gs = np.sort(grp, axis=-1)[:, -2:].sum(-1)
            keep = np.argsort(-gs, kind="stable")[:used]
            m = np.full((G, E // G), -np.inf)
            m[keep] = grp[keep]
            s_t = m.reshape(-1)
        idx = np.argsort(-s_t, kind="stable")[:K]
        w = probs[t, idx]
        if norm:
            w = w / max(w.sum(), 6.1e-5)
        w = w * scale
        for e, we in zip(idx, w):
            gt, up = xin[t] @ g("ffn_gate_exps.weight")[e].T, xin[t] @ g("ffn_up_exps.weight")[e].T
            a = gt / (1 + np.exp(-gt))
            if clamp > 1e-6:                                       # llama-graph.cpp:2504-2515
                a, up = np.minimum(a, clamp), np.clip(up, -clamp, clamp)
            out[t] += we * ((a * up) @ g("ffn_down_exps.weight")[e].T)
    return out @ g("ffn_latent_up.weight").T if latent else out


def run2(tmp_path, K, groups=None, latent=0, clamp=None, **kw):
    path, row, T, o, hl = build2(tmp_path, K=K, groups=groups, latent=latent, clamp=clamp, **kw)
    G, used = groups or (1, 1)
    o = {**o, "ffn_hook": lambda i, x: moe_ref(T, i, x, K[i], G, used, latent, (clamp or [0] * NL)[i])}
    model, _ = SK.load_cpu(path, row=row)
    got = np.array(model(mx.array(IDS[None]))[0].astype(mx.float32))
    want = TS.ref(T, o, hl, IDS)
    return model, np.abs(got - want).max(), want


def test_group_limited_routing(tmp_path):
    model, diff, want = run2(tmp_path, K=(2, 2), groups=(4, 2))
    print("groups 4/2 max abs diff vs numpy", diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max())
    s = model.layers[0].mlp.moe_spec
    assert (s.n_group, s.group_used) == (4, 2)
    _, d2, _ = run2(tmp_path, K=(2, 2), groups=(4, 1))        # dropping more groups changes the routing: the reference follows
    assert d2 < 2e-4 * max(1.0, np.abs(want).max())


def test_latent_projection_from_the_tensors(tmp_path):
    model, diff, want = run2(tmp_path, K=(2, 2), latent=6)
    print("latent 6 max abs diff vs numpy", diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max())
    assert model.layers[0].mlp.moe_spec.latent is True


def test_per_layer_top_k_and_expert_count_arrays(tmp_path):
    model, diff, want = run2(tmp_path, K=(1, 3), E=(8, 6))
    print("per-layer top-k [1,3], experts [8,6] max abs diff vs numpy", diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max())
    assert [l.mlp.moe_spec.top_k for l in model.layers] == [1, 3]
    assert [l.mlp.moe_spec.n_expert for l in model.layers] == [8, 6]


def test_scalar_top_k_still_works(tmp_path):
    _, diff, want = run2(tmp_path, K=(2, 2), k_scalar=True)
    assert diff < 2e-4 * max(1.0, np.abs(want).max())


def test_per_layer_clamp_array(tmp_path):
    clamp = (0.0, 0.05)
    model, diff, want = run2(tmp_path, K=(2, 2), clamp=clamp)
    print("per-layer clamp max abs diff vs numpy", diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max())
    assert [l.mlp.moe_spec.clamp for l in model.layers] == [pytest.approx(c, rel=1e-6) for c in clamp]


def test_mtp_block_after_the_decoded_layers_is_skipped(tmp_path):
    model, diff, want = run2(tmp_path, K=(2, 2), nextn=1)
    assert len(model.layers) == NL and diff < 2e-4 * max(1.0, np.abs(want).max())


def test_refusals_of_what_is_not_placed(tmp_path):
    path, row, *_ = build2(tmp_path, groups=(3, 1))
    with pytest.raises(SK.SkeletonError, match="do not fit"):
        SK.load_cpu(path, row=row)
    path, row, *_ = build2(tmp_path, K=(2, 9))
    with pytest.raises(SK.SkeletonError, match="top_k 9"):
        SK.load_cpu(path, row=row)
    path, row, *_ = build2(tmp_path, latent=6)
    from kyojin_mlx import foreign_gguf as GF2
    g = GF2.GgufFile(path)
    kv = dict(g.kv)
    kv["synth.moe_latent_size"] = 5
    with pytest.raises(SK.SkeletonError, match="moe_latent_size"):
        SM.detect("synth", kv, {k: ti.ne for k, ti in g.tensors.items()}, row)
    g.close()
