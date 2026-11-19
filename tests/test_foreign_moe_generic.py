# SPDX-License-Identifier: Apache-2.0
"""foreign_moe_generic: every spec field against an independent float64 numpy re-implementation of llama.cpp's build_moe_ffn (llama-graph.cpp:2252-2667)
on synthetic tensors. CPU only, no Metal (the Route / kind checks build objects, they launch no kernel)."""
import itertools

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import foreign_moe_generic as G

mx.set_default_device(mx.cpu)
H, FF, E, K, T = 16, 12, 8, 3, 5


def sig(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def act_ref(kind, g, u, clamp=0.0, mode="post_act", beta=2.0, lb=3.0):
    gelu = lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    if g is None:
        return {"silu": lambda x: x * sig(x), "gelu": gelu, "relu": lambda x: np.maximum(x, 0), "relu2": lambda x: np.maximum(x, 0) ** 2}[kind](u)
    if kind == "silu":
        if clamp > 1e-6:
            if mode == "pre_act":
                gg = np.minimum(g, clamp)
                return gg * sig(gg) * np.clip(u, -clamp, clamp)
            return np.minimum(g * sig(g), clamp) * np.clip(u, -clamp, clamp)
        return g * sig(g) * u
    if kind == "gelu":
        return gelu(g) * u
    if kind == "relu":
        return np.maximum(g, 0) * u
    if kind == "situ":
        return beta * np.tanh(g / beta) * sig(g) * (lb * np.tanh(u / lb))
    if kind == "swiglu_oai":
        gg = np.minimum(g, 7.0)
        return gg * sig(1.702 * gg) * (np.clip(u, -7, 7) + 1)


def ffn_ref(x, g, u, d, kind, **kw):
    return act_ref(kind, None if g is None else x @ g.T, x @ u.T, **kw) @ d.T


def moe_ref(x, s, t):
    """x [T, H] float64. Row by row, like the graph, no vectorised tricks."""
    out = []
    for xr in x:
        lg = t["gate_inp"] @ xr
        if s.get("router_bias"):
            lg = lg + t["gate_inp_b"]
        lg = lg / s.get("temperature", 1.0)
        g = s.get("gating", "softmax")
        probs = {"softmax": softmax, "sigmoid": sig, "softmax_weight": lambda v: v, "none": lambda v: v, "sqrt_softplus": lambda v: np.sqrt(np.log1p(np.exp(v)))}[g](lg)
        sel = probs + (t["exp_probs_b"] if s.get("select_bias") else 0)
        ng = s.get("n_group", 1)
        if ng > 1:
            grp = sel.reshape(ng, -1)
            sc = np.sort(grp, axis=-1)[:, -2:].sum(-1)
            keep = np.argsort(-sc, kind="stable")[:s["group_used"]]
            m = np.full_like(grp, -np.inf); m[keep] = grp[keep]; sel = m.reshape(-1)
        idx = np.argsort(-sel, kind="stable")[:s["top_k"]]
        w = probs[idx]
        if g == "softmax_weight":
            w = softmax(w)
        if s.get("norm_weights"):
            tot = w.sum()
            w = w / (tot + s["norm_eps"] if s.get("norm_eps", 0) > 0 else max(tot, 6.103515625e-5))
        if s.get("weight_scale", 1.0) not in (0.0, 1.0):
            w = w * s["weight_scale"]
        xe = t["routed_down"] @ xr if s.get("latent") else xr
        kw = dict(clamp=s.get("clamp", 0.0), mode=s.get("clamp_mode", "post_act"), beta=s.get("situ_beta", 2.0), lb=s.get("situ_linear_beta", 3.0))
        acc = 0
        for wi, e in zip(w, idx):
            acc = acc + wi * ffn_ref(xe, t["gate_exps"][e] if s.get("gated", True) else None, t["up_exps"][e], t["down_exps"][e], s.get("act", "silu"), **kw)
        if s.get("latent"):
            if s.get("latent_norm"):
                acc = acc / np.sqrt((acc ** 2).mean() + 1e-6) * t["routed_norm"]
            acc = t["routed_up"] @ acc
        if s.get("shared"):
            sh = ffn_ref(xr, t["gate_shexp"], t["up_shexp"], t["down_shexp"], s.get("shared_act", "silu"))
            if s.get("shared_gate"):
                sh = sh * sig(t["gate_inp_shexp"].reshape(-1) @ xr)
            acc = acc + sh
        out.append(acc)
    return np.stack(out)


def tensors(seed=0, latent=0, shared=True, router_bias=False, ff=FF):
    r = np.random.RandomState(seed)
    n = lambda *sh: (r.randn(*sh) * 0.5).astype(np.float32)
    W = latent or H
    t = dict(gate_inp=n(E, H), gate_inp_b=n(E) * 0.3, exp_probs_b=n(E) * 0.3, gate_exps=n(E, ff, W), up_exps=n(E, ff, W), down_exps=n(E, W, ff),
             gate_shexp=n(2 * ff, H), up_shexp=n(2 * ff, H), down_shexp=n(H, 2 * ff), gate_inp_shexp=n(1, H),
             ffn_gate=n(20, H), ffn_up=n(20, H), ffn_down=n(H, 20))
    if latent:
        t.update(routed_down=n(latent, H), routed_up=n(H, latent), routed_norm=(1 + n(latent) * 0.1))
    return t


def run(spec, t, x=None, rtol=2e-4):
    x = x if x is not None else (np.random.RandomState(3).randn(T, H) * 0.7).astype(np.float32)
    blk = G.ffn_factory(1, dict(n_expert=E, top_k=K, **spec), {k: mx.array(v) for k, v in t.items()})
    got = np.array(blk(mx.array(x)[None])[0]).astype(np.float64)
    ref = moe_ref(x.astype(np.float64), dict(top_k=K, **spec), {k: v.astype(np.float64) for k, v in t.items()})
    err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
    assert err < rtol, (spec, err)
    return blk


@pytest.mark.parametrize("gating,norm", list(itertools.product(("softmax", "sigmoid", "softmax_weight", "none", "sqrt_softplus"), (False, True))))
def test_gating_and_norm(gating, norm):
    run(dict(gating=gating, norm_weights=norm), tensors())


def test_norm_eps_scale_temperature_biases():
    t = tensors()
    run(dict(gating="sigmoid", norm_weights=True, norm_eps=1e-3, weight_scale=2.5), t)
    run(dict(gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=1.8), t)
    run(dict(gating="softmax", router_bias=True), t)
    run(dict(gating="softmax", temperature=0.5, norm_weights=True), t)


def test_select_bias_only_changes_the_choice():
    t = tensors()
    t["exp_probs_b"] = np.zeros(E, np.float32); t["exp_probs_b"][0] = 10.0      # expert 0 always chosen, its weight is still the plain score
    blk = run(dict(gating="sigmoid", select_bias=True, norm_weights=False), t)
    inds, w = blk.route(mx.random.normal((4, H)))
    assert (np.array(inds) == 0).any(axis=-1).all() and float(np.array(w).max()) <= 1.0


@pytest.mark.parametrize("ng,gu", [(2, 1), (4, 2), (2, 2)])
def test_groups(ng, gu):
    run(dict(gating="sigmoid", select_bias=True, norm_weights=True, n_group=ng, group_used=gu), tensors())


@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "situ", "swiglu_oai"])
def test_gated_acts(act):
    run(dict(act=act, situ_beta=2.0, situ_linear_beta=3.0), tensors())


@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "relu2"])
def test_non_gated_experts(act):
    t = tensors(); del t["gate_exps"]
    run(dict(act=act, gated=False), t)


@pytest.mark.parametrize("mode", ["post_act", "pre_act"])
def test_clamp(mode):
    t = tensors()
    t["gate_exps"] = t["gate_exps"] * 4; t["up_exps"] = t["up_exps"] * 4
    run(dict(clamp=0.7, clamp_mode=mode), t)
    # the two modes really differ
    x = (np.random.RandomState(1).randn(T, H) * 2).astype(np.float32)
    a = np.array(G.ffn_factory(0, dict(n_expert=E, top_k=K, clamp=0.7, clamp_mode="post_act"), {k: mx.array(v) for k, v in t.items()})(mx.array(x)))
    b = np.array(G.ffn_factory(0, dict(n_expert=E, top_k=K, clamp=0.7, clamp_mode="pre_act"), {k: mx.array(v) for k, v in t.items()})(mx.array(x)))
    assert np.abs(a - b).max() > 1e-4


@pytest.mark.parametrize("gate", [False, True])
def test_shared_expert(gate):
    run(dict(gating="softmax", shared=True, shared_gate=gate), tensors())
    run(dict(gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=2.0, shared=True, shared_gate=gate), tensors())


def test_parallel_dense_and_second_input():
    t = tensors()
    s = dict(n_expert=E, top_k=K, gating="softmax", norm_weights=True, parallel_dense=True)
    blk = G.ffn_factory(0, s, {k: mx.array(v) for k, v in t.items()})
    x = np.random.RandomState(5).randn(1, T, H).astype(np.float32) * 0.6
    xm = np.random.RandomState(6).randn(1, T, H).astype(np.float32) * 0.6
    got = np.array(blk(mx.array(x), mx.array(xm)))[0].astype(np.float64)
    t64 = {k: v.astype(np.float64) for k, v in t.items()}
    ref = moe_ref(xm[0].astype(np.float64), dict(s), t64) + ffn_ref(x[0].astype(np.float64), t64["ffn_gate"], t64["ffn_up"], t64["ffn_down"], "silu")
    assert np.abs(got - ref).max() / np.abs(ref).max() < 2e-4


@pytest.mark.parametrize("norm", [False, True])
def test_latent(norm):
    t = tensors(latent=6)
    run(dict(gating="sigmoid", select_bias=True, norm_weights=True, latent=True, latent_norm=norm, shared=True, act="situ", situ_beta=2.0, situ_linear_beta=3.0), t)


def test_interleaved_dense_layers():
    t = tensors()
    s = dict(n_expert=E, top_k=K, n_dense_leading=1, moe_every=2)
    kinds = [type(G.ffn_factory(i, s, {k: mx.array(v) for k, v in t.items()})).__name__ for i in range(5)]
    assert kinds == ["DenseFFN", "_ForeignMoeKind", "DenseFFN", "_ForeignMoeKind", "DenseFFN"]
    assert G.MoeSpec(dense_layers=(0, 3)).is_moe_layer(1) and not G.MoeSpec(dense_layers=(0, 3)).is_moe_layer(3)
    x = np.random.randn(1, 3, H).astype(np.float32)
    d = G.ffn_factory(0, s, {k: mx.array(v) for k, v in t.items()})
    ref = ffn_ref(x[0].astype(np.float64), t["ffn_gate"].astype(np.float64), t["ffn_up"].astype(np.float64), t["ffn_down"].astype(np.float64), "silu")
    assert np.abs(np.array(d(mx.array(x)))[0] - ref).max() < 1e-4 * np.abs(ref).max()


def test_names_and_aliases():
    t = tensors()
    g = {"blk.4.ffn_gate_inp.weight": t["gate_inp"], "blk.4.exp_probs_b.bias": t["exp_probs_b"], "blk.4.ffn_gate_exps.weight": t["gate_exps"], "blk.4.ffn_up_exps.weight": t["up_exps"],
         "blk.4.ffn_down_exps.weight": t["down_exps"], "blk.4.ffn_gate_shexp.weight": t["gate_shexp"], "blk.4.ffn_up_shexp.weight": t["up_shexp"], "blk.4.ffn_down_shexp.weight": t["down_shexp"]}
    g = {k: mx.array(v) for k, v in g.items()}
    blk = G.ffn_factory(4, dict(n_expert=E, top_k=K, gating="sigmoid", select_bias=True, norm_weights=True, shared=True), g)
    assert blk.sel_b is not None and blk.shared_ffn is not None
    with pytest.raises(KeyError):
        G.ffn_factory(4, dict(n_expert=E, top_k=K, gating="sigmoid", select_bias=True), {"gate_inp": g["blk.4.ffn_gate_inp.weight"]})


def test_describe_kinds_and_reasons():
    d = lambda **k: G.describe(G.MoeSpec(n_expert=E, top_k=K, **k))
    assert d(gating="softmax", norm_weights=True)["kind"] == "foreign_moe"
    assert d(gating="softmax_weight")["kind"] == "foreign_moe"
    assert d(gating="softmax", shared=True)["kind"] == "foreign_moe"
    assert d(gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=2.0, shared=True, latent=True, act="situ")["kind"] == "glm_route"
    for kw, frag in [(dict(router_bias=True), "bias"), (dict(gating="none"), "select kernel"), (dict(gating="sqrt_softplus"), "select kernel"), (dict(temperature=0.5), "temperature"),
                     (dict(gated=False, act="relu2"), "non-gated"), (dict(act="gelu"), "kernel kind"), (dict(clamp=1.0), "pre-activation"), (dict(n_group=4, group_used=2), "group"),
                     (dict(gating="softmax", select_bias=True), "selection bias"), (dict(gating="softmax", weight_scale=2.0), "scale"), (dict(gating="softmax", latent=True), "latent")]:
        r = d(**kw)
        assert r["kind"] == "stock" and any(frag in w for w in r["why"]), (kw, r)
    assert d(clamp=1.0, clamp_mode="pre_act")["kind"] == "foreign_moe"


def test_attributes_of_each_kind():
    t = {k: mx.array(v) for k, v in tensors().items()}
    f = G.ffn_factory(0, dict(n_expert=E, top_k=K, gating="softmax_weight", shared=True), t)           # ungated shared: gate 0 and x2
    assert f.gate.weight.dtype == mx.float32 and f.gate.weight.shape == (E, H) and f.top_k == K and f.num_experts == E and f.norm_topk_prob is True
    assert not hasattr(f, "moe_route") and float(mx.abs(f.shared_expert_gate.weight).max()) == 0.0
    x = mx.random.normal((3, H))
    sg = 0.5 * 2 * np.array(f.shared_ffn(x))                                                              # sigmoid(0) * (2 * shared)
    assert np.allclose(sg, np.array(f.shared_ffn(x)), atol=1e-6) and np.allclose(np.array(f.shared_expert(x)) * 0.5, np.array(f.shared_ffn(x)), atol=1e-6)
    g = G.ffn_factory(0, dict(n_expert=E, top_k=K, gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=2.0, shared=True, shared_gate=True), t)
    assert not hasattr(g, "top_k") and not hasattr(g, "gate") and callable(g.moe_route)
    s = G.ffn_factory(0, dict(n_expert=E, top_k=K, gating="none"), t)
    assert not hasattr(s, "moe_route") and not hasattr(s, "top_k")


def test_route_description_matches_block():
    try:
        from kyojin_mlx import glm_moe
    except Exception as e:                                                                               # pragma: no cover
        pytest.skip(f"glm_moe not importable here: {e}")
    t = {k: mx.array(v) for k, v in tensors(latent=6).items()}
    g = G.ffn_factory(0, dict(n_expert=E, top_k=K, gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=2.0, shared=True, shared_gate=True, latent=True,
                              latent_norm=True), t)
    r = glm_moe.route_of(g)
    assert r.top_k == K and r.E == E and r.norm and r.scale == 2.0 and r.weight.dtype == mx.float32 and r.width == 6 and r.pre is not None and r.post is not None
    assert np.allclose(np.array(r.bias), np.array(t["exp_probs_b"]))
    x = mx.random.normal((2, H))
    # the Route's pieces recombine to the block: shared + post(sum_k w_k * experts(pre(x)))
    inds, w = g.route(x)
    y = g.switch_mlp(r.pre(x), inds)
    out = r.post((y.astype(mx.float32) * w[..., None]).sum(-2).astype(x.dtype)) + r.shared(x)
    assert np.abs(np.array(out) - np.array(g(x))).max() < 1e-5
    g2 = G.ffn_factory(0, dict(n_expert=E, top_k=K, gating="sigmoid"), t)
    assert float(np.abs(np.array(glm_moe.route_of(g2).bias)).max()) == 0.0                              # no selection bias -> zero bias vector


def test_bf16_and_f16_streams():
    t = {k: mx.array(v) for k, v in tensors().items()}
    blk = G.ffn_factory(0, dict(n_expert=E, top_k=K, gating="softmax", norm_weights=True, shared=True, shared_gate=True), t)
    x = mx.random.normal((1, 3, H))
    ref = np.array(blk(x))
    for dt in (mx.float16, mx.bfloat16):
        y = blk(x.astype(dt))
        assert y.dtype == dt and np.abs(np.array(y.astype(mx.float32)) - ref).max() < 0.05 * np.abs(ref).max()


def test_describe_with_expert_formats_never_calls_an_unreadable_variant_fused():
    """Metal run of 17/11: sigmoid + bias + scale + gated shared on q4_k experts was `glm_route` / fused although glm_moe falls back (g2a_pack): the fused decode reads nvfp4 experts only."""
    sig = G.MoeSpec(n_expert=8, top_k=2, gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=2.5, shared=True, shared_gate=True)
    soft = G.MoeSpec(n_expert=8, top_k=2, gating="softmax_weight")
    assert G.describe(sig)["kind"] == "glm_route" and G.describe(sig, None)["kind"] == "glm_route"        # no format known: the spec alone
    d = G.describe(sig, ("q4_k", "q4_k", "q6_k"))
    assert d["kind"] == "stock" and "g2a_pack" in d["why"][0] and "q4_k" in d["why"][0]
    assert G.describe(sig, ("nvfp4",) * 3)["kind"] == "glm_route"
    assert G.describe(soft, ("q4_k", "q4_k", "q6_k"))["kind"] == "foreign_moe"
    assert G.describe(soft, ("dense",) * 3)["kind"] == "stock"


def test_moe_block_class_follows_the_expert_format():
    class P:
        def __init__(self, fmt):
            self.fmt = fmt

    class SM:
        g, u, d = P("q4_k"), P("q4_k"), P("q4_k")
    assert G.expert_fmts({"switch_mlp": SM()}) == ("q4_k",) * 3 and G.expert_fmts({}) is None
