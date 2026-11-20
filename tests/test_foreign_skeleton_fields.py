"""Skeleton spec fields for the flagship MoE family (sliding-window masks, per-layer rope, NoPE query scale, attention output gate, softcap, residual
forms, swiglu_oai) on tiny synthetic GGUF files, against an independent float64 numpy decoder written from the llama.cpp graphs (cited per test).
CPU device only. The architecture name of the synthetic files is not in the rows: the row is passed to the loader."""
from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_skeleton as SK, foreign_synth as SY  # noqa: E402

D, H, KV, HD, FF, V, NL = 16, 4, 2, 4, 24, 20, 4
IDS = np.array([3, 7, 1, 19, 4, 4, 11, 0, 15, 2, 8, 9])
EPS = 1e-5


def silu(x):
    return x / (1 + np.exp(-x))


def swiglu_oai(g, u, alpha=1.702, limit=7.0):                      # ggml_swiglu_oai
    g = np.minimum(g, limit)
    u = np.clip(u, -limit, limit)
    return g / (1 + np.exp(-alpha * g)) * (u + 1.0)


def rms(x, w=None, eps=EPS):
    y = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)
    return y * w if w is not None else y


def rope(x, pos, base, dims):                                      # neox, x [L, heads, hd]
    half = dims // 2
    ang = pos[:, None] * (base ** (-np.arange(half) * 2.0 / dims))[None]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    out = x.copy()
    a, b = x[..., :half], x[..., half:dims]
    out[..., :half], out[..., half:dims] = a * c - b * s, a * s + b * c
    return out


OPT = dict(window=0, pattern=None, period=None, rope_on="all", base=100.0, base_swa=0.0, temp=0.0, temp_len=8, gate=None, gate_bias=False, o_norm=False,
           softcap=0.0, post_eps=0.0, resid="plain", sandwich=False, embed_rms=False, ffn_post_first=False, act="silu", logit=1.0, heads=None)


def make(tmp_path, rng, **kw):
    o = {**OPT, **kw}
    hl = o["heads"] or [(H, KV)] * NL
    kv = {"general.architecture": "synth", "synth.block_count": NL, "synth.embedding_length": D, "synth.feed_forward_length": FF,
          "synth.attention.head_count": H, "synth.attention.head_count_kv": KV, "synth.attention.key_length": HD, "synth.context_length": 64,
          "synth.rope.dimension_count": HD, "synth.rope.freq_base": o["base"], "synth.attention.layer_norm_rms_epsilon": EPS}
    if o["window"]:
        kv["synth.attention.sliding_window"] = o["window"]
        if o["pattern"] is not None:
            kv["synth.attention.sliding_window_pattern"] = [int(x) for x in o["pattern"]]
    if o["base_swa"]:
        kv["synth.rope.freq_base_swa"] = o["base_swa"]
    if o["temp"]:
        kv["synth.attention.temperature_scale"], kv["synth.attention.temperature_length"] = o["temp"], o["temp_len"]
    if o["softcap"]:
        kv["synth.final_logit_softcapping"] = o["softcap"]
    if o["post_eps"]:
        kv["synth.post_norm_epsilon"] = o["post_eps"]
    if o["logit"] != 1.0:
        kv["synth.logit_scale"] = o["logit"]
    T = {}

    def w(name, *shape, scale=0.3):
        T[name] = (rng.standard_normal(shape) * scale).astype(np.float32)

    def nw(name, n):
        T[name + ".weight"] = (1 + 0.2 * rng.standard_normal(n)).astype(np.float32)
    w("token_embd.weight", V, D, scale=1.0)
    w("output.weight", V, D)
    nw("output_norm", D)
    for i in range(NL):
        p = f"blk.{i}."
        nw(p + "attn_norm", D)
        nw(p + "ffn_norm", D)
        if o["sandwich"]:
            nw(p + "post_attention_norm", D)
            nw(p + "post_ffw_norm", D)
        for n, rows in (("q", H * HD), ("k", KV * HD), ("v", KV * HD)):
            w(p + f"attn_{n}.weight", rows, D)
        w(p + "attn_output.weight", D, H * HD)
        T[p + "attn_q_norm.weight"] = (1 + 0.2 * rng.standard_normal(HD)).astype(np.float32)
        T[p + "attn_k_norm.weight"] = (1 + 0.2 * rng.standard_normal(HD)).astype(np.float32)
        if o["gate"]:
            w(p + "attn_gate.weight", H if o["gate"] == "head" else H * HD, D, scale=0.8)
            if o["gate_bias"]:
                w(p + "attn_gate.bias", H, scale=0.5)
        if o["resid"] == "vector":
            for k in ("attn_resid_alpha", "attn_resid_beta", "ffn_resid_alpha", "ffn_resid_beta"):
                T[p + k + ".weight"] = (1 + 0.3 * rng.standard_normal(D)).astype(np.float32)
        w(p + "ffn_gate.weight", FF, D)
        w(p + "ffn_up.weight", FF, D)
        w(p + "ffn_down.weight", D, FF)
    path = str(tmp_path / "m.gguf")
    SY.write_gguf(path, kv, [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a)) for n, a in T.items()])
    row = dict(norm="rms", act=o["act"], rope="neox", norm_pos="sandwich" if o["sandwich"] else "pre")
    if o["period"] is not None:
        row["swa_period"] = o["period"]
    if o["rope_on"] != "all":
        row["rope_on"] = o["rope_on"]
    if o["resid"] != "plain":
        row["residual"] = o["resid"]
    if o["o_norm"]:
        row["attn_out_norm"] = True
    if o["ffn_post_first"]:
        row["ffn_post_norm_mul_first"] = True
    if o["embed_rms"]:
        row["embed_norm_plain"] = True
    return path, row, T, o


def swa_layers(o):
    if not o["window"]:
        return [False] * NL
    if o["pattern"] is not None:
        return [bool(x) for x in o["pattern"]][:NL]
    per = o["period"]
    return [(il % per) < per - 1 for il in range(NL)]


def ref(T, o, ids):
    """float64 decoder from llama.cpp: beam.cpp / muse-glimmer.cpp / muse-spark.cpp / step35.cpp (graph) + llama-hparams.h:557 (mask)."""
    g = lambda n: T[n].astype(np.float64) if n in T else None      # noqa: E731
    L = len(ids)
    pos = np.arange(L).astype(np.float64)
    x = g("token_embd.weight")[ids]
    if o["embed_rms"]:
        x = rms(x)                                                 # weightless: muse-glimmer.cpp:70
    sw = swa_layers(o)
    peps = o["post_eps"] or EPS
    for i in range(NL):
        p = f"blk.{i}."
        nrm = rms(x, g(p + "attn_norm.weight"))
        q, k, v = (nrm @ g(p + f"attn_{n}.weight").T for n in "qkv")
        q, k, v = q.reshape(L, H, HD), k.reshape(L, KV, HD), v.reshape(L, KV, HD)
        q, k = rms(q, g(p + "attn_q_norm.weight")), rms(k, g(p + "attn_k_norm.weight"))
        has_rope = o["rope_on"] == "all" or (o["rope_on"] == "swa") == sw[i]
        if has_rope:
            base = o["base_swa"] if (sw[i] and o["base_swa"]) else o["base"]
            q, k = rope(q, pos, base, HD), rope(k, pos, base, HD)
        elif o["temp"]:                                            # llama-graph.cpp:318-323, attn_temp_pow, offset 1
            q = q * np.exp(o["temp"] * np.log1p((pos + 1.0) / o["temp_len"]))[:, None, None]
        out = np.zeros((L, H, HD))
        for hh in range(H):
            sc = q[:, hh] @ k[:, hh // (H // KV)].T * HD ** -0.5
            dist = pos[:, None] - pos[None, :]
            ok = dist >= 0
            if sw[i]:
                ok = ok & (dist < o["window"])                     # is_masked_swa STANDARD: masked iff p1 - p0 >= n_swa
            sc = np.where(ok, sc, -np.inf)
            e = np.exp(sc - sc.max(-1, keepdims=True))
            out[:, hh] = (e / e.sum(-1, keepdims=True)) @ v[:, hh // (H // KV)]
        if o["o_norm"]:
            out = rms(out)                                         # muse-spark.cpp:129-133: weightless, per head
        if o["gate"]:
            gt = nrm @ g(p + "attn_gate.weight").T
            gt = gt[:, :, None] if o["gate"] == "head" else gt.reshape(L, H, HD)   # step35.cpp:253-266 (per head) / beam.cpp:174-178 (per element)
            if o["gate_bias"]:
                gt = gt + g(p + "attn_gate.bias")[None, :, None]   # bias [n_head] broadcast over the head dim (beam.cpp:177)
            out = out * (1 / (1 + np.exp(-gt)))
        a = out.reshape(L, -1) @ g(p + "attn_output.weight").T
        if o["sandwich"]:
            a = rms(a, g(p + "post_attention_norm.weight"), peps)
        x = _resid(x, a, o, g, p, "attn", 2 * (i + 1))
        n2 = rms(x, g(p + "ffn_norm.weight"))
        if o.get("ffn_hook") is not None:                          # a test replaces the FFN (MoE): hook(layer, normed input) -> array
            f = o["ffn_hook"](i, n2)
        else:
            gt_, up = n2 @ g(p + "ffn_gate.weight").T, n2 @ g(p + "ffn_up.weight").T
            hid = swiglu_oai(gt_, up) if o["act"] == "swiglu_oai" else silu(gt_) * up
            f = hid @ g(p + "ffn_down.weight").T
        if o["sandwich"]:
            f = rms(f * g(p + "post_ffw_norm.weight"), None, peps) if o["ffn_post_first"] else rms(f, g(p + "post_ffw_norm.weight"), peps)
        x = _resid(x, f, o, g, p, "ffn", 2 * (i + 1) + 1)
    x = rms(x, g("output_norm.weight"))
    out = x @ g("output.weight").T * o["logit"]
    if o["softcap"]:
        out = np.tanh(out / o["softcap"]) * o["softcap"]           # beam.cpp:247-251
    return out


def _resid(x, branch, o, g, p, kind, idx):
    if o["resid"] == "depth":                                      # beam.cpp:109-113: branch / sqrt(i) + stream * sqrt((i - 1) / i)
        return branch * (1 / math.sqrt(idx)) + x * math.sqrt((idx - 1) / idx)
    if o["resid"] == "vector":                                     # muse-spark.cpp:153-156: stream * alpha + branch * beta
        return x * g(p + f"{kind}_resid_alpha.weight") + branch * g(p + f"{kind}_resid_beta.weight")
    return x + branch


def run(tmp_path, name, **kw):
    rng = np.random.default_rng(zlib.crc32(name.encode()) % 1000)
    path, row, T, o = make(tmp_path, rng, **kw)
    model, rep = SK.load_cpu(path, row=row)
    got = np.array(model(mx.array(IDS[None]))[0]).astype(np.float64)
    want = ref(T, o, IDS)
    return model, got, want, np.abs(got - want).max()


def cache_consistent(model, got):
    """Token-by-token decode through mlx-lm KVCache equals the full-sequence forward (prefill 5, then one token at a time)."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    steps = [np.array(model(mx.array(IDS[None, :5]), cache=cache))[0]] + [np.array(model(mx.array(IDS[None, j:j + 1]), cache=cache))[0] for j in range(5, len(IDS))]
    return np.abs(np.concatenate(steps) - got).max()


MASK_CASES = {
    "swa_all_layers": dict(window=3, pattern=[1, 1, 1, 1]),
    "swa_pattern_array_global_every_third": dict(window=3, pattern=[1, 1, 0, 1], base_swa=7.0),
    "swa_integer_period_row": dict(window=4, period=2),
    "swa_rope_only_on_windowed_layers_nope_global": dict(window=3, pattern=[1, 0, 1, 0], rope_on="swa", base_swa=50.0),
    "swa_nope_global_temp_scale": dict(window=3, pattern=[1, 0, 1, 0], rope_on="swa", temp=0.1, temp_len=4),
    "window_wider_than_sequence_is_causal": dict(window=64, pattern=[1, 0, 1, 0]),
}


@pytest.mark.parametrize("name", list(MASK_CASES))
def test_mask_kinds(tmp_path, name):
    model, got, want, diff = run(tmp_path, name, **MASK_CASES[name])
    print(name, "max abs diff vs numpy", diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max()), (name, diff)
    cdiff = cache_consistent(model, got)
    print(name, "cache consistency", cdiff)
    assert cdiff < 1e-4, (name, cdiff)


def test_window_actually_changes_the_output(tmp_path):
    _, got_w, _, _ = run(tmp_path, "w1", window=3, pattern=[1, 1, 1, 1])
    _, got_g, _, _ = run(tmp_path, "w1", window=64, pattern=[1, 1, 1, 1])
    assert np.abs(got_w - got_g).max() > 1e-3


def test_batch_cache_window_mask(tmp_path):
    from mlx_lm.models.cache import BatchKVCache
    rng = np.random.default_rng(5)
    path, row, T, o = make(tmp_path, rng, window=3, pattern=[1, 0, 1, 0], rope_on="swa", temp=0.1, temp_len=4)
    model, _ = SK.load_cpu(path, row=row)
    ids = mx.array(np.stack([IDS[:6], IDS[3:9]]))
    cache = [BatchKVCache([0, 0]) for _ in model.layers]
    model(ids[:, :4], cache=cache)
    got2 = np.array(model(ids[:, 4:5], cache=cache))
    want = np.concatenate([np.array(model(ids[i:i + 1, :5]))[0] for i in range(2)])
    assert np.abs(got2[:, 0] - np.stack([want[4], want[9]])).max() < 1e-4


def test_refusals(tmp_path):
    path, row, T, o = make(tmp_path, np.random.default_rng(1), window=3, pattern=[1, 1, 1, 1])
    kv_bad = dict(row)
    spec = SK.build_spec  # window without pattern and without period refuses
    from kyojin_mlx import foreign_gguf as GF
    g = GF.GgufFile(path)
    kv = dict(g.kv)
    del kv["synth.attention.sliding_window_pattern"]
    with pytest.raises(SK.SkeletonError, match="without a layer pattern"):
        spec("synth", kv, {k: ti.ne for k, ti in g.tensors.items()}, row=kv_bad)
    kv["synth.attention.sliding_window_pattern"] = [1, 1]
    with pytest.raises(SK.SkeletonError, match="2 entries for 4 layers"):
        spec("synth", kv, {k: ti.ne for k, ti in g.tensors.items()}, row=kv_bad)
    g.close()


GATE_CASES = {
    "gate_per_head": dict(gate="head"),
    "gate_per_head_with_bias": dict(gate="head", gate_bias=True),
    "gate_per_element_bias_per_head": dict(gate="element", gate_bias=True),
    "gate_per_element_no_bias_with_o_norm": dict(gate="element", o_norm=True),
    "gate_with_window_and_nope": dict(gate="element", gate_bias=True, window=3, pattern=[1, 0, 1, 0], rope_on="swa", temp=0.1, temp_len=4),
}


@pytest.mark.parametrize("name", list(GATE_CASES))
def test_attention_output_gate(tmp_path, name):
    model, got, want, diff = run(tmp_path, name, **GATE_CASES[name])
    print(name, "max abs diff vs numpy", diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max()), (name, diff)
    cdiff = cache_consistent(model, got)
    print(name, "cache consistency", cdiff)
    assert cdiff < 1e-4


def test_gate_changes_the_output(tmp_path):
    _, a, _, _ = run(tmp_path, "g", gate="element")
    _, b, _, _ = run(tmp_path, "g")
    assert np.abs(a - b).max() > 1e-3


FIELD_CASES = {
    "softcap_with_logit_scale": dict(softcap=3.0, logit=0.4),
    "post_norm_eps_sandwich": dict(sandwich=True, post_eps=1e-8),
    "post_norm_eps_large": dict(sandwich=True, post_eps=0.05),
    "embed_rms_weightless": dict(embed_rms=True),
    "residual_depth_scaled": dict(resid="depth", sandwich=True),
    "residual_vector_alpha_beta": dict(resid="vector", sandwich=True, post_eps=1e-8, ffn_post_first=True),
    "all_flagship_fields_together": dict(window=3, pattern=[1, 0, 1, 0], rope_on="swa", base_swa=30.0, gate="element", o_norm=True, softcap=2.5, logit=0.3,
                                         sandwich=True, post_eps=1e-6, embed_rms=True, resid="vector", ffn_post_first=True),
}


@pytest.mark.parametrize("name", list(FIELD_CASES))
def test_softcap_residual_forms(tmp_path, name):
    model, got, want, diff = run(tmp_path, name, **FIELD_CASES[name])
    print(name, "max abs diff vs numpy", diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max()), (name, diff)
    cdiff = cache_consistent(model, got)
    print(name, "cache consistency", cdiff)
    assert cdiff < 1e-4


def test_residual_vector_refuses_missing_tensors(tmp_path):
    path, row, T, o = make(tmp_path, np.random.default_rng(1), sandwich=True)
    row["residual"] = "vector"
    with pytest.raises(SK.SkeletonError, match="residual vector form without"):
        SK.load_cpu(path, row=row)


def test_swiglu_oai_dense_ffn(tmp_path):
    model, got, want, diff = run(tmp_path, "oai", act="swiglu_oai")
    print("swiglu_oai dense max abs diff vs numpy", diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max())
    assert cache_consistent(model, got) < 1e-4
    _, silu_out, _, _ = run(tmp_path, "oai")
    assert np.abs(got - silu_out).max() > 1e-3


def test_swiglu_oai_limits_are_active(tmp_path):
    """Large pre-activations reach the limit 7: the clamp of gate (one-sided) and up (two-sided) is in the formula."""
    x = mx.array([[-9.0, 9.0, 3.0, 20.0]])
    y = mx.array([[9.0, -9.0, 0.5, 20.0]])
    got = np.array(SK.GLU_ACTS["swiglu_oai"](x, y))[0].astype(np.float64)
    gx, uy = np.minimum(np.array(x)[0].astype(np.float64), 7.0), np.clip(np.array(y)[0].astype(np.float64), -7, 7)
    assert np.abs(got - gx / (1 + np.exp(-1.702 * gx)) * (uy + 1)).max() < 1e-5


def test_swiglu_oai_needs_a_gated_ffn(tmp_path):
    path, row, T, o = make(tmp_path, np.random.default_rng(1), act="swiglu_oai")
    from kyojin_mlx import foreign_gguf as GF
    g = GF.GgufFile(path)
    ne = {k: ti.ne for k, ti in g.tensors.items() if "ffn_gate" not in k}
    sp = SK.build_spec("synth", g.kv, ne, row=row)
    g.close()
    with pytest.raises(SK.SkeletonError, match="needs a gated FFN"):
        SK.Model(sp)
