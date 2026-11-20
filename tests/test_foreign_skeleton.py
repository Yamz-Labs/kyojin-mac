"""Decoder skeleton (kyojin_mlx/foreign_skeleton.py): every spec field on tiny synthetic GGUF files, against an independent float64 numpy decoder. CPU device only.
The architecture name of the synthetic files is not in the rows: the row is passed to the loader, so no architecture is needed for the skeleton to build."""
from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_skeleton as SK, foreign_synth as SY  # noqa: E402

D, H, KV, HD, FF, V, NL = 16, 4, 2, 4, 24, 20, 2


def _erf(x):
    return np.vectorize(math.erf)(x)


ACT = {"silu": lambda x: x / (1 + np.exp(-x)), "gelu": lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))),
       "gelu_erf": lambda x: 0.5 * x * (1 + _erf(x / math.sqrt(2))), "gelu_quick": lambda x: x / (1 + np.exp(-1.702 * x)), "relu": lambda x: np.maximum(x, 0),
       "relu2": lambda x: np.maximum(x, 0) ** 2}


def norm(x, w, b, kind, eps):
    if kind == "rms":
        y = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)
    else:
        y = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)
    y = y * w if w is not None else y
    return y + b if b is not None else y


def rope(x, pos, dims, base, kind, scale=1.0, factors=None):          # x [L, heads, hd]
    out = x.copy()
    half = dims // 2
    freqs = base ** (-np.arange(half) * 2.0 / dims)
    if factors is not None:
        freqs = freqs / factors
    ang = pos[:, None] * scale * freqs[None]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if kind == "neox":
        a, b = x[..., :half], x[..., half:dims]
        out[..., :half], out[..., half:dims] = a * c - b * s, a * s + b * c
    else:
        a, b = x[..., 0:dims:2], x[..., 1:dims:2]
        out[..., 0:dims:2], out[..., 1:dims:2] = a * c - b * s, a * s + b * c
    return out


def make(tmp_path, rng, **o):
    """-> (path, row, tensors dict name -> array (logical shape), options). Options default to a llama-like block."""
    o = {**dict(norm="rms", norm_w=True, norm_b=False, gated=True, act="silu", rope="neox", rope_dims=HD, qkv="split", bias=False, ffn_bias=False,
                tied=True, pos=False, qk=None, qk_after=False, parallel=False, shared=False, norm_pos="pre", emb=1.0, logit=1.0, res=1.0, attn_scale=0.0,
                clamp=0.0, rope_scale=1.0, alpha=0.0, factors=False, heads=None, final_b=False, extra_kv={}), **o}
    hl = o["heads"] or [(H, KV)] * NL
    kv = {"general.architecture": "synth", "synth.block_count": NL, "synth.embedding_length": D, "synth.feed_forward_length": FF,
          "synth.attention.head_count": [h for h, _ in hl] if o["heads"] else H, "synth.attention.head_count_kv": [k for _, k in hl] if o["heads"] else KV,
          "synth.attention.key_length": HD, "synth.context_length": 64, "synth.rope.dimension_count": o["rope_dims"], "synth.rope.freq_base": 100.0,
          ("synth.attention.layer_norm_epsilon" if o["norm"] == "layer" else "synth.attention.layer_norm_rms_epsilon"): 1e-5}
    if o["parallel"]:
        kv["synth.use_parallel_residual"] = True
    for k, key in (("emb", "embedding_scale"), ("logit", "logit_scale"), ("res", "residual_scale")):
        if o[k] != 1.0:
            kv["synth." + key] = o[k]
    if o["attn_scale"]:
        kv["synth.attention.scale"] = o["attn_scale"]
    if o["clamp"]:
        kv["synth.attention.clamp_kqv"] = o["clamp"]
    if o["rope_scale"] != 1.0:
        kv["synth.rope.scaling.type"], kv["synth.rope.scaling.factor"] = "linear", 1.0 / o["rope_scale"]
    if o["alpha"]:
        kv["synth.rope.scaling.alpha"] = o["alpha"]
    kv.update(o["extra_kv"])
    T = {}

    def w(name, *shape, scale=0.3):
        T[name] = (rng.standard_normal(shape) * scale).astype(np.float32)

    def nw(name, n):
        if o["norm_w"]:
            T[name + ".weight"] = (1 + 0.2 * rng.standard_normal(n)).astype(np.float32)
        if o["norm_b"]:
            T[name + ".bias"] = (0.1 * rng.standard_normal(n)).astype(np.float32)
    w("token_embd.weight", V, D, scale=1.0)
    if o["pos"]:
        w("position_embd.weight", 64, D, scale=0.5)
    nw("output_norm", D)
    if not o["tied"]:
        w("output.weight", V, D)
    if o["factors"]:
        T["rope_freqs.weight"] = (1 + rng.random(o["rope_dims"] // 2)).astype(np.float32)
    for i, (h, k) in enumerate(hl):
        p = f"blk.{i}."
        nw(p + "attn_norm", D)
        if not (o["shared"]):
            nw(p + "ffn_norm", D)
        if o["norm_pos"] == "sandwich":
            nw(p + "post_attention_norm", D)
            nw(p + "post_ffw_norm", D)
        if o["qkv"] == "fused":
            w(p + "attn_qkv.weight", (h + 2 * k) * HD, D)
            if o["bias"]:
                w(p + "attn_qkv.bias", (h + 2 * k) * HD)
        else:
            for n, rows in (("q", h * HD), ("k", k * HD), ("v", k * HD)):
                w(p + f"attn_{n}.weight", rows, D)
                if o["bias"]:
                    w(p + f"attn_{n}.bias", rows)
        w(p + "attn_output.weight", D, h * HD)
        if o["bias"]:
            w(p + "attn_output.bias", D)
        if o["qk"]:
            n = HD if o["qk"] == "head" else None
            T[p + "attn_q_norm.weight"] = (1 + 0.2 * rng.standard_normal(n or h * HD)).astype(np.float32)
            T[p + "attn_k_norm.weight"] = (1 + 0.2 * rng.standard_normal(n or k * HD)).astype(np.float32)
        if o["gated"]:
            w(p + "ffn_gate.weight", FF, D)
        w(p + "ffn_up.weight", FF, D)
        w(p + "ffn_down.weight", D, FF)
        if o["ffn_bias"]:
            w(p + "ffn_up.bias", FF)
            w(p + "ffn_down.bias", D)
    path = str(tmp_path / "m.gguf")
    SY.write_gguf(path, kv, [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a)) for n, a in T.items()])
    row = dict(norm=o["norm"], act=o["act"], rope=o["rope"], norm_pos=o["norm_pos"], qk_norm_after_rope=o["qk_after"])
    return path, row, T, o, hl


def ref(T, o, hl, ids):
    """float64 decoder written from the llama.cpp graphs, independent of the module under test."""
    g = lambda n: T[n].astype(np.float64) if n in T else None      # noqa: E731
    L = len(ids)
    pos = np.arange(L).astype(np.float64)
    x = g("token_embd.weight")[ids] * o["emb"]
    if o["pos"]:
        x = x + g("position_embd.weight")[ids * 0 + np.arange(L)]
    kind, eps = o["norm"], 1e-5
    nm = lambda x, n: norm(x, g(n + ".weight"), g(n + ".bias"), kind, eps)      # noqa: E731
    factors = g("rope_freqs.weight")
    base = 100.0 * (o["alpha"] ** (HD / (HD - 2)) if o["alpha"] else 1.0)
    for i, (h, k) in enumerate(hl):
        p = f"blk.{i}."

        def attn(inp):
            if o["qkv"] == "fused":
                z = inp @ g(p + "attn_qkv.weight").T + (g(p + "attn_qkv.bias") if o["bias"] else 0)
                q, kk, v = z[:, :h * HD], z[:, h * HD:(h + k) * HD], z[:, (h + k) * HD:]
            else:
                q, kk, v = (inp @ g(p + f"attn_{n}.weight").T + (g(p + f"attn_{n}.bias") if o["bias"] else 0) for n in "qkv")
            if o["clamp"]:
                q, kk, v = (np.clip(a, -o["clamp"], o["clamp"]) for a in (q, kk, v))
            if o["qk"] == "full":
                q, kk = norm(q, g(p + "attn_q_norm.weight"), None, kind, eps), norm(kk, g(p + "attn_k_norm.weight"), None, kind, eps)
            q, kk, v = q.reshape(L, h, HD), kk.reshape(L, k, HD), v.reshape(L, k, HD)
            qn = lambda a, w: norm(a, g(p + w), None, kind, eps)      # noqa: E731
            if o["qk"] == "head" and not o["qk_after"]:
                q, kk = qn(q, "attn_q_norm.weight"), qn(kk, "attn_k_norm.weight")
            if o["rope"] != "none" and not o["pos"]:
                q = rope(q, pos, o["rope_dims"], base, o["rope"], o["rope_scale"], factors)
                kk = rope(kk, pos, o["rope_dims"], base, o["rope"], o["rope_scale"], factors)
            if o["qk"] == "head" and o["qk_after"]:
                q, kk = qn(q, "attn_q_norm.weight"), qn(kk, "attn_k_norm.weight")
            out = np.zeros((L, h, HD))
            sc = o["attn_scale"] or HD ** -0.5
            for hh in range(h):
                s = (q[:, hh] @ kk[:, hh // (h // k)].T) * sc
                s = np.where(np.tril(np.ones((L, L), bool)), s, -1e30)
                e = np.exp(s - s.max(-1, keepdims=True))
                out[:, hh] = (e / e.sum(-1, keepdims=True)) @ v[:, hh // (h // k)]
            return out.reshape(L, -1) @ g(p + "attn_output.weight").T + (g(p + "attn_output.bias") if o["bias"] else 0)

        def ffn(inp):
            if o.get("ffn_hook") is not None:                      # a test replaces the FFN of some layers (MoE): hook(layer, inp) -> array or None
                alt = o["ffn_hook"](i, inp)
                if alt is not None:
                    return alt
            up = inp @ g(p + "ffn_up.weight").T + (g(p + "ffn_up.bias") if o["ffn_bias"] else 0)
            a = ACT[o["act"]]
            hid = a(inp @ g(p + "ffn_gate.weight").T) * up if o["gated"] else a(up)
            return hid @ g(p + "ffn_down.weight").T + (g(p + "ffn_down.bias") if o["ffn_bias"] else 0)
        r = o["res"]
        if o["norm_pos"] == "post":
            x = nm(x + r * attn(x), p + "attn_norm")
            x = nm(x + r * ffn(x), p + "ffn_norm")
        elif o["parallel"]:
            n1 = nm(x, p + "attn_norm")
            x = x + r * attn(n1) + r * ffn(n1 if o["shared"] else nm(x, p + "ffn_norm"))
        else:
            a = attn(nm(x, p + "attn_norm"))
            if o["norm_pos"] == "sandwich":
                a = nm(a, p + "post_attention_norm")
            x = x + r * a
            f = ffn(nm(x, p + "ffn_norm"))
            if o["norm_pos"] == "sandwich":
                f = nm(f, p + "post_ffw_norm")
            x = x + r * f
    x = nm(x, "output_norm")
    out = x @ (g("output.weight") if "output.weight" in T else g("token_embd.weight")).T
    return out * o["logit"]


CASES = {
    "llama_like": {},
    "layernorm_bias_gelu_nongated_learned_pos_fused_qkv_untied": dict(norm="layer", norm_b=True, act="gelu", gated=False, pos=True, qkv="fused", bias=True, ffn_bias=True, tied=False, rope="none"),
    "weightless_layernorm_rope_norm_clamp": dict(norm="layer", norm_w=False, rope="norm", clamp=0.4),
    "parallel_residual_shared_norm_partial_rope": dict(norm="layer", norm_b=True, act="gelu", gated=False, parallel=True, shared=True, rope_dims=2, qkv="fused"),
    "parallel_residual_two_norms": dict(parallel=True),
    "sandwich_norms": dict(norm_pos="sandwich"),
    "post_norm": dict(norm_pos="post", norm="layer", norm_b=True),
    "qk_norm_head_before_rope": dict(qk="head"),
    "qk_norm_head_after_rope": dict(qk="head", qk_after=True),
    "qk_norm_full_width": dict(qk="full"),
    "scales": dict(emb=2.5, logit=0.5, res=0.7, attn_scale=0.31),
    "act_relu2": dict(act="relu2", gated=False),
    "act_gelu_erf": dict(act="gelu_erf"),
    "act_gelu_quick": dict(act="gelu_quick"),
    "act_relu": dict(act="relu"),
    "rope_linear_scale": dict(rope_scale=0.5),
    "rope_alpha_ntk": dict(alpha=4.0),
    "rope_factors_tensor": dict(factors=True),
    "per_layer_heads": dict(heads=[(4, 2), (2, 1)]),
    "mha_no_gqa": dict(heads=[(4, 4), (4, 4)]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_field(tmp_path, name):
    rng = np.random.default_rng(zlib.crc32(name.encode()) % 1000)
    path, row, T, o, hl = make(tmp_path, rng, **CASES[name])
    model, rep = SK.load_cpu(path, row=row)
    ids = np.array([3, 7, 1, 19, 4, 4, 11, 0, 15])
    got = np.array(model(mx.array(ids[None]))[0]).astype(np.float64)
    want = ref(T, o, hl, ids)
    assert np.abs(got - want).max() < 2e-4 * max(1.0, np.abs(want).max()), (name, np.abs(got - want).max())
    from mlx_lm.models.cache import make_prompt_cache         # the serving contract: layers + mlx-lm caches, step by step == one forward
    cache = make_prompt_cache(model)
    steps = [np.array(model(mx.array(ids[None, :5]), cache=cache))[0]] + [np.array(model(mx.array(ids[None, j:j + 1]), cache=cache))[0] for j in range(5, len(ids))]
    assert np.abs(np.concatenate(steps) - got).max() < 1e-4


def test_batch_cache_with_per_row_offsets(tmp_path):
    path, row, T, o, hl = make(tmp_path, np.random.default_rng(1), pos=True, rope="none")
    model, _ = SK.load_cpu(path, row=row)
    from mlx_lm.models.cache import BatchKVCache
    ids = mx.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    cache = [BatchKVCache([0, 0]) for _ in model.layers]
    got = model(ids, cache=cache)
    want = mx.concatenate([model(ids[i:i + 1]) for i in range(2)])
    assert float(mx.abs(got - want).max()) < 1e-4


def test_ffn_factory_hook_is_called_per_layer_with_the_layers_tensors(tmp_path):
    path, row, T, o, hl = make(tmp_path, np.random.default_rng(2))
    seen = []

    def factory(il, spec, tensors):
        seen.append((il, sorted(tensors)))
        m = SK.dense_ffn_factory(il, spec, tensors)
        return m
    model, _ = SK.load_cpu(path, row=row, ffn_factory=factory)
    assert [i for i, _ in seen] == [0, 1] and "ffn_gate.weight" in seen[0][1]
    assert type(model.model.layers[0].mlp) is SK.DenseFFN

    class Double(SK.DenseFFN):
        def __call__(self, x):
            return 2 * super().__call__(x)
    m2, _ = SK.load_cpu(path, row=row, ffn_factory=lambda il, sp, t: Double(sp, il, t))
    ids = mx.array([[1, 2, 3]])
    assert float(mx.abs(m2(ids) - model(ids)).max()) > 1e-3


@pytest.mark.parametrize("bad", [dict(extra_kv={"synth.attention.sliding_window": 8}), dict(extra_kv={"synth.rope.scaling.type": "yarn"}),
                                 dict(extra_kv={"synth.expert_count": 4})])
def test_refuses_what_it_cannot_place(tmp_path, bad):
    path, row, *_ = make(tmp_path, np.random.default_rng(3), **bad)
    with pytest.raises(SK.SkeletonError):
        SK.load_cpu(path, row=row)


def test_unknown_block_tensor_is_refused_and_no_row_is_refused(tmp_path):
    path, row, *_ = make(tmp_path, np.random.default_rng(4))
    with pytest.raises(SK.SkeletonError):
        SK.load_cpu(path)                                     # the architecture name `synth` has no row
    SY.write_gguf(path, {"general.architecture": "synth", "synth.block_count": 1, "synth.embedding_length": D, "synth.attention.head_count": H,
                         "synth.attention.layer_norm_rms_epsilon": 1e-5, "synth.feed_forward_length": FF},
                  [("token_embd.weight", (D, V), "F32", np.zeros((V, D), np.float32)), ("blk.0.ssm_a", (4,), "F32", np.zeros(4, np.float32))])
    with pytest.raises(SK.SkeletonError):
        SK.load_cpu(path, row=row)


def test_arch_with_a_class_keeps_its_path_unless_forced(tmp_path):
    from kyojin_mlx import foreign_archkit_cpu as KC
    assert KC._has_class("gptneox") and not KC._has_class("hunyuan-dense")
    assert SK.has_rows("gptneox") and SK.has_rows("hunyuan-dense")
