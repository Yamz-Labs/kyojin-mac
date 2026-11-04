"""Prefill kernels of the DS4.1 attention block: sparse pooled attention over gathered keys (kyojin_mlx.attn_prefill) vs an fp32 reference built
from the definition, and the prompt-row mHC kernels vs the stock mlx-vlm functions (bit for bit). Needs a GPU with NAX (M5)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import attn_prefill as AP, metal_nax as NX
if not NX.nax_available():
    pytest.skip("no neural accelerators on this GPU", allow_module_level=True)


def reference(q, window, pool, idxs, scale, sinks, W):
    """fp32 numpy: query t sees window rows [t + hist - W + 1, t + hist] (those >= 0) and pool rows idxs[t] (>= 0); K = V; the sink adds exp(sink) to the denominator."""
    q, window, pool, idxs, sinks = (np.array(a.astype(mx.float32)) if isinstance(a, mx.array) and a.dtype != mx.int32 else np.array(a) for a in (q, window, pool, idxs, sinks))
    _, H, T, D = q.shape
    hist = window.shape[1] - T
    out = np.zeros((1, H, T, D), np.float32)
    for t in range(T):
        rows = [r for r in range(t + hist - W + 1, t + hist + 1) if r >= 0]
        keys = np.concatenate([window[0, rows], pool[0, [i for i in idxs[0, t] if i >= 0]]], axis=0)
        s = scale * (q[0, :, t] @ keys.T)                                  # [H, nk]
        m = np.maximum(s.max(-1), sinks)
        e = np.exp(s - m[:, None])
        out[0, :, t] = (e / (e.sum(-1) + np.exp(sinks - m))[:, None]) @ keys
    return out


def make(T, hist, Np, topk, H=32, D=256, seed=0, ratio=2):
    rng = np.random.default_rng(seed)
    q = mx.array((rng.standard_normal((1, H, T, D)) * 0.6).astype(np.float16))
    window = mx.array((rng.standard_normal((1, hist + T, D)) * 0.6).astype(np.float16))
    pool = mx.array((rng.standard_normal((1, Np, D)) * 0.6).astype(np.float16))
    idx = np.full((1, T, topk), -1, np.int32)
    for t in range(T):
        n = min(Np, (hist + t + 1) // ratio)
        k = min(topk, n)
        if k:
            idx[0, t, :k] = np.sort(rng.choice(n, k, replace=False))
    return q, window, pool, mx.array(idx), mx.array(rng.standard_normal(H).astype(np.float32))


@pytest.mark.parametrize("T,hist,Np,topk,chunk", [(70, 0, 40, 48, 512), (96, 127, 80, 64, 32), (65, 30, 33, 64, 512), (130, 127, 150, 100, 64)])
def test_sparse_attention_vs_reference(T, hist, Np, topk, chunk, monkeypatch):
    monkeypatch.setattr(AP, "CHUNK", chunk)
    q, window, pool, idxs, sinks = make(T, hist, Np, topk)
    scale = 256 ** -0.5
    got = AP.sparse_attention(q, window, pool, idxs, scale, sinks, 128)
    ref = reference(q, window, pool, idxs, scale, sinks, 128)
    g = np.array(got.astype(mx.float32))
    assert got.shape == q.shape and got.dtype == mx.float16
    assert np.linalg.norm(g - ref) / np.linalg.norm(ref) < 2e-3


def test_sparse_attention_window_only_and_all_masked_pool():
    """No pooled key at all for the first queries (every idx -1): only the window and the sink."""
    q, window, pool, idxs, sinks = make(64, 0, 8, 16, ratio=1000)
    assert int((idxs >= 0).sum()) == 0
    got = AP.sparse_attention(q, window, pool, idxs, 256 ** -0.5, sinks, 128)
    ref = reference(q, window, pool, idxs, 256 ** -0.5, sinks, 128)
    assert np.linalg.norm(np.array(got.astype(mx.float32)) - ref) / np.linalg.norm(ref) < 2e-3


@pytest.mark.parametrize("T,hist,Np,topk,chunk", [(70, 0, 40, 48, 512), (96, 127, 80, 64, 32), (65, 30, 33, 64, 512),
                                                  (130, 127, 150, 100, 64), (72, 127, 40, 512, 512)])
def test_sparse_attention_online_vs_reference(T, hist, Np, topk, chunk, monkeypatch):
    """pf_attn_online: K2 folded into K1 (per-tile max in threadgroup memory, K3 rescales from the tile arrays). Same math as the fp32
    reference up to the fp16 rounding of P and the order of the reductions; against the K1/K2/K3 path the difference is of the same order as
    that rounding (one extra fp16 rounding when the block factor is applied)."""
    monkeypatch.setattr(AP, "CHUNK", chunk)
    q, window, pool, idxs, sinks = make(T, hist, Np, topk, H=64, D=512, seed=T)
    scale = 512 ** -0.5
    ref = reference(q, window, pool, idxs, scale, sinks, 128)
    out = {}
    for on in (False, True):
        monkeypatch.setattr(AP, "ONLINE", on)
        got = AP.sparse_attention(q, window, pool, idxs, scale, sinks, 128)
        out[on] = np.array(got.astype(mx.float32))
        assert got.shape == q.shape and got.dtype == mx.float16
        assert np.linalg.norm(out[on] - ref) / np.linalg.norm(ref) < 2e-3
    assert np.linalg.norm(out[True] - out[False]) / np.linalg.norm(out[False]) < 1e-3


def test_attn_online_flag_documented_and_off():
    from kyojin_mlx import flags as FL
    f = FL.FLAGS["pf_attn_online"]
    assert f.env == "KYOJIN_ATTN_PF_ONLINE" and f.default == "1" and f.runtime
    FL.set("pf_attn_online", "1")
    assert AP.ONLINE
    FL.set("pf_attn_online", "0")
    assert not AP.ONLINE


def test_sparse_attention_matches_stock_function():
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.deepseek_v4.language import _sparse_pooled_attention
    T, H, D = 96, 32, 256
    q, window, pool, idxs, sinks = make(T, 127, 90, 64, H=H, D=D, seed=3)
    scale = D ** -0.5
    hist = window.shape[1] - T
    pos = mx.arange(T)[:, None] + hist
    keyp = mx.arange(window.shape[1])[None, :]
    mask = ((keyp <= pos) & (keyp > pos - 128))[None, None]
    stock = _sparse_pooled_attention(q.astype(mx.float32), window, pool, idxs, mask, (idxs != -1)[:, None], scale, sinks).astype(mx.float16)
    got = AP.sparse_attention(q, window, pool, idxs, scale, sinks, 128)
    a, b = np.array(got.astype(mx.float32)), np.array(stock.astype(mx.float32))
    assert np.linalg.norm(a - b) / np.linalg.norm(b) < 2e-3


def test_unsupported_shapes_fall_back():
    q, window, pool, idxs, sinks = make(70, 0, 40, 48)
    assert AP.supported(q, window, pool)
    assert not AP.supported(q[:, :, :8], window, pool)              # a few rows (decode, verify): stock code
    assert not AP.supported(q.astype(mx.float32), window, pool)
    assert not AP.supported(q[:, :16], window, pool)                 # heads the kernel is not built for


@pytest.mark.parametrize("S", [2, 7, 300])
def test_mhc_prompt_rows_bit_equal_to_stock(S):
    pytest.importorskip("mlx_vlm")
    import importlib
    from kyojin_mlx import fuse as F
    L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    rng = np.random.default_rng(S)
    D = 512
    x = mx.array(rng.standard_normal((1, S, D)).astype(np.float16))
    res = mx.array(rng.standard_normal((1, S, 4, D)).astype(np.float16))
    post = mx.array(rng.uniform(0, 2, (1, S, 4)).astype(np.float32))
    comb = mx.array(rng.uniform(0, 1, (1, S, 4, 4)).astype(np.float32))
    pre = mx.array(rng.uniform(0, 1, (1, S, 4)).astype(np.float32))
    stock_pre = L.DeepseekV41Block.__dict__["hc_pre"]
    stock_pre = stock_pre.__func__ if isinstance(stock_pre, staticmethod) else stock_pre
    assert bool(mx.array_equal(F._impl_mhc_expand(L.hc_expand)(x, res, post, comb), L.hc_expand(x, res, post, comb)).item())
    assert bool(mx.array_equal(F._impl_mhc_pre(stock_pre)(res, pre), stock_pre(res, pre)).item())


def _local_attention_module():
    import importlib, json
    from mlx.utils import tree_map
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    c = dict(model_type="deepseek_v41_text", vocab_size=64, hidden_size=512, moe_intermediate_size=64, num_hidden_layers=1, num_attention_heads=32,
             num_key_value_heads=1, head_dim=128, qk_rope_head_dim=32, q_lora_rank=128, o_lora_rank=64, o_groups=4, rms_norm_eps=1e-6, rope_theta=10000,
             n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, sliding_window=128, compress_ratios=[0], compress_rope_theta=160000,
             kv_source_layer_ids=[], index_source_layer_ids=[], candidate_source_layer_id=-1, index_n_heads=4, index_head_dim=32, index_topk=16, hc_mult=4,
             max_position_embeddings=4096, rope_scaling=None, scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True, routed_scaling_factor=1.5)
    try:
        cfg = ModelConfig.from_dict(c)
    except Exception as e:                                                    # config fields differ between mlx-vlm builds: the module test needs the real one
        pytest.skip(f"ModelConfig: {e}")
    A = L.DeepseekV41Attention(cfg, 0)
    rng = np.random.default_rng(0)
    A.update(tree_map(lambda a: mx.array((rng.standard_normal(a.shape) * (0.05 if a.ndim > 1 else 0.5)).astype(np.float16)), A.parameters()))
    A.attn_sink = mx.array(rng.standard_normal(32).astype(np.float32))
    mx.eval(A.parameters())
    return L, A, rng


def test_window_only_attention_module_chunked_prompt_equals_stock(monkeypatch):
    """Ratio-0 block: two prompt chunks (the second sees the ring of the first) through the replacement vs the stock method (fp8 fake quant off in both:
    a bin flip of the quantizer turns a 1e-4 difference into a 6 % one on that element)."""
    from kyojin_mlx import attn_prefill as AP2
    L, A, rng = _local_attention_module()
    monkeypatch.setattr(L, "fake_quant_fp8_ue8m0", lambda x: x)
    stock = L.DeepseekV41Attention._attention
    new = AP2._impl_attention_local(stock)
    xs = [mx.array(rng.standard_normal((1, n, 512)).astype(np.float16)) for n in (200, 90)]
    outs = {}
    for name, fn in (("stock", stock), ("new", new)):
        cache = L.DeepseekV41Cache(1)
        res = []
        for x in xs:
            res.append(fn(A, cache, x))
            cache.advance(x.shape[1])
        outs[name] = [np.array(r.astype(mx.float32)) for r in res]
    for a, b in zip(outs["new"], outs["stock"]):
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 3e-3


@pytest.mark.parametrize("T", [64, 100, 301])
def test_mix_prompt_rows_vs_fp64_and_stock(T):
    """mixes = rms_norm(h) @ hc_fn.T for prompt rows: the NAX kernel (hi + lo weight halves) is as close to fp64 as an fp32 matmul should be; the stock
    fp32 matmul rounds through the tensor units (4e-4)."""
    from kyojin_mlx import mhc_prefill as MX
    rng = np.random.default_rng(T)
    D = 1024
    h = mx.array((rng.standard_normal((1, T, 4, D)) * 3).astype(np.float16))
    fn = mx.array((rng.standard_normal((24, 4 * D)) * 0.02).astype(np.float32))
    got = np.array(MX.mixes(h, fn, 1e-20))[0]
    x = np.array(h.reshape(T, 4 * D).astype(mx.float32)).astype(np.float64)
    ref = (x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-20)) @ np.array(fn).astype(np.float64).T
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 5e-6
    stock = np.array(mx.fast.rms_norm(h.flatten(-2).astype(mx.float32), None, 1e-20) @ fn.T)[0]
    assert np.linalg.norm(got - stock) / np.linalg.norm(ref) < 3e-3
    assert MX.supported(h, fn, 4) and not MX.supported(h[:, :8], fn, 4)
