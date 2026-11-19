"""KYOJIN_GLM_COMPILE: a compiled linear-attention decode layer equals the eager layer (CPU, tiny stock glm5_next layer, no model load)."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from kyojin_mlx import glm_compile as GC

mx.set_default_device(mx.cpu)


def _cfg(sparse):
    from mlx_vlm.models.glm5_next.config import TextConfig
    return TextConfig(vocab_size=64, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                      n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, kv_lora_rank=16, q_lora_rank=16, qk_nope_head_dim=16, v_head_dim=16,
                      qk_head_dim=16, linear_head_dim=16, linear_num_heads=4, index_topk=16, index_kpool=4, index_head_dim=16, index_n_heads=2,
                      layer_types=["linear_attention"], mlp_layer_types=["sparse" if sparse else "dense"], indexer_types=["full"], hc_mult=4)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("rows", [1, 3])
def test_compiled_layer_matches_eager(sparse, rows):
    from mlx_vlm.models.glm5_next.language import DecoderLayer
    from mlx_vlm.models.cache import ArraysCache
    mx.random.seed(0)
    cfg = _cfg(sparse)
    layer = DecoderLayer(cfg, 0)
    layer.set_dtype(mx.float32)
    mx.eval(layer.parameters())

    px = mx.random.normal((rows, 5, 4, 64))

    def fresh():
        c = ArraysCache(size=2)
        layer(px, None, c)                                           # prefill leaves both states
        return c
    GC.install(type("L", (), {"model": type("M", (), {"layers": [layer]})()}))
    x = mx.random.normal((rows, 1, 4, 64))
    GC.ENABLED = False
    ce = fresh()
    ye, _ = layer(x, None, ce)
    GC.ENABLED = True
    GC.STATS.update(fast=0, fallback=0, traced=0)
    cc = fresh()
    y0, _ = layer(x, None, cc)                                      # first call of the shape: stock
    cc = fresh()
    y1, _ = layer(x, None, cc)                                      # traced + compiled
    mx.eval(ye, y1, cc.cache, ce.cache)
    assert GC.STATS["fast"] == 1 and GC.STATS["traced"] == 1, GC.STATS
    assert mx.allclose(ye, y1, atol=1e-4, rtol=1e-4).item()
    for a, b in zip(ce.cache, cc.cache):
        assert mx.allclose(a, b, atol=1e-4, rtol=1e-4).item()
    GC.ENABLED = False


def _dsa_cfg(sparse, kind):
    from mlx_vlm.models.glm5_next.config import TextConfig
    c = _cfg(sparse)
    return TextConfig(**{**c.__dict__, "layer_types": ["deepseek_sparse_attention"], "indexer_types": [kind]})


def _model(layers):
    return type("L", (), {"model": type("M", (), {"layers": layers})()})()


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("kind", ["full", "shared"])
def test_compiled_dsa_layer_matches_eager(sparse, kind):
    """A / eager core / B around the attention of a DSA layer equal the stock layer over several decode steps (growing cache, pooled indexer)."""
    from mlx_vlm.models.glm5_next.language import DecoderLayer
    from mlx_vlm.models.cache import CacheList, KVCache, PoolingCache
    mx.random.seed(1)
    cfg = _dsa_cfg(sparse, kind)
    layer = DecoderLayer(cfg, 0)
    layer.set_dtype(mx.float32)
    mx.eval(layer.parameters())
    ind = layer.self_attn.indexer

    def fresh():
        c = CacheList(KVCache(), KVCache()) if ind is None else CacheList(KVCache(), KVCache(), PoolingCache(ind.index_kpool), KVCache())
        px = mx.random.normal((1, 40, 4, 64), key=mx.random.key(5))
        prev = mx.zeros((1, 40, 16), dtype=mx.int32) if ind is None else None
        layer(px, None, c, prev)
        return c
    GC.install(_model([layer]))
    xs = [mx.random.normal((1, 1, 4, 64), key=mx.random.key(10 + i)) for i in range(6)]
    prev = mx.arange(16, dtype=mx.int32).reshape(1, 1, 16) if ind is None else None
    GC.ENABLED = False
    ce, ye = fresh(), []
    for x in xs:
        y, _ = layer(x, None, ce, prev)
        ye.append(y)
    GC.ENABLED = True
    GC.STATS.update({k: 0 for k in GC.STATS})
    cc, yc = fresh(), []
    for x in xs:
        y, _ = layer(x, None, cc, prev)
        yc.append(y)
    mx.eval(ye, yc)
    assert GC.STATS["dsa_fast"] == len(xs) - 1 and GC.STATS["dsa_traced"] == 1, GC.STATS
    for a, b in zip(ye, yc):
        assert mx.allclose(a, b, atol=1e-4, rtol=1e-4).item()
    GC.ENABLED = False


def test_g3b_slot_crosses_compiled_regions(monkeypatch):
    """With the G3b layer call installed (fake pure-mx kernels), the partial sums travel layer to layer through glm_hc.SLOT: compiled regions must take / hand them over
    as explicit inputs / outputs (same result as eager, nothing left in the slot)."""
    from mlx_vlm.models.glm5_next.config import TextConfig
    from mlx_vlm.models.glm5_next.language import DecoderLayer
    from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache, PoolingCache
    from kyojin_mlx import glm_hc as H

    calls = {"partials": 0}

    def partials(x, fn):
        calls["partials"] += 1
        return mx.stack([x.sum(axis=(2, 3)), (x * x).mean(axis=(2, 3))], axis=-1).reshape(-1, 2) * fn.sum()

    def pre(x, part, hc, w, eps):
        xo = x.mean(axis=2) * w * (1.0 + 0.1 * part[:, :1].reshape(x.shape[0], x.shape[1], 1))
        return xo, mx.concatenate([part, part * 0.5], axis=-1)

    def expand(out, x, pc, fn_next):
        x2 = x * 0.9 + out[:, :, None, :] * (0.1 + 0.01 * pc.sum())
        return x2, (None if fn_next is None else partials(x2, fn_next))

    monkeypatch.setattr(H, "partials", partials)
    monkeypatch.setattr(H, "pre", pre)
    monkeypatch.setattr(H, "expand", expand)
    monkeypatch.setattr(H, "ok_shape", lambda x, hc: True)
    monkeypatch.setattr(H, "ENABLED", True)

    mx.random.seed(2)
    base = _cfg(True)
    cfg = TextConfig(**{**base.__dict__, "num_hidden_layers": 3, "layer_types": ["linear_attention", "deepseek_sparse_attention", "linear_attention"],
                        "mlp_layer_types": ["sparse"] * 3, "indexer_types": ["full"] * 3})
    layers = [DecoderLayer(cfg, i) for i in range(3)]
    for ly in layers:
        ly.set_dtype(mx.float32)
        ly.attn_hc.fn = mx.random.normal((24, 256)) * 0.05
        ly.ffn_hc.fn = mx.random.normal((24, 256)) * 0.05
    mx.eval([ly.parameters() for ly in layers])
    ind = layers[1].self_attn.indexer
    lm = type("LM", (), {"language_model": _model(layers), "model": None})()
    assert H.install(lm, log=lambda *_: None) == 3
    GC.install(lm.language_model)

    def caches():
        cs = [ArraysCache(size=2), CacheList(KVCache(), KVCache(), PoolingCache(ind.index_kpool), KVCache()), ArraysCache(size=2)]
        x = mx.random.normal((1, 40, 4, 64), key=mx.random.key(3))
        GC.ENABLED = False
        for ly, c in zip(layers, cs):
            x, _ = ly(x, None, c, None)
        mx.eval(x)
        return cs

    def step(cs, x):
        for ly, c in zip(layers, cs):
            x, _ = ly(x, None, c, None)
        return x

    xs = [mx.random.normal((1, 1, 4, 64), key=mx.random.key(20 + i)) for i in range(5)]
    GC.ENABLED = False
    ce = caches()
    ye = [step(ce, x) for x in xs]
    cc = caches()
    GC.ENABLED = True
    GC.STATS.update({k: 0 for k in GC.STATS})
    yc = [step(cc, x) for x in xs]
    mx.eval(ye, yc)
    for a, b in zip(ye, yc):
        assert mx.allclose(a, b, atol=1e-4, rtol=1e-4).item()
    assert GC.STATS["fast"] == 2 * 4 and GC.STATS["dsa_fast"] == 4, GC.STATS
    GC.ENABLED = False
