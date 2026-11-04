"""step_regions: compiled decode regions equal the stock Qwen3.5 forward bit for bit (tiny random hybrid model, 4 layers = 3 gated-delta + 1 attention)."""
import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache

from kyojin_mlx import step_regions as SR


def _model():
    tc = dict(model_type="qwen3_5", hidden_size=64, num_hidden_layers=8, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
              linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
              vocab_size=128, rms_norm_eps=1e-6, rope_theta=1000.0, partial_rotary_factor=0.5, max_position_embeddings=512,
              full_attention_interval=4, tie_word_embeddings=False, rope_parameters={"type": "default", "rope_theta": 1000.0, "partial_rotary_factor": 0.5})
    m = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(dict(model_type="qwen3_5", text_config=tc)))
    mx.random.seed(3)
    m.update(tree_map(lambda a: (mx.random.normal(a.shape) * 0.1).astype(mx.float32) if a.dtype == mx.float32 else a, m.parameters()))
    m.eval()
    mx.eval(m.parameters())
    return m


def _run(m, steps=20):
    cache = make_prompt_cache(m)
    ids = mx.array(np.arange(1, 40) % 120)[None]
    lg = m(ids, cache=cache)[:, -1]
    t = mx.argmax(lg, axis=-1)
    out = []
    for _ in range(steps):
        lg = m(t[:, None], cache=cache)[:, -1]
        t = mx.argmax(lg, axis=-1)
        mx.eval(lg, t)
        out.append(np.array(lg))
    return out


def test_regions_bit_equal():
    m = _model()
    ref = _run(m)
    assert SR.enable(m) or not SR.ENABLED
    got = _run(m)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    if SR.ENABLED:
        assert SR.STATS["fast"] >= 20


def test_wide_cache_pick():
    """KYOJIN_STEP_REGIONS_WIDE: off = bare KVCache only; on = also WinKVCache and a one-cache multiuser RowsKV around either (not a 2-user batch)."""
    from mlx_lm.models.cache import KVCache
    from kyojin_mlx.kvwin import WinKVCache
    RowsKV = type("RowsKV", (), {})
    kv, win = KVCache(), WinKVCache.__new__(WinKVCache)
    one, two = RowsKV(), RowsKV()
    one.caches, two.caches = [win], [kv, kv]
    w0 = SR.WIDE
    try:
        SR.WIDE = False
        assert [SR._solo(c) for c in (kv, win, one)] == [kv, None, None]
        SR.WIDE = True
        assert [SR._solo(c) for c in (kv, win, one, two)] == [kv, win, win, None]
    finally:
        SR.WIDE = w0
