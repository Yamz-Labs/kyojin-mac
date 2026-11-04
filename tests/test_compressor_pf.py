"""Vectorized prefill pooling of the DS4.1 compressor (kyojin_mlx/compressor_pf.py) against the stock per-token loop: pooled values and the final
state must be bit for bit the same, for every start alignment, chunk length and ratio, over a chain of chunks, and through a whole tiny model."""
import os
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import compressor_pf as PF  # noqa: E402


def make_pair(ratio, seed=0):
    import mlx_vlm.models.deepseek_v41.language as L
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    cfg = ModelConfig.from_dict({**M.CFG, "compress_ratios": [0, 0, ratio, ratio, 1, 1, 1, 1]})
    ref, new = L.Compressor(cfg, 2), L.Compressor(cfg, 2)
    mx.random.seed(seed)
    from mlx.utils import tree_flatten, tree_unflatten
    w = [(k, mx.random.normal(v.shape) * (v.shape[-1] ** -0.5 if v.ndim > 1 else 1.0)) for k, v in tree_flatten(ref.parameters())]
    ref.update(tree_unflatten(w))
    new.update(tree_unflatten(w))
    mx.eval(ref.parameters(), new.parameters())
    from kyojin_mlx.ced import _override
    _override(new, "__call__", PF._pool_call)
    return L, cfg, ref, new


def twin_caches(L, ref, start, seed):
    """Two caches in the same state: `start` tokens already pooled by the stock code (so the compressor state holds a realistic partial group)."""
    c1, c2 = L.DeepseekV41Cache(8), L.DeepseekV41Cache(8)
    if start:
        x = mx.random.normal((1, start, 64), key=mx.random.key(seed)).astype(mx.bfloat16)
        ref(x, 0, c1)
        c2.kv_state[2], c2.score_state[2] = c1.kv_state[2], c1.score_state[2]
        mx.eval(c1.kv_state[2], c1.score_state[2])
    return c1, c2


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("start", [64, 65, 67, 130])
@pytest.mark.parametrize("n", [64, 65, 70, 128, 131])
def test_chunk_equals_the_stock_loop(ratio, start, n):
    L, cfg, ref, new = make_pair(ratio)
    c1, c2 = twin_caches(L, ref, start, 3)
    x = mx.random.normal((1, n, 64), key=mx.random.key(11)).astype(mx.bfloat16)
    a = ref(x, start, c1)
    b = new(x, start, c2)
    assert same(a, b)
    assert same(c1.kv_state[2], c2.kv_state[2]) and same(c1.score_state[2], c2.score_state[2])


@pytest.mark.parametrize("ratio", [2, 4])
def test_chain_of_chunks_with_odd_lengths(ratio):
    L, cfg, ref, new = make_pair(ratio, seed=2)
    c1, c2 = L.DeepseekV41Cache(8), L.DeepseekV41Cache(8)
    s, outs = 0, []
    for k, n in enumerate([70, 64, 97, 5, 1, 1, 80, 64, 129]):          # prefill chunks, a few decode-sized ones in between (stock path)
        x = mx.random.normal((1, n, 64), key=mx.random.key(k)).astype(mx.bfloat16)
        a, b = ref(x, s, c1), new(x, s, c2)
        assert same(a, b), (k, n, s)
        assert same(c1.kv_state[2], c2.kv_state[2]) and same(c1.score_state[2], c2.score_state[2]), (k, n, s)
        s += n


def test_short_chunks_and_ratio_one_take_the_stock_code():
    L, cfg, ref, new = make_pair(2)
    c1, c2 = twin_caches(L, ref, 66, 5)
    x = mx.random.normal((1, PF.MIN_ROWS - 1, 64), key=mx.random.key(1)).astype(mx.bfloat16)
    assert same(ref(x, 66, c1), new(x, 66, c2))
    assert same(c1.kv_state[2], c2.kv_state[2])


def test_batched_state_of_another_size_is_left_to_the_stock_code():
    L, cfg, ref, new = make_pair(2)
    c1, c2 = L.DeepseekV41Cache(8), L.DeepseekV41Cache(8)
    x0 = mx.random.normal((2, 66, 64), key=mx.random.key(2)).astype(mx.bfloat16)
    ref(x0, 0, c1)
    c2.kv_state[2], c2.score_state[2] = c1.kv_state[2], c1.score_state[2]
    x = mx.random.normal((1, 70, 64), key=mx.random.key(3)).astype(mx.bfloat16)           # state has 2 rows, the call 1: stock path
    assert same(ref(x, 66, c1), new(x, 66, c2)) and same(c1.kv_state[2], c2.kv_state[2])


def test_install_arms_ratio_two_compressors_once_and_a_whole_model_is_unchanged():
    lm1, cfg = M.make()
    lm2, _ = M.make()
    assert PF.install(lm2) == 1 and PF.install(lm2) == 0                      # only KV source layer 2 has a ratio-2 compressor (layer 4 is ratio 1)
    ids = M.tokens(300)
    c1, c2 = lm1.make_cache(), lm2.make_cache()
    s = 0
    for e in (70, 134, 200, 300):                                              # chunks of 70, 64, 66, 100 tokens
        a = lm1(mx.array([ids[s:e]]), cache=c1).logits
        b = lm2(mx.array([ids[s:e]]), cache=c2).logits
        mx.eval(a, b)
        assert same(a, b), e
        s = e
    assert all(same(x, y) for x, y in zip(c1[0].state, c2[0].state))


def test_negative_control_a_wrong_slot_roll_is_caught():
    L, cfg, ref, new = make_pair(2)
    c1, c2 = twin_caches(L, ref, 65, 9)
    x = mx.random.normal((1, 71, 64), key=mx.random.key(4)).astype(mx.bfloat16)          # ends at 136: even, e = 0
    ref(x, 65, c1)
    new(x, 65, c2)
    assert same(c1.kv_state[2], c2.kv_state[2])
    c2.kv_state[2] = mx.concatenate([c2.kv_state[2][:, 1:], c2.kv_state[2][:, :1]], axis=1)       # swapped slots
    assert not same(c1.kv_state[2], c2.kv_state[2])
