"""Decode indexer cut (kyojin_mlx.fuse_index) against the stock `Indexer.__call__`, on a synthetic indexer (random weights, no checkpoint): same indices and same
candidate mask to the bit, for every compression ratio a DS4.1 layer can have (1 = the last 20 layers of the real model, 2, 4), with and without a candidate source / a
candidate consumer, fewer keys than the top-k and more, both dtypes."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
L = pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
from types import SimpleNamespace                                            # noqa: E402

from mlx.utils import tree_map                                               # noqa: E402
from mlx_vlm.models.deepseek_v41.config import ModelConfig                   # noqa: E402

from kyojin_mlx import fuse_index as FI                                      # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
TOPK = 16


def make(ratio, cand_layer, dtype):
    cfg = ModelConfig()
    cfg.compress_ratios = [ratio, ratio]
    cfg.kv_source_layer_ids = [0]
    cfg.candidate_source_layer_id = cand_layer
    cfg.index_topk = TOPK
    cfg.candidate_topk_blocks = 6
    cfg.candidate_block_size = 4
    cfg.hidden_size = 256
    cfg.q_lora_rank = 128
    cfg.index_n_heads = 8
    cfg.index_head_dim = 128
    cfg.qk_rope_head_dim = 64
    mx.random.seed(11)
    idx = L.Indexer(cfg, 1)
    idx.update(tree_map(lambda p: (p * 8).astype(dtype), idx.parameters()))
    mx.eval(idx.parameters())
    return idx, cfg


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("cand_layer", [-1, 0, 1], ids=["plain", "consumer", "source"])
def test_decode_indexer_bit_identical(ratio, cand_layer, dtype):
    idx, cfg = make(ratio, cand_layer, dtype)
    fused = FI._impl_indexer(L.Indexer.__call__)
    rng = np.random.default_rng(ratio * 10 + cand_layer)
    for start_pos in (ratio - 1, 3 * ratio + ratio - 1, 9 * ratio + 1, 40 * ratio + ratio - 1, 41 * ratio, 200 * ratio + 3):   # n = 1, 4, 9, 41, 41, 200 keys
        n = (start_pos + 1) // ratio
        x = mx.array(rng.standard_normal((1, 1, cfg.hidden_size)).astype(np.float32)).astype(dtype)
        qr = mx.array(rng.standard_normal((1, 1, cfg.q_lora_rank)).astype(np.float32)).astype(dtype)
        keys = mx.array(rng.standard_normal((1, 400, cfg.index_head_dim)).astype(np.float32)).astype(dtype)
        cand = mx.array(rng.random((1, 1, n)) < 0.7) if cand_layer == 0 else None
        out = {}
        for name, fn in (("stock", L.Indexer.__call__), ("fused", fused)):
            cache = SimpleNamespace(index_k=keys, candidates=cand)
            r = fn(idx, x, qr, None, start_pos, 0, cache)
            mx.eval(r)
            out[name] = (np.array(r), None if cache.candidates is None else np.array(cache.candidates))
        assert out["stock"][0].shape == out["fused"][0].shape == (1, 1, min(TOPK, n))
        assert np.array_equal(out["stock"][0], out["fused"][0]), (ratio, start_pos)
        if cand_layer == 1:
            assert np.array_equal(out["stock"][1], out["fused"][1]), (ratio, start_pos)


def test_fused_indexer_defers_to_stock_off_the_decode_shape():
    """Prefill chunks (several tokens) and a disabled switch run the stock code."""
    idx, cfg = make(2, -1, mx.bfloat16)
    calls = []

    def spy(self, *a):
        calls.append(a[0].shape)
        return "stock"
    fused = FI._impl_indexer(spy)
    cache = SimpleNamespace(index_k=mx.zeros((1, 50, cfg.index_head_dim), dtype=mx.bfloat16), candidates=None)
    x4 = mx.zeros((1, 4, cfg.hidden_size), dtype=mx.bfloat16)
    assert fused(idx, x4, None, None, 10, 0, cache) == "stock" and calls == [(1, 4, cfg.hidden_size)]
    old, FI.ENABLED = FI.ENABLED, False
    try:
        assert fused(idx, x4[:, :1], None, None, 10, 0, cache) == "stock"
    finally:
        FI.ENABLED = old
