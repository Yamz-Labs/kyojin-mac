"""dsa_decode kernels through the fused decode indexer (32 index heads like DS4.1) against the stock `Indexer.__call__`: same indices (>= 99.5 % of the set,
exact in nearly every case) and the same candidate mask, with / without a candidate source / consumer, both dtypes, several ratios and key counts."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
L = pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
from types import SimpleNamespace                                            # noqa: E402

from mlx.utils import tree_map                                               # noqa: E402
from mlx_vlm.models.deepseek_v41.config import ModelConfig                   # noqa: E402
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0 as fq4   # noqa: E402

from kyojin_mlx import fuse_index as FI                                      # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
TOPK = 64


def make(ratio, cand_layer, dtype):
    cfg = ModelConfig()
    cfg.compress_ratios = [ratio, ratio]
    cfg.kv_source_layer_ids = [0]
    cfg.candidate_source_layer_id = cand_layer
    cfg.index_topk = TOPK
    cfg.candidate_topk_blocks = 40
    cfg.candidate_block_size = 8
    cfg.hidden_size = 256
    cfg.q_lora_rank = 128
    cfg.index_n_heads = 32
    cfg.index_head_dim = 128
    cfg.qk_rope_head_dim = 64
    mx.random.seed(11)
    idx = L.Indexer(cfg, 1)
    idx.update(tree_map(lambda p: (p * 8).astype(dtype), idx.parameters()))
    mx.eval(idx.parameters())
    return idx, cfg


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("cand_layer", [-1, 0, 1], ids=["plain", "consumer", "source"])
def test_dsa_indexer_matches_stock(ratio, cand_layer, dtype):
    idx, cfg = make(ratio, cand_layer, dtype)
    fused = FI._impl_indexer(L.Indexer.__call__)
    rng = np.random.default_rng(ratio * 10 + cand_layer)
    for start_pos in (ratio - 1, 9 * ratio + 1, 70 * ratio + ratio - 1, 600 * ratio + 3, 3001 * ratio + 1):
        n = (start_pos + 1) // ratio
        x = mx.array(rng.standard_normal((1, 1, cfg.hidden_size)).astype(np.float32)).astype(dtype)
        qr = mx.array(rng.standard_normal((1, 1, cfg.q_lora_rank)).astype(np.float32)).astype(dtype)
        keys = fq4(mx.array(rng.standard_normal((1, n + 11, cfg.index_head_dim)).astype(np.float32)).astype(dtype))
        cand = mx.array(np.repeat(rng.random((1, 1, -(-n // 8))) < 0.7, 8, axis=-1)[..., :n]) if cand_layer == 0 else None   # block-aligned like select_candidate_blocks
        out = {}
        for name, fn in (("stock", L.Indexer.__call__), ("fused", fused)):
            cache = SimpleNamespace(index_k=keys, candidates=cand)
            r = fn(idx, x, qr, None, start_pos, 0, cache)
            mx.eval(r)
            out[name] = (np.array(r), None if cache.candidates is None else np.array(cache.candidates))
        a, b = out["stock"][0].reshape(-1), out["fused"][0].reshape(-1)
        assert out["stock"][0].shape == out["fused"][0].shape == (1, 1, min(TOPK, n))
        assert len(set(a.tolist()) & set(b.tolist())) >= 0.995 * len(a), (ratio, start_pos)
        if cand_layer == 1:
            same = (out["stock"][1].reshape(-1) == out["fused"][1].reshape(-1)).mean()
            assert same >= 0.99, (ratio, start_pos, same)
