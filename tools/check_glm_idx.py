"""deepseek_v32 Indexer (GLM) prefill replacement vs stock on random weights: top-k sets equal. python tests/test_glm_idx.py [T]"""
import sys, numpy as np, mlx.core as mx
from types import SimpleNamespace as NS
from mlx_vlm.models.deepseek_v32 import language as V
from mlx_vlm.models.base import create_attention_mask
from kyojin_mlx import ds_pf_idx as I
T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
a = NS(hidden_size=4096, index_n_heads=32, index_head_dim=128, qk_rope_head_dim=64, index_topk=512, q_lora_rank=1536, rope_theta=10000.0, max_position_embeddings=65536, rope_scaling=None)
ix = V.Indexer(a); mx.eval(ix.parameters())
rng = np.random.default_rng(0)
x = mx.array(rng.standard_normal((1, T, 4096)).astype(np.float16)); qr = mx.array(rng.standard_normal((1, T, 1536)).astype(np.float16))
mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))[None, None]
ix.set_dtype(mx.float16)
ref = V.Indexer.__call__(ix, x, qr, mask); new = I._impl_v32_indexer(V.Indexer.__call__)(ix, x, qr, mask)
r, n = np.sort(np.array(ref), -1), np.sort(np.array(new), -1)
# rows before the causal limit hold -inf ties: compare only rows with > topk valid keys
ok = [bool((r[0, 0, i] == n[0, 0, i]).all()) for i in range(a.index_topk + 8, T)]
print("T", T, "rows equal", sum(ok), "/", len(ok))
ov = np.mean([len(set(r[0, 0, i]) & set(n[0, 0, i])) / a.index_topk for i in range(a.index_topk + 8, T)])
print("mean top-k overlap", round(float(ov), 5))
