#!/usr/bin/env python3
"""Compressor prefill pooling, stock per-token loop against kyojin_mlx/compressor_pf.py, at the real DS4.1 sizes (hidden 5120, head_dim 512, ratio 2).

usage: compressor_bench.py [--rows 1024,2048,4096] [--ratio 2] [--reps 3]
Per row count: ms of one chunk that starts at a non-zero position (the case the stock loop handles token by token), stock and vectorized, and whether
the pooled values and the final state are bit for bit equal. A DS4.1 prefill has 3 to 4 such compressors per chunk (KV source layers of the encoder half)."""
import argparse
import os
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mlx_vlm.models.deepseek_v41.language as L  # noqa: E402
from mlx_vlm.models.deepseek_v41.config import ModelConfig  # noqa: E402
from kyojin_mlx import compressor_pf as PF  # noqa: E402
from kyojin_mlx.ced import _override  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--rows", default="1024,2048,4096")
ap.add_argument("--ratio", type=int, default=2)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
H, D = 5120, 512
cfg = ModelConfig.from_dict(dict(
    model_type="deepseek_v41", vocab_size=96, hidden_size=H, moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=64, num_key_value_heads=1,
    head_dim=D, qk_rope_head_dim=64, q_lora_rank=32, o_lora_rank=32, o_groups=2, rms_norm_eps=1e-20, n_routed_experts=8, n_shared_experts=1,
    num_experts_per_tok=2, sliding_window=128, compress_ratios=[a.ratio, a.ratio], kv_source_layer_ids=[0], index_source_layer_ids=[0],
    candidate_source_layer_id=0, index_n_heads=4, index_head_dim=128, index_topk=24, candidate_topk_blocks=12, candidate_block_size=4, hc_mult=2,
    engram_layer_ids=[], engram_num_embeddings=[]))
ref, new = L.Compressor(cfg, 0), L.Compressor(cfg, 0)
mx.random.seed(0)
w = [(k, (mx.random.normal(v.shape) * 0.02).astype(v.dtype)) for k, v in tree_flatten(ref.parameters())]
ref.update(tree_unflatten(w))
new.update(tree_unflatten(w))
mx.eval(ref.parameters(), new.parameters())
_override(new, "__call__", PF._pool_call)


def timed(comp, x, start, cache):
    t0 = time.perf_counter()
    out = comp(x, start, cache)
    mx.eval(out, cache.kv_state[0], cache.score_state[0])
    return time.perf_counter() - t0, out


for n in [int(v) for v in a.rows.split(",")]:
    x = (mx.random.normal((1, n, H)) * 0.5).astype(mx.bfloat16)
    start = 65                                      # odd: the chunk starts inside an open group
    res = {}
    for name, comp in (("stock", ref), ("vector", new)):
        ts = []
        for r in range(a.reps + 1):
            cache = L.DeepseekV41Cache(2)
            comp(x[:, :65], 0, cache)               # a first chunk with an open group
            mx.eval(cache.kv_state[0])
            t, out = timed(comp, x, start, cache)
            ts.append(t)
        res[name] = (min(ts[1:]), out, cache)
    eq = bool(mx.array_equal(res["stock"][1], res["vector"][1]).item()) and all(
        bool(mx.array_equal(p, q).item()) for p, q in ((res["stock"][2].kv_state[0], res["vector"][2].kv_state[0]),
                                                      (res["stock"][2].score_state[0], res["vector"][2].score_state[0])))
    s, v = res["stock"][0], res["vector"][0]
    print(f"{n:5d} rows ratio {a.ratio}: stock loop {1000 * s:7.1f} ms ({1e6 * s / n:5.1f} us/row)   vectorized {1000 * v:6.1f} ms   x{s / v:5.1f}   bit exact: {eq}", flush=True)
