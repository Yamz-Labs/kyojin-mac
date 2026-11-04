# SPDX-License-Identifier: Apache-2.0
"""pfroof2: exact DS4.1 prefill, second roofline pass (flag KYOJIN_PFROOF2, default OFF, read at call time; sub-flag KYOJIN_PFROOF2_TOPK, default 1).

What this module holds:
  1. Prefill indexer, causal width (lever A). A query block of a prompt chunk only has finite scores in the columns n < clen of its LAST row,
     clen = (start + t0 + n_rows) // ratio. Every column from there on is -inf for every row of the block (the post kernel writes -INFINITY for n >= clen,
     the candidate mask can only add more -inf). The stock path scores, negates and argsorts all Tc columns anyway. Here a block uses the first W columns only,
         W = min(Tc, ceil64(max(clen_last, topk))).
     The output is the same, value for value: ordering is (score desc, index asc), the columns >= W are all -inf with an index above every column < W, and
     topk <= W, so the first topk of the full ordering lie inside [0, W). The scores of the columns < W are the same dot products (q . k is per column).
     At start = 0 (a fresh 8K prompt) the blocks average about 55 % of the columns: the score matmul, the post kernel and the sort shrink by that factor.
     The candidate-source layer keeps the stock full width (its block maxima and mask need every column).
  2. Optional exact radix top-k (`idxfast.topk_mt`, already bit-identical to the stock chain and used by the long-past path) instead of argsort + sort.
  3. A byte/flop budget model of one 8192-row chunk (`budget`), a transient-memory planner (`indexer_workspace_bytes`) and a CPU mirror of the score
     kernel (`ref_scores`) for the unit tests."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from . import fired as FD

ENV = "KYOJIN_PFROOF2"
ENV_TOPK = "KYOJIN_PFROOF2_TOPK"
COL_ALIGN = 64                                    # W is a multiple of this (aligned-N matmul path); the extra columns are masked, so no value changes


def enabled() -> bool:
    return os.environ.get(ENV, "0") not in ("0", "")


def topk_mt_enabled() -> bool:
    return os.environ.get(ENV_TOPK, "0") not in ("0", "")


def causal_width(start: int, t0: int, n: int, ratio: int, Tc: int, topk: int) -> int:
    """Columns [0, W) a block of rows [t0, t0 + n) of the chunk needs: every column at or above the last row's causal limit is -inf for the whole block."""
    last = (start + t0 + n) // ratio                                  # clen of the block's last row
    w = max(last, topk, 1)
    return min(Tc, -(-w // COL_ALIGN) * COL_ALIGN)


def select_topk(s: mx.array, k: int) -> mx.array:
    """[n, W] scores -> [n, k] int32 ascending, the stock `IT.topk`. On the GPU with KYOJIN_PFROOF2_TOPK=1 (off by default: slower than argsort at index ratio 4 on M5 Ultra): the exact multi-threadgroup radix select."""
    from . import idx_topk as IT
    if k >= s.shape[1]:                                               # every column is kept: the stock sort(argsort(-s)[:k]) is just 0 .. W-1 (first block of a fresh prompt, W == topk)
        return mx.broadcast_to(mx.arange(s.shape[1], dtype=mx.int32), (s.shape[0], s.shape[1]))
    if topk_mt_enabled() and k <= s.shape[1] and mx.metal.is_available() and mx.default_device() == mx.gpu:
        from . import idxfast as IX
        FD.hit("pfroof2_topk")
        return IX.topk_mt(s, mx.full((s.shape[0],), s.shape[1], dtype=mx.int32), k).astype(mx.int32)
    return IT.topk(s, k)


# ------------------------------------------------------------------------------------------------------------- CPU mirror of the score kernel
def ref_scores(q: np.ndarray, keys: np.ndarray, w: np.ndarray, start: int, ratio: int, t0: int = 0, n0: int = 0) -> np.ndarray:
    """What `ds_pf_idx.scores_range` computes, in numpy, in the kernel's order. q [T, NH, 128] fp16, keys [Tc, 128] fp16, w [T, NH] fp16 -> [T, Tc] float32.
    q . k in fp32 then rounded to half (the matmul product), relu in fp32, times the half head weight, heads summed h = 0..NH-1 in fp32, then the causal
    mask: column n + n0 >= clen(row) -> -inf, with clen = (start + t0 + t + 1) // ratio."""
    T, NH, _ = q.shape
    Tc = keys.shape[0]
    prod = np.einsum("thd,cd->thc", q.astype(np.float32), keys.astype(np.float32)).astype(np.float16).astype(np.float32)
    acc = np.zeros((T, Tc), dtype=np.float32)
    for h in range(NH):
        acc = (acc + np.maximum(prod[:, h], 0.0) * w[:, h:h + 1].astype(np.float32)).astype(np.float32)
    clen = ((start + t0 + np.arange(T) + 1) // ratio)[:, None]
    return np.where(np.arange(Tc)[None, :] + n0 < clen, acc, -np.inf).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------- transient memory planner
def indexer_workspace_bytes(T: int, Tc: int, topk: int, qblock: int, start: int = 0, ratio: int = 1, nh: int = 32, max_elems: int = 1 << 26,
                            restricted: bool = True) -> int:
    """Peak transient bytes of one indexer query block (matmul product half + scores fp32 + negation fp32 + argsort int32), worst block of a chunk of T rows.
    Depends on the block size and the key width only, never on T: a chunk of 8192, 16384 or 32768 rows at the same Tc has the same figure."""
    rows = T if (not qblock or qblock >= T) else qblock
    worst = 0
    for t0 in range(0, T, rows):
        n = min(rows, T - t0)
        W = causal_width(start, t0, n, ratio, Tc, topk) if restricted else Tc
        bt = max(1, min(n, max_elems // (nh * W)))                    # scores_range splits the queries so the half product stays under max_elems
        prod = bt * nh * W * 2
        worst = max(worst, prod + 3 * 4 * n * W)
    return worst


# ------------------------------------------------------------------------------------------------------------- budget model (labels: measured / model)
def budget(chunk: int = 8192, layers: int = 40, past: int = 0) -> list[dict]:
    """Rows of the op x time x roof table of one chunk. 'anchor' says where the time comes from; every 'roof' and 'gain' is a model (flops / peak, bytes / BW).
    Peaks used: 90 TFLOPS fp16 NAX dense (matmul roof), 1000 GB/s. `past` is the number of tokens already cached before the chunk."""
    TF, BW = 90e12, 1000e9
    T = chunk
    ffn = 495e6 * T * layers                                           # routed top-6 + shared experts, flops of the chunk, all layers
    dense = 253e6 * T * layers                                         # dense attention linears per layer x layers
    sparse = 687e9 * layers * (T / 8192)                               # S = q.k and P.v over 640 keys, 64 heads
    ratio_tc = [(2, 3), (1, 4), ]                                      # (ratio, count) of plain indexer layers (layer 20 is the candidate source)
    full_cols = 0.0
    rest_cols = 0.0
    for ratio, cnt in ratio_tc:
        Tc = (past + T) // ratio
        full_cols += cnt * Tc
        tot = 0
        for t0 in range(0, T, 512):
            tot += min(512, T - t0) * causal_width(past, t0, min(512, T - t0), ratio, Tc, 512)
        rest_cols += cnt * tot / T
    idx_full = 2 * 32 * 128 * T * full_cols
    idx_rest = 2 * 32 * 128 * T * rest_cols
    rows = [
        dict(op="routed+shared FFN GEMM (decode + NAX)", anchor="measured 3.2 s at ~50 TF (dsgap/ultraperf)", roof_s=ffn / TF, time_s=3.2),
        dict(op="dense attention linears (wq/wkv/wo)", anchor="measured 1.0-1.3 s (dsgap 1.3 s = 18 %; ultraperf 28 %: inconsistent, both kept)", roof_s=dense / TF,
             time_s=1.3),
        dict(op="sparse attention K1/K2/K3", anchor="measured 0.53 s (ultraperf)", roof_s=sparse / TF + layers * 4e9 / BW, time_s=0.53),
        dict(op="MoE rot_pf + mid_pf + final_pf", anchor="measured 0.25 s (rot)", roof_s=0.1, time_s=0.25),
        dict(op="HC glue (pf_glue fused)", anchor="measured ~1 s (dsgap) vs <0.05 s (pf_glue row): inconsistent", roof_s=0.12, time_s=1.0),
        dict(op="prefill indexer, 7 plain layers, stock", anchor="model (longctx linear fit, 8K from scratch)", roof_s=idx_full / TF, time_s=0.16),
        dict(op="prefill indexer, 7 plain layers, KYOJIN_PFROOF2", anchor="model: stock x columns kept", roof_s=idx_rest / TF,
             time_s=0.16 * (rest_cols / full_cols if full_cols else 1.0)),
    ]
    return rows
