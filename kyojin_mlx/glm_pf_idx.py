# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x indexer pool selection for prompt chunks on the DS4.1 prefill indexer kernels (fused NAX score kernel + multi-threadgroup exact radix top-k).

Stock `_exact_pool_select` (mlx_vlm glm5_next), per block of 512 query rows: scores = relu(q.f32 @ pool_keys.f32^T) * (weights * scale) summed over the 32 heads as
[512, 32, pools] fp32 temporaries (written and read back), a masked copy, `argpartition` of the whole row. At 128K tokens (32768 pools) that is ~2 GB of fp32 traffic per block.
Here (batch 1, fp16, more pools than select_k):

    scores   ds_pf_idx's fused kernel (one query row per threadgroup, 32 heads x 32 keys matmul2d tiles, the product never leaves the threadgroup) with two changes for
             GLM: the product stays fp32 (the stock GLM matmul is fp32; the DS4.1 kernel rounds it to half) and the head weights are the fp32 `weights * scale` of the stock
             code. The causal / validity mask is the stock one (`candidates`), applied on the [n, pools] scores.
    top-k    idxfast.topk_mt: exact (value desc, lower pool first on ties) multi-threadgroup radix select, instead of argpartition on the whole row.

Same selected set as the stock selection up to the order of the fp32 accumulation inside a score (the set can differ only for pools whose scores tie within ~1e-6).

Switch: KYOJIN_GLM_PF_IDX (flag glm_pf_idx, default off until measured)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import archs as _A
from . import ds_pf_idx as DI
from . import fired as FD
from . import metal_nax as NX

ENABLED = os.environ.get("KYOJIN_GLM_PF_IDX", "0") != "0"
MIN_ROWS = 64
VIEW = os.environ.get("KYOJIN_DSA_INDEXER_PF_VIEW", "0") != "0"      # reshape views instead of `a[0]` (a Gather = a full copy: q 67 MB + pool keys per 512-row block, 2 launches x 176 per chunk)
BLOCK = int(os.environ.get("KYOJIN_GLM_PF_IDX_BLOCK", "512"))
_BIG = 0x00FFFFFF

_SRC = DI._FSCORE
assert _SRC.count("float(half(cT[i]))") == 1 and _SRC.count("Wh[tid] = float(w[ulong(t) * 32 + tid]);") == 1, "ds_pf_idx._FSCORE changed"
_SRC = _SRC.replace("float(half(cT[i]))", "cT[i]")
_kern: dict = {}


def scores(q, keys, w, t0: int, n: int):
    """[n, Tc] fp32 = sum_h w[t, h] * relu(q[t, h] . keys[c]) for query rows [t0, t0 + n); q [1, T, 32, 128] fp16, keys [1, Tc, 128] fp16, w [1, T, 32] fp32. No mask."""
    Tc = keys.shape[1]
    k = _kern.get("f")
    if k is None:
        k = _kern["f"] = mx.fast.metal_kernel(name="kyojin_glm_idx_fscore", header=NX.HEADER_NAX, input_names=["Q", "K", "w", "prm"], output_names=["S"], source=_SRC)
    prm = mx.array([Tc, _BIG, 1, t0, 0, DI.FSCORE_TPT] + [0] * 10, dtype=mx.uint32)      # start = _BIG, ratio 1: clen far above every column (no causal mask here)
    if VIEW:
        qs, ks, ws = q.reshape(-1, 32, 128)[t0:t0 + n].reshape(-1, 128), keys.reshape(Tc, 128), w.reshape(-1, 32)[t0:t0 + n]
    else:
        qs, ks, ws = q[0][t0:t0 + n].reshape(-1, 128), keys[0], w[0][t0:t0 + n]
    return k(inputs=[qs, ks, ws, prm], grid=(n * 128, -(-Tc // (128 * DI.FSCORE_TPT)), 1), threadgroup=(128, 1, 1),
             output_shapes=[(n, Tc)], output_dtypes=[mx.float32])[0]


def why_not(q, pool_keys, weights, select_k):
    """None when the fused prompt selection serves this call, else the reason (flag audit key; the guards of `supported`, same order)."""
    if not ENABLED:
        return "flag_off"
    if q is None or weights is None or q.ndim != 4:
        return "q_none"
    if q.shape[1] < MIN_ROWS:
        return f"rows<{MIN_ROWS}"
    if q.shape[0] != 1:
        return f"batch={q.shape[0]}"
    if q.shape[2] != 32 or q.shape[3] != 128:
        return "heads_dim"
    if q.dtype != mx.float16 or pool_keys.dtype != mx.float16:
        return f"dtype q={str(q.dtype).split('.')[-1]} keys={str(pool_keys.dtype).split('.')[-1]}"
    if pool_keys.shape[0] != 1 or pool_keys.shape[1] < 128:
        return "few_pools"
    if select_k >= pool_keys.shape[1]:
        return "select_k>=pools"
    if weights.dtype != mx.float32:
        return "weights_dtype=" + str(weights.dtype).split(".")[-1]
    if not DI.METAL_OPT or not NX.nax_available():
        return "no_nax"
    return None


def supported(q, pool_keys, weights, select_k) -> bool:
    return why_not(q, pool_keys, weights, select_k) is None


def select(q, pool_keys, weights, pool_ends, pool_valid, query_positions, select_k, scale, block: int | None = None):
    """(selected [1, T, select_k] int32, valid [1, T, select_k] bool): the stock `_exact_pool_select` result for batch 1."""
    from . import idxfast as IX
    T = query_positions.shape[0]
    pc = pool_keys.shape[1]
    w = weights * scale
    bs = block or BLOCK
    sel_chunks, val_chunks = [], []
    lens = None
    for a in range(0, T, bs):
        n = min(bs, T - a)
        cand = pool_valid[:, None] & (pool_ends[:, None] <= query_positions[None, a:a + n, None])        # [1, n, pc]
        sc = scores(q, pool_keys, w, a, n)
        sc = mx.where(cand.reshape(n, pc) if VIEW else cand[0], sc, -mx.inf)
        if lens is None or lens.shape[0] != n:
            lens = mx.full((n,), pc, dtype=mx.int32)
        sel = IX.topk_mt(sc, lens, select_k).astype(mx.int32)                                          # [n, k] ascending
        sel_chunks.append(sel[None])
        val_chunks.append(mx.take_along_axis(cand, sel[None], axis=-1))
    if len(sel_chunks) == 1:
        return sel_chunks[0], val_chunks[0]
    return mx.concatenate(sel_chunks, axis=1), mx.concatenate(val_chunks, axis=1)


def install(model) -> int:
    """Wrap the module-level `_exact_pool_select` of glm5_next.language (once per process). 1 when installed."""
    if not ENABLED:
        return 0
    try:
        L = _A.lang(_A.GLM5)
    except Exception:
        return 0
    layers = getattr(getattr(model, "language_model", None), "layers", None) or []
    if not any(type(getattr(ly, "self_attn", None)).__name__ == "Glm5NextAttention" for ly in layers):
        return 0
    if getattr(L, "_kyojin_pf_idx", False):
        return 1
    stock = L._exact_pool_select

    def patched(q, pool_keys, weights, pool_ends, pool_valid, query_positions, select_k, scale, chunk_size=512):
        why = why_not(q, pool_keys, weights, select_k)
        if why is not None:
            FD.fb("pf_idx", why)
            return stock(q, pool_keys, weights, pool_ends, pool_valid, query_positions, select_k, scale, chunk_size)
        FD.fire("pf_idx")
        return select(q, pool_keys, weights, pool_ends, pool_valid, query_positions, select_k, scale)

    L._exact_pool_select = patched
    L._kyojin_pf_idx = True
    return 1
