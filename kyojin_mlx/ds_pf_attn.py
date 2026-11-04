# SPDX-License-Identifier: Apache-2.0
"""DS4.1 / GLM sparse attention for prefill (S > 1, batch 1): replaces `DeepseekV41Attention._sparse_attention`.

Stock path: float32 everywhere, the scores of every query against the whole local window array (T x T per head) and a gather of
[T, topk, D] pooled rows for all queries at once, then the window and pooled parts in separate softmax pieces.

Here every query owns ONE key list of `window + topk` row ids into the source [window rows; pooled rows] (-1 = none: not yet visible / not selected),
so the window band and the selected pooled rows are the same thing: queries go in blocks of BQ, per block

    gather   G[q, n, :] = source[ids[q, n]]                       (one kernel, 4 rows per 64-lane group)
    scores   S[q] = Qs[q] (H x D)  @  G[q]^T                       (batched half matmul, MLX)
    softmax  P[q, h, :] = softmax([S, sink])  masked by ids >= 0   (one kernel, one simdgroup per (q, h) row held in registers, normalized, half)
    output   O[q] = P[q] @ G[q]                                    (batched half matmul, MLX)

Nothing of size T x T or T x H x Tc is built, no float32 copy of the gathered rows, no combine pass. Scores and weights are half (fp32 accumulate inside the
matmuls): relative rms difference to the stock path about 6e-4 (the stock path rounds the outputs to half too).

Switch: module attribute ENABLED (env KYOJIN_DS_PF=0). Block size: env KYOJIN_DS_PF_BQ (default 128)."""
from __future__ import annotations

from . import archs as _A
import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_DS_PF", "1") != "0"
BQ = int(os.environ.get("KYOJIN_DS_PF_BQ", "128"))
NAX = os.environ.get("KYOJIN_DS_PF_NAX", "1") != "0"     # fp16 prompts on NAX devices: attn_prefill's gather-fused kernels (2x faster at T = 8192, closer to stock)
MIN_T = 32

_GATHER_SRC = """
    uint tid = thread_position_in_grid.x;           // 64 threads per row (one 16-byte chunk each); a group of 64 lanes copies 4 rows
    uint c = tid % 64, rg = tid / 64;
    const uint D = prm[0], nrow = prm[1];
    for (uint r = 0; r < 4; ++r) {
        uint row = rg * 4 + r;
        if (row >= nrow) break;
        int s = ids[row];
        if (s < 0) s = 0;
        ((device uint4*)(G + ulong(row) * D))[c] = ((const device uint4*)(src + ulong(s) * D))[c];
    }
"""

_SOFT_SRC = """
    // one simdgroup per (q, h) row of NK scores, 8 rows per threadgroup: the row is read once into registers (NV values per lane),
    // softmax over [valid keys | sink], weights written normalized in half
    const uint lane = thread_index_in_simdgroup;
    const uint row = thread_position_in_grid.y;      // q * H + h
    const uint NK = prm[0], H = prm[1], NR = prm[2];
    if (row >= NR) return;
    const uint q = row / H, h = row % H;
    const ulong base = ulong(row) * NK;
    const device int* idq = ids + ulong(q) * NK;
    const float sk = sink[h];
    float v[NV];
    float m = sk;
    for (uint i = 0; i < NV; ++i) {
        uint n = lane + i * 32;
        v[i] = (n < NK && idq[n] >= 0) ? float(S[base + n]) : -INFINITY;
        m = max(m, v[i]);
    }
    m = simd_max(m);
    float den = 0.0f;
    for (uint i = 0; i < NV; ++i) { v[i] = exp(v[i] - m); den += v[i]; }
    const float inv = 1.0f / (simd_sum(den) + exp(sk - m));
    for (uint i = 0; i < NV; ++i) {
        uint n = lane + i * 32;
        if (n < NK) P[base + n] = half(v[i] * inv);
    }
"""
_kern: dict = {}


def _prm(*v):
    return mx.array(list(v) + [0] * (16 - len(v)), dtype=mx.uint32)      # >= 16 entries: passed as a device pointer


def gather_rows(src, ids):
    """src [N, D] half, ids [BQ, NK] int32 (-1 -> row 0, masked later) -> [BQ, NK, D]."""
    BQ_, NK = ids.shape
    D = src.shape[-1]
    k = _kern.get("g")
    if k is None:
        k = _kern["g"] = mx.fast.metal_kernel(name="kyojin_dspf_gather", input_names=["src", "ids", "prm"], output_names=["G"], source=_GATHER_SRC)
    return k(inputs=[src, ids.reshape(-1), _prm(D, BQ_ * NK)], grid=((D // 8) * (-(-BQ_ * NK // 4)), 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(BQ_, NK, D)], output_dtypes=[src.dtype])[0]


def softmax_rows(S, ids, sink, H):
    """S [BQ, H, NK] half scores, ids [BQ, NK] -> P [BQ, H, NK] half (softmax with the sink in the denominator, invalid keys 0)."""
    BQ_, _, NK = S.shape
    k = _kern.get("s")
    if k is None:
        k = _kern["s"] = mx.fast.metal_kernel(name="kyojin_dspf_softmax", input_names=["S", "ids", "sink", "prm"], output_names=["P"], source=_SOFT_SRC)
    NR = BQ_ * H
    return k(inputs=[S, ids.reshape(-1), sink, _prm(NK, H, NR)], template=[("NV", -(-NK // 32))], grid=(32, -(-NR // 8) * 8, 1), threadgroup=(32, 8, 1),
             output_shapes=[S.shape], output_dtypes=[mx.float16])[0]


def key_ids(T: int, Tp: int, Wn: int, Tc: int, idxs, win: int):
    """[Tp, win + K] int32 row ids into concat([window (Wn rows), pool (Tc rows)]): window rows j = rel + i - win + 1 + c (>= 0), pool rows idx + Wn."""
    rel = Wn - T
    i = mx.arange(Tp)[:, None]
    j = rel + i - (win - 1) + mx.arange(win)[None, :]
    wid = mx.where((j >= 0) & (i < T), j, -1).astype(mx.int32)
    pid = mx.where(idxs[0] >= 0, idxs[0] + Wn, -1).astype(mx.int32)
    if Tp != T:
        pid = mx.pad(pid, [(0, Tp - T), (0, 0)], constant_values=-1)
    return mx.concatenate([wid, pid], axis=1)


def sparse_pf(q, window, pool, idxs, wmask, scale, sink, win):
    """q [1, H, T, D]; window [1, Wn, D] (history + this chunk); pool [1, Tc, D]; idxs [1, T, K] (-1 = none); sink [H]. Returns [1, H, T, D]."""
    _, H, T, D = q.shape
    Wn, Tc = window.shape[1], pool.shape[1]
    nb = -(-T // BQ)
    Tp = nb * BQ
    dt = q.dtype
    qs = (q[0] * scale).astype(dt).transpose(1, 0, 2)                          # [T, H, D]
    if Tp != T:
        qs = mx.pad(qs, [(0, Tp - T), (0, 0), (0, 0)])
    src = mx.concatenate([window[0].astype(dt), pool[0].astype(dt)], axis=0)
    ids = key_ids(T, Tp, Wn, Tc, idxs, win)
    sink32 = sink.astype(mx.float32)
    outs = []
    for b in range(nb):
        ib = ids[b * BQ:(b + 1) * BQ]
        g = gather_rows(src, ib)                                                # [BQ, NK, D]
        s = qs[b * BQ:(b + 1) * BQ] @ g.transpose(0, 2, 1)                      # [BQ, H, NK]
        outs.append(softmax_rows(s, ib, sink32, H) @ g)                         # [BQ, H, D]
    out = mx.concatenate(outs, axis=0) if nb > 1 else outs[0]
    return out[:T].transpose(1, 0, 2)[None]


def _impl_sparse(orig):
    def _sparse_attention(self, q, window, pool, indices, window_mask):
        if not (ENABLED and q.shape[0] == 1 and q.shape[2] >= MIN_T and q.dtype in (mx.float16, mx.bfloat16)):
            return orig(self, q, window, pool, indices, window_mask)
        if NAX and indices.ndim == 3 and indices.shape[1] == q.shape[2] and window.shape[1] >= q.shape[2]:
            from . import attn_prefill as _ap
            if _ap.supported(q, window, pool):
                return _ap.sparse_attention(q, window, pool, indices, self.scale, self.attn_sink, self.window_size)
        return sparse_pf(q, window, pool, indices, window_mask, self.scale, self.attn_sink, self.window_size)
    return _sparse_attention


_done: set = set()


def install(model) -> int:
    """Replace `_sparse_attention` of the DS4.1 attention class (once per process). Returns 1 when installed."""
    if not ENABLED:
        return 0
    try:
        from mlx_vlm.models.deepseek_v32 import language as v32
        from . import ds_pf_idx as _pi
        if type(getattr(model.language_model.layers[0], "self_attn", None)).__module__.startswith("mlx_vlm.models.glm_moe_dsa") and not getattr(v32.Indexer, "_kyojin_pf", False):
            v32.Indexer.__call__ = _pi._impl_v32_indexer(v32.Indexer.__call__); v32.Indexer._kyojin_pf = True
    except Exception:
        pass
    cls = type(model.language_model.layers[0].attn) if hasattr(model.language_model, "layers") and hasattr(model.language_model.layers[0], "attn") else None
    if cls is None or cls.__name__ != _A.vendor().attn or cls in _done:
        return 0
    cls._sparse_attention = _impl_sparse(cls._sparse_attention)
    from . import ds_pf_idx
    mod = __import__(cls.__module__, fromlist=["Indexer"])
    mod.Indexer.__call__ = ds_pf_idx._impl_indexer(mod.Indexer.__call__)
    from . import ds_pf_mhc
    ds_pf_mhc.install(mod, getattr(mod, _A.vendor().block))
    _done.add(cls)
    return 1
