# SPDX-License-Identifier: Apache-2.0
"""Decode indexer at depth in ONE launch: scores + exact chunk-tree top-k (+ candidate mask) (flag KYOJIN_CIDXFUSE, default 0).

Served chain at depth is scores (iscore kernel, width = capacity) -> select (topk_tree or the 7-launch idxfast chain) -> candidate mask, three or more
dependent launches with a [N, capacity] score buffer between them. `idxfuse` fuses all three but keeps the scores of the row for one last threadgroup, so it
stops at 32768 row-keys; `topk_tree` selects wide rows in one launch but reads a score buffer. This kernel is the join of the two:

  phase 0  each (row, chunk) threadgroup scores ITS chunk of CS keys straight into threadgroup memory: lane = head, simdgroup = KPS keys, the same 128-step fp32
           FMA chain per head and the same head sum in order 0..NH-1 as `idxfuse.scores_fast` / `dsv4_core._ISCORE`, then T(total). A key past lens, or dropped
           by the candidate mask, is T(-inf). No score buffer in device memory, and a chunk past lens[u] does nothing: the work follows the row's count, not the capacity.
  phase 1, 2  the text of `topk_tree._BODY` unchanged except that the chunk's scores are read from threadgroup memory (per-chunk K best, last-arriver ticket,
           K best of the chunk lists, -1 tail, block level + bool mask for a candidate-source layer).

Exact: the indices and the mask equal `topk_tree` / `idxfast._run` applied to the scores of `idxfuse.scores_fast` (and so to `index_scores(rt=True)`), value desc,
lower index first on ties, -1 tail. They are NOT claimed equal to a selection over the matrix-unit scores of `dsa_decode.scores_rows` (iscore_mma): that kernel
sums in another order. Gate: fp16 / bf16, <= topk_tree.MAX_ROWS rows, keys >= topk_tree.MIN_KEYS, chunk <= MAX_CS keys (threadgroup memory), else the old chain."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD
from . import fp4pool as FP
from . import idxfuse as IXF
from . import topk_tree as TT
from . import topk_tree_model as M

ENABLED = os.environ.get("KYOJIN_CIDXFUSE", "0") not in ("0", "")
KPS = IXF.KPS
MAX_CS = 8192                                                                  # chunk scores live in threadgroup memory (2 B each); with q (8 KB) and the chunk table it stays under 32 KB
TG_BYTES = 32768

_SCORE_STAGE = r"""
    threadgroup T qs[NH * DI];
    threadgroup T scs[@CS@];
    const uint Ck = ip[3];
    const uint Tc = ip[4];
    for (uint i = tid; i < NH * DI; i += 1024) qs[(i % DI) * NH + i / DI] = q[u * NH * DI + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        constexpr uint KPT = 32 * KPS;
        const uint Lcs = (L > c0) ? min(L - c0, uint(@CS@)) : 0u;
        const float wv = lane < NH ? float(wts[u * NH + lane]) : 0.0f;
        for (uint base = 0; base < Lcs; base += KPT) {
            const uint tl = base + sg * KPS;
            if (tl >= Lcs) continue;
            const uint t0 = c0 + tl;
            bool ok[KPS];
            bool any = false;
            for (uint j = 0; j < KPS; ++j) {
                bool o = tl + j < Lcs;
                if (CAND && o) o = cmask[ulong(u) * Tc + t0 + j] != 0;
                ok[j] = o;
                any = any || o;
            }
            float acc[KPS];
            for (uint j = 0; j < KPS; ++j) acc[j] = 0.0f;
            if (any) {
                const device T* kp = keys + (ulong(sess[u]) * Ck + t0) * DI;
                for (uint dc = 0; dc < DI; dc += 8) {
                    float kf[KPS][8];
                    for (uint j = 0; j < KPS; ++j) for (uint i = 0; i < 8; ++i) kf[j][i] = ok[j] ? float(kp[j * DI + dc + i]) : 0.0f;
                    for (uint i = 0; i < 8; ++i) {
                        const float qv = lane < NH ? float(qs[(dc + i) * NH + lane]) : 0.0f;
                        for (uint j = 0; j < KPS; ++j) acc[j] = fma(qv, kf[j][i], acc[j]);
                    }
                }
            }
            for (uint j = 0; j < KPS; ++j) {
                const T tile = T(acc[j]);
                const T rel = tile > T(0) ? tile : T(0);
                const float pr = float(T(float(rel) * wv));
                float total = 0.0f;
                for (uint h = 0; h < NH; ++h) total += simd_shuffle(pr, ushort(h));
                if (lane == 0 && tl + j < Lcs) scs[tl + j] = ok[j] ? T(total) : T(-INFINITY);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_SC_DECL = "    const device T* sc = scores + ulong(u) * Tm;\n"

_k: dict = {}


def _source(k: int, kb: int, bs: int, cs: int, r1: int, r2: int, ncm: int, rb: int) -> str:
    """topk_tree's kernel text with the chunk's scores computed in threadgroup memory instead of read from a buffer."""
    s = TT._source(k, kb, bs, cs, r1, r2, ncm, rb)
    assert s.count(_SC_DECL) == 1 and s.count("sc[c0 + i]") == 1 and s.count("sc[t]") == (1 if kb > 0 else 0)
    s = s.replace("sc[c0 + i]", "scs[i]").replace("sc[t]", "scs[t - c0]")
    return s.replace(_SC_DECL, _SCORE_STAGE.replace("@CS@", str(cs)))


def smem_bytes(nh: int, di: int, cs: int, ncm: int) -> int:
    return 2 * nh * di + 2 * cs + 4 * (ncm + 1) + 4 * 32 + 4


def applies(N: int, Tm: int, k: int, dt, nh: int, di: int, topk_blocks: int = 0, bs: int = 8) -> bool:
    """True when the fused kernel can serve the call (the caller checks ENABLED): the tree's own gate, the score kernel's geometry, the chunk in threadgroup memory."""
    if dt not in (mx.float16, mx.bfloat16) or not IXF.scores_supported(nh, di, dt) or not TT.applies(N, Tm, k, dt, topk_blocks, bs):
        return False
    cs, nc, r1, r2, ncm, rb = M.plan(Tm, k, bs)
    return cs <= MAX_CS and smem_bytes(nh, di, cs, ncm) <= TG_BYTES


def fused(q, keys, wts, lens, t_max, k, cand=None, sess=None, topk_blocks=0, bs=8):
    """(idx [N, k] int32 ascending, -1 tail; mask [N, t_max] bool | None): the contract of topk_tree.topk_tree over the scores of
    idxfuse.scores_fast(q, keys, wts, lens, t_max, cand, sess). q [N, NH, DI], keys [S, C, DI] (or a stackpack stack), wts [N, NH], lens [N] int32."""
    N, NH, DI = q.shape
    assert applies(N, t_max, k, q.dtype, NH, DI, topk_blocks, bs) and q.dtype == keys.dtype == wts.dtype and t_max <= keys.shape[1]
    FD.hit("cidxfuse")
    cs, nc, r1, r2, ncm, rb = M.plan(t_max, k, bs)
    src = topk_blocks > 0
    nb = -(-t_max // bs)
    name = f"kyojin_cidxfuse_{k}_{topk_blocks}_{bs}_{cs}"
    pk = hasattr(keys, "codes")
    ker = _k.get(name + "_pk" if pk else name)
    names = ["q", "kcodes", "kscales", "wts", "lens", "cmask", "sess", "ip", "cnt"] if pk else ["q", "keys", "wts", "lens", "cmask", "sess", "ip", "cnt"]
    if ker is None:
        text = _source(k, topk_blocks, bs, cs, r1, r2, ncm, rb)
        hdr = TT._HDR
        if pk:
            text, hdr = IXF.pack_source(text), hdr + IXF._PK_HELPER
        ker = _k[name + "_pk" if pk else name] = mx.fast.metal_kernel(name=name + ("_pk" if pk else ""), input_names=names,
                                                                      output_names=["out", "mout", "cand", "cnts", "bk"], source=text, header=hdr)
    sess = IXF.C._identity(N) if sess is None else sess
    c = cand if cand is not None else mx.zeros((1,), dtype=mx.bool_)
    tc = cand.shape[-1] if cand is not None else 1
    assert cand is None or tc >= t_max
    kin = [keys.codes, keys.scales] if pk else [keys]
    out, mout, _, _, _ = ker(
        inputs=[q, *kin, wts, lens.astype(mx.int32), c, sess, IXF.C._u32((t_max, nc, nb, FP.stride(keys), tc)), IXF._counter(N)],
        template=[("T", q.dtype), ("NH", NH), ("DI", DI), ("KPS", KPS), ("CAND", cand is not None)],
        grid=(N * nc * M.NT, 1, 1), threadgroup=(M.NT, 1, 1),
        output_shapes=[(N, k), (N, t_max) if src else (1,), (N, nc * k), (N, nc), (N, nb) if src else (1,)],
        output_dtypes=[mx.int32, mx.bool_, mx.uint32, mx.uint32, mx.uint32])
    return out, (mout if src else None)
