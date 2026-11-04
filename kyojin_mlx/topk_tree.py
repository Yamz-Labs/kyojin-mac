# SPDX-License-Identifier: Apache-2.0
"""Chunk-tree exact top-k of the decode indexer (lever KYOJIN_TOPK_TREE): the select of a score row in ONE launch whatever the width.

`idxfast.topk_mt` / `select_fused` read the scores with a hist -> thr -> hist -> thr -> count -> prefix -> write chain (7 dependent launches for fp16, plus
two zero fills), each one a wide grid fed by device-memory atomics, so their time grows with the context and is mostly launch latency and atomic traffic
(LEVERS: 0.31 / 0.47 / 0.76 ms per call at 128K / 512K / 1M keys, N = 1). `idxblk` (dec_topk_blocks) cut the bytes but kept ~20 launches. This kernel:

  phase 1  one threadgroup per (row, chunk of CS keys, CS = 4096 at 262144 keys): the keys sit in registers, the chunk's K best (key desc, lower index first)
           are found with the same 16-step bitwise search as idxfuse (a threadgroup reduction per step, no atomics), compacted in ascending index order into
           a per-row candidate list as 31-bit words key16 << 15 | local index. A candidate-source layer also writes one word per 8-key block (block max key,
           finite flag).
  phase 2  every threadgroup counts itself on a per-row device counter (fence + atomic, the idxfuse ticket: nobody waits); the LAST one concatenates the chunk
           lists in chunk order (ascending global index, no holes), selects the K best of them with the same search, writes the indices ascending with the -1
           tail, and, for a candidate-source layer, runs the block level (kb - 1 best blocks, last block pinned, finite only) and expands it to the bool mask.

Exact, same indices / mask as `idxfast._run` (value desc, lower index on ties, -1 tail; mask on t < lens): the global top-K under the total order (key desc,
index asc) lies inside the union of the per-chunk top-Ks under the same order. The model of the whole algorithm is `topk_tree_model` (numpy, no mlx), which the
unit tests check against a brute-force sort; the kernel text is written against it and idxfuse's proven `_LEVEL` / ticket code. Runtime widths (lens, ip) as in
idxfast: the grid follows the capacity Tm, nothing inside a CS class is a template constant, so a width that moves every token builds no new pipeline.

Cost model (not measured): 1 launch, ~2 x 17 threadgroup-reduction steps on the critical path, the scores read once, 128 KB of candidates per row at most."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

from . import idxfuse as IXF
from . import topk_tree_model as M

MIN_KEYS = int(os.environ.get("KYOJIN_TOPK_TREE_MIN", "16384"))                # capacity from which the tree runs (the radix chain wins on short rows)
MAX_ROWS = 16                                                                  # decode / verify rows; a prefill tile (64-512 rows) keeps the radix select
MAX_K = 1024
MIN_K = 128
RB_MAX = 64                                                                    # block words per thread of the candidate-source variant (<= 524288 keys)
NT = M.NT

_BODY = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint Tm = ip[0];
    const uint NC = ip[1];
    const uint NB = ip[2];
    const uint u = tg / NC;
    const uint c = tg % NC;
    const uint L = uint(min(uint(max(lens[u], 0)), Tm));
    const uint c0 = c * @CS@;
    threadgroup uint red[32];
    threadgroup uint flag;
    threadgroup uint coff[@NCM@ + 1];
    if (c0 >= L && c != 0) return;                                            // chunks past the row's count do nothing (chunk 0 always counts: an empty row still gets its -1 tail)
    device atomic_uint* cand_a = ((device atomic_uint*)cand) + ulong(u) * (NC * @K@);
    device atomic_uint* cnts_a = ((device atomic_uint*)cnts) + ulong(u) * NC;
@BKDECL@
    const device T* sc = scores + ulong(u) * Tm;
    // ---- phase 1: this chunk's K best, ascending local index
    {
        const uint Lc = (L > c0) ? min(L - c0, uint(@CS@)) : 0u;
        uint kk[@R1@];
        bool sel[@R1@];
        for (uint r = 0; r < @R1@; ++r) {
            const uint i = tid * @R1@ + r;
            kk[r] = i < Lc ? IXF_KEY(sc[c0 + i]) : 0u;
        }
@LEVEL1@
        {
            uint cn = 0;
            for (uint r = 0; r < @R1@; ++r) cn += sel[r] ? 1u : 0u;
            uint total;
            uint pos = fz_excl(cn, red, sg, lane, total);
            for (uint r = 0; r < @R1@; ++r) {
                if (sel[r]) { atomic_store_explicit(cand_a + (c * @K@ + pos), (kk[r] << 15) | (tid * @R1@ + r), memory_order_relaxed); ++pos; }
            }
            if (tid == 0) atomic_store_explicit(cnts_a + c, total, memory_order_relaxed);
        }
@BLK1@
    }
    // ---- publish: this chunk's stores are visible on the device before it is counted
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint need = max(1u, (L + @CS@ - 1) / @CS@);
        device atomic_uint* ctr = ((device atomic_uint*)cnt) + u;
        const uint prev = atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
        const bool lastg = (prev + 1 == need);
        if (lastg) atomic_store_explicit(ctr, 0u, memory_order_relaxed);          // ready for the next call
        flag = lastg ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (flag == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    // ---- phase 2 (last chunk of the row): K best of the concatenated chunk lists
    const uint nact = max(1u, (L + @CS@ - 1) / @CS@);
    if (tid < nact) coff[tid + 1] = atomic_load_explicit(cnts_a + tid, memory_order_relaxed);
    if (tid == 0) coff[0] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { for (uint j = 1; j <= nact; ++j) coff[j] += coff[j - 1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nv = coff[nact];
    {
        uint kk[@R2@];
        uint gx[@R2@];
        bool sel[@R2@];
        const uint p0 = tid * @R2@;
        uint lo = 0, hi = nact;                                                // the chunk holding compact position p0: largest ch with coff[ch] <= p0
        while (hi - lo > 1) { const uint mid = (lo + hi) / 2; if (coff[mid] <= p0) lo = mid; else hi = mid; }
        uint ch = lo;
        for (uint r = 0; r < @R2@; ++r) {
            const uint p = p0 + r;
            kk[r] = 0u;
            gx[r] = 0u;
            if (p < nv) {
                while (ch + 1 < nact && coff[ch + 1] <= p) ++ch;
                const uint v = atomic_load_explicit(cand_a + (ch * @K@ + (p - coff[ch])), memory_order_relaxed);
                kk[r] = v >> 15;
                gx[r] = ch * @CS@ + (v & 0x7FFFu);
            }
        }
@LEVEL2@
        uint cn = 0;
        for (uint r = 0; r < @R2@; ++r) cn += sel[r] ? 1u : 0u;
        uint total;
        uint pos = fz_excl(cn, red, sg, lane, total);
        for (uint r = 0; r < @R2@; ++r) if (sel[r]) out[ulong(u) * @K@ + pos++] = int(gx[r]);
        for (uint i = total + tid; i < @K@; i += 1024) out[ulong(u) * @K@ + i] = -1;
    }
@BLK2@
"""

# candidate source, phase 1: one word per 8-key block of the chunk: bit 16 = the block max is finite (> -inf), low 16 bits = the order key of the max
_BLK1 = r"""
        for (uint lb = tid; lb < @CS@ / @BS@; lb += 1024) {
            const uint t0 = c0 + lb * @BS@;
            if (t0 >= L) break;
            const uint te = min(L, t0 + @BS@);
            float m = -INFINITY;
            for (uint t = t0; t < te; ++t) m = max(m, float(sc[t]));
            atomic_store_explicit(bk_a + (t0 / @BS@), (m > -INFINITY ? 0x10000u : 0u) | fz_key16(as_type<ushort>(T(m))), memory_order_relaxed);
        }
"""

# candidate source, phase 2: the idxfuse block level on the block words (last block pinned, finite only), expanded to the bool mask
_BLK2 = r"""
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nel = (L + @BS@ - 1) / @BS@;
        const uint nd = nel > 0 ? nel - 1 : 0;
        uint kb[@RB@];
        bool fb[@RB@];
        bool selb[@RB@];
        for (uint r = 0; r < @RB@; ++r) {
            const uint b = tid * @RB@ + r;
            const uint x = b < nd ? atomic_load_explicit(bk_a + b, memory_order_relaxed) : 0u;
            fb[r] = (x >> 16) != 0u;
            kb[r] = x & 0xFFFFu;
        }
@LEVELB@
        for (uint r = 0; r < @RB@; ++r) {
            const uint b = tid * @RB@ + r;
            if (b < nd) { const bool keep = selb[r] && fb[r]; const uint e = min(L, (b + 1) * @BS@); for (uint t = b * @BS@; t < e; ++t) mout[ulong(u) * Tm + t] = keep; }
        }
        if (tid == 0 && nel > 0) {
            const bool fin = (atomic_load_explicit(bk_a + (nel - 1), memory_order_relaxed) >> 16) != 0u;
            for (uint t = (nel - 1) * @BS@; t < L; ++t) mout[ulong(u) * Tm + t] = fin;
        }
    }
"""

_k: dict = {}


def _source(k: int, kb: int, bs: int, cs: int, r1: int, r2: int, ncm: int, rb: int) -> str:
    src = kb > 0

    def lvl(cnt: str, kd: int, rr: int, tag: str = "") -> str:
        s = IXF._LEVEL.replace("@CNT@", cnt).replace("@KD@", str(kd)).replace("@RR@", str(rr))
        return s.replace("kk[", "kb[").replace("sel[", "selb[") if tag == "b" else s

    s = _BODY.replace("@LEVEL1@", lvl("Lc", k, r1)).replace("@LEVEL2@", lvl("nv", k, r2))
    s = s.replace("@BKDECL@", "    device atomic_uint* bk_a = ((device atomic_uint*)bk) + ulong(u) * NB;" if src else "")
    s = s.replace("@BLK1@", _BLK1 if src else "").replace("@BLK2@", _BLK2.replace("@LEVELB@", lvl("nd", kb - 1, rb, "b")) if src else "")
    return (s.replace("@CS@", str(cs)).replace("@NCM@", str(ncm)).replace("@K@", str(k)).replace("@R1@", str(r1)).replace("@R2@", str(r2))
             .replace("@RB@", str(rb)).replace("@BS@", str(bs)))


_HDR = IXF._HEADER + r"""
#define IXF_KEY(v) fz_key16(as_type<ushort>(v))
"""


def applies(N: int, Tm: int, k: int, dt, topk_blocks: int = 0, bs: int = 8) -> bool:
    """True when the tree can serve this call (the caller checks the flag, idxfast.TOPK_TREE): fp16 / bf16 scores, few rows, a wide row, a k the candidate list holds. topk_blocks > 0 = the
    candidate-source variant (mask), which needs the block words of the whole row in registers (<= RB_MAX per thread)."""
    if dt not in (mx.float16, mx.bfloat16) or N > MAX_ROWS or Tm < MIN_KEYS or not (MIN_K <= k <= MAX_K):
        return False
    if topk_blocks and (topk_blocks < 2 or bs not in (4, 8, 16)):
        return False
    try:
        cs, nc, r1, r2, ncm, rb = M.plan(Tm, k, bs)
    except ValueError:
        return False
    return not topk_blocks or rb <= RB_MAX


def topk_tree(scores: mx.array, lens: mx.array, k: int, topk_blocks: int = 0, bs: int = 8):
    """(idx [N, k] int32 ascending with a -1 tail, mask [N, Tm] bool | None): the contract of idxfast._run (kb = topk_blocks, mask only when kb > 0)."""
    N, Tm = scores.shape
    FD.hit("topk_tree")
    cs, nc, r1, r2, ncm, rb = M.plan(Tm, k, bs)
    src = topk_blocks > 0
    nb = -(-Tm // bs)
    name = f"kyojin_topk_tree_{k}_{topk_blocks}_{bs}_{cs}"
    ker = _k.get(name)
    if ker is None:
        ker = _k[name] = mx.fast.metal_kernel(name=name, input_names=["scores", "lens", "ip", "cnt"], output_names=["out", "mout", "cand", "cnts", "bk"],
                                              source=_source(k, topk_blocks, bs, cs, r1, r2, ncm, rb), header=_HDR)
    out, mout, _, _, _ = ker(
        inputs=[scores, lens.astype(mx.int32), IXF.C._u32((Tm, nc, nb)), IXF._counter(N)],
        template=[("T", scores.dtype)], grid=(N * nc * NT, 1, 1), threadgroup=(NT, 1, 1),
        output_shapes=[(N, k), (N, Tm) if src else (1,), (N, nc * k), (N, nc), (N, nb) if src else (1,)],
        output_dtypes=[mx.int32, mx.bool_, mx.uint32, mx.uint32, mx.uint32])
    return out, (mout if src else None)
