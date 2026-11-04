# SPDX-License-Identifier: Apache-2.0
"""Fast exact top-k for any DSA-style index scorer (DS4.1, GLM): multi-threadgroup histogram top-k with the candidate mask fused in.

The stock selection (mx.argsort/mx.sort/where, the dsa_decode single-threadgroup radix select, dsv4_core index_topk_rt / candidate_mask_rt) scans the
scores with one threadgroup: latency- and grid-bound (~0.2-1.3 ms), 1 of 80 GPU cores busy. These kernels do the same selection on a wide grid:

    hist   (wide)    per (row, chunk) partial 8-bit histograms of the order-preserving score keys, top byte first;
                    the fused variant histograms the block-max keys in the same read of the scores (one block per thread)
    thr    (tiny)    reduce the partials over the chunks, extend the K-th key prefix by one byte
    ...             one (hist, thr) pair per key byte (2 for fp16/bf16, 4 for fp32)
    count  (wide)   per (row, chunk) counts of keys strictly above / equal to the K-th key
    prefix (tiny)   exclusive prefix of the counts over the chunks of a row (+ the ties still needed, the totals)
    write  (wide)   compact the selected indices in ascending order; the fused variant expands the kept blocks to the bool mask

Inputs are runtime width: scores [N, Tm] over the capacity-sized buffer, lens [N] int32 (valid keys per row); only t < lens[u] is read and the grid
depends on the capacity, never on the position, so a recorded verify region stays valid (same contract as index_topk_rt / candidate_mask_rt).
Exactness: same order-preserving keys (fp16/bf16 16-bit with -0/+0 tied, fp32 32-bit), same K-th-key threshold, same tie rule (lower index first),
same pinned last block and finite-only mask as the stock chain: the index set (and the mask on t < lens) is bit-identical to index_topk_rt /
candidate_mask_rt, which equal the stock sort/argsort chain.

Chunking: CS = 256 * BS items per chunk (one BS block per thread), NC = ceil(Tm / CS) chunks; the grid is N * NC threadgroups of 256 threads.

Switch: KYOJIN_IDX_FAST=1 (default off until the real-pack gate).
"""
from __future__ import annotations

import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_IDX_FAST", "1") not in ("0", "")
TOPK_TREE = os.environ.get("KYOJIN_TOPK_TREE", "0") not in ("0", "")           # chunk-tree exact top-k in one launch (topk_tree), set by the flag table
DEC_TOPK_BLOCKS = os.environ.get("KYOJIN_DEC_TOPK_BLOCKS", "0") not in ("0", "")   # D3: candidate-block exact top-k (idxblk), set by the flag table

_HDR = """using namespace metal;
inline uint idxf_key16(ushort b) {
    if ((b & 0x7FFF) == 0) b = 0;
    return (b & 0x8000) ? uint(ushort(~b)) : uint(b | 0x8000);
}
inline uint idxf_key32(uint b) {
    if (b == 0x80000000u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
inline uint idxf_ekey(const device half* s, uint i) { return idxf_key16(as_type<ushort>(s[i])); }
inline uint idxf_ekey(const device bfloat16_t* s, uint i) { return idxf_key16(as_type<ushort>(s[i])); }
inline uint idxf_ekey(const device float* s, uint i) { return idxf_key32(as_type<uint>(s[i])); }
inline uint idxf_bkey16(float m) { return idxf_key16(as_type<ushort>(half(m))); }
inline uint idxf_bkey32(float m) { return idxf_key32(as_type<uint>(m)); }
"""

# Partial 8-bit histograms of pass P over the chunks of every row. FUSE = 1 also histograms the block-max keys (BS keys per block, the last
# block of a row excluded from the block partials and pinned to the top by the consumers). Thread i of a threadgroup owns block b0 + i and reads
# its BS scores once, feeding the element partials and, fused, the block partials (second half of `part`) from that single read.
_HIST = r"""
    const uint Tm = dims[0];
    const uint NC = dims[1];
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint u = tg / NC;
    const uint c = tg % NC;
    const uint L = min(uint(max(lens[u], 0)), uint(Tm));
    const device T* sc = scores + ulong(u) * Tm;
    threadgroup atomic_uint pe[256];
    threadgroup atomic_uint pbf[256];
    for (uint b = tid; b < 256u; b += 256u) { atomic_store_explicit(&pe[b], 0u, memory_order_relaxed); if (FUSE) atomic_store_explicit(&pbf[b], 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint shift = 8u * (NP - 1u - PS);
    const uint nel = (L + BS - 1) / BS;
    const uint lastb = (L - 1) / BS;
    const uint b0 = c * 256, b1 = min(nel, b0 + 256u);
    const uint bb = b0 + tid;
    if (bb < b1) {
        const uint t0 = bb * BS;
        const uint te = min(L, t0 + BS);
        float bm = -INFINITY;
        for (uint t = t0; t < te; ++t) {
            bm = max(bm, float(sc[t]));
            const uint ek = idxf_ekey(sc, t);
            if (PS == 0 || (ek >> (shift + 8u)) == st[ulong(u) * 4 + 0]) atomic_fetch_add_explicit(&pe[(ek >> shift) & 255u], 1u, memory_order_relaxed);
        }
        if (FUSE && bb != lastb) {
            const uint bk = (KW == 16) ? idxf_bkey16(bm) : idxf_bkey32(bm);
            if (PS == 0 || (bk >> (shift + 8u)) == st[ulong(u) * 4 + 2]) atomic_fetch_add_explicit(&pbf[(bk >> shift) & 255u], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = tid; b < 256u; b += 256u) {
        part[(ulong(u) * NC + c) * 512 + b] = atomic_load_explicit(&pe[b], memory_order_relaxed);
        if (FUSE) part[(ulong(u) * NC + c) * 512 + 256 + b] = atomic_load_explicit(&pbf[b], memory_order_relaxed);
    }
"""

# Reduce one row's partials (NC x 256) and extend the K-th key prefix by the next byte.
# above[u] = keys strictly above pre[u] so far; Kd is min(K, count) for the element (KDE) or block (KDB) side.
_THRESH = r"""
    const uint Tm = dims[0];
    const uint NC = dims[1];
    const uint u = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    uint cole = 0, colb = 0;
    for (uint c = 0; c < NC; ++c) {
        cole += part[(ulong(u) * NC + c) * 512 + tid];
        if (FUSE) colb += part[(ulong(u) * NC + c) * 512 + 256 + tid];
    }
    threadgroup uint rede[256], redb[256];
    rede[tid] = cole;
    if (FUSE) redb[tid] = colb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint L = min(uint(max(lens[u], 0)), uint(Tm));
        const uint nel = (L + BS - 1) / BS;
        uint need = (KDE < L ? KDE : L) - st[ulong(u) * 4 + 1];
        uint cum = 0, b = 255;
        for (;;) { uint h = rede[b]; if (cum + h >= need || b == 0) break; cum += h; --b; }
        stn[ulong(u) * 4 + 0] = (st[ulong(u) * 4 + 0] << 8) | b;
        stn[ulong(u) * 4 + 1] = st[ulong(u) * 4 + 1] + cum;
        if (FUSE) {
            const uint nd = nel > 0 ? nel - 1 : 0;
            uint needb = (KDB < nd ? KDB : nd) - st[ulong(u) * 4 + 3];
            uint cum2 = 0, b2 = 255;
            for (;;) { uint h = redb[b2]; if (cum2 + h >= needb || b2 == 0) break; cum2 += h; --b2; }
            stn[ulong(u) * 4 + 2] = (st[ulong(u) * 4 + 2] << 8) | b2;
            stn[ulong(u) * 4 + 3] = st[ulong(u) * 4 + 3] + cum2;
        }
    }
"""

# Per (row, chunk) counts of keys strictly above / equal to the K-th key (elements; FUSE: and blocks, last block excluded like the stock nd).
_COUNT = r"""
    const uint Tm = dims[0];
    const uint NC = dims[1];
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint u = tg / NC;
    const uint c = tg % NC;
    const uint L = min(uint(max(lens[u], 0)), uint(Tm));
    const device T* sc = scores + ulong(u) * Tm;
    const uint nel = (L + BS - 1) / BS;
    const uint nd = nel > 0 ? nel - 1 : 0;
    const uint lastb = (L - 1) / BS;
    const uint T0 = st[ulong(u) * 4 + 0];
    const uint T0b = st[ulong(u) * 4 + 2];
    const bool alle = L <= KDE;
    const bool allb = nd <= KDB;
    const uint b0 = c * 256, b1 = min(nel, b0 + 256u);
    const uint bb = b0 + tid;
    uint gt = 0, eq = 0, gtb = 0, eqb = 0;
    if (bb < b1) {
        const uint t0 = bb * BS;
        const uint te = min(L, t0 + BS);
        float bm = -INFINITY;
        for (uint t = t0; t < te; ++t) {
            bm = max(bm, float(sc[t]));
            if (alle) { gt += 1; continue; }
            const uint ek = idxf_ekey(sc, t);
            gt += (ek > T0); eq += (ek == T0);
        }
        if (FUSE && bb != lastb && bb < nd) {
            if (allb) { gtb += 1; }
            else {
                const uint bk = (KW == 16) ? idxf_bkey16(bm) : idxf_bkey32(bm);
                gtb += (bk > T0b); eqb += (bk == T0b);
            }
        }
    }
    threadgroup uint sh[4][256];
    sh[0][tid] = gt; sh[1][tid] = eq; sh[2][tid] = gtb; sh[3][tid] = eqb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (uint i = 0; i < 256; ++i) { s0 += sh[0][i]; s1 += sh[1][i]; s2 += sh[2][i]; s3 += sh[3][i]; }
        cnt[(ulong(u) * NC + c) * 4 + 0] = s0; cnt[(ulong(u) * NC + c) * 4 + 1] = s1;
        cnt[(ulong(u) * NC + c) * 4 + 2] = s2; cnt[(ulong(u) * NC + c) * 4 + 3] = s3;
    }
"""

# Exclusive prefix of the counts over a row's chunks: OFF[u, c] = (selected elements before c, equal elements before c,
# greater blocks before c, equal blocks before c); TAKE[u] = (element ties still needed, block ties still needed, totals).
_PREFIX = r"""
    const uint Tm = dims[0];
    const uint NC = dims[1];
    const uint u = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint L = min(uint(max(lens[u], 0)), uint(Tm));
    const uint nel = (L + BS - 1) / BS;
    const uint nd = nel > 0 ? nel - 1 : 0;
    const uint Ke = KDE < L ? KDE : L;
    const uint Kb = FUSE ? (KDB < nd ? KDB : nd) : 0;
    if (tid == 0) {
        uint gte = 0, gtbe = 0;
        for (uint c = 0; c < NC; ++c) {
            gte += cnt[(ulong(u) * NC + c) * 4 + 0];
            if (FUSE) gtbe += cnt[(ulong(u) * NC + c) * 4 + 2];
        }
        const uint needE = Ke > gte ? Ke - gte : 0;
        const uint needB = FUSE ? (Kb > gtbe ? Kb - gtbe : 0) : 0;
        uint se = 0, ee = 0, gb = 0, eb = 0;
        for (uint c = 0; c < NC; ++c) {
            const uint gt = cnt[(ulong(u) * NC + c) * 4 + 0];
            const uint eq = cnt[(ulong(u) * NC + c) * 4 + 1];
            const uint g2 = cnt[(ulong(u) * NC + c) * 4 + 2];
            const uint e2 = cnt[(ulong(u) * NC + c) * 4 + 3];
            off[(ulong(u) * NC + c) * 4 + 0] = se; off[(ulong(u) * NC + c) * 4 + 1] = ee;
            off[(ulong(u) * NC + c) * 4 + 2] = gb; off[(ulong(u) * NC + c) * 4 + 3] = eb;
            const uint rem = needE > ee ? needE - ee : 0;
            const uint take = eq < rem ? eq : rem;
            se += gt + take;
            ee += take;
            if (FUSE) {
                const uint remb = needB > eb ? needB - eb : 0;
                const uint takeb = e2 < remb ? e2 : remb;
                gb += g2 + takeb;
                eb += takeb;
            }
        }
        take[ulong(u) * 4 + 0] = needE;
        take[ulong(u) * 4 + 1] = needB;
        take[ulong(u) * 4 + 2] = se;
        take[ulong(u) * 4 + 3] = gb;
    }
"""

# Compact in ascending index order: elements to out[u, 0..K-1] (-1 tail from chunk 0); FUSE: kept blocks expanded to the bool mask
# (last block: finite only). Item (tid, j) of a chunk is global index c * CS + tid * BS + j: ascending in (tid, j).
_WRITE = r"""
    const uint Tm = dims[0];
    const uint NC = dims[1];
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint u = tg / NC;
    const uint c = tg % NC;
    const uint L = min(uint(max(lens[u], 0)), uint(Tm));
    const device T* sc = scores + ulong(u) * Tm;
    const uint nel = (L + BS - 1) / BS;
    const uint lastb = (L - 1) / BS;
    const uint nd = nel > 0 ? nel - 1 : 0;
    const uint T0 = st[ulong(u) * 4 + 0];
    const uint T0b = st[ulong(u) * 4 + 2];
    const uint b0 = c * 256, b1 = min(nel, b0 + 256u);
    const uint off0 = off[(ulong(u) * NC + c) * 4 + 0];
    const uint eq0 = off[(ulong(u) * NC + c) * 4 + 1];
    const uint gtb0 = off[(ulong(u) * NC + c) * 4 + 2];
    const uint eqb0 = off[(ulong(u) * NC + c) * 4 + 3];
    const uint needE = take[ulong(u) * 4 + 0];
    const uint needB = take[ulong(u) * 4 + 1];
    const uint totE = take[ulong(u) * 4 + 2];
    const bool alle = L <= KDE;
    const bool allb = FUSE && nd <= KDB;
    threadgroup uint pse[256], eqe[256], gbb[256], ebb[256];
    uint eqi = 0, seli = 0, gbi = 0, ebi = 0;
    float bmax = -INFINITY;
    const uint bb = b0 + tid;
    if (bb < b1) {
        const uint t0 = bb * BS;
        const uint te = min(L, t0 + BS);
        for (uint t = t0; t < te; ++t) {
            bmax = max(bmax, float(sc[t]));
            if (alle) { seli += 1; continue; }
            const uint ek = idxf_ekey(sc, t);
            eqi += (ek == T0);
        }
        if (FUSE) {
            if (allb) { gbi = 1; }
            else {
                const uint bk = (KW == 16) ? idxf_bkey16(bmax) : idxf_bkey32(bmax);
                gbi = (bk > T0b); ebi = (bk == T0b);
            }
        }
    }
    pse[tid] = 0; eqe[tid] = eqi; gbb[tid] = gbi; ebb[tid] = ebi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint ebe = eq0;
    for (uint i = 0; i < tid; ++i) ebe += eqe[i];
    uint ebprep = eqb0;
    for (uint i = 0; i < tid; ++i) ebprep += ebb[i];
    if (!alle && bb < b1) {
        const uint t0 = bb * BS;
        const uint te = min(L, t0 + BS);
        uint ein = 0;
        for (uint t = t0; t < te; ++t) {
            const uint ek = idxf_ekey(sc, t);
            if (ek > T0) seli += 1;
            else if (ek == T0) { if (ebe + (ein++) < needE) seli += 1; }
        }
    }
    pse[tid] = seli;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = off0;
    for (uint i = 0; i < tid; ++i) base += pse[i];
    if (bb < b1) {
        const uint t0 = bb * BS;
        const uint te = min(L, t0 + BS);
        if (alle) {
            for (uint t = t0; t < te; ++t) out[ulong(u) * K + base + (t - t0)] = int(t);
        } else {
            uint ein = 0, s = 0;
            for (uint t = t0; t < te; ++t) {
                const uint ek = idxf_ekey(sc, t);
                bool sel = ek > T0;
                if (ek == T0) sel = (ebe + (ein++) < needE);
                if (sel) out[ulong(u) * K + base + (s++)] = int(t);
            }
        }
    }
    if (FUSE) {
        if (bb == lastb && bb < nel) {
            for (uint t = bb * BS; t < L; ++t) mout[ulong(u) * Tm + t] = (bmax > -INFINITY);
        } else if (bb < b1 && bb != lastb && bb < nd) {
            bool keep = allb || (gbi == 1) || (ebi == 1 && (ebprep < needB));
            keep = keep && (bmax > -INFINITY);
            const uint t0 = bb * BS;
            const uint te = min(L, t0 + BS);
            for (uint t = t0; t < te; ++t) mout[ulong(u) * Tm + t] = keep;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0 && c == 0) {
        for (uint i = totE; i < uint(K); ++i) out[ulong(u) * K + i] = -1;
    }
"""

_k: dict = {}


def _dts(dt) -> str:
    return {mx.float16: "f16", mx.bfloat16: "bf16", mx.float32: "f32"}[dt]


def _kernel(name: str, inputs: list, outputs: list, source: str):
    k = _k.get(name)
    if k is None:
        k = _k[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source, header=_HDR)
    return k


def _run(scores: mx.array, lens: mx.array, k: int, kb: int, bs: int, fuse: bool):
    """Shared engine: exact top-k (and, fused, the candidate mask) of scores [N, Tm] with runtime widths lens [N].
    k = elements kept, kb = blocks kept (fused only), bs = block size. Returns (idx [N, k] int32, mask [N, Tm] bool or None)."""
    N, Tm = scores.shape
    assert lens.shape == (N,) and k >= 1 and (not fuse or kb >= 2)
    dt = scores.dtype
    assert dt in (mx.float16, mx.bfloat16, mx.float32), dt
    kw = 16 if dt != mx.float32 else 32
    npass = kw // 8
    assert Tm >= 1
    nc = max(1, -(-Tm // (256 * bs)))
    assert bs >= 1 and 256 * bs >= 1
    tmpl_c = {"BS": bs, "KW": kw, "KDE": k, "KDB": kb - 1 if fuse else 1, "FUSE": fuse}
    dims = mx.array([Tm, nc], dtype=mx.uint32)              # runtime widths: a template width would build ~7 new pipelines (~200 ms) per new Tm
    ncb = nc
    part = mx.zeros((N * ncb, 512), dtype=mx.uint32)
    st = mx.zeros((N, 4), dtype=mx.uint32)
    for p in range(npass):
        kh = _kernel(f"kyojin_idxf_hist_{_dts(dt)}_{int(fuse)}", ["scores", "lens", "st", "dims"], ["part"], _HIST)
        part = kh(inputs=[scores, lens, st, dims], template=[("T", dt), ("PS", p), ("NP", npass), *tmpl_c.items()],
                  grid=(N * ncb * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N * ncb, 512)],
                  output_dtypes=[mx.uint32])[0]
        kt = _kernel("kyojin_idxf_thr", ["part", "lens", "st", "dims"], ["stn"], _THRESH)
        st = kt(inputs=[part, lens, st, dims], template=[*tmpl_c.items()],
                grid=(N * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, 4)],
                output_dtypes=[mx.uint32])[0]
    kc = _kernel(f"kyojin_idxf_count_{_dts(dt)}_{int(fuse)}", ["scores", "lens", "st", "dims"], ["cnt"], _COUNT)
    cnt = kc(inputs=[scores, lens, st, dims], template=[("T", dt), *tmpl_c.items()],
             grid=(N * ncb * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N * ncb, 4)],
             output_dtypes=[mx.uint32])[0]
    kp = _kernel("kyojin_idxf_prefix", ["cnt", "lens", "dims"], ["off", "take"], _PREFIX)
    off, take = kp(inputs=[cnt, lens, dims], template=[*tmpl_c.items()],
                    grid=(N * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(N * ncb, 4), (N, 4)],
                    output_dtypes=[mx.uint32, mx.uint32])
    kwr = _kernel(f"kyojin_idxf_write_{_dts(dt)}_{int(fuse)}", ["scores", "lens", "st", "off", "take", "dims"], ["out", "mout"], _WRITE)
    mshape = (N, Tm) if fuse else (1,)
    out, mout = kwr(inputs=[scores, lens, st, off, take, dims],
                    template=[("T", dt), ("K", k), *tmpl_c.items()],
                    grid=(N * ncb * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, k), mshape],
                    output_dtypes=[mx.int32, mx.bool_])
    return out, (mout if fuse else None)


def topk_mt(scores: mx.array, lens: mx.array, k: int) -> mx.array:
    """The `k` best columns t < lens[u] of scores [N, Tm], int32 [N, k] ascending with a -1 tail: bit-identical to
    dsv4_core.index_topk_rt (and the stock sort/argsort chain), on a multi-threadgroup grid."""
    if TOPK_TREE:
        from . import topk_tree as TT
        if TT.applies(scores.shape[0], scores.shape[1], k, scores.dtype):
            return TT.topk_tree(scores, lens, k)[0]
    if DEC_TOPK_BLOCKS:
        from . import idxblk
        if idxblk.applies(scores.shape[0], scores.shape[1], k):
            return idxblk.topk_blocks(scores, lens, k)
    out, _ = _run(scores, lens, k, 2, 8, False)
    return out


def select_fused(scores: mx.array, lens: mx.array, k: int, topk_blocks: int, block_size: int = 8):
    """Fused exact selection for a candidate-source indexer layer: (idx [N, k] int32, mask [N, Tm] bool), bit-identical to
    (index_topk_rt, candidate_mask_rt). Any block_size with 256 * block_size covering a chunk (one block per thread)."""
    if TOPK_TREE:
        from . import topk_tree as TT
        if TT.applies(scores.shape[0], scores.shape[1], k, scores.dtype, topk_blocks, block_size):
            return TT.topk_tree(scores, lens, k, topk_blocks, block_size)
    return _run(scores, lens, k, topk_blocks, block_size, True)
