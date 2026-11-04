# SPDX-License-Identifier: Apache-2.0
"""Indexer decode chain in ONE launch: scores + exact top-k + candidate mask (KYOJIN_IDX_FUSE=1, default off).

The stock rt chain is three dependent launches per indexer layer (dsv4_core.index_scores, index_topk_rt, candidate_mask_rt). Both halves are latency
bound at decode (one thread per key runs a 4096-FMA chain; one threadgroup radix-selects with contended threadgroup atomics). This kernel:

  1. scores: a simdgroup works on KPS keys, one lane per HEAD. A lane runs the same 128-step fp32 FMA chain per head as the stock thread (d = 0 .. 127 from 0),
     the head terms are summed in head order 0 .. 31 from 0 through simd_shuffle: bit-identical scores, 32x shorter dependent chain.
  2. the threadgroup stores its scores, then counts itself on a per-row device counter (fence + atomic). The LAST threadgroup of a row runs the select; nobody
     waits (no spin, no deadlock whatever the scheduling order). The counter is a persistent buffer the last threadgroup zeroes again.
  3. select (1024 threads, keys in registers, no atomics): the K-th key is found by a 16-step bitwise search with threadgroup reductions, ties go to the lower
     index (contiguous ownership + exclusive scans), indices are written in increasing order, -1 tail. For a candidate-source layer the block level
     (block max, last block pinned, finite only) follows in the same threadgroup.

Same contract as the rt kernels: runtime width (lens), columns >= lens never read, grid depends on the capacity only. Result == (index_scores(rt=True) ->
index_topk_rt [-> candidate_mask_rt]) bit for bit (tests/test_idxfuse.py). Keys in registers bound the width: the caller passes lmax >= every lens
(rounded to a power-of-two class), up to MAX_KEYS.
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import dsv4_core as C
from . import fp4pool as FP

ENABLED = os.environ.get("KYOJIN_IDX_FUSE", "1") not in ("0", "")
MAX_KEYS = 65536                                                     # keys-in-registers bound of the select (the caller gates rows x keys with KYOJIN_IDX_FUSE_MAX)
KPS = 4                                                              # keys per simdgroup (x 32 simdgroups = 64 keys per threadgroup)
NT = 1024

# ----------------------------------------------------------------------------------------------------------------- score-only kernel
# The score half of _BODY without the select: a simdgroup owns KPS keys, one lane per head, the same 128-step fp32 FMA chain per head
# (d = 0 .. DI - 1 from 0) and the same head sum in increasing head order 0 .. NH-1 as dsv4_core._ISCORE, so the scores are bit-identical.
# It exists for the widths where the in-launch select would lose (the last threadgroup walks every key from registers): there the fused
# kernel is gated off and this scores for the normal exact select. One threadgroup = 32 * KPS keys; NT threads.
_SCORES = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint Tm = ip[0];
    const uint Ck = ip[1];
    const uint Tc = ip[2];
    constexpr uint KPT = 32 * KPS;
    const uint nb = (Tm + KPT - 1) / KPT;
    const uint u = tg / nb;
    const uint gi = tg % nb;
    const uint L = uint(min(uint(max(lens[u], 0)), Tm));
    threadgroup T qs[NH * DI];
    for (uint i = tid; i < NH * DI; i += NT) qs[(i % DI) * NH + i / DI] = q[u * NH * DI + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t0 = gi * KPT + sg * KPS;
    if (t0 >= Tm) return;
    bool ok[KPS];
    bool any = false;
    for (uint j = 0; j < KPS; ++j) {
        const uint t = t0 + j;
        bool o = t < Tm && int(t) < int(L);
        if (CAND && o) o = cand[ulong(u) * Tc + t] != 0;
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
                const float qv = float(qs[(dc + i) * NH + lane]);
                for (uint j = 0; j < KPS; ++j) acc[j] = fma(qv, kf[j][i], acc[j]);
            }
        }
    }
    const float wv = float(wts[u * NH + lane]);
    for (uint j = 0; j < KPS; ++j) {
        const T tile = T(acc[j]);
        const T rel = tile > T(0) ? tile : T(0);
        const float pr = float(T(float(rel) * wv));
        float total = 0.0f;
        for (uint h = 0; h < NH; ++h) total += simd_shuffle(pr, ushort(h));
        const uint t = t0 + j;
        if (lane == 0 && t < Tm) out[ulong(u) * Tm + t] = ok[j] ? T(total) : T(-INFINITY);
    }
"""

_k: dict = {}
_cnt: dict = {}


def scores_supported(nh: int, di: int, dt) -> bool:
    """Geometry of the score-only kernel: one lane per head (nh <= 32), 8-wide channel steps, query tile in threadgroup memory."""
    return 0 < nh <= 32 and di % 8 == 0 and dt in (mx.float16, mx.bfloat16, mx.float32) and nh * di * (4 if dt == mx.float32 else 2) <= 24 * 1024


def scores_fast(q, keys, wts, lens, t_max, cand=None, sess=None) -> mx.array:
    """Head-weighted rectified index scores [N, t_max] in q's dtype, bit-identical to dsv4_core.index_scores(rt=True), on a lane-per-head
    grid. Same contract: -inf where t >= lens[u] or the candidate mask drops the key; columns >= t_max are not written."""
    N, NH, DI = q.shape
    assert scores_supported(NH, DI, q.dtype) and q.dtype == keys.dtype == wts.dtype and t_max <= keys.shape[1]
    name = "kyojin_idxscore_fast_" + {mx.float16: "f16", mx.bfloat16: "bf16", mx.float32: "f32"}[q.dtype]
    pk = hasattr(keys, "codes")                                         # stackpack: keys is a stackpack.PackedStack, loaded from the codes
    ker = _k.get(name + "_pk" if pk else name)
    if ker is None and pk:
        ker = _k[name + "_pk"] = mx.fast.metal_kernel(name=name + "_pk", input_names=["q", "kcodes", "kscales", "wts", "lens", "cand", "sess", "ip"],
                                                      output_names=["out"], source=pack_source(_SCORES), header=_HEADER + _PK_HELPER)
    elif ker is None:
        ker = _k[name] = mx.fast.metal_kernel(name=name, input_names=["q", "keys", "wts", "lens", "cand", "sess", "ip"],
                                              output_names=["out"], source=_SCORES, header=_HEADER)
    sess = C._identity(N) if sess is None else sess
    c = cand if cand is not None else mx.zeros((1,), dtype=mx.bool_)
    tc = cand.shape[-1] if cand is not None else 1
    assert cand is None or tc >= t_max
    nt = 1024
    kin = [keys.codes, keys.scales] if pk else [keys]
    return ker(inputs=[q, *kin, wts, lens, c, sess, C._u32((t_max, FP.stride(keys), tc))],
               template=[("T", q.dtype), ("NH", NH), ("DI", DI), ("KPS", KPS), ("NT", nt), ("CAND", cand is not None)],
               grid=(N * (-(-t_max // (32 * KPS))) * nt, 1, 1), threadgroup=(nt, 1, 1),
               output_shapes=[(N, t_max)], output_dtypes=[q.dtype])[0]

_HEADER = C._CONTRACT_OFF + r"""
inline uint fz_key16(ushort b) {
    if ((b & 0x7FFF) == 0) b = 0;
    return (b & 0x8000) ? uint(ushort(~b)) : uint(b | 0x8000);
}
inline uint fz_sum(uint v, threadgroup uint* red, uint sg, uint lane) {
    const uint s = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return simd_sum(red[lane]);
}
inline uint fz_excl(uint v, threadgroup uint* red, uint sg, uint lane, thread uint& total) {
    const uint p = simd_prefix_exclusive_sum(v);
    const uint s = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint off = 0;
    total = 0;
    for (uint j = 0; j < 32; ++j) { const uint x = red[j]; off += (j < sg) ? x : 0u; total += x; }
    return off + p;
}
"""

# One selection level. ITEMS (live values in `kk[]`, `RR` per thread, contiguous ownership) are keys of the `cnt` valid elements, `Kd` of them are wanted.
# Fills sel[] (bool per owned element) = in the K best (key desc, lower index first on ties).
_LEVEL = r"""
    {
        const uint all_ = (@CNT@) <= (@KD@);
        uint thr_ = 0, needeq_ = 0;
        if (!all_) {
            for (int b_ = 15; b_ >= 0; --b_) {
                const uint try_ = thr_ | (1u << uint(b_));
                uint c_ = 0;
                for (uint r_ = 0; r_ < @RR@; ++r_) c_ += (tid * @RR@ + r_ < (@CNT@) && kk[r_] >= try_) ? 1u : 0u;
                if (fz_sum(c_, red, sg, lane) >= (@KD@)) thr_ = try_;
            }
            uint g_ = 0, e_ = 0;
            for (uint r_ = 0; r_ < @RR@; ++r_) { const bool v_ = tid * @RR@ + r_ < (@CNT@); g_ += (v_ && kk[r_] > thr_) ? 1u : 0u; e_ += (v_ && kk[r_] == thr_) ? 1u : 0u; }
            const uint gt_ = fz_sum(g_, red, sg, lane);
            needeq_ = (@KD@) - gt_;
            uint tot_;
            const uint pref_ = fz_excl(e_, red, sg, lane, tot_);
            uint take_ = (pref_ >= needeq_) ? 0u : min(e_, needeq_ - pref_);
            for (uint r_ = 0; r_ < @RR@; ++r_) {
                const bool v_ = tid * @RR@ + r_ < (@CNT@);
                bool s_ = v_ && kk[r_] > thr_;
                if (v_ && kk[r_] == thr_) { s_ = take_ > 0; if (take_ > 0) --take_; }
                sel[r_] = s_;
            }
        } else {
            for (uint r_ = 0; r_ < @RR@; ++r_) sel[r_] = tid * @RR@ + r_ < (@CNT@);
        }
    }
"""

_BODY = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint Tm = ip[0];
    const uint C = ip[1];
    const uint Tc = ip[2];
    constexpr uint KPT = 32 * KPS;
    const uint nb = (Tm + KPT - 1) / KPT;
    const uint u = tg / nb;
    const uint gi = tg % nb;
    const uint Lrow = uint(min(uint(max(lens[u], 0)), Tm));
    const bool act = !(gi * KPT >= Lrow && gi != 0);                  // groups past the row's count do nothing (group 0 always counts: a row with no keys still gets its -1 tail)
    threadgroup T qs[NH * DI];
    threadgroup uint red[32];
    threadgroup uint flag;
    if (!act) return;
    // ---- scores: lane = head, simdgroup = KPS keys
    for (uint i = tid; i < NH * DI; i += 1024) qs[(i % DI) * NH + i / DI] = q[u * NH * DI + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint t0 = gi * KPT + sg * KPS;
        bool ok[KPS];
        bool any = false;
        for (uint j = 0; j < KPS; ++j) {
            const uint t = t0 + j;
            bool o = t < Tm && int(t) < lens[u];
            if (CAND && o) o = cand[ulong(u) * Tc + t] != 0;
            ok[j] = o;
            any = any || o;
        }
        if (t0 < Tm) {
            float acc[KPS];
            for (uint j = 0; j < KPS; ++j) acc[j] = 0.0f;
            if (any) {
                const device T* kp = keys + (ulong(sess[u]) * C + t0) * DI;
                for (uint dc = 0; dc < DI; dc += 8) {
                    float kf[KPS][8];
                    for (uint j = 0; j < KPS; ++j) for (uint i = 0; i < 8; ++i) kf[j][i] = ok[j] ? float(kp[j * DI + dc + i]) : 0.0f;
                    for (uint i = 0; i < 8; ++i) {
                        const float qv = float(qs[(dc + i) * NH + lane]);
                        for (uint j = 0; j < KPS; ++j) acc[j] = fma(qv, kf[j][i], acc[j]);
                    }
                }
            }
            const float wv = float(wts[u * NH + lane]);
            for (uint j = 0; j < KPS; ++j) {
                const T tile = T(acc[j]);
                const T rel = tile > T(0) ? tile : T(0);
                const float pr = float(T(float(rel) * wv));
                float total = 0.0f;
                for (uint h = 0; h < NH; ++h) total += simd_shuffle(pr, ushort(h));
                const uint t = t0 + j;
                if (lane == 0 && t < Tm) {
                    const T sv = ok[j] ? T(total) : T(-INFINITY);
                    scores[ulong(u) * Tm + t] = sv;
                    atomic_store_explicit(((device atomic_uint*)sc32) + ulong(u) * Tm + t, uint(as_type<ushort>(sv)), memory_order_relaxed);
                }
            }
        }
    }
    // ---- publish: the stores of the group are visible on the device before the group is counted
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint need = max(1u, (Lrow + KPT - 1) / KPT);
        device atomic_uint* c = ((device atomic_uint*)cnt) + u;
        const uint prev = atomic_fetch_add_explicit(c, 1u, memory_order_relaxed);
        const bool lastg = (prev + 1 == need);
        if (lastg) atomic_store_explicit(c, 0u, memory_order_relaxed);        // ready for the next call
        flag = lastg ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (flag == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    // ---- select (last group of the row): top-k indices
    device atomic_uint* sc = ((device atomic_uint*)sc32) + ulong(u) * Tm;
    const uint L = Lrow;
    uint kk[@R@];
    bool sel[@R@];
    for (uint r = 0; r < @R@; ++r) {
        const uint i = tid * @R@ + r;
        kk[r] = i < L ? fz_key16(ushort(atomic_load_explicit(sc + i, memory_order_relaxed))) : 0u;
    }
@LEVEL_TOPK@
    {
        uint c = 0;
        for (uint r = 0; r < @R@; ++r) c += sel[r] ? 1u : 0u;
        uint total;
        uint pos = fz_excl(c, red, sg, lane, total);
        for (uint r = 0; r < @R@; ++r) if (sel[r]) out[ulong(u) * @K@ + pos++] = int(tid * @R@ + r);
        for (uint i = total + tid; i < @K@; i += 1024) out[ulong(u) * @K@ + i] = -1;
    }
@BLOCKS@
"""

_BLOCKS = r"""
    // ---- block level (candidate source): last block pinned, finite-only mask
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint nel = (L + @BS@ - 1) / @BS@;
        const uint nd = nel > 0 ? nel - 1 : 0;
        uint kb[@RB@];
        bool fb[@RB@];
        bool selb[@RB@];
        for (uint r = 0; r < @RB@; ++r) {
            const uint b = tid * @RB@ + r;
            float m = -INFINITY;
            if (b < nd) { const uint e = min(L, (b + 1) * @BS@); for (uint t = b * @BS@; t < e; ++t) m = max(m, float(as_type<T>(ushort(atomic_load_explicit(sc + t, memory_order_relaxed))))); }
            fb[r] = m > -INFINITY;
            kb[r] = b < nd ? fz_key16(as_type<ushort>(T(m))) : 0u;
        }
        // reuse the level code on the block keys
        {
@LEVEL_BLK@
        }
        for (uint r = 0; r < @RB@; ++r) {
            const uint b = tid * @RB@ + r;
            if (b < nd) { const bool keep = selb[r] && fb[r]; const uint e = min(L, (b + 1) * @BS@); for (uint t = b * @BS@; t < e; ++t) mout[ulong(u) * Tm + t] = keep; }
        }
        if (tid == 0 && nel > 0) {
            float m = -INFINITY;
            for (uint t = (nel - 1) * @BS@; t < L; ++t) m = max(m, float(as_type<T>(ushort(atomic_load_explicit(sc + t, memory_order_relaxed)))));
            const bool fin = m > -INFINITY;
            for (uint t = (nel - 1) * @BS@; t < L; ++t) mout[ulong(u) * Tm + t] = fin;
        }
    }
"""


def _source(k: int, kb: int, bs: int, cs: bool, r: int) -> str:
    rb = max(1, -(-r * NT // bs) // NT + (1 if (-(-r * NT // bs)) % NT else 0))
    lv_top = _LEVEL.replace("@CNT@", "L").replace("@KD@", str(k)).replace("@RR@", str(r))
    s = _BODY.replace("@LEVEL_TOPK@", lv_top)
    if cs:
        lv = _LEVEL.replace("kk[", "kb[").replace("sel[", "selb[").replace("@CNT@", "nd").replace("@KD@", str(kb - 1)).replace("@RR@", str(rb))
        s = s.replace("@BLOCKS@", _BLOCKS.replace("@LEVEL_BLK@", lv))
    else:
        s = s.replace("@BLOCKS@", "")
    return s.replace("@RB@", str(rb)).replace("@BS@", str(bs)).replace("@K@", str(k)).replace("@R@", str(r))


# ----------------------------------------------------------------------------------------------------------------- packed keys (flag stackpack)
# The key load of the two score kernels above, `float(kp[j * DI + dc + i])`, replaced by the same value taken from the fp4 codes and the ue8m0 scale byte of the
# stackpack stack (fp4pack.unpack(codes, scales, 32, False, T)): nibble (low = even element), LUT magnitude, sign bit 3, 2^(s - 127) built from the exponent byte
# (an exact power of two), one fp32 product (exact) and one cast to T, like the `.astype(dtype)` of the unpack. Everything else of the kernel text is untouched;
# the dense form stays what flag 0 binds.
_PK_HELPER = r"""
constant float PKL[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
template <typename T>
inline float pk_key(const device uchar* kc, const device uchar* ksc, uint j, uint d, uint DI) {
    const uint b = uint(kc[ulong(j) * (DI / 2) + (d >> 1)]);
    const uint c = (d & 1u) ? (b >> 4) : (b & 15u);
    const float mag = PKL[c & 7u];
    const float v = (c & 8u) ? -mag : mag;
    const float sc = as_type<float>(uint(ksc[ulong(j) * (DI / 32) + (d >> 5)]) << 23);
    return float(T(v * sc));
}
"""
_PK_LOAD = "kf[j][i] = ok[j] ? float(kp[j * DI + dc + i]) : 0.0f;"


def pack_source(src: str) -> str:
    """`src` (a score kernel text above) with its key pointer and key load reading the codes: kcodes / kscales instead of keys."""
    for v in ("Ck", "C"):
        old = f"const device T* kp = keys + (ulong(sess[u]) * {v} + t0) * DI;"
        if src.count(old) == 1:
            assert src.count(_PK_LOAD) == 1
            new = (f"const device uchar* kc = kcodes + (ulong(sess[u]) * {v} + t0) * (DI / 2);\n"
                   f"                const device uchar* ksc = kscales + (ulong(sess[u]) * {v} + t0) * (DI / 32);")
            return src.replace(old, new).replace(_PK_LOAD, "kf[j][i] = ok[j] ? pk_key<T>(kc, ksc, j, dc + i, DI) : 0.0f;")
    raise AssertionError("key pointer line not found")


_POOL = 16


def _counter(n: int) -> mx.array:
    """Round-robin over a few zeroed counter buffers (>= 64 words so MLX binds them as device memory): two calls never share one,
    so kernels MLX may overlap cannot meet on a counter."""
    pool = _cnt.get(n)
    if pool is None:
        pool = _cnt[n] = [mx.zeros((max(n, 64),), dtype=mx.uint32) for _ in range(_POOL)] + [0]
        mx.eval(*pool[:-1])
    pool[-1] = (pool[-1] + 1) % _POOL
    return pool[pool[-1]]


def reg_class(lmax: int) -> int:
    """Keys per thread (power of two >= lmax / NT)."""
    r = 1
    while r * NT < lmax:
        r *= 2
    return r


def supported(lmax: int, nh: int, dt) -> bool:
    return nh == 32 and lmax <= MAX_KEYS and dt in (mx.float16, mx.bfloat16)


def fused(q, keys, wts, lens, t_max, k, cand=None, sess=None, topk_blocks=0, block_size=8, lmax=None):
    """(idx [N, k] int32, mask [N, t_max] bool | None, scores): bit-identical to index_scores(rt=True) -> index_topk_rt [-> candidate_mask_rt].
    topk_blocks > 0 = candidate-source layer (mask returned). q [N, 32, DI], keys [S, C, DI], wts [N, 32], lens [N] int32 (all <= lmax)."""
    N, NH, DI = q.shape
    lmax = min(t_max, lmax or t_max)
    assert supported(lmax, NH, q.dtype) and q.dtype == keys.dtype == wts.dtype and DI % 8 == 0 and t_max <= keys.shape[1]
    cs = topk_blocks > 0
    assert not cs or topk_blocks >= 2
    r = reg_class(lmax)
    name = f"kyojin_idxfuse_{k}_{topk_blocks}_{block_size}_{r}"
    pk = hasattr(keys, "codes")                                         # stackpack: keys is a stackpack.PackedStack, loaded from the codes
    ker = _k.get(name + "_pk" if pk else name)
    if ker is None and pk:
        ker = _k[name + "_pk"] = mx.fast.metal_kernel(
            name=name + "_pk", input_names=["q", "kcodes", "kscales", "wts", "lens", "cand", "sess", "ip", "cnt"], output_names=["scores", "out", "mout", "sc32"],
            source=pack_source(_source(k, topk_blocks, block_size, cs, r)), header=_HEADER + _PK_HELPER)
    elif ker is None:
        ker = _k[name] = mx.fast.metal_kernel(
            name=name, input_names=["q", "keys", "wts", "lens", "cand", "sess", "ip", "cnt"], output_names=["scores", "out", "mout", "sc32"],
            source=_source(k, topk_blocks, block_size, cs, r), header=_HEADER)
    sess = C._identity(N) if sess is None else sess
    c = cand if cand is not None else mx.zeros((1,), dtype=mx.bool_)
    tc = cand.shape[-1] if cand is not None else 1
    assert cand is None or tc >= t_max
    cnt = _counter(N)
    kin = [keys.codes, keys.scales] if pk else [keys]
    scores, idx, mask, _ = ker(
        inputs=[q, *kin, wts, lens, c, sess, C._u32((t_max, FP.stride(keys), tc, t_max)), cnt],
        template=[("T", q.dtype), ("NH", NH), ("DI", DI), ("KPS", KPS), ("CAND", cand is not None)],
        grid=(N * (-(-t_max // (32 * KPS))) * NT, 1, 1), threadgroup=(NT, 1, 1),
        output_shapes=[(N, t_max), (N, k), (N, t_max) if cs else (1,), (N, t_max)], output_dtypes=[q.dtype, mx.int32, mx.bool_, mx.uint32])
    return idx, (mask if cs else None), scores
