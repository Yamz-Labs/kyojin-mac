# SPDX-License-Identifier: Apache-2.0
"""DS4.1 attention core for N users in one dispatch per op (multi-user decode, one token per user).

The model's own decode attention keeps private state at a private offset per user (window ring, compressor state, compressed pool, index keys)
and runs the whole core once per user: about 110 launches per user per indexer layer. Here the state of all users is stacked and ragged:

    window ring   [N, W, D]        slot p % W holds position p
    pool          [N, cap, D]      user u owns rows 0 .. (pos_u + 1) // ratio - 1
    index keys    [N, cap, DI]     same row count as the pool of the layer that owns them
    compressor    [N, ratio, D]    fp32 kv / score slots, slot p % ratio

Positions, row counts and who completes a compression group on this step are known on the host (every user advances by one token), so the
ragged parts are plain index arrays and no step reads anything back from the GPU.

Exactness: the three reductions whose summation order the stock ops tie to shapes (indexer scores over heads and channels, the compressor
softmax-pool, the window + top-k softmax attention) are Metal kernels with one fixed summation order per (user, head, entry): a row does
not depend on the number of users, on the padding of the other users, or on the capacity of the buffers. Selection (sort / top-k), norms,
rotary and the fake-quant blocks are per-row ops of MLX. Entries beyond a user's own length are skipped, never added as zeros of a
different shape. The solo decode (one user) is the same code with N = 1.
"""
from __future__ import annotations

from . import archs as _A
import math
import os
import sys
import time
from types import SimpleNamespace

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import fired as FD
from . import ablate as AZ
from . import cdecpack as CD
from . import mirrorq as MQ
from . import idxshort as IDXS
from . import longctx as LC
from . import decfilter as DFL
from . import stackpack as SP
from . import idxblocks as IXB
from . import decidx8 as IX8
from . import decblk as DBK
from . import decsel as DCS
from . import keyscodes as KC
from . import fp4pool as FP

# indexer query / key rotary + FP4 round trip as one kernel per call (ds_pf_idx.fq4_rot, bit-identical, one simdgroup per row) instead of the
# stock chain (complex64 rotary, ~17 elementwise launches on the serial path of every indexer layer). Switch: KYOJIN_DEC_IDX_FUSE=0.
ROT_FQ4 = os.environ.get("KYOJIN_DEC_IDX_FUSE", "1") != "0"           # (its own name: IDX_FUSE below is the fused indexer select, flag idx_fuse)

_CONTRACT_OFF = "#pragma METAL fp contract(off)\n"

# ----------------------------------------------------------------------------------------------------------------- membership (flag mu_keep)
# A member joining or leaving the batched decode used to cost a whole-state round trip of EVERY member: the engine dissolved the batch
# (unbatch -> sync_rows -> one extract per row, O(rows) each), re-merged it (one extract per session) and re-stacked it (one dense copy of every
# session's pool / index keys, plus the fp4 unpack under it). At 131072 tokens a session's stacked pool + key rows are 419 MB dense (2.5 rows per
# token x (1024 + 256) B), so the join and the leave each moved several GB.
#
# KYOJIN_MU_KEEP=1 (default 1) keeps the stack and updates only the rows that change: the session axis of the stacked buffers is allocated with
# `mu_spare()` spare slots, each session owns one slot (`cache.slots`), `Plan.sess` holds the slot of a row and `Plan.NS` the slot count (so every
# kernel and every scatter addresses a slot, while `Plan.S` stays the number of sessions). A leave drops the slot from the map (no stacked buffer
# is read or written) after writing back that one row; a join writes the joining session's rows into a free slot. The small stacked state (window
# rings, compressor kv / score slots) rides along on the same slot axis, so it is not copied either. Exact: same rows, same order, same bits; the
# only shape change is a wider session axis, which no read depends on (a row's arithmetic does not depend on the batch width).
# `HITS["copy_bytes"]` counts what the membership path actually copies, so a test can hold a stayer to zero.
def mu_keep() -> bool:
    """Flag mu_keep, read per stack. Off under KYOJIN_VERIFY_REPLAY (the recorded command stream is shape-stable by construction)."""
    return os.environ.get("KYOJIN_MU_KEEP", "1") not in ("0", "") and os.environ.get("KYOJIN_VERIFY_REPLAY", "0") in ("0", "")


def mu_spare() -> int:
    """Spare session slots the stack reserves (KYOJIN_MU_KEEP_SPARE, default 1): one join then copies no row of the members that stay. 0 = the
    stack grows on a join instead (one copy of the live slots, still no re-pack)."""
    try:
        return max(0, int(os.environ.get("KYOJIN_MU_KEEP_SPARE", "1") or 1))
    except ValueError:
        return 1


HITS = {"stack_slots": 0, "kept_rows": 0, "adopt": 0, "drop": 0, "extract_row": 0, "sync_row": 0, "grow": 0,
        "copy_bytes": 0, "keep_bytes": 0}

# ----------------------------------------------------------------------------------------------------------------- attention kernel
# One threadgroup (4 simdgroups) per (user, group of HG heads). Entries j = 0 .. W + K - 1: j < W is window position base + j (ring slot
# p % W), j >= W is pool row idx[u, j - W] (-1 = none). Scores: lane l of a simdgroup owns channels 16 l .. 16 l + 15 (fp32 fma in order),
# then simd_sum. Softmax over the entries and the sink: max (exact), exp, per-lane partial sums over j = l, l + 32, ... in order, simd_sum.
# Output: thread t owns channels DPT t .. DPT t + DPT - 1 and accumulates the valid entries in increasing j.
_ATTN = r"""
    constexpr uint NSG = 4;
    constexpr uint TPG = NSG * 32;
    constexpr uint DPL = D / 32;
    constexpr uint DPT = D / TPG;
    constexpr uint NHG = H / HG;
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint u = tg / NHG;
    const uint h0 = (tg % NHG) * HG;
    const uint K = ip[0];
    const uint C = ip[1];
    const uint M = W + K;
    const float scale = fp[0];
    const int base = pos[u] - int(W) + 1;
    threadgroup float sc[HG * MAXM];
    threadgroup int ent[MAXM];
    threadgroup float lsum[HG];
    for (uint j = tid; j < M; j += TPG) {
        int e;
        if (j < W) {
            const int p = base + int(j);
            e = p >= 0 ? (p % int(W)) : -1;
        } else {
            e = idx[u * K + (j - W)];
            if (e >= int(C)) e = -1;
        }
        ent[j] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qv[HG][DPL];
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPL; ++i)
            qv[hh][i] = float(q[(u * H + h0 + hh) * D + lane * DPL + i]) * scale;
    for (uint j = sg; j < M; j += NSG) {
        const int e = ent[j];
        float s[HG];
        if (e < 0) {
            for (uint hh = 0; hh < HG; ++hh) s[hh] = -INFINITY;
        } else {
            const device T* kp = (j < W) ? (win + (ulong(u) * W + uint(e)) * D) : (pool + (ulong(sess[u]) * C + uint(e)) * D);
            float kv[DPL];
            for (uint i = 0; i < DPL; ++i) kv[i] = float(kp[lane * DPL + i]);
            for (uint hh = 0; hh < HG; ++hh) {
                float a = 0.0f;
                for (uint i = 0; i < DPL; ++i) a = fma(qv[hh][i], kv[i], a);
                s[hh] = simd_sum(a);
            }
        }
        if (lane == 0)
            for (uint hh = 0; hh < HG; ++hh) sc[hh * MAXM + j] = s[hh];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint hh = sg; hh < HG; hh += NSG) {
        const float sk = sink[h0 + hh];
        float m = sk;
        for (uint j = lane; j < M; j += 32) m = max(m, sc[hh * MAXM + j]);
        m = simd_max(m);
        float part = 0.0f;
        for (uint j = lane; j < M; j += 32) {
            const float ex = ent[j] >= 0 ? exp(sc[hh * MAXM + j] - m) : 0.0f;
            sc[hh * MAXM + j] = ex;
            part += ex;
        }
        const float tot = simd_sum(part);
        if (lane == 0) lsum[hh] = tot + exp(sk - m);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[HG][DPT];
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPT; ++i) acc[hh][i] = 0.0f;
    for (uint j = 0; j < M; ++j) {
        const int e = ent[j];
        if (e < 0) continue;
        const device T* kp = (j < W) ? (win + (ulong(u) * W + uint(e)) * D) : (pool + (ulong(sess[u]) * C + uint(e)) * D);
        float v[DPT];
        for (uint i = 0; i < DPT; ++i) v[i] = float(kp[tid * DPT + i]);
        for (uint hh = 0; hh < HG; ++hh) {
            const float p = sc[hh * MAXM + j];
            for (uint i = 0; i < DPT; ++i) acc[hh][i] = fma(p, v[i], acc[hh][i]);
        }
    }
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPT; ++i)
            out[(u * H + h0 + hh) * D + tid * DPT + i] = T(acc[hh][i] / lsum[hh]);
"""

# Split-key variant (flash decoding): the one-threadgroup-per-head-group kernel above runs W + K entries serially and fills only N * H / HG
# threadgroups (16 for one user on a 80-core GPU), so a decode step pays ~250 us per layer for 84 MFLOP. Here entry range s of SPLIT parts
# goes to its own threadgroup, which writes its local max, sum of exp and unnormalised fp32 output; a second kernel merges the parts with
# the sink. Same formulas, a different summation order (parts merged in increasing s): rounding-level differences only, still row independent.
_ATTN_PART = r"""
    constexpr uint NSG = 4;
    constexpr uint TPG = NSG * 32;
    constexpr uint DPL = D / 32;
    constexpr uint DPT = D / TPG;
    constexpr uint NHG = H / HG;
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint sp = tg % SPLIT;
    const uint u = (tg / SPLIT) / NHG;
    const uint h0 = ((tg / SPLIT) % NHG) * HG;
    const uint K = ip[0];
    const uint C = ip[1];
    const uint M = W + K;
    const uint j0 = sp * MS;
    const uint n = j0 < M ? min(uint(MS), M - j0) : 0;
    const float scale = fp[0];
    const int base = pos[u] - int(W) + 1;
    threadgroup float sc[HG * MS];
    threadgroup int ent[MS];
    threadgroup float lmax[HG];
    for (uint jj = tid; jj < n; jj += TPG) {
        const uint j = j0 + jj;
        int e;
        if (j < W) {
            const int p = base + int(j);
            e = p >= 0 ? (p % int(W)) : -1;
        } else {
            e = idx[u * K + (j - W)];
            if (e >= int(C)) e = -1;
        }
        ent[jj] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qv[HG][DPL];
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPL; ++i)
            qv[hh][i] = float(q[(u * H + h0 + hh) * D + lane * DPL + i]) * scale;
    for (uint jj = sg; jj < n; jj += NSG) {
        const uint j = j0 + jj;
        const int e = ent[jj];
        float s[HG];
        if (e < 0) {
            for (uint hh = 0; hh < HG; ++hh) s[hh] = -INFINITY;
        } else {
            const device T* kp = (j < W) ? (win + (ulong(u) * W + uint(e)) * D) : (pool + (ulong(sess[u]) * C + uint(e)) * D);
            float kv[DPL];
            for (uint i = 0; i < DPL; ++i) kv[i] = float(kp[lane * DPL + i]);
            for (uint hh = 0; hh < HG; ++hh) {
                float a = 0.0f;
                for (uint i = 0; i < DPL; ++i) a = fma(qv[hh][i], kv[i], a);
                s[hh] = simd_sum(a);
            }
        }
        if (lane == 0)
            for (uint hh = 0; hh < HG; ++hh) sc[hh * MS + jj] = s[hh];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong ob = (ulong(u) * H + h0) * SPLIT + sp;          // part row of head h0 + hh: ob + hh * SPLIT
    for (uint hh = sg; hh < HG; hh += NSG) {
        float m = -INFINITY;
        for (uint jj = lane; jj < n; jj += 32) m = max(m, ent[jj] >= 0 ? sc[hh * MS + jj] : -INFINITY);
        m = simd_max(m);
        float part = 0.0f;
        for (uint jj = lane; jj < n; jj += 32) {
            const float ex = ent[jj] >= 0 ? exp(sc[hh * MS + jj] - m) : 0.0f;
            sc[hh * MS + jj] = ex;
            part += ex;
        }
        const float tot = simd_sum(part);
        if (lane == 0) { lmax[hh] = m; ml[(ob + hh * SPLIT) * 2] = m; ml[(ob + hh * SPLIT) * 2 + 1] = tot; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[HG][DPT];
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPT; ++i) acc[hh][i] = 0.0f;
    for (uint jj = 0; jj < n; ++jj) {
        const uint j = j0 + jj;
        const int e = ent[jj];
        if (e < 0) continue;
        const device T* kp = (j < W) ? (win + (ulong(u) * W + uint(e)) * D) : (pool + (ulong(sess[u]) * C + uint(e)) * D);
        float v[DPT];
        for (uint i = 0; i < DPT; ++i) v[i] = float(kp[tid * DPT + i]);
        for (uint hh = 0; hh < HG; ++hh) {
            const float p = sc[hh * MS + jj];
            for (uint i = 0; i < DPT; ++i) acc[hh][i] = fma(p, v[i], acc[hh][i]);
        }
    }
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPT; ++i)
            part_o[(ob + hh * SPLIT) * D + tid * DPT + i] = acc[hh][i];
"""

# merge: one threadgroup (128 threads) per (user, head); thread t owns channels DPT t ..; parts in increasing s
_ATTN_MERGE = r"""
    constexpr uint DPT = D / 128;
    const uint r = threadgroup_position_in_grid.x;              // u * H + h
    const uint tid = thread_position_in_threadgroup.x;
    const float sk = sink[r % H];
    float m = sk;
    for (uint s = 0; s < SPLIT; ++s) m = max(m, ml[(r * SPLIT + s) * 2]);
    float l = exp(sk - m);
    float acc[DPT];
    for (uint i = 0; i < DPT; ++i) acc[i] = 0.0f;
    for (uint s = 0; s < SPLIT; ++s) {
        const float ms = ml[(r * SPLIT + s) * 2];
        if (ms == -INFINITY) continue;
        const float w = exp(ms - m);
        l = fma(ml[(r * SPLIT + s) * 2 + 1], w, l);
        for (uint i = 0; i < DPT; ++i) acc[i] = fma(part_o[(r * SPLIT + s) * D + tid * DPT + i], w, acc[i]);
    }
    for (uint i = 0; i < DPT; ++i) out[r * D + tid * DPT + i] = T(acc[i] / l);
"""
# Part size is a constant (not a function of N, K or the buffers): part boundaries sit at fixed entries, so a row keeps the same bits whatever
# the number of users or the padding of idx (padded parts are all -1 and the merge skips them). 0 = one-pass kernel.
SPLIT_MS = int(os.environ.get("KYOJIN_DSV4_SPLIT_MS", "64"))
SPLIT_HG = int(os.environ.get("KYOJIN_DSV4_SPLIT_HG", "2"))     # heads per part threadgroup (they share the key reads)

# ----------------------------------------------------------------------------------------------------------------- q / inverse rope in the kernels
# The q rope of a row (the tail dims of every head, mx.fast.rope traditional=True) rides in the attention kernel's q load: the lane already holds the
# DPL-dim chunk it scores, and the rope factors come from mx.fast.rope itself (Plan.attn_rope). Each component is rounded to T like the stock op before
# the scale. The inverse rope rides in the merge epilogue (the thread that writes the 4-dim chunk), same table, same rounding.
_Q_LOAD = """    float qv[HG][DPL];
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPL; ++i)
            qv[hh][i] = float(q[(u * H + h0 + hh) * D + lane * DPL + i]) * scale;
"""
_Q_ROPE = """    float qv[HG][DPL];
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPL; ++i)
            qv[hh][i] = float(q[(u * H + h0 + hh) * D + lane * DPL + i]);
    if (lane * DPL >= uint(D - RD)) {                                 // rope tail: DPL / 2 interleaved pairs per lane
        const device float* rtr = rt + ulong(u) * RD;
        const uint p0 = (lane * DPL - uint(D - RD)) / 2;
        for (uint hh = 0; hh < HG; ++hh)
            for (uint j = 0; j < DPL / 2; ++j) {
                const float c = rtr[2 * (p0 + j)], s = rtr[2 * (p0 + j) + 1];
                const float x1 = qv[hh][2 * j], x2 = qv[hh][2 * j + 1];
                qv[hh][2 * j] = float(static_cast<T>(x1 * c - x2 * s));
                qv[hh][2 * j + 1] = float(static_cast<T>(x1 * s + x2 * c));
            }
    }
    for (uint hh = 0; hh < HG; ++hh)
        for (uint i = 0; i < DPL; ++i) qv[hh][i] = qv[hh][i] * scale;
"""
_MERGE_OUT = """    for (uint i = 0; i < DPT; ++i) out[r * D + tid * DPT + i] = T(acc[i] / l);
"""
_MERGE_INV = """    float v[DPT];
    for (uint i = 0; i < DPT; ++i) v[i] = float(T(acc[i] / l));
    if (tid * DPT >= uint(D - RD)) {                                  // inverse rope tail (the table holds cos / sin of the negated angles)
        const device float* itr = it + ulong(r / H) * RD;
        const uint p0 = (tid * DPT - uint(D - RD)) / 2;
        for (uint j = 0; j < DPT / 2; ++j) {
            const float c = itr[2 * (p0 + j)], s = itr[2 * (p0 + j) + 1];
            const float x1 = v[2 * j], x2 = v[2 * j + 1];
            v[2 * j] = float(static_cast<T>(x1 * c - x2 * s));
            v[2 * j + 1] = float(static_cast<T>(x1 * s + x2 * c));
        }
    }
    for (uint i = 0; i < DPT; ++i) out[r * D + tid * DPT + i] = T(v[i]);
"""
assert _ATTN.count(_Q_LOAD) == 1 and _ATTN_PART.count(_Q_LOAD) == 1 and _ATTN_MERGE.count(_MERGE_OUT) == 1
_ATTN_Q, _ATTN_PART_Q, _ATTN_MERGE_Q = _ATTN.replace(_Q_LOAD, _Q_ROPE), _ATTN_PART.replace(_Q_LOAD, _Q_ROPE), _ATTN_MERGE.replace(_MERGE_OUT, _MERGE_INV)

# attn_merge_tail (KYOJIN_ATTN_MERGE_TAIL): the merge rides in the tail of the part kernel. Every part threadgroup stores its partials, then arrives on the counter
# of its (row, head group); the last arriver (counter reset by it, so the buffer is zero for the next launch) runs the statements of _ATTN_MERGE[_Q] for the HG heads
# of the group on the partials the others stored (read back with relaxed atomic loads after device fences, the moe_chain protocol: metal_src._CH_ARRIVE).
# Same arithmetic in the same order: bit-identical to the two-launch path. No early return before the arrival (checked below).
ATTN_MERGE_TAIL = os.environ.get("KYOJIN_ATTN_MERGE_TAIL", "0") not in ("0", "")
_MT_LDF = """
#define LDF(ptr) as_type<float>(atomic_load_explicit((const device atomic_uint*)((ptr)), memory_order_relaxed))
"""
_MT_ARRIVE = """
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup uint mt_last;
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit(((device atomic_uint*)(ctr)) + (tg / SPLIT), 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const bool lst = (old + 1u) == uint(SPLIT);
        if (lst) atomic_store_explicit(((device atomic_uint*)(ctr)) + (tg / SPLIT), 0u, memory_order_relaxed);
        mt_last = lst ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (mt_last == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
"""
_MT_R = "    const uint r = threadgroup_position_in_grid.x;              // u * H + h\n    const uint tid = thread_position_in_threadgroup.x;\n"


def merge_tail_source(part: str, merge: str) -> str:
    assert "return" not in part and merge.count(_MT_R) == 1, "attn part / merge template changed"
    body = merge.replace(_MT_R, "").replace("part_o[(r * SPLIT + s) * D + tid * DPT + i]", "LDF(part_o + (r * SPLIT + s) * D + tid * DPT + i)")
    body = body.replace("ml[(r * SPLIT + s) * 2 + 1]", "LDF(ml + (r * SPLIT + s) * 2 + 1)").replace("ml[(r * SPLIT + s) * 2]", "LDF(ml + (r * SPLIT + s) * 2)")
    assert body.count("LDF(") == 4 and "part_o[" not in body and "ml[" not in body
    head = "    for (uint hh = 0; hh < HG; ++hh) {\n    const uint r = (u * H + h0 + hh);\n"
    return _MT_LDF + part + _MT_ARRIVE + head + body + "    }\n"


_mt_ctr: dict = {}


def _attn_ctr(n: int) -> mx.array:
    """Zeroed arrival counters, one per (row, head group). The kernel leaves them zero; >= 16 words (MLX binds arrays under 8 elements as `constant`)."""
    n = max(16, n)
    c = _mt_ctr.get(n)
    if c is None:
        c = _mt_ctr[n] = mx.zeros((n,), dtype=mx.uint32)
        mx.eval(c)
    return c

# ----------------------------------------------------------------------------------------------------------------- kv chain in one launch
# One threadgroup (128 threads) per (row, ring slot). The slot a row writes recomputes its kv row (rms_norm in mx.fast.rms_norm's single-row layout:
# 4 contiguous values per thread, simd_sum, then the simdgroup partials; the rope tail and the fp8 round trip of _KV_RING) and writes it; every other
# slot copies the row's session ring. Same arithmetic, same bits as kv_norm + rope + fake_quant_fp8_ue8m0 + the overlay scatter.
_KV_CHAIN = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint b = tg / uint(W);
    const uint s = tg % uint(W);
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, sl = thread_index_in_simdgroup;
    const int e = wmap[b * uint(W) + s];                       // kv row that writes this slot of row b's ring, -1 = keep the base ring
    const ulong o = (ulong(b) * uint(W) + s) * uint(D);
    if (e < 0) {
        if (lid < 64) {
            const ulong q = (ulong(sess[b]) * uint(W) + s) * uint(D) + lid * 8;
            *(device uint4*)(out + o + lid * 8) = *(const device uint4*)(ring + q);
        }
        return;
    }
    const device T* xr = kvr + ulong(e) * uint(D);
    threadgroup float ls[32];
    threadgroup T nb[D];
    float acc = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const float f = float(xr[lid * 4 + i]);
        acc += f * f;
    }
    acc = simd_sum(acc);
    if (sg == 0) ls[sl] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sl == 0) ls[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float t = simd_sum(ls[sl]);
        if (sl == 0) ls[0] = metal::precise::rsqrt(t / float(D) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = ls[0];
    for (uint i = 0; i < 4; ++i) nb[lid * 4 + i] = w[lid * 4 + i] * static_cast<T>(float(xr[lid * 4 + i]) * inv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid >= 64) return;
    float v[8];
    for (uint i = 0; i < 8; ++i) v[i] = float(nb[lid * 8 + i]);
    if (lid * 8 >= uint(D - RD)) {                             // rope tail: 4 interleaved pairs per thread (mx.fast.rope traditional=True)
        const device float* rtr = rt + ulong(e) * RD;
        const uint p0 = (lid * 8 - uint(D - RD)) / 2;
        for (uint j = 0; j < 4; ++j) {
            const float c = rtr[2 * (p0 + j)], s = rtr[2 * (p0 + j) + 1];
            const float x1 = v[2 * j], x2 = v[2 * j + 1];
            v[2 * j] = float(static_cast<T>(x1 * c - x2 * s));
            v[2 * j + 1] = float(static_cast<T>(x1 * s + x2 * c));
        }
    }
    float amax = 0.0f;
    for (uint i = 0; i < 8; ++i) { amax = metal::max(amax, metal::abs(v[i])); }
    amax = metal::max(amax, simd_shuffle_xor(amax, 1u));
    amax = metal::max(amax, simd_shuffle_xor(amax, 2u));
    amax = metal::max(amax, 1e-4f);
    const uint bits = as_type<uint>(amax * (1.0f / 448.0f));
    const uint exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0);
    const float scale = as_type<float>(exponent << 23);
    for (uint i = 0; i < 8; ++i) {
        const float vv = metal::clamp(v[i] / scale, -448.0f, 448.0f);
        const uint vbits = as_type<uint>(metal::abs(vv));
        const uint qexp = metal::max(int(vbits >> 23) - 3, 118);
        const float step = as_type<float>(qexp << 23);
        out[o + lid * 8 + i] = static_cast<T>((metal::rint(vv / step) * step) * scale);
    }
"""


# kv chain + the commit's slot write in one launch: one threadgroup per (slot of the stacked buffer, ring position). A slot a row of the pass owns
# computes exactly the _KV_CHAIN result and writes it to `out` (the row's ring, read by the attention) and to `buf` (the stacked buffer the commit
# keeps); every other slot copies the base buffer into `buf`. rowof[slot] = the row that owns the slot (plain decode: one row per session), -1 = none.
_KV_CHAIN_STK = _KV_CHAIN.replace("""    const uint tg = threadgroup_position_in_grid.x;
    const uint b = tg / uint(W);
    const uint s = tg % uint(W);
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, sl = thread_index_in_simdgroup;
    const int e = wmap[b * uint(W) + s];                       // kv row that writes this slot of row b's ring, -1 = keep the base ring
    const ulong o = (ulong(b) * uint(W) + s) * uint(D);
    if (e < 0) {
        if (lid < 64) {
            const ulong q = (ulong(sess[b]) * uint(W) + s) * uint(D) + lid * 8;
            *(device uint4*)(out + o + lid * 8) = *(const device uint4*)(ring + q);
        }
        return;
    }
""", """    const uint tg = threadgroup_position_in_grid.x;
    const uint slot = tg / uint(W);
    const uint s = tg % uint(W);
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, sl = thread_index_in_simdgroup;
    const int rw = rowof[slot];
    const ulong ob = (ulong(slot) * uint(W) + s) * uint(D);
    if (rw < 0) {
        if (lid < 64) *(device uint4*)(buf + ob + lid * 8) = *(const device uint4*)(ring + ob + lid * 8);
        return;
    }
    const uint b = uint(rw);
    const int e = wmap[b * uint(W) + s];
    const ulong o = (ulong(b) * uint(W) + s) * uint(D);
    if (e < 0) {
        if (lid < 64) {
            const uint4 v4 = *(const device uint4*)(ring + ob + lid * 8);
            *(device uint4*)(out + o + lid * 8) = v4;
            *(device uint4*)(buf + ob + lid * 8) = v4;
        }
        return;
    }
""")
_old="        out[o + lid * 8 + i] = static_cast<T>((metal::rint(vv / step) * step) * scale);\n"
_new="        const T ov = static_cast<T>((metal::rint(vv / step) * step) * scale);\n        out[o + lid * 8 + i] = ov;\n        buf[ob + lid * 8 + i] = ov;\n"
assert _KV_CHAIN_STK.count(_old) == 1
_KV_CHAIN_STK = _KV_CHAIN_STK.replace(_old, _new)
assert "rowof[slot]" in _KV_CHAIN_STK


# ----------------------------------------------------------------------------------------------------------------- indexer scores kernel
# score[u, t] = T( sum_h T( relu(T(q_h . key_t)) * w_h ) ): the roundings of the stock fp16 einsum / relu / weight product, fp32 sums in
# increasing channel / head order. One thread per (user, key); -inf for t >= lens[u] and for keys the candidate mask drops.
_ISCORE = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint Tm = ip[0];
    const uint C = ip[1];
    const uint Tc = ip[2];
    const uint nb = (Tm + TB - 1) / TB;
    const uint u = tg / nb;
    const uint t = (tg % nb) * TB + tid;
    if (RT && (tg % nb) * TB >= uint(max(lens[u], 0))) return;                    // runtime width: the grid covers the capacity, groups past the row's count do nothing (the whole group leaves, no barrier is skipped)
    threadgroup T qs[NH * DI];
    for (uint i = tid; i < NH * DI; i += TB) qs[i] = q[u * NH * DI + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t >= Tm) return;
    bool ok = int(t) < lens[u];
    if (CAND && ok) ok = cand[ulong(u) * Tc + t] != 0;
    if (!ok) {
        out[ulong(u) * Tm + t] = T(-INFINITY);
        return;
    }
    float acc[NH];
    for (uint h = 0; h < NH; ++h) acc[h] = 0.0f;
    const device T* kp = keys + (ulong(sess[u]) * C + t) * DI;
    for (uint dc = 0; dc < DI; dc += 8) {
        float kf[8];
        for (uint i = 0; i < 8; ++i) kf[i] = float(kp[dc + i]);
        for (uint h = 0; h < NH; ++h) {
            float a = acc[h];
            for (uint i = 0; i < 8; ++i) a = fma(float(qs[h * DI + dc + i]), kf[i], a);
            acc[h] = a;
        }
    }
    float total = 0.0f;
    for (uint h = 0; h < NH; ++h) {
        const T tile = T(acc[h]);
        const T rel = tile > T(0) ? tile : T(0);
        const T pr = T(float(rel) * float(wts[u * NH + h]));
        total += float(pr);
    }
    out[ulong(u) * Tm + t] = T(total);
"""

# Same scores, register-blocked (KYOJIN_ISCORE_KB, or KYOJIN_LONGCTX): the kernel above issues one threadgroup-memory read of q per FMA (4096 per key), which
# bounds it far below the bandwidth roof at long context. Here a thread holds KB = 4 consecutive keys x NH / 4 heads: every q read feeds 4 FMAs and every key
# load feeds NH / 4 heads. Each (key, head) chain runs the same FMAs in the same order (d = 0 .. DI - 1 from 0), the head terms are gathered over the 4 lanes of
# the key quad by simd_shuffle and summed in head order from 0 as above: bit-identical output. Same grid (TB = 128 keys per threadgroup) and runtime width.
_ISCORE_KB = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint Tm = ip[0];
    const uint C = ip[1];
    const uint Tc = ip[2];
    const uint nb = (Tm + 127) / 128;
    const uint u = tg / nb;
    const uint base = (tg % nb) * 128;
    if (RT && base >= uint(max(lens[u], 0))) return;
    threadgroup T qs[NH * DI];
    for (uint i = tid; i < NH * DI; i += 128) qs[(i % DI) * NH + i / DI] = q[u * NH * DI + i];   // transposed [d][h]: the 4 head groups of a quad read 4 banks
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint HG = NH / 4;
    const uint hg = tid & 3;
    const uint t0 = base + (tid >> 2) * 4;                     // 4 consecutive keys per quad of lanes
    const int ln = lens[u];
    bool ok[4];
    bool any = false;
    for (uint j = 0; j < 4; ++j) {
        const uint t = t0 + j;
        bool o = t < Tm && int(t) < ln;
        if (CAND && o) o = cand[ulong(u) * Tc + t] != 0;
        ok[j] = o;
        any = any || o;
    }
    float acc[4][HG];
    for (uint j = 0; j < 4; ++j) for (uint h = 0; h < HG; ++h) acc[j][h] = 0.0f;
    if (any) {
        const device T* kp = keys + (ulong(sess[u]) * C + t0) * DI;
        for (uint dc = 0; dc < DI; dc += 8) {
            float kf[4][8];
            for (uint j = 0; j < 4; ++j) for (uint i = 0; i < 8; ++i) kf[j][i] = ok[j] ? float(kp[j * DI + dc + i]) : 0.0f;
            for (uint hh = 0; hh < HG; ++hh) {
                const uint h = hg * HG + hh;
                for (uint i = 0; i < 8; ++i) {
                    const float qv = float(qs[(dc + i) * NH + h]);
                    for (uint j = 0; j < 4; ++j) acc[j][hh] = fma(qv, kf[j][i], acc[j][hh]);
                }
            }
        }
    }
    const uint q0 = lane & ~3u;
    for (uint j = 0; j < 4; ++j) {
        float pr[HG];
        for (uint hh = 0; hh < HG; ++hh) {
            const T tile = T(acc[j][hh]);
            const T rel = tile > T(0) ? tile : T(0);
            pr[hh] = float(T(float(rel) * float(wts[u * NH + hg * HG + hh])));
        }
        float total = 0.0f;
        for (uint g = 0; g < 4; ++g)
            for (uint hh = 0; hh < HG; ++hh) {
                const float v = simd_shuffle(pr[hh], ushort(q0 + g));
                total += v;
            }
        const uint t = t0 + j;
        if (hg == 0 && t < Tm) out[ulong(u) * Tm + t] = ok[j] ? T(total) : T(-INFINITY);
    }
"""
ISCORE_KB = os.environ.get("KYOJIN_ISCORE_KB", "0") not in ("0", "")       # flags.py iscore_kb (switched at call time)
ISCORE_KB_MIN = int(os.environ.get("KYOJIN_ISCORE_KB_MIN", "0"))           # keys from which it runs: 0 = short contexts too (flags.py iscore_kb_min)
# Why it was gated at 256K keys and off: on the Ultra (FAST_PLAN 30/10 ab53) the same 32K/128K decode with KYOJIN_LONGCTX=1 read 60.7 / 60.5 tok/s
# with this kernel against 73.1 / 52.1 without it, i.e. behind at 32K and ahead only at 128K, so the gate sat above every measured crossover. The lever is
# now a runtime flag with its own threshold: flipping iscore_kb raises the switch and iscore_kb_min sets where it starts, short contexts included.

# ----------------------------------------------------------------------------------------------------------------- runtime-width selection kernels
# The stock selection (mx.argsort / mx.sort / where over a width that follows the pooled count, bucketed to 32) bakes the width into the graph: every step of the bucket is another
# command stream, so a recorded verify region (verify_replay) is valid for 32 positions only. These kernels take the count from `lens` (a per-round device array that the region
# already updates), run one threadgroup per row over the capacity-sized buffers and touch only t < lens: the stream is the same at every position.
# Order = the stock one: descending score, ties by increasing index (mx.argsort is stable); -0 and +0 tie. Selection is exact (two byte-histogram passes find the threshold key, a
# third pass compacts in index order with a running count), so the result equals sort(argsort(-scores)[:k]) with the entries >= lens set to -1 and a -1 tail up to k.
_SEL_HEADER = _CONTRACT_OFF + r"""
inline uint kyojin_key16(ushort b) {
    if ((b & 0x7FFF) == 0) b = 0;
    return (b & 0x8000) ? uint(ushort(~b)) : uint(b | 0x8000);
}
template <typename T> inline uint kyojin_tkey(const device T* sc, uint i) { return kyojin_key16(as_type<ushort>(sc[i])); }
template <typename T> inline uint kyojin_bkey(const device T* sc, uint b, uint L, uint bs, thread bool& fin) {
    float m = -INFINITY;
    const uint e = min(L, (b + 1) * bs);
    for (uint t = b * bs; t < e; ++t) m = max(m, float(sc[t]));
    fin = m > -INFINITY;
    return kyojin_key16(as_type<ushort>(T(m)));
}
"""
# BLK = 0: out[u, 0..K-1] = indices of the K best t < L in increasing order, -1 after. BLK = 1: block level (bs keys per block, the last block pinned first): out[u, t] = the
# block of t is kept (rank < K among the blocks, finite block score).
_SELECT = r"""
    constexpr uint NSG = TPG / 32;
    const uint u = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const uint Tm = ip[0];
    const uint Tc = ip[1];
    const uint L = uint(min(uint(max(lens[u], 0)), Tm));
    const device T* sc = scores + ulong(u) * Tm;
    const uint nel = BLK ? (L + BS - 1) / BS : L;
    const uint nd = (BLK && nel > 0) ? nel - 1 : nel;
    const uint Kd = BLK ? K - 1 : K;
    const bool all = nd <= Kd;
    threadgroup atomic_uint hist[256];
    threadgroup uint sgtot[2][NSG];
    threadgroup uint sh[6];
    uint thr = 0, need_eq = 0;
    if (!all) {
        for (uint b = tid; b < 256; b += TPG) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < nd; i += TPG) {
            bool fin; const uint kk = BLK ? kyojin_bkey(sc, i, L, BS, fin) : kyojin_tkey(sc, i);
            atomic_fetch_add_explicit(&hist[kk >> 8], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint cum = 0, hb = 0;
            for (int b = 255; b >= 0; --b) {
                const uint h = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + h >= Kd) { hb = uint(b); break; }
                cum += h;
            }
            sh[0] = hb; sh[1] = cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = tid; b < 256; b += TPG) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hb = sh[0];
        for (uint i = tid; i < nd; i += TPG) {
            bool fin; const uint kk = BLK ? kyojin_bkey(sc, i, L, BS, fin) : kyojin_tkey(sc, i);
            if ((kk >> 8) == hb) atomic_fetch_add_explicit(&hist[kk & 255], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            const uint need = Kd - sh[1];
            uint cum = 0, lb = 0;
            for (int b = 255; b >= 0; --b) {
                const uint h = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + h >= need) { lb = uint(b); break; }
                cum += h;
            }
            sh[2] = (hb << 8) | lb; sh[3] = need - cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        thr = sh[2]; need_eq = sh[3];
    }
    if (tid == 0) { sh[4] = 0; sh[5] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0; c < nd; c += TPG) {
        const uint i = c + tid;
        const bool in = i < nd;
        uint kk = 0; bool fin = true;
        if (in) kk = BLK ? kyojin_bkey(sc, i, L, BS, fin) : kyojin_tkey(sc, i);
        const bool g = in && (all || kk > thr);
        const bool e = in && !all && kk == thr;
        const uint ep = simd_prefix_exclusive_sum(uint(e));
        const uint et = simd_sum(uint(e));
        if (lane == 0) sgtot[0][sgi] = et;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint eoff = sh[5];
        for (uint j = 0; j < sgi; ++j) eoff += sgtot[0][j];
        const bool sel = g || (e && (eoff + ep) < need_eq);
        const uint sp = simd_prefix_exclusive_sum(uint(sel));
        const uint st = simd_sum(uint(sel));
        if (lane == 0) sgtot[1][sgi] = st;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint soff = sh[4];
        for (uint j = 0; j < sgi; ++j) soff += sgtot[1][j];
        if (BLK) {
            if (in) {
                const bool keep = sel && fin;
                const uint hi = min(L, (i + 1) * BS);
                for (uint t = i * BS; t < hi; ++t) out[ulong(u) * Tc + t] = keep;
            }
        } else if (sel) {
            out[ulong(u) * K + soff + sp] = int(i);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint a = 0, b = 0;
            for (uint j = 0; j < NSG; ++j) { a += sgtot[0][j]; b += sgtot[1][j]; }
            sh[5] += a; sh[4] += b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (BLK) {
        if (tid == 0 && nel > 0) {
            bool fin; (void)kyojin_bkey(sc, nel - 1, L, BS, fin);
            const uint hi = L;
            for (uint t = (nel - 1) * BS; t < hi; ++t) out[ulong(u) * Tc + t] = fin;
        }
    } else {
        const uint total = sh[4];
        for (uint i = total + tid; i < K; i += TPG) out[ulong(u) * K + i] = -1;
    }
"""


def _sel_kernel(name: str, inputs: list):
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["out"], source=_SELECT, header=_SEL_HEADER)
    return k


def index_topk_rt(scores: mx.array, lens: mx.array, k: int) -> mx.array:
    """The `k` best columns t < lens[u] of scores [N, Tm] (an `index_scores(rt=True)` result), as int32 [N, k]: increasing column order, -1 after the last. Equals
    where(sort(argsort(-scores)[:, :k]) < lens, ., -1) of the stock chain for any width >= lens, and does not depend on Tm."""
    N, Tm = scores.shape
    tpg = 1024
    ker = _sel_kernel("kyojin_dsv4_topk_rt", ["scores", "lens", "ip"])
    return ker(inputs=[scores, lens, _u32((Tm, 1))], template=[("T", scores.dtype), ("K", k), ("BLK", 0), ("BS", 1), ("TPG", tpg)],
               grid=(N * tpg, 1, 1), threadgroup=(tpg, 1, 1), output_shapes=[(N, k)], output_dtypes=[mx.int32])[0]


def candidate_mask_rt(scores: mx.array, lens: mx.array, topk_blocks: int, block_size: int) -> mx.array:
    """Level one of the two-level top-k (`select_candidate_blocks` of the model) for scores [N, Tm] of `index_scores(rt=True)`: bool [N, Tm], valid for t < lens[u] only."""
    N, Tm = scores.shape
    tpg = 1024
    assert topk_blocks >= 2
    ker = _sel_kernel("kyojin_dsv4_cand_rt", ["scores", "lens", "ip"])
    return ker(inputs=[scores, lens, _u32((Tm, Tm))], template=[("T", scores.dtype), ("K", topk_blocks), ("BLK", 1), ("BS", block_size), ("TPG", tpg)],
               grid=(N * tpg, 1, 1), threadgroup=(tpg, 1, 1), output_shapes=[(N, Tm)], output_dtypes=[mx.bool_])[0]


# ----------------------------------------------------------------------------------------------------------------- compressor pool kernel
# softmax over the ratio slots of the score state, weighted sum of the kv state, for the users listed in `rows` (those that complete a group).
_POOL = r"""
    const uint g = thread_position_in_grid.x;
    const uint i = g / D;
    const uint d = g % D;
    const uint u = rows[i];
    const device float* a = kvs + ulong(u) * R * D + d;
    const device float* b = scs + ulong(u) * R * D + d;
    float m = -INFINITY;
    for (uint j = 0; j < R; ++j) m = max(m, b[j * D]);
    float o = 0.0f;
    if (m > -INFINITY) {
        float s = 0.0f;
        for (uint j = 0; j < R; ++j) s += exp(b[j * D] - m);
        for (uint j = 0; j < R; ++j) o = fma(a[j * D], exp(b[j * D] - m) / s, o);
    }
    out[g] = o;
"""

_kernels: dict = {}


def _kernel(name: str, inputs: list, source: str):
    k = _kernels.get(name)
    if k is None:
        k = _kernels[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=["out"], source=source, header=_CONTRACT_OFF)
    return k


_const: dict = {}
_CONST_KEEP = 4096                                           # kernel argument arrays are cached by value: the key count moves every step at long context, so the cache is bounded


def _cached(key, make):
    a = _const.get(key)
    if a is None:
        if len(_const) >= _CONST_KEEP:
            _const.clear()
        a = _const[key] = make()
    return a


def _u32(vals: tuple) -> mx.array:
    return _cached(vals, lambda: mx.array(list(vals), dtype=mx.uint32))


def _f32(vals: tuple) -> mx.array:
    return _cached(("f",) + vals, lambda: mx.array(list(vals), dtype=mx.float32))


def _identity(n: int) -> mx.array:
    a = _const.get(("id", n))
    if a is None:
        a = _const[("id", n)] = mx.arange(n, dtype=mx.uint32)
    return a


def attend(q: mx.array, win: mx.array, pool: mx.array, idx: mx.array, pos: mx.array, sink: mx.array, scale: float, max_topk: int,
           sess: mx.array | None = None, rt: mx.array | None = None, it: mx.array | None = None) -> mx.array:
    """Window + top-k pooled sparse attention with sink for N rows. q [N, H, D]; win [N, W, D] (the window ring each row sees); pool [S, C, D]
    (one pool per session); idx [N, K] int32 (-1 = no entry); pos [N] int32 (position of the query token); sink [H] fp32; sess [N] uint32:
    the pool of row u is pool[sess[u]] (None = row u owns pool u). Returns [N, H, D] in q's dtype.
    `max_topk` bounds K (threadgroup memory), it is a property of the model, not of the batch. rt / it [N, RD] fp32: the forward / inverse rope
    tables of the rows (Plan.attn_rope); with rt the q rope is applied in the kernel's q load (q is then the pre-rope wq_b output), with it the
    inverse rope rides in the merge epilogue."""
    N, H, D = q.shape
    W = win.shape[1]
    K = idx.shape[1]
    assert D % 128 == 0 and K <= max_topk and q.dtype == win.dtype == pool.dtype, (q.shape, win.shape, pool.shape, K, max_topk)
    maxm = W + max_topk
    hg = 4
    while hg > 1 and (hg * maxm + maxm + hg) * 4 > 30 * 1024:
        hg //= 2
    assert H % hg == 0
    sess = _identity(N) if sess is None else sess
    M = W + K
    assert rt is None or (D - rt.shape[-1]) % (D // 32) == 0, "q rope tail must start on a lane boundary"
    assert it is None or (D - it.shape[-1]) % (D // 128) == 0, "inverse rope tail must start on a thread boundary"
    qs = "" if rt is None else "_q"
    rd = () if rt is None else (("RD", rt.shape[-1]),)
    if SPLIT_MS > 0:
        hgs = min(SPLIT_HG, H)
        while H % hgs:
            hgs //= 2
        ms = SPLIT_MS
        sp = -(-M // ms)
        fk = _fk() if rt is not None else ""
        if ATTN_MERGE_TAIL:
            nm = "kyojin_dsv4_attn_part_mt" + qs + ("" if it is None else "_i") + fk
            kt = _kernels.get(nm)
            if kt is None:
                src = merge_tail_source(_ATTN_PART if rt is None else rope_src(_ATTN_PART_Q), _ATTN_MERGE if it is None else rope_src(_ATTN_MERGE_Q))
                kt = _kernels[nm] = mx.fast.metal_kernel(
                    name=nm, input_names=["q", "win", "pool", "idx", "pos", "sess", "ip", "fp"] + ([] if rt is None else ["rt"]) + ["sink"] + ([] if it is None else ["it"]) + ["ctr"],
                    output_names=["part_o", "ml", "out"], source=src, header=_CONTRACT_OFF)
            ngrp = N * (H // hgs)
            return kt(inputs=[q, win, pool, idx, pos, sess, _u32((K, pool.shape[1])), _f32((scale,))] + ([] if rt is None else [rt]) + [sink] + ([] if it is None else [it]) + [_attn_ctr(ngrp)],
                      template=[("T", q.dtype), ("H", H), ("HG", hgs), ("D", D), ("W", W), ("MS", ms), ("SPLIT", sp)] + (list(rd) if rt is not None else ([] if it is None else [("RD", it.shape[-1])])),
                      grid=(ngrp * sp * 128, 1, 1), threadgroup=(128, 1, 1),
                      output_shapes=[(N, H, sp, D), (N, H, sp, 2), (N, H, D)], output_dtypes=[mx.float32, mx.float32, q.dtype])[2]
        kp = _kernels.get("kyojin_dsv4_attn_part" + qs + fk)
        if kp is None:
            kp = _kernels["kyojin_dsv4_attn_part" + qs + fk] = mx.fast.metal_kernel(
                name="kyojin_dsv4_attn_part" + qs + fk, input_names=["q", "win", "pool", "idx", "pos", "sess", "ip", "fp"] + ([] if rt is None else ["rt"]),
                output_names=["part_o", "ml"], source=_ATTN_PART if rt is None else rope_src(_ATTN_PART_Q), header=_CONTRACT_OFF)
        km = _kernels.get("kyojin_dsv4_attn_merge" + qs + fk)
        if km is None:
            km = _kernels["kyojin_dsv4_attn_merge" + qs + fk] = mx.fast.metal_kernel(
                name="kyojin_dsv4_attn_merge" + qs + fk, input_names=["part_o", "ml", "sink"] + ([] if it is None else ["it"]),
                output_names=["out"], source=_ATTN_MERGE if it is None else rope_src(_ATTN_MERGE_Q), header=_CONTRACT_OFF)
        po, ml = kp(inputs=[q, win, pool, idx, pos, sess, _u32((K, pool.shape[1])), _f32((scale,))] + ([] if rt is None else [rt]),
                    template=[("T", q.dtype), ("H", H), ("HG", hgs), ("D", D), ("W", W), ("MS", ms), ("SPLIT", sp)] + list(rd),
                    grid=(N * (H // hgs) * sp * 128, 1, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(N, H, sp, D), (N, H, sp, 2)], output_dtypes=[mx.float32, mx.float32])
        return km(inputs=[po, ml, sink] + ([] if it is None else [it]),
                  template=[("T", q.dtype), ("H", H), ("D", D), ("SPLIT", sp)] + ([] if it is None else [("RD", it.shape[-1])]),
                  grid=(N * H * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(N, H, D)], output_dtypes=[q.dtype])[0]
    ker = _kernel("kyojin_dsv4_attn" + qs + (_fk() if rt is not None else ""), ["q", "win", "pool", "idx", "pos", "sink", "sess", "ip", "fp"] + ([] if rt is None else ["rt"]),
                  _ATTN if rt is None else rope_src(_ATTN_Q))
    return ker(inputs=[q, win, pool, idx, pos, sink, sess, _u32((K, pool.shape[1])), _f32((scale,))] + ([] if rt is None else [rt]),
               template=[("T", q.dtype), ("H", H), ("HG", hg), ("D", D), ("W", W), ("MAXM", maxm)] + list(rd),
               grid=(N * (H // hg) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(N, H, D)], output_dtypes=[q.dtype])[0]


def index_scores(q: mx.array, keys: mx.array, wts: mx.array, lens: mx.array, t_max: int, cand: mx.array | None = None,
                 sess: mx.array | None = None, rt: bool = False) -> mx.array:
    """Head-weighted rectified index scores. q [N, NH, DI]; keys [S, C, DI] (one key buffer per session, row u reads keys[sess[u]], None =
    keys[u]); wts [N, NH]; lens [N] int32 (valid keys per row); t_max = columns of the result (>= max lens, <= C); cand [N, >= t_max] bool or
    None. Returns [N, t_max] in q's dtype, -inf where invalid. `rt` (runtime width): t_max is the capacity and the columns past lens[u] are NOT
    written (undefined): the consumers (index_topk_rt, candidate_mask_rt) read only t < lens[u], so the stream does not move with the position."""
    N, NH, DI = q.shape
    assert q.dtype == keys.dtype == wts.dtype and DI % 8 == 0 and t_max <= keys.shape[1]
    tb = 128
    ker = _kernel("kyojin_dsv4_iscore", ["q", "keys", "wts", "lens", "cand", "sess", "ip"], _ISCORE)
    sess = _identity(N) if sess is None else sess
    c = cand if cand is not None else mx.zeros((1,), dtype=mx.bool_)
    tc = cand.shape[-1] if cand is not None else 1
    assert cand is None or tc >= t_max
    if ISCORE_KB and NH % 4 == 0 and t_max >= ISCORE_KB_MIN:
        LC.HITS["iscore_kb"] += 1
        ker = _kernel("kyojin_dsv4_iscore_kb", ["q", "keys", "wts", "lens", "cand", "sess", "ip"], _ISCORE_KB)
        return ker(inputs=[q, keys, wts, lens, c, sess, _u32((t_max, keys.shape[1], tc))],
                   template=[("T", q.dtype), ("NH", NH), ("DI", DI), ("CAND", cand is not None), ("RT", rt)],
                   grid=(N * (-(-t_max // 128)) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(N, t_max)], output_dtypes=[q.dtype])[0]
    return ker(inputs=[q, keys, wts, lens, c, sess, _u32((t_max, keys.shape[1], tc))],
               template=[("T", q.dtype), ("NH", NH), ("DI", DI), ("TB", tb), ("CAND", cand is not None), ("RT", rt)],
               grid=(N * (-(-t_max // tb)) * tb, 1, 1), threadgroup=(tb, 1, 1), output_shapes=[(N, t_max)], output_dtypes=[q.dtype])[0]


def pool_groups(kvs: mx.array, scs: mx.array, rows: mx.array) -> mx.array:
    """Softmax-weighted pooling of the compressor state for the users in `rows`. kvs / scs [N, R, D] fp32; rows [n] int32 -> [n, D] fp32."""
    N, R, D = kvs.shape
    n = rows.shape[0]
    ker = _kernel("kyojin_dsv4_pool", ["kvs", "scs", "rows"], _POOL)
    return ker(inputs=[kvs, scs, rows], template=[("R", R), ("D", D)], grid=(n * D, 1, 1), threadgroup=(min(256, D), 1, 1),
               output_shapes=[(n, D)], output_dtypes=[mx.float32])[0]


# The same pool over the state a pass would snapshot: the base state (kvs / scs [S, R, D], the committed slots) plus the pass's own compressor rows
# (kvc / scc [n, D], row t's slot is pos[t] % R, the last row of the session at or before the pooled row wins). `_POOL`'s arithmetic over the same slot
# values, so a pooled row is bit-identical to pool_groups of the snapshot it replaces; no [n, R, D] copy is built.
_POOL_OV = r"""
    const uint g = thread_position_in_grid.x;
    const uint i = g / D;
    const uint d = g % D;
    const uint u = rows[i];
    const uint ss = sess[u];
    const uint t0 = fst[u];
    float m = -INFINITY;
    for (uint j = 0; j < R; ++j) {
        int src = -1;
        for (int t = int(u); t >= int(t0); --t) if (uint(pos[t]) % R == j) { src = t; break; }
        const float b = src >= 0 ? scc[ulong(src) * D + d] : scs[(ulong(ss) * R + j) * D + d];
        m = max(m, b);
    }
    float o = 0.0f;
    if (m > -INFINITY) {
        float s = 0.0f;
        for (uint j = 0; j < R; ++j) {
            int src = -1;
            for (int t = int(u); t >= int(t0); --t) if (uint(pos[t]) % R == j) { src = t; break; }
            const float b = src >= 0 ? scc[ulong(src) * D + d] : scs[(ulong(ss) * R + j) * D + d];
            s += exp(b - m);
        }
        for (uint j = 0; j < R; ++j) {
            int src = -1;
            for (int t = int(u); t >= int(t0); --t) if (uint(pos[t]) % R == j) { src = t; break; }
            const float a = src >= 0 ? kvc[ulong(src) * D + d] : kvs[(ulong(ss) * R + j) * D + d];
            const float b = src >= 0 ? scc[ulong(src) * D + d] : scs[(ulong(ss) * R + j) * D + d];
            o = fma(a, exp(b - m) / s, o);
        }
    }
    out[g] = o;
"""


def pool_overlay(kvs: mx.array, scs: mx.array, kvc: mx.array, scc: mx.array, rows: mx.array, sess: mx.array, fst: mx.array, pos: mx.array) -> mx.array:
    """Pooling of `rows` (row indices of the pass) over the state the pass would have snapshotted: base kvs / scs [S, R, D], the pass's compressor rows
    kvc / scc [n, D], sess / fst / pos int32 [n] (session, first row of the session in the pass, position) -> [len(rows), D] fp32."""
    S, R, D = kvs.shape
    n = rows.shape[0]
    ker = _kernel("kyojin_dsv4_pool_ov", ["kvs", "scs", "kvc", "scc", "rows", "sess", "fst", "pos"], _POOL_OV)
    return ker(inputs=[kvs, scs, kvc, scc, rows, sess, fst, pos], template=[("R", R), ("D", D)], grid=(n * D, 1, 1),
               threadgroup=(min(256, D), 1, 1), output_shapes=[(n, D)], output_dtypes=[mx.float32])[0]



# ----------------------------------------------------------------------------------------------------------------- reference (stock ops)
def attend_stock(q, win, pool, idx, pos, sink, scale):
    """The model's own sparse attention (stock ops, one user per call) on the same stacked inputs: reference for tests and the negative
    control of the batched kernel. Returns [N, H, D]."""
    from mlx_vlm.models.deepseek_v4.language import _sparse_pooled_attention
    N, H, D = q.shape
    W = win.shape[1]
    outs = []
    for u, p in enumerate(int(v) for v in pos.tolist()):
        base = max(0, p - W + 1)
        part = mx.take(win[u:u + 1], mx.arange(base, p + 1) % W, axis=1)
        positions = mx.arange(base, p + 1)
        valid = ((positions <= p) & (positions > p - W))[None, None, None]
        row = idx[u:u + 1, None, :]
        o = _sparse_pooled_attention(q[u:u + 1, :, None, :].astype(mx.float32), part[:, None], pool[u:u + 1], row, valid,
                                     (row != -1)[:, None], scale, sink.astype(mx.float32))
        outs.append(o[:, :, 0, :].astype(q.dtype))
    return mx.concatenate(outs, axis=0)


# ----------------------------------------------------------------------------------------------------------------- stacked state
def _lang():
    L = _A.lang()
    return L


class Plan:
    """Host-side facts of one pass: the positions of its rows, which rows complete a compression group, how rows map to sessions.
    Session s owns the rows first[s] .. first[s] + counts[s] - 1, consecutive positions p0[s] + i. A plain decode step is the case of one row
    per session (`plain`): row u is session u.

    `slots` (flag mu_keep): the slot of each session's state in the stacked buffers, `NS` the number of slots. Without it session s lives in
    slot s and NS == S, which is the stack of every other lever (and what `sess`/`sess_arr`/`sess_idx` mean: the buffer slot of a row)."""

    def __init__(self, pos: list, counts: list | None = None, slots: list | None = None, nslots: int | None = None):
        n = len(pos)
        counts = [1] * n if counts is None else list(counts)
        assert sum(counts) == n and min(counts) >= 1, (counts, n)
        S = len(counts)
        slots = list(range(S)) if slots is None else list(slots)
        assert len(slots) == S and len(set(slots)) == S and min(slots) >= 0, (slots, S)
        self.key = (tuple(pos), tuple(counts), None if slots == list(range(S)) else tuple(slots), None if nslots is None else int(nslots))
        self.pos, self.counts, self.n, self.S = list(pos), counts, n, S
        self.slots = slots
        self.NS = S if nslots is None else max(int(nslots), max(slots) + 1)
        self.N = n
        self.plain = all(c == 1 for c in counts)
        self.first = [sum(counts[:s]) for s in range(S)]
        self.sess = [slots[s] for s, c in enumerate(counts) for _ in range(c)]
        self.rank = [s for s, c in enumerate(counts) for _ in range(c)]    # the session of a row in this pass; `sess` is its buffer slot, `first` is per session
        self.p0 = [self.pos[f] for f in self.first]
        self.pos_arr = mx.array(self.pos, dtype=mx.int32)
        self.rows_all = mx.arange(n, dtype=mx.int32)
        self.sess_arr = mx.array(self.sess, dtype=mx.uint32)             # kernel input: pool / key buffer of a row
        self.sess_idx = mx.array(self.sess, dtype=mx.int32)              # gather index: the state of a row's session
        self.slot_arr = mx.array(self.slots, dtype=mx.int32)             # session -> slot (one row per session) for the commit scatter
        self._first_arr = None
        self._acc = {}
        self._slot, self._emit, self._cs, self._dummy, self._ov = {}, {}, {}, {}, {}
        self._wmap, self._arope = {}, {}

    def slot(self, m: int) -> mx.array:
        a = self._slot.get(m)
        if a is None:
            a = self._slot[m] = mx.array([p % m for p in self.pos], dtype=mx.int32)
        return a

    def first_arr(self) -> mx.array:
        """Per row the first row of its session in the pass (int32 [n]): the fused row core reads the kv rows first[r] .. r of row r's session."""
        if self._first_arr is None:
            self._first_arr = mx.array([self.first[s] for s in self.rank], dtype=mx.int32)
        return self._first_arr

    def accepted(self, p_last: list, r: int) -> SimpleNamespace:
        """The rows a commit with `p_last[s]` (position of the last accepted row of session s) keeps: row / session / group-slot arrays for the
        in-place state writes of a pass (STATE_NOCOPY), one set per (r, accepted counts). One entry per (session, slot): the compressor state is a
        ring of r slots, so with more than r rows in a session the later row wins, exactly the overlay the snapshot applies (a scatter with a
        repeated index would leave that to the backend)."""
        key = (tuple(p_last), r)
        e = self._acc.get(key)
        if e is None:
            seen = {}
            for s, p in enumerate(p_last):
                for i in range(self.first[s], self.first[s] + p - self.p0[s] + 1):
                    seen[(s, self.pos[i] % r)] = i
            rows = [seen[k] for k in sorted(seen, key=lambda k: seen[k])]
            e = self._acc[key] = SimpleNamespace(
                n=len(rows), rows=rows, idx=mx.array(rows, dtype=mx.int32), prefix=rows == list(range(len(rows))),
                sess=mx.array([self.sess[i] for i in rows], dtype=mx.int32),
                slots=mx.array([self.pos[i] % r for i in rows], dtype=mx.int32))
        return e

    def overlay(self, m: int) -> SimpleNamespace:
        """What a row adds to its session's ring of `m` slots: the new entries of the rows up to itself (slot p % m; two rows of a session
        sharing a slot: the later one wins). rows / slots / src index arrays, src None when row j writes its own entry (plain step)."""
        o = self._ov.get(m)
        if o is None:
            rows, slots, src = [], [], []
            for j in range(self.n):
                seen = {}
                for i in range(self.first[self.rank[j]], j + 1):
                    seen[self.pos[i] % m] = i
                for sl, i in seen.items():
                    rows.append(j), slots.append(sl), src.append(i)
            o = self._ov[m] = SimpleNamespace(
                rows=mx.array(rows, dtype=mx.int32), slots=mx.array(slots, dtype=mx.int32),
                src=None if src == rows else mx.array(src, dtype=mx.int32))
        return o

    def rowof(self) -> mx.array:
        """int32 [NS]: the row that owns each slot of the stacked buffers in a plain pass (one row per session), -1 = no row (attn_core_fuse kv chain)."""
        a = self._wmap.get("rowof")
        if a is None:
            ro = [-1] * self.NS
            for j, sl in enumerate(self.sess):
                ro[sl] = j
            a = self._wmap["rowof"] = mx.array(ro, dtype=mx.int32)
        return a

    def wmap(self, m: int) -> mx.array:
        """Per (row, slot) the kv row that slot holds after the rows up to this one, int32 [n * m] (-1 = keep the base ring). The overlay scatter
        as one dense lookup for the kv-chain kernel (one kv row may write several snapshots)."""
        a = self._wmap.get(m)
        if a is None:
            wm = [-1] * (self.n * m)
            for j in range(self.n):
                for i in range(self.first[self.rank[j]], j + 1):
                    wm[j * m + self.pos[i] % m] = i
            a = self._wmap[m] = mx.array(wm, dtype=mx.int32)
        return a

    def attn_rope(self, rope):
        """(forward, inverse) rope tables [n, rope.dims] of this pass's positions, taken from mx.fast.rope itself (fuse_attn.rope_tables /
        rope_range: the same cos / sin the stock op uses), one row per row."""
        e = self._arope.get(id(rope))
        if e is None:
            from . import fuse_attn as FA
            if all(self.pos[i] == self.pos[0] + i for i in range(self.n)):          # a pass of consecutive positions: one rope call for all rows
                f, i = FA.rope_range(rope, self.pos[0], self.n)
            else:
                rows = [FA.rope_tables(rope, p) for p in self.pos]
                f, i = mx.stack([x[0] for x in rows]), mx.stack([x[1] for x in rows])
            e = self._arope[id(rope)] = (rope, f, i)                                # keeps the module alive: the id key stays unique
        return e[1], e[2]

    def emit(self, r: int) -> SimpleNamespace:
        """Rows completing a group of `r` tokens on this pass: row indices, session of each, write index (= row count - 1), row counts of
        every row after its own token."""
        e = self._emit.get(r)
        if e is None:
            lens = [(p + 1) // r for p in self.pos]
            rows = [u for u, p in enumerate(self.pos) if (p + 1) % r == 0]
            widx = [lens[u] - 1 for u in rows]
            e = self._emit[r] = SimpleNamespace(
                lens=lens, tmax=max(lens), lens_arr=mx.array(lens, dtype=mx.int32), rows=rows, n=len(rows),
                rows_arr=mx.array(rows, dtype=mx.int32) if rows else None, widx=widx, need=max(widx) + 1 if rows else 0,
                widx_arr=mx.array(widx, dtype=mx.int32) if rows else None,
                sess_arr=mx.array([self.sess[u] for u in rows], dtype=mx.int32) if rows else None,
                positions=[(lens[u] - 1) * r for u in rows])
        return e

    def cs(self, positions: list, theta: float, yarn: tuple, rope_dim: int, shape: tuple):
        """cos / sin of the rotary at `positions`, reshaped to `shape` (+ the pair axis)."""
        key = (tuple(positions), theta, yarn, rope_dim, shape)
        c = self._cs.get(key)
        if c is None:
            L = _lang()
            if _CANON_POOLS:                                                  # the whole cached table (capacity multiple of 256): no slice length that moves with max(positions)
                cos, sin = L._cached_index_cos_sin(((max(positions) + 1 + 255) // 256) * 256, rope_dim, theta, yarn)
            else:
                cos, sin = L._index_cos_sin(max(positions) + 1, rope_dim, theta, yarn)
            pa = mx.array(positions, dtype=mx.int32)
            c = self._cs[key] = (cos[pa].reshape(*shape, -1), sin[pa].reshape(*shape, -1))
        return c

    def dummy(self, D: int, dtype):
        d = self._dummy.get((D, dtype))
        if d is None:
            d = self._dummy[(D, dtype)] = (mx.zeros((self.n, 1, D), dtype=dtype), mx.full((self.n, 1), -1, dtype=mx.int32))
        return d


# Explicit state writes (one switch): every buffer the verify forward writes is a fresh copy of the committed state (never an in-place write into a buffer MLX may donate,
# whose copy-on-write behaviour depends on who else holds a reference). Written sites: the window ring snapshot + overlay scatter, the compressor kv / score snapshots, the
# compressed pool and indexer key pool scatters (_put). Default off; KYOJIN_VERIFY_REPLAY turns it on (the recorded command stream must not depend on reference counts).
TMAX_BUCKET = 32
_CANON_POOLS = os.environ.get("KYOJIN_STATE_EXPLICIT", os.environ.get("KYOJIN_VERIFY_REPLAY", "0")) not in ("0", "")
# Runtime-width selection (index_topk_rt / candidate_mask_rt, see above): with the explicit state, the indexer scores the whole capacity-sized key buffer (columns past the count are not
# computed), the top-k has the model's fixed width and the pooled count is read from a device array: nothing in the stream moves with the bucketed count. Same selection as the stock chain.
RT_SELECT = _CANON_POOLS and os.environ.get("KYOJIN_RT_SELECT", "1") not in ("0", "")
# Fast multi-threadgroup selection (kyojin_mlx.idxfast) with the candidate mask fused into the select, same runtime-width inputs and the same
# selection as the rt kernels above. Opt-in (KYOJIN_IDX_FAST=1, default off until the real-pack gate); independent of the explicit state.
IDX_FAST = os.environ.get("KYOJIN_IDX_FAST", os.environ.get("KYOJIN_LONGCTX", "0")) not in ("0", "")
# Plain steps take the fast select from this many indexed keys (max over the sessions); below it the stock / rt chain (fewer launches wins at short
# context: Ultra microbench tools/longctx_topk_bench.py, serial latency crosses over between 8K and 32K keys). The explicit state ignores it (no switch in the stream).
IDX_FAST_MIN = int(os.environ.get("KYOJIN_IDX_FAST_MIN", "16384"))
# Scores + select + candidate mask in ONE launch (kyojin_mlx.idxfuse, KYOJIN_IDX_FUSE=1, default off): the last score threadgroup of a row runs the select. Up to
# KYOJIN_IDX_FUSE_MAX rows x keys (default 32768: score-compute bound above, the stock chain wins; idxfast wins at long n); takes precedence over IDX_FAST in that range. Bit-identical to the rt chain.
IDX_FUSE = os.environ.get("KYOJIN_IDX_FUSE", "1") not in ("0", "")
IDX_FUSE_MAX = int(os.environ.get("KYOJIN_IDX_FUSE_MAX", "32768"))
# KYOJIN_CIDXFUSE=1 (default off): at depth, scores + chunk-tree top-k + candidate mask in one launch (kyojin_mlx.cidxfuse); see its header for the exactness statement.
CIDXFUSE = os.environ.get("KYOJIN_CIDXFUSE", "0") not in ("0", "")
# Score-only lane-per-head index kernel (kyojin_mlx.idxfuse.scores_fast) instead of the one-thread-per-key kernel, then the usual exact
# select: bit-identical scores (same FMA and head order), a wider grid, worth it where the fused select is gated off and the width is small
# enough that the extra lanes pay (in-chain micro-bench: 8K-32K keys ahead, 64K and above behind).
IDX_SCORE_FAST = os.environ.get("KYOJIN_IDX_SCORE_FAST", "1") not in ("0", "")
IDX_SCORE_FAST_MAX = int(os.environ.get("KYOJIN_IDX_SCORE_FAST_MAX", "32768"))
# The rows of a batched pass (verify round, multi-session step) score on the simdgroup-matrix kernel the solo decode step already serves
# (dsa_decode.scores_rows: one launch over every row) instead of the one-thread-per-key fp32 chain. Gated on keys: at short context the chain is
# cheap and the wider matrix grid does not pay. Rounding-level against _ISCORE (4 partial head sums vs the 32-step order), so it is a drift lever.
ISCORE_MMA = os.environ.get("KYOJIN_ISCORE_MMA", "1") not in ("0", "")
ISCORE_MMA_MIN = int(os.environ.get("KYOJIN_ISCORE_MMA_MIN", "262144"))
# One launch for the kv chain of a pass (kv_norm + rope tail + fp8 round trip + window ring scatter), and the q / inverse rope of the core in the
# kernels that already read / write those dims. Both bit-identical to the stock ops; see _KV_CHAIN and the _Q_ROPE / _MERGE_INV substitutions.
CPROJ = os.environ.get("KYOJIN_DEC_CPROJ", "0") not in ("0", "")          # compressor wkv + wgate of a kv-source layer: x32 cast + two f32 matmuls -> one launch (stagek2.xgemv)
KV_CHAIN = os.environ.get("KYOJIN_KV_CHAIN", "1") not in ("0", "")
# attn_core_fuse: one launch less per layer in the plain decode attention core, same bits (default off): the kv chain writes the stacked window buffer the commit
# keeps, so the commit's slot scatter (a whole-buffer copy when the stack has spare slots) is gone. See _KV_CHAIN_STK. (A part + merge fold through a
# last-threadgroup counter was measured 25-45 us per layer SLOWER and dropped.)
ATTN_CORE_FUSE = os.environ.get("KYOJIN_ATTN_CORE_FUSE", "0") not in ("0", "")
ROPE_FUSE = os.environ.get("KYOJIN_ROPE_FUSE", "1") not in ("0", "")
# The rows of a served pass (RowsCache with a pass open) through the fused row core: the window ring and the attention of row j read the session's
# committed ring plus the pass's own kv rows (fuse_attn.attend_rows_m / kv_apply_m) instead of a per-row copy of the ring, and the ring a commit keeps is
# built from the base ring at the commit (a rejected row's ring is never written anywhere). Off = the batched core as before.
ROWS_CORE_FUSE = os.environ.get("KYOJIN_ROWS_CORE_FUSE", "0") not in ("0", "")
# The compressor state (kvs / scs) of a pass: no per-row snapshot. The pool of a closing group is read through the base state plus the pass's own
# compressor rows in one kernel (pool_overlay) and the accepted prefix of those rows is written into the state at the commit (bit-identical to the
# snapshot + overlay + pick of the batched core). Off = the per-layer snapshot as before.
STATE_NOCOPY = os.environ.get("KYOJIN_ROWS_STATE_NOCOPY", "0") not in ("0", "")
# rope_fma (KYOJIN_ROPE_FMA, default 3 = auto; 0 = the plain products below). WHY: the fused rope kernels run under `fp contract(off)`, so
# `x1 * c - x2 * s` is two roundings of the products, while the stock op (mx.fast.rope, kernels `rope*_float16` of the MLX metallib) is built with clang's
# default contraction and its AIR holds, for the forward rotation, `fmuladd(x1, c, (-s) * x2)` and `fmuladd(x1, s, c * x2)` (tools/air_rope_form.py reads it
# out of the shipped mlx.metallib; tests/data/air_rope_single_float16.ll is the excerpt). That is form 1 below: the FIRST product of each expression is fused,
# the second is rounded. The fp32 pair then differs in the last bit from the plain form on about 1 pair in 3, and a fp16 result flips on about 1 element
# in 10^4 (kv rope, q rope, inverse rope of every layer), which is enough to move an emitted-token md5 over a few hundred tokens.
# 0 = plain products (the old text); 1 = form 1 (what the shipped MLX does); 2 = second product fused (kept for the probe); 3 = auto: the form the stock
# op is MEASURED to use on this device, resolved once when the flag is read (0 if no form matches). The value stored in ROPE_FMA is always the resolved
# form 0 / 1 / 2, part of every kernel key and name.
_ROPE_PLAIN = ("x1 * c - x2 * s", "x1 * s + x2 * c")
_ROPE_FORMS = {1: ("fma(x1, c, -(x2 * s))", "fma(x1, s, x2 * c)"), 2: ("fma(-x2, s, x1 * c)", "fma(x2, c, x1 * s)")}
ROPE_DETECT: dict = {}                                   # last detect_rope_form result: {"form": int | None, "bad": {form: mismatches}, "n": elements}


def detect_rope_form(n: int = 4096, dims: int = 64, pos0: int = 1237, stream=None) -> int | None:
    """The fma form (0 plain, 1, 2) whose fp32 result equals mx.fast.rope bit for bit on random fp32 pairs, or None when none or several match.
    fp32 in and out (no fp16 rounding to hide a last-bit difference); the cos / sin come from the same op on a [1, 0] probe (y1 = cos, y2 = sin exactly
    in every form). The fp32 and fp16 kernels share their arithmetic (same AIR dataflow), so this decides the fp16 calls too. `stream` = device to ask
    (default: the default device)."""
    import numpy as np
    from .ref_numpy import rope_pairs_f32
    f = (10000.0 ** (np.arange(0, dims, 2, dtype=np.float32) / dims)).astype(np.float32)
    kw = dict(traditional=True, base=None, scale=1.0, offset=pos0, freqs=mx.array(f))
    if stream is not None:
        kw["stream"] = stream
    rng = np.random.default_rng(12345)
    x = (rng.standard_normal((1, 1, n, dims)) * np.exp(rng.uniform(-2, 2, (1, 1, n, dims)))).astype(np.float32)
    probe = np.tile(np.array([1.0, 0.0], np.float32), (1, 1, n, dims // 2))
    outs = []
    for a in (probe, x):
        y = mx.fast.rope(mx.array(a), dims, **kw)
        mx.eval(y)
        outs.append(np.array(y))
    tab, got = outs
    c, s = tab[..., 0::2], tab[..., 1::2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    bad = {}
    for form in (0, 1, 2):
        r1, r2 = rope_pairs_f32(x1, x2, c, s, form)
        bad[form] = int((r1.view(np.uint32) != got[..., 0::2].view(np.uint32)).sum() + (r2.view(np.uint32) != got[..., 1::2].view(np.uint32)).sum())
    ok = [f_ for f_, b in bad.items() if b == 0]
    ROPE_DETECT.clear()
    ROPE_DETECT.update(form=ok[0] if len(ok) == 1 else None, bad=bad, n=int(x.size))
    return ROPE_DETECT["form"]


def resolve_rope_fma(v: int) -> int:
    """0 / 1 / 2 as given; 3 = auto (detect_rope_form on the default device; no Metal, or no unique match = 0, the old text)."""
    if v != 3:
        return v
    if not mx.metal.is_available():
        return 0
    form = detect_rope_form()
    return 0 if form is None else form


ROPE_FMA = resolve_rope_fma(int(os.environ.get("KYOJIN_ROPE_FMA", "3")))


def rope_src(src: str, form: int | None = None) -> str:
    """`src` with the rope-tail products of the fused kernels in the fma form `form` (default ROPE_FMA); 0 = unchanged text."""
    form = ROPE_FMA if form is None else form
    if not form:
        return src
    a, b = _ROPE_FORMS[form]
    return src.replace(_ROPE_PLAIN[0], a).replace(_ROPE_PLAIN[1], b)


def _fk() -> str:
    """Kernel name / cache key suffix of the rope form in effect."""
    return f"_fma{ROPE_FMA}" if ROPE_FMA else ""


_rows_cls = None


def _rows_cache_class():
    global _rows_cls
    if _rows_cls is None:
        Base = _A.cls("batch_cache")

        class RowsCache(Base):
            """BatchDeepseekV41Cache whose attention state lives stacked (see the module docstring) while the decode core runs. The per-user
            caches in `rows` keep the offsets and the Engram state; their attention arrays are refreshed by `sync_rows` when something
            reads them (extract, state, filter, extend).

            One pass = one forward over `n` rows (the batch dimension of the model), the rows of session s being consecutive positions.
            Plain decode (`_counts` None): one row per session, the state is committed by the core itself. `begin(counts)` starts a pass of
            counts[s] rows for session s; the core leaves the state it computed in `pend` and `commit(accepted)` keeps, per session, the
            first accepted[s] + 1 rows of it (nothing to undo for the others: their state is never adopted)."""

            _counts = None

            def stacked_arrays(self) -> list:
                pend = [a for pd in getattr(self, "pend", {}).values() for v in vars(pd).values() if isinstance(v, (mx.array, SP.PackedStack)) for a in SP.arrays_of(v)]
                return [a for lst in (self.win, self.cmp, self.keys, self.kvs, self.scs) for v in lst if v is not None for a in SP.arrays_of(v)] + pend

            def begin(self, counts: list) -> None:
                assert len(counts) == len(self.rows) and self._counts is None, "a pass is open"
                self._counts = list(counts)
                self.pend = {}

            def advance(self, width):
                if self._counts is None:
                    Base.advance(self, width)

            def commit_layer(self, l: int, last: list | None, p_last: list | None = None, idx=None, slots: list | None = None) -> None:
                """Adopt the pending state of layer l. last[s] = row (of the pass) whose state session s keeps (None: row s, plain step);
                p_last[s] = position of that row. `slots` (mu_keep): the slot of session s in the stacked buffers (None = s)."""
                pd = self.pend.pop(l, None)
                if pd is None:
                    return
                self._dirty = True
                S = getattr(self, "S_cap", len(self.rows))
                if last is not None and idx is None:
                    idx = mx.array(last, dtype=mx.int32)
                pick = (lambda a: a) if last is None else (lambda a, i=idx: a[i])
                if getattr(pd, "win_stk", None) is not None:                        # attn_core_fuse: the kv chain already wrote the stacked buffer
                    self.win[l] = pd.win_stk
                elif pd.win is not None:
                    self.win[l] = _keep_slots(self.win[l], pick(pd.win), slots)
                if pd.kvr is not None and p_last is not None:                      # STATE_NOCOPY: the compressor state of the kept rows, written once
                    plan, r = pd.kvr
                    e = plan.accepted(p_last, r)
                    kvc, scc = pd.kvrow, pd.scrow
                    kb, sb = self.kvs[l], self.scs[l]
                    if kb is None:
                        kb = mx.zeros((S, r, kvc.shape[-1]), dtype=kvc.dtype)
                    if sb is None:
                        sb = mx.full((S, r, scc.shape[-1]), -math.inf, dtype=scc.dtype)
                    self.kvs[l] = _write_rows(kb, e, kvc)
                    self.scs[l] = _write_rows(sb, e, scc)
                if pd.kvs is not None:
                    self.kvs[l] = _keep_slots(self.kvs[l], pick(pd.kvs), slots, fill=0.0)
                    self.scs[l] = _keep_slots(self.scs[l], pick(pd.scs), slots, fill=-math.inf)
                if pd.cmp is not None:
                    cnt = [(p + 1) // pd.r for p in p_last] if p_last is not None else pd.cnt
                    self.cnt[l] = _per_slot(cnt, slots, S)
                    self.cmp[l] = CD.settle(pd.cmp, self.cnt[l]) if CD.is_view(pd.cmp) else pd.cmp      # cdecpack: the valid count follows the commit
                if pd.keys is not None:
                    kcnt = [(p + 1) // pd.r for p in p_last] if p_last is not None else pd.kcnt
                    self.kcnt[l] = _per_slot(kcnt, slots, S)
                    self.keys[l] = CD.settle(pd.keys, self.kcnt[l]) if CD.is_view(pd.keys) else pd.keys
                dec = getattr(pd, "dec", None)                                  # decsel: publish the selection of the row the commit keeps
                if dec is not None:
                    DCS.publish(getattr(self, "dec_sel", None), l, self.rows, dec, last)

            def commit(self, accepted: list, plan: Plan, engram: list | None = None) -> None:
                """End of a pass: session s keeps its first accepted[s] + 1 rows. engram[s]: the session's new token history (None = untouched)."""
                last = [plan.first[s] + a for s, a in enumerate(accepted)]
                p_last = [plan.p0[s] + a for s, a in enumerate(accepted)]
                ident = last == list(range(len(last)))
                from . import hostsync
                idx = mx.array(last, dtype=mx.int32) if hostsync.enabled() and not ident else None      # one index array for every layer of the commit
                for l in list(self.pend):
                    self.commit_layer(l, None if ident else last, p_last, idx, plan.slots)
                for s, a in enumerate(accepted):
                    row = self.rows[s]
                    row.offset += a + 1
                    if engram is not None and engram[s] is not None:
                        row.engram = engram[s][:, :row.offset]
                self._counts = None
                self._dirty = True

            def sync_row(self, u: int) -> None:
                """Write one row's stacked state back into its own cache (`rows[u]`): only slot slots[u] of the stacked buffers is read, the
                rows of the other members are neither read nor written (a leave with mu_keep). The pool / key layers go back through
                `fp4pool.write_back`, which packs only the rows the stack appended (sync_tail)."""
                row = self.rows[u]
                s = u if self.slots is None else self.slots[u]
                for l in range(len(self.win)):
                    if self.win[l] is not None:
                        row.window[l] = self.win[l][s:s + 1][...]
                        HITS["copy_bytes"] += self.win[l][s:s + 1].nbytes
                    else:
                        row.window[l] = None
                    n = self.cnt[l][s]
                    if CD.is_view(self.cmp[l]):                                # cdecpack: the codes are the session's own entry, already written in place
                        pass
                    elif SP.is_packed(self.cmp[l]):                            # stackpack: the codes go back as they are (a session entry, no unpack, no repack)
                        row.compress.set_raw(l, SP.session_entry(self.cmp[l], s, n) if n else None)
                    else:
                        if self.cmp[l] is not None and n:
                            row.compress.write_back(l, self.cmp[l][s:s + 1], n)
                        else:
                            row.compress[l] = None
                    row._compress_buffers[l] = None
                    n = self.kcnt[l][s]
                    if CD.is_view(self.keys[l]):
                        pass                                                   # cdecpack: same, the key codes are the session's own entry
                    elif SP.is_packed(self.keys[l]):
                        row.keys.set_raw(l, SP.session_entry(self.keys[l], s, n) if n else None)
                    else:
                        if self.keys[l] is not None and n:
                            if l in (getattr(self, "mq_keys", None) or ()):       # mirrorq: the stack holds these rows, no mirror to feed
                                with MQ.writeback(row.keys):
                                    row.keys.write_back(l, self.keys[l][s:s + 1], n)
                            else:
                                row.keys.write_back(l, self.keys[l][s:s + 1], n)
                        else:
                            row.keys[l] = None
                    row._key_buffers[l] = None
                    if self.kvs[l] is not None:
                        row.kv_state[l] = self.kvs[l][s:s + 1][...]
                        HITS["copy_bytes"] += self.kvs[l][s:s + 1].nbytes
                    else:
                        row.kv_state[l] = None
                    if self.scs[l] is not None:
                        row.score_state[l] = self.scs[l][s:s + 1][...]
                        HITS["copy_bytes"] += self.scs[l][s:s + 1].nbytes
                    else:
                        row.score_state[l] = None
                    row.reset_handoff()
                HITS["sync_row"] += 1

            def sync_rows(self):
                if not self._dirty:
                    return
                for u in range(len(self.rows)):
                    self.sync_row(u)
                self._dirty = False

            def unstack(self):
                assert self._counts is None, "unstack inside a pass"
                self.sync_rows()
                for a in ("win", "cmp", "keys", "kvs", "scs", "cnt", "kcnt", "hand", "_plan", "_dirty", "pend", "_counts", "taps", "fnorm", "i8k", "packed", "mq_keys", "dec_sel",
                          "slots", "S_cap"):
                    self.__dict__.pop(a, None)
                self.__class__ = Base

            # ---------------------------------------------------------------------------------------- flag mu_keep: membership in place
            def extract(self, index):
                if self.slots is None:
                    self.sync_rows()
                else:
                    self.sync_row(index)                                       # one row: the members that stay keep the stack
                HITS["extract_row"] += 1
                return Base.extract(self, index)

            def release_rows(self, indices: list) -> dict:
                """mu_keep: hand the sessions at `indices` their own cache back (the write-back of that one row each) and nothing of the rows
                that stay. Returns {index: per-session cache}."""
                out = {}
                for i in indices:
                    self.sync_row(i)
                    out[i] = Base.extract(self, i)
                HITS["drop"] += len(indices)
                return out

            def keep_only(self, indices: list) -> None:
                """mu_keep: the sessions at `indices` stay in the batch, in that order; the slot map follows. No stacked buffer is read or
                written: the slots of the members that stay keep their rows and their ids."""
                assert self._counts is None, "a pass is open"
                self.rows = [self.rows[i] for i in indices]
                if self.slots is not None:
                    self.slots = [self.slots[i] for i in indices]
                self.left_padding = [self.left_padding[i] for i in indices]
                self._lengths = [self._lengths[i] for i in indices]
                self._plan = None
                self.hand = SimpleNamespace(index_k=None, candidates=None, compress_kv=None, topk_idxs=None)
                HITS["kept_rows"] += len(indices)

            def adopt(self, src) -> int:
                """mu_keep: add the session of `src` (its own per-session cache, the class of `rows`) to the batch, its state written into a
                free slot. Only that slot's rows are written: the slots of the members that stay are not copied. Returns the slot."""
                assert self._counts is None, "a pass is open"
                if self.slots is None:
                    raise RuntimeError("mu_keep: adopt on a cache without slots")
                used = set(self.slots)
                free = [x for x in range(self.S_cap) if x not in used]
                if not free:
                    self._grow_slots(self.S_cap + max(1, mu_spare()))
                    used = set(self.slots)
                    free = [x for x in range(self.S_cap) if x not in used]
                s = free[0]
                step = type(src).step
                for l in range(len(self.win)):
                    w = src.window[l]
                    buf = self.win[l]
                    if buf is None and w is not None:
                        buf = mx.zeros((self.S_cap, *w.shape[1:]), dtype=w.dtype)
                        self.win[l] = buf
                    if buf is not None:
                        buf[s:s + 1] = w if w is not None else mx.zeros((1, *buf.shape[1:]), dtype=buf.dtype)
                        HITS["copy_bytes"] += buf[s:s + 1].nbytes
                    for dst, cnts, get in ((self.cmp, self.cnt, lambda r: r.compress), (self.keys, self.kcnt, lambda r: r.keys)):
                        sl = get(src)
                        v = sl[l] if sl is not None else None
                        raw = sl.raw(l) if (v is not None and hasattr(sl, "raw")) else None
                        n = 0 if v is None else int(v.shape[1])
                        cnts[l][s] = n
                        if n == 0 and dst[l] is None:
                            continue
                        dst[l] = self._slot_pool(dst[l], s, v, raw, step)
                for dst, name, fill in ((self.kvs, "kv_state", 0.0), (self.scs, "score_state", -math.inf)):
                    src_states = getattr(src, name)
                    for l in range(len(self.win)):
                        st = src_states[l]
                        buf = dst[l]
                        if buf is None and st is not None:
                            shape = (self.S_cap, *st.shape[1:])
                            buf = mx.zeros(shape, dtype=st.dtype) if fill == 0.0 else mx.full(shape, fill, dtype=st.dtype)
                            dst[l] = buf
                        if buf is not None:
                            buf[s:s + 1] = st if st is not None else mx.full((1, *buf.shape[1:]), fill, dtype=buf.dtype)
                            HITS["copy_bytes"] += buf[s:s + 1].nbytes
                self.rows.append(src)
                self.slots.append(s)
                self.left_padding.append(0)
                self._lengths.append(None)
                self._plan = None
                for store in (getattr(self, "fnorm", None), getattr(self, "i8k", None)):
                    for e in (store or {}).values():                  # the slot's cached key norms / int8 codes were the session that left
                        have = getattr(e, "have", None)
                        if have is not None and s < len(have):
                            have[s] = 0
                self.hand = SimpleNamespace(index_k=None, candidates=None, compress_kv=None, topk_idxs=None)
                HITS["adopt"] += 1
                return s

            def _slot_pool(self, buf, s: int, v, raw, step: int):
                """Stacked pool / key layer `buf` with slot `s` = the rows of the joining session (`v` dense [1, n, D], `raw` its packed
                entry). Only slot s is written; a layer that is not a writable stack (a cdecpack solo view) is materialised once, counted."""
                n = 0 if v is None else int(v.shape[1])
                if CD.is_view(buf):                                             # cdecpack (one session, read-only codes): a real stack now
                    d = CD.dense_of(buf)
                    HITS["keep_bytes"] += d.nbytes
                    HITS["copy_bytes"] += d.nbytes
                    nb = mx.zeros((self.S_cap, *d.shape[1:]), dtype=d.dtype)
                    nb[0:1] = d
                    buf, HITS["grow"] = nb, HITS["grow"] + 1
                if buf is None:
                    cap = -(-(max(n, 1) + 1) // step) * step
                    buf = mx.zeros((self.S_cap, cap, v.shape[-1]), dtype=v.dtype)
                if SP.is_packed(buf) and not isinstance(raw, FP._Pk):
                    d = SP.dense_of(buf)                                    # the joining layer is off the grid: the stack goes dense
                    HITS["keep_bytes"] += d.nbytes
                    HITS["copy_bytes"] += d.nbytes
                    buf = mx.zeros((self.S_cap, *d.shape[1:]), dtype=d.dtype)
                    buf[self.slots] = d[self.slots]
                if SP.is_packed(buf):                                           # stackpack: codes + scales, no unpack
                    end = n
                    if end > buf.codes.shape[1]:
                        from . import longctx as LC
                        cap = LC.capacity(end, step)
                        HITS["grow"] += 1
                        nc = mx.zeros((self.S_cap, cap, buf.codes.shape[2]), dtype=mx.uint8)
                        ns = mx.zeros((self.S_cap, cap, buf.scales.shape[2]), dtype=mx.uint8)
                        old = buf.codes.shape[1]
                        nc[:, :old], ns[:, :old] = buf.codes, buf.scales
                        HITS["copy_bytes"] += buf.codes.nbytes + buf.scales.nbytes
                        HITS["keep_bytes"] += buf.codes.nbytes + buf.scales.nbytes
                        buf = SP.PackedStack(nc, ns, buf.dtype, buf.block, buf.e4m3, buf.segs)
                    buf.codes[s:s + 1, :end] = raw.codes[:, :end]
                    buf.scales[s:s + 1, :end] = raw.scales[:, :end]
                    HITS["copy_bytes"] += raw.codes[:, :end].nbytes + raw.scales[:, :end].nbytes
                    return buf
                end = n
                if end > buf.shape[1]:
                    from . import longctx as LC
                    HITS["grow"] += 1
                    nb = mx.zeros((buf.shape[0], LC.capacity(end, step), *buf.shape[2:]), dtype=buf.dtype)
                    nb[self.slots, :buf.shape[1]] = buf[self.slots]
                    HITS["copy_bytes"] += buf[self.slots].nbytes
                    HITS["keep_bytes"] += buf[self.slots].nbytes
                    buf = nb
                if end:
                    buf[s:s + 1, :end] = v
                    HITS["copy_bytes"] += v.nbytes
                return buf

            def _grow_slots(self, n: int) -> None:
                """mu_keep: a session axis of `n` slots, the live slots copied (the one copy a join pays when no spare slot is free). The slot
                ids do not move, so `slots` and the per-slot counts stay."""
                assert n > self.S_cap
                old = list(self.slots)
                idx = mx.array(old, dtype=mx.int32)
                HITS["grow"] += 1

                def grow(b, fill=None):
                    if b is None:
                        return None
                    if SP.is_packed(b):
                        codes = mx.zeros((n, *b.codes.shape[1:]), dtype=b.codes.dtype)
                        scales = mx.zeros((n, *b.scales.shape[1:]), dtype=b.scales.dtype)
                        codes[idx], scales[idx] = b.codes[idx], b.scales[idx]
                        HITS["copy_bytes"] += b.codes[idx].nbytes + b.scales[idx].nbytes
                        HITS["keep_bytes"] += b.codes[idx].nbytes + b.scales[idx].nbytes
                        return SP.PackedStack(codes, scales, b.dtype, b.block, b.e4m3, b.segs)
                    if CD.is_view(b):
                        b = CD.dense_of(b)
                    if fill is None:
                        nb = mx.zeros((n, *b.shape[1:]), dtype=b.dtype)
                    else:
                        nb = mx.full((n, *b.shape[1:]), fill, dtype=b.dtype)
                    nb[idx] = b[idx]
                    HITS["copy_bytes"] += b[idx].nbytes
                    HITS["keep_bytes"] += b[idx].nbytes
                    return nb

                self.win = [grow(x) for x in self.win]
                self.cmp = [grow(x) for x in self.cmp]
                self.keys = [grow(x) for x in self.keys]
                self.kvs = [grow(x, 0.0) for x in self.kvs]
                self.scs = [grow(x, -math.inf) for x in self.scs]
                self.cnt = [list(c) + [0] * (n - len(c)) for c in self.cnt]
                self.kcnt = [list(c) + [0] * (n - len(c)) for c in self.kcnt]
                self.S_cap = n
                self._plan = None

            @property
            def state(self):
                self.sync_rows()
                return Base.state.fget(self)

            def filter(self, indices):
                self.unstack()
                Base.filter(self, indices)

            def extend(self, other):
                if isinstance(other, RowsCache):
                    other.unstack()
                self.unstack()
                Base.extend(self, other)

        _rows_cls = RowsCache
    return _rows_cls


def _per_slot(values: list, slots: list | None, S: int) -> list:
    """A per-session list -> the per-slot list of a stacked buffer (0 for a slot no session owns). `slots` None: the list itself."""
    if slots is None:
        return values
    out = [0] * max(S, max(slots) + 1)
    for u, s in enumerate(slots):
        out[s] = values[u]
    return out


def _keep_slots(buf, vals, slots: list | None, fill: float | None = None):
    """`buf` with the rows of `vals` (one row per session, in session order) written into their slots. Identity slots on a buffer of the same
    session count: `vals` is the buffer, no copy (the stack of every other lever). mu_keep: the buffer has spare slots, so a row goes back to
    its own slot and the other slots are left alone."""
    if slots is not None and vals.shape[0] > len(slots):                      # a pass with rejected drafts whose kept rows are the first ones (`last` None): only those rows go back
        vals = vals[:len(slots)]
    if slots is None or (vals.shape[0] == len(slots) and slots == list(range(len(slots)))
                         and (buf is None or buf.shape[0] == vals.shape[0])):
        return vals
    S = max(max(slots) + 1, vals.shape[0])
    if buf is None or buf.shape[0] < S:
        nb = mx.zeros((S, *vals.shape[1:]), dtype=vals.dtype) if fill is None else mx.full((S, *vals.shape[1:]), fill, dtype=vals.dtype)
        if buf is not None:
            nb[:buf.shape[0]] = buf
        buf = nb
    buf[mx.array(slots, dtype=mx.int32)] = vals
    return buf


def _stack_plain(arrs: list, slots: list | None = None, S: int | None = None):
    have = [a for a in arrs if a is not None]
    if not have:
        return None
    if slots is None:                                                         # mu_keep off: the layout of every other lever
        return mx.concatenate([a if a is not None else mx.zeros_like(have[0]) for a in arrs], axis=0)
    buf = mx.zeros((S, *have[0].shape[1:]), dtype=have[0].dtype)
    for u, a in enumerate(arrs):
        if a is not None:
            buf[slots[u]:slots[u] + 1] = a
    return buf


def _stack_ragged(arrs: list, step: int, slots: list | None = None, S: int | None = None):
    """Per-user [1, n_u, ...] arrays -> one [N, cap, ...] buffer (cap > max n_u) and the row counts. mu_keep: [S, cap, ...] with the user at
    its slot and the counts per slot."""
    cnt = [0 if a is None else a.shape[1] for a in arrs]
    have = [a for a in arrs if a is not None]
    if not have or max(cnt) == 0:
        return None, _per_slot(cnt, slots, S)
    cap = -(-(max(cnt) + 1) // step) * step
    buf = mx.zeros((len(arrs) if slots is None else S, cap, *have[0].shape[2:]), dtype=have[0].dtype)
    for u, a in enumerate(arrs):
        if cnt[u]:
            s = u if slots is None else slots[u]
            buf[s:s + 1, :cnt[u]] = a
    return buf, _per_slot(cnt, slots, S)


def _stack_state(arrs: list, fill: float, slots: list | None = None, S: int | None = None):
    have = [a for a in arrs if a is not None]
    if not have:
        return None
    if slots is None:
        return mx.concatenate([a if a is not None else mx.full(have[0].shape, fill, dtype=have[0].dtype) for a in arrs], axis=0)
    buf = mx.full((S, *have[0].shape[1:]), fill, dtype=have[0].dtype)
    for u, a in enumerate(arrs):
        if a is not None:
            buf[slots[u]:slots[u] + 1] = a
    return buf


def stack_dedupe() -> bool:
    """Flag stack_dedupe (read per stack): while a batch is stacked, the dense stacked pool / key buffers are the only copy of those layers."""
    return os.environ.get("KYOJIN_STACK_DEDUPE", "0") not in ("0", "")


def _release_rows(rows: list, name: str, buffers: str, l: int) -> None:
    """Layer l of every row's `name` slots (pool or index keys) was copied into the stack: drop the row's own copy (packed codes + scales, dense mirror, memo,
    capacity buffer). The stack holds the same values bit for bit and `sync_rows` writes the layer back (re-packed on the fp4 grid, so the same codes)."""
    for r in rows:
        getattr(r, name)[l] = None
        getattr(r, buffers)[l] = None


def stack_rows(cache):
    """The cache of a batched step, with the attention state of its users stacked (idempotent)."""
    cls = _rows_cache_class()
    if type(cache) is cls:
        return cache
    rows = cache.rows
    n_layers = len(rows[0].window)
    step = type(rows[0]).step
    cache.__class__ = cls
    keep = mu_keep()                                                          # mu_keep: a slot per session and a spare, a join copies nothing
    slots = list(range(len(rows))) if keep else None
    S = len(rows) + mu_spare() if keep else len(rows)
    cache.slots, cache.S_cap = slots, S
    HITS["stack_slots"] += 1 if keep else 0
    cache.win = [_stack_plain([r.window[l] for r in rows], slots, S) for l in range(n_layers)]
    cache.cmp, cache.cnt, cache.keys, cache.kcnt = [], [], [], []
    packed = cache.packed = SP.ENABLED and len(rows) > 1                      # stackpack: one session keeps the exact current path
    dedupe = stack_dedupe() and len(rows) > 1         # N = 1 keeps its own layers: the solo path is the code and the allocations it had
    solo = CD.ENABLED and len(rows) == 1 and slots is None                    # cdecpack: one session reads its own packed layers, no dense stack (a slot stack must be writable: mu_keep off)
    mq = MQ.on() and len(rows) == 1 and not solo and slots is None            # mirrorq: the pool as its codes, the key stack unpacked from the codes (a slot stack must be writable: mu_keep off)
    if mq:
        cache.mq_keys = set()                                                 # mirrorq: layers whose dense key stack came from the codes (mirror dropped)
    held = []
    for l in range(n_layers):
        for dst, cnts, get, name, bufs in ((cache.cmp, cache.cnt, lambda r: r.compress, "compress", "_compress_buffers"), (cache.keys, cache.kcnt, lambda r: r.keys, "keys", "_key_buffers")):
            v = CD.view(get(rows[0]), l) if solo else (MQ.pool_view(get(rows[0]), l) if mq and name == "compress" else None)
            if v is not None:                                                 # the codes themselves: no unpack, no mirror, no copy
                dst.append(v)
                cnts.append([v.n])
                continue
            if mq and name == "keys":                                         # mirrorq: the dense key stack the core reads, unpacked from the codes
                got = MQ.stack_dense([get(r) for r in rows], l, step)
                if got is not None:
                    dst.append(got[0])
                    cnts.append(got[1])
                    cache.mq_keys.add(l)                                      # its mirror is gone: the write-back must not read the layer back
                    continue
            pk = SP.stack([get(r) for r in rows], l, step, S, slots) if packed else None
            if pk is not None:
                b, c = pk
                held.append(([get(r) for r in rows], l))
            else:
                if packed:
                    SP.HITS["dense_layer"] += 1
                b, c = _stack_ragged([get(r)[l] for r in rows], step, slots, S)
                if dedupe and b is not None:
                    _release_rows(rows, name, bufs, l)
            dst.append(b)
            cnts.append(c)
    cache.kvs = [_stack_state([r.kv_state[l] for r in rows], 0.0, slots, S) for l in range(n_layers)]
    cache.scs = [_stack_state([r.score_state[l] for r in rows], -math.inf, slots, S) for l in range(n_layers)]
    cache.hand = SimpleNamespace(index_k=None, candidates=None, compress_kv=None, topk_idxs=None)
    cache.dec_sel = {}                                 # decsel: the published index selection per (session row, layer) + its key count / age
    cache.fnorm = {}                                   # decfilter: committed key norms per layer (rebuilt from the keys at every stack)
    cache.blks = {}                                    # dec_blk_sum: block summary keys per layer (rebuilt from the keys at every stack)
    cache.i8k = {}                                     # dec_idx_i8: int8 codes of the committed index keys per layer (rebuilt from the keys at every stack)
    cache._plan = None
    cache._dirty = dedupe                              # stack_dedupe: the rows' own pool / key layers are gone, the first sync_rows rebuilds them from the stack
    cache.pend = {}
    if os.environ.get("KYOJIN_TICK_TRACE", "0") == "1":
        t0 = time.perf_counter()
        for lst in (cache.cmp, cache.keys):
            mx.eval([b for b in lst if isinstance(b, mx.array)])
            print(f"tick_trace stack n={sum(1 for b in lst if isinstance(b, mx.array))} eval {(time.perf_counter() - t0) * 1e3:.0f} ms bytes {sum(b.nbytes for b in lst if isinstance(b, mx.array)) / 2**30:.2f} GiB", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
    if held:                                                                  # the batch holds the only copy of these layers until sync_rows writes them back
        for slots, l in held:
            SP.release(slots, l)
        cache._dirty = True
    return cache


def plan_of(cache) -> Plan:
    counts = cache._counts
    if counts is None:
        pos = [r.offset for r in cache.rows]
    else:
        pos = [r.offset + i for r, c in zip(cache.rows, counts) for i in range(c)]
    slots = getattr(cache, "slots", None)
    nslots = getattr(cache, "S_cap", None)
    key = (tuple(pos), tuple(counts if counts is not None else [1] * len(pos)),
           None if slots is None or slots == list(range(len(slots))) else tuple(slots), None if nslots is None else int(nslots))
    p = cache._plan
    if p is None or p.key != key:
        p = cache._plan = Plan(pos, counts, slots, nslots)
    return p


def _put(lst: list, l: int, S: int, tail: tuple, dtype, em: SimpleNamespace, vals, step: int, pack: tuple | None = None):
    """lst[l][session of each emitting row, write index] = vals, growing the buffer (capacity in multiples of `step`) when a row would
    overflow. lst[l] is a pool / key buffer [S, cap, ...] with one slot range per session. The slot of the list takes the result at once and
    the old reference is dropped first, so MLX scatters in place when nothing else holds the buffer (a copy of the whole pool per emitting
    pass otherwise: the entries past the committed count are never read by the committed state, so the early adoption is invisible)."""
    need = em.need
    buf, lst[l] = lst[l], None
    if CD.is_view(buf):                                                       # cdecpack: the session's own packed layer, no dense scatter buffer
        lst[l] = CD.put(buf, l, em, vals, step, LC.capacity)
        return lst[l]
    if SP.is_packed(buf):                                                     # stackpack: the layer is stacked as codes + scales
        lst[l] = SP.put(buf, S, tail[0], buf.dtype, em, vals, step, buf.block, buf.e4m3, LC.capacity, _CANON_POOLS)
        return lst[l]
    if buf is None and pack is not None and tail[0] % pack[0] == 0 and dtype in (mx.float16, mx.bfloat16, mx.float32):
        lst[l] = SP.put(None, S, tail[0], dtype, em, vals, step, pack[0], pack[1], LC.capacity, _CANON_POOLS)
        return lst[l]
    if _CANON_POOLS and buf is not None:
        buf = buf[mx.arange(buf.shape[0])]                                # always a gather copy: the stream (gather, scatter into the fresh copy) must not depend on who else holds the buffer (MLX donation)
    if buf is None:
        buf = mx.zeros((S, -(-need // step) * step, *tail), dtype=dtype)
    elif need > buf.shape[1]:
        new = mx.zeros((S, LC.capacity(need, step), *tail), dtype=buf.dtype)
        new[:, :buf.shape[1]] = buf
        buf = new
    buf[em.sess_arr, em.widx_arr] = vals
    lst[l] = buf
    return buf


def _snapshot(base, plan: Plan, shape: tuple, dtype, fill: float):
    """The state each row starts from: its session's committed state (base [S, ...], None = never written). mu_keep: base is [NS, ...] and the
    rows are gathered by slot, so a row of a session in slot s reads slot s and a dead slot is never read."""
    if base is None:
        return mx.zeros((plan.n, *shape), dtype=dtype) if fill == 0.0 else mx.full((plan.n, *shape), fill, dtype=dtype)
    if plan.plain and base.shape[0] == plan.n and not _CANON_POOLS:
        return base[...]
    return base[plan.sess_idx]


def _write_rows(buf, e, vals):
    """buf[session, slot] = vals[row] for the rows `e` a commit keeps: the compressor state of a pass, written once, at the commit (STATE_NOCOPY).
    Same values as the snapshot-then-pick it replaces, so the state is the one a pass that never drafted would have left."""
    buf[e.sess, e.slots] = vals[:e.n] if e.prefix else vals[e.idx]
    return buf


# ----------------------------------------------------------------------------------------------------------------- the core
_kc: dict = {}


def kv_chain(kvr: mx.array, w: mx.array, base: mx.array, sess: mx.array, wmap: mx.array, rt: mx.array, eps: float) -> mx.array:
    """kvr [n, D] (the raw kv projection) -> the new window rings [n, W, D] of the pass: base[sess[b]] with the slots this pass writes replaced by
    kv_norm + rope tail + fp8 round trip of their kv row. One launch (same arithmetic and bits as the stock chain plus the overlay scatter)."""
    n, D = kvr.shape
    W = base.shape[1]
    k = _kc.get(("kvchain", D, W, ROPE_FMA))
    if k is None:
        k = _kc[("kvchain", D, W, ROPE_FMA)] = mx.fast.metal_kernel(name="kyojin_dsv4_kv_chain" + _fk(), input_names=["kvr", "w", "ring", "sess", "wmap", "rt", "eps"],
                                                                    output_names=["out"], source=rope_src(_KV_CHAIN), header=_CONTRACT_OFF)
    return k(inputs=[kvr, w, base, sess, wmap, rt, _f32((eps,))], template=[("T", kvr.dtype), ("D", D), ("W", W), ("RD", rt.shape[-1])],
             grid=(n * W * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(n, W, D)], output_dtypes=[kvr.dtype])[0]


def kv_chain_stack(kvr: mx.array, w: mx.array, base: mx.array, sess: mx.array, wmap: mx.array, rowof: mx.array, rt: mx.array, eps: float):
    """kv_chain plus the commit's slot write: (rings [n, W, D], stacked buffer [NS, W, D] = base with the slots of the rows replaced). One launch."""
    n, D = kvr.shape
    NS, W = base.shape[0], base.shape[1]
    key = ("kvchain_stk", D, W, ROPE_FMA)
    k = _kc.get(key)
    if k is None:
        k = _kc[key] = mx.fast.metal_kernel(name="kyojin_dsv4_kv_chain_stk" + _fk(), input_names=["kvr", "w", "ring", "wmap", "rowof", "rt", "eps"],
                                            output_names=["out", "buf"], source=rope_src(_KV_CHAIN_STK), header=_CONTRACT_OFF)
    out, buf = k(inputs=[kvr, w, base, wmap, rowof, rt, _f32((eps,))], template=[("T", kvr.dtype), ("D", D), ("W", W), ("RD", rt.shape[-1])],
                 grid=(NS * W * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(n, W, D), (NS, W, D)], output_dtypes=[kvr.dtype, kvr.dtype])
    return out, buf


def rot_fq4(x: mx.array, cos: mx.array, sin: mx.array, rd: int) -> mx.array:
    """fake_quant_fp4_ue8m0(_apply_index_rotary(x, cos, sin, rd)) for x [..., heads, 128] half with one cos / sin row per position (broadcast over the
    heads): one kernel when ROT_FQ4, else the stock chain."""
    L = _lang()
    D = x.shape[-1]
    T = cos.size // (rd // 2)
    if not ROT_FQ4 or x.dtype != mx.float16 or D != 128 or x.size % (T * D):
        return L.fake_quant_fp4_ue8m0(L._apply_index_rotary(x, cos, sin, rd))
    from . import ds_pf_idx
    return ds_pf_idx.fq4_rot(x.reshape(1, T, x.size // (T * D), D), cos.reshape(T, -1), sin.reshape(T, -1), rd).reshape(x.shape)


def _cproj_ok(comp, x: mx.array) -> bool:
    """dec_cproj covers plain nn.Linear compressor projections (f32 / f16 / bf16 weights of one dtype, no bias) on a decode pass."""
    from . import stagek2 as SK2
    w0, w1 = getattr(comp.wkv, "weight", None), getattr(comp.wgate, "weight", None)
    return bool(w0 is not None and w1 is not None and type(comp.wkv).__name__ == "Linear" and type(comp.wgate).__name__ == "Linear"
                and "bias" not in comp.wkv and "bias" not in comp.wgate and x.ndim == 3 and SK2.xgemv_supported(x.reshape(x.shape[0], -1), w0, w1)
                and SK2.xgemv_usable())


def core(attn, x: mx.array, cache, pre=None, stage: bool = False, ex: dict | None = None) -> mx.array:
    """One pass of a DeepseekV41Attention module over n rows (x [n, 1, H], the cache of a batched step). Plain decode: one row per session.
    After `cache.begin(counts)`: counts[s] consecutive positions of session s, each row seeing the state its predecessors in the pass left
    (the state of the whole pass stays in `cache.pend` until `cache.commit`). The arithmetic of a row does not depend on the layout.
    `pre` (decstage): (qr, q, kv) already projected by the block path's `in` stage (q pre-rope [n, 1, H, D], kv the raw wkv output [n, 1, D], its norm below), so the
    core runs the compressor / indexer / ring / attention / merge only. `stage`: the decstage `attn` stage, which turns the kv chain, the q /
    inverse rope fuse and the fused index selection on for this call (the same bit-identical kernels, gated by their own flags when off).
    `ex` (decstage k1-k2 levers): "win" = the window rings the stage already wrote (stagek2.qkv_stage, same bits as the kv_chain below), "qi" = the indexer
    wq_b output the stage already projected (same-input group with the attention wq_b); either missing: this function computes it as before."""
    L = _lang()
    st = stack_rows(cache)
    plan = plan_of(st)
    n = x.shape[0]
    assert n == plan.n, (n, plan.n)
    S = plan.NS                                                               # mu_keep: the slot axis of the stacked buffers (== plan.S when off)
    l = attn.layer_idx
    H, D, W, cfg = attn.n_heads, attn.head_dim, attn.window_size, attn.config
    assert max(plan.counts) <= W, "more rows per session than window slots"
    dt = x.dtype
    step = type(st.rows[0]).step
    fq8 = L.fake_quant_fp8_ue8m0
    if pre is None:
        xq = fq8(x)
        qr = attn.q_norm(attn.wq_a(xq))
        q = attn.wq_b(fq8(qr)).reshape(n, H, 1, D)
    else:                                                                       # decstage `in`: the q latent, the pre-rope q and the raw kv
        qr, q, kv = pre
        q = q.reshape(n, H, 1, D)
    if AZ.ACTIVE and AZ.on("attn"):                                             # ablate: zeros for the attention output; no kv write, compressor, indexer or attend; `pend` empty = cache untouched
        AZ.hit("attn")
        o = AZ.zeros_like_shape((n, attn.o_groups, 1, (H // attn.o_groups) * D), dt)
        if stage:
            return o
        return attn.wo_b(fq8(attn.wo_a(o).transpose(0, 2, 1, 3).flatten(-2)))
    rope_fuse = (ROPE_FUSE or stage) and D == KG.HEAD_DIM and attn.rope.dims == KG.ROPE_DIMS
    kvfuse = (KV_CHAIN or stage) and D == KG.HEAD_DIM and attn.rope.dims == KG.ROPE_DIMS
    kvfuse = kvfuse and (x if pre is None else kv).dtype == mx.float16 and attn.kv_norm.weight.dtype == mx.float16
    # The rows of an open pass through the fused row core (fuse_attn.attend_rows_m / kv_apply_m): the session's ring and pool are read as they are and
    # the pass's own kv rows are read from `kvp`, so no per-row copy of the ring is built for the attention and the ring a commit keeps is built once,
    # for the accepted row, at the commit. Plain decode (no pass open) keeps the batched path below.
    rowfuse = bool(ROWS_CORE_FUSE and st._counts is not None and not (ex and ex.get("win") is not None) and D == KG.HEAD_DIM
                   and dt in (mx.float16, mx.bfloat16) and attn.rope.dims % 2 == 0 and max(plan.counts) <= W
                   and getattr(st.hand, "dbg_o", None) is None)
    rope_f, rope_i = plan.attn_rope(attn.rope) if (kvfuse or rope_fuse or rowfuse) else (None, None)
    if not rope_fuse and not rowfuse:                                           # (the hoisted q of decstage `in` is pre-rope)
        q = attn.rope(q, plan.pos_arr)
    pd = st.pend[l] = SimpleNamespace(win=None, kvs=None, scs=None, cmp=None, keys=None, r=0, cnt=None, kcnt=None, kvr=None, kvrow=None, scrow=None, win_stk=None)
    FD.hit("kvchain") if kvfuse else None
    FD.hit("ropefuse") if rope_fuse else None
    FD.hit("rows_core_fuse") if rowfuse else None
    if rowfuse:                                                                 # kv_norm (stock) + rope tail + fp8 round trip of the pass's rows, one launch
        from . import fuse_attn as FA
        rbase = st.win[l] if st.win[l] is not None else mx.zeros((S, W, D), dtype=dt)
        if pre is None:
            kv = attn.wkv(xq)
        kvp = FA.kv_proc(attn.kv_norm(kv).reshape(n, D), rope_f, attn.rope.dims)
        win = None
        pd.win = FA.kv_apply_m(rbase, kvp, plan.sess_arr, plan.first_arr(), plan.pos_arr)     # lazy: read at the commit only
    elif kvfuse:                                                                # kv_norm + rope + fp8 + ring scatter in one launch

        base = st.win[l] if st.win[l] is not None else mx.zeros((S, W, D), dtype=dt)
        if pre is None:
            kv = attn.wkv(xq)
        if (ATTN_CORE_FUSE and not (ex and ex.get("win") is not None) and st._counts is None and plan.plain and base.shape[0] == plan.NS
                and not (plan.slots == list(range(plan.NS)))):                  # attn_core_fuse: the commit's slot scatter rides in the kv chain launch
            FD.hit("attn_core_fuse_kv")
            win, pd.win_stk = kv_chain_stack(kv.reshape(n, D), attn.kv_norm.weight, base, plan.sess_arr, plan.wmap(W), plan.rowof(), rope_f, attn.kv_norm.eps)
        else:
            win = ex["win"] if ex and ex.get("win") is not None else kv_chain(kv.reshape(n, D), attn.kv_norm.weight, base, plan.sess_arr, plan.wmap(W), rope_f, attn.kv_norm.eps)
    else:
        if pre is None:
            kv = attn.kv_norm(attn.wkv(xq))
        else:
            kv = attn.kv_norm(kv)
        kv = fq8(attn.rope(kv[:, None], plan.pos_arr).reshape(n, D))
        win = _snapshot(st.win[l], plan, (W, D), dt, 0.0)
        ov = plan.overlay(W)
        win[ov.rows, ov.slots] = kv if ov.src is None else kv[ov.src]
    pd.win = pd.win if rowfuse else win
    hand = st.hand
    idxs = None
    dsel = None                                                                # decsel: set when this layer's selection came from the candidate set
    t_max = 0
    r = attn.compress_ratio
    if r:
        em = plan.emit(r)
        rd = cfg.qk_rope_head_dim
        idxr = attn.indexer
        if attn.is_kv_source:
            comp = attn.compressor
            lat = None
            if r == 1:
                lat = comp.norm(comp.wkv(x))[:, 0]
            else:
                if CPROJ and _cproj_ok(comp, x):                             # dec_cproj: the f32 cast and both compressor matmuls in one launch
                    from . import stagek2 as SK2
                    FD.hit("dec_cproj")
                    kvc, scc = SK2.xgemv(x.reshape(n, -1), comp.wkv.weight, comp.wgate.weight)
                    kvc, scc = kvc.reshape(n, 1, -1), scc.reshape(n, 1, -1)
                else:
                    xf = x.astype(mx.float32)
                    kvc, scc = comp.wkv(xf), comp.wgate(xf)
                if STATE_NOCOPY and st._counts is not None:                   # no per-row snapshot: the pool reads base + the pass's rows, the commit writes them
                    pd.kvr, pd.kvrow, pd.scrow = (plan, r), kvc[:, 0], scc[:, 0]
                    if em.n:
                        kb = st.kvs[l] if st.kvs[l] is not None else mx.zeros((S, r, D), dtype=mx.float32)
                        sb = st.scs[l] if st.scs[l] is not None else mx.full((S, r, D), -math.inf, dtype=mx.float32)
                        FD.hit("rows_state_nocopy")
                        lat = comp.norm(pool_overlay(kb, sb, kvc[:, 0], scc[:, 0], em.rows_arr, plan.sess_idx, plan.first_arr(), plan.pos_arr).astype(dt))
                else:
                    ks = _snapshot(st.kvs[l], plan, (r, D), mx.float32, 0.0)
                    ss = _snapshot(st.scs[l], plan, (r, D), mx.float32, -math.inf)
                    ovr = plan.overlay(r)
                    ks[ovr.rows, ovr.slots] = kvc[:, 0] if ovr.src is None else kvc[:, 0][ovr.src]
                    ss[ovr.rows, ovr.slots] = scc[:, 0] if ovr.src is None else scc[:, 0][ovr.src]
                    pd.kvs, pd.scs = ks, ss
                    if em.n:
                        lat = comp.norm(pool_groups(ks, ss, em.rows_arr).astype(dt))

            if em.n:
                cos, sin = plan.cs(em.positions, attn._latent_theta, attn._latent_yarn, rd, (em.n, 1))
                pk = L.fake_quant_fp4_e4m3(L._apply_index_rotary(lat[:, None, :], cos, sin, rd))
                hand.compress_kv = hand.index_k = None                    # the hand of the previous pass must not pin the buffers
                pd.cmp = _put(st.cmp, l, S, (D,), dt, em, pk[:, 0], step, (16, True) if st.packed else None)
                pd.r = r
                if idxr is not None and idxr.owns_k:
                    cosk, sink_ = plan.cs(em.positions, idxr.rope_theta, idxr.yarn, rd, (em.n, 1))
                    k = rot_fq4(idxr.k_norm(idxr.wk(lat[:, None, :])), cosk, sink_, rd)
                    pd.keys = _put(st.keys, l, S, (idxr.index_head_dim,), dt, em, k[:, 0], step, (32, False) if st.packed else None)
            hand.compress_kv = pd.cmp if pd.cmp is not None else st.cmp[l]
            if idxr is not None and idxr.owns_k:
                hand.index_k = pd.keys if pd.keys is not None else st.keys[l]
        t_max = em.tmax
        if _CANON_POOLS and t_max > 0 and hand.compress_kv is not None:                      # capacity bucket: the stream must not move with max(lens) over the sessions (selection is masked by lens, results are identical)
            cap = CD.rows(hand.compress_kv) if hand.index_k is None else min(CD.rows(hand.compress_kv), CD.rows(hand.index_k))
            t_max = min(cap, -(-t_max // TMAX_BUCKET) * TMAX_BUCKET)
        fast = IDX_FAST and (_CANON_POOLS or em.tmax >= IDX_FAST_MIN)
        cap_rt = 0 if hand.compress_kv is None else (CD.rows(hand.compress_kv) if hand.index_k is None else min(CD.rows(hand.compress_kv), CD.rows(hand.index_k)))
        fuse = (IDX_FUSE or stage) and t_max > 0 and hand.compress_kv is not None and idxr is not None and n * (cap_rt if _CANON_POOLS else em.tmax) <= IDX_FUSE_MAX
        if fuse and hand.index_k is not None:                                  # geometry gate BEFORE rt: a fused call that cannot run must not widen t_max to the capacity
            from . import idxfuse as IXF
            fuse = IXF.supported(cap_rt, idxr.n_heads, hand.index_k.dtype)
        rt = (RT_SELECT or fast or fuse) and t_max > 0 and hand.compress_kv is not None and idxr is not None
        if rt:
            t_max = CD.rows(hand.compress_kv) if hand.index_k is None else min(CD.rows(hand.compress_kv), CD.rows(hand.index_k))
        if t_max > 0 and hand.compress_kv is not None:
            short = IDXS.decide(idxr, cfg, em, rt, t_max, n, _CANON_POOLS) if idxr is not None and IDXS.ENABLED else None
            if short is None and AZ.ACTIVE and AZ.on("idx") and idxr is not None:     # ablate: the first-k positions below each row's length (the idx_short form), no wq_b, scores or select
                AZ.hit("idx")
                short = (IDXS.idx_for(em, idxr.index_topk if rt else min(idxr.index_topk, t_max))[:, None, :],
                         IDXS.all_mask(n, t_max) if getattr(idxr, "is_candidate_source", False) else None)
            if short is not None:
                # flag idx_short: every pooled key of every row fits in the top-k, so the exact selection is the identity (see idxshort.py): no wq_b, no scores, no select
                FD.hit("idx_short")
                hand.topk_idxs = short[0]
                if short[1] is not None:
                    hand.candidates = short[1]
            elif idxr is not None:
                nh, dI = idxr.n_heads, idxr.index_head_dim
                cosq, sinq = plan.cs(plan.pos, idxr.rope_theta, idxr.yarn, rd, (n, 1, 1))
                qraw = ex["qi"] if ex and ex.get("qi") is not None else idxr.wq_b(fq8(qr))
                qi = rot_fq4(qraw.reshape(n, 1, nh, dI), cosq, sinq, rd)
                wts = idxr.weights_proj(x) * (idxr.scale * nh ** -0.5)
                cand = hand.candidates[:, 0, :] if idxr.uses_candidates else None
                if fuse:                                                       # the fused score+select holds for 32 index heads
                    from . import idxfuse as IXF
                    fuse = IXF.supported(t_max, nh, qi.dtype)
                dsel = None
                if rt:                                                         # decsel: the candidate set of the previous selection + the appended keys
                    dsel = DCS.layer(st, l, idxr, CD.dense_of(hand.index_k), cand, qi[:, 0], wts[:, 0], em, t_max, plan)
                fres = None
                if dsel is None and DFL.ENABLED and rt and cand is None and DFL.applies(idxr, CD.dense_of(hand.index_k), qi, em.tmax, n):    # dec_filter: certified two-level select (plain layer or candidate source)
                    fres = DFL.layer(st, l, idxr, qi[:, 0], CD.dense_of(hand.index_k), wts[:, 0], plan, em, t_max)
                fblk = None
                if dsel is None and IXB.ENABLED and rt and IXB.applies(idxr, CD.dense_of(hand.index_k), cand, qi, t_max):                    # iscore_blocks: a consumer layer's kept blocks only
                    fblk = IXB.layer(idxr, CD.dense_of(hand.index_k), cand, qi[:, 0], wts[:, 0], em, t_max, plan.sess_arr)
                fbs = None
                if dsel is None and DBK.ENABLED and rt and fres is None and fblk is None and hand.index_k is not None:                       # dec_blk_sum: block-summary first pass, exact rescore of the picked blocks
                    fbs = DBK.layer(st, l, idxr, CD.dense_of(hand.index_k), cand, qi[:, 0], wts[:, 0], em, t_max, plan.sess_arr)
                f8 = None
                if dsel is None and IX8.ENABLED and fbs is None and rt and fres is None and fblk is None and hand.index_k is not None:                        # dec_idx_i8: int8 first pass, exact rescore of the picked blocks
                    f8 = IX8.layer(st, l, idxr, CD.dense_of(hand.index_k), cand, qi[:, 0], wts[:, 0], em, t_max, plan.sess_arr)
                cif = None
                if CIDXFUSE and dsel is None and rt and not fuse and hand.index_k is not None and fres is None and fblk is None and fbs is None and f8 is None:
                    from . import cidxfuse as _cif
                    if (qi.dtype == hand.index_k.dtype == wts.dtype and (cand is None or cand.shape[-1] >= t_max) and t_max <= hand.index_k.shape[1] and _cif.applies(n, t_max, idxr.index_topk, qi.dtype, nh, dI, idxr.candidate_topk_blocks if idxr.is_candidate_source else 0,
                                                                                      idxr.candidate_block_size)):
                        cif = _cif
                if dsel is not None:                                           # the top-k of the candidate set (previous selection + appended keys), rt form
                    FD.hit("decsel")
                    hand.topk_idxs = dsel[:, None, :]
                    scores = None
                elif fres is not None:                                         # certificate held: the exact stock selection, rt form (ascending, -1 tail)
                    FD.hit("dec_filter")
                    hand.topk_idxs = fres[0][:, None, :]
                    if fres[1] is not None:
                        hand.candidates = fres[1][:, None, :]
                    scores = None
                elif fblk is not None:                                         # the same selection over the 16384 keys the mask kept: flat in the context
                    FD.hit("iscore_blocks")
                    hand.topk_idxs = fblk[:, None, :]
                    scores = None
                elif fbs is not None:                                          # the top-k of the blocks the summaries picked, rescored exactly
                    FD.hit("dec_blk_sum")
                    hand.topk_idxs = fbs[0][:, None, :]
                    if fbs[1] is not None:
                        hand.candidates = fbs[1][:, None, :]
                    scores = None
                elif f8 is not None:                                           # the top-k of the picked blocks, rescored exactly (mask: the source's kept blocks)
                    FD.hit("dec_idx_i8")
                    hand.topk_idxs = f8[0][:, None, :]
                    if f8[1] is not None:
                        hand.candidates = f8[1][:, None, :]
                    scores = None
                elif cif is not None:                                          # cidxfuse: scores + chunk-tree select + mask in one launch (topk_tree over the scores_fast bits)
                    FD.hit("cidxfuse_layer")
                    CD.keys_hit(hand.index_k)
                    idx, m = cif.fused(qi[:, 0], hand.index_k, wts[:, 0], em.lens_arr, t_max, idxr.index_topk, cand, plan.sess_arr,
                                       idxr.candidate_topk_blocks if idxr.is_candidate_source else 0, idxr.candidate_block_size)
                    if m is not None:
                        hand.candidates = m[:, None, :]
                    hand.topk_idxs = idx[:, None, :]
                    scores = None
                elif rt and fuse:
                    from . import idxfuse as IXF
                    LC.HITS["dec_fuse"] += 1
                    FD.hit("idx_fuse")
                    cs = idxr.is_candidate_source
                    CD.keys_hit(hand.index_k)
                    idx, m, _ = IXF.fused(qi[:, 0], hand.index_k, wts[:, 0], em.lens_arr, t_max, idxr.index_topk, cand, plan.sess_arr,
                                          idxr.candidate_topk_blocks if cs else 0, idxr.candidate_block_size)
                    if cs:
                        hand.candidates = m[:, None, :]
                    hand.topk_idxs = idx[:, None, :]
                    scores = None
                else:
                    scores = None
                    if KC.ENABLED and ISCORE_MMA and rt and t_max >= ISCORE_MMA_MIN:      # dec_keys_codes: the packed layer's codes are read by the score kernel, no whole-layer unpack
                        scores = KC.scores(qi[:, 0], wts[:, 0], hand.index_k, em.lens_arr, t_max, cand, plan.sess_arr)
                        if scores is not None:
                            FD.hit("dec_keys_codes")
                            CD.keys_hit(hand.index_k)
                    if scores is None and ISCORE_MMA and rt and t_max >= ISCORE_MMA_MIN and getattr(CD.dense_of(hand.index_k), "dtype", None) == qi.dtype:
                        from . import dsa_decode as DSA                       # not `D`: core's local D is the head width, a module there breaks q.reshape below
                        if DSA.rows_supported(nh, dI, qi.dtype):
                            LC.HITS["dec_score_mma"] += 1
                            FD.hit("iscore_mma")
                            scores = DSA.scores_rows(qi[:, 0], wts[:, 0], CD.dense_of(hand.index_k), em.lens_arr, t_max, cand, plan.sess_arr)
                    if scores is None:
                        if IDX_SCORE_FAST and t_max <= IDX_SCORE_FAST_MAX and hand.index_k is not None:
                            from . import idxfuse as IXF
                            if IXF.scores_supported(nh, dI, qi.dtype):
                                LC.HITS["dec_score_fast"] += 1
                                CD.keys_hit(hand.index_k)
                                scores = IXF.scores_fast(qi[:, 0], hand.index_k, wts[:, 0], em.lens_arr, t_max, cand, plan.sess_arr)
                            else:
                                scores = index_scores(qi[:, 0], CD.dense_of(hand.index_k), wts[:, 0], em.lens_arr, t_max, cand, plan.sess_arr, rt=rt)
                        else:
                            scores = index_scores(qi[:, 0], CD.dense_of(hand.index_k), wts[:, 0], em.lens_arr, t_max, cand, plan.sess_arr, rt=rt)
                if scores is None:
                    pass
                elif rt and fast:
                    LC.HITS["dec_fast"] += 1
                    FD.hit("idx_fast")
                    from . import idxfast as IX
                    if idxr.is_candidate_source:
                        idx, m = IX.select_fused(scores, em.lens_arr, idxr.index_topk, idxr.candidate_topk_blocks,
                                                idxr.candidate_block_size)
                        hand.candidates = m[:, None, :]
                        hand.topk_idxs = idx[:, None, :]
                    else:
                        hand.topk_idxs = IX.topk_mt(scores, em.lens_arr, idxr.index_topk)[:, None, :]
                elif rt:
                    if idxr.is_candidate_source:
                        hand.candidates = candidate_mask_rt(scores, em.lens_arr, idxr.candidate_topk_blocks, idxr.candidate_block_size)[:, None, :]
                    hand.topk_idxs = index_topk_rt(scores, em.lens_arr, idxr.index_topk)[:, None, :]
                else:
                    scores = scores[:, None, :]
                    if idxr.is_candidate_source:
                        hand.candidates = L.select_candidate_blocks(scores, em.lens_arr[:, None, None], idxr.candidate_topk_blocks,
                                                                    idxr.candidate_block_size)
                    topk = min(idxr.index_topk, t_max)
                    ids = mx.sort(mx.argsort(-scores, axis=-1)[..., :topk], axis=-1).astype(mx.int32)
                    hand.topk_idxs = mx.where(ids < em.lens_arr[:, None, None], ids, -1).astype(mx.int32)
            idxs = hand.topk_idxs
            if DCS.active() and rt and idxr is not None and idxs is not None and idxs.shape[-1] == idxr.index_topk:
                pd.dec = (idxs[:, 0, :], em.lens, dsel is None)                 # published at the commit with the row the commit keeps
    if idxs is not None:
        pool, idx = hand.compress_kv, idxs[:, 0, :]
        if MQ.on() and MQ.is_view(pool):                                        # mirrorq: the rows the top-k names, decoded from the codes into a slab
            pool, idx = MQ.decode_rows(pool, idx)
        elif CD.is_view(pool):                                                  # cdecpack: the top set from the codes, one slab of K rows per row; no dense pool
            pool, idx = CD.compact(pool, idx)
        elif SP.is_packed(pool):                                                # stackpack: the top set from the codes, in a slab per session; no dense pool
            pool, idx = SP.gather_slab(pool, idx, plan.sess_idx, plan.rows_all - plan.first_arr(), S, max(plan.counts))
        elif rowfuse and hasattr(pool, "codes"):
            pool = pool.dense()
    else:
        if rowfuse:
            pool, idx = mx.zeros((S, 1, D), dtype=dt), mx.full((n, 1), -1, dtype=mx.int32)
        else:
            pool, idx = plan.dummy(D, dt)
    if rowfuse:
        from . import fuse_attn as FA
        o = FA.attend_rows_m(q.reshape(n, H, D), rbase, kvp, pool, idx, attn.attn_sink.astype(mx.float32), rope_f, rope_i, plan.pos_arr,
                             plan.sess_arr, plan.first_arr(), H, attn.rope.dims, attn.scale)
        inv_fuse = True
    else:
        inv_fuse = rope_fuse and SPLIT_MS > 0
        o = attend(q.reshape(n, H, D), win, pool, idx, plan.pos_arr, attn.attn_sink.astype(mx.float32), attn.scale, cfg.index_topk, plan.sess_arr,
                   rt=rope_f if rope_fuse else None, it=rope_i if inv_fuse else None)
    if getattr(hand, 'dbg_o', None) is not None:
        hand.dbg_o.append(o)
        hand.dbg_o.extend([q.reshape(n, H, D), win, pool, idx, plan.pos_arr])
    if not inv_fuse:
        o = attn.rope(o.reshape(n, H, 1, D), plan.pos_arr, inverse=True)
    o = o.reshape(n, attn.o_groups, -1, 1, D).transpose(0, 1, 3, 2, 4).flatten(-2)
    if stage:                                                         # decstage `out` runs wo_a / wo_b (fp8 folds) and the expand
        if st._counts is None:
            st.commit_layer(l, None, [p for p in plan.pos], slots=plan.slots)
        return o
    o = attn.wo_a(o).transpose(0, 2, 1, 3).flatten(-2)
    if st._counts is None:                                            # plain decode: every row is its session's last, commit at once
        st.commit_layer(l, None, [p for p in plan.pos], slots=plan.slots)
    return attn.wo_b(fq8(o))
