# SPDX-License-Identifier: Apache-2.0
"""Device-side MoE slot sort (TF lever 5, first third): count / scan / place in three parallel GPU passes.

Replaces the host `_plan` of `moe_prefill` (device->host sync of the router ids, numpy bincount + stable
argsort, re-uploads) with GPU kernels. Inputs ids i32 [P] / wts f32 [P] stay on device; outputs are device
arrays: tok u32 [P] (sorted slot -> token), wts_s f32 [P] (sorted slot -> routing weight), inv u32 [P]
(original pair -> sorted slot), cum u32 [E+1] (expert offsets: expert e owns slots [cum[e], cum[e+1]]).

Two single-threadgroup kernels (all barriers are inside one threadgroup, so init + atomics fuse safely):
countscan (zero, histogram, prefix sum) and place (cursor init, scatter). Placement is deterministic
(thread order fixed), but that is belt and braces: a slot's downstream arithmetic never depends on where
the slot sits, so any consistent permutation is bit-identical.
"""
from __future__ import annotations

import mlx.core as mx

_SORTONE_SRC = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint E = params[0], P = params[1], TOPK = params[2];
    for (uint e = tid; e < E; e += 256u) counts[e] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < P; p += 256u) {
        const uint e = uint(ids[p]);
        atomic_fetch_add_explicit(((device atomic_uint*)counts) + uint(rank[e]), 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s = 0;
        for (uint e = 0; e < E; e++) { const uint c = counts[e]; cum[e] = s; s += c; }
        cum[E] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e <= E; e += 256u) cursor[e] = cum[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < P; p += 256u) {
        const uint e0 = uint(ids[p]);
        const uint e = uint(rank[e0]);
        const uint slot = atomic_fetch_add_explicit(((device atomic_uint*)cursor) + e, 1u, memory_order_relaxed);
        tok[slot] = p / TOPK;
        ids_s[slot] = int(e0);
        wts_s[slot] = wts[p];
        inv[p] = slot;
    }
"""

_kc: dict = {}


def _kernel(name, src, inputs, outputs):
    k = _kc.get(name)
    if k is None:
        k = _kc[name] = mx.fast.metal_kernel(name=f"kyojin_sort_{name}", input_names=inputs,
                                             output_names=outputs, source=src)
    return k


def plan_device(ids: mx.array, wts: mx.array, experts: int, topk: int, rank: mx.array | None = None):
    """Device-side slot sort. ids i32 [P], wts f32 [P] (any T, topk with T * topk = P).
    rank u32 [E] (optional): proc-position of each expert (e.g. grouped by bitrate class);
    counts/cum follow rank order. Returns
    (tok u32 [P], ids_s i32 [P] (sorted slot -> expert), wts_s f32 [P], inv u32 [P], cum u32 [E+1]).
    One kernel launch (count, scan and place as barrier-separated phases of a single threadgroup)."""
    import numpy as _np
    P = int(ids.size)
    assert wts.size == P and P % topk == 0
    if rank is None:
        rank = mx.array(_np.arange(experts, dtype=_np.uint32))
        mx.eval(rank)
    params = mx.array([experts, P, topk], dtype=mx.uint32)
    k = _kernel("sortone", _SORTONE_SRC, ["ids", "wts", "rank", "params"],
                ["counts", "cum", "cursor", "tok", "ids_s", "wts_s", "inv"])
    counts, cum, _, tok, ids_s, wts_s, inv = k(
        inputs=[ids.astype(mx.int32), wts.astype(mx.float32), rank, params], grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(experts,), (experts + 1,), (experts + 1,), (P,), (P,), (P,), (P,)],
        output_dtypes=[mx.uint32] * 3 + [mx.uint32, mx.int32, mx.float32, mx.uint32])
    return tok, ids_s, wts_s, inv, cum
