# SPDX-License-Identifier: Apache-2.0
"""Router gate GEMV + top-k selection in one launch (launchcut group `router`; KYOJIN_LAUNCHCUT=1).

The V4.1 router is two launches: rowinv.rowdot (logits = x @ gate.T, fp32, one threadgroup per expert) and fuse.router_select (sqrt(softplus), the biased
top-k, the weight normalisation, one threadgroup per row). Here the gate GEMV is the rowdot kernel unchanged (same text, same reduction layout, so the same logits
bit for bit); each threadgroup stores its logit with a relaxed atomic store, then bumps a per-row-group counter (the idiom of metal_src._FUSED_TAIL); the
threadgroup that completes the group (N arrivals) resets the counter and runs the selection for the rows of the group. The selection text is fuse._ROUTER's, with
the 128 threads of its threadgroup replaced by the 32 of one simdgroup: every step is per element, the key choice is the exact (min key, min index) pair and the
weight sum is sequential in one thread, so the result does not depend on the thread count.

Needs KS = 1 (pick_ks(K, E) == 1, true for the gate: 384 outputs), fp32 logits, an fp32 gate bias, a gate weight in fp32 or its exact narrow copy (lowp.exact_lowp), fewer than 64 row groups. Anything else, and the
non-Metal case, runs the two stock launches. `route_ref` is the stock composition (the reference of the Metal test)."""
from __future__ import annotations

import mlx.core as mx

from . import fuse as F
from . import rowinv as RI

_NCTR = 64                                                                         # row groups (of RI.ROWDOT_MAX rows) one counter array serves
_k: dict = {}
_ctr: dict = {}
_par: dict = {}

_SELECT = r"""
    // ---- selection (fuse._ROUTER with one simdgroup): the last arrival of the row group, rows rb .. rb + MR - 1 ----
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup float sval[E];
    threadgroup float skey[E];
    threadgroup uint chosen[TK];
    const float inf = metal::numeric_limits<float>::infinity();
    const device atomic_uint* lgp = (const device atomic_uint*)(y);
    for (uint rr = 0; rr < MR; ++rr) {
        const uint r = rb + rr;
        if (r >= ROWS) { break; }
        for (uint e = lane; e < E; e += 32) {
            const float x = as_type<float>(atomic_load_explicit(lgp + ulong(r) * E + e, memory_order_relaxed));
            const float maxv = metal::max(x, 0.0f);
            const float minv = metal::min(x, 0.0f);
            float sp = (minv == -inf || maxv == inf) ? maxv : (maxv + k_log1p(metal::exp(minv - maxv)));
            if (metal::isnan(x)) { sp = metal::numeric_limits<float>::quiet_NaN(); }
            const float s = metal::precise::sqrt(sp);
            sval[e] = s;
            const float b = BIAS_EXPR;
            skey[e] = -(s + b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < TK; ++j) {
            float bk = inf;
            uint bi = 0xffffffffu;
            for (uint e = lane; e < E; e += 32) {
                const float k = skey[e];
                if (k < bk || (k == bk && e < bi)) { bk = k; bi = e; }
            }
            const float mk = simd_min(bk);
            const uint mi = simd_min(bk == mk ? bi : 0xffffffffu);
            if (lane == 0) {
                chosen[j] = mi;
                skey[mi] = inf;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            float w[TK];
            float sum = 0.0f;
            for (uint j = 0; j < TK; ++j) { w[j] = sval[chosen[j]]; sum += w[j]; }
            const float den = sum + par[1];
            for (uint j = 0; j < TK; ++j) {
                ids[ulong(r) * TK + j] = int(chosen[j]);
                const float wn = NORM ? (w[j] / den) : w[j];
                wts[ulong(r) * TK + j] = wn * par[0];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_ARRIVE = r"""
    threadgroup uint last_;
    if (sg == 0 && lane == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit(((device atomic_uint*)(ctr)) + threadgroup_position_in_grid.y, 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last_ = (old + 1u == N) ? 1u : 0u;
        if (last_ != 0u) { atomic_store_explicit(((device atomic_uint*)(ctr)) + threadgroup_position_in_grid.y, 0u, memory_order_relaxed); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last_ == 0u) { return; }
"""

_STORE = "if (rb + r < ROWS) y[ulong(rb + r) * N + o] = static_cast<YT>(s);"
_STORE_ATOM = "if (rb + r < ROWS) { atomic_store_explicit(((device atomic_uint*)(y)) + ulong(rb + r) * N + o, as_type<uint>(s), memory_order_relaxed); }"


def _source(masked: bool) -> str:
    assert _STORE in RI._ROWDOT
    gemv = RI._ROWDOT.replace(_STORE, _STORE_ATOM)
    sel = _SELECT.replace("BIAS_EXPR", "(msk[r] ? bias_vl[e] : bias[e])" if masked else "bias[e]")
    return gemv + _ARRIVE + sel


def route_ref(x: mx.array, weight: mx.array, bias: mx.array, top_k: int, norm: bool, scale: float, bias_vl=None, image_mask=None):
    """The two launches this one replaces: rowdot + router_select (Metal). x [R, K] (fp16 / bf16), weight [E, K] fp32."""
    return F.router_select(RI.rowdot(x, weight), bias, top_k, norm, scale, bias_vl, image_mask)


def route_ok(x: mx.array, weight: mx.array, bias: mx.array, image_mask=None) -> bool:
    """x [R, K]: the fused launch applies (see the module doc)."""
    if x.ndim != 2 or weight.ndim != 2 or x.dtype not in (mx.float16, mx.bfloat16) or weight.dtype not in (mx.float32, mx.bfloat16, mx.float16) or bias.dtype != mx.float32:
        return False
    R, K = x.shape
    E = weight.shape[0]
    return bool(RI.pick_ks(K, E) == 1 and -(-R // min(R, RI.ROWDOT_MAX)) <= _NCTR and E % 32 == 0 and bias.shape == (E,) and weight.shape[1] == K
                and (image_mask is None or image_mask.size == R))


def _counter(weight: mx.array) -> mx.array:
    """The arrival counters of one gate (zero between launches: the completing threadgroup resets its own). Created and evaluated outside a trace (the first,
    stock call of a block reaches the router before the regions trace)."""
    c = _ctr.get(id(weight))
    if c is None:
        arr = mx.zeros((_NCTR,), dtype=mx.uint32)
        mx.eval(arr)
        c = _ctr[id(weight)] = (weight, arr)                                       # the weight is kept alive so its id is not reused
    return c[1]


def route(x: mx.array, weight: mx.array, bias: mx.array, top_k: int, norm: bool, scale: float, bias_vl=None, image_mask=None):
    """x [R, K], weight [E, K] -> (ids int32 [R, top_k], weights fp32 [R, top_k]) in one launch; the values of route_ref."""
    masked = image_mask is not None
    kern = _k.get(masked)
    if kern is None:
        kern = _k[masked] = mx.fast.metal_kernel(
            name=f"kyojin_route{int(masked)}", input_names=["x", "w", "params", "bias"] + (["bias_vl", "msk"] if masked else []) + ["par", "ctr"],
            output_names=["y", "ids", "wts"], header=F._HEADER + F._LOG1P + RI._LD8, source=_source(masked))
    R, K = x.shape
    E = weight.shape[0]
    mr = min(R, RI.ROWDOT_MAX)
    p = RI._params.get((K, E, R))
    if p is None:
        p = RI._params[(K, E, R)] = mx.array([K, E, R], dtype=mx.uint32)
    q = _par.get(scale)
    if q is None:
        q = _par[scale] = mx.array([scale, 1e-20], dtype=mx.float32)
    ins = [x, weight, p, bias] + ([bias_vl, image_mask.reshape(R)] if masked else []) + [q, _counter(weight)]
    return kern(inputs=ins, template=[("MR", mr), ("KS", 1), ("YT", mx.float32), ("V8", int(K % 8 == 0)), ("E", E), ("TK", top_k), ("NORM", bool(norm))],
                grid=(E * 32, (R + mr - 1) // mr, 1), threadgroup=(32, 1, 1),
                output_shapes=[(R, E), (R, top_k), (R, top_k)], output_dtypes=[mx.float32, mx.int32, mx.float32])[1:]
