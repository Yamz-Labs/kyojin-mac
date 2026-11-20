# SPDX-License-Identifier: Apache-2.0
"""Adapted in part (the structure: previous expand folded into the next pre kernel, sinkhorn / comb off the dependent chain) from oMLX (https://github.com/jundot/omlx), Apache-2.0,
omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py (`hc_pre_fused`, `hc_post_mm`, `hc_expand_one`); the Metal sources here are our own.

GLM-5.3 mHC glue at decode, ONE launch on the dependent chain per branch (flag `KYOJIN_GLM_HC_DEFER`, G10d; needs KYOJIN_GLM_HC_FUSE).

G3b (`glm_hc`) runs per branch   A (expand of the previous branch + partial sums)  ->  B (mixes, sinkhorn, collapse, norm)  ->  [branch]: two launches in series, and B is one
threadgroup per token whose sinkhorn (20 iterations of a simdgroup) is on the critical path. The branch input only needs `pre` (4 of the 24 mixes) and the norm; `post` / `comb`
are only needed by the expand AFTER the branch. So per branch:

    P  (`_P`)   6 threadgroups per token. Each recomputes the new streams h = round(post * y + comb^T x) of the previous branch (the expand, same fp32 order as glm_hc A) from
                (y = the previous branch output, x = its input streams, post / comb), then 4 of the 24 mix dot products and sum(h^2) against the mix weight. Threadgroup 0
                (mixes 0..3 = pre) also collapses the streams with pre, applies the branch's input RMSNorm and writes h (the next P and the final expand read it).
    Q  (`_Q`)   sinkhorn / post / comb of the 24 mixes (glm_hc B's arithmetic), one simdgroup per token. Independent of the branch: it runs beside it.
    [branch]
    E           (only at the end of the model) the expand of the last branch: the stock A kernel.
The array returned by a layer is the LAZY expand E(y, h, post, comb): the next layer's P does not read it (it takes y, h, pc from a one-slot side channel keyed by its identity),
so the launch exists only if something else consumes it (the head, a hook, a miss). Nothing but identity is trusted.

Exactness: the expand, pre / post / comb, collapse and norm are glm_hc's arithmetic; the mix dot products are summed in another fixed order (per-thread, simdgroup, threadgroup) than
G3b's per-64-wide-chunk partials: fp32 rounding-level differences of the mixes (1e-7 relative). Every token is reduced on its own (row invariance)."""
from __future__ import annotations

import functools
import os

import mlx.core as mx

from . import fired as FD
from . import glm_hc as HC

ENABLED = os.environ.get("KYOJIN_GLM_HC_DEFER", "0") not in ("0", "")
STATS = {"fused": 0, "miss": 0, "stock": 0}

_P_COMMON = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.x, tok = threadgroup_position_in_grid.y;
    constexpr uint K = 4 * H, NV = H / 1024;
    constexpr float EPS = EPS_INT * 1e-9f;
    constexpr float NEPS = NEPS_INT * 1e-9f;
    threadgroup float red[8 * 5];
    threadgroup float tot[5];
    threadgroup float wsum[8];
    float4 hv[NV][4];
    DEFER_LOAD
    float ss = 0.0f;
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = 0; i < NV; ++i) {
        const uint h4 = tid + i * 256;
        for (uint s = 0; s < 4; ++s) {
            const float4 v = hv[i][s];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            for (uint r = 0; r < 4; ++r) {
                const float4 f = float4(((const device vec<FT, 4>*)(fn + (size_t)(4 * g + r) * K + s * H))[h4]);
                d[r] += v.x * f.x + v.y * f.y + v.z * f.z + v.w * f.w;
            }
        }
    }
    {
        const float r0 = simd_sum(ss), r1 = simd_sum(d[0]), r2 = simd_sum(d[1]), r3 = simd_sum(d[2]), r4 = simd_sum(d[3]);
        if (lane == 0) { red[sg * 5] = r0; red[sg * 5 + 1] = r1; red[sg * 5 + 2] = r2; red[sg * 5 + 3] = r3; red[sg * 5 + 4] = r4; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 5) {
        float t = 0.0f;
        for (uint q = 0; q < 8; ++q) t += red[q * 5 + tid];
        tot[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rs = metal::precise::rsqrt(tot[0] / float(K) + NEPS);
    if (tid < 4) { mixes[tok * 24 + 4 * g + tid] = tot[1 + tid] * rs; }
    if (g != 0) { return; }

    // threadgroup 0: pre (mixes 0..3), collapse, the branch's input RMSNorm
    float p[4];
    for (uint s = 0; s < 4; ++s) {
        const float z = tot[1 + s] * rs * scale[0] + base[s];
        p[s] = 1.0f / (1.0f + metal::fast::exp(-z)) + EPS;
    }
    float4 col[NV];
    float ss2 = 0.0f;
    for (uint i = 0; i < NV; ++i) {
        float4 a = p[0] * hv[i][0];
        a = metal::fma(float4(p[1]), hv[i][1], a);
        a = metal::fma(float4(p[2]), hv[i][2], a);
        a = metal::fma(float4(p[3]), hv[i][3], a);
        a = float4(vec<T, 4>(a));
        col[i] = a;
        ss2 += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    ss2 = simd_sum(ss2);
    if (lane == 0) wsum[sg] = ss2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t2 = 0.0f;
    for (uint q = 0; q < 8; ++q) t2 += wsum[q];
    const float nrm = metal::precise::rsqrt(t2 / float(H) + NEPS);
    for (uint i = 0; i < NV; ++i) {
        const uint h4 = tid + i * 256;
        const vec<T, 4> qv = vec<T, 4>(col[i] * nrm);
        const vec<T, 4> w4 = ((const device vec<T, 4>*)nw)[h4];
        ((device vec<T, 4>*)xo)[tok * (H / 4) + h4] = w4 * qv;
    }
    WRITE_H
"""

_DEFER_LOAD = r"""
    const device float* pcp = pc + tok * 20;
    for (uint i = 0; i < NV; ++i) {
        const uint h4 = tid + i * 256;
        const float4 yv = float4(((const device vec<T, 4>*)(y + tok * H))[h4]);
        float4 xv[4];
        for (uint j = 0; j < 4; ++j) xv[j] = float4(((const device vec<T, 4>*)(x + (tok * 4 + j) * H))[h4]);
        for (uint s = 0; s < 4; ++s) {
            float4 a = pcp[s] * yv;
            for (uint j = 0; j < 4; ++j) a += pcp[4 + j * 4 + s] * xv[j];
            hv[i][s] = float4(vec<T, 4>(a));
        }
    }
"""
_PLAIN_LOAD = r"""
    for (uint i = 0; i < NV; ++i) {
        const uint h4 = tid + i * 256;
        for (uint s = 0; s < 4; ++s) hv[i][s] = float4(((const device vec<T, 4>*)(x + (tok * 4 + s) * H))[h4]);
    }
"""
_WRITE_H = r"""
    for (uint i = 0; i < NV; ++i) {
        const uint h4 = tid + i * 256;
        for (uint s = 0; s < 4; ++s) ((device vec<T, 4>*)(hout + (tok * 4 + s) * H))[h4] = vec<T, 4>(hv[i][s]);
    }
"""

_Q = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint tok = threadgroup_position_in_grid.x;
    constexpr float EPS = EPS_INT * 1e-9f;
    const device float* mixv = mixes + tok * 24;
    const float post_scale = scale[1], comb_scale = scale[2];
    const float active = (lane < 4u) ? 1.0f : 0.0f;
    const uint llane = metal::min(lane, 3u);
    float post_z = mixv[4 + llane] * post_scale + base[4 + llane];
    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    if (lane < 4u) { pc[tok * 20 + lane] = post_v; }
    float4 v4 = (float4(mixv[8 + llane * 4], mixv[9 + llane * 4], mixv[10 + llane * 4], mixv[11 + llane * 4]) * comb_scale
                 + *(const device float4*)(base + 8 + llane * 4)) * active;
    float row_max = metal::max(metal::max(v4.x, v4.y), metal::max(v4.z, v4.w));
    float4 e = metal::fast::exp(v4 - row_max) * active;
    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    r *= col_inv;
    for (int it = 1; it < ITERS; ++it) {
        r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
        col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
        r *= col_inv;
    }
    if (lane < 4u) *(device float4*)(pc + tok * 20 + 4 + lane * 4) = r;
"""


@functools.lru_cache(maxsize=None)
def _kernel_p(defer: bool):
    src = _P_COMMON.replace("DEFER_LOAD", _DEFER_LOAD if defer else _PLAIN_LOAD).replace("WRITE_H", _WRITE_H if defer else "")
    ins = (["pc", "y"] if defer else []) + ["x", "fn", "scale", "base", "nw"]
    outs = ["xo", "mixes"] + (["hout"] if defer else [])
    return mx.fast.metal_kernel(name="kyojin_glm_hc_p" + ("d" if defer else ""), input_names=ins, output_names=outs, source=src)


@functools.lru_cache(maxsize=None)
def _kernel_q():
    return mx.fast.metal_kernel(name="kyojin_glm_hc_q", input_names=["mixes", "scale", "base"], output_names=["pc"], source=_Q)


def pre(x, hc, norm_w, norm_eps, deferred=None):
    """P: (branch input [B, L, H], mixes [T, 24] fp32, new streams h [B, L, 4, H] or None).
    deferred = (y, pc): `x` are the INPUT streams of the previous branch, y its output, pc its post / comb (Q): the streams of this branch are recomputed here."""
    B, L, _, H = x.shape
    T = B * L
    d = deferred is not None
    ins = ([deferred[1], deferred[0]] if d else []) + [x, hc.fn, hc.scale, hc.base, HC._norm_w(norm_w, x.dtype)]
    shapes = [(B, L, H), (T, 24)] + ([x.shape] if d else [])
    dts = [x.dtype, mx.float32] + ([x.dtype] if d else [])
    r = _kernel_p(d)(inputs=ins, template=[("T", x.dtype), ("FT", hc.fn.dtype), ("H", H), ("EPS_INT", round(hc.hc_eps / 1e-9)), ("NEPS_INT", round(norm_eps / 1e-9))],
                     grid=(6 * 256, T, 1), threadgroup=(256, 1, 1), output_shapes=shapes, output_dtypes=dts)
    return r[0], r[1], (r[2] if d else None)


def post_comb(mixes, hc):
    """Q: post [T, 4] / comb [T, 16] as one [T, 20] fp32 array (the layout of glm_hc's pc)."""
    T = int(mixes.shape[0])
    return _kernel_q()(inputs=[mixes, hc.scale, hc.base], template=[("ITERS", hc.sinkhorn_iters), ("EPS_INT", round(hc.hc_eps / 1e-9))], grid=(T * 32, 1, 1), threadgroup=(32, 1, 1),
                       output_shapes=[(T, 20)], output_dtypes=[mx.float32])[0]


# ------------------------------------------------------------------------------------------------------------------------------ layer call
class _Slot:
    x = None
    y = None
    h = None
    pc = None
    a = None


SLOT = _Slot()


def _take(x, hc):
    s = SLOT
    if s.x is x and s.a is hc:
        r = (s.y, s.h, s.pc)
        s.x = s.y = s.h = s.pc = s.a = None
        return r
    return None


def layer_call(self, x, mask=None, cache=None, prev_topk_indices=None):
    a, f = self.attn_hc, self.ffn_hc
    STATS["fused"] += 1
    FD.fire("hc_defer")
    d = _take(x, a)
    if d is None:
        if SLOT.x is not None or STATS["fused"] > 1:
            STATS["miss"] += 1
        xo, mx_a, _ = pre(x, a, self.input_layernorm.weight, self.input_layernorm.eps)
        h_a = x
    else:
        xo, mx_a, h_a = pre(d[1], a, self.input_layernorm.weight, self.input_layernorm.eps, deferred=(d[0], d[2]))
    pc_a = post_comb(mx_a, a)
    if self.is_linear:
        o = self.self_attn(xo, mask, cache)
        topk = prev_topk_indices
    else:
        o, topk = self.self_attn(xo, mask, cache, prev_topk_indices)
    xo, mx_f, h_f = pre(h_a, f, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps, deferred=(o, pc_a))
    pc_f = post_comb(mx_f, f)
    o2 = self.mlp(xo)
    x2, _ = HC.expand(o2, h_f, pc_f, None)                                  # lazy: only a consumer other than the next layer's P evaluates it
    nxt = self.__dict__.get("_g3b_next")                                    # already layers[i+1].attn_hc (glm_hc.install), not the layer itself
    if nxt is not None:
        SLOT.x, SLOT.y, SLOT.h, SLOT.pc, SLOT.a = x2, o2, h_f, pc_f, nxt
    return x2, topk
