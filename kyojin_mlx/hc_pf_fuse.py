# SPDX-License-Identifier: Apache-2.0
"""Prompt-chunk mHC (hyper-connection) fusions for the sub-block seams of a decoder layer (op level: any model whose layers are
`x = hc.apply_branch(x, norm, branch)` with hc_mult = 4 and an rms_norm after the collapse; installed for GLM-5.x by glm_pf_glue).

Stock chain per sub-block at 8K rows (G0 / P1-PFPROF, ms of GPU, 268 MB per [S, 4, D] fp16 stream):
    pf_expand 0.55 -> (gather 0.52: `x[0]`, fixed by glm_pf_glue.VIEW) -> hcmix 0.61 -> sinkhorn + collapse 0.30 -> [cast f16->f32 0.18, rms_norm f32 0.20] or [rms_norm f16 0.08-0.16]
Here
  collapse_norm  (flag KYOJIN_HC_MIX_NORM)   ONE kernel for sinkhorn + collapse + the sub-block's rms_norm: the collapsed row is rounded to the stream dtype (as the stock kernel stores it),
                 then rms_norm exactly as MLX's single-row kernel (1024 threads x 4 contiguous elements, simd_sum, threadgroup sum, precise rsqrt: the same reduction tree, so the bits are the
                 stock bits). Output dtype = result_type(x, weight) like `mx.fast.rms_norm` (f16 x f16 w -> f16; f16 x f32 w -> f32, so the cast + f32 norm launches go).
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

NORM = os.environ.get("KYOJIN_HC_MIX_NORM", "0") != "0"
STATS = {"norm": 0}

_SINK = r"""
    if (sg == 0) {
        const float pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint llane = metal::min(lane, (uint)(HC - 1));
        float pre_z = mix[llane] * pre_scale + base[llane];
        float post_z = mix[HC + llane] * post_scale + base[HC + llane];
        float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
        if (lane < (uint)HC) { pre_shared[lane] = pre_v; post_out[lane] = post_v; }
        float4 v = (*(const device float4*)(mix + BASE_OFF + llane * HC) * comb_scale + *(const device float4*)(base + BASE_OFF + llane * HC)) * active;
        float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
        float4 e = metal::fast::exp(v - row_max) * active;
        float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
        float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
        r *= col_inv;
        for (int iter = 1; iter < ITERS; ++iter) {
            r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
            col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
            r *= col_inv;
        }
        if (lane < (uint)HC) { *(device float4*)(comb_out + lane * HC) = r; }
    }
"""

# the stock hc_sinkhorn_collapse (mlx_vlm deepseek_v4.hyper_connection) with 1024 threads, one float4 of the collapsed row each (D == 4096), + MLX's rms_norm single-row reduction
_COLLAPSE_NORM = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint lane = tid % 32;
    uint sg = tid / 32;
    constexpr int MIX = (2 + HC) * HC;
    constexpr int BASE_OFF = 2 * HC;
    constexpr float EPS = EPS_INT * 1e-9;
    const device float* mix = (const device float*)mixes + row * MIX;
    device float* post_out = (device float*)post + row * HC;
    device float* comb_out = (device float*)comb + row * HC * HC;
    threadgroup float pre_shared[HC];
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
""" + _SINK + r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float p0 = pre_shared[0], p1 = pre_shared[1], p2 = pre_shared[2], p3 = pre_shared[3];
    const device T* x_row = (const device T*)x_in + row * (HC * D);
    using T4 = vec<T, 4>;
    const device T4* x_row0 = (const device T4*)(x_row + 0 * D);
    const device T4* x_row1 = (const device T4*)(x_row + 1 * D);
    const device T4* x_row2 = (const device T4*)(x_row + 2 * D);
    const device T4* x_row3 = (const device T4*)(x_row + 3 * D);
    float4 x0 = float4(x_row0[tid]);
    float4 x1 = float4(x_row1[tid]);
    float4 x2 = float4(x_row2[tid]);
    float4 x3 = float4(x_row3[tid]);
    float4 result = fma(float4(p0), x0, fma(float4(p1), x1, fma(float4(p2), x2, float4(p3) * x3)));
    const float4 c = float4(T4(result));                                   // the stock kernel stores the collapse in the stream dtype; the norm reads that
    float acc = 0.0f;
    { float xi = c.x; acc += xi * xi; } { float xi = c.y; acc += xi * xi; } { float xi = c.z; acc += xi * xi; } { float xi = c.w; acc += xi * xi; }
    acc = simd_sum(acc);
    if (sg == 0) { local_sums[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { local_sums[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) { local_inv[0] = metal::precise::rsqrt(acc / float(D) + nrm_eps[0]); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = local_inv[0];
    const float cc[4] = {c.x, c.y, c.z, c.w};
    OT o[4];
    for (uint i = 0; i < 4; ++i) {
        const float w = float(nw[tid * 4 + i]);
        o[i] = OUT_EXPR;
    }
    device OT* out_row = (device OT*)normed + ulong(row) * D + tid * 4;
    for (uint i = 0; i < 4; ++i) { out_row[i] = o[i]; }
    TWIN_STORE
"""
# f16 stream x f16 weight: w * T(x * inv) in T (a half product);  f16 stream x f32 weight (output f32): w * (x * inv) in fp32
_OUT_SAME = "static_cast<OT>(w * float(static_cast<T>(cc[i] * inv)))"
_OUT_PROMOTE = "w * (cc[i] * inv)"

# the f32 (promoted) output also as fp16 = `normed.astype(float16)`: the MoE experts / shared expert read that cast (2 launches per MoE layer, 0.2-0.6 ms each at 8K rows)
_TWIN = r"""
    device half* outh = (device half*)normed_h + ulong(row) * D + tid * 4;
    for (uint i = 0; i < 4; ++i) { outh[i] = half(o[i]); }
"""
TWIN = os.environ.get("KYOJIN_HC_MIX_TWIN", "0") != "0"
_twin = [None, None]                                  # (the f32 array, its fp16 twin): the last norm output of a promoted sub-block


def twin_of(x):
    """fp16 twin of `x` when `x` is the f32 output of the last fused norm, else None (callers use it where they would `x.astype(float16)`)."""
    t = _twin
    return t[1] if (TWIN and t[0] is x) else None


_K = {}


def _kern(promote: bool, twin: bool = False):
    key = (promote, twin)
    k = _K.get(key)
    if k is None:
        src = _COLLAPSE_NORM.replace("OUT_EXPR", _OUT_PROMOTE if promote else _OUT_SAME)
        src = src.replace("TWIN_STORE", _TWIN if twin else "")
        k = _K[key] = mx.fast.metal_kernel(name="kyojin_hc_collapse_norm" + ("_p" if promote else "") + ("t" if twin else ""), input_names=["x_in", "mixes", "scale", "base", "nw", "nrm_eps"],
                                               output_names=["normed", "post", "comb"] + (["normed_h"] if twin else []), source=src, ensure_row_contiguous=True)
    return k


def _norm_type_why(norm):
    """None when `norm` is MLX's own RMSNorm (y = w * x * rsqrt(mean(x^2) + eps), the math of the kernel) with a float eps, else the reason. LayerNorm (centred, bias) and the model-local RMSNorm
    classes that scale by (1 + w) (Gemma family) have a `.weight` and an `.eps` too and would be fused with the wrong math: a type check, not duck typing."""
    import mlx.nn as nn
    t = type(norm)
    if not isinstance(norm, nn.RMSNorm) or t.__call__ is not nn.RMSNorm.__call__:
        return "norm_type=" + t.__name__
    if not isinstance(getattr(norm, "eps", None), (int, float)):
        return "norm_eps"
    return None


def norm_why(x, norm, hc_mult: int, D: int):
    """None when `collapse_norm` serves this call, else the reason (flag audit key)."""
    if not NORM:
        return "flag_off"
    if hc_mult != 4 or D != 4096:
        return "shape"                                   # one float4 per thread of a 1024-thread group = MLX's rms_norm single-row geometry for 4096
    if x.dtype != mx.float16:
        return "x_dtype=" + str(x.dtype).split(".")[-1]
    why = _norm_type_why(norm)
    if why:
        return why
    w = getattr(norm, "weight", None)
    if w is None or w.shape != (D,) or w.dtype not in (mx.float16, mx.float32):
        return "norm_weight"
    if mx.default_device() != mx.gpu:
        return "no_gpu"
    return None


def collapse_norm(x, mixes, scale, base, w, nrm_eps: float, hc_eps: float, iters: int):
    """x [1, S, 4, D] fp16, mixes [1, S, 24] fp32 -> (rms_norm(collapse(x)) [1, S, D] in result_type(x, w), post [1, S, 4], comb [1, S, 4, 4])."""
    B, L, H, D = x.shape
    promote = w.dtype == mx.float32
    out_t = mx.float32 if promote else x.dtype
    STATS["norm"] += 1
    twin = promote and TWIN
    k = _kern(promote, twin)
    outs = k(inputs=[x, mixes, scale, base, w, mx.array([nrm_eps], dtype=mx.float32)],
             template=[("T", x.dtype), ("OT", out_t), ("HC", H), ("ITERS", iters), ("D", D), ("EPS_INT", round(hc_eps / 1e-9))],
             grid=(B * L * 1024, 1, 1), threadgroup=(1024, 1, 1),
             output_shapes=[(B, L, D), (B, L, H), (B, L, H, H)] + ([(B, L, D)] if twin else []),
             output_dtypes=[out_t, mx.float32, mx.float32] + ([mx.float16] if twin else []))
    _twin[0], _twin[1] = (outs[0], outs[3]) if twin else (None, None)
    return outs[0], outs[1], outs[2]
