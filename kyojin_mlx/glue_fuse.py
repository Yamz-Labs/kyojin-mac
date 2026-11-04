# SPDX-License-Identifier: Apache-2.0
"""Decode-step glue: the mHC mix / collapse / RMSNorm chain of a block, with fewer launches per sub-layer.

The block body of a DS4.1 decode layer runs, per sub-layer, four small launches of pure glue around the attention / FFN:

    mixes = rms_norm(h.flatten(-2)) @ hc_fn.T          (2 launches: the norm of the flattened streams, then the 24 x hc*hidden row dot)
    mix   = hc_split_sinkhorn(mixes, ...)              (1)
    y     = hc_pre(h, pre_mix)                         (1)
    x     = attn_norm(y)                               (1)

`mix_norm` merges the last two into one launch: the collapse feeds only the norm that consumes it, so y is never written to memory. The norm's
element layout and reduction are the ones of `mx.fast.rms_norm` (4 contiguous values per thread, stride threads_per_threadgroup * 4, fp32
partials, simd_sum, then the threadgroup partials in simdgroup index order; the same layout decfuse's add+RMSNorm kernel reproduces), so the
norm output is bit-identical to the stock pair. The sinkhorn text is the mlx-vlm kernel's, verbatim (as in fuse._MIXPRE), in the same
threadgroup after the norm's reduction.

The first two are left alone: fusing them (one threadgroup per row, the mix dots reading the normalised row out of registers instead of the 20480-wide
fp32 buffer) needs another summation order for the dots, and on the random-weight stack that moved the logits by rel L2 1e-1 while costing 35 % more
time (the mix coefficients feed the sinkhorn, which is not contractive in this fixture): measured and rejected, see the report.

One threadgroup per row, so a row's bits do not depend on how many rows share the launch (multi-user decode, verify rows). Switch: module
attribute ENABLED (env KYOJIN_GLUE_FUSE=1 and KYOJIN_GLUE_FUSE_NORM=1, default off).

Measured on the synthetic 12-layer stack (tools/ds41_glue_bench.py, interleaved steps, paired medians of 25-41 steps): 359 -> 331 launches per
step (12 fewer RMSNorm), but the step is at best unchanged and with the mix+collapse fusion off (fuse.MIXPRE=0, the arrangement of the batched
served path) it is 0.146 ms/12 layers SLOWER. A chain of the block glue alone (stub sublayers, 86 chained blocks) puts the same number at
+22 us/block. Cause: a fused row reduction serialises inside one threadgroup what two small launches overlap on the grid, and the launch it saves
costs less than the barrier and the serial tail it adds; the long dense / MoE kernels of a real step partly hide it, which is why the full-step
difference is inside the noise of the machine. The same mechanism makes the existing one-launch mix+collapse (fuse.mhc_mix_pre, KYOJIN_FUSE_MIXPRE,
on by default) a loss on that path: turning it off is bit-identical and -0.31 .. -0.69 ms per 8-14 layers in 4 of 4 runs (90/122 steps faster). So this module is kept as
the bit-identical reference of the fused chain, default OFF; the number that pays is not the launch count but the split. Details in
oc-runs/glueroof-ds_REPORT.md.
"""
from __future__ import annotations

from . import archs as _A
import os

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import ablit as AB
from . import fired as FD

# One glue lever (cloud/day2 merge): KYOJIN_GLUE_FUSE=1 runs the gluefuse cuts (mixvec, mixargs, mix+collapse fold off); this module's
# collapse + norm launch is the extra opt-in sub-switch KYOJIN_GLUE_FUSE_NORM=1 on top (measured neutral to slower above, so off by default).
ENABLED = os.environ.get("KYOJIN_GLUE_FUSE", "1") == "1" and os.environ.get("KYOJIN_GLUE_FUSE_NORM", "0") == "1"

_SINKHORN = r"""
    // mlx-vlm hc_sinkhorn_split, verbatim for one row (one simdgroup, 32 lanes): pre' / post / comb of the mix.
    {
        const uint lane = thread_index_in_simdgroup;
        constexpr int HC = 4;
        constexpr int MIX = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr float EPS = EPS_INT * 1e-9;
        const device float* mix      = (const device float*)mixes + b * MIX;
        device float*       pre_out  = (device float*)pre_o + b * HC;
        device float*       post_out = (device float*)post + b * HC;
        device float*       comb_out = (device float*)comb + b * HC * HC;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint  llane  = metal::min(lane, (uint)(HC - 1));
        float pre_z  = mix[llane]      * pre_scale  + base[llane];
        float post_z = mix[HC + llane] * post_scale + base[HC + llane];
        float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
        if (lane < (uint)HC) {
            pre_out[lane]  = pre_v;
            post_out[lane] = post_v;
        }
        float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                        * comb_scale
                  + *(const device float4*)(base + BASE_OFF + llane * HC))
                 * active;
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
        if (lane < (uint)HC) {
            *(device float4*)(comb_out + lane * HC) = r;
        }
    }
"""

# Collapse + RMSNorm, one threadgroup per row: the layout and the reduction of mx.fast.rms_norm.
_COLLAPSE = r"""
                float a = 0.0f;
                {
#pragma METAL fp contract(off)
                    for (uint j = 0; j < 4; ++j) a += pre[b * 4 + j] * float(x[(ulong(b) * 4 + j) * D + i0 + i]);
                }
"""

_MIXNORM = r"""
    const uint b = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint sl = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint ntg = threads_per_threadgroup.x;
    threadgroup float part[32];
    T keep[NG][4];                                              // the rounded collapse of this thread's own elements: no second pass over x
    float acc = 0.0f;
    for (uint g = 0; g < NG; ++g) {
        const uint i0 = lid * 4 + g * ntg * 4;
        for (uint i = 0; i < 4; i++) {
            if (i0 + i < D) {
                COLLAPSE
                const T hv = static_cast<T>(is_same<T, half>::value ? clamp(a, -65504.0f, 65504.0f) : a);
                keep[g][i] = hv;
                const float f = float(hv);
                acc += f * f;
            }
        }
    }
    acc = simd_sum(acc);
    if (sl == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float t = simd_sum(part[sl]);
        if (sl == 0) part[0] = metal::precise::rsqrt(t / (float)D + eps[0]);
        SINKHORN
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = part[0];
    for (uint g = 0; g < NG; ++g) {
        const uint i0 = lid * 4 + g * ntg * 4;
        for (uint i = 0; i < 4; i++) {
            if (i0 + i < D) nout[ulong(b) * D + i0 + i] = w[i0 + i] * static_cast<T>(float(keep[g][i]) * inv);
        }
    }
"""

_k: dict = {}
_const: dict = {}


def _kernel(name: str, inputs: list, outputs: list, source: str):
    k = _k.get(name)
    if k is None:
        k = _k[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source)
    return k


def _f32(v: float) -> mx.array:
    a = _const.get(v)
    if a is None:
        a = _const[v] = mx.array([v], dtype=mx.float32)
    return a


def mix_norm(mixes: mx.array, x: mx.array, pre: mx.array, w: mx.array, eps: float, scale: mx.array, base: mx.array, iters: int,
             hc_eps: float):
    """hc_pre(x, pre) followed by rms_norm of it, in one launch. mixes [B, 1, 24] fp32 (sinkhorn input), x [B, 1, 4, D] half, pre [B, 1, 4]
    fp32, w [D] half -> (normed [B, 1, D], pre' [B, 1, 4], post [B, 1, 4], comb [B, 1, 4, 4]). Bit-identical to mhc_pre + nn.RMSNorm."""
    B, S, H, D = x.shape
    assert S == 1 and H == 4 and pre.dtype == mx.float32 and mixes.shape == (B, 1, 24) and w.dtype == x.dtype, (x.shape, mixes.shape, w.dtype)
    k = _kernel("kyojin_glue_mix_norm", ["mixes", "scale", "base", "x", "pre", "w", "eps"], ["nout", "pre_o", "post", "comb"],
                _MIXNORM.replace("COLLAPSE", _COLLAPSE).replace("SINKHORN", _SINKHORN))
    tg = min(1024, 32 * (-(-D // 128)))
    return k(inputs=[mixes.astype(mx.float32), scale.astype(mx.float32), base.astype(mx.float32), x, pre, w, _f32(eps)],
             template=[("T", x.dtype), ("D", D), ("ITERS", iters), ("EPS_INT", round(hc_eps / 1e-9)),
                       ("NG", -(-D // (tg * 4)))],
             grid=(tg, B, 1), threadgroup=(tg, 1, 1),
             output_shapes=[(B, 1, D), (B, 1, 4), (B, 1, 4), (B, 1, 4, 4)],
             output_dtypes=[x.dtype, mx.float32, mx.float32, mx.float32])


# --------------------------------------------------------------------------------------------------------------- block body (one row)
def block_ok(blk, h, pre_mix, image_mask, cache) -> bool:
    """The fused body applies to one row of a DS4.1 block whose norms are plain fp16 RMSNorms and whose mixers are the 24 -> 4 mHC ones."""
    import mlx.nn as nn
    if not ENABLED or h.ndim != 4 or h.shape[1] != 1 or h.shape[2] != KG.HC or h.dtype != mx.float16 or pre_mix.dtype != mx.float32 \
            or pre_mix.shape != (h.shape[0], 1, KG.HC) or getattr(blk, "hc_mult", 0) != KG.HC or h.shape[-1] % KG.HIDDEN_MULT:
        return False
    if image_mask is not None and (image_mask.dtype != mx.bool_ or image_mask.size != h.shape[0]):
        return False                                                     # the mask only steers the router: any value passes through to the FFN
    for nm in ("attn_norm", "ffn_norm"):
        nrm = getattr(blk, nm, None)
        if type(nrm) is not nn.RMSNorm or nrm.weight.dtype != h.dtype or nrm.weight.shape != (h.shape[-1],):
            return False
    return True


def site_installed(blk) -> bool:
    """True when the block's `hc_mixes` is the rows site (rows.mix_vector + sinkhorn). The site is a method of the per-instance subclass rows._override
    creates; later overrides (CED / speculative `__call__` subclasses) stack another subclass on top, so the marker is searched along the MRO,
    the way attribute lookup finds the method."""
    for c in type(blk).__mro__:
        m = c.__dict__.get("hc_mixes")
        if m is not None:
            return getattr(m, "_kyojin_site", None) is not None
    return False


def mixes_of(blk, x, fn) -> mx.array:
    """The block's mix coefficients [B, 1, 24] fp32, through the same chain the ambient block body uses: rows.mix_vector (rms_norm + rowdot) when
    the mHC site is installed, else the model's own rms_norm + matmul, so the glue kernels keep the bits of the unfused path."""
    from . import rows as RW
    if not site_installed(blk):
        z = mx.fast.rms_norm(x.flatten(-2).astype(mx.float32), None, blk.norm_eps)     # the model's own chain when the mHC site is absent
        return z @ fn.T
    return RW.mix_vector(blk, x, fn)


def block(blk, h, pre_mix, image_mask, cache):
    """DeepseekV41Block.__call__ for one row with the mix / collapse / norm glue fused (everything else runs the patched ops)."""
    L = _lang()
    FD.hit("glue_fuse_norm")
    residual = h
    mixes = mixes_of(blk, h, blk.hc_attn_fn)
    x, attn_pre, attn_post, attn_comb = mix_norm(mixes, h, pre_mix, blk.attn_norm.weight, blk.attn_norm.eps, blk.hc_attn_scale,
                                                 blk.hc_attn_base, blk.hc_sinkhorn_iters, blk.hc_eps)
    x = AB.attn_out(blk, blk.attn(x, cache))
    x = L.hc_expand(x, residual, attn_post, attn_comb)
    residual = x
    mixes = mixes_of(blk, x, blk.hc_ffn_fn)
    y, ffn_pre, ffn_post, ffn_comb = mix_norm(mixes, x, attn_pre, blk.ffn_norm.weight, blk.ffn_norm.eps, blk.hc_ffn_scale,
                                              blk.hc_ffn_base, blk.hc_sinkhorn_iters, blk.hc_eps)
    y = AB.ffn_out(blk, blk.ffn(y, image_mask))
    y = L.hc_expand(y, x, ffn_post, ffn_comb)
    return y, ffn_pre


def _lang():
    L = _A.lang()
    return L
