# SPDX-License-Identifier: Apache-2.0
"""Launch cuts for the decode step: small ops that the stock model spreads over many kernels become one kernel each, with the arithmetic of the
stock sequence (same formulas, same summation order, `fp contract(off)`), so the result is bit-identical to the unfused path. tests/test_fuse.py
compares every kernel with the stock op sequence on many random inputs (ties and extremes included).

Why: a decode token of a 40-layer model is ~5000 launches; the CPU needs ~5 us per launch (python build + encode), so on a wide GPU (M5 Ultra) the
step is CPU bound, not bandwidth bound. Every launch removed is ~5 us of the token time.

Each cut has a switch (module attribute, env KYOJIN_FUSE=0 turns all off) to A/B inside one process. Every kernel is row independent: a row gets
the same bits whatever the number of rows (multi-user decode and speculative verify rely on it).

  router_select   sqrt(softplus(logits)) + correction bias, top-k by (value desc, index asc), weights normalised and scaled: 14 launches -> 2
  mhc_expand      hyper-connection expand (post * x + comb^T residual) for one token: 5 launches -> 1, also writes the fp32 copy the next mix reads
  mhc_pre         hyper-connection collapse (sum of pre-weighted copies): 3 launches -> 1
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import ablate as AZ                                                     # probe stand-ins (KYOJIN_ABLATE), off = one attribute test

ENABLED = os.environ.get("KYOJIN_FUSE", "1") != "0"
ROUTER = True
MHC = True
MIXPRE = os.environ.get("KYOJIN_FUSE_MIXPRE", "1") != "0"

_HEADER = "#pragma METAL fp contract(off)\n"

# log1p of mlx/backend/metal/kernels/utils.h (Goldberg), so that softplus matches mx.logaddexp(x, 0) bit for bit
_LOG1P = r"""
inline float k_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
  if (xp1 == 1.0f) { return x; }
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}
"""

_ROUTER = r"""
    // one threadgroup (128 threads) per row: scores, selection key, K rounds of (min key, min index), normalised weights.
    const uint r = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float sval[E];
    threadgroup float skey[E];
    threadgroup float rk[4];
    threadgroup uint ri[4];
    threadgroup uint chosen[K];
    const float inf = metal::numeric_limits<float>::infinity();
    for (uint e = t; e < E; e += 128) {
        const float x = lg[ulong(r) * E + e];
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
    for (uint j = 0; j < K; ++j) {
        float bk = inf;
        uint bi = 0xffffffffu;
        for (uint e = t; e < E; e += 128) {
            const float k = skey[e];
            if (k < bk || (k == bk && e < bi)) { bk = k; bi = e; }
        }
        const float mk = simd_min(bk);
        const uint mi = simd_min(bk == mk ? bi : 0xffffffffu);
        if (lane == 0) { rk[sg] = mk; ri[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float fk = rk[0];
            uint fi = ri[0];
            for (uint g = 1; g < 4; ++g) {
                if (rk[g] < fk || (rk[g] == fk && ri[g] < fi)) { fk = rk[g]; fi = ri[g]; }
            }
            chosen[j] = fi;
            skey[fi] = inf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        float w[K];
        float sum = 0.0f;
        for (uint j = 0; j < K; ++j) { w[j] = sval[chosen[j]]; sum += w[j]; }
        const float den = sum + par[1];
        for (uint j = 0; j < K; ++j) {
            ids[ulong(r) * K + j] = int(chosen[j]);
            const float wn = NORM ? (w[j] / den) : w[j];
            wts[ulong(r) * K + j] = wn * par[0];
        }
    }
"""
_k_router: dict = {}
_par: dict = {}


def router_select(logits: mx.array, bias: mx.array, top_k: int, norm: bool, scale: float, bias_vl: mx.array | None = None,
                  image_mask: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """logits fp32 [R, E] (router matmul), bias fp32 [E] -> (ids int32 [R, K], weights fp32 [R, K]); the stock sequence
    scores = sqrt(softplus(logits)); ids = argpartition(-(scores + bias), K - 1)[:K]; w = take(scores, ids); w = w / (w.sum() + 1e-20); w * scale.
    image_mask (bool, R values) with bias_vl selects the bias per row like mx.where(image_mask[..., None], bias_vl, bias)."""
    masked = image_mask is not None
    from . import alucut as AC
    r1 = AC.router1()                                                              # KYOJIN_ALU_ROUTER1: one barrier per selection round (exact)
    kern = _k_router.get((masked, r1))
    if kern is None:
        kern = _k_router[(masked, r1)] = mx.fast.metal_kernel(
            name=f"kyojin_router_select{int(masked)}" + ("_r1" if r1 else ""), input_names=["lg", "bias"] + (["bias_vl", "msk"] if masked else []) + ["par"],
            output_names=["ids", "wts"], header=_HEADER + _LOG1P,
            source=(AC.router_source(_ROUTER) if r1 else _ROUTER).replace("BIAS_EXPR", "(msk[r] ? bias_vl[e] : bias[e])" if masked else "bias[e]"))
    R, E = logits.shape
    p = _par.get(scale)
    if p is None:
        p = _par[scale] = mx.array([scale, 1e-20], dtype=mx.float32)
    ins = [logits, bias] + ([bias_vl, image_mask.reshape(R)] if masked else []) + [p]
    return kern(inputs=ins, template=[("E", E), ("K", top_k), ("NORM", bool(norm))], grid=(128 * R, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[(R, top_k), (R, top_k)], output_dtypes=[mx.int32, mx.float32])


# --------------------------------------------------------------------------------------------------------------- hyper-connections (mHC)
_EXPAND = r"""
    // y[b, j, d] = T(post[b, j] * x[b, d] + sum_a comb[b, a, j] * res[b, a, d]); the sum is sequential from 0 in fp32 (products rounded first),
    // the last multiply-add is an fma (the stock compiled kernel contracts it); y32 is the exact fp32 copy of the rounded y.
    const uint d = thread_position_in_grid.x;
    const uint b = thread_position_in_grid.y;
    if (d >= D) { return; }
    float res[4];
    for (uint a = 0; a < 4; ++a) { res[a] = float(r[(ulong(b) * 4 + a) * D + d]); }
    const float xv = float(x[ulong(b) * D + d]);
    for (uint j = 0; j < 4; ++j) {
        float mixed = 0.0f;
        for (uint a = 0; a < 4; ++a) { mixed += comb[(ulong(b) * 4 + a) * 4 + j] * res[a]; }
        const float v = FMA_LAST ? fma(post[b * 4 + j], xv, mixed) : (post[b * 4 + j] * xv + mixed);
        const T o = static_cast<T>(is_same<T, half>::value ? clamp(v, -65504.0f, 65504.0f) : v);   // saturate at the fp16 range: see fuse.sat
        y[(ulong(b) * 4 + j) * D + d] = o;
        y32[(ulong(b) * 4 + j) * D + d] = float(o);
    }
"""
_PRE = r"""
    // y[b, d] = T(sum_j pre[b, j] * x[b, j, d]) sequential from 0 in fp32.
    const uint d = thread_position_in_grid.x;
    const uint b = thread_position_in_grid.y;
    if (d >= D) { return; }
    float acc = 0.0f;
    for (uint j = 0; j < 4; ++j) { acc += pre[b * 4 + j] * float(x[(ulong(b) * 4 + j) * D + d]); }
    y[ulong(b) * D + d] = static_cast<T>(is_same<T, half>::value ? clamp(acc, -65504.0f, 65504.0f) : acc);
"""
_MIXPRE = r"""
    // Two independent jobs per row in one launch (grid x: threadgroups 0 .. NDT - 1, then one more; grid y: the row b):
    //  hc_pre   y[d] = T(sum_j pre[j] * x[j, d]) sequential from 0 in fp32 (products rounded first: contraction off for this scope, like _PRE)
    //  sinkhorn the mlx-vlm kernel hc_sinkhorn_split (pre / post / comb of the mix just computed), verbatim for one row, default contraction like that kernel
    const uint tg = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    if (tg < NDT) {
        const uint d = tg * 128 + thread_position_in_threadgroup.x;
        if (d >= D) { return; }
        float acc = 0.0f;
        {
#pragma METAL fp contract(off)
            for (uint j = 0; j < 4; ++j) { acc += pre[b * 4 + j] * float(x[(ulong(b) * 4 + j) * D + d]); }
        }
        y[ulong(b) * D + d] = static_cast<T>(is_same<T, half>::value ? clamp(acc, -65504.0f, 65504.0f) : acc);
        return;
    }
    if (simdgroup_index_in_threadgroup != 0) { return; }
    uint lane = thread_position_in_threadgroup.x;
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
"""
_k_mhc: dict = {}
FMA_LAST = False


def mhc_expand(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> tuple[mx.array, mx.array]:
    """x [B, 1, D] (sublayer output), residual [B, 1, 4, D], post [B, 1, 4] fp32, comb [B, 1, 4, 4] fp32 -> (y [B, 1, 4, D] in x.dtype, y32 = y as fp32).
    Equals language.hc_expand(x, residual, post, comb)."""
    B, S, D = x.shape
    assert S == 1 and residual.shape == (B, 1, 4, D) and post.dtype == mx.float32 and comb.dtype == mx.float32
    from . import alucut as AC
    if AC.noy32():                                                                 # KYOJIN_ALU_NOY32: the fp32 copy is not stored (y32 is None)
        k = _k_mhc.get("expand_n")
        if k is None:
            k = _k_mhc["expand_n"] = mx.fast.metal_kernel(name="kyojin_mhc_expand_n", input_names=["x", "r", "post", "comb"], output_names=["y"],
                                                          source=AC.expand_source_no_y32(_EXPAND), header=_HEADER)
        y = k(inputs=[x, residual, post, comb], template=[("T", x.dtype), ("D", D), ("FMA_LAST", FMA_LAST)], grid=(((D + 127) // 128) * 128, B, 1),
              threadgroup=(128, 1, 1), output_shapes=[(B, 1, 4, D)], output_dtypes=[x.dtype])[0]
        return y, None
    k = _k_mhc.get("expand")
    if k is None:
        k = _k_mhc["expand"] = mx.fast.metal_kernel(name="kyojin_mhc_expand", input_names=["x", "r", "post", "comb"], output_names=["y", "y32"],
                                                    source=_EXPAND, header=_HEADER)
    y, y32 = k(inputs=[x, residual, post, comb], template=[("T", x.dtype), ("D", D), ("FMA_LAST", FMA_LAST)], grid=(((D + 127) // 128) * 128, B, 1),
               threadgroup=(128, 1, 1), output_shapes=[(B, 1, 4, D), (B, 1, 4, D)], output_dtypes=[x.dtype, mx.float32])
    return y, y32


def mhc_pre(x: mx.array, pre: mx.array) -> mx.array:
    """x [B, 1, 4, D], pre [B, 1, 4] fp32 -> [B, 1, D] in x.dtype. Equals the stock hc_pre."""
    B, S, H, D = x.shape
    assert S == 1 and H == 4 and pre.dtype == mx.float32
    k = _k_mhc.get("pre")
    if k is None:
        k = _k_mhc["pre"] = mx.fast.metal_kernel(name="kyojin_mhc_pre", input_names=["x", "pre"], output_names=["y"], source=_PRE, header=_HEADER)
    return k(inputs=[x, pre], template=[("T", x.dtype), ("D", D)], grid=(((D + 127) // 128) * 128, B, 1), threadgroup=(128, 1, 1),
             output_shapes=[(B, 1, D)], output_dtypes=[x.dtype])[0]


def mhc_mix_pre(mixes: mx.array, scale: mx.array, base: mx.array, x: mx.array, pre: mx.array, iters: int, eps: float):
    """One launch for `hc_split_sinkhorn(mixes, scale, base, 4, iters, eps)` (-> pre', post, comb) and `hc_pre(x, pre)` (-> y): the two are independent.
    mixes [B, 1, 24] fp32, scale [3], base [24] fp32, x [B, 1, 4, D], pre [B, 1, 4] fp32 -> (y [B, 1, D] in x.dtype, pre' [B, 1, 4], post [B, 1, 4], comb [B, 1, 4, 4]).
    Bit-identical to the two separate launches (the sinkhorn text is the mlx-vlm kernel's, the pre sum is `mhc_pre`'s); a row does not depend on B (one grid row each)."""
    B, S, H, D = x.shape
    assert S == 1 and H == 4 and pre.dtype == mx.float32 and mixes.shape == (B, 1, 24) and pre.shape == (B, 1, 4)
    k = _k_mhc.get("mixpre")
    if k is None:
        k = _k_mhc["mixpre"] = mx.fast.metal_kernel(name="kyojin_mhc_mix_pre", input_names=["mixes", "scale", "base", "x", "pre"], output_names=["y", "pre_o", "post", "comb"],
                                                    source=_MIXPRE)
    ndt = (D + 127) // 128
    from . import gluefuse as _GF                                                    # stays a no-op unless KYOJIN_GLUE_FUSE=1 (read per call)
    y, pre_o, post, comb = k(inputs=[_GF.f32_arg(mixes), _GF.f32_arg(scale), _GF.f32_arg(base), x, pre],
                             template=[("T", x.dtype), ("D", D), ("NDT", ndt), ("ITERS", iters), ("EPS_INT", round(eps / 1e-9))],
                             grid=((ndt + 1) * 128, B, 1), threadgroup=(128, 1, 1),
                             output_shapes=[(B, 1, D), (B, 1, 4), (B, 1, 4), (B, 1, 4, 4)], output_dtypes=[x.dtype, mx.float32, mx.float32, mx.float32])
    return y, pre_o, post, comb


# ------------------------------------------------------------------------------------------- mHC identity form (2 hc mix rows, no comb)
# The identity form of the hyper-connection (foreign_skeleton_hc: fn has 2 hc rows, no sinkhorn, the residual passes as is). Its seams in two
# launches around the sub-layer (the mix itself is gluefuse.mixvec, any T and N = 2 hc):
#   mhc_pre_identity     split (pre / post sigmoids) + collapse (_PRE's sum) [+ the sub-layer's pre rms_norm]   (stock: ~13 launches + 1 norm)
#   mhc_expand_identity  [the sandwich post rms_norm of the sub-layer output] + expand h_j + post_j y          (stock: ~4 launches + 1 norm)
# Any stream dtype T (fp32 / fp16 / bf16), any hc (HC), any D % 4 == 0, one threadgroup of 1024 threads per row (row independent). The arithmetic is
# the stock op sequence of foreign_skeleton_hc + foreign_skeleton.Norm (fp contract off; each stock op rounds to its output dtype, the kernel
# rounds at the same points; mx.sigmoid's 1 / (1 + exp(-|z|)) form). The row reduction of the norms is the rms_looped / single-row layout of MLX
# (thread lid owns 4 lid .. 4 lid + 3 of every 4096, rounds in order, two-level simd_sum, precise rsqrt), the text of gluefuse._MIXVEC.
# No clamp at the fp16 range: the stock identity path has none either.
_HC_RMS = r"""
    acc = simd_sum(acc);
    if (sg == 0) { ls[ln] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ln == 0) { ls[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(ls[ln]);
        if (ln == 0) { inv_s[0] = metal::precise::rsqrt(acc / float(D) + NEPS_EXPR); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[0];
"""
_PRE_ID = r"""
    // mixes [R, 2 HC] fp32, scale [>= 2], base [2 HC], prm = [eps, magnitude, post_eps, norm_eps] fp32, x [R, HC, D] T, nw [D] T (NORM only)
    // -> y [R, D] T = T(sum_j pre_j x_j) or its rms_norm (NORM), post [R, HC] fp32
    const uint b = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, ln = thread_index_in_simdgroup;
    threadgroup float ls[32];
    threadgroup float inv_s[1];
    const float s0 = scale[0], s1 = scale[1];
    float pre[HC];
    for (uint j = 0; j < HC; ++j) {
        const float z = mixes[b * 2 * HC + j] * s0 + base[j];
        const float e = 1.0f / (1.0f + metal::precise::exp(-metal::abs(z)));
        pre[j] = ((z < 0.0f) ? 1.0f - e : e) + prm[0];
    }
    if (lid < HC) {
        const float z = mixes[b * 2 * HC + HC + lid] * s1 + base[HC + lid];
        const float e = 1.0f / (1.0f + metal::precise::exp(-metal::abs(z)));
        float p = prm[1] * ((z < 0.0f) ? 1.0f - e : e);
        if (POST_EPS) { p = p + prm[0]; }
        post[b * HC + lid] = p;
    }
    const device T* xr = x + ulong(b) * HC * D;
    device T* yr = y + ulong(b) * D;
    float acc = 0.0f;
    for (uint r = 0; r < ROUNDS; ++r) {
        const uint off = r * 4096 + lid * 4;
        if (off + 4 <= D) {
            for (uint i = 0; i < 4; ++i) {
                float c = 0.0f;
                for (uint j = 0; j < HC; ++j) { c += pre[j] * float(xr[j * D + off + i]); }
                const float ct = float(static_cast<T>(c));
                if (NORM) { acc += ct * ct; } else { yr[off + i] = static_cast<T>(c); }
            }
        }
    }
    if (NORM) {
""" + _HC_RMS.replace("NEPS_EXPR", "prm[3]") + r"""
        for (uint r = 0; r < ROUNDS; ++r) {
            const uint off = r * 4096 + lid * 4;
            if (off + 4 <= D) {
                for (uint i = 0; i < 4; ++i) {
                    float c = 0.0f;
                    for (uint j = 0; j < HC; ++j) { c += pre[j] * float(xr[j * D + off + i]); }
                    const float ct = float(static_cast<T>(c));
                    yr[off + i] = static_cast<T>(float(nw[off + i]) * float(static_cast<T>(ct * inv)));
                }
            }
        }
    }
"""
_EXPAND_ID = r"""
    // yv [R, D] Y (sub-layer output), h [R, HC, D] T, post [R, HC] fp32, nw [D] Y (NORM only), prm = [norm_eps] fp32
    // -> out [R, HC, D] T = T(h_j + post_j * n), n = yv, or its rms_norm w * Y(v * inv) (NORM == 1), or Y(Y(v * w) * inv) (NORM == 2: rms_norm(w * v))
    const uint b = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, ln = thread_index_in_simdgroup;
    threadgroup float ls[32];
    threadgroup float inv_s[1];
    const device Y* yr = yv + ulong(b) * D;
    float inv = 1.0f;
    if (NORM) {
        float acc = 0.0f;
        for (uint r = 0; r < ROUNDS; ++r) {
            const uint off = r * 4096 + lid * 4;
            if (off + 4 <= D) {
                for (uint i = 0; i < 4; ++i) {
                    float v = float(yr[off + i]);
                    if (NORM == 2) { v = float(static_cast<Y>(v * float(nw[off + i]))); }
                    acc += v * v;
                }
            }
        }
""" + _HC_RMS.replace("const float inv = inv_s[0];", "inv = inv_s[0];").replace("NEPS_EXPR", "prm[0]") + r"""
    }
    float pj[HC];
    for (uint j = 0; j < HC; ++j) { pj[j] = post[b * HC + j]; }
    for (uint r = 0; r < ROUNDS; ++r) {
        const uint off = r * 4096 + lid * 4;
        if (off + 4 <= D) {
            for (uint i = 0; i < 4; ++i) {
                float v = float(yr[off + i]);
                if (NORM == 1) { v = float(static_cast<Y>(float(nw[off + i]) * float(static_cast<Y>(v * inv)))); }
                if (NORM == 2) { v = float(static_cast<Y>(float(static_cast<Y>(v * float(nw[off + i]))) * inv)); }
                for (uint j = 0; j < HC; ++j) {
                    const ulong k = (ulong(b) * HC + j) * D + off + i;
                    out[k] = static_cast<T>(float(h[k]) + pj[j] * v);
                }
            }
        }
    }
"""
_k_hcid: dict = {}
_prm_cache: dict = {}


def _prm(*vals) -> mx.array:
    a = _prm_cache.get(vals)
    if a is None:
        a = _prm_cache[vals] = mx.array(list(vals), dtype=mx.float32)
    return a


def mhc_pre_identity(x: mx.array, mixes: mx.array, scale: mx.array, base: mx.array, eps: float, magnitude: float, post_eps: bool,
                     nw: mx.array | None = None, norm_eps: float = 0.0) -> tuple[mx.array, mx.array]:
    """x [R, hc, D] stream (T), mixes [R, 2 hc] / scale / base fp32 -> (y [R, D] T, post [R, hc] fp32). y = collapse(x, pre) as T, or
    rms_norm(that, nw, norm_eps) when nw (dtype T, shape [D]) is given. Equals foreign_skeleton_hc.split (identity form) + collapse [+ Norm]."""
    R, H, D = x.shape
    norm = nw is not None
    k = _k_hcid.get("pre")
    if k is None:
        k = _k_hcid["pre"] = mx.fast.metal_kernel(name="kyojin_mhc_pre_identity", input_names=["mixes", "scale", "base", "prm", "x", "nw"],
                                                  output_names=["y", "post"], source=_PRE_ID, header=_HEADER)
    y, post = k(inputs=[mixes, scale, base, _prm(eps, magnitude, eps, norm_eps), x, nw if norm else x],
                template=[("T", x.dtype), ("HC", H), ("D", D), ("ROUNDS", -(-D // 4096)), ("NORM", norm), ("POST_EPS", bool(post_eps))],
                grid=(1024, R, 1), threadgroup=(1024, 1, 1), output_shapes=[(R, D), (R, H)], output_dtypes=[x.dtype, mx.float32])
    return y, post


def mhc_expand_identity(y: mx.array, h: mx.array, post: mx.array, nw: mx.array | None = None, norm_eps: float = 0.0,
                        mul_first: bool = False) -> mx.array:
    """y [R, D] sub-layer output (Y), h [R, hc, D] stream (T), post [R, hc] fp32 -> [R, hc, D] T = h_j + post_j * n, n = y, or Norm(y): rms_norm(y, nw)
    (nw dtype Y), or rms_norm(y * nw, ones) when mul_first. Equals foreign_skeleton_hc.expand(Norm(y), h, post, None)."""
    R, H, D = h.shape
    mode = 0 if nw is None else (2 if mul_first else 1)
    k = _k_hcid.get("expand")
    if k is None:
        k = _k_hcid["expand"] = mx.fast.metal_kernel(name="kyojin_mhc_expand_identity", input_names=["yv", "h", "post", "nw", "prm"],
                                                     output_names=["out"], source=_EXPAND_ID, header=_HEADER)
    return k(inputs=[y, h, post, nw if mode else y, _prm(norm_eps)],
             template=[("T", h.dtype), ("Y", y.dtype), ("HC", H), ("D", D), ("ROUNDS", -(-D // 4096)), ("NORM", mode)],
             grid=(1024, R, 1), threadgroup=(1024, 1, 1), output_shapes=[(R, H, D)], output_dtypes=[h.dtype])[0]


# --------------------------------------------------------------------------------------------------------------- wiring
MIN_PROMPT_ROWS = 64
_last = (None, None)          # (array, its fp32 copy) written by the last mhc_expand: the next mHC mix reads the copy instead of casting


def take_f32(x: mx.array) -> mx.array | None:
    """fp32 copy of `x` when `x` is the array the last mhc_expand returned, else None."""
    from . import alucut as AC
    return _last[1] if _last[0] is x and not AC.noy32() else None


def region_f32(h: mx.array) -> mx.array | None:
    """What a compiled region returns as the fp32 copy of its output: take_f32, or a scalar placeholder under KYOJIN_ALU_NOY32 (a compiled region cannot
    return None; take_f32 never hands the placeholder out, the next block runs without a copy)."""
    t = take_f32(h)
    if t is None:
        from . import alucut as AC
        if AC.noy32():
            return mx.zeros((), mx.float32)
    return t


def sat(y: mx.array) -> mx.array:
    """Saturate a stream array at the fp16 range. The model is bf16 (range 1e38), the engine runs it in fp16 (limit 65504): the last-layer streams come within
    x4 of the limit on ordinary text and the BOS input overflows from layer 22, and one inf is NaN at the next norm (the row then samples token 0 for ever).
    The expand and collapse kernels do the same inline (the collapse weights are sigmoids, their sum goes above 1); bit-identical below the limit."""
    return mx.clip(y, -65504, 65504) if y.dtype == mx.float16 else y


def _impl_mhc_expand(orig):
    from . import pf_glue as PG
    def expand(x, residual, post, comb):
        global _last
        if AZ.ACTIVE and AZ.on("mhc") and x.ndim == 3 and x.shape[1] == 1 and residual.ndim == 4:      # ablate: plain add (the hc mixes are constants)
            _last = (None, None)
            return AZ.mhc_expand(x, residual)
        if ENABLED and MHC and x.ndim == 3 and x.shape[1] == 1 and residual.ndim == 4 and residual.shape[2] == 4 and post.dtype == mx.float32 \
                and comb.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            y, y32 = mhc_expand(x, residual, post, comb)
            _last = (y, y32) if y32 is not None else (None, None)
            return y
        if ENABLED and MHC and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] > 1 and residual.shape == (1, x.shape[1], 4, x.shape[2]) and post.dtype == mx.float32 \
                and comb.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            S, D = x.shape[1], x.shape[2]                                      # prompt rows: one grid row per token, the arithmetic of the 1-row case
            if PG.ENABLED and D % 8 == 0 and S >= MIN_PROMPT_ROWS:
                _last = (None, None)
                return PG.expand(x, residual, post, comb)
            y, _ = mhc_expand(x.reshape(S, 1, D), residual.reshape(S, 1, 4, D), post.reshape(S, 1, 4), comb.reshape(S, 1, 4, 4))
            _last = (None, None)
            return y.reshape(1, S, 4, D)
        return sat(orig(x, residual, post, comb))
    expand.saturates = True                                                    # every branch returns a stream already clamped at the fp16 range
    return expand


def _impl_mhc_pre(orig):
    def pre(x, pre_mix):
        if AZ.ACTIVE and AZ.on("mhc") and x.ndim == 4 and x.shape[1] == 1:                           # ablate: the stream mean
            return AZ.mhc_pre(x)
        if ENABLED and MHC and x.ndim == 4 and x.shape[1] == 1 and x.shape[2] == 4 and pre_mix.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            return mhc_pre(x, pre_mix)
        if ENABLED and MHC and x.ndim == 4 and x.shape[0] == 1 and x.shape[1] > 1 and x.shape[2] == 4 and pre_mix.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            S, D = x.shape[1], x.shape[3]
            return mhc_pre(x.reshape(S, 1, 4, D), pre_mix.reshape(S, 1, 4)).reshape(1, S, D)
        return sat(orig(x, pre_mix))
    pre.saturates = True                                                       # every branch returns a collapse already clamped at the fp16 range
    return pre


IMPLS = {"mhc_expand": _impl_mhc_expand, "mhc_pre": _impl_mhc_pre}
from . import fuse_attn as _fa                                                 # noqa: E402  (attention decode cut, see fuse_attn)
IMPLS["attn_decode"] = _fa._impl_attention
IMPLS["indexer"] = lambda orig: __import__(f"{__package__}.fuse_index", fromlist=["_impl_indexer"])._impl_indexer(orig)
IMPLS["compress_part"] = _fa._impl_compress_part
IMPLS["rope_positions"] = _fa._impl_rope_positions
IMPLS["sparse_prefill"] = lambda orig: __import__(f"{__package__}.attn_prefill", fromlist=["_impl_sparse"])._impl_sparse(orig)
IMPLS["mix_prompt"] = lambda orig: __import__(f"{__package__}.mhc_prefill", fromlist=["_impl_mix"])._impl_mix(orig)
IMPLS["moe_prompt"] = lambda orig: __import__(f"{__package__}.pf_glue", fromlist=["_impl_moe"])._impl_moe(orig)
IMPLS["shared_mlp"] = lambda orig: __import__(f"{__package__}.pf_glue", fromlist=["_impl_shared_mlp"])._impl_shared_mlp(orig)
IMPLS["block_prompt"] = lambda orig: __import__(f"{__package__}.pf_glue", fromlist=["_impl_block"])._impl_block(orig)
IMPLS["local_prefill"] = lambda orig: __import__(f"{__package__}.attn_prefill", fromlist=["_impl_attention_local"])._impl_attention_local(orig)
IMPLS["block_decode"] = lambda orig: __import__(f"{__package__}.fuse_layer", fromlist=["_impl_block"])._impl_block(orig)     # compiled decode regions of the block (fuse_layer imports this module)
PREFILL_IMPLS = {"sparse_prefill", "local_prefill", "mix_prompt", "mhc_expand", "mhc_pre", "moe_prompt", "shared_mlp", "block_prompt"}              # replacements that serve prompts: installed by patch_model itself
_installed: set = set()


def install(spec, only=None) -> int:
    """Replace the functions listed in `spec.fuse` (process wide, once). Returns the number of replacements."""
    import importlib
    if spec.fuse is None or not ENABLED:
        return 0
    if os.environ.get("KYOJIN_GLM_DSA_FAST", "0") == "1" and spec.index_patch == "glm":
        from .glm_dsa import install as _dsa_install
        _dsa_install()
    if os.environ.get("KYOJIN_DSA_INDEX_GLM", "0") == "1" and spec.index_patch == "glm":
        from .glm_index import install as _glm_install
        _glm_install()
    n = 0
    for path, attr, impl in spec.fuse.funcs:
        if only is not None and impl not in only:
            continue
        key = (path, attr, impl)
        if key in _installed:
            continue
        modpath, _, cls = path.partition(":")
        owner = importlib.import_module(modpath)
        if cls:
            owner = getattr(owner, cls)
        orig = owner.__dict__[attr] if cls else getattr(owner, attr)
        static = isinstance(orig, staticmethod)
        fn = IMPLS[impl](orig.__func__ if static else orig)
        setattr(owner, attr, staticmethod(fn) if static else fn)
        _installed.add(key)
        n += 1
    return n
