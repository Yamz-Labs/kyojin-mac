# SPDX-License-Identifier: Apache-2.0
# Portions adapted from sushi (MIT), src/dsv41/dsv41_kernels.zig@0f4f00b (role layout of the fused hyper-connection stage: one threadgroup per
# role, docs/dsv41-arch.md:122-127).
"""Fused hyper-connection stages of a decode block (KYOJIN_PORT_SUGLUE=1, default off).

A DS4.1 decode layer runs ten glue launches around its two sublayers (per sublayer: the RMSNorm of the flattened streams, the 24 x 4*hidden mix
dot, the sinkhorn + collapse, the sublayer norm, and the expand that closes the sublayer). Sushi fuses them into one stage whose threadgroups
take different roles. This module does the same in three launches per layer, keeping OUR arithmetic (the bits of gluefuse.mixvec,
glue_fuse.mix_norm and fuse.mhc_expand) inside Sushi's grid layout:

  stage A  (before the attention)   grid (25 x row): threadgroup 0 = collapse of h with the incoming pre_mix + RMSNorm (attn_norm); threadgroups
                                    1..24 = one mix dot each (every one recomputes the rms scale of the 4*D stream). -> x, z_attn
  stage B  (between the sublayers)  every threadgroup splits z_attn with the sinkhorn (one simdgroup), then re-derives the expanded stream
                                    x2 = expand(attn_out, h) element by element. Threadgroup 0 writes x2 and the FFN input (collapse with the
                                    new pre' + ffn_norm); threadgroups 1..24 do the FFN mix dots on x2. -> ffn_in, x2, z_ffn
  stage E  (after the FFN)          splits z_ffn (sinkhorn), expands (ffn_out, x2) -> h', fp32 copy of h', pre' for the next block.

This replaces 10 launches by 3 per layer. A row's bits do not depend on the number of rows (one grid row each); the sinkhorn text is the mlx-vlm
kernel's verbatim (default fp contraction, as in fuse._MIXPRE), everything else runs with contraction off like fuse._EXPAND / gluefuse._MIXVEC.
Solo decode only (the chained verify keeps the separate launches).
"""
from __future__ import annotations

import os

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import ablit as AB
from .rowinv import pick_ks

ENABLED = os.environ.get("KYOJIN_PORT_SUGLUE", "0") == "1"   # off: +6 ms per plain step on the M5 Ultra (bare7 30/10)
STATS = {"A": 0, "B": 0, "E": 0}

_HEADER = r"""
#pragma METAL fp contract(off)
template <typename T>
inline float sg_x2(const device T* ao, const device T* h, const threadgroup float* post, const threadgroup float* comb, uint j, uint d, uint D) {
    float mixed = 0.0f;
    for (uint a = 0; a < 4; ++a) { mixed += comb[a * 4 + j] * float(h[a * D + d]); }
    const float v = post[j] * float(ao[d]) + mixed;
    return float(static_cast<T>(is_same<T, half>::value ? clamp(v, -65504.0f, 65504.0f) : v));   // saturate at the fp16 range: see fuse.sat
}
"""

# mlx-vlm hc_sinkhorn_split, verbatim for one row (one simdgroup, 32 lanes). Reads MIXROW (device float*), writes the threadgroup arrays
# tpre / tpost / tcomb. SCALE / BASE name the device arrays.
_SINK = r"""
    {
#pragma METAL fp contract(on)
        const uint lane = sl;
        constexpr int HC = 4;
        constexpr int BOFF = 2 * HC;
        constexpr float EPS = EPS_INT * 1e-9;
        const device float* mix = MIXROW;
        const float pre_scale  = SCALE[0];
        const float post_scale = SCALE[1];
        const float comb_scale = SCALE[2];
        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint  llane  = metal::min(lane, (uint)(HC - 1));
        float pre_z  = mix[llane]      * pre_scale  + BASE[llane];
        float post_z = mix[HC + llane] * post_scale + BASE[HC + llane];
        float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
        if (lane < (uint)HC) {
            tpre[lane]  = pre_v;
            tpost[lane] = post_v;
        }
        float4 v = (*(const device float4*)(mix  + BOFF + llane * HC)
                        * comb_scale
                  + *(const device float4*)(BASE + BOFF + llane * HC))
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
            tcomb[lane * HC + 0] = r.x; tcomb[lane * HC + 1] = r.y; tcomb[lane * HC + 2] = r.z; tcomb[lane * HC + 3] = r.w;
        }
    }
"""

# One mix dot (role o of a threadgroup): the rms scale of the K stream (mlx rms_looped layout), then rowdot's k order (gluefuse._MIXVEC).
# SG_ELEM(e) is the stream element e as float.
_DOT = r"""
        {
            float acc = 0.0f;
            for (uint r = 0; r < ROUNDS; ++r) {
                const uint off = r * 4096 + lid * 4;
                if (off + 4 <= K) {
                    for (uint i = 0; i < 4; ++i) {
                        const float v = SG_ELEM(off + i);
                        acc += v * v;
                    }
                }
            }
            acc = simd_sum(acc);
            if (sg == 0) { ls[sl] = 0.0f; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sl == 0) { ls[sg] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                acc = simd_sum(ls[sl]);
                if (sl == 0) { inv_s[0] = metal::precise::rsqrt(acc / float(K) + eps_mix[0]); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = inv_s[0];
            if (sg < KS) {
                const device float* wp = fn + ulong(t - 1) * K;
                float d = 0.0f;
                for (uint kk = (sg * 32 + sl) * 8; kk < K; kk += 256 * KS) {
                    float wv[8];
                    for (uint j = 0; j < 8; ++j) { wv[j] = (kk + j < K) ? wp[kk + j] : 0.0f; }
                    for (uint j = 0; j < 8; ++j) {
                        const float tv = (kk + j < K) ? SG_ELEM(kk + j) * inv : 0.0f;
                        d = fma(wv[j], tv, d);
                    }
                }
                const float s1 = simd_sum(d);
                if (sl == 0) { part[sg] = s1; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0 && sl == 0) {
                float s = part[0];
                for (uint i = 1; i < KS; ++i) { s += part[i]; }
                zout[ulong(b) * 24 + (t - 1)] = s;
            }
        }
"""

# collapse + RMSNorm of threadgroup 0 (glue_fuse._MIXNORM): COLLAPSE_ELEM computes `a` for element index d (and may write side outputs).
_NORM = r"""
        {
            T keep[NG][4];
            float acc = 0.0f;
            for (uint g = 0; g < NG; ++g) {
                const uint i0 = lid * 4 + g * 1024 * 4;
                for (uint i = 0; i < 4; i++) {
                    if (i0 + i < D) {
                        const uint d = i0 + i;
                        float a = 0.0f;
                        COLLAPSE
                        const T hv = static_cast<T>(is_same<T, half>::value ? clamp(a, -65504.0f, 65504.0f) : a);
                        keep[g][i] = hv;
                        const float f = float(hv);
                        acc += f * f;
                    }
                }
            }
            acc = simd_sum(acc);
            if (sl == 0) ls[sg] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                const float tt = simd_sum(ls[sl]);
                if (sl == 0) inv_s[0] = metal::precise::rsqrt(tt / (float)D + eps_n[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = inv_s[0];
            for (uint g = 0; g < NG; ++g) {
                const uint i0 = lid * 4 + g * 1024 * 4;
                for (uint i = 0; i < 4; i++) {
                    if (i0 + i < D) nout[ulong(b) * D + i0 + i] = wn[i0 + i] * static_cast<T>(float(keep[g][i]) * inv);
                }
            }
        }
"""

_PRELUDE = r"""
    const uint t = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, sl = thread_index_in_simdgroup;
    threadgroup float ls[32];
    threadgroup float part[KS];
    threadgroup float inv_s[1];
    threadgroup float tpre[4], tpost[4], tcomb[16];
"""

_SRC_A = _PRELUDE + r"""
    const device T* hr = h + ulong(b) * K;
    if (t == 0) {
        NORM
    } else {
#define SG_ELEM(e) float(hr[(e)])
        DOT
#undef SG_ELEM
    }
"""

_SRC_B = _PRELUDE + r"""
    const device T* hr = h + ulong(b) * K;
    const device T* aor = ao + ulong(b) * D;
    if (sg == 0) {
        SINK
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        NORM
    } else {
#define SG_ELEM(e) sg_x2<T>(aor, hr, tpost, tcomb, (e) / D, (e) % D, D)
        DOT
#undef SG_ELEM
    }
"""

_COLLAPSE_A = r"""
                        for (uint j = 0; j < 4; ++j) a += pre[b * 4 + j] * float(hr[j * D + d]);
"""
_COLLAPSE_B = r"""
                        for (uint j = 0; j < 4; ++j) {
                            const float xv = sg_x2<T>(aor, hr, tpost, tcomb, j, d, D);
                            x2o[(ulong(b) * 4 + j) * D + d] = static_cast<T>(xv);
                            a += tpre[j] * xv;
                        }
"""

_SRC_E = r"""
    const uint d = thread_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint sl = thread_index_in_simdgroup;
    threadgroup float tpre[4], tpost[4], tcomb[16];
    if (simdgroup_index_in_threadgroup == 0) {
        SINK
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadgroup_position_in_grid.x == 0 && thread_position_in_threadgroup.x < 4) { pre_o[b * 4 + thread_position_in_threadgroup.x] = tpre[thread_position_in_threadgroup.x]; }
    if (d >= D) { return; }
    float res[4];
    for (uint a = 0; a < 4; ++a) { res[a] = float(x2[(ulong(b) * 4 + a) * D + d]); }
    const float xv = float(y[ulong(b) * D + d]);
    for (uint j = 0; j < 4; ++j) {
        float mixed = 0.0f;
        for (uint a = 0; a < 4; ++a) { mixed += tcomb[a * 4 + j] * res[a]; }
        const float v = tpost[j] * xv + mixed;
        const T o = static_cast<T>(is_same<T, half>::value ? clamp(v, -65504.0f, 65504.0f) : v);   // saturate at the fp16 range: see fuse.sat
        h2[(ulong(b) * 4 + j) * D + d] = o;
        h232[(ulong(b) * 4 + j) * D + d] = float(o);
    }
"""

_k: dict = {}
_const: dict = {}


def _f32(v: float) -> mx.array:
    a = _const.get(v)
    if a is None:
        a = _const[v] = mx.array([v], dtype=mx.float32)
    return a


def _build_a() -> str:
    s = _SRC_A.replace("NORM", _NORM.replace("COLLAPSE", _COLLAPSE_A)).replace("DOT", _DOT)
    return s


def _build_b() -> str:
    sink = _SINK.replace("MIXROW", "(const device float*)za + b * 24").replace("SCALE", "sca").replace("BASE", "bsa")
    s = _SRC_B.replace("SINK", sink).replace("NORM", _NORM.replace("COLLAPSE", _COLLAPSE_B)).replace("DOT", _DOT)
    return s


def _build_e() -> str:
    sink = _SINK.replace("MIXROW", "(const device float*)zf + b * 24").replace("SCALE", "scf").replace("BASE", "bsf")
    return _SRC_E.replace("SINK", sink)


def _kern(name, inputs, outputs, src):
    k = _k.get(name)
    if k is None:
        k = _k[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=src, header=_HEADER)
    return k


def stage_ok(blk, h) -> bool:
    """The stages apply to the rows (one token each) of a DS4.1 block with 4 streams, fp32 mix matrices and plain RMSNorms."""
    import mlx.nn as nn
    if not ENABLED or h.ndim != 4 or h.shape[1] != 1 or h.shape[2] != KG.HC or h.dtype not in (mx.float16, mx.bfloat16):
        return False
    D = h.shape[-1]
    if getattr(blk, "hc_mult", 0) != KG.HC or D % KG.HIDDEN_MULT or D < KG.STAGE_MIN_HIDDEN or 32 * pick_ks(4 * D, 24) > 1024:
        return False
    for fn in (blk.hc_attn_fn, blk.hc_ffn_fn):
        if fn.dtype != mx.float32 or fn.shape != (24, 4 * D):
            return False
    for nm in ("attn_norm", "ffn_norm"):
        nrm = getattr(blk, nm, None)
        if type(nrm) is not nn.RMSNorm or nrm.weight.dtype != h.dtype or nrm.weight.shape != (D,):
            return False
    return True


def _tmpl(h, D, norm_ng=True):
    K = 4 * D
    return [("T", h.dtype), ("D", D), ("K", K), ("KS", pick_ks(K, 24)), ("ROUNDS", -(-K // 4096)), ("NG", -(-D // 4096))]


def stage_a(blk, h, pre_mix):
    """h [B, 1, 4, D], pre_mix [B, 1, 4] fp32 -> (normed attention input [B, 1, D], z_attn [B, 1, 24] fp32)."""
    B, _, _, D = h.shape
    STATS["A"] += 1
    k = _kern("kyojin_suglue_a", ["h", "pre", "wn", "fn", "eps_mix", "eps_n"], ["nout", "zout"], _build_a())
    x, z = k(inputs=[h, pre_mix, blk.attn_norm.weight, blk.hc_attn_fn, _f32(blk.norm_eps), _f32(blk.attn_norm.eps)],
             template=_tmpl(h, D), grid=(25 * 1024, B, 1), threadgroup=(1024, 1, 1),
             output_shapes=[(B, 1, D), (B, 1, 24)], output_dtypes=[h.dtype, mx.float32])
    return x, z


def stage_b(blk, ao, h, z_attn):
    """ao [B, 1, D] attention output, h [B, 1, 4, D] residual, z_attn [B, 1, 24] -> (ffn input [B, 1, D], x2 [B, 1, 4, D], z_ffn [B, 1, 24])."""
    B, _, _, D = h.shape
    STATS["B"] += 1
    k = _kern("kyojin_suglue_b", ["ao", "h", "za", "sca", "bsa", "wn", "fn", "eps_mix", "eps_n"], ["nout", "x2o", "zout"], _build_b())
    yn, x2, zf = k(inputs=[ao, h, z_attn, blk.hc_attn_scale.astype(mx.float32), blk.hc_attn_base.astype(mx.float32), blk.ffn_norm.weight, blk.hc_ffn_fn,
                           _f32(blk.norm_eps), _f32(blk.ffn_norm.eps)],
                   template=_tmpl(h, D) + [("ITERS", blk.hc_sinkhorn_iters), ("EPS_INT", round(blk.hc_eps / 1e-9))],
                   grid=(25 * 1024, B, 1), threadgroup=(1024, 1, 1),
                   output_shapes=[(B, 1, D), (B, 1, 4, D), (B, 1, 24)], output_dtypes=[h.dtype, h.dtype, mx.float32])
    return yn, x2, zf


def stage_e(blk, y, x2, z_ffn):
    """y [B, 1, D] FFN output, x2 [B, 1, 4, D], z_ffn [B, 1, 24] -> (h' [B, 1, 4, D], h' as fp32, pre' [B, 1, 4])."""
    B, _, _, D = x2.shape
    STATS["E"] += 1
    k = _kern("kyojin_suglue_e", ["y", "x2", "zf", "scf", "bsf"], ["h2", "h232", "pre_o"], _build_e())
    return k(inputs=[y, x2, z_ffn, blk.hc_ffn_scale.astype(mx.float32), blk.hc_ffn_base.astype(mx.float32)],
             template=[("T", x2.dtype), ("D", D), ("ITERS", blk.hc_sinkhorn_iters), ("EPS_INT", round(blk.hc_eps / 1e-9))],
             grid=(((D + 127) // 128) * 128, B, 1), threadgroup=(128, 1, 1),
             output_shapes=[(B, 1, 4, D), (B, 1, 4, D), (B, 1, 4)], output_dtypes=[x2.dtype, mx.float32, mx.float32])


_XV_Y = "const float xv = float(y[ulong(b) * D + d]);"
_XV_COMB = "const float xv = float(static_cast<T>(y[ulong(b) * D + d] + float(sh[ulong(b) * D + d])));"      # launchcut comb: the stock (routed + shared) rounded to T, then read back


def stage_e_comb(blk, routed, shared, x2, z_ffn):
    """stage_e whose FFN output is the MoE combine, folded into the prologue: routed [B, 1, D] fp32 (the summed routed experts), shared [B, 1, D] (the shared
    expert, x dtype) -> the same (h', h' as fp32, pre') as stage_e(blk, (routed + shared.astype(fp32)).astype(x dtype), x2, z_ffn). The sum is one fp32 add
    rounded to T exactly like the compiled elementwise launch it replaces, then read back as float, so the bits of stage E do not change."""
    B, _, _, D = x2.shape
    STATS["E"] += 1
    src = _build_e()
    assert _XV_Y in src
    k = _kern("kyojin_suglue_ec", ["y", "sh", "x2", "zf", "scf", "bsf"], ["h2", "h232", "pre_o"], src.replace(_XV_Y, _XV_COMB))
    return k(inputs=[routed, shared, x2, z_ffn, blk.hc_ffn_scale.astype(mx.float32), blk.hc_ffn_base.astype(mx.float32)],
             template=[("T", x2.dtype), ("D", D), ("ITERS", blk.hc_sinkhorn_iters), ("EPS_INT", round(blk.hc_eps / 1e-9))],
             grid=(((D + 127) // 128) * 128, B, 1), threadgroup=(128, 1, 1),
             output_shapes=[(B, 1, 4, D), (B, 1, 4, D), (B, 1, 4)], output_dtypes=[x2.dtype, mx.float32, mx.float32])


def block(blk, h, pre_mix, image_mask, cache):
    """DeepseekV41Block.__call__ for single-token rows with the three stages around the stock sublayers (the layers the compiled regions do not take:
    local-attention layers, N-user rows)."""
    from . import fuse as F
    x, z = stage_a(blk, h, pre_mix)
    x = AB.attn_out(blk, blk.attn(x, cache))
    yn, x2, zf = stage_b(blk, x, h, z)
    y = AB.ffn_out(blk, blk.ffn(yn, image_mask))
    h2, h232, pre = stage_e(blk, y, x2, zf)
    F._last = (h2, h232)
    return h2, pre


# ---------------------------------------------------------------------------------------------------------- q_norm + kv_norm in one launch
# The two RMSNorms that follow the grouped q / kv projections are independent: one launch, one threadgroup per norm. Each role keeps the layout of
# mx.fast.rms_norm's single-row kernel for its own axis (4 contiguous values per thread, threads = 32 * ceil(axis / 128), fp32 partials, simd_sum, then
# the simdgroup partials); the threads of the wider role that the narrower one does not need add exact zeros.
_QKN = r"""
    const uint t = threadgroup_position_in_grid.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, sl = thread_index_in_simdgroup;
    threadgroup float ls[32];
    threadgroup float inv_s[1];
    const uint N = t == 0 ? NQ : NK;
    const device T* xr = (t == 0 ? q : kv) + ulong(r) * N;
    const device T* wr = t == 0 ? wq : wk;
    device T* orow = (t == 0 ? qo : kvo) + ulong(r) * N;
    T v[4];
    float acc = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const uint e = lid * 4 + i;
        v[i] = e < N ? xr[e] : static_cast<T>(0.0f);
        const float f = float(v[i]);
        acc += f * f;
    }
    acc = simd_sum(acc);
    if (sg == 0) { ls[sl] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sl == 0) { ls[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float tt = simd_sum(ls[sl]);
        if (sl == 0) { inv_s[0] = metal::precise::rsqrt(tt / (float)N + (t == 0 ? eq[0] : ek[0])); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[0];
    for (uint i = 0; i < 4; ++i) {
        const uint e = lid * 4 + i;
        if (e < N) { orow[e] = wr[e] * static_cast<T>(float(v[i]) * inv); }
    }
"""


def qk_norm_ok(attn, q, kv) -> bool:
    import mlx.nn as nn
    nq, nk = q.shape[-1], kv.shape[-1]
    return bool(ENABLED and type(attn.q_norm) is nn.RMSNorm and type(attn.kv_norm) is nn.RMSNorm and q.dtype in (mx.float16, mx.bfloat16) and kv.dtype == q.dtype
                and attn.q_norm.weight.dtype == q.dtype and attn.kv_norm.weight.dtype == q.dtype and nq % 4 == 0 and nk % 4 == 0
                and 0 < nq <= 4096 and 0 < nk <= 4096 and q.size // nq == kv.size // nk and attn.q_norm.weight.shape == (nq,)
                and attn.kv_norm.weight.shape == (nk,))


def qk_norm(attn, q, kv):
    """(q_norm(q), kv_norm(kv)) in one launch; q [..., NQ], kv [..., NK] with the same leading rows."""
    nq, nk = q.shape[-1], kv.shape[-1]
    rows = q.size // nq
    nt = max(32 * -(-nq // 128), 32 * -(-nk // 128))
    k = _k.get("qkn")
    if k is None:
        k = _k["qkn"] = mx.fast.metal_kernel(name="kyojin_suglue_qkn", input_names=["q", "kv", "wq", "wk", "eq", "ek"], output_names=["qo", "kvo"], source=_QKN)
    STATS["qkn"] = STATS.get("qkn", 0) + 1
    qo, kvo = k(inputs=[q, kv, attn.q_norm.weight, attn.kv_norm.weight, _f32(attn.q_norm.eps), _f32(attn.kv_norm.eps)],
                template=[("T", q.dtype), ("NQ", nq), ("NK", nk)], grid=(2 * nt, rows, 1), threadgroup=(nt, 1, 1),
                output_shapes=[q.shape, kv.shape], output_dtypes=[q.dtype, q.dtype])
    return qo, kvo
