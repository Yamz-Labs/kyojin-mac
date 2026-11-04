# SPDX-License-Identifier: Apache-2.0
"""Decode glue fusion: the small ops around the linears of a decoder layer, one launch each instead of a chain.

A 1-row decode step is a chain of dependent launches (the layer stack of one token is ~1500 dispatches; the glue around the linears -- norms,
the hyper-connection mixes, the residual adds, the fp8 round trips, the head/layout copies -- is ~200 of them and ~1.8 ms of a 20 ms step on the
M5 Ultra). Every launch costs a gap of ~2-4 us before the kernel even starts, which is what this module removes; the arithmetic is kept exactly
the arithmetic of the ops it replaces (same lane layout, same summation order, same roundings), so a step stays bit-identical.

Switches: KYOJIN_GLUE_FUSE=1 turns the module on (default OFF, the lead's real-pack gate decides the default); KYOJIN_GLUE_FUSE_MIXVEC=0 /
_MIXARGS=0 switch one cut off, for an A/B inside one process.

  1. mixvec   -- the mHC mix vector of DS4.1 (and any "RMSNorm then small linear" seam): `rowdot(rms_norm(x.flatten(-2), None, eps), fn)` is
     one launch. The norm of the flattened streams and the 24 x 20480 product along the same row; each threadgroup of 1024 threads recomputes
     the row scale (the rms_looped layout of MLX: thread lid owns 4 lid .. 4 lid + 3 of every 4096, rounds in order, two-level simd_sum) and
     then the first 8 of its simdgroups run `rowinv.rowdot`'s k order, so x is read in its own dtype (the fp32 cast of the stock sequence is
     exact) and the fp32 norm output array is never written.
  2. mixargs  -- `fuse.mhc_mix_pre` no longer copies its fp32 scale / base / mixes arguments (mx's astype always copies, even to the same
     dtype: three 1..24 element copies per mHC mix per sub-layer, each one a full launch).
  3. mixpre_off -- the mHC mix+collapse fold (KYOJIN_FUSE_MIXPRE, fuse.mhc_mix_pre) is turned off: its single launch serialises the collapse,
     the norm that follows it and the sinkhorn that is independent of both, and two overlapping small launches cost less than that serialisation
     (measured on perf/glueroof: -0.31 ms per 12 layers, 4/4 runs; here it also drops the argument copies, -4 dispatches per layer).
  4. mixnorm  -- OPTIONAL (KYOJIN_GLUE_FUSE_MIXNORM=1, needs the fold): `hc_pre` + `RMSNorm` + sinkhorn in one launch instead of three. It
     removes two launches per layer but hosts independent work in one threadgroup, which measured time-neutral here and slightly negative on the
     plain (uncompiled) body on perf/glueroof; kept for the launch count, off by default.
  5. norm     -- OPTIONAL (KYOJIN_GLUE_FUSE_NORM=1): perf/glueroof's glue_fuse.mix_norm, the collapse + RMSNorm + sinkhorn launch that does not
     need the fold (so it runs with cut 3 on) and the same fused body on the batched / N-user path. Measured neutral to slower there; off by default.
"""
from __future__ import annotations

import os

import mlx.core as mx

from .rowinv import pick_ks

ENABLED = os.environ.get("KYOJIN_GLUE_FUSE", "1") == "1"
MIXVEC = os.environ.get("KYOJIN_GLUE_FUSE_MIXVEC", "1") != "0"
MIXARGS = os.environ.get("KYOJIN_GLUE_FUSE_MIXARGS", "1") != "0"
MIXNORM = os.environ.get("KYOJIN_GLUE_FUSE_MIXNORM", "0") == "1"     # needs the mHC fold (KYOJIN_FUSE_MIXPRE=1): off by default, see the report
MIXPREOFF = os.environ.get("KYOJIN_GLUE_FUSE_MIXPRE_OFF", "1") != "0"  # the fold measured slower than the two overlapping launches (perf/glueroof)
PARTS = ("MIXVEC", "MIXARGS", "MIXNORM", "MIXPREOFF")
STATS = {"mixvec": 0, "mixargs": 0, "mixnorm": 0, "mixpre_off": 0, "fallback": 0}

HEADER = "#pragma METAL fp contract(off)\n"

# one threadgroup of 1024 threads per (output o, row b):
#   phase 1 (all threads): inv = rsqrt(mean(x^2) + eps) over the K row, mlx rms_looped's element split, per-thread order and two-level simd_sum
#   phase 2 (simdgroups 0..KS-1): y[b, o] = sum_k w[o, k] * (x[k] * inv), the k order and the reduction of rowinv.rowdot (fma, fp contract off)
_MIXVEC = r"""
    const uint o = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float ls[32];
    threadgroup float part[KS];
    threadgroup float inv_s[1];
    const device T* xr = x + ulong(b) * K;
    float acc = 0.0f;
    for (uint r = 0; r < ROUNDS; ++r) {
        const uint off = r * 4096 + lid * 4;
        if (off + 4 <= K) {
            for (uint i = 0; i < 4; ++i) {
                const float v = float(xr[off + i]);
                acc += v * v;
            }
        }
    }
    acc = simd_sum(acc);
    if (sg == 0) { ls[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { ls[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(ls[lane]);
        if (lane == 0) { inv_s[0] = metal::precise::rsqrt(acc / float(K) + eps[0]); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[0];
    if (sg < KS) {
        const device W* wp = w + ulong(o) * K;
        float d = 0.0f;
        for (uint kk = (sg * 32 + lane) * 8; kk < K; kk += 256 * KS) {
            float wv[8];
            for (uint j = 0; j < 8; ++j) { wv[j] = (kk + j < K) ? float(wp[kk + j]) : 0.0f; }
            for (uint j = 0; j < 8; ++j) {
                const float t = (kk + j < K) ? float(xr[kk + j]) * inv : 0.0f;
                d = fma(wv[j], t, d);
            }
        }
        const float t = simd_sum(d);
        if (lane == 0) { part[sg] = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float s = part[0];
        for (uint i = 1; i < KS; ++i) { s += part[i]; }
        y[ulong(b) * N + o] = s;
    }
"""
_kern: dict = {}                       # (mixbar, mixpf) -> kernel (alucut: KYOJIN_ALU_MIXBAR / KYOJIN_ALU_MIXPF, read per build)
_eps_cache: dict = {}


def _eps(e: float) -> mx.array:
    a = _eps_cache.get(e)
    if a is None:
        a = _eps_cache[e] = mx.array([e], dtype=mx.float32)
    return a


def mixvec(x: mx.array, fn: mx.array, eps: float) -> mx.array:
    """x [B, K] (fp16 / bf16 / fp32, K = hc_mult * hidden) and fn [N, K] fp32 -> [B, N] fp32, equals
    rowdot(mx.fast.rms_norm(x.astype(mx.float32), None, eps), fn); None when the shapes do not fit the kernel."""
    B, K = x.shape
    N = fn.shape[0]
    if K % 4 or K < 4096 or fn.shape[1] != K or fn.dtype != mx.float32 or x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return None
    ks = pick_ks(K, N)
    if 32 * ks > 1024:                                                              # the rms layout needs exactly 1024 threads
        return None
    from . import alucut as AC
    ak = (AC.mixbar(), AC.mixpf())
    kern = _kern.get(ak)
    if kern is None:
        kern = _kern[ak] = mx.fast.metal_kernel(name="kyojin_glue_mixvec" + ("_b" if ak[0] else "") + ("_p" if ak[1] else ""), input_names=["x", "w", "eps"],
                                                output_names=["y"], source=AC.mixvec_source(_MIXVEC, *ak), header=HEADER)
    STATS["mixvec"] += 1
    return kern(inputs=[x, fn, _eps(eps)], template=[("T", x.dtype), ("W", fn.dtype), ("K", K), ("N", N), ("KS", ks), ("ROUNDS", -(-K // 4096))],
                 grid=(N * 1024, B, 1), threadgroup=(1024, 1, 1), output_shapes=[(B, N)], output_dtypes=[mx.float32])[0]


# ---------------------------------------------------------------------------------------------------------------- 2. hc_pre + RMSNorm in one launch
# One threadgroup of 1024 threads per row: the collapse of the 4 residual copies (hc_pre) and the RMSNorm of its result are one kernel, because
# the norm needs the whole row and hc_pre produces it element by element. The element split, the per-thread order and the two-level simd_sum are
# mlx `rms_looped`'s (the same text as pf_glue._PRENORM, the prompt-row kernel that is bit-tested against nn.RMSNorm there); the sinkhorn of the
# mix that is already in `mixes` rides along in simdgroup 0 (verbatim from fuse._MIXPRE), since it is independent of the row reduction.
_MIXNORM = r"""
    const uint b = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup, ln = thread_index_in_simdgroup;
    if (lid < 32) {                                                       // hc_sinkhorn_split of this row (32 lanes, no barrier)
#pragma METAL fp contract(on)                                             // the stock sinkhorn kernel runs with default contraction (fuse._MIXPRE)
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
        const float active = (lid < (uint)HC) ? 1.0f : 0.0f;
        const uint  llane  = metal::min(lid, (uint)(HC - 1));
        float pre_z  = mix[llane]      * pre_scale  + base[llane];
        float post_z = mix[HC + llane] * post_scale + base[HC + llane];
        float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
        if (lid < (uint)HC) {
            pre_out[lid]  = pre_v;
            post_out[lid] = post_v;
        }
        float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC) * comb_scale
                  + *(const device float4*)(base + BASE_OFF + llane * HC)) * active;
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
        if (lid < (uint)HC) { *(device float4*)(comb_out + lid * HC) = r; }
    }
    float v[4 * ROUNDS];
    float acc = 0.0f;
    for (uint r = 0; r < ROUNDS; ++r) {
        const uint off = r * 4096 + lid * 4;
        if (off + 4 <= D) {
            for (uint i = 0; i < 4; ++i) {
                const ulong at = ulong(b) * 4 * D + off + i;
                float a = 0.0f;
                for (uint j = 0; j < 4; ++j) { a += pre[b * 4 + j] * float(x[at + ulong(j) * D]); }
                const float h = float(static_cast<T>(is_same<T, half>::value ? clamp(a, -65504.0f, 65504.0f) : a));
                v[r * 4 + i] = h;
                acc += h * h;
            }
        }
    }
    threadgroup float ls[32];
    threadgroup float inv_s[1];
    acc = simd_sum(acc);
    if (sg == 0) { ls[ln] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ln == 0) { ls[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(ls[ln]);
        if (ln == 0) { inv_s[0] = metal::precise::rsqrt(acc / float(D) + nrm_eps[0]); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[0];
    for (uint r = 0; r < ROUNDS; ++r) {
        const uint off = r * 4096 + lid * 4;
        for (uint i = 0; i < 4; ++i) {
            if (off + 4 <= D) { y[ulong(b) * D + off + i] = w[off + i] * static_cast<T>(v[r * 4 + i] * inv); }
        }
    }
"""
_k_mixnorm = None


def mixnorm_ok(x: mx.array, w: mx.array) -> bool:
    return (bool(x.ndim == 4 and x.shape[1] == 1 and x.shape[2] == 4) and x.dtype in (mx.float16, mx.bfloat16)
            and w.dtype == x.dtype and w.ndim == 1 and w.shape[0] == x.shape[3] and x.shape[3] >= 4096 and x.shape[3] % 4 == 0)


def mixpre_norm(mixes: mx.array, scale: mx.array, base: mx.array, x: mx.array, pre: mx.array, w: mx.array, iters: int,
                eps: float, norm_eps: float):
    """One launch for `hc_split_sinkhorn(mixes, ...)` -> (pre', post, comb), `hc_pre(x, pre)` and `RMSNorm(w)(hc_pre(x, pre))` -> y.
    Equals fuse.mhc_mix_pre(...)[0] passed through nn.RMSNorm(w, eps=norm_eps) (its element split and roundings are mlx rms_looped's)."""
    global _k_mixnorm
    B, S, H, D = x.shape
    assert S == 1 and H == 4
    if _k_mixnorm is None:
        _k_mixnorm = mx.fast.metal_kernel(name="kyojin_glue_mixpre_norm", input_names=["mixes", "scale", "base", "x", "pre", "w", "nrm_eps"],
                                          output_names=["y", "pre_o", "post", "comb"], source=_MIXNORM, header=HEADER)
    STATS["mixnorm"] += 1
    return _k_mixnorm(inputs=[mixes, scale, base, x, pre, w, _eps(norm_eps)],
                      template=[("T", x.dtype), ("D", D), ("ROUNDS", -(-D // 4096)), ("ITERS", iters), ("EPS_INT", round(eps / 1e-9))],
                      grid=(1024, B, 1), threadgroup=(1024, 1, 1),
                      output_shapes=[(B, 1, D), (B, 1, 4), (B, 1, 4), (B, 1, 4, 4)],
                      output_dtypes=[x.dtype, mx.float32, mx.float32, mx.float32])


# ---------------------------------------------------------------------------------------------------------------- wiring

def mix_vector(self, x, hc_fn):
    """Replacement of rows.mix_vector: the fused mix, or the stock two launches when the shapes do not fit."""
    if ENABLED and MIXVEC and x.ndim == 4 and x.shape[1] == 1 and x.shape[2] == 4 and (x.shape[2] * x.shape[-1]) % 4 == 0:
        y = mixvec(x.reshape(x.shape[0], -1), hc_fn, self.norm_eps)
        if y is not None:
            return y[:, None]
        STATS["fallback"] += 1
    return None


def f32_arg(a: mx.array) -> mx.array:
    """a as fp32 without the copy mx's astype makes when a already is fp32 (fuse.mhc_mix_pre's scale / base / mixes arguments).
    Called from fuse.py with the module read per call, so KYOJIN_GLUE_FUSE / _MIXARGS A/B inside one process."""
    if ENABLED and MIXARGS and a.dtype == mx.float32:
        STATS["mixargs"] += 1
        return a
    return a.astype(mx.float32)


def install() -> int:
    """Install the cuts (process wide, idempotent). Returns the number installed."""
    if not ENABLED:
        return 0
    n = 0
    from . import fuse as F                                            # every cut is wired once; MIXVEC / MIXARGS are read per call (one-process A/B)
    if MIXPREOFF and F.MIXPRE:                                          # read when a compiled region is built: set it here, before any trace
        F.MIXPRE = False
        STATS["mixpre_off"] += 1
        n += 1
    from . import rows as RW
    if not getattr(RW.mix_vector, "_kyojin_glue", False):
        stock_mv = RW.mix_vector

        def mv(self, x, hc_fn):
            y = mix_vector(self, x, hc_fn)
            return stock_mv(self, x, hc_fn) if y is None else y

        mv._kyojin_glue = True
        RW.mix_vector = mv
        n += 1
    return n
