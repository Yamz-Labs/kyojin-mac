# SPDX-License-Identifier: Apache-2.0
"""dec_stage one-launch stages for the mHC glue of a decode block (lever dec_stage, sub-flags dec_stage_in / dec_stage_ffn; generic, no model names).

A sub-layer's mHC glue is, on the served lane, mix_vector (rms_norm + row dot, 2 launches), the sinkhorn split, hc_pre, the block RMSNorm and, after
the sub-layer, hc_expand: 6 launches per sub-layer, 12 per layer. Two kernels replace them, a stage each, f32 inside and one rounding at the end
(the technique of a fused decode stage: one launch, the x rows stay in registers, rounding where the stage ends):

  mix_stage     one launch, grid (25 threadgroups, rows). Threadgroups 0..23 each take ONE of the 24 mix rows (the weight row is read once per
                row of the pass, the 24 rows spread over the cores: a single-threadgroup dot is the slow form glue_fuse measured), recompute the
                flat sum of squares of the row, and write z_m = (x . fn_m) * rsqrt(mean(x^2) + eps) (f32); the first four also write
                pre' = sigmoid(z * scale0 + base) + eps. Threadgroup 24 collapses the four streams with the incoming pre (f32, no fp16 round of the
                collapse) and writes h = T(w * collapse * rsqrt(mean(collapse^2) + eps)): the only rounding is the last one.
  expand_stage  one launch, grid (D / 128 threadgroups, rows). Every threadgroup runs the sinkhorn split of the row's z (one simdgroup, the text of
                the mlx-vlm kernel) into threadgroup memory, then y[j] = T(post[j] * f + sum_a comb[a][j] * res[a]) with f32 products and one
                rounding; it also writes the fp32 copy of the rounded y that the next mix reads (fuse.take_f32). The sinkhorn is no launch of its own.

Every output element of a row depends on that row only, with one reduction order (a function of D alone): a row has the same bits whatever the
number of rows of the pass, so a plain step, a verify of 2..24 rows and a replay agree. The arithmetic is NOT the chain's (the dot and the sums of
squares are accumulated in another order, the collapse is not rounded to fp16 before its norm), so the outputs are not bit-equal to the lane's
chain; they are closer to an f32 reference than the chain is (tests/test_dec_stage_k.py measures both against float64 per output).
`mix_ref` / `expand_ref` are the same math in mx ops (CPU too): the reference of the Metal tests and the first-use self check, which turns the stage
off, with a printed line, if the kernels do not compile or do not agree with it on the machine.
"""
from __future__ import annotations

import mlx.core as mx

NT = 256                                       # threads of every stage threadgroup (8 simdgroups)
NMIX = 24                                      # (2 + hc) * hc mix rows, hc = 4
STATE = {"checked": None}                      # first-use self check: None = not run, True = kernels agree with the reference, False = off


_MIX = r"""
    // role 0..23: mix row `role`; role 24: collapse + RMSNorm. One threadgroup per (role, row): a row never sees the other rows.
    constexpr uint NT = 256, NSG = 8, K = 4 * D;
    const uint role = threadgroup_position_in_grid.x;
    const uint b = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sl = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float part[2 * NSG];
    threadgroup float inv_s[1];
    if (role < 24) {
        const device float* f = fn + ulong(role) * K;
        const device T* xr = x + ulong(b) * K;
        float dot = 0.0f, ss = 0.0f;
        for (uint i = tid * 4; i < K; i += NT * 4) {
            for (uint j = 0; j < 4; ++j) {
                const float xv = float(xr[i + j]);
                dot = fma(xv, f[i + j], dot);
                ss = fma(xv, xv, ss);
            }
        }
        dot = simd_sum(dot);
        ss = simd_sum(ss);
        if (sl == 0) { part[sg] = dot; part[NSG + sg] = ss; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float d = 0.0f, s = 0.0f;
            for (uint g = 0; g < NSG; ++g) { d += part[g]; s += part[NSG + g]; }
            const float zz = d * metal::precise::rsqrt(s / float(K) + cst[0]);
            z[b * 24 + role] = zz;
            if (role < 4) {
                pre_o[b * 4 + role] = 1.0f / (1.0f + metal::precise::exp(-(zz * scale[0] + base[role]))) + cst[2];
            }
        }
        return;
    }
    const device T* xb = x + ulong(b) * K;
    float keep[NG][4];
    float acc = 0.0f;
    for (uint g = 0; g < NG; ++g) {
        const uint i0 = tid * 4 + g * NT * 4;
        for (uint j = 0; j < 4; ++j) {
            float a = 0.0f;
            if (i0 + j < D) {
                for (uint s = 0; s < 4; ++s) { a = fma(cpre[b * 4 + s], float(xb[ulong(s) * D + i0 + j]), a); }
                acc = fma(a, a, acc);
            }
            keep[g][j] = a;
        }
    }
    acc = simd_sum(acc);
    if (sl == 0) { part[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint g = 0; g < NSG; ++g) { t += part[g]; }
        inv_s[0] = metal::precise::rsqrt(t / float(D) + cst[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[0];
    for (uint g = 0; g < NG; ++g) {
        const uint i0 = tid * 4 + g * NT * 4;
        for (uint j = 0; j < 4; ++j) {
            if (i0 + j < D) { hn[ulong(b) * D + i0 + j] = static_cast<T>(float(w[i0 + j]) * keep[g][j] * inv); }
        }
    }
"""

# The sinkhorn text is the one of glue_fuse._SINKHORN / the mlx-vlm kernel (one row, one simdgroup, 32 lanes) writing post / comb to threadgroup memory.
_EXPAND = r"""
    constexpr int HC = 4;
    constexpr int BASE_OFF = 2 * HC;
    constexpr float EPS = EPS_INT * 1e-9;
    const uint b = threadgroup_position_in_grid.y;
    const uint d = thread_position_in_grid.x;
    threadgroup float pc[20];                                    // post[4], comb[4][4] (row a, column j)
    if (simdgroup_index_in_threadgroup == 0) {
        const uint lane = thread_index_in_simdgroup;
        const device float* mix = z + b * 24;
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint llane = metal::min(lane, (uint)(HC - 1));
        float post_z = mix[HC + llane] * post_scale + base[HC + llane];
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
        float4 v = (*(const device float4*)(mix + BASE_OFF + llane * HC) * comb_scale
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
        if (lane < (uint)HC) {
            pc[lane] = post_v;
            pc[4 + lane * 4 + 0] = r.x;
            pc[4 + lane * 4 + 1] = r.y;
            pc[4 + lane * 4 + 2] = r.z;
            pc[4 + lane * 4 + 3] = r.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= D) { return; }
    float res[4];
    for (uint a = 0; a < 4; ++a) { res[a] = float(rs[(ulong(b) * 4 + a) * D + d]); }
    const float xv = float(y[ulong(b) * D + d]);
    for (uint j = 0; j < 4; ++j) {
        float mixed = 0.0f;
        for (uint a = 0; a < 4; ++a) { mixed = fma(pc[4 + a * 4 + j], res[a], mixed); }
        const float v = fma(pc[j], xv, mixed);
        const T o = static_cast<T>(is_same<T, half>::value ? clamp(v, -65504.0f, 65504.0f) : v);   // saturate at the fp16 range: see fuse.sat
        out[(ulong(b) * 4 + j) * D + d] = o;
        out32[(ulong(b) * 4 + j) * D + d] = float(o);
    }
"""

_k: dict = {}
_c: dict = {}


def _cst(*v: float) -> mx.array:
    a = _c.get(v)
    if a is None:
        a = _c[v] = mx.array(list(v), dtype=mx.float32)
    return a


def _f32(a: mx.array) -> mx.array:
    return a if a.dtype == mx.float32 else a.astype(mx.float32)


def mix_stage(x: mx.array, cpre: mx.array, fn: mx.array, scale: mx.array, base: mx.array, w: mx.array, eps_flat: float, eps_norm: float,
              hc_eps: float):
    """x [R, 1, 4, D] (fp16 / bf16 residual streams), cpre [R, 1, 4] fp32 (the incoming pre), fn [24, 4D] fp32, scale [3], base [24], w [D] (x.dtype)
    -> (h [R, 1, D] x.dtype, z [R, 24] fp32 raw mixes, pre' [R, 4] fp32). One launch."""
    R, _, H, D = x.shape
    assert H == 4 and fn.shape == (NMIX, 4 * D) and cpre.shape == (R, 1, 4) and w.shape == (D,) and D % 4 == 0
    k = _k.get("mix")
    if k is None:
        k = _k["mix"] = mx.fast.metal_kernel(name="kyojin_stage_mix", input_names=["x", "cpre", "fn", "scale", "base", "w", "cst"],
                                             output_names=["hn", "z", "pre_o"], source=_MIX)
    hn, z, pre_o = k(inputs=[x, _f32(cpre), _f32(fn), _f32(scale), _f32(base), w, _cst(eps_flat, eps_norm, hc_eps)],
                     template=[("T", x.dtype), ("D", D), ("NG", -(-D // (NT * 4)))], grid=((NMIX + 1) * NT, R, 1), threadgroup=(NT, 1, 1),
                     output_shapes=[(R, 1, D), (R, NMIX), (R, 4)], output_dtypes=[x.dtype, mx.float32, mx.float32])
    return hn, z, pre_o


def expand_stage(y: mx.array, res: mx.array, z: mx.array, scale: mx.array, base: mx.array, iters: int, hc_eps: float):
    """y [R, 1, D] (the sub-layer output, after the ablit hook), res [R, 1, 4, D], z [R, 24] fp32 (mix_stage) -> (out [R, 1, 4, D] y.dtype, out32 fp32 copy
    of out). The sinkhorn split runs inside. One launch."""
    R, _, D = y.shape
    assert res.shape == (R, 1, 4, D) and z.shape == (R, NMIX)
    k = _k.get("expand")
    if k is None:
        k = _k["expand"] = mx.fast.metal_kernel(name="kyojin_stage_expand", input_names=["y", "rs", "z", "scale", "base"], output_names=["out", "out32"],
                                                source=_EXPAND)
    return k(inputs=[y, res, _f32(z), _f32(scale), _f32(base)], template=[("T", y.dtype), ("D", D), ("ITERS", int(iters)), ("EPS_INT", round(hc_eps / 1e-9))],
             grid=(-(-D // 128) * 128, R, 1), threadgroup=(128, 1, 1), output_shapes=[(R, 1, 4, D), (R, 1, 4, D)], output_dtypes=[y.dtype, mx.float32])


# ---------------------------------------------------------------------------------------------------------------- references (mx ops, f32)
def mix_ref(x, cpre, fn, scale, base, w, eps_flat, eps_norm, hc_eps):
    """The math of mix_stage in mx ops: f32 everywhere, h rounded to x.dtype at the end."""
    R, _, _, D = x.shape
    xf = x.astype(mx.float32).reshape(R, 4 * D)
    z = (xf @ fn.T) * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps_flat)
    pre = mx.sigmoid(z[:, :4] * scale[0] + base[:4]) + hc_eps
    h = (cpre.reshape(R, 4, 1) * x.astype(mx.float32).reshape(R, 4, D)).sum(axis=1)
    hn = (w.astype(mx.float32) * h * mx.rsqrt(mx.mean(h * h, axis=-1, keepdims=True) + eps_norm)).astype(x.dtype)
    return hn.reshape(R, 1, D), z, pre


def sinkhorn_ref(z, scale, base, iters, hc_eps):
    """post [R, 4], comb [R, 4, 4] of the split of z [R, 24] (the algorithm of the kernel text: row softmax + eps, then alternate column / row normalisation)."""
    post = 2.0 * mx.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    v = (z[:, 8:24] * scale[2] + base[8:24]).reshape(-1, 4, 4)
    r = mx.softmax(v, axis=-1) + hc_eps
    r = r / (r.sum(axis=-2, keepdims=True) + hc_eps)
    for _ in range(1, iters):
        r = r / (r.sum(axis=-1, keepdims=True) + hc_eps)
        r = r / (r.sum(axis=-2, keepdims=True) + hc_eps)
    return post, r


def expand_ref(y, res, z, scale, base, iters, hc_eps):
    post, comb = sinkhorn_ref(z, scale, base, iters, hc_eps)
    R, _, D = y.shape
    mixed = mx.einsum("raj,rad->rjd", comb, res.astype(mx.float32).reshape(R, 4, D))
    out = post[:, :, None] * y.astype(mx.float32).reshape(R, 1, D) + mixed
    out = (mx.clip(out, -65504, 65504) if y.dtype == mx.float16 else out).astype(y.dtype)
    return out.reshape(R, 1, 4, D)


# ---------------------------------------------------------------------------------------------------------------- first use
def usable() -> bool:
    """True when the two kernels compile on this machine and agree with the references (run once, on a small synthetic block). False: the caller keeps
    the lane's chain."""
    if STATE["checked"] is not None:
        return STATE["checked"]
    ok = False
    try:
        if not mx.metal.is_available():
            raise RuntimeError("no Metal")
        import numpy as np
        rng = np.random.default_rng(7)
        R, D = 3, 384                                                          # NG = 1, a ragged tail group (384 < 1024), 3 rows
        x = mx.array((rng.standard_normal((R, 1, 4, D)) * 0.5).astype(np.float16))
        cpre = mx.array(rng.uniform(0.1, 1.0, (R, 1, 4)).astype(np.float32))
        fn = mx.array((rng.standard_normal((NMIX, 4 * D)) * 0.02).astype(np.float32))
        scale = mx.array(np.array([0.7, 0.9, 1.1], np.float32))
        base = mx.array((rng.standard_normal(NMIX) * 0.3).astype(np.float32))
        w = mx.array(rng.uniform(0.5, 1.5, (D,)).astype(np.float16))
        hn, z, pre = mix_stage(x, cpre, fn, scale, base, w, 1e-6, 1e-6, 1e-6)
        hr, zr, pr = mix_ref(x, cpre, fn, scale, base, w, 1e-6, 1e-6, 1e-6)
        y = mx.array((rng.standard_normal((R, 1, D)) * 0.5).astype(np.float16))
        out, out32 = expand_stage(y, x, zr, scale, base, 20, 1e-6)
        orf = expand_ref(y, x, zr, scale, base, 20, 1e-6)
        mx.eval(hn, z, pre, hr, zr, pr, out, out32, orf)
        err = lambda a, b: float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
        # z and pre' against float64, not against mix_ref's `xf @ fn.T`: that matmul keeps 10 bits of each operand on the tensor units (fuse_attn.trunc10), which can put an
        # fp32 `fn` ~1e-3 off exact; the kernel's order is ~1e-6 off exact (tests/test_cmd5_kernel_refs.py)
        xe = np.array(x.astype(mx.float32)).reshape(R, 4 * D).astype(np.float64)
        z_ex = (xe @ np.array(fn).astype(np.float64).T) / np.sqrt((xe * xe).mean(-1, keepdims=True) + 1e-6)
        pre_ex = 1 / (1 + np.exp(-(z_ex[:, :4] * 0.7 + np.array(base)[:4]))) + 1e-6
        ok = (err(hn, hr) < 2e-2 and float(np.abs(np.array(z) - z_ex).max()) < 1e-4 and float(np.abs(np.array(pre) - pre_ex).max()) < 1e-4
              and err(out, orf) < 2e-2 and err(out32, out.astype(mx.float32)) == 0.0)
        if not ok:
            print(f"[kyojin] dec_stage mix/expand kernels disagree with the reference (hn {err(hn, hr):.2e} z {err(z, zr):.2e} out {err(out, orf):.2e}): stage glue off", flush=True)
    except Exception as e:                                                      # a Metal compile error surfaces here, at the first eval
        print(f"[kyojin] dec_stage mix/expand kernels unavailable ({type(e).__name__}: {str(e)[:160]}): stage glue off", flush=True)
    STATE["checked"] = ok
    return ok
