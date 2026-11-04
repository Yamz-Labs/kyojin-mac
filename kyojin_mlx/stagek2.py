# SPDX-License-Identifier: Apache-2.0
"""dec_stage k1-k2 kernels: the stage between the wq_a | wkv projection and the attention kernel (generic: any MLA-style block with a q latent
norm and a window ring of kv rows; geometry comes from the arrays, nothing here knows a model name).

qkv_stage (lever dec_qkv, KYOJIN_DEC_QKV): the q latent RMSNorm (q_norm) and the kv chain (kv_norm + rope tail + fp8 round trip + window ring scatter,
dsv4_core.kv_chain) are independent reads of the one group launch's two outputs, and ran as two dependent-looking launches per layer. One launch now,
two roles by threadgroup index and the grid stays wide:

    threadgroups 0 .. R-1      q_norm of row r (128 threads, 4 contiguous values per thread, strided by 512, f32 sum of squares, simd_sum, then the
                               simdgroup partials; qr = w * T(x * rsqrt(mean(x^2) + eps)), one rounding)
    threadgroups R ..          the kv chain: the TEXT of dsv4_core._KV_CHAIN with the threadgroup index shifted by R (one threadgroup per (row, ring
                               slot), 128 threads): bit-identical to kv_chain, which the Metal test asserts.

Why it is wide where glue_fuse_norm / port_suglue measured slower: it merges two launches that were independent, so no stage waits for another
inside the launch (a role never reads what another role of the launch wrote), and the larger role (R * W copy threadgroups) is not serialised
behind the narrower one. The q role is NOT mx.fast.rms_norm's bits (its single-row kernel uses ceil(N / 4 / 32) * 32 threads; this one 128, so the
sum of squares adds in another order): within f32 reference tolerance, closer to it than the stream-dtype norm is no worse (tests/test_dec_stage_k2.py).
`qkv_ref` is the same math in mx ops (CPU too): the reference of the tests and of the first-use self check, which turns the lever off, with a
printed line, if the kernel does not compile or does not agree with it on the machine.
"""
from __future__ import annotations

import mlx.core as mx

from . import dsv4_core as DC

NT = 128                                       # threads of every threadgroup (the kv chain's width)
STATE = {"checked": None}                      # first-use self check: None = not run, True = kernel agrees, False = off
STATS = {"qkv": 0}

_SHIFT = "const uint tg = threadgroup_position_in_grid.x;"
assert DC._KV_CHAIN.count(_SHIFT) == 1

_Q_ROLE = r"""
    threadgroup float qls[32];
    if (threadgroup_position_in_grid.x < uint(QT)) {
        const uint b = threadgroup_position_in_grid.x;
        const uint qlid = thread_position_in_threadgroup.x;
        const uint qsg = simdgroup_index_in_threadgroup, qsl = thread_index_in_simdgroup;
        const device T* xr = qa + ulong(b) * uint(NQ);
        float acc = 0.0f;
        for (uint e0 = qlid * 4; e0 < uint(NQ); e0 += 128 * 4) {
            for (uint i = 0; i < 4; ++i) {
                const float f = float(xr[e0 + i]);
                acc += f * f;
            }
        }
        acc = simd_sum(acc);
        if (qsg == 0) qls[qsl] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qsl == 0) qls[qsg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qsg == 0) {
            const float t = simd_sum(qls[qsl]);
            if (qsl == 0) qls[0] = metal::precise::rsqrt(t / float(NQ) + eq[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = qls[0];
        for (uint e0 = qlid * 4; e0 < uint(NQ); e0 += 128 * 4) {
            for (uint i = 0; i < 4; ++i) {
                qr[ulong(b) * uint(NQ) + e0 + i] = wq[e0 + i] * static_cast<T>(float(xr[e0 + i]) * inv);
            }
        }
        return;
    }
"""
SOURCE = _Q_ROLE + DC._KV_CHAIN.replace(_SHIFT, "const uint tg = threadgroup_position_in_grid.x - uint(QT);")
_k: dict = {}


def supported(qa: mx.array, kvr: mx.array, ring: mx.array) -> bool:
    """fp16 rows, a q latent a multiple of 4 wide, the kv chain's own geometry (D = 4 * 128 threads, a ring of D-wide rows)."""
    return bool(qa.dtype == mx.float16 and kvr.dtype == mx.float16 and qa.ndim == 2 and kvr.ndim == 2 and qa.shape[0] == kvr.shape[0]
                and qa.shape[1] % 4 == 0 and kvr.shape[1] == 4 * NT and ring.ndim == 3 and ring.shape[2] == kvr.shape[1])


def qkv_stage(qa: mx.array, wq: mx.array, eps_q: float, kvr: mx.array, w: mx.array, ring: mx.array, sess: mx.array, wmap: mx.array, rt: mx.array,
              eps_kv: float):
    """qa [R, NQ] (the q latent, wq_a output), kvr [R, D] (raw wkv output), ring [S, W, D], sess [R] u32, wmap [R * W] i32, rt [R, RD] f32
    -> (qr [R, NQ] = q_norm(qa), win [R, W, D] = dsv4_core.kv_chain(...)). One launch."""
    R, NQ = qa.shape
    D = kvr.shape[1]
    W = ring.shape[1]
    kq = "qkv" + DC._fk()
    k = _k.get(kq)
    if k is None:
        k = _k[kq] = mx.fast.metal_kernel(name="kyojin_stage_qkv" + DC._fk(), input_names=["qa", "wq", "eq", "kvr", "w", "ring", "sess", "wmap", "rt", "eps"],
                                          output_names=["qr", "out"], source=DC.rope_src(SOURCE), header=DC._CONTRACT_OFF)
    STATS["qkv"] += 1
    qr, win = k(inputs=[qa, wq, DC._f32((eps_q,)), kvr, w, ring, sess, wmap, rt, DC._f32((eps_kv,))],
                template=[("T", qa.dtype), ("NQ", NQ), ("QT", R), ("D", D), ("W", W), ("RD", rt.shape[-1])],
                grid=((R + R * W) * NT, 1, 1), threadgroup=(NT, 1, 1), output_shapes=[(R, NQ), (R, W, D)], output_dtypes=[qa.dtype, kvr.dtype])
    return qr, win


# ---------------------------------------------------------------------------------------------------------------- reference (mx ops)
def qnorm_ref(qa, wq, eps):
    """q_norm in f32 with one rounding: w * T(x * rsqrt(mean(x^2) + eps))."""
    x = qa.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return wq * (x * inv).astype(qa.dtype)


def kv_ref(kvr, w, ring, sess, wmap, rt, eps):
    """The stock chain of dsv4_core.kv_chain in mx ops: kv_norm, rope tail (interleaved pairs), fp8 ue8m0 round trip, ring overlay. fp16 rounding where
    the kernel rounds."""
    R, D = kvr.shape
    W = ring.shape[1]
    RD = rt.shape[-1]
    x = kvr.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    nb = (w * (x * inv).astype(kvr.dtype)).astype(mx.float32)
    cs = rt.reshape(R, RD // 2, 2)
    c, s = cs[..., 0], cs[..., 1]
    tail = nb[:, D - RD:].reshape(R, RD // 2, 2)
    x1, x2 = tail[..., 0], tail[..., 1]
    r1 = (x1 * c - x2 * s).astype(kvr.dtype).astype(mx.float32)
    r2 = (x1 * s + x2 * c).astype(kvr.dtype).astype(mx.float32)
    v = mx.concatenate([nb[:, :D - RD], mx.stack([r1, r2], axis=-1).reshape(R, RD)], axis=-1)
    g = v.reshape(R, D // 32, 32)
    amax = mx.maximum(mx.max(mx.abs(g), axis=-1, keepdims=True), 1e-4)
    sc = mx.power(2.0, mx.ceil(mx.log2(amax / 448.0)))
    vv = mx.clip(g / sc, -448.0, 448.0)
    e = mx.maximum(mx.floor(mx.log2(mx.maximum(mx.abs(vv), 1e-30))) - 3, -9)
    step = mx.power(2.0, e)
    q = (mx.round(vv / step) * step * sc).reshape(R, D).astype(kvr.dtype)
    base = ring[sess.astype(mx.int32)]                                   # [R, W, D]
    wm = wmap.reshape(R, W)
    take = mx.where((wm < 0)[..., None], base, q[mx.maximum(wm, 0)])
    return take


def qkv_ref(qa, wq, eps_q, kvr, w, ring, sess, wmap, rt, eps_kv):
    return qnorm_ref(qa, wq, eps_q), kv_ref(kvr, w, ring, sess, wmap, rt, eps_kv)


# ---------------------------------------------------------------------------------------------------------------- first use
def usable() -> bool:
    """True when the kernel compiles on this machine and agrees with dsv4_core.kv_chain bit for bit and with the q_norm reference (run once, small
    synthetic pass). False: the caller keeps the stage's two launches."""
    if STATE["checked"] is not None:
        return STATE["checked"]
    ok = False
    try:
        if not mx.metal.is_available():
            raise RuntimeError("no Metal")
        import numpy as np
        rng = np.random.default_rng(11)
        R, NQ, D, W, RD = 2, 1280, 512, 8, 64                             # NQ = 2.5 passes of 512: a ragged tail
        qa = mx.array((rng.standard_normal((R, NQ)) * 0.8).astype(np.float16))
        wq = mx.array(rng.uniform(0.5, 1.5, NQ).astype(np.float16))
        kvr = mx.array((rng.standard_normal((R, D)) * 0.8).astype(np.float16))
        w = mx.array(rng.uniform(0.5, 1.5, D).astype(np.float16))
        ring = mx.array((rng.standard_normal((R, W, D)) * 0.5).astype(np.float16))
        sess = mx.array([1, 0], dtype=mx.uint32)
        wm = [-1] * (R * W)
        wm[0 * W + 3], wm[1 * W + 5], wm[1 * W + 2] = 0, 1, 0
        wmap = mx.array(wm, dtype=mx.int32)
        rt = mx.array(rng.uniform(-1, 1, (R, RD)).astype(np.float32))
        qr, win = qkv_stage(qa, wq, 1e-6, kvr, w, ring, sess, wmap, rt, 1e-6)
        want_win = DC.kv_chain(kvr, w, ring, sess, wmap, rt, 1e-6)
        want_q = qnorm_ref(qa, wq, 1e-6)
        mx.eval(qr, win, want_win, want_q)
        ok = bool(mx.array_equal(win, want_win)) and float(mx.max(mx.abs(qr.astype(mx.float32) - want_q.astype(mx.float32)))) < 2e-2
        if not ok:
            print("[kyojin] dec_qkv kernel disagrees with the kv chain / q_norm reference: lever off", flush=True)
    except Exception as e:                                                  # a Metal compile error surfaces here, at the first eval
        print(f"[kyojin] dec_qkv kernel unavailable ({type(e).__name__}: {str(e)[:160]}): lever off", flush=True)
    STATE["checked"] = ok
    return ok


# ---------------------------------------------------------------------------------------------------------------- x-shared f32 GEMV pair
# lever dec_cproj (KYOJIN_DEC_CPROJ): the two compressor projections of a kv-source layer read the same row: the chain casts x to f32 (a launch), then
# runs two f32 matmuls (MLX promotes the stored weight to f32 for the product when it is fp16 / bf16). One launch instead: one simdgroup per weight row
# (lanes stride the row by 4 values, f32 FMAs, simd_sum), 8 weight rows per threadgroup, 4 x rows per grid.y step, the weight read in its stored dtype
# (no f32 copy of the weight, no cast of x). The products are the chain's (f32 x * f32 value of the weight); the sum order differs (lane partials), so
# within f32 reference tolerance, not bit-equal. The grid is (N0 + N1) / 8 threadgroups: wide, no role waits for another.
_XGEMV = r"""
    constexpr uint RB = 4;
    const uint sg = simdgroup_index_in_threadgroup, sl = thread_index_in_simdgroup;
    const uint j = threadgroup_position_in_grid.x * 8 + sg;
    if (j >= uint(N0 + N1)) return;
    const uint r0 = threadgroup_position_in_grid.y * RB;
    const device WT* wr = j < uint(N0) ? w0 + ulong(j) * K : w1 + ulong(j - uint(N0)) * K;
    float acc[RB] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = sl * 4; k < uint(K); k += 128) {
        float wv[4];
        for (uint i = 0; i < 4; ++i) wv[i] = float(wr[k + i]);
        for (uint r = 0; r < RB; ++r) {
            if (r0 + r < uint(R)) {
                const device T* xr = x + ulong(r0 + r) * K;
                for (uint i = 0; i < 4; ++i) acc[r] = fma(wv[i], float(xr[k + i]), acc[r]);
            }
        }
    }
    for (uint r = 0; r < RB; ++r) {
        const float s = simd_sum(acc[r]);
        if (sl == 0 && r0 + r < uint(R)) {
            if (j < uint(N0)) y0[ulong(r0 + r) * N0 + j] = s;
            else y1[ulong(r0 + r) * N1 + (j - uint(N0))] = s;
        }
    }
"""


def xgemv_supported(x: mx.array, w0: mx.array, w1: mx.array) -> bool:
    return bool(x.dtype in (mx.float16, mx.bfloat16, mx.float32) and w0.dtype in (mx.float16, mx.bfloat16, mx.float32) and w1.dtype == w0.dtype
                and w0.ndim == 2 and w1.ndim == 2 and w0.shape[1] == w1.shape[1] == x.shape[-1] and x.shape[-1] % 128 == 0)


def xgemv(x: mx.array, w0: mx.array, w1: mx.array):
    """x [R, K], w0 [N0, K], w1 [N1, K] (nn.Linear weights, any of f32 / f16 / bf16) -> (x32 @ w0.T [R, N0], x32 @ w1.T [R, N1]) in f32. One launch."""
    R, K = x.shape
    N0, N1 = w0.shape[0], w1.shape[0]
    k = _k.get("xgemv")
    if k is None:
        k = _k["xgemv"] = mx.fast.metal_kernel(name="kyojin_stage_xgemv", input_names=["x", "w0", "w1"], output_names=["y0", "y1"], source=_XGEMV)
    return k(inputs=[x, w0, w1], template=[("T", x.dtype), ("WT", w0.dtype), ("K", K), ("N0", N0), ("N1", N1), ("R", R)],
             grid=(-(-(N0 + N1) // 8) * 256, -(-R // 4), 1), threadgroup=(256, 1, 1), output_shapes=[(R, N0), (R, N1)],
             output_dtypes=[mx.float32, mx.float32])


_EXACT_CHUNK = 1 << 24                                                          # elements of the [R, n, K] product held at once (64 MiB fp32)


def exact_f32_matmul_t(x, w):
    """x [R, K] @ w [N, K].T in fp32 with the operands untouched, on the elementwise units. The stock fp32 `@` of a GPU with tensor units (M5) rounds
    both operands to ~10 mantissa bits for 2+ rows (relative error ~1e-3: ULTRA.md, tests/test_attn_prefill.py), so it is no reference for a kernel
    that multiplies in fp32. Product and sum stay fp32 (error ~1e-7 relative); only a reference, never on a served path."""
    xf, wf = x.astype(mx.float32), w.astype(mx.float32)
    R, K = xf.shape
    N = wf.shape[0]
    step = max(1, _EXACT_CHUNK // max(1, R * K))
    cols = [(xf[:, None, :] * wf[None, i:i + step, :]).sum(-1) for i in range(0, N, step)]
    return cols[0] if len(cols) == 1 else mx.concatenate(cols, axis=1)


def xgemv_ref(x, w0, w1):
    return exact_f32_matmul_t(x, w0), exact_f32_matmul_t(x, w1)


def xgemv_usable() -> bool:
    if STATE.get("xgemv") is not None:
        return STATE["xgemv"]
    ok = False
    try:
        if not mx.metal.is_available():
            raise RuntimeError("no Metal")
        import numpy as np
        rng = np.random.default_rng(5)
        ok = True
        for wdt, npdt in ((mx.float16, np.float16), (mx.float32, np.float32)):
            x = mx.array(rng.standard_normal((6, 640)).astype(np.float16))                  # 6 rows: one full and one ragged row block; K = 5 strides
            w0 = mx.array((rng.standard_normal((20, 640)) * 0.05).astype(npdt))
            w1 = mx.array((rng.standard_normal((13, 640)) * 0.05).astype(npdt))
            y0, y1 = xgemv(x, w0, w1)
            mx.eval(y0, y1)
            # float64 reference: xgemv_ref's fp32 matmul keeps 10 bits of each operand on the tensor units (fuse_attn.trunc10), which puts an fp32 weight ~1e-3 off exact;
            # the kernel's order is ~1e-6 off exact (tests/test_cmd5_kernel_refs.py)
            xe = np.array(x.astype(mx.float32)).astype(np.float64)
            e0 = np.abs(np.array(y0).astype(np.float64) - xe @ np.array(w0.astype(mx.float32)).astype(np.float64).T).max()
            e1 = np.abs(np.array(y1).astype(np.float64) - xe @ np.array(w1.astype(mx.float32)).astype(np.float64).T).max()
            ok = ok and e0 < 1e-4 and e1 < 1e-4
        if not ok:
            print("[kyojin] dec_cproj kernel disagrees with the f32 matmul: lever off", flush=True)
    except Exception as e:
        print(f"[kyojin] dec_cproj kernel unavailable ({type(e).__name__}: {str(e)[:160]}): lever off", flush=True)
    STATE["xgemv"] = ok
    return ok
