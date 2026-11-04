# SPDX-License-Identifier: Apache-2.0
"""numpy emulations of the reduction order of two decode-stage kernels, and of the tensor-unit fp32 matmul, so a Metal test can be judged on a CPU.

xgemv_order   stagek2._XGEMV: lane l of a simdgroup adds k = 4 l + 128 m + i (i = 0..3, m = 0..) with one fp32 fma each, then simd_sum (butterfly 16, 8, 4, 2, 1).
mix_z_order   stagek._MIX roles 0..23: thread t adds i = 4 t + 1024 g + j (g outer, j inner) for the dot and for the sum of squares, simd_sum per simdgroup of 32,
              then the 8 simdgroup partials in order; z = dot * precise::rsqrt(ss / K + eps).
trunc10       fuse_attn.trunc10: the operand of an fp32 matmul on the tensor units keeps 10 mantissa bits (rounded toward zero) -- the repo's own measurement.
fma is emulated as an exact float64 product (24 + 24 bits fit in 53) added to the fp32 accumulator and rounded to fp32 once.
"""
from __future__ import annotations

import numpy as np


def f32(a):
    return np.asarray(a, np.float32)


def trunc10(a):
    a = f32(a)
    return (a.view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32)


def _butterfly(v):
    """simd_sum over the last axis of 32 lanes."""
    s = v
    for off in (16, 8, 4, 2, 1):
        s = f32(s[..., :off] + s[..., off:2 * off])
    return s[..., 0]


def xgemv_order(x, w):
    """x [R, K] f32 values, w [N, K] f32 values -> [R, N] f32 in the order of stagek2._XGEMV."""
    R, K = x.shape
    N = w.shape[0]
    acc = np.zeros((R, N, 32), np.float32)
    for m in range(K // 128):
        for i in range(4):
            idx = np.arange(32) * 4 + 128 * m + i
            acc = f32(acc.astype(np.float64) + x[:, None, idx].astype(np.float64) * w[None, :, idx].astype(np.float64))
    return _butterfly(acc)


def mix_z_order(x, fn, eps=1e-6):
    """x [R, K] f32 values (K = 4 D), fn [24, K] f32 -> z [R, 24] f32 in the order of stagek._MIX."""
    R, K = x.shape
    dot = np.zeros((R, fn.shape[0], 256), np.float32)
    ss = np.zeros((R, 256), np.float32)
    for g in range(0, K, 1024):
        for j in range(4):
            idx = g + np.arange(256) * 4 + j
            ok = idx < K
            idx = np.where(ok, idx, 0)
            xv = x[:, idx] * ok
            fv = fn[:, idx] * ok
            dot = f32(dot.astype(np.float64) + xv[:, None, :].astype(np.float64) * fv[None, :, :].astype(np.float64))
            ss = f32(ss.astype(np.float64) + xv.astype(np.float64) ** 2)
    d = _butterfly(dot.reshape(R, fn.shape[0], 8, 32))        # [R, 24, 8]
    s = _butterfly(ss.reshape(R, 8, 32))                      # [R, 8]
    dd = np.zeros(d.shape[:2], np.float32)
    sv = np.zeros(s.shape[:1], np.float32)
    for g in range(8):
        dd = f32(dd + d[..., g])
        sv = f32(sv + s[..., g])
    inv = f32(1.0 / np.sqrt(f32(f32(sv / np.float32(K)) + np.float32(eps))))
    return f32(dd * inv[:, None])


def exact_z(x, fn, eps=1e-6):
    xf = x.astype(np.float64)
    return (xf @ fn.astype(np.float64).T) / np.sqrt((xf * xf).mean(-1, keepdims=True) + eps)


def trunc10_z(x, fn, eps=1e-6):
    """The z of the reference mix (`xf @ fn.T` on the tensor units, then the rms factor in f64)."""
    xf = x.astype(np.float64)
    inv = 1.0 / np.sqrt((xf * xf).mean(-1, keepdims=True) + eps)
    return (trunc10(x).astype(np.float64) @ trunc10(fn).astype(np.float64).T) * inv


def weight_as(w32, dtype: str):
    """The f32 values of a weight stored as float32 / float16 / bfloat16."""
    if dtype == "float32":
        return f32(w32)
    if dtype == "float16":
        return f32(np.asarray(w32, np.float16))
    import ml_dtypes
    return f32(np.asarray(w32, ml_dtypes.bfloat16))
