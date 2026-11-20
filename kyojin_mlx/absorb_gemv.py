# SPDX-License-Identifier: Apache-2.0
"""Flag absorb_gemv (KYOJIN_ABSORB_GEMV, default 0): the MLA absorb products of a 1 .. 8 row pass (qa = q_nope @ k_b^T, o = (P @ latent) @ v_b^T) as ONE
per-head GEMV launch over a 16-bit or Q8_0 copy of k_b / v_b, instead of the stock fp32 matmul over the stored fp32 weights.

Hy4 keeps k_b [nh, kv_lora, nope] and v_b [nh, v, kv_lora] as fp32 (the GGUF tensors are Q8_0 and are dequantised at load): ~2.1 GB per step (30 layers of 64 heads, 12 of 32) through MLX's
batched fp32 gemv (n10: the gemv_float32_bm4 bucket, 2.1 ms of 21.8 ms GPU busy at ~1 TB/s: bytes-bound). The copies here move half (mode 1, fp16) or a quarter (mode 2, Q8_0: int8 + one fp16
scale per 32 along K) of those bytes. Accumulation is fp32; x stays in its own dtype (fp32 in Hy4), the output is fp32 (x fp32) or the x dtype.

Mode 2 rebuilds the Q8_0 blocks from the dequantised fp32 weights (d = amax / 127, q = round(w / d)); for a tensor that really was Q8_0 this returns the stored
blocks (d fp16-exact, max |q| = 127). `prepare` records the largest relative rebuild error per layer (`REPORT`)."""
from __future__ import annotations

import os

import mlx.core as mx

MODE = int(os.environ.get("KYOJIN_ABSORB_GEMV", "0"))          # flag absorb_gemv: 0 off, 1 fp16 weights, 2 Q8_0 weights
MAX_ROWS = 8
SGS = 4                                                         # simdgroups per threadgroup
RPS = 4                                                         # output rows per simdgroup (they share the simdgroup's x registers)
REPORT: dict = {"layers": 0, "q8_max_rel_err": 0.0, "bytes_f16": 0, "bytes_q8": 0}

_HEAD = r"""
    const uint h = threadgroup_position_in_grid.y;
    const uint r0 = threadgroup_position_in_grid.z * MR;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint n0 = (threadgroup_position_in_grid.x * SGS + sg) * RPS;
    if (n0 >= N) return;
"""
_F16 = _HEAD + r"""
    constexpr uint I = (K + 63) / 64;
    float2 xv[MR][I];
    for (uint r = 0; r < MR; ++r) {
        const uint rr = min(r0 + r, uint(B) - 1);
        auto xp = x + (ulong(h) * B + rr) * K;
        for (uint i = 0; i < I; ++i) {
            const uint e = 64 * i + 2 * lane;
            xv[r][i] = e < K ? float2(float(xp[e]), float(xp[e + 1])) : float2(0.0f);
        }
    }
    for (uint j = 0; j < RPS; ++j) {
        const uint n = n0 + j;
        if (n >= N) break;
        auto wp = (device const half2*)(w + (ulong(h) * N + n) * K);
        float acc[MR];
        for (uint r = 0; r < MR; ++r) acc[r] = 0.0f;
        for (uint i = 0; i < I; ++i) {
            const uint e = 64 * i + 2 * lane;
            if (e < K) {
                const float2 wv = float2(wp[e / 2]);
                for (uint r = 0; r < MR; ++r) acc[r] = fma(wv.x, xv[r][i].x, fma(wv.y, xv[r][i].y, acc[r]));
            }
        }
        for (uint r = 0; r < MR; ++r) {
            const float t = simd_sum(acc[r]);
            if (lane == 0 && r0 + r < uint(B)) y[(ulong(h) * B + r0 + r) * N + n] = static_cast<YT>(t);
        }
    }
"""
_Q8 = _HEAD + r"""
    constexpr uint I = (K + 127) / 128;
    float4 xv[MR][I];
    for (uint r = 0; r < MR; ++r) {
        const uint rr = min(r0 + r, uint(B) - 1);
        auto xp = x + (ulong(h) * B + rr) * K;
        for (uint i = 0; i < I; ++i) {
            const uint e = 128 * i + 4 * lane;
            xv[r][i] = e < K ? float4(float(xp[e]), float(xp[e + 1]), float(xp[e + 2]), float(xp[e + 3])) : float4(0.0f);
        }
    }
    for (uint j = 0; j < RPS; ++j) {
        const uint n = n0 + j;
        if (n >= N) break;
        auto wp = (device const char4*)(wq + (ulong(h) * N + n) * K);
        auto sp = ws + (ulong(h) * N + n) * (K / 32);
        float acc[MR];
        for (uint r = 0; r < MR; ++r) acc[r] = 0.0f;
        for (uint i = 0; i < I; ++i) {
            const uint e = 128 * i + 4 * lane;
            if (e < K) {
                const float4 q = float4(wp[e / 4]);
                const float s = float(sp[e / 32]);
                for (uint r = 0; r < MR; ++r) acc[r] = fma(s, dot(q, xv[r][i]), acc[r]);
            }
        }
        for (uint r = 0; r < MR; ++r) {
            const float t = simd_sum(acc[r]);
            if (lane == 0 && r0 + r < uint(B)) y[(ulong(h) * B + r0 + r) * N + n] = static_cast<YT>(t);
        }
    }
"""
_k: dict = {}


def _kern(q8: bool):
    k = _k.get(q8)
    if k is None:
        k = _k[q8] = mx.fast.metal_kernel(name="kyojin_absorb_q8" if q8 else "kyojin_absorb_f16", input_names=["x", "wq", "ws"] if q8 else ["x", "w"], output_names=["y"], source=_Q8 if q8 else _F16)
    return k


def pack(w: mx.array, mode: int):
    """w fp32 / fp16 [H, N, K] -> the arrays `apply` reads: mode 1 (w16,), mode 2 (int8 [H, N, K], fp16 scales [H, N, K / 32]). Also returns the largest relative rebuild error (mode 2)."""
    H, N, K = w.shape
    if mode == 1:
        w16 = w if w.dtype == mx.float16 else w.astype(mx.float16)
        mx.eval(w16)
        return (w16,), 0.0
    assert K % 32 == 0, K
    wf = w.astype(mx.float32).reshape(H, N, K // 32, 32)
    amax = mx.max(mx.abs(wf), axis=-1, keepdims=True)
    d = (amax / 127.0).astype(mx.float16)
    d32 = d.astype(mx.float32)
    q = mx.clip(mx.round(wf / mx.where(d32 == 0, 1.0, d32)), -127, 127)
    err = mx.max(mx.abs(q * d32 - wf)) / mx.maximum(mx.max(amax), 1e-30)
    wq, ws = q.astype(mx.int8).reshape(H, N, K), d.reshape(H, N, K // 32)
    mx.eval(wq, ws, err)
    return (wq, ws), float(err.item())


def apply(x: mx.array, pk: tuple, out_dtype=None) -> mx.array:
    """x [H, B, K] (fp32 / fp16, 1 <= B <= 8) against the packed per-head weights -> [H, B, N] (fp32 accumulate)."""
    H, B, K = x.shape
    q8 = len(pk) == 2
    N = pk[0].shape[1]
    assert 1 <= B <= MAX_ROWS and pk[0].shape[0] == H and pk[0].shape[2] == K and K % (32 if q8 else 2) == 0 and x.dtype in (mx.float16, mx.float32)
    mr = min(B, 4)
    tgx = (N + SGS * RPS - 1) // (SGS * RPS)
    yt = out_dtype or x.dtype
    return _kern(q8)(inputs=[x, *pk], template=[("K", K), ("N", N), ("B", B), ("MR", mr), ("SGS", SGS), ("RPS", RPS), ("YT", yt)],
                     grid=(tgx * 32 * SGS, H, (B + mr - 1) // mr), threadgroup=(32 * SGS, 1, 1), output_shapes=[(H, B, N)], output_dtypes=[yt])[0]


def prepare(model) -> None:
    """Eager, host side (Model.__call__): the packed k_b / v_b copy of the active mode on every MLA layer, once per mode (outside the module's parameters, constants for the compiled layers)."""
    if MODE not in (1, 2):
        return
    for layer in model.layers:
        at = getattr(layer, "self_attn", None)
        if at is None or not hasattr(at, "k_b") or at.k_b.ndim != 3:
            continue
        d = at.__dict__.setdefault("_ag", {})
        if MODE in d:
            continue
        k16, v16 = at.__dict__.get("_kb16"), at.__dict__.get("_vb16")
        shared = MODE == 1 and k16 is not None and v16 is not None          # attn_fp16.prepare (KYOJIN_HY4_PF_FP16, in the lane flags) already holds the same fp16 cast: ONE resident copy
        (kb, e1), (vb, e2) = pack(k16 if shared else at.k_b, MODE), pack(v16 if shared else at.v_b, MODE)
        d[MODE] = {"k": kb, "v": vb}
        REPORT["layers"] += 1
        REPORT["q8_max_rel_err"] = max(REPORT["q8_max_rel_err"], e1, e2)
        REPORT["shared_f16_layers"] = REPORT.get("shared_f16_layers", 0) + shared
        if not shared:
            REPORT["bytes_f16" if MODE == 1 else "bytes_q8"] += sum(a.nbytes for a in kb + vb)


def active(at, x: mx.array, w: mx.array, kind: str):
    """The packed weights of this layer (kind "k" / "v") when the product x [1, nh, L, K] @ w^T (w = the stored fp32 k_b / v_b) runs through the flag's kernel, else None."""
    if MODE not in (1, 2) or w.dtype != mx.float32 or x.ndim != 4 or x.shape[0] != 1 or not 1 <= x.shape[2] <= MAX_ROWS or x.dtype not in (mx.float16, mx.float32):
        return None
    d = at.__dict__.get("_ag", {}).get(MODE)
    if d is None or not (mx.default_device() == mx.gpu):
        return None
    return d[kind]
