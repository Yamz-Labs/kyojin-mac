# SPDX-License-Identifier: Apache-2.0
"""DS4.1 mHC mix coefficients for prompt rows: mixes = rms_norm(h.flatten(-2)) @ hc_fn.T of every token, then the Sinkhorn split.

The stock `hc_mix_coeffs` casts the [T, 4 * D] fp16 residual stream to fp32, normalises it and runs an fp32 matmul against a 24-row weight (4 passes over
T * 4 * D * 4 bytes: 9.5 ms per call at T = 2048, two calls per block). Here one NAX kernel reads the fp16 stream once: the 24 dot products on the tensor units
with the weight split in two fp16 halves (hi + lo / 4096, 22 bits, two accumulators: the fp32 product of an fp16 input with the fp32 weight to 1e-7), K split over the 4 simdgroups of
a threadgroup; a second small kernel gives the row scales rsqrt(mean(h^2) + eps) in fp32. The Sinkhorn split is the stock kernel."""
from __future__ import annotations

import functools
import os

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import metal_nax as NX

ENABLED = os.environ.get("KYOJIN_MIX_PF", "auto")           # "1" force, "0" off, "auto" = on when the GPU has neural accelerators
MIN_ROWS = int(os.environ.get("KYOJIN_MIX_PF_MIN_ROWS", "64"))

_GEMM = r"""
    const uint sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint M = params[0], K = params[1];
    const uint mg = threadgroup_position_in_grid.y * 32;
    threadgroup float part[4 * 32 * 32];
    constexpr auto desc = matmul2d_descriptor(32, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)x, dextents<int32_t, 2>(int(K), int(M)));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)whi, dextents<int32_t, 2>(int(K), 32));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tL((device half*)wlo, dextents<int32_t, 2>(int(K), 32));
    auto sA0 = tA.slice(0, int(mg)); auto sH0 = tH.slice(0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sH0), float>();
    auto cL = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sH0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) { cT[i] = 0.0f; cL[i] = 0.0f; }
    const uint kq = K / 4;
    for (uint k = sg * kq; k < (sg + 1) * kq; k += 64) {
        auto sA = tA.slice(int(k), int(mg));
        auto sH = tH.slice(int(k), 0);
        auto sL = tL.slice(int(k), 0);
        op.run(sA, sH, cT);
        op.run(sA, sL, cL);
    }
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
        const auto ix = cT.get_multidimensional_index(i);
        part[sg * 1024 + uint(ix[1]) * 32 + uint(ix[0])] = cT[i] + cL[i] * (1.0f / 4096.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 32 * 24; e += 128) {
        const uint r = e / 24, n = e % 24;
        if (mg + r < M) {
            const uint o = r * 32 + n;
            mixes[ulong(mg + r) * 24 + n] = (part[o] + part[1024 + o] + part[2048 + o] + part[3072 + o]) * rs[mg + r];
        }
    }
"""

_RS = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint M = params[0], K = params[1];
    if (row >= M) return;
    const device half4* xp = (const device half4*)(x + ulong(row) * K);
    float s = 0.0f;
    for (uint i = lane; i < K / 4; i += 32) { const float4 v = float4(xp[i]); s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
    s = simd_sum(s);
    if (lane == 0) rs[row] = metal::precise::rsqrt(s / float(K) + EPS);
"""


@functools.lru_cache(maxsize=None)
def _kernels(eps: float):
    k1 = mx.fast.metal_kernel(name="kyojin_mix_rs", input_names=["x", "params"], output_names=["rs"], source=_RS.replace("EPS", repr(float(eps)) + "f"))
    k2 = mx.fast.metal_kernel(name="kyojin_mix_gemm", header=NX.HEADER_NAX, input_names=["x", "whi", "wlo", "rs", "params"], output_names=["mixes"], source=_GEMM)
    return k1, k2


_split: dict = {}               # id(hc_fn) -> (hc_fn, hi, lo) padded to 32 rows


def split_weight(fn: mx.array):
    ent = _split.get(id(fn))
    if ent is None or ent[0] is not fn:
        w = mx.concatenate([fn.astype(mx.float32), mx.zeros((32 - fn.shape[0], fn.shape[1]), dtype=mx.float32)], axis=0)
        hi = w.astype(mx.float16)
        lo = ((w - hi.astype(mx.float32)) * 4096.0).astype(mx.float16)       # scaled: the residual of a 0.02 weight (1e-5) is an fp16 subnormal
        mx.eval(hi, lo)
        ent = _split[id(fn)] = (fn, hi, lo)
    return ent[1], ent[2]


def available() -> bool:
    return NX.site_on(ENABLED)


def mixes(x: mx.array, hc_fn: mx.array, norm_eps: float) -> mx.array:
    """x [1, T, 4, D] fp16, hc_fn [24, 4 * D] fp32 -> [1, T, 24] fp32 = rms_norm(x.flatten(-2)) @ hc_fn.T."""
    _, T, H, D = x.shape
    K = H * D
    hi, lo = split_weight(hc_fn)
    k1, k2 = _kernels(norm_eps)
    p = mx.array([T, K], dtype=mx.uint32)
    xf = x.reshape(T, K)
    rs = k1(inputs=[xf, p], grid=(-(-T // 4) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(T,)], output_dtypes=[mx.float32])[0]
    return k2(inputs=[xf, hi, lo, rs, p], grid=(128, -(-T // 32), 1), threadgroup=(128, 1, 1), output_shapes=[(T, 24)], output_dtypes=[mx.float32])[0].reshape(1, T, 24)


def supported(x: mx.array, hc_fn: mx.array, hc_mult: int) -> bool:
    return (x.ndim == 4 and x.shape[0] == 1 and x.shape[1] >= MIN_ROWS and x.shape[2] == KG.HC and hc_mult == KG.HC and x.dtype == mx.float16 and hc_fn.dtype == mx.float32
            and hc_fn.shape == (KG.MIXES, KG.HC * x.shape[3]) and (KG.HC * x.shape[3]) % 256 == 0 and available())


def _impl_mix(orig):
    """Replacement of `hc_mix_coeffs(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_eps)` for prompt rows."""
    import importlib
    HC = importlib.import_module("mlx_vlm.models.deepseek_v4.hyper_connection")

    def hc_mix_coeffs(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_eps):
        if not supported(x, hc_fn, hc_mult):
            return orig(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps, norm_eps)
        return HC.hc_split_sinkhorn(mixes(x, hc_fn, norm_eps), hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
    return hc_mix_coeffs
