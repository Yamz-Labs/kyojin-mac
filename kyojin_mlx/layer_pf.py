# SPDX-License-Identifier: Apache-2.0
"""Fused decoder-layer seams for the dense prefill (Qwen3-Next / Qwen3.5 `DecoderLayer`, EXL3 linears).

Stock layer:  h = x + attn(norm1(x));  out = h + mlp(norm2(h)).  With EXL3 linears every seam is a chain of memory-bound passes over [rows, hidden]:
    finalize(out_proj) -> add -> rms_norm -> rotation(gate), rotation(up)          (seam 1)
    finalize(down)     -> add                                                       (seam 2)
Seam 1 is ONE launch here: a threadgroup per row, one simdgroup per 128-block (the rows of `mx.fast.rms_norm` single-row kernel use the same 4-elements-per-lane layout, so the
reduction order is the stock one), reading the matmul product of out_proj, the residual, and writing h and the rotated inputs of gate and up. Seam 2 is one finalize + add launch.
Each piece keeps the arithmetic and roundings of the separate steps (finalize result rounded to fp16, fp16 add, rms_norm: fp32 sum of squares over 4-element lanes, simd_sum,
threadgroup sum, precise rsqrt, w * half(x * inv), rotation as `kernels.rot`): bit-identical logits.
The out_proj call of the attention module returns its matmul product un-finalized (`Exl3Linear._defer`).
Switch: module attribute ENABLED (env KYOJIN_LAYER_PF=0 turns it off); needs mlp_pf (the fused MLP middle)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import kernels as KN
from . import metal_src as S
from . import mlp_pf as MP
from . import tuning as TN

ENABLED = (os.environ["KYOJIN_LAYER_PF"] != "0") if "KYOJIN_LAYER_PF" in os.environ else None       # None = tuning table ("prefill".layer_pf, default on)
MIN_ROWS = int(os.environ["KYOJIN_LAYER_PF_MIN_ROWS"]) if "KYOJIN_LAYER_PF_MIN_ROWS" in os.environ else None
# Largest chunk with the one-kernel seam 1: above it the stock seams win on the Air (4B: 0.97 vs 0.91 at 2048 rows). Cause (27/10 ablation): the 3-output kernel (fin + add + rms_norm + both rotations in
# one launch) loses in the graph although it is 2x faster alone; fin + add fused alone (seam 2) and the same work split in two launches (seam1_split) both match the stock seams there.
MAX_ROWS = int(os.environ["KYOJIN_LAYER_PF_MAX_ROWS"]) if "KYOJIN_LAYER_PF_MAX_ROWS" in os.environ else None
SPLIT_MAX_ROWS = int(os.environ["KYOJIN_LAYER_PF_SPLIT_MAX_ROWS"]) if "KYOJIN_LAYER_PF_SPLIT_MAX_ROWS" in os.environ else None


def enabled() -> bool:
    return bool(TN.pf("layer_pf", 1)) if ENABLED is None else ENABLED


def min_rows() -> int:
    return int(TN.pf("layer_min_rows", 64)) if MIN_ROWS is None else MIN_ROWS


def max_rows() -> int:
    return int(TN.pf("layer_max_rows", 1024)) if MAX_ROWS is None else MAX_ROWS


def split_max_rows() -> int:
    return int(TN.pf("layer_split_max_rows", 0)) if SPLIT_MAX_ROWS is None else SPLIT_MAX_ROWS


_FIN = r"""
    {
        const float4 acc = float4(*((const device half4*)(PTR + o)));
        float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
        const float4 sv = float4(*((const device half4*)(SVH + s * 128 + lane * 4)));
        DST = half4(float4(v0, v1, v2, v3) * RSQRT * sv);
    }
"""

_ROT = r"""
    {
        const float4 sv = float4(*((const device half4*)(SUH + s * 128 + lane * 4)));
        float v0 = float(nh[0]) * sv.x, v1 = float(nh[1]) * sv.y, v2 = float(nh[2]) * sv.z, v3 = float(nh[3]) * sv.w;
BUTTERFLY
        STORE
    }
"""

_SEAM1 = r"""
    const uint s = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint lid = thread_position_in_threadgroup.x;
    const uint r = threadgroup_position_in_grid.x;
    const uint K = params[0];
    const float eps = epsv[0];
    const ulong o = ulong(r) * K + s * 128 + lane * 4;
    half4 y;
FIN
    const half4 hh = *((const device half4*)(x + o)) + y;
    *((device half4*)(h + o)) = hh;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    float thread_x[4];
    float acc = 0;
    for (int i = 0; i < 4; i++) { thread_x[i] = float(hh[i]); acc += thread_x[i] * thread_x[i]; }
    acc = simd_sum(acc);
    if (s == 0) local_sums[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) local_sums[s] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / K + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half nh[4];
    for (int i = 0; i < 4; i++) nh[i] = nw[s * 128 + lane * 4 + i] * static_cast<half>(thread_x[i] * local_inv_mean[0]);
ROT_G
ROT_U
"""

_SEAM2 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y;
    const uint N = params[1];
    if (blk * 128 >= N) return;
    const ulong o = ulong(r) * N + blk * 128 + lane * 4;
    float4 acc = float4(*((const device half4*)(part + o)));
    float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
    const float4 sv = float4(*((const device half4*)(svh + blk * 128 + lane * 4)));
    const half4 yv = half4(float4(v0, v1, v2, v3) * RSQRT * sv);
    *((device half4*)(out + o)) = *((const device half4*)(res + o)) + yv;
"""

_ROTPAIR = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint s = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y;
    const uint K = params[0];
    if (s * 128 >= K) return;
    const ulong o = ulong(r) * K + s * 128 + lane * 4;
    const half4 nh4 = *((const device half4*)(n + o));
    const half nh[4] = {nh4.x, nh4.y, nh4.z, nh4.w};
ROT_G
ROT_U
"""

_cache: dict = {}
_params: dict = {}


def _rot_macro(suh, out, perm):
    store = (S.perm_store(f"{out} + ulong(r) * K + s * 128", "float4(v0, v1, v2, v3) * RSQRT", True) if perm
             else f"*((device half4*)({out} + o)) = half4(float4(v0, v1, v2, v3) * RSQRT);")
    return _ROT.replace("SUH", suh).replace("STORE", store).replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD)


def _fin_macro() -> str:
    return _FIN.replace("PTR", "inner").replace("SVH", "svh").replace("DST", "y").replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD)


def _seam1_source(perm_g: bool, perm_u: bool) -> str:
    return _SEAM1.replace("FIN", _fin_macro()).replace("ROT_G", _rot_macro("suhg", "xhg", perm_g)).replace("ROT_U", _rot_macro("suhu", "xhu", perm_u))


def _seam1n_source() -> str:
    """seam 1 without the rotations: stores the normalized rows `n` instead."""
    return _SEAM1.replace("FIN", _fin_macro()).replace("ROT_G", "*((device half4*)(n + o)) = half4(nh[0], nh[1], nh[2], nh[3]);").replace("ROT_U", "")


def seam1_split(inner: mx.array, svh: mx.array, x: mx.array, nw: mx.array, eps: float, suh_g: mx.array, suh_u: mx.array, perm_g: bool, perm_u: bool):
    """`seam1` in two launches (fin + add + rms_norm -> h and the normalized rows, then both rotations of those rows): the same bits."""
    rows, K = inner.shape
    k1 = _cache.get("split1")
    if k1 is None:
        k1 = _cache["split1"] = mx.fast.metal_kernel(name="kyojin_seam1n", header=S.HEADER, input_names=["inner", "svh", "x", "nw", "epsv", "params"], output_names=["h", "n"], source=_seam1n_source())
    key = ("split2", perm_g, perm_u)
    k2 = _cache.get(key)
    if k2 is None:
        k2 = _cache[key] = mx.fast.metal_kernel(name=f"kyojin_rotpair_{int(perm_g)}{int(perm_u)}", header=S.HEADER, input_names=["n", "suhg", "suhu", "params"], output_names=["xhg", "xhu"],
                                                source=_ROTPAIR.replace("ROT_G", _rot_macro("suhg", "xhg", perm_g)).replace("ROT_U", _rot_macro("suhu", "xhu", perm_u)))
    ent = _params.get((K, eps))
    if ent is None:
        ent = _params[(K, eps)] = (mx.array([K], dtype=mx.uint32), mx.array([eps], dtype=mx.float32))
    h, n = k1(inputs=[inner, svh, x, nw, ent[1], ent[0]], grid=(rows * (K // 4), 1, 1), threadgroup=(K // 4, 1, 1), output_shapes=[(rows, K)] * 2, output_dtypes=[mx.float16] * 2)
    grid, tg = S.had_grid(K // 128, rows)
    xhg, xhu = k2(inputs=[n, suh_g, suh_u, ent[0]], grid=grid, threadgroup=tg, output_shapes=[(rows, K)] * 2, output_dtypes=[mx.float16] * 2)
    return h, xhg, xhu


def seam1(inner: mx.array, svh: mx.array, x: mx.array, nw: mx.array, eps: float, suh_g: mx.array, suh_u: mx.array, perm_g: bool, perm_u: bool):
    """inner [rows, K] fp16: matmul product of out_proj (before its finalize); x [rows, K] fp16 residual; nw fp16 norm weight. Returns (h, xh_gate, xh_up), fp16 [rows, K]."""
    rows, K = inner.shape
    key = (perm_g, perm_u)
    kern = _cache.get(key)
    if kern is None:
        kern = _cache[key] = mx.fast.metal_kernel(name=f"kyojin_seam1_{int(perm_g)}{int(perm_u)}", header=S.HEADER, input_names=["inner", "svh", "x", "nw", "epsv", "suhg", "suhu", "params"],
                                                  output_names=["h", "xhg", "xhu"], source=_seam1_source(perm_g, perm_u))
    ent = _params.get((K, eps))
    if ent is None:
        ent = _params[(K, eps)] = (mx.array([K], dtype=mx.uint32), mx.array([eps], dtype=mx.float32))
    return kern(inputs=[inner, svh, x, nw, ent[1], suh_g, suh_u, ent[0]], grid=(rows * (K // 4), 1, 1), threadgroup=(K // 4, 1, 1),
                output_shapes=[(rows, K)] * 3, output_dtypes=[mx.float16] * 3)


def seam2(inner: mx.array, svh: mx.array, res: mx.array) -> mx.array:
    """res + finalize(inner): inner [rows, N] fp16 matmul product, res [rows, N] fp16."""
    rows, N = inner.shape
    kern = _cache.get("seam2")
    if kern is None:
        kern = _cache["seam2"] = mx.fast.metal_kernel(name="kyojin_seam2", header=S.HEADER, input_names=["part", "svh", "res", "params"], output_names=["out"],
                                                      source=_SEAM2.replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD))
    prm = _params.get(("s2", N))
    if prm is None:
        prm = _params[("s2", N)] = mx.array([1, N], dtype=mx.uint32)
    grid, tg = S.had_grid(N // 128, rows)
    return kern(inputs=[inner, svh, res, prm], grid=grid, threadgroup=tg, output_shapes=[(rows, N)], output_dtypes=[mx.float16])[0]


_stock = None


def _layer_call(self, x, mask=None, cache=None):
    if enabled() and MP.enabled() and x.ndim == 3 and x.shape[0] == 1 and min_rows() <= x.shape[1] <= max(max_rows(), split_max_rows()) and x.dtype == mx.float16:
        from .exl3_linear import Exl3Linear
        mlp = self.mlp
        g, u, d = getattr(mlp, "gate_proj", None), getattr(mlp, "up_proj", None), getattr(mlp, "down_proj", None)
        attn = self.linear_attn if self.is_linear else self.self_attn
        ol = getattr(attn, "out_proj", None) or getattr(attn, "o_proj", None)
        rows, K = x.shape[1], x.shape[2]
        nw = self.post_attention_layernorm.weight
        if (type(g) is Exl3Linear and type(u) is Exl3Linear and type(d) is Exl3Linear and type(ol) is Exl3Linear and nw.dtype == mx.float16 and K % 128 == 0 and K // 4 <= 1024
                and ol.out_features == K and d.out_features == K and g.in_features == K and u.in_features == K and g.out_features == u.out_features == d.in_features and g.out_features % 128 == 0):
            mo, mg, mu, md = ol.dense_mode(rows), g.dense_mode(rows), u.dense_mode(rows), d.dense_mode(rows)
            if mo and mg and mu and md:
                ol._defer = True
                try:
                    r_inner = attn(self.input_layernorm(x), mask, cache)
                finally:
                    ol._defer = False
                h, xhg, xhu = (seam1 if rows <= max_rows() else seam1_split)(r_inner.reshape(rows, K), ol._svh, x.reshape(rows, K), nw, float(self.post_attention_layernorm.eps),
                                    g._suh if g._suh.dtype == mx.float16 else g._suh.astype(mx.float16), u._suh if u._suh.dtype == mx.float16 else u._suh.astype(mx.float16),
                                    mg == "sh", mu == "sh")
                d_inner = MP.mlp_inner(mlp, xhg, xhu, rows)
                return seam2(d_inner.reshape(rows, K), d._svh, h).reshape(1, rows, K)
    return _stock(self, x, mask, cache)


def install():
    """Replace the `__call__` of mlx_lm's Qwen3.5 DecoderLayer (dense MLP) by the fused-seam version for prefill chunks; everything else runs the stock call."""
    global _stock
    from mlx_lm.models import qwen3_5 as Q5
    if _stock is None:
        _stock = Q5.DecoderLayer.__call__
        Q5.DecoderLayer.__call__ = _layer_call
