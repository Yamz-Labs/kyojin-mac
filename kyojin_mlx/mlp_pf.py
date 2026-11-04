# SPDX-License-Identifier: Apache-2.0
"""Fused dense-MLP middle for prefill: finalize of gate and up, swiglu, input rotation of down in ONE launch.

Stock prefill MLP (Qwen3-Next / Qwen3.5 `MLP`: down(swiglu(gate(x), up(x)))) with three EXL3 linears costs, between the two matmuls, five memory-bound passes over [rows, F]:
fin(gate) r+w, fin(up) r+w, swiglu r2+w, rot(down) r+w. All of them work on one 128-wide block of F at a time (the Hadamard blocks of the finalize and of the rotation
line up with the columns of the activation), so one simdgroup per (128-block, row) does the whole chain on registers: it reads the two fp16 matmul products (before the
finalize) and writes the rotated fp16 input of `down` (in the k permutation of the fused NAX kernels when `perm`). Same arithmetic, same roundings (fin result rounded to the
output dtype, swiglu in that dtype like the compiled mlx_lm `swiglu`, rotation as `kernels.rot`): bit-identical to the three separate steps.
Switch: module attribute ENABLED (env KYOJIN_MLP_PF=0 turns it off); None = the tuning table entry "prefill" (tuning.pf: mlp_pf, mlp_min_rows)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import kernels as KN
from . import metal_src as S
from . import tuning as TN

ENABLED = (os.environ["KYOJIN_MLP_PF"] != "0") if "KYOJIN_MLP_PF" in os.environ else None          # None = tuning table, default on
MIN_ROWS = int(os.environ["KYOJIN_MLP_PF_MIN_ROWS"]) if "KYOJIN_MLP_PF_MIN_ROWS" in os.environ else None


def enabled() -> bool:
    return bool(TN.pf("mlp_pf", 1)) if ENABLED is None else ENABLED


def min_rows() -> int:
    return int(TN.pf("mlp_min_rows", 64)) if MIN_ROWS is None else MIN_ROWS


_FIN = r"""
    {
        const float4 acc = float4(*((const device half4*)(PTR + o)));
        float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
        const float4 sv = float4(*((const device half4*)(SVH + blk * 128 + lane * 4)));
        DST = half4(float4(v0, v1, v2, v3) * RSQRT * sv);
    }
"""

_BODY = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y;
    const uint F = params[0];
    if (blk * 128 >= F) return;
    const ulong o = ulong(r) * F + blk * 128 + lane * 4;
    half4 gh, uh;
FIN_G
FIN_U
    half ah[4];
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        SWIGLU
    }
    const float4 xv = float4(half4(ah[0], ah[1], ah[2], ah[3]));
    const float4 sv = float4(*((const device half4*)(suh + blk * 128 + lane * 4)));
    float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
BUTTERFLY
    STORE
"""

_SW = r"""const half g = gh[i], u = uh[i];
        const float gf = float(g);
        const float ef = 1.0f / (1.0f + metal::precise::exp(metal::abs(gf)));
        const half sg = half((gf < 0.0f) ? ef : 1.0f - ef);          // MLX sigmoid of a half: evaluated in float, rounded once
        ah[i] = (g * sg) * u;"""

_cache: dict = {}
_params: dict = {}


def _source(perm: bool) -> str:
    fin = lambda ptr, svh, dst: _FIN.replace("PTR", ptr).replace("SVH", svh).replace("DST", dst).replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD)
    store = (S.perm_store("xh + ulong(r) * F + blk * 128", "float4(v0, v1, v2, v3) * RSQRT", True) if perm else "*((device half4*)(xh + o)) = half4(float4(v0, v1, v2, v3) * RSQRT);")
    return (_BODY.replace("FIN_G", fin("gi", "svg", "gh")).replace("FIN_U", fin("ui", "svu", "uh")).replace("SWIGLU", _SW)
            .replace("BUTTERFLY", S._BUTTERFLY).replace("STORE", store).replace("RSQRT", S.RSQRT_HAD))


def swiglu_rot(gi: mx.array, ui: mx.array, svg: mx.array, svu: mx.array, suh_d: mx.array, perm: bool) -> mx.array:
    """gi, ui: fp16 [rows, F] matmul products of gate and up (before their finalize); svg, svu: their fp16 svh; suh_d: fp16 su of down. Returns the rotated input of down, fp16 [rows, F]."""
    rows, F = gi.shape
    key = perm
    ent = _cache.get(key)
    if ent is None:
        ent = _cache[key] = mx.fast.metal_kernel(name=f"kyojin_mlp_pf_{int(perm)}", header=S.HEADER, input_names=["gi", "ui", "svg", "svu", "suh", "params"],
                                                 output_names=["xh"], source=_source(perm))
    grid, tg = S.had_grid(F // 128, rows)
    prm = _params.get(F)
    if prm is None:
        prm = _params[F] = mx.array([F], dtype=mx.uint32)
    return ent(inputs=[gi, ui, svg, svu, suh_d, prm], grid=grid, threadgroup=tg, output_shapes=[(rows, F)], output_dtypes=[mx.float16])[0]



def mlp_inner(mlp, xhg, xhu, rows):
    """MLP middle from the rotated inputs of gate and up (layer_pf seam 1): returns the matmul product of down before its finalize, [1, rows, n]."""
    g, u, d = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    gi = g._dense(None, xh=xhg, nofin=True).reshape(rows, -1)
    ui = u._dense(None, xh=xhu, nofin=True).reshape(rows, -1)
    xh = swiglu_rot(gi, ui, g._svh, u._svh, d._suh if d._suh.dtype == mx.float16 else d._suh.astype(mx.float16), d.dense_mode(rows) == "sh")
    return d._dense(None, xh=xh, nofin=True)


_stock = None


def _mlp_call(self, x):
    g, u, d = self.gate_proj, self.up_proj, self.down_proj
    if enabled() and x.ndim == 3 and x.shape[0] == 1 and x.shape[1] >= min_rows():
        from .exl3_linear import Exl3Linear
        if type(g) is Exl3Linear and type(u) is Exl3Linear and type(d) is Exl3Linear and g.out_features == u.out_features == d.in_features and g.out_features % 128 == 0 \
                and g.in_features == u.in_features and x.dtype in (mx.float16, mx.float32):
            rows = x.shape[1]
            mg, mu, md = g.dense_mode(rows), u.dense_mode(rows), d.dense_mode(rows)
            if mg and mu and md:
                x2 = x.reshape(rows, -1)
                gi = g._dense(x2, nofin=True).reshape(rows, -1)
                ui = u._dense(x2, nofin=True).reshape(rows, -1)
                xh = swiglu_rot(gi, ui, g._svh, u._svh, d._suh if d._suh.dtype == mx.float16 else d._suh.astype(mx.float16), md == "sh")
                return d._dense(None, xh=xh).reshape(1, rows, d.out_features)
    return _stock(self, x)


def install():
    """Replace the `__call__` of mlx_lm's Qwen3-Next / Qwen3.5 MLP (down(swiglu(gate, up))) by the fused prefill middle; other row counts and layer kinds run the stock call."""
    global _stock
    from mlx_lm.models import qwen3_next as Q
    if _stock is None:
        _stock = Q.Qwen3NextMLP.__call__
        Q.Qwen3NextMLP.__call__ = _mlp_call
