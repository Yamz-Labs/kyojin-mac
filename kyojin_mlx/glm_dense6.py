# SPDX-License-Identifier: Apache-2.0
"""BP3-Q6: dense GLM-5.3 linears as symmetric 6-bit, one fp16 scale per 32 weights (6.5 bits per weight against 8.25 for the served MXFP8 = -21 % bytes).

Why a custom kernel: MLX's affine 6-bit qmv reads 362 GB/s on one token's dense linears (tools/g4q6_gemv_bench.py), the served FP8 G3 GEMV ~1000 GB/s.
Stored layout (the weights of one row, per chunk of 512 columns = 32 lanes x one unit of 16 weights): 96 uint32 words = 3 planes of 32 words (plane p, lane l at
word chunk * 96 + 32 * p + l), so every lane load of a simdgroup is one contiguous 128-byte line.
  A unit's weights are the 8 pairs (j, j + 8) j = 0..7; code c = q + 32 in 0..63 (q the signed value, the scale is s per 32 columns = 2 units).
  plane 0: pairs 0..3: nibble i (bits 4 i) = low 4 bits of the first weight of pair i, bits 16 + 4 i = of the second
  plane 1: pairs 4..7 likewise (i = pair - 4)
  plane 2: pair i: bits 2 i..2 i + 1 = high 2 bits of the first weight, bits 16 + 2 i.. = of the second
Decode: half2 bits (0x6400 | c) = 1024 + c exactly; minus 1056 = q. y = sum_groups s * sum q x.
"""
from __future__ import annotations

import functools

import numpy as np
import mlx.core as mx

from . import foreign_metal as FM, kernels as KN

QMAX = 31
PROBE = (0.0, 0.25, 0.5, -0.25)          # scale = amax / (31 + m): m > 0 clips the largest weights for a finer grid (squared-error minimum per group, 4 candidates)


def quantize(w: np.ndarray, probe=PROBE):
    """w float32 [n, k] (k % 512 == 0) -> (words uint32 [n, k / 512 * 96], scales float16 [n, k / 32], dequantised float32 [n, k])."""
    n, k = w.shape
    assert k % 512 == 0, k
    wb = w.reshape(n, k // 32, 32).astype(np.float32)
    amax = np.abs(wb).max(-1, keepdims=True)
    best_q = best_s = best_e = None
    for m in probe:
        s = (amax / (QMAX + m)).astype(np.float16).astype(np.float32)
        s = np.where(s == 0, np.float32(1.0), s)
        q = np.clip(np.rint(wb / s), -32, 31)
        e = ((wb - q * s) ** 2).sum(-1, keepdims=True)
        if best_q is None:
            best_q, best_s, best_e = q, s, e
        else:
            mk = e < best_e
            best_q, best_s, best_e = np.where(mk, q, best_q), np.where(mk, s, best_s), np.where(mk, e, best_e)
    q = best_q.reshape(n, k)
    return pack_codes((q + 32).astype(np.uint32)), best_s[..., 0].astype(np.float16), (best_q * best_s).reshape(n, k)


def pack_codes(c: np.ndarray) -> np.ndarray:
    """codes uint32 [n, k] (0..63) -> words uint32 [n, k / 512 * 96] in the layout above."""
    n, k = c.shape
    u = c.reshape(n, k // 512, 32, 2, 8)                       # [row, chunk, lane, half (j // 8), pair (j % 8)]
    lo, hi = u & 15, u >> 4
    sh_lo = (4 * (np.arange(8) % 4)).astype(np.uint32)
    sh_hi = (2 * np.arange(8)).astype(np.uint32)
    half_sh = np.array([0, 16], np.uint32)[:, None]
    w = np.zeros((n, k // 512, 3, 32), np.uint32)
    lw = (lo << (sh_lo + half_sh))                              # [.., 2, 8] shifted nibbles
    w[:, :, 0, :] = lw[..., :, 0:4].sum(axis=(-1, -2)).astype(np.uint32)
    w[:, :, 1, :] = lw[..., :, 4:8].sum(axis=(-1, -2)).astype(np.uint32)
    w[:, :, 2, :] = (hi << (sh_hi + half_sh)).sum(axis=(-1, -2)).astype(np.uint32)
    return np.ascontiguousarray(w.reshape(n, k // 512 * 96))


def dequant(words: np.ndarray, scales: np.ndarray, n: int, k: int) -> np.ndarray:
    """Reference decode of the stored arrays (float32 [n, k])."""
    w = np.asarray(words, np.uint32).reshape(n, k // 512, 3, 32)
    out = np.zeros((n, k // 512, 32, 2, 8), np.float32)
    for h in range(2):
        for p in range(8):
            lo = (w[:, :, p // 4, :] >> np.uint32(4 * (p % 4) + 16 * h)) & 15
            hi = (w[:, :, 2, :] >> np.uint32(2 * p + 16 * h)) & 3
            out[:, :, :, h, p] = (lo | (hi << 4)).astype(np.float32) - 32.0
    return out.reshape(n, k // 32, 32) * np.asarray(scales, np.float32)[..., None] if False else \
        (out.reshape(n, k // 32, 32) * np.asarray(scales, np.float32)[..., None]).reshape(n, k)


def source(r: int, nr: int, u: int, nsg: int, xmode: str = "h", limit: float = 0.0, v: int = 0) -> str:
    """G3 structure (foreign_metal.gemv8g3_source) on the 6-bit layout. Inputs x, qs (uint32 [N, KIN / 512 * 96]), mt (half [N, KIN / 32]), gs (float [N], 1.0)."""
    nv = nr * r
    HI_DECODE = ("""                const uint hw = ww.z >> (2u * i);
                const uint bits = (lw & 0x000F000Fu) | ((hw & 0x00030003u) << 4) | 0x64006400u;""" if v == 0 else """                const uint hq = i <= 2u ? (ww.z << (4u - 2u * i)) : (ww.z >> (2u * i - 4u));
                const uint bits = (lw & 0x000F000Fu) | (hq & 0x00300030u) | 0x64006400u;""")
    stage = {"h": "xs4[r_ * (uint(KC) / 8u) + c_] = X4[(r_ * uint(KIN) + kbase) / 8u + c_];",
             "f": "{ device const float4* XF = (device const float4*)x; const uint b_ = (r_ * uint(KIN) + kbase + c_ * 8u) / 4u; xs4[r_ * (uint(KC) / 8u) + c_] = g3_pack_f(XF[b_], XF[b_ + 1u]); }",
             "b": "xs4[r_ * (uint(KC) / 8u) + c_] = g3_pack_b(X4[(r_ * uint(KIN) + kbase) / 8u + c_]);",
             "glu": f"{{ const uint b_ = (r_ * 2u * uint(KIN) + kbase) / 8u + c_; xs4[r_ * (uint(KC) / 8u) + c_] = g3_glu(X4[b_], X4[b_ + uint(KIN) / 8u], {float(limit)!r}f); }}"}[xmode]
    return f"""
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint row0 = (threadgroup_position_in_grid.x * {nsg}u + sg) * {nr}u;
    constexpr uint NCH = uint(KIN) / 512u, RW = NCH * 96u, UPR = uint(KIN) / 32u;
    threadgroup uint4 xs4[{r} * uint(KC) / 8u];
    device const uint4* X4 = (device const uint4*)x;
    uint rn[{nr}];
    UNR for (uint j = 0; j < {nr}u; ++j) rn[j] = min(row0 + j, uint(N) - 1u);
    float acc[{nv}];
    UNR for (uint i = 0; i < {nv}u; ++i) acc[i] = 0.0f;
    for (uint ph = 0; ph < uint(NPH); ++ph) {{
      const uint kbase = ph * uint(KC), kcn = min(uint(KC), uint(KIN) - kbase);
      if (ph > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < {r}u * kcn / 8u; i += {nsg * 32}u) {{ const uint r_ = i / (kcn / 8u), c_ = i % (kcn / 8u); {stage} }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint nit = kcn / 512u;
      if (row0 < uint(N)) {{
        for (uint it0 = 0; it0 < nit; it0 += {u}u) {{
          uint3 w[{u}][{nr}]; half d[{u}][{nr}];
          UNR for (uint u_ = 0; u_ < {u}u; ++u_) {{
            const uint ch = kbase / 512u + min(it0 + u_, nit - 1u);
            UNR for (uint j = 0; j < {nr}u; ++j) {{
              device const uint* wp = qs + rn[j] * RW + ch * 96u + lane;
              w[u_][j] = uint3(wp[0], wp[32], wp[64]);
              d[u_][j] = mt[rn[j] * UPR + ch * 16u + (lane >> 1)];
            }}
          }}
          UNR for (uint u_ = 0; u_ < {u}u; ++u_) {{
            if (it0 + u_ >= nit) break;
            const uint kl = (it0 + u_) * 32u + lane;
            float xv[{r}][16];
            UNR for (uint r_ = 0; r_ < {r}u; ++r_) {{
              const uint4 xa = xs4[r_ * (uint(KC) / 8u) + 2u * kl], xb = xs4[r_ * (uint(KC) / 8u) + 2u * kl + 1u];
              const uint xw[8] = {{xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w}};
              UNR for (uint i = 0; i < 8u; ++i) {{ const half2 h = as_type<half2>(xw[i]); xv[r_][2u * i] = float(h.x); xv[r_][2u * i + 1u] = float(h.y); }}
            }}
            UNR for (uint j = 0; j < {nr}u; ++j) {{
              const uint3 ww = w[u_][j];
              float2 q[8];
              UNR for (uint i = 0; i < 8u; ++i) {{
                const uint lw = (i < 4u ? ww.x : ww.y) >> (4u * (i & 3u));
{HI_DECODE}
                q[i] = float2(as_type<half2>(bits) - half2(1056.0h));
              }}
              const float sc = float(d[u_][j]);
              UNR for (uint r_ = 0; r_ < {r}u; ++r_) {{
                float s = 0.0f;
                UNR for (uint i = 0; i < 8u; ++i) {{ s = fma(q[i].x, xv[r_][i], s); s = fma(q[i].y, xv[r_][i + 8u], s); }}
                acc[j * {r}u + r_] = fma(s, sc, acc[j * {r}u + r_]);
              }}
            }}
          }}
        }}
      }}
    }}
    if (row0 >= uint(N)) return;
    {FM._rs_epilogue(nv, r)}
"""


@functools.lru_cache(maxsize=None)
def _kern(r, nr, u, nsg, xmode="h", limit=0.0, v=0):
    return KN._kern(("fg6", r, nr, u, nsg, xmode, limit, v), name=f"kyojin_fg_g6{'v' + str(v) if v else ''}_r{r}_n{nr}_u{u}_s{nsg}" + xmode + (f"{limit:g}".replace(".", "p") if xmode == "glu" else ""),
                    input_names=["x", "qs", "mt", "gs"], output_names=["y"], header=FM.G3_HDR, source=source(r, nr, u, nsg, xmode, limit, v))


def gemv(x2, qs, mt, gs, n, k, limit=None, v=0):
    """x2 [R, k] half / float32 / bfloat16 (cast while staging) -> [R, n] half."""
    R = x2.shape[0]
    nr, nsg, u = FM.gemv8g3_config(R, n, k)
    kc = FM.gemv8g3_kc(R, k)
    xm = "glu" if limit is not None else {mx.float16: "h", mx.float32: "f", mx.bfloat16: "b"}[x2.dtype]
    return _kern(R, nr, u, nsg, xm, float(limit or 0.0), v)(
        inputs=[x2, qs, mt, gs], template=[("KIN", k), ("N", n), ("KC", kc), ("NPH", -(-k // kc))], grid=(-(-n // (nr * nsg)) * nsg * 32, 1, 1),
        threadgroup=(nsg * 32, 1, 1), output_shapes=[(R, n)], output_dtypes=[mx.float16])[0]


def quantize_mx(w: mx.array, probe=PROBE):
    """Same as quantize() on the device: w [n, k] (any float dtype, k % 512 == 0) -> (words uint32 [n, k / 512 * 96], scales float16 [n, k / 32]). Row chunks keep the
    float32 temporaries small (the caller passes a few thousand rows at a time)."""
    n, k = w.shape
    wb = w.astype(mx.float32).reshape(n, k // 32, 32)
    amax = mx.max(mx.abs(wb), axis=-1, keepdims=True)
    best_q = best_s = best_e = None
    for m in probe:
        s = (amax / (QMAX + m)).astype(mx.float16).astype(mx.float32)
        s = mx.where(s == 0, 1.0, s)
        q = mx.clip(mx.round(wb / s), -32, 31)
        e = mx.sum((wb - q * s) ** 2, axis=-1, keepdims=True)
        if best_q is None:
            best_q, best_s, best_e = q, s, e
        else:
            mk = e < best_e
            best_q, best_s, best_e = mx.where(mk, q, best_q), mx.where(mk, s, best_s), mx.where(mk, e, best_e)
    c = (best_q + 32.0).astype(mx.uint32).reshape(n, k // 512, 32, 2, 8)
    lo, hi = c & 15, c >> 4
    sh_lo = mx.array([0, 4, 8, 12, 0, 4, 8, 12], mx.uint32)
    sh_hi = mx.array([0, 2, 4, 6, 8, 10, 12, 14], mx.uint32)
    half = mx.array([0, 16], mx.uint32).reshape(2, 1)
    lw, hw = lo << (sh_lo + half), hi << (sh_hi + half)
    w0 = mx.sum(lw[..., :4], axis=(-1, -2))
    w1 = mx.sum(lw[..., 4:], axis=(-1, -2))
    w2 = mx.sum(hw, axis=(-1, -2))
    words = mx.stack([w0, w1, w2], axis=2).reshape(n, k // 512 * 96)
    return words, best_s[..., 0].astype(mx.float16)


def dequant_mx(words: mx.array, scales: mx.array, n: int, k: int) -> mx.array:
    """Device decode of the stored arrays -> float16 [n, k] (the PPL gate's all-rows path, KYOJIN_GLM_DENSE_Q_ALL, and tests)."""
    w = words.reshape(n, k // 512, 3, 32)
    lo_w = mx.stack([w[:, :, 0]] * 4 + [w[:, :, 1]] * 4, axis=-1)                       # [n, nch, 32, 8]: the plane of pair p
    half = mx.array([0, 16], mx.uint32).reshape(2, 1)
    sh_lo = mx.array([0, 4, 8, 12, 0, 4, 8, 12], mx.uint32)
    sh_hi = mx.array([0, 2, 4, 6, 8, 10, 12, 14], mx.uint32)
    lo = (lo_w[..., None, :] >> (sh_lo + half)) & 15                                      # [n, nch, 32, 2, 8]
    hi = (w[:, :, 2][..., None, None] >> (sh_hi + half)) & 3
    q = (lo | (hi << 4)).astype(mx.float32) - 32.0
    return (q.reshape(n, k // 32, 32) * scales.astype(mx.float32)[..., None]).astype(mx.float16).reshape(n, k)
