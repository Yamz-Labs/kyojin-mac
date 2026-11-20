# SPDX-License-Identifier: Apache-2.0
"""Flag foreign_fmm8 (KYOJIN_FOREIGN_FMM8, default 0 = off): few-row simdgroup-matrix product for stored Q8_0 linears (2 .. 16 rows: verify rows of a speculative round).

One launch reads each weight once for 8 * NT weight rows x up to 16 activation rows; the NSG simdgroups of a threadgroup split K (units of 32 weights, interleaved) and
reduce through threadgroup memory. Port of llama.cpp ggml-metal kernel_mul_mv_mma_blk (Q8_0, rt 1): the weights are read from the stored bytes in the A fragment layout
(lane = weight row fm, 8 consecutive bytes c * 8 .. c * 8 + 7 of the 32-unit), a byte becomes the exact half q via the "1024 + q" bit trick (sign bit flipped, OR 0x6400,
minus 1152), the activation fragment is read unpermuted from x (lane fm holds x[b], x[b + 1] / x[b + 4], x[b + 5], b = 8 (fm / 2) + 2 (fm % 2)), the MMA is half x half
-> float over each 32-unit, the unit scale (mt, half) multiplies the unit's partial sum, gs (per row float) multiplies at the store. Same products as gemv8, different sum
order (fp32): equal to the fp32 reference to ~1e-6 relative.

Tiling (llama ggml_metal_op_mul_mat_mma_tiling): NSG = 16 (N <= 6144) or 8 (more rows), halved to the unit count; NT = 4, halved while NT > N / 1024 or the K-reduction
buffer (NSG * NT * RT * 256 B) exceeds 16 KB. The head: the dense matrix is packed once into the same Q8_0 layout (Q8Pack.from_dense)."""
from __future__ import annotations

import functools
import os

import numpy as np
import mlx.core as mx

from . import foreign_metal as FM
from . import kernels as KN

MIN_ROWS = int(os.environ.get("KYOJIN_FOREIGN_FMM8", "0") or 0)        # flag foreign_fmm8 (flags.py rewrites it at runtime): first row count served, 0 = off
MAX_ROWS = int(os.environ.get("KYOJIN_FOREIGN_FMM8_MAX", "8") or 8)      # flag foreign_fmm8_max: last row count served (<= 16)
ROWS_CAP = 16
SMEM_MAX = 16384
NSG_MAX = 16                                                            # threads per threadgroup = 32 NSG (llama uses 32 for <= 64 rows: not needed here)


def tiling(R: int, n: int, k: int) -> tuple[int, int, int]:
    """(NSG, NT, RT) for R rows of an [n, k] matrix."""
    rt = 1 if R <= 8 else 2
    nsg = NSG_MAX if n > 64 and n <= 6144 else 8 if n > 6144 else NSG_MAX
    steps = k // 32
    while nsg > 1 and nsg > steps:
        nsg //= 2
    lim = min(SMEM_MAX // (nsg * rt * 256), n // 1024)
    nt = 4
    while nt > 1 and nt > lim:
        nt //= 2
    return nsg, nt, rt


def eligible(R: int, n: int, k: int) -> bool:
    return 1 <= R <= ROWS_CAP and k % 32 == 0 and n >= 8


@functools.lru_cache(maxsize=None)
def source(yout: str = "h", xin: str = "h") -> str:
    """Body. Template ints: R (1..16), KIN (% 32), N, NT, NSG. Inputs x (half [R, KIN]), qs (uint32 [N, KIN/4], natural byte order), mt (half [N, KIN/32]), gs (float [N]).
    Output y [R, N] half (yout "h", saturated at +-65504) or float32 (yout "f"). x half (xin "h") or float32 (xin "f": rounded to half, saturated, as the cast the caller would launch)."""
    store = "half(clamp(v, -65504.0f, 65504.0f))" if yout == "h" else "v"
    return r"""
    constexpr uint RT = (uint(R) + 7u) / 8u;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = sg * 32u + lane;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    const uint bo = 8u * (fm >> 1) + 2u * (fm & 1u);
    const uint UPR = uint(KIN) / 32u;
    const uint n0 = threadgroup_position_in_grid.x * (8u * uint(NT));
    device const uint2* qp[NT];
    device const half* mp[NT];
    UNR for (uint t = 0; t < uint(NT); ++t) {
      const uint r = min(n0 + 8u * t + fm, uint(N) - 1u);
      qp[t] = (device const uint2*)qs + r * (uint(KIN) / 8u) + c;
      mp[t] = mt + r * UPR;
    }
    device const XT* xp[RT][2];
    UNR for (uint rt = 0; rt < RT; ++rt) UNR for (uint e = 0; e < 2; ++e) {
      const uint r = min(8u * rt + fn + e, uint(R) - 1u);
      xp[rt][e] = (device const XT*)(x + r * uint(KIN) + bo);
    }
    float acc[RT][NT][2];
    UNR for (uint rt = 0; rt < RT; ++rt) UNR for (uint t = 0; t < uint(NT); ++t) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }

    uint2 q[NT]; float d[NT]; half2 b0[RT][2]; half2 b1[RT][2];
    uint u = min(sg, UPR - 1u);
    UNR for (uint t = 0; t < uint(NT); ++t) { q[t] = qp[t][u * 4u]; d[t] = float(mp[t][u]); }
    UNR for (uint rt = 0; rt < RT; ++rt) UNR for (uint e = 0; e < 2; ++e) { b0[rt][e] = XL(xp[rt][e][u * 16u]); b1[rt][e] = XL(xp[rt][e][u * 16u + 2u]); }

    for (uint ib = sg; ib < UPR; ib += uint(NSG)) {
      uint2 qc[NT]; float dc[NT]; half2 c0[RT][2]; half2 c1[RT][2];
      UNR for (uint t = 0; t < uint(NT); ++t) { qc[t] = q[t] ^ uint2(0x80808080u); dc[t] = d[t]; }
      UNR for (uint rt = 0; rt < RT; ++rt) UNR for (uint e = 0; e < 2; ++e) { c0[rt][e] = b0[rt][e]; c1[rt][e] = b1[rt][e]; }
      const uint un = min(ib + uint(NSG), UPR - 1u);
      UNR for (uint t = 0; t < uint(NT); ++t) { q[t] = qp[t][un * 4u]; d[t] = float(mp[t][un]); }
      UNR for (uint rt = 0; rt < RT; ++rt) UNR for (uint e = 0; e < 2; ++e) { b0[rt][e] = XL(xp[rt][e][un * 16u]); b1[rt][e] = XL(xp[rt][e][un * 16u + 2u]); }

      simdgroup_matrix<float, 8, 8> mm[RT][NT];
      UNR for (uint rt = 0; rt < RT; ++rt) UNR for (uint t = 0; t < uint(NT); ++t) mm[rt][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      UNR for (uint s = 0; s < 4u; ++s) {
        simdgroup_matrix<half, 8, 8> mb[RT];
        UNR for (uint rt = 0; rt < RT; ++rt) {
          const half2 v0 = (s >> 1) ? c1[rt][0] : c0[rt][0];
          const half2 v1 = (s >> 1) ? c1[rt][1] : c0[rt][1];
          mb[rt].thread_elements()[0] = (s & 1u) ? v0.y : v0.x;
          mb[rt].thread_elements()[1] = (s & 1u) ? v1.y : v1.x;
        }
        UNR for (uint t = 0; t < uint(NT); ++t) {
          const uint wv = (s >> 1) ? qc[t].y : qc[t].x;
          const half2 h = as_type<half2>(((wv >> (8u * (s & 1u))) & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h);
          simdgroup_matrix<half, 8, 8> ma;
          ma.thread_elements()[0] = h.x; ma.thread_elements()[1] = h.y;
          UNR for (uint rt = 0; rt < RT; ++rt) simdgroup_multiply_accumulate(mm[rt][t], ma, mb[rt], mm[rt][t]);
        }
      }
      UNR for (uint t = 0; t < uint(NT); ++t) UNR for (uint rt = 0; rt < RT; ++rt) {
        acc[rt][t][0] = fma(dc[t], mm[rt][t].thread_elements()[0], acc[rt][t][0]);
        acc[rt][t][1] = fma(dc[t], mm[rt][t].thread_elements()[1], acc[rt][t][1]);
      }
    }

    threadgroup float red[NSG * RT * NT * 64];
    UNR for (uint rt = 0; rt < RT; ++rt) UNR for (uint t = 0; t < uint(NT); ++t) {
      const uint base = ((sg * RT + rt) * uint(NT) + t) * 64u + fm * 8u + fn;
      red[base] = acc[rt][t][0]; red[base + 1u] = acc[rt][t][1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < RT * uint(NT) * 64u; idx += 32u * uint(NSG)) {
      const uint w = idx & 7u, a = (idx >> 3) & 7u, tile = idx >> 6;
      const uint t = tile % uint(NT), rt = tile / uint(NT);
      float v = 0.0f;
      for (uint g = 0; g < uint(NSG); ++g) v += red[((g * RT + rt) * uint(NT) + t) * 64u + w * 8u + a];
      const uint nn = n0 + 8u * t + w, row = 8u * rt + a;
      if (nn < uint(N) && row < uint(R)) { v *= gs[nn]; y[row * uint(N) + nn] = STORE; }
    }
""".replace("STORE", store).replace("XT", "half2" if xin == "h" else "float2").replace("XL(", "(" if xin == "h" else "fmm8_h2(")


_HDR = FM.UNR + "inline half2 fmm8_h2(float2 v) { return half2(clamp(v, float2(-65504.0f), float2(65504.0f))); }\n"


def _kern(yout: str, xin: str = "h"):
    return KN._kern(("fmm8", yout, xin), name=f"kyojin_fmm8_q8_x{xin}y{yout}", input_names=["x", "qs", "mt", "gs"], output_names=["y"], header=_HDR, source=source(yout, xin))


def run(x, qs, mt, gs, n: int, k: int, out_dtype=mx.float16):
    """x [R, k] half -> [R, n] out_dtype (half or float32). qs uint32 [n, k / 4], mt half [n, k / 32], gs float [n]."""
    R = x.shape[0]
    xin = "h"
    if x.dtype == mx.float32:
        xin = "f"                                       # read as stored, rounded to half in the kernel (no cast launch)
    elif x.dtype != mx.float16:
        x = x.astype(mx.float32); xin = "f"
    nsg, nt, _ = tiling(R, n, k)
    ntg = -(-n // (8 * nt))
    return _kern("f" if out_dtype == mx.float32 else "h", xin)(
        inputs=[x, qs, mt, gs], template=[("R", R), ("KIN", k), ("N", n), ("NT", nt), ("NSG", nsg)], grid=(ntg * 32 * nsg, 1, 1), threadgroup=(32 * nsg, 1, 1),
        output_shapes=[(R, n)], output_dtypes=[out_dtype])[0]


class Q8Pack:
    """A dense matrix [n, k] in the stored Q8_0 layout (symmetric int8, one fp16 scale per 32 weights, gs = ones)."""

    def __init__(self, qs, mt, gs, n: int, k: int):
        self.qs, self.mt, self.gs, self.n, self.k = qs, mt, gs, n, k

    @classmethod
    def from_dense(cls, w, rows: int = 8192):
        """Packed in row chunks (no float32 copy of the whole matrix beyond one chunk)."""
        n, k = w.shape
        qs_parts, mt_parts = [], []
        for r0 in range(0, n, rows):
            blk = w[r0:r0 + rows].astype(mx.float32).reshape(-1, k // 32, 32)
            d = (mx.max(mx.abs(blk), axis=-1, keepdims=True) / 127.0).astype(mx.float16)
            df = mx.maximum(d.astype(mx.float32), 1e-30)
            q = mx.clip(mx.round(blk / df), -127, 127).astype(mx.int8).reshape(-1, k)
            qs_parts.append(mx.contiguous(q).view(mx.uint32))
            mt_parts.append(d.reshape(-1, k // 32))
            mx.eval(qs_parts[-1], mt_parts[-1])
        qs = mx.concatenate(qs_parts, axis=0) if len(qs_parts) > 1 else qs_parts[0]
        mt = mx.concatenate(mt_parts, axis=0) if len(mt_parts) > 1 else mt_parts[0]
        gs = mx.ones((n,), dtype=mx.float32)
        mx.eval(qs, mt, gs)
        return cls(qs, mt, gs, n, k)

    def apply(self, x, out_dtype=mx.float16):
        return run(x, self.qs, self.mt, self.gs, self.n, self.k, out_dtype)


_HEADS: dict = {}                                               # id(weight) -> (Q8Pack, weight): the weight reference keeps the id valid


def head_pack(w):
    """Q8Pack of the dense head matrix `w` [V, D] (built once), or None when it is not a dense 2-D float matrix with D % 32 == 0."""
    if not isinstance(w, mx.array) or w.ndim != 2 or w.dtype not in (mx.float32, mx.bfloat16, mx.float16) or w.shape[-1] % 32 or w.shape[0] < 8:
        return None
    hit = _HEADS.get(id(w))
    if hit is None:
        hit = _HEADS[id(w)] = (Q8Pack.from_dense(w), w)
    return hit[0]


def np_reference(x, qs, mt, gs):
    """float64 numpy product of the stored Q8_0 bytes: x [R, k], qs uint32 [n, k/4], mt [n, k/32], gs [n] -> [R, n]."""
    q = np.ascontiguousarray(np.array(qs)).view(np.int8).reshape(qs.shape[0], -1).astype(np.float64)
    n, k = q.shape
    w = (q.reshape(n, k // 32, 32) * np.array(mt).astype(np.float64)[:, :, None]).reshape(n, k) * np.array(gs).astype(np.float64)[:, None]
    return np.array(x).astype(np.float64) @ w.T
