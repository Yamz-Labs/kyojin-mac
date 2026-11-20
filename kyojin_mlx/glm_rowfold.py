# SPDX-License-Identifier: Apache-2.0
"""N2-ROWDOT: the KDA `fbg_a_proj` row-dot folded into the KDA in-projection GEMV (flag KYOJIN_GLM_ROWFOLD, default off).

Stock decode of a GLM-5.3 linear-attention (KDA) layer launches, right after the mHC collapse (`glm_hc_b`), TWO independent kernels on the same x:
    kyojin_fg_g6_r<R>_*   the 6-bit qkv in-projection  [4096 -> 24576]   (one launch, 384 threadgroups at R=1)
    kyojin_rowdot2        the fp16 fbg_a projection      [4096 ->   320]   (rowinv.rowdot: one 32-lane threadgroup per output, 320 of them, ~0 bytes)
G2h charged 3.5 ms / step to the 34 + 11 `rowdot2` launches, but the charge rule gives overlapped time to the buffer that finishes last: the pair is
138 us per KDA layer against 115 us for the in-projection alone (G2h raw data), i.e. the row-dot really costs ~23 us / layer (~0.8 ms / step) of dispatch + tail,
not 100 us. Here the row-dot becomes extra threadgroups of the in-projection launch:

    grid = [ main threadgroups (the unchanged G6 body) | ceil(NX / NSG) row-dot threadgroups ]
    row-dot threadgroup: simdgroup s takes output o = tg * NSG + s, lane l sums k = 8 l + 256 i (8 values in order, fp32 fma), simd_sum, one fp16 store.
This is rowinv.rowdot's arithmetic for KS = 1 exactly (same k order, same fma, same rounding), so y2 is bit-identical to rowinv.rowdot(x, w) and y to
glm_dense6.gemv (the main body is the same source text; the role test is the only addition and is uniform per threadgroup, before any barrier).
Rows R = 1 .. DENSE_GEMV_ROWS (decode: one token per user), x fp16, K % 8 == 0, KS(K, NX) == 1 (pick_ks: NX >= 192 or K small). One launch less per KDA layer
(34 per step); the pair becomes one launch of the in-projection's length plus the 320 extra rows (+2.6 MB read against 82 MB).
Not covered (stock two launches): verify rows of one session (L > 1), non-fp16 x, a bias, a missing 6-bit copy, other module types.
"""
from __future__ import annotations

import functools
import os

import mlx.core as mx

from . import fired as FD
from . import foreign_metal as FM
from . import glm_dense6 as G6
from . import kernels as KN
from . import rowinv as RI

ENABLED = os.environ.get("KYOJIN_GLM_ROWFOLD", "0") not in ("0", "")
FRONT = os.environ.get("KYOJIN_GLM_ROWFOLD_FRONT", "1") not in ("0", "")      # row-dot threadgroups first in the grid (beside the main body), 0 = after it (tail)
STATS = {"fold": 0, "stock": 0}

# 8 values of a row as float (rowinv._LD8's helpers, own names: the G3 header owns other symbols)
_HDR = FM.G3_HDR + r"""
template <typename T> inline void rf_ld8(const device T* p, thread float* v) { for (uint j = 0; j < 8; ++j) v[j] = float(p[j]); }
inline void rf_ld8(const device half* p, thread float* v) {
    const float4 a = float4(*(const device half4*)p), b = float4(*(const device half4*)(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
"""


def _role(r: int, nr: int, nsg: int, front: bool) -> str:
    """front: the row-dot threadgroups are the FIRST of the grid (they run beside the main body instead of as its tail), the main body then reads its index minus NXTG."""
    cond = "tgi < uint(NXTG)" if front else "tgi >= ntg_main"
    first = "tgi" if front else "(tgi - ntg_main)"
    return f"""
    {{
      const uint ntg_main = (uint(N) + {nr * nsg}u - 1u) / {nr * nsg}u;
      const uint tgi = threadgroup_position_in_grid.x;
      if ({cond}) {{                                   // row-dot role (uniform per threadgroup, no barrier before it)
        const uint o = {first} * {nsg}u + simdgroup_index_in_threadgroup;
        if (o >= uint(NX)) return;
        const uint ln = thread_index_in_simdgroup;
        float a2[{r}];
        for (uint r_ = 0; r_ < {r}u; ++r_) a2[r_] = 0.0f;
        const device half* wp = w2 + ulong(o) * uint(KIN);
        for (uint kk = ln * 8u; kk < uint(KIN); kk += 256u) {{
          float wv[8];
          rf_ld8(wp + kk, wv);
          for (uint r_ = 0; r_ < {r}u; ++r_) {{
            float xv[8];
            rf_ld8(x + ulong(r_) * uint(KIN) + kk, xv);
            for (uint j = 0; j < 8u; ++j) a2[r_] = fma(wv[j], xv[j], a2[r_]);
          }}
        }}
        for (uint r_ = 0; r_ < {r}u; ++r_) {{
          const float t = simd_sum(a2[r_]);
          if (ln == 0u) y2[ulong(r_) * uint(NX) + o] = static_cast<half>(t);
        }}
        return;
      }}
    }}
"""


@functools.lru_cache(maxsize=None)
def _kern(r: int, nr: int, u: int, nsg: int, front: bool):
    main = G6.source(r, nr, u, nsg, "h").replace("threadgroup_position_in_grid.x", "(threadgroup_position_in_grid.x - uint(NXTG))")
    return KN._kern(("rowfold", r, nr, u, nsg, front), name=f"kyojin_fg_g6rf{'f' if front else 't'}_r{r}_n{nr}_u{u}_s{nsg}h", input_names=["x", "qs", "mt", "gs", "w2"],
                    output_names=["y", "y2"], header=_HDR, source=_role(r, nr, nsg, front) + main)


def fold_ok(k: int, n: int, nx: int, r: int) -> bool:
    """The row-dot role equals rowinv.rowdot only for KS == 1 and K % 8 == 0 (pick_ks is a function of (K, N) of the row-dot)."""
    return k % 512 == 0 and k % 8 == 0 and RI.pick_ks(k, nx) == 1 and 1 <= r <= 4


def gemv_fold(x2, qs, mt, gs, n: int, k: int, w2):
    """x2 [R, k] fp16, the 6-bit copy (qs, mt, gs) of an [n, k] matrix, w2 [NX, k] fp16 -> (y [R, n] fp16 = glm_dense6.gemv, y2 [R, NX] fp16 = rowinv.rowdot(x2, w2))."""
    R = int(x2.shape[0])
    NX = int(w2.shape[0])
    nr, nsg, u = FM.gemv8g3_config(R, n, k)
    kc = FM.gemv8g3_kc(R, k)
    main = -(-n // (nr * nsg))
    extra = -(-NX // nsg)
    return _kern(R, nr, u, nsg, FRONT)(
        inputs=[x2, qs, mt, gs, w2], template=[("KIN", k), ("N", n), ("KC", kc), ("NPH", -(-k // kc)), ("NX", NX), ("NXTG", extra if FRONT else 0)], grid=((main + extra) * nsg * 32, 1, 1),
        threadgroup=(nsg * 32, 1, 1), output_shapes=[(R, n), (R, NX)], output_dtypes=[mx.float16, mx.float16])


def _plain_fp16(m, rows: int, cols: int) -> bool:
    """A bias-free fp16 [rows, cols] linear: nn.Linear or the row-invariant RowLinear / RowLinearS the served load path swaps in."""
    return (type(m).__name__ in ("Linear", "RowLinear", "RowLinearS") and "bias" not in m and m.weight.shape == (rows, cols) and m.weight.dtype == mx.float16)


def qkv_fbg(qkv_proj, fbg_proj, x):
    """(qkv [B, 1, n] fp16, fbg [B, 1, NX] fp16) in ONE launch for a decode activation x [B, 1, K] fp16, or None when the stock two-launch path must run."""
    if not ENABLED or x.ndim != 3 or x.shape[1] != 1 or x.dtype != mx.float16:
        return None
    dev = getattr(qkv_proj, "dev", None)
    B, K = int(x.shape[0]), int(x.shape[2])
    if dev is None or getattr(qkv_proj, "_in_perm", None) is not None or getattr(qkv_proj, "_bias", None) is not None or qkv_proj.out_dtype != mx.float16:
        return None
    if dev.g6 is None or not dev.g3_ok(B) or dev.k != K or "weight" not in fbg_proj:
        return None
    nx = int(fbg_proj.weight.shape[0])
    if not _plain_fp16(fbg_proj, nx, K) or not fold_ok(K, dev.n, nx, B):
        return None
    FD.fire("glm_rowfold")
    y, y2 = gemv_fold(x.reshape(B, K), dev.g6[0], dev.g6[1], dev._ones(), dev.n, K, fbg_proj.weight)
    return y.reshape(B, 1, dev.n), y2.reshape(B, 1, nx)
