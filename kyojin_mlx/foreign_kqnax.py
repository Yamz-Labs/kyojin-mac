# SPDX-License-Identifier: Apache-2.0
# Adapted from oMLX (https://github.com/jundot/omlx), Apache-2.0, omlx/patches/m5_gather_qmm_nax.py (the "seg" schedule: BM x 64 single-expert tiles, the weight tile of each K step
# decoded ONCE into threadgroup memory between two barriers, tile-on-x grid layout, split gate/up operands with the paired-row epilogue). Re-written for our NVFP4 stack:
# fp16 activations in natural k order, fp16 decode (e2m1 x e4m3 is exact in fp16), the per-expert global scale applied in the epilogue on the fp32 accumulators,
# the tile list found in-kernel from tb / off (kqgmm.csort with MT = BM), the clamped SwiGLU of GLM-5.3 fused behind gate|up (no concatenated copy of the weights).
"""NAX sorted grouped GEMM on the stored NVFP4 rows, weights decoded once per tile (flag KYOJIN_KQ_GMM_SCHED=omlx, default off).

Same stored bytes as kqgmm (qs uint32 [E, n, k / 8] = nibble j of word w is k = 8 w + j, meta = e4m3 scale bytes [E, n, k / 16], gs float [E]); the only difference is the schedule:
kqgmm feeds the matrix units from registers decoded per simdgroup; here a threadgroup of BM / 32 x 2 simdgroups owns a BM x 64 output tile of ONE expert, decodes the 64 x BK
weight tile once into threadgroup memory (fp16) and every simdgroup multiplies its 32 x 32 block against it. The activations are read natural-order from the sorted rows (no xperm).
Output rows are in sorted order (plain), or scattered to the slot order through orow (the down projection). gate|up: one launch, both operands, silu(clamp(g)) * clamp(u) in the epilogue."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx

from . import kernels as KN

BN = 64
_MLX_UTILS = ("mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/bf16.h", "mlx/backend/metal/kernels/bf16_math.h", "mlx/backend/metal/kernels/complex.h",
              "mlx/backend/metal/kernels/defines.h", "mlx/backend/metal/kernels/logging.h")
_MLX_MM = ("mlx/backend/metal/kernels/steel/gemm/nax.h", "mlx/backend/metal/kernels/fp4.h", "mlx/backend/metal/kernels/fp8.h")


def sched_on() -> bool:
    """KYOJIN_KQ_GMM_SCHED=omlx turns the tile-decode schedule on (default off). Read per call."""
    return os.environ.get("KYOJIN_KQ_GMM_SCHED", "") == "omlx"


def map_on() -> bool:
    """Row-mapped activations in the gate|up kernel (default on; KYOJIN_KQ_NAX_MAP=0 reads the gathered copy). Read per call."""
    return os.environ.get("KYOJIN_KQ_NAX_MAP", "1") not in ("0", "")


def _mlx_headers() -> str:
    root = Path(mx.__file__).parent / "include"
    seen = {root / p for p in _MLX_UTILS}

    def expand(rel: str) -> str:
        path = root / rel
        if path in seen:
            return ""
        seen.add(path)
        lines = []
        for line in path.read_text().splitlines():
            s = line.strip()
            if s.startswith('#include "mlx/') and s.endswith('"'):
                lines.append(expand(s[len('#include "'):-1]))
            elif s != "#pragma once":
                lines.append(line)
        return "\n".join(lines)
    return "\n".join(expand(p) for p in _MLX_MM)


_HEADER = r"""
using namespace metal;
using namespace mlx::steel;

namespace kq_nax {
STEEL_CONST int kBN = 64;
STEEL_CONST int kWN = 2;
STEEL_CONST short kSM = 32;
STEEL_CONST short kSN = 32;
STEEL_CONST short kSK = 32;
STEEL_CONST short kTM = 2;
STEEL_CONST short kTN = 2;
STEEL_CONST short kTK = 2;

template <int BM, int BK>
struct Geo {
  STEEL_CONST int kWM = BM / kSM;
  STEEL_CONST int kThreads = kWM * kWN * 32;
  STEEL_CONST int kLT = (kThreads & (kThreads - 1)) == 0 ? kThreads : (kThreads > 256 ? 256 : (kThreads > 128 ? 128 : 64));
  STEEL_CONST int kVPT = kBN * BK / kLT;
  STEEL_CONST int kTPR = BK / kVPT;
  STEEL_CONST int kBKP = BK + 8;
  static_assert(BM % kSM == 0 && BK % kSK == 0, "tile geometry");
  static_assert(kTPR >= 1 && kTPR * kVPT == BK && kVPT % 16 == 0, "loader split");
};

// fused gate|up: 16-row blocks of the 64-row weight tile alternate gate / up rows of the same 16 output columns (row of the operand: pair_row(r, 0)).
METAL_FUNC int pair_row(const int r) { return ((r >> 5) << 4) + (r & 15); }

// Weight-tile loader: thread lid owns row lid / kTPR of the 64 x BK tile and the kVPT values from column (lid % kTPR) * kVPT (kNG groups of 16, one e4m3 scale each).
template <typename G, bool PAIR>
struct TileLoader {
  STEEL_CONST int kVPT = G::kVPT;
  STEEL_CONST int kWords = kVPT / 8;
  STEEL_CONST int kNG = kVPT / 16;
  STEEL_CONST int kWPG = 2;
  const device uint32_t* src;
  const device uint8_t* sc;
  const short row;
  const short col;
  uint32_t raw[kWords];
  float p[kNG];

  METAL_FUNC TileLoader(const device uint8_t* w_tile, const device uint8_t* s_tile, const int K, const uint lid) thread
      : row(short(lid / G::kTPR)), col(short((lid % G::kTPR) * kVPT)) {
    const size_t w_off = PAIR ? size_t(pair_row(row)) : size_t(row);
    src = (const device uint32_t*)(w_tile + w_off * (K / 2) + col / 2);
    sc = s_tile + w_off * (K / 16);
  }
  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* ptr = src + kb * (G::kBKP - 8) / 8;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) raw[i] = ptr[i];
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      uint8_t sb = sc[(kb * (G::kBKP - 8) + col + g * 16) / 16];
      p[g] = float(static_cast<bfloat>(*(thread fp8_e4m3*)(&sb)));
    }
  }
  METAL_FUNC void store(threadgroup half* Ws) const thread {
    threadgroup half* dst = Ws + row * G::kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      vec<half, 8> v;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < 8; j++) {
        uint8_t qb = uint8_t((raw[i] >> (4 * j)) & 15u);
        v[j] = static_cast<half>(p[i / kWPG] * float(*(thread fp4_e2m1*)(&qb)));
      }
      *(threadgroup vec<half, 8>*)(dst + i * 8) = v;
    }
  }
};

template <int BKP, bool FULL>
METAL_FUNC void sub_step(thread NAXTile<float, kTM, kTN>& Dtile, const device half* xn, const threadgroup half* ws, const int K, const short sgp_sm) {
  NAXTile<half, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<half, kTM, kTK> Atile;
    volatile int compiler_barrier;
    Atile.load(xn, K);
    Btile.template load<half, BKP, 1>(ws);
    tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
    (void)compiler_barrier;
  } else {
    Btile.template load<half, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<half, 1, kTK> Arow;
        Arow.load_safe(xn + mm * 16 * K, K, short2(kSK, sgp_sm - mm * 16));
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(Dtile.frag_at(mm, nn), Dtile.frag_at(mm, nn + 1), Arow.frag_at(0, kk), metal::bool_constant<false>{}, Btile.frag_at(nn, kk), Btile.frag_at(nn + 1, kk),
                             metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}


// Row-mapped activations (oMLX load_a_map / sub_step_map): sorted row r of the product is token row order[r] / KTOP of x, read in place (no replicated [S, k] copy).
// a_off[i][h] = this lane's element offset of activation row i * 16 + h * 8 + sc.y of its block (rows past the tile point at its last row).
METAL_FUNC void load_a_map(thread NAXTile<half, kTM, kTK>& A, const device half* xk, const thread uint (&a_off)[kTM][2], const short i0, const short nr) {
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < nr; r++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const device half* xp = xk + a_off[i0 + r][h];
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < kTK; kk++) {
        const vec<half, 4> v = *(const device vec<half, 4>*)(xp + kk * 16);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c++) A.frag_at(r, kk)[h * 4 + c] = v[c];
      }
    }
  }
}

template <int BKP, bool FULL>
METAL_FUNC void sub_step_map(thread NAXTile<float, kTM, kTN>& Dtile, const device half* xk, const thread uint (&a_off)[kTM][2], const threadgroup half* ws, const short sgp_sm) {
  NAXTile<half, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<half, kTM, kTK> Atile;
    volatile int compiler_barrier;
    load_a_map(Atile, xk, a_off, 0, kTM);
    Btile.template load<half, BKP, 1>(ws);
    tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
    (void)compiler_barrier;
  } else {
    const short2 sc = BaseNAXFrag::get_coord();
    Btile.template load<half, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<half, 1, kTK> Arow;
        {
          NAXTile<half, kTM, kTK> Atmp;
          load_a_map(Atmp, xk, a_off, 0, kTM);
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) Arow.frag_at(0, kk) = Atmp.frag_at(mm, kk);
        }
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; h++) {
          if (mm * 16 + h * 8 + sc.y >= sgp_sm) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < kTK; kk++) {
              STEEL_PRAGMA_UNROLL
              for (short c = 0; c < 4; c++) Arow.frag_at(0, kk)[h * 4 + c] = half(0);
            }
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(Dtile.frag_at(mm, nn), Dtile.frag_at(mm, nn + 1), Arow.frag_at(0, kk), metal::bool_constant<false>{}, Btile.frag_at(nn, kk), Btile.frag_at(nn + 1, kk),
                             metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}

// EPI 0: y[row] = clamp(acc * gs) as half, row = orow[sorted row] (SCAT) or the sorted row; N output columns.
// EPI 1 / 2: fused gate|up (split operands), y = silu(g) * u (EPI 2: g = min(g, lim), u = clamp(u, -lim, lim) first), both rounded to half first like the two stored projections; HN output columns.
template <int BM, int BK, int EPI, bool SCAT, bool MAP, int KTOP>
METAL_FUNC void gather_seg(const device half* x, const device uint8_t* w, const device uint8_t* sc, const device uint8_t* wu, const device uint8_t* su, const float gs_a, const float gs_b,
                           const float lim, const device uint32_t* orow, const device uint32_t* rmap, const int row_start, const uint ex, const int rows, const int y_col, device half* y, const int N, const int K,
                           threadgroup half* Ws, const uint sgid, const uint lane) {
  using G = Geo<BM, BK>;
  constexpr bool kPair = EPI != 0;
  constexpr int BKP = G::kBKP;
  const int K_it = K / BK;
  const int HN = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;
  const size_t w_row = size_t(ex) * (kPair ? HN : N) + w_col;
  const uint lid = sgid * 32 + lane;
  const bool up_row = kPair && (((lid / G::kTPR) >> 4) & 1);
  const device uint8_t* wb = up_row ? wu : w;
  const device uint8_t* sb = up_row ? su : sc;
  TileLoader<G, kPair> loader(wb + w_row * (K / 2), sb + w_row * (K / 16), K, lid);
  const bool loads = G::kLT == G::kThreads || lid < uint(G::kLT);

  if constexpr (!MAP) x += size_t(row_start) * K;
  const short tm = kSM * short(sgid / kWN);
  const short tn = kSN * short(sgid % kWN);
  const short sgp_sm = short(min(int(kSM), max(0, rows - int(tm))));
  const bool sg_active = sgp_sm > 0;
  NAXTile<float, kTM, kTN> Dtile;
  Dtile.clear();
  uint a_off[kTM][2];
  const short2 sc0 = BaseNAXFrag::get_coord();
  if constexpr (MAP) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short h = 0; h < 2; h++) {
        const int r = min(int(tm) + i * 16 + h * 8 + int(sc0.y), rows - 1);
        a_off[i][h] = (rmap[row_start + r] / uint(KTOP)) * uint(K) + uint(sc0.x);
      }
    }
  }
  const device half* xn = MAP ? x : x + tm * K;
  const threadgroup half* ws = Ws + tn * BKP;

  dispatch_bool(sgp_sm == kSM, [&](auto kAlignedM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        loader.fetch(k);
        loader.store(Ws);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < BK; kk1 += kSK) {
        if (sg_active) {
          if constexpr (MAP) sub_step_map<BKP, kAlignedM.value>(Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          else sub_step<BKP, kAlignedM.value>(Dtile, xn + kk1, ws + kk1, K, sgp_sm);
        }
      }
      xn += BK;
    }
    if (!sg_active) return;
    const short2 sc2 = BaseNAXFrag::get_coord();
    if constexpr (kPair) {
      device half* yb = y + size_t(row_start + tm) * HN + w_col + tn / 2 + sc2.x;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < BaseNAXFrag::kElemRows; h++) {
          const int r = i * BaseNAXFrag::kFragRows + h * BaseNAXFrag::kElemRowsJump + sc2.y;
          if (r < sgp_sm) {
            vec<half, BaseNAXFrag::kElemCols> v;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < BaseNAXFrag::kElemCols; j++) {
              const short e = h * BaseNAXFrag::kElemCols + j;
              float a = float(half(clamp(Dtile.frag_at(i, 0)[e] * gs_a, -65504.0f, 65504.0f)));
              float b = float(half(clamp(Dtile.frag_at(i, 1)[e] * gs_b, -65504.0f, 65504.0f)));
              if constexpr (EPI == 2) {
                a = min(a, lim);
                b = clamp(b, -lim, lim);
              }
              v[j] = half(a * (1.0f / (1.0f + metal::precise::exp(-a))) * b);
            }
            *(device vec<half, BaseNAXFrag::kElemCols>*)(yb + size_t(r) * HN) = v;
          }
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < BaseNAXFrag::kElemRows; h++) {
          const int r = i * BaseNAXFrag::kFragRows + h * BaseNAXFrag::kElemRowsJump + sc2.y;
          if (r < sgp_sm) {
            const size_t yr = SCAT ? size_t(orow[row_start + tm + r]) : size_t(row_start + tm + r);
            device half* yp = y + yr * N + y_col + tn + sc2.x;
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < kTN; jj++) {
              vec<half, BaseNAXFrag::kElemCols> v;
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < BaseNAXFrag::kElemCols; j++) v[j] = half(clamp(Dtile.frag_at(i, jj)[h * BaseNAXFrag::kElemCols + j] * gs_a, -65504.0f, 65504.0f));
              *(device vec<half, BaseNAXFrag::kElemCols>*)(yp + jj * 16) = v;
            }
          }
        }
      }
    }
  });
}
}  // namespace kq_nax
"""

# Body. Template ints: BM BK GX PAD N K E EPI; bool SCAT. N = rows of the weight operand per expert (plain) or 2 x half_n (fused: the grid covers gate|up).
_BODY = r"""
    using G = kq_nax::Geo<BM, BK>;
    threadgroup half Ws[kq_nax::kBN * G::kBKP + PAD / 2];
    const uint n_cols = uint((N + kq_nax::kBN - 1) / kq_nax::kBN);
    uint ti, c;
    if (GX > 0) { ti = (threadgroup_position_in_grid.y / n_cols) * uint(GX) + threadgroup_position_in_grid.x; c = threadgroup_position_in_grid.y % n_cols; }
    else { ti = threadgroup_position_in_grid.y; c = threadgroup_position_in_grid.x; }
    if (ti >= tb[uint(E)]) return;
    uint e_lo = 0u, e_hi = uint(E);
    while (e_hi - e_lo > 1u) { const uint e_mid = (e_lo + e_hi) >> 1; if (tb[e_mid] <= ti) e_lo = e_mid; else e_hi = e_mid; }
    const uint ex = e_lo;
    const uint rs = off[ex] + (ti - tb[ex]) * uint(BM);
    const int rows = int(min(uint(BM), off[ex + 1u] - rs));
    kq_nax::gather_seg<BM, BK, EPI, SCAT, MAP, KTOP>(x, (const device uint8_t*)w, (const device uint8_t*)sc, WU, SU, GSA, GSB, LIM, ORW, RMP, int(rs), ex, rows, int(c) * kq_nax::kBN, y, N, K, Ws,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

_PLAIN_SRC = (_BODY.replace("WU", "(const device uint8_t*)w").replace("SU", "(const device uint8_t*)sc").replace("GSA", "gs[ex]").replace("GSB", "0.0f").replace("LIM", "0.0f")
              .replace("ORW", "orow").replace("RMP", "tb"))
_PLAIN_SCAT_SRC = _PLAIN_SRC
_FUSED_SRC = (_BODY.replace("WU", "(const device uint8_t*)wu").replace("SU", "(const device uint8_t*)su").replace("GSA", "gsg[ex]").replace("GSB", "gsu[ex]").replace("LIM", "lim[0]")
              .replace("ORW", "tb").replace("RMP", "order"))

_EPI_OLD = "v[j] = half(a * (1.0f / (1.0f + metal::precise::exp(-a))) * b);"


def act_header(hdr: str, act):
    """Activation kind of the fused gate | up epilogue: act None = the silu-gated form (header returned unchanged); ("situ", scale_gate, scale_lin) = ((sg tanh(a / sg)) * sigmoid(a)) * (sl tanh(b / sl)) on the
    half-rounded projections, no clamp (EPI 1 path)."""
    if act is None:
        return hdr
    kind, sg, sl = act
    assert kind == "situ" and sg > 0.0 and sl > 0.0 and hdr.count(_EPI_OLD) == 1, act
    return hdr.replace(_EPI_OLD, f"v[j] = half((({float(sg)!r}f * metal::precise::tanh(a / {float(sg)!r}f)) * (1.0f / (1.0f + metal::precise::exp(-a)))) * ({float(sl)!r}f * metal::precise::tanh(b / {float(sl)!r}f)));")


_K: dict = {}


def _kern(kind: str, act=None):
    key = kind if act is None else (kind, act)
    if key not in _K:
        hdr = _mlx_headers() + act_header(_HEADER, act)
        if kind == "plain":
            _K[key] = mx.fast.metal_kernel(name="kyojin_kq_nax_plain", input_names=["x", "w", "sc", "gs", "tb", "off"], output_names=["y"], header=hdr, source=_PLAIN_SRC.replace("orow", "tb"))
        elif kind == "scat":
            _K[key] = mx.fast.metal_kernel(name="kyojin_kq_nax_scat", input_names=["x", "w", "sc", "gs", "tb", "off", "orow"], output_names=["y"], header=hdr, source=_PLAIN_SCAT_SRC)
        else:
            _K[key] = mx.fast.metal_kernel(name="kyojin_kq_nax_gateup" + ("_situ" if act else ""), input_names=["x", "w", "sc", "wu", "su", "gsg", "gsu", "lim", "tb", "off", "order"], output_names=["y"], header=hdr, source=_FUSED_SRC)
    return _K[key]


def plan(per_expert: float):
    """(BM, BK, GX, PAD) from the mean rows per expert (oMLX measured plan: 128 x 128 + 8 KB pad from 96 rows per expert)."""
    ov = os.environ.get("KYOJIN_KQ_NAX_PLAN")
    if ov:
        return tuple(int(v) for v in ov.split(","))
    return 96, 128, 32, 0                                   # best or tied at 16K .. 64K slots, 57 .. 228 rows per expert (G5f sweep)


def meta_bytes(dp):
    """e4m3 scale bytes [E, n, k / 16] of a stored nvfp4 pack (the stored uint16 meta, same memory)."""
    return mx.view(dp.meta, mx.uint8).reshape(dp.E, dp.n, dp.k // 16)


def ok(dp, BK: int = 128) -> bool:
    return dp.fmt == "nvfp4" and bool(dp.E) and dp.k % BK == 0 and dp.n % BN == 0 and not dp.rowgs


def _grid(N: int, S: int, E: int, BM: int, GX: int):
    maxt = S // BM + E
    n_cols = (N + BN - 1) // BN
    tg = (GX, ((maxt + GX - 1) // GX) * n_cols) if GX else (n_cols, maxt)
    return (tg[0] * 32, tg[1] * 2, BM // 32), (32, 2, BM // 32)


def gmm(dp, x, tb, off, S: int, pl, orow=None):
    """x half [S, k] natural order, sorted by expert (tb / off from csort with MT = BM). y half [S, n] (sorted order, or slot order through orow)."""
    BM, BK, GX, PAD = pl
    grid, tgs = _grid(dp.n, S, dp.E, BM, GX)
    ins = [x, dp.qs, meta_bytes(dp), dp.gs, tb, off] + ([orow] if orow is not None else [])
    return _kern("scat" if orow is not None else "plain")(
        inputs=ins, template=[("BM", BM), ("BK", BK), ("GX", GX), ("PAD", PAD), ("N", dp.n), ("K", dp.k), ("E", dp.E), ("EPI", 0), ("SCAT", orow is not None), ("MAP", False), ("KTOP", 1)],
        grid=grid, threadgroup=tgs, output_shapes=[(S, dp.n)], output_dtypes=[mx.float16])[0]


def gate_up(g, u, x, tb, off, S: int, pl, limit=None, order=None, K: int = 8, act=None):
    """x half [S, k] natural sorted, or (order given) the token rows [T, k] read in place: sorted row r = x[order[r] // K]; returns silu(min(gate, L)) * clip(up, +-L) half [S, n] (both projections rounded to half first, as the stored path)."""
    BM, BK, GX, PAD = pl
    grid, tgs = _grid(2 * g.n, S, g.E, BM, GX)
    lim = mx.array([0.0 if limit is None else float(limit)], dtype=mx.float32)
    ins = [x, g.qs, meta_bytes(g), u.qs, meta_bytes(u), g.gs, u.gs, lim, tb, off, order if order is not None else tb]
    return _kern("gateup", act)(
        inputs=ins, template=[("BM", BM), ("BK", BK), ("GX", GX), ("PAD", PAD), ("N", 2 * g.n), ("K", g.k), ("E", g.E), ("EPI", 1 if (limit is None or act) else 2), ("SCAT", False), ("MAP", order is not None), ("KTOP", K)],
        grid=grid, threadgroup=tgs, output_shapes=[(S, g.n)], output_dtypes=[mx.float16])[0]


_GATHER = r"""
    const uint g = thread_position_in_grid.x;
    const uint s = g / (uint(KIN) / 8u), p = (g - s * (uint(KIN) / 8u)) * 8u;
    *(device uint4*)(o + ulong(s) * uint(KIN) + p) = *(device const uint4*)(x + ulong(order[s] / uint(K)) * uint(KIN) + p);
"""


def gather(x2, order, K: int):
    """x2 half [T, k] -> half [S, k]: row s = x2[order[s] // K], natural column order."""
    S, k = int(order.shape[0]), int(x2.shape[1])
    kern = KN._kern(("kq_nax_gather",), name="kyojin_kq_nax_gather", input_names=["x", "order"], output_names=["o"], header="", source=_GATHER)
    return kern(inputs=[x2, order], template=[("KIN", k), ("K", K)], grid=(S * k // 8, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, k)], output_dtypes=[mx.float16])[0]


def mlp(glu, x2, e32, K: int):
    """ForeignSwitchGLU prefill on the stored rows with the tile-decode schedule: gate|up fused (clamped SwiGLU in the epilogue), then down scattered to slot order.
    Returns None when this shape is not eligible (the caller keeps kqgmm)."""
    from . import foreign_kqgmm as KQ
    g, u, d = glu.g, glu.u, glu.d
    S = int(e32.shape[0])
    pl = plan(S / max(1, g.E))
    if not (ok(g, pl[1]) and ok(u, pl[1]) and ok(d, pl[1]) and g.k == u.k and g.n == u.n and d.k == g.n and g.E <= 1024):
        return None
    order, tb, off = KQ.csort(e32, g.E, pl[0])
    act = getattr(glu, "act_kind", None)
    h = gate_up(g, u, x2, tb, off, S, pl, glu.limit, order=order, K=K, act=act) if map_on() else gate_up(g, u, gather(x2, order, K), tb, off, S, pl, glu.limit, act=act)
    return gmm(d, h, tb, off, S, pl, orow=order)
