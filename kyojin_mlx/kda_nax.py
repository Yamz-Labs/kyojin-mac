"""Chunked KDA (vector-gated delta rule, GLM-5.3 linear attention) prefill on the Metal 4 matrix units.

Fork of kyojin_mlx.gdn_nax (itself a fork of MLX's gated_delta_update_nax.h) for a PER-CHANNEL forget gate. Per head, state S [Dv, Dk], one chunk of C = 16
tokens, G_i[d] = sum_{j<=i} lg_j[d] (lg = lower_bound * sigmoid(exp(A_log) * (a + dt_bias)) in [lower_bound, 0], so G >= 16 * lower_bound = -80 and
exp(+-G) stays inside fp32 for the GLM safe gate):
    Qh_i = q_i e^{G_i}   Kh_i = k_i e^{G_i}   Kt_j = k_j e^{-G_j}   Kd_j = k_j e^{G_C - G_j}
    A = beta tril(Kh Kt^T, -1);  delta = (I + A)^-1 (beta V - beta Kh S^T);  Y = Qh S^T + tril(Qh Kt^T) delta;  S <- S diag(e^{G_C}) + delta^T Kd
(tools/kda_chunk_ref.py checks the algebra against the sequential recurrence in float64: 1e-16, float32 2e-7.)
The gate is evaluated inside the kernel from the raw f_b_proj output `a` (no fp32 [T, H, Dk] gate tensor); the per-chunk cumulative tables e^{G}, e^{-G}
live in threadgroup memory and are shared by the 4 simdgroups of a threadgroup (64 value rows of one head). The 16x16 triangular solve is the Horner form of gdn_nax.
The MLX headers are read from the installed package and inlined (not vendored), like gdn_nax."""
import functools
import os
import mlx.core as mx
from . import fired as FD
from . import gdn_nax as GN


# The NAX matrix unit TRUNCATES fp32 operands (toward zero, a biased error) to a 10-bit mantissa (tf32-like; the fp32 accumulator is exact): one "fp32" MMA has ~1e-3 relative error
# (tools/g5e: mma_prec). A product A B is recovered to ~1e-6 with x = hi + lo (hi = x with the low 13 mantissa bits cleared, lo = x - hi, both exact in the MMA) and
# A B ~ Ah Bh + Ah Bl + Al Bh (the Al Bl term is 2^-20). SPLIT bits (KYOJIN_GLM_KDA_SPLIT, default 127 = all): 1 = Kh S^T, 16 = Qh S^T, 2 = state update delta^T Kd (32: with delta^T from the solve, 64: the three products into three tiles instead of one accumulator chain, 2.4x less latency), 4 = pre-pass (Kh Kt^T, Qh Kt^T, Horner solve), 8 = delta / out tail. State error vs float64 (T=8192): 1e-3 without, 4e-6 with.
_HELP = r"""
METAL_FUNC float tf32_hi(float x) { return as_type<float>(as_type<uint>(x) & 0xffffe000u); }
template <typename T>
METAL_FUNC void split_t(const thread T& x, thread T& hi, thread T& lo) {
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < T::kElemsPerFrag; e++) {
      const float v = x.frag_at(0, f)[e];
      const float h = tf32_hi(v);
      hi.frag_at(0, f)[e] = h;
      lo.frag_at(0, f)[e] = v - h;
    }
  }
}
// two frags (f0, f0 + 1) of a wider tile
template <typename T, typename U>
METAL_FUNC void split_sl(const thread T& x, short f0, thread U& hi, thread U& lo) {
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < U::kNumFrags; f++) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < U::kElemsPerFrag; e++) {
      const float v = x.frag_at(0, f0 + f)[e];
      const float h = tf32_hi(v);
      hi.frag_at(0, f)[e] = h;
      lo.frag_at(0, f)[e] = v - h;
    }
  }
}
"""

# ---- shared pieces (python-composed into the kernels below) -------------------------------------------------------------------------------------------------------------
_COMMON_SETUP = r"""
  constexpr int SPLIT = SPLITV;                                       // bitmask of the MMAs run as 3-term hi/lo products (see _HELP)
  constexpr int C = 16;
  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
  const short fn4 = ((qid & 2) | (simd_lane_id & 1)) * 4;
  const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
  const float lbound = lb[0];
  threadgroup float EG[C * Dk];                                       // e^{G_i[d]}
  threadgroup float EN[C * Dk];                                       // e^{-G_i[d]}
  threadgroup float EL[Dk];                                           // e^{G_C-1[d]}
  float beta_fm[2];
  mlx::steel::NAXTile<float, 1, 2> K_tile, Q_tile, Kt_tile;
  mlx::steel::NAXTile<float, 1, 1> V_tile, KS_tile, QS_tile, delta_tile, QKt_tile, out_tile, Tinv_tile, A_tile, KKt_tile, I_tile, TMP_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
    AT_NAX(I_tile, _i) = (_c.x == _c.y) ? 1.0f : 0.0f;
  }
"""

# gate tables of one chunk: NT threads, thread t handles the channels d = t, t + NT, ... (`apre` = a_ at the chunk's first row); a_, decay, dtb, valid_rows in scope
_TABLES = r"""
    {
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < Dk / NT; dd++) {
        const short d = tdx + dd * NT;
        const float dtb_d = dtb[hv_idx * Dk + d];
        APRE_LOAD
        float cum = 0.0f;
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < C; r++) {
          if (r < valid_rows) {
            const float x = decay * (apre[r] + dtb_d);
            const float e = EXPF(-metal::abs(x));                       // sigmoid(x) = x >= 0 ? 1 / (1 + e) : e / (1 + e)
            const float rc = 1.0f / (1.0f + e);
            cum += lbound * ((x < 0) ? e * rc : rc);
          }
          const float eg = EXPF(cum);
          EG[r * Dk + d] = eg;
          EN[r * Dk + d] = EXPF(-cum);
        }
        EL[d] = EXPF(cum);
        APRE_NEXT
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
_APRE_DIRECT = ("float apre[C];\n        STEEL_PRAGMA_UNROLL\n        for (short r = 0; r < C; r++) { apre[r] = (r < valid_rows) ? float(a_[r * Hv * Dk + d]) : 0.0f; }", "")
# main scan: the gate pre-activations of the next chunk are loaded into registers while this chunk's tables / MMAs run (one channel per thread: Dk == NT)
_APRE_CARRY = ("", """const short vn = short(min(C, T - (chunk_i + 1) * C));
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < C; r++) { apre[r] = (r < vn) ? float(a_[(C + r) * Hv * Dk + d]) : 0.0f; }""")


def _tab(mode):
    return _TABLES.replace("APRE_LOAD", mode[0]).replace("APRE_NEXT", mode[1])


# Qh = q e^G, Kh = k e^G (and Kt = k e^-G when WITH_KT) for the k-step at kk
_SCALE_QK = r"""
      mlx::steel::NAXTile<InT, 1, 2> Qb, Kb;
      load_seq(Qb, q_ + kk, Hv * Dk);
      load_seq(Kb, k_ + kk, Hv * Dk);
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < 2; f++) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < mlx::steel::BaseNAXFrag::kElemsPerFrag; e++) {
          const short r = fm + (e >> 2) * 8;
          const short cidx = kk + f * 16 + fn4 + (e & 3);
          const float eg = EG[r * Dk + cidx];
          const float kf = float(Kb.frag_at(0, f)[e]);
          const float qf = float(Qb.frag_at(0, f)[e]);
          K_tile.frag_at(0, f)[e] = kf * eg;
          Q_tile.frag_at(0, f)[e] = qf * eg;
          WITH_KT
        }
      }
"""

# A = beta tril(Kh Kt^T, -1) (select, never multiply: the masked half holds e^{+G} products); Tinv = (I + A)^-1 by the Horner form; QKt = tril(Qh Kt^T) incl. the diagonal
_SOLVE = r"""
    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(A_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
      AT_NAX(A_tile, _i) = (_c.x >= _c.y) ? 0.0f : AT_NAX(KKt_tile, _i) * beta_fm[_i >> 2];
      AT_NAX(QKt_tile, _i) = (_c.x > _c.y) ? 0.0f : AT_NAX(QKt_tile, _i);
    }
    SUB_NAX(Tinv_tile, I_tile, A_tile);
    if constexpr (SPLIT & 4) {
      mlx::steel::NAXTile<float, 1, 1> Ah_, Al_, Th_, Tl_;
      split_t(A_tile, Ah_, Al_);
      STEEL_PRAGMA_UNROLL
      for (int step = 0; step < 15; step++) {
        split_t(Tinv_tile, Th_, Tl_);
        MM16x16x16(TMP_tile, 0, Ah_, false, 0, Th_, false, 0);
        MMA16x16x16(TMP_tile, 0, Ah_, false, 0, Tl_, false, 0);
        MMA16x16x16(TMP_tile, 0, Al_, false, 0, Th_, false, 0);
        SUB_NAX(Tinv_tile, I_tile, TMP_tile);
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int step = 0; step < 15; step++) {
        MM16x16x16(TMP_tile, 0, A_tile, false, 0, Tinv_tile, false, 0);
        SUB_NAX(Tinv_tile, I_tile, TMP_tile);
      }
    }
"""

_LOAD_SEQ = r"""
    auto load_seq = [&](thread auto& tile, auto src, int ld) {
      if constexpr (B) { tile.load_rows(src, ld, valid_rows); } else { tile.load(src, ld); }
    };
"""

_CHUNK_LOOP = r"""
  int t = 0;
  for (; t + C <= T; t += C) {
    process_chunk(C, metal::false_type{});
    a_ += C * Hv * Dk; q_ += C * Hv * Dk; k_ += C * Hv * Dk; v_ += C * Hv * Dv; beta_ += C * Hv; y_ += C * Hv * Dv; chunk_i++;
  }
  if (t < T) { process_chunk(short(T - t), metal::true_type{}); }
"""


def _mm3(c, a, b, ta, tb, co="0", ao="0", bo="0", mma="MMA16x16x32", terms=("hh", "hl", "lh")):
    """hi/lo product c += a b for operands already split into a_h / a_l and b_h / b_l tiles: the listed partial products (default: all but lo*lo)"""
    return "".join(f"        {mma}({c}, {co}, {a}{t[0]}_, {ta}, {ao}, {b}{t[1]}_, {tb}, {bo});\n" for t in terms)


_UPD_TERMS = ("hh", "hl", "lh")


_KKT_BLOCK = r"""if constexpr (SPLIT & 4) {
        mlx::steel::NAXTile<float, 1, 2> Kh_, Kl_, Qh_, Ql_, Th_, Tl_;
        split_t(K_tile, Kh_, Kl_); split_t(Q_tile, Qh_, Ql_); split_t(Kt_tile, Th_, Tl_);
""" + _mm3("KKt_tile", "K", "T", "false", "true") + _mm3("QKt_tile", "Q", "T", "false", "true") + r"""      } else {
        MMA16x16x32(KKt_tile, 0, K_tile, false, 0, Kt_tile, true, 0);
        MMA16x16x32(QKt_tile, 0, Q_tile, false, 0, Kt_tile, true, 0);
      }"""
_KS_BLOCK = r"""mlx::steel::NAXTile<float, 1, 2> Sh_, Sl_;
      if constexpr (SPLIT & 17) { split_sl(S_tile, kk / 16, Sh_, Sl_); }
      if constexpr (SPLIT & 1) {
        mlx::steel::NAXTile<float, 1, 2> Kh_, Kl_;
        split_t(K_tile, Kh_, Kl_);
""" + _mm3("KS_tile", "K", "S", "false", "true") + r"""      } else {
        MMA16x16x32(KS_tile, 0, K_tile, false, 0, S_tile, true, kk / 16);
      }
      if constexpr (SPLIT & 16) {
        mlx::steel::NAXTile<float, 1, 2> Qh_, Ql_;
        split_t(Q_tile, Qh_, Ql_);
""" + _mm3("QS_tile", "Q", "S", "false", "true") + r"""      } else {
        MMA16x16x32(QS_tile, 0, Q_tile, false, 0, S_tile, true, kk / 16);
      }"""

# full kernel: everything inside the sequential chunk loop (4 simdgroups x 16 value rows per threadgroup share the tables)
_SRC = _COMMON_SETUP + r"""
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto dv_idx = thread_position_in_grid.y * 16;                       // this simdgroup owns 16 value rows of the head
  const short sg_id = thread_position_in_threadgroup.y;
  const short tdx = sg_id * 32 + thread_position_in_threadgroup.x;    // 0..127
  constexpr short NT = 128;
  auto a_ = a + b_idx * T * Hv * Dk + hv_idx * Dk;
  auto q_ = q + b_idx * T * Hv * Dk + hv_idx * Dk;
  auto k_ = k + b_idx * T * Hv * Dk + hv_idx * Dk;
  auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto beta_ = beta + b_idx * T * Hv;
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;
  const float decay = adec[hv_idx];
  int chunk_i = 0;
  const int nchunk = (T + C - 1) / C;
  mlx::steel::NAXTile<float, 1, Dk / 16> S_tile;                      // state rows dv_idx..+15 (fp32 master)
  S_tile.load(i_state, Dk);
  float apre[C];                                                      // this thread's channel of the gate pre-activation, 16 rows of the coming chunk
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < C; r++) { apre[r] = (r < T) ? float(a_[r * Hv * Dk + tdx]) : 0.0f; }

  auto process_chunk = [&](const short valid_rows, auto bounded_tag) __attribute__((always_inline)) {
    constexpr bool B = decltype(bounded_tag)::value;
""" + _LOAD_SEQ + r"""
    // ---- gate tables
""" + _tab(_APRE_CARRY) + r"""
    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;
    // one pass over the K / Q k-steps: Kh Kt^T, Qh Kt^T, Kh S^T, Qh S^T (fp32 tiles)
    KKt_tile.clear();
    QKt_tile.clear();
    KS_tile.clear();
    QS_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
""" + _SCALE_QK.replace("WITH_KT", "Kt_tile.frag_at(0, f)[e] = kf * EN[r * Dk + cidx];") + r"""
KKT_BLOCK
      KS_BLOCK
    }
""" + _SOLVE + r"""
    STATE_TAIL
  };
""" + _CHUNK_LOOP + r"""
  S_tile.store(o_state, Dk);
"""

# state-dependent tail shared by the full and the main kernel: delta, out, state update
_STATE_TAIL = r"""
    // delta = Tinv (beta V - beta Kh S^T);  out = Qh S^T + tril(Qh Kt^T) delta
    SCALE_BETA_NAX(KS_tile, beta_fm);
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    SUB_NAX(TMP_tile, V_tile, KS_tile);
    out_tile = QS_tile;
    if constexpr (SPLIT & 8) {
      mlx::steel::NAXTile<float, 1, 1> Th_, Tl_, Xh_, Xl_, Dh_, Dl_, Qh_, Ql_;
      split_t(Tinv_tile, Th_, Tl_); split_t(TMP_tile, Xh_, Xl_);
      MM16x16x16(delta_tile, 0, Th_, false, 0, Xh_, false, 0);
      MMA16x16x16(delta_tile, 0, Th_, false, 0, Xl_, false, 0);
      MMA16x16x16(delta_tile, 0, Tl_, false, 0, Xh_, false, 0);
      split_t(delta_tile, Dh_, Dl_); split_t(QKt_tile, Qh_, Ql_);
      MMA16x16x16(out_tile, 0, Qh_, false, 0, Dh_, false, 0);
      MMA16x16x16(out_tile, 0, Qh_, false, 0, Dl_, false, 0);
      MMA16x16x16(out_tile, 0, Ql_, false, 0, Dh_, false, 0);
    } else {
      MM16x16x16(delta_tile, 0, Tinv_tile, false, 0, TMP_tile, false, 0);
      MMA16x16x16(out_tile, 0, QKt_tile, false, 0, delta_tile, false, 0);
    }
    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(out_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
      if (_c.y < valid_rows) { y_[_c.y * Hv * Dv + dv_idx + _c.x] = static_cast<InT>(AT_NAX(out_tile, _i)); }
    }
    // S <- S diag(e^{G_C}) + delta^T (K e^{G_C - G_j})
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < Dk / 16; f++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < mlx::steel::BaseNAXFrag::kElemsPerFrag; e++) {
        S_tile.frag_at(0, f)[e] *= EL[f * 16 + fn4 + (e & 3)];
      }
    }
    mlx::steel::NAXTile<float, 1, 1> DDh_, DDl_;
    if constexpr (SPLIT & 32) {                                       // delta^T straight from the solve: the update MMAs then take a plain (not transposed) A operand
      mlx::steel::NAXTile<float, 1, 1> Th2_, Tl2_, Xh2_, Xl2_, dT_;
      split_t(Tinv_tile, Th2_, Tl2_); split_t(TMP_tile, Xh2_, Xl2_);
      MM16x16x16(dT_, 0, Xh2_, true, 0, Th2_, true, 0);
      MMA16x16x16(dT_, 0, Xh2_, true, 0, Tl2_, true, 0);
      MMA16x16x16(dT_, 0, Xl2_, true, 0, Th2_, true, 0);
      split_t(dT_, DDh_, DDl_);
    } else if constexpr (SPLIT & 2) { split_t(delta_tile, DDh_, DDl_); }
    for (int kk = 0; kk < Dk; kk += 32) {
      mlx::steel::NAXTile<InT, 1, 2> Kb;
      load_seq(Kb, k_ + kk, Hv * Dk);
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < 2; f++) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < mlx::steel::BaseNAXFrag::kElemsPerFrag; e++) {
          const short r = fm + (e >> 2) * 8;
          const short cidx = kk + f * 16 + fn4 + (e & 3);
          K_tile.frag_at(0, f)[e] = float(Kb.frag_at(0, f)[e]) * EL[cidx] * EN[r * Dk + cidx];
        }
      }
      if constexpr (SPLIT & 34) {
        mlx::steel::NAXTile<float, 1, 2> Kh_, Kl_;
        split_t(K_tile, Kh_, Kl_);
        if constexpr (SPLIT & 64) {                                   // three independent products (no accumulator dependency chain), summed afterwards
          mlx::steel::NAXTile<float, 1, 2> Pa_, Pb_, Pc_;
          if constexpr (SPLIT & 32) {
            MM16x32x16(Pa_, 0, DDh_, false, 0, Kh_, false, 0);
            MM16x32x16(Pb_, 0, DDh_, false, 0, Kl_, false, 0);
            MM16x32x16(Pc_, 0, DDl_, false, 0, Kh_, false, 0);
          } else {
            MM16x32x16(Pa_, 0, DDh_, true, 0, Kh_, false, 0);
            MM16x32x16(Pb_, 0, DDh_, true, 0, Kl_, false, 0);
            MM16x32x16(Pc_, 0, DDl_, true, 0, Kh_, false, 0);
          }
          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < 2; f++) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < mlx::steel::BaseNAXFrag::kElemsPerFrag; e++) {
              S_tile.frag_at(0, kk / 16 + f)[e] += (Pb_.frag_at(0, f)[e] + Pc_.frag_at(0, f)[e]) + Pa_.frag_at(0, f)[e];
            }
          }
        } else if constexpr (SPLIT & 32) {
""" + _mm3("S_tile", "DD", "K", "false", "false", co="kk / 16", mma="MMA16x32x16", terms=_UPD_TERMS) + r"""        } else {
""" + _mm3("S_tile", "DD", "K", "true", "false", co="kk / 16", mma="MMA16x32x16", terms=_UPD_TERMS) + r"""        }
      } else {
        MMA16x32x16(S_tile, kk / 16, delta_tile, true, 0, K_tile, false, 0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                  // the tables are rewritten by the next chunk
"""
_SRC = _SRC.replace("STATE_TAIL", _STATE_TAIL)

# prep kernel (one simdgroup per head and chunk, all chunks in parallel): Tinv and tril(Qh Kt^T), stored as fp32 [B, nchunk, H, 16, 16]
_SRC_PREP = _COMMON_SETUP + r"""
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  const int chunk = thread_position_in_grid.y;
  const int nchunk = (T + C - 1) / C;
  const int t0 = chunk * C;
  const short valid_rows = short(min(C, T - t0));
  const short tdx = thread_position_in_threadgroup.x;
  constexpr short NT = 32;
  auto a_ = a + (b_idx * T + t0) * Hv * Dk + hv_idx * Dk;
  auto q_ = q + (b_idx * T + t0) * Hv * Dk + hv_idx * Dk;
  auto k_ = k + (b_idx * T + t0) * Hv * Dk + hv_idx * Dk;
  auto beta_ = beta + (b_idx * T + t0) * Hv;
  const float decay = adec[hv_idx];
  const bool B = valid_rows < C;
""" + _LOAD_SEQ.replace("if constexpr (B)", "if (B)") + r"""
""" + _tab(_APRE_DIRECT) + r"""
  beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
  beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;
  KKt_tile.clear();
  QKt_tile.clear();
  for (int kk = 0; kk < Dk; kk += 32) {
""" + _SCALE_QK.replace("WITH_KT", "Kt_tile.frag_at(0, f)[e] = kf * EN[r * Dk + cidx];") + r"""
KKT_BLOCK
  }
""" + _SOLVE + r"""
  auto o_ = (b_idx * nchunk + chunk) * Hv + hv_idx;
  Tinv_tile.store(tinv + o_ * 256, 16);
  QKt_tile.store(qkt + o_ * 256, 16);
"""

# main kernel after the prep: only the state-dependent MMAs; Tinv and QKt come from the prep outputs
_SRC_MAIN = _SRC.replace(r"""    KKt_tile.clear();
    QKt_tile.clear();
    KS_tile.clear();""", """    KS_tile.clear();""").replace(
    _SCALE_QK.replace("WITH_KT", "Kt_tile.frag_at(0, f)[e] = kf * EN[r * Dk + cidx];"), _SCALE_QK.replace("WITH_KT", "")).replace(
    "KKT_BLOCK\n", "").replace(
    _SOLVE, r"""
    {
      auto pm = (b_idx * nchunk + chunk_i) * Hv + hv_idx;
      Tinv_tile.load(tinv + pm * 256, 16);
      QKt_tile.load(qkt + pm * 256, 16);
    }
""")


# simdgroups (16 value rows each) per threadgroup of the main scan: 4 = the shipped shape (128 threads, one gate channel each, next chunk's gate loads carried in registers);
# 2 / 1 = more, smaller threadgroups (64 / 32 threads, 2 / 4 channels per thread, direct gate loads): more resident groups per core to hide the latency of the serial chunk chain
NSG_OK = (1, 2, 4)                                                                    # the only shapes the main kernel is written for (Dk / NT channels per thread: NSG = 8 gives 0 and never writes the gate tables)


def _env_nsg():
    try:
        return int(os.environ.get("KYOJIN_KDA_SCAN_NSG", "4"))
    except ValueError:
        return 0                                                                      # not a number: nsg_eff() takes the shipped 4 and says why


NSG = _env_nsg()
_SRC_MAIN_D = _SRC_MAIN.replace(_tab(_APRE_CARRY), _tab(_APRE_DIRECT))
assert _SRC_MAIN_D != _SRC_MAIN
EXP = os.environ.get("KYOJIN_GLM_KDA_EXP", "fast")                                   # exp of the gate tables: fast (exp2 based, ~5e-6 relative) | precise
PREP = os.environ.get("KYOJIN_GLM_KDA_PREP", "1") != "0"                   # 1: Tinv / tril(Qh Kt^T) of all chunks in a parallel pre-pass; 0: inside the sequential scan


SPLIT = int(os.environ.get("KYOJIN_GLM_KDA_SPLIT", "127"))                 # bitmask of the MMAs run as 3-term hi/lo products (fp32-accurate): 1 Kh S^T, 16 Qh S^T, 2 state update, 4 pre-pass, 8 tail


def nsg_eff(nsg=None):
    """(simdgroups the scan launches with, None) when NSG is one of 1 / 2 / 4, else (4, reason): an unvalidated NSG = 8 builds a kernel with Dk / NT = 0 channels per thread whose
    gate tables are never written (silent garbage), so anything else takes the shipped 4 and the flag audit says why (key kda_scan_nsg)."""
    n = NSG if nsg is None else nsg
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 4, f"nsg={n}"
    return (n, None) if n in NSG_OK else (4, f"nsg={n}")


def set_nsg(nsg):
    """simdgroups per threadgroup of the main scan at runtime (benches); the kernels are rebuilt under a new name"""
    global NSG
    NSG = int(nsg)
    pass                                                                # _kernels_for is keyed by (SPLIT, NSG)


def set_split(split):
    """change the split mask at runtime (tests / benches); the kernels are rebuilt under a new name"""
    global SPLIT
    SPLIT = int(split)
    pass                                                                # _kernels_for is keyed by (SPLIT, NSG)


def _x(src, nsg=4):
    src = src.replace("constexpr short NT = 128;", f"constexpr short NT = {32 * nsg};").replace("KKT_BLOCK", _KKT_BLOCK).replace("KS_BLOCK", _KS_BLOCK).replace("SPLITV", str(SPLIT))
    return GN._norefs(src).replace("EXPF", "metal::fast::exp" if EXP == "fast" else "metal::precise::exp")


def _kernels():
    return _kernels_for(SPLIT, nsg_eff()[0])


@functools.lru_cache(maxsize=None)
def _kernels_for(SPLIT, NSG):
    hdr = GN.header() + _HELP + "\n"
    names = ["q", "k", "v", "a", "beta", "adec", "dtb", "lb", "state_in", "T"]
    full = mx.fast.metal_kernel(name=f"kyojin_kda_nax_s{SPLIT}", input_names=names, output_names=["y", "state_out"], header=hdr, source=_x(_SRC))
    main = mx.fast.metal_kernel(name=f"kyojin_kda_nax_main_s{SPLIT}" + ("" if NSG == 4 else f"_g{NSG}"), input_names=names + ["tinv", "qkt"], output_names=["y", "state_out"], header=hdr,
                                source=_x(_SRC_MAIN if NSG == 4 else _SRC_MAIN_D, NSG))
    prep = mx.fast.metal_kernel(name=f"kyojin_kda_nax_prep_s{SPLIT}", input_names=["q", "k", "a", "beta", "adec", "dtb", "lb", "T"], output_names=["tinv", "qkt"], header=hdr,
                                source=_x(_SRC_PREP))
    return full, main, prep


def kda_nax(q, k, v, a, beta, a_log, dt_bias, lower_bound, state, prep=None):
    """q, k, v, a [B,T,H,128] (bf16/fp16; a = raw f_b_proj output), beta [B,T,H] (sigmoid(b), cast to fp32), a_log [H] or [H,1], dt_bias [H,128] or [H*128],
    state [B,H,128,128] fp32 (Dv rows, Dk cols) -> y [B,T,H,128] (q dtype), state_out fp32. Safe gate only: lower_bound in [-5, 0) so that |G| <= 80 inside a 16-token chunk."""
    B, T, H, Dk = k.shape
    Dv = v.shape[3]
    assert Dk == 128 and Dv == 128 and v.shape[2] == H and q.shape[2] == H and a.shape == k.shape and lower_bound < 0
    prep = PREP if prep is None else prep
    nsg, why = nsg_eff()
    if why is not None:
        FD.fb("kda_scan_nsg", why)
    full, main, pre = _kernels_for(SPLIT, nsg)
    adec = mx.exp(a_log.astype(mx.float32)).reshape(H)
    beta = beta.astype(mx.float32)
    dtb = dt_bias.astype(mx.float32).reshape(H, Dk)
    lb = mx.array([lower_bound], dtype=mx.float32)
    tpl = [("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hv", H)]
    ins = [q, k, v, a, beta, adec, dtb, lb, state, T]
    nchunk = (T + 15) // 16
    if prep:
        tinv, qkt = pre(inputs=[q, k, a, beta, adec, dtb, lb, T], template=tpl, grid=(32, nchunk, B * H), threadgroup=(32, 1, 1),
                        output_shapes=[(B, nchunk, H, 16, 16)] * 2, output_dtypes=[mx.float32] * 2)
        ins += [tinv, qkt]
    return (main if prep else full)(inputs=ins, template=tpl, grid=(32, Dv // 16, B * H), threadgroup=(32, nsg if prep else 4, 1),
                                    output_shapes=[(B, T, H, Dv), state.shape], output_dtypes=[q.dtype, mx.float32])
