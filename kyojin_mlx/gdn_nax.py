"""Fork of MLX's NAX chunked gated-delta prefill kernel (mlx/backend/metal/kernels/gated_delta_update_nax.h) as an mx.fast.metal_kernel.
Differences from mx.fast.gated_delta_update: g and beta stay fp32 (the stock op casts them to the q dtype: bf16 g near 1 is a 4e-3 grid), no vjp buffers,
one K / Q load pass per chunk, exp factors hoisted, delta = Tinv_s (beta V - e^gamma (beta K) S^T): no 16x128 W panel).
The MLX headers are read from the installed package and inlined (not vendored)."""
import os, re, functools
import mlx.core as mx

_INC = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")

def _read(rel):
    t = open(os.path.join(_INC, rel)).read()
    t = re.sub(r'^\s*#\s*include\s+"mlx/[^"]*"\s*$', "", t, flags=re.M)
    return t.replace("#pragma once", "")

def _norefs(t):
    """decltype(x)::member also works when x is an array element (a reference type)."""
    return re.sub(r"decltype\((\w+)\)::", r"metal::remove_reference_t<decltype(\1)>::", t)

@functools.lru_cache(maxsize=None)
def header():
    h = "\n".join(_read(f) for f in ("steel/defines.h", "steel/utils/type_traits.h", "steel/utils/integral_constant.h", "steel/gemm/nax.h", "gated_delta_nax_ops.h"))
    h = _norefs(h)
    return "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp;\nusing namespace mpp::tensor_ops;\n" + h

_SRC = r"""
  constexpr int C = 16;                                               // chunk rows (one 16x16 NAX fragment)
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);
  auto dv_idx = thread_position_in_grid.y * 16;                       // this simdgroup owns 16 value rows of the head
  const short sg_id = thread_position_in_threadgroup.y;
  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
  const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;

  auto g_ = g + b_idx * T * Hv;
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto beta_ = beta + b_idx * T * Hv;
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  threadgroup float gamma_all[C * 8];
  threadgroup float* gamma = gamma_all + sg_id * C;
  float beta_fm[2];

  mlx::steel::NAXTile<float, 1, Dk / 16> S_tile;                      // state rows dv_idx..+15 (fp32 master)
  S_tile.load(i_state, Dk);
  mlx::steel::NAXTile<float, 1, 2> K_tile, Q_tile;
  mlx::steel::NAXTile<float, 1, 1> V_tile, KS_tile, QS_tile, delta_tile, QKt_tile, out_tile, Tinv_tile, A_tile, KKt_tile, I_tile, TMP_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
    AT_NAX(I_tile, _i) = (_c.x == _c.y) ? 1.0f : 0.0f;
  }

  auto process_chunk = [&](const short valid_rows, auto bounded_tag) __attribute__((always_inline)) {
    constexpr bool B = decltype(bounded_tag)::value;
    auto load_seq = [&](thread auto& tile, auto src, int ld) {
      if constexpr (B) { tile.load_rows(src, ld, valid_rows); } else { tile.load(src, ld); }
    };
    // gamma_i = sum_{j<=i} log g_j (inclusive prefix over the chunk rows)
    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(metal::max(float(g_[thread_index_in_simdgroup * Hv + hv_idx]), 1e-6)) : 0.0f;
    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) { gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val); }

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;
    float eg[2], dk2[2];                                              // e^gamma_row and e^(gamma_last - gamma_row) for the lane's two fragment rows
    eg[0] = metal::fast::exp(gamma[fm]); eg[1] = metal::fast::exp(gamma[fm1]);
    dk2[0] = metal::fast::exp(gamma[C - 1] - gamma[fm]); dk2[1] = metal::fast::exp(gamma[C - 1] - gamma[fm1]);

    // one pass over the K / Q k-steps: K K^T and Q K^T (input-dtype tiles: exact products), (beta K) S^T and Q S^T (fp32 tiles) share the loads
    KKt_tile.clear();
    QKt_tile.clear();
    KS_tile.clear();
    QS_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      mlx::steel::NAXTile<InT, 1, 2> Qb, Kb;
      load_seq(Qb, q_ + kk, Hk * Dk);
      load_seq(Kb, k_ + kk, Hk * Dk);
      MMA16x16x32(KKt_tile, 0, Kb, false, 0, Kb, true, 0);
      MMA16x16x32(QKt_tile, 0, Qb, false, 0, Kb, true, 0);
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < 2; f++) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < mlx::steel::BaseNAXFrag::kElemsPerFrag; e++) {
          K_tile.frag_at(0, f)[e] = float(Kb.frag_at(0, f)[e]);
          Q_tile.frag_at(0, f)[e] = float(Qb.frag_at(0, f)[e]);
        }
      }
      SCALE_BETA_NAX(K_tile, beta_fm);
      MMA16x16x32(KS_tile, 0, K_tile, false, 0, S_tile, true, kk / 16);
      MMA16x16x32(QS_tile, 0, Q_tile, false, 0, S_tile, true, kk / 16);
    }

    // A = beta tril(K K^T, -1);  Tinv = (I + A)^-1 = (I - A)(I + A^2)(I + A^4)(I + A^8)  (A^16 = 0: 6 products instead of 15 dependent steps)
    A_tile = KKt_tile;
    SCALE_TRIEQ_NAX1(A_tile, beta_fm);
    SUB_NAX(Tinv_tile, I_tile, A_tile);
    // Horner form T <- I - A T (15 steps, A^16 = 0): self-correcting. The product form (I - A)(I + A^2)(I + A^4)(I + A^8) is 6 products but
    // blows up (1e10) on strongly correlated keys, where A^8 has huge entries and the rounding of the relaxed fp32 MMA is amplified.
    STEEL_PRAGMA_UNROLL
    for (int step = 0; step < 15; step++) {
      MM16x16x16(TMP_tile, 0, A_tile, false, 0, Tinv_tile, false, 0);
      SUB_NAX(Tinv_tile, I_tile, TMP_tile);
    }
    SCALE_TRI_NAX(Tinv_tile, gamma)                                   // Tinv_s[i][j] = Tinv[i][j] e^(gamma_i - gamma_j), j <= i
    SCALE_TRI_NAX(QKt_tile, gamma)

    // delta = Tinv_s (beta V - e^gamma (beta K) S^T);  out = e^gamma (Q S^T) + QKt_s delta
    SCALE_BETA_NAX(KS_tile, eg);
    SCALE_BETA_NAX(QS_tile, eg);
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    SUB_NAX(TMP_tile, V_tile, KS_tile);
    MM16x16x16(delta_tile, 0, Tinv_tile, false, 0, TMP_tile, false, 0)
    out_tile = QS_tile;
    MMA16x16x16(out_tile, 0, QKt_tile, false, 0, delta_tile, false, 0);
    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(out_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
      if (_c.y < valid_rows) { y_[_c.y * Hv * Dv + dv_idx + _c.x] = static_cast<InT>(AT_NAX(out_tile, _i)); }
    }

    // S <- e^gamma_last S + delta^T (K e^(gamma_last - gamma_i))
    SCALE_NAX(S_tile, metal::fast::exp(gamma[C - 1]));
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Hk * Dk);
      SCALE_BETA_NAX(K_tile, dk2);
      MMA16x32x16(S_tile, kk / 16, delta_tile, true, 0, K_tile, false, 0);
    }
  };

  int t = 0;
  for (; t + C <= T; t += C) {
    process_chunk(C, metal::false_type{});
    q_ += C * Hk * Dk; k_ += C * Hk * Dk; v_ += C * Hv * Dv; beta_ += C * Hv; y_ += C * Hv * Dv; g_ += C * Hv;
  }
  if (t < T) { process_chunk(short(T - t), metal::true_type{}); }
  S_tile.store(o_state, Dk);
"""

@functools.lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(name="kyojin_gdn_nax", input_names=["q", "k", "v", "g", "beta", "state_in", "T"], output_names=["y", "state_out"],
                                header=header() + "\n", source=_norefs(_SRC))

def gated_delta_nax(q, k, v, g, beta, state):
    """q, k [B,T,Hk,128], v [B,T,Hv,128] (bf16/fp16/fp32), g beta [B,T,Hv] fp32, state [B,Hv,128,128] fp32 -> y (q dtype), state_out fp32."""
    B, T, Hk, Dk = q.shape; Hv, Dv = v.shape[2], v.shape[3]
    return _kernel()(inputs=[q, k, v, g.astype(mx.float32), beta.astype(mx.float32), state, T],
        template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)], grid=(32, Dv // 16, B * Hv), threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, mx.float32])
