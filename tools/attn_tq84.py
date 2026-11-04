"""hd=256 flash prefill, d-split with TQ q row-blocks per simdgroup (every K / V tile loaded from memory feeds TQ matmuls instead of 1).
Own kernel body, written after MLX's attention_nax_dsplit (same masks / online softmax, causal + GQA, no mask array, no sinks, no lse). Headers inlined from the installed mlx."""
import sys, time, functools
import mlx.core as mx
import attn_ds84 as A

BODY = r"""
  ulong3 tidl{tid.x, tid.y, tid.z};
  Q += tidl.z * params->Q_strides[0] + tidl.y * params->Q_strides[1] + tidl.x * BQ * params->Q_strides[2];
  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += tidl.z * params->V_strides[0] + kv_head_idx * params->V_strides[1];
  O += tidl.z * params->O_strides[0] + tidl.y * params->O_strides[1] + tidl.x * BQ * params->O_strides[2];
  const metal::uniform<float> scale2 = make_uniform(params->scale) * make_uniform(1.44269504089f);

  constexpr short kU = 16;
  constexpr int TD = BD / kU;
  constexpr short TK = BK / kU;
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;
  static_assert(BQ == WM * TQ * kU, "BQ = WM * TQ * 16");
  static_assert(TDh % 2 == 0 && TK % 2 == 0, "pairs");

  const short row_group = simd_group_id / WN;
  const short d_group = simd_group_id % WN;
  using otile_t = NAXTile<AccumType, TQ, TDh>;
  otile_t Otile;
  Otile.clear();
  const short tm = kU * TQ * row_group;
  Q += tm * int(params->Q_strides[2]) + d_group * BDh;
  K += d_group * BDh;
  V += d_group * BDh;
  O += tm * int(params->O_strides[2]) + d_group * BDh;

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { max_score[i] = Limits<AccumType>::finite_min; }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;
  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = min(params->NK, (q_max + BK - 1) / BK);
    int q_min = max(0, int(tid.x) * BQ + params->qL_off);
    kb_min_causal = (q_min / BK);
  }
  const bool is_last_q = int(tid.x) == (params->NQ_aligned);
  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  using stile_t = NAXTile<AccumType, TQ, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
  threadgroup AccumType s_xchg[WM][WN][TQ * TK * kEPF * 32];

  NAXTile<T, 1, 1> Qtiles[TQ][TDh];
  STEEL_PRAGMA_UNROLL
  for (short tq = 0; tq < TQ; tq++) {
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id++) {
      const int off = tq * kU * int(params->Q_strides[2]) + id * kU;
      if (!align_Q && is_last_q) { Qtiles[tq][id].load_rows(Q + off, int(params->Q_strides[2]), lim_rows_q - tq * kU); }
      else { Qtiles[tq][id].load(Q + off, int(params->Q_strides[2])); }
    }
  }
  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));
    stile_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;
        if (!align_K && is_last_k) { Ktile.load_rows(K + K_load_off, int(params->K_strides[2]), lim_rows_k - ik * kU); }
        else { Ktile.load(K + K_load_off, int(params->K_strides[2])); }
        STEEL_PRAGMA_UNROLL
        for (short tq = 0; tq < TQ; tq++) {
          stile_t::NAXFrag_t::mma(Stile.frag_at(tq, ik), Stile.frag_at(tq, ik + 1), Qtiles[tq][id].frag_at(0, 0), metal::false_type{},
                                  Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
        }
      }
    }
    if (WN > 1) {
      threadgroup AccumType* slot = s_xchg[row_group][d_group];
      const short base = short(simd_lane_id) * (TQ * TK * kEPF);
      STEEL_PRAGMA_UNROLL
      for (short tq = 0; tq < TQ; tq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& s = Stile.frag_at(tq, ik);
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kEPF; i++) { slot[base + (tq * TK + ik) * kEPF + i] = s[i]; }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short tq = 0; tq < TQ; tq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& s = Stile.frag_at(tq, ik);
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kEPF; i++) { s[i] = s_xchg[row_group][0][base + (tq * TK + ik) * kEPF + i]; }
          STEEL_PRAGMA_UNROLL
          for (short pg = 1; pg < WN; pg++) {
            const threadgroup AccumType* peer = s_xchg[row_group][pg];
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kEPF; i++) { s[i] += peer[base + (tq * TK + ik) * kEPF + i]; }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) { Stile.elems()[ii] *= float(scale2); }

    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short tq = 0; tq < TQ; tq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(tq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short tq = 0; tq < TQ; tq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(tq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r = base_row + tq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { new_max[i] = max_score[i]; }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { factor[i] = fast::exp2(max_score[i] - new_max[i]); max_score[i] = new_max[i]; }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { sum_score[i] = sum_score[i] * factor[i]; }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);
    simdgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;
        if (!align_K && is_last_k) { Vtile.load_rows(V + V_load_off, int(params->V_strides[2]), lim_rows_k - ik * kU); }
        else { Vtile.load(V + V_load_off, int(params->V_strides[2])); }
        STEEL_PRAGMA_UNROLL
        for (short tq = 0; tq < TQ; tq++) {
          otile_t::NAXFrag_t::mma(Otile.frag_at(tq, id), Otile.frag_at(tq, id + 1), Stile.frag_at(tq, ik), metal::false_type{},
                                  Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
        }
      }
    }
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  threadgroup_barrier(mem_flags::mem_none);
  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { rcp[i] = 1.f / sum_score[i]; }
  Otile.template row_bin_op<MulOp>(rcp);
  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0) return;
    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
"""

BODY2 = r"""
  ulong3 tidl{tid.x, tid.y, tid.z};
  Q += tidl.z * params->Q_strides[0] + tidl.y * params->Q_strides[1] + tidl.x * BQ * params->Q_strides[2];
  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += tidl.z * params->V_strides[0] + kv_head_idx * params->V_strides[1];
  O += tidl.z * params->O_strides[0] + tidl.y * params->O_strides[1] + tidl.x * BQ * params->O_strides[2];
  const metal::uniform<float> scale2 = make_uniform(params->scale) * make_uniform(1.44269504089f);

  constexpr short kU = 16;
  constexpr int TD = BD / kU;
  constexpr short TK = BK / kU;
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;
  static_assert(BQ == WM * TQ * kU, "BQ = WM * TQ * 16");
  static_assert(TDh % 2 == 0 && TK % 2 == 0, "pairs");

  const short row_group = simd_group_id / WN;
  const short d_group = simd_group_id % WN;
  using otile_t = NAXTile<AccumType, TQ, TDh>;
  otile_t Otile;
  Otile.clear();
  const short tm = kU * TQ * row_group;
  Q += tm * int(params->Q_strides[2]) + d_group * BDh;
  K += d_group * BDh;
  V += d_group * BDh;
  O += tm * int(params->O_strides[2]) + d_group * BDh;

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { max_score[i] = Limits<AccumType>::finite_min; }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;
  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = min(params->NK, (q_max + BK - 1) / BK);
    int q_min = max(0, int(tid.x) * BQ + params->qL_off);
    kb_min_causal = (q_min / BK);
  }
  const bool is_last_q = int(tid.x) == (params->NQ_aligned);
  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  using stile_t = NAXTile<AccumType, TQ, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
  threadgroup AccumType s_xchg[2][WM][WN][TQ * TK * kEPF * 32];

  NAXTile<T, 1, 1> Qtiles[TQ][TDh];
  STEEL_PRAGMA_UNROLL
  for (short tq = 0; tq < TQ; tq++) {
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id++) {
      const int off = tq * kU * int(params->Q_strides[2]) + id * kU;
      if (!align_Q && is_last_q) { Qtiles[tq][id].load_rows(Q + off, int(params->Q_strides[2]), lim_rows_q - tq * kU); }
      else { Qtiles[tq][id].load(Q + off, int(params->Q_strides[2])); }
    }
  }
  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // software pipeline: the partial scores of block kb+1 are computed and published while the peers still reduce block kb (double-buffered exchange, one barrier per block)
  auto partial_scores = [&](int kbi, thread stile_t& S) {
    const bool last_k = (kbi == (params->NK_aligned));
    const device T* Kb = K + kbi * BK * int(params->K_strides[2]);
    S.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;
        if (!align_K && last_k) { Ktile.load_rows(Kb + K_load_off, int(params->K_strides[2]), lim_rows_k - ik * kU); }
        else { Ktile.load(Kb + K_load_off, int(params->K_strides[2])); }
        STEEL_PRAGMA_UNROLL
        for (short tq = 0; tq < TQ; tq++) {
          stile_t::NAXFrag_t::mma(S.frag_at(tq, ik), S.frag_at(tq, ik + 1), Qtiles[tq][id].frag_at(0, 0), metal::false_type{},
                                  Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
        }
      }
    }
  };
  const short xbase = short(simd_lane_id) * (TQ * TK * kEPF);
  auto publish = [&](int buf, thread stile_t& S) {
    threadgroup AccumType* slot = s_xchg[buf][row_group][d_group];
    STEEL_PRAGMA_UNROLL
    for (short tq = 0; tq < TQ; tq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& s = S.frag_at(tq, ik);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) { slot[xbase + (tq * TK + ik) * kEPF + i] = s[i]; }
      }
    }
  };
  stile_t Snext;
  partial_scores(0, Snext);
  publish(0, Snext);
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));
    const int buf = kb & 1;
    stile_t Stile;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short tq = 0; tq < TQ; tq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& s = Stile.frag_at(tq, ik);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) { s[i] = s_xchg[buf][row_group][0][xbase + (tq * TK + ik) * kEPF + i]; }
        STEEL_PRAGMA_UNROLL
        for (short pg = 1; pg < WN; pg++) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kEPF; i++) { s[i] += s_xchg[buf][row_group][pg][xbase + (tq * TK + ik) * kEPF + i]; }
        }
      }
    }
    if (kb + 1 < kb_lim) { partial_scores(kb + 1, Snext); publish(buf ^ 1, Snext); }
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) { Stile.elems()[ii] *= float(scale2); }

    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short tq = 0; tq < TQ; tq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(tq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short tq = 0; tq < TQ; tq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(tq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r = base_row + tq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { new_max[i] = max_score[i]; }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { factor[i] = fast::exp2(max_score[i] - new_max[i]); max_score[i] = new_max[i]; }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) { sum_score[i] = sum_score[i] * factor[i]; }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);
    simdgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;
        const int V_load_off = (kb * BK + ik * kU) * int(params->V_strides[2]) + id * kU;
        if (!align_K && is_last_k) { Vtile.load_rows(V + V_load_off, int(params->V_strides[2]), lim_rows_k - ik * kU); }
        else { Vtile.load(V + V_load_off, int(params->V_strides[2])); }
        STEEL_PRAGMA_UNROLL
        for (short tq = 0; tq < TQ; tq++) {
          otile_t::NAXFrag_t::mma(Otile.frag_at(tq, id), Otile.frag_at(tq, id + 1), Stile.frag_at(tq, ik), metal::false_type{},
                                  Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_none);
  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) { rcp[i] = 1.f / sum_score[i]; }
  Otile.template row_bin_op<MulOp>(rcp);
  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0) return;
    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
"""

@functools.lru_cache(maxsize=None)
def _kernel(name, align_q, align_k, body):
    return mx.fast.metal_kernel(name=name, input_names=["Q", "K", "V", "params_buf"], output_names=["O"], header=A.header(align_q, align_k), source=A.PRE + body)

def attention(q, k, v, scale, TQ=2, BK=32, WM=2, WN=4, body=BODY, tag="tq"):
    B, H, qL, D = q.shape; Hkv, kL = k.shape[1], k.shape[2]
    BQ = WM * TQ * 16
    kern = _kernel(f"attn_tq84_{tag}_{TQ}_{BK}_{WM}_{WN}", qL % BQ == 0, kL % BK == 0, body)
    p = A.make_params(B, H, Hkv, qL, kL, D, BQ, BK, scale)
    NQ = (qL + BQ - 1) // BQ
    return kern(inputs=[q, k, v, p], template=[("T", q.dtype), ("BQ", BQ), ("BK", BK), ("BD", D), ("WM", WM), ("WN", WN), ("TQ", TQ)],
                grid=(NQ * 32 * WM * WN, H, B), threadgroup=(32 * WM * WN, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])[0]

if __name__ == "__main__":
    o = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    L = 2048; N = o + L
    q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16); k = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); mx.eval(q, k, v)
    S = mx.fast.scaled_dot_product_attention
    ref = S(q, k, v, scale=0.0625, mask="causal"); mx.eval(ref)
    fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
    cfgs = [("tq", BODY, (1, 32, 4, 2)), ("pipe", BODY2, (1, 32, 4, 2)), ("pipe", BODY2, (1, 32, 4, 4)), ("pipe", BODY2, (1, 32, 2, 2)), ("pipe", BODY2, (1, 64, 4, 2))]
    for tag, body, c in cfgs:
        try:
            y = attention(q, k, v, 0.0625, *c, body=body, tag=tag); mx.eval(y)
            d = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max())
            t = A.bench(lambda: attention(q, k, v, 0.0625, *c, body=body, tag=tag)); tm = A.bench(lambda: S(q, k, v, scale=0.0625, mask="causal"))
            print(f"{tag} TQ{c[0]} BK{c[1]} WM{c[2]} WN{c[3]}: {t*1e3:6.2f} ms {fl/t/1e12:5.2f} TF  (mlx {fl/tm/1e12:5.2f})  max|d| {d:.2g}", flush=True)
        except Exception as e:
            print(tag, c, "ERR", str(e).split("error:")[-1][:160].strip().replace("\n", " "), flush=True)
