# SPDX-License-Identifier: Apache-2.0
"""Decode-once fused EXL3 + NAX dense GEMM with the output Hadamard in the epilogue (prefill, 64..~1024 rows).

Like `metal_nax_sh` (the weights of a k step are decoded once per threadgroup into threadgroup memory, then every simdgroup runs `matmul2d` against the shared tile), but a
threadgroup owns one whole 128-column output block: 4 column slabs of 32 x NRB row blocks of 64 rows, one 64 x 32 `matmul2d` accumulator per simdgroup. The finalize of the
dense path (`kernels.fin_kernel`: y = H128(round_fp16(acc)) / sqrt(128) * svh) then runs on the accumulators through threadgroup memory, so the fp16 inner product never
reaches memory and one launch disappears. Same arithmetic as the unfused pair: bit-identical."""
from __future__ import annotations

import functools
import itertools

import mlx.core as mx

from . import cbill as CB
from . import metal_nax as NX
from . import metal_src as S

_SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint KT = params[0], NT = params[1], M = params[2];
    const uint nt0 = threadgroup_position_in_grid.x * 8;
    if (nt0 >= NT) return;
    const uint cs = sg & 3u, rb = sg >> 2;                                  // 32-column slab and 64-row block of this simdgroup
    const uint mg = (threadgroup_position_in_grid.y * NRB + rb) * 64;
    const bool active = mg < M;
    const ulong KD = ulong(KT) * 16, N = ulong(NT) * 16;
    const uint n0 = nt0 * 16;
    const uint G = lane >> 3, nb = lane & 7;
    threadgroup half btg[2][128 * BKSV];
    constexpr auto desc = matmul2d_descriptor(64, 32, BKSV, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)A, dextents<int32_t, 2>(int(KD), int(M)));
    auto tA0 = tA.slice(0, int(min(mg, M - 1)));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB0((threadgroup half*)btg[0], dextents<int32_t, 2>(BKSV, 32));
    auto cT = op.get_destination_cooperative_tensor<mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>, mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(tB0)>, float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const uint TB = TBV;
    uint B0 = BSTR * nb + KK + 8u * G * KK + TB - 16u; if (B0 >= TB) B0 -= TB;
    const device uint* tbase = tr0;
    uint p = 0;
    for (uint kt = 0; kt < KT; kt += NQ) {
      half wv[NU * 8];
BODY
      threadgroup half* dst = btg[p];
      #pragma unroll
      for (uint u = 0; u < NU; ++u) {
        const uint id = sg + 4u * NRB * u, q = id >> 3, nj = id & 7u;
        #pragma unroll
        for (uint hi = 0; hi < 2; ++hi)
          *((threadgroup half4*)(dst + (nj * 16 + nb + 8 * hi) * BKSV + q * 16 + 4 * G)) = half4(wv[u * 8 + hi * 4], wv[u * 8 + hi * 4 + 1], wv[u * 8 + hi * 4 + 2], wv[u * 8 + hi * 4 + 3]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active) {
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB((threadgroup half*)(dst + cs * 32 * BKSV), dextents<int32_t, 2>(BKSV, 32));
        auto sA = tA.slice(int(kt) * 16, int(mg));
        op.run(sA, tB, cT);
      }
      p ^= 1u;
    }
    threadgroup half* eb = ((threadgroup half*)btg) + rb * (16 * 128);
    const float4 sv = float4(*((const device half4*)(svh + n0 + lane * 4)));
    for (uint pass = 0; pass < 4; ++pass) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active) {
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
          const auto ix = cT.get_multidimensional_index(i);
          const uint r = uint(ix[1]);
          if ((r >> 4) == pass) eb[(r & 15u) * 128 + cs * 32 + uint(ix[0])] = half(cT[i]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active) {
        for (uint j = 0; j < 4; ++j) {
          const uint r = cs * 4 + j, m = mg + pass * 16 + r;
          if (m >= M) break;
          float4 acc = float4(*((threadgroup half4*)(eb + r * 128 + lane * 4)));
          float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
          const float4 o4 = float4(v0, v1, v2, v3) * RSQRT * sv;
          *((device vec<OUTT, 4>*)(Y + ulong(m) * N + n0 + lane * 4)) = vec<OUTT, 4>(o4);
        }
      }
    }
"""


@functools.lru_cache(maxsize=None)
def shw_source(K: int, cb: str, nq: int, out_t: str, nrb: int = 4, abl: str = "") -> str:
    """Threadgroup = 4 column slabs x nrb row blocks of 64 rows (4 * nrb simdgroups). Per k step (nq k tiles of 16) the 8 * nq trellis tiles of the 128-column block are split over the simdgroups."""
    TB = 256 * K
    nsg = 4 * nrb
    assert (8 * nq) % nsg == 0
    nu = 8 * nq // nsg
    dec = []
    for u in range(nu):
        dec += ["      " + l for l in NX._tile_decode(K, cb, f"tbase + (ulong(kt + ((sg + {nsg * u}u) >> 3)) * NT + nt0 + ((sg + {nsg * u}u) & 7u)) * {8 * K}u", "wv", u * 8)]
    bks = 16 * nq
    assert nrb * 16 * 128 <= 2 * 128 * bks, "epilogue buffer must fit in the shared tile"
    src = _SRC
    if abl == "noepi":
        i = src.index("    for (uint pass = 0;")
        src = src[:i] + "    if (active) for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) { const auto ix = cT.get_multidimensional_index(i); const uint m = mg + uint(ix[1]); if (m < M) Y[ulong(m) * N + n0 + cs * 32 + uint(ix[0])] = OUTT(cT[i]); }\n"
    return (src.replace("BODY", "\n".join(dec)).replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD).replace("NRB", str(nrb)).replace("NU", str(nu))
            .replace("OUTT", out_t).replace("BKSV", str(bks)).replace("NQ", str(nq)).replace("TBV", f"{TB}u").replace("BSTR", f"{32 * K}u").replace("KK", f"{K}u"))


_kc: dict = {}
_ids = itertools.count()
_p: dict = {}
_T = {mx.float16: "half", mx.bfloat16: "bfloat", mx.float32: "float"}


def dense_shw(st: NX.SingleStack, A: mx.array, rows: int, svh: mx.array, out_dtype=mx.float16, nq: int = 2, nrb: int = 4, abl: str = "") -> mx.array:
    """Y[rows, N] = H128(A @ Wr^T) / sqrt(128) * svh, the finalized linear. A half [rows, K] in the k-permuted layout (kernels.rot(..., perm=True)); st one dense stack with kt % nq == 0,
    nt % 8 == 0; svh half [N]."""
    assert st.kt % nq == 0 and st.nt % 8 == 0
    K = st.ks[0]
    cbk = NX.dec_cb(st.cb)
    key = (K, cbk, nq, out_dtype, nrb, abl)
    kern = _kc.get(key)
    if kern is None:
        kern = _kc[key] = mx.fast.metal_kernel(name=f"kyojin_nax_shw_{cbk}_{K}_q{nq}r{nrb}_{_T[out_dtype]}{abl}_{next(_ids)}", header=NX.HEADER_NAX,
                                               input_names=["tr0", "A", "svh", "params"], output_names=["Y"], source=shw_source(K, cbk, nq, _T[out_dtype], nrb, abl))
    pk = (st.kt, st.nt, rows)
    params = _p.get(pk)
    if params is None:
        params = _p[pk] = mx.array(pk, dtype=mx.uint32)
    return kern(inputs=[CB.bind(st.words, st.bview), A, svh, params], grid=(st.nt // 8 * 128 * nrb, -(-rows // (nrb * 64)), 1), threadgroup=(128 * nrb, 1, 1),
                output_shapes=[(rows, st.nout)], output_dtypes=[out_dtype])[0]
