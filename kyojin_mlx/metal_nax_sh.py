# SPDX-License-Identifier: Apache-2.0
"""Decode-once fused EXL3 + NAX dense GEMM (prefill, 64..~1024 rows).

`metal_nax.coop_source` gives every simdgroup its own 32 columns and decodes the trellis straight into its right-input cooperative tensor, so every
64-row block of the chunk decodes the same weights again. Here the four simdgroups of a threadgroup share ONE 32-column slab and own four different
row blocks: per 64-wide k step each simdgroup decodes one 16-wide k tile of the slab (two 16x16 trellis tiles) into threadgroup memory (double
buffered, one barrier per step), then all four run `matmul2d` against the shared 32 x 64 fp16 tile. Decode work per weight drops from rows/64 to
rows/256. Same fp16 weights and the same matmul2d tiles as `coop_source`.
Lane layout of the decode: lane = 8 * G + nb, element i of a tile = (k', n') = (4 * G + (i & 3), nb + 8 * (i >> 2)) (see metal_nax docstring)."""
from __future__ import annotations

import functools
import itertools

import mlx.core as mx

from . import cbill as CB
from . import metal_nax as NX

_SRC = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint KT = params[0], NT = params[1], M = params[2];
    const uint nt0 = threadgroup_position_in_grid.x * 2;
    if (nt0 >= NT) return;
    const uint mg = (threadgroup_position_in_grid.y * NSGV + sg) * ROWS;
    const bool active = mg < M;
    const ulong KD = ulong(KT) * 16, N = ulong(NT) * 16;
    const uint n0 = nt0 * 16;
    const uint G = lane >> 3, nb = lane & 7;
    threadgroup half btg[2][32 * BKSV];
    constexpr auto desc = matmul2d_descriptor(BMU, 32, BKSV, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)A, dextents<int32_t, 2>(int(KD), int(M)));
    device OUTT* Yp = (device OUTT*)Y;
    auto tA0 = tA.slice(0, int(min(mg, M - 1)));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB0((threadgroup half*)btg[0], dextents<int32_t, 2>(BKSV, 32));
DECL
ZERO
    const uint TB = TBV;
    uint B0 = BSTR * nb + KK + 8u * G * KK + TB - 16u; if (B0 >= TB) B0 -= TB;
    const device uint* tbase = tr0;
    uint p = 0;
CONSTS
PRE0
    for (uint kt = 0; kt < KT; kt += NKT) {
      half wv[NQ * 16];
PRE1
BODY
      threadgroup half* dst = btg[p];
      #pragma unroll
      for (uint q = 0; q < NQ; ++q) {
        #pragma unroll
        for (uint jj = 0; jj < 2; ++jj) {
          #pragma unroll
          for (uint hi = 0; hi < 2; ++hi)
            *((threadgroup half4*)(dst + (jj * 16 + nb + 8 * hi) * BKSV + (q * NSGV + sg) * 16 + 4 * G)) = half4(wv[q * 16 + jj * 8 + hi * 4], wv[q * 16 + jj * 8 + hi * 4 + 1], wv[q * 16 + jj * 8 + hi * 4 + 2], wv[q * 16 + jj * 8 + hi * 4 + 3]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (active) {
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB((threadgroup half*)dst, dextents<int32_t, 2>(BKSV, 32));
        MATMULS
      }
      p ^= 1u;
    }
    if (!active) return;
STORE
"""


@functools.lru_cache(maxsize=None)
def sh_source(K: int, cb: str, bmu: int, nmb: int, out_half: bool, abl: str = "", nq: int = 1, pf: bool = False, nsg: int = 4) -> str:
    TB = 256 * K
    dec = []
    for q, jj in itertools.product(range(nq), range(2)):
        if abl == "nodec":
            dec += [f"      for (uint i = 0; i < 8; ++i) wv[{q * 16 + jj * 8} + i] = half(float(i + kt) * 0.001f);"]
            continue
        dec += ["      " + l for l in NX._tile_decode(K, cb, f"tbase + (ulong(kt + sg + {nsg * q}u) * NT + nt0 + {jj}u) * {8 * K}u", "wv", q * 16 + jj * 8)]
    mm = " ".join(f"{'if (mg + %du < M) ' % (bmu * mb) if mb else ''}{{ auto sA = tA.slice(int(kt) * 16, int(mg) + {bmu * mb}); op.run(sA, tB, cT{mb}); }}" for mb in range(nmb))
    ra = "mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t"
    decl = "\n".join(f"    auto cT{mb} = op.get_destination_cooperative_tensor<{ra}<decltype(tA0)>, {ra}<decltype(tB0)>, float>();" for mb in range(nmb))
    zero = "\n".join(f"    for (uint16_t i = 0; i < cT{mb}.get_capacity(); ++i) if (cT{mb}.is_valid_element(i)) cT{mb}[i] = 0.0f;" for mb in range(nmb))
    store = "\n".join(
        f"    for (uint16_t i = 0; i < cT{mb}.get_capacity(); ++i) if (cT{mb}.is_valid_element(i)) {{ const auto ix = cT{mb}.get_multidimensional_index(i);"
        f" const uint m = mg + {bmu * mb}u + uint(ix[1]); if (m < M) Yp[ulong(m) * N + n0 + uint(ix[0])] = OUTT(cT{mb}[i]); }}" for mb in range(nmb))
    if abl == "nomm":
        mm = "if (kt == 0xffffffffu) { " + mm + " }"
    if abl == "nopf":
        pass
    nw = 3 * len(NX.chunk_plan(K)); ntl = 2 * nq
    tile = lambda q, jj, kt: f"tbase + (ulong({kt} + sg + {nsg * q}u) * NT + nt0 + {jj}u) * {8 * K}u"
    pre0 = ["    {", "      const uint kt = 0;"] + ["      " + l for q, jj in itertools.product(range(nq), range(2)) for l in NX._tile_loads(K, tile(q, jj, "kt"), f"nw_[{q * 2 + jj}]")] + ["    }"]
    pre1 = ["      if (kt + NKT < KT) {"] + ["        " + l for q, jj in itertools.product(range(nq), range(2)) for l in NX._tile_loads(K, tile(q, jj, "(kt + NKT)"), f"nw_[{q * 2 + jj}]")] + ["      }"]
    if pf and abl != "nodec":
        dec = ["      " + l for q, jj in itertools.product(range(nq), range(2)) for l in NX._tile_decode_w(K, cb, f"cw[{q * 2 + jj}]", "wv", q * 16 + jj * 8)]
    consts = "\n".join("    " + l for l in NX._chunk_consts(K))
    if pf:
        consts += f"\n    uint cw[{ntl}][{nw}], nw_[{ntl}][{nw}];"
        pre1 = [f"      for (uint t_ = 0; t_ < {ntl}; ++t_) for (uint w_ = 0; w_ < {nw}; ++w_) cw[t_][w_] = nw_[t_][w_];"] + pre1
    else:
        consts, pre0, pre1 = "", [], []
    return (_SRC.replace("CONSTS", consts).replace("NTL", str(ntl)).replace("NWD", str(nw)).replace("PRE0", "\n".join(pre0)).replace("PRE1", "\n".join(pre1)).replace("BODY", "\n".join(dec)).replace("MATMULS", mm).replace("DECL", decl).replace("ZERO", zero).replace("STORE", store)
            .replace("OUTT", "half" if out_half else "float").replace("ROWS", str(bmu * nmb)).replace("BMU", str(bmu))
            .replace("BKSV", str(16 * nsg * nq)).replace("NKT", f"{nsg * nq}u").replace("NSGV", f"{nsg}u").replace("NQ", str(nq)).replace("TBV", f"{TB}u").replace("BSTR", f"{32 * K}u").replace("KK", f"{K}u"))


_kc: dict = {}
_ids = itertools.count()
_p: dict = {}


def dense_sh(st: NX.SingleStack, A: mx.array, rows: int, cfg=(32, 2), out_half: bool = True, abl: str = "", nq: int = 1, pf: bool = False, nsg: int = 4) -> mx.array:
    """Y[rows, N] = A @ Wr^T. A half [rows, K] in the k-permuted layout (kernels.rot(..., perm=True)); st one dense stack with kt % 4 == 0, nt even."""
    assert st.kt % (nsg * nq) == 0 and st.nt % 2 == 0
    bmu, nmb = cfg
    K = st.ks[0]
    cbk = NX.dec_cb(st.cb)
    key = (K, cbk, bmu, nmb, out_half, abl, nq, pf, nsg)
    kern = _kc.get(key)
    if kern is None:
        kern = _kc[key] = mx.fast.metal_kernel(name=f"kyojin_nax_sh_{cbk}_{K}_{bmu}_{nmb}_{int(out_half)}{abl}_q{nq}{'p' if pf else ''}s{nsg}_{next(_ids)}", header=NX.HEADER_NAX,
                                               input_names=["tr0", "A", "params"], output_names=["Y"], source=sh_source(K, cbk, bmu, nmb, out_half, abl, nq, pf, nsg))
    pk = (st.kt, st.nt, rows)
    params = _p.get(pk)
    if params is None:
        params = _p[pk] = mx.array(pk, dtype=mx.uint32)
    per_tg = nsg * bmu * nmb
    return kern(inputs=[CB.bind(st.words, st.bview), A, params], grid=(-(-st.nt // 2) * 32 * nsg, -(-rows // per_tg), 1), threadgroup=(32 * nsg, 1, 1),
                output_shapes=[(rows, st.nout)], output_dtypes=[mx.float16 if out_half else mx.float32])[0]
