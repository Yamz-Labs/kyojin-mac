# SPDX-License-Identifier: Apache-2.0
"""Decode-once fused EXL3 + NAX expert GEMM: the weight slab of a k step is decoded ONCE per threadgroup into threadgroup memory and shared by the four
simdgroups, which each multiply their own 32 rows of the expert against it (metal_nax.coop_source decodes the weights in registers inside every simdgroup,
for every 64-row block of the expert: with ~128 rows per expert the decode ALU dominates and the tensor units idle).

Threadgroup = 4 simdgroups = one 32-column slab (two 16x16 trellis tiles) x 128 rows of one expert. Per k step of 64 (4 k blocks): simdgroup s decodes k block s of
both tiles (the same 16 weights per lane per tile as the register path, same `KPERM` order), writes them to threadgroup memory B[n][k'] (double buffered, one
barrier per step), then every simdgroup runs `matmul2d` (32 x 32 x 64) of its rows against the shared B. The fp32 accumulation order over k is the register
path's, so the result equals `metal_nax.expert_gemm`. Needs kt % 4 == 0.

Grid (NT / 2 * 128, ceil(maxrows / 128), nproj * G), threadgroup 128."""
from __future__ import annotations

import itertools

import mlx.core as mx

from . import metal_nax as NX

ROWS_TG = 128

_BODY = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint z = threadgroup_position_in_grid.z;
    const uint KT = params[0], NT = params[1], GL = params[2];
    const uint e = uint(eids[z % GL]);
    TRSEL
    const uint row0 = rowoff[z], M = rowoff[z + 1] - row0;
    const uint mg = threadgroup_position_in_grid.y * 128u;
    const uint nt0 = threadgroup_position_in_grid.x * 2u;
    if (mg >= M || nt0 >= NT) return;
    const uint K = uint(kk[e]);
    const ulong KD = ulong(KT) * 16, N = ulong(NT) * 16;
    const uint n0 = nt0 * 16;
    const device uint* tbase = trellis + off[e];
    const uint mgs = mg + sg * 32u;

    constexpr auto desc = matmul2d_descriptor(32, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(A + ulong(row0) * KD), dextents<int32_t, 2>(int(KD), int(M)));
    device OUTT* Yp = (device OUTT*)(Y + ulong(row0) * N);
PREVCOPY
    threadgroup half bt[2][32 * 64];
    auto tA0 = tA.slice(0, int(mgs));
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB0(bt[0], dextents<int32_t, 2>(64, 32));
    auto sB0 = tB0.slice(0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(tA0), decltype(sB0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const ushort i0 = 0;
    const auto ix0 = rB.get_multidimensional_index(i0);
    const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
BODY
    {
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
            const auto ix = cT.get_multidimensional_index(i);
            const uint m = mgs + uint(ix[1]);
            if (m < M) Yp[ulong(m) * N + n0 + uint(ix[0])] = OUTT(cT[i]);
        }
    }
"""


def tg_source(Ks, cb="mul1", nproj=1, out_half=True, prev=False) -> str:
    bodies = []
    for K in Ks:
        TB = 256 * K
        lines = [f"if (K == {K}u) {{",
                 f"  uint B0 = {32 * K}u * nb + {K}u + 8u * G * {K}u + {TB}u - 16u; if (B0 >= {TB}u) B0 -= {TB}u;",
                 "  uint p = 0;",
                 "  for (uint kt = 0; kt < KT; kt += 4u) {",
                 "    half wv[16];"]
        for jj in range(2):
            lines += ["    " + l for l in NX._tile_decode(K, cb, f"tbase + (ulong(kt + sg) * NT + nt0 + {jj}u) * {8 * K}u", "wv", jj * 8)]
        lines += ["    threadgroup half* b = bt[p];",
                  "    #pragma unroll",
                  "    for (uint jj = 0; jj < 2; ++jj) {",
                  "      #pragma unroll",
                  "      for (uint j = 0; j < 8; ++j) b[(16u * jj + nb + 8u * (j >> 2)) * 64u + sg * 16u + 4u * G + (j & 3u)] = wv[jj * 8 + j];",
                  "    }",
                  "    threadgroup_barrier(mem_flags::mem_threadgroup);",
                  "    if (mgs < M) {",
                  "      tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(b, dextents<int32_t, 2>(64, 32));",
                  "      auto sA = tA.slice(int(kt) * 16, int(mgs)); auto sB = tB.slice(0, 0);",
                  "      op.run(sA, sB, cT);",
                  "    }",
                  "    p ^= 1u;",
                  "  }",
                  "}"]
        bodies.append("\n".join(lines))
    body = " else ".join(bodies)
    src = _BODY.replace("BODY", body).replace("OUTT", "half" if out_half else "float").replace("TRSEL", NX._TRSEL[nproj])
    if prev:
        miss = " && ".join(f"K != {K}u" for K in Ks)
        src = src.replace("PREVCOPY", f"    if ({miss}) {{ for (uint i = lane; i < 32u * 32u; i += 32u) {{ const uint m = mgs + i / 32u, c = n0 + i % 32u; "
                                      "if (m < M) Yp[ulong(m) * N + c] = Yprev[(ulong(row0) + m) * N + c]; } return; }")
    else:
        src = src.replace("PREVCOPY", "")
    return src


_kc: dict = {}
_ids = itertools.count()


def expert_gemm_tg(stacks, eids, rowoff, A, rows_out, maxrows, out_half=True, ks=None, prev=None):
    s0, nproj, GL = stacks[0], len(stacks), int(eids.shape[0])
    assert s0.kt % 4 == 0 and s0.nt % 2 == 0, "decode-once kernel: kt % 4 and nt % 2"
    Ks = ks or tuple(sorted(set().union(*[set(s.ks) for s in stacks])))
    cbk = NX.dec_cb(s0.cb)
    key = (Ks, cbk, nproj, out_half, prev is not None)
    kern = _kc.get(key)
    if kern is None:
        names = ["eids", "rowoff"] + [f"tr{i}" for i in range(nproj)] + [f"off{i}" for i in range(nproj)] + [f"kk{i}" for i in range(nproj)]
        kern = _kc[key] = mx.fast.metal_kernel(
            name=f"kyojin_naxtg_{cbk}_{'_'.join(map(str, Ks))}_p{nproj}{'_pv' if prev is not None else ''}_{next(_ids)}", header=NX.HEADER_NAX,
            input_names=names + ["A", "params"] + (["Yprev"] if prev is not None else []), output_names=["Y"],
            source=tg_source(Ks, cbk, nproj, out_half, prev is not None))
    pk = (s0.kt, s0.nt, GL)
    params = NX._params.get(pk)
    if params is None:
        params = NX._params[pk] = mx.array(pk, dtype=mx.uint32)
    ins = [eids, rowoff] + [s.bound for s in stacks] + [NX._device_array(s.offset) for s in stacks] + [NX._device_array(s.k) for s in stacks] + [A, params] + ([prev] if prev is not None else [])
    grid = (s0.nt // 2 * 128, -(-maxrows // ROWS_TG), nproj * GL)
    return kern(inputs=ins, grid=grid, threadgroup=(128, 1, 1), output_shapes=[(rows_out, s0.nout)],
                output_dtypes=[mx.float16 if out_half else mx.float32])[0]
