# SPDX-License-Identifier: Apache-2.0
"""Comparison arms of the S6 probe (tools/probe_nax.py): the same trellis decode into a threadgroup tile, then matmul2d (NAX) or
simdgroup_matrix 8x8 (the pre-NAX tensor path). Not used by the engine."""
from __future__ import annotations

import functools

from kyojin_mlx import metal_src as S

_STAGED = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint z = threadgroup_position_in_grid.z;
    const uint KT = params[0], NT = params[1];
    const uint e = uint(eids[z]);
    const uint row0 = rowoff[z], M = rowoff[z + 1] - row0;
    const uint mg = threadgroup_position_in_grid.y * ROWS;
    const uint ntb = threadgroup_position_in_grid.x * 8;
    if (mg >= M || ntb >= NT) return;
    const uint K = uint(kk[e]);
    const ulong KD = ulong(KT) * 16, N = ulong(NT) * 16;
    const device uint* tbase = tr + off[e];
    threadgroup half Bs[128 * 32];
    const uint g = tid & 7, tix = tid >> 3, kt2 = tix >> 3, ntl = tix & 7;
    const bool live = ntb + ntl < NT;
    const uint n0 = (ntb + sg * 2) * 16;
    device OUTT* Yp = (device OUTT*)(Y + ulong(row0) * N);
ENGINE_PRE
    for (uint kt = 0; kt < KT; kt += 2) {
        DECODE
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ENGINE_MM
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
ENGINE_POST
"""


@functools.lru_cache(maxsize=None)
def staged_source(Ks: tuple[int, ...], cb: str = "mul1", engine: str = "nax", rows: int = 64, out_half: bool = True,
                  mode: str = "full") -> str:
    """Comparison arms. Same decode as `decode_pf` (a lane decodes a whole run with static windows) but into a threadgroup tile
    Bs[128 n][32 k] instead of device memory; then either matmul2d on the threadgroup tile (engine 'nax') or simdgroup_matrix
    8x8x8 MMA (engine 'simd'). A in natural k order. Four simdgroups: 32 columns each; `rows` rows per threadgroup.
    Grid (ceil(NT / 8) * 128, ceil(maxrows / rows), G), threadgroup 128."""
    decs = []
    for K in Ks:
        loads, unpack = S.gen_loads(K)
        ld = [f"{t} {v} = {e};" for v, t, e in loads] + unpack
        run = S.run_decode_lines(K, lambda p, kk, hi, e: f"{'whi' if hi else 'wlo'}[{kk}] = {e};", cb)
        body = (f"{{ constexpr uint TW = {8 * K}u; const uint wprev = ({K}u * g + TW - 1u) % TW;\n"
                f"  const device uint* tp = tbase + (ulong(kt + kt2) * NT + ntb + ntl) * TW;\n  " + "\n  ".join(ld) + "\n"
                f"  threadgroup half* wlo = Bs + (ntl * 16 + g) * 32 + kt2 * 16; threadgroup half* whi = Bs + (ntl * 16 + 8 + g) * 32 + kt2 * 16;\n  "
                + "\n  ".join(run) + " }")
        decs.append(f"if (K == {K}u) " + body)
    decode = "if (live) { " + " else ".join(decs) + " }" if mode != "nodec" else "Bs[tid] = half(kt);"
    if engine == "nax":
        nmb = rows // 32
        ra = "mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t"
        pre = (f"constexpr auto desc = matmul2d_descriptor({rows}, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);\n"
               "matmul2d<desc, execution_simdgroup> op;\n"
               "tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(A + ulong(row0) * KD), dextents<int32_t, 2>(int(KD), int(M)));\n"
               "tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(Bs, dextents<int32_t, 2>(32, 128));\n"
               "auto tA0 = tA.slice(0, int(mg)); auto tB0 = tB.slice(0, int(sg) * 32);\n"
               f"auto cT = op.get_destination_cooperative_tensor<{ra}<decltype(tA0)>, {ra}<decltype(tB0)>, float>();\n"
               "for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;\n")
        mm = "{ auto sA = tA.slice(int(kt) * 16, int(mg)); auto sB = tB.slice(0, int(sg) * 32); op.run(sA, sB, cT); }"
        post = ("for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) { const auto ix = cT.get_multidimensional_index(i);"
                " const uint m = mg + uint(ix[1]); if (m < M) Yp[ulong(m) * N + n0 + uint(ix[0])] = OUTT(cT[i]); }")
    else:
        fm_ = rows // 8
        pre = ("const ushort qid = lane / 4; const ushort fm = (qid & 4) + ((lane / 2) % 4); const ushort fn = (qid & 2) * 2 + (lane % 2) * 2;\n"
               f"simdgroup_matrix<float, 8, 8> acc[{fm_}][4];\n"
               f"_Pragma(\"clang loop unroll(full)\") for (uint i = 0; i < {fm_}; ++i) _Pragma(\"clang loop unroll(full)\") for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);\n")
        mm = (f"_Pragma(\"clang loop unroll(full)\") for (uint k8 = 0; k8 < 32; k8 += 8) {{\n"
              f"  simdgroup_matrix<half, 8, 8> bf[4];\n"
              f"  _Pragma(\"clang loop unroll(full)\") for (uint j = 0; j < 4; ++j) simdgroup_load(bf[j], Bs + (sg * 32 + j * 8) * 32 + k8, 32, ulong2(0, 0), true);\n"
              f"  _Pragma(\"clang loop unroll(full)\") for (uint i = 0; i < {fm_}; ++i) {{\n"
              f"    simdgroup_matrix<half, 8, 8> af;\n"
              f"    simdgroup_load(af, A + (ulong(row0) + mg + i * 8) * KD + kt * 16 + k8, KD);\n"
              f"    _Pragma(\"clang loop unroll(full)\") for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], af, bf[j], acc[i][j]);\n"
              f"  }}\n}}")
        post = (f"for (uint i = 0; i < {fm_}; ++i) for (uint j = 0; j < 4; ++j) {{\n"
                f"  const uint m = mg + i * 8 + fm; const auto v = acc[i][j].thread_elements();\n"
                f"  if (m < M) {{ Yp[ulong(m) * N + n0 + j * 8 + fn] = OUTT(v[0]); Yp[ulong(m) * N + n0 + j * 8 + fn + 1] = OUTT(v[1]); }}\n}}")
    if mode == "nomm":
        mm = "if (kt == 0xffffffffu) { " + mm + " }"
    return (_STAGED.replace("ENGINE_PRE", pre).replace("ENGINE_MM", mm).replace("ENGINE_POST", post).replace("DECODE", decode)
            .replace("OUTT", "half" if out_half else "float").replace("ROWS", str(rows)))
