# SPDX-License-Identifier: Apache-2.0
"""Wide-row products on stored weights (9..64 activation rows): one NAX kernel family for every foreign block format with a few-row MMA decoder.

A simdgroup owns 32 output columns. A lane owns 4 weight rows (n0 + nb + 8 j) and unit G of every 128-k chunk; the half2 decoders of the few-row MMA (foreign_metal.fmma_source)
write straight into the right-input cooperative tensor of a `matmul2d` (layout of q4mma.q4nax), the activation rows (permuted to the decoder's pair order, wide_xperm_index_reg)
are the left input read in place. The stored bytes are read once for all rows, the decode ALU runs once per weight, the MMA runs on the neural accelerator with fp32 accumulation.
The simdgroups of a threadgroup split k and reduce through threadgroup memory. No resident copy, no persistent buffer."""
from __future__ import annotations

from . import foreign_metal as FM

BK = 128            # k per chunk (4 units)

NAX_HDR = r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define RA mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t
"""


def mt_for(rows: int) -> int:
    """M tile of the matmul2d (activation rows of one threadgroup, padded up to it): a multiple of 16 to 48; above 48 rows the grid splits the rows in tiles of 32
    (the 64-row tile runs at 0.55 of the 32 / 48 row tile's row rate: registers), each half decodes its weights again but reads them from the cache."""
    return 16 * -(-rows // 16) if rows <= 48 else 32


def wide_header(fmt: str) -> str:
    return NAX_HDR + (FM.UNR if fmt in FM.FORMATS8 else FM.fmma_header(fmt))


def wide_inputs(fmt: str):
    if fmt in FM.FORMATS8:
        return FM.gemv8_inputs(fmt)
    return ["x"] + FM.fmma_inputs(fmt)[1:]


def wide_layout(fmt: str) -> str:
    """Key of the activation permutation: formats with the same letter share one index."""
    return "8" if fmt in FM.FORMATS8 else FM._FMMA_LAYOUT[fmt]


def wide_xperm_index(fmt: str, k: int):
    """int32 [k]: position p = kb * 128 + ur * 32 + 8 q + 2 t + e of the permuted activation holds natural input idx[p]. The tile holds the weights in the order the half2
    decoders of the few-row MMA produce them (unit ur of the block, word q, pair t = nibbles t and t + 4, element e); the k sum does not care, the activations follow."""
    import numpy as np
    assert k % BK == 0
    p = np.arange(k)
    e, t, q, ur, kb = p & 1, (p >> 1) & 3, (p >> 3) & 3, (p >> 5) & 3, p >> 7
    u = 4 * kb + ur
    if fmt in FM.FORMATS8:       # one byte per weight: pair (q, t) = bytes (t & 1) and (t & 1) + 2 of word 2 q + (t >> 1) of the 32-byte unit, element e picks one of the two
        return (32 * u + 8 * q + 4 * (t >> 1) + (t & 1) + 2 * e).astype(np.int32)
    nj = t + 4 * e
    L = FM._FMMA_LAYOUT[fmt]
    if L in ("K", "KG", "K2", "K1"):
        first, second = FM._pair_elements(L, 4 * q + t)
        return (32 * u + np.where(e == 0, first, second)).astype(np.int32)
    if L == "B":
        off = 4 * q + t + 16 * e
    else:
        off = 8 * q + nj if L == "N" else 4 * q + (nj >> 1) + (16 if L == "H" else 32) * (nj & 1)
    if L == "Q":
        u8 = u & 7
        unit = 256 * (u >> 3) + 64 * (u8 >> 1) + 16 * (u8 & 1)
    else:
        unit = 32 * u
    return (unit + off).astype(np.int32)


def _blk_dec8(fmt: str):
    """(scale, kd, blk, dec, lutinit) of the 8-bit formats. A lane loads the 32 bytes of its unit per row as 8 words; pair (q, t) is bytes (t & 1) and (t & 1) + 2 of word 2 q + (t >> 1).
    fp8 (E4M3): half bits = sign | magnitude << 7 is value / 256 exactly (subnormals included), the 256 and the row scale gs[n] come back on the fp32 sum.
    q8_0: the byte xor 0x80 or-ed under 0x6400 is 1024 + 128 + v; minus 1152 is v, times the block scale d (half [N, K/32], one per unit)."""
    if fmt == "fp8":
        dec = ("const uint y_ = wd[j][2u * q + (t >> 1)] >> (8u * (t & 1u)); const half2 w2 = as_type<half2>(((y_ << 7) & 0x3F803F80u) | ((y_ << 8) & 0x80008000u));")
        extra, scale = "", "gs[n0 + nl] * 256.0f"
    else:
        dec = ("const uint y_ = wd[j][2u * q + (t >> 1)] >> (8u * (t & 1u));"
               " const half2 w2 = (as_type<half2>((y_ & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h)) * S[j];")
        extra, scale = "S[j] = half2(((device const half*)MTP)[ulong(rn) * UPR + uu]);", "gs[n0 + nl]"
    xr = " ^ 0x80808080u" if fmt == "q8_0" else ""
    blk = f"""uint wd[NT][8]; half2 S[NT];
      const uint uu = kb * 4u + c;
      UNR for (uint j = 0; j < uint(NT); ++j) {{
        const uint rn = rowt[j];
        const uint4 a_ = q4[ulong(rn) * (uint(KIN) / 16u) + 2u * uu], b_ = q4[ulong(rn) * (uint(KIN) / 16u) + 2u * uu + 1u];
        wd[j][0] = a_.x{xr}; wd[j][1] = a_.y{xr}; wd[j][2] = a_.z{xr}; wd[j][3] = a_.w{xr}; wd[j][4] = b_.x{xr}; wd[j][5] = b_.y{xr}; wd[j][6] = b_.z{xr}; wd[j][7] = b_.w{xr};
        S[j] = half2(0.0h); {extra}
      }}"""
    return scale, False, blk, dec, ""


def wide_source_reg(fmt: str, rowgs: bool = False) -> str:
    """Register-fed variant: no threadgroup tile. A simdgroup owns 32 output columns, a lane 4 weight rows (n0 + nb + 8 j) and unit G of every 128-k chunk; the half2 decoders of the few-row
    MMA write straight into the right-input cooperative tensor of the matmul2d (layout of q4mma.q4nax: element 16 s + 4 j + 2 cp + e holds k' = 16 s + 4 G + 2 cp + e of a 64-k step, row nb + 8 j).
    Step = q >> 1 of the unit's word, s = pair t, cp = q & 1; wide_xperm_index_reg puts x in that order. The grid is (column tiles x WS simdgroups, row tiles of MT): a threadgroup
    owns x rows mb .. mb + R (R = x_shape[0] is not a template: no recompile per row count). The WS simdgroups split k and reduce RC rows at a time through threadgroup memory.
    Template ints: MT, KIN (% 128), N, WS, RC."""
    if fmt in FM.FORMATS8:
        scale, kd, blk, dec, lutinit = _blk_dec8(fmt)
    else:
        scale = "1.0f" if (fmt in FM._FMMA_K) else "gs[n0 + nl]" if rowgs else "gs[0]"
        kd, blk, dec = FM._fmma_blk_dec(fmt)
        lutinit = ("threadgroup half2 lut[256];\n    for (uint i = tid; i < 256u; i += 32u * uint(WS)) lut[i] = half2(half(KVC[i & 15u]), half(KVC[i >> 4]));\n"
                   "    threadgroup_barrier(mem_flags::mem_threadgroup);") if FM._FMMA_PAIR[fmt] == "lut" else ""
    return f"""
    constexpr uint NT = 4;
    const uint mb = threadgroup_position_in_grid.y * uint(MT);
    const uint R = min(uint(MT), uint(x_shape[0]) - mb);
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint n0 = threadgroup_position_in_grid.x * 32u;
    const uint KB = uint(KIN) / 128u;
    const uint UPR = uint(KIN) / 32u;
    {'auto MTP = mt;' if fmt != 'fp8' else ''}
    {FM.ktab_decl(fmt, False) if kd else ''}
    {lutinit}
    device const uint4* q4 = (device const uint4*)qs;
    constexpr auto desc = matmul2d_descriptor(MT, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)x + ulong(mb) * uint(KIN), dextents<int32_t, 2>(int(KIN), int(R)));
    auto rB0 = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto rB1 = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cT = op.template get_destination_cooperative_tensor<RA<decltype(tA)>, RA<decltype(rB0)>, float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    const ushort i0 = 0;
    const auto ix0 = rB0.get_multidimensional_index(i0);
    const uint c = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
    uint rowt[NT];
    UNR for (uint j = 0; j < NT; ++j) rowt[j] = min(n0 + nb + 8u * j, uint(N) - 1u);
    for (uint kb = warp; kb < KB; kb += uint(WS)) {{
      {blk}
      UNR for (uint step = 0; step < 2; ++step) {{
        UNR for (uint t = 0; t < 4; ++t) UNR for (uint cp = 0; cp < 2; ++cp) {{
          const uint q = 2u * step + cp;
          UNR for (uint j = 0; j < NT; ++j) {{
            {dec}
            if (step == 0) {{ rB0[16 * t + 4 * j + 2 * cp] = w2.x; rB0[16 * t + 4 * j + 2 * cp + 1] = w2.y; }}
            else {{ rB1[16 * t + 4 * j + 2 * cp] = w2.x; rB1[16 * t + 4 * j + 2 * cp + 1] = w2.y; }}
          }}
        }}
        auto sA = tA.slice(int(kb * 128u + step * 64u), 0);
        if (step == 0) op.run(sA, rB0, cT); else op.run(sA, rB1, cT);
      }}
    }}
    threadgroup float red[WS][RC * 32];
    for (uint m0 = 0; m0 < R; m0 += uint(RC)) {{
      for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {{
        const auto ix = cT.get_multidimensional_index(i);
        const uint m = uint(ix[1]);
        if (m >= m0 && m < m0 + uint(RC)) red[warp][(m - m0) * 32u + uint(ix[0])] = cT[i];
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint idx = tid; idx < uint(RC) * 32u; idx += 32u * uint(WS)) {{
        const uint m = m0 + (idx >> 5), nl = idx & 31u;
        if (m < R && n0 + nl < uint(N)) {{
          float v = 0.0f;
          UNR for (uint s = 0; s < uint(WS); ++s) v += red[s][idx];
          y[ulong(mb + m) * uint(N) + n0 + nl] = half(clamp(v * {scale}, -65504.0f, 65504.0f));
        }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""


def wide_xperm_index_reg(fmt: str, k: int):
    """int32 [k] for wide_source_reg: position kb * 128 + 64 (q >> 1) + 16 t + 4 ur + 2 (q & 1) + e holds the natural input of unit 4 kb + ur, word q, pair t, element e."""
    import numpy as np
    nat = wide_xperm_index(fmt, k)                   # natural input at p = kb * 128 + 32 ur + 8 q + 2 t + e
    p = np.arange(k)
    e, t, q, ur, kb = p & 1, (p >> 1) & 3, (p >> 3) & 3, (p >> 5) & 3, p >> 7
    dst = kb * 128 + 64 * (q >> 1) + 16 * t + 4 * ur + 2 * (q & 1) + e
    out = np.empty(k, np.int32)
    out[dst] = nat
    return out
