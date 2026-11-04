# SPDX-License-Identifier: Apache-2.0
"""S6: fused EXL3 trellis decode + M5 neural-accelerator GEMM (Metal 4 tensor ops, `matmul2d`).

Y[m, n] = sum_k A[m, k'] * Wr[n, k]  with Wr^T the inner (rotated) weight of one projection of one expert, decoded in registers
and never written to memory. The result is the same as `decode_pf` (fp16 weights) followed by `mx.gather_mm`, up to the order of the
fp32 accumulation inside the matmul.

Register path (`coop_source`): one simdgroup owns a 32-column slab (two 16x16 trellis tiles) and the whole 32*NMB-row block of one
expert. Per 16-wide k-step each lane decodes its 16 weights straight into the right-input cooperative tensor of the simdgroup's
`matmul2d`. The cooperative layout of the right input (measured on M5, `layout1.py`) is
    element i = 8 * tile + 4 * hi + c :   (k', n') = (4 * G + c, nb + 16 * tile + 8 * hi)     G = c0(i=0) / 4,  nb = c1(i=0)
so a lane works on run g = nb of both tiles and trellis positions p = 8 * G .. 8 * G + 7 of that run (consecutive, one 64-bit window
register per tile). The stored weight order inside a 16-wide k-block is permuted so that the lane's four k' values land on one
trellis run:
    k' = 4 * G + c   <->   k = 2 * G + (c & 1) + 8 * (c >> 1)         (`KPERM`)
The activations A must carry the same permutation inside every 16-block of k (the rotation kernels that produce A write it for free).
"""
from __future__ import annotations

import functools
import itertools
import logging
import os
import sys

import mlx.core as mx

from . import cbill as CB
from . import metal_src as S
from . import fired as FD

HEADER_NAX = S.HEADER_MOE + S.HEADER_MOE_ROOF + r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

# Row sub-blocks of a tail block (an expert's last bmu * nmb rows) that hold no row skip their matmul: 1 = on
TG_MIN_ROWS = int(os.environ.get("KYOJIN_NAX_TG", "0"))      # >0: expert GEMMs of experts with at least this many rows (maxrows) use the decode-once threadgroup kernel (metal_nax_tg); 0 = off
SKIP_TAIL = os.environ.get("KYOJIN_NAX_SKIP_TAIL", "1") != "0"
# pf_dec: the trellis decode of the fused NAX kernels (routed prefill GEMMs, the gate/up epilogue kernel, the dense stacks) of an exact mul1 pack: 1 = dec_mul1r1 (lane sum by one imad),
# 2 = dec_mul1r2 (also the byte pairs by a subtract-imad). The same fp16 weight bit for bit (exhaustive over the 65536 states in tests/test_moeroof.py), 2-3 integer ops
# fewer per decoded weight. The decode is the ALU limit of these kernels (pfroof: 35-45 TF with it, 85-91 TF without), unlike the decode GEMVs (dense_dec r1: memory bound).
PF_DEC = int(os.environ.get("KYOJIN_PF_DEC", "2"))     # ON at 2 (31/10: 32K 2757 -> 2834 tok/s, 128K 2851 -> 2891, all pairs up)


def dec_cb(cb: str) -> str:
    """Codebook name the NAX kernels decode with: the exact mul1 swaps to its cheaper integer form under PF_DEC (read at kernel build, part of every kernel cache key)."""
    if PF_DEC:
        FD.hit("pf_dec" if cb == "mul1" and PF_DEC in (1, 2) else "pf_dec_miss")
    return f"mul1r{PF_DEC}" if cb == "mul1" and PF_DEC in (1, 2) else cb


# row map (pf_moe_rowmap): simdgroups per threadgroup. One threadgroup stages its row block once per k block and every simdgroup
# runs its own bns columns against it, so a bigger threadgroup divides the staging work (and the x reads) over more columns.
RM_NSG = int(os.environ.get("KYOJIN_PF_MOE_RM_NSG", "8"))

# software prefetch of the trellis words of the next k step (loads of step kt + nk issued before the decode of step kt): 1 = on
PREFETCH = os.environ.get("KYOJIN_NAX_PREFETCH", "1") != "0"
# opalu: software-pipeline the trellis decode against the tensor op. The register path decodes step k into the simdgroup's one right-input
# cooperative tensor and then runs the matmul on it, so the ALU of step k+1 cannot start until that matmul has read the tensor: decode and
# tensor op alternate instead of overlapping (pfroof: 35-45 TF with the decode against 85-91 TF without). With PIPE the kernel keeps TWO right-input
# tensors and decodes step k+1 into the idle one before the matmul of step k is issued, so the tensor unit runs behind the ALU. Same tiles, same
# decode, same k order into the same accumulators: bit-identical. MEASURED (docs/reports/opalu_REPORT.md, tools/opalu_dec.py): 1.24-2.42x SLOWER
# in every form built (bks 64 / 32 / 16, bns 32 / 16) - a second right-input tile costs more registers than the overlap it buys, so this GPU gives
# one simdgroup no overlap between `matmul2d` and the integer ALU. 1 = on (default 0, a dead end kept for the record).
PIPE = os.environ.get("KYOJIN_NAX_PIPE", "0") != "0"
# cpftrellis: the k loop of the register path unrolled by two steps with two alternating word sets (wa / wb), instead of one word set copied into the next
# (`cw = nw_`, nsub * nw = 24 register moves per step, a loop-carried copy the compiler can only drop by unrolling). Same loads, same decode, same k
# order into the same accumulators: bit-identical, 24 fewer ALU instructions per 64-weight step (792 -> ~768 in opalu's count). Default 0 until timed.
UNROLL2 = os.environ.get("KYOJIN_NAX_UNROLL2", "0") != "0"
# cpftrellis: row sub-blocks of 16 instead of 32 in the routed prefill GEMM (cfg (32, n, ..) becomes (16, 2n, ..): the same rows per block and the same accumulator
# registers, twice the matmul runs per k step of half the rows each). An expert's tail pads to 16 rows instead of 32 (SKIP_TAIL skips the empty 16-row runs); the
# weight decode per block is unchanged. Model on skewed routing (tests/test_cpftrellis.py): padded matmul rows -5 %, expert GEMM -2.9..-4.3 % IF a 16-row run costs
# its per-row share of a 32-row run (not measured: tools/cpftrellis_gemm.py). fp32 order inside the tensor op may differ by descriptor: bits are checked, not assumed.
BMU16 = os.environ.get("KYOJIN_PF_BMU16", "0") != "0"
DIRECT_RB = os.environ.get("KYOJIN_NAX_DIRECT_RB", "1") != "0"      # decode straight into the cooperative operand (no half[] staging array)

# permutation of k inside a 16-block: stored position k' = 4 * G + c holds the weight of k = KPERM[k']
KPERM = [2 * (kp // 4) + (kp & 1) + 8 * ((kp & 3) >> 1) for kp in range(16)]


def chunk_plan(K: int) -> list[tuple[int, int]]:
    """[(first window, number of windows)] over the 8 windows of a lane: one 64-bit register when 7K + 16 <= 64, else two."""
    return [(0, 8)] if 7 * K + 16 <= 64 else [(0, 4), (4, 4)]


def _tile_decode(K: int, cb: str, tile_expr: str, dst: str, base: int = 0) -> list[str]:
    """Lines decoding the lane's 8 weights of the tile whose words start at `tile_expr` (device uint*) into dst[0..7]."""
    out = [f"{{ const device uint* tp = {tile_expr};"]
    for ci, (i0, n) in enumerate(chunk_plan(K)):
        out.append(f"  {{ uint b = B0 + {i0 * K}u; if (b >= {256 * K}u) b -= {256 * K}u;")
        out.append(f"    const uint a0 = b >> 5, sh = b & 31u;")
        out.append(f"    uint a1 = a0 + 1u; if (a1 >= {8 * K}u) a1 -= {8 * K}u;")
        out.append(f"    uint a2 = a1 + 1u; if (a2 >= {8 * K}u) a2 -= {8 * K}u;")
        out.append(f"    const ulong S = (((ulong(tp[a0]) << 32) | ulong(tp[a1])) << sh) | (ulong(tp[a2]) >> (32u - sh));")
        for i in range(n):
            out.append(f"    {dst}[{base + i0 + i}] = dec_{cb}(uint((S >> {48 - i * K}) & 0xffffUL));")
        out.append("  }")
    out.append("}")
    return out


def _chunk_consts(K: int) -> list[str]:
    """Lane constants of the word windows (loop invariant): per chunk the word indices a0 a1 a2 and the bit shift."""
    out = []
    for ci, (i0, n) in enumerate(chunk_plan(K)):
        out += [f"uint b{ci} = B0 + {i0 * K}u; if (b{ci} >= {256 * K}u) b{ci} -= {256 * K}u;",
                f"const uint a0_{ci} = b{ci} >> 5, sh_{ci} = b{ci} & 31u;",
                f"uint a1_{ci} = a0_{ci} + 1u; if (a1_{ci} >= {8 * K}u) a1_{ci} -= {8 * K}u;",
                f"uint a2_{ci} = a1_{ci} + 1u; if (a2_{ci} >= {8 * K}u) a2_{ci} -= {8 * K}u;"]
    return out


def _tile_loads(K: int, tile_expr: str, dst: str) -> list[str]:
    """Loads of the lane's trellis words of one tile into dst[0 .. 3 * chunks - 1]."""
    out = [f"{{ const device uint* tp = {tile_expr};"]
    for ci in range(len(chunk_plan(K))):
        out.append(f"  {dst}[{3 * ci}] = tp[a0_{ci}]; {dst}[{3 * ci + 1}] = tp[a1_{ci}]; {dst}[{3 * ci + 2}] = tp[a2_{ci}];")
    out.append("}")
    return out


def _tile_decode_w(K: int, cb: str, src: str, dst: str, base: int = 0) -> list[str]:
    """Like _tile_decode, from the words already loaded in src[] (_tile_loads). The 64-bit window register is two 32-bit words (funnel shifts by the lane's
    bit offset, written so that a zero shift is defined) and every 16-bit window is ONE extract_bits of a word, or of the 32-bit word that straddles the
    two (windows sit at compile-time offsets): no 64-bit shift per weight."""
    out = []
    for ci, (i0, n) in enumerate(chunk_plan(K)):
        w0, w1, w2 = (f"{src}[{3 * ci + j}]" for j in range(3))
        sh = f"sh_{ci}"
        out.append(f"{{ const uint shi = ({w0} << {sh}) | (({w1} >> 1u) >> (31u - {sh}));       // bits 63..32 of the 64-bit window")
        out.append(f"  const uint slo = ({w1} << {sh}) | (({w2} >> 1u) >> (31u - {sh}));       // bits 31..0")
        out.append("  const uint smid = insert_bits(slo >> 16, shi, 16u, 16u);                  // bits 47..16")
        for i in range(n):
            c = 48 - i * K
            if c >= 32:
                e = f"extract_bits(shi, {c - 32}u, 16u)"
            elif c + 16 <= 32:
                e = f"extract_bits(slo, {c}u, 16u)"
            else:
                e = f"extract_bits(smid, {c - 16}u, 16u)"
            out.append(f"  {dst}[{base + i0 + i}] = dec_{cb}({e});")
        out.append("}")
    return out


def _mm_text(nmb: int, bmu: int, skip_tail: bool, tname: str, ktexpr: str) -> str:
    """The matmul runs of one k step against right-input tensor `tname` (one per 32-row sub-block), from the k tile `ktexpr`."""
    return " ".join(f"{'if (mg + %du < M) ' % (bmu * mb) if skip_tail and mb else ''}"
                    f"{{ auto sA = tA.slice(int({ktexpr}) * 16, int(mg) + {bmu * mb}); op.run(sA, {tname}, cT{mb}); }}"
                    for mb in range(nmb))


@functools.lru_cache(maxsize=None)
def coop_source(Ks: tuple[int, ...], cb: str = "mul1", bmu: int = 32, nmb: int = 2, bns: int = 32, bks: int = 16,
                out_half: bool = True, mode: str = "full", nproj: int = 1, skip_tail: bool = True, fin: str = "", prev: bool = False,
                persist: bool = False, pipe: bool | None = None, persist_y: int = 0, rowend: bool = False, unroll: bool = False) -> str:
    """Inputs: eids i32 [GL], rowoff u32 [nproj * GL + 1] (rows of z = pj * GL + local expert: rowoff[z] .. rowoff[z+1] of A / Y),
    tr0 (tr1) u32 trellis words, off0 (off1) u32 [E+1], kk0 (kk1) u8 [E], A half [rows, KD] (k permuted inside 16-blocks),
    params u32 [KT, NT, GL]. Output Y [rows, N]. nproj 2 = gate and up in one launch (z >= GL selects the second stack).
    A simdgroup owns `bns` columns (bns / 16 tiles) and bmu * nmb rows, steps k by `bks`. Four simdgroups per threadgroup.
    Grid (ceil(NT / (4 * bns / 16)) * 128, ceil(maxrows / (bmu * nmb)), G), threadgroup 128.
    rowend (persist, nproj 2): rowoff has 2 * GL + 2 entries, a gate range and an up range each closed by its own end (see the persist block below).
    prev: extra input Yprev [rows, N] (same dtype as Y); a tile whose expert bitrate is not in Ks copies Yprev instead of writing zeros (per-bitrate passes
    over one slot list: the second pass carries the first pass's tiles through, no host ids)."""
    nt_sg, nk = bns // 16, bks // 16
    ksp = mode == "ks"          # k-split: the grid's y carries (row block, k split); partial sums in fp32 [KS, rows, N] (the GEMV partial layout)
    pf = mode in ("pf", "ks", "pfw") or (mode == "full" and PREFETCH)
    direct = DIRECT_RB and mode != "pfw"
    if mode in ("pf", "nopf", "ks", "pfw"):
        mode = "full"
        if ksp:
            out_half, fin = False, ""
    bodies = []
    pipe = PIPE if pipe is None else pipe
    pipe = bool(pipe) and mode == "full" and pf and direct
    unroll = bool(unroll) and not pipe and mode == "full" and pf and direct
    for K in Ks:
        TB = 256 * K
        nw = 3 * len(chunk_plan(K))
        nsub = nk * nt_sg
        lines = [f"if (K == {K}u) {{",
                 f"  uint B0 = {32 * K}u * nb + {K}u + 8u * G * {K}u + {TB}u - 16u; if (B0 >= {TB}u) B0 -= {TB}u;"]
        if pipe:
            # opalu: decode step k+1 into the idle right-input tensor before the matmul of step k is issued (see PIPE above).
            lines += ["  " + l for l in _chunk_consts(K)]
            tile = lambda kk2, jj, kt="kt": f"tbase + (ulong({kt} + {kk2}u) * NT + nt0 + {jj}u) * {8 * K}u"
            lines += [f"  uint wa[{nsub}][{nw}], wb[{nsub}][{nw}];", "  if (KT0 < KT1) {"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["    " + l for l in _tile_loads(K, tile(kk2, jj, "KT0"), f"wa[{kk2 * nt_sg + jj}]")]
            lines += ["  }"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["  " + l for l in _tile_decode_w(K, cb, f"wa[{kk2 * nt_sg + jj}]", "rB", (kk2 * nt_sg + jj) * 8)]
            lines += [f"  for (uint kt = KT0; kt + {2 * nk}u <= KT1; kt += {2 * nk}u) {{"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["    " + l for l in _tile_loads(K, tile(kk2, jj, f"(kt + {nk}u)"), f"wb[{kk2 * nt_sg + jj}]")]
            lines += ["    " + _mm_text(nmb, bmu, skip_tail, "rB", "kt")]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["    " + l for l in _tile_decode_w(K, cb, f"wb[{kk2 * nt_sg + jj}]", "rB2", (kk2 * nt_sg + jj) * 8)]
            lines += [f"    if (kt + {2 * nk}u < KT1) {{"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["      " + l for l in _tile_loads(K, tile(kk2, jj, f"(kt + {2 * nk}u)"), f"wa[{kk2 * nt_sg + jj}]")]
            lines += ["    }", "    " + _mm_text(nmb, bmu, skip_tail, "rB2", f"kt + {nk}u"),
                      f"    if (kt + {2 * nk}u < KT1) {{"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["      " + l for l in _tile_decode_w(K, cb, f"wa[{kk2 * nt_sg + jj}]", "rB", (kk2 * nt_sg + jj) * 8)]
            lines += ["    }", "  }",
                      # a pass whose k-tile count is not a multiple of 2 * nk finishes in the plain order (its steps were never run above)
                      f"  for (uint kt = KT0 + ((KT1 - KT0) / {2 * nk}u) * {2 * nk}u; kt < KT1; kt += {nk}u) {{"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["    " + l for l in _tile_loads(K, tile(kk2, jj, "kt"), f"wa[{kk2 * nt_sg + jj}]")]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["    " + l for l in _tile_decode_w(K, cb, f"wa[{kk2 * nt_sg + jj}]", "rB", (kk2 * nt_sg + jj) * 8)]
            lines += ["    " + _mm_text(nmb, bmu, skip_tail, "rB", "kt"), "  }"]
        elif unroll:
            # cpftrellis (pf_unroll2): two steps per iteration, word sets wa / wb alternate (no copy between them); the order of loads, decodes and matmuls is the stock one
            lines += ["  " + l for l in _chunk_consts(K)]
            tile = lambda kk2, jj, kt="kt": f"tbase + (ulong({kt} + {kk2}u) * NT + nt0 + {jj}u) * {8 * K}u"
            tiles = [(kk2, jj) for kk2 in range(nk) for jj in range(nt_sg)]
            lines += [f"  uint wa[{nsub}][{nw}], wb[{nsub}][{nw}];", "  if (KT0 < KT1) {"]
            for kk2, jj in tiles:
                lines += ["    " + l for l in _tile_loads(K, tile(kk2, jj, "KT0"), f"wa[{kk2 * nt_sg + jj}]")]
            lines += ["  }", f"  for (uint kt = KT0; kt < KT1; kt += {2 * nk}u) {{", f"    const bool two = kt + {nk}u < KT1;", "    if (two) {"]
            for kk2, jj in tiles:
                lines += ["      " + l for l in _tile_loads(K, tile(kk2, jj, f"(kt + {nk}u)"), f"wb[{kk2 * nt_sg + jj}]")]
            lines += ["    }"]
            for kk2, jj in tiles:
                lines += ["    " + l for l in _tile_decode_w(K, cb, f"wa[{kk2 * nt_sg + jj}]", "rB", (kk2 * nt_sg + jj) * 8)]
            lines += ["    " + _mm_text(nmb, bmu, skip_tail, "rB", "kt"), "    if (two) {", f"      if (kt + {2 * nk}u < KT1) {{"]
            for kk2, jj in tiles:
                lines += ["        " + l for l in _tile_loads(K, tile(kk2, jj, f"(kt + {2 * nk}u)"), f"wa[{kk2 * nt_sg + jj}]")]
            lines += ["      }"]
            for kk2, jj in tiles:
                lines += ["      " + l for l in _tile_decode_w(K, cb, f"wb[{kk2 * nt_sg + jj}]", "rB", (kk2 * nt_sg + jj) * 8)]
            lines += ["      " + _mm_text(nmb, bmu, skip_tail, "rB", f"kt + {nk}u"), "    }", "  }"]
        elif pf and mode == "full":
            lines += ["  " + l for l in _chunk_consts(K)]
            tile = lambda kk2, jj, kt="kt": f"tbase + (ulong({kt} + {kk2}u) * NT + nt0 + {jj}u) * {8 * K}u"
            lines += [f"  uint cw[{nsub}][{nw}], nw_[{nsub}][{nw}];", "  if (KT0 < KT1) {"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["    " + l for l in _tile_loads(K, tile(kk2, jj, "KT0"), f"nw_[{kk2 * nt_sg + jj}]")]
            lines += ["  }", f"  for (uint kt = KT0; kt < KT1; kt += {nk}u) {{", f"    half wv[{nsub * 8}];",
                      f"    for (uint t_ = 0; t_ < {nsub}; ++t_) for (uint w_ = 0; w_ < {nw}; ++w_) cw[t_][w_] = nw_[t_][w_];",
                      f"    if (kt + {nk}u < KT1) {{"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["      " + l for l in _tile_loads(K, tile(kk2, jj, f"(kt + {nk}u)"), f"nw_[{kk2 * nt_sg + jj}]")]
            lines += ["    }"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    lines += ["    " + l for l in _tile_decode_w(K, cb, f"cw[{kk2 * nt_sg + jj}]", "rB" if direct else "wv", (kk2 * nt_sg + jj) * 8)]
        else:
            lines += [f"  for (uint kt = KT0; kt < KT1; kt += {nk}u) {{",
                      f"    half wv[{nk * nt_sg * 8}];"]
            for kk2 in range(nk):
                for jj in range(nt_sg):
                    if mode != "nodec":
                        lines += ["    " + l for l in _tile_decode(K, cb, f"tbase + (ulong(kt + {kk2}u) * NT + nt0 + {jj}u) * {8 * K}u", "wv", (kk2 * nt_sg + jj) * 8)]
        if mode == "nodec":
            lines += [f"    for (uint i = 0; i < {nk * nt_sg * 8}; ++i) wv[i] = half(float(i + kt) * 0.001f);"]
        if pipe or unroll:
            lines += ["}"]
        else:
            if not (pf and mode == "full" and direct):
                lines += ["    #pragma unroll", f"    for (uint i = 0; i < {nk * nt_sg * 8}; ++i) rB[i] = wv[i];"]
            lines += ["    MATMULS" if mode != "nomm" else "    if (kt == 0xffffffffu) { MATMULS }", "  }", "}"]
        bodies.append("\n".join(lines))
    body = " else ".join(bodies)
    mm = " ".join(f"{'if (mg + %du < M) ' % (bmu * mb) if skip_tail and mb else ''}{{ auto sA = tA.slice(int(kt) * 16, int(mg) + {bmu * mb}); op.run(sA, rB, cT{mb}); }}"
                  for mb in range(nmb))
    zero = "\n".join(f"  for (uint16_t i = 0; i < cT{mb}.get_capacity(); ++i) if (cT{mb}.is_valid_element(i)) cT{mb}[i] = 0.0f;" for mb in range(nmb))
    ra = "mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t"
    decl = "\n".join(f"  auto cT{mb} = op.get_destination_cooperative_tensor<{ra}<decltype(tA0)>, {ra}<decltype(rB)>, float>();" for mb in range(nmb))
    store = "\n".join(
        f"  for (uint16_t i = 0; i < cT{mb}.get_capacity(); ++i) if (cT{mb}.is_valid_element(i)) {{ const auto ix = cT{mb}.get_multidimensional_index(i);"
        f" const uint m = mg + {bmu * mb}u + uint(ix[1]); if (m < M) Yp[ulong(m) * N + n0 + uint(ix[0])] = OUTT(cT{mb}[i]); }}" for mb in range(nmb))
    if fin:
        assert bns == 32 and nproj == 1, "the fused finalize needs one 128-column block per threadgroup"
        store = _FIN_STORE.replace("STORE_ROWS", "\n".join(
            f"  for (uint16_t i = 0; i < cT{mb}.get_capacity(); ++i) if (cT{mb}.is_valid_element(i)) {{ const auto ix = cT{mb}.get_multidimensional_index(i);"
            f" fb[({bmu * mb}u + uint(ix[1])) * 128 + sg * 32 + uint(ix[0])] = half(cT{mb}[i]); }}" for mb in range(nmb))
        ).replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD).replace("ROWS", str(bmu * nmb)).replace("FINT", fin)
    src = (_COOP.replace("BODY", body).replace("MATMULS", mm).replace("ZERO", zero).replace("DECL", decl).replace("STORE", store)
           .replace("OUTT", "half" if out_half else "float").replace("ROWS", str(bmu * nmb)).replace("BMU", str(bmu))
           .replace("BNS", str(bns)).replace("BKS", str(bks)).replace("NTSG", str(nt_sg)))
    src = src.replace("PIPERB2", "  auto rB2 = op.get_right_input_cooperative_tensor<half, half, float>();" if pipe else "")
    src = src.replace("TRSEL", _TRSEL[nproj]).replace("FINDT", fin or "half")
    if prev:
        assert not fin and not ksp, "prev: plain store modes only"
        miss = " && ".join(f"K != {K}u" for K in Ks)
        src = src.replace("PREVCOPY", f"    if ({miss}) {{ const uint m1 = min(M, mg + ROWS), w = NTSG * 16u; for (uint i = lane; i < (m1 - mg) * w; i += 32u) {{ "
                                      "const uint m = mg + i / w, c = n0 + i % w; if (c < N) Yp[ulong(m) * N + c] = Yprev[(ulong(row0) + m) * N + c]; } return; }"
                          .replace("ROWS", str(bmu * nmb)).replace("NTSG", str(nt_sg)))
    else:
        src = src.replace("PREVCOPY", "")
    if ksp:
        src = (src.replace("KSINIT", "const uint KS = params[3], KCH = params[4], MTOT = params[5]; const uint ksi = threadgroup_position_in_grid.y % KS; "
                                     "const uint KT0 = min(ksi * KCH, KT), KT1 = min(KT0 + KCH, KT);")
               .replace("MGEXPR", "(threadgroup_position_in_grid.y / KS)").replace("YPOFF", "(ulong(ksi) * MTOT + row0)"))
    else:
        src = src.replace("KSINIT", "const uint KT0 = 0, KT1 = KT;").replace("MGEXPR", "threadgroup_position_in_grid.y").replace("YPOFF", "ulong(row0)")
    if persist:
        # pf_moe_nosync: the row blocks of an expert are walked by PERSIST_Y threadgroups instead of one per block, so the row count never reaches
        # the host. Every block is the unrolled body of the sync path, run once: the same tiles, the same per-row arithmetic, bit-identical output.
        assert not ksp and not fin and not prev, "persist: plain row-block loop only"
        old = f"const uint mg = threadgroup_position_in_grid.y * {bmu * nmb};"
        assert src.count(old) == 1, "persist: row-block line moved"
        # grid.y = PERSIST_Y threadgroups per (tile, expert), block j = PERSIST_Y threadgroups apart: the row count stays on the device and the
        # launch keeps the same parallelism as one threadgroup per block while an expert has <= PERSIST_Y blocks (above that it loops).
        src = src.replace(old, f"for (uint mg = threadgroup_position_in_grid.y * {bmu * nmb}; mg < M; mg += {(persist_y or PERSIST_Y) * bmu * nmb}) {{")
        src = src + "\n}"
        if rowend:
            # pf_nosync_rowend: nproj = 2 reads rowoff with GL + 1 entries per projection ([gate starts .. gate end, up starts .. up end], 2 * GL + 2 in all)
            # instead of 2 * GL + 1 with the up block glued to the gate block. In the glued form the last gate expert ends where the FIRST up expert starts
            # (class start + P), so it walks every slot between its own start and that bound, as many as P - (the class's rows before it): garbage rows
            # that nothing reads, on 8 threadgroups, at every class of every layer.
            assert nproj == 2, "rowend: the gate / up launch only"
            old2 = "const uint row0 = rowoff[z], M = rowoff[z + 1] - row0;"
            assert src.count(old2) == 1, "rowend: row range line moved"
            src = src.replace(old2, "const uint zo = z + (z >= GL ? 1u : 0u); const uint row0 = rowoff[zo], M = rowoff[zo + 1] - row0;")
    return src


# Finalize folded into the epilogue (bns = 32: the 4 simdgroups of a threadgroup own one 128-column block of the output). The accumulators go to
# threadgroup memory as half (the rounding of the fp16 partial the separate finalize launch reads), then every simdgroup runs H128 / sqrt(128) * svh on
# its rows with the same butterfly and operand order as metal_src.final_source: the bits equal the two-launch path.
_FIN_STORE = r"""
  threadgroup half fb[ROWS * 128];
STORE_ROWS
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = sg; r < ROWS; r += 4) {
    if (mg + r >= M) continue;
    const half4 hv = *((threadgroup half4*)(fb + r * 128 + lane * 4));
    float v0 = float(hv.x), v1 = float(hv.y), v2 = float(hv.z), v3 = float(hv.w);
BUTTERFLY
    const float4 sv = float4(*((const device half4*)(svh + threadgroup_position_in_grid.x * 128 + lane * 4)));
    const float4 o4 = float4(v0, v1, v2, v3) * RSQRT * sv;
    *((device vec<FINT, 4>*)(Yf + ulong(mg + r) * N + threadgroup_position_in_grid.x * 128 + lane * 4)) = vec<FINT, 4>(o4);
  }
"""

_TRSEL = {1: "const device uint* trellis = tr0; const device uint* off = off0; const device uint8_t* kk = kk0;",
          2: ("const bool pj = z >= GL; const device uint* trellis = pj ? tr1 : tr0; "
              "const device uint* off = pj ? off1 : off0; const device uint8_t* kk = pj ? kk1 : kk0;")}

_COOP = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint z = threadgroup_position_in_grid.z;
    const uint KT = params[0], NT = params[1], GL = params[2];
    KSINIT
    const uint e = uint(eids[z % GL]);
    TRSEL
    const uint row0 = rowoff[z], M = rowoff[z + 1] - row0;
    const uint mg = MGEXPR * ROWS;
    const uint nt0 = (threadgroup_position_in_grid.x * 4 + sg) * NTSG;
    if (mg >= M || nt0 >= NT) return;
    const uint K = uint(kk[e]);
    const ulong KD = ulong(KT) * 16, N = ulong(NT) * 16;
    const uint n0 = nt0 * 16;
    const device uint* tbase = trellis + off[e];

    constexpr auto desc = matmul2d_descriptor(BMU, BNS, BKS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(A + ulong(row0) * KD), dextents<int32_t, 2>(int(KD), int(M)));
    device OUTT* Yp = (device OUTT*)(Y + YPOFF * N);
    device FINDT* Yf = (device FINDT*)(Y + ulong(row0) * N);
PREVCOPY
    auto tA0 = tA.slice(0, int(mg));
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
PIPERB2
DECL
ZERO
    const ushort i0 = 0;
    const auto ix0 = rB.get_multidimensional_index(i0);
    const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
BODY
STORE
"""


# ---------------------------------------------------------------------------------------------------------------- row map (pf_moe_rowmap)
# The gate / up GEMM reads the token rows of x through the slot -> token map and applies the projection's input transform
# (multiply by the expert's suh, H128 / sqrt(128), k permutation inside 16-blocks) while it fills the A tile in threadgroup
# memory. Same values as `metal_pf.rot_pf_source(perm=True)` followed by the device-A kernel, so the bits match; no [2P, K]
# fp16 copy is written or read. Every threadgroup of a row block and 128-column slab stages the rows it needs (the weight
# decode is already repeated per column slab), in 128-k blocks because the Hadamard spans 128 columns.
_COOP_RM = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint z = threadgroup_position_in_grid.z;
    const uint KT = params[0], NT = params[1], GL = params[2];
    const uint e = uint(eids[z % GL]);
    TRSEL
    const uint row0 = rowoff[z], M = rowoff[z + 1] - row0;
    const uint mg = threadgroup_position_in_grid.y * ROWS;
    const uint nt0 = (threadgroup_position_in_grid.x * NSG + sg) * NTSG;
    if (mg >= M) return;
    const uint K = uint(kk[e]);
    const ulong KD = ulong(KT) * 16, N = ulong(NT) * 16;
    const uint n0 = nt0 * 16;
    const uint rblk = min((uint)ROWS, M - mg);
    const device uint* tbase = trellis + off[e];
    const device half* suh = SUHSEL + ulong(e) * KD;

    constexpr auto desc = matmul2d_descriptor(BMU, BNS, BKS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    threadgroup half atg[ROWS * 128];
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA((threadgroup half*)atg, dextents<int32_t, 2>(128, ROWS));
    device OUTT* Yp = (device OUTT*)(Y + ulong(row0) * N);
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
DECL
ZERO
    const ushort i0 = 0;
    const auto ix0 = rB.get_multidimensional_index(i0);
    const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
    const bool live = nt0 < NT;
    // rows past rblk (the padding of a tail block) are never transformed: their A rows stay zero, as when they were staged from a zero input
    for (uint i = sg * 32u + lane; i < ROWS * 128u; i += NSG * 32u) atg[i] = half(0.0f);
    for (uint kb = 0; kb < KT / 8u; ++kb) {
      // four rows per simdgroup at a time: the four row loads are issued together, so the butterflies hide their latency
      for (uint r0 = sg * 4u; r0 < ROWS; r0 += 4u * NSG) {
        float4 xv[4], sv[4];
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
          const uint r = r0 + j;
          if (r < rblk) {
            const ulong o = ulong(tok[row0 + mg + r]) * KD + kb * 128u + lane * 4;
            xv[j] = float4(*((const device half4*)(x + o)));
            sv[j] = float4(*((const device half4*)(suh + kb * 128u + lane * 4)));
          } else { xv[j] = float4(0.0f); sv[j] = float4(0.0f); }
        }
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
          const uint r = r0 + j;
          if (r >= rblk) continue;                                   // rblk <= ROWS: padded rows skip the butterfly
          float v0 = xv[j].x * sv[j].x, v1 = xv[j].y * sv[j].y, v2 = xv[j].z * sv[j].z, v3 = xv[j].w * sv[j].w;
BUTTERFLY
          { const float4 pv = float4(v0, v1, v2, v3) * RSQRT;
            threadgroup half* pd = (threadgroup half*)(atg + r * 128) + (lane >> 2) * 16 + 8 * (lane & 1) + 2 * ((lane >> 1) & 1);
            pd[0] = half(pv.x); pd[1] = half(pv.y); pd[4] = half(pv.z); pd[5] = half(pv.w); }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      BODY
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live) return;
STORE
"""


@functools.lru_cache(maxsize=None)
def coop_rm_source(Ks: tuple[int, ...], cb: str = "mul1", bmu: int = 32, nmb: int = 2, bns: int = 32, bks: int = 16,
                   out_half: bool = True, nproj: int = 1, skip_tail: bool = True, nsg: int = 4) -> str:
    """Row-map twin of `coop_source` (mode "full", no finalize): inputs eids i32 [GL], rowoff u32 [nproj * GL + 1],
    tr0 (tr1) u32, off0 (off1) u32 [E+1], kk0 (kk1) u8 [E], x half [T, KD], tok u32 [rows] (A row -> token row),
    suh0 (suh1) half [E, KD], params u32 [KT, NT, GL]. Output Y [rows, N]."""
    nt_sg, nk = bns // 16, bks // 16
    assert 8 % nk == 0, "the row map stages A in 128-k blocks: bks must divide 128"
    decl = "\n".join(f"  auto cT{mb} = op.get_destination_cooperative_tensor<{ 'mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t' }<decltype(tA)>, "
                     f"{ 'mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t' }<decltype(rB)>, float>();" for mb in range(nmb))
    zero = "\n".join(f"  for (uint16_t i = 0; i < cT{mb}.get_capacity(); ++i) if (cT{mb}.is_valid_element(i)) cT{mb}[i] = 0.0f;" for mb in range(nmb))
    store = "\n".join(
        f"  for (uint16_t i = 0; i < cT{mb}.get_capacity(); ++i) if (cT{mb}.is_valid_element(i)) {{ const auto ix = cT{mb}.get_multidimensional_index(i);"
        f" const uint m = mg + {bmu * mb}u + uint(ix[1]); if (m < M) Yp[ulong(m) * N + n0 + uint(ix[0])] = OUTT(cT{mb}[i]); }}" for mb in range(nmb))
    bodies = []
    for K in Ks:
        tb = 256 * K
        lines = [f"if (K == {K}u) {{",
                 f"  uint B0 = {32 * K}u * nb + {K}u + 8u * G * {K}u + {tb}u - 16u; if (B0 >= {tb}u) B0 -= {tb}u;",
                 f"  for (uint ks = 0; ks < 8u; ks += {nk}u) {{"]
        for t in range(nk):
            kt = f"kb * 8u + ks + {t}u"
            for jj in range(nt_sg):
                lines += ["    " + l for l in _tile_decode(K, cb, f"tbase + (ulong({kt}) * NT + nt0 + {jj}u) * {8 * K}u", "rB", (t * nt_sg + jj) * 8)]
        mm = " ".join(f"if (live{' && rblk > %du' % (bmu * mb) if skip_tail and mb else ''}) {{ auto sA = tA.slice(int(ks) * 16, {bmu * mb}); op.run(sA, rB, cT{mb}); }}"
                      for mb in range(nmb))
        lines += ["    " + mm, "  }", "}"]
        bodies.append("\n".join(lines))
    body = " else ".join(bodies)
    src = (_COOP_RM.replace("BODY", body).replace("ZERO", zero).replace("DECL", decl).replace("STORE", store)
           .replace("BUTTERFLY", S._BUTTERFLY).replace("RSQRT", S.RSQRT_HAD)
           .replace("OUTT", "half" if out_half else "float").replace("ROWS", str(bmu * nmb)).replace("BMU", str(bmu))
           .replace("BNS", str(bns)).replace("BKS", str(bks)).replace("NTSG", str(nt_sg)).replace("NSG", str(nsg)))
    src = src.replace("TRSEL", _TRSEL[nproj]).replace("SUHSEL", "suh0" if nproj == 1 else "((z >= GL) ? suh1 : suh0)")
    return src


# ---------------------------------------------------------------------------------------------------------------- launcher
_kc: dict = {}
_ids = itertools.count()


# ---------------------------------------------------------------------------------------------------------------- capability probe
# One probe per process, cached: KYOJIN_NAX = auto (default) | 1 | 0.
#   auto  NAX when the GPU is applegpu_g17 or newer (M5, A19) AND a 32x32x64 `matmul2d` compiles, runs and matches numpy; M1-M4 take the non-NAX kernels.
#   1     skip the architecture test (a Metal 4 GPU the table does not know yet), still require the compile + run probe: never a crash on a GPU without it.
#   0     NAX off everywhere (bitwise day-1 gates; the same switch simulates an M1-M4 on the M5).
# The per-site switches (KYOJIN_PF_FUSED, KYOJIN_DENSE_FUSED, KYOJIN_ATTN_PF, KYOJIN_MIX_PF) take "1" / "0" / "auto" through `site_on`.
MODE = os.environ.get("KYOJIN_NAX", "auto")
_PROBE = r"""
    constexpr auto desc = matmul2d_descriptor(32, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)a, dextents<int32_t, 2>(64, 32));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*)b, dextents<int32_t, 2>(64, 32));
    auto sA = tA.slice(0, 0); auto sB = tB.slice(0, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    op.run(sA, sB, cT);
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
        const auto ix = cT.get_multidimensional_index(i);
        c[uint(ix[1]) * 32 + uint(ix[0])] = cT[i];
    }
"""
_status: dict | None = None
_forced: bool | None = None


def _architecture() -> str:
    try:
        return str(mx.device_info()["architecture"]) if hasattr(mx, "device_info") else str(mx.metal.device_info()["architecture"])
    except Exception:
        return ""


def _arch_nax(arch: str) -> bool:
    digits = "".join(itertools.takewhile(str.isdigit, arch.split("applegpu_g")[-1])) if "applegpu_g" in arch else ""
    return bool(digits) and int(digits) >= 17


def _compile_probe() -> tuple[bool, str]:
    """Compile and run one matmul2d tile, compare with numpy: (ok, reason)."""
    try:
        import numpy as np
        rng = np.random.default_rng(1)
        a = rng.standard_normal((32, 64)).astype(np.float16)
        b = rng.standard_normal((32, 64)).astype(np.float16)
        k = mx.fast.metal_kernel(name="kyojin_nax_probe", header=HEADER_NAX, input_names=["a", "b"], output_names=["c"], source=_PROBE)
        c = k(inputs=[mx.array(a), mx.array(b)], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(32, 32)], output_dtypes=[mx.float32])[0]
        mx.eval(c)
        ref = a.astype(np.float32) @ b.astype(np.float32).T
        err = float(np.abs(np.array(c) - ref).max() / (np.abs(ref).max() + 1e-9))
        return err < 1e-2, f"matmul2d probe max-rel {err:.1e}" if err < 1e-2 else f"matmul2d probe gave a WRONG result (max-rel {err:.1e})"
    except Exception as e:
        msg = str(e).strip().splitlines()
        return False, "matmul2d probe failed to compile/run" + (": " + msg[0][:120] if msg else "")


def nax_status() -> dict:
    """The cached probe result {mode, arch, arch_nax, compiles, on, reason}; the first call probes and logs one line (KYOJIN_NAX_LOG=0 silences stderr)."""
    global _status
    if _status is None:
        arch = _architecture()
        arch_nax = _arch_nax(arch)
        if MODE == "0":
            compiles, reason = False, "KYOJIN_NAX=0"
        elif MODE != "1" and not arch_nax:
            compiles, reason = False, f"architecture {arch or 'unknown'} has no neural accelerators"
        else:
            compiles, reason = _compile_probe()
        _status = dict(mode=MODE, arch=arch, arch_nax=arch_nax, compiles=compiles, on=compiles, reason=reason)
        line = f"[kyojin] NAX {'ON' if compiles else 'OFF (non-NAX kernels)'}: {arch or 'unknown gpu'}, KYOJIN_NAX={MODE}, {reason}"
        logging.getLogger("kyojin.nax").info(line)
        if os.environ.get("KYOJIN_NAX_LOG", "1") != "0":
            print(line, file=sys.stderr, flush=True)
    return _status


def nax_available() -> bool:
    """True when the NAX kernels may run: a GPU with neural accelerators (applegpu_g17 and newer: M5, A19) whose matmul2d probe passed, and KYOJIN_NAX != 0."""
    return nax_status()["on"]


def site_on(setting: str) -> bool:
    """A call site's switch ("1" force, "0" off, anything else auto): auto follows the device probe; "1" forces the NAX path wherever matmul2d
    compiles and runs (KYOJIN_NAX=0 still wins); a GPU without it falls back to the non-NAX kernel."""
    global _forced
    if setting == "0" or MODE == "0":
        return False
    if setting == "1":
        if _forced is None:
            _forced = True if nax_status()["compiles"] else _compile_probe()[0]
        return _forced
    return nax_available()


def permute_k(a):
    """numpy: out[..., 16 b + k'] = a[..., 16 b + KPERM[k']], the layout of A that `expert_gemm` expects (host-side twin of the perm=True kernels)."""
    import numpy as np
    idx = (np.arange(a.shape[-1]).reshape(-1, 16) // 16 * 16 + np.array(KPERM)[None, :]).reshape(-1)
    return a[..., idx]


# Time of one row block (bmu * nmb rows) for a group of 8 experts on the Air (ms, w1 5120 x 2304 K = 3, DS4.1; tools/sweep_nax_cfg.py, quiet Air):
# the cost model of `pick_cfg`. A block pays the weight decode once, so 16 rows cost 0.78 and 64 rows 1.52: bigger blocks amortize it.
NOSYNC_Y = int(os.environ.get("KYOJIN_PF_NOSYNC_Y", "0"))     # pf_nosync_y: 0 = PERSIST_Y
PERSIST_Y = 8          # pf_moe_nosync: row-block threadgroups per (tile, expert) in the persist GEMM (blocks are strided over them)
_BLOCK_COST = {16: 0.78, 32: 1.01, 64: 1.52}
# M5 Ultra (64 cores, tools/sweep_nax_cfg.py-style sweep on the real DS4.1 layer, 8 experts, ms per launch at M rows per expert): the weight decode is almost the whole
# block cost, so a bigger block costs about the same as a small one: 16 rows 0.35, 32 rows 0.37, 64 rows 0.42 (the Air table above makes (16,1) win the small experts)
# Second Ultra sweep (sustained clocks: 40 launches per timing after a burn-in; the first one measured single launches at a ramping clock): one 16-column tile per
# simdgroup (bns = 16) halves the registers of the kernel and the occupancy more than doubles - 8 experts, ms per launch: 32 rows 0.347 -> 0.118, 64 rows 0.486 -> 0.200,
# 128 rows 0.635 -> 0.395, 256 rows 0.967 -> 0.689 with (32,2,32,32). Block costs of the bns = 16 tiles: 16 rows 0.097, 32 rows 0.118, 64 rows 0.20.
_BLOCK_COST_ULTRA = {16: 0.097, 32: 0.118, 64: 0.20}
_BLOCK_CFG_ULTRA = {16: (16, 1, 16, 64), 32: (32, 1, 16, 64), 64: (32, 2, 16, 64)}
_BIG_CFG_ULTRA = (32, 2, 32, 32)        # experts with >= 160 rows on average
_COST_MODE = os.environ.get("KYOJIN_NAX_COST", "air")
# "mixed": the Air table with the 16-row block on bns = 16 tiles (production-scale launch, 384 experts x 16 rows, gate + up: 18.5 -> 8.4 ms; 32 / 64-row blocks stay on bns = 32: 10.5 / 17.2 ms vs 11.7 / 21.6)
# Recheck 30/10 (tools/pfroof_buckets.py, 16 GiB cache limit, register path): 16 rows per expert gate + up air16 7.95 ms vs b16 8.53, so bns = 32 wins and the default stays;
# the 18.5 ms above was fresh-buffer page faults under a 256 MiB cache limit. Measured block costs (0.76 / 1.01 / 1.68) match the default table.
_BLOCK_COST_MIXED = {16: 0.81, 32: 1.01, 64: 1.65}
_BLOCK_CFG_MIXED = {16: (16, 1, 16, 64), 32: (32, 1, 32, 64), 64: (32, 2, 32, 64)}
if _COST_MODE == "mixed":
    _BLOCK_COST = _BLOCK_COST_MIXED
if _COST_MODE == "ultra":   # OFF by default: the 8-expert microbenches that suggested it do not fill the GPU; at 384 experts (production S3 call) the tables tie
    _BLOCK_COST = _BLOCK_COST_ULTRA
_BLOCK_CFG = {16: (16, 1, 32, 64), 32: (32, 1, 32, 64), 64: (32, 2, 32, 64)}
if _BLOCK_COST is _BLOCK_COST_ULTRA:
    _BLOCK_CFG = _BLOCK_CFG_ULTRA
if _BLOCK_COST is _BLOCK_COST_MIXED:
    _BLOCK_CFG = _BLOCK_CFG_MIXED


def _expert_cost(c: int, rows: int, nmb: int, table: dict | None = None) -> float:
    """Summed block cost of an expert with `c` rows under a config of `rows` = bmu * nmb rows per block. A tail block that fits in the first
    sub-block (SKIP_TAIL) costs what a one-sub-block config of half the rows costs."""
    table = table or _BLOCK_COST
    full, tail = divmod(c, rows)
    cost = full * table[rows]
    if tail:
        half = rows // nmb
        cost += table[half] if SKIP_TAIL and nmb > 1 and tail <= half else table[rows]
    return cost


# pf_blk128 (KYOJIN_PF_BLK128, default 0): a fourth row block of 128 rows (bmu 32 x nmb 4, one 16-column tile per simdgroup so the accumulators stay at 64 registers) for
# the routed prefill GEMMs. Every row block of an expert decodes its weights again and the decode is the ALU limit of these kernels, so an expert with 128+ rows
# (hot experts, and every expert once the MoE pass is 8192 rows) pays half the decodes per row of the 64-row block. The cost of the block is NOT measured: it is
# extrapolated linearly from the 32 and 64-row entries of the table (cost64 + 2 * (cost64 - cost32)), which is what `tools/bench_pf_levers.py` checks on the Ultra.
BLK128 = os.environ.get("KYOJIN_PF_BLK128", "0") != "0"
_CFG_128 = (32, 4, 16, 64)


def block_costs(with128: bool = False) -> dict:
    """The block cost table of pick_cfg, plus the extrapolated 128-row block when asked."""
    if not with128:
        return _BLOCK_COST
    return {**_BLOCK_COST, 128: _BLOCK_COST[64] + 2.0 * (_BLOCK_COST[64] - _BLOCK_COST[32])}


def pick_cfg(counts, allow128: bool = False) -> tuple[int, int, int, int]:
    """(bmu, nmb, bns, bks) for experts that have `counts` rows each: the row block that minimizes the summed block cost (every row
    block of an expert decodes its weights again). The cfgs of _BLOCK_CFG step k by 64 (`expert_gemm` falls back to 32 when kt % 4).
    allow128 (the routed prefill GEMMs, pf_blk128): the 128-row block is a candidate too."""
    cfg = _pick_cfg(counts, allow128)
    if BMU16 and allow128 and cfg[0] == 32:                         # allow128 = the plain routed GEMM pair (the epilogue / row map kernels are not sized for it)
        FD.hit("pf_bmu16")
        return (16, 2 * cfg[1]) + tuple(cfg[2:])
    return cfg


def _pick_cfg(counts, allow128: bool = False) -> tuple[int, int, int, int]:
    with128 = allow128 and BLK128
    table = block_costs(with128)
    cfgs = {**_BLOCK_CFG, 128: _CFG_128} if with128 else _BLOCK_CFG
    best = min(table, key=lambda b: sum(_expert_cost(int(c), b, cfgs[b][1], table) for c in counts))
    if best == 128:
        return _CFG_128
    if _BLOCK_CFG is _BLOCK_CFG_ULTRA and best == 64 and len(counts) and sum(int(c) for c in counts) >= 160 * len(counts):
        return _BIG_CFG_ULTRA
    return _BLOCK_CFG[best]


class SingleStack:
    """One dense EXL3 linear (or `groups` equal slices laid one after the other in `words`, e.g. the 8 groups of DS4.1 wo_a) shaped as a stack
    for `expert_gemm` (words u32 flat, kt x nt tiles per slice, one bitrate K)."""

    def __init__(self, words: mx.array, K: int, kt: int, nt: int, cb: str = "mul1", groups: int = 1):
        self.words, self.kt, self.nt, self.cb, self.ks = words, kt, nt, cb, (K,)
        self.bview = CB.make_view(words)                              # KYOJIN_DENSE_VIEWS: bound by metal_nax_sh / _shw instead of the whole array
        per = kt * nt * 8 * K
        assert int(words.size) == groups * per, (int(words.size), groups, per)
        self.offset = _device_array(mx.array([g * per for g in range(groups + 1)], dtype=mx.uint32))      # padded once: a call adds no concatenate / fill node
        self.k = _device_array(mx.array([K] * groups, dtype=mx.uint8))
        mx.eval(self.offset, self.k)
        self.kin, self.nout, self.experts = kt * 16, nt * 16, groups
        self.bound = words                                                                  # a dense linear: no bank view (same attribute as moe.Stack.bound)


_params: dict = {}      # (kt, nt, GL[, KS, KCH, rows]) -> params array, built once


def nproj_of(stacks) -> int:
    return len(stacks)


_FIN_T = {mx.float16: "half", mx.bfloat16: "bfloat", mx.float32: "float"}


def _device_array(a: mx.array) -> mx.array:
    """MLX passes inputs of a few bytes in the `constant` address space; the kernel reads the stack tables through device pointers."""
    return a if a.size >= 16 else mx.concatenate([a, mx.zeros(16 - a.size, dtype=a.dtype)])


def expert_gemm(stacks, eids: mx.array, rowoff: mx.array, A: mx.array | None, rows_out: int, maxrows: int,
                cfg: tuple[int, int, int, int] | None = None, out_half: bool = True, mode: str = "full",
                fin: tuple[mx.array, mx.Dtype] | None = None, ksplit: int = 1, ks: tuple[int, ...] | None = None,
                prev: mx.array | None = None, rowmap: tuple | None = None, persist: bool = False, rowend: bool = False) -> mx.array:
    """Y[rows_out, N] (half) with Y[r] = A[r] @ Wr_e^T for the rows r of expert eids[z % GL] given by rowoff (z = pj * GL + local expert).
    stacks: 1 stack, or 2 (gate and up: rows pj * n .. of A / Y belong to stacks[pj]); A half [rows, KD] with the k permutation inside
    16-blocks (`kernels.rot(..., perm=True)`, `metal_pf.rot_pf_source(..., perm=True)`). The decoded weights never leave the registers.
    rowmap=(x, tok, suh0[, suh1]): A is None; the kernel reads x [T, KD] (no k permutation) at the token row tok[r] and applies the
    projection's own input transform (suh, H128 / sqrt(128), permutation) per tile (pf_moe_rowmap).
    persist (pf_moe_nosync): PERSIST_Y threadgroups per (tile, expert) share that expert's row blocks (block j on threadgroup j % PERSIST_Y)
    instead of one threadgroup per block, so the row counts stay on the device (`rowoff`); rows are computed block by block, same bits.
    `maxrows` is then ignored (the block count comes from rowoff).
    rowend (persist, two stacks; pf_nosync_rowend): rowoff = [gate starts, gate end, up starts, up end] (2 * GL + 2 entries), so the last gate expert stops at its class end."""
    s0, nproj, GL = stacks[0], len(stacks), int(eids.shape[0])
    if persist:
        assert not fin and prev is None and ksplit == 1 and mode == "full" and rowmap is None, "persist: plain full-mode row loop only"
    assert not rowend or (persist and len(stacks) == 2), "rowend: persist gate / up only"
    if (not persist and TG_MIN_ROWS and maxrows >= TG_MIN_ROWS and mode == "full" and not fin and ksplit == 1 and s0.kt % 4 == 0 and s0.nt % 2 == 0
            and rowmap is None and (cfg is None or cfg[1] > 1)):
        from . import metal_nax_tg as TG                                  # decode-once kernel (threadgroup-shared weight slab): same bits, opt-in only (Ultra 384 experts: 20-35 % slower than the register path at 32-256 rows)
        return TG.expert_gemm_tg(stacks, eids, rowoff, A, rows_out, maxrows, out_half, ks, prev)
    Ks = ks or tuple(sorted(set().union(*[set(s.ks) for s in stacks])))      # ks: the bitrates of the experts in eids only (a smaller kernel)
    bmu, nmb, bns, bks = cfg or _BLOCK_CFG[64]
    if bks % 64 == 0 and s0.kt % (bks // 16):      # k steps of bks / 16 tiles: fall back to 32 (kt is even for every fused linear)
        bks = 32
    if ksplit > 1:
        assert not fin and nproj_of(stacks) == 1
        mode, out_half = "ks", False
    rm = rowmap is not None
    if rm:
        assert not fin and prev is None and ksplit == 1 and mode == "full" and nproj_of(stacks) in (1, 2)
        assert s0.kin % 128 == 0, "the row map stages A in 128-k blocks"
        assert len(rowmap) == 2 + nproj_of(stacks), "rowmap = (x, tok, suh0[, suh1])"
    fint = _FIN_T[fin[1]] if fin else ""
    sgt = RM_NSG if rm else 4                     # simdgroups per threadgroup (the row map stages A once per threadgroup)
    nsg = sgt * bns // 16
    cbk = dec_cb(s0.cb)
    pipe = PIPE and not rm and not persist and mode == "full"
    unroll = UNROLL2 and not pipe and not rm and mode == "full" and DIRECT_RB and PREFETCH
    py = (NOSYNC_Y or PERSIST_Y) if persist else 0                  # threadgroups per (tile, expert) of the persist GEMM
    key = (Ks, cbk, nproj, bmu, nmb, bns, bks, out_half, mode, SKIP_TAIL, fint, prev is not None, rm, sgt, persist, pipe, unroll) + ((py,) if py != PERSIST_Y and persist else ()) + (("re",) if rowend else ())
    kern = _kc.get(key)
    if kern is None:
        names = ["eids", "rowoff"] + [f"tr{i}" for i in range(nproj)] + [f"off{i}" for i in range(nproj)] + [f"kk{i}" for i in range(nproj)]
        if rm:
            names = names + ["x", "tok"] + [f"suh{i}" for i in range(nproj)] + ["params"]
            src = coop_rm_source(Ks, cbk, bmu, nmb, bns, bks, out_half, nproj, SKIP_TAIL, RM_NSG)
            suffix = "_rm"
        else:
            names = names + ["A", "params"] + (["svh"] if fint else []) + (["Yprev"] if prev is not None else [])
            src = coop_source(Ks, cbk, bmu, nmb, bns, bks, out_half, mode, nproj, SKIP_TAIL, fint, prev is not None, persist, pipe, py if py != PERSIST_Y else 0, rowend, unroll)
            suffix = ("_f" + fint) if fint else ""
        kern = _kc[key] = mx.fast.metal_kernel(
            name=f"kyojin_nax_{cbk}_{'_'.join(map(str, Ks))}_p{nproj}_{bmu}_{nmb}_{bns}_{bks}_{mode}{suffix}{'_pv' if prev is not None else ''}{'_pl' if persist else ''}{'_re' if rowend else ''}{'_pp' if pipe else ''}{'_u2' if unroll else ''}_{next(_ids)}",
            header=HEADER_NAX, input_names=names, output_names=["Y"], source=src)
    pk = (s0.kt, s0.nt, GL)
    if ksplit > 1:
        pk = pk + (ksplit, -(-s0.kt // (ksplit * (bks // 16))) * (bks // 16), rows_out)
    params = _params.get(pk)
    if params is None:
        params = _params[pk] = mx.array(pk, dtype=mx.uint32)
    mid = [eids, rowoff] + [s.bound for s in stacks] + [_device_array(s.offset) for s in stacks] + [_device_array(s.k) for s in stacks]
    if rm:
        ins = mid + [rowmap[0], rowmap[1]] + list(rowmap[2:]) + [params]
    else:
        ins = mid + [A, params] + ([fin[0]] if fin else []) + ([prev] if prev is not None else [])
    odt = fin[1] if fin else (mx.float16 if out_half else mx.float32)
    ks = max(ksplit, 1)
    gy = py if persist else -(-maxrows // (bmu * nmb)) * ks
    return kern(inputs=ins, grid=(-(-s0.nt // nsg) * 32 * sgt, gy, nproj * GL), threadgroup=(32 * sgt, 1, 1),
                output_shapes=[(rows_out * ks, s0.nout)], output_dtypes=[odt])[0]
