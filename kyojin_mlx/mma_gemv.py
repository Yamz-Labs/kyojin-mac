# SPDX-License-Identifier: Apache-2.0
"""Small-batch GEMV on simdgroup_matrix (4..16 rows). Same inputs / outputs as metal_src.gemv_source (xh f32 [rows, KD], trellis, params [KT, NT, TGK];
part f32 [TGK, rows, N]), so rot and finalize are the ones of the GEMV.

C^T = W^T x^T: the decoded weights are the A operand (8 n x 8 k fragment), the rotated input the B operand (8 k x 8 rows), accumulators C^T[n][row].
Fragment layout of simdgroup_matrix 8x8 (thread_elements): element 0 / 1 sit at (fm, fn) / (fm, fn + 1) with
fm = 4 * bit4(lane) + ((lane >> 1) & 3), fn = 4 * bit3(lane) + 2 * bit0(lane).
Lane (g = fm, c = fn / 2) of a simdgroup decodes, for each of the 4 tiles it covers, positions p = 8c + 4nh + 2kh + e of run g (nh: column half g / g+8,
kh: k half, e in 0..1): the weights of A[n = 8 nh + g][l = 2c + e] for k = 8 kh + l. The 8 windows of a lane are bits [(8c+1)K-16 + jK, +16) of the run's stream:
the lane loads the 2-4 words that hold them (runtime word offset) and funnel-shifts the stream by the runtime bit offset once, so the 8 windows are
compile-time extracts. Decode work per weight = the GEMV's; the row FMAs become MMA on the fragments."""
from __future__ import annotations

import functools

import mlx.core as mx

from . import cbill as CB
from . import kernels as KN, metal_src as S

KS = tuple(range(1, 9))                                           # bit widths served

_SRC = r"""
    constexpr uint TW = 8 * BITS;
    constexpr uint WS = WSP;                                // k-splits per threadgroup (x 2 column halves)
    constexpr uint RB = (MR + 7) / 8;                       // row blocks of 8
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;        // 128-column output block
    const uint kblock = threadgroup_position_in_grid.y;
    const uint KT = params[0], NT = params[1], TGK = params[2];
    const uint KD = KT * 16, N = NT * 16;
    const uint half_ = warp & 1u, split = warp >> 1;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    const uint g = fm;
    const uint nt0 = blk * 8 + half_ * 4;

    const uint S = TGK * 4;                                 // total k-splits; kblock runs over TGK * 4 / WS
    const uint chunk = (KT + S - 1) / S;
    const uint ks0 = (kblock * WS + split) * chunk;
    const uint myn = ks0 < KT ? min(chunk, KT - ks0) : 0u;

    // stream bits of this lane's windows: window j ends at (8c + j + 1) * BITS; the string starts 16 bits before the first window's end
    const int bstart = int((8u * c + 1u) * BITS) - 16;
    const int iw0 = bstart >> 5;                            // word index relative to the run (-1: the word before, circular)
    const uint ob = uint(bstart) & 31u;                     // bit offset of the first window inside word iw0
    const int ir = int(BITS * g) + iw0;                     // run-relative index of the first word
    const uint i0 = ir >= 0 ? uint(ir) : TW - 1;
    const uint ilast = BITS * g + BITS - 1;                 // last word of the run

    simdgroup_matrix<float, 8, 8> acc[4][2][RB];
    _Pragma("clang loop unroll(full)") for (uint t = 0; t < 4; ++t) for (uint h = 0; h < 2; ++h) for (uint r = 0; r < RB; ++r) acc[t][h][r] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const device uint* tp = trellis + (ulong(ks0) * NT + nt0) * TW;
    const ulong tstride = ulong(NT) * TW;
    const device float* xk = xh + ks0 * 16 + fm;
    for (uint i = 0; i < myn; ++i) {
        // B fragments: x^T[l][row] = x[row][16 i + 8 kh + l]; this thread: l = fm, rows 8 rb + fn, +1 (clamped)
        simdgroup_matrix<float, 8, 8> B[2][RB];
        _Pragma("clang loop unroll(full)") for (uint kh = 0; kh < 2; ++kh) for (uint r = 0; r < RB; ++r) {
            const uint r0 = min(8u * r + fn, MR - 1u), r1 = min(8u * r + fn + 1u, MR - 1u);
            B[kh][r].thread_elements()[0] = xk[r0 * KD + 8 * kh];
            B[kh][r].thread_elements()[1] = xk[r1 * KD + 8 * kh];
        }
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < 4; ++t) {
            const device uint* tt = tp + t * TW;
            LOAD_WORDS
            DECODE_MMA
        }
        tp += tstride;
        xk += 16;
    }

    // ---- k-splits of this threadgroup -> 128-column partial (fixed order) ----
    threadgroup float red[WS][128][8];
    _Pragma("clang loop unroll(full)") for (uint r = 0; r < RB; ++r) {
        _Pragma("clang loop unroll(full)") for (uint t = 0; t < 4; ++t) for (uint h = 0; h < 2; ++h) {
            const uint col = half_ * 64 + t * 16 + h * 8 + fm;
            red[split][col][fn] = acc[t][h][r].thread_elements()[0];
            red[split][col][fn + 1] = acc[t][h][r].thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < 8 * 128; idx += 64 * WS) {
            const uint rr = idx >> 7, cc = idx & 127;
            if (8 * r + rr < MR) {
                float v = 0.0f;
                for (uint j = 0; j < WS; ++j) v += red[j][cc][rr];
                part[(ulong(kblock) * MR + 8 * r + rr) * N + blk * 128 + cc] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


@functools.lru_cache(maxsize=None)
def mma_gemv_source(K: int, cb: str = "mul1") -> str:
    nvw = ((7 * K + 15) >> 5) + 1                                 # stream words v_0 .. holding windows 0..7
    nu = nvw + 1                                                  # source words (funnel needs the next one)
    ld = [f"const uint u{0} = tt[i0];"]
    ld += [f"const uint u{i} = tt[min(uint(ir + {i}), ilast)];" for i in range(1, nu)]
    ld += [f"const uint v{i} = (u{i} << ob) | ((u{i + 1} >> 1) >> (31u - ob));" for i in range(nvw - 1)]
    ld.append(f"const uint v{nvw - 1} = (u{nvw - 1} << ob) | ((u{nvw} >> 1) >> (31u - ob));")
    mids = {}
    win = []
    for j in range(8):
        b = j * K
        i, s = b >> 5, b & 31
        if s + 16 <= 32:
            win.append(f"extract_bits(v{i}, {16 - s}u, 16u)")
        else:
            mids[i] = f"const uint mm{i} = (v{i} << 16) | (v{i + 1} >> 16);"
            win.append(f"extract_bits(mm{i}, {32 - s}u, 16u)")
    ld += list(mids.values())
    dec = []
    for nh in range(2):
        for kh in range(2):
            ds = [f"{'' if e else 'half '}d{e} = dec_{cb}({win[4 * nh + 2 * kh + e]});" for e in range(2)]
            dec.append("{ simdgroup_matrix<float, 8, 8> A;\n"
                       f"  half d0 = dec_{cb}({win[4 * nh + 2 * kh]}), d1 = dec_{cb}({win[4 * nh + 2 * kh + 1]});\n"
                       "  A.thread_elements()[0] = float(d0); A.thread_elements()[1] = float(d1);\n"
                       f"  _Pragma(\"clang loop unroll(full)\") for (uint r = 0; r < RB; ++r) simdgroup_multiply_accumulate(acc[t][{nh}][r], A, B[{kh}][r], acc[t][{nh}][r]); }}")
    return _SRC.replace("LOAD_WORDS", "\n            ".join(ld)).replace("DECODE_MMA", "\n            ".join(dec))


def partial_mma(g: KN.Mul1Gemv, xh: mx.array, rows: int, ws: int = 4) -> mx.array:
    """GEMV partials of `rows` (<= 16) rows on simdgroup_matrix: part f32 [tgk, rows, n], the layout of Mul1Gemv.partial."""
    kern = KN._kern(("mma", g.K, g.cb), name=f"kyojin_{g.cb}_mma_k{g.K}", input_names=["xh", "trellis", "params"], output_names=["part"],
                    source=mma_gemv_source(g.K, g.cb))
    return kern(inputs=[xh, CB.bind(g.words, g.bview), g.params], template=[("BITS", g.K), ("MR", rows), ("WSP", ws)],
                grid=(g.blocks * 64 * ws, g.tgk * 4 // ws, 1), threadgroup=(64 * ws, 1, 1),
                output_shapes=[(g.tgk * 4 // ws, rows, g.n)], output_dtypes=[mx.float32])[0]


_fp: dict = {}


def finalize_rows(g: KN.Mul1Gemv, part: mx.array, rows: int) -> mx.array:
    """Finalize of `rows` rows: same kernel as Mul1Gemv.finalize, any row count."""
    grid, tg = S.had_grid(g.blocks, rows)
    tgk = part.shape[0]
    fparams = _fp.get((tgk, g.n, rows))
    if fparams is None:
        fparams = _fp[(tgk, g.n, rows)] = mx.array([tgk, g.n, rows], dtype=mx.uint32)
    return g.fin(inputs=[part, g.svh, fparams], template=[("OT", g.out_dtype)], grid=grid, threadgroup=tg,
                 output_shapes=[(rows, g.n)], output_dtypes=[g.out_dtype])[0]
