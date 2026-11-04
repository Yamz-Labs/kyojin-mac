# SPDX-License-Identifier: Apache-2.0
"""Candidate-block exact top-k (lever D3, KYOJIN_DEC_TOPK_BLOCKS): the top-k of scores [N, Tm] without the 4 full reads of `idxfast.topk_mt`.

    1. block max   one read of the scores: m[b] = max of the valid scores of block b (BS = 8 keys), -inf past lens
    2. block pick  exact top-k of the block maxima (idxfast.topk_mt over Tm / 8 values, ties to the lower block)
    3. candidates  gather the k picked blocks (k * 8 scores, ascending positions)
    4. key pick    exact top-k of the candidates, mapped back to global positions

Exact, same set and order as `topk_mt` (value desc, lower index first on ties, -1 tail): rank the blocks by their best element (value, then lowest
index of that value). If a top-k element x lay in an unpicked block B, the k picked blocks would hold k elements ranked before B's best, hence
before x: contradiction. Ties between block maxima go to the lower block, which is the lower index of the best element, so the order is the same.
Invalid keys (t >= lens) sit in one contiguous suffix of the candidates (the pad of the last block, then the -1 blocks); the second pick gets that
count as its width, so they never win.

Cost: the scores are read once (plus k * 8 candidates) instead of one read per radix byte + count + write, and no launch scales with the context
except the block max. Applies when the row has clearly more blocks than picks (NB >= 4 * k) and N * Tm >= 2^20 (launch-bound below)."""
from __future__ import annotations

import mlx.core as mx

from . import fired as FD

BS = 8
MIN_RATIO = 4          # blocks per pick from which the candidate path runs
MAX_ROWS = 0           # >0: score tiles with more rows keep the radix select (flag dec_topk_blocks_rows)
MIN_ELEMS = 1 << 20    # N * Tm from which it pays (about 20 launches, ~0.5 ms flat, against a 4-read select that grows with the width)

_HDR = "using namespace metal;\n"
_BMAX = r"""
    const uint Tm = dims[0];
    const uint NB = dims[1];
    const uint gid = thread_position_in_grid.x;
    const uint u = gid / NB;
    const uint b = gid % NB;
    const uint L = min(uint(max(lens[u], 0)), Tm);
    const uint t0 = b * BS;
    float m = -INFINITY;
    if (t0 < L) {
        const device T* p = scores + ulong(u) * Tm + t0;
        const uint n = min(uint(BS), L - t0);
        for (uint i = 0; i < n; ++i) m = max(m, float(p[i]));
    }
    out[ulong(u) * NB + b] = T(m);
"""
_GATHER = r"""
    const uint Tm = dims[0];
    const uint KB = dims[1];
    const uint gid = thread_position_in_grid.x;
    const uint u = gid / (KB * BS);
    const uint j = gid % (KB * BS);
    const int b = bidx[ulong(u) * KB + j / BS];
    const uint L = min(uint(max(lens[u], 0)), Tm);
    const uint pos = uint(max(b, 0)) * BS + (j % BS);
    const bool ok = (b >= 0) && (pos < L);
    cand[gid] = ok ? scores[ulong(u) * Tm + pos] : T(-INFINITY);
    gpos[gid] = ok ? int(pos) : -1;
"""

_k: dict = {}


def _kern(name: str, inputs: list, outputs: list, source: str):
    k = _k.get(name)
    if k is None:
        k = _k[name] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source, header=_HDR)
    return k


def applies(N: int, Tm: int, k: int) -> bool:
    return k >= 1 and (-(-Tm // BS)) >= MIN_RATIO * k and N * Tm >= MIN_ELEMS and (not MAX_ROWS or N <= MAX_ROWS)


def block_max(scores: mx.array, lens: mx.array) -> mx.array:
    N, Tm = scores.shape
    nb = -(-Tm // BS)
    dims = mx.array([Tm, nb], dtype=mx.uint32)
    kern = _kern("kyojin_idxblk_bmax", ["scores", "lens", "dims"], ["out"], _BMAX)
    return kern(inputs=[scores, lens, dims], template=[("T", scores.dtype), ("BS", BS)], grid=(N * nb, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(N, nb)], output_dtypes=[scores.dtype])[0]


def topk_blocks(scores: mx.array, lens: mx.array, k: int) -> mx.array:
    """Same contract as idxfast.topk_mt (int32 [N, k], ascending, -1 tail); requires applies(Tm, k)."""
    from . import idxfast as IX
    N, Tm = scores.shape
    FD.hit("dec_topk_blocks")
    if MAX_ROWS:
        FD.hit("dec_topk_blocks_rows")
    lens = lens.astype(mx.int32)
    nb = -(-Tm // BS)
    bm = block_max(scores, lens)
    blens = (lens + (BS - 1)) // BS
    bidx = IX._run(bm, blens, k, 2, 8, False)[0]                     # [N, k] ascending block ids, -1 tail
    dims = mx.array([Tm, k], dtype=mx.uint32)
    kern = _kern("kyojin_idxblk_gather", ["scores", "bidx", "lens", "dims"], ["cand", "gpos"], _GATHER)
    cand, gpos = kern(inputs=[scores, bidx, lens, dims], template=[("T", scores.dtype), ("BS", BS)], grid=(N * k * BS, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(N, k * BS), (N, k * BS)], output_dtypes=[scores.dtype, mx.int32])
    clen = (gpos >= 0).sum(axis=1).astype(mx.int32)
    sel = IX._run(cand, clen, k, 2, 8, False)[0]                     # [N, k] positions in cand, ascending, -1 tail
    got = mx.take_along_axis(gpos, mx.maximum(sel, 0), axis=1)
    return mx.where(sel >= 0, got, -1).astype(mx.int32)
