# SPDX-License-Identifier: Apache-2.0
"""Pure-numpy model of the chunk-tree top-k (kyojin_mlx.topk_tree): same plan, same bitwise threshold search, same tie rule, same candidate packing.

No mlx import: the unit tests run it anywhere, and the kernel's source is written line by line against these functions (the model is the spec,
`ref_topk` / `ref_mask` are the brute-force truth it is checked against).

    phase 1  every chunk of CS keys keeps its own K best (key desc, lower index first) in ascending index order, as 31-bit words key16 << 15 | local index
    phase 2  the chunk lists are concatenated in chunk order (so ascending global index, no holes) and the K best of that list are the answer
Exactness: the global top-K under the total order (key desc, index asc) is inside the union of the per-chunk top-K under the same order.
"""
from __future__ import annotations

import numpy as np

NT = 1024                 # threads of a threadgroup
MAXC = 32768              # candidate slots of a row the phase-2 threadgroup holds in registers (32 per thread)
MIN_CHUNK = 4096
MAX_CHUNK = 32768         # local index is 15 bits of the packed word
LOCAL_BITS = 15


def key16(bits: np.ndarray) -> np.ndarray:
    """Order-preserving 16-bit key of fp16 / bf16 bit patterns (+-0 tied) as uint32: the kernel's fz_key16."""
    b = bits.astype(np.uint32) & 0xFFFF
    b = np.where((b & 0x7FFF) == 0, 0, b)
    return np.where(b & 0x8000, (~b) & 0xFFFF, b | 0x8000).astype(np.uint32)


def f16_keys(scores: np.ndarray) -> np.ndarray:
    return key16(np.asarray(scores, dtype=np.float16).view(np.uint16))


def pow2_ceil(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def plan(Tm: int, k: int, bs: int = 8) -> tuple[int, int, int, int, int, int]:
    """(CS, NC, R1, R2, NCM, RB): chunk keys, chunks of this width, keys per thread in phase 1, candidate slots per thread in phase 2, chunk-count bound,
    block words per thread in the block level. Every kernel constant is a function of (k, CS, bs) only, never of Tm inside a CS class: the served width moves
    every token and a new constant is a new pipeline (~200 ms). Raises ValueError when the shape is outside the kernel (the caller keeps the radix select)."""
    if k < 1 or Tm < 1:
        raise ValueError("empty")
    cs = max(MIN_CHUNK, pow2_ceil(-(-Tm * k // MAXC)))
    if cs > MAX_CHUNK:
        raise ValueError(f"width {Tm} with k {k} needs chunks of {cs} > {MAX_CHUNK}")
    nc = -(-Tm // cs)
    ncm = pow2_ceil(-(-MAXC // k))
    assert nc <= -(-MAXC // k)
    r2 = -(-(-(-MAXC // k) * k) // NT)
    rb = -(-(-(-MAXC // k) * cs // bs) // NT)
    return cs, nc, cs // NT, r2, ncm, rb


def level(keys: np.ndarray, cnt: int, kd: int) -> np.ndarray:
    """One selection level (the kernel's idxfuse._LEVEL): bool per element, True = in the `kd` best of the first `cnt` (key desc, lower position first).
    A 16-step bitwise search for the kd-th key, then ties by contiguous ownership (lower position first)."""
    n = len(keys)
    valid = np.arange(n) < cnt
    if cnt <= kd:
        return valid
    thr = 0
    for b in range(15, -1, -1):
        try_ = thr | (1 << b)
        if int(np.count_nonzero(valid & (keys >= try_))) >= kd:
            thr = try_
    gt = int(np.count_nonzero(valid & (keys > thr)))
    need = kd - gt
    eq = valid & (keys == thr)
    take = eq & (np.cumsum(eq) <= need)             # the first `need` ties in position order
    return (valid & (keys > thr)) | take


def chunk_phase(keys_row: np.ndarray, L: int, cs: int, k: int) -> list[np.ndarray]:
    """Phase 1 of one row: per active chunk, the packed words of its k best, ascending local index."""
    out = []
    for c in range(max(1, -(-L // cs))):
        c0 = c * cs
        lc = max(0, min(L - c0, cs))
        kk = np.zeros(cs, dtype=np.uint32)
        kk[:lc] = keys_row[c0:c0 + lc]
        sel = level(kk, lc, k)
        idx = np.nonzero(sel)[0]
        out.append(((kk[idx] << LOCAL_BITS) | idx.astype(np.uint32)).astype(np.uint32))
    return out


def topk_row(keys_row: np.ndarray, L: int, k: int, cs: int) -> np.ndarray:
    """Phase 1 + 2 of one row: int32 [k] ascending global index, -1 tail."""
    chunks = chunk_phase(keys_row, L, cs, k)
    words = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint32)
    gidx = np.concatenate([ci * cs + (w & ((1 << LOCAL_BITS) - 1)).astype(np.int64) for ci, w in enumerate(chunks)])
    kk2 = (words >> LOCAL_BITS).astype(np.uint32)
    sel = level(kk2, len(kk2), k)
    got = gidx[sel].astype(np.int32)
    return np.concatenate([got, np.full(k - len(got), -1, dtype=np.int32)])


def topk(scores: np.ndarray, lens: np.ndarray, k: int) -> np.ndarray:
    """The kernel's contract: scores [N, Tm] fp16 (or bf16 bit patterns viewed as uint16 by `topk_bits`), lens [N] -> int32 [N, k]."""
    N, Tm = scores.shape
    cs = plan(Tm, k)[0]
    keys = f16_keys(scores)
    return np.stack([topk_row(keys[u], int(min(max(lens[u], 0), Tm)), k, cs) for u in range(N)])


def block_phase(scores_row: np.ndarray, L: int, bs: int) -> np.ndarray:
    """Phase 1 block words of one row: bit 16 = the block max is finite (> -inf), low 16 = key of the block max; one word per block of the row (ceil(L/bs))."""
    nel = -(-L // bs)
    out = np.zeros(nel, dtype=np.uint32)
    for b in range(nel):
        seg = np.asarray(scores_row[b * bs:min(L, (b + 1) * bs)], dtype=np.float16).astype(np.float32)
        m = np.float16(seg.max())
        out[b] = (0x10000 if float(m) > -np.inf else 0) | int(key16(np.array([m], dtype=np.float16).view(np.uint16))[0])
    return out


def mask_row(scores_row: np.ndarray, L: int, kb: int, bs: int) -> np.ndarray:
    """Candidate mask [L] bool of one row: the kb - 1 best blocks (first nel - 1 blocks, key desc, lower block first), finite only; the last block is pinned
    (kept when its max is finite)."""
    w = block_phase(scores_row, L, bs)
    nel = len(w)
    out = np.zeros(L, dtype=bool)
    if nel == 0:
        return out
    nd = nel - 1
    keys = (w[:nd] & 0xFFFF).astype(np.uint32)
    fin = (w[:nd] >> 16) != 0
    sel = level(keys, nd, kb - 1) if nd else np.zeros(0, dtype=bool)
    keep = sel & fin
    for b in range(nd):
        out[b * bs:(b + 1) * bs] = keep[b]
    out[nd * bs:L] = (w[nd] >> 16) != 0
    return out


# ----------------------------------------------------------------------------------------------------------------- brute force truth
def ref_topk_row(scores_row: np.ndarray, L: int, k: int) -> np.ndarray:
    keys = f16_keys(scores_row)[:L].astype(np.int64)
    order = np.lexsort((np.arange(L), -keys))[:k]
    got = np.sort(order).astype(np.int32)
    return np.concatenate([got, np.full(k - len(got), -1, dtype=np.int32)])


def ref_topk(scores: np.ndarray, lens: np.ndarray, k: int) -> np.ndarray:
    Tm = scores.shape[1]
    return np.stack([ref_topk_row(scores[u], int(min(max(lens[u], 0), Tm)), k) for u in range(scores.shape[0])])


def ref_mask_row(scores_row: np.ndarray, L: int, kb: int, bs: int) -> np.ndarray:
    nel = -(-L // bs)
    out = np.zeros(L, dtype=bool)
    if nel == 0:
        return out
    nd = nel - 1
    mx_ = np.array([np.asarray(scores_row[b * bs:min(L, (b + 1) * bs)], dtype=np.float16).max() for b in range(nel)], dtype=np.float16)
    keys = f16_keys(mx_[:nd]).astype(np.int64)
    fin = mx_[:nd].astype(np.float32) > -np.inf
    chosen = np.zeros(nd, dtype=bool)
    if nd:
        chosen[np.lexsort((np.arange(nd), -keys))[:kb - 1]] = True
    for b in range(nd):
        out[b * bs:(b + 1) * bs] = chosen[b] and fin[b]
    out[nd * bs:L] = float(mx_[nd]) > -np.inf
    return out
