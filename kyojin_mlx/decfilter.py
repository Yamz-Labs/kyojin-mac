# SPDX-License-Identifier: Apache-2.0
"""Certified two-level decode indexer (lever KYOJIN_DEC_FILTER, default 0): the exact top-k of the index scores without running the 4096-FMA chain over every key.

Why (docs/reports/cflatdec_REPORT.md, term T1): the decode scores of the plain index layers are the one term that grows with the context AND with the rows of a
verify pass (`dsv4_core._ISCORE`: one thread per (row, key), compute bound, 7x above the byte roof; 8192 FLOP per key and row). Nothing a score row computes can be
shared between steps (the query changes), so a cache of scores is useless; what can be cut is the number of keys that are scored exactly.

Level 1 (cheap, whole width): the scores of every key with ONE fp16 matmul per session, [R * heads, dim] x [dim, T], then relu, the head weight and the head sum
in fp32. These are the kernel's arithmetic (tile rounded to the dtype, relu, product rounded to the dtype, head sum) except the ORDER of the additions inside the
matmul and the head sum. For every key a rigorous bound `delta` on the distance between this approximation and the pre-rounding fp32 total of the exact kernel is
computed from quantities that are known (see `bound`), so the exact score of a key lies in [a - delta, a + delta] and the stored score is the monotone rounding
of it. The keys are cut in blocks of B; a block carries `A = max a` (to rank it) and `U = max (a + delta)` (to exclude it).
Level 2 (exact, bounded): the `M` best blocks by `A` are gathered (B contiguous keys each), scored with the EXISTING exact kernel (`index_scores`, bit-identical for a
key whatever its neighbours: the kernel is per key), and the exact top-k of that set is taken with the same select as the stock chain (`topk_mt`: value descending,
lower index first, -1 tail, ascending order, and the blocks are in ascending order so the compact order is the global order).
Certificate (per row, on the device): let `tau` be the k-th best exact score of the gathered set. Every key outside the gathered blocks has an exact stored score
<= round(U_rest) where U_rest is the largest U among the blocks left out. If round(U_rest) < tau (strictly: a tie could still beat the k-th by index) no left-out
key can enter the top-k, so the selection is exactly the stock one. Rows with a non-finite value, fewer than k valid keys while some block was left out, or a
failed certificate are NOT trusted: the caller (core) runs the stock chain for the layer. Correctness never depends on the ranking, only on the bound and on the
certificate; a failed certificate costs the time of the stock path plus the filter. HITS counts both outcomes.

What the certificate costs on the host: one boolean read per filtered layer (a sync). It is only taken from KYOJIN_DEC_FILTER_MIN index keys, where a layer is worth
milliseconds. A graph without any sync would need the fallback inside a kernel; that is not built here (nothing can be compiled or timed in the writing sandbox).

Extra memory: the key norms the bound needs, fp32 per key per index layer that owns keys, filled incrementally (committed rows only: O(new rows) per pass, the rows of
the open pass are recomputed on the fly). 4 B x (3 x L/2 + L) = 10 MiB at L = 1M, 1.25 MiB at 128K. Per call: the gathered keys (rows x M x B x 256 B) and the
[R * heads, TILE] fp16 matmul tile (R = 6, TILE = 131072: ~50 MB, with ~5 temporaries of the same size).

Scope: plain index layers (no candidate mask, not the candidate source), fp16 / bf16, any head count and dim. The candidate source (it must hand out the block mask of
ALL keys) and the consumers (they score only the kept blocks) keep the stock path. The solo plain step (`fuse_index`, MMA scores) is not wired.

Bound (per key t, head h; u = unit roundoff of the dtype: 2^-11 fp16, 2^-8 bf16; the kernel computes acc = fp32 FMA chain, tile = round(acc), rel = max(tile, 0),
pr = round(rel * w), total = sum_h pr in fp32, score = round(total); the approximation x = matmul tile, same roundings, other order):
  |acc - acc'| <= 2 * 128 * 2^-24 * S_h  with S_h = sum_d |q_hd k_td| <= |q_h| |k_t|                (two fp32 accumulations of n = dim terms)
  |tile - x| <= |acc - acc'| + ulp(x) * 2                                                          (two roundings, ulp(x) <= 2u |x| + tiny)
  |pr - pr'| <= |w| |tile - x| + 2 * ulp(pr') ; sum of the heads: the same sum plus the fp32 reordering 2 * heads * 2^-24 * sum |pr|
Collected with a 2x margin on every constant:  delta_t = C_K |k_t| sum_h |w_h| |q_h|  +  C_X (sum_h |w_h| |x_h| + sum_h |pr'_h|)  +  C_0
with C_K = 4e-5 (dim 128, scaled by dim / 128), C_X = 8 u, C_0 = 8 * 2^-24 * (sum_h |w_h| + heads) (the subnormal quantum of tile and product). tests/test_decfilter.py checks the bound against a numpy
emulation of the kernel on adversarial data (cancellation, huge and tiny scale, mixed-sign weights, quantized ties); tools/cflatdec_bench.py checks it again on the real
Metal matmul (the claim `matmul accumulates in fp32` is an assumption about the GPU library, printed as `bound slack` there, must stay < 1)."""
from __future__ import annotations

import math
import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_DEC_FILTER", "0") not in ("0", "")
MIN_KEYS = int(os.environ.get("KYOJIN_DEC_FILTER_MIN", "131072"))        # index-key width (rows' longest) from which a layer takes the filter
BUDGET = int(os.environ.get("KYOJIN_DEC_FILTER_BUDGET", "2"))             # blocks rescored exactly = BUDGET x index_topk
BLOCK = int(os.environ.get("KYOJIN_DEC_FILTER_BLOCK", "16"))              # keys per block (contiguous gather unit)
TILE = int(os.environ.get("KYOJIN_DEC_FILTER_TILE", "131072"))            # keys per matmul tile

HITS = {"ok": 0, "fallback": 0, "skip": 0, "norm_rows": 0}
LAST = {}                                                                 # last call: rows, kept blocks, margin (for the bench / the gate)

_EPS = {mx.float16: 2.0 ** -11, mx.bfloat16: 2.0 ** -8}


def eps_of(dt) -> float:
    return _EPS[dt]


# ----------------------------------------------------------------------------------------------------------------- incremental key norms
def row_norms(keys: mx.array) -> mx.array:
    """|k_t| as an fp32 upper bound (relative 1e-5 up): keys [..., dim] -> [...]"""
    k = keys.astype(mx.float32)
    return mx.sqrt((k * k).sum(axis=-1)) * (1.0 + 1e-5)


class Norms:
    """Key norms of the committed rows of every session of one layer: buf [S, cap] fp32, have[s] rows final. A committed row never changes (append-only). The rows past
    the committed count (an open pass) are written into the same buffer as scratch at every call and never counted in `have`, so a rejected row that the next pass
    overwrites is recomputed. Memory: 4 B per key; an in-place write may copy the buffer (4 MiB per layer at 1M), 1/64 of the key bytes."""

    def __init__(self, S: int):
        self.buf = None
        self.have = [0] * S

    def get(self, keys: mx.array, committed: list, T: int) -> mx.array:
        """[S, T] fp32 norms of keys[:, :T]; `committed[s]` = rows of session s that are final."""
        S = keys.shape[0]
        if self.buf is None or self.buf.shape[1] < T:
            cap = -(-T // 4096) * 4096 + (T >> 3)
            nb = mx.zeros((S, cap), dtype=mx.float32)
            if self.buf is not None:
                nb[:, :self.buf.shape[1]] = self.buf
            self.buf = nb
        for s in range(S):
            nc = min(int(committed[s]), T)
            if self.have[s] > nc:
                self.have[s] = 0
            if self.have[s] < nc:
                self.buf[s, self.have[s]:nc] = row_norms(keys[s, self.have[s]:nc])
                HITS["norm_rows"] += nc - self.have[s]
                self.have[s] = nc
            if T > nc:                                                         # the open pass: scratch rows, rewritten at every call (`have` never covers them)
                self.buf[s, nc:T] = row_norms(keys[s, nc:T])
        return self.buf[:, :T]


# ----------------------------------------------------------------------------------------------------------------- level 1
def approx_session(q: mx.array, K: mx.array, w: mx.array, knorm: mx.array):
    """q [R, NH, DI] (dtype dt), K [T, DI], w [R, NH], knorm [T] fp32 -> (a [R, T] fp32, delta [R, T] fp32)."""
    R, NH, DI = q.shape
    dt = q.dtype
    u = eps_of(dt)
    x = (q.reshape(R * NH, DI) @ K.T).reshape(R, NH, K.shape[0])             # tile rounded to dt (fp32 accumulation in the matmul)
    rel = mx.maximum(x, 0)
    pr = rel * w[:, :, None]                                                 # rounded to dt like the kernel's product
    a = pr.astype(mx.float32).sum(axis=1)
    wf = mx.abs(w).astype(mx.float32)
    ax = (mx.abs(x).astype(mx.float32) * wf[:, :, None]).sum(axis=1)
    ap = mx.abs(pr).astype(mx.float32).sum(axis=1)
    qn = mx.sqrt((q.astype(mx.float32) ** 2).sum(axis=-1))                   # [R, NH]
    wq = (wf * qn).sum(axis=-1)                                              # [R]
    ck = 4e-5 * (DI / 128.0)
    cx = 8.0 * u
    c0 = 8.0 * 2.0 ** -24 * (wf.sum(axis=-1) + NH)                           # [R]: the subnormal quantum of the tile and of the product, per head
    delta = ck * knorm[None, :] * wq[:, None] + cx * (ax + ap) + c0[:, None]
    return a, delta


def approx_all(q, keys, w, sess, lens, T, norms, tile=None):
    """a, delta [N, T] fp32 for every row (rows of one session are consecutive)."""
    N = q.shape[0]
    tile = tile or TILE
    a_rows, d_rows = [], []
    u = 0
    while u < N:
        v = u
        while v + 1 < N and sess[v + 1] == sess[u]:
            v += 1
        s = sess[u]
        parts_a, parts_d = [], []
        for t0 in range(0, T, tile):
            t1 = min(T, t0 + tile)
            a, d = approx_session(q[u:v + 1], keys[s, t0:t1], w[u:v + 1], norms[s, t0:t1])
            parts_a.append(a)
            parts_d.append(d)
        a_rows.append(parts_a[0] if len(parts_a) == 1 else mx.concatenate(parts_a, axis=1))
        d_rows.append(parts_d[0] if len(parts_d) == 1 else mx.concatenate(parts_d, axis=1))
        u = v + 1
    return (a_rows[0] if len(a_rows) == 1 else mx.concatenate(a_rows, axis=0)), (d_rows[0] if len(d_rows) == 1 else mx.concatenate(d_rows, axis=0))


# ----------------------------------------------------------------------------------------------------------------- the two levels
def select(q, keys, w, sess, lens, T, k, exact_fn, topk_fn, norms, block=None, budget=None, tile=None, kb=0, cap=None):
    """Exact top-k of the index scores of rows q [N, NH, DI] against keys [S, C, DI] (row u reads keys[sess[u]], first lens[u] keys), through the certified filter.
    sess: python list (non-decreasing), lens: python list of ints; T: keys looked at (multiple of `block`, >= max lens, <= C); norms [S, >= T] fp32.
    exact_fn(q [N, NH, DI], keys_g [N, W, DI], w [N, NH], lens [N] int32) -> [N, W] scores in q's dtype: the stock exact kernel on gathered keys.
    topk_fn(scores [N, W], lens [N] int32, k) -> int32 [N, k]: ascending indices of the k best columns t < lens, value descending / lower index first, -1 tail.
    kb > 0 = the candidate source: `block` must be its block size; the kb best blocks by exact block maximum (the block of the last valid key pinned first, ties to the lower
    block, a block kept only if its maximum is not -inf) come out as a bool mask [N, cap] (valid for t < lens), certified by the same left-out bound.
    Returns (idx [N, k] int32 ascending with the -1 tail, ok [N] bool) or, for the source, (idx, mask, ok): rows with ok False must be recomputed by the stock chain."""
    N, NH, DI = q.shape
    B = block or BLOCK
    dt = q.dtype
    assert T % B == 0 and T >= max(lens) and T <= keys.shape[1]
    nb = T // B
    Mb = min(nb, (budget or BUDGET) * max(k, kb))
    a, d = approx_all(q, keys, w, sess, lens, T, norms, tile)
    lens_a = mx.array(lens, dtype=mx.int32)
    pos = mx.arange(T, dtype=mx.int32)[None, :]
    valid = pos < lens_a[:, None]
    ninf = mx.array(-mx.inf, dtype=mx.float32)
    A = mx.where(valid, a, ninf).reshape(N, nb, B).max(axis=-1)               # block ranking value
    U = mx.where(valid, a + d, ninf).reshape(N, nb, B).max(axis=-1)           # block exclusion value
    nbv = (lens_a + (B - 1)) // B                                             # blocks with at least one valid key
    finite = mx.all(mx.where(mx.arange(nb, dtype=mx.int32)[None, :] < nbv[:, None], mx.isfinite(A) & mx.isfinite(U), True), axis=-1)
    lastb = (mx.maximum(lens_a, 1) - 1) // B                                  # block of the last valid key
    if kb:
        A = mx.where(mx.arange(nb, dtype=mx.int32)[None, :] == lastb[:, None], mx.array(mx.inf, dtype=mx.float32), A)    # pinned: always gathered
    blk = topk_fn(A, nbv, Mb)                                                 # [N, Mb] ascending block ids, -1 tail
    bok = blk >= 0
    bclamp = mx.where(bok, blk, 0)
    sess_a = mx.array(sess, dtype=mx.int32)
    gpos = (bclamp[:, :, None] * B + mx.arange(B, dtype=mx.int32)[None, None, :]).reshape(N, Mb * B)
    gk = keys[sess_a[:, None], gpos]                                          # [N, Mb * B, DI]: a row gather straight from the [S, C, DI] buffer (no reshape of a sliced buffer: that copies when S > 1)
    gvalid = mx.repeat(bok, B, axis=1) & (gpos < lens_a[:, None])
    s = exact_fn(q, gk, w, mx.full((N,), Mb * B, dtype=mx.int32))
    s = mx.where(gvalid, s, mx.array(-mx.inf, dtype=dt))
    ci = topk_fn(s, mx.full((N,), Mb * B, dtype=mx.int32), k)                 # compact ascending == global ascending
    civ = ci >= 0
    cc = mx.where(civ, ci, 0)
    got_valid = civ & mx.take_along_axis(gvalid, cc, axis=1)
    idx = mx.where(got_valid, mx.take_along_axis(gpos, cc, axis=1), -1).astype(mx.int32)
    sk = mx.take_along_axis(s, cc, axis=1)
    nvalid = got_valid.sum(axis=-1)
    full = nvalid >= k
    tau = mx.where(got_valid, sk, mx.array(mx.inf, dtype=dt)).min(axis=-1)    # k-th best exact score of the gathered set
    taken = mx.zeros((N, nb + 1), dtype=mx.bool_)
    taken[mx.arange(N, dtype=mx.int32)[:, None], mx.where(bok, blk, nb)] = True
    rest = mx.where(taken[:, :nb], ninf, U)
    u_rest = rest.max(axis=-1)
    nothing_left = u_rest == -mx.inf
    gap = u_rest.astype(dt) < tau                                              # strict: a tie at the k-th score could still win on the index
    fin_s = mx.all(mx.where(gvalid, mx.isfinite(s), True), axis=-1)           # a non-finite exact score anywhere in the gathered set: do not trust
    ok = finite & fin_s & ((full & (gap | nothing_left)) | (~full & nothing_left))
    LAST.update(blocks=Mb, nb=nb)
    if not kb:
        return idx, ok
    rows = mx.arange(N, dtype=mx.int32)[:, None]
    Xb = s.reshape(N, Mb, B).max(axis=-1)                                      # exact block maxima of the gathered blocks (-inf: pad or no valid key)
    pinned = bok & (blk == lastb[:, None])
    Xk = mx.where(pinned, mx.array(mx.inf, dtype=dt), Xb)
    sel = topk_fn(Xk, mx.full((N,), Mb, dtype=mx.int32), kb)                  # ascending positions in the gathered list == ascending global block ids
    sv = sel >= 0
    sc_ = mx.where(sv, sel, 0)
    xs = mx.take_along_axis(Xb, sc_, axis=1)
    ps = mx.take_along_axis(pinned, sc_, axis=1)
    kept = sv & (ps | (xs > mx.array(-mx.inf, dtype=dt)))
    taub = mx.where(kept & ~ps, xs, mx.array(mx.inf, dtype=dt)).min(axis=-1)  # k-th best non-pinned exact block maximum
    fullb = kept.sum(axis=-1) >= kb
    gapb = u_rest.astype(dt) < taub
    ok = finite & fin_s & ((full & fullb & gap & gapb) | nothing_left)       # both the k-th key and the weakest kept block must beat every left-out block
    gb = mx.take_along_axis(mx.where(bok, blk, 0), sc_, axis=1)
    ncap = (cap or keys.shape[1]) // B
    m = mx.zeros((N, ncap + 1), dtype=mx.bool_)
    m[rows, mx.where(kept, gb, ncap)] = True
    mask = mx.repeat(m[:, :ncap], B, axis=1)
    return idx, mask, ok


# ----------------------------------------------------------------------------------------------------------------- defaults for the Metal path and for the host
def default_exact(q, gk, w, lens):
    from . import dsv4_core as DC
    return DC.index_scores(q, gk, w, lens, gk.shape[1], None, None, rt=False)


def default_topk(scores, lens, k):
    from . import idxfast as IX
    return IX.topk_mt(scores, lens, k)


def can_sync() -> bool:
    """False where the certificate read is not allowed: a gated Engram step is being built (one command buffer per step; a sync inside makes the gate give up for good,
    `serve/backends.py` launch_gated) or the verify-replay / explicit-state mode (recorded command buffers). `dsv4_core.core` itself runs eagerly between the compiled
    regions of fuse_layer (A and B are compiled, the core is not), so no trace is open here."""
    from . import dsv4_core as DC, engram_mlx as EM
    return not EM.DEFER_DEPTH and not DC._CANON_POOLS


def applies(idxr, hand_keys, q, tmax: int, n: int) -> bool:
    if not ENABLED or hand_keys is None or idxr.uses_candidates or not mx.metal.is_available() or not can_sync():
        return False
    src = idxr.is_candidate_source
    B = idxr.candidate_block_size if src else BLOCK
    need = max(idxr.index_topk, idxr.candidate_topk_blocks if src else 0)
    return (tmax >= MIN_KEYS and q.dtype in _EPS and hand_keys.dtype == q.dtype and idxr.index_head_dim % 8 == 0 and tmax > BUDGET * need * B
            and hand_keys.shape[1] >= tmax and (not src or idxr.candidate_topk_blocks >= 2))


def layer(st, l: int, idxr, q, keys, w, plan, em, t_max: int):
    """The filter for one index layer of `core` without a candidate mask: q [n, NH, DI], keys [S, C, DI], w [n, NH] -> (idx [n, k] int32 in rt form, mask or None) or None
    (stock path). A plain layer returns mask None; the candidate source returns its bool mask [n, t_max] (valid for t < lens). The caller keeps the stock chain for None."""
    src = idxr.is_candidate_source
    B = idxr.candidate_block_size if src else BLOCK
    T = min(keys.shape[1], -(-em.tmax // B) * B)
    if T < MIN_KEYS or T < em.tmax:
        HITS["skip"] += 1
        return None
    store = st.__dict__.setdefault("fnorm", {})
    nrm = store.get(l)
    if nrm is None or len(nrm.have) != keys.shape[0]:
        nrm = store[l] = Norms(keys.shape[0])
    norms = nrm.get(keys, st.kcnt[l], T)
    out = select(q, keys, w, plan.sess, em.lens, T, idxr.index_topk, default_exact, default_topk, norms, block=B,
                 kb=idxr.candidate_topk_blocks if src else 0, cap=t_max)
    ok = out[-1]
    if not bool(mx.all(ok).item()):                                           # the certificate read: one sync per filtered layer
        HITS["fallback"] += 1
        return None
    HITS["ok"] += 1
    return (out[0], out[1]) if src else (out[0], None)
