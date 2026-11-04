# SPDX-License-Identifier: Apache-2.0
"""int8 first pass for the decode index scores (lever KYOJIN_DEC_IDX_I8, default 0; KYOJIN_DEC_IDX_I8_MARGIN, default 128).

Why. One `dsv4_core.core` pass scores every row against its session's whole index-key buffer (`dsa_decode.scores_rows`, the `iscore_mma` branch):
2.5 L keys per row on DS4.1 at 1M (three ratio-2 layers, the ratio-1 candidate source) - 4.5 ms of the depth curve at n = 6 (T1 of
docs/reports/flatdec3_REPORT.md), and that kernel is at the key-byte roof, so the only lever is to read fewer bytes. Prefill already does this
(`pf_flat.I8` / `pf_flat.I8_SRC`): matmul2d on int8 operands runs at twice the half rate and reads half the bytes, and the index keys / queries sit
on the FP4 grid, so one power of two per key (per query head) maps a row to int8 without loss (`pf_flat.i8pack`, `pf_flat.i8_on_grid`). The same
first pass applies to the decode rows.

What. `select`: the int8 block-max kernel (`pf_flat.bmax_source` / `i8_source` / `i8_fast_source` with the decode row axis patched in, see
`bmax_source`) writes the maximum of every 8-key block of every row, over the keys below the row's own `lens`; the block ranking picks
`topk + MARGIN` blocks (the candidate source: `max(candidate_topk_blocks, topk) + MARGIN`, the last visible block pinned, ties to the lower block -
`pf_flat.pick_blocks`; `topk` blocks are what holds a row's top-k keys, the margin is the rounding shield); the picked keys are gathered per row out
of the session's buffer and rescored by the EXACT kernel the path it stands in for uses (`dsa_decode.scores_rows`, `nax=False`: the masked path's
own kernel, one key per lane, so a key's value does not depend on which keys share the tile); the top-k of those scores (`idxfast.topk_mt`, the same
select as `index_topk_rt`) mapped back through the position table is the answer, and a candidate-source layer rebuilds its kept block mask from the
exact block maxima of the picked ones (`pf_flat.rerank_blocks` + `pf_flat.mask_from_ids`). A consumer layer's candidate mask ranks blocks only
(`bm = -inf` on a block the mask dropped), so no unkept key can enter the answer.

Exactness. The rescored values are the full-width path's values bit for bit, so the top-k is exact whenever the true top-k blocks are among the
picked ones; the margin covers the int8 rounding of the (row, head) products that names them. `tests/test_dec_idx_i8.py` checks the selection against
`index_topk_rt` / `select_fused` on FP4-grid keys at margins 64 / 128 / 256. Two divergences, the same ones `decfilter` and `idxblocks` carry: a row
with fewer finite keys than `index_topk` gets a -1 tail here where the full-width select returns the (masked) filler positions, and the head-sum
rounding is the matrix kernel's (`iscore_mma` already carries it).

Memory. The int8 form of the keys (`Pack8`) is derived from the dense key buffer, never from a code view (an int8 code of an fp4 code is not a
view), and it is held per layer for the committed rows only: 128 B of codes + 4 B of power-of-two scale per key, i.e. 132 B per token per index-key
layer (2.5 L keys on DS4.1: ~330 MB at 1M). Rows of an open pass are repacked at every call (they may be rolled back), so an append costs O(new
rows) and a committed row is packed once. The fp16 key buffer is not copied and not held: `pack` reads the one the caller scored before.

Scope. fp16 keys / queries of 32 heads x 128 channels (`dsa_decode.rows_supported`), the runtime-width path (`rt`), the `iscore_mma` geometry and
`KYOJIN_DEC_IDX_I8_MIN` (262144) index keys; a candidate mask only with the 8-key block size it was built with. Everything else - another dtype,
another head shape, a mask of another block size or a narrower width, keys shorter than `t_max`, a key buffer whose last block would run past the
capacity, a row that cannot have `index_topk` finite keys (see above), no Metal, a device without matmul2d, keys off the int8 grid - keeps the
exact path: `layer` returns None and `dsv4_core.core` runs the chain it ran before. Nothing changes at flag 0.
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import dsa_decode as D
from . import idxfast as IX
from . import metal_nax as NX
from . import pf_flat as PF

ENABLED = os.environ.get("KYOJIN_DEC_IDX_I8", "0") not in ("0", "")        # flags.py dec_idx_i8 (switched at call time)
MARGIN = int(os.environ.get("KYOJIN_DEC_IDX_I8_MARGIN", "128"))           # blocks picked beyond the ones the answer needs
MIN_KEYS = int(os.environ.get("KYOJIN_DEC_IDX_I8_MIN", "262144"))         # index keys from which a layer takes the first pass

BS = PF.BS                                                               # keys per block (the kernels are written for 8)
NEG = mx.array(-mx.inf, dtype=mx.float32)
HITS = {"layer": 0, "source": 0, "masked": 0, "pack_rows": 0, "pack_open": 0}
_kern: dict = {}

# ----------------------------------------------------------------------------------------------------------------- first pass (int8 block maxima)
_CLEN = "    const uint clen = (start + t0 + t + 1) / ratio;\n"
_PROLOGUE = ("    const uint kcap = prm[6];"                                # keys per session: the stride of the key buffer and of its int8 scale
             "                                             // prm[6]\n"
             "    const ulong kof = ulong(sess[t]) * ulong(kcap);               // this row's key buffer and scales\n"
             "    const device int8_t* Kb = K + kof * 128;\n")
_FAST = [("a *= float4(E[n.x], E[n.y], E[n.z], E[n.w]);", "a *= float4(E[n.x + kof], E[n.y + kof], E[n.z + kof], E[n.w + kof]);"),
         ("a = select(float4(-INFINITY), a, n + n0 < clen);", "a = select(float4(-INFINITY), a, n < uint(lens[t]));")]


def bmax_source(m) -> str:
    """The `pf_flat` int8 block-max kernel with the decode row axis: the key buffer and its int8 scale are picked by `sess[t]` and the visibility cut
    is the row's own `lens[t]` instead of the prefill chunk's causal formula (`(start + t0 + t + 1) / ratio`). The arithmetic, the head order, the
    register post (`m` from `pf_flat.i8_tile_map`) and the block write are the prefill kernel's, line for line."""
    s = PF.i8_fast_source(m) if m else PF.i8_source()
    for old, new in [(_CLEN, _CLEN + _PROLOGUE),
                     ("(device int8_t*)(K + ulong(ks) * 128)", "(device int8_t*)(Kb + ulong(ks) * 128)"),
                     ("s = n + n0 < clen ? acc * E[n] : -INFINITY;", "s = n < uint(lens[t]) ? acc * E[n + kof] : -INFINITY;")] + (_FAST if m else []):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s


def bmax_rows(qi, w8, ki, ke, lens, sess, t_max: int, kcap: int, n: int):
    """[n, ceil(t_max / 8)] float32 block maxima of the int8 first pass. qi [n * 32, 128] int8, ki [S, kcap, 128] int8 with its scales
    ke [S, kcap] float32, lens / sess int32 [n] (row u reads key buffer sess[u], blocks with a key below lens[u])."""
    k = _kern.get("bm")
    if k is None:
        m = PF.i8_tile_map() if PF.I8_REG else None
        k = _kern["bm"] = mx.fast.metal_kernel(name="kyojin_decidx8_bmax" + ("r" if m else ""), header=NX.HEADER_NAX,
                                               input_names=["Q", "K", "w", "E", "lens", "sess", "prm"], output_names=["S"], source=bmax_source(m))
    tpt = max(PF.I8_TPT, 1)
    prm = mx.array([t_max, 0, 1, 0, 0, tpt, kcap] + [0] * 9, dtype=mx.uint32)
    return k(inputs=[qi, ki, w8, ke, lens.astype(mx.int32), sess, prm], grid=(n * 128, -(-t_max // (128 * tpt)), 1), threadgroup=(128, 1, 1),
             output_shapes=[(n, -(-t_max // BS))], output_dtypes=[mx.float32])[0]


class Pack8:
    """The int8 form of the index keys of every session of one layer: codes [S, cap, 128] int8 + one power-of-two scale [S, cap] float32
    (`pf_flat.i8pack`), 132 B per key. Filled for the committed rows once (`committed[s]`, the buffer count before the pass), the rows of an open
    pass at every call (a rejected row of a pass is overwritten by the next one), so an append costs O(new rows) and a rolled-back count repacks."""

    def __init__(self, S: int):
        self.ki, self.ke = None, None
        self.have = [0] * S

    def get(self, keys, committed, T: int):
        S, C, DI = keys.shape
        if self.ki is None or self.ki.shape[1] < T:
            cap = -(-T // 4096) * 4096
            ki = mx.zeros((S, cap, DI), dtype=mx.int8)
            ke = mx.zeros((S, cap), dtype=mx.float32)
            if self.ki is not None:
                ki[:, :self.ki.shape[1]] = self.ki
                ke[:, :self.ke.shape[1]] = self.ke
            self.ki, self.ke = ki, ke
        for s in range(S):
            nc = min(max(int(committed[s]), 0), T)
            if self.have[s] > nc:
                self.have[s] = 0
            if self.have[s] < nc:
                c, e = PF.i8pack(keys[s, self.have[s]:nc])
                self.ki[s, self.have[s]:nc] = c
                self.ke[s, self.have[s]:nc] = e
                HITS["pack_rows"] += nc - self.have[s]
                self.have[s] = nc
            if T > nc:                                                          # the open pass: the rows a commit may still reject
                c, e = PF.i8pack(keys[s, nc:T])
                self.ki[s, nc:T] = c
                self.ke[s, nc:T] = e
                HITS["pack_open"] += T - nc
        return self.ki, self.ke


def _compact(ids, nb: int):
    """Ascending block ids of a picked set [r, P] (-1 = none), compacted (at most P, -1 padding): the gathered columns are then the picked blocks in
    ascending block order with the empty slots at the end, which is what makes the real slots a prefix (`pf_flat.ids_from_mask`)."""
    r, P = ids.shape
    flag = mx.zeros((r, nb + 1), dtype=mx.bool_)
    flag = mx.put_along_axis(flag, mx.where(ids >= 0, ids, nb), mx.ones((r, P), dtype=mx.bool_), axis=1)[:, :nb]
    return PF.ids_from_mask(flag, P)


# ----------------------------------------------------------------------------------------------------------------- selection
def select(q, keys, w, lens, sess, t_max: int, topk: int, pack: Pack8, committed, cand=None, kb: int = 0):
    """(top-k [n, topk] int32 ascending with the -1 tail, mask [n, t_max] bool or None) of the index scores of the rows q [n, NH, DI] / w [n, NH]
    against keys [S, K, DI] (row u reads keys[sess[u]], keys below lens[u]; the candidate mask `cand` [n, >= t_max] bool of 8-key blocks drops whole
    blocks), through the int8 first pass. `kb` > 0 = the candidate source layer: the top-k AND its kept block mask."""
    n = q.shape[0]
    nb = -(-t_max // BS)
    cl = mx.minimum(lens.astype(mx.int32), mx.array(int(t_max), dtype=mx.int32))
    ki, ke = pack.get(keys, committed, t_max)
    qi, qe = PF.i8pack(q.reshape(-1, q.shape[-1]))
    w8 = w.astype(mx.float32) * qe.reshape(n, q.shape[1])
    bm = bmax_rows(qi, w8, ki, ke, lens, sess, t_max, ki.shape[1], n)           # the codes' own capacity is the per-session stride
    if cand is not None:                                                       # a consumer layer: only the blocks its source kept, and only the visible ones
        bm = mx.where(cand[:, ::BS][:, :nb] & (mx.arange(nb, dtype=mx.int32)[None, :] * BS < cl[:, None]), bm, NEG)
    pick = min(nb, (max(kb, topk) + MARGIN) if kb else (topk + MARGIN))         # the row's top-k keys live in its top-k blocks; the source also needs its `kb` kept ones
    ids = PF.pick_blocks(bm, cl, pick, kb > 0)
    fin = mx.take_along_axis(bm, mx.maximum(ids, 0), axis=1) > NEG              # a block with no visible key ranks in when the last one is pinned: drop it
    ids = _compact(mx.where((ids >= 0) & fin, ids, mx.array(-1, dtype=mx.int32)), nb)
    P = ids.shape[1]
    W = P * BS
    bok = ids >= 0
    gpos = (mx.where(bok, ids, 0)[:, :, None] * BS + mx.arange(BS, dtype=mx.int32)).reshape(n, W)
    real = mx.repeat(bok, BS, axis=1) & (gpos < cl[:, None])                    # ascending, real slots first: the tie rule of the select is the full-width one
    gk = keys[sess.astype(mx.int32)[:, None], gpos]                            # [n, W, DI]: the picked keys of every row, straight out of its session
    sc = D.scores_rows(q, w, gk, real.sum(axis=-1).astype(mx.int32), W, None, mx.arange(n, dtype=mx.int32), nax=False)
    sc = mx.where(real, sc, mx.array(-mx.inf, dtype=q.dtype))                  # the exact kernel of the path this stands in for, per key: same values
    nreal = real.sum(axis=-1).astype(mx.int32)
    ci = IX.topk_mt(sc, nreal, topk)
    idx = mx.where(ci >= 0, mx.take_along_axis(gpos, mx.maximum(ci, 0), axis=1), mx.array(-1, dtype=mx.int32)).astype(mx.int32)
    if not kb:
        HITS["masked" if cand is not None else "layer"] += 1
        return idx, None
    mask = PF.mask_from_ids(PF.rerank_blocks(sc, ids, cl, kb), nb, t_max) & (mx.arange(t_max, dtype=mx.int32)[None, :] < cl[:, None])
    HITS["source"] += 1
    return idx, mask


def applies(idxr, hand_keys, cand, q, w, lens, sess, t_max: int) -> bool:
    """Can this layer take the int8 first pass? Geometry the row kernel serves (fp16, 32 heads x 128 channels), the runtime width, enough keys,
    a mask of the block size the kernels are written for, and keys on the int8 grid (`pf_flat.i8_on_grid`, probed once per process)."""
    if not ENABLED or not mx.metal.is_available() or hand_keys is None:
        return False
    if q.ndim != 3 or hand_keys.ndim != 3 or w.ndim != 2:
        return False
    dt = q.dtype
    if dt != mx.float16 or hand_keys.dtype != dt:
        return False
    if not D.rows_supported(idxr.n_heads, idxr.index_head_dim, dt):
        return False
    n = q.shape[0]
    if w.shape != (n, idxr.n_heads) or tuple(lens.shape) != (n,) or tuple(sess.shape) != (n,):
        return False
    if hand_keys.shape[-1] != q.shape[-1] or hand_keys.shape[0] < 1:               # the key row the kernel strides is the query's row
        return False
    if not max(MIN_KEYS, 32) <= t_max <= hand_keys.shape[1]:                       # the kernel reads a 32-key tile at a time
        return False
    if -(-t_max // BS) * BS > hand_keys.shape[1]:                              # the gathered last block must stay inside the key buffer
        return False
    if not 1 <= int(getattr(idxr, "index_topk", 0) or 0) <= t_max:
        return False
    if cand is not None:
        if cand.ndim != 2 or cand.shape[0] != n or cand.shape[-1] < t_max or int(getattr(idxr, "candidate_block_size", 0) or 0) != BS:
            return False
    if getattr(idxr, "is_candidate_source", False):
        kb = int(getattr(idxr, "candidate_topk_blocks", 0) or 0)
        if cand is not None or not 2 <= kb <= t_max:
            return False
    return PF.i8_on_grid(hand_keys[:1])


def layer(st, l: int, idxr, hand_keys, cand, q, w, em, t_max: int, sess):
    """The int8 first pass for one index layer of `core`: (top-k [n, topk] int32 in rt form, mask [n, t_max] bool or None) or None (the exact path
    keeps the layer). st / l key the int8 codes of this layer's committed keys (`Pack8`, dropped with the stacked state)."""
    if not applies(idxr, hand_keys, cand, q, w, em.lens_arr, sess, t_max):
        return None
    S = hand_keys.shape[0]
    store = getattr(st, "i8k", None)
    if store is None:
        store = st.i8k = {}
    pk = store.get(l)
    if pk is None or len(pk.have) != S or pk.ki is not None and pk.ki.shape[0] != S:
        pk = store[l] = Pack8(S)
    committed = st.kcnt[l] if getattr(st, "kcnt", None) is not None else [0] * S
    src = bool(getattr(idxr, "is_candidate_source", False))
    return select(q, hand_keys, w, em.lens_arr, sess, t_max, int(idxr.index_topk), pk, committed, cand,
                  kb=int(idxr.candidate_topk_blocks) if src else 0)
