# SPDX-License-Identifier: Apache-2.0
"""Packed-key prefill scan for the plain indexer layers (KYOJIN_PF_IDX_CPACK, default 0; track cpackscan).

Where the time goes (docs/cloud/rival/WORK_LEDGER.md, 01/11 08:00, one 8192-row chunk at 1M tokens): the 3 plain layers on 500K keys cost ~1.8 s. That is the exact
arithmetic of the scores, 2 x 32 x 128 x 8192 x Tc FLOPs per layer (pf_flat docstring), not key traffic: the fused score kernel runs one query row per threadgroup and is
FLOP bound, so reading 68 B/key of fp4 codes (KeySlots) instead of 256 B/key of fp16 cannot make the dense scan cheaper, and `mx.quantized_matmul` over the codes writes the
[rows * 32, Tc] half product (pf_idx_qmm, ledger nightq26 pending). The only exact way under that floor is to not score key blocks that provably hold no top-k key
(pf_bound). This module is that route, rebuilt on the packed storage:

  1. keys        the fp16 keys of the layer are decoded from the codes + scales of the KeySlots entry (fp4pack.unpack, bit-identical to the vendor rounding) in chunks into a
                 TRANSIENT array that lives for this layer call only. The depth-flat key mirror (256 B/key, kept for the whole session) is never built or grown by the prefill:
                 `PackedSlots.defer_dense` stops `append_rows` from reading the layer back dense, and an older mirror of the layer is dropped.
  2. summaries   per superblock of SB keys: the box (centre, half width), the mean and, per GROUP of dims, the radius around the mean, plus the largest key norm.
  3. bound       per row, head and superblock: x = q_h . k lies in [lo, hi] with, per group of dims g, the tighter of the box bound (q_g . c_g +- sum |q_d| h_d) and the ball bound
                 (q_g . m_g +- |q_g| r_g); the group bounds ADD. The sum of per-group minima is never above the sum of the box bounds, but the group radii can add up to
                 more than the whole-vector radius, so the result is also capped by the whole-vector ball bound: GROUP = 128 is exactly the pf_bound bound and smaller groups are
                 never looser (tests/test_cpackscan.py checks both). The score sum_h w_h relu(x_h) is at most sum_h (w_h >= 0 ? w_h relu(hi_h) : w_h
                 relu(lo_h)); pf_bound's margin covers the half rounding of q . k, the fp32 accumulations, the head sum and the rounding of the summaries.
  4. refinement  the 2 ceil(topk / SB) superblocks with the highest bound are scored exactly (pf_flat.gathered_scores); tau = their topk-th best score is at most the true
                 topk-th best score of the row. Only superblocks whose bound reaches tau are rescored (same gathered kernel, ascending positions), and the exact top-k of those
                 columns is returned: ties go to the lower position, as in the stock sort(argsort(-s)). Exactness argument and fallbacks are pf_bound's (a query block whose
                 survivors pass MAXPCT percent runs the stock chain; two such blocks in a chunk send the rest of the chunk stock; two missed chunks in a row skip SKIP_CHUNKS).

What it is NOT: a measured gain. The bound is data dependent (isotropic keys prune nothing, ledger gbound2: survivors 97.8 % on random keys; tools/pf_bound_probe.py: 99.8 % on
gauss / smooth / lowrank, 54-57 % on runs) and nothing here ran on a GPU. The first chunk of a layer that misses costs the decode, the summaries, the bound and the seed on top of the
stock chain. The candidate source is left to its own path: its output includes the candidate block set (top candidate_topk_blocks block maxima, last block forced), so a threshold at
the topk-th score would change what the consumers see. Decode (S = 1) is untouched and rebuilds the key mirror on its first dense read."""
from __future__ import annotations

import os
from contextlib import contextmanager

import mlx.core as mx

from . import fired as FD
from . import fp4pack
from . import idx_topk as IT
from . import pf_bound as PB
from . import pf_flat as PF

ENABLED = os.environ.get("KYOJIN_PF_IDX_CPACK", "0") not in ("0", "")
MIN_KEYS = int(os.environ.get("KYOJIN_PF_IDX_CPACK_MIN", "131072"))          # keys (Tc) from which a plain layer takes the packed path
GROUP = int(os.environ.get("KYOJIN_PF_IDX_CPACK_GROUP", "32"))               # dims per bound group
DECODE_ROWS = 1 << 16                                                        # rows decoded per step of the transient dense array (bounds the fp32 temporaries)
HITS = {"chunks": 0, "blocks": 0, "fallback_blocks": 0, "skipped_chunks": 0, "survivor_sb": 0, "total_sb": 0, "decoded_rows": 0, "mirror_dropped": 0, "missed_chunks": 0,
        "mirror_avoided": 0}
_state: dict = {}                                                            # (layer key) -> [consecutive missed chunks, chunks left to skip]


class _Miss(Exception):
    """The first query block of a chunk cannot use the bound: the whole chunk goes to the normal chain, nothing but that block's bound work was spent."""


def on() -> bool:
    return ENABLED


def group_size(d: int = 128) -> int:
    g = GROUP if GROUP in (16, 32, 64, 128) else 32
    return g if d % g == 0 else d


# ----------------------------------------------------------------------------------------------------------------- packed storage
def packed_entry(cache, layer: int):
    """(slots, entry) of the KeySlots layer when it is packed fp16 with the DS4.1 index key shape and one session, else None."""
    slots = getattr(cache, "keys", None)
    if not hasattr(slots, "raw"):
        return None
    from .fp4pool import _Pk
    try:
        e = slots.raw(layer)
    except IndexError:
        return None
    if not isinstance(e, _Pk) or e.codes.shape[0] != 1 or e.dtype != mx.float16 or e.codes.shape[2] * 2 != 128:
        return None
    return slots, e


def decode(slots, e, n: int, step: int = DECODE_ROWS):
    """[1, n, 128] fp16: the first n keys of the packed layer, decoded in steps of `step` rows (bit-identical to the dense read; no mirror is touched)."""
    parts = []
    for a in range(0, n, step):
        b = min(n, a + step)
        d = fp4pack.unpack(e.codes[:, a:b], e.scales[:, a:b], slots.BLOCK, slots.E4M3, e.dtype)
        mx.eval(d)
        parts.append(d)
    HITS["decoded_rows"] += n
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    mx.eval(out)
    return out


def drop_mirror(e) -> int:
    """Free the dense mirror of the layer (bytes freed). The packed codes stay the ground truth; a later dense read rebuilds it."""
    nb = 0 if e.mirror is None else int(e.mirror.nbytes)
    if e.mirror is not None or e.mn or e.memo is not None:
        e.mirror, e.mn, e.memo = None, 0, None
        HITS["mirror_dropped"] += 1
    return nb


@contextmanager
def deferred(cache, on_: bool):
    """Around an index-key publish: the append does not read the layer back dense (no mirror build / growth)."""
    slots = getattr(cache, "keys", None) if on_ else None
    if slots is None or not hasattr(slots, "defer_dense"):
        yield
        return
    slots.defer_dense = True
    HITS["mirror_avoided"] += 1
    try:
        yield
    finally:
        slots.defer_dense = False


def mirror_bytes(cache) -> int:
    """Bytes of the dense mirrors held by the key slots of a cache (the retained cost this flag removes)."""
    slots = getattr(cache, "keys", None)
    if not hasattr(slots, "raw"):
        return 0
    from .fp4pool import _Pk
    return sum(int(e.mirror.nbytes) for e in list.__iter__(slots) if isinstance(e, _Pk) and e.mirror is not None)


# ----------------------------------------------------------------------------------------------------------------- eligibility
def layer_ok(indexer, start: int, seqlen: int) -> bool:
    """Before the publish: the flag is on, this layer owns its keys and is a plain layer (neither candidate source nor consumer), and the chunk reaches the key floor."""
    if not ENABLED or not indexer.owns_k or indexer.is_candidate_source or indexer.uses_candidates:
        return False
    ratio = indexer.compress_ratio
    sb = PB.sb_size()
    topk = indexer.index_topk
    return bool(ratio >= 1 and (start + seqlen) // ratio >= max(MIN_KEYS, 8 * sb) and (start + 1) // ratio >= max(topk, (2 * -(-topk // sb) + 1) * sb))


class shape_of:
    """A stand-in for the decoded keys [1, Tc, 128] fp16, for `applies` before anything is decoded."""
    ndim, dtype = 3, mx.float16

    def __init__(self, tc: int):
        self.shape = (1, tc, 128)


def applies(q, keys, w, topk: int, start: int, ratio: int) -> bool:
    """The bound may serve this chunk (same conditions as pf_bound.applies, without its own flag): indexer shapes, enough keys, every row sees at least `topk` keys, the
    fused rescoring kernel on the GPU, and not the qmm score numerics."""
    from . import ds_pf_idx as PI
    from .modeldesc import KERNEL as KG
    sb = PB.sb_size()
    Tc = keys.shape[1]
    if not (ENABLED and topk >= 1 and Tc >= max(MIN_KEYS, 8 * sb) and (start + 1) // ratio >= max(topk, (2 * -(-topk // sb) + 1) * sb)
            and q.shape[2] == KG.IDX_HEADS and q.shape[3] == KG.IDX_DIM and PB.MAXPCT > 0 and not PI.qmm_on()):
        return False
    if PF.METAL_OPT:
        return bool(PF.GSCORE and "gs" not in PF._failed and PF.fscore_ok(q, keys, w))
    return True


# ----------------------------------------------------------------------------------------------------------------- summaries and bounds
def summaries(keys, sb: int, gsz: int, group: int = 1 << 12):
    """keys [1, Tc, D] -> fp32 summaries of the nsb = ceil(Tc / sb) superblocks: cb, hb (box centre, half width) [nsb, D], cm (mean) [nsb, D], rg (largest |k_g - cm_g| per
    group of `gsz` dims) [nsb, D / gsz], rho (largest |k|) [nsb]. A short last superblock repeats its last key."""
    Tc, D = keys.shape[1], keys.shape[2]
    ng = D // gsz
    nsb = -(-Tc // sb)
    full = Tc // sb
    parts = []

    def one(blk):                                                          # blk [g, sb, D] float32
        mx_, mn_ = blk.max(axis=1), blk.min(axis=1)
        cm = blk.mean(axis=1)
        dev = (blk - cm[:, None]) ** 2
        rg = mx.sqrt(dev.reshape(blk.shape[0], sb, ng, gsz).sum(axis=-1).max(axis=1))
        rb = mx.sqrt(dev.sum(axis=-1).max(axis=1))
        rho = mx.sqrt((blk * blk).sum(axis=-1).max(axis=1))
        return ((mx_ + mn_) * 0.5, (mx_ - mn_) * 0.5, cm, rg, rb, rho)

    for a in range(0, full, group):
        b = min(full, a + group)
        parts.append(one(keys[0, a * sb:b * sb].astype(mx.float32).reshape(b - a, sb, D)))
        mx.eval(*parts[-1])
    if nsb > full:
        tail = keys[0, full * sb:].astype(mx.float32)
        tail = mx.concatenate([tail, mx.broadcast_to(tail[-1:], (sb - tail.shape[0], D))], axis=0)[None]
        parts.append(one(tail))
    cat = [parts[0][i] if len(parts) == 1 else mx.concatenate([pp[i] for pp in parts], axis=0) for i in range(6)]
    cb, hb, cm, rg, rb, rho = cat
    s = 1.0 + 2.0 ** -12                                                   # the radii and widths are fp32 reductions: a hair of slack, far below the margin
    return {"cb": cb, "hb": hb * s, "cm": cm, "rg": rg * s, "rb": rb * s, "rho": rho * s, "nsb": nsb, "sb": sb, "gsz": gsz}


def bounds(q, w, summ, start: int, ratio: int, t0: int, n: int):
    """float32 [n, nsb]: for rows t0 .. t0 + n of the chunk an upper bound (margin included) on the score of every key of every superblock, -inf for a superblock no key of which
    the row can see. `pf_bound.bounds` with the box / ball minimum taken per group of dims."""
    NH, D = q.shape[2], q.shape[3]
    nsb, sb, gsz = summ["nsb"], summ["sb"], summ["gsz"]
    ng = D // gsz
    cb, hb, cm, rg, rb, rho = summ["cb"], summ["hb"], summ["cm"], summ["rg"], summ["rb"], summ["rho"]
    r = max(1, min(n, PB.MAX_ELEMS // (NH * nsb)))
    cols = mx.arange(nsb, dtype=mx.int32)[None, :] * sb
    outs = []
    for a in range(0, n, r):
        m = min(r, n - a)
        qf = q[0, t0 + a:t0 + a + m].astype(mx.float32).reshape(m * NH, D)
        nq = mx.sqrt((qf * qf).sum(axis=-1, keepdims=True))                # [m * NH, 1]
        hi = lo = qm = None
        for g in range(ng):
            sl = slice(g * gsz, (g + 1) * gsz)
            qg = qf[:, sl]
            ngq = mx.sqrt((qg * qg).sum(axis=-1, keepdims=True))
            qc, qa = qg @ cb[:, sl].T, mx.abs(qg) @ hb[:, sl].T
            qmg = qg @ cm[:, sl].T
            ball = ngq * rg[:, g][None, :]
            hg, lg = mx.minimum(qc + qa, qmg + ball), mx.maximum(qc - qa, qmg - ball)
            hi, lo, qm = (hg, lg, qmg) if hi is None else (hi + hg, lo + lg, qm + qmg)
        hi = mx.minimum(hi, qm + nq * rb[None, :]).reshape(m, NH, nsb)          # the whole-vector ball: the group radii can add up to more than it
        lo = mx.maximum(lo, qm - nq * rb[None, :]).reshape(m, NH, nsb)
        wf = w[0, t0 + a:t0 + a + m].astype(mx.float32)[:, :, None]
        u = mx.where(wf >= 0, wf * mx.maximum(hi, 0.0), wf * mx.maximum(lo, 0.0)).sum(axis=1)       # [m, nsb]
        aw = mx.abs(wf)
        mag = (aw * mx.maximum(mx.abs(hi), mx.abs(lo))).sum(axis=1)
        scale = (aw * nq.reshape(m, NH, 1)).sum(axis=1)
        u = u + (PB.MARGIN_HALF * mag + PB.MARGIN_ACC * scale * rho[None, :] + PB.MARGIN_ABS * aw.sum(axis=1))
        clen = ((start + t0 + a + mx.arange(m, dtype=mx.int32) + 1) // ratio)[:, None]
        u = mx.where(cols < clen, u, -mx.inf)
        mx.eval(u)
        outs.append(u)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


# ----------------------------------------------------------------------------------------------------------------- one query block
def _block(q, keys, kb, w, summ, start, ratio, topk, t0, n):
    """Exact top-k [n, topk] int32 of rows t0 .. t0 + n through the bound, or None (survivors above the cap, or fewer than topk finite seed scores)."""
    nsb, sb = summ["nsb"], summ["sb"]
    nblk = kb.shape[0]
    u = bounds(q, w, summ, start, ratio, t0, n)
    m = min(nsb, 2 * -(-topk // sb))
    seed = mx.argsort(-u, axis=-1)[:, :m].astype(mx.int32)
    sc = PF.gathered_scores(q, keys, kb, w, PB.blocks_of(seed, nblk, sb), start, ratio, t0)[:n]
    tau = mx.topk(sc, topk, axis=-1).min(axis=-1)                         # the topk-th best exact score among the seed keys
    flag = u >= tau[:, None]
    cnt = flag.sum(axis=-1)
    kmax_i, finite_i, total_i = mx.stack([mx.max(cnt), mx.all(tau > -mx.inf).astype(cnt.dtype), cnt.sum()]).tolist()
    HITS["blocks"] += 1
    HITS["total_sb"] += n * nsb
    HITS["survivor_sb"] += int(total_i)
    kmax_i, finite_i = int(kmax_i), bool(finite_i)
    if not finite_i or kmax_i * 100 > PB.MAXPCT * nsb or kmax_i < 1:
        HITS["fallback_blocks"] += 1
        return None
    ids = PF.ids_from_mask(flag, kmax_i)
    return PF.select_from_blocks(q, keys, kb, w, PB.blocks_of(ids, nblk, sb), start, ratio, t0, topk)


def try_plain(q, keys, w, start: int, ratio: int, topk: int, seqlen: int, key=None, stock=None):
    """Top-k indices [seqlen, topk] int32 (ascending absolute key positions) of a plain indexer layer through the grouped superblock bound, over `keys` [1, Tc, 128] (the
    transient decode); None when the chunk is left to the normal chain (the layer sits out its skip window, or the first query block misses)."""
    st = _state.setdefault(key, [0, 0])
    if st[1] > 0:
        st[1] -= 1
        HITS["skipped_chunks"] += 1
        return None
    if stock is None:
        from . import ds_pf_idx as PI

        def stock(t0, n):
            if not PF.METAL_OPT:                                           # no Metal (CPU tests): numpy scoring in the kernels' order
                from . import pfroof2 as PR
                import numpy as np
                return IT.topk(mx.array(PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(keys[0]), np.array(w[0, t0:t0 + n]), start, ratio, t0)), topk)
            return PI.topk_block(PI.scores_range(q, keys, w, start, ratio, t0, n), topk)
    HITS["chunks"] += 1
    FD.hit("pf_idx_cpack")
    sb = PB.sb_size()
    summ = summaries(keys, sb, group_size(keys.shape[2]))
    kb = PF.key_blocks(keys)
    missed = [0, 0]                                                        # fallback blocks, blocks

    def red(t0, tn):
        n = tn[1]
        got = None if (missed[0] >= 2 and missed[0] == missed[1]) else _block(q, keys, kb, w, summ, start, ratio, topk, t0, n)
        missed[1] += 1
        if got is None:
            if missed[1] == 1:
                raise _Miss()
            missed[0] += 1
            return stock(t0, n)
        return got

    try:
        out = IT.scan(lambda t0, n: (t0, n), seqlen, IT.qblock(seqlen), lambda t0, tn: red(tn[0], tn))
    except _Miss:
        HITS["missed_chunks"] += 1
        st[0] += 1
        if st[0] >= 2:
            st[0], st[1] = 0, PB.SKIP_CHUNKS
        return None
    st[0] = st[0] + 1 if missed[0] * 2 > missed[1] else 0
    if st[0] >= 2:
        st[0], st[1] = 0, PB.SKIP_CHUNKS
    return out


def survivor_fraction() -> float:
    return HITS["survivor_sb"] / HITS["total_sb"] if HITS["total_sb"] else 0.0
