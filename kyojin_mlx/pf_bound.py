# SPDX-License-Identifier: Apache-2.0
"""Exact two-level bound for the prefill indexer of a plain (non-candidate) layer, past `MIN_KEYS` keys (cloud/cflatpf-2, KYOJIN_PF_IDX_BOUND, default 0).

Why this exists. Under the served CED path only the encoder layers that own an index (DS4.1: 2, 8, 14) score all 8192 rows of a chunk against the whole past
(Tc = p / ratio keys); the candidate source and its consumers score the 128 tail rows only. Their cost is 2 * NH * D * T * Tc FLOPs per layer, exact arithmetic
that grows linearly with the context and that no traffic lever (bmax, gscore, selfuse, cand_ids) removes: block maxima still need every score. The only exact way
to do less of it is to prove that most key blocks cannot hold a top-k key and never score them.

What it does, per layer per chunk (nothing is kept between chunks, so fork / restore / kv-disk can never serve a stale summary):

  1. summaries   one read of the keys: per superblock of SB keys the box (centre, half width), the mean centre and the radius around it, and the largest norm.
  2. bounds      per query row and superblock, in fp32 matmuls: x = q_h . k lies in [lo, hi] for every key k of the superblock, with the tighter of the box bound
                 (q . c +- sum |q_d| h_d) and the ball bound (q . m +- |q| r). The score sum_h w_h relu(x_h) is at most sum_h (w_h >= 0 ? w_h relu(hi_h) : w_h relu(lo_h)).
                 A margin sum_h |w_h| (2^-10 max|x_h| + 2^-11 |q_h| rho) + 2^-24 sum_h |w_h| covers the half rounding of q . k, the fp32 accumulations, the head sum and the rounding of the summaries.
  3. threshold   the 2 ceil(topk / SB) superblocks with the highest bound are scored exactly (pf_flat.gathered_scores: bitwise the fscore column values); tau = the
                 topk-th best of those scores is at most the true topk-th best score of the row.
  4. survivors   a superblock whose bound is below tau holds no key of the top-k (every key in it scores below tau <= the topk-th value, ties included), so only
                 superblocks with bound >= tau are rescored (the same gathered kernel, 8-key blocks, ascending positions) and the exact top-k of those columns is returned.
                 Ties go to the lower position (columns ascend, the argsort is stable), as in the stock sort(argsort(-s)).

Exactness: the returned set is the stock set, bit for bit, whenever the row sees at least topk keys (checked per chunk: (start + 1) // ratio >= topk, else the stock
path runs). A query block whose survivors exceed MAXPCT percent of the superblocks, or whose seed has fewer than topk finite scores, runs the stock block instead;
after two such blocks the rest of the chunk goes stock, and a layer that missed in two chunks in a row skips the bound for the next SKIP_CHUNKS chunks.

What it is NOT: a measured gain. The bound is data dependent (isotropic keys prune nothing: the stock path then runs and the cost is the summaries, the bound
matmuls and the seed), and nothing here ran on a GPU. tools/pf_bound_probe.py prints the survivor fraction and the time on the Mac before any lane time is spent.
Everything here is plain MLX ops except the rescoring, which is the pf_flat gathered kernel (numpy reference without Metal, so the CPU tests cover the whole logic). On the GPU it needs
KYOJIN_PF_IDX_GSCORE=1 (the gathered-copy rescoring moves more bytes than the score tile it saves) and is off while KYOJIN_PF_IDX_QMM is on (different score numerics)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD
from . import idx_topk as IT
from . import pf_flat as PF

ENABLED = os.environ.get("KYOJIN_PF_IDX_BOUND", "0") not in ("0", "")
MIN_KEYS = int(os.environ.get("KYOJIN_PF_IDX_BOUND_MIN", "131072"))        # keys (Tc) from which a plain layer takes the bound path
SB = int(os.environ.get("KYOJIN_PF_IDX_BOUND_SB", "64"))                   # keys per superblock (a multiple of 8)
MAXPCT = int(os.environ.get("KYOJIN_PF_IDX_BOUND_MAXPCT", "25"))           # most survivors (percent of the superblocks, worst row of a query block) before the stock block runs
SKIP_CHUNKS = 16                                                            # chunks a layer sits out after two missed chunks in a row
MARGIN_HALF = 2.0 ** -10        # half rounding of q . k (2^-11 relative to |x|), x2 for the rounding of the bound itself
MARGIN_ACC = 2.0 ** -11         # fp32 accumulation of 128 terms (2^-17), the box terms (x sqrt 128), the head sum, the summaries: all relative to |q| rho
MARGIN_ABS = 2.0 ** -24         # subnormal half products
MAX_ELEMS = 1 << 25                                                         # elements of one [rows * 32, nsb] fp32 temporary
HITS = {"chunks": 0, "blocks": 0, "fallback_blocks": 0, "skipped_chunks": 0, "survivor_sb": 0, "total_sb": 0}
_state: dict = {}                                                           # layer key -> [consecutive missed chunks, chunks left to skip]


def sb_size() -> int:
    return max(PF.BS, SB // PF.BS * PF.BS)


# ----------------------------------------------------------------------------------------------------------------- summaries
def summaries(keys, sb: int, group: int = 1 << 12):
    """keys [1, Tc, D] -> dict of fp32 arrays over the nsb = ceil(Tc / sb) superblocks: cb, hb (box centre and half width) [nsb, D], cm (mean) [nsb, D],
    rb (largest |k - cm|) [nsb], rho (largest |k|) [nsb]. A short last superblock repeats its last key (a duplicate changes no maximum, minimum or radius)."""
    Tc, D = keys.shape[1], keys.shape[2]
    nsb = -(-Tc // sb)
    full = Tc // sb
    parts = []

    def one(blk):                                                          # blk [g, sb, D] float32
        mx_, mn_ = blk.max(axis=1), blk.min(axis=1)
        cm = blk.mean(axis=1)
        rb = mx.sqrt(((blk - cm[:, None]) ** 2).sum(axis=-1).max(axis=1))
        rho = mx.sqrt((blk * blk).sum(axis=-1).max(axis=1))
        return ((mx_ + mn_) * 0.5, (mx_ - mn_) * 0.5, cm, rb, rho)

    for a in range(0, full, group):
        b = min(full, a + group)
        parts.append(one(keys[0, a * sb:b * sb].astype(mx.float32).reshape(b - a, sb, D)))
        mx.eval(*parts[-1])
    if nsb > full:
        tail = keys[0, full * sb:].astype(mx.float32)
        tail = mx.concatenate([tail, mx.broadcast_to(tail[-1:], (sb - tail.shape[0], D))], axis=0)[None]
        parts.append(one(tail))
    cat = [parts[0][i] if len(parts) == 1 else mx.concatenate([pp[i] for pp in parts], axis=0) for i in range(5)]
    cb, hb, cm, rb, rho = cat
    s = 1.0 + 2.0 ** -12                                                   # the radii and widths are fp32 reductions: a hair of slack, far below MARGIN
    return {"cb": cb, "hb": hb * s, "cm": cm, "rb": rb * s, "rho": rho * s, "nsb": nsb, "sb": sb}


# ----------------------------------------------------------------------------------------------------------------- bounds
def bounds(q, w, summ, start: int, ratio: int, t0: int, n: int):
    """float32 [n, nsb]: for rows t0 .. t0 + n of the chunk an upper bound (margin included) on the score of every key of every superblock, -inf for a superblock
    no key of which the row can see. Rows go in sub-batches so the [rows * 32, nsb] temporaries stay under MAX_ELEMS."""
    NH, D = q.shape[2], q.shape[3]
    nsb, sb = summ["nsb"], summ["sb"]
    cb, hb, cm, rb, rho = summ["cb"], summ["hb"], summ["cm"], summ["rb"], summ["rho"]
    r = max(1, min(n, MAX_ELEMS // (NH * nsb)))
    cols = mx.arange(nsb, dtype=mx.int32)[None, :] * sb
    outs = []
    for a in range(0, n, r):
        m = min(r, n - a)
        qf = q[0, t0 + a:t0 + a + m].astype(mx.float32).reshape(m * NH, D)
        nq = mx.sqrt((qf * qf).sum(axis=-1, keepdims=True))                # [m * NH, 1]
        qc, qa = qf @ cb.T, mx.abs(qf) @ hb.T
        qm = qf @ cm.T
        ball = nq * rb[None, :]
        hi = mx.minimum(qc + qa, qm + ball).reshape(m, NH, nsb)
        lo = mx.maximum(qc - qa, qm - ball).reshape(m, NH, nsb)
        wf = w[0, t0 + a:t0 + a + m].astype(mx.float32)[:, :, None]
        u = mx.where(wf >= 0, wf * mx.maximum(hi, 0.0), wf * mx.maximum(lo, 0.0)).sum(axis=1)       # [m, nsb]
        aw = mx.abs(wf)
        mag = (aw * mx.maximum(mx.abs(hi), mx.abs(lo))).sum(axis=1)         # sum_h |w_h| max |x_h| over the superblock: the half rounding of q . k is relative to it
        scale = (aw * nq.reshape(m, NH, 1)).sum(axis=1)                     # sum_h |w_h| |q_h|: the fp32 accumulations and the summaries are relative to |q| rho
        u = u + (MARGIN_HALF * mag + MARGIN_ACC * scale * rho[None, :] + MARGIN_ABS * aw.sum(axis=1))
        clen = ((start + t0 + a + mx.arange(m, dtype=mx.int32) + 1) // ratio)[:, None]
        u = mx.where(cols < clen, u, -mx.inf)
        mx.eval(u)
        outs.append(u)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


def blocks_of(sb_ids, nblk: int, sb: int):
    """8-key block ids [r, K * sb / 8] of superblock ids [r, K] (-1 = none, or a block past the last key)."""
    per = sb // PF.BS
    ids = sb_ids[..., None] * per + mx.arange(per, dtype=mx.int32)
    ids = mx.where((sb_ids[..., None] >= 0) & (ids < nblk), ids, -1)
    return ids.reshape(sb_ids.shape[0], -1)


# ----------------------------------------------------------------------------------------------------------------- one query block
def _block(q, keys, kb, w, summ, start, ratio, topk, t0, n):
    """Exact top-k [n, topk] int32 of rows t0 .. t0 + n through the bound, or None (survivors above the cap, or fewer than topk finite seed scores)."""
    nsb, sb = summ["nsb"], summ["sb"]
    nblk = kb.shape[0]
    u = bounds(q, w, summ, start, ratio, t0, n)
    m = min(nsb, 2 * -(-topk // sb))
    seed = mx.argsort(-u, axis=-1)[:, :m].astype(mx.int32)
    sc = PF.gathered_scores(q, keys, kb, w, blocks_of(seed, nblk, sb), start, ratio, t0)[:n]
    tau = mx.topk(sc, topk, axis=-1).min(axis=-1)                         # the topk-th best exact score among the seed keys
    flag = u >= tau[:, None]
    cnt = flag.sum(axis=-1)
    kmax_i, finite_i, total_i = mx.stack([mx.max(cnt), mx.all(tau > -mx.inf).astype(cnt.dtype), cnt.sum()]).tolist()       # one readback of three scalars
    HITS["blocks"] += 1
    HITS["total_sb"] += n * nsb
    HITS["survivor_sb"] += int(total_i)
    kmax_i, finite_i = int(kmax_i), bool(finite_i)
    if not finite_i or kmax_i * 100 > MAXPCT * nsb or kmax_i < 1:
        HITS["fallback_blocks"] += 1
        return None
    ids = PF.ids_from_mask(flag, kmax_i)
    return PF.select_from_blocks(q, keys, kb, w, blocks_of(ids, nblk, sb), start, ratio, t0, topk)


# ----------------------------------------------------------------------------------------------------------------- entry points (ds_pf_idx._impl_indexer)
def applies(q, keys, w, topk: int, start: int, ratio: int, plain: bool) -> bool:
    """The bound path may serve this layer: flag on, a plain layer, enough keys, every row sees at least `topk` keys (so the stock top-k has no -inf filler), the indexer
    shapes of the kernels, and on the GPU the fused rescoring kernel (`pf_idx_gscore`): without it the rescoring is the gathered copy, which at the survivor cap moves
    more bytes than the stock score tile it replaces. The qmm score path is not bitwise equal to the fscore numerics the bound and the rescoring use, so it excludes this path."""
    from . import ds_pf_idx as PI
    from .modeldesc import KERNEL as KG
    sb = sb_size()
    Tc = keys.shape[1]
    if not (ENABLED and plain and topk >= 1 and Tc >= max(MIN_KEYS, 8 * sb) and (start + 1) // ratio >= max(topk, (2 * -(-topk // sb) + 1) * sb)
            and q.shape[2] == KG.IDX_HEADS and q.shape[3] == KG.IDX_DIM and MAXPCT > 0 and not PI.qmm_on()):
        return False
    if PF.METAL_OPT:
        return bool(PF.GSCORE and "gs" not in PF._failed and PF.fscore_ok(q, keys, w))
    return True


def try_plain(q, keys, w, start: int, ratio: int, topk: int, seqlen: int, key=None, stock=None):
    """Top-k indices [seqlen, topk] int32 (ascending absolute key positions) of a plain indexer layer through the superblock bound; None when this chunk is
    left to the stock path (the layer sits out its skip window). `stock(t0, n)` -> [n, topk] is the block the fallback runs (default: the fscore chain)."""
    st = _state.setdefault(key, [0, 0])
    if st[1] > 0:
        st[1] -= 1
        HITS["skipped_chunks"] += 1
        return None
    if stock is None:
        from . import ds_pf_idx as PI

        def stock(t0, n):
            return PI.topk_block(PI.scores_range(q, keys, w, start, ratio, t0, n), topk)
    HITS["chunks"] += 1
    FD.hit("pf_idx_bound")
    sb = sb_size()
    summ = summaries(keys, sb)
    kb = PF.key_blocks(keys)
    missed = [0, 0]                                                        # fallback blocks, blocks

    def red(t0, tn):
        n = tn[1]
        got = None if (missed[0] >= 2 and missed[0] == missed[1]) else _block(q, keys, kb, w, summ, start, ratio, topk, t0, n)
        missed[1] += 1
        if got is None:
            missed[0] += 1
            return stock(t0, n)
        return got

    out = IT.scan(lambda t0, n: (t0, n), seqlen, IT.qblock(seqlen), lambda t0, tn: red(tn[0], tn))
    st[0] = st[0] + 1 if missed[0] * 2 > missed[1] else 0
    if st[0] >= 2:
        st[0], st[1] = 0, SKIP_CHUNKS
    return out


def survivor_fraction() -> float:
    return HITS["survivor_sb"] / HITS["total_sb"] if HITS["total_sb"] else 0.0
