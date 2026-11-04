# SPDX-License-Identifier: Apache-2.0
"""Prefill indexer: score one candidate block set per query block instead of every key block (KYOJIN_PF_IDX_CLUSTER, default 0, approximate by design).

Why. A plain indexer layer of a prompt chunk scores every row against the whole past: 2 x NH x D x T x Tc FLOPs and T x Tc x 256 B of key reads, both
linear in the keys. At 1M tokens the three plain layers of a DS4.1 chunk are ~1.8 s of a 6.8 s chunk and no traffic lever (bmax, gscore, cand_ids) removes
the score pass itself: bmax still computes every score, it only writes 1/8 of them. The exact bound (pf_bound) tries to prove that most key blocks cannot
hold a top-k key, and on trained keys it prunes too little for its own price. The remaining way under the arithmetic floor is to score fewer (row, key)
pairs, which cannot be exact for the rows it skips: this lever is therefore approximate, and it ships off by default with its recall counted
(pf_idx_cluster_recall) and asserted in tests/test_pf_cluster.py.

What it does, per query block of `idx_topk.qblock` rows (512 by default) of a plain layer:

  1. probes     `probes` rows spread over the block (the last row is always one of them: it sees every key the block can see) run the exact full-width
                selection (`scores_range` + `topk_block`, the same values as the layer would compute) and keep their `mult x topk` best keys.
  2. union      those keys, mapped to the 8-key blocks the rescoring kernel works in, give one candidate block set for the whole block (<= `cap` blocks;
                above it, or with fewer than `topk` keys, the block runs the stock path and `fallback` counts).
  3. rescoring  every row of the block is scored exactly on the union only (`pf_flat.gathered_scores`: the same bit-exact column values, the same
                relu / head order / causal mask) and its exact top-k inside the union is the answer (`pf_flat.select_from_blocks`, ties to the lower
                position as in the stock sort).

Cost of one block, against the stock block: `probes / rows` full-width rows plus `rows x |union|` pairs, i.e. probes/rows + |union|/Tc of the stock pass.
With the shipped defaults (16 probes, mult 4, a 512-row block, topk 512) the union is at most 16 x 2048 keys = 32K keys, so a 500K-key layer pays
about 9 % (3 % probes + 6 % union), and less when a probe's own top-k keys sit in few blocks. A union wider than `maxpct` percent of the keys is
not worth scoring at all: that block runs the stock pass, and after two such blocks the rest of the chunk runs stock without paying probes.
The candidate source and the consumer layers are NOT served by this path: they already score 128 rows (or only their kept blocks), so a shared set can
only make their universe bigger. Recall of the rows between the probes is what the flag trades; the probes themselves return their exact set.

Switch: KYOJIN_PF_IDX_CLUSTER=1 (flags.py pf_idx_cluster) plus the knobs below. `HITS` counts the calls, the union size, the fallbacks and the recall
sample (overlap with the exact full-width top-k of one row per query block, in % of topk; see pf_idx_cluster_recall)."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from . import fired as FD
from . import idx_topk as IT
from . import pf_flat as PF

ENABLED = os.environ.get("KYOJIN_PF_IDX_CLUSTER", "0") not in ("0", "")
MIN_KEYS = int(os.environ.get("KYOJIN_PF_IDX_CLUSTER_MIN", "65536"))       # keys (Tc) from which a plain layer takes this path
PROBES = int(os.environ.get("KYOJIN_PF_IDX_CLUSTER_PROBES", "16"))         # probe rows per query block
MULT = int(os.environ.get("KYOJIN_PF_IDX_CLUSTER_MULT", "4"))              # keys kept per probe = MULT x topk
CAP = int(os.environ.get("KYOJIN_PF_IDX_CLUSTER_CAP", "8192"))             # most 8-key blocks of one union
MAXPCT = int(os.environ.get("KYOJIN_PF_IDX_CLUSTER_MAXPCT", "25"))         # widest union (percent of the keys) still worth scoring: else the stock block
SKIP_BLOCKS = 2                                                            # fallback blocks after which the rest of the chunk runs stock (no probes)
RECALL = os.environ.get("KYOJIN_PF_IDX_CLUSTER_RECALL", "0") not in ("0", "")

HITS = {"calls": 0, "blocks": 0, "fallback": 0, "probe_keys": 0, "union_blocks": 0, "union_keys": 0, "recall_hit": 0, "recall_total": 0}


def applies(q, keys, w, Tc: int, topk: int, start: int, ratio: int, plain: bool) -> bool:
    """A plain indexer layer of this chunk may take the union path: flag on, enough keys for the union to be worth its probes, every row sees at least
    `topk` keys (so no row is answered by -inf filler picks), and the layer is neither the candidate source nor one of its consumers."""
    from .modeldesc import KERNEL as KG
    if not (ENABLED and plain and topk >= 1 and Tc >= max(MIN_KEYS, 8 * topk) and PROBES >= 1 and MULT >= 1 and 0 < MAXPCT <= 100 and CAP >= -(-topk // PF.BS)
            and (start + 1) // ratio >= topk and q.shape[2] == KG.IDX_HEADS and q.shape[3] == KG.IDX_DIM):
        return False
    return True if not PF.METAL_OPT else PF.fscore_ok(q, keys, w)


def probe_rows(n: int, p: int) -> list:
    """Probe rows of a block of `n` rows: `p` spread over it plus the last row (its causal width covers every key the block can see); a block no longer
    than `p` rows is probed in full, i.e. served exactly."""
    if p >= n:
        return list(range(n))
    return sorted({(i * n) // p for i in range(p)} | {n - 1})


def block_topk(q, keys, w, start: int, ratio: int, t0: int, n: int, topk: int):
    """[n, topk] int32: the layer's own full-width selection of the rows t0 .. t0 + n (the fallback of a block whose union does not fit, and the reference
    the recall sample is compared against). Metal: `scores_range`; without Metal, the numpy score mirror (tests)."""
    from . import ds_pf_idx as PI
    if PF.METAL_OPT:
        s = PI.scores_range(q, keys, w, start, ratio, t0, n)
    else:
        from . import pfroof2 as PR
        s = mx.array(PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(keys[0]), np.array(w[0, t0:t0 + n]), start, ratio, t0))
    return PI.topk_block(s, topk)


def scores_rows(q, keys, w, start: int, ratio: int, rows, m: int):
    """[len(rows), m] int32: the `m` best keys of each row of `rows` (ascending positions)."""
    outs = [block_topk(q, keys, w, start, ratio, int(t), 1, m) for t in rows]
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


def union(q, keys, w, start: int, ratio: int, rows, topk: int):
    """(block ids int32 [1, K] ascending, K) of the candidate set of the rows `rows`, or (None, blocks) when the set is empty, too small to hold `topk`
    keys, or wider than `cap` (the caller runs the stock block then). Row 0 is always a probe, so its own top-k keys are in the set and every row of the
    block keeps at least `topk` finite candidates."""
    Tc = keys.shape[1]
    nblk = -(-Tc // PF.BS)
    m = min(Tc, max(topk, MULT * topk))
    pos = scores_rows(q, keys, w, start, ratio, rows, m)
    HITS["probe_keys"] += int(pos.size)
    flag = mx.zeros((1, nblk), dtype=mx.bool_)
    blk = (pos // PF.BS).reshape(1, -1).astype(mx.uint32)
    flag = mx.put_along_axis(flag, blk, mx.ones(blk.shape, dtype=mx.bool_), axis=1)
    cnt = int(flag.sum())
    if cnt < 1 or cnt * PF.BS < topk or cnt > CAP or cnt * PF.BS * 100 > MAXPCT * Tc:      # too narrow, or too wide to beat the stock pass
        return None, cnt
    HITS["union_blocks"] += cnt
    HITS["union_keys"] += cnt * PF.BS
    return PF.ids_from_mask(flag, cnt)[:, :cnt], cnt


def _block(q, keys, kb, w, start: int, ratio: int, t0: int, n: int, topk: int):
    """Exact top-k [n, topk] of the rows t0 .. t0 + n through the union of the block's probes, or None (the caller runs the stock block)."""
    rows = probe_rows(n, PROBES)
    ids, cnt = union(q, keys, w, start, ratio, [t0 + r for r in rows], topk)
    if ids is None:
        return None
    HITS["blocks"] += 1
    out = PF.select_from_blocks(q, keys, kb, w, mx.broadcast_to(ids, (n, cnt)), start, ratio, t0, topk)
    if RECALL:
        _recall(q, keys, w, start, ratio, t0, n, topk, rows, out)
    return out


def _recall(q, keys, w, start, ratio, t0, n, topk, rows, out) -> None:
    """Score one non-probe row of the block at full width and count how many of its exact top-k keys the union path returned for it."""
    t = next((r for r in (n // 3, n // 3 + 1, n // 2, 1, n - 2) if 0 <= r < n and r not in rows), None)
    if t is None:
        return
    exact = np.array(scores_rows(q, keys, w, start, ratio, [t0 + t], topk)[0])
    got = np.array(out[t])
    HITS["recall_hit"] += int(np.isin(exact, got).sum())
    HITS["recall_total"] += int(exact.size)


def plain_topk(q, keys, w, start: int, ratio: int, topk: int, seqlen: int):
    """Top-k indices [seqlen, topk] int32 (ascending absolute key positions) of a plain indexer layer, one candidate block set per query block."""
    HITS["calls"] += 1
    FD.hit("pf_idx_cluster")
    kb = PF.key_blocks(keys)
    qblock = IT.qblock(seqlen) or IT.QBLOCK_DEFAULT       # the group the probes are shared over: the indexer's row block, never the whole chunk
    missed = [0]

    def red(t0, tn):
        n = tn[1]
        if missed[0] < SKIP_BLOCKS:                       # a chunk whose unions are too wide pays the probes twice, then runs stock
            got = _block(q, keys, kb, w, start, ratio, t0, n, topk)
            if got is not None:
                return got
            missed[0] += 1
        HITS["fallback"] += 1
        return block_topk(q, keys, w, start, ratio, t0, n, topk)

    return IT.scan(lambda t0, n: (t0, n), seqlen, qblock, lambda t0, tn: red(tn[0], tn))


def recall_pct() -> float:
    """Recall of the sampled non-probe rows, in percent of topk (0 when the sample is off or empty)."""
    tot = HITS["recall_total"]
    return 100.0 * HITS["recall_hit"] / tot if tot else 0.0
