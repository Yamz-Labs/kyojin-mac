# SPDX-License-Identifier: Apache-2.0
"""Reuse the decode index selection between rounds (lever KYOJIN_DEC_SEL_EVERY, default 0).

Why. An indexer layer of the batched core scores every pooled key of every row of the pass and selects the `index_topk` best
(`dsv4_core.core`, T1-T4 of docs/reports/flatdec3_REPORT.md). Successive decode queries of one session select nearly the same keys: the
last `index_topk` keys of the previous round are the base of this round's answer, and only the keys appended since then can be new. So
most rounds do not need the full scan.

What. One entry per (session row, layer), published at commit: the key count `pos` the selection was taken at, the selection itself
(int32 [index_topk], ascending, -1 tail, in the layer's own key index space) and `age` = rounds since the last full scan. On a round
whose `age + 1 < EVERY` and whose every session has an entry with `pos <= lens` (the row's key count: nothing the entry references was
rolled back), the layer scores a CANDIDATE SET instead of the whole width:

    candidates(u) = entry(session(u)).idx  ++  [ pos(session(u)), lens(u) )

ascending and duplicate-free by construction (the entry's keys are all < pos, the appended range starts at pos), so the tie rule of
the select ("value descending, lower index first") is the full scan's rule restricted to the set. The candidate positions are gathered
out of the row's own session buffer into one dense [n, W, DI] tile (the same gather `idxblocks` and `decidx8` use), the same score
kernel scores the tile, and the top-k of the tile (`idxfast.topk_mt`, bit-identical to `index_topk_rt`) mapped back through the
position table is the answer. A consumer layer (candidate mask) also drops the appended keys whose 8-key block the source layer did not
keep this round; the entries of the previous selection are accepted as they are (they were inside the mask when they were taken).

A full scan runs (a) on a layer with no usable entry, (b) every EVERY rounds, (c) always when a session has no entry - a new request, a
session that joined the batch, a rewind/checkpoint restore that moved the committed count below `pos`, or a session whose committed
keys are fewer than `index_topk` - and (d) at flag 0, which is today's path with no bookkeeping at all (EVERY = 1 keeps the
bookkeeping and never reuses: the same answer).

Why the counter is the commit's. The state a selection cache needs is per (layer, session row): `pos`, the selection of the row that is
KEPT, and the age. The cache is therefore written in `RowsCache.commit_layer`, from the pending selection of the row the commit keeps
(`Plan.accepted`), never from a row the commit drops: a rejected draft leaves no trace, so a rollback cannot publish a selection of
keys that never became committed state. The pending per-layer record (`pd.dec`) is dropped with the layer's pending state.

Approximate, by design. A key outside the candidate set is never scored, so the answer is the exact top-k WHENEVER the true top-k is
inside the set (then the answer is bit-identical to the full scan: same kernel, per key, same tie rule). `dec_sel_check` runs the full
scan on the reused rounds too and counts the overlap per layer (/v1/stats engage -> `decsel` -> `ov<layer>` / `slots<layer>`), which
is the recall of the lever on real text in one served run. Cost of the check: the full scan it measures plus one host read per reused
layer per round.

Scope. The candidate source layer is NOT covered (its answer is a full-width block mask the consumer layers read: a selection over a
candidate set cannot rebuild that mask), so it keeps its full scan; the seven other index layers (plain and masked consumers) take the
lever. fp16 / bf16 keys of the query's own dtype, the runtime-width path, dense keys (a packed key view is unpacked by the caller; the
lever does not build a dense buffer of its own), and `is_candidate_source` off. Off under the explicit-state / verify-replay mode
(`dsv4_core._CANON_POOLS`: a recorded command stream must not depend on the round), like `dec_filter` and `idx_short`.
"""
from __future__ import annotations

import os

import mlx.core as mx

EVERY = int(os.environ.get("KYOJIN_DEC_SEL_EVERY", "0"))                   # flags.py dec_sel_every: rounds between full scans (0 = today, 1 = a scan every round)
CHECK = os.environ.get("KYOJIN_DEC_SEL_CHECK", "0") not in ("0", "")      # flags.py dec_sel_check: also run the full scan, count the overlap

HITS = {"reuse": 0, "full": 0, "check": 0, "gate": 0, "prune": 0}
STATS: dict = {}                                                          # per layer: ov<l> / slots<l> / reuse<l> / full<l> (engage reads it)

_CAP = 2048                                                               # entries kept in a store before the rows of other passes are dropped


def active() -> bool:
    """The lever is armed: every layer reads the cache and publishes its selection at the commit. EVERY = 1 keeps the bookkeeping and
    never reuses (age + 1 < 1 is false), so the answer is the full scan's on every round - the identity the flag has to hold."""
    return EVERY >= 1


def _bump(d: dict, key: str, n: int = 1) -> None:
    d[key] = d.get(key, 0) + n


# ----------------------------------------------------------------------------------------------------------------------- the store
class Entry:
    """The selection of one (session row, layer): the key count it was taken at (`pos`), the selection itself (`idx` int32 [k],
    ascending, -1 tail) and the rounds since the last full scan (`age`). `row` keeps the session object alive so the id() key of the
    store stays unique."""

    __slots__ = ("pos", "idx", "age", "row")

    def __init__(self, pos: int, idx: mx.array, age: int, row):
        self.pos, self.idx, self.age, self.row = int(pos), idx, int(age), row


def store_of(st) -> dict:
    """The selection cache of a stacked batch (created on first use, dropped with the stacked state in `RowsCache.unstack`)."""
    d = getattr(st, "dec_sel", None)
    if d is None:
        d = st.dec_sel = {}
    return d


def _prune(store: dict, rows: list) -> None:
    """Bound the store: only the rows of the passes seen recently can hold an entry worth keeping (a session that left the batch has
    no committed key count any more that a later pass could reuse)."""
    if len(store) <= _CAP:
        return
    live = {id(r) for r in rows}
    for k in [k for k in store if k[0] not in live]:
        store.pop(k, None)
    _bump(HITS, "prune")


def publish(store: dict | None, l: int, rows: list, dec, last) -> None:
    """Write the cache of layer `l` from the pass's pending selection `dec` = (selection [n, k], lens per row, full scan?). `last[s]`
    is the row of session s that the commit kept (None: row s, a plain step). Called from `RowsCache.commit_layer`, so a rejected
    row is never published."""
    if store is None:
        return
    sel, lens, full = dec
    for s, row in enumerate(rows):
        r = s if last is None else last[s]
        key = (id(row), l)
        old = store.get(key)
        age = 0 if full else (old.age + 1 if old is not None else 0)
        store[key] = Entry(int(lens[r]), sel[r], age, row)
    _prune(store, rows)


# ----------------------------------------------------------------------------------------------------------------------- the gate
def applies(idxr, keys, q, w, t_max: int, k: int) -> bool:
    """Can this layer take the candidate path at all (geometry, width, the source layer excluded)? Independent of the cache: a layer
    that clears this gate still runs a full scan when no entry is usable."""
    from . import cdecpack as CD, dsv4_core as DC
    if not active() or not mx.metal.is_available() or DC._CANON_POOLS:
        return False
    if getattr(idxr, "is_candidate_source", False):
        return False
    if k <= 0 or t_max < 4 * k:                                                # below that the full width is already cheap (and the
        return False                                                           # candidate tile would be as wide as the scan)
    if keys is None or not isinstance(keys, mx.array) or q.ndim != 3 or keys.ndim != 3:
        return False
    n, nh, di = q.shape
    if keys.dtype != q.dtype or keys.shape[-1] != di or w.shape != (n, nh):
        return False
    if CD.packed_keys(keys):                                                   # a packed view: the gather would unpack the whole layer
        return False
    return di % 8 == 0                                          # the chain serves any such width, the matrix kernel 32 x 128 (see _scores)


# ----------------------------------------------------------------------------------------------------------------------- candidates
def build(cached: mx.array, pos: mx.array, lens: mx.array, t_max: int, cand: mx.array | None, maxnew: int):
    """The candidate positions of every row: [n, W] int32 ascending with the real entries first (a prefix, which is what the select's
    tie rule and `scores_rows`' cut need) and `t_max` in the slots past the row's count. `cached` [n, k] is the previous selection
    (-1 = none), `pos` [n] the key count it was taken at, `lens` [n] the row's current count, `cand` [n, t_max] the source layer's
    block mask or None (only the APPENDED keys are checked against it: the previous selection was inside the mask when it was taken)."""
    n, k = cached.shape
    sent = mx.array(int(t_max), dtype=mx.int32)
    a = mx.where(cached >= 0, cached, sent)
    nb = mx.arange(maxnew, dtype=mx.int32)[None, :] + pos[:, None]
    ok = nb < lens[:, None]
    if cand is not None:
        ok = ok & mx.take_along_axis(cand[:, :t_max], mx.minimum(nb, t_max - 1), axis=1)
    b = mx.where(ok, nb, sent)
    g = mx.sort(mx.concatenate([a, b], axis=1), axis=-1)
    real = g < lens[:, None]
    return mx.where(real, g, mx.zeros((1,), dtype=mx.int32)), real


def _scores(q, w, keys, lens, t_max: int, cand, sess, masked: bool):
    """The score path the stock chain would take for this layer: the simdgroup-matrix row kernel when `iscore_mma` would take it (with
    the mask form when the layer has a mask, so the values are the masked path's), else the thread-per-key chain. One key's value does
    not depend on the other keys of the tile, so a gathered tile holds the full-width values."""
    from . import dsv4_core as DC, dsa_decode as DSA
    if DC.ISCORE_MMA and t_max >= DC.ISCORE_MMA_MIN and keys.dtype == q.dtype and DSA.rows_supported(q.shape[1], q.shape[-1], q.dtype):
        return DSA.scores_rows(q, w, keys, lens, t_max, cand, sess, nax=(False if masked else None))
    return DC.index_scores(q, keys, w, lens, t_max, cand, sess, rt=True)


def full_select(q, w, keys, lens, sess, t_max: int, cand, k: int) -> mx.array:
    """The stock full-width selection of one layer (rt form), for `dec_sel_check` only: the reference the reused selection is
    compared with. It is the default chain (matrix kernel / chain + `index_topk_rt`), not the dec_filter / dec_idx_i8 / iscore_blocks
    paths, so with one of those armed the overlap of an exact candidate set reads slightly below 1."""
    from . import dsv4_core as DC
    s = _scores(q, w, keys, lens, t_max, cand, sess, cand is not None)
    return DC.index_topk_rt(s, lens, k)


def overlap(a: mx.array, b: mx.array, t_max: int) -> list:
    """Per row the number of columns where two rt-form selections hold the same key. Both are ascending with a -1 tail; the padding
    slots get values above every key index, unique inside one array and disjoint between the two, so they never count as a match;
    the merged sort then has one adjacent equal pair per common key."""
    ka, kb = a.shape[1], b.shape[1]
    ua = int(t_max) + 1 + mx.arange(ka, dtype=mx.int32)[None, :]
    ub = int(t_max) + 1 + ka + mx.arange(kb, dtype=mx.int32)[None, :]
    y = mx.sort(mx.concatenate([mx.where(a >= 0, a, ua), mx.where(b >= 0, b, ub)], axis=1), axis=-1)
    return [int(v) for v in (y[:, 1:] == y[:, :-1]).sum(axis=-1).tolist()]


# ----------------------------------------------------------------------------------------------------------------------- the layer
def layer(st, l: int, idxr, keys, cand, q, w, em, t_max: int, plan) -> mx.array | None:
    """The selection of one index layer of `core` through the candidate set: idx [n, index_topk] int32 (rt form, ascending, -1 tail)
    or None when the layer keeps the full scan (no gate, no usable entry, P reached, a session switched or rolled back). `q` [n, NH,
    DI], `w` [n, NH], `keys` [S, C, DI] dense, `cand` [n, t_max] bool or None, `em` the pass's emit record, `plan` the pass."""
    if not active():
        return None
    n = q.shape[0]
    k = int(getattr(idxr, "index_topk", 0) or 0)
    if not applies(idxr, keys, q, w, t_max, k) or (cand is not None and (cand.ndim != 2 or cand.shape[1] < t_max)):
        _bump(HITS, "gate")
        _bump(STATS, f"full{l}", n)
        return None
    S = len(plan.counts)
    store = store_of(st)
    rows, ent = st.rows, []
    for s in range(S):
        e = store.get((id(rows[s]), l))
        lo = min(em.lens[plan.first[s]:plan.first[s] + plan.counts[s]])       # the row that sees the fewest keys of this session
        if (e is None or e.age + 1 >= EVERY or e.pos > lo or e.pos < k or e.idx.shape[0] != k or e.idx.dtype != mx.int32):
            _bump(HITS, "full")
            _bump(STATS, f"full{l}", n)
            return None
        ent.append(e)
    rank = mx.array(plan.rank, dtype=mx.int32)                                # `ent` is per session of the pass; plan.sess is the buffer slot of `keys`
    cached = mx.stack([e.idx for e in ent])[rank]                             # [n, k]: every row's own session selection
    pos = mx.array([e.pos for e in ent], dtype=mx.int32)[rank]                # [n]
    lens = em.lens_arr
    maxnew = max(em.lens[u] - ent[plan.rank[u]].pos for u in range(n))
    gp, real = build(cached, pos, lens, t_max, cand, maxnew)
    W = gp.shape[1]
    nreal = real.sum(axis=-1).astype(mx.int32)
    gk = keys[plan.sess_idx[:, None], gp]                                     # [n, W, DI]: the candidates of every row
    s = _scores(q, w, gk, nreal, W, None, mx.arange(n, dtype=mx.int32), cand is not None)
    from . import idxfast as IX
    ci = IX.topk_mt(s, nreal, k)
    idx = mx.where(ci >= 0, mx.take_along_axis(gp, mx.maximum(ci, 0), axis=1), mx.array(-1, dtype=mx.int32)).astype(mx.int32)
    _bump(HITS, "reuse")
    _bump(STATS, f"reuse{l}", n)
    if CHECK:
        ref = full_select(q, w, keys, lens, plan.sess_idx, t_max, cand, k)
        ov = overlap(idx, ref, t_max)
        _bump(HITS, "check")
        _bump(STATS, f"ov{l}", sum(ov))
        _bump(STATS, f"slots{l}", n * k)
    return idx
