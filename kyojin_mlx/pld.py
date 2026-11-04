# SPDX-License-Identifier: Apache-2.0
"""Prompt-lookup drafts: when the last committed tokens plus the round's first token occurred earlier in the
session's own committed tokens, the tokens that followed are a draft with no draft-head forward. Only committed
tokens are indexed; the verify pass decides every emitted token, so a lookup draft is just another point proposal
(greedy: compare, sampled: sample the row and compare, exactly like an MTP draft).

The idea and the gate are re-implemented from mlx-serve's src/engines/mlx/spec/mtp_lookup.zig (MIT, see
THIRD_PARTY_NOTICES.md), in numpy / Python for this engine. The index is a 4-gram chained hash table: one vectorised
numpy insert per prefill chunk, no Python loop per token, fixed-size arrays (flat at 1M context).
"""
from __future__ import annotations

import os

import numpy as np

ENABLED = os.environ.get("KYOJIN_SPEC_LOOKUP", "1") not in ("0", "")     # flags.py spec_lookup (engine reads the env per round)
NGRAM = 4
MIN_SUFFIX = 8            # shorter context matches land too few drafts to pay for their verify
MAX_DRAFT = 7             # an ordinary lookup's verify stays at <= 8 rows, inside the fused paths
STRONG_SUFFIX = 16        # a 16-token agreement already marks a quoted span
SUFFIX_CAP = 64           # agreement tokens compared backwards
CANDIDATES = 8            # earlier occurrences match_best compares
TOKENS_DEFAULT = 1 << 20  # positions held (4 MB tokens + 4 MB prev); head is 2 x that in int32

_MIX = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xC2B2AE3D27D4EB4F)
_M3 = np.uint64(0x165667B19E3779F9)
_M4 = np.uint64(0x85EBCA77C2B2AE63)
_M5 = np.uint64(0x27D4EB2F165667C5)
_M6 = np.uint64(0xD6E8FEB86659FD93)


def _hash4(a, b, c, d):
    """64-bit mix of a 4-gram (numpy arrays or scalars). Collisions only add a chain step: the caller checks the actual tokens."""
    with np.errstate(over="ignore"):
        x = (np.uint64(a) * _MIX) ^ (np.uint64(b) * _M2) ^ (np.uint64(c) * _M3) ^ (np.uint64(d) * _M4)
        x ^= x >> np.uint64(30)
        x *= _M5
        x ^= x >> np.uint64(27)
        x *= _M6
        x ^= x >> np.uint64(31)
    return x


def verify_cost(rows: int) -> float:
    """Relative verify cost over `rows` positions (fit of the fused MoE rows / GDN recurSeq paths; only the fallback unit)."""
    r = float(max(1, rows))
    return 11.4 + 2.6 * (r - 1.0) if rows <= 8 else 37.0 + 1.8 * (r - 9.0)


def mtp_fit_cost(width: int) -> float:
    """Relative cost of an MTP round at `width` drafts: verify(width + 1) + a draft-head term."""
    return verify_cost(width + 1) + 0.75 * float(max(0, width))


def ema_step(ema: float, accepted: int) -> float:
    """Accepted lookup drafts per lookup round, smoothed."""
    return 0.7 * ema + 0.3 * float(accepted)


def drift_step(ema: float) -> float:
    """Pull the lookup EMA back toward the ordinary cap on non-lookup rounds, so lookups get retried."""
    return 0.98 * ema + 0.02 * float(MAX_DRAFT)


class Match:
    __slots__ = ("draft", "suffix", "votes")

    def __init__(self, draft, suffix: int):
        self.draft = draft
        self.suffix = int(suffix)
        self.votes = 1

    def __repr__(self):
        return f"Match(draft={self.draft!r}, suffix={self.suffix})"


class Index:
    """Chained 4-gram index over a session's committed tokens. `head[hash]` = latest position p whose 4-gram is
    toks[p-4:p]; `prev[p]` = the previous position with the same hash. Positions are token indices (the token after
    the 4-gram). Fixed arrays: 4 + 4 bytes per position, 8 bytes per hash bucket (2 ^ ceil(log2(cap)) buckets)."""

    def __init__(self, cap: int | None = None):
        cap = int(TOKENS_DEFAULT if cap is None else cap)
        self.cap = max(NGRAM + 1, cap)
        size = 1
        while size < self.cap:
            size <<= 1
        self.head = np.full(size, -1, dtype=np.int32)
        self.mask = size - 1
        self.prev = np.empty(self.cap, dtype=np.int32)
        self.toks = np.empty(self.cap, dtype=np.int32)
        self.n = 0
        self.inserts = 0

    def __len__(self) -> int:
        return self.n

    def fork(self) -> "Index":
        """An own copy (a few ms where an insert of the same tokens takes 100 ms per 1M): the same bytes as an index that took these tokens itself."""
        o = Index.__new__(Index)
        o.cap, o.mask, o.n, o.inserts = self.cap, self.mask, self.n, self.inserts
        o.head = self.head.copy()
        o.prev, o.toks = np.empty(self.cap, dtype=np.int32), np.empty(self.cap, dtype=np.int32)
        o.prev[:self.n], o.toks[:self.n] = self.prev[:self.n], self.toks[:self.n]
        return o

    def extend(self, ids) -> int:
        """Index a batch of committed tokens; returns the positions inserted. Vectorised (no Python loop per token)."""
        ids = np.asarray(ids, dtype=np.int32).ravel()
        n0 = self.n
        if n0 >= self.cap or ids.size == 0:
            return 0
        ids = ids[: self.cap - n0]
        n1 = n0 + int(ids.size)
        self.toks[n0:n1] = ids
        self.n = n1
        start = max(NGRAM, n0)
        if n1 <= start:
            return 0
        p = np.arange(start, n1, dtype=np.int32)
        a, b, c, d = self.toks[p - 4], self.toks[p - 3], self.toks[p - 2], self.toks[p - 1]
        h = (_hash4(a, b, c, d) & np.uint64(self.mask)).astype(np.int64)
        self._insert(p, h)
        self.inserts += int(p.size)
        return int(p.size)

    def _insert(self, p: np.ndarray, slots: np.ndarray) -> None:
        """Insert ascending positions p at hash slots; groups sharing a slot are chained in position order."""
        order = np.lexsort((p, slots))
        ss, ps = slots[order], p[order]
        new = np.empty(order.size, dtype=bool)
        new[0] = True
        new[1:] = ss[1:] != ss[:-1]
        prv = np.empty(order.size, dtype=np.int32)
        prv[0] = self.head[ss[0]]
        if order.size > 1:
            prv[1:] = np.where(new[1:], self.head[ss[1:]], ps[:-1])
        last = np.empty(order.size, dtype=bool)
        last[-1] = True
        last[:-1] = ss[1:] != ss[:-1]
        self.head[ss[last]] = ps[last]
        self.prev[ps] = prv

    def _agreement(self, p: int, t1: int) -> int:
        """Tokens before p that agree with the committed tail (plus t1), capped."""
        suffix = 0
        a, b, n = int(p), self.n + 1, self.n
        toks = self.toks
        while suffix < SUFFIX_CAP and a > 0:
            a -= 1
            b -= 1
            want = t1 if b == n else int(toks[b])
            if int(toks[a]) != want:
                break
            suffix += 1
        return suffix

    def _latest(self, t1: int):
        """Latest position whose exact 4-gram is the committed tail plus t1, or -1."""
        if self.n < NGRAM - 1:
            return -1
        a, b, c = (int(x) for x in self.toks[self.n - (NGRAM - 1): self.n])
        p = int(self.head[int(_hash4(a, b, c, t1) & np.uint64(self.mask))])
        while p >= 0:
            if np.array_equal(self.toks[p - NGRAM: p], np.array([a, b, c, t1], dtype=np.int32)):
                return p
            p = int(self.prev[p])
        return -1

    def match(self, t1: int, max_draft: int) -> Match | None:
        if max_draft <= 0:
            return None
        p = self._latest(int(t1))
        if p < 0:
            return None
        draft = self.toks[p: min(p + int(max_draft), self.n)]
        return Match([int(x) for x in draft], self._agreement(p, int(t1)))

    def match_best(self, t1: int, max_draft: int, candidates: int = CANDIDATES) -> Match | None:
        """The occurrence that agrees furthest back wins; the latest among equals."""
        if max_draft <= 0 or self.n < NGRAM - 1:
            return None
        a, b, c = (int(x) for x in self.toks[self.n - (NGRAM - 1): self.n])
        p = int(self.head[int(_hash4(a, b, c, t1) & np.uint64(self.mask))])
        best = None
        seen = 0
        while p >= 0 and seen < candidates:
            if np.array_equal(self.toks[p - NGRAM: p], np.array([a, b, c, int(t1)], dtype=np.int32)):
                suffix = self._agreement(p, int(t1))
                if best is None or suffix > best.suffix:
                    draft = self.toks[p: min(p + int(max_draft), self.n)]
                    best = Match([int(x) for x in draft], suffix)
                if suffix >= SUFFIX_CAP:
                    break
                seen += 1
            p = int(self.prev[p])
        return best


    def match_vote(self, t1: int, max_draft: int, candidates: int = 16) -> Match | None:
        """Flag spec_lookup_vote. Like `match_best`, but the draft is built by a vote of several earlier occurrences: the candidates that agree
        with the tail within half of the best agreement (at least NGRAM) each vote for the token that followed them, weighted by their agreement;
        the draft follows the winner while it holds more than half of the weight of the candidates still alive, so it stops at a position the
        earlier occurrences disagree on (one verify row there is a coin flip) and a stale most-recent copy no longer decides a position its older
        siblings contradict. `votes` = candidates that backed the first token. With one candidate this is `match_best`."""
        if max_draft <= 0 or self.n < NGRAM - 1:
            return None
        a, b, c = (int(x) for x in self.toks[self.n - (NGRAM - 1): self.n])
        want = np.array([a, b, c, int(t1)], dtype=np.int32)
        p = int(self.head[int(_hash4(a, b, c, int(t1)) & np.uint64(self.mask))])
        found: list = []
        while p >= 0 and len(found) < int(candidates):
            if np.array_equal(self.toks[p - NGRAM: p], want):
                found.append((p, self._agreement(p, int(t1))))
            p = int(self.prev[p])
        if not found:
            return None
        best = max(sf for _, sf in found)
        floor = max(NGRAM, (best + 1) // 2)
        alive = [(pp, sf) for pp, sf in found if sf >= floor]                        # newest first (the chain order)
        draft: list = []
        votes = 0
        for j in range(int(max_draft)):
            tally: dict = {}
            for pp, sf in alive:
                if pp + j < self.n:
                    tok = int(self.toks[pp + j])
                    tally[tok] = tally.get(tok, 0.0) + float(sf)
            if not tally:
                break
            total = sum(tally.values())
            win = max(tally, key=tally.get)
            if tally[win] * 2.0 <= total:
                break
            if j == 0:
                votes = sum(1 for pp, _ in alive if pp < self.n and int(self.toks[pp]) == win)
            draft.append(win)
            alive = [(pp, sf) for pp, sf in alive if pp + j < self.n and int(self.toks[pp + j]) == win]
        if not draft:
            return None
        m = Match(draft, best)
        m.votes = votes
        return m


LONG_TIER = 14            # flag spec_lookup_long: the middle width between an ordinary draft and the row cap (rival: 14 on a strong match)
COST_WINDOW = 8           # lookup-round ms samples kept per draft count


def tier_caps(max_rows: int) -> list:
    """Draft widths a strong match may take under flag spec_lookup_long: MAX_DRAFT, LONG_TIER, the row cap less the pending row."""
    top = strong_cap(max_rows)
    return sorted({t for t in (MAX_DRAFT, LONG_TIER, top) if t <= top})


class LookupCosts:
    """Measured ms of solo lookup rounds per draft count (the first round of a count builds its verify graph and is dropped; a count is priced from its
    second sample, by the median). `estimate` interpolates / extrapolates the measured counts; with fewer than two it answers None and the caller keeps
    its own fit. `unlocked` widens the strong-match cap one tier at a time, only after the tier below was measured."""

    def __init__(self):
        self.ms: dict = {}
        self._warm: set = set()

    def observe(self, k: int, ms: float) -> None:
        k = int(k)
        if k not in self._warm:
            self._warm.add(k)
            return
        self.ms[k] = (self.ms.get(k, []) + [float(ms)])[-COST_WINDOW:]

    def median(self, k: int):
        v = self.ms.get(int(k), [])
        return sorted(v)[(len(v) - 1) // 2] if len(v) >= 2 else None

    def points(self) -> list:
        return sorted((k, self.median(k)) for k in self.ms if self.median(k) is not None)

    def estimate(self, k: int):
        pts = self.points()
        got = self.median(k)
        if got is not None:
            return got
        if len(pts) < 2:
            return None
        lo = [p for p in pts if p[0] < k]
        hi = [p for p in pts if p[0] > k]
        (k0, m0), (k1, m1) = (lo[-1], hi[0]) if lo and hi else (pts[-2], pts[-1]) if not hi else (pts[0], pts[1])
        return max(1e-3, m0 + max(0.0, (m1 - m0) / max(1, k1 - k0)) * (k - k0))

    def unlocked(self, max_rows: int) -> int:
        """Widest tier whose lower tier has a measured round of at least that many drafts."""
        tiers = tier_caps(max_rows)
        cap = tiers[0]
        for lo, hi in zip(tiers, tiers[1:]):
            if any(k >= lo for k, _ in self.points()):
                cap = hi
            else:
                break
        return cap


SUFFIX_EDGES = (NGRAM, 6, MIN_SUFFIX)       # flag spec_lookup_adapt: short-suffix buckets [4, 6), [6, 8); 8 and longer keep the ordinary gate
SUFFIX_PRIOR = (1.0, 1.8)                   # accepted drafts per lookup round a bucket is assumed to land before it is measured (optimistic enough to be tried)


class SuffixBuckets:
    """Per-session accepted-drafts-per-round EMA of lookup rounds whose agreement was shorter than MIN_SUFFIX. A short agreement is a gamble the
    fixed floor refused outright; here it is priced like any lookup (`gate(min_suffix=NGRAM, lookup_ema=expect(suffix))`), so it runs while its
    measured acceptance pays against the MTP round and stops when it does not. A bucket that lost drifts back toward its prior on MTP rounds (retry)."""

    def __init__(self):
        self.ema = list(SUFFIX_PRIOR)
        self.rounds = [0] * len(SUFFIX_PRIOR)

    @staticmethod
    def bucket(suffix: int) -> int:
        """Bucket of a short agreement, or -1 for one the ordinary gate owns (>= MIN_SUFFIX) or below the 4-gram."""
        for i in range(len(SUFFIX_EDGES) - 1):
            if SUFFIX_EDGES[i] <= int(suffix) < SUFFIX_EDGES[i + 1]:
                return i
        return -1

    def expect(self, suffix: int) -> float:
        b = self.bucket(suffix)
        return self.ema[b] if b >= 0 else 0.0

    def observe(self, suffix: int, accepted: int) -> None:
        b = self.bucket(suffix)
        if b >= 0:
            self.ema[b] = ema_step(self.ema[b], accepted)
            self.rounds[b] += 1

    def drift(self) -> None:
        for i, pr in enumerate(SUFFIX_PRIOR):
            self.ema[i] = 0.98 * self.ema[i] + 0.02 * pr


def strong_cap(max_rows: int) -> int:
    """Draft cap for a >= STRONG_SUFFIX agreement: the highest verify row count the backend supports, less the pending row."""
    return max(MAX_DRAFT, int(max_rows) - 1)


def gate(m: Match | None, remaining: int, lookup_ema: float, mtp_ema: float, mtp_width: int, streak: bool,
         lookup_ms: float | None = None, mtp_ms: float | None = None, max_rows: int = MAX_DRAFT + 1, tiers: list | None = None, min_suffix: int | None = None) -> int:
    """Draft length for a lookup match, or 0 to run the planned MTP round. Sizes the draft to what lookups have been
    landing (or the cap right after one landed every draft), then takes it only when its expected tokens per ms beat
    the MTP round's. Measured ms values are used where the engine has them; a side not yet measured is its fit in
    the other side's unit (as in the rival)."""
    if m is None or m.suffix < (MIN_SUFFIX if min_suffix is None else int(min_suffix)) or remaining <= 0:
        return 0
    cap = strong_cap(max_rows) if m.suffix >= STRONG_SUFFIX else MAX_DRAFT
    if tiers is not None and m.suffix >= STRONG_SUFFIX:                  # flag spec_lookup_long: the caller's unlocked widths, each priced; the best rate wins
        cap = min(cap, max(tiers))
    sized = cap if streak else int(np.ceil(max(float(lookup_ema), 0.0))) + 2
    k = min(cap, sized, len(m.draft), int(remaining))
    if k <= 0:
        return 0
    if tiers is not None and m.suffix >= STRONG_SUFFIX:
        ks = sorted({k, *(min(t, k) for t in tiers if t > 0)})
        best, best_rate = 0, 0.0
        for kk in ks:
            r = _lookup_rate(kk, lookup_ema, mtp_ema, mtp_width, streak, lookup_ms, mtp_ms)
            if r is not None and r[0] > r[1] and r[0] > best_rate:
                best, best_rate = kk, r[0]
        return int(best)
    lk_fit = verify_cost(k + 1)
    mt_fit = mtp_fit_cost(int(mtp_width))
    if callable(lookup_ms):
        lookup_ms = lookup_ms(k)
    if lookup_ms is not None and mtp_ms is not None:
        lk_cost, mt_cost = max(1e-6, float(lookup_ms)), max(1e-6, float(mtp_ms))
    elif mtp_ms is not None:
        unit = max(1e-6, float(mtp_ms)) / mt_fit
        lk_cost, mt_cost = lk_fit * unit, float(mtp_ms)
    elif lookup_ms is not None:
        unit = max(1e-6, float(lookup_ms)) / lk_fit
        lk_cost, mt_cost = float(lookup_ms), mt_fit * unit
    else:
        lk_cost, mt_cost = lk_fit, mt_fit
    lk_rate = ((float(k) if streak else min(float(lookup_ema), float(k))) + 1.0) / lk_cost
    mt_rate = (float(mtp_ema) + 1.0) / mt_cost
    return int(k) if lk_rate > mt_rate else 0


def _lookup_rate(k: int, lookup_ema: float, mtp_ema: float, mtp_width: int, streak: bool, lookup_ms, mtp_ms):
    """(lookup tokens per cost unit, MTP tokens per cost unit) for a draft of k, priced as in `gate`."""
    lk_fit = verify_cost(k + 1)
    mt_fit = mtp_fit_cost(int(mtp_width))
    if callable(lookup_ms):
        lookup_ms = lookup_ms(k)
    if lookup_ms is not None and mtp_ms is not None:
        lk_cost, mt_cost = max(1e-6, float(lookup_ms)), max(1e-6, float(mtp_ms))
    elif mtp_ms is not None:
        unit = max(1e-6, float(mtp_ms)) / mt_fit
        lk_cost, mt_cost = lk_fit * unit, float(mtp_ms)
    elif lookup_ms is not None:
        unit = max(1e-6, float(lookup_ms)) / lk_fit
        lk_cost, mt_cost = float(lookup_ms), mt_fit * unit
    else:
        lk_cost, mt_cost = lk_fit, mt_fit
    lk_rate = ((float(k) if streak else min(float(lookup_ema), float(k))) + 1.0) / lk_cost
    return lk_rate, (float(mtp_ema) + 1.0) / mt_cost
