# SPDX-License-Identifier: Apache-2.0
"""Copy drafts, v2 (flag copy_v2): the request's own tokens (prompt + output) are the draft source when the committed tail plus the round's
first token matches an earlier 4-gram. Pure functions on token lists / the `pld.Index`, no model.

What differs from the lookup gate in pld.py:
  * the agreement count (tokens that agree backwards from the match, capped) is the confidence; a copy needs MIN_AGREE of them (rival: 16);
  * the draft length comes from that confidence: P(the copied span lasts i more tokens | it lasted a so far) = a / (a + i)  (a span of unknown
    length has a heavy tail; the form is scale free, so no text or prompt constant), the length maximises expected tokens per priced ms;
  * a source with no more than one draft block left is not a copy (the draft head does that job at its own price): same-round fallback to block;
  * a copy round that lands less than half of its draft halves the session's confidence scale (a spent match), a full landing restores it,
    so the round after a spent match drafts short or hands over to the block route instead of paying a wide verify for one token.
The route of a round is copy, then block, then plain (the engine owns the last two)."""
from __future__ import annotations

import os

from . import pld as PLD

MIN_AGREE = 16            # tokens of agreement that make a match a copy
SCALE_FLOOR = 0.25        # lowest confidence scale after repeated spent matches
SRC_COPY, SRC_BLOCK, SRC_PLAIN = "copy", "block", "plain"


def min_agree() -> int:
    try:
        return max(PLD.NGRAM, int(os.environ.get("KYOJIN_COPY_V2_MIN", str(MIN_AGREE)) or MIN_AGREE))
    except ValueError:
        return MIN_AGREE


def max_len(row_cap: int) -> int:
    """Longest copy draft: flag copy_v2_max when set (> 0), else the verify row cap less the pending row."""
    try:
        v = int(os.environ.get("KYOJIN_COPY_V2_MAX", "0") or 0)
    except ValueError:
        v = 0
    top = max(1, int(row_cap) - 1)
    return min(top, v) if v > 0 else top


def survival(agree: float, i: int) -> float:
    """P(a copied span that agreed `agree` tokens goes on for `i` more) = a / (a + i)."""
    a = max(0.0, float(agree))
    return a / (a + float(i)) if a > 0 and i > 0 else (1.0 if i <= 0 else 0.0)


def expected_accept(agree: float, k: int) -> float:
    """Expected accepted drafts of a k-token copy draft."""
    return sum(survival(agree, i) for i in range(1, int(k) + 1))


def price(k: int, lookup_ms, mtp_ms, mtp_width: int) -> tuple:
    """(copy round cost, block-or-plain round cost) in one unit: measured ms when both exist, else the other side's fit scaled by the measured one
    (the unit rule of pld.gate)."""
    if callable(lookup_ms):
        lookup_ms = lookup_ms(k)
    lk_fit, mt_fit = PLD.verify_cost(k + 1), PLD.mtp_fit_cost(int(mtp_width))
    if lookup_ms is not None and mtp_ms is not None:
        return max(1e-6, float(lookup_ms)), max(1e-6, float(mtp_ms))
    if mtp_ms is not None:
        unit = max(1e-6, float(mtp_ms)) / mt_fit
        return lk_fit * unit, float(mtp_ms)
    if lookup_ms is not None:
        unit = max(1e-6, float(lookup_ms)) / lk_fit
        return float(lookup_ms), mt_fit * unit
    return lk_fit, mt_fit


def choose_len(agree: float, avail: int, kmax: int, lookup_ms=None, mtp_ms=None, mtp_width: int = 0, mtp_ema: float = 0.0) -> int:
    """Draft length 1..min(avail, kmax) with the best expected tokens per cost, or 0 when the block / plain round pays more per cost."""
    top = min(int(avail), int(kmax))
    if top <= 0:
        return 0
    alt_tok = (float(mtp_ema) + 1.0) if mtp_width > 0 else 1.0
    best, best_rate = 0, 0.0
    for k in range(1, top + 1):
        lk_cost, mt_cost = price(k, lookup_ms, mtp_ms, mtp_width)
        rate = (1.0 + expected_accept(agree, k)) / lk_cost
        if rate > alt_tok / mt_cost and rate > best_rate * (1.0 + 1e-9):
            best, best_rate = k, rate
    return best


class Plan:
    __slots__ = ("k", "draft", "agree", "eff")

    def __init__(self, k: int, draft: list, agree: int, eff: float):
        self.k, self.draft, self.agree, self.eff = int(k), list(draft), int(agree), float(eff)

    def __repr__(self):
        return f"Plan(k={self.k}, agree={self.agree}, eff={self.eff:.1f})"


def plan(index, t1: int, room: int, row_cap: int, block: int, scale: float = 1.0, lookup_ms=None, mtp_ms=None, mtp_width: int = 0,
         mtp_ema: float = 0.0, floor: int | None = None):
    """The copy draft of this round, or None (route on to block / plain). `index` = pld.Index over the committed tokens, `t1` the round's first
    token, `room` tokens the request may still emit, `block` the draft head's block (a source with <= block tokens left is the head's job)."""
    if room <= 0:
        return None
    m = index.match_best(int(t1), max_len(row_cap))
    if m is None:
        return None
    floor = min_agree() if floor is None else int(floor)
    eff = float(m.suffix) * float(scale)
    if m.suffix < floor or eff < floor * SCALE_FLOOR or len(m.draft) <= int(block):
        return None
    k = choose_len(eff, min(len(m.draft), int(room)), max_len(row_cap), lookup_ms, mtp_ms, mtp_width, mtp_ema)
    if k <= 0:
        return None
    return Plan(k, m.draft[:k], m.suffix, eff)


def update_scale(scale: float, drafted: int, accepted: int) -> float:
    """Confidence scale after a copy round: a full landing restores it, a miss of more than half of the draft halves it."""
    if drafted <= 0:
        return scale
    if accepted >= drafted:
        return 1.0
    if accepted * 2 < drafted:
        return max(SCALE_FLOOR, scale * 0.5)
    return scale
