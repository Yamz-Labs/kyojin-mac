# SPDX-License-Identifier: Apache-2.0
"""Acceptance-driven speculation depth (lever spec_depth_auto, env KYOJIN_SPEC_DEPTH_AUTO). Generic: any MTP backend that drafts a chain of k tokens.

A fixed depth is wrong in both directions. On predictable text (code continuation, edits, tool output, repeated context) the head's drafts land
~every time, so d1 leaves 1-3 tokens per round on the table; on prose a deep draft is a wasted row. This module keeps, per session, an EMA of the
conditional acceptance of every draft position (P(draft k lands | drafts < k landed), updated only for the positions a round reached) and picks the
depth that maximises expected tokens per ms:

    E[T](d) = 1 + p_1 + p_1 p_2 + ... + p_1 ... p_d        rate(d) = E[T](d) / cost(d)        d* = argmax over 0..cap  (d = 0: the plain step, E[T] = 1)

* Cost: the measured median ms of the warm solo rounds the engine ran at a depth (Engine._d_ms) where it has them; else the prior shape
  `PRIOR_COST` (ms of a round at depth d over the plain step, measured on GLM-5.3 with VROWS / POOLSEL / HEAD_F16 / LAG1: `docs`/STATUS_SPECDEEP.md) scaled
  by the nearest measured depth (or by the measured plain step). KYOJIN_SPEC_DEPTH_COST = "1,1.5,..." overrides the shape.
* A position never drafted has no estimate: it is priced at the last observed p x EXTRAP^(distance) (optimism that lets a streak of landings
  go deeper; one miss corrects it), and the depth moves UP by one per round at most (a new row count pays a cold graph once) and DOWN at once.
* Plain: the best drafted rate below PLAIN_MARGIN x the plain rate (0.9, the engine's RATE_MARGIN) runs the plain step; every PROBE rounds one
  d1 round re-measures (gap doubles after a failed probe, resets after a landing), so a text that turns predictable gets drafts back.
* Several sessions: the policy is solo only. A crowd keeps the engine's own rule, capped at depth 1 and at CROWD_ROWS verify rows (the rows the
  served d1 lane had), MULTI_PLAIN still sends it to the plain step.

The verify decides every token and the plan never samples: the depth only moves how many rows are verified. Pure Python, no mlx."""
from __future__ import annotations

import os

ENV = "KYOJIN_SPEC_DEPTH_AUTO"
COST_ENV = "KYOJIN_SPEC_DEPTH_COST"
ALPHA = 0.15                  # EMA rate of a reached position (memory ~ 6 rounds)
ALPHA_PROBE = 0.5             # a probe round after a plain spell: the old estimates are stale, so it moves them further
PRIOR_P = 0.7                 # first-position estimate of a session that has no round yet (the acceptance the d1 lane measures on mixed text)
EXTRAP = 0.9                  # per-position decay of an estimate beyond the deepest observed one
TIE = 0.02                    # a deeper depth must win by this much, else the shallower (cheaper) round stays
PLAIN_MARGIN = 0.9            # plain only when the best drafted rate is below 0.9 x plain's
PROBE = 16                    # plain rounds between two d1 probes
DRIFT = 0.1                   # per round, a position the round did not reach moves toward EXTRAP x the one before it (text predictability is shared; a stale low estimate must be re-tried)
PROBE_MAX = 128               # longest gap (doubles after a probe that did not land)
CROWD_ROWS = 4                # verify rows a crowd (2+ sessions) may build: the lane's served SPEC_ROWS before the deep depths
PLAIN_CROWD = 3               # sessions from which the crowd takes the true plain tick: 2 x (1 + 1) rows fill CROWD_ROWS, 3 sessions have none left to verify
CTX_EDGES = (16384, 65536)    # context buckets of the measured round costs: <= 16K ("8K"), <= 64K ("32K"), above ("128K"): a d3 round priced at 8K is not the price at 32K
PRIOR_COST = (1.0, 1.42, 1.74, 2.10, 2.77, 3.2, 3.6, 4.0, 4.4)   # round ms at depth d over the plain step, index 0 = plain; measured 8K-32K (STATUS_SPECDEEP): d4 = 5 rows leaves the GEMV band


def ctx_bucket(n: int) -> int:
    """Context bucket of a session (0 = '8K', 1 = '32K', 2 = '128K'): the key under which the engine keeps the measured ms per depth."""
    for i, e in enumerate(CTX_EDGES):
        if n <= e:
            return i
    return len(CTX_EDGES)


def enabled() -> bool:
    return os.environ.get(ENV, "0") not in ("0", "")


def prior_cost() -> tuple:
    raw = os.environ.get(COST_ENV, "")
    if raw:
        try:
            v = tuple(float(x) for x in raw.split(","))
            if len(v) >= 2 and all(x > 0 for x in v):
                return v
        except ValueError:
            pass
    return PRIOR_COST


class State:
    """One session: p[k-1] = EMA of the acceptance of draft position k; last = the depth planned last round; idle / gap = the plain-spell probe clock."""
    __slots__ = ("p", "last", "idle", "gap", "probing", "n")

    def __init__(self):
        self.p: list = []
        self.last = 0                                     # no round yet: the first draft is d1
        self.idle, self.gap, self.probing, self.n = 0, PROBE, False, 0

    def prob(self, k: int) -> float:
        """Estimate of position k (1-based): the EMA if observed, else the last observed one decayed by EXTRAP per position of distance."""
        if k <= len(self.p):
            return self.p[k - 1]
        if not self.p:
            return PRIOR_P * EXTRAP ** (k - 1)
        return self.p[-1] * EXTRAP ** (k - len(self.p))

    def observe(self, drafted: int, accepted: int) -> None:
        """A verified round that drafted `drafted` heads' tokens and landed `accepted` of them: only positions 1..min(drafted, accepted + 1) were reached."""
        if drafted <= 0:
            return
        self.n += 1
        reached = min(drafted, accepted + 1)
        for i in range(reached):
            x = 1.0 if i < accepted else 0.0
            if i < len(self.p):
                self.p[i] += (ALPHA_PROBE if self.probing else ALPHA) * (x - self.p[i])
            else:                                         # first look at this position: half the estimate it had, half the observation
                self.p.append(0.5 * (x + self.prob(i + 1)))
        for i in range(reached, len(self.p)):
            self.p[i] += DRIFT * (self.p[i - 1] * EXTRAP - self.p[i])
        if self.probing:
            self.probing = False
            self.gap = PROBE if accepted else min(2 * self.gap, PROBE_MAX)

    def tokens(self, d: int) -> float:
        t, run = 1.0, 1.0
        for k in range(1, d + 1):
            run *= self.prob(k)
            t += run
        return t


def _median(v) -> float:
    s = sorted(v)
    return s[(len(s) - 1) // 2]


def cost_table(cap: int, plain_ms, d_ms: dict, prior: tuple | None = None, min_n: int = 2):
    """ms of a round at depth 0..cap: [plain, d1, ...] from the engine's measured medians where >= min_n warm samples exist, else the prior shape scaled by the
    nearest measured depth (the plain step when none). Returns (list, plain_ref_ms)."""
    prior = prior or prior_cost()

    def shape(d):
        return prior[d] if d < len(prior) else prior[-1] + (d - len(prior) + 1) * (prior[-1] - prior[-2])
    pts = {d: _median(v) for d, v in d_ms.items() if d >= 1 and len(v) >= min_n}
    ref = plain_ms or (sum(m / shape(d) for d, m in pts.items()) / len(pts) if pts else 1.0)
    out = [ref]
    for d in range(1, cap + 1):
        if d in pts:
            out.append(pts[d])
        elif pts:
            near = min(pts, key=lambda x: (abs(x - d), x))
            out.append(shape(d) * pts[near] / shape(near))
        else:
            out.append(shape(d) * ref)
    return out, ref


def choose(st: State, cap: int, plain_ms, d_ms: dict, prior: tuple | None = None) -> int:
    """Depth of this round in 0..cap. Updates st.last / the probe clock."""
    if cap <= 0:
        st.last = 0
        return 0
    ms, ref = cost_table(cap, plain_ms, d_ms, prior)
    best, best_r = 0, 0.0
    for d in range(1, cap + 1):
        r = st.tokens(d) / ms[d]
        if best == 0 or r > best_r * (1.0 + TIE):
            best, best_r = d, r
    if best_r < PLAIN_MARGIN / ref:                                       # plain wins: probe a d1 round every `gap` plain rounds
        st.idle += 1
        if st.idle >= st.gap:
            st.idle, st.probing = 0, True
            st.last = 1
            return 1
        st.last = 0
        return 0
    st.idle = 0
    d = min(best, st.last + 1) if st.last >= 1 else 1                   # up by one per round, down at once; leaving a plain spell starts at d1
    st.last = d
    return d
