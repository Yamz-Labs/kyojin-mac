# SPDX-License-Identifier: Apache-2.0
"""Measured convex cost ladder for the verify width (lever spec_ladder, env KYOJIN_SPEC_LADDER).

The rival's `chooseRows` (mlx-serve src/engines/mlx/spec/dflash/policy.zig) picks the verify width from a
measured cost ladder instead of a fixed table: cost(rows) is linear between measured points, expected tokens
come from online per-position acceptance, and the width is the argmax of tokens per ms. Drafting backs off
when it stops paying (a losing round owes plain rounds, the period doubles to 32, a win clears the debt).

This module is the generic (any MTP / DSpark backend) version of that idea, re-implemented for the serving
engine. It is pure Python: the engine feeds it each round's wall time and token count, and asks it for the
next width.

* Cost ladder: per depth d (drafts, 1 = plain) the median ms of the rounds the engine actually ran at that
  depth (`observe`). The FIRST round the engine runs at a new width is not a cost sample: it builds the
  verify / draft graph for that row count (the engine's `_learn_k` skips it for the same reason), and one
  compile stall in a one-sample median pins that width as unaffordable for ever. A width between two measured
  depths is priced on the line between them (the rival's interpolation); a width past the deepest measured
  one is never priced, it may only be *probed*, one row past the current best, to get measured. The
  least-squares line of >= MIN_POINTS depths (`fit`) is kept for the stats only: it used to price the deep
  widths from three shallow points, and rows 5+ leave the GEMV band those points were measured in.
* Expected tokens: 1 + p_1 + p_1 p_2 + ... from the per-position conditional acceptance the engine tracks
  (the brief's acc(d) * (1 + d) with the conditioning made explicit).
* Choice: argmax expected tokens / priced ms over d in 1..cap (cap from the backend's max_spec_rows, the
  drafter block, the session room and MAX_ROWS). A width within ROW_TIE of the best loses to the cheaper
  round. If the best priced rate is below the measured plain rate (1 token per plain_ms) the choice is 0: the
  caller runs the ordinary plain tick.
* Leaving plain is not free: the engine's plain rounds launch the next step before reading the one in flight,
  and a drafted round that follows a plain one must drain that step (the engine reads it outside the round's
  timed window) and feed the draft head the taps the plain rounds deferred. So a round that leaves plain is
  judged in `observe` against `ms + plain_pass / run_ema` plus ROW_TIE, not against `ms`: `run_ema` is the
  measured drafted rounds per stretch, so a one-off probe pays most of the drained step and a width that has
  been holding for 20 rounds pays 1/20 of it. The fixed 5 % LEAVE_PLAIN stays as a floor on the choice. And
  once drafting, the ladder does not return to plain on a rate estimate alone: only a round that LOSES stops
  the stretch (`owed`). Otherwise a noisy median that dips below plain sends one plain round and the next
  drafted round pays the drain again, every few rounds.
* Uncertainty: while fewer than MIN_POINTS widths are priced, the ladder measures the SHALLOWEST width that
  has no warm sample yet, every round (it has to measure to price). It never re-measures `best + 1` while a
  cheaper width is unpriced: an early losing sample at depth 1 then cannot pin the ladder at depth 1 for ever
  (the old loop probed `best + 1`, so `best == 0` meant depth 1 every round and depths 2..cap were never
  measured). After that, at a low rate (PROBE_PERIOD_MIN rounds, doubling per losing probe up to MAX_PERIOD,
  halving per winning probe). A losing drafted round puts the ladder in debt: it then returns 0 for `period`
  rounds (skip drafting), the period doubling up to MAX_PERIOD; a winning round clears the debt and halves
  the period. The debt is bounded, so a better acceptance profile is priced again within MAX_PERIOD rounds
  (no latch at 0).

The width only steers the plan: the verify decides every token, and the plan never samples. Greedy output is
identical to serial decode for any width ONLY where the backend's verify is row invariant (same kernel and
same arithmetic at every verify width). That is a property of the backend, not of this module: this engine's
Exl3 linears switch the row-FMA class at `metal_src.HALF_ROWS` and the kernel band at
`Exl3Linear.GEMV_MAX_ROWS`, so two widths can round the same row differently. `tests/test_spec_ladder.py`
checks the part the ladder owns (same width -> same tokens, accept logic width independent).
"""
from __future__ import annotations

ENV = "KYOJIN_SPEC_LADDER"     # flags.py spec_ladder: "0" = off (the ordinary depth policy)

MAX_ROWS = 8                   # widths priced: 1..8 drafts (the caller clamps further by max_spec_rows, the drafter block and the room)
MIN_POINTS = 3                 # distinct warm depths the ladder prices before it stops measuring every unpriced width
MAX_PERIOD = 32                # longest debt / probe gap, in rounds (rival Drafting.MAX_PERIOD)
PROBE_PERIOD_MIN = 8           # low rate at which a priced ladder widens one row past its best
ROW_TIE = 0.02                 # a width within 2 % of the best rate loses to the cheaper one
LEAVE_PLAIN = 0.05             # floor on the margin a drafted round needs over plain (the drained step is priced separately, see `observe`)
WINDOW = 5                     # ms samples per depth behind its median

# Engine side: "0" (the ordinary depth policy) / "1" (the ladder). Read per round so POST /admin/flags switches
# a loaded server; this literal is the one the flag row and the engine property agree on.
ENABLED_DEFAULT = "0"


def _median(vals: list) -> float:
    s = sorted(vals)
    return s[(len(s) - 1) // 2]


class Ladder:
    """Engine-wide cost ladder (one round = all sessions of the verify pass: the row cap is shared)."""

    def __init__(self):
        self.ms: dict = {}                  # depth -> [round ms], last WINDOW warm samples
        self._visits: dict = {}             # depth -> rounds observed (the first one builds that width's graphs: not a cost sample)
        self._followed_plain = False        # the round in flight drains the plain step the previous round launched ahead
        self.run_ema = 1.0                  # measured drafted rounds per stretch (the horizon a leave-plain transition is paid over)
        self.stretch = 0                    # drafted rounds of the stretch in flight
        self.owed = 0                       # plain rounds still owed before the next drafted round (a losing streak)
        self.period = 1                     # plain rounds the next losing round will owe (doubles to MAX_PERIOD)
        self.probe_gap = 1                  # rounds between two widening probes (doubles per losing probe)
        self.since_probe = 0
        self.probes = 0                     # rounds planned one row past the best (stats)
        self.skips = 0                      # rounds skipped by the debt (stats)
        self._planned: tuple | None = None  # (depth, was_a_probe) of the round in flight
        self.last = 0                       # width of the previous round (0 = plain, whose next step is already in flight)

    # ------------------------------------------------------------------------------------------------ measurements
    def observe(self, depth: int, ms: float, tokens: float, plain_per_tok: float,
                plain_pass: float | None = None) -> bool | None:
        """A round of `tokens` tokens that took `ms` ms at `depth` drafts. True when it beat the measured plain rate
        (tokens * plain_per_tok >= ms, plus the drained plain step when this round followed a plain one), False when it
        lost, None when the sample is unusable. Feeds the ladder and the drafting backoff (rival Drafting.record).
        `plain_pass`: ms of a plain pass of the whole pass (default: `plain_per_tok`, the n = 1 case); the first round
        at a width is not a cost sample (its graph build is not the width's cost)."""
        try:
            ms, tokens, plain_per_tok = float(ms), float(tokens), float(plain_per_tok)
            plain_pass = plain_per_tok if plain_pass is None else float(plain_pass)
        except (TypeError, ValueError):
            return None
        if depth <= 0 or not (ms > 0) or not (tokens > 0) or not (plain_per_tok > 0) or ms != ms:
            return None
        d = int(depth)
        self._visits[d] = self._visits.get(d, 0) + 1
        if self._visits[d] > 1:                       # the first round at a width builds its verify / draft graph: not a cost
            self.ms.setdefault(d, []).append(ms)
            self.ms[d] = self.ms[d][-WINDOW:]
        drain = plain_pass / max(1.0, self.run_ema) if self._followed_plain else 0.0   # the queued step, paid over a drafted stretch
        won = tokens * plain_per_tok >= (ms + drain) * (1.0 + ROW_TIE)
        probe = bool(self._planned and self._planned[0] == depth and self._planned[1])
        self._planned = None
        if probe:
            if won:
                self.probe_gap = max(1, self.probe_gap // 2)
            else:
                self.probe_gap = min(self.probe_gap * 2, MAX_PERIOD)
        if won:
            self.owed = 0
            self.period = max(1, self.period // 2)
        else:
            self.owed = self.period
            self.period = min(self.period * 2, MAX_PERIOD)
        return won

    # ------------------------------------------------------------------------------------------------ ladder
    def fit(self) -> tuple | None:
        """(s_ms, slope_ms) of cost(d) = s + slope * d from >= MIN_POINTS distinct measured depths, else None."""
        pts = [(d, _median(v)) for d, v in self.ms.items() if v]
        if len(pts) < MIN_POINTS:
            return None
        md = sum(d for d, _ in pts) / len(pts)
        mm = sum(m for _, m in pts) / len(pts)
        den = sum((d - md) ** 2 for d, _ in pts)
        if den <= 0:
            return None
        slope = sum((d - md) * (m - mm) for d, m in pts) / den
        return max(0.0, mm - slope * md), max(1e-3, slope)

    def cost(self, d: int) -> float | None:
        """Priced ms of a round at d drafts: its measured median, else the line between the two measured depths around it, else None. A width past
        the deepest measured one (or below the shallowest) is never extrapolated: rows 5+ leave the fused GEMV band the shallow rounds were
        measured in (MMA / NAX / row-group kernels, longer draft chains), so a line through depths 1-3 underprices them; it can only be probed."""
        d = int(d)
        v = self.ms.get(d)
        if v:
            return max(1e-3, _median(v))
        lo = [k for k, m in self.ms.items() if m and k < d]
        hi = [k for k, m in self.ms.items() if m and k > d]
        if not lo or not hi:
            return None
        a, b = max(lo), min(hi)
        ca, cb = _median(self.ms[a]), _median(self.ms[b])
        return max(1e-3, ca + (cb - ca) * (d - a) / (b - a))

    def priced(self, d: int) -> bool:
        return self.cost(d) is not None

    @staticmethod
    def tokens(acc, d: int) -> float:
        """Expected tokens of a round at d drafts: 1 + sum_k prod_{i<=k} p_i (acc = conditional per-position acceptance)."""
        acc = list(acc or ())
        run, tok = 1.0, 1.0
        for i in range(max(0, int(d))):
            p = acc[i] if i < len(acc) else (acc[-1] if acc else 0.7)
            p = min(1.0, max(0.0, float(p)))
            run *= p
            tok += run
        return tok

    def plan(self, cap: int, acc, plain_per_tok: float, n: int = 1) -> int:
        """Drafts to ask for this round (0 = plain): argmax expected tokens / priced ms, never a width that was not
        measured or fitted; measures every unpriced width once before pricing; probes one row past the best at the probe
        rate; a debt from a losing streak returns 0. `n` = sessions of the pass: a round is priced for the whole pass
        (ms), so its tokens are the n sessions' tokens (plain_per_tok = the plain pass ms over n, as `observe` gets it)."""
        self._followed_plain = (self.last == 0)                # this round will drain the plain step launched ahead
        d = self._plan(cap, acc, plain_per_tok, max(1, int(n)))
        if d > 0:                                              # the measured length of a drafted stretch sets what a leave-plain
            self.stretch = self.stretch + 1 if self.last > 0 else 1   # transition is amortized over (observe)
        elif self.last > 0 and self.stretch > 0:
            self.run_ema = 0.5 * self.run_ema + 0.5 * self.stretch
            self.stretch = 0
        self.last = d
        return d

    def _unpriced(self, cap: int) -> int | None:
        """Shallowest width in 1..cap with no warm cost sample yet (the width the ladder still has to measure to price)."""
        for d in range(1, cap + 1):
            if self.cost(d) is None:
                return d
        return None

    def _plan(self, cap: int, acc, plain_per_tok: float, n: int) -> int:
        cap = max(0, min(int(cap), MAX_ROWS))
        if cap <= 0:
            return 0
        try:
            plain_per_tok = float(plain_per_tok)
        except (TypeError, ValueError):
            plain_per_tok = 0.0
        if not (plain_per_tok > 0):
            self._planned = (cap, False)
            return cap                                        # no plain measurement yet: draft (the caller measures plain on its ticks)
        if self.owed > 0:                                     # a losing streak: skip drafting, bounded by MAX_PERIOD
            self.owed -= 1
            self.skips += 1
            self.since_probe += 1
            self._planned = (0, False)
            return 0
        plain_rate = 1.0 / plain_per_tok                      # tokens per ms of the measured plain step
        best, best_rate = 0, plain_rate
        lead = 1.0 + ROW_TIE + (LEAVE_PLAIN if self.last == 0 else 0.0)
        for d in range(1, cap + 1):
            c = self.cost(d)
            if c is None:                                     # never price a width that has neither a round nor a fit
                continue
            r = n * self.tokens(acc, d) / c
            if r > best_rate * (lead if best == 0 else 1.0 + ROW_TIE):   # ties go to the cheaper round
                best, best_rate = d, r
        if best == 0 and self.last > 0 and self.cost(self.last) is not None and self.last <= cap:
            # Already drafting: a rate estimate that dipped below plain is not a reason to leave the pipeline. Leaving costs the queued step
            # (drained off the engine's timed window) plus the draft head's deferred feeds; only a round that LOSES (observe -> debt) stops a
            # drafted stretch, so a noisy median cannot flap plain<->draft and pay that drain twice.
            best = self.last
        self.since_probe += 1
        priced = sum(1 for d in range(1, cap + 1) if self.cost(d) is not None)
        miss = self._unpriced(cap) if priced < min(MIN_POINTS, cap) else None
        if miss is not None:                                  # measure every width once (not `best + 1` again): a losing
            self.since_probe = 0                              # sample at depth 1 must not hide depths 2..cap for ever
            self.probes += 1
            self._planned = (miss, True)
            return miss
        if self.since_probe >= max(PROBE_PERIOD_MIN, self.probe_gap):
            self.since_probe = 0
            if cap > best:                                    # widen the ladder one row past the best, at a low rate
                self.probes += 1
                self._planned = (min(cap, best + 1), True)
                return min(cap, best + 1)
        self._planned = (best, False)
        return best

    # ------------------------------------------------------------------------------------------------ stats
    def snapshot(self) -> dict:
        """What /v1/stats shows for the ladder: measured depths, fit, probe/debt state (JSON-safe)."""
        f = self.fit()
        return {"depths": {str(d): round(_median(v), 2) for d, v in sorted(self.ms.items()) if v},
                "visits": {str(d): n for d, n in sorted(self._visits.items())},
                "s_ms": None if f is None else round(f[0], 2), "slope_ms": None if f is None else round(f[1], 3),
                "probes": self.probes, "skips": self.skips, "owed": self.owed, "period": self.period,
                "probe_gap": self.probe_gap, "run_ema": round(self.run_ema, 2), "stretch": self.stretch}
