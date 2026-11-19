"""GLM-5.3 adaptive MTP depth: how many drafts to verify each round, from the MEASURED cost of a verify pass and the running acceptance (flag glm_adaptive).

Adapted from oMLX (https://github.com/jundot/omlx), Apache-2.0, omlx/patches/mlx_lm_mtp/batch_generator.py (_DepthController) and omlx/patches/mlx_lm_mtp/batch_policy.py
(BatchPolicy). What is kept from there: the score expected tokens / cost, acceptance as a per-position conditional EMA, a wall-clock cost EMA with a spike guard, a 3 % hysteresis,
bidirectional duty-bounded probes of a rival that is within 15 % of the best (so every estimate has a bounded age), a clear-loss test against the plain step. What differs:
  * the unit is the whole pass (n sessions, rows = n + drafts), because a verify row costs ~12 ms on this MoE and the sessions share the weights read: the cost table is keyed (n, rows);
  * the cost of a key nobody measured yet is the static G8a table (glm_cost.Costs), scaled by the measured / table ratio of the nearest measured keys, so the table's SHAPE survives
    but its LEVEL is the live one (G2c / G3b / G10 lower the verify rows: nothing here is a constant of today's kernels);
  * acceptance is kept per (session, text class) and pooled per class: a session that moves from code to prose (agent turns) resumes the right estimate, and a new session starts from its
    class pool instead of a flat prior;
  * a per-rival cooldown (doubles each time a probe confirms the rival loses) keeps the steady-state probe cost under 1 % of the rounds.
The verify decides every token: no depth policy changes the greedy output (tests/test_glm_adaptive.py).

Units: ms everywhere; the controller's clock is the sum of the cycles it was told about (no wall clock inside: deterministic tests).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import glm_cost as GC

# ---- constants (oMLX names in the comments)
ALPHA_TAU_MS = 400.0          # TAU_MS: cost EMA horizon, wall clock
SPIKE_RATIO, SPIKE_DAMP = 2.0, 0.25
HYSTERESIS = 1.03             # HYSTERESIS: switch vector only for a > 3 % score gain
PROBE_PERIOD_MS = 1000.0      # PROBE_PERIOD_MS
PROBE_PERIOD_MAX_MS = 5000.0  # PROBE_PERIOD_MAX_MS: a key older than this is stale
PROBE_LEN = 4                 # PROBE_LEN
PROBE_DUTY = 0.15             # PROBE_DUTY
PROBE_MARGIN = 1.15           # PROBE_MARGIN: a rival within this score ratio is worth measuring
COOLDOWN_MAX = 4.0            # per-rival probe period multiplier, reached by doubling
PARK_MARGIN = 0.03            # drafts must beat the plain pass by this to be planned at all (HYSTERESIS)
ACC_DECAY = 0.95              # per reached round (~20 round window; the engine's own counters use 0.9: a deeper position is reached p1 of the rounds, 10 was too noisy to price a 4 % gain)
ACC_PRIOR_N = 2.0
POOL_DECAY = 0.995
MIN_POOL_N = 6.0              # trials before a class pool replaces the static prior
EXPLORE_N = 6.0               # a position with fewer decayed trials than this is "unexplored" (a probe is informative; a verdict on it needs this many)
UNREACHED_DECAY = 0.98        # a position the session did not reach this round forgets slowly (half-life ~35 rounds): its estimate drifts back to the prior and gets re-probed
CLASS_EVERY = 8               # rounds between two class readings of a session
CLASS_WINDOW = 48             # output tokens read for the class
MAX_SESSIONS_KEPT = 64
EXHAUSTIVE_N = 4
WARM_N = 4                    # samples of a key averaged flat before the EMA takes over


def text_class(tail: list) -> str:
    """A token-id-only text class of the last output tokens (no tokenizer in the engine): the share of distinct ids. Code and maths repeat identifiers, brackets and
    indentation ('rep'), prose varies ('var'). Only keys the acceptance memory; the session's own counters decide the depth."""
    if len(tail) < 24:
        return "all"
    r = len(set(tail[-CLASS_WINDOW:])) / len(tail[-CLASS_WINDOW:])
    return "rep" if r < 0.5 else ("mix" if r < 0.72 else "var")


@dataclass
class SessView:
    """What the controller needs of one session of the pass."""
    key: object                  # stable id (Session.sid)
    cap: int                     # most drafts this round (room left, mtp depth, rows)
    tail: list = field(default_factory=list)   # last output tokens (class)


class RoundCosts:
    """Live round cost per (sessions n, rows) with the static table behind it."""

    def __init__(self, prior: GC.Costs | None = None):
        self.prior = prior or GC.Costs()
        self.t: dict = {}              # (n, rows) -> ms EMA (rows == n never: that is the plain pass, kept in plain)
        self.age: dict = {}            # key -> ms since it was measured
        self.cnt: dict = {}            # key -> samples
        self.plain: dict = {}          # n -> ms EMA of the plain pass of n sessions
        self.plain_age: dict = {}
        self.plain_cnt: dict = {}

    # -- the prior of a key: the table's round for an even spread of the drafts over the sessions
    def prior_ms(self, n: int, rows: int) -> float:
        d = max(0, rows - n)
        if d == 0:
            return self.prior.plain(n)
        base, extra = divmod(d, n)
        return self.prior.round_ms([base + 1] * extra + [base] * (n - extra))

    def tick(self, ms: float) -> None:
        for k in self.age:
            self.age[k] += ms
        for k in self.plain_age:
            self.plain_age[k] += ms

    @staticmethod
    def _ema(prev, x: float, ms: float, cnt: int = 0) -> float:
        """Wall-clock EMA (horizon ALPHA_TAU_MS) with a spike guard; the first WARM_N samples of a key are a plain mean (a probe of PROBE_LEN rounds settles it)."""
        if prev is None:
            return x
        a = 1.0 - math.exp(-max(min(ms, prev), 1e-3) / ALPHA_TAU_MS)          # a stall does not get the weight of its own length
        if x > SPIKE_RATIO * prev:
            a *= SPIKE_DAMP
        elif cnt < WARM_N:
            a = max(a, 1.0 / (cnt + 1))
        return (1.0 - a) * prev + a * x

    def observe(self, n: int, rows: int, ms: float) -> None:
        key = (n, rows)
        if rows <= n:
            self.observe_plain(n, ms)
            return
        self.t[key] = self._ema(self.t.get(key), ms, ms, self.cnt.get(key, 0))
        self.age[key] = 0.0
        self.cnt[key] = self.cnt.get(key, 0) + 1

    def observe_plain(self, n: int, ms: float) -> None:
        self.plain[n] = self._ema(self.plain.get(n), ms, ms, self.plain_cnt.get(n, 0))
        self.plain_age[n] = 0.0
        self.plain_cnt[n] = self.plain_cnt.get(n, 0) + 1

    def _ratio(self, n: int, rows: int) -> float:
        """Live / table ratio around (n, rows): the measured keys weighted by 1 / (1 + rows distance), same n first."""
        num = den = 0.0
        for (kn, kr), v in self.t.items():
            w = 1.0 / (1.0 + abs(kr - rows)) * (1.0 if kn == n else 0.35)
            num += w * v / self.prior_ms(kn, kr)
            den += w
        return num / den if den else 1.0

    def est(self, n: int, rows: int) -> float:
        if rows <= n:
            if n in self.plain:
                return self.plain[n]
            return self.prior.plain(n)                                 # never scaled by the rows' ratio: only a measured plain pass (a probe) moves it
        if (n, rows) in self.t:
            return self.t[(n, rows)]
        return self.prior_ms(n, rows) * self._ratio(n, rows)

    def stale(self, n: int, rows: int) -> bool:
        if rows <= n:
            return self.plain_age.get(n, math.inf) > PROBE_PERIOD_MAX_MS
        return self.age.get((n, rows), math.inf) > PROBE_PERIOD_MAX_MS

    def describe(self) -> str:
        a = ", ".join(f"n{n}r{r}:{v:.1f}" for (n, r), v in sorted(self.t.items()))
        p = ", ".join(f"n{n}:{v:.1f}" for n, v in sorted(self.plain.items()))
        return f"plain[{p}] spec[{a}]"


class Acceptance:
    """Per-position conditional acceptance, per (session, class), pooled per class."""

    def __init__(self):
        self.sess: dict = {}           # key -> {class -> ([succ], [trial])}
        self.pool: dict = {}           # class -> ([succ], [trial])
        self.cls: dict = {}            # key -> (current class, candidate, count, rounds since reading)

    @staticmethod
    def _grow(st, n: int) -> None:
        while len(st[1]) < n:
            st[0].append(0.0); st[1].append(0.0)

    def classify(self, key, tail: list, enabled: bool = True) -> str:
        cur, cand, cnt, since = self.cls.get(key, ("all", "all", 0, CLASS_EVERY))
        if not enabled:
            return "all"
        since += 1
        if since >= CLASS_EVERY:
            since, c = 0, text_class(tail)
            if c == cur:
                cand, cnt = c, 0
            elif c == cand:
                cnt += 1
                if cnt >= 1:                            # two readings in a row
                    cur, cnt = c, 0
            else:
                cand, cnt = c, 0
        self.cls[key] = (cur, cand, cnt, since)
        if len(self.cls) > MAX_SESSIONS_KEPT:
            for k in list(self.cls)[:len(self.cls) - MAX_SESSIONS_KEPT]:
                self.cls.pop(k, None); self.sess.pop(k, None)
        return cur

    def _prior(self, cls: str, j: int) -> float:
        pr = GC.PRIOR[min(j, len(GC.PRIOR) - 1)]
        st = self.pool.get(cls)
        if st is not None and j < len(st[1]) and st[1][j] >= MIN_POOL_N:
            return (st[0][j] + pr * ACC_PRIOR_N) / (st[1][j] + ACC_PRIOR_N)
        return pr

    def trials(self, key, cls: str, j: int) -> float:
        st = self.sess.get(key, {}).get(cls)
        return st[1][j] if st is not None and j < len(st[1]) else 0.0

    def p(self, key, cls: str, j: int) -> float:
        """P(draft j accepted | drafts before it accepted). A position nobody measured yet takes the prior's shape at this session's level (never above the position before it)."""
        st = self.sess.get(key, {}).get(cls)
        pr = self._prior(cls, j)
        if st is not None and j < len(st[1]) and st[1][j] > 0:
            return (st[0][j] + pr * ACC_PRIOR_N) / (st[1][j] + ACC_PRIOR_N)
        if j == 0:
            return pr
        p0 = self.p(key, cls, 0)
        lvl = min(1.15, max(0.5, p0 / self._prior(cls, 0)))
        return min(self.p(key, cls, j - 1), pr * lvl)

    def observe(self, key, cls: str, drafted: int, accepted: int) -> None:
        if drafted <= 0:
            return
        accepted = max(0, min(accepted, drafted))
        reached = min(drafted, accepted + 1)
        for st in (self.sess.setdefault(key, {}).setdefault(cls, ([], [])), self.pool.setdefault(cls, ([], []))):
            self._grow(st, reached)
            dec = ACC_DECAY if st is not self.pool[cls] else POOL_DECAY
            for j in range(len(st[1])):
                if j < reached:
                    st[1][j] = dec * st[1][j] + 1
                    st[0][j] = dec * st[0][j] + (1 if j < accepted else 0)
                elif dec == ACC_DECAY:                                    # session counters only: the class pool keeps its weight
                    st[1][j] *= UNREACHED_DECAY
                    st[0][j] *= UNREACHED_DECAY

    def forget(self, key) -> None:
        self.sess.pop(key, None); self.cls.pop(key, None)


class DepthController:
    """One per engine. choose() before a pass, observe() after it (and observe_plain() after a plain pass)."""

    def __init__(self, costs: GC.Costs | None = None, max_rows: int = 0, use_class: bool = True):
        self.costs = RoundCosts(costs)
        self.acc = Acceptance()
        self.max_rows = max_rows
        self.use_class = use_class
        self.clock = 0.0               # ms of cycles seen
        self.cur: tuple | None = None  # the vector in force (hysteresis)
        self.cur_keys: tuple = ()
        self.probe_left = 0
        self.probe_vec: tuple | None = None
        self.probe_base: float | None = None
        self.last_probe = 0.0
        self.last_explore = 0.0
        self.cool: dict = {}           # rival vector -> period multiplier
        self.cycle_ema = 40.0
        self.rounds = 0
        self.stats = {"rounds": 0, "probe_rounds": 0, "probes": 0, "park": 0, "switch": 0, "cooldown_up": 0}
        self._last_choice: tuple | None = None
        self._cls: list = []

    # ---------------------------------------------------------------- scoring
    def _tokens(self, v: SessView, cls: str, d: int) -> float:
        t, c = 1.0, 1.0
        for j in range(d):
            c *= self.acc.p(v.key, cls, j)
            t += c
        return t

    def _rate(self, views, cls, ds) -> float:
        n = len(views)
        rows = n + sum(ds)
        if self.max_rows and rows > self.max_rows:
            return -1.0
        return sum(self._tokens(v, c, d) for v, c, d in zip(views, cls, ds)) / self.costs.est(n, rows)

    def _best(self, views, cls) -> tuple:
        n = len(views)
        caps = [max(0, v.cap) for v in views]
        if n <= EXHAUSTIVE_N:
            import itertools
            best, br = (0,) * n, self._rate(views, cls, (0,) * n)
            for ds in itertools.product(*[range(c + 1) for c in caps]):
                r = self._rate(views, cls, ds)
                if r > br * (1 + 1e-9):
                    best, br = ds, r
            return best
        u = max(caps) if caps else 0
        cand = [tuple(min(d, c) for c in caps) for d in range(u + 1)]
        best = max(cand, key=lambda ds: self._rate(views, cls, ds))
        br = self._rate(views, cls, best)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for dd in (-1, 1):
                    c = list(best); c[i] += dd
                    if 0 <= c[i] <= caps[i]:
                        r = self._rate(views, cls, tuple(c))
                        if r > br * (1 + 1e-9):
                            best, br, improved = tuple(c), r, True
        return best

    def _neighbors(self, views, ds) -> list:
        out = []
        for i, v in enumerate(views):
            for dd in (-1, 1):
                c = list(ds); c[i] += dd
                if 0 <= c[i] <= max(0, v.cap):
                    out.append(tuple(c))
        return out

    def _informative(self, views, cls, ds) -> bool:
        """A probe teaches something: its cost key is stale, or one of its positions has hardly been tried by that session."""
        n = len(views)
        if self.costs.stale(n, n + sum(ds)):
            return True
        for v, c, d in zip(views, cls, ds):
            for j in range(d):
                if self.acc.trials(v.key, c, j) < EXPLORE_N:
                    return True
        return False

    # ---------------------------------------------------------------- the decision
    def choose(self, views: list) -> list:
        n = len(views)
        if n == 0:
            return []
        cls = [self.acc.classify(v.key, v.tail, self.use_class) for v in views]
        self._cls = cls
        self.rounds += 1
        self.stats["rounds"] += 1
        zero = (0,) * n
        keys = tuple(v.key for v in views)
        caps = [max(0, v.cap) for v in views]
        if not any(caps):
            self._last_choice = zero
            return [0] * n
        if keys != self.cur_keys:                                        # a join / leave: the vector in force and the probe are about another pass
            self.cur, self.cur_keys, self.probe_left, self.probe_vec = None, keys, 0, None
        if self.probe_left > 0 and self.probe_vec is not None and len(self.probe_vec) == n \
                and all(d <= c for d, c in zip(self.probe_vec, caps)) \
                and not (self.max_rows and n + sum(self.probe_vec) > self.max_rows):
            self.probe_left -= 1
            self.stats["probe_rounds"] += 1
            self._last_choice = self.probe_vec
            return list(self.probe_vec)
        self.probe_left, self.probe_vec = 0, None
        best = self._best(views, cls)
        base = self._rate(views, cls, zero)
        br = self._rate(views, cls, best)
        if br <= base * (1 + PARK_MARGIN):                               # drafts do not beat the plain pass: park
            best, br = zero, base
            self.stats["park"] += 1
        if self.cur is not None and len(self.cur) == n and all(d <= c for d, c in zip(self.cur, caps)):
            cr = self._rate(views, cls, self.cur)
            if cr > 0 and br < cr * HYSTERESIS:                          # hysteresis: stay unless the gain is above 3 %
                best, br = self.cur, cr
        if self.cur is not None and best != self.cur:
            self.stats["switch"] += 1
            self.cool.clear()
        self.cur = best
        # ---- probe: the best rival within 15 % whose measurement is stale / unexplored, at a duty-bounded cadence
        period = max(PROBE_PERIOD_MS, PROBE_LEN * self.cycle_ema / PROBE_DUTY)
        if self.clock - self.last_probe >= period:
            rivals = []
            for r in self._neighbors(views, best) + ([zero] if best != zero else []):
                if r == best:
                    continue
                rr = self._rate(views, cls, r)
                if rr * PROBE_MARGIN >= br and self._informative(views, cls, r) \
                        and self.clock - self.last_probe >= period * self.cool.get(r, 1.0):
                    rivals.append((rr, r))
            if not rivals and self.clock - self.last_explore >= max(PROBE_PERIOD_MAX_MS, 2.0 * period):
                # staleness exploration (oMLX _most_stale): no rival is within the margin, but a neighbour nobody tried for a long time may have moved (acceptance, costs)
                far = [r for r in self._neighbors(views, best) + ([zero] if best != zero else [])
                       if r != best and self._informative(views, cls, r)
                       and self.clock - self.last_explore >= max(PROBE_PERIOD_MAX_MS, 2.0 * period) * self.cool.get(r, 1.0)
                       * (1.0 + min(1.0, 10.0 * max(0.0, 1.0 - self._rate(views, cls, r) / br)))]      # the worse the rival looks, the rarer its exploration (up to 2x: a plain pass 25 % under is tried half as often)
                if far:
                    rivals = [(self._rate(views, cls, r), r) for r in far]
                    self.last_explore = self.clock
            if rivals:
                rr, r = max(rivals)
                self.probe_vec, self.probe_left, self.probe_base = r, PROBE_LEN - 1, br
                self.last_probe = self.clock
                self.stats["probes"] += 1
                self._last_choice = r
                return list(r)
        self._last_choice = best
        return list(best)

    # ---------------------------------------------------------------- the feedback
    def observe(self, views: list, depths: list, drafted: list, accepted: list, cycle_ms: float | None) -> None:
        """One verify pass ran: depths planned (all drafted unless the block was cut), drafted / accepted counts per session, the cycle ms (None: not a clean sample)."""
        n = len(views)
        cls = self._cls if len(self._cls) == n else ["all"] * n
        for v, c, dr, ac in zip(views, cls, drafted, accepted):
            self.acc.observe(v.key, c, int(dr), int(ac))
        if cycle_ms is None:
            return
        rows = n + sum(int(d) for d in drafted)
        self.clock += cycle_ms
        self.costs.tick(cycle_ms)
        self.costs.observe(n, rows, cycle_ms)
        self.cycle_ema += 0.1 * (cycle_ms - self.cycle_ema)
        self._probe_verdict(views, cls, tuple(depths))

    def observe_plain(self, n: int, ms: float) -> None:
        self.clock += ms
        self.costs.tick(ms)
        self.costs.observe_plain(n, ms)
        self.cycle_ema += 0.1 * (ms - self.cycle_ema)

    def _probe_verdict(self, views, cls, ds) -> None:
        """A finished probe that did not beat the vector in force by the hysteresis doubles that rival's cooldown."""
        if self.probe_vec is None or self.probe_left != 0 or ds != self.probe_vec or self.probe_base is None:
            return
        rr = self._rate(views, cls, ds)
        inc = self._rate(views, cls, self.cur) if self.cur is not None and len(self.cur) == len(ds) else self.probe_base
        tried = all(self.acc.trials(v.key, c, j) >= EXPLORE_N for v, c, d in zip(views, cls, ds) for j in range(d))
        if rr < inc or (rr < inc * HYSTERESIS and tried):                               # the rival did not beat the vector in force by the hysteresis, on enough trials: probe it less often
            m = self.cool.get(ds, 1.0)
            if m < COOLDOWN_MAX:
                self.cool[ds] = min(COOLDOWN_MAX, m * 2.0)
                self.stats["cooldown_up"] += 1
        self.probe_vec = None

    def finish(self, key) -> None:
        self.acc.forget(key)

    def describe(self) -> str:
        return f"{self.costs.describe()} stats={self.stats}"
