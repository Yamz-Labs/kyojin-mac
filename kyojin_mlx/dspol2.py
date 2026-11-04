# SPDX-License-Identifier: Apache-2.0
"""One calibrated draft policy with a measured per-row cost (flag dspol_v2, default 0).

Replaces the confidence gate (a fixed threshold that cuts rows) plus the depth cut (kyojin_mlx.dspol + Engine._plan_depth) by one rule:

    p_k   = P(accept draft k | drafts 0..k-1 kept), calibrated from the confidence head
    E[T]  = 1 + p_1 + p_1 p_2 + ...                       (R rows verified = R - 1 drafts, R = 1 is a plain step)
    R*    = argmax_R E[T](R) / cost(R); every R within TIE (2 %) of the best gain loses to the cheaper (smaller) R

Calibration (`Calibration`): bins on x = -log10(1 - conf), dense near conf 0.9-1 where the draft head lives; a small fitted logistic
prior p = sigmoid(a + b x) worth PRIOR_N pseudo-counts per bin; per-bin forgetting FORGET applied to the bin that receives a new trial
(a bin that is not visited keeps its history). A trial is a verified position that was reached (every draft before it was kept),
a success is a position that was accepted: so the table is P(accept k | kept so far) by construction.

Cost (`RowCost`): a measured table by verify width (whole round ms, the draft call included), scaled by the live plain-step ms over the
plain ms the table was measured at. R = 1 is the live plain step itself, so prose can fall back when speculation does not pay.

Nothing here knows a prompt or a model: the prior and the table are overridable (KYOJIN_DSPOL_V2_PRIOR, KYOJIN_DSPOL_V2_COSTS) and
`fit_prior` refits the prior from a round trace. Speculative decode stays exact for any R (the trunk verifies every token).

The prior is fitted online: every REFIT_EVERY trials the logistic (a, b) is refitted on the calibration's own (decayed) bin counts by a
weighted Newton fit over the bin centres, ridged toward the cold-start curve, so bins nobody has visited yet inherit what the visited ones
learned. The cold-start curve (PRIOR_A, PRIOR_B) is a hand-set curve through the accept rates the ledger reports (0.73-0.83 at gate 0.5,
~0.9 near conf 0.9), NOT a fit to a trace; no round trace with confidences exists in the repo. `fit_prior` refits from a round_log trace
(tools/dspol2_calib.py --refit) and KYOJIN_DSPOL_V2_PRIOR sets the cold start from it, as an env value, never a code default.
"""
from __future__ import annotations

import json
import math
import os

ENV = "KYOJIN_DSPOL_V2"
ENV_COSTS = "KYOJIN_DSPOL_V2_COSTS"      # "2:36.8,3:42.0,..." or a JSON file {"1": plain, "2": ms, ...}
ENV_PRIOR = "KYOJIN_DSPOL_V2_PRIOR"      # "a,b" of the logistic prior

# Whole-round ms by verify width (rows incl. the pending token, draft call included), ledger "Whole round width sweep" (opr6) and the
# plain step it was measured at (opplain, 20.57 ms). Rows 7 interpolates; past 8 the mean slope w2..w8 carries on.
PLAIN_REF_MS = 20.57
WIDTH_MS = {2: 36.8, 3: 42.0, 4: 46.7, 5: 52.1, 6: 58.1, 8: 63.0}

TIE = 0.02              # spec/dflash/policy.zig ROW_TIE: within 2 % of the best gain the cheaper row count wins
BIN_W = 0.25            # bin width on x = -log10(1 - conf)
X_MAX = 3.0             # conf 0.999 and above share the last bin
FORGET = 0.97           # per-bin decay applied when the bin receives a trial
PRIOR_N = 3.0           # pseudo-counts of the prior
PRIOR_A, PRIOR_B = -0.2, 1.9      # sigmoid(a + b x): 0.6 at conf 0.5 (x 0.3), 0.85 at conf 0.9, 0.97 at conf 0.99
REFIT_EVERY = 32        # trials between two online refits of the prior (a, b) on the calibration's own bin counts
REFIT_MIN = 40.0        # decayed trials needed before the first refit
REFIT_RIDGE = 2.0       # pseudo-trials pulling the refit toward the cold-start curve
TRACK_ALPHA = 0.4       # per-position EMA of x over the rounds that drafted (plan-time estimate of this round's confidences)
EPS = 1e-6


def enabled() -> int:
    """Mode of the flag: 0 off, 1 policy at plan time (lazy draft kept), 2 plus the per-round choice after the draft (eager draft)."""
    try:
        return max(0, int(os.environ.get(ENV, "0") or 0))
    except ValueError:
        return 0


def xof(conf: float) -> float:
    """-log10(1 - conf), clipped to [0, X_MAX]."""
    c = min(1.0 - EPS, max(0.0, float(conf)))
    return min(X_MAX, -math.log10(1.0 - c))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def load_prior() -> tuple:
    try:
        v = os.environ.get(ENV_PRIOR, "").strip()
        if v:
            a, b = (float(x) for x in v.split(",")[:2])
            return a, b
    except Exception:
        pass
    return PRIOR_A, PRIOR_B


# ----------------------------------------------------------------------------------------------------------------------- cost
class RowCost:
    """cost(R) in ms: the measured width table scaled by live plain ms / table plain ms; R = 1 is the live plain step."""

    def __init__(self, table: dict | None = None, ref_plain: float = PLAIN_REF_MS):
        t = {int(k): float(v) for k, v in (table or WIDTH_MS).items() if int(k) >= 2}
        self.ref_plain = float(table.get(1, ref_plain)) if table else float(ref_plain)
        assert self.ref_plain > 0 and len(t) >= 1
        self.knots = sorted(t.items())
        lo, hi = self.knots[0], self.knots[-1]
        self.tail_slope = (hi[1] - lo[1]) / (hi[0] - lo[0]) if hi[0] > lo[0] else max(0.0, hi[1] - self.ref_plain)

    def table_ms(self, rows: int) -> float:
        """Unscaled ms at `rows` verify rows (1 = the table's own plain step)."""
        if rows <= 1:
            return self.ref_plain
        k = self.knots
        if rows <= k[0][0]:                                  # between the plain step and the first width
            return self.ref_plain + (k[0][1] - self.ref_plain) * (rows - 1) / (k[0][0] - 1)
        for (r0, m0), (r1, m1) in zip(k, k[1:]):
            if rows <= r1:
                return m0 + (m1 - m0) * (rows - r0) / (r1 - r0)
        return k[-1][1] + self.tail_slope * (rows - k[-1][0])

    def ms(self, rows: int, plain_ms: float | None = None) -> float:
        """Round ms at `rows` verify rows given the live plain step (None: the table's own)."""
        if rows <= 1:
            return float(plain_ms) if plain_ms else self.ref_plain
        scale = float(plain_ms) / self.ref_plain if plain_ms else 1.0
        return self.table_ms(rows) * scale

    @staticmethod
    def parse(v: str) -> "RowCost":
        """'2:36.8,3:42,...' (a '1:' entry sets the table's plain ms), a JSON object or a JSON file; the defaults when it does not parse."""
        try:
            v = (v or "").strip()
            if not v:
                return RowCost()
            if v.startswith("{"):
                d = json.loads(v)
            elif ":" in v:
                d = {p.split(":")[0]: float(p.split(":")[1]) for p in v.split(",")}
            else:
                d = json.loads(open(v).read())
            d = {int(k): float(x) for k, x in d.items()}
            return RowCost(d)
        except Exception:
            return RowCost()

    @staticmethod
    def load() -> "RowCost":
        return RowCost.parse(os.environ.get(ENV_COSTS, ""))


# ----------------------------------------------------------------------------------------------------------------------- calibration
class Calibration:
    """P(accept draft k | kept so far) by bin of x = -log10(1 - conf). Shared by every session (the head's calibration is the model's)."""

    def __init__(self, prior: tuple | None = None, prior_n: float = PRIOR_N, forget: float = FORGET, bin_w: float = BIN_W):
        self.a, self.b = prior if prior is not None else load_prior()
        self.prior_n, self.forget, self.bin_w = float(prior_n), float(forget), float(bin_w)
        self.nbins = int(round(X_MAX / self.bin_w)) + 1
        self.a0, self.b0 = self.a, self.b                 # the cold-start curve the online refit is ridged toward
        self.succ = [0.0] * self.nbins
        self.trial = [0.0] * self.nbins
        self.seen = 0
        self.refits = 0

    def bin_of(self, conf: float) -> int:
        return min(self.nbins - 1, int(xof(conf) / self.bin_w))

    def prior(self, x: float) -> float:
        return _sigmoid(self.a + self.b * x)

    def _centre(self, b: int) -> float:
        return min(X_MAX, (b + 0.5) * self.bin_w)

    def prob(self, conf: float) -> float:
        b = self.bin_of(conf)
        return (self.succ[b] + self.prior_n * self.prior(self._centre(b))) / (self.trial[b] + self.prior_n)

    def prob_x(self, x: float) -> float:
        return self.prob(1.0 - 10.0 ** (-min(X_MAX, max(0.0, x))))

    def probs(self, conf: list) -> list:
        return [self.prob(c) for c in conf]

    def observe(self, conf: list, drafted: int, accepted: int) -> None:
        """A verified round: positions 0..accepted were reached (the first miss is the last trial), positions < accepted succeeded."""
        reached = min(int(drafted), int(accepted) + 1, len(conf or ()))
        for i in range(reached):
            b = self.bin_of(conf[i])
            self.trial[b] = self.forget * self.trial[b] + 1.0
            self.succ[b] = self.forget * self.succ[b] + (1.0 if i < accepted else 0.0)
            self.seen += 1
            if self.seen % REFIT_EVERY == 0:
                self.refit()

    def refit(self) -> bool:
        """Weighted Newton fit of sigmoid(a + b x) on the bins' own counts (x = bin centre), ridged toward the cold-start curve. False: too few
        trials, or every trial one class (nothing to fit)."""
        pts = [(self._centre(b), self.succ[b], self.trial[b]) for b in range(self.nbins) if self.trial[b] > 1e-6]
        n, k = sum(t for _, _, t in pts), sum(sc for _, sc, _ in pts)
        if n < REFIT_MIN or k < 1.0 or k > n - 1.0:
            return False
        a, b = self.a, self.b
        for _ in range(12):
            g0 = REFIT_RIDGE * (a - self.a0)
            g1 = REFIT_RIDGE * (b - self.b0)
            h00 = h11 = REFIT_RIDGE
            h01 = 0.0
            for x, sc, t in pts:
                p = _sigmoid(a + b * x)
                w = t * p * (1 - p) + 1e-9
                g0 += t * p - sc; g1 += (t * p - sc) * x
                h00 += w; h01 += w * x; h11 += w * x * x
            det = h00 * h11 - h01 * h01
            if det <= 1e-12:
                return False
            da, db = (h11 * g0 - h01 * g1) / det, (-h01 * g0 + h00 * g1) / det
            a, b = a - da, b - db
            if abs(da) + abs(db) < 1e-6:
                break
        self.a, self.b = min(3.0, max(-6.0, a)), min(8.0, max(0.1, b))
        self.refits += 1
        return True


class Track:
    """Per-session plan-time estimate of the next block's confidences: an EMA of x per block position over the rounds that drafted
    (the head computes the whole block's confidence whatever depth was verified)."""

    def __init__(self, alpha: float = TRACK_ALPHA):
        self.alpha, self.x = float(alpha), []

    def update(self, conf: list) -> None:
        for i, c in enumerate(conf or ()):
            x = xof(c)
            if i < len(self.x):
                self.x[i] += self.alpha * (x - self.x[i])
            else:
                self.x.append(x)

    @property
    def ready(self) -> bool:
        return bool(self.x)

    def probs(self, cal: Calibration, cap: int) -> list:
        return [cal.prob_x(x) for x in self.x[:cap]]


# ----------------------------------------------------------------------------------------------------------------------- choice
def expected_tokens(probs: list, rows: int) -> float:
    """E[tokens of a round that verifies `rows` rows] = 1 + p_1 + p_1 p_2 + ... (rows - 1 terms)."""
    tot, run = 1.0, 1.0
    for i in range(min(max(0, rows - 1), len(probs))):
        run *= min(1.0, max(0.0, probs[i]))
        tot += run
    return tot


class Choice:
    __slots__ = ("rows", "et", "ms", "gain", "plain_gain")

    def __init__(self, rows, et, ms, gain, plain_gain):
        self.rows, self.et, self.ms, self.gain, self.plain_gain = rows, et, ms, gain, plain_gain

    def as_dict(self) -> dict:
        return {"R": self.rows, "et": round(self.et, 3), "cms": round(self.ms, 2)}


def choose_rows(probs: list, cost: RowCost, plain_ms: float | None = None, max_rows: int | None = None, tie: float = TIE) -> Choice:
    """R in 1..min(max_rows, len(probs) + 1) maximising E[T](R) / cost(R); every R within `tie` of the best gain loses to the smaller R."""
    top = len(probs) + 1 if max_rows is None else max(1, min(int(max_rows), len(probs) + 1))
    gains = []
    for r in range(1, top + 1):
        et, ms = expected_tokens(probs, r), cost.ms(r, plain_ms)
        gains.append((et / ms, r, et, ms))
    best = max(g[0] for g in gains)
    g, r, et, ms = next(x for x in gains if x[0] >= best * (1.0 - tie))        # ascending R: the first within the tie is the cheapest
    return Choice(r, et, ms, g, gains[0][0])


class State:
    """Per-session state: the confidence track, plus what the last decision was (for the round trace)."""

    def __init__(self):
        self.track = Track()
        self.last: dict | None = None
        self.fresh = False            # the track already took this round's confidences (mode 2 reads them before the verify)


# ----------------------------------------------------------------------------------------------------------------------- trace analysis
def calibration_report(records: list, nbins: int = 6) -> dict:
    """Calibration error from round_log records that carry `pk` (the p_k the choice used, one per drafted position of the traced R):
    per position i < len(pk) a trial exists when the round reached it (acc >= i), a success when acc > i. Returns
      pos:   per position {n, pred, real}  (mean predicted p_k vs realised rate over reached rounds)
      bins:  the same by predicted-p bin
      et:    {pred, real}  mean E[T] the policy used vs mean tokens realised, over verify rounds
      ece:   weighted mean |pred - real| over the predicted-p bins (expected calibration error)
    A plain round (pk absent) carries no outcome and is skipped."""
    pos: dict = {}
    bins = [[0, 0.0, 0.0] for _ in range(nbins)]
    et_p = et_r = n_r = 0.0
    for r in records:
        pk = r.get("pk")
        if not pk or r.get("k") != "r" or r.get("lk"):
            continue
        acc = int(r.get("acc") or 0)
        n_r += 1
        et_r += int(r.get("tok") or 0)
        et_p += float(r.get("et") or expected_tokens(pk, len(pk) + 1))
        for i, p in enumerate(pk):
            if acc < i:
                break
            s = 1.0 if acc > i else 0.0
            e = pos.setdefault(i, [0, 0.0, 0.0])
            e[0] += 1; e[1] += p; e[2] += s
            b = bins[min(nbins - 1, int(min(max(p, 0.0), 1.0 - 1e-9) * nbins))]
            b[0] += 1; b[1] += p; b[2] += s
    out_pos = {i: {"n": e[0], "pred": e[1] / e[0], "real": e[2] / e[0]} for i, e in sorted(pos.items())}
    out_bins = [{"n": b[0], "pred": b[1] / b[0], "real": b[2] / b[0]} for b in bins if b[0]]
    tot = sum(b["n"] for b in out_bins)
    ece = sum(b["n"] * abs(b["pred"] - b["real"]) for b in out_bins) / tot if tot else 0.0
    return {"pos": out_pos, "bins": out_bins, "ece": ece,
            "et": {"pred": et_p / n_r if n_r else 0.0, "real": et_r / n_r if n_r else 0.0, "rounds": int(n_r)}}


def fit_prior(samples: list, iters: int = 25) -> tuple:
    """Fit sigmoid(a + b x) by Newton steps on the log-loss. samples: (conf, success) pairs of reached positions, e.g. from a trace's
    `cf` (the block confidences) and `acc`. Returns the PRIOR_A, PRIOR_B default when there are fewer than 20 samples or one class only."""
    pts = [(xof(c), 1.0 if s else 0.0) for c, s in samples]
    if len(pts) < 20 or not 0 < sum(y for _, y in pts) < len(pts):
        return PRIOR_A, PRIOR_B
    a, b = 0.0, 1.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in pts:
            p = _sigmoid(a + b * x)
            w = p * (1 - p) + 1e-9
            g0 += p - y; g1 += (p - y) * x
            h00 += w; h01 += w * x; h11 += w * x * x
        det = h00 * h11 - h01 * h01 + 1e-9                    # tiny ridge: a separable sample does not blow up
        da, db = (h11 * g0 - h01 * g1) / det, (-h01 * g0 + h00 * g1) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-6:
            break
    return a, b


def samples_from_trace(records: list) -> list:
    """(conf, success) pairs of the reached positions of verify rounds that logged `cf` (and `R` rows, so which positions were verified)."""
    out = []
    for r in records:
        cf = r.get("cf")
        if r.get("k") != "r" or not cf:
            continue
        drafted, acc = int(r.get("draft") or 0) - int(r.get("cut") or 0), int(r.get("acc") or 0)
        for i in range(min(drafted, acc + 1, len(cf))):
            out.append((cf[i], i < acc))
    return out
