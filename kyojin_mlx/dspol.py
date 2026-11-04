# SPDX-License-Identifier: Apache-2.0
"""Adaptive DSpark verify-width policy (rival lever #4).

Sushi picks the verify width R = 1..6 per cycle from measured cost priors
(`chooseRows`, src/dflash_policy.zig:61: 28 ms plain + 9 ms per step);
oMLX sets depth = min(max, accepted + 1) and primes the drafter ring with the
prompt taps. This module is the generic (any MTP/DSpark model) version of
that idea for the serving engine:

* cost model:  cost(R) = S + slope * (R - 1) + draft   (ms, R = verify rows
  = D drafts + 1; the one draft call per round costs `draft`, 0 when D = 0).
  Read from a small measured table (env/JSON, see `CostModel.load`), with
  defaults from mac_results.log: plain 44-46 tok/s -> S = 22.0 ms; ours
  0.40-0.49 S per extra row, Sushi prior 9 ms/step -> slope = 9.5 ms;
  draft 5.5 ms measured (line 14) at S = 22 -> draft = 3.3 ms here (the
  engine default draft_cost 0.15 in S units; the 5.5 ms number carries the
  CPU-serial part of that box).
* expected tokens from the confidence head: per-position acceptance probs
  p_i, E[T|D] = 1 + p_1 + p_1 p_2 + ... ; the width is argmax_D E[T]/cost.
  The probs come from a `Predictor`: per-position empirical acceptance
  shrunk toward the confidence head's calibrated value (a position never
  drafted inherits its predecessor, as in Engine._plan_measured, but through
  the calibrated confidence instead of the pooled geometric prior).

The policy only steers the planned depth D. Speculative decode stays exact
for any D (the trunk verifies), so the output is token-identical to plain
greedy whatever the policy picks; tests/test_dspol.py checks that.

Round anatomy this is tuned against (mac_results.log:14, served fixed d2,
230 rounds): verify 49 ms/round (GPU ~35 + CPU build ~12-14 serial) +
draft 5.5 + commit 1 at 2.4 tok/round = 43 tok/s; the row model held
(measured 1.46/1.72/2.07 solo steps at 2/3/4 rows, dsgap_REPORT.md).
"""
from __future__ import annotations

import json
import os

# Defaults from mac_results.log (see module doc).
DEFAULT_S_MS = 22.0        # plain solo step: 44-46 tok/s gated, "plain_ms now 22 ms" (line 12)
DEFAULT_SLOPE_MS = 9.5     # one more verify row: ours 0.40-0.49 S, Sushi prior 9 ms/step
DEFAULT_DRAFT_MS = 3.3     # one draft call per drafted round: engine default draft_cost 0.15 S

ENV_COSTS = "KYOJIN_DSPOL_COSTS"     # JSON file, inline JSON, or "S_ms,slope_ms[,draft_ms]"
ENV_S, ENV_SLOPE, ENV_DRAFT = "KYOJIN_DSPOL_S", "KYOJIN_DSPOL_SLOPE", "KYOJIN_DSPOL_DRAFT"
ENV_ENABLE = "KYOJIN_DSPOL"          # "0" turns the engine integration off


def explicit() -> bool:
    """True when the operator set a cost table (any of the env knobs): it wins over the engine's row_cost/draft_cost."""
    return any(os.environ.get(k) for k in (ENV_COSTS, ENV_S, ENV_SLOPE, ENV_DRAFT))


class CostModel:
    """cost(D drafts) = S + slope * D + draft (draft only when D > 0), in ms."""

    def __init__(self, s_ms: float = DEFAULT_S_MS, slope_ms: float = DEFAULT_SLOPE_MS,
                 draft_ms: float = DEFAULT_DRAFT_MS):
        self.s_ms, self.slope_ms, self.draft_ms = float(s_ms), float(slope_ms), float(draft_ms)

    def cost_ms(self, d: int) -> float:
        """Ms of a round that verifies D drafts (D + 1 rows)."""
        return self.s_ms + self.slope_ms * d + (self.draft_ms if d > 0 else 0.0)

    @staticmethod
    def from_rows(rows_ms: list) -> "CostModel":
        """Fit S + slope from a measured per-row table [ms at 1..R rows] (draft folded into the slope)."""
        r = [float(x) for x in rows_ms]
        assert len(r) >= 1 and r[0] > 0
        slope = sum(b - a for a, b in zip(r, r[1:])) / max(1, len(r) - 1) if len(r) > 1 else DEFAULT_SLOPE_MS
        return CostModel(r[0], max(0.0, slope), 0.0)

    @staticmethod
    def load() -> "CostModel":
        """Cost model from env/JSON, else the mac_results.log defaults. Never raises."""
        s, slope, draft = DEFAULT_S_MS, DEFAULT_SLOPE_MS, DEFAULT_DRAFT_MS
        try:
            v = os.environ.get(ENV_COSTS, "").strip()
            if v.startswith("{"):
                d = json.loads(v)
                s, slope, draft = float(d.get("S_ms", s)), float(d.get("slope_ms", slope)), float(d.get("draft_ms", draft))
            elif "," in v:
                parts = [float(x) for x in v.split(",")]
                s, slope = parts[0], parts[1]
                if len(parts) > 2:
                    draft = parts[2]
            elif v:
                d = json.loads(open(v).read())
                if "rows_ms" in d:
                    return CostModel.from_rows(d["rows_ms"])
                s, slope, draft = float(d.get("S_ms", s)), float(d.get("slope_ms", slope)), float(d.get("draft_ms", draft))
            if os.environ.get(ENV_S):
                s = float(os.environ[ENV_S])
            if os.environ.get(ENV_SLOPE):
                slope = float(os.environ[ENV_SLOPE])
            if os.environ.get(ENV_DRAFT):
                draft = float(os.environ[ENV_DRAFT])
        except Exception:
            pass
        return CostModel(s, slope, draft)


def expected_tokens(probs: list, d: int) -> float:
    """E[tokens of a round that verifies D drafts]: 1 + p_1 + p_1 p_2 + ... (d terms)."""
    tot, run = 1.0, 1.0
    for i in range(min(d, len(probs))):
        run *= min(1.0, max(0.0, probs[i]))
        tot += run
    return tot


def choose_width(probs: list, cost: CostModel, cap: int) -> int:
    """Drafts D in 0..cap maximizing expected tokens per ms. Ties go to the smaller D."""
    best, best_g = 0, 1.0 / cost.cost_ms(0)
    for d in range(1, cap + 1):
        g = expected_tokens(probs, d) / cost.cost_ms(d)
        if g > best_g:
            best, best_g = d, g
    return best


class Calibrator:
    """Maps the confidence head's value to measured acceptance: decayed
    per-bin P(accept | conf), shrunk toward the pooled rate. The confidence
    head is a per-position proposal score, so a bin trial is a verified
    position: reached (i <= accepted) = trial, kept (i < accepted) = success."""

    def __init__(self, nbins: int = 10, prior_trials: float = 2.0):
        self.nbins, self.prior_trials = nbins, prior_trials
        self.succ = [0.0] * nbins
        self.trial = [0.0] * nbins
        self.all_succ = 0.0
        self.all_trial = 0.0

    def _bin(self, c: float) -> int:
        return min(self.nbins - 1, max(0, int(c * self.nbins)))

    def observe(self, conf: list, drafted: int, accepted: int) -> None:
        reached = min(drafted, accepted + 1, len(conf or ()))
        for i in range(reached):
            b = self._bin(float(conf[i]))
            self.trial[b] = 0.9 * self.trial[b] + 1
            self.succ[b] = 0.9 * self.succ[b] + (1 if i < accepted else 0)
            self.all_trial = 0.9 * self.all_trial + 1
            self.all_succ = 0.9 * self.all_succ + (1 if i < accepted else 0)

    @property
    def pooled(self) -> float:
        return (self.all_succ + 0.7 * 2) / (self.all_trial + 2)

    @property
    def ready(self) -> bool:
        return self.all_trial > 0

    def prob(self, c: float) -> float:
        """Calibrated P(accept | confidence c); the pooled rate when the bin is empty."""
        b = self._bin(float(c))
        return (self.succ[b] + self.pooled * self.prior_trials) / (self.trial[b] + self.prior_trials)


class Predictor:
    """Per-session acceptance state: per-position empirical rates plus the
    mean confidence seen at each position, mapped through the Calibrator.
    predict(cap) returns one acceptance prob per draft position: measured
    positions shrink their empirical rate toward their calibrated confidence,
    unmeasured ones inherit the predecessor (streaky text: a draft that lands
    3 tokens tends to land the 4th), the first one the pooled rate."""

    SHRINK = 2.0            # trials' worth of the calibrated value, as Engine._plan_depth's prior 0.7 x 2

    def __init__(self, cal: Calibrator | None = None):
        self.cal = cal if cal is not None else Calibrator()      # the engine passes its shared one: confidence -> acceptance is a model property
        self.pos_succ: list = []
        self.pos_trial: list = []
        self.pos_conf: list = []      # decayed mean confidence of verified positions
        self.pos_confn: list = []

    def observe(self, conf: list | None, drafted: int, accepted: int) -> None:
        reached = min(drafted, accepted + 1)
        while len(self.pos_trial) < reached:
            self.pos_succ.append(0.0)
            self.pos_trial.append(0.0)
            self.pos_conf.append(0.0)
            self.pos_confn.append(0.0)
        for i in range(reached):
            self.pos_trial[i] = 0.9 * self.pos_trial[i] + 1
            self.pos_succ[i] = 0.9 * self.pos_succ[i] + (1 if i < accepted else 0)
            if conf is not None and i < len(conf):
                self.pos_confn[i] = 0.9 * self.pos_confn[i] + 1
                self.pos_conf[i] = 0.9 * self.pos_conf[i] + float(conf[i])
        if conf is not None:
            self.cal.observe(list(conf), drafted, accepted)

    @property
    def ready(self) -> bool:
        return self.cal.ready

    def predict(self, cap: int) -> list:
        out, prior = [], self.cal.pooled
        for i in range(cap):
            c = self.pos_conf[i] / self.pos_confn[i] if i < len(self.pos_confn) and self.pos_confn[i] > 0 else None
            anchor = self.cal.prob(c) if c is not None else prior
            if i < len(self.pos_trial) and self.pos_trial[i] > 0:
                out.append((self.pos_succ[i] + self.SHRINK * anchor) / (self.pos_trial[i] + self.SHRINK))
            else:
                out.append(anchor)
            prior = out[-1]
        return out
