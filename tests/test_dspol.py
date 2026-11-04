# SPDX-License-Identifier: Apache-2.0
"""Adaptive DSpark width policy (kyojin_mlx.dspol): cost model loading, width
choice, confidence calibration, and engine exactness under the policy on a
tiny random model (a seeded fake backend with random drafts and acceptance-
correlated confidences). The policy only steers the planned depth, so greedy
output must equal plain greedy whatever it picks."""
import random

import pytest

from kyojin_mlx import dspol as DSPOL
from kyojin_mlx.serve.engine import Engine
from test_serve_engine import Fake, V, collect, nxt
from test_serve_spec import run, solo_all


# ------------------------------------------------------------------------------------------------ pure policy
def test_cost_defaults_from_mac_results():
    c = DSPOL.CostModel.load()
    assert (c.s_ms, c.slope_ms, c.draft_ms) == (22.0, 9.5, 3.3)
    assert c.cost_ms(0) == 22.0
    assert c.cost_ms(3) == pytest.approx(22.0 + 3 * 9.5 + 3.3)      # S + slope*(R-1) + draft, R = D+1


def test_cost_from_rows_table():
    c = DSPOL.CostModel.from_rows([20.0, 29.5, 39.0])                # slope 9.5 fitted
    assert (c.s_ms, c.slope_ms, c.draft_ms) == (20.0, 9.5, 0.0)


def test_cost_env_inline_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv(DSPOL.ENV_COSTS, '{"S_ms": 18.0, "slope_ms": 4.5}')
    assert DSPOL.CostModel.load().cost_ms(2) == pytest.approx(18.0 + 9.0 + 3.3)
    f = tmp_path / "costs.json"
    f.write_text('{"rows_ms": [20.0, 24.0, 28.0]}')
    monkeypatch.setenv(DSPOL.ENV_COSTS, str(f))
    assert DSPOL.CostModel.load().cost_ms(2) == pytest.approx(28.0)
    monkeypatch.setenv(DSPOL.ENV_COSTS, "18.0,4.5,1.0")
    assert DSPOL.CostModel.load().cost_ms(1) == pytest.approx(23.5)
    monkeypatch.delenv(DSPOL.ENV_COSTS)
    monkeypatch.setenv(DSPOL.ENV_SLOPE, "5.0")                       # operator override wins over the file/defaults
    assert DSPOL.CostModel.load().slope_ms == 5.0


def test_choose_width_known_points():
    cost = DSPOL.CostModel(22.0, 9.5, 3.3)
    assert DSPOL.choose_width([0.9] * 5, cost, 5) == 4               # high acceptance: deep but not max
    assert DSPOL.choose_width([0.97] * 5, cost, 5) == 5              # code-like: the whole block
    assert DSPOL.choose_width([0.2] * 5, cost, 5) == 0               # hopeless: plain steps
    assert DSPOL.choose_width([0.68] * 5, cost, 5) == 2              # prose-like: short drafts (48.4 vs 48.3 tok/s at D = 1)
    assert DSPOL.choose_width([0.68] * 5, cost, 3) <= 3
    assert DSPOL.choose_width([], cost, 5) == 0


def test_choose_width_ties_go_shallow():
    cost = DSPOL.CostModel(10.0, 0.0, 0.0)                           # free rows and hopeless drafts: every depth gains 1 token
    assert DSPOL.choose_width([0.0] * 5, cost, 5) == 0


def test_calibrator_learns_confidence():
    cal = DSPOL.Calibrator()
    assert not cal.ready
    assert cal.prob(0.99) == pytest.approx(cal.pooled)               # no data: the pooled rate
    for _ in range(30):
        cal.observe([0.95, 0.9, 0.85, 0.8, 0.75], 5, 5)              # confident rounds land
        cal.observe([0.1, 0.15, 0.1, 0.2, 0.1], 5, 0)                # diffident rounds miss
    assert cal.ready
    assert cal.prob(0.95) > 0.8 > cal.prob(0.1)
    assert cal.prob(0.5) == pytest.approx(cal.pooled)                # an unseen bin: the pooled rate


def test_predictor_anchors_unmeasured_positions_in_confidence():
    pol = DSPOL.Predictor()
    for _ in range(20):                                              # only position 0 ever drafted, always lands, high conf
        pol.observe([0.9, 0.8, 0.7], 1, 1)
    assert pol.ready
    p = pol.predict(4)
    assert p[0] > 0.8                                                # measured: empirical, shrunk to the calibrated conf
    assert p[1] > 0.5                                                # never drafted: inherits through the calibrated confidence,
    assert p[1] <= p[0]                                              # not through the pessimistic pooled geometric prior


# ------------------------------------------------------------------------------------------------ tiny random model + engine
class RandSpec(Fake):
    """Fake with a draft head: per-position acceptance ~ p_i (seeded), confidences correlated with acceptance
    (high when the draft lands). supports_spec, so the engine runs the full draft/verify/commit path."""

    kind = "randspec"
    supports_spec = True
    max_spec_rows = 8

    def __init__(self, seed=0, p=(0.9, 0.75, 0.6, 0.45, 0.3), conf_lo=0.4, conf_hi=0.6):
        self.rng = random.Random(seed)
        self.p, self.conf_lo, self.conf_hi = p, conf_lo, conf_hi
        self.depth_asked, self.fed_total = [], 0
        self._confs = None

    def new_draft(self):
        return {"fed": 0}

    def draft_snapshot(self, d):
        return dict(d)

    def draft_restore(self, snap):
        return dict(snap)

    def draft_nbytes(self, d):
        return 8

    def draft_arrays(self, d):
        return []

    def prefill_chunk(self, cache, ids, draft=None):
        if draft is not None:                                        # priming: every prompt chunk feeds the draft ring
            draft["fed"] += len(ids)
            self.fed_total += len(ids)
        return super().prefill_chunk(cache, ids)

    def draft_many(self, jobs):
        out, confs = [], []
        for d, pending, p0, depth, history in jobs:
            self.depth_asked.append(depth)
            ids, oo = history
            assert oo[-1] == pending and p0 == len(ids) + len(oo) - 1
            h, dr, cf = list(ids) + list(oo), [], []
            for i in range(depth):
                t = nxt(h)
                accept = self.rng.random() < self.p[min(i, len(self.p) - 1)]
                c = self.rng.uniform(self.conf_hi, 1.0) if accept else self.rng.uniform(0.0, self.conf_lo)
                dr.append(t if accept else (t + 1 + self.rng.randrange(V - 1)) % V)
                cf.append(c)
                h = h + [dr[-1]]
            out.append(dr)
            confs.append(cf)
        self._confs = confs
        return out

    def draft_confs(self):
        return self._confs

    def spec_verify_bc(self, bc, token_lists):
        self.passes = getattr(self, "passes", []) + [sum(len(t) for t in token_lists)]
        from test_serve_spec import SpecPass, onehot
        rows, starts = [], []
        for cache, toks in zip(bc, token_lists):
            starts.append(len(rows))
            rows += [nxt(cache["h"] + toks[:j + 1]) for j in range(len(toks))]
        return SpecPass(onehot(rows), starts, bc, token_lists)


def reqs():
    return {f"r{i}": ([(i * 7 + j * 3) % V for j in range(6 + 4 * i)], dict(max_new=30 + 9 * i)) for i in range(4)}


@pytest.mark.parametrize("dspol", ["1", "0"])
def test_policy_matches_plain_greedy_on_random_model(dspol, monkeypatch):
    monkeypatch.setenv(DSPOL.ENV_ENABLE, dspol)
    rr = reqs()
    want = solo_all(rr)
    be = RandSpec(seed=7)
    got, st, _ = run(rr, be, sessions=1, mtp_depth=5, prefix_bytes=0)
    for n in rr:
        assert got[n][0] == want[n][0] and got[n][1] == want[n][1] == "length", n
    assert st["spec_accepted"] > 0
    assert len(set(be.depth_asked)) >= 3                             # the policy actually varies the width
    assert be.fed_total == sum(len(ids) for ids, _ in rr.values())   # every prompt token fed the draft ring: primed


def test_dspol_drafts_deep_on_streaky_text(monkeypatch):
    """Near-certain drafts (streaky text): the policy must draft at the cap and accept almost everything,
    still token-identical to plain greedy."""
    monkeypatch.setenv(DSPOL.ENV_ENABLE, "1")
    be = RandSpec(seed=3, p=(0.99, 0.99, 0.99, 0.99, 0.99), conf_lo=0.05, conf_hi=0.95)
    rr = {"a": ([1, 2, 3], dict(max_new=120))}
    got, st, _ = run(rr, be, sessions=1, mtp_depth=5, prefix_bytes=0)
    assert got["a"][0] == solo_all(rr)["a"][0]
    assert st["spec_accepted"] / max(st["spec_drafted"], 1) > 0.9
    assert 5 in be.depth_asked                                        # confident rounds draft at the cap
