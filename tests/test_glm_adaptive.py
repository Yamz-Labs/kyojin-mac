"""Flag glm_adaptive (kyojin_mlx.glm_adaptive): the depth controller on a simulated served pass (live ms by rows, scripted acceptance) and the engine wiring on a scripted
backend (no model). No policy ever changes the greedy output: the verify decides every token."""
import random

import pytest

import test_spec_lookup as TSL
from kyojin_mlx import glm_adaptive as GA
from kyojin_mlx import glm_cost as GC
from kyojin_mlx.serve import engine as E

C = GC.Costs()
V = TSL.V


# ------------------------------------------------------------------------------------------------ simulated served passes
def g0_round(n, rows):
    """The served 1-session round (G0 verify wall 32.5 / 39.5 / 51.0 / 62.6 ms for 1-4 rows + commit + draft); more sessions: the G8a table."""
    if n == 1:
        v = {1: 32.5, 2: 39.5, 3: 51.0, 4: 62.6}[rows]
        return 31.1 if rows == 1 else v + C.post(rows) + 4.0 + 2.9 * (rows - 2)
    return C.round_ms([0] * n) if rows == n else C.round_ms([(rows - n) // n + (1 if i < (rows - n) % n else 0) for i in range(n)])


def cheap_round(n, rows, row_ms=4.0):
    """The same pass after the verify rows got cheap (G2c / G3b / G10 landed): a draft row costs row_ms + its head step."""
    return 31.1 if rows == n else 31.1 + (rows - n) * (row_ms + 2.9) + 4.0


class Env:
    def __init__(self, cost, p, seed=1):
        self.cost, self.p, self.rng = cost, p, random.Random(seed)       # p: per-position conditional acceptance, or a function of the round index

    def accept(self, d, rnd):
        p = self.p(rnd) if callable(self.p) else self.p
        a = 0
        while a < d and self.rng.random() < p[min(a, len(p) - 1)]:
            a += 1
        return a


def drive(ctrl, env, rounds, n=1, cap=3, tail=None):
    """choose -> run -> observe, as the engine does; returns [(depth vector, tokens, ms)]."""
    log = []
    views = [GA.SessView(i, cap, []) for i in range(n)]
    for r in range(rounds):
        if tail is not None:
            for v in views:
                v.tail = tail(r)
        ds = ctrl.choose(views)
        rows = n + sum(ds)
        ms = env.cost(n, rows)
        if not any(ds):
            ctrl.observe_plain(n, ms)
            log.append((tuple(ds), n, ms))
            continue
        acc = [env.accept(d, r) for d in ds]
        ctrl.observe(views, ds, ds, acc, ms)
        log.append((tuple(ds), n + sum(acc), ms))
    return log


def rate(log):
    return 1000.0 * sum(t for _, t, _ in log) / sum(m for _, _, m in log)


def fixed_rate(env, d, p, n=1):
    t, c = 1.0, 1.0
    for j in range(d):
        c *= p[min(j, len(p) - 1)]
        t += c
    return 1000.0 * t / env.cost(n, n + d)


CODE, FR, MATH, EDIT = (0.91, 0.66, 0.66), (0.81, 0.26, 0.26), (0.95, 0.79, 0.79), (0.90, 0.55, 0.55)


# ------------------------------------------------------------------------------------------------ the controller
def test_costs_of_today_keep_depth_one_on_every_text():
    for p in (CODE, FR, EDIT):
        ctrl = GA.DepthController(GC.Costs())
        log = drive(ctrl, Env(g0_round, p), 400)
        tail = log[100:]
        assert sum(1 for d, _, _ in tail if d == (1,)) / len(tail) > 0.8, (p, ctrl.describe())
        best = max(fixed_rate(Env(g0_round, p), d, p) for d in (0, 1, 2, 3))
        assert rate(tail) > 0.95 * best                                   # the probes cost under 5 % of the best fixed depth


def test_maths_at_todays_costs_depth_two_is_a_4_percent_prize_the_controller_never_loses():
    """G0's 51.0 ms 3-row verify: 2.70 tokens in 60.7 ms against depth 1's 1.95 in 45.5 (+3.8 %), a gap inside the noise of a 20-round estimate and barely above the 3 %
    hysteresis: the controller may dither between the two, it must not fall under depth 1."""
    env = Env(g0_round, MATH)
    ctrl = GA.DepthController(GC.Costs())
    tail = drive(ctrl, env, 600)[150:]
    assert fixed_rate(env, 2, MATH) > 1.03 * fixed_rate(env, 1, MATH)
    assert rate(tail) > 0.99 * fixed_rate(env, 1, MATH), ctrl.describe()


def test_cheap_verify_rows_move_the_depth_up_and_pay():
    """After the verify rows get cheap, depth 2-3 wins on code and maths and the controller finds it from the live ms alone (no constant changed)."""
    for p in (CODE, MATH):
        env = Env(cheap_round, p)
        ctrl = GA.DepthController(GC.Costs())
        log = drive(ctrl, env, 600)
        tail = log[200:]
        best_d = max((0, 1, 2, 3), key=lambda d: fixed_rate(env, d, p))
        assert best_d >= 2
        assert sum(1 for d, _, _ in tail if d[0] >= 2) / len(tail) > 0.7, (p, ctrl.describe())
        assert rate(tail) > 0.95 * fixed_rate(env, best_d, p)
        assert rate(tail) > 1.04 * fixed_rate(env, 1, p)                  # and it beats the shipped depth 1


def test_prose_stays_shallow_even_when_rows_are_cheap():
    env = Env(cheap_round, FR)
    ctrl = GA.DepthController(GC.Costs())
    log = drive(ctrl, env, 600)
    tail = log[200:]
    best = max(fixed_rate(env, d, FR) for d in (0, 1, 2, 3))
    assert rate(tail) > 0.95 * best


def test_follows_a_change_of_text_and_remembers_the_class():
    """Code -> French prose -> code on one session, cheap rows: the depth follows within ~40 rounds each way."""
    prof = lambda r: CODE if (r < 300 or r >= 600) else FR
    env = Env(cheap_round, prof)
    ctrl = GA.DepthController(GC.Costs())
    log = drive(ctrl, env, 900, tail=lambda r: [1, 2, 3, 4] * 12 if prof(r) is CODE else list(range(60)))
    def share(lo, hi, f):
        seg = log[lo:hi]
        return sum(1 for d, _, _ in seg if f(d[0])) / len(seg)
    assert share(200, 300, lambda d: d >= 2) > 0.7
    assert share(360, 600, lambda d: d <= 1) > 0.7
    assert share(700, 900, lambda d: d >= 2) > 0.7


def test_contention_parks_and_a_probe_comes_back():
    slow = lambda n, rows: g0_round(n, rows) * (1.0 if rows == n else 1.6)       # drafts got expensive: plain wins
    ctrl = GA.DepthController(GC.Costs())
    log = drive(ctrl, Env(slow, CODE), 400)
    tail = log[150:]
    assert sum(1 for d, _, _ in tail if d == (0,)) / len(tail) > 0.8
    assert any(d != (0,) for d, _, _ in tail)                                         # the probes still measure the rival
    assert ctrl.stats["cooldown_up"] > 0
    back = drive(ctrl, Env(g0_round, CODE), 2500)                                       # the load goes: depth 1 again
    assert sum(1 for d, _, _ in back[-100:] if d == (1,)) / 100 > 0.7


def test_probe_cadence_is_bounded():
    ctrl = GA.DepthController(GC.Costs())
    log = drive(ctrl, Env(g0_round, CODE), 1500)
    seg = log[500:]
    env = Env(g0_round, CODE)
    best = max(fixed_rate(env, d, CODE) for d in (0, 1, 2, 3))
    assert rate(seg) > 0.97 * best, (rate(seg), best)                                  # what the probes cost (rivals within 15 % are measured, the plain pass and depth 3 rarely)
    assert sum(1 for d, _, _ in seg if d == (0,)) / len(seg) < 0.05


def test_a_crowd_plans_plain_and_two_sessions_too():
    for n in (2, 4):
        ctrl = GA.DepthController(GC.Costs())
        log = drive(ctrl, Env(g0_round, CODE), 300, n=n)
        tail = log[100:]
        assert sum(1 for d, _, _ in tail if not any(d)) / len(tail) > 0.8, n


def test_caps_and_max_rows_are_respected():
    ctrl = GA.DepthController(GC.Costs(), max_rows=3)
    env = Env(cheap_round, MATH)
    views = [GA.SessView(0, 3), GA.SessView(1, 0)]
    for _ in range(200):
        ds = ctrl.choose(views)
        assert ds[1] == 0 and 2 + sum(ds) <= 3 and ds[0] <= 3
        ctrl.observe(views, ds, ds, [min(d, 1) for d in ds], 40.0 if any(ds) else None)
    assert ctrl.choose([GA.SessView(0, 0)]) == [0]
    assert ctrl.choose([]) == []


def test_unmeasured_keys_take_the_table_shape_at_the_live_level():
    rc = GA.RoundCosts(C)
    assert rc.est(1, 2) == pytest.approx(C.round_ms([1]))
    for _ in range(6):
        rc.observe(1, 2, 0.8 * C.round_ms([1]))
    assert rc.est(1, 2) == pytest.approx(0.8 * C.round_ms([1]), rel=1e-6)
    assert rc.est(1, 3) == pytest.approx(0.8 * C.round_ms([2]), rel=1e-6)           # the live ratio carries to the neighbour keys
    assert rc.est(1, 1) == C.plain(1)                                                 # the plain pass is the table's until a plain pass is measured
    rc.observe_plain(1, 25.0)
    assert rc.est(1, 1) == 25.0


def test_cost_ema_guards_spikes_and_averages_the_first_samples():
    rc = GA.RoundCosts(C)
    for x in (50.0, 40.0, 45.0, 45.0):
        rc.observe(1, 2, x)
    assert rc.t[(1, 2)] == pytest.approx(45.0, abs=1.5)                              # flat mean of the first samples
    base = rc.t[(1, 2)]
    rc.observe(1, 2, 400.0)                                                           # one stall (a prefill chunk, a page-in)
    assert rc.t[(1, 2)] < base + 0.05 * (400.0 - base)


def test_acceptance_is_per_session_and_per_class_with_a_pooled_prior():
    acc = GA.Acceptance()
    for _ in range(30):
        acc.observe("a", "rep", 2, 2)
        acc.observe("a", "var", 2, 0)
    assert acc.p("a", "rep", 0) > 0.9 and acc.p("a", "var", 0) < 0.2                   # one session, two texts
    assert acc.p("b", "rep", 0) > acc.p("b", "var", 0)                                 # a new session starts from its class pool
    assert acc.p("b", "var", 1) <= acc.p("b", "var", 0)                                # a deeper position never above the one before it until measured
    assert acc.trials("b", "rep", 0) == 0


def test_text_class_buckets_and_session_hysteresis():
    assert GA.text_class(list(range(10))) == "all"
    assert GA.text_class([1, 2, 3, 4] * 12) == "rep"
    assert GA.text_class(list(range(60))) == "var"
    acc = GA.Acceptance()
    seq = [1, 2, 3, 4] * 12
    got = [acc.classify("s", seq) for _ in range(GA.CLASS_EVERY * 3)]
    assert got[-1] == "rep" and got[0] == "all"
    assert acc.classify("s", seq, enabled=False) == "all"


def test_a_leaver_changes_the_pass_and_drops_the_probe():
    ctrl = GA.DepthController(GC.Costs())
    drive(ctrl, Env(g0_round, CODE), 60, n=2)
    ds = ctrl.choose([GA.SessView(0, 3)])                                              # session 1 left: another pass, no stale vector of length 2
    assert len(ds) == 1


# ------------------------------------------------------------------------------------------------ the engine on a scripted backend
def nxt(hist):
    return TSL.nxt8(hist)


class Scripted(TSL.FakeLookup):
    """A draft head that is right with a scripted pattern (accepted run lengths 0..d vary by position): the verify rejects the rest, the cache rolls back."""
    kind = "fakeglm"
    glm_costs = GC.Costs()
    max_spec_rows = 16

    def __init__(self, miss_every=3):
        super().__init__()
        self.miss_every, self.depths = miss_every, []

    def draft(self, d, pending, p0, depth, history):
        ids, out = history
        self.depths.append(depth)
        h, res = ids + out, []
        for j in range(depth):
            t = nxt(h)
            if (len(h) + j) % self.miss_every == 0 and j >= (len(h) % 2):              # deterministic misses at varying positions
                t = (t + 1) % V
            res.append(t)
            h = h + [t]
        return res

    def spec_verify_bc(self, bc, token_lists):
        self.passes.append(sum(len(t) for t in token_lists))
        rows, starts = [], []
        for cache, toks in zip(bc, token_lists):
            starts.append(len(rows))
            rows += [nxt(cache["h"] + toks[:j + 1]) for j in range(len(toks))]
        return TSL.SpecPass(TSL.onehot(rows), starts, bc, token_lists)


def run_engine(monkeypatch, flag, sessions=1, n=40, depth=3, override=None, chaos=None, backend=None):
    monkeypatch.setenv("KYOJIN_GLM_ADAPTIVE", flag)
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    if chaos is not None:
        monkeypatch.setattr(GA.DepthController, "choose", chaos)
    reqs = {f"r{i}": ([80 + i, 5, 9, 11, 13 + i], {"max_new": n}) for i in range(sessions)}
    kw = {} if override is None else {"depth_override": override}
    return TSL.run(reqs, backend=backend or Scripted(), sessions=sessions, mtp_depth=depth, **kw)


def toks(out):
    return {k: (v[0], v[1]) for k, v in out.items()}


def test_adaptive_keeps_the_greedy_output(monkeypatch):
    base, _, _ = run_engine(monkeypatch, "0", override=0)
    for ov in (1, 2, 3):
        out, _, _ = run_engine(monkeypatch, "0", override=ov)
        assert toks(out) == toks(base)
    ad, st, _ = run_engine(monkeypatch, "1")
    assert toks(ad) == toks(base)
    assert st["spec_ad_rounds"] > 0 and st["spec_ad_obs"] > 0


def test_any_depth_vector_every_round_keeps_the_greedy_output(monkeypatch):
    """The rollback under a depth that changes every round, per session (1, 2 and 3 sessions): random vectors, scripted misses: same tokens as the plain run."""
    for sessions in (1, 2, 3):
        base, _, _ = run_engine(monkeypatch, "0", sessions=sessions, override=0)
        rng = random.Random(7 + sessions)

        def chaos(self, views, rng=rng):
            self.rounds += 1
            ds = [rng.randint(0, v.cap) for v in views]
            while self.max_rows and len(ds) + sum(ds) > self.max_rows:
                ds[ds.index(max(ds))] -= 1
            return ds
        out, st, be = run_engine(monkeypatch, "1", sessions=sessions, chaos=chaos)
        assert toks(out) == toks(base), sessions
        assert st["spec_ad_d0"] > 0 and st["spec_ad_d1"] > 0 and st["spec_ad_d3"] > 0     # the vectors really varied
        assert st["spec_accepted"] < st["spec_drafted"]                                   # and the head really missed (rollback ran)


def test_a_fixed_depth_wins_over_the_flag(monkeypatch):
    out, st, _ = run_engine(monkeypatch, "1", override=2)
    assert st.get("spec_ad_rounds", 0) == 0


def test_flag_off_is_the_old_path(monkeypatch):
    out, st, _ = run_engine(monkeypatch, "0")
    assert st["spec_ad_rounds"] == 0 and st["spec_ad_obs"] == 0


def test_sampled_session_runs_under_the_flag(monkeypatch):
    out, st, _ = run_engine(monkeypatch, "1", sessions=2)
    assert all(len(v[0]) > 0 for v in out.values())
