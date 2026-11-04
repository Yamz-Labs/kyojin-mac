"""Verify-width cost ladder (flag spec_ladder) on synthetic timings and a fake backend: ladder fit / argmax from
measured depths, never pricing a width without a measurement or a >= 3-point fit, probe widening + plain-round
debt after a loss, the plain fallback, recovery after the acceptance profile improves (no D = 0 latch), the shared
row cap and the session room, greedy output equal to serial decode, and the lookup/ablit round path. No model."""
import pytest

from kyojin_mlx import ladder as LADDER
from kyojin_mlx.ladder import Ladder
from kyojin_mlx.serve.engine import Engine, Session
from kyojin_mlx.serve.sampling import Sampling
from test_serve_spec import FakeSpec, prompts, run, solo_all
from test_spec_lookup import FakeLookup, repeated_prompt, run as run_lookup

PLAIN_MS = 25.0
P_GOOD = [0.9] * 8


class TimedPlain:
    """The real serving backend's plain tick (step_feed), which measures plain_ms and runs the ladder's plain fallback; the
    bare fakes only have the eager verify pass."""

    def step_feed(self, bc, tokens, drafts, p0s):
        return self.step(bc, tokens)


class LadderSpec(TimedPlain, FakeSpec):
    pass


class LadderLookup(TimedPlain, FakeLookup):
    pass


def seeded(costs: dict) -> Ladder:
    lad = Ladder()
    lad.ms = {d: [float(ms)] for d, ms in costs.items()}
    return lad


# ------------------------------------------------------------------------------------------------ ladder arithmetic
def test_widths_are_interpolated_between_measured_depths_never_extrapolated():
    """(a) A line through the shallow depths 1-3 (2-4 rows, the fused GEMV band) must not price the deep widths (5+ rows: MMA / NAX / row-group
    kernels, longer draft chains): a width is priced from its own rounds or from the line between the two measured depths around it."""
    lad = seeded({1: 26.0, 2: 34.0, 4: 60.0})
    s, slope = lad.fit()                                              # the stats line is still there
    assert slope > 0 and s >= 0
    assert lad.cost(3) == pytest.approx((34.0 + 60.0) / 2)            # between 2 and 4
    assert lad.cost(5) is None and lad.cost(8) is None                # past the deepest measured depth: only probed
    assert lad.cost(0) is None
    two = seeded({1: 26.0, 2: 34.0})
    assert two.fit() is None and two.cost(3) is None
    assert two.priced(1) and two.priced(2) and not two.priced(3)


def test_deep_widths_are_reached_only_by_the_probe():
    lad = seeded({1: 26.0, 2: 34.0, 3: 42.0})
    d = lad.plan(8, P_GOOD, PLAIN_MS)
    assert d <= 3                                                     # the fit through 1-3 would have said 8
    lad.since_probe = LADDER.PROBE_PERIOD_MIN - 1
    d = lad.plan(8, P_GOOD, PLAIN_MS)
    assert d == 4 and lad._planned == (4, True)                       # one row past the deepest priced width
    assert lad.cost(4) is None


def test_argmax_expected_tokens_per_ms():
    lad = seeded({1: 22.0, 2: 26.0, 3: 30.0, 4: 34.0})               # 4 ms per row, measured 1..4
    assert Ladder.tokens(P_GOOD, 3) == pytest.approx(1 + 0.9 + 0.81 + 0.729)
    d = lad.plan(8, P_GOOD, PLAIN_MS)
    assert d == 4                                                    # the deepest measured width; 5..8 are not priced
    assert lad.cost(d) is not None


def test_never_prices_an_unmeasured_width_and_probes_to_measure_it():
    lad = seeded({1: 26.0, 2: 34.0})                                  # no fit: only 1 and 2 are priced
    assert lad.cost(3) is None
    d = lad.plan(8, P_GOOD, PLAIN_MS)                                 # cold exploration: widen one row past the best
    assert d == 3 and lad._planned == (3, True)
    assert lad.priced(1) and lad.priced(2) and not lad.priced(3)


def test_probe_cadence_and_plain_debt_after_a_loss():
    lad = seeded({1: 22.0, 2: 26.0, 3: 30.0, 4: 34.0})
    lad.since_probe = LADDER.PROBE_PERIOD_MIN - 1
    d = lad.plan(8, P_GOOD, PLAIN_MS)
    assert d == 5 and lad._planned == (5, True)                       # one row past the best (4), at the low rate
    assert lad.probes == 1
    assert lad.observe(d, 400.0, Ladder.tokens(P_GOOD, d), PLAIN_MS) is False   # the probe loses badly
    assert lad.owed == 1 and lad.period == 2                          # the next losing round owes a longer stretch
    assert lad.plan(8, P_GOOD, PLAIN_MS) == 0                         # debt: one plain round
    assert lad.skips == 1
    d = lad.plan(8, P_GOOD, PLAIN_MS)
    assert d > 0 and lad.priced(d)                                    # debt paid: a priced width again
    assert lad.observe(d, 30.0, Ladder.tokens(P_GOOD, d), PLAIN_MS) is True     # a win clears the debt
    assert lad.owed == 0 and lad.period == 1


def test_plain_fallback_when_the_priced_best_loses_to_plain():
    lad = seeded({1: 50.0, 2: 60.0, 3: 70.0, 4: 80.0})
    assert lad.plan(4, [0.4] * 4, PLAIN_MS) == 0                      # 1.4 tok / 50 ms < 1 tok / 25 ms
    lad.owed = 0
    assert lad.cost(4) is not None


def test_latch_case_returns_to_depth_after_the_profile_improves():
    """The comment at engine.py:361: a policy stuck at D = 0 after a bad stretch. The ladder's debt is bounded by
    MAX_PERIOD, so a better acceptance profile is drafted again within MAX_PERIOD rounds."""
    lad = seeded({1: 26.0, 2: 34.0, 3: 42.0, 4: 50.0})
    bad = [0.2] * 8
    for _ in range(4000):                                             # a long losing streak: period saturates at 32
        d = lad.plan(8, bad, PLAIN_MS)
        if d:
            lad.observe(d, 100.0, Ladder.tokens(bad, d), PLAIN_MS)
        if lad.period == LADDER.MAX_PERIOD and lad.owed == LADDER.MAX_PERIOD:
            break
    assert lad.period == LADDER.MAX_PERIOD and lad.owed == LADDER.MAX_PERIOD
    zeros = 0
    while zeros <= LADDER.MAX_PERIOD:
        d = lad.plan(8, P_GOOD, PLAIN_MS)                             # the text changed: acceptance is high now
        if d > 0:
            break
        zeros += 1
    assert d > 0 and zeros <= LADDER.MAX_PERIOD, (d, zeros)
    assert lad.observe(d, 30.0, Ladder.tokens(P_GOOD, d), PLAIN_MS) is True


# ------------------------------------------------------------------------------------------------ engine wiring
def session(max_new: int = 1 << 20) -> Session:
    return Session(sid=0, ids=[1, 2, 3], max_new=max_new, sampling=Sampling(0.0, 1.0, 0, 0.0, 0),
                   stop_ids=frozenset(), handle=None)


def ladder_engine(monkeypatch, sessions: int = 1, rows: int = 8, depth: int = 8) -> Engine:
    monkeypatch.setenv(LADDER.ENV, "1")
    eng = Engine(FakeSpec(err=3, rows=rows), sessions=sessions, ctx=1 << 20, chunk=4, block=8, mtp_depth=depth)
    eng.plain_ms = PLAIN_MS
    eng._ladder.ms = {d: [20.0 + 8.0 * d] for d in range(1, 9)}       # a priced ladder at every width
    eng.members = [session()]
    return eng


def test_row_cap_is_shared_by_the_sessions_of_the_pass(monkeypatch):
    eng = ladder_engine(monkeypatch, sessions=2, rows=8, depth=8)
    eng.members = [session(), session()]
    depths = eng._ladder_depths(eng.members)
    assert len(depths) == 2 and max(depths) <= 8 // 2 - 1 == 3        # two sessions: one row each + drafts <= 8
    eng.members = [session()]
    assert max(eng._ladder_depths(eng.members)) <= 8 - 1              # solo: the backend cap still applies
    assert max(eng._ladder_depths(eng.members)) <= LADDER.MAX_ROWS


def test_room_clamps_the_chosen_width(monkeypatch):
    eng = ladder_engine(monkeypatch)
    s = session()
    s.n_gen, s.max_new = 1 << 20, (1 << 20) + 4                       # 3 tokens of reply left
    eng.members = [s]
    assert max(eng._ladder_depths(eng.members)) <= 3
    s.n_gen = s.max_new - 1
    assert eng._ladder_depths(eng.members) == [0]


def test_cold_engine_measures_plain_then_drafts(monkeypatch):
    eng = ladder_engine(monkeypatch)
    eng.plain_ms = None
    eng.members = [session()]
    assert eng._ladder_depths(eng.members)[0] == 1                    # cold: a one-draft round (the cap of the first rounds is unmeasured: it was the 100-340 ms rounds)
    assert eng._ladder_depths(eng.members)[0] == 1
    assert eng._ladder_depths(eng.members)[0] == 0                    # then plain ticks until a plain rate exists
    eng.plain_ms = PLAIN_MS
    assert eng._ladder_depths(eng.members)[0] > 0


def test_greedy_output_equals_serial_decode(monkeypatch):
    reqs = prompts(n=3)
    want = solo_all(reqs)                                             # mtp_depth 0: serial plain decode
    monkeypatch.setenv(LADDER.ENV, "1")
    got, st, be = run(reqs, LadderSpec(err=3), mtp_depth=6, prefix_bytes=1 << 20)
    for n in reqs:
        assert got[n][0] == want[n][0] and got[n][1] == want[n][1] == "length", n
    assert st["spec_ladder"]["depths"]                                # the ladder measured its rounds
    assert max(d + 1 for d in be.depth_asked) <= be.max_spec_rows


def test_lookup_rounds_keep_the_row_cap_and_the_ablit_hook(monkeypatch):
    monkeypatch.setenv(LADDER.ENV, "1")
    ids = repeated_prompt()
    want, _, _ = run_lookup({"a": (ids, dict(max_new=12))}, LadderLookup(), sessions=1, mtp_depth=0, prefix_bytes=0)
    got, st, be = run_lookup({"a": (ids, dict(max_new=12))}, LadderLookup(), sessions=1, mtp_depth=4, prefix_bytes=0)
    assert got["a"][0] == want["a"][0]
    assert be.hooks > 0                                               # the verify pass ran through the ablit hook
    assert max(be.passes) <= be.max_spec_rows                         # lookup rows share the same cap


# ------------------------------------------------------------------------------------------------ several sessions (the served lane)
def test_tokens_of_a_pass_scale_with_its_sessions():
    """A round is priced in ms for the whole pass, so its expected tokens are the n sessions' tokens: at n = 4 a width whose single-session
    rate loses to plain can win when the plain pass is priced per session (plain pass / n)."""
    costs = {1: 60.0, 2: 64.0, 3: 68.0, 4: 72.0}
    acc = [0.6] * 4
    plain_pass_ms = 100.0                                              # 4 sessions: 25 ms per token
    solo = seeded(costs)
    assert solo.plan(4, acc, plain_pass_ms / 4, n=1) == 0              # mis-scaled (the old call): drafting looks 4 x too poor
    pooled = seeded(costs)
    assert pooled.plan(4, acc, plain_pass_ms / 4, n=4) > 0


def test_leaving_plain_pays_for_the_step_in_flight():
    """plain -> drafted drains the pipelined plain step: a width that beats plain by less than LEAVE_PLAIN stays plain; once drafting, it stays."""
    lad = seeded({1: 40.0, 2: 41.8})
    acc = [0.5, 0.5]
    gain = Ladder.tokens(acc, 2) / 41.8 * PLAIN_MS                    # rate over plain at width 2
    assert 1.0 + LADDER.ROW_TIE < gain < 1.0 + LADDER.ROW_TIE + LADDER.LEAVE_PLAIN, gain
    lad.since_probe = -10 ** 9                                         # no probe in this test
    assert lad.plan(2, acc, PLAIN_MS) == 0 and lad.last == 0
    lad.last = 2                                                       # already drafting: only the ordinary tie margin applies
    assert lad.plan(2, acc, PLAIN_MS) == 2


def test_first_round_at_a_width_is_not_a_cost_sample():
    """The first round the engine runs at a new verify width builds that width's graphs (0.1-0.6 s at a kernel the load did not warm): one such
    sample in a one-sample median prices the width as unaffordable for the rest of the reply. The engine's `_learn_k` skips the first round at a
    depth for that reason; the ladder must too, and the width stays unpriced so the exploration comes back for a warm sample."""
    lad = Ladder()
    lad.observe(3, 700.0, 3.4, PLAIN_MS, PLAIN_MS)                    # a compile stall, not the width's cost
    assert lad.cost(3) is None and not lad.priced(3)
    lad.observe(3, 41.0, 3.4, PLAIN_MS, PLAIN_MS)                     # the warm round prices it
    assert lad.cost(3) == pytest.approx(41.0)


def test_exploration_measures_every_width_because_depth_one_can_lose():
    """The old loop probed `best + 1`: while depth 1 lost to plain, best stayed 0, so it re-probed depth 1 for ever and depths 2..cap were
    never measured (the ladder could not see a width that pays). Exploration now walks the shallowest width with no warm sample."""
    lad = Ladder()
    cost = {1: 40.0, 2: 36.0, 3: 34.0}                                # depth 1 loses to plain (25 ms), depths 2-3 win
    asked = []
    for _ in range(10):
        d = lad.plan(3, P_GOOD, PLAIN_MS)
        asked.append(d)
        if d > 0:                                                     # the debt may return a plain round in between
            lad.observe(d, cost[d], Ladder.tokens(P_GOOD, d), PLAIN_MS, PLAIN_MS)
    assert {1, 2, 3} <= set(asked), asked                             # every width was run, not depth 1 ten times
    assert set(lad.ms) == {1, 2, 3} and lad.priced(3)
    assert lad.plan(3, P_GOOD, PLAIN_MS) == 3                         # and the deep width that pays is the choice


def test_drafting_does_not_return_to_plain_on_a_rate_dip():
    """Leaving the pipeline costs the queued plain step plus the draft head's deferred feeds, so a noisy rate estimate must not send one plain
    round between two drafted ones (the next drafted round pays the drain again). Only a round that LOSES stops a drafted stretch."""
    lad = seeded({1: 30.0, 2: 32.0, 3: 34.0})
    lad.since_probe = -10 ** 9
    assert lad.plan(3, P_GOOD, PLAIN_MS) == 3 and lad.last == 3
    lad.ms[3] = [90.0]                                                # the estimate dips: plain now measures better than width 3
    assert lad.plan(3, P_GOOD, PLAIN_MS) > 0 and lad.last > 0          # still drafting (a shallower width), never back to plain
    lad.last = 3
    assert lad.observe(3, 90.0, Ladder.tokens(P_GOOD, 3), PLAIN_MS, PLAIN_MS) is False   # a round that loses owes plain rounds
    assert lad.plan(3, P_GOOD, PLAIN_MS) == 0


def test_the_drained_step_is_paid_over_the_measured_drafted_stretch():
    """A probe pays most of the drained step; a width that has been holding for a long stretch pays 1/stretch of it (the transition is a
    one-off, the stretch is what it buys). Same round numbers, two stretches."""
    acc = [0.8, 0.7, 0.6]
    tok = Ladder.tokens(acc, 3)
    ms = 44.0
    assert tok * PLAIN_MS > ms * (1.0 + LADDER.ROW_TIE)               # a steady win over plain ...
    assert tok * PLAIN_MS < (ms + PLAIN_MS) * (1.0 + LADDER.ROW_TIE)  # ... that a one-off probe cannot pay for, with the step drained
    lad = Ladder()
    lad.last = 0
    lad.plan(3, acc, PLAIN_MS, 1)                                     # leaves plain: this round drains the step
    assert lad.observe(3, ms, tok, PLAIN_MS, PLAIN_MS) is False
    assert lad.owed > 0
    lad2 = Ladder()
    lad2.last = 0
    lad2.run_ema = 20.0                                               # a width that has been holding for ~20 rounds
    lad2.plan(3, acc, PLAIN_MS, 1)
    assert lad2.observe(3, ms, tok, PLAIN_MS, PLAIN_MS) is True


def test_a_shared_pass_is_never_widened_past_the_per_session_plan(monkeypatch):
    """Several sessions share one read of the weights: the per-session policy prices a plain row at its own row + 1/n of the fixed step, so a
    pass where it chose plain must not be dragged into drafting by a noisy pass measurement. The ladder may only narrow that plan."""
    eng = ladder_engine(monkeypatch, sessions=2, rows=16, depth=3)
    eng.members = [session(), session()]
    eng._round_D = None
    eng._plain_n[2] = 26.0                                            # the pass of n = 2 has its own measured plain rate
    assert eng._ladder_depths(eng.members)[0] > 0                     # the ladder's own width says draft
    eng._plan_depth = lambda s: 0                                     # the shared-step policy says plain
    assert eng._round_depths(eng.members) == [0, 0]
    eng._plan_depth = lambda s: 1                                     # it may lower the ladder's width, not raise it
    assert eng._round_depths(eng.members) == [1, 1]


def test_the_drafters_block_caps_the_planned_width(monkeypatch):
    """A head that proposes one block per call cannot fill a deeper plan: the verify pays for the rows, the accept loop cuts the rest, and the
    ladder would price a width it never ran (the engine counts `spec_block_cut`). The plan stops at the block."""
    from types import SimpleNamespace
    eng = ladder_engine(monkeypatch, rows=24, depth=8)
    eng.backend.drafter = SimpleNamespace(block=2)
    eng.members = [session()]
    assert max(eng._ladder_depths(eng.members)) <= 2



def test_several_sessions_use_their_own_plain_rate_and_never_sit_at_the_cap(monkeypatch):
    """plain_ms is the solo rate: with n sessions the ladder needs the plain pass of n sessions. Cold, it drafts two rounds at the cap, then plain
    ticks measure it (engine._plain_n); a backend that never times keeps a short draft, not the cap."""
    eng = ladder_engine(monkeypatch, sessions=2, rows=16, depth=8)
    eng.members = [session(), session()]
    eng._round_D = None
    assert eng.plain_ms == PLAIN_MS and 2 not in eng._plain_n
    depths = [eng._ladder_depths(eng.members)[0] for _ in range(12)]
    assert depths[0] == depths[1] == 1                                 # cold: one draft, not the cap of the pass
    assert depths[2:8] == [0] * 6                                      # plain ticks that measure the n = 2 pass
    assert max(depths[8:]) <= 2, depths                                # never the cap forever
    assert eng._ladder_for(2) is not eng._ladder and eng._ladder_for(1) is eng._ladder


def test_mtp_cost_for_the_lookup_gate_is_measured_or_falls_through(monkeypatch):
    """(c) PLD.gate prices an MTP round against a lookup round with _mtp_cost_ms: an unmeasured width must fall through to the engine's own cost
    tables, not take an extrapolated (underpriced) ladder line that would make the MTP round beat the cheap lookup."""
    eng = ladder_engine(monkeypatch)
    eng._ladder.ms = {1: [26.0], 2: [34.0], 3: [42.0]}
    assert eng._mtp_cost_ms(100, 2) == pytest.approx(34.0)
    assert eng._mtp_cost_ms(100, 3) == pytest.approx(42.0)
    deep = eng._mtp_cost_ms(100, 7)
    assert deep is not None and deep > 58.0, deep                     # the fit would have said 18 + 8 * 7 = 74 only from three points; the tables say >= plain * (1 + ...)
    assert deep != pytest.approx(18.0 + 8.0 * 7)
