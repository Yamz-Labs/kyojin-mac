"""Flag glm_multi_plain (KYOJIN_GLM_MULTI_PLAIN, MULTI_PLAIN v2): the MTP head drafts only while ONE session decodes; with two or more every session runs the TRUE plain
step (_decode_tick: no verify pass, no head feed, lookahead on, the tuner not consulted). Policy unit tests (no model) and the engine on a scripted backend shaped like
GlmBackend (4 rows, step_feed, no spec_ok; the head misses on a schedule, the verify rolls back): 1 -> 2 -> 1 sessions joining and leaving mid-stream keep every
session's greedy text identical to its depth-0 run. No GPU."""
import time

import pytest

import test_glm_adaptive as TGA
import test_spec_lookup as TSL
from kyojin_mlx.serve import engine as E
from kyojin_mlx.serve.sampling import Sampling

FLAG = "KYOJIN_GLM_MULTI_PLAIN"


def _engine(monkeypatch, flag, n_members, depth=1, **kw):
    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv("KYOJIN_GLM_ADAPTIVE", "0")
    monkeypatch.setenv("KYOJIN_GLM_PLAN", "0")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    eng = E.Engine(TGA.Scripted(), sessions=4, mtp_depth=depth, ctx=4096, chunk=4, block=8, prefix_bytes=0, **kw)
    eng.members = [E.Session(sid=i, ids=[1, 2, 3, 4, 5], max_new=200, sampling=Sampling(), stop_ids=frozenset(), handle=None) for i in range(n_members)]
    return eng


def plans(eng):
    return [eng._plan_depth(s) for s in eng.members]


# ------------------------------------------------------------------------------------------------ the policy
def test_registered_default_off():
    from kyojin_mlx import flags as F
    f = F.FLAGS["glm_multi_plain"]
    assert f.env == FLAG and f.default == "0"


def test_solo_keeps_the_depth_and_a_crowd_plans_plain(monkeypatch):
    assert plans(_engine(monkeypatch, "1", 1)) == [1]
    for n in (2, 3, 4):
        assert plans(_engine(monkeypatch, "1", n)) == [0] * n


def test_join_and_leave_transitions(monkeypatch):
    eng = _engine(monkeypatch, "1", 1)
    s0 = eng.members[0]
    seen = [plans(eng)]                                               # alone
    eng.members.append(E.Session(sid=9, ids=[7, 8], max_new=50, sampling=Sampling(), stop_ids=frozenset(), handle=None))
    seen.append(plans(eng))                                           # a joiner
    eng.members.append(E.Session(sid=10, ids=[7, 8], max_new=50, sampling=Sampling(), stop_ids=frozenset(), handle=None))
    seen.append(plans(eng))                                           # a third
    del eng.members[2]
    seen.append(plans(eng))                                           # one leaves: still two
    del eng.members[1]
    seen.append(plans(eng))                                           # the other leaves: s0 alone again
    assert seen == [[1], [0, 0], [0, 0, 0], [0, 0], [1]] and eng.members == [s0]


def test_flag_off_keeps_the_old_depth_at_every_count(monkeypatch):
    for n in (1, 2, 3):
        eng = _engine(monkeypatch, "0", n)
        eng.backend.multi_depth = 1                                   # the legacy crowd rule (a measured fixed depth for several sessions)
        assert plans(eng) == [1] * n
        monkeypatch.setenv(FLAG, "1")
        assert plans(eng) == ([1] if n == 1 else [0] * n)


def test_a_fixed_depth_wins_and_the_cap_is_still_the_cap(monkeypatch):
    assert plans(_engine(monkeypatch, "1", 2, depth_override=1)) == [1, 1]
    eng = _engine(monkeypatch, "1", 1)
    eng.members[0].max_new = 1                                        # no room for a draft: 0 as ever
    assert plans(eng) == [0]


def test_the_policy_is_not_glm_only(monkeypatch):
    """Any MTP backend (here the plain lookup fake, no glm_costs) takes the same session-count policy."""
    eng = _engine(monkeypatch, "1", 3)
    eng.backend = TSL.FakeLookup()
    assert eng._glm_multi_plain


# ------------------------------------------------------------------------------------------------ the engine, sessions joining and leaving
class Real(TGA.Scripted):
    """The scripted head on a backend shaped like GlmBackend: 4 rows (KYOJIN_GLM_SPEC_ROWS), step_feed (the plain step of a speculative epoch, it feeds the head),
    no spec_ok, no bind_drafts. A few ms per pass so a second request lands mid-stream. log = (kind, sessions, rows per session)."""
    max_spec_rows = 4

    def __init__(self):
        super().__init__()
        self.log = []

    def spec_verify_bc(self, bc, token_lists):
        self.log.append(("verify", len(token_lists), [len(t) for t in token_lists]))
        time.sleep(0.003)
        return super().spec_verify_bc(bc, token_lists)

    def step_feed(self, bc, tokens, drafts, p0s):
        self.log.append(("feed", len(bc), [1] * len(bc)))
        time.sleep(0.003)
        return TSL.FakeLookup.step(self, bc, tokens)

    def step(self, bc, tokens):
        self.log.append(("step", len(bc), [1] * len(bc)))
        time.sleep(0.003)
        return super().step(bc, tokens)


REQS = {"a": ([80, 5, 9, 11, 13], {"max_new": 150}), "b": ([81, 6, 9, 12, 14], {"max_new": 24}), "c": ([82, 7, 8, 11, 15], {"max_new": 30}),
        "d": ([83, 8, 7, 10, 16], {"max_new": 36})}


def _env(monkeypatch, flag, tuner="1"):
    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv("KYOJIN_MU_TUNER", tuner)
    monkeypatch.setenv("KYOJIN_GLM_ADAPTIVE", "0")
    monkeypatch.setenv("KYOJIN_GLM_PLAN", "0")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")


def solo_ref(monkeypatch, name):
    _env(monkeypatch, "0")
    out, _, _ = TSL.run({name: REQS[name]}, backend=Real(), sessions=1, mtp_depth=1, depth_override=0)
    return out[name][:2]


def served(monkeypatch, flag, names, delay, tuner="1"):
    _env(monkeypatch, flag, tuner)
    be = Real()
    out, st, _ = TSL.run({k: REQS[k] for k in names}, backend=be, sessions=4, delay=delay, mtp_depth=1)
    return out, st, be


def crowd_kinds(be):
    return {k for k, n, _ in be.log if n > 1}


def test_policy_on_a_four_row_backend(monkeypatch):
    eng = _engine(monkeypatch, "1", 4)
    eng.backend = Real()
    assert plans(eng) == [0] * 4


def test_one_two_one_sessions_are_byte_identical_to_the_plain_runs(monkeypatch):
    ref = {k: solo_ref(monkeypatch, k) for k in REQS}
    out, st, be = served(monkeypatch, "1", ["a", "b"], 0.15)          # a alone, b joins mid-stream and leaves first, a alone again
    assert {k: v[:2] for k, v in out.items()} == {k: ref[k] for k in ("a", "b")}
    solo = [rows for k, n, rows in be.log if n == 1 and k == "verify"]
    assert any(max(r) > 1 for r in solo), "no draft while alone"
    assert crowd_kinds(be) == {"step"}, f"a crowd round was not the plain step: {crowd_kinds(be)}"   # no verify pass, no step_feed (head feed) with 2 sessions
    seq = [n for k, n, _ in be.log if k != "feed"]
    first2, last2 = seq.index(2), len(seq) - 1 - seq[::-1].index(2)
    assert 1 in seq[:first2] and 1 in seq[last2 + 1:], "not 1 -> 2 -> 1"
    assert st["spec_multi_plain_rounds"] > 0 and st["spec_multi_solo_rounds"] > 0
    assert st["spec_accepted"] < st["spec_drafted"]                   # the head missed while alone: the rollback ran


def test_three_and_four_sessions_and_the_flag_off_arm_keep_the_same_text(monkeypatch):
    ref = {k: solo_ref(monkeypatch, k) for k in REQS}
    for names, delay in ((["a", "b", "c"], 0.08), (["a", "b", "c", "d"], 0.05)):
        for flag in ("1", "0"):
            out, st, be = served(monkeypatch, flag, names, delay)
            assert {k: v[:2] for k, v in out.items()} == {k: ref[k] for k in names}, (names, flag)
            if flag == "1":
                assert crowd_kinds(be) <= {"step"}, crowd_kinds(be)
                assert any(n > 1 for k, n, _ in be.log if k == "step")
            else:
                assert st.get("spec_multi_plain_rounds", 0) == 0


def test_the_tuner_is_not_consulted_and_keeps_no_table(monkeypatch):
    """Two sessions from the start (the tuner's arm 1 would enter _spec_tick): the flag runs plain steps and the tuner table (stored on the backend) stays empty."""
    out, st, be = served(monkeypatch, "1", ["a", "b"], 0.0)
    assert crowd_kinds(be) <= {"step"} and st["spec_multi_plain_rounds"] > 0
    assert not st.get("tuner_ms"), st.get("tuner_ms")
    assert not getattr(be, "_mu_tuner", None) or not be._mu_tuner.cost


def test_the_crowd_looks_ahead(monkeypatch):
    """The plain epoch of a speculative engine launches the next step before it reads the one in flight (Engine.lookahead is off for such a backend)."""
    seen = []
    real = E.Engine._consume

    def spy(self, y, leaving, allow_yield):
        if not self._in_spec and len(self.members) > 1:
            seen.append(self.nxt is not None)
        return real(self, y, leaving, allow_yield)
    monkeypatch.setattr(E.Engine, "_consume", spy)
    served(monkeypatch, "1", ["a", "b", "c"], 0.0)
    assert seen and sum(seen) > len(seen) // 2, (sum(seen), len(seen))


def test_a_backend_without_glm_costs_gets_the_same_policy(monkeypatch):
    """The scripted backend without glm_costs (not GLM-shaped): 3 sessions with the flag run plain steps and keep the same greedy texts as without."""
    class NoCosts(Real):
        glm_costs = None
    outs = {}
    for flag in ("1", "0"):
        _env(monkeypatch, flag)
        be = NoCosts()
        out, st, _ = TSL.run({k: REQS[k] for k in ("a", "b", "c")}, backend=be, sessions=3, delay=0.08, mtp_depth=1)
        outs[flag] = {k: v[:2] for k, v in out.items()}
        if flag == "1":
            assert st["spec_multi_plain_rounds"] > 0 and crowd_kinds(be) <= {"step"}
    assert outs["1"] == outs["0"]
