"""Flag spec_plain_min_sessions (KYOJIN_SPEC_PLAIN_MIN_SESSIONS = N, MP-THRESH): the MTP head drafts while FEWER than N sessions decode; with N or more every session runs the TRUE plain
step (the MULTI_PLAIN v2 rounds). N = 0 is off; KYOJIN_GLM_MULTI_PLAIN=1 is the alias for N = 2. Policy tests and the engine on a scripted backend with sessions joining and
leaving at thresholds 2 / 3 / 4. No GPU."""
import pytest

import test_glm_adaptive as TGA
import test_glm_multi_plain as TM
import test_spec_lookup as TSL
from kyojin_mlx.serve import engine as E
from kyojin_mlx.serve.sampling import Sampling

FLAG = "KYOJIN_SPEC_PLAIN_MIN_SESSIONS"
ALIAS = "KYOJIN_GLM_MULTI_PLAIN"


def _eng(monkeypatch, n_thr, n_members, alias="0", **kw):
    monkeypatch.setenv(FLAG, str(n_thr))
    eng = TM._engine(monkeypatch, alias, n_members, **kw)         # sets the alias env too
    monkeypatch.setenv(FLAG, str(n_thr))
    eng.backend.multi_depth = 1                                       # the legacy crowd rule: a measured fixed depth for several sessions
    return eng


def _sess(i):
    return E.Session(sid=i, ids=[7, 8], max_new=50, sampling=Sampling(), stop_ids=frozenset(), handle=None)


def test_registered_default_off():
    from kyojin_mlx import flags as F
    f = F.FLAGS["spec_plain_min_sessions"]
    assert f.env == FLAG and f.default == "0"


@pytest.mark.parametrize("thr", [2, 3, 4])
def test_threshold_policy(monkeypatch, thr):
    for n in (1, 2, 3, 4):
        assert TM.plans(_eng(monkeypatch, thr, n)) == ([0] * n if n >= thr else [1] * n), (thr, n)


def test_off_and_alias(monkeypatch):
    for n in (1, 2, 3, 4):
        assert TM.plans(_eng(monkeypatch, 0, n, alias="0")) == [1] * n                # flag off = legacy
    base = TM.plans(_eng(monkeypatch, 0, 2, alias="0"))
    eng = _eng(monkeypatch, 0, 2, alias="1")                           # alias alone = N 2
    assert eng._plain_min_sessions == 2 and TM.plans(eng) == [0, 0] and base != [0, 0]
    eng = _eng(monkeypatch, 3, 2, alias="1")                           # the new flag wins over the alias
    assert eng._plain_min_sessions == 3 and TM.plans(eng) == base


def test_bad_values_are_off(monkeypatch):
    for v in ("", "x", "-3", "0"):
        assert _eng(monkeypatch, v, 4)._plain_min_sessions == 0


def test_fixed_depth_wins(monkeypatch):
    assert TM.plans(_eng(monkeypatch, 3, 4, depth_override=1)) == [1] * 4
    assert _eng(monkeypatch, 3, 4, depth_override=1)._plain_min_sessions == 0


@pytest.mark.parametrize("thr", [2, 3, 4])
def test_join_and_leave_transitions(monkeypatch, thr):
    eng = _eng(monkeypatch, thr, 1)
    seen = [TM.plans(eng)]
    for i in (9, 10, 11):                                             # three joiners: 2, 3, 4 sessions
        eng.members.append(_sess(i))
        seen.append(TM.plans(eng))
    for _ in range(3):                                                # three leavers: 3, 2, 1
        eng.members.pop()
        seen.append(TM.plans(eng))
    want = [[0] * n if n >= thr else [1] * n for n in (1, 2, 3, 4, 3, 2, 1)]
    assert seen == want


def test_threshold_is_live(monkeypatch):
    """Read per round: moving N between rounds (what /admin/flags does) moves the plan at once."""
    eng = _eng(monkeypatch, 4, 3)
    assert TM.plans(eng) == [1] * 3
    monkeypatch.setenv(FLAG, "3")
    assert TM.plans(eng) == [0] * 3
    monkeypatch.setenv(FLAG, "0")
    assert TM.plans(eng) == [1] * 3


def _served(monkeypatch, thr, names, delay):
    TM._env(monkeypatch, "0")
    monkeypatch.setenv(FLAG, str(thr))
    be = TM.Real()
    out, st, _ = TSL.run({k: TM.REQS[k] for k in names}, backend=be, sessions=4, delay=delay, mtp_depth=1)
    return out, st, be


@pytest.mark.parametrize("thr", [2, 3, 4])
def test_engine_text_is_the_plain_text_and_the_crowd_rounds_follow_the_threshold(monkeypatch, thr):
    ref = {k: TM.solo_ref(monkeypatch, k) for k in TM.REQS}
    for names, delay in ((["a", "b"], 0.15), (["a", "b", "c"], 0.08), (["a", "b", "c", "d"], 0.05)):
        out, st, be = _served(monkeypatch, thr, names, delay)
        assert {k: v[:2] for k, v in out.items()} == {k: ref[k] for k in names}, (thr, names)
        # a pass of >= thr sessions is never a verify / feed pass; a pass of fewer sessions than thr and > 1 may be
        assert all(k == "step" for k, n, _ in be.log if n >= thr), (thr, names, {(k, n) for k, n, _ in be.log})
        if len(names) >= thr:
            assert st["spec_multi_plain_rounds"] > 0
            assert any(k == "step" and n >= thr for k, n, _ in be.log)
        else:
            assert st.get("spec_multi_plain_rounds", 0) == 0


def test_two_sessions_still_draft_at_threshold_three(monkeypatch):
    """The point of N = 3: two decoding sessions keep the verify rounds (d1 wins there)."""
    out, st, be = _served(monkeypatch, 3, ["a", "b"], 0.0)
    assert any(k in ("verify", "feed") and n == 2 for k, n, _ in be.log), {(k, n) for k, n, _ in be.log}
    assert st.get("spec_multi_plain_rounds", 0) == 0
