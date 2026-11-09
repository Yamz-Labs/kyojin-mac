"""Relaxed ("typical") acceptance, flag spec_typical (kyojin_mlx.specsam.typical_mask + the served round): rule boundaries on synthetic logits, the
engine with the flag off equal to today, and the flag on in solo / mixed greedy + sampled / lookup rounds."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import specsam
from kyojin_mlx.serve.sampling import Sampling
from test_serve_engine import V, nxt
from test_serve_spec import FakeSpec, run, solo_all
from test_specsam import FakeLazy

T1 = Sampling(1.0, 1.0, 0, 0.0, 3)


def probs(lg, s=T1):
    return np.exp(np.asarray(specsam.logp(mx.array(lg, dtype=mx.float32), s)))


def mask(lg, drafts, eq, eps=0.09, delta=0.3, s=T1, hmax=None):
    rows = mx.array(np.asarray(lg, dtype=np.float32))
    return specsam.typical_mask(rows, mx.array(drafts, dtype=mx.int32), mx.array(eq), s, eps, delta, hmax).tolist()


def test_rule_boundaries():
    lg = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    ent = -(np.exp(lg) * lg).sum()
    thr = min(0.09, 0.3 * np.exp(-ent))                               # H ~ 1.1: delta * exp(-H) ~ 0.1 > eps, so thr = 0.09
    assert thr == pytest.approx(0.09)
    assert mask(lg, [1], [False]) == [True]                           # p 0.3 >= thr
    assert mask(lg, [2], [False]) == [True]                           # p 0.15 >= thr
    assert mask(lg, [3], [False]) == [False]                          # p 0.05 < thr
    assert mask(lg, [3], [True]) == [True]                            # equality always accepts
    assert mask(lg, [2], [False], eps=0.2, delta=1.0) == [False]      # thr = min(0.2, 0.32) = 0.2 > 0.15
    assert mask(lg, [2], [False], eps=1.0, delta=0.1) == [True]       # thr = 0.1 * exp(-H) ~ 0.033 <= 0.15
    

def test_entropy_zero_and_cut_tokens_never_pass():
    lg = np.array([[30.0, 0.0, 0.0, 0.0]])                            # H ~ 0: thr = min(eps, delta) = 0.09
    assert mask(lg, [1], [False]) == [False] and mask(lg, [0], [False]) == [True]
    top_p = Sampling(1.0, 0.5, 0, 0.0, 3)
    lg = np.log(np.array([[0.6, 0.2, 0.1, 0.1]]))                     # top_p 0.5 keeps only token 0
    assert probs(lg[0], top_p)[1] == 0.0
    assert mask(lg, [1], [False], eps=1e-9, delta=1e-9, s=top_p) == [False]
    assert mask(lg, [0], [False], s=top_p) == [True]


def test_high_entropy_uses_delta_bound():
    lg = np.zeros((1, 64))                                            # p = 1/64 = 0.0156, H = ln 64: delta * exp(-H) = 0.3 / 64 = 0.0047
    assert mask(lg, [5], [False], eps=0.09, delta=0.3) == [True]
    assert mask(lg, [5], [False], eps=0.09, delta=1.2) == [False]     # 1.2 / 64 > 1 / 64


def test_hmax_gate():
    lg = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))                   # H ~ 1.14 nats, draft 1 has p 0.3 (relaxed)
    ent = float(-(np.exp(lg) * lg).sum())
    assert mask(lg, [1], [False], hmax=ent + 0.1) == [True]           # H below hmax: relaxed as before
    assert mask(lg, [1], [False], hmax=ent - 0.1) == [False]          # H above hmax: never relaxed, even with a high p
    assert mask(lg, [1], [True], hmax=0.0) == [True]                  # an exact match still passes
    flat = np.zeros((1, 64))                                          # H = ln 64 ~ 4.16
    assert mask(flat, [3], [False], eps=0.01, delta=1.0, hmax=5.0) == [True]
    assert mask(flat, [3], [False], eps=0.01, delta=1.0, hmax=4.0) == [False]


def test_hmax_off_is_todays_mask():
    rng = np.random.default_rng(0)
    lg = rng.normal(size=(6, 40)) * 2
    d = [int(x) for x in rng.integers(0, 40, 6)]
    eq = [False, True, False, False, True, False]
    base = mask(lg, d, eq, eps=0.05, delta=0.5)
    assert mask(lg, d, eq, eps=0.05, delta=0.5, hmax=None) == base
    assert mask(lg, d, eq, eps=0.05, delta=0.5, hmax=float("inf")) == base
    assert mask(lg, d, eq, eps=0.05, delta=0.5, hmax=1e9) == base


def test_hmax_flag_and_engine_parse(monkeypatch):
    from kyojin_mlx import flags
    from kyojin_mlx.serve import engine as E
    f = flags.FLAGS["spec_typical_hmax"]
    assert f.env == "KYOJIN_SPEC_TYPICAL_HMAX" and f.default == "" and f.runtime and f.kind == "float"
    assert flags._norm(f, "2.5") == "2.5" and flags._norm(f, "inf") == "inf" and flags._norm(f, "") == ""
    with pytest.raises(ValueError):
        flags._norm(f, "-1")
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    for v, want in (("", None), ("inf", None), ("2.5", 2.5)):
        monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_HMAX", v)
        assert E.Engine._typical(None)[2] == want


def test_flags_registered_and_validated():
    from kyojin_mlx import flags
    for n, d in (("spec_typical", "0"), ("spec_typical_eps", "0.09"), ("spec_typical_delta", "0.3")):
        assert flags.FLAGS[n].default == d and flags.FLAGS[n].runtime
    assert flags._norm(flags.FLAGS["spec_typical_eps"], "0.2") == "0.2"
    with pytest.raises(ValueError):
        flags._norm(flags.FLAGS["spec_typical_eps"], "abc")
    with pytest.raises(ValueError):
        flags._norm(flags.FLAGS["spec_typical_eps"], "-1")


def reqs(sampling=None, n=1):
    return {f"a{i}": ([1 + i, 2, 3, 4, 5], dict(max_new=60, **({"sampling": sampling} if sampling else {}))) for i in range(n)}


S_PEAK = Sampling(1e-3, 1.0, 0, 0.0, 3)                               # near one-hot rows: the verify sample is the argmax whatever the key


def test_flag_off_is_bit_identical(monkeypatch):
    monkeypatch.delenv("KYOJIN_SPEC_TYPICAL", raising=False)
    s = Sampling(1.0, 0.95, 0, 0.0, 7)
    a = run(reqs(s), FakeSpec(err=3), sessions=1, mtp_depth=3, prefix_bytes=0)
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "0")
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_EPS", "0.0")
    b = run(reqs(s), FakeSpec(err=3), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert a[0]["a0"][:2] == b[0]["a0"][:2] and a[1]["spec_accepted"] == b[1]["spec_accepted"] and a[1].get("spec_typical_relaxed", 0) == 0


def test_flag_on_peaked_target_equals_plain(monkeypatch):
    """A one-hot target: the rule (p of a wrong draft = 0) never relaxes anything, tokens equal plain, nothing counted as relaxed."""
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    rr = reqs(S_PEAK)
    want = solo_all(rr)["a0"][0]
    got, st, _ = run(rr, FakeSpec(err=3), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert got["a0"][0] == want and got["a0"][1] == "length"
    assert st.get("spec_typical_relaxed", 0) == 0 and st["spec_accepted"] > 0


class Soft(FakeSpec):
    """Verify rows put 0.5 on the true next token and 0.4 on the (wrong) draft: sampled, the rule accepts the draft where equality may not."""
    def spec_verify_bc(self, bc, token_lists):
        out = super().spec_verify_bc(bc, token_lists)
        z = np.full((out.logits.shape[0], V), -30.0, dtype=np.float32)
        for k, toks in enumerate(token_lists):
            for j in range(len(toks)):
                r = out.starts[k] + j
                true = int(mx.argmax(out.logits[r]).item())
                z[r, true] = np.log(0.5)
                if j + 1 < len(toks):
                    z[r, toks[j + 1]] = max(z[r, toks[j + 1]], np.log(0.4)) if toks[j + 1] != true else z[r, true]
        out.logits = mx.array(z)
        return out


def test_flag_on_relaxes_and_counts_and_never_exceeds_depth(monkeypatch):
    s = Sampling(1.0, 1.0, 0, 0.0, 11)
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "0")
    off = run(reqs(s), Soft(err=1), sessions=1, mtp_depth=3, prefix_bytes=0)
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    on = run(reqs(s), Soft(err=1), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert off[1].get("spec_typical_relaxed", 0) == 0
    assert on[1]["spec_typical_relaxed"] > 0                            # err=1: every draft is wrong, p = 0.4 >= 0.09: relaxed accepts
    assert on[1]["spec_accepted"] > off[1]["spec_accepted"]             # the depth policy's accepted count includes them
    assert on[1]["spec_accepted"] <= on[1]["spec_drafted"]              # never above what was drafted
    assert len(on[0]["a0"][0]) == 60 and on[0]["a0"][1] == "length"


def test_mixed_greedy_and_sampled_members(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    rr = {"g": ([1, 2, 3, 4, 5], dict(max_new=50)), "s": ([2, 3, 4, 5, 6], dict(max_new=50, sampling=Sampling(1.0, 1.0, 0, 0.0, 5)))}
    want = solo_all({"g": rr["g"]})["g"][0]
    got, st, be = run(rr, Soft(err=1), sessions=2, mtp_depth=3, delay=0, prefix_bytes=0)
    assert got["g"][0] == want                                          # the greedy member is untouched by the rule
    assert len(got["s"][0]) == 50 and max(be.passes) <= be.max_spec_rows


def test_lazy_drafts_device_tokens(monkeypatch):
    """The lazy path builds the mask from the device draft tokens (lh.toks): the peaked target keeps the plain tokens."""
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    monkeypatch.setenv("KYOJIN_SPEC_FULL", "1")
    rr = reqs(S_PEAK)
    want = solo_all(rr)["a0"][0]
    got, st, _ = run(rr, FakeLazy(err=0), sessions=1, mtp_depth=3, depth_override=3, prefix_bytes=0, ctx=4096)
    assert got["a0"][0] == want and st["spec_accepted"] > 0 and st.get("spec_typical_relaxed", 0) == 0


def test_lookup_drafts_device_tokens_feed_the_mask():
    """The engine reads a lookup session's draft tokens from LookupDrafts.toks[slot[k]][:n - 1]: a session without lookup rows is not in slot, a short row is zero padded."""
    from kyojin_mlx.serve.backends import LookupDrafts
    lh = LookupDrafts({0: [5, 6, 7], 2: [9, 8]}, [3, 0, 2])
    assert lh.slot == {0: 0, 2: 2}
    assert lh.toks[lh.slot[0]][:3].tolist() == [5, 6, 7] and lh.toks[lh.slot[2]][:2].tolist() == [9, 8]
    lg = np.full((3, V), -30.0, dtype=np.float32)
    lg[:, 5] = 0.0
    m = specsam.typical_mask(mx.array(lg), lh.toks[lh.slot[0]][:3], mx.array([True, False, False]), T1, 0.09, 0.3)
    assert m.tolist() == [True, False, False]                          # token 5 is the target's; 6 and 7 have p ~ 0


# ------------------------------------------------------------------------------------------------ flag spec_typical_lookup
def two_way_backend():
    """The lookup fake, with every verify row putting 0.5 on the true next token and 0.5 on the next id: a sampled row often differs from a correct draft
    while the draft still has p = 0.5 (the relaxed rule accepts it, equality does not)."""
    import test_spec_lookup as L

    class Be(L.FakeLookup):
        def spec_verify_bc(self, bc, token_lists):
            self.passes.append(sum(len(t) for t in token_lists))
            rows, starts = [], []
            for cache, toks in zip(bc, token_lists):
                starts.append(len(rows))
                rows += [L.nxt8(cache["h"] + toks[:j + 1]) for j in range(len(toks))]
            z = np.full((len(rows), V), -30.0, dtype=np.float32)
            for r, true in enumerate(rows):
                z[r, true] = z[r, (true + 1) % V] = np.log(0.5)
            return L.SpecPass(mx.array(z), starts, bc, token_lists)
    return Be()


def lookup_run(monkeypatch, typical_lookup, lookup="1"):
    """(stats, calls): calls = [(round had a lookup draft, rows)] for every typical_mask the engine built."""
    import test_spec_lookup as L
    from kyojin_mlx.serve import engine as E
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", lookup)
    if typical_lookup is None:
        monkeypatch.delenv("KYOJIN_SPEC_TYPICAL_LOOKUP", raising=False)
    else:
        monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_LOOKUP", typical_lookup)
    plan, calls = {"lookups": {}}, []
    real_plan, real_mask = E.Engine._lookup_plan, E.SPECSAM.typical_mask

    def spy_plan(self, *a, **k):
        r = real_plan(self, *a, **k)
        plan["lookups"] = dict(r[1])
        return r

    def spy_mask(rows, *a, **k):
        calls.append((bool(plan["lookups"]), rows.shape[0]))
        return real_mask(rows, *a, **k)
    monkeypatch.setattr(E.Engine, "_lookup_plan", spy_plan)
    monkeypatch.setattr(E.SPECSAM, "typical_mask", spy_mask)
    ids = L.repeated_prompt()
    got, st, _ = L.run({"a": (ids, dict(max_new=64, sampling=Sampling(1.0, 1.0, 0, 0.0, 5)))}, two_way_backend(), mtp_depth=4, prefix_bytes=0)
    assert len(got["a"][0]) == 64
    return st, calls


def test_typical_lookup_flag_registered():
    from kyojin_mlx import flags
    f = flags.FLAGS["spec_typical_lookup"]
    assert f.env == "KYOJIN_SPEC_TYPICAL_LOOKUP" and f.default == "rep" and f.runtime


def test_lookup_round_never_relaxed_with_the_flag_at_0(monkeypatch):
    for v in ("0",):
        st, calls = lookup_run(monkeypatch, v)
        assert st["spec_lookup_rounds"] > 0 and st["spec_typical_lookup_skipped"] > 0
        assert calls and not any(lk for lk, _ in calls)                # no relaxed mask is built for a lookup round
        assert st["spec_typical_relaxed"] > 0                          # the MTP rounds of the same request still relax


def test_lookup_round_relaxed_with_the_flag_at_1(monkeypatch):
    st, calls = lookup_run(monkeypatch, "1")
    assert st["spec_lookup_rounds"] > 0 and st.get("spec_typical_lookup_skipped", 0) == 0
    assert any(lk for lk, _ in calls)                                  # the old behavior: lookup rounds get the mask
    assert st["spec_typical_relaxed"] > 0


def test_mtp_rounds_keep_the_relaxed_rule_whatever_the_flag(monkeypatch):
    for v in ("0", "1", None):
        st, calls = lookup_run(monkeypatch, v, lookup="0")
        assert st.get("spec_lookup_rounds", 0) == 0 and st.get("spec_typical_lookup_skipped", 0) == 0
        assert calls and st["spec_typical_relaxed"] > 0


# ------------------------------------------------------------------------------------------------ spec_typical_lookup=rep (default)
def test_repeats_rule():
    loop = list(range(100, 120))                                      # a 20-token span
    assert not specsam.repeats(loop + loop[:11], loop[11], n=12)      # the span seen once: the new copy is its 2nd occurrence
    assert specsam.repeats(loop + loop + loop[:11], loop[11], n=12)   # seen twice: a third is refused
    assert not specsam.repeats(loop + loop + loop[:11], 999, n=12)    # another token does not complete the n-gram
    assert not specsam.repeats(loop[:5], loop[5], n=12)               # shorter than the n-gram
    assert not specsam.repeats(loop + loop + loop[:11], loop[11], n=12, w=15)   # the earlier copies fell out of the window


def test_rep_default_keeps_lookup_copies_of_the_prompt_relaxed(monkeypatch):
    st, calls = lookup_run(monkeypatch, None)                         # default = rep
    assert st["spec_lookup_rounds"] > 0 and st.get("spec_typical_lookup_skipped", 0) == 0
    assert any(lk for lk, _ in calls) and st["spec_typical_relaxed"] > 0
    assert st.get("spec_typical_rep_refused", 0) == 0                 # nothing of the session's own output repeats


def test_rep_refuses_a_repeated_span_and_equality_still_accepts(monkeypatch):
    from kyojin_mlx.serve import engine as E
    st0, _ = lookup_run(monkeypatch, None)
    monkeypatch.setattr(E.SPECSAM, "repeats", lambda *a, **k: True)  # every relaxed-only token extends a repeated span
    st1, _ = lookup_run(monkeypatch, None)
    assert st1["spec_typical_rep_refused"] > 0 and st1.get("spec_typical_relaxed", 0) == 0
    assert st1["spec_accepted"] > 0                                   # equal drafts still accept
    assert st1["spec_accepted"] < st0["spec_accepted"]                # and the refused ones no longer count
    st2, _ = lookup_run(monkeypatch, "1")                             # the arm 1 has no check
    assert st2.get("spec_typical_rep_refused", 0) == 0 and st2["spec_typical_relaxed"] > 0


def test_rep_applies_to_mtp_drafts_too(monkeypatch):
    from kyojin_mlx.serve import engine as E
    monkeypatch.setattr(E.SPECSAM, "repeats", lambda *a, **k: True)
    st, _ = lookup_run(monkeypatch, None, lookup="0")
    assert st["spec_typical_rep_refused"] > 0 and st.get("spec_typical_relaxed", 0) == 0


def test_rep_check_off_the_greedy_path(monkeypatch):
    from kyojin_mlx.serve import engine as E
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    monkeypatch.setattr(E.SPECSAM, "repeats", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    rr = {"g": ([1, 2, 3, 4, 5], dict(max_new=40))}
    want = solo_all(rr)["g"][0]
    got, st, _ = run(rr, Soft(err=1), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert got["g"][0] == want


# ------------------------------------------------------------------------------------------------ flag spec_typical_phase
OPEN, CLOSE = 1000, 1001                                              # think tags outside the fake's vocabulary: the fake never generates them


def phase_run(monkeypatch, phase, prompt, think=True, sampling=None, n=60):
    from kyojin_mlx.serve import engine as E
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    if phase is None:
        monkeypatch.delenv("KYOJIN_SPEC_TYPICAL_PHASE", raising=False)
    else:
        monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_PHASE", phase)
    monkeypatch.setattr(E.Engine, "think_ids", (OPEN, CLOSE) if think else None)
    s = sampling or Sampling(1.0, 1.0, 0, 0.0, 11)
    got, st, _ = run({"a": (prompt, dict(max_new=n, sampling=s))}, Soft(err=1), sessions=1, mtp_depth=3, prefix_bytes=0)
    return got["a"][0], st


def test_phase_flag_registered():
    from kyojin_mlx import flags
    f = flags.FLAGS["spec_typical_phase"]
    assert f.env == "KYOJIN_SPEC_TYPICAL_PHASE" and f.default == "all" and f.runtime and f.choices == ("answer", "all")
    assert flags.FLAGS["loop_guard"].default == "1"                   # the neighbouring entry is still there


def test_thinking_tokens_never_relaxed_under_answer(monkeypatch):
    p = [1, 2, 3, OPEN]                                               # the template opened the block: the whole reply is thinking
    toks, st = phase_run(monkeypatch, "answer", p)
    assert len(toks) == 60 and st.get("spec_typical_relaxed", 0) == 0 and st["spec_typical_think_skipped"] > 0


def test_default_phase_is_all(monkeypatch):
    p = [1, 2, 3, OPEN]
    toks, st = phase_run(monkeypatch, None, p)                        # unset = all: thinking is relaxed too
    assert toks == phase_run(monkeypatch, "all", p)[0] and st["spec_typical_relaxed"] > 0 and st.get("spec_typical_think_skipped", 0) == 0


def test_answer_tokens_still_relaxed(monkeypatch):
    for p in ([1, 2, 3, 4, 5], [1, OPEN, 2, CLOSE, 3]):              # no tag at all / a closed block
        toks, st = phase_run(monkeypatch, None, p)
        assert len(toks) == 60 and st["spec_typical_relaxed"] > 0 and st.get("spec_typical_think_skipped", 0) == 0


def test_all_is_todays_behaviour(monkeypatch):
    p = [1, 2, 3, OPEN]
    a, sa = phase_run(monkeypatch, "all", p)
    b, sb = phase_run(monkeypatch, "all", p, think=False)             # tags unknown: the same run
    assert a == b and sa["spec_typical_relaxed"] > 0 and sa.get("spec_typical_think_skipped", 0) == 0


def test_model_without_thinking_is_always_answer(monkeypatch):
    toks, st = phase_run(monkeypatch, None, [1, 2, 3, OPEN], think=False)
    assert st["spec_typical_relaxed"] > 0 and st.get("spec_typical_think_skipped", 0) == 0


def test_greedy_and_spec_rs_unchanged(monkeypatch):
    p = [1, 2, 3, OPEN]
    g = [phase_run(monkeypatch, ph, p, sampling=Sampling(0.0, 1.0, 0, 0.0, 1))[0] for ph in ("answer", "all")]
    assert g[0] == g[1]
    monkeypatch.setenv("KYOJIN_SPEC_RS", "1")                         # the q-sampled path: its rows never reach the typical rule
    from kyojin_mlx.serve import engine as E
    monkeypatch.setattr(E.Engine, "think_ids", (OPEN, CLOSE))
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    r = []
    for ph in ("all", "answer"):
        monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_PHASE", ph)
        got, st, be = run({"a": (p, dict(max_new=60, sampling=Sampling(1.0, 1.0, 0, 0.0, 3)))}, FakeLazy(err=1), sessions=1, mtp_depth=3, depth_override=3, prefix_bytes=0, ctx=4096)
        assert be.want_logits is True
        r.append((got["a"][0], st))
    assert r[0][0] == r[1][0] and r[1][1].get("spec_typical_think_skipped", 0) == 0 and r[1][1].get("spec_typical_relaxed", 0) == 0


def test_in_think_state_is_incremental(monkeypatch):
    from types import SimpleNamespace
    from kyojin_mlx.serve.engine import Engine, Session
    e = object.__new__(Engine)
    e.think_ids = (OPEN, CLOSE)
    mk = lambda ids: SimpleNamespace(ids=ids, out=[], th_in=False, th_n=-1)
    s = mk([5, OPEN, 6, CLOSE, 7, OPEN, 8])                           # the last tag of the prompt decides
    assert e._in_think(s) is True
    s.out += [9, 9, CLOSE]
    assert e._in_think(s) is False and s.th_n == 3
    s.out += [OPEN]
    assert e._in_think(s) is True and s.th_n == 4
    assert e._in_think(s) is True and s.th_n == 4                     # nothing new: nothing scanned
    assert e._in_think(mk([1, 2, 3])) is False and e._in_think(mk([1, CLOSE])) is False
    e.think_ids = None
    assert e._in_think(mk([OPEN])) is False


# ------------------------------------------------------------------------------------------------ flag spec_typical_cooldown
def cd_engine(monkeypatch, cd="100", detok=True):
    from types import SimpleNamespace
    from kyojin_mlx.serve.engine import Engine
    import collections
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_COOLDOWN", cd)
    e = object.__new__(Engine)
    e.stats = collections.Counter()
    e.detok = (lambda ids: "".join(chr(i) for i in ids)) if detok else None
    mk = lambda: SimpleNamespace(out=[], cd_n=0, cd_end=0, cd_win=0, cd_last=-1, cd_watch=None)
    return e, mk


LONG = "the quick brown fox jumps over the lazy dog again\n"


def feed(s, text):
    s.out += [ord(c) for c in text]


def test_cooldown_flag_registered_and_parsed(monkeypatch):
    from kyojin_mlx import flags
    f = flags.FLAGS["spec_typical_cooldown"]
    assert f.env == "KYOJIN_SPEC_TYPICAL_COOLDOWN" and f.default == "512" and f.kind == "int"
    e, _ = cd_engine(monkeypatch, "0")
    assert e._cooldown() == 0
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_COOLDOWN", "x")
    assert e._cooldown() == 0


def test_cooldown_no_signal_stays_relaxed(monkeypatch):
    e, mk = cd_engine(monkeypatch)
    s = mk()
    feed(s, "".join(f"line number {i} with enough distinct words in it to be long\n" for i in range(30)))
    assert e._cd_armed(s, 100) is False and not e.stats


def test_cooldown_window_exact_then_relaxed_again(monkeypatch):
    e, mk = cd_engine(monkeypatch)
    s = mk()
    feed(s, LONG)
    assert e._cd_armed(s, 100) is False                              # first copy: no signal
    feed(s, LONG)
    assert e._cd_armed(s, 100) is True and e.stats["spec_typical_cooldown_triggers"] == 1
    end = s.cd_end
    assert end == len(s.out) + 100
    s.out += [65] * 99
    assert e._cd_armed(s, 100) is True                               # one token before the end
    s.out += [65]
    assert len(s.out) == end and e._cd_armed(s, 100) is False        # exactly the window
    assert e.stats["spec_typical_cooldown_tokens"] == 100


def test_cooldown_doubles_and_caps(monkeypatch):
    e, mk = cd_engine(monkeypatch)
    s = mk()
    e._cd_trigger(s, 100)
    assert s.cd_win == 100
    s.out += [1] * 10
    e._cd_trigger(s, 100)                                            # inside the debounce burst: counts once
    assert s.cd_win == 100 and e.stats["spec_typical_cooldown_triggers"] == 1
    s.out += [1] * 100
    e._cd_trigger(s, 100)                                            # armed: doubles
    assert s.cd_win == 200
    s.out += [1] * 250                                               # window over (200 from 110), next window still open
    e._cd_trigger(s, 100)
    assert s.cd_win == 400
    s.out += [1] * 5000                                              # far away: back to the base
    e._cd_trigger(s, 100)
    assert s.cd_win == 100
    for _ in range(20):
        s.out += [1] * 70
        e._cd_trigger(s, 100)
    assert s.cd_win == 16384 or s.cd_win <= 16384


def test_cooldown_fenced_code_line_never_triggers(monkeypatch):
    e, mk = cd_engine(monkeypatch)
    s = mk()
    feed(s, "```\n" + LONG * 3 + "```\n")
    assert e._cd_armed(s, 100) is False and not e.stats


def test_cooldown_line_outside_window_ignored():
    from kyojin_mlx.serve.loopguard import LineWatch
    w = LineWatch()
    assert w.feed(LONG) is False
    assert w.feed("x" * 20000 + "\n") is False
    assert w.feed(LONG) is False                                     # the first copy is further back than 16K chars
    assert w.feed(LONG) is True


def test_cooldown_without_detok_is_inert(monkeypatch):
    e, mk = cd_engine(monkeypatch, detok=False)
    s = mk()
    feed(s, LONG * 3)
    assert e._cd_armed(s, 100) is False


def test_cooldown_zero_is_todays_run_and_default_is_inert_without_signal(monkeypatch):
    runs = []
    for cd in ("0", None):
        if cd is None:
            monkeypatch.delenv("KYOJIN_SPEC_TYPICAL_COOLDOWN", raising=False)
        else:
            monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_COOLDOWN", cd)
        runs.append(lookup_run(monkeypatch, None, lookup="0")[0])
    assert runs[0].get("spec_typical_cooldown_triggers", 0) == 0 and runs[1].get("spec_typical_cooldown_triggers", 0) == 0
    assert runs[0]["spec_typical_relaxed"] == runs[1]["spec_typical_relaxed"] and runs[0]["spec_accepted"] == runs[1]["spec_accepted"]


def test_cooldown_rep_refusal_arms_exact_window(monkeypatch):
    from kyojin_mlx.serve import engine as E
    monkeypatch.setattr(E.SPECSAM, "repeats", lambda *a, **k: True)
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_COOLDOWN", "0")
    st0, _ = lookup_run(monkeypatch, None, lookup="0")
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_COOLDOWN", "30")
    st1, calls = lookup_run(monkeypatch, None, lookup="0")
    assert st0["spec_typical_rep_refused"] > 0 and st1["spec_typical_cooldown_triggers"] >= 1
    assert st1["spec_typical_cooldown_tokens"] >= 30
    assert st1["spec_typical_rep_refused"] < st0["spec_typical_rep_refused"]   # armed rounds build no mask, so no more refusals


def test_cooldown_greedy_and_spec_rs_untouched(monkeypatch):
    from kyojin_mlx.serve import engine as E
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    monkeypatch.setattr(E.Engine, "_cd_armed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    rr = {"g": ([1, 2, 3, 4, 5], dict(max_new=40))}
    want = solo_all(rr)["g"][0]
    got, st, _ = run(rr, Soft(err=1), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert got["g"][0] == want


# ------------------------------------------------------------------------------------------------ flag spec_typical_rewind
def rw_engine(monkeypatch, rewind="3", think=True):
    from types import SimpleNamespace
    from kyojin_mlx.serve.engine import Engine
    import collections
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_REWIND", rewind)
    monkeypatch.setattr(Engine, "think_ids", (OPEN, CLOSE) if think else None)
    e = object.__new__(Engine)
    e.stats = collections.Counter()
    e.detok = lambda ids: "".join(chr(i) if i < 900 else "<>" for i in ids)
    e.backend = SimpleNamespace(replay=lambda *a: None)

    def mk(out, ids=(1, OPEN), cut=-1, **kw):
        return SimpleNamespace(out=list(out), ids=list(ids), th_in=False, th_n=-1, cd_n=len(out), cd_end=0, cd_win=100, cd_last=-1, cd_watch=object(),
                               rw_cut=cut, rw_n=0, rw_tok=0, rw_drop=False, n_gen=len(out), last=out[-1] if out else 0, lookup_index=object(),
                               lk_streak=True, key=mx.random.key(5), seed=5, sampling=T1, vl=False, **kw)
    return e, mk


def key_of(s):
    return np.asarray(s.key).tolist()


def test_rewind_flag_registered():
    from kyojin_mlx import flags
    f = flags.FLAGS["spec_typical_rewind"]
    assert f.env == "KYOJIN_SPEC_TYPICAL_REWIND" and f.default == "3" and f.kind == "int"


def test_rewind_drops_exactly_the_span_and_resets_state(monkeypatch):
    e, mk = rw_engine(monkeypatch)
    s = mk(list(range(2, 42)), cut=25)
    k0 = key_of(s)
    assert e._rw_apply(s) is True
    assert s.out == list(range(2, 27)) and s.last == 26 and s.n_gen == 25
    assert s.rw_tok == 15 and s.rw_n == 1 and s.rw_cut == -1 and s.rw_drop is True
    assert e.stats["spec_typical_rewinds"] == 1 and e.stats["spec_typical_rewound_tokens"] == 15
    assert s.th_n == -1 and s.cd_watch is None and s.cd_n == 0 and s.lookup_index is None and s.lk_streak is False
    assert s.cd_end == 25 + 100 and s.cd_last == 25                  # exact window from the kept prefix
    assert key_of(s) != k0                                           # the key moved on


def test_rewind_key_advances_each_time(monkeypatch):
    e, mk = rw_engine(monkeypatch)
    keys = []
    for _ in range(2):
        s = mk(list(range(2, 42)), cut=25)
        e._rw_apply(s)
        keys.append(key_of(s))
    assert keys[0] == keys[1]                                        # deterministic from the same state...
    s = mk(list(range(2, 42)), cut=25)
    s.rw_n = 1
    e._rw_apply(s)
    assert key_of(s) != keys[0]                                      # ... and a later rewind draws from further along


def test_rewind_cap_per_reply(monkeypatch):
    e, mk = rw_engine(monkeypatch, rewind="2")
    s = mk(list(range(2, 80)))
    for want in (True, True, False):
        s.rw_cut = 10
        assert e._rw_apply(s) is want
        s.out += list(range(2, 40))                                  # the session goes on generating
        s.n_gen = len(s.out)
    assert s.rw_n == 2 and e.stats["spec_typical_rewinds"] == 2


def test_rewind_zero_and_no_think_state_refuse(monkeypatch):
    e, mk = rw_engine(monkeypatch, rewind="0")
    s = mk(list(range(2, 42)), cut=10)
    assert e._rw_apply(s) is False and len(s.out) == 40 and s.rw_tok == 0 and not s.rw_drop
    e, mk = rw_engine(monkeypatch, think=False)                      # think state unknown: never
    s = mk(list(range(2, 42)), cut=10)
    assert e._rw_apply(s) is False and len(s.out) == 40
    e, mk = rw_engine(monkeypatch)
    e.backend = object()                                             # a backend that cannot rebuild a dropped cache: cooldown only
    s = mk(list(range(2, 42)), cut=10)
    assert e._rw_apply(s) is False and len(s.out) == 40


def test_no_rewind_in_the_answer_or_across_a_closed_block(monkeypatch):
    e, mk = rw_engine(monkeypatch)
    s = mk(list(range(2, 42)), ids=(1, OPEN, 5, CLOSE), cut=10)      # the prompt closed the block: answer phase
    assert e._rw_apply(s) is False and len(s.out) == 40
    s = mk(list(range(2, 20)) + [CLOSE] + list(range(20, 40)), cut=10)   # the answer began inside the span
    assert e._rw_apply(s) is False and len(s.out) == 39
    s = mk(list(range(2, 20)) + [CLOSE, OPEN] + list(range(20, 40)), cut=25)   # a block closed and reopened before the cut: the span itself is inside the open block
    assert e._rw_apply(s) is True and len(s.out) == 25
    assert e.stats["spec_typical_rewinds"] == 1


def test_rewind_cut_edges(monkeypatch):
    e, mk = rw_engine(monkeypatch)
    for cut in (-1, 0, 40, 99):
        s = mk(list(range(2, 42)), cut=cut)
        assert e._rw_apply(s) is False and len(s.out) == 40


def test_line_signal_marks_the_first_copy(monkeypatch):
    e, mk = rw_engine(monkeypatch)
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_COOLDOWN", "100")
    s = mk([])
    s.cd_watch, s.cd_n = None, 0
    pre = "short intro line\n"
    s.out += [ord(c) for c in pre + LONG + "other words in between of the lines\n"]
    assert e._cd_armed(s, 100) is False and s.rw_cut == -1
    s.out += [ord(c) for c in LONG]
    assert e._cd_armed(s, 100) is True
    assert s.rw_cut == len(pre)                                      # tokens (chars here) before the first copy
    assert "".join(chr(c) for c in s.out[:s.rw_cut]) == pre
    s.rw_cut = -1
    s.cd_last = -1
    e.stats.clear()
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_REWIND", "0")
    s2 = mk([])
    s2.cd_watch, s2.cd_n = None, 0
    s2.out += [ord(c) for c in LONG * 2]
    assert e._cd_armed(s2, 100) is True and s2.rw_cut == -1          # rewind 0: cooldown only


def test_first_repeat_helper():
    h = [9, 9] + list(range(20, 32)) + [5, 5, 5] + list(range(20, 32))[:11]
    assert specsam.first_repeat(h, 31) == 2 and specsam.first_repeat(h, 99) == -1
    two = list(range(12)) * 2 + list(range(11))
    assert specsam.first_repeat(two, 11) == 0 and specsam.repeats(two, 11, times=2)


def test_linewatch_first_at():
    from kyojin_mlx.serve import loopguard as LG
    w = LG.LineWatch()
    assert not w.feed("hello\n" + LONG + "x\n") and w.first_at is None
    assert w.feed(LONG) and w.first_at == len("hello\n")
    assert not w.feed("fresh start\n") and w.first_at is None


def rw_run(monkeypatch, rewind, think=True, n=200):
    """A sampled thinking session whose every refused relaxed round is a loop signal (the repetition test is forced true)."""
    from kyojin_mlx.serve import engine as E
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", "1")
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_LOOKUP", "rep")
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_COOLDOWN", "20")
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_REWIND", rewind)
    monkeypatch.setattr(E.Engine, "think_ids", (OPEN, CLOSE) if think else None)
    monkeypatch.setattr(E.SPECSAM, "repeats", lambda *a, **k: True)
    monkeypatch.setattr(E.SPECSAM, "first_repeat", lambda hist, tok, *a, **k: 3 if len(hist) > 3 else -1)
    be = Soft(err=1)
    got, st, _ = run({"a": ([1, 2, 3, 4, OPEN], dict(max_new=n, sampling=Sampling(1.0, 1.0, 0, 0.0, 11)))}, be, sessions=1, mtp_depth=3, prefix_bytes=0)
    return got["a"], st, be


def test_engine_rewinds_resume_exact_and_stop_at_the_cap(monkeypatch):
    (toks, reason, usage), st, be = rw_run(monkeypatch, "2")
    assert st["spec_typical_rewinds"] == 2 and st["spec_typical_rewound_tokens"] > 0
    assert st["rebuilds"] >= 2 and reason == "length"
    assert len(toks) == usage["completion_tokens"] == 200            # streamed tokens = tokens generated, rewound ones included; max_new counts them
    assert be.replays and be.replays[0] == 2                         # the first rebuild replays out[:cut] minus the pending token: cut = 3
    assert st["spec_typical_cooldown_triggers"] >= 2                 # every rewind arms the exact window


def test_engine_rewind_off_equals_cooldown_only(monkeypatch):
    a, sta, _ = rw_run(monkeypatch, "0")
    b, stb, _ = rw_run(monkeypatch, "3", think=False)                # no think state: no rewind possible
    assert a[:2] == b[:2] and sta.get("spec_typical_rewinds", 0) == 0 and stb.get("spec_typical_rewinds", 0) == 0
    assert sta["spec_typical_relaxed"] == stb["spec_typical_relaxed"] and sta["spec_accepted"] == stb["spec_accepted"]
    assert sta["spec_typical_cooldown_triggers"] == stb["spec_typical_cooldown_triggers"]


def test_engine_no_signal_session_untouched(monkeypatch):
    runs = []
    for rw in ("0", "3"):
        monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_REWIND", rw)
        runs.append(lookup_run(monkeypatch, None, lookup="0")[0])
    assert runs[0]["spec_accepted"] == runs[1]["spec_accepted"] and runs[0]["spec_typical_relaxed"] == runs[1]["spec_typical_relaxed"]
    assert runs[1].get("spec_typical_rewinds", 0) == 0


def test_engine_no_rewind_in_the_answer(monkeypatch):
    from kyojin_mlx.serve import engine as E
    monkeypatch.setattr(E.SPECSAM, "first_repeat", lambda hist, tok, *a, **k: 3)
    (toks, reason, usage), st, be = rw_run(monkeypatch, "3")
    assert st["spec_typical_rewinds"] >= 1
    # same session whose prompt already closed the think block: answer phase, nothing rewinds, signals only arm the window
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL_REWIND", "3")
    got, st2, _ = run({"a": ([1, 2, 3, OPEN, 7, CLOSE], dict(max_new=120, sampling=Sampling(1.0, 1.0, 0, 0.0, 11)))}, Soft(err=1), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert st2.get("spec_typical_rewinds", 0) == 0 and st2["spec_typical_cooldown_triggers"] >= 1


# ------------------------------------------------------------------------------------------------ flag think_temperature
def tt_run(monkeypatch, val, prompt, sampling=None, typical="0", n=40):
    from kyojin_mlx.serve import engine as E
    if val is None:
        monkeypatch.delenv("KYOJIN_THINK_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("KYOJIN_THINK_TEMPERATURE", val)
    monkeypatch.setenv("KYOJIN_SPEC_TYPICAL", typical)
    monkeypatch.setattr(E.Engine, "think_ids", (OPEN, CLOSE))
    seen, masks = [], []
    real, real_mask = E.sample_rows, E.SPECSAM.typical_mask

    def spy(lg, params, keys):
        seen.append([p.temperature for p in params])
        return real(lg, params, keys)

    def spy_mask(rows, drafts, eq, smp, *a):
        masks.append(smp.temperature)
        return real_mask(rows, drafts, eq, smp, *a)
    monkeypatch.setattr(E, "sample_rows", spy)
    monkeypatch.setattr(E.SPECSAM, "typical_mask", spy_mask)
    s = sampling or Sampling(1.0, 0.95, 0, 0.0, 11)
    got, st, _ = run({"a": (prompt, dict(max_new=n, sampling=s))}, Soft(err=1), sessions=1, mtp_depth=3, prefix_bytes=0)
    return got["a"][0], seen, masks


def test_think_temperature_registered():
    from kyojin_mlx import flags
    f = flags.FLAGS["think_temperature"]
    assert f.env == "KYOJIN_THINK_TEMPERATURE" and f.default == "" and f.runtime and f.kind == "float"
    assert flags._norm(f, "0.7") == "0.7"
    assert flags.FLAGS["spec_typical_cooldown"].default == "512"      # the neighbouring entry is still there


def test_think_temperature_off_is_identical(monkeypatch):
    for p in ([1, 2, 3, OPEN], [1, OPEN, 2, CLOSE, 3]):
        a, sa, _ = tt_run(monkeypatch, None, p)
        b, sb, _ = tt_run(monkeypatch, "", p)
        assert a == b and all(t == 1.0 for r in sa + sb for t in r)


def test_think_temperature_thinking_uses_flag_answer_uses_request(monkeypatch):
    _, seen, _ = tt_run(monkeypatch, "0.7", [1, 2, 3, OPEN])          # the whole reply is thinking
    assert seen and all(t == 0.7 for r in seen for t in r)
    _, seen, _ = tt_run(monkeypatch, "0.7", [1, OPEN, 2, CLOSE, 3])   # closed block: answer
    assert seen and all(t == 1.0 for r in seen for t in r)


def test_think_temperature_verify_and_typical_use_the_same_temperature(monkeypatch):
    _, seen, masks = tt_run(monkeypatch, "0.7", [1, 2, 3, OPEN], typical="1")
    assert any(len(r) > 1 for r in seen), "no verify round ran"
    assert all(t == 0.7 for r in seen for t in r) and masks and set(masks) == {0.7}
    _, seen, masks = tt_run(monkeypatch, "0.7", [1, 2, 3, 4], typical="1")
    assert all(t == 1.0 for r in seen for t in r) and set(masks) == {1.0}


def test_think_temperature_greedy_unaffected(monkeypatch):
    a, sa, _ = tt_run(monkeypatch, None, [1, 2, 3, OPEN], sampling=Sampling())
    b, sb, _ = tt_run(monkeypatch, "0.7", [1, 2, 3, OPEN], sampling=Sampling())
    assert a == b and all(t == 0.0 for r in sb for t in r)


def test_think_temperature_helper_ignores_bad_values(monkeypatch):
    from types import SimpleNamespace as NS
    from kyojin_mlx.serve import engine as E
    e = E.Engine.__new__(E.Engine)
    monkeypatch.setattr(E.Engine, "think_ids", (OPEN, CLOSE))
    s = NS(sampling=Sampling(1.0, 0.95, 20, 0.0, 5), ids=[1, OPEN], out=[], th_n=-1, th_in=False)
    for v in ("abc", "0", "-1"):
        monkeypatch.setenv("KYOJIN_THINK_TEMPERATURE", v)
        assert e._samp(s) is s.sampling
    monkeypatch.setenv("KYOJIN_THINK_TEMPERATURE", "0.7")
    r = e._samp(s)
    assert (r.temperature, r.top_p, r.top_k, r.seed) == (0.7, 0.95, 20, 5) and s.sampling.temperature == 1.0
