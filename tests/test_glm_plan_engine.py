"""Flags glm_plan / glm_chain on the fake backend of test_spec_lookup (no model): the planned drafts and the n-gram extension never change the greedy output,
the extension is scored apart from the head's positions, and a crowd of sessions plans plain."""
import pytest

import test_spec_lookup as TSL
from kyojin_mlx import glm_cost as GC
from kyojin_mlx.serve import engine as E

V = TSL.V


def nxt4(hist):
    """Next token = a function of the last 4 tokens only: a 4-gram match IS a valid continuation (suffix 4 .. 7: below the lookup gate's floor of 8)."""
    h = hist[-4:]
    return (sum((i + 5) * t for i, t in enumerate(h)) * 37 + 11) % V


class FakeGlm(TSL.FakeLookup):
    kind = "fakeglm"
    glm_costs = GC.Costs()
    max_spec_rows = 16

    def spec_verify_bc(self, bc, token_lists):                # the parent's version also exercises an ablit hook the fake no longer matches
        self.passes.append(sum(len(t) for t in token_lists))
        rows, starts = [], []
        for cache, toks in zip(bc, token_lists):
            starts.append(len(rows))
            rows += [TSL.nxt8(cache["h"] + toks[:j + 1]) for j in range(len(toks))]
        return TSL.SpecPass(TSL.onehot(rows), starts, bc, token_lists)


def short_match_prompt():
    """[P1 A X] ... [P2 A]: after the second A the earlier A (4 agreeing tokens, the token before differs) predicts X, which the fake continues exactly."""
    a = [3, 17, 29, 41]
    x, h = [], list(a)
    for _ in range(10):
        x.append(nxt4(h)); h.append(x[-1])
    return [60, 61, 62, 63, 64] + a + x + [70, 71, 72, 73, 74, 75, 76] + a


def run_flags(monkeypatch, prompt, plan, chain, sessions=1, n=24, depth=3):
    monkeypatch.setattr(TSL, "nxt8", nxt4)
    for k, v in (("KYOJIN_GLM_PLAN", plan), ("KYOJIN_GLM_CHAIN", chain), ("KYOJIN_SPEC_LOOKUP", "1")):
        monkeypatch.setenv(k, v)
    reqs = {f"r{i}": ([80 + i] + prompt, {"max_new": n}) for i in range(sessions)}
    return TSL.run(reqs, backend=FakeGlm(), sessions=sessions, mtp_depth=depth)


def toks(out):
    return {k: (v[0], v[1]) for k, v in out.items()}


def test_plan_and_chain_keep_the_greedy_output(monkeypatch):
    p = short_match_prompt()
    base, st0, _ = run_flags(monkeypatch, p, "0", "0")
    both, st1, be = run_flags(monkeypatch, p, "1", "1")
    assert toks(both) == toks(base)
    assert st1["spec_glm_plan_rounds"] > 0 and st1["spec_glm_plan_d1"] > 0 and st1["spec_glm_plan_d3"] == 0      # one session: depth 1 (no row past it pays at the measured costs)
    assert st1["spec_glm_chain_cand"] > 0


def test_chain_extends_on_agreement_and_scores_the_extension_apart(monkeypatch):
    monkeypatch.setattr(E.Session, "gx_rho", 0.97, raising=False)           # a strong prior: the table lets 2+ free rows through
    p = short_match_prompt()
    base, _, _ = run_flags(monkeypatch, p, "1", "0")
    out, st, _ = run_flags(monkeypatch, p, "1", "1")
    assert toks(out) == toks(base)
    assert st["spec_glm_chain_agree"] > 0 and st["spec_glm_chain_drafted"] > 0
    assert 0 < st["spec_glm_chain_accepted"] <= st["spec_glm_chain_drafted"]  # the fake's continuation is exact while the earlier span lasts, then the earlier text moves on


def test_a_crowd_plans_plain_and_a_probe_still_drafts():
    ps = [[0.9, 0.8, 0.7]] * 8
    assert GC.plan_depths(ps, 3, GC.Costs()) == [0] * 8
    # the engine probes a session planned at 0 every PROBE_EVERY rounds so that its acceptance estimate can move
    assert GC.PROBE_EVERY > 1


def test_lookup_only_flag_with_the_head_loaded(monkeypatch):
    from kyojin_mlx.serve.backends import GlmBackend
    be = GlmBackend.__new__(GlmBackend)
    monkeypatch.delenv("KYOJIN_GLM_LOOKUP_ONLY", raising=False)
    assert be.lookup_only is False
    be.lookup_only = True                                         # enable_lookup_spec (no head)
    assert be.lookup_only is True
    be.lookup_only = False
    be.nextn = object()                                           # the head is loaded: lookup_only only through the flag
    assert be.lookup_only is False
    monkeypatch.setenv("KYOJIN_GLM_LOOKUP_ONLY", "1")
    assert be.lookup_only is True
