"""Flag lazy_trim on the engine with a fake draft head (no model, no Metal): the draft ask follows what the confidence gate keeps; the output is the
plain decode's, the rows verified per pass fall."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from test_serve_spec import FakeSpec, run, solo_all  # noqa: E402
from kyojin_mlx import flags as FL  # noqa: E402


class FakeLazy:
    """What backends.LazyDrafts hands the engine: the PLANNED proposals (the verify is built at that depth) and, once read, how many the gate keeps."""
    is_lookup = False

    def __init__(self, be, jobs):
        self.be = be
        self.depths = [j[3] for j in jobs]
        self.counts = list(self.depths)
        self.drafts = [FakeSpec.draft(be, j[0], j[1], j[2], j[3], j[4]) if j[3] > 0 else [] for j in jobs]
        self.keep = None
        self.confs = None

    def tolist(self):
        self.keep = [min(len(d), self.be.keep) for d in self.drafts]
        self.confs = [None] * len(self.drafts)
        return self.drafts


class Gated(FakeSpec):
    """A draft head whose confidence gate keeps at most `keep` proposals of a round (prose: the head is sure of the first token or two).
    Eager: the gate cuts before the verify. LazyGated: the verify is built at the planned depth and the gate cuts after it."""

    def __init__(self, keep, err=0):
        super().__init__(err=err)
        self.keep = keep

    def draft(self, d, pending, p0, depth, history):
        return super().draft(d, pending, p0, depth, history)[:self.keep]


class LazyGated(Gated):
    def draft_lazy(self, jobs, want_logits=False):
        return FakeLazy(self, jobs) if any(j[3] > 0 for j in jobs) else None

    def spec_verify_lazy(self, bc, lh, pendings):
        return self.spec_verify_bc(bc, [[p] + d for p, d in zip(pendings, lh.drafts)])


REQS = {"g": ([1, 2, 3, 4, 5, 6], dict(max_new=80))}


def go(monkeypatch, on, be, **kw):
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    monkeypatch.setenv("KYOJIN_LAZY_TRIM", "1" if on else "0")
    return run(REQS, be, sessions=1, mtp_depth=5, depth_override=5, prefix_bytes=0, **kw)


def test_flag_registered_off_by_default():
    f = FL.FLAGS["lazy_trim"]
    assert f.default == "0" and f.runtime


def test_output_is_the_plain_decode_and_the_ask_follows_the_gate(monkeypatch):
    want = solo_all(REQS)["g"][0]
    base, st0, be0 = go(monkeypatch, False, Gated(1))
    got, st1, be1 = go(monkeypatch, True, Gated(1))
    assert got["g"][0] == base["g"][0] == want and len(want) == 80
    assert st0.get("spec_trim_rows", 0) == 0 and max(be0.depth_asked) == 5
    assert st1["spec_trim_rows"] > 0 and st1["spec_trim_rounds"] > 0
    assert sorted(set(be1.depth_asked)) == [2, 5]                            # ceil(1) + 1 between the untrimmed probes


def test_a_lazy_verify_pays_fewer_rows_for_the_same_tokens(monkeypatch):
    want = solo_all(REQS)["g"][0]
    base, st0, be0 = go(monkeypatch, False, LazyGated(1))
    got, st1, be1 = go(monkeypatch, True, LazyGated(1))
    assert got["g"][0] == base["g"][0] == want
    assert st0["spec_lazy_rounds"] > 0 and st1["spec_lazy_rounds"] > 0
    assert be0.passes.count(6) > len(be0.passes) // 2                        # today: the planned depth is verified (the last rounds are shorter), the gate keeps one proposal
    assert sum(be1.passes) < 0.7 * sum(be0.passes) and max(be1.passes) == 6  # the probe round still asks the whole plan


def test_a_gate_that_keeps_nothing_asks_one_row(monkeypatch):
    got, st, be = go(monkeypatch, True, Gated(0))
    assert got["g"][0] == solo_all(REQS)["g"][0]
    assert sorted(set(be.depth_asked)) == [1, 5]


def test_a_head_that_keeps_every_ask_is_not_trimmed(monkeypatch):
    got, st, be = go(monkeypatch, True, FakeSpec(err=0))
    assert got["g"][0] == solo_all(REQS)["g"][0]
    assert st.get("spec_trim_rows", 0) == 0 and set(be.depth_asked) == {5} or max(be.depth_asked) == 5


def test_the_ask_grows_back_when_the_gate_opens(monkeypatch):
    class Opens(Gated):
        def draft(self, d, pending, p0, depth, history):
            self.keep = 1 if p0 < 40 else 5
            return super().draft(d, pending, p0, depth, history)

    be = Opens(1)
    got, st, be = go(monkeypatch, True, be)
    assert got["g"][0] == solo_all(REQS)["g"][0]
    late = be.depth_asked[-6:]
    assert max(late) == 5 and sum(1 for x in late if x == 5) >= 3            # the EMA climbed back to the full ask


def test_oracle_and_full_modes_are_left_alone(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_FULL", "1")
    got, st, be = go(monkeypatch, True, Gated(1))
    assert st.get("spec_trim_rows", 0) == 0 and got["g"][0] == solo_all(REQS)["g"][0]
