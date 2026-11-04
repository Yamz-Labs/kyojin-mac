"""flag lazy_fit: the lazy draft decision from two measured numbers (the draft wait the verify build hides, the rows the gate cuts), and the
engine side that measures them. No model: a stub backend runs the real VlmBackend.draft_lazy."""
import mlx.core as mx

from kyojin_mlx import flags as FL
from kyojin_mlx.serve import backends as B
from kyojin_mlx.serve.engine import Engine
from test_serve_engine import collect
from test_serve_spec import FakeSpec


class Stub:
    """Just enough of VlmBackend for draft_lazy: the cut ema, the measured budget, a drafter whose lazy pass returns arrays."""
    _cut_ema = 0.0
    lazy_budget = None

    def __init__(self, ema=0.0, budget=None):
        self._cut_ema, self.lazy_budget, self._lazy_skip = ema, budget, 0
        self.drafter = self

    def _flush_lazy(self, ds):
        pass

    def draft_many_lazy(self, ctxs, pendings, p0s, depths, want_logits=False):
        n = len(ctxs)
        self.depths = list(depths)
        return mx.zeros((n, 4), dtype=mx.int32), mx.zeros((n, 4), dtype=mx.float32)


JOBS = [({"ring": 1}, 7, 11, 3, None)]


def test_cut_rate_rule_still_runs_the_eager_path_when_no_budget_is_set():
    st = Stub(ema=0.9, budget=None)
    assert B.VlmBackend.draft_lazy(st, JOBS) is None                    # the old rule: too many cuts, no overlap
    st = Stub(ema=0.0, budget=None)
    assert isinstance(B.VlmBackend.draft_lazy(st, JOBS), B.LazyDrafts)


def test_the_measured_budget_decides_and_says_why():
    st = Stub(ema=0.9, budget=(9.0, 2.0))                               # the overlap hides more than the cut rows cost: lazy
    assert isinstance(B.VlmBackend.draft_lazy(st, JOBS), B.LazyDrafts) and st._lazy_skip == 0
    st = Stub(ema=0.0, budget=(1.0, 2.5))                               # the wasted rows cost more: eager, counted
    assert B.VlmBackend.draft_lazy(st, JOBS) is None and st._lazy_skip == 1


def test_a_depth_zero_pass_or_want_logits_keeps_the_old_path():
    st = Stub(ema=0.0, budget=(9.0, 2.0))
    assert B.VlmBackend.draft_lazy(st, [({"ring": 1}, 7, 11, 0, None)]) is None
    st = Stub(ema=0.9, budget=(1.0, 2.5))
    assert isinstance(B.VlmBackend.draft_lazy(st, JOBS, want_logits=True), B.LazyDrafts)   # spec_rs: no gate, the budget does not apply


def test_flag_is_off_by_default_and_read_per_round(monkeypatch):
    f = FL.FLAGS["lazy_fit"]
    assert f.default == "0" and f.runtime and f.env == "KYOJIN_LAZY_FIT"
    eng = Engine(FakeSpec(err=2), sessions=1, ctx=2048, chunk=64, block=64, mtp_depth=3)
    assert eng._lazy_fit is False
    monkeypatch.setenv("KYOJIN_LAZY_FIT", "1")
    assert eng._lazy_fit is True


class LazyDrafter:
    """Just enough of drafter.Drafter for the real draft_many_lazy: the block, a full ring, a tail-row count and the launch it records."""
    keep_margin = False
    window = 4
    block = 5

    def __init__(self):
        self.calls = []

    def _rows_cap(self):
        return 4

    def _tail_rows(self, depths):
        return max(1, min(self.block, max(int(d) for d in depths)))

    def _rows_launch(self, ctxs, pendings, p0s, want_logits=False, nrows=None):
        self.calls.append(nrows)
        n = len(ctxs)
        return mx.zeros((n, self.block), dtype=mx.int32), mx.zeros((n, self.block), dtype=mx.float32)


class Ctx:
    def __init__(self, window):
        self.rows = [mx.zeros((1, window), dtype=mx.float32)]


def test_lazy_pass_carries_the_planned_depths_into_the_tail_cut():
    """draft_tail_cut runs the draft tail on the deepest planned depth; the lazy pass must be told that depth (it read a local of another
    function and raised NameError on the first lazy round a lever ever took)."""
    from kyojin_mlx.dspark import Drafter
    d = LazyDrafter()
    ctxs = [Ctx(d.window), Ctx(d.window)]
    out = Drafter.draft_many_lazy(d, ctxs, [7, 9], [11, 12], [2, 4])
    assert d.calls == [4]                                       # the deepest plan of the pass, not the block
    assert out is not None and len(out) == 2
    Drafter.draft_many_lazy(d, ctxs, [7, 9], [11, 12], [3, 1], want_logits=True)
    assert d.calls[-1] is None                                  # spec_rs: the biased logits need the whole block


def test_engine_measures_the_two_sides_and_sets_the_budget(monkeypatch):
    """One session on the fake drafter: the engine updates its emas from the round's own sections and hands the backend (hiding ms, waste ms)."""
    monkeypatch.setenv("KYOJIN_LAZY_FIT", "1")
    be = FakeSpec(err=3)
    eng = Engine(be, sessions=1, ctx=2048, chunk=64, block=64, mtp_depth=3)
    eng.start_depth = 3
    seen = []
    real = getattr(be, "draft_lazy", None)

    def spy(jobs, want_logits=False):                                   # the fake has no draft_lazy: record the budget the backend would read
        seen.append(be.lazy_budget)
        return real(jobs, want_logits) if real is not None else None

    be.draft_lazy = spy
    eng.start()
    try:
        toks = collect(eng.submit([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], max_new=24))
    finally:
        eng.stop()
    toks, reason, _st = toks
    assert len(toks) >= 8 and seen, (toks, reason)
    assert all(isinstance(s, tuple) and len(s) == 2 for s in seen if s is not None), seen[:4]
    assert eng._dwait_ms >= 0.0 and eng._row_ms >= 0.0 and eng._cut_rows >= 0.0
    if eng.stats["spec_gate_cut"] == 0:                                 # no cut on the fake: the waste side is zero
        assert eng._cut_rows == 0.0
