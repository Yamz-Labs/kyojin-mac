"""Flag dspol_v2 inside the engine, on a fake backend with a confidence head (needs mlx, no model): output equals plain decode for any R, the
default leaves the policy untouched, the plain step is the fallback when speculation does not pay, and the round trace carries p_k / R / tokens."""
import json

import pytest

from kyojin_mlx.serve import roundlog as RL
from test_serve_spec import FakeSpec, prompts, run, solo_all


class FakeConf(FakeSpec):
    """FakeSpec whose draft head reports a confidence per block position (block of 5); draft_confs() aligns to the jobs of the last draft_many."""

    def __init__(self, conf=0.99, **kw):
        super().__init__(**kw)
        self.conf, self.last = conf, None

    def draft_many(self, jobs):
        out = super().draft_many(jobs)
        self.last = [[self.conf] * 5 if j[3] > 0 else None for j in jobs]
        return out

    def draft_confs(self):
        return self.last


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv("KYOJIN_DSPOL_V2", raising=False)
    monkeypatch.setenv("KYOJIN_ROUND_LOG", "0")
    monkeypatch.setenv("KYOJIN_ROUND_LOG_PATH", str(tmp_path / "r.jsonl"))
    RL.reset_sink()
    yield
    RL.reset_sink()


def test_default_never_touches_the_v2_path():
    reqs = {"a": ([1, 2, 3], dict(max_new=40))}
    got, st, be = run(reqs, FakeConf(err=0), sessions=1, mtp_depth=5)
    assert not [k for k in st if "dsp2" in k]                              # status() carries the engine counters: nothing v2 ran


@pytest.mark.parametrize("mode", ["1", "2"])
@pytest.mark.parametrize("err", [0, 3, 1])
def test_v2_output_equals_plain(monkeypatch, mode, err):
    monkeypatch.setenv("KYOJIN_DSPOL_V2", mode)
    reqs = prompts(3)
    want = solo_all(reqs)
    for n, r in reqs.items():
        got, st, be = run({n: r}, FakeConf(conf=0.99 if err == 0 else 0.5, err=err), sessions=1, mtp_depth=5, prefix_bytes=0)
        assert got[n][0] == want[n][0] and got[n][1] == "length", (n, mode, err)


@pytest.mark.parametrize("mode", ["1", "2"])
def test_v2_widens_on_good_drafts(monkeypatch, mode):
    monkeypatch.setenv("KYOJIN_DSPOL_V2", mode)
    reqs = {"a": ([1, 2, 3], dict(max_new=120))}
    got, st, be = run(reqs, FakeConf(conf=0.99, err=0), sessions=1, mtp_depth=5)
    assert st["spec_accepted"] > 0 and st["spec_rounds"] < 120 // 3
    assert max(be.passes) >= 5


@pytest.mark.parametrize("mode", ["1", "2"])
def test_v2_falls_back_to_plain_when_drafts_never_land(monkeypatch, mode):
    monkeypatch.setenv("KYOJIN_DSPOL_V2", mode)
    reqs = {"a": ([1, 2, 3], dict(max_new=160))}
    got, st, be = run(reqs, FakeConf(conf=0.6, err=1), sessions=1, mtp_depth=5, probe_every=8)
    assert got["a"][0] == run(reqs, sessions=1)[0]["a"][0]
    assert st["spec_d0"] >= 100, st["spec_d0"]                       # the calibration learned 0.6 is wrong: most rounds are the plain step
    assert 1 in be.depth_asked[2:]                                         # and it keeps probing


@pytest.mark.parametrize("mode", ["1", "2"])
def test_round_trace_carries_pk_R_and_realised_tokens(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("KYOJIN_DSPOL_V2", mode)
    monkeypatch.setenv("KYOJIN_ROUND_LOG", "1")
    reqs = {"a": ([1, 2, 3], dict(max_new=80))}
    got, st, be = run(reqs, FakeConf(conf=0.97, err=3), sessions=1, mtp_depth=5)
    recs = [json.loads(l) for l in open(tmp_path / "r.jsonl") if l.startswith("{") and '"k"' in l]
    rr = [r for r in recs if r["k"] == "r"]
    assert rr and all("pk" in r and "R" in r and "et" in r and "cms" in r for r in rr[1:])
    for r in rr[1:]:
        assert len(r["pk"]) == r["R"] - 1 and r["R"] - 1 >= r["draft"] - r["cut"] - 0 and r["tok"] == r["acc"] + 1
    from kyojin_mlx import dspol2 as D
    rep = D.calibration_report(recs)
    assert rep["et"]["rounds"] > 0 and 0.0 <= rep["ece"] <= 1.0


class FakeStep(FakeConf):
    """FakeConf with a plain decode step (step_feed): a round with nothing to verify is the plain step of the batched cache, as on VlmBackend."""

    def step_feed(self, bc, tokens, drafts, p0s):
        return self.step(bc, tokens)


def test_mode2_runs_the_plain_step_when_the_cut_leaves_nothing_to_verify(monkeypatch):
    monkeypatch.setenv("KYOJIN_DSPOL_V2", "2")
    reqs = {"a": ([1, 2, 3], dict(max_new=100))}
    got, st, be = run(reqs, FakeStep(conf=0.5, err=1), sessions=1, mtp_depth=5, probe_every=8)
    assert got["a"][0] == run(reqs, sessions=1)[0]["a"][0]
    assert st["dsp2_plain_after_draft"] >= 1                               # the cold round drafted, the calibrated p_k said plain: no one-row verify
    assert st["spec_d0"] >= 60
    assert max(be.passes, default=0) <= 6


def test_mode2_files_the_round_under_the_width_it_verified(monkeypatch):
    """The cut changes the verify width: depths, the block-cut counter and the engine-wide measured cost fit (_d_ms, read by the rate rule of every
    other arm on the same server) must see the verified width, not the planned one."""
    import time
    from kyojin_mlx.serve.engine import Engine
    from test_serve_engine import collect
    monkeypatch.setenv("KYOJIN_DSPOL_V2", "2")
    be = FakeConf(conf=0.9, err=3)
    eng = Engine(be, sessions=1, ctx=4096, chunk=4, block=8, mtp_depth=5)
    eng.start()
    try:
        collect(eng.submit([1, 2, 3], max_new=120))
    finally:
        eng.stop()
    assert eng.stats["dsp2_cut_rows"] > 0                                  # the cut did fire
    assert eng.stats["spec_block_cut"] == 0
    verified = {p - 1 for p in be.passes}
    assert set(eng._d_ms) <= verified, (set(eng._d_ms), verified)
