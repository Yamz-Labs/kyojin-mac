"""The gated lazy verify (KYOJIN_GATED_VERIFY=1, VlmBackend._verify_gated): call order with fakes, the fallbacks to the ungated verify, the
verify-shaped row fill of an EngramStage, and on Metal (subprocess, one command buffer per eval) the held pass against the stock row embed."""
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kyojin_mlx import engram_mlx as EM
from kyojin_mlx.serve import backends as B

CGATE = os.path.join(HERE, "..", "tools", "commitgate", "libcommitgate.dylib")


class Stage:
    def __init__(self, calls, cg_k=3, rows=True, fail=False):
        self.calls, self.cg_k, self.rows, self.fail, self.pending = calls, cg_k, rows, fail, []

    def fill(self, host):
        self.calls.append(("fill", host.tolist()))
        if self.fail:
            raise OSError("pread failed")
        self.pending = []

    def resync(self):
        self.calls.append("resync")


class Gate:
    def __init__(self, stage):
        self.stage = stage

    def reset(self, bc):
        self.stage.calls.append("reset")


class CG:
    def __init__(self, calls):
        self.calls = calls

    def cg_hold(self, n):
        self.calls.append(("hold", n))

    def cg_release(self):
        self.calls.append("release")
        return 0


def backend(monkeypatch, calls, flag=True, **kw):
    st = Stage(calls, **kw)

    class R:
        @staticmethod
        def verify(model, bc, xs, stage=None):
            calls.append(("verify", stage is not None))
            if stage is not None and stage.rows:
                stage.pending = ["slot"]
            ps = type("Pass", (), {})()
            ps.logits, ps.taps, ps.starts, ps.eh = "logits", {"t": "tap"}, [0], mx.array([[1, 2]])
            ps.entry = type("Entry", (), {"_counts": [3], "pend": {"k": 1}})()
            be.entries.append(ps.entry)
            return ps

    monkeypatch.setattr(B, "GATED_VERIFY", flag)
    monkeypatch.setattr(EM, "_commit_gate", lambda: CG(calls))
    monkeypatch.setattr(B.mx, "async_eval", lambda *a: calls.append(("async_eval", [x if isinstance(x, str) else "eh" for x in a])))
    be = object.__new__(B.VlmBackend)
    be.gated, be.R, be.model, be.entries = Gate(st), R, None, []
    lh = object.__new__(B.LazyDrafts)
    lh.toks, lh.slot, lh.depths = mx.array([[7, 8, 9]]), {0: 0}, [2]
    return be, lh


def test_gated_verify_holds_the_pass_until_the_rows_are_in(monkeypatch):
    calls = []
    be, lh = backend(monkeypatch, calls)
    bp = be.spec_verify_lazy([None], lh, [5])
    assert calls == ["reset", ("verify", True), ("async_eval", ["eh"]), ("hold", B.VERIFY_HOLD), ("async_eval", ["logits", "tap"]),
                     ("fill", [[1, 2]]), "release", "resync"]
    assert bp.logits == "logits" and be.stats_gated_verify == 1


@pytest.mark.parametrize("why", ["flag off", "no calibrated gate", "no Engram row", "verify replay"])
def test_gated_verify_falls_back_to_the_ungated_pass(monkeypatch, why):
    calls = []
    kw = {"flag off": {"flag": False}, "no calibrated gate": {"cg_k": None}, "no Engram row": {"rows": False}, "verify replay": {}}[why]
    be, lh = backend(monkeypatch, calls, **kw)
    from kyojin_mlx import verify_replay as VR
    monkeypatch.setattr(VR, "ENABLED", why == "verify replay")
    be.spec_verify_lazy([None], lh, [5])
    assert not any(c == "release" or (isinstance(c, tuple) and c[0] in ("hold", "fill")) for c in calls), calls
    assert ("verify", why == "no Engram row") in calls


def test_gated_verify_releases_and_turns_off_when_the_fill_fails(monkeypatch):
    calls = []
    be, lh = backend(monkeypatch, calls, fail=True)
    with pytest.raises(OSError):
        be.spec_verify_lazy([None], lh, [5])
    assert calls[-2:] == [("fill", [[1, 2]]), "release"] and B.GATED_VERIFY is False and be.gated.stage.pending == []
    assert be.entries[0]._counts is None and be.entries[0].pend == {}                # the uncommitted pass left no pending state


def test_stage_fills_verify_shaped_rows(tmp_path):
    """A verify pass reads L > 1 positions per session: the fill takes [B, L, layer, cols] hash ids, bit for bit the stock embed."""
    from kyojin_mlx import engram_rows as ER
    from test_engram_rows import make_shards, spec_with_rows
    sp, rows = spec_with_rows()
    make_shards(tmp_path, sp, rows)
    t = ER.EngramTables(str(tmp_path), sp)
    stage = EM.EngramStage(t)
    emb, ref = EM.StagedRowEmbedding(stage, 1), EM.ParallelRowEmbedding(t, 1)
    idx = np.random.default_rng(5).integers(0, rows[1], (2, 4, 24))
    want = np.array(ref(mx.array(idx)).astype(mx.float32))
    with stage.deferred():
        lazy = emb(mx.array(idx))
    host = np.zeros((2, 4, 2, 24), np.int64)
    host[:, :, 1, :] = idx
    assert stage.fill(host) == idx.size
    assert np.array_equal(np.array(lazy.astype(mx.float32)), want)
    t.close()


METAL_SCRIPT = """
import sys, numpy as np
import mlx.core as mx
sys.path.insert(0, sys.argv[2])
from kyojin_mlx import engram_mlx as EM, engram_rows as ER
from kyojin_mlx.serve import backends as B
from test_engram_rows import make_shards, spec_with_rows
sp, rows = spec_with_rows()
import pathlib
make_shards(pathlib.Path(sys.argv[1]), sp, rows)
t = ER.EngramTables(sys.argv[1], sp)
stage = EM.EngramStage(t)
stage.poison = True                                                # a pass that ran before its rows are in gives NaN
emb, ref = EM.StagedRowEmbedding(stage, 1), EM.ParallelRowEmbedding(t, 1)
B.GATED_VERIFY = True
rng = np.random.default_rng(6)
x = mx.random.normal((1024, 1024))
mx.eval(x)
nohold = sys.argv[3] == "nohold"                                   # the control: the same pass with a gate that holds nothing
if nohold:
    class NoHold:
        def cg_hold(self, n): pass
        def cg_release(self): return 0
        def cg_seen(self): return 0
    EM._commit_gate = lambda: NoHold()
stage.cg_k = 1
be = object.__new__(B.VlmBackend)
early = 0
for it in range(4):
    idx = rng.integers(0, rows[1], (2, 4, 24))
    host = np.zeros((2, 4, 2, 24), np.int64)
    host[:, :, 1, :] = idx
    want = np.array(ref(mx.array(idx)).astype(mx.float32).sum(-1))
    d = x
    for _ in range(40):                                            # the drafter: long GPU work submitted before the verify
        d = mx.tanh(d @ x)
    mx.async_eval(d)
    class R:
        @staticmethod
        def verify(model, bc, xs, stage=None):
            with stage.deferred():
                eh = mx.array(host) + (d[0, 0] > 1e30).astype(mx.int64)   # the hash ids wait for the drafts, as the drafted tokens do
                out = emb(eh[:, :, 1, :]).astype(mx.float32).sum(-1)
            ps = type("Pass", (), {})()
            ps.logits, ps.taps, ps.starts, ps.eh = out, {}, [0], eh
            return ps
    be.gated, be.R, be.model = type("G", (), {"stage": stage})(), R, None
    bp = be._verify_gated(None, [])
    got = np.array(bp.logits)
    if nohold:
        early += bool(np.isnan(got).any())
        continue
    assert np.array_equal(got, want), (it, got, want)
    assert EM._commit_gate().cg_release() == 0, "a buffer stayed held"
print("NOHOLD EARLY", early) if nohold else print("GATED VERIFY OK", be.stats_gated_verify)
"""


def metal(tmp_path, mode):
    os.makedirs(tmp_path / mode)
    env = dict(os.environ, MLX_MAX_OPS_PER_BUFFER="100000", MLX_MAX_MB_PER_BUFFER="1000000",
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, ".."), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", METAL_SCRIPT, str(tmp_path / mode), HERE, mode], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-1500:]
    return r.stdout


@pytest.mark.skipif(not os.path.exists(CGATE), reason="tools/commitgate/libcommitgate.dylib not built")
def test_gated_verify_on_metal_equals_the_stock_embed(tmp_path):
    assert "GATED VERIFY OK 4" in metal(tmp_path, "hold")
    out = metal(tmp_path, "nohold")                                 # the control: without the hold the pass reads the poisoned slots
    assert "NOHOLD EARLY" in out and not out.strip().endswith(" 0"), out
