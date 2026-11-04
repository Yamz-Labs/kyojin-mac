"""KYOJIN_VERIFY_ASYNC (hostsync): the served MTP round with fewer forced host syncs and fewer graph nodes, bit-identical to the current path.

1. Engine rounds on a fake lazy-draft backend (CPU, no model): greedy token streams equal with the flag off and on over many rounds and mixed accept
   lengths (drafter wrong every 1 / 3 positions, never; depth 1 / 3), and the blocking syncs per round drop (hostsync.Probe counts them).
2. Rows pass on the tiny synthetic DS4.1 model (CPU): logits and the committed state equal with the flag off and on over mixed row counts and
   accept lengths; the commit builds one index array per commit instead of one per layer.
3. Compiled mHC / FFN regions of the rows pass on a real slice (needs KYOJIN_SHRUNK=<dir with ds2/> and Metal): logits equal to the bit."""
import hashlib
import os
import sys
import types

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import hostsync as HS  # noqa: E402
from kyojin_mlx.serve.backends import LazyDrafts, VlmBackend  # noqa: E402
from test_serve_engine import V, nxt, onehot  # noqa: E402
from test_serve_spec import FakeSpec, SpecPass, prompts, run  # noqa: E402


class FakeLazy(FakeSpec):
    """FakeSpec with the lazy-draft surface of VlmBackend: proposals as GPU arrays, spec_verify_lazy, and the real flush_drafts / settle code."""
    kind = "fakelazy"
    _ring = staticmethod(lambda d: [] if d is None else [d["ring"]])

    def __init__(self, err=3, rows=8):
        super().__init__(err, rows)
        self.drafter = types.SimpleNamespace(conf_keep=lambda toks, conf, depths: list(depths))
        self.host = {}

    def new_draft(self):
        return {"fed": 0, "ring": mx.zeros((4,))}

    def _flush_lazy(self, ds):
        pass

    def _note_cut(self, cut):
        pass

    flush_drafts = VlmBackend.flush_drafts
    settle = VlmBackend.settle

    def draft_lazy(self, jobs):
        live = [i for i, j in enumerate(jobs) if j[3] > 0]
        if not live:
            return None
        width = max(jobs[i][3] for i in live)
        rows = []
        for i in live:
            d, pend, p0, depth, hist = jobs[i]
            t = self.draft(d, pend, p0, depth, hist)
            rows.append(t + [0] * (width - len(t)))
        self.host = {i: rows[k][:jobs[i][3]] for k, i in enumerate(live)}      # the fake verifies from the host copy (no read of the GPU arrays)
        toks = mx.array(rows, dtype=mx.int32) + 0
        conf = mx.ones((len(live), width)) * 0.9
        mx.async_eval(toks, conf)
        return LazyDrafts(self, jobs, live, toks, conf)

    def spec_verify_lazy(self, bc, lh, pendings):
        toks = [[p] + self.host.get(i, []) for i, p in enumerate(pendings)]
        return self.spec_verify_bc(bc, toks)


class RingPass(SpecPass):
    def commit(self, accepted, drafts=None, p0s=None):
        super().commit(accepted, drafts, p0s)
        for d in drafts or []:
            if d is not None:
                d["ring"] = d["ring"] + 1                                       # lazy state that settle must evaluate


_orig = FakeSpec.spec_verify_bc


def _verify_bc(self, bc, token_lists):
    sp = _orig(self, bc, token_lists)
    return RingPass(sp.logits, sp.starts, sp.bc, sp.token_lists)


FakeLazy.spec_verify_bc = _verify_bc


def _served(monkeypatch, on, err, depth):
    monkeypatch.setenv("KYOJIN_VERIFY_ASYNC", "1" if on else "0")
    monkeypatch.setenv("KYOJIN_SPEC_RATE", "0")                                 # sync counts per drafted round: the fake's 0 ms plain step sends the rate rule plain
    with HS.Probe() as pr:
        got, st, be = run(prompts(), FakeLazy(err=err), sessions=1, mtp_depth=depth, prefix_bytes=0, delay=0)
    return got, st, pr


@pytest.mark.parametrize("err", [0, 3, 1])
@pytest.mark.parametrize("depth", [1, 3])
def test_streams_equal_and_syncs_drop(monkeypatch, err, depth):
    off, st_off, p_off = _served(monkeypatch, False, err, depth)
    on, st_on, p_on = _served(monkeypatch, True, err, depth)
    for n in off:
        assert on[n][0] == off[n][0] and on[n][1] == off[n][1] == "length", n
    # no round-count equality: the served depth follows measured wall time (engine _learn_k / plain_ms), so two arms differ by a few rounds
    per = lambda p, st: p.blocking() / st["spec_rounds"]
    assert per(p_on, st_on) < per(p_off, st_off)
    if err == 0:                                                               # every round drafted: the full drop shows
        assert p_off.n["eval"] + p_off.n["tolist"] >= 4 * st_off["spec_rounds"]       # flush eval, vals / toks / conf tolist, settle eval
        assert p_on.n["eval"] + p_on.n["tolist"] <= 2.2 * st_on["spec_rounds"] + 20   # flush eval + one packed eval per round (+ the fake's own prompt work)
        assert p_on.n["read"] >= 3 * st_on["spec_rounds"] - 20                       # the copies are free reads, not waits
    print(f"err={err} depth={depth}: blocking syncs per round {per(p_off, st_off):.2f} -> {per(p_on, st_on):.2f}")


def test_async_settle_and_flush_do_not_wait(monkeypatch):
    be = FakeLazy()
    d = be.new_draft()
    d["ring"] = d["ring"] + 1
    for on in ("0", "1"):
        monkeypatch.setenv("KYOJIN_VERIFY_ASYNC", on)
        with HS.Probe() as pr:
            be.settle([], [d])
            be.flush_drafts([d])
        assert (pr.n["eval"], pr.n["async_eval"]) == ((2, 0) if on == "0" else (1, 1))      # settle goes async, flush stays blocking (the engine resets t0 after it)


def test_flag_in_table_default_off():
    from kyojin_mlx import flags
    assert flags.FLAGS["verify_async"].default == "0" and flags.FLAGS["rows_regions"].default == "0"


_SEEN: list = []


def _drive(model, on, monkeypatch, rounds=24):
    from kyojin_mlx import rows as R
    import mini_ds41
    monkeypatch.setenv("KYOJIN_VERIFY_ASYNC", "1" if on else "0")
    cache, _ = R.prefill(model, mini_ds41.tokens(40))
    bc = R.batch([cache])                                                        # one session: the stock modules of the tiny model batch one only
    rng = np.random.default_rng(3)
    h = hashlib.md5()
    _SEEN.clear()
    for r in range(rounds):
        Rr = [1, 4, 2, 3][r % 4]
        toks = [rng.integers(5, 90, Rr).tolist()]
        ps = R.verify(model, bc, toks)
        mx.eval(ps.logits)
        h.update(np.asarray(ps.logits.astype(mx.float32)).tobytes())
        acc = [min(r % 3, Rr - 1)]
        for l in range(4):                                                       # the stock modules leave no pending state: stage window rows as the patched core does
            ps.entry.pend[l] = types.SimpleNamespace(win=mx.array(rng.standard_normal((Rr, 2, 4)).astype(np.float32)), kvs=None, scs=None, cmp=None, keys=None)
        cls = type(ps.entry)
        if "spy" not in cls.__dict__:
            orig = cls.commit_layer

            def spy(self, l, last, p_last=None, idx=None, _o=orig):
                if last is not None:
                    _SEEN.append(idx)
                return _o(self, l, last, p_last, idx)
            cls.commit_layer, cls.spy = spy, True
        ps.commit(acc)
        arrs = [a for c in bc for a in c.stacked_arrays()] + [ps.entry.win[l] for l in range(4)]
        mx.eval(*arrs)
        for a in arrs:
            h.update(np.asarray(a.astype(mx.float32)).tobytes())
    return h.hexdigest(), list(_SEEN)


@pytest.fixture
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def test_rows_commit_state_equal_and_one_index_array(monkeypatch, cpu):
    import mini_ds41
    lm, _ = mini_ds41.make()
    model = types.SimpleNamespace(language_model=lm)
    d_off, seen_off = _drive(model, False, monkeypatch)
    d_on, seen_on = _drive(model, True, monkeypatch)
    assert d_on == d_off
    assert seen_off and all(i is None for i in seen_off)                       # off: every layer builds its own index array
    assert seen_on and all(i is not None for i in seen_on) and 2 * len({id(i) for i in seen_on}) <= len(seen_on)   # on: the layers of a commit share one


SHRUNK = os.environ.get("KYOJIN_SHRUNK")

_SLICE = r"""
import hashlib, os, sys
import numpy as np
import mlx.core as mx
from kyojin_mlx import fuse_layer as FL, patch as P, rows as R, spec as S
pack = sys.argv[1]
model, spec, cfg = P.build_model(pack)
P.patch_model(model, pack, spec, cfg)
R.enable(model, spec)
S.install(model, spec)
Vs = cfg["text_config"]["vocab_size"]
def go(flag):
    os.environ["KYOJIN_VERIFY_ASYNC"] = flag
    FL.ROWS = flag == "R"                                                       # regions only on the explicit KYOJIN_ROWS_REGIONS opt-in
    rng = np.random.default_rng(5)
    cache, _ = R.prefill(model, rng.integers(16, Vs - 16, 150).tolist())
    bc = R.batch([R.fork(cache)])
    h = hashlib.md5()
    for r in range(24):
        Rr = [1, 4, 2, 3][r % 4]
        ps = R.verify(model, bc, [rng.integers(16, Vs - 16, Rr).tolist()])
        mx.eval(ps.logits)
        h.update(np.asarray(ps.logits.astype(mx.float32)).tobytes())
        ps.commit([r % Rr])
    return h.hexdigest()
off = go("0"); calls_off = FL.STATS["call_rows"]
va = go("1"); calls_va = FL.STATS["call_rows"]                                  # VERIFY_ASYNC alone: no regions
on = go("R"); on2 = go("R")
print("RESULT", off == va == on == on2, calls_off, calls_va, FL.STATS["call_rows"])
"""


@pytest.mark.skipif(not mx.metal.is_available() or not SHRUNK, reason="needs Metal and KYOJIN_SHRUNK=<dir with ds2/>")
def test_rows_regions_equal_the_eager_rows_pass_on_a_slice():
    """Own process: the tiny CPU model of the other tests leaves a patched attention class and a CPU default device behind."""
    import subprocess
    pack = os.path.join(SHRUNK, "ds2")
    if not os.path.isdir(pack):
        pytest.skip(f"{pack} missing")
    r = subprocess.run([sys.executable, "-c", _SLICE, pack], capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join([os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("PYTHONPATH", "")])})
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert line, r.stderr[-2000:]
    _, same, calls_off, calls_va, calls_on = line[0].split()
    assert same == "True"                                                        # off, verify_async, regions, regions again: the same logits to the bit
    assert calls_off == "0" and calls_va == "0" and int(calls_on) > 0            # only the explicit opt-in enters the regions
