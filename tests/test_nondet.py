# SPDX-License-Identifier: Apache-2.0
"""Run-to-run reproducibility (docs/reports/nondet_REPORT.md).

Two things make a T=0 request come out different on two runs of the same build:

  1. the served round width is chosen from MEASURED wall time (serve/engine.py `_plan_depth` -> `_plan_rate` at 1954, `_fit_cost` at 2033,
     dspol / the measured ladder), and a row's arithmetic moves with the width of the pass (`metal_src.HALF_ROWS`,
     `Exl3Linear.GEMV_MAX_ROWS` = 4, docs/cloud/LEVERS.md spec_ladder). flag `nondet_fix` takes the width from the request geometry only.
  2. a token id outside the fixture's vocabulary makes `mx.take` read PAST the embedding table: the engine is fine, the input is not.

Test 1 is CPU-only (a fake backend with injectable timings). Test 2 needs Metal and runs the same forward in fresh interpreters: the digest of
the logits and of the decode tokens has to be identical in every one of them (that is the shape the served md5 gate has).
"""
import json
import os
import subprocess
import sys

import mlx.core as mx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


# ------------------------------------------------------------------------------------------------------- 1. the plan may not read a clock
def _engine(monkeypatch, timings, depth=4):
    """An Engine on the timed fake backend of tests/test_depthpol_timing.py (no model, no Metal)."""
    from kyojin_mlx.serve.engine import Engine
    import test_depthpol_timing as TD
    be = TD.TimedFakeSpec(plain_ms=timings[0], draft_ms=timings[1], accept_probs=(0.9,) * 5, seed=3)
    return Engine(be, sessions=1, mtp_depth=depth, ctx=4096, chunk=4, block=8, prefix_bytes=0), be


def _session(eng, ids, max_new):
    from kyojin_mlx.serve.engine import Session
    from kyojin_mlx.serve.sampling import Sampling
    s = Session(sid=1, ids=list(ids), max_new=max_new, sampling=Sampling(), stop_ids=frozenset(), handle=None)
    eng.members = [s]
    return s


def _plan(monkeypatch, plain_ms: float, d_ms: list, cap_tok: float, depth: int = 4) -> int:
    """The depth the policy asks for on one round, with the measured tables written by hand: plain_ms = the median plain round, d_ms = the measured
    ms of the cap's rounds, cap_tok = its tokens per round, cap_n = the rounds of the window (RATE_MIN or more, or the policy drafts before it decides)."""
    from kyojin_mlx.serve import engine as E
    eng, be = _engine(monkeypatch, (plain_ms, {1: 30.0, 2: 42.0, 3: 54.0, 4: 66.0}), depth=depth)
    s = _session(eng, [1, 2, 3, 4, 5], 200)
    eng.plain_ms, eng._plain_ring = plain_ms, [plain_ms] * 5
    eng._d_ms[depth] = list(d_ms)
    s.cap_tok, s.cap_n = cap_tok, 8
    return eng._plan_depth(s)


def test_nondet_fix_takes_the_measured_clock_out_of_the_plan(monkeypatch):
    """Two identical requests whose rounds were measured at different ms get the same plan with nondet_fix=1, and not without it."""
    from kyojin_mlx.serve import engine as E
    slow = dict(plain_ms=22.0, d_ms=[30.0, 31.0], cap_tok=2.6)        # 2.6 x 22 = 57 tok/ms of a cap round beats 0.9 x 30.5: draft at the cap
    fast = dict(plain_ms=8.0, d_ms=[30.0, 31.0], cap_tok=0.9)         # 0.9 x 8 = 7.2 < 27.5: the same request goes plain
    monkeypatch.setattr(E, "NONDET_FIX", False)
    c, d = _plan(monkeypatch, **slow), _plan(monkeypatch, **fast)
    assert (c, d) == (4, 0), ("the measured-ms rule no longer decides (spec_rate's branch moved): the control arm of this test is gone", c, d)
    monkeypatch.setattr(E, "NONDET_FIX", True)
    a, b = _plan(monkeypatch, **slow), _plan(monkeypatch, **fast)
    assert a == b == 4, ("the plan moved with the measured ms under nondet_fix", a, b)


def test_nondet_fix_never_exceeds_the_room_or_the_model_width(monkeypatch):
    """The cap is still the request's room and max_spec_rows - 1: the flag removes the clock, not the bounds."""
    from kyojin_mlx.serve import engine as E
    monkeypatch.setattr(E, "NONDET_FIX", True)
    eng, be = _engine(monkeypatch, (22.0, {1: 30.0, 2: 42.0, 3: 54.0, 4: 66.0}), depth=8)
    s = _session(eng, [1, 2, 3], 4)                                          # room = 4 - 1 = 3 drafts
    assert eng._plan_depth(s) == 3
    be.max_spec_rows = 3
    assert eng._plan_depth(s) == 2


# ------------------------------------------------------------------------------------------------------- 2. fresh processes, same bits
CHILD = r"""
import hashlib, json, sys
sys.path.insert(0, %(root)r); sys.path.insert(0, %(tests)r)
import numpy as np, mlx.core as mx
import decstage_fixture as FX
from kyojin_mlx import archs, rows as RW

def dg(a):
    mx.eval(a)
    return hashlib.sha1(np.array(a).tobytes()).hexdigest()[:16]

m, _ = FX.build(**FX.REAL)
RW.enable(m, archs.arch_of({"model_type": "deepseek_v41"}))
pc = m.make_cache()
mx.eval(m(mx.array([FX.toks(8192, 5)]), cache=pc).logits)
cache = RW.batch([pc])
p = RW.verify(m, cache, [FX.toks(4, 900)])
mx.eval(p.logits)
ids = mx.argmax(p.logits, -1).astype(mx.int32)[-1:]
p.commit([3])
toks, lgs = [], []
for i in range(6):
    lg = RW.step(m, cache, ids)
    mx.eval(lg)
    ids = mx.argmax(lg, -1).astype(mx.int32)
    toks.append(int(ids[0])); lgs.append(dg(lg))
print(json.dumps({"verify": dg(p.logits), "toks": toks, "logits": lgs}))
"""


def _run_child():
    env = dict(os.environ)
    vlm = os.environ.get("VLM_PKG", os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
    env["PYTHONPATH"] = os.pathsep.join([ROOT, HERE, vlm])
    r = subprocess.run([sys.executable, "-c", CHILD % {"root": ROOT, "tests": HERE}], capture_output=True, text=True, env=env,
                       timeout=180)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.skipif(not os.path.isdir(os.environ.get("VLM_PKG", os.path.expanduser("~/kyojin-mlx/vlm_pkg"))), reason="no vendored mlx_vlm")
def test_forward_is_identical_in_fresh_processes():
    """The same forward (8K prefill, one 4-row verify, six plain steps) digest-and-token identical in three fresh interpreters.

    Before the fixture ids were fixed this failed about half the time: with list(range(5, 8197)) against the 128-row embedding table, 4 fresh
    processes of the same forward gave 3 distinct token strings (the ids past the table read other buffers of the process)."""
    runs = [_run_child() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2], runs
    assert runs[0]["toks"], runs[0]


def test_fixture_token_ids_stay_in_the_vocabulary():
    """The fixture's embedding table has vocab_size = 128 rows: a token id >= 128 makes mx.take read past it (see decstage_fixture.toks)."""
    import decstage_fixture as FX
    for n, start in ((300, 5), (4, 400), (4, 900), (8192, 5), (1, 0)):
        assert max(FX.toks(n, start)) < FX.DIM["vocab_size"], (n, start)


# ------------------------------------------------------------------------------------------------------- 3. a width is not a bit
@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_pass_width_moves_a_row(monkeypatch):
    """Why the plan decides the tokens: the same token, consumed in a 1-row pass and in a 4-row pass, is not bit-equal.

    First difference: layer 0 `q` = the EXL3 GEMV of the block input (Exl3Linear.GEMV_MAX_ROWS = 4, metal_src.HALF_ROWS), about 1e-3 relative on
    this random fixture, and it is enough to flip the indexer selection of the next layer (a different key set at layer 1). Both paths are
    deterministic on their own: rerunning one gives the same bits. docs/cloud/LEVERS.md, spec_ladder: "a width change can change the
    continuation (open fault, measured as an md5 change in orch5)"."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    import decstage_fixture as FX
    from kyojin_mlx import archs, rows as RW
    m, _ = FX.build(**FX.REAL)
    RW.enable(m, archs.arch_of({"model_type": "deepseek_v41"}))
    toks = FX.toks(4, 777)

    def fresh():
        pc = m.make_cache()
        mx.eval(m(mx.array([FX.toks(300, 5)]), cache=pc).logits)
        return RW.batch([pc])

    def verify(cache, tokens):
        p = RW.verify(m, cache, [tokens])
        mx.eval(p.logits)
        return mx.stop_gradient(p.logits)

    a = verify(fresh(), toks)
    b = verify(fresh(), toks)
    assert mx.array_equal(a, b).item(), "the 4-row pass is not deterministic on its own"
    c = fresh()
    step = []
    for t in toks:
        lg = RW.step(m, c, mx.array([t], dtype=mx.int32))
        mx.eval(lg)
        step.append(lg)
    one = mx.concatenate(step, axis=0)
    width_differs = not mx.array_equal(a, one).item()
    assert width_differs, ("a 1-row pass and a 4-row pass agree bit for bit now: the width fault is fixed, drop flag nondet_fix's reason and "
                           "re-check the served md5 gate (see docs/reports/nondet_REPORT.md)")
