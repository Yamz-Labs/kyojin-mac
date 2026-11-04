"""pf_moe_one_pass: one pass over the expert banks per MoE call (up to the 8192-row cap), off by default.

The pass structure is what the change moves: a prefill chunk handed to `MoePrefill` in two 4096-row passes builds the plan, the device sort,
the `cum` readback and one `mx.eval` twice, and each pass re-decodes the trellis of every expert it touches. With the flag the call is one
pass (up to the cap), the same arithmetic per row. Checked here: the pass split, the row cap, the default, and that the routed output is
unchanged bit for bit against the two-pass path (same tiles, same per-row arithmetic).
"""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
from test_moe import make_stack, E, TOPK, KS  # noqa: E402
from test_pf_small import route, xs  # noqa: E402
from kyojin_mlx import flags as FL, moe as M, moe_prefill as MP  # noqa: E402


def _moe(cb="mul1", kin=256, ninter=384):
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return M.MoeGather(g, u, d, TOPK)


def test_flag_is_off_by_default():
    assert FL.FLAGS["pf_moe_one_pass"].default == "0"
    assert FL.FLAGS["pf_moe_one_pass"].env == "KYOJIN_PF_ONE_PASS"
    assert MP.ONE_PASS in (False, True)                     # the module attribute the flag writes


def test_pass_rows_follow_the_flag(monkeypatch):
    pf = MP.MoePrefill(_moe(), chunk=4096)
    monkeypatch.setattr(MP, "ONE_PASS", False)
    assert pf.pass_rows(8192) == 4096                       # off: the chunk size
    assert pf.pass_rows(128) == 4096                        # a tail call is one pass anyway (T <= chunk)
    monkeypatch.setattr(MP, "ONE_PASS", True)
    assert pf.pass_rows(8192) == 8192                       # on: the whole call ...
    assert pf.pass_rows(16384) == MP.PASS_CAP == 8192       # ... up to the cap (16384-row passes panicked the Ultra)
    assert pf.pass_rows(128) == 128
    explicit = MP.MoePrefill(_moe(), chunk=2048)
    assert explicit.pass_rows(8192) == 8192                 # the flag wins over an explicit chunk size


def test_one_pass_matches_two_passes(monkeypatch):
    """Same routed output as the two-pass path (same tiles, same per-row arithmetic), on rows > one pass."""
    kin = 256
    moe = _moe("mcg", kin)
    T = E * TOPK                                        # one row per expert, so both arms touch every expert
    ids, wts = route(T, 5)
    x = xs(T, kin, 6)
    mx.eval(ids, wts, x)

    monkeypatch.setattr(MP, "ONE_PASS", False)
    two = MP.MoePrefill(moe, chunk=128)
    calls = []
    orig = MP.MoePrefill._chunk
    monkeypatch.setattr(MP.MoePrefill, "_chunk", lambda self, *a: calls.append(a[0].shape[0]) or orig(self, *a))
    y2 = np.array(two(x, ids, wts))
    assert len(calls) == (T + 127) // 128                   # off: one call per 128-row pass

    calls.clear()
    monkeypatch.setattr(MP, "ONE_PASS", True)
    one = MP.MoePrefill(moe, chunk=128)
    y1 = np.array(one(x, ids, wts))
    assert calls == [T]                                     # on: one pass for the whole call
    assert np.array_equal(y1.view(np.uint32), y2.view(np.uint32))
