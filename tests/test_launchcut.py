# SPDX-License-Identifier: Apache-2.0
"""launchcut (KYOJIN_LAUNCHCUT=1): CPU tests of the plumbing, the combine arithmetic and the launch counts; Metal tests behind KYOJIN_LAUNCHCUT_METAL=1.

Everything here runs on the CPU stream (no kernel is compiled or dispatched). The census tests run tools/launchcut_census.py in a subprocess (every
metal_kernel replaced by a one-node stub) and assert the dispatches per layer role. The Metal tests compare each fused kernel with the launches it replaces,
for 1 and 4 rows; run them on the Ultra:  KYOJIN_LAUNCHCUT_METAL=1 PYTHONPATH=.:vlm_pkg pytest tests/test_launchcut.py"""
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn

from kyojin_mlx import flags as FL
from kyojin_mlx import fuse_router as FR
from kyojin_mlx import launchcut as LC
from kyojin_mlx import rowinv as RI
from kyojin_mlx import suglue as SG

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METAL = os.environ.get("KYOJIN_LAUNCHCUT_METAL", "0") == "1"
metal_only = pytest.mark.skipif(not METAL, reason="Metal kernels: set KYOJIN_LAUNCHCUT_METAL=1 (run on the Ultra)")


@pytest.fixture(autouse=True)
def _cpu_default():
    """CPU by default inside this module; the caller's default device comes back after each test."""
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def same(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.fixture
def lc_on(monkeypatch):
    monkeypatch.setattr(LC, "ENABLED", True)
    monkeypatch.setattr(LC, "SKIP", set())


# ---------------------------------------------------------------------------------------------------------------------- flag
def test_flag_registered_default_off():
    assert FL.FLAGS["launchcut"].default == "0"
    assert os.environ.get("KYOJIN_LAUNCHCUT", "0") != "1" or LC.ENABLED
    assert LC.GROUPS == ("chain", "comb", "router")


def test_on_honours_enabled_and_skip(monkeypatch):
    monkeypatch.setattr(LC, "ENABLED", False)
    assert not any(LC.on(g) for g in LC.GROUPS)
    monkeypatch.setattr(LC, "ENABLED", True)
    monkeypatch.setattr(LC, "SKIP", {"comb"})
    assert LC.on("chain") and LC.on("router") and not LC.on("comb")
    with pytest.raises(AssertionError):
        LC.on("nope")


# ---------------------------------------------------------------------------------------------------------------------- comb arithmetic
def _bf16_round(f32):
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16                       # round to nearest even
    return (r.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("dt", ["float16", "bfloat16"])
@pytest.mark.parametrize("R", [1, 4])
def test_combine_ref_is_one_fp32_add_rounded_to_t(dt, R):
    """The prologue of stage E_comb reads float(T(y + float(sh))): the numpy rounding of that expression equals the stock combine."""
    rng = np.random.default_rng(R)
    D = 4096
    routed = (rng.standard_normal((R, 1, D)) * 0.3).astype(np.float32)
    shared_f = (rng.standard_normal((R, 1, D)) * 0.3).astype(np.float32)
    mdt = getattr(mx, dt)
    shared = mx.array(shared_f).astype(mdt)
    sh32 = np.array(shared.astype(mx.float32))
    s = routed + sh32
    want = s.astype(np.float16).astype(np.float32) if dt == "float16" else _bf16_round(s)
    got = np.array(LC.combine_ref(mx.array(routed), shared, mdt).astype(mx.float32))
    assert got.tobytes() == want.tobytes()


def test_source_changes_are_confined():
    """stage_e_comb differs from stage_e in the xv line only; the router kernel differs from rowdot in the logit store (+ the arrival and selection tails)."""
    e = SG._build_e()
    assert SG._XV_Y in e and SG._XV_COMB not in e
    out = e.replace(SG._XV_Y, SG._XV_COMB)
    assert out.count("sh[") == 1
    for masked in (False, True):
        src = FR._source(masked)
        assert FR._STORE not in src and FR._STORE_ATOM in src
        assert src.startswith(RI._ROWDOT.replace(FR._STORE, FR._STORE_ATOM))
        for name in ("MR", "KS", "YT", "E", "TK", "NORM"):                      # template names must not be redeclared by the kernel body
            assert f"constant uint {name}" not in src and f"const uint {name} " not in src
        assert ("bias_vl[e]" in src) == masked
        assert "atomic_load_explicit(lgp" in src and "y[ulong" not in src.split("selection")[1]      # the selection reads the other groups' logits atomically, never with plain loads


# ---------------------------------------------------------------------------------------------------------------------- router predicate / site
def test_route_ok_predicate():
    x = mx.zeros((4, 4096), dtype=mx.float16)
    w = mx.zeros((384, 4096), dtype=mx.float32)
    b = mx.zeros((384,), dtype=mx.float32)
    assert FR.route_ok(x, w, b)
    assert not FR.route_ok(x.astype(mx.float32), w, b)
    assert FR.route_ok(x, w.astype(mx.float16), b) and FR.route_ok(x, w.astype(mx.bfloat16), b)       # router_w16: the exact narrow copy of the gate
    assert not FR.route_ok(x, w.astype(mx.int8), b)
    assert not FR.route_ok(x, w, b.astype(mx.float16))
    assert not FR.route_ok(x, mx.zeros((380, 4096), dtype=mx.float32), mx.zeros((380,), dtype=mx.float32))   # E % 32
    assert not FR.route_ok(x, w, b, image_mask=mx.zeros((3,), dtype=mx.bool_))                                # mask size != rows
    assert FR.route_ok(x, w, b, image_mask=mx.zeros((4,), dtype=mx.bool_))


def _gate(R):
    g = types.SimpleNamespace(weight=mx.zeros((64, 128), dtype=mx.float32), bias=mx.zeros((64,), dtype=mx.float32), bias_vl=mx.zeros((64,), dtype=mx.float32),
                              top_k=6, norm_topk_prob=True, routed_scaling_factor=2.5)
    return g, mx.zeros((R, 1, 128), dtype=mx.float16)


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("R", [1, 4])
def test_router_site_dispatch(monkeypatch, on, R):
    from kyojin_mlx import fuse as F
    from kyojin_mlx import rows as RW
    calls = []
    ids, wts = mx.zeros((R, 6), dtype=mx.int32), mx.zeros((R, 6), dtype=mx.float32)
    monkeypatch.setattr(F, "ENABLED", True)
    monkeypatch.setattr(F, "ROUTER", True)
    monkeypatch.setattr(LC, "ENABLED", on)
    monkeypatch.setattr(LC, "SKIP", set())
    monkeypatch.setattr(FR, "route_ok", lambda *a, **k: True)
    monkeypatch.setattr(FR, "route", lambda *a, **k: calls.append("route") or (ids, wts))
    monkeypatch.setattr(RW, "rowdot", lambda *a, **k: calls.append("rowdot") or mx.zeros((R, 64)))
    monkeypatch.setattr(F, "router_select", lambda *a, **k: calls.append("select") or (ids, wts))
    g, x = _gate(R)
    out_i, out_w = RW._router_sqrtsoftplus(g, None, x)
    assert out_i.shape == (R, 1, 6) and out_w.shape == (R, 1, 6)
    assert calls == (["route"] if on else ["rowdot", "select"])


# ---------------------------------------------------------------------------------------------------------------------- ffn_stage_e plumbing
def _moe_cls():
    pytest.importorskip("mlx_vlm")
    L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")

    class FakeMoE(L.DeepseekV41MoE):
        def __init__(self, D, dtype):
            nn.Module.__init__(self)
            self.D, self.dt, self.n_calls = D, dtype, 0

        def gate(self, x, mask=None):
            n = x.shape[0]
            return mx.zeros((n, 1, 2), dtype=mx.int32), mx.ones((n, 1, 2), dtype=mx.float32)

        def switch_mlp(self, x, inds, scores):
            n = x.shape[0]
            return (mx.arange(n * 2 * self.D, dtype=mx.float32).reshape(n, 1, 2, self.D) * 0.01).astype(self.dt)

        def shared_experts(self, x):
            return (x * 0.5).astype(self.dt)

        def __call__(self, x, mask=None):
            self.n_calls += 1
            routed, shared = LC.moe_parts(self, x, mask)
            return LC.combine_ref(routed, shared, x.dtype)

    return FakeMoE


@pytest.mark.parametrize("R", [1, 4])
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("comb", [False, True])
def test_ffn_stage_e_plumbing(monkeypatch, comb, dt, R):
    FakeMoE = _moe_cls()
    D = 64
    ffn = FakeMoE(D, dt)
    yn = mx.ones((R, 1, D), dtype=dt)
    x2, zf = mx.zeros((R, 1, 4, D), dtype=dt), mx.zeros((R, 1, 24))
    seen = {}
    monkeypatch.setattr(SG, "stage_e", lambda blk, y, x2_, z: seen.update(e=y) or ("e",))
    monkeypatch.setattr(SG, "stage_e_comb", lambda blk, r, s, x2_, z: seen.update(r=r, s=s) or ("c",))
    monkeypatch.setattr(LC, "ENABLED", comb)
    monkeypatch.setattr(LC, "SKIP", set())
    res = LC.ffn_stage_e(None, ffn, yn, None, x2, zf)
    if comb:
        assert res == ("c",) and ffn.n_calls == 0
        assert seen["r"].dtype == mx.float32 and seen["s"].dtype == dt
        # what stage E_comb folds into its prologue equals the output of the stock FFN
        assert same(LC.combine_ref(seen["r"], seen["s"], dt), ffn(yn))
    else:
        assert res == ("e",) and ffn.n_calls == 1 and seen["e"].dtype == dt
    assert LC.comb_ok(ffn, yn) == comb
    assert not LC.comb_ok(ffn, yn[:, 0])                                   # not one token per row
    assert not LC.comb_ok(lambda x: x, yn)                                  # not the stock MoE


# ---------------------------------------------------------------------------------------------------------------------- census (dispatches per layer role)
def _census(tmp_path, rows):
    pkg = importlib.util.find_spec("mlx_vlm")
    if pkg is None:
        pytest.skip("mlx_vlm (vlm_pkg with deepseek_v41) not on PYTHONPATH")
    vlm = os.path.dirname(os.path.dirname(pkg.origin or list(pkg.submodule_search_locations)[0]))
    out = tmp_path / f"c{rows}.json"
    env = dict(os.environ, KYOJIN_VLM_PKG=vlm, PYTHONPATH=os.pathsep.join([ROOT, vlm]))
    for k in ("KYOJIN_LAUNCHCUT", "KYOJIN_PORT_SUGLUE", "KYOJIN_LAUNCHCUT_SKIP"):
        env.pop(k, None)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "launchcut_census.py"), "--layers", "6", "--arms", "sg,lc", "--moe", "--steps", "1", "--json", str(out)]
    if rows > 1:
        cmd += ["--rows", str(rows)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        pytest.skip("census cannot run here: " + r.stderr[-300:])
    return json.load(open(out))


def test_census_solo_plain_layer(tmp_path):
    c = _census(tmp_path, 1)
    sg, lc = c["sg"]["per_layer"], c["lc"]["per_layer"]
    assert [d for _, d in sg[3:]] == [22, 22, 22]
    assert [d for _, d in lc[3:]] == [20, 20, 20]                           # comb -1, router -1
    assert c["lc"]["dispatches"] < c["sg"]["dispatches"] and c["lc"]["nodes"] < c["sg"]["nodes"]


@pytest.mark.parametrize("rows", [2, 3, 4])
def test_census_chained_verify_plain_layer(tmp_path, rows):
    c = _census(tmp_path, rows)
    sg, lc = c["sg"]["per_layer"], c["lc"]["per_layer"]
    assert all(d == 30 for _, d in sg[4:]) and all(d == 20 for _, d in lc[4:])    # ten glue launches -> three, comb, router
    assert c["lc"]["dispatches"] < 0.7 * c["sg"]["dispatches"]


# ---------------------------------------------------------------------------------------------------------------------- Metal (Ultra)
@metal_only
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("R", [1, 4])
def test_stage_e_comb_matches_stage_e_of_combine(dt, R):
    from test_port_suglue import make_blk
    mx.set_default_device(mx.gpu)
    try:
        rng = np.random.default_rng(R)
        D = 4096
        blk = make_blk(rng, D, dt)
        routed = mx.array((rng.standard_normal((R, 1, D)) * 0.3).astype(np.float32))
        shared = mx.array((rng.standard_normal((R, 1, D)) * 0.3).astype(np.float32)).astype(dt)
        x2 = mx.array((rng.standard_normal((R, 1, 4, D)) * 0.05).astype(np.float32)).astype(dt)
        zf = mx.array(rng.standard_normal((R, 1, 24)).astype(np.float32))
        a = SG.stage_e_comb(blk, routed, shared, x2, zf)
        b = SG.stage_e(blk, LC.combine_ref(routed, shared, dt), x2, zf)
        mx.eval(a, b)
        assert all(same(p, q) for p, q in zip(a, b))
    finally:
        mx.set_default_device(mx.cpu)                                       # the autouse fixture restores the caller device


@metal_only
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("R", [1, 4])
@pytest.mark.parametrize("tied", [False, True])
def test_route_matches_rowdot_plus_select(masked, R, tied):
    mx.set_default_device(mx.gpu)
    try:
        rng = np.random.default_rng(R + 10 * masked)
        K, E, TK = 4096, 384, 6
        x = mx.array((rng.standard_normal((R, K)) * 0.5).astype(np.float32)).astype(mx.float16)
        w = rng.standard_normal((E, K)).astype(np.float32) * 0.02
        if tied:
            w[7] = w[3]                                                   # equal logits ...
        w = mx.array(w)
        bias = rng.standard_normal(E).astype(np.float32) * 0.1
        bvl = rng.standard_normal(E).astype(np.float32) * 0.1
        if tied:
            bias[7], bvl[7] = bias[3], bvl[3]                             # ... and equal keys: the tie goes to the lower index in both paths
        bias, bvl = mx.array(bias), mx.array(bvl)
        msk = mx.array(rng.integers(0, 2, R).astype(bool)).reshape(R, 1) if masked else None
        args = (TK, True, 2.5, bvl if masked else None, msk)
        assert FR.route_ok(x, w, bias, msk)
        for _ in range(100):                                                # the arrival counter must be back at zero after each launch; a race shows up here
            ids, wts = FR.route(x, w, bias, *args)
            rid, rw = FR.route_ref(x, w, bias, *args)
            mx.eval(ids, wts, rid, rw)
            assert same(ids, rid) and same(wts, rw)
    finally:
        mx.set_default_device(mx.cpu)                                       # the autouse fixture restores the caller device
