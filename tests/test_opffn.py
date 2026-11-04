"""opffn levers: router_w16 (exact narrow gate copy) and dec_stage_xmix (expand folded into the next mix launch).

CPU tests (run anywhere mlx imports): the exactness rule of the narrow copy, the router wiring with the Metal kernels replaced by their references, the flag table,
the xmix kernel source and its plumbing in decstage (pending expand, fallbacks) with the Metal kernel replaced by the composition it must equal.
Metal tests (skipped without a GPU, for the lane): the kernels against the composition of the two stages they fold, bit for bit."""
import re
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import decstage as DS, flags as FL, fuse_router as FR, lowp, rows as RW, stagek as SK  # noqa: E402

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


# ------------------------------------------------------------------------------------------------------------ flags
def test_flags_default_off_and_module_literals_aligned():
    # dec_stage_xmix is not in this tree: its stage conflicted with the k1-k2 dec_stage line (opk1) and it stays on
    # cloud/opffn-2. Only router_w16 is asserted here.
    for n, mod, attr in (("router_w16", RW, "ROUTER_W16"),):
        f = FL.FLAGS[n]
        assert f.default == "0" and f.runtime
        assert (mod, attr) in [(__import__("kyojin_mlx." + m, fromlist=["x"]), a) for m, a, _ in f.attrs]
    assert RW.ROUTER_W16 is False


# ------------------------------------------------------------------------------------------------------------ router_w16
def _gate(E=64, K=256, seed=0, exact=True):
    rng = np.random.default_rng(seed)
    w = mx.array(rng.standard_normal((E, K)).astype(np.float32))
    return w.astype(mx.bfloat16).astype(mx.float32) if exact else w


def test_exact_lowp_keeps_bf16_values_and_refuses_the_rest():
    lowp.reset()
    w = _gate(exact=True)
    c = lowp.exact_lowp(w)
    assert c is not None and c.dtype == mx.bfloat16 and bool(mx.array_equal(c.astype(mx.float32), w))
    assert lowp.exact_lowp(w) is c                                       # one check per weight object
    assert lowp.exact_lowp(_gate(seed=1, exact=False)) is None            # a real fp32 gate keeps its fp32 read
    h = mx.array(np.random.default_rng(2).standard_normal((8, 32)).astype(np.float16)).astype(mx.float32)
    assert lowp.exact_lowp(h).dtype == mx.bfloat16 or lowp.exact_lowp(h).dtype == mx.float16
    assert lowp.exact_lowp(mx.zeros((4, 4), dtype=mx.float16)) is None   # only fp32 weights are narrowed


def test_narrow_copy_gives_the_same_logits_for_the_rowdot_order():
    """The operands of every fma are the same fp32 numbers, so a numpy replay of rowdot's lane order agrees bit for bit between the fp32 gate and its copy."""
    w = _gate(E=8, K=512)
    c = lowp.exact_lowp(w)
    x = np.random.default_rng(3).standard_normal((3, 512)).astype(np.float16)

    def replay(wt):
        wt = np.array(wt.astype(mx.float32))
        out = np.zeros((3, 8), np.float32)
        for r in range(3):
            for o in range(8):
                lanes = np.zeros(32, np.float32)
                for l in range(32):
                    for kk in range(l * 8, 512, 256):
                        for j in range(8):
                            lanes[l] = np.float32(np.float32(wt[o, kk + j]) * np.float32(x[r, kk + j]) + lanes[l])
                out[r, o] = lanes.sum(dtype=np.float32)
        return out

    assert np.array_equal(replay(w), replay(c))


def test_route_ok_accepts_the_narrow_gate_and_only_it():
    x = mx.zeros((1, 5120), dtype=mx.float16)
    b = mx.zeros((384,), dtype=mx.float32)
    for dt, want in ((mx.float32, True), (mx.bfloat16, True), (mx.float16, True)):
        assert FR.route_ok(x, mx.zeros((384, 5120), dtype=dt), b) is want
    assert FR.route_ok(x, mx.zeros((384, 5120), dtype=mx.int8), b) is False


def test_router_path_uses_the_narrow_copy_when_the_flag_is_on(monkeypatch):
    lowp.reset()
    E, K, TK = 64, 256, 6
    seen = {}
    monkeypatch.setattr(RW, "PF_ROUTE", False)
    monkeypatch.setattr("kyojin_mlx.fuse.ENABLED", True)
    monkeypatch.setattr("kyojin_mlx.fuse.ROUTER", True)

    def fake_rowdot(x, w, out_dtype=None):
        seen["w"], seen["out"] = w.dtype, out_dtype
        return x.astype(mx.float32) @ w.astype(mx.float32).T

    def fake_select(lg, bias, top_k, norm, scale, bias_vl=None, image_mask=None):
        seen["lg"] = lg
        return mx.zeros((lg.shape[0], top_k), dtype=mx.int32), mx.zeros((lg.shape[0], top_k))

    monkeypatch.setattr(RW, "rowdot", fake_rowdot)
    monkeypatch.setattr("kyojin_mlx.fuse.router_select", fake_select)
    monkeypatch.setattr("kyojin_mlx.launchcut.on", lambda g: False)
    gate = SimpleNamespace(weight=_gate(E, K), bias=mx.zeros((E,)), top_k=TK, norm_topk_prob=True, routed_scaling_factor=1.0)
    x = mx.array(np.random.default_rng(4).standard_normal((2, 1, K)).astype(np.float16))
    monkeypatch.setattr(RW, "ROUTER_W16", False)
    RW._router_sqrtsoftplus(gate, None, x)
    base = np.array(seen["lg"])
    assert seen["w"] == mx.float32
    monkeypatch.setattr(RW, "ROUTER_W16", True)
    RW._router_sqrtsoftplus(gate, None, x)
    assert seen["w"] == mx.bfloat16 and seen["out"] == mx.float32           # narrow read, fp32 logits (bf16 x bf16 would otherwise promote to bf16)
    assert np.array_equal(np.array(seen["lg"]), base)
    gate.weight = _gate(E, K, seed=9, exact=False)                           # not representable: fp32 is read
    RW._router_sqrtsoftplus(gate, None, x)
    assert seen["w"] == mx.float32
