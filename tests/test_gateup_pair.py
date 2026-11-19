# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_EXPERT_GATEUP_PAIR (flags.expert_gateup_pair): routed experts on stored bytes, decode slots.

  * gate | up in ONE gather pass (DevPack.gather_pair_raw, the stored bytes of both, one launch) instead of two gathers,
  * silu(gate) * up folded into the down launch's input staging (DevPack.gather_glu) where the down type has that kernel, else one activation launch,
  * the router-weighted sum over the K slots of GenericMoE in one kernel (moe_reduce) instead of astype / mul / sum / astype.
The plan is decided once per block (ForeignSwitchGLU._gu_plan, cached): no host work per call.

Numeric difference vs the flag off: the activation is computed in f32 inside the kernel (the stock path rounds gate and up to half first, then the product to half);
moe_reduce adds the K f32 products in slot order (the stock reduction may associate them differently). Not bit-identical: the gate is greedy ids.

CPU part: flag wiring, plan decisions on stubs (once, cached, each branch), the reduce arithmetic against the stock formula, kernel text.
Metal part (KYOJIN_METAL_TESTS=1, stage 0 of the c99 cell): moe_reduce vs stock; ForeignSwitchGLU flag on vs off on random q3_k / q4_k expert stacks.
"""
import os
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import flags, foreign_linear as FL, foreign_moe_generic as FMG

METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1" and mx.metal.is_available()
needs_metal = pytest.mark.skipif(not METAL, reason="KYOJIN_METAL_TESTS=1 (stage 0 of the c99 cell)")


def test_flag_wiring():
    f = flags.FLAGS["expert_gateup_pair"]
    assert f.env == "KYOJIN_EXPERT_GATEUP_PAIR" and f.default == "0" and f.runtime
    old = (FL.GATEUP_PAIR, FMG.GATEUP_PAIR, os.environ.get(f.env))
    g0 = flags.GEN
    try:
        flags.set("expert_gateup_pair", 1)
        assert FL.GATEUP_PAIR is True and FMG.GATEUP_PAIR is True and flags.GEN == g0 + 1
        flags.set("expert_gateup_pair", 0)
        assert FL.GATEUP_PAIR is False and FMG.GATEUP_PAIR is False
    finally:
        FL.GATEUP_PAIR, FMG.GATEUP_PAIR = old[0], old[1]
        if old[2] is None:
            os.environ.pop(f.env, None)
        else:
            os.environ[f.env] = old[2]


class _P:
    def __init__(self, fmt="q3_k", n=64, k=256, pair=True, g2a=False, glu=True, gemv=True):
        self.fmt, self.n, self.k, self.E = fmt, n, k, 4
        self._pair, self._g2a, self._glu, self._gemv = pair, g2a, glu, gemv
        self.calls = 0

    def gemv_ok(self):
        self.calls += 1
        return self._gemv

    def pair_ok(self, o):
        return self._pair

    def g2a_ok(self, pk=False):
        return self._g2a

    def glu_ok(self):
        return self._glu


def _mk(g, u, d, limit=None, act=None):
    m = FL.ForeignSwitchGLU.__new__(FL.ForeignSwitchGLU)
    m.__dict__.update(g=g, u=u, d=d, limit=limit, _act_kind=act)
    return m


def test_plan_branches():
    assert _mk(_P(n=64), _P(n=64), _P(n=256, k=64))._gu_plan() == "glu_down"
    assert _mk(_P(n=64), _P(n=64), _P(n=256, k=64, glu=False))._gu_plan() == "rows"             # down type without the glu kernel (q4_k class)
    assert _mk(_P(n=64), _P(n=64), _P(n=256, k=64), limit=7.0)._gu_plan() == "rows"             # clamped swiglu: the activation kernel carries the clamp
    assert _mk(_P(n=64), _P(n=64), _P(n=256, k=64), act=("situ", 1.0, 1.0))._gu_plan() == "rows"
    assert _mk(_P(n=62), _P(n=62), _P(n=256, k=62, glu=False))._gu_plan() == "stock"           # n % 4 != 0: no vector activation kernel
    assert _mk(_P(pair=False), _P(), _P(k=64))._gu_plan() is None                               # gate / up stored differently: today's path
    assert _mk(_P(g2a=True), _P(), _P(k=64))._gu_plan() is None                                 # nvfp4: its own pair path
    assert _mk(_P(gemv=False), _P(), _P(k=64))._gu_plan() is None
    assert _mk(_P(fmt="dense"), _P(), _P(k=64))._gu_plan() is None
    assert _mk(_P(n=64), _P(n=64), _P(n=256, k=128))._gu_plan() is None                         # down k != gate n: not a gated MLP


def test_plan_decided_once():
    m = _mk(_P(n=64), _P(n=64), _P(n=256, k=64))
    for _ in range(5):
        assert m._gu_plan() == "glu_down"
    assert m.g.calls == 1 and m.u.calls == 1 and m.d.calls == 1


def test_off_is_untouched(monkeypatch):
    """Flag off: __call__ never asks for the plan (no extra work on the old path)."""
    m = _mk(_P(), _P(), _P(k=64))
    monkeypatch.setattr(FL, "GATEUP_PAIR", False)
    monkeypatch.setattr(m, "_gu_plan", lambda: pytest.fail("plan asked with the flag off"), raising=False)
    src = open(FL.__file__).read()
    assert "if GATEUP_PAIR and S <= GATHER_MAX:" in src


def _stock_reduce(y, w, dt):
    return (y.astype(mx.float32) * w[..., None]).sum(-2).astype(dt)


def _slot_order(y, w, dt):
    """The kernel's arithmetic: f32 products (no fma), added in slot order."""
    y, w = np.array(y.astype(mx.float32)), np.array(w)
    acc = np.zeros((y.shape[0], y.shape[2]), np.float32)
    for k in range(y.shape[1]):
        acc = acc + (y[:, k, :] * w[:, k, None]).astype(np.float32)
    return mx.array(acc).astype(dt)


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16, mx.float32])
def test_reduce_arithmetic_vs_stock(dt):
    rng = np.random.default_rng(3)
    T, K, W = 3, 8, 512
    y = mx.array(rng.standard_normal((T, K, W)).astype(np.float32) * 4).astype(mx.float16)
    w = mx.array(rng.uniform(0.01, 1.5, (T, K)).astype(np.float32))
    a, b = _stock_reduce(y, w, dt), _slot_order(y, w, dt)
    a32, b32 = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    ulp = {mx.float16: 2 ** -10, mx.bfloat16: 2 ** -7, mx.float32: 2 ** -23}[dt]
    # bound: the two associations differ by <= (K - 1) f32 half-ulps of the partial sums, then one rounding to dt: at most 1 ulp of dt where they land on different sides
    assert np.all(np.abs(a32 - b32) <= 2 * ulp * np.maximum(np.abs(a32), 1e-3))
    if dt == mx.float32:
        assert np.max(np.abs(a32 - b32)) < 1e-5


def test_reduce_gates_off_cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)                                  # run the CPU part on any host: no skip
    try:
        y = mx.zeros((2, 8, 16), dtype=mx.float16)
        assert not FMG.moe_reduce_ok(y, mx.zeros((2, 8)))
    finally:
        mx.set_default_device(prev)


def test_reduce_kernel_text():
    assert "contract(off)" in FMG._RED_HDR and "acc += p;" in FMG._RED_SRC and "for (uint k = 0; k < K; k++)" in FMG._RED_SRC


# ---------------------------------------------------------------------------------------------------------------- Metal
def _rand_stack(tname, E, n, k, rng):
    from kyojin_mlx import foreign_gguf as GF
    bs, nb = {"Q3_K": (110, 108), "Q4_K": (144, 0)}[tname]
    row = k // 256 * bs
    packs = []
    for _ in range(E):
        raw = rng.integers(0, 256, (n, row), dtype=np.uint8)
        for b in range(k // 256):                                     # small finite f16 super-block scales
            o = b * bs + nb
            raw[:, o:o + 2] = np.array([0x00, 0x1c], np.uint8)        # f16 0x1c00 = 0.0039 (0.0625 saturated the f16 output: gate x up x down scale d^3)
            if tname == "Q4_K":
                raw[:, o + 2:o + 4] = np.array([0x00, 0x18], np.uint8)
        packs.append(GF.to_pack(tname, raw, k))
    return FL.DevPack(packs)


@needs_metal
def test_metal_reduce_vs_stock():
    rng = np.random.default_rng(5)
    for dt in (mx.float16, mx.float32):
        y = mx.array(rng.standard_normal((4, 8, 1000)).astype(np.float32)).astype(mx.float16)
        w = mx.array(rng.uniform(0.01, 1.5, (4, 8)).astype(np.float32))
        a, b = _stock_reduce(y, w, dt), FMG.moe_reduce(y, w, dt)
        mx.eval(a, b)
        d = np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32)))
        assert d.max() <= (2 ** -10 if dt == mx.float16 else 1e-5) * max(1.0, float(np.abs(np.array(a.astype(mx.float32))).max()))


@needs_metal
@pytest.mark.parametrize("types", [("Q3_K", "Q3_K", "Q3_K"), ("Q4_K", "Q4_K", "Q3_K"), ("Q3_K", "Q3_K", "Q4_K"), ("Q4_K", "Q4_K", "Q4_K")])
@pytest.mark.parametrize("rows", [1, 2])
def test_metal_pair_vs_stock(types, rows, monkeypatch):
    rng = np.random.default_rng(11)
    E, K, H, F = 16, 8, 512, 512      # F = 512: the down stack's k must be a whole gemv unit (k % 512), as Hy4's real experts are
    g, u, d = _rand_stack(types[0], E, F, H, rng), _rand_stack(types[1], E, F, H, rng), _rand_stack(types[2], E, H, F, rng)
    sw = FL.ForeignSwitchGLU(g, u, d)
    x = mx.array(rng.standard_normal((rows, 1, H)).astype(np.float32) * 0.5).astype(mx.float16)
    inds = mx.array(np.stack([rng.choice(E, K, replace=False) for _ in range(rows)])[:, None, :].astype(np.uint32))
    monkeypatch.setattr(FL, "GATEUP_PAIR", False)
    ref = sw(x, inds)
    monkeypatch.setattr(FL, "GATEUP_PAIR", True)
    plan = sw._gu_plan()
    assert plan in ("glu_down", "rows")
    out = sw(x, inds)
    mx.eval(ref, out)
    r, o = np.array(ref.astype(mx.float32)), np.array(out.astype(mx.float32))
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(o))
    # the activation is not rounded to half before the down product: relative to the output scale
    assert np.abs(r - o).max() <= 2e-2 * max(1e-3, float(np.abs(r).max()))
