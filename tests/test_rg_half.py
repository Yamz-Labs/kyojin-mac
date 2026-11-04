"""rg_half (KYOJIN_RG_HALF, default 0): the many-rows GEMV of a 5..8 row block multiplies-accumulates in packed fp16 instead of fp32 rows, the class the
<= MAX_ROWS fused launch takes from HALF_ROWS rows on.

The ledger's per-kernel table (opgpu) puts the dense EXL3 linears at 28 % of a 6-row verify round and calls them the class with the largest slope (+2.0 ms per
row); above GEMV_MAX_ROWS (4) a block leaves the fused launch, and the many-rows GEMV it lands on (`gemv_rg_source`) is fp32 rows today while the fused
launch from HALF_ROWS rows on is packed fp16. This file checks (a) the flag and the kernel text without a GPU, (b) the gate the module applies, and (c) on
Metal, that a row of the half many-rows launch is bit for bit the row of the fused launch at the same row count (same half4 groups, same order), that the
row-group size does not move the bits, and how far the half bits sit from today's fp32 rows.
"""
import copy
import re

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import flags as FL
from kyojin_mlx import kernels as KN
from kyojin_mlx import metal_src as S

ROWS = 6                      # a verify block of 5 drafts + the pending token
KIN, NOUT, K = 1280, 4096, 3


@pytest.fixture(autouse=True)
def half_rows_4(monkeypatch):
    """This file is about the packed-fp16 row class; tests/conftest.py pins HALF_ROWS=0 for the bitwise tests, so put the shipped default back."""
    monkeypatch.setattr(S, "HALF_ROWS", 4)


# ------------------------------------------------------------------------------------------------ flag, text, gate (no GPU needed)
def test_flag_registered_default_off():
    f = FL.FLAGS["rg_half"]
    assert f.env == "KYOJIN_RG_HALF" and f.default == "0" and f.kind == "bool" and f.runtime is True
    assert f.attrs[0][:2] == ("exl3_linear", "RG_HALF")
    assert f.attrs[0][2]("1") is True and f.attrs[0][2]("0") is False


def test_module_literal_matches_flag_default():
    src = open(EL.__file__).read()
    m = re.search(r'^RG_HALF = os\.environ\.get\("KYOJIN_RG_HALF", "(\w*)"\) not in \("0", ""\)', src, re.M)
    assert m and m.group(1) == FL.FLAGS["rg_half"].default


def test_many_rows_kernel_text_is_the_half_launch():
    """half=True builds the half4 x groups of the fused prologue from the plain fp16 rot rows; the fp32 form keeps its text."""
    h = S.gemv_rg_source(K, "mul1", 4, half=True)
    f = S.gemv_rg_source(K, "mul1", 4)
    assert "half4 hacc0_lo" in h and "hacc" not in f                    # packed-fp16 accumulators only in the half form
    assert "const device half4* xq4 = (const device half4*)" in h
    assert "xq4 += 4;" in h and "xp += 4;" in f
    # xh{r}_j = (x[2j], x[2j+1], x[8+2j], x[9+2j]) of the tile's 16-slice: the grouping run_decode_half pairs the decoded weights with
    for j in range(4):
        line = next(l for l in h.splitlines() if f"half4 xh0_{j} = half4(" in l)
        for p in (2 * j, 2 * j + 1, 8 + 2 * j, 9 + 2 * j):
            assert f"(KD / 4) + {p // 4}].{'xyzw'[p % 4]}" in line, (j, p, line)
    # row-group addressing and the store guard are the fp32 form's, unchanged
    for t in (h, f):
        assert "const uint RGN = (ROWS + MR - 1) / MR" in t
        assert "if (rbase + r < ROWS) part[((ulong(kblock) * GN + zg) * ROWS + rbase + r) * N + blk * 128 + c] = v;" in t
    assert "half4" not in f


def test_gate_follows_half_rows_and_the_fused_cap():
    """The class a row takes must not depend on which path the row count lands on: the gate is `_half_on`'s."""
    lin = EL.Exl3Linear.__new__(EL.Exl3Linear)
    lin.out_features, lin.in_features, lin.K, lin.cb = NOUT, KIN, K, "mul1"
    lin.dcb, lin._bias, lin._weff, lin.out_dtype = "mul1", None, None, mx.float16
    half_rows, fuse_rows, flag = S.HALF_ROWS, KN.FUSE_ROWS, EL.RG_HALF
    try:
        S.HALF_ROWS = 4                                                # shipped default
        EL.RG_HALF = True
        assert lin._rg_half(5) is True and lin._rg_half(6) is True
        assert lin._rg_half(3) is False                                # below HALF_ROWS the fused launch is fp32 too
        S.HALF_ROWS = 2                                                # the arm that makes 2..8 one class
        assert lin._rg_half(2) is True and lin._rg_half(6) is True
        S.HALF_ROWS = 0
        assert lin._rg_half(6) is False                                # fp32 rows everywhere: nothing to match
        S.HALF_ROWS = 4
        EL.RG_HALF = False
        assert lin._rg_half(6) is False                                # flag off: today's fp32 many-rows path
        EL.RG_HALF = True
        KN.FUSE_ROWS = 4                                               # cap under the row count: the fused form could not take these rows
        assert lin._rg_half(6) is False
        KN.FUSE_ROWS = 8
        lin.cb = "mul1fd"                                              # an affine-folded decode is never the half class
        assert KN.S.dense_cb("mul1fd") == "mul1fd" or not KN.S.af_fold("mul1fd")
    finally:
        S.HALF_ROWS, KN.FUSE_ROWS, EL.RG_HALF = half_rows, fuse_rows, flag


# ------------------------------------------------------------------------------------------------ Metal
metal = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")


def trellis(kin, nout, k, seed):
    rng = np.random.default_rng(seed)
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * k), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    return tr, suh, svh


def x_of(rows, kin, seed=0):
    return mx.array((np.random.default_rng(seed).standard_normal((rows, kin)) * 3).astype(np.float16))


def gemv(g, rows):
    """A Mul1Gemv of `rows` rows built the way exl3_linear._gemv builds one (the constructor asserts <= MAX_ROWS)."""
    c = KN.Mul1Gemv(*g, rows=1, out_dtype=mx.float16, cb="mul1")
    d = copy.copy(c)
    d.rows = rows
    d.fused_init()
    d.fparams = mx.array([d.tgk, d.n, rows], dtype=mx.uint32)
    return d


def many(g, x, half, mr=None):
    """One linear through the many-rows path: rot, one many-rows GEMV, finalize (what `Exl3Linear._many` runs)."""
    rows = x.shape[0]
    d = gemv(g, rows)
    p = mx.array([d.kt, d.nt, d.tgk, rows, 1], dtype=mx.uint32)
    mr = -(-rows // -(-rows // KN.RG_ROWS)) if mr is None else mr
    return d.finalize(d.partial_rows(KN.rot(x, d.suh, mx.float16 if half else mx.float32), rows, p, mr, half=half))


@pytest.fixture(scope="module")
def g():
    return trellis(KIN, NOUT, K, 7)


@metal
def test_half_row_bits_are_the_fused_launch_bits(g):
    """The claim: at the same row count a row of the many-rows half launch is bit for bit the fused launch's row (same half4 groups, same k order, same
    fp32 accumulator flush), so the choice of path cannot move a token by itself."""
    x = x_of(ROWS, KIN)
    fused = gemv(g, ROWS)
    if not (fused.can_fuse(ROWS) and S.half_on(ROWS)):
        pytest.skip("this shape / row count is not on the fused half path")
    want = np.array(fused.fused(x))
    got = np.array(many(g, x, half=True))
    assert np.array_equal(want.view(np.uint16), got.view(np.uint16))


@metal
def test_half_row_bits_do_not_depend_on_the_row_group(g):
    """Equal row groups, the same bits: the group size is launch geometry (mr 3 x 2 groups against mr 6 x 1)."""
    x = x_of(ROWS, KIN)
    a, b = np.array(many(g, x, half=True, mr=3)), np.array(many(g, x, half=True, mr=6))
    assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


@metal
def test_half_against_fp32_rows_is_finite_and_small(g):
    """Not bit-identical to today's fp32 rows: that is the trade the shipped half_rows makes too. The bound is stated in that class: the fp16 accumulator
    carries HALF_FLUSH tiles before it is flushed into the fp32 one, so an element differs by at most ~1e-3 of the row's magnitude (0.0008 measured here,
    shipped half_rows' NLL +0.002 nats is the same mechanism), and relative error is large only where the row itself cancels."""
    x = x_of(ROWS, KIN)
    h, f = np.array(many(g, x, half=True)).astype(np.float64), np.array(many(g, x, half=False)).astype(np.float64)
    assert np.isfinite(h).all() and np.isfinite(f).all()
    assert np.abs(h - f).max() <= 2e-3 * np.abs(f).max(), np.abs(h - f).max() / np.abs(f).max()
    rel = np.abs(h - f) / np.maximum(np.abs(f), 1e-6)
    assert np.percentile(rel, 99) < 5e-2, np.percentile(rel, 99)
    # the fp32 path is the call it always was: the flag never reaches it
    assert np.array_equal(np.array(many(g, x, half=False)).view(np.uint16), np.array(many(g, x, half=False, mr=4)).view(np.uint16))


@metal
def test_many_rows_half_reaches_through_the_module(g, monkeypatch):
    """The module path: with the flag on, ROW_SPLIT_MAX above GEMV_MAX_ROWS and no MMA / NAX band, 6 rows land on the half many-rows launch and the fired
    counter sees it; with the flag off the same call is the fp32 one."""
    from kyojin_mlx import fired as FD
    lin = EL.Exl3Linear(*g, out_dtype=mx.float16, cb="mul1", tune={})
    x = x_of(ROWS, KIN)
    saved = (EL.ROW_SPLIT_MAX, EL.MMA_ROWS, EL.RG_HALF, EL.GEMV_MAX_ROWS, EL.ROWS_NAX)
    try:
        EL.GEMV_MAX_ROWS, EL.ROW_SPLIT_MAX, EL.MMA_ROWS, EL.ROWS_NAX = 4, 16, 0, False
        monkeypatch.setattr(KN.Mul1Gemv, "can_fuse", lambda self, rows: False)   # the fused launch refuses these rows (smem / tgk): the block lands on the many-rows GEMV
        EL.RG_HALF = False
        monkeypatch.setattr(FD, "ON", True)                                   # the engage counter is off by default (flag fired)
        FD.reset()
        off = np.array(lin(x))
        EL.RG_HALF = True
        on = np.array(lin(x))
        assert FD.HITS.get("rg_half", 0) > 0, "rg_half never fired: rows did not land on the many-rows path"
        assert on.shape == off.shape == (ROWS, NOUT)
        assert on.view(np.uint16).tobytes() != off.view(np.uint16).tobytes()   # fp16 rows, not the fp32 ones
        assert np.isfinite(on).all()
    finally:
        EL.ROW_SPLIT_MAX, EL.MMA_ROWS, EL.RG_HALF, EL.GEMV_MAX_ROWS, EL.ROWS_NAX = saved
