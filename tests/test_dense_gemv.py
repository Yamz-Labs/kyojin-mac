"""KYOJIN_DENSE_DEC: the dense (non-routed) trellis decode cut (metal_src.dense_cb). 0 = today's dec_mul1, 1 = dec_mul1r1 (lane sum by one imad,
the same fp16 value bit for bit), 2 = dec_mul1fd (bare byte sum with the affine folded out: NOT bit-identical, checked against the f32 reference).
The flag must be a normal int lever, arm through the module attribute, and the outputs of the single linear, the same-input group and the grouped
wo_a must keep today's bits at 1-4 rows (decode 1) or stay within kernel numerics of the f32 reference (decode 2). The abliteration projection
is a post-output linear map: it must still apply and commute with the decode choice the same way.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import ablit as AB
from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import flags as FL
from kyojin_mlx import fused_group as FG
from kyojin_mlx import grouped as G
from kyojin_mlx import metal_src as S
from kyojin_mlx import ref_numpy as R


def arrays(kin, nout, K=5, seed=0):
    return R.random_linear(kin, nout, K, seed=seed)


def lin(kin, nout, K=5, seed=0):
    t, u, v = arrays(kin, nout, K, seed)
    m = EL.Exl3Linear(mx.array(t), mx.array(u), mx.array(v), cb="mul1", out_dtype=mx.float16)
    mx.eval(m.parameters())
    return m, t, u, v


def bits(y):
    return np.array(y).view(np.uint16)


def cond(m, rows, kin, seed=0, dt=np.float16):
    return mx.array((np.random.default_rng(seed).standard_normal((rows, kin)) * 0.7).astype(dt))


# ------------------------------------------------------------------ flag plumbing
def test_flag_registered_and_aligned():
    f = FL.FLAGS["dense_dec"]
    assert f.env == "KYOJIN_DENSE_DEC" and f.default == "0" and f.kind == "int" and f.runtime
    assert f.attrs == (("metal_src", "DENSE_DEC", int),)
    assert FL.value("dense_dec") == "0"
    assert S.dense_cb("mul1") == "mul1"
    assert S.dense_header("mul1") == S.HEADER


def test_runtime_switch_rebuilds_the_kernel(monkeypatch):
    S.DENSE_DEC = 0
    m, *_ = lin(256, 256)
    y0 = m(cond(m, 1, 256)); mx.eval(y0)
    assert ("mul1", 1, S.PF2) in m._g[1]._kerns
    S.DENSE_DEC = 1
    y1 = m(cond(m, 1, 256, seed=1)); mx.eval(y1)
    assert m._g[1].dcb == "mul1r1"
    assert any(k[-2] == "mul1r1" for k in m._g[1]._fc)          # the fused kernel was rebuilt for the new decode
    S.DENSE_DEC = 0
    y2 = m(cond(m, 1, 256, seed=0)); mx.eval(y2)
    assert m._g[1].dcb == "mul1"
    assert np.array_equal(bits(y0), bits(y2))


# ------------------------------------------------------------------ same bits at 1..4 rows
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_single_r1_is_bit_identical(rows):
    m, *_ = lin(384, 256)
    x = cond(m, rows, 384)
    S.DENSE_DEC = 0
    try:
        a = m(x); mx.eval(a)
        S.DENSE_DEC = 1
        b = m(x); mx.eval(b)
    finally:
        S.DENSE_DEC = 0
    assert np.array_equal(bits(a), bits(b))


@pytest.mark.parametrize("rows", [1, 4])
def test_group_r1_is_bit_identical(rows):
    a, *_ = lin(384, 128, seed=1)
    b, *_ = lin(384, 384, seed=2)
    g = FG.Exl3Group([a, b])
    x = cond(a, rows, 384)
    S.DENSE_DEC = 0
    try:
        y0 = g.run(x); mx.eval(*y0)
        S.DENSE_DEC = 1
        y1 = g.run(x); mx.eval(*y1)
    finally:
        S.DENSE_DEC = 0
    assert all(np.array_equal(bits(p), bits(q)) for p, q in zip(y0, y1))


def test_grouped_r1_is_bit_identical():
    rng = np.random.default_rng(3)
    Gg, kt, nt, K = 4, 16, 64, 5
    tr = rng.integers(-32768, 32768, (Gg, kt, nt, 16 * K)).astype(np.int16)
    suh = (rng.choice([-1.0, 1.0], (Gg, kt * 16)) * rng.uniform(.5, 1.5, (Gg, kt * 16))).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], (Gg, nt * 16)) * rng.uniform(.5, 1.5, (Gg, nt * 16))).astype(np.float16)
    x = mx.array((rng.standard_normal((1, Gg, 1, kt * 16)) * .3).astype(np.float16))
    S.DENSE_DEC = 0
    try:
        y0 = G.GroupedGemv(mx.array(tr), mx.array(suh), mx.array(svh))(x); mx.eval(y0)
        S.DENSE_DEC = 1
        y1 = G.GroupedGemv(mx.array(tr), mx.array(suh), mx.array(svh))(x); mx.eval(y1)
    finally:
        S.DENSE_DEC = 0
    assert np.array_equal(bits(y0), bits(y1))


# ------------------------------------------------------------------ error against the f32 reference
def _err(y, ref):
    y = np.array(y, dtype=np.float32); ref = ref.astype(np.float32)
    return float(np.linalg.norm(y - ref) / np.linalg.norm(ref)), float(np.abs(y - ref).max())


def test_fd_error_next_to_today(capsys):
    m, t, u, v = lin(256, 256)
    x = cond(m, 1, 256)
    xn = np.array(x).astype(np.float32)
    ref = R.gemv(xn, t, u, v, dtype=np.float32)
    C = (np.random.default_rng(7).standard_normal((256, 32)) / 16).astype(np.float32)
    out = {}
    for dec in (0, 2):
        S.DENSE_DEC = dec
        try:
            y = m(x); mx.eval(y)
        finally:
            S.DENSE_DEC = 0
        out[dec] = np.array(y).astype(np.float32)
        rel, mxerr = _err(y, ref)
        print(f"dense_dec={dec}: rel {rel:.3e}  max abs {mxerr:.3e}  top1 {int(np.argmax(out[dec] @ C))}")
    rel0, max0 = _err(out[0], ref)
    rel2, max2 = _err(out[2], ref)
    assert rel2 <= max(10 * rel0, 2e-3), (rel0, rel2)          # fd adds one folded-affine rounding, not a new class of error
    assert np.array_equal(np.argsort(out[0] @ C)[-5:], np.argsort(out[2] @ C)[-5:]) or rel2 < 1e-3


def test_fd_rows4_never_uses_the_half_stage():
    m, *_ = lin(384, 256)
    S.DENSE_DEC = 2
    try:
        m(cond(m, 1, 384)); mx.eval(m(cond(m, 1, 384)))
        assert S.af_fold(m._g[1].dcb)
        assert m._g[1]._half_on(4) is False                     # the bare byte sum has no affine: the packed-fp16 stage would drop it
        y = m(cond(m, 4, 384)); mx.eval(y)
    finally:
        S.DENSE_DEC = 0
    assert np.isfinite(np.array(y, dtype=np.float32)).all()


# ------------------------------------------------------------------ abliteration hook
def test_ablit_projection_still_applies_and_commutes():
    m, *_ = lin(256, 256)
    x = cond(m, 1, 256)
    Rm = mx.array((np.random.default_rng(9).standard_normal((2, 256)) / 16).astype(np.float32))
    w = 0.7
    ys = {}
    for dec in (0, 1, 2):
        S.DENSE_DEC = dec
        try:
            y = m(x); mx.eval(y)
        finally:
            S.DENSE_DEC = 0
        ys[dec] = np.array(AB._project(y, w, Rm), dtype=np.float32)
    # the hook is a linear map of the output: with the bit-identical decode it is bit-identical, with fd within kernel numerics
    assert np.array_equal(ys[0].view(np.uint32), ys[1].view(np.uint32))
    assert np.linalg.norm(ys[2] - ys[0]) / np.linalg.norm(ys[0]) < 2e-3
    # and it did something (off == the unprojected output, on != it): the lever cannot skip it
    y = None
    S.DENSE_DEC = 0
    y = m(x); mx.eval(y)
    assert not np.array_equal(np.array(AB._project(y, w, Rm)), np.array(y))
