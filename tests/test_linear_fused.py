"""One-launch decode linear (rot + GEMV + finalize in one kernel, KYOJIN_LINEAR_FUSED) vs the three-launch path: bit-identical for the single linear, the
same-input group and the grouped wo_a, 1-4 rows, both input dtypes, with and without the folded fp8 fake quant, repeated calls (counter sets), odd k ranges."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import fused_group as FG
from kyojin_mlx import kernels as KN
from kyojin_mlx import metal_src as S
from kyojin_mlx.grouped import GroupedGemv


def trellis(kin, nout, K, seed):
    rng = np.random.default_rng(seed)
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    return tr, suh, svh


def bits(a):
    a = np.array(a)
    return a.view(np.uint16) if a.dtype == np.float16 else a.view(np.uint32)


def x_of(rows, kin, dt, seed=0):
    return mx.array((np.random.default_rng(seed).standard_normal((rows, kin)) * 3).astype(dt))


SHAPES = [(256, 256), (640, 1152), (1280, 4096), (2304, 5120), (5120, 1280), (4096, 1024), (8192, 5120)]


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("K", [2, 3, 5])
@pytest.mark.parametrize("kin,nout", SHAPES)
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_single(cb, K, kin, nout, rows):
    tr, su, sv = trellis(kin, nout, K, kin + K)
    for od in (mx.float16, mx.float32):
        g = KN.Mul1Gemv(tr, su, sv, rows=rows, out_dtype=od, cb=cb)
        assert g.can_fuse(rows)
        for dt, fq in ((np.float16, False), (np.float16, True), (np.float32, False)):
            x = x_of(rows, kin, dt, rows)
            want = g(x, fq8=fq)
            for _ in range(10):                                     # the counter sets rotate: every call sees different counters
                assert np.array_equal(bits(g.fused(x, fq8=fq)), bits(want))


@pytest.mark.parametrize("kin,nout", [(1280, 4096), (2304, 5120), (4096, 1024)])
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_single_rot2(monkeypatch, kin, nout, rows):
    """Two-launch form (rot kernel, then GEMV + finalize tail) above KN.ROT2_MIN_BLOCKS: bit-identical too."""
    monkeypatch.setattr(KN, "ROT2_MIN_BLOCKS", 1)
    tr, su, sv = trellis(kin, nout, 3, kin)
    g = KN.Mul1Gemv(tr, su, sv, rows=rows, out_dtype=mx.float32)
    for dt, fq in ((np.float16, False), (np.float16, True), (np.float32, False)):
        x = x_of(rows, kin, dt, rows)
        want = g(x, fq8=fq)
        for _ in range(10):
            assert np.array_equal(bits(g.fused(x, fq8=fq)), bits(want))


def test_fq8_extremes():
    """The folded fp8 round trip with power-of-two reciprocals vs the stock fake-quant kernel followed by the plain rot."""
    fq = pytest.importorskip("mlx_vlm.models.deepseek_v41.fakequant")
    rng = np.random.default_rng(1)
    k = 1024
    suh = mx.ones((k,), dtype=mx.float16)
    for trial in range(40):
        sc = 10.0 ** rng.uniform(-9, 4.6, size=(1, k // 32, 1))
        x = rng.standard_normal((1, k // 32, 32)) * sc
        if trial % 7 == 0:
            x[:, ::3] = 0
        if trial % 11 == 0:
            x[:, 1, 5] = 65000
        xm = mx.array(np.clip(x, -65000, 65000).reshape(1, k).astype(np.float16))
        assert np.array_equal(bits(KN.rot(xm, suh, mx.float32, fq8=True)), bits(KN.rot(fq.fake_quant_fp8_ue8m0(xm), suh, mx.float32)))


def test_module_bias_and_dispatch(monkeypatch):
    tr, su, sv = trellis(512, 1024, 3, 5)
    m = EL.Exl3Linear(mx.array(tr), mx.array(su), mx.array(sv), bias=mx.array(np.random.default_rng(2).standard_normal(1024).astype(np.float16)))
    x = mx.array(np.random.default_rng(3).standard_normal((1, 1, 512)).astype(np.float16))
    monkeypatch.setattr(EL, "LINEAR_FUSED", False)
    want = m(x)
    monkeypatch.setattr(EL, "LINEAR_FUSED", True)
    assert np.array_equal(bits(m(x)), bits(want))
    x4 = mx.array(np.random.default_rng(4).standard_normal((1, 4, 512)).astype(np.float16))
    monkeypatch.setattr(EL, "LINEAR_FUSED", False)
    want = m(x4)
    monkeypatch.setattr(EL, "LINEAR_FUSED", True)
    assert np.array_equal(bits(m(x4)), bits(want))


GROUPS = [("gate_up", 512, [1024, 1024], 2, "mcg"), ("qkv", 512, [1024, 256, 256], 3, "mcg"), ("ragged_tgk", 4096, [1024, 16384], 2, "mul1"),
          ("wq_a_wkv", 5120, [1280, 512], 5, "mul1"), ("four", 1024, [512, 512, 512, 512], 4, "mul1")]


@pytest.mark.parametrize("name,kin,outs,K,cb", GROUPS, ids=[g[0] for g in GROUPS])
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_group(name, kin, outs, K, cb, rows):
    mem = [EL.Exl3Linear(*(mx.array(a) for a in trellis(kin, n, K, 11 * i + K)), cb=cb) for i, n in enumerate(outs)]
    g = FG.Exl3Group(mem)
    assert g.can_fuse(rows)
    for dt, fq in ((np.float16, False), (np.float16, True), (np.float32, False)):
        x = x_of(rows, kin, dt, 7)
        want = g.fin(g.gemv(g.rot(x, fq), rows), rows)
        for _ in range(10):
            for a, b in zip(g.fused(x, fq), want):
                assert np.array_equal(bits(a), bits(b))


@pytest.mark.parametrize("cb,K,G", [("mul1", 5, 8), ("mul1", 3, 4), ("mcg", 2, 3)])
@pytest.mark.parametrize("R", [1, 2, 4])
def test_grouped(monkeypatch, cb, K, G, R):
    rng = np.random.default_rng(K)
    tr = mx.array(rng.integers(-32768, 32768, (G, 4096 // 16, 1024 // 16, 16 * K), dtype=np.int16))
    suh = mx.array((rng.choice([-1.0, 1.0], (G, 4096)) * rng.uniform(0.5, 1.5, (G, 4096))).astype(np.float16))
    svh = mx.array((rng.choice([-1.0, 1.0], (G, 1024)) * rng.uniform(0.5, 1.5, (G, 1024))).astype(np.float16))
    gg = GroupedGemv(tr, suh, svh, cb)
    assert gg.fusable
    for dt, fq in ((np.float16, False), (np.float16, True), (np.float32, False)):
        x = mx.array((rng.standard_normal((1, G, R, 4096)) * 2).astype(dt))
        monkeypatch.setattr(EL, "LINEAR_FUSED", False)
        want = gg(x, fq8=fq)
        monkeypatch.setattr(EL, "LINEAR_FUSED", True)
        for _ in range(10):
            assert np.array_equal(bits(gg(x, fq8=fq)), bits(want))


def test_independent_calls_in_flight():
    """Up to FUSED_SETS independent calls of one module in a single graph (they may overlap on the GPU) keep their bits."""
    tr, su, sv = trellis(1280, 4096, 5, 1)
    g = KN.Mul1Gemv(tr, su, sv, rows=1, out_dtype=mx.float16)
    xs = [x_of(1, 1280, np.float16, s) for s in range(KN.Mul1Gemv.FUSED_SETS)]
    want = [g(x) for x in xs]
    for _ in range(20):
        got = [g.fused(x) for x in xs]
        mx.eval(got)
        for a, b in zip(got, want):
            assert np.array_equal(bits(a), bits(b))


def test_smem_budget_guard():
    """A k range that does not fit the threadgroup buffer is refused (the caller keeps the three-launch path)."""
    tr, su, sv = trellis(16384, 128, 2, 1)
    g = KN.Mul1Gemv(tr, su, sv, rows=4)
    assert g.can_fuse(4) == (max(4 * S.fused_dims(g.kt, g.tgk), 512 * 4) <= KN.Mul1Gemv.FUSED_SMEM_FLOATS)


@pytest.mark.parametrize("cb", ["mul1", "mcg"])
@pytest.mark.parametrize("K", [2, 3, 4, 5])
@pytest.mark.parametrize("kin,nout", SHAPES)
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_half_rows(cb, K, kin, nout, rows, monkeypatch):
    """Packed-fp16 row FMAs (metal_src.HALF_ROWS) in the fused launch: a row of >= HALF_ROWS rows is within a couple of fp16 ulps of the fp32 path,
    fewer rows keep the fp32 bits."""
    tr, su, sv = trellis(kin, nout, K, kin + K + 7)
    g = KN.Mul1Gemv(tr, su, sv, rows=rows, out_dtype=mx.float16, cb=cb)
    x = x_of(rows, kin, np.float16, rows)
    want = np.array(g(x)).astype(np.float64)
    monkeypatch.setattr(S, "HALF_ROWS", 2)
    got = np.array(g.fused(x)).astype(np.float64)
    if rows < 2:
        assert np.array_equal(got, want)
    else:
        err = np.sqrt(((got - want) ** 2).mean() / (want ** 2).mean())
        assert err < 2e-3, err
        for _ in range(5):                                           # counter sets rotate: same result every call
            assert np.array_equal(np.array(g.fused(x)).astype(np.float64), got)
    monkeypatch.setattr(S, "HALF_ROWS", 0)
    assert np.array_equal(bits(g.fused(x)), bits(g(x)))              # switch off again: the fp32 bits
