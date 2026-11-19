"""G2e: oMLX-structure expert kernels against the G2c rows kernels (+ glm_moe.combine) on tiny synthetic NVFP4 packs.
g2m / inl / dc / seq(g2m) are bit-identical; omx (oMLX lane mapping, other sum order) is within 1 ulp per projection (2 after the glu product)."""
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
from kyojin_mlx import glm_moe as GM  # noqa: E402
from kyojin_mlx import glm_moe_e as GE  # noqa: E402
from kyojin_mlx import glm_moe_rows as R  # noqa: E402

E, H, I, K, LIMIT = 24, 2048, 1024, 8, 10.0


def mk(k, n, seed):
    rng = np.random.default_rng(seed)
    qs = mx.array(rng.integers(0, 2 ** 32, size=(E, n, k // 8), dtype=np.uint32))
    mt = mx.array((rng.integers(0x2C, 0x44, size=(E, n, k // 32)) | (rng.integers(0x2C, 0x44, size=(E, n, k // 32)) << 8)).astype(np.uint16))
    return types.SimpleNamespace(qs=qs, meta=mt, gs=mx.full((E,), 0.004, dtype=mx.float32), k=k, n=n)


def ulp(a, b):
    return int(np.abs(np.asarray(a.view(mx.int16)).astype(np.int32) - np.asarray(b.view(mx.int16)).astype(np.int32)).max())


def setup(T, overlap):
    g, u, d = mk(H, I, 1), mk(H, I, 2), mk(I, H, 3)
    rng = np.random.default_rng(T)
    rows = []
    for t in range(T):
        rows.append(list(rows[-1][:5]) + [x for x in rng.permutation(E) if x not in rows[-1][:5]][:K - 5] if (overlap and t) else list(rng.choice(E, size=K, replace=False)))
    inds = mx.array(np.array(rows, np.uint32).reshape(-1))
    wts = mx.array(rng.uniform(0.05, 0.4, size=(T * K,)).astype(np.float32))
    x2 = mx.array((rng.normal(size=(T, H)) * 0.5).astype(np.float16))
    sh = mx.array((rng.normal(size=(T, H)) * 0.1).astype(np.float16))
    tab = R.group(inds, T, K) if T > 1 else None
    h0 = R.pair_glu(g, u, x2, inds, tab, T, K, LIMIT)
    y0 = R.down(d, h0, inds, tab, T, K)
    return g, u, d, x2, inds, wts, sh, h0, y0, GM.combine(y0, wts, sh, T, K, H)


@pytest.mark.parametrize("T", [1, 2, 3, 4])
@pytest.mark.parametrize("overlap", [False, True])
def test_exact_variants(T, overlap):
    g, u, d, x2, inds, wts, sh, h0, y0, o0 = setup(T, overlap)
    for rps, nsg in ((1, 4), (2, 2), (4, 1)):
        assert mx.array_equal(GE.pair_glu("g2m", g, u, x2, inds, T, K, LIMIT, rps, nsg), h0)
        assert mx.array_equal(GE.down("g2m", d, h0, inds, T, K, rps, nsg), y0)
        assert mx.array_equal(GE.down_combine_seq(d, h0, inds, wts, sh, T, K, rps, nsg, "g2m"), o0)
    assert mx.array_equal(GE.pair_glu("inl", g, u, x2, inds, T, K, LIMIT), h0)
    assert mx.array_equal(GE.down("inl", d, h0, inds, T, K), y0)
    assert mx.array_equal(GE.down_combine(d, h0, inds, wts, sh, T, K), o0)


@pytest.mark.parametrize("T", [1, 2, 4])
def test_omx_within_ulp(T):
    g, u, d, x2, inds, wts, sh, h0, y0, o0 = setup(T, True)
    assert ulp(GE.pair_glu("omx", g, u, x2, inds, T, K, LIMIT), h0) <= 2
    assert ulp(GE.down("omx", d, h0, inds, T, K), y0) <= 1


def test_ok_limits():
    d = types.SimpleNamespace(k=2048, n=4096)
    assert GE.ok(d, d, 4, 8) and not GE.ok(d, d, 5, 8) and not GE.ok(d, types.SimpleNamespace(k=2000, n=4096), 1, 8)


@pytest.mark.parametrize("T", [1, 2, 4])
@pytest.mark.parametrize("overlap", [False, True])
def test_exact_x16_unrolled(T, overlap):
    """BP-MOE2: x read as 4 x 16 B per lane (xv) and the unit loop fully unrolled (iu) are bit-identical to G2c, for several (rows per simdgroup, simdgroups) shapes."""
    g, u, d, x2, inds, wts, sh, h0, y0, o0 = setup(T, overlap)
    for rps, nsg in ((1, 4), (2, 2), (4, 2), (4, 4), (8, 1)):
        for xv, iu in ((1, 0), (0, 1), (1, 1)):
            assert mx.array_equal(GE.pair_glu("g2m", g, u, x2, inds, T, K, LIMIT, rps, nsg, 0, xv, iu), h0), (rps, nsg, xv, iu)
            assert mx.array_equal(GE.down("g2m", d, h0, inds, T, K, rps, nsg, 0, xv, iu), y0), (rps, nsg, xv, iu)


def test_down_cfg_env(monkeypatch):
    assert GE.down_cfg(1) == (*GE.cfg(1), 0, 0)
    monkeypatch.setenv("KYOJIN_GLM_MOE_E_DOWN", "2x2+iu")
    assert GE.down_cfg(1) == (2, 2, 0, 1)
    assert GE.down_cfg(2) == (*GE.cfg(2), 0, 0)
    monkeypatch.setenv("KYOJIN_GLM_MOE_E_DOWN", "4x4+xv+iu")
    assert GE.down_cfg(1) == (4, 4, 1, 1)


@pytest.mark.parametrize("T", [1, 2, 3, 4])
@pytest.mark.parametrize("overlap", [False, True])
def test_rows_unroll_bit_identical(T, overlap):
    """V2-VERIFY (KYOJIN_MOE_ROWS2_UNROLL): the pair loops unrolled over TMAX (registers instead of a runtime index) give the same bits as the dynamic loops, including with duplicated experts (cnt > 1)."""
    g, u, d, x2, inds, wts, sh, h0, y0, o0 = setup(T, overlap)
    for rps, nsg in ((1, 4), (2, 2), (4, 1)):
        assert mx.array_equal(GE.pair_glu("g2m", g, u, x2, inds, T, K, LIMIT, rps, nsg, 0, 0, 0, 1), h0)
        assert mx.array_equal(GE.down("g2m", d, h0, inds, T, K, rps, nsg, 0, 0, 0, 1), y0)
    assert mx.array_equal(GE.pair_glu("g2m", g, u, x2, inds, T, K, LIMIT, 2, 2, 0, 1, 1, 1), h0)
    assert mx.array_equal(GE.down("g2m", d, h0, inds, T, K, 2, 2, 0, 1, 1, 1), y0)


def test_rows_unroll_same_experts():
    """every row picks the same 8 experts: cnt == T in every threadgroup"""
    T = 3
    g, u, d = mk(H, I, 1), mk(H, I, 2), mk(I, H, 3)
    rng = np.random.default_rng(5)
    a = list(rng.choice(E, size=K, replace=False))
    inds = mx.array(np.array([a] * T, np.uint32).reshape(-1))
    x2 = mx.array((rng.normal(size=(T, H)) * 0.5).astype(np.float16))
    h0 = GE.pair_glu("g2m", g, u, x2, inds, T, K, LIMIT, 2, 2)
    assert mx.array_equal(GE.pair_glu("g2m", g, u, x2, inds, T, K, LIMIT, 2, 2, 0, 0, 0, 1), h0)
    assert mx.array_equal(GE.down("g2m", d, h0, inds, T, K, 2, 2, 0, 0, 0, 1), GE.down("g2m", d, h0, inds, T, K, 2, 2))
