"""GGUF Q4_1 / Q5_0 / Q5_1 stored kernels on the Metal GPU (flag KYOJIN_LEGACY5_STORED): output against the float32 numpy decode of the same bytes.
Needs the GPU: run through gpu_archkit/q5_types_metal.sh (RULE 7 micro jobs). The CPU half is tests/test_legacy5.py. Small shapes (< 1 MiB of weights)."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY

TYPES5 = ("Q4_1", "Q5_0", "Q5_1")
RNG = lambda s=0: np.random.default_rng(s)


def _gpu():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu(), reason="no Metal GPU")


def _check(y, ref, tol=2e-3):
    y = np.array(y, np.float32)
    assert np.isfinite(y).all()
    err = np.abs(y - ref).max() / max(np.abs(ref).max(), 1e-6)
    assert err <= tol, err


def _small(t, n, k, seed=0):
    """Blocks with small d and m, so the fp16 products stay in range."""
    raw = SY.random_blocks(t, n, k, RNG(seed))
    b3 = raw.reshape(n, k // 32, GF.TYPES_BY_NAME[t][2])
    SY.set_scales(t, b3, RNG(seed + 1).uniform(1e-4, 5e-4, (n, k // 32)))
    if t != "Q5_0":
        b3[..., 2:4] = RNG(seed + 2).uniform(-1e-2, 1e-2, (n, k // 32)).astype("<f2").view(np.uint8).reshape(n, k // 32, 2)
    return GF.to_pack(t, raw, k), raw


@pytest.fixture(autouse=True)
def _flag(monkeypatch):
    monkeypatch.setenv("KYOJIN_LEGACY5_STORED", "1")
    monkeypatch.setenv("KYOJIN_KSTORED", "1")


@pytest.mark.parametrize("t", TYPES5)
@pytest.mark.parametrize("R", (1, 2, 4))
def test_gemv_one_and_few_rows(t, R):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 128, 1024)
    x = RNG(5).standard_normal((R, 1024)).astype(np.float16)
    d = FL.DevPack(p)
    assert d.w16 is None and d.gemv_ok()
    y = d.apply(mx.array(x))
    mx.eval(y)
    assert y.dtype == mx.float16 and y.shape == (R, 128)
    _check(y, x.astype(np.float32) @ GF.dequant(t, raw, 1024).T)


@pytest.mark.parametrize("t", TYPES5)
@pytest.mark.parametrize("R", (5, 9, 16, 40))
def test_fmma_and_rows_above(t, R):
    """R 5..16: the few-row MMA (decoder of the GEMV, lane per unit); above: transient decode + matmul. Ragged n = 72."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 72, 1024)
    x = RNG(4).standard_normal((R, 1024)).astype(np.float16)
    d = FL.DevPack(p)
    ref = x.astype(np.float32) @ GF.dequant(t, raw, 1024).T
    _check(d.apply(mx.array(x)), ref)
    if 5 <= R <= 16:
        _check(d._fmma(mx.array(x)), ref)


@pytest.mark.parametrize("t", TYPES5)
def test_dequant_kernel_is_the_numpy_decode(t):
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 64, 512)
    w = np.array(FL.DevPack(p).dequant(), np.float32)
    ref = GF.dequant(t, raw, 512)
    assert np.abs(w - ref).max() <= 1e-3 * max(np.abs(ref).max(), 1e-6)          # fp16 rounding of the decode only


@pytest.mark.parametrize("t", TYPES5)
@pytest.mark.parametrize("k", (640, 704))
def test_widths_off_the_gemv_grid_stay_stored(t, k):
    """The census down-expert widths (k = 640, 704: not a multiple of 512): no resident copy, the rows go through the transient decode."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 64, k)
    d = FL.DevPack(p)
    assert d.w16 is None and not d.gemv_ok()
    x = RNG(6).standard_normal((3, k)).astype(np.float16)
    _check(d.apply(mx.array(x)), x.astype(np.float32) @ GF.dequant(t, raw, k).T)


def _silu(a):
    return a / (1.0 + np.exp(-a))


@pytest.mark.parametrize("t", TYPES5)
@pytest.mark.parametrize("S", (1, 3, 40))
def test_routed_experts_gather_and_prefill(t, S):
    """E = 4 stacked experts through ForeignSwitchGLU: S = 1, 3 tokens (2, 6 slots) = the gather kernels (one launch per projection), S = 40 tokens (80 slots, above GATHER_MAX if it is below 80) = the sorted prefill path."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    E, H, I = 4, 512, 512
    gp, gr = zip(*[_small(t, I, H, 10 + e) for e in range(E)])
    up, ur = zip(*[_small(t, I, H, 20 + e) for e in range(E)])
    dp, dr = zip(*[_small(t, H, I, 30 + e) for e in range(E)])
    glu = FL.ForeignSwitchGLU(FL.DevPack(list(gp)), FL.DevPack(list(up)), FL.DevPack(list(dp)))
    assert glu.g.E == E and glu.g.w16 is None
    x = RNG(7).standard_normal((S, 1, H)).astype(np.float16)
    inds = ((np.arange(S * 2).reshape(S, 2) * 3 + 1) % E)           # [S, K = 2] expert ids of the S tokens
    y = glu(mx.array(x), mx.array(inds.reshape(S, 1, 2).astype(np.uint32)))
    mx.eval(y)
    ref = np.zeros((S, 2, H), np.float32)
    W = lambda raws, e, k: GF.dequant(t, raws[e], k)
    for s in range(S):
        for j in range(2):
            e = int(inds[s, j])
            a = x[s, 0].astype(np.float32)
            h = _silu(W(gr, e, H) @ a) * (W(ur, e, H) @ a)
            ref[s, j] = W(dr, e, I) @ h
    _check(np.array(y, np.float32).reshape(S, 2, H), ref, tol=2e-2)
