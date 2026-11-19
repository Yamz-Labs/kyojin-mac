"""GGUF types 2 on the Metal GPU: Q2_0 (flag KYOJIN_Q2_0_STORED) and the partial-chunk K-class GEMV / gather (flag KYOJIN_GEMVK_PARTIAL): output against the float32 numpy decode of
the same bytes. Needs the GPU: run through gpu_archkit/types2_metal.sh (RULE 7 micro jobs). The CPU halves are tests/test_q2_0.py and tests/test_gemvk_partial.py.
Small shapes (< 1 MiB of weights per matrix)."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_synth as SY

RNG = lambda s=0: np.random.default_rng(s)
LEGACY = ("Q4_1", "Q5_0", "Q5_1")


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
    """Blocks with small scales, so the fp16 products stay in range."""
    be, bb = GF.TYPES_BY_NAME[t][1:]
    raw = SY.random_blocks(t, n, k, RNG(seed))
    b3 = raw.reshape(n, k // be, bb)
    SY.set_scales(t, b3, RNG(seed + 1).uniform(1e-4, 5e-4, (n, k // be)))
    if t in ("Q4_1", "Q5_1"):
        b3[..., 2:4] = RNG(seed + 2).uniform(-1e-2, 1e-2, (n, k // be)).astype("<f2").view(np.uint8).reshape(n, k // be, 2)
    return GF.to_pack(t, raw, k), raw


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    monkeypatch.setenv("KYOJIN_LEGACY5_STORED", "1")
    monkeypatch.setenv("KYOJIN_Q2_0_STORED", "1")
    monkeypatch.setenv("KYOJIN_KSTORED", "1")


@pytest.fixture
def partial(monkeypatch):
    monkeypatch.setenv("KYOJIN_GEMVK_PARTIAL", "1")


def _spy_gemvk(monkeypatch):
    from kyojin_mlx import foreign_linear as FL
    calls = []
    orig = FL.DevPack._gemvk
    monkeypatch.setattr(FL.DevPack, "_gemvk", lambda self, x2, nbr=None: (calls.append(self.k), orig(self, x2, nbr))[1])
    return calls


# ---------------------------------------------------------------- item 1: Q2_0 on the stock widths (k % 512 == 0)
@pytest.mark.parametrize("R", (1, 2, 4))
def test_q2_0_gemv_one_and_few_rows(R):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small("Q2_0", 128, 1024)
    x = RNG(5).standard_normal((R, 1024)).astype(np.float16)
    d = FL.DevPack(p)
    assert d.w16 is None and d.gemv_ok()
    y = d.apply(mx.array(x))
    mx.eval(y)
    assert y.dtype == mx.float16 and y.shape == (R, 128)
    _check(y, x.astype(np.float32) @ GF.dequant("Q2_0", raw, 1024).T)


@pytest.mark.parametrize("R", (5, 9, 16, 40))
def test_q2_0_fmma_and_rows_above(R):
    """R 5..16: the few-row MMA (decoder of the GEMV); above: transient decode + matmul. Ragged n = 72."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small("Q2_0", 72, 1024)
    x = RNG(4).standard_normal((R, 1024)).astype(np.float16)
    d = FL.DevPack(p)
    ref = x.astype(np.float32) @ GF.dequant("Q2_0", raw, 1024).T
    _check(d.apply(mx.array(x)), ref)
    if 5 <= R <= 16:
        _check(d._fmma(mx.array(x)), ref)


@pytest.mark.parametrize("t,k", [("Q2_0", 512), ("Q2_0", 704), ("Q5_1", 640)])
def test_dequant_kernel_is_the_numpy_decode(t, k):
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 64, k)
    w = np.array(FL.DevPack(p).dequant(), np.float32)
    ref = GF.dequant(t, raw, k)
    assert np.abs(w - ref).max() <= 1e-3 * max(np.abs(ref).max(), 1e-6)          # fp16 rounding of the decode only


# ---------------------------------------------------------------- item 2: widths off the chunk, flag off = transient decode, flag on = GEMV
@pytest.mark.parametrize("t", LEGACY + ("Q2_0",))
@pytest.mark.parametrize("k", (640, 704))
def test_off_grid_flag_off_stays_on_the_transient_decode(t, k, monkeypatch):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.delenv("KYOJIN_GEMVK_PARTIAL", raising=False)
    p, raw = _small(t, 64, k)
    d = FL.DevPack(p)
    assert d.w16 is None and not d.gemv_ok()
    x = RNG(6).standard_normal((3, k)).astype(np.float16)
    _check(d.apply(mx.array(x)), x.astype(np.float32) @ GF.dequant(t, raw, k).T)


def _spy_fmma(monkeypatch):
    from kyojin_mlx import foreign_linear as FL
    calls = []
    orig = FL.DevPack._fmma
    monkeypatch.setattr(FL.DevPack, "_fmma", lambda self, x2, *a, **kw: (calls.append(self.k), orig(self, x2, *a, **kw))[1])
    return calls


def _check_partial_route(d, t, raw, k, R, seed, monkeypatch):
    """apply() must run a stored kernel, never the transient decode: the few-row MMA (fmma_ok: R >= the per-format minimum, k % 128 == 0) or else the GEMV.
    The GEMV (partial chunk when k % 512 != 0) is then also called directly, so it is checked at every row count."""
    import mlx.core as mx
    gv, fm = _spy_gemvk(monkeypatch), _spy_fmma(monkeypatch)
    x = RNG(seed).standard_normal((R, k)).astype(np.float16)
    ref = x.astype(np.float32) @ GF.dequant(t, raw, k).T
    stored_mma = d.fmma_ok(R)
    y = d.apply(mx.array(x))
    mx.eval(y)
    assert (fm, gv) == (([k], []) if stored_mma else ([], [k]))
    _check(y, ref)
    if stored_mma:
        y2 = d._gemvk(mx.array(x))
        mx.eval(y2)
        assert gv == [k]
        _check(y2, ref)


@pytest.mark.parametrize("t", LEGACY + ("Q2_0",))
@pytest.mark.parametrize("k", (640, 704, 768, 1024))
@pytest.mark.parametrize("R", (1, 3, 4, 8))
def test_partial_gemv_rows(t, k, R, partial, monkeypatch):
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 128, k, seed=k)
    d = FL.DevPack(p)
    assert d.w16 is None and d.gemv_ok() and d._part() == (k % 512 != 0)
    assert d.fmma_ok(R) == (R >= 4 and k % 128 == 0)                 # documents the route: min 4 for these four types
    _check_partial_route(d, t, raw, k, R, 6, monkeypatch)


@pytest.mark.parametrize("t,k", [("Q5_K", 768), ("Q6_K", 768), ("Q5_K", 1280), ("Q6_K", 1280)])
@pytest.mark.parametrize("R", (1, 2, 4, 8))
def test_partial_gemv_k_class(t, k, R, partial, monkeypatch):
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 128, k, seed=k)
    d = FL.DevPack(p)
    assert d.gemv_ok() and d._part()
    _check_partial_route(d, t, raw, k, R, 8, monkeypatch)


def _silu(a):
    return a / (1.0 + np.exp(-a))


@pytest.mark.parametrize("t", LEGACY + ("Q2_0",))
@pytest.mark.parametrize("H,I", [(640, 704), (704, 640), (1024, 1024)])
@pytest.mark.parametrize("S", (1, 3, 40))
def test_routed_experts_gather_and_grouped(t, H, I, S, partial):
    """E = 4 stacked experts through ForeignSwitchGLU: S = 1, 3 tokens (2, 6 slots) = the gather kernels (gate | up, glu, down: widths H and I), S = 40 tokens = the sorted prefill."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    E = 4
    gp, gr = zip(*[_small(t, I, H, 10 + e) for e in range(E)])
    up, ur = zip(*[_small(t, I, H, 20 + e) for e in range(E)])
    dp, dr = zip(*[_small(t, H, I, 30 + e) for e in range(E)])
    glu = FL.ForeignSwitchGLU(FL.DevPack(list(gp)), FL.DevPack(list(up)), FL.DevPack(list(dp)))
    assert glu.g.E == E and glu.g.w16 is None and glu.g.gemv_ok() and glu.d.gemv_ok()
    x = RNG(7).standard_normal((S, 1, H)).astype(np.float16)
    inds = ((np.arange(S * 2).reshape(S, 2) * 3 + 1) % E)
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
