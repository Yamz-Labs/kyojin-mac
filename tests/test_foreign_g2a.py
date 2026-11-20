"""Flag g2a: the register-fed NVFP4 routed-expert gather GEMV (foreign_metal.gemv_g2_source), the gate | up pair launch, the one-launch gated activation and the
ForeignSwitchGLU fast path, against an independent float64 dequant of the stored bytes and against the flag-off kernels. Metal only; small shapes."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_linear as FL  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")

E = 12
LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)


def _dev(n, k, seed):
    r = np.random.default_rng(seed)
    d = object.__new__(FL.DevPack)
    d.fmt, d.n, d.k, d.E, d.rowgs, d._views, d.w, d.w16, d.mq = "nvfp4", n, k, E, False, {}, None, None, None
    d.qs = mx.array(r.integers(0, 2 ** 32, (E, n, k // 8), dtype=np.uint32))
    sc = r.integers(0, 0x7F, (E, n, k // 32, 2), dtype=np.uint16)               # every e4m3 code except the NaN one (0x7F), subnormals included
    sc[..., 0][:, :4] = 3                                                         # force some subnormal scales
    d.meta = mx.array((sc[..., 0] | (sc[..., 1] << 8)).astype(np.uint16))
    d.gs = mx.array((r.uniform(0.5, 1.5, E) * 1e-4).astype(np.float32))
    return d


def _ref(d, e):
    """float64 [n, k] of expert e from the stored layout (unit = 32 natural-order weights = 16 B; two e4m3 scales per unit)."""
    qs, meta = np.array(d.qs[e]).astype(np.uint32), np.array(d.meta[e]).astype(np.uint32)
    n = qs.shape[0]
    v = LUT[((qs[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(n, -1)]

    def e4m3(b):
        b = (b & 127).astype(np.int64)
        ex, m = b >> 3, b & 7
        return np.where(ex == 0, m * 2.0 ** -9, (1 + m / 8.0) * 2.0 ** (ex - 7.0))
    s = np.repeat(np.stack([e4m3(meta & 255), e4m3(meta >> 8)], -1).reshape(n, -1), 16, axis=1)
    return v * s * float(np.array(d.gs[e]))


def _set(monkeypatch, on):
    monkeypatch.setenv("KYOJIN_FOREIGN_G2A", "1" if on else "0")


@pytest.mark.parametrize("n,k", [(64, 1024), (128, 2048), (256, 4096)])
@pytest.mark.parametrize("S", [1, 8, 16, 32])
def test_gather_and_pair(monkeypatch, n, k, S):
    a, b = _dev(n, k, 1), _dev(n, k, 2)
    r = np.random.default_rng(3)
    ids = mx.array(r.integers(0, E, S).astype(np.uint32))
    rows = max(1, S // 8)
    x = mx.array(r.normal(0, 0.5, (rows, k)).astype(np.float16))
    tok = mx.array((np.arange(S) % rows).astype(np.uint32))
    _set(monkeypatch, True)
    assert a.g2a_ok()
    ya, yb = a.gather_pair(b, x, tok, ids)
    yg = a.gather(x, tok, ids)
    _set(monkeypatch, False)
    assert not a.g2a_ok()
    za, zb = a.gather_pair(b, x, tok, ids)
    mx.eval(ya, yb, yg, za, zb)
    xs, idn, tn = np.array(x).astype(np.float64), np.array(ids), np.array(tok)
    for s in range(S):
        for d, y, z in ((a, ya, za), (b, yb, zb)):
            ref = _ref(d, int(idn[s])) @ xs[tn[s]]
            got, old = np.array(y[s]).astype(np.float64), np.array(z[s]).astype(np.float64)
            tol = 2e-3 * np.abs(ref).max() + 1e-6
            assert np.abs(got - ref).max() <= tol, (s, np.abs(got - ref).max(), tol)
            assert np.abs(got - old).max() <= tol
    assert np.array_equal(np.array(yg), np.array(ya))                              # the single gather and the pair read the same way


@pytest.mark.parametrize("limit", [None, 7.0])
def test_glu_rows(limit):
    S, n = 8, 256
    y2 = mx.array(np.random.default_rng(5).normal(0, 6, (2 * S, n)).astype(np.float16))
    sm = FL.ForeignSwitchGLU.__new__(FL.ForeignSwitchGLU)
    sm.limit = limit
    ref = FL.ForeignSwitchGLU._glu(sm, y2[:S], y2[S:])
    got = FL.glu_rows(y2, S, n, limit)
    mx.eval(ref, got)
    r, g = np.array(ref).astype(np.float32), np.array(got).astype(np.float32)
    assert np.abs(r - g).max() <= 2e-3 * np.abs(r).max()
    assert (r == g).mean() > 0.9                                                    # same math; the exp differs in the last bit now and then (1 fp16 ulp)


@pytest.mark.parametrize("R", [1, 2, 3, 4])
def test_switch_glu(monkeypatch, R):
    H, I = 1024, 1024
    g, u, d = _dev(I, H, 1), _dev(I, H, 2), _dev(H, I, 3)
    sm = FL.ForeignSwitchGLU(g, u, d, limit=7.0)
    r = np.random.default_rng(R)
    x = mx.array(r.normal(0, 1.0, (R, H)).astype(np.float16))
    inds = mx.array(np.stack([r.permutation(E)[:8] for _ in range(R)]).astype(np.uint32))
    _set(monkeypatch, True)
    y = sm(x, inds)
    _set(monkeypatch, False)
    z = sm(x, inds)
    mx.eval(y, z)
    y, z = np.array(y).astype(np.float64), np.array(z).astype(np.float64)
    assert y.shape == (R, 8, H)
    assert np.abs(y - z).max() <= 5e-3 * np.abs(z).max()
