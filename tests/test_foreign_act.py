"""Activation kind (SiTU) through the foreign-format routed-expert kernels: g2a glu_rows, tile gather_glu, kqgmm silu_perm, kqnax fused gate | up and the foreign_moe elementwise
kernel, each against the same math on the stored NVFP4 bytes (independent float64 dequant, fp16 roundings of the stored path). Metal only; small shapes (12 experts, 1024 x 1024)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_linear as FL, foreign_kqgmm as KQ, foreign_kqnax as NX, foreign_moe as FMO, fp4tile as FT  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")

E, N, KD = 12, 1024, 1024
SITU = ("situ", 4.0, 25.0)
LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)


def _dev(n, k, seed, gscale=1e-4):
    r = np.random.default_rng(seed)
    d = object.__new__(FL.DevPack)
    d.fmt, d.n, d.k, d.E, d.rowgs, d._views, d.w, d.w16, d.mq, d.tl = "nvfp4", n, k, E, False, {}, None, None, None, None
    d.qs = mx.array(r.integers(0, 2 ** 32, (E, n, k // 8), dtype=np.uint32))
    sc = r.integers(0, 0x70, (E, n, k // 32, 2), dtype=np.uint16)
    d.meta = mx.array((sc[..., 0] | (sc[..., 1] << 8)).astype(np.uint16))
    d.gs = mx.array((r.uniform(0.5, 1.5, E) * gscale).astype(np.float32))
    return d


def _ref_w(d, e):
    qs, meta = np.array(d.qs[e]).astype(np.uint32), np.array(d.meta[e]).astype(np.uint32)
    n = qs.shape[0]
    v = LUT[((qs[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(n, -1)]

    def e4m3(b):
        b = (b & 127).astype(np.int64)
        ex, m = b >> 3, b & 7
        return np.where(ex == 0, m * 2.0 ** -9, (1 + m / 8.0) * 2.0 ** (ex - 7.0))
    s = np.repeat(np.stack([e4m3(meta & 255), e4m3(meta >> 8)], -1).reshape(n, -1), 16, axis=1)
    return v * s * float(np.array(d.gs[e]))


def _situ(a, b, sg=4.0, sl=25.0):
    return (sg * np.tanh(a / sg)) * (1.0 / (1.0 + np.exp(-a))) * (sl * np.tanh(b / sl))


def _silu(a, b):
    return a / (1.0 + np.exp(-a)) * b


def _experts(seed=0):
    # gate / up scaled so the pre-activations span the nonlinear range of SiTU (|g| up to ~20)
    return _dev(N, KD, seed + 1, 2e-3), _dev(N, KD, seed + 2, 2e-3), _dev(KD, N, seed + 3, 3e-4)


def _ref_mlp(g, u, d, x, ids, K, act=_situ):
    xs, idn = np.array(x).astype(np.float64), np.array(ids)
    out, cache = [], {}
    for s in range(len(idn)):
        e = int(idn[s])
        if e not in cache:
            cache[e] = tuple(_ref_w(p, e) for p in (g, u, d))
        wg, wu, wd = cache[e]
        a = (wg @ xs[s // K]).astype(np.float16).astype(np.float64)
        b = (wu @ xs[s // K]).astype(np.float16).astype(np.float64)
        h = act(a, b).astype(np.float16).astype(np.float64)
        out.append(wd @ h)
    return np.stack(out)


def _close(got, ref, tol=3e-3):
    got = np.array(got).astype(np.float64).reshape(ref.shape)
    err, top = np.abs(got - ref).max(), np.abs(ref).max()
    assert np.isfinite(got).all() and top > 0
    print(f"REL_ERR {err / top:.2e} (max abs {err:.3e}, max ref {top:.3e})")
    assert err <= tol * top + 1e-6, (err, top, err / top)
    return err / top


def _inputs(T, K, seed=7):
    r = np.random.default_rng(seed)
    x = mx.array(r.normal(0, 1.0, (T, KD)).astype(np.float16))
    ids = mx.array(r.integers(0, E, T * K).astype(np.uint32))
    return x, ids


def test_kinds_differ():
    """The test input is in the nonlinear range: SiTU and silu differ by more than the tolerance (the check discriminates)."""
    g, u, d = _experts()
    x, ids = _inputs(2, 4)
    a = _ref_mlp(g, u, d, x, ids, 4, _situ)
    b = _ref_mlp(g, u, d, x, ids, 4, _silu)
    assert np.abs(a - b).max() > 0.05 * np.abs(a).max()


def test_glu_rows_situ():
    S, n = 16, 1024
    y2 = mx.array(np.random.default_rng(5).normal(0, 8, (2 * S, n)).astype(np.float16))
    sm = FL.ForeignSwitchGLU.__new__(FL.ForeignSwitchGLU)
    sm.limit = None
    sm.set_act_kind(SITU)
    ref = np.array(FL.ForeignSwitchGLU._glu(sm, y2[:S], y2[S:])).astype(np.float64)
    a, b = np.array(y2[:S]).astype(np.float64), np.array(y2[S:]).astype(np.float64)
    assert np.abs(ref - _situ(a, b)).max() <= 2e-3 * np.abs(ref).max()         # the stock-op fallback is the formula
    got = np.array(FL.glu_rows(y2, S, n, 10.0, SITU)).astype(np.float64)       # limit is unused with an activation kind
    assert np.abs(got - ref).max() <= 2e-3 * np.abs(ref).max() + 1e-6


@pytest.mark.parametrize("g2a", ["1", "0"])
def test_decode_slots(monkeypatch, g2a):
    """S <= GATHER_MAX: g2a pair + glu_rows(situ) + down (flag g2a) and the generic gather + stock _glu (flag off)."""
    monkeypatch.setenv("KYOJIN_FOREIGN_G2A", g2a)
    g, u, d = _experts()
    sm = FL.ForeignSwitchGLU(g, u, d, act_kind=SITU)
    assert g.g2a_ok() == (g2a == "1")
    x, ids = _inputs(3, 4)
    calls = []
    real = FL.glu_rows
    monkeypatch.setattr(FL, "glu_rows", lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    y = sm(x.reshape(1, 3, KD), ids.reshape(1, 3, 4)).reshape(12, KD)
    assert (len(calls) == 1 and calls[0][-1] == SITU) == (g2a == "1")            # the fused activation launch took the kind (flag off: generic gather + stock ops)
    _close(y, _ref_mlp(g, u, d, x, ids, 4))


def test_tile_slots(monkeypatch):
    monkeypatch.setenv("KYOJIN_FOREIGN_G2A", "1")
    g, u, d = _experts()
    ref_in = _inputs(3, 4)
    ref = _ref_mlp(g, u, d, *ref_in, 4)
    for p in (g, u, d):
        assert FT.tile_ok(p.n, p.k, FL.TILE_UB)
        p.tl = FT.to_tile(p.qs, p.meta, FL.TILE_UB)
        mx.eval(p.tl)
        p.qs = p.meta = None
    sm = FL.ForeignSwitchGLU(g, u, d, act_kind=SITU)
    assert sm.tile_ok()
    x, ids = ref_in
    calls = []
    real = FT.gather_glu
    monkeypatch.setattr(FT, "gather_glu", lambda *a, **k: (calls.append(k), real(*a, **k))[1])
    _close(sm(x.reshape(1, 3, KD), ids.reshape(1, 3, 4)).reshape(12, KD), ref)
    assert len(calls) == 1 and calls[0]['act'] == SITU


def test_kqgmm_prefill(monkeypatch):
    monkeypatch.setenv("KYOJIN_KQ_GMM_GLM", "1")
    monkeypatch.delenv("KYOJIN_KQ_GMM_SCHED", raising=False)
    g, u, d = _experts()
    sm = FL.ForeignSwitchGLU(g, u, d, act_kind=SITU)
    T, K = 40, 4                                                                  # 160 slots > GATHER_MAX
    x, ids = _inputs(T, K)
    assert all(KQ.ok(p) for p in (g, u, d))
    y = KQ.mlp(sm, x, ids, K, 32)
    assert y is not None
    _close(y, _ref_mlp(g, u, d, x, ids, K))
    y2 = sm(x.reshape(1, T, KD), ids.reshape(1, T, K))                           # through the call: the kqgmm branch takes it
    _close(y2, _ref_mlp(g, u, d, x, ids, K))


def test_kqnax_prefill(monkeypatch):
    monkeypatch.setenv("KYOJIN_KQ_GMM_GLM", "1")
    monkeypatch.setenv("KYOJIN_KQ_GMM_SCHED", "omlx")
    g, u, d = _experts()
    sm = FL.ForeignSwitchGLU(g, u, d, act_kind=SITU)
    T, K = 160, 4                                                                 # 640 slots >= 48 x 12 experts
    x, ids = _inputs(T, K)
    assert NX.ok(g) and NX.ok(d)
    y = NX.mlp(sm, x, ids, K)
    assert y is not None
    _close(y, _ref_mlp(g, u, d, x, ids, K))


def test_foreign_moe_elementwise():
    a = mx.array(np.random.default_rng(1).normal(0, 8, (6, 512)).astype(np.float16))
    b = mx.array(np.random.default_rng(2).normal(0, 8, (6, 512)).astype(np.float16))
    sm = FL.ForeignSwitchGLU.__new__(FL.ForeignSwitchGLU)
    sm.limit = None
    sm.set_act_kind(SITU)
    ref = np.array(FL.ForeignSwitchGLU._glu(sm, a, b)).astype(np.float64)
    got = np.array(FMO._silu_mul(a, b, SITU)).astype(np.float64)
    assert np.abs(got - ref).max() <= 2e-3 * np.abs(ref).max() + 1e-6
    assert np.array_equal(np.array(FMO._silu_mul(a, b)), np.array(FMO._silu_mul(a, b, None)))
