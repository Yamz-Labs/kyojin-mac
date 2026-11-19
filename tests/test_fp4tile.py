"""fp4tile (flag glmtile): the decode tile layout of NVFP4 routed experts. Metal only; small shapes (H = 512 / I = 256 stand for GLM's 4096 / 2048)."""
import numpy as np
import pytest


def _gpu():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no Metal GPU")
E, H, I = 5, 1024, 512


def _stack(n, k, seed):
    import mlx.core as mx
    rng = np.random.default_rng(seed)
    qs = rng.integers(0, 2 ** 32, size=(E, n, k // 8), dtype=np.uint32)
    sc = rng.integers(0x28, 0x40, size=(E, n, k // 16), dtype=np.uint8)
    sc[0, :3] = np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7, 0x7F, 8, 0x30, 0x38, 0x40, 0x41, 0x10, 0x03], np.uint8), k // 16 // 16 + 1)[:k // 16]   # zero / subnormal / NaN-code scales
    meta = sc.view("<u2").reshape(E, n, k // 32)
    gs = rng.uniform(0.5, 2.0, E).astype(np.float32)
    return qs, meta, gs


@needs_gpu
def test_roundtrip_exact():
    import mlx.core as mx
    from kyojin_mlx import fp4tile as FT
    qs, meta, gs = _stack(I, H, 1)
    for ub in (2, 4):
        tl = FT.to_tile(mx.array(qs), mx.array(meta), ub)
        assert tl.nbytes == qs.nbytes + meta.nbytes                          # same bytes: the stored layout is replaced, not added to
        qs2, meta2 = FT.from_tile(tl, I, H, ub)
        assert np.array_equal(np.array(qs2), qs)
        m1, m2 = np.array(meta2).view(np.uint8) & 127, meta.view(np.uint8) & 127
        assert np.array_equal(np.where(m2 == 127, 0, m2), np.where(m1 == 127, 0, m1))   # scale codes round-trip (sign bit / NaN code are ignored by every kernel)


@needs_gpu
@pytest.mark.parametrize("n,k", [(I, H), (H, I)])
def test_gather_matches_float64_reference(n, k):
    import mlx.core as mx
    from kyojin_mlx import fp4tile as FT
    qs, meta, gs = _stack(n, k, 2)
    W = FT.tile_ref(qs, meta, gs)
    rng = np.random.default_rng(3)
    T, S = 3, 11
    x = (rng.standard_normal((T, k)) * 2).astype(np.float16)
    xr, ei = rng.integers(0, T, S).astype(np.uint32), rng.integers(0, E, S).astype(np.uint32)
    ref = np.stack([W[ei[s]] @ x[xr[s]].astype(np.float64) for s in range(S)])
    tl = FT.to_tile(mx.array(qs), mx.array(meta), 2)
    for nsg in (4, 8):
        y = np.array(FT.gather(mx.array(x), mx.array(xr), mx.array(ei), tl, mx.array(gs), n, k, 2, nsg)).astype(np.float64)
        assert np.abs(y - ref).max() / np.abs(ref).max() < 1e-3                # fp16 output rounding only
    wq = np.array(FT.dequant(tl, mx.array(gs), E, n, k, 2)).astype(np.float64)
    assert np.abs(wq - W).max() / np.abs(W).max() < 5e-4


@needs_gpu
@pytest.mark.parametrize("limit", [10.0, None])
def test_fused_glu(limit):
    import mlx.core as mx
    from kyojin_mlx import fp4tile as FT
    qg, mg, gg = _stack(I, H, 4)
    qu, mu, gu = _stack(I, H, 5)
    Wg, Wu = FT.tile_ref(qg, mg, gg), FT.tile_ref(qu, mu, gu)
    rng = np.random.default_rng(6)
    S = 9
    x = (rng.standard_normal((2, H)) * (1.0 if limit else 0.01)).astype(np.float16)
    xr, ei = rng.integers(0, 2, S).astype(np.uint32), rng.integers(0, E, S).astype(np.uint32)
    h = np.array(FT.gather_glu(mx.array(x), mx.array(xr), mx.array(ei), FT.to_tile(mx.array(qg), mx.array(mg), 2), FT.to_tile(mx.array(qu), mx.array(mu), 2),
                               mx.array(gg), mx.array(gu), I, H, limit, 2, 8)).astype(np.float64)
    a = np.stack([Wg[ei[s]] @ x[xr[s]].astype(np.float64) for s in range(S)]).astype(np.float16).astype(np.float64)
    b = np.stack([Wu[ei[s]] @ x[xr[s]].astype(np.float64) for s in range(S)]).astype(np.float16).astype(np.float64)
    if limit:
        a, b = np.minimum(a, limit), np.clip(b, -limit, limit)
    ref = a / (1 + np.exp(-a)) * b
    assert np.abs(h - ref).max() / np.abs(ref).max() < 2e-3


@needs_gpu
def test_switch_glu_flag(monkeypatch):
    """ForeignSwitchGLU on the same packs, stored layout vs flag glmtile: decode rows (fused tile launches) agree to fp16 rounding, prefill rows (untiled copy) are identical."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL, foreign_pack as FP
    rng = np.random.default_rng(7)

    def packs(n, k):
        return [FP.Pack("nvfp4", n, k, rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8), rng.integers(0x28, 0x40, size=(n, k // 16), dtype=np.uint8).view('<u2').reshape(n, k // 32), float(rng.uniform(.5, 2)))
                for _ in range(E)]
    pk = [packs(I, H), packs(I, H), packs(H, I)]

    def build(tile):
        monkeypatch.setattr(FL, "TILE", tile)
        return FL.ForeignSwitchGLU(*[FL.DevPack(p) for p in pk], limit=10.0)
    a, b = build(False), build(True)
    assert all(d.tl is not None and d.qs is None and d.meta is None for d in (b.g, b.u, b.d)) and all(d.tl is None for d in (a.g, a.u, a.d))
    for T in (1, 4, 20):                                                       # 8 / 32 slots decode, 160 slots > GATHER_MAX: prefill
        x = (mx.random.normal((T, H)) * 0.004).astype(mx.float16)
        inds = mx.array(np.stack([rng.choice(E, 4, replace=False) for _ in range(T)]).astype(np.uint32))
        ya, yb = np.array(a(x, inds)).astype(np.float64), np.array(b(x, inds)).astype(np.float64)
        if T * 4 > FL.GATHER_MAX:
            assert np.array_equal(ya, yb)
        else:
            assert np.abs(ya - yb).max() / np.abs(ya).max() < 2e-3
