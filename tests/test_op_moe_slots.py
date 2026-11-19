"""M2: in-kernel expert dedupe of the NVFP4 decode block above 32 slots (two ballots). Micro tests on random tensors (Metal); S <= 32 sources are untouched."""
import pytest

mx = pytest.importorskip("mlx.core")
import numpy as np  # noqa: E402

if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)


def _ge_setup(T, K, overlap, seed=0):
    import types
    from kyojin_mlx import glm_moe_rows as R
    E, H, I = 24, 2048, 1024

    def mk(k, n, sd):
        rng = np.random.default_rng(sd)
        qs = mx.array(rng.integers(0, 2 ** 32, size=(E, n, k // 8), dtype=np.uint32))
        mt = mx.array((rng.integers(0x2C, 0x44, size=(E, n, k // 32)) | (rng.integers(0x2C, 0x44, size=(E, n, k // 32)) << 8)).astype(np.uint16))
        return types.SimpleNamespace(qs=qs, meta=mt, gs=mx.full((E,), 0.004, dtype=mx.float32), k=k, n=n)
    g, u, d = mk(H, I, 1), mk(H, I, 2), mk(I, H, 3)
    rng = np.random.default_rng(T * 100 + K + seed)
    rows = []
    for t in range(T):
        if overlap and t:
            keep = list(rows[-1][:K // 2])
            rows.append(keep + [x for x in rng.permutation(E) if x not in keep][:K - len(keep)])
        else:
            rows.append(list(rng.choice(E, size=K, replace=False)))
    inds = mx.array(np.array(rows, np.uint32).reshape(-1))
    x2 = mx.array((rng.normal(size=(T, H)) * 0.5).astype(np.float16))
    tab = R.group(inds, T, K) if T > 1 else None
    h0 = R.pair_glu(g, u, x2, inds, tab, T, K, 10.0)
    y0 = R.down(d, h0, inds, tab, T, K)
    return g, u, d, x2, inds, h0, y0


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("T,K", [(4, 10), (3, 12), (4, 16), (5, 8), (4, 8)])
def test_g2m_wide_dedupe(T, K, overlap):
    """T*K above 32 (up to 64): g2m / inl / omx give the G2c rows result (bit-identical; omx within 2 ulp), the old 32-slot shape (4, 8) included."""
    from kyojin_mlx import glm_moe_e as GE
    if T * K > GE.S_MAX_WIDE or T > 8:
        pytest.skip("beyond the wide limit")
    g, u, d, x2, inds, h0, y0 = _ge_setup(T, K, overlap)
    if T > GE.TMAX:                                         # more rows than the per-row mem[] of the g2m / omx kernels: only the limit check applies
        assert not GE.ok(d, d, T, K)
        return
    assert GE.ok(d, d, T, K) == (T * K <= GE.S_MAX_WIDE)
    for rps, nsg in ((1, 4), (2, 2)):
        h = GE.pair_glu("g2m", g, u, x2, inds, T, K, 10.0, rps, nsg)
        assert mx.array_equal(h, h0)
        assert mx.array_equal(GE.down("g2m", d, h0, inds, T, K, rps, nsg), y0)
    assert mx.array_equal(GE.pair_glu("inl", g, u, x2, inds, T, K, 10.0), h0)
    assert mx.array_equal(GE.down("inl", d, h0, inds, T, K), y0)
    ho = GE.pair_glu("omx", g, u, x2, inds, T, K, 10.0)
    ulp = int(np.abs(np.asarray(ho.view(mx.int16)).astype(np.int32) - np.asarray(h0.view(mx.int16)).astype(np.int32)).max())
    assert ulp <= 2
    print(f"T {T} K {K} slots {T * K} overlap {overlap}: g2m/inl bit-identical to rows, omx max {ulp} ulp")


def test_g2m_limits_are_parameters():
    import types
    from kyojin_mlx import glm_moe_e as GE
    d = types.SimpleNamespace(k=2048, n=4096)
    assert GE.ok(d, d, 4, 10) and GE.ok(d, d, 4, 16) and not GE.ok(d, d, 4, 17)
    assert GE.dc_ok(d, 4, 8) and not GE.dc_ok(d, 4, 10)         # down_combine keeps one simdgroup per slot: 32 slots = 1024 threads




@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("T,K", [(4, 10), (3, 12), (4, 16)])
def test_g2m_wide_with_rows_unroll(T, K, overlap):
    """KYOJIN_MOE_ROWS2_UNROLL (ju) together with the two-ballot dedupe (wide, T*K > 32): same bits as the rows result, and the kernels are the ju + wide variants."""
    from kyojin_mlx import glm_moe_e as GE
    g, u, d, x2, inds, h0, y0 = _ge_setup(T, K, overlap)
    assert T * K > GE.S_MAX
    k = GE._g2m_kern(True, 10.0, 0, 0, 1, 1, None)
    assert k is not GE._g2m_kern(True, 10.0, 0, 0, 0, 1, None) and k is not GE._g2m_kern(True, 10.0, 0, 0, 1, 0, None)
    for rps, nsg in ((1, 4), (2, 2)):
        assert mx.array_equal(GE.pair_glu("g2m", g, u, x2, inds, T, K, 10.0, rps, nsg, 0, 0, 0, 1), h0)
        assert mx.array_equal(GE.down("g2m", d, h0, inds, T, K, rps, nsg, 0, 0, 0, 1), y0)
    assert mx.array_equal(GE.pair_glu("g2m", g, u, x2, inds, T, K, 10.0, 2, 2, 0, 1, 1, 1), h0)
    assert mx.array_equal(GE.down("g2m", d, h0, inds, T, K, 2, 2, 0, 1, 1, 1), y0)
