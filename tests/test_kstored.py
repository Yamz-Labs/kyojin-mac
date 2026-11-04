"""Flag kstored (KYOJIN_KSTORED): GGUF Q5_K / Q6_K / IQ4_XS, and the 3-bit class Q3_K / IQ3_S / IQ3_XXS (GEMV only), read as stored.

CPU half: the repack (foreign_gguf.to_pack) read back by the device-layout reader equals the ggml reference decoder; flag default; sources memoized.
Metal half (needs_gpu): GEMV, dequant kernel and the routed paths against a float32 reference, off-grid shapes, saturation, the resident fp16 copy. Small shapes (< 1 GiB)."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_metal as FM, foreign_pack as FP, foreign_synth as SY

RNG = lambda s=0: np.random.default_rng(s)
TYPES_MMA = ("Q5_K", "Q6_K", "IQ4_XS")
LOW = ("Q3_K", "IQ3_S", "IQ3_XXS", "Q2_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M")
TYPES = TYPES_MMA + LOW


def _pack(t, n, k, seed=0):
    return GF.to_pack(t, SY.random_blocks(t, n, k, RNG(seed)), k)


@pytest.mark.parametrize("t", TYPES)
def test_repack_is_the_ggml_decoder(t):
    raw = SY.random_blocks(t, 12, 1024, RNG(3))
    p = GF.to_pack(t, raw, 1024)
    assert p.fmt == t.lower() and p.qs.shape == (12, 1024 // 32 * FP.K_QS_BYTES[p.fmt])
    assert np.array_equal(FP.dequant(p), GF.dequant(t, raw, 1024))
    assert p.nbytes <= raw.nbytes * (1.012 if t in TYPES_MMA else 1.08)       # same bytes (Q6_K pads two per block; Q3_K keeps its 16 scales as int8)


def test_flag_default_copy_and_bad_shapes_refused(monkeypatch):
    from kyojin_mlx import foreign_linear as FL, flags
    monkeypatch.delenv("KYOJIN_KSTORED", raising=False)
    assert flags.FLAGS["kstored"].default == "1" and FL.kstored() and not FL.k_f16copy()
    monkeypatch.setenv("KYOJIN_KSTORED", "0")
    assert not FL.kstored() and not FL.k_f16copy()
    for v, st, cp in (("1", True, False), ("2", True, True)):
        monkeypatch.setenv("KYOJIN_KSTORED", v)
        assert (FL.kstored(), FL.k_f16copy()) == (st, cp)
    p = _pack("Q6_K", 8, 256)
    with pytest.raises(ValueError):
        FP.Pack("q6_k", 8, 256, p.qs, p.meta[:, :4], extra=p.extra)


def test_k_sources_are_memoized_and_clamped():
    for f in FM.FORMATSK:
        assert FM.gemvk_source(f, 2) is FM.gemvk_source(f, 2) and FM.dequantk_source(f) is FM.dequantk_source(f)
        assert "65504" in FM.gemvk_source(f, 2)


def test_mmak_source_memoized_clamped_and_perm_is_a_permutation():
    for f in FM.FORMATSK_MMA:
        assert FM.mmak_source(f, 2) is FM.mmak_source(f, 2) and "65504" in FM.mmak_source(f, 2)
        idx = FM.xpermk_index(f, 512)
        assert sorted(idx.tolist()) == list(range(512))


# ------------------------------------------------------------------------------------------------------------------ Mac only
def _gpu():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no Metal GPU")


def _check(y, ref, tol=2e-3):
    y = np.array(y, np.float32)
    assert np.isfinite(y).all()
    assert np.abs(y - ref).max() <= tol * max(np.abs(ref).max(), 1e-6), np.abs(y - ref).max() / np.abs(ref).max()


def _small(t, n, k, seed=0):
    """Blocks with small scales so the fp16 matmul reference does not overflow."""
    raw = SY.random_blocks(t, n, k, RNG(seed))
    _, be, bb = GF.TYPES_BY_NAME[t]
    b3 = raw.reshape(n, k // be, bb)
    SY.set_scales(t, b3, RNG(seed + 1).uniform(1e-4, 5e-4, (n, k // be)))
    return GF.to_pack(t, raw, k), raw


@needs_gpu
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("R", (1, 2, 4, 5, 8, 9, 12, 16, 32, 33, 40))
def test_apply_matches_float32_reference(t, R, monkeypatch):
    """R <= GEMVK_ROWS: the stored GEMV; above: decode + matmul. Against x @ W^T in float32 of the ggml decode."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "1")
    p, raw = _small(t, 128, 1024)
    x = RNG(5).standard_normal((R, 1024)).astype(np.float16)
    y = FL.DevPack(p).apply(mx.array(x))
    mx.eval(y)
    assert y.dtype == mx.float16 and y.shape == (R, 128)
    _check(y, x.astype(np.float32) @ GF.dequant(t, raw, 1024).T)


@needs_gpu
@pytest.mark.parametrize("t", TYPES)
def test_dequant_kernel_matches_reader(t):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 48, 512)
    w = np.array(FL.DevPack(p).dequant(), np.float32)
    assert np.allclose(w, GF.dequant(t, raw, 512), rtol=1e-3, atol=1e-6)


@needs_gpu
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("R", (1, 4))
def test_shapes_off_the_gemv_grid_take_the_decoded_path(t, R):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    for n, k in ((24, 1024), (64, 768)):
        p, raw = _small(t, n, k, seed=n)
        d = FL.DevPack(p)
        assert not d.gemv_ok()
        x = RNG(6).standard_normal((R, k)).astype(np.float16)
        _check(d.apply(mx.array(x)), x.astype(np.float32) @ GF.dequant(t, raw, k).T)


@needs_gpu
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("R", (1, 3, 8, 12))
def test_overflow_saturates_never_inf(t, R, monkeypatch):
    """A row sum past the fp16 range writes +-65504, not inf (the GEMV kernel and the decode kernel)."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    if t not in TYPES_MMA and R > FL.GEMVK_ROWS:
        pytest.skip("3-bit types above the GEMV rows use the plain fp16 matmul (as the copy arm does)")
    monkeypatch.setenv("KYOJIN_KSTORED", "1")
    raw = SY.random_blocks(t, 32, 512, RNG(9))
    _, be, bb = GF.TYPES_BY_NAME[t]
    b3 = raw.reshape(32, 512 // be, bb)
    SY.set_scales(t, b3, np.full((32, 512 // be), 60000.0))
    d = FL.DevPack(GF.to_pack(t, raw, 512))
    x = np.full((R, 512), 100.0, np.float16)
    y = np.array(d.apply(mx.array(x)), np.float32)
    assert np.isfinite(y).all() and np.abs(y).max() == 65504.0
    assert np.isfinite(np.array(d.dequant(), np.float32)).all()


@needs_gpu
@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("R", (1, 3, 8, 12, 16, 40))
def test_f16_copy_arm(t, R, monkeypatch):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "2")
    p, raw = _small(t, 96, 1024)
    d = FL.DevPack(p)
    assert d.w16 is not None and d.w16.shape == (96, 1024)
    x = RNG(8).standard_normal((R, 1024)).astype(np.float16)
    _check(d.apply(mx.array(x)), x.astype(np.float32) @ GF.dequant(t, raw, 1024).T)
    monkeypatch.setenv("KYOJIN_KSTORED", "1")
    assert FL.DevPack(p).w16 is None


@needs_gpu
@pytest.mark.parametrize("t", TYPES_MMA)
@pytest.mark.parametrize("R", (2, 5, 9, 17, 32))
@pytest.mark.parametrize("ntn", (1, 2))
def test_mmak_kernel_matches_reference_and_off_grid_rows(t, R, ntn):
    """The stored simdgroup-matrix kernel, N not a multiple of 8 NTN, against the float32 decode."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p, raw = _small(t, 100, 1024, seed=R)
    x = RNG(11).standard_normal((R, 1024)).astype(np.float16)
    y = FL.DevPack(p)._mmak(mx.array(x), ntn)
    mx.eval(y)
    assert y.shape == (R, 100)
    _check(y, x.astype(np.float32) @ GF.dequant(t, raw, 1024).T)


@needs_gpu
@pytest.mark.parametrize("t", TYPES_MMA)
def test_mmak_route(t, monkeypatch):
    """Rows 9..32 go through mmak (rows up to 16 of the fmma formats go through fmma), above through decode + matmul; KYOJIN_FOREIGN_MMAK_ROWS=0 switches it off."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "1")
    monkeypatch.setenv("KYOJIN_FOREIGN_WIDE", "0")                       # the wide NAX product takes rows 9..32 first (tests/test_widerows.py)
    d = FL.DevPack(_small(t, 64, 512)[0])
    calls = []
    orig = d._mmak
    monkeypatch.setattr(d, "_mmak", lambda x, ntn=2: (calls.append(x.shape[0]), orig(x, ntn))[1])
    for R in (1, 8, 9, 32, 33):
        d.apply(mx.zeros((R, 512), mx.float16))
    assert calls == ([32] if t.lower() in FL.FM._FMMA_PAIR else [9, 32])
    monkeypatch.setattr(FL, "MMAK_ROWS", 0)
    d.apply(mx.zeros((6, 512), mx.float16))
    assert calls == ([32] if t.lower() in FL.FM._FMMA_PAIR else [9, 32])


@needs_gpu
@pytest.mark.parametrize("t", LOW)
def test_three_bit_types_never_take_mmak(t, monkeypatch):
    """Without the fp16 copy the rows above the GEMV go to decode + matmul (no stored simdgroup-matrix kernel for these types)."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "1")
    d = FL.DevPack(_small(t, 64, 512)[0])
    monkeypatch.setattr(d, "_mmak", lambda *a: pytest.fail("mmak called"))
    x = RNG(2).standard_normal((12, 512)).astype(np.float16)
    _check(d.apply(mx.array(x)), x.astype(np.float32) @ GF.dequant(t, _small(t, 64, 512)[1], 512).T)


@needs_gpu
@pytest.mark.parametrize("R", (2, 5, 9, 16))
@pytest.mark.parametrize("t", LOW)
def test_fmma_of_the_low_bit_types_matches_reference(t, R, monkeypatch):
    """The 3- and 2-bit classes have the few-row MMA (the lane decodes its unit with the GEMV decoder): float32 reference, ragged n, perm is a permutation."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_KSTORED", "1")
    pk, raw = _small(t, 72, 1024)
    d = FL.DevPack(pk)
    x = RNG(4).standard_normal((R, 1024)).astype(np.float16)
    _check(d._fmma(mx.array(x)), x.astype(np.float32) @ GF.dequant(t, raw, 1024).T)
    assert sorted(FM.fmma_xperm_index(t.lower(), 512).tolist()) == list(range(512))


@pytest.mark.parametrize("t", ("Q2_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M"))
def test_two_bit_class_repack_equals_reference_on_ragged_rows(t):
    raw = SY.random_blocks(t, 5, 2048, RNG(11))
    assert np.array_equal(FP.dequant(GF.to_pack(t, raw, 2048)), GF.dequant(t, raw, 2048))
