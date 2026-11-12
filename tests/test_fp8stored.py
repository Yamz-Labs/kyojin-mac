"""Flag fp8stored (KYOJIN_FP8STORED): FP8 E4M3 and GGUF Q8_0 linears read as stored (one byte per weight).

CPU half: Pack layouts and their float32 readers; Q8_0 to_pack equals the GGUF decoder; flag off keeps Q8_0 out of the kernel types and the 4-bit kernel sources are untouched.
Metal half (needs_gpu): GEMV, the decoded path above GEMV8_ROWS and the non-GEMV shapes against a float32 reference, same-input groups, saturation. Small shapes (< 1 GiB)."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_metal as FM, foreign_pack as FP, foreign_synth as SY

RNG = lambda s=0: np.random.default_rng(s)


def _fp8_pack(n, k, seed=0, scale=None):
    r = RNG(seed)
    q = r.integers(0, 256, (n, k), dtype=np.uint8)
    q[(q & 127) == 127] = 0x38                                        # the NaN codes are not weights
    return FP.Pack("fp8", n, k, q, None, (r.uniform(0.01, 2.0, n) if scale is None else scale).astype(np.float32))


def _fp8b_pack(n, k, seed=0, bn=128, bk=128):
    """fp8b as the loader builds it: E4M3 bytes, the block scale of each 32-unit (half, constant over bn x bk blocks) and a power-of-two row scale."""
    r = RNG(seed)
    q = r.integers(0, 256, (n, k), dtype=np.uint8)
    q[(q & 127) == 127] = 0x38
    g = r.uniform(0.25, 2.0, (-(-n // bn), -(-k // bk))).astype(np.float32)
    meta = g[np.arange(n) // bn][:, (np.arange(k // 32) * 32) // bk].astype(np.float16)
    return FP.Pack("fp8b", n, k, q, meta, np.float32(2.0 ** -6))


def _q8_pack(n, k, seed=0):
    return GF.to_pack("Q8_0", SY.random_blocks("Q8_0", n, k, RNG(seed)), k)


def test_e4m3_table_values():
    t = np.array(FM.E4M3_TABLE, np.float32)
    assert np.array_equal(t, FP.E4M3_TAB)
    assert t[0x38] == 1.0 and t[0x7E] == 448.0 and t[0xFE] == -448.0 and t[1] == 2.0 ** -9 and t[0x7F] == 0 and t[0xFF] == 0


def test_fp8_reader_applies_row_scale():
    p = _fp8_pack(16, 64)
    w = FP.dequant(p)
    assert np.array_equal(w, FP.E4M3_TAB[p.qs] * p.gscale[:, None])
    assert np.array_equal(FP.dequant(p, g=False), FP.E4M3_TAB[p.qs])


def test_q8_0_to_pack_is_the_gguf_decoder():
    raw = SY.random_blocks("Q8_0", 8, 128, RNG(3))
    p = GF.to_pack("Q8_0", raw, 128)
    assert p.qs.shape == (8, 128) and p.meta.shape == (8, 4) and p.qs.dtype == np.uint8
    assert np.array_equal(FP.dequant(p), GF.dequant("Q8_0", raw, 128))


def test_flag_off_keeps_the_old_types_and_kernels():
    assert "Q8_0" not in GF.KERNEL_TYPES and "q8_0" not in FM.FORMATS and "fp8" not in FM.FORMATS and "q8_0" not in FP.FORMATS
    for f in FM.FORMATS:                                             # the 4-bit sources do not mention the 8-bit table
        assert "E4M3T" not in FM.gemv_source(f) and "E4M3T" not in FM.dequant_source(f)


def test_flag_default_is_the_f16_copy_arm(monkeypatch):
    from kyojin_mlx import foreign_linear as FL, flags
    monkeypatch.delenv("KYOJIN_FP8STORED", raising=False)
    assert flags.FLAGS["fp8stored"].default == "1" and FL.fp8stored() and not FL.fp8_f16copy()
    for v, st, cp in (("0", False, False), ("1", True, False), ("2", True, True)):
        monkeypatch.setenv("KYOJIN_FP8STORED", v)
        assert (FL.fp8stored(), FL.fp8_f16copy()) == (st, cp)


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


def _pack(fmt, n, k, seed=0):
    return _fp8_pack(n, k, seed) if fmt == "fp8" else _fp8b_pack(n, k, seed) if fmt == "fp8b" else _q8_pack(n, k, seed)


@needs_gpu
@pytest.mark.parametrize("fmt", FM.FORMATS8)
@pytest.mark.parametrize("R", (1, 2, 4, 8, 9, 12, 13, 16, 24, 32, 33, 56, 57, 70))
def test_apply_matches_float32_reference(fmt, R):
    """R <= GEMV8_ROWS: the stored GEMV; to DENSE8_ROWS: the simdgroup-matrix kernel (32-row slices); above: decode + matmul. All against x @ W^T in float32."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p = _pack(fmt, 128, 1024)
    x = RNG(5).standard_normal((R, 1024)).astype(np.float16)
    y = FL.DevPack(p).apply(mx.array(x))
    mx.eval(y)
    assert y.dtype == mx.float16 and y.shape == (R, 128)
    _check(y, x.astype(np.float32) @ FP.dequant(p).T)


@needs_gpu
@pytest.mark.parametrize("fmt", FM.FORMATS8)
@pytest.mark.parametrize("R", (1, 4, 16))
def test_shapes_off_the_gemv_grid_take_the_decoded_path(fmt, R):
    """n not a multiple of 16: gemv_ok is False, the simdgroup-matrix kernel answers (any n, k % 32 == 0); k off the 512 grid (k % 32 == 0) runs the GEMV's partial last chunk."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    for n, k in ((24, 1024), (64, 288)):
        p = _pack(fmt, n, k, seed=n)
        d = FL.DevPack(p)
        assert d.gemv_ok() == (n % 16 == 0)
        x = RNG(6).standard_normal((R, k)).astype(np.float16)
        _check(d.apply(mx.array(x)), x.astype(np.float32) @ FP.dequant(p).T)


@needs_gpu
@pytest.mark.parametrize("fmt", FM.FORMATS8)
def test_dequant_kernel_matches_reader(fmt):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p = _pack(fmt, 48, 512)
    w = np.array(FL.DevPack(p).dequant(), np.float32)
    assert np.allclose(w, FP.dequant(p), rtol=1e-3, atol=1e-7)


@needs_gpu
@pytest.mark.parametrize("fmt", FM.FORMATS8)
@pytest.mark.parametrize("R", (1, 8, 20))
def test_same_input_group_equals_members(fmt, R):
    """DevPack.grouped over 8-bit members (per-row scales kept) = each member alone, split at the cuts."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    ps = [_pack(fmt, n, 1024, seed=n) for n in (64, 32, 48)]
    devs = [FL.DevPack(p) for p in ps]
    x = mx.array(RNG(7).standard_normal((R, 1024)).astype(np.float16))
    g = FL.DevPack.grouped(devs)
    outs = g.apply_split(x, [0, 64, 96, 144])
    for d, o in zip(devs, outs):
        _check(o, np.array(d.apply(x), np.float32), tol=2e-3)


@needs_gpu
@pytest.mark.parametrize("fmt", FM.FORMATS8)
@pytest.mark.parametrize("R", (1, 12))
def test_overflow_saturates_never_inf(fmt, R):
    """A row sum past the fp16 range writes +-65504, not inf (the GEMV kernel; the decoded path above GEMV8_ROWS is an MLX fp16 matmul, as the old dense path was)."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    p = _pack(fmt, 32, 512)
    if fmt == "fp8":
        p.gscale = np.full(32, 1.0e5, np.float32)
    else:
        p.meta = np.full_like(p.meta, 1.0e4)
        p.qs = np.full_like(p.qs, 100 if fmt == "q8_0" else 0x70)          # int8 100 / E4M3 256
        if fmt == "fp8b":
            p.gscale = np.float32(1.0)
    x = np.full((R, 512), 100.0, np.float16)
    y = np.array(FL.DevPack(p).apply(mx.array(x)), np.float32)
    assert np.isfinite(y).all() and np.abs(y).max() == 65504.0


@needs_gpu
@pytest.mark.parametrize("fmt", FM.FORMATS8)
@pytest.mark.parametrize("R", (1, 8, 15, 16, 40, 70))
def test_f16_copy_arm(fmt, R, monkeypatch):
    """KYOJIN_FP8STORED=2 keeps a resident fp16 copy (same values as the dequant kernel) and runs the matmul from F16_ROWS rows; answers stay on the float32 reference."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_FP8STORED", "2")
    assert FL.fp8stored() and FL.fp8_f16copy()
    p = _pack(fmt, 96, 1024)
    d = FL.DevPack(p)
    assert d.w16 is not None and d.w16.shape == (96, 1024)
    x = RNG(8).standard_normal((R, 1024)).astype(np.float16)
    _check(d.apply(mx.array(x)), x.astype(np.float32) @ FP.dequant(p).T)
    monkeypatch.setenv("KYOJIN_FP8STORED", "1")
    assert FL.DevPack(p).w16 is None and not FL.fp8_f16copy()


def test_8bit_sources_are_memoized():
    assert FM.gemv8_source("q8_0", 2) is FM.gemv8_source("q8_0", 2) and FM.dequant8_source("fp8") is FM.dequant8_source("fp8") and FM.mma8_source("fp8", 2) is FM.mma8_source("fp8", 2)
    assert sorted(FM.xperm8_index(64).tolist()) == list(range(64))
