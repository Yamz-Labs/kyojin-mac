"""Metal kernels vs the numpy reference. Run on the Mac (Apple Silicon, mlx installed); skipped elsewhere."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import kernels as KN, ref_numpy as R


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_decode_bit_exact(case):
    """Trellis decode to fp16 Wr: must match the numpy reference (and so the engine) bit for bit."""
    _, K, t, _, _ = case
    kt, nt, _ = t.shape
    got = np.array(KN.decode_wr(KN.to_words(t), K, kt, nt))
    want = R.decode_tiles(t).T
    bad = int((got.view(np.uint16) != want.view(np.uint16)).sum())
    assert bad == 0, f"{bad} of {got.size} fp16 values differ"


def test_rot():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 512)).astype(np.float32)
    suh = (rng.choice([-1, 1], 512) * rng.uniform(0.5, 1.5, 512)).astype(np.float16)
    got = np.array(KN.rot(mx.array(x), mx.array(suh)))
    want = R.fwht128(x * suh.astype(np.float32), axis=1) * R.INV_SQRT_HAD
    assert _rel(got, want) < 1e-6


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("prefetch", [True, False])
def test_gemv(case, rows, prefetch):
    name, K, t, su, sv = case
    x = np.random.default_rng(7).standard_normal((rows, t.shape[0] * 16)).astype(np.float32)
    ref = R.gemv(x, t, su, sv)
    outs = []
    for tgk in (None, 1, 2):
        lin = KN.Mul1Gemv(t, su, sv, rows=rows, prefetch=prefetch, tgk=tgk)
        y = np.array(lin(mx.array(x)))
        assert _rel(y, ref) < 1e-4, (name, tgk, _rel(y, ref))
        y2 = np.array(lin(mx.array(x)))
        assert np.array_equal(y, y2), "not deterministic"
        outs.append(y)


@pytest.mark.parametrize("cb", ["mcg", "3inst"])
@pytest.mark.parametrize("K", [2, 4])
def test_decode_other_codebooks(K, cb):
    """Published EXL3 packs use mcg; same decode kernel, other codebook."""
    t, su, sv = R.random_linear(256, 256, K, seed=5)
    got = np.array(KN.decode_wr(KN.to_words(t), K, 16, 16, cb))
    assert np.array_equal(got.view(np.uint16), R.decode_tiles(t, cb).T.view(np.uint16))


@pytest.mark.parametrize("K", [2, 4])
def test_gemv_mcg(K):
    t, su, sv = R.random_linear(256, 256, K, seed=6)
    x = np.random.default_rng(2).standard_normal((1, 256)).astype(np.float32)
    y = np.array(KN.Mul1Gemv(t, su, sv, cb="mcg")(mx.array(x)))
    assert _rel(y, R.gemv(x, t, su, sv, cb="mcg")) < 1e-4


def _v0_kernels():
    """The first barrier-based Hadamard kernels (threadgroup memory), kept as the bitwise reference of the simdgroup ones."""
    from kyojin_mlx import metal_src as S
    rot0 = mx.fast.metal_kernel(name="v0_rot", input_names=["x", "suh", "params"], output_names=["xh"], source=S.ROT_SOURCE_V0)
    fin0 = mx.fast.metal_kernel(name="v0_fin", input_names=["part", "svh", "params"], output_names=["y"], source=S.FINAL_SOURCE_V0)
    return rot0, fin0


@pytest.mark.parametrize("in_dtype", [mx.float32, mx.float16])
def test_rot_simdgroup_bitwise_vs_barrier_kernel(in_dtype):
    rot0, _ = _v0_kernels()
    rng = np.random.default_rng(3)
    rows, k = 5, 1024 + 128                       # 9 blocks: not a multiple of the 4 blocks per threadgroup
    x = mx.array(rng.standard_normal((rows, k)).astype(np.float32)).astype(in_dtype)
    suh = mx.array((rng.choice([-1, 1], k) * rng.uniform(0.5, 1.5, k)).astype(np.float16))
    old = rot0(inputs=[x, suh, mx.array([k], dtype=mx.uint32)], grid=(k, rows, 1), threadgroup=(128, 1, 1),
               output_shapes=[(rows, k)], output_dtypes=[mx.float32])[0]
    new32, new16 = KN.rot(x, suh), KN.rot(x, suh, mx.float16)
    assert np.array_equal(np.array(old).view(np.uint32), np.array(new32).view(np.uint32))
    assert np.array_equal(np.array(old.astype(mx.float16)).view(np.uint16), np.array(new16).view(np.uint16))


@pytest.mark.parametrize("part_dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("out_dtype", [mx.float32, mx.float16])
def test_final_simdgroup_bitwise_vs_barrier_kernel(part_dtype, out_dtype):
    _, fin0 = _v0_kernels()
    rng = np.random.default_rng(4)
    tgk, rows, n = 3, 2, 640                      # 5 blocks
    part = mx.array(rng.standard_normal((tgk, rows, n)).astype(np.float32)).astype(part_dtype)
    svh = mx.array((rng.choice([-1, 1], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    fp = mx.array([tgk, n, rows], dtype=mx.uint32)
    old = fin0(inputs=[part.astype(mx.float32), svh, fp], template=[("OT", out_dtype)], grid=(n, rows, 1), threadgroup=(128, 1, 1),
               output_shapes=[(rows, n)], output_dtypes=[out_dtype])[0]
    grid, tg = KN.S.had_grid(n // 128, rows)
    new = KN.fin_kernel(out_dtype, part_dtype)(inputs=[part, svh, fp], template=[("OT", out_dtype)], grid=grid, threadgroup=tg,
                                               output_shapes=[(rows, n)], output_dtypes=[out_dtype])[0]
    view = np.uint32 if out_dtype == mx.float32 else np.uint16
    assert np.array_equal(np.array(old).view(view), np.array(new).view(view))
