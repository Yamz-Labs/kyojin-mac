"""cmd5: the fp32 reference of the Metal kernel tests must be exact fp32. CPU tests (run anywhere mlx imports).

On a GPU with tensor units (M5) the stock fp32 `@` rounds both operands to ~10 mantissa bits for 2+ rows (ULTRA.md, tests/test_attn_prefill.py), so
`x @ w.T` is ~8e-4 away from the exact product on the shapes of test_xgemv_matches_the_f32_matmul, 4x above that test's 2e-4 bound, while a kernel that
multiplies in fp32 is within ~1e-5. Only float32 weights at 2+ rows lose bits (fp16 / bf16 weights fit in 10 bits, one row is a GEMV): the failing
set was exactly float32 at R = 2, 5, 6, 24. The Metal confirmation is tools/cmd5_check.py."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import stagek2 as SK2  # noqa: E402

BOUND = 2e-4                                    # tests/test_dec_stage_k2.py::test_xgemv_matches_the_f32_matmul, unchanged


def trunc10(a):
    """Round-to-nearest to 10 mantissa bits, the operand width of the tensor-unit fp32 matmul."""
    u = a.astype(np.float32).view(np.uint32)
    return (((u + (1 << 12)) >> 13) << 13).view(np.float32)


def shapes(R):
    rng = np.random.default_rng(R)
    x = (rng.standard_normal((R, 5120)) * 0.7).astype(np.float16)
    w = (rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)
    return x, w, x.astype(np.float64) @ w.astype(np.float64).T


@pytest.mark.parametrize("R", [1, 2, 5, 6, 24])
def test_exact_reference_is_within_fp32_rounding_of_float64(R):
    x, w, want = shapes(R)
    got = np.array(SK2.exact_f32_matmul_t(mx.array(x), mx.array(w)))
    assert got.dtype == np.float32 and got.shape == want.shape
    err = np.abs(got - want).max()
    assert err < 3e-5, err                      # measured 5.8e-6 .. 1.2e-5 on |y| up to ~4: fp32 sum of 5120 terms; 6.7x under BOUND


@pytest.mark.parametrize("R", [2, 5, 6, 24])
def test_a_tensor_unit_matmul_would_break_the_bound(R):
    """The operand truncation alone puts the old reference above the bound (measured 6.8e-4 .. 9.4e-4 for these R)."""
    x, w, want = shapes(R)
    old = trunc10(x).astype(np.float32) @ trunc10(w).T
    assert np.abs(old - want).max() > BOUND


def test_chunking_does_not_change_the_result(monkeypatch):
    x, w, _ = shapes(3)
    one = np.array(SK2.exact_f32_matmul_t(mx.array(x), mx.array(w[:70])))
    monkeypatch.setattr(SK2, "_EXACT_CHUNK", 3 * 5120 * 16)            # 16 weight rows per chunk: 5 chunks, the last ragged
    many = np.array(SK2.exact_f32_matmul_t(mx.array(x), mx.array(w[:70])))
    np.testing.assert_array_equal(one, many)


def test_xgemv_ref_takes_every_stored_dtype():
    x, w, _ = shapes(2)
    for dt in (mx.float16, mx.bfloat16, mx.float32):
        r0, r1 = SK2.xgemv_ref(mx.array(x), mx.array(w[:20]).astype(dt), mx.array(w[20:33]).astype(dt))
        assert r0.shape == (2, 20) and r1.shape == (2, 13) and r0.dtype == mx.float32
