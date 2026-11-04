"""Kernel index / window logic, run on any machine through kyojin_mlx.emulate (numpy only)."""
import numpy as np
import pytest

from kyojin_mlx import emulate as E, ref_numpy as R


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
def test_decode_matches_reference_bit_exact(K):
    trellis, _, _ = R.random_linear(128, 128, K, seed=10 + K)
    kt, nt = 8, 8
    words = R.trellis_words(trellis)
    got = E.decode_wr_t(words, K, kt, nt)
    want = R.decode_tiles(trellis).T
    assert np.array_equal(got.view(np.uint16), want.view(np.uint16))


@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("tgk", [1, 2, 4])
def test_gemv_mapping(K, tgk):
    kt, nt, rows = 24, 8, 2                       # k=384, n=128; tgk=4 leaves empty k-splits (myn == 0)
    trellis, suh, svh = R.random_linear(kt * 16, nt * 16, K, seed=3)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((rows, kt * 16)).astype(np.float32)
    xh = (R.fwht128(x * suh.astype(np.float32), axis=1) * R.INV_SQRT_HAD).astype(np.float32)
    part = E.gemv_partial(R.trellis_words(trellis), K, kt, nt, xh, tgk)
    y = E.finalize(part, svh)
    ref = R.gemv(x, trellis, suh, svh)
    err = np.abs(y - ref).max() / np.abs(ref).max()
    assert err < 1e-5, err


@pytest.mark.parametrize("cb", ["mcg", "3inst"])
@pytest.mark.parametrize("K", [2, 4])
def test_decode_other_codebooks(K, cb):
    trellis, _, _ = R.random_linear(128, 128, K, seed=40 + K)
    got = E.decode_wr_t_cb(R.trellis_words(trellis), K, 8, 8, cb)
    want = R.decode_tiles(trellis, cb).T
    assert np.array_equal(got.view(np.uint16), want.view(np.uint16))


def test_codebook_scalar_decoders_agree_with_reference():
    s = np.arange(65536, dtype=np.uint16)
    for cb in ("mul1", "mcg", "3inst"):
        want = R.codebook_values(s, cb)
        got = np.array([E.DECODERS[cb](int(v)) for v in s[::37]], np.float16)
        assert np.array_equal(got.view(np.uint16), want[::37].view(np.uint16)), cb
