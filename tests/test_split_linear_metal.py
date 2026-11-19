# SPDX-License-Identifier: Apache-2.0
"""SplitLinear (mixed GGUF types of one sanitize-fused linear, members stored) gives the fp16 fused matrix's output within fp16 rounding (Metal)."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_linear as FL, foreign_pack as FP

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")


def _raw(rng, n, k):
    nb = k // 256
    r = rng.integers(0, 256, (n, nb * 210), dtype=np.uint8)
    r.reshape(n, nb, 210)[:, :, 208:210] = np.array([0x00, 0x1C], np.uint8)       # fp16 d = 0.0039
    return r


@pytest.mark.parametrize("rows", [1, 4, 40])
def test_split_matches_the_fp16_fused_matrix(rows):
    rng = np.random.default_rng(rows)
    k = 1024
    a = GF.to_pack("Q6_K", _raw(rng, 96, k), k)
    b = FP.Pack("q8_0", 40, k, rng.integers(0, 256, (40, k), dtype=np.uint8), (rng.standard_normal((40, k // 32)) * 0.001).astype(np.float16))
    sp = FL._concat_rows_exact([a, b])
    lin = FL.make_linear(sp)
    assert isinstance(lin, FL.SplitLinear) and lin.out_features == 136
    w16 = np.concatenate([FP.dequant(a), FP.dequant(b)], 0).astype(np.float16)            # the removed path: one resident fp16 matrix
    ref = FL.ForeignLinear(FP.Pack("dense", 136, k, w16))
    x = mx.array(rng.standard_normal((1, rows, k)).astype(np.float16))
    y, r = lin(x), ref(x)
    assert y.shape == r.shape == (1, rows, 136) and y.dtype == mx.float16
    y, r = np.array(y.astype(mx.float32)), np.array(r.astype(mx.float32))
    ex = x.astype(mx.float32).reshape(rows, k) @ mx.array(w16.astype(np.float32)).T
    tol = 2e-3 * float(np.abs(np.array(ex)).max()) + 1e-3                               # fp16 rounding of the two orders of summation
    assert np.abs(y.reshape(rows, 136) - np.array(ex)).max() <= tol and np.abs(y - r).max() <= 2 * tol
