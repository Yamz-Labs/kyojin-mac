import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import qwen3_next as QN

from kyojin_mlx import gdn_norm as GN


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("rows", [1, 32, 100, 512 * 32, 4096 * 32])
def test_gated_norm_bit_equal_to_stock(dt, rows):
    mx.random.seed(rows)
    h = (mx.random.normal((1, rows // 32 + (rows < 32), 32 if rows >= 32 else rows, 128)) * 3).astype(dt)
    z = (mx.random.normal(h.shape) * 3).astype(dt)
    m = QN.Qwen3NextRMSNormGated(128, eps=1e-6)
    m.weight = (mx.random.normal((128,)) * 0.5 + 1).astype(mx.float32)
    x = mx.fast.rms_norm(h, m.weight, m.eps)
    ref = (nn.silu(z.astype(mx.float32)) * x.astype(mx.float32)).astype(dt)
    out = GN.gated_norm(h, z, m.weight, m.eps)
    mx.eval(ref, out)
    assert out.dtype == dt and out.shape == h.shape
    assert bool(mx.array_equal(out, ref))
