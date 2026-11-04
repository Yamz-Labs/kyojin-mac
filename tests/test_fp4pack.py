import sys, os
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx"))
import mlx.core as mx, numpy as np, pytest
from mlx_lm.models.deepseek_v41 import _fake_quant_fp4
from kyojin_mlx import fp4pack

@pytest.mark.parametrize("block,e4", [(16, True), (32, False)])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 40.0])
def test_roundtrip_bit_exact(block, e4, scale):
    mx.random.seed(3)
    x = (mx.random.normal((2, 37, 512)) * scale).astype(mx.float16)
    x = mx.concatenate([x, mx.zeros((2, 1, 512), mx.float16)], axis=1)   # an all-zero row (floor scale)
    ref = _fake_quant_fp4(x, block_size=block, e4m3_scale=e4)
    c, s = fp4pack.pack(x, block, e4)
    got = fp4pack.unpack(c, s, block, e4)
    assert c.shape == (2, 38, 256) and c.dtype == mx.uint8 and s.shape == (2, 38, 512 // block)
    assert np.array_equal(np.array(got.view(mx.uint16)), np.array(ref.view(mx.uint16)))


def test_attend_packed_pool_is_bit_identical():
    from kyojin_mlx import dsa_attend
    rng = np.random.default_rng(5)
    H, D, W, RD, n, K = 64, 512, 128, 64, 4000, 300
    pool = _fake_quant_fp4(mx.array(rng.standard_normal((1, n, D)).astype(np.float32) * 2).astype(mx.float16), block_size=16, e4m3_scale=True)
    ring = mx.array(rng.standard_normal((1, W, D)).astype(np.float32)).astype(mx.float16)
    q = mx.array(rng.standard_normal((1, 1, H * D)).astype(np.float32)).astype(mx.float16)
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    ang = rng.uniform(0, 6, RD // 2)
    rt = mx.array(np.stack([np.cos(ang), np.sin(ang)], -1).reshape(-1).astype(np.float32))
    it = mx.array(np.stack([np.cos(ang), -np.sin(ang)], -1).reshape(-1).astype(np.float32))
    idx = rng.choice(n, K, replace=False).astype(np.int32); idx[::7] = -1
    idx = mx.array(idx)
    pos = mx.array([9000], dtype=mx.int32)
    ref = dsa_attend.attend(q, ring, pool, idx, sink, rt, it, pos, H, RD, 0.0442, False)
    c, s = fp4pack.pack(pool, 16, True)
    got = dsa_attend.attend_packed(q, ring, c, s, idx, sink, rt, it, pos, H, RD, 0.0442, False)
    mx.eval(ref, got)
    assert np.array_equal(np.array(got.view(mx.uint16)), np.array(ref.view(mx.uint16)))
