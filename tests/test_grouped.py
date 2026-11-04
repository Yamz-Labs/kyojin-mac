"""Grouped wo_a (kyojin_mlx.grouped.GroupedGemv, wired into patch.Exl3MultiLinear) vs the per-group Exl3Linear path: bit-identical for batch 1 and 1-4 rows per group."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import kernels as KN
from kyojin_mlx.grouped import GroupedGemv
from kyojin_mlx.patch import Exl3MultiLinear


def make(G, kin, nout, K, seed):
    rng = np.random.default_rng(seed)
    tr = rng.integers(-32768, 32768, (G, kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], (G, kin)) * rng.uniform(0.5, 1.5, (G, kin))).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], (G, nout)) * rng.uniform(0.5, 1.5, (G, nout))).astype(np.float16)
    return mx.array(tr), mx.array(suh), mx.array(svh)


def per_group(ml, x):
    return mx.stack([ml.groups[g](x[:, g]) for g in range(ml.num_groups)], axis=1)


def bits(a):
    return np.array(a).view(np.uint16)


@pytest.fixture(scope="module", params=[("mul1", 5, 8), ("mul1", 3, 4), ("mcg", 5, 8), ("mcg", 2, 3)], ids=lambda p: f"{p[0]}-K{p[1]}-G{p[2]}")
def layer(request):
    cb, K, G = request.param
    ml = Exl3MultiLinear(*make(G, 512, 256, K, 7 + K), cb)
    return ml


@pytest.mark.parametrize("bs", [(1, 1), (1, 2), (1, 3), (1, 4)])
@pytest.mark.parametrize("dt", [np.float16, np.float32])
def test_bit_exact(layer, bs, dt):
    B, Sq = bs
    ml = layer
    x = mx.array((np.random.default_rng(B * 10 + Sq).standard_normal((B, ml.num_groups, Sq, ml.in_features)) * 2).astype(dt))
    got, want = ml.grouped(x), per_group(ml, x)
    assert got.dtype == want.dtype == mx.float16 and got.shape == want.shape == (B, ml.num_groups, Sq, ml.out_features)
    assert np.array_equal(bits(got), bits(want))


def test_tgk_is_the_per_group_one(layer):
    assert layer.grouped.tgk == layer.groups[0]._g[1].tgk


def test_zero_rows_keep_signed_zero(layer):
    ml = layer
    x = np.random.default_rng(3).standard_normal((1, ml.num_groups, 2, ml.in_features)).astype(np.float16)
    x[:, :, 1] = 0
    x = mx.array(x)
    assert np.array_equal(bits(ml.grouped(x)), bits(per_group(ml, x)))


def test_call_dispatch(layer, monkeypatch):
    from kyojin_mlx import exl3_linear as EL
    ml = layer
    x = mx.array(np.random.default_rng(1).standard_normal((1, ml.num_groups, 3, ml.in_features)).astype(np.float16))
    assert np.array_equal(bits(ml(x)), bits(per_group(ml, x)))              # grouped path
    x = mx.array(np.random.default_rng(2).standard_normal((1, ml.num_groups, 9, ml.in_features)).astype(np.float16))
    with monkeypatch.context() as m:
        m.setattr(EL, "DENSE_FUSED_MAX", 0)
        assert np.array_equal(bits(ml(x)), bits(per_group(ml, x)))          # more than 4 rows, no fused crossover: per-group decode + matmul
    got, want = np.array(ml(x)).astype(np.float32), np.array(per_group(ml, x)).astype(np.float32)
    assert np.linalg.norm(got - want) <= 5e-3 * np.linalg.norm(want)        # with the crossover: fused NAX launch set, fp16 rounding class


def test_batch_above_one_uses_per_group_path(layer):
    ml = layer
    x = mx.array(np.random.default_rng(4).standard_normal((2, ml.num_groups, 2, ml.in_features)).astype(np.float16))
    assert np.array_equal(bits(ml(x)), bits(per_group(ml, x)))
