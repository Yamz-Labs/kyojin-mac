"""Fp8 fake quant folded into the rot kernels (metal_src.with_fq8): rot(fq(x)) == rot(x, fq8=True) to the bit for the single, grouped-gather and multi-member rots,
random inputs over many scales plus exact zeros, powers of two and block ties. Needs a Mac (Metal) and mlx-vlm 0.7.10."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41.fakequant")
from kyojin_mlx import kernels as KN, metal_src as S                                  # noqa: E402
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq          # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def inputs(seed, k, rows):
    rng = np.random.default_rng(seed)
    sc = float(rng.choice([1e-6, 1e-3, 0.1, 1.0, 30.0, 300.0, 3000.0]))
    x = (rng.standard_normal((rows, k)) * sc).astype(np.float32)
    if seed % 5 == 0:
        x *= rng.choice([0.0, 1.0, 1.0], size=x.shape)
    if seed % 7 == 0:
        x[:, ::3] = np.sign(x[:, ::3]) * 2.0 ** rng.integers(-12, 9, size=x[:, ::3].shape)
    return mx.array(x).astype(mx.float16), mx.array(rng.choice([-1.0, 1.0], size=k).astype(np.float16))


def same(a, b):
    mx.eval(a, b)
    return np.array_equal(np.array(a).view(np.uint32), np.array(b).view(np.uint32))


@pytest.mark.parametrize("k", [128, 1280, 2304, 4096, 5120])
def test_single_rot(k):
    for seed in range(40):
        rows = 1 + seed % 4
        x, suh = inputs(seed, k, rows)
        assert same(KN.rot(fq(x), suh), KN.rot(x, suh, fq8=True)), (k, seed)


def test_gather_rot_and_multi_rot():
    from kyojin_mlx.moe import _kernel
    from kyojin_mlx import fused_group as FG
    k = 2304
    for seed in range(30):
        x, suh = inputs(seed, k, 1)
        suh2 = mx.concatenate([suh, suh[::-1]])
        ids = mx.array([0, 1, 0], dtype=mx.int32)
        out = []
        for fold in (False, True):
            xin = x if fold else fq(x)
            kr = _kernel(("t_rotg", fold), name=f"t_rotg_{int(fold)}", input_names=["x", "ids", "suh0", "suh1", "params"], output_names=["xh"],
                         source=S.rot_gather_source(2, True, fold))
            grid, tg = S.had_grid(k // 128, 6)
            out.append(kr(inputs=[xin, ids, mx.concatenate([suh[None], suh[None][:, ::-1]]), mx.concatenate([suh[None][:, ::-1], suh[None]]),
                                  mx.array([k, 3], dtype=mx.uint32)], grid=grid, threadgroup=tg, output_shapes=[(6, k)], output_dtypes=[mx.float32])[0])
        assert same(out[0], out[1]), seed
        res = []
        for fold in (False, True):
            kr = KN._kern(("t_s5rot", fold), name=f"t_s5rot_{int(fold)}", input_names=["x", "suh", "params"], output_names=["xh"], source=FG.rot_multi_source(True, fold))
            grid, tg = S.had_grid(k // 128, 1)
            res.append(kr(inputs=[x if fold else fq(x), suh2, mx.array([k, 1], dtype=mx.uint32)], grid=(grid[0], grid[1], 2), threadgroup=tg,
                          output_shapes=[(2, k)], output_dtypes=[mx.float32])[0])
        assert same(res[0], res[1]), seed
