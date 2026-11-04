"""mlp_pf.swiglu_rot is bit-identical to fin(gate), fin(up), mlx_lm swiglu, rot(down)."""
import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.activations import swiglu
from kyojin_mlx import kernels as KN, mlp_pf as MP


@pytest.mark.parametrize("rows,F,sc", [(64, 2560, 1.0), (333, 9216, 3.0), (512, 9216, 30.0), (65, 1024, 0.01), (1, 128, 1.0)])
@pytest.mark.parametrize("perm", [False, True])
def test_swiglu_rot_bit_exact(rows, F, sc, perm):
    rng = np.random.default_rng(3)
    h = lambda shape, s: mx.array((rng.standard_normal(shape) * s).astype(np.float16))
    gi, ui = h((rows, F), sc), h((rows, F), sc)
    svg = mx.array(rng.uniform(0.3, 2.0, F).astype(np.float16)) * mx.array(rng.choice([-1, 1], F).astype(np.float16))
    svu = mx.array(rng.uniform(0.3, 2.0, F).astype(np.float16))
    suhd = mx.array(rng.choice([-1.0, 1.0], F).astype(np.float16))
    fp = mx.array([1, F, rows], dtype=mx.uint32)
    grid, tg = KN.S.had_grid(F // 128, rows)
    fin = lambda inner, sv: KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner.reshape(1, rows, F), sv, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg,
                                                                  output_shapes=[(rows, F)], output_dtypes=[mx.float16])[0]
    ref = KN.rot(swiglu(fin(gi, svg), fin(ui, svu)), suhd, mx.float16, perm=perm)
    assert bool(mx.array_equal(ref, MP.swiglu_rot(gi, ui, svg, svu, suhd, perm)).item())
