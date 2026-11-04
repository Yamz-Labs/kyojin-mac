"""Bit test of mlp_pf.swiglu_rot against fin(gate), fin(up), mlx_lm swiglu, rot(down). usage: pf2k_mlp_test.py"""
import mlx.core as mx, numpy as np
from mlx_lm.models.activations import swiglu
from kyojin_mlx import kernels as KN, mlp_pf as MP
rng = np.random.default_rng(3)
def h(shape, sc): return mx.array((rng.standard_normal(shape) * sc).astype(np.float16))
for rows, F, sc in ((64, 2560, 1.0), (333, 9216, 3.0), (512, 9216, 30.0), (65, 1024, 0.01), (130, 4096, 300.0)):
    gi, ui = h((rows, F), sc), h((rows, F), sc)
    svg = mx.array(rng.uniform(0.3, 2.0, F).astype(np.float16)) * mx.array(rng.choice([-1, 1], F).astype(np.float16))
    svu = mx.array(rng.uniform(0.3, 2.0, F).astype(np.float16)); suhd = mx.array((rng.choice([-1.0, 1.0], F)).astype(np.float16))
    fp = mx.array([1, F, rows], dtype=mx.uint32); grid, tg = KN.S.had_grid(F // 128, rows)
    fin = lambda inner, sv: KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner.reshape(1, rows, F), sv, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg, output_shapes=[(rows, F)], output_dtypes=[mx.float16])[0]
    act = swiglu(fin(gi, svg), fin(ui, svu))
    for perm in (False, True):
        ref = KN.rot(act, suhd, mx.float16, perm=perm)
        for mode in ("",):
            out = MP.swiglu_rot(gi, ui, svg, svu, suhd, perm); mx.eval(ref, out)
            print(rows, F, sc, "perm", perm, "bitexact", bool(mx.array_equal(ref, out).item()), "nbad", int((ref != out).sum().item()), "nan", bool(mx.isnan(ref).any().item()))
