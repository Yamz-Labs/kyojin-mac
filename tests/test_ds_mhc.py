"""Prefill mhc kernels vs the stock compiled ops, bit for bit: python tests/test_ds_mhc.py"""
import numpy as np, mlx.core as mx
import importlib
L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
from kyojin_mlx import ds_pf_mhc as M
stock_exp, stock_pre = L.hc_expand, L.DeepseekV41Block.__dict__["hc_pre"].__func__
exp, pre = M._wrap_expand(stock_exp), M._wrap_pre(stock_pre)
rng = np.random.default_rng(1)
for dt in (mx.float16, mx.bfloat16):
    for S in (33, 512, 2048):
        D = 5120
        x = mx.array(rng.standard_normal((1, S, D))).astype(dt); r = mx.array(rng.standard_normal((1, S, 4, D))).astype(dt)
        post = mx.array(rng.random((1, S, 4)).astype(np.float32)); comb = mx.array(rng.random((1, S, 4, 4)).astype(np.float32))
        a, b = stock_exp(x, r, post, comb), exp(x, r, post, comb)
        c, d = stock_pre(r, post), pre(r, post)
        print(dt, S, "expand equal", bool(mx.array_equal(a, b)), "pre equal", bool(mx.array_equal(c, d)))
