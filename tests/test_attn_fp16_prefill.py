# SPDX-License-Identifier: Apache-2.0
"""Flag attn_fp16 on the prefill path (stock KV cache): the attention runs on fp16 rows with fp32 accumulation and stays within fp16 rounding of the fp32 path
(CPU, no Metal): global layer (kpool selection, "causal" mask) and windowed layer, chunks then decode rows; the cache rows are fp16 and saturated."""
import numpy as np

import mlx.core as mx

from kyojin_mlx import attn_fp16 as AF, foreign_skeleton_mla as MLA
from test_hy4_attn_slice import _cache, _kp_model, _swa_model  # noqa: E402

mx.set_default_device(mx.cpu)
TOL16 = 3e-2          # relative to max(1, max|out|): fp16 rounding of q / latent / scores through 3 chained products


def _go(m, xs, window=0, swa=False, on=False):
    from mlx_lm.models.base import create_causal_mask
    AF.ON = on
    m.__dict__.pop("_kb16", None); m.__dict__.pop("_vb16", None)
    if on:
        m.__dict__["_kb16"], m.__dict__["_vb16"] = m.k_b.astype(mx.float16), m.v_b.astype(mx.float16)
    c, outs = _cache(), []
    for x in xs:
        L = x.shape[1]
        mask = create_causal_mask(L, c.offset, window_size=window) if swa else ("causal" if L > 1 else None)
        outs.append(m(x, mask=mask, cache=c))
    mx.eval(outs)
    return outs, c


def _check(m, rng, window=0, swa=False):
    xs = [mx.array((0.3 * rng.standard_normal((1, n, 64))).astype(np.float32)) for n in (300, 200)] + [mx.array((0.3 * rng.standard_normal((1, 1, 64))).astype(np.float32)) for _ in range(2)]
    try:
        ref, c0 = _go(m, xs, window, swa, False)
        new, c1 = _go(m, xs, window, swa, True)
    finally:
        AF.ON = False
    assert c1.keys.dtype == mx.float16 and c0.keys.dtype == mx.float32
    mag = max(1.0, max(float(mx.max(mx.abs(x)).item()) for x in ref))
    d = max(float(mx.max(mx.abs(x - y)).item()) for x, y in zip(ref, new)) / mag
    print(f"attn_fp16 prefill: max rel diff {d:.3g} (max out {mag:.3g})")
    assert 0 < d <= TOL16


def test_global_layer_fp16_path():
    """The selection keeps every pool (top_k past the context): only the fp16 rounding differs. (A selection that bites can flip a borderline pool: that is the noise band's job.)"""
    rng = np.random.default_rng(21)
    _check(_kp_model(rng, 64, 4, 32, 24, 8, 16, 2, 16, top_k=4096), rng)


def test_windowed_layer_fp16_path():
    rng = np.random.default_rng(22)
    _check(_swa_model(rng, 64, 4, 32, 24, 8, 16, 128), rng, 128, True)
