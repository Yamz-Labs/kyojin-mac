# SPDX-License-Identifier: Apache-2.0
"""N2 hyper-connections x4 against the served references on synthetic tensors (CPU):
identity form vs mlx-vlm hy_v4 hyper_connection (the vendor class of the identity form) and vs DS4.1's served hc_expand with comb = I;
Sinkhorn form vs the DS4.1 served path (mlx-vlm deepseek_v41 hc_mix_coeffs / DeepseekV41Block.hc_pre / hc_expand, deepseek_v4 HyperHead)."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import foreign_skeleton_hc as HC

D, HCN, EPS, NEPS = 64, 4, 1e-6, 1e-6


def _rng(seed):
    return np.random.default_rng(seed)


def _stream(rng, B=2, L=3, dtype=mx.float32):
    return mx.array(rng.standard_normal((B, L, HCN, D)).astype(np.float32)).astype(dtype)


def _w(rng, rows):
    fn = mx.array((rng.standard_normal((rows, HCN * D)) * 0.05).astype(np.float32))
    base = mx.array((rng.standard_normal((rows,)) * 0.3).astype(np.float32))
    return fn, base


def _eq(a, b):
    mx.eval(a, b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert mx.array_equal(a, b).item(), float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_form_of():
    assert HC.form_of({"hyper_connection.count": 4}) == "identity"
    assert HC.form_of({"hyper_connection.count": 4, "hyper_connection.sinkhorn_iterations": 20}) == "sinkhorn"


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("mag", [1.0, 2.0])
def test_identity_vs_hy_v4(dtype, mag):
    from mlx_vlm.models.hy_v4 import hyper_connection as V
    rng = _rng(1)
    cfg = SimpleNamespace(hc_mult=HCN, hc_eps=EPS, hc_magnitude=mag, rms_norm_eps=NEPS, hidden_size=D)
    ref = V.IdentityHyperConnectionPre(cfg)
    ref.hc_fn, ref.hc_base = _w(rng, 2 * HCN)
    ref.hc_scale = mx.array([0.7, 1.3], dtype=mx.float32)
    h = _stream(rng, dtype=dtype)
    rx, rpost = ref(h)
    pre, post, comb = HC.split(HC.mixes(h, ref.hc_fn, NEPS), ref.hc_scale, ref.hc_base, HCN, EPS, mag)
    assert comb is None
    _eq(HC.collapse(h, pre), rx)
    _eq(post, rpost)
    # the patch's +eps on post (M:5414-5415) is exactly V's post + eps
    _, post_e, _ = HC.split(HC.mixes(h, ref.hc_fn, NEPS), ref.hc_scale, ref.hc_base, HCN, EPS, mag, post_eps=True)
    _eq(post_e, rpost + EPS)
    y = mx.array(rng.standard_normal((2, 3, D)).astype(np.float32)).astype(dtype)
    _eq(HC.expand(y, h, post), V.hc_expand(y, h, rpost))
    head = V.IdentityHyperHead(cfg)
    head.hc_head_fn, head.hc_head_base = _w(rng, HCN)
    head.hc_head_scale = mx.array([0.9], dtype=mx.float32)
    _eq(HC.head(h, head.hc_head_fn, head.hc_head_base, head.hc_head_scale, EPS, NEPS), head(h))


def test_identity_expand_vs_ds41_comb_eye():
    from mlx_vlm.models.deepseek_v41 import language as L41
    rng = _rng(2)
    h, y = _stream(rng), mx.array(rng.standard_normal((2, 3, D)).astype(np.float32))
    post = mx.array(rng.random((2, 3, HCN)).astype(np.float32))
    eye = mx.broadcast_to(mx.eye(HCN, dtype=mx.float32), (2, 3, HCN, HCN))
    _eq(HC.expand(y, h, post), L41.hc_expand(y, h, post, eye))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("iters", [1, 20])
def test_sinkhorn_vs_ds41_served(dtype, iters):
    from mlx_vlm.models.deepseek_v41 import language as L41
    rng = _rng(3 + iters)
    fn, base = _w(rng, (2 + HCN) * HCN)
    scale = mx.array([0.8, 1.1, 1.4], dtype=mx.float32)
    h = _stream(rng, dtype=dtype)
    rpre, rpost, rcomb = L41.hc_mix_coeffs(h, fn, scale, base, HCN, iters, EPS, NEPS)
    pre, post, comb = HC.split(HC.mixes(h, fn, NEPS), scale, base, HCN, EPS, 2.0, iters)
    _eq(pre, rpre)
    _eq(post, rpost)
    _eq(comb, rcomb)
    _eq(HC.collapse(h, pre), L41.DeepseekV41Block.hc_pre(h, rpre))
    y = mx.array(rng.standard_normal((2, 3, D)).astype(np.float32)).astype(dtype)
    _eq(HC.expand(y, h, post, comb), L41.hc_expand(y, h, rpost, rcomb))
    # one sub-layer through HcStream = the served chain (sub = a fixed linear map)
    W = mx.array((rng.standard_normal((D, D)) * 0.1).astype(np.float32)).astype(dtype)
    sub = lambda x: x @ W
    st = HC.HcStream(fn, base, scale, HCN, EPS, 2.0, NEPS, iters)
    _eq(st(h, sub), L41.hc_expand(sub(L41.DeepseekV41Block.hc_pre(h, rpre)), h, rpost, rcomb))


@pytest.mark.parametrize("L", [1, 40])
def test_head_vs_ds4_hyperhead(L):
    from mlx_vlm.models.deepseek_v4 import hyper_connection as V4
    rng = _rng(9)
    ref = V4.HyperHead(SimpleNamespace(hc_mult=HCN, rms_norm_eps=NEPS, hc_eps=EPS, hidden_size=D))
    ref.fn, ref.base = _w(rng, HCN)
    ref.scale = mx.array([1.2], dtype=mx.float32)
    h = _stream(rng, L=L)
    _eq(HC.head(h, ref.fn, ref.base, ref.scale, EPS, NEPS), ref(h))


def test_embed():
    x = mx.arange(2 * 3 * D, dtype=mx.float32).reshape(2, 3, D)
    s = HC.embed(x, HCN)
    assert s.shape == (2, 3, HCN, D) and all(mx.array_equal(s[:, :, i], x).item() for i in range(HCN))
