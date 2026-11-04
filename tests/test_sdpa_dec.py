"""Split-K decode attention (kyojin_mlx/sdpa_dec.py) against fp32 stock SDPA: causal tail, bool array masks (per head / broadcast, a fully masked
split), and the routing rules (bool masks in, additive masks / sinks / TurboQuant caches out)."""
import numpy as np
import mlx.core as mx
import pytest

from kyojin_mlx import sdpa_dec as SD

pytestmark = pytest.mark.skipif(not SD.nax_ok(), reason="needs the Neural Accelerator tiles (M5 / A19)")


def _ref(q, k, v, mask):
    f = lambda x: x.astype(mx.float32)
    return mx.fast.scaled_dot_product_attention(f(q), f(k), f(v), scale=q.shape[-1] ** -0.5, mask=mask)


def _err(a, b):
    return mx.abs(a.astype(mx.float32) - b).max().item()


@pytest.mark.parametrize("B,Hq,Hk,D,T,N", [(1, 16, 4, 256, 1, 4096), (1, 16, 4, 256, 5, 4096), (2, 32, 8, 128, 2, 3000), (1, 64, 1, 512, 1, 2048)])
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("per_head", [False, True])
def test_masked(B, Hq, Hk, D, T, N, dt, per_head):
    rng = np.random.default_rng(0)
    q, k, v = (mx.random.normal(s, key=mx.random.key(i)).astype(dt) for i, s in enumerate([(B, Hq, T, D), (B, Hk, N, D), (B, Hk, N, D)]))
    m = rng.random((B, Hq if per_head else 1, T, N)) > 0.4
    m[..., :600] = False                                       # the first splits see no key at all
    m = mx.array(m)
    tol = 4 * max(_err(mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask=m), _ref(q, k, v, m)), 1e-4)
    for ns in (None, 1, 4, 16):
        assert _err(SD.sdpa_decode(q, k, v, D ** -0.5, ns=ns, mask=m), _ref(q, k, v, m)) <= tol


@pytest.mark.parametrize("T", [1, 3, 5])
def test_causal(T):
    q = mx.random.normal((1, 16, T, 256)).astype(mx.float16)
    k = mx.random.normal((1, 4, 5000, 256)).astype(mx.float16)
    v = mx.random.normal((1, 4, 5000, 256)).astype(mx.float16)
    mask = "causal" if T > 1 else None
    tol = 4 * max(_err(mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / 16, mask=mask), _ref(q, k, v, mask)), 1e-4)
    assert _err(SD.sdpa_decode(q, k, v, 1 / 16), _ref(q, k, v, mask)) <= tol


def test_route_rules(monkeypatch):
    monkeypatch.setattr(SD, "_chip", lambda: dict(cores=10, tpc=2.0, min_n1={256: 1024}, min_n={256: 1024}))
    q = mx.zeros((2, 16, 1, 256), dtype=mx.float16)
    k = mx.zeros((2, 4, 2048, 256), dtype=mx.float16)
    bm = mx.ones((2, 1, 1, 2048), dtype=mx.bool_)
    assert SD.route(q, k, k, None, 0.06, None) is not None
    assert SD.route(q, k, k, None, 0.06, bm) is not None
    assert SD.route(q, k, k, None, 0.06, bm.astype(mx.float16)) is None             # additive mask: stock
    assert SD.route(q, k, k, None, 0.06, mx.ones((3, 1, 1, 2048), dtype=mx.bool_)) is None
    assert SD.route(q, k, k, None, 0.06, None, sinks=mx.zeros((16,))) is None

    class TurboQuantKVCache:
        pass
    assert SD.route(q, k, k, TurboQuantKVCache(), 0.06, None) is None
    q5 = mx.zeros((1, 64, 1, 512), dtype=mx.float16)                                  # D 512 one row: no min_n1 entry, stock
    k5 = mx.zeros((1, 1, 2048, 512), dtype=mx.float16)
    assert SD.route(q5, k5, k5, None, 0.04, bm[:1]) is None
