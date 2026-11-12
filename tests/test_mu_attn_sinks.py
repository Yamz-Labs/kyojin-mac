"""Pooled multi-session decode attention (mu_attn) with learned attention sinks (gpt-oss) == per-session mx.fast SDPA with the same sinks."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import mu_attn as MA, sdpa_dec as SD

needs_nax = pytest.mark.skipif(not getattr(SD, "nax_ok", lambda: True)(), reason="NAX tiles unavailable")


@needs_nax
@pytest.mark.parametrize("with_sinks", [False, True])
@pytest.mark.parametrize("P", [0, 512])
def test_pooled_matches_sdpa(with_sinks, P):
    rng = np.random.default_rng(5)
    Nu, Hk, G, D = 3, 8, 8, 64
    Hq = Hk * G
    lens = [37, 5, 120]
    cap = 256
    f = lambda *s: mx.array(rng.standard_normal(s).astype(np.float16))
    q = f(Nu, Hq, 1, D)
    pk, pv = (f(1, Hk, P, D), f(1, Hk, P, D)) if P else (None, None)
    tk, tv = f(Nu, Hk, cap, D), f(Nu, Hk, cap, D)
    sinks = mx.array((rng.standard_normal(Hq) * 2).astype(np.float32)) if with_sinks else None
    scale = D ** -0.5
    out = MA.pooled_attention(q, pk, pv, tk, tv, mx.array(lens, dtype=mx.int32), scale, sinks)
    refs = []
    for u in range(Nu):
        k = tk[u:u + 1, :, :lens[u]]
        v = tv[u:u + 1, :, :lens[u]]
        if P:
            k, v = mx.concatenate([pk, k], axis=2), mx.concatenate([pv, v], axis=2)
        refs.append(mx.fast.scaled_dot_product_attention(q[u:u + 1], k, v, scale=scale, mask=None,
                                                         sinks=sinks.astype(q.dtype) if sinks is not None else None))
    ref = mx.concatenate(refs, axis=0)
    mx.eval(out, ref)
    a, b = np.array(out, np.float32), np.array(ref, np.float32)
    assert np.abs(a - b).max() <= 2e-3 * max(1.0, np.abs(b).max()), np.abs(a - b).max()
