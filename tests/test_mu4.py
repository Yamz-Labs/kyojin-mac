"""TQKVCache.update_many (one rotation + one quantize launch for a group of sessions) is bit-identical to one update_and_fetch per session."""
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def test_update_many_equals_per_session():
    from kyojin_mlx import tq as TQ
    g, H, d = 4, 4, 256
    mx.random.seed(3)
    def mk():
        mx.random.seed(3)
        cs = [TQ.TQKVCache(6, 6, 4, 64) for _ in range(g)]
        for c in cs:
            c.update_and_fetch(mx.random.normal((1, H, 200, d)).astype(mx.bfloat16), mx.random.normal((1, H, 200, d)).astype(mx.bfloat16))
        return cs
    a, b = mk(), mk()
    for c, e in zip(a, b):                                   # same prefill content
        pass
    mx.random.seed(5)
    for n in (3, 1, 3):
        k = mx.random.normal((g, H, n, d)).astype(mx.bfloat16); v = mx.random.normal((g, H, n, d)).astype(mx.bfloat16)
        for j, c in enumerate(a):
            c.update_and_fetch(k[j:j + 1], v[j:j + 1])
        TQ.TQKVCache.update_many(b, k, v)
    for c, e in zip(a, b):
        assert c.offset == e.offset
        mx.eval(c.kq, c.vq, c.rk, c.rv, e.kq, e.vq, e.rk, e.rv)
        for x, y in zip((*c.kq, *c.vq, c.rk, c.rv), (*e.kq, *e.vq, e.rk, e.rv)):
            L = c.offset
            assert x.shape == y.shape
        assert mx.array_equal(c.rk, e.rk).item() and mx.array_equal(c.rv, e.rv).item()
        assert mx.array_equal(c.kq[1][:, :, :c.offset], e.kq[1][:, :, :e.offset]).item()
        assert mx.array_equal(c.vq[0][:, :, :c.offset], e.vq[0][:, :, :e.offset]).item()
