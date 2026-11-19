"""step_compile: a runtime flag read inside a wrapped layer / mlp is honoured after a flags.set (GEN bump retraces the compiled program)."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache

from kyojin_mlx import flags as FL
from kyojin_mlx import step_compile as SC


class _Mlp(nn.Module):
    def __call__(self, x):
        return x * (3.0 if FL.value("moe_ksplit") == "1" else 2.0)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Mlp()

    def __call__(self, x, cache=None):
        cache.cache[0] = x
        return x * (10.0 if FL.value("moe_ksplit") == "1" else 1.0)


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer()]


def test_runtime_flag_after_compile_is_honoured():
    FL.set("moe_ksplit", "0")
    h = _Holder()
    SC.ENABLED = SC.ACTIVE = True
    assert SC.install(h, log=lambda *_: None, force=True) == 1
    l = h.layers[0]
    x = mx.ones((1, 1, 4))
    c = ArraysCache(size=1)
    for _ in range(4):                                   # warm, trace, replay
        assert mx.array_equal(l(x, cache=c), x).item()
        assert mx.array_equal(l.mlp(x), x * 2).item()
    assert SC.STATS["compiled"] >= 2
    try:
        FL.set("moe_ksplit", "1")
        assert mx.array_equal(l(x, cache=c), x * 10).item()
        assert mx.array_equal(l.mlp(x), x * 3).item()
        assert mx.array_equal(l(x, cache=c), x * 10).item()
        assert mx.array_equal(l.mlp(x), x * 3).item()
    finally:
        FL.set("moe_ksplit", "0")
        SC.ACTIVE = False


# ---- attention split: several call sites (static branch), closure inside a hyper-connection stream, passthrough mask, array-side host reads ----
import types
import pytest
from kyojin_mlx.foreign_skeleton_hc import HcStream
from kyojin_mlx.foreign_skeleton import _host_offset


class _KV:
    def __init__(self):
        self.offset, self.buf = 0, []


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.p = nn.Linear(d, d, bias=False)

    def __call__(self, x, mask, cache, freqs=None):
        cache.buf.append(self.p(x))
        cache.offset += 1
        k = mx.concatenate(cache.buf, axis=-2) if x.ndim == 3 else mx.concatenate([b.reshape(1, -1, b.shape[-1]) for b in cache.buf], axis=1)
        return (x.reshape(1, -1, x.shape[-1]) * 0 + k.mean(axis=1, keepdims=True) + mask.sum() * 0.001).reshape(x.shape)


class _Blk(nn.Module):
    def __init__(self, mode, d=8):
        super().__init__()
        self.mode = mode
        self.attn, self.mlp, self.n1, self.n2 = _Attn(d), nn.Linear(d, d), nn.RMSNorm(d), nn.RMSNorm(d)
        if mode == "hc":
            self.hcs = HcStream(mx.random.normal((4, 2 * d)) * 0.1, mx.random.normal((4,)) * 0.1, mx.array([1.0, 1.0]), 2, 1e-6, 1.0, 1e-6)

    def __call__(self, x, mask, cache, freqs=None):
        m = self.mode
        if m == "hc":
            def attn(y):
                return self.attn(self.n1(y), mask, cache, freqs)

            def ffn(y):
                return self.mlp(self.n2(y))
            h = self.hcs(x, attn)
            return self.hcs(h, ffn)
        if m == "post":
            h = self.n1(x + self.attn(x, mask, cache, freqs))
            return self.n2(h + self.mlp(h))
        a = self.attn(self.n1(x), mask, cache, freqs)
        if m == "par":
            return x + a + self.mlp(self.n2(x))
        h = x + a
        return h + self.mlp(self.n2(h))


@pytest.mark.parametrize("mode", ["post", "par", "seq", "hc"])
def test_attention_split_bit_equal(mode):
    mx.random.seed(3)
    h = _Holder.__new__(_Holder)
    nn.Module.__init__(h)
    h.layers = [_Blk(mode)]
    shape = (1, 1, 2, 8) if mode == "hc" else (1, 1, 8)
    xs = [mx.random.normal(shape) for _ in range(6)]
    SC.ENABLED = SC.ACTIVE = SC.ATTN = True
    SC.STATS.update(compiled=0, eager_warm=0, bad=0, stock=0, bad_reasons=[], attn_pre=0, attn_post=0, ast_refused=[])
    try:
        assert SC.install(h, log=lambda *_: None, force=True) == 1
        l = h.layers[0]
        outs = {}
        for active in (False, True):
            SC.ACTIVE = active
            c, o = _KV(), []
            for i, x in enumerate(xs):
                mask = mx.arange(i + 1).reshape(1, -1)               # grows every step: a passthrough, not a shape key
                o.append(l(x, mask, c))
            outs[active] = o
        for a, b in zip(outs[False], outs[True]):
            assert mx.array_equal(a, b).item()
        assert SC.STATS["bad"] == 0, SC.STATS["bad_reasons"]
        assert bool(SC.STATS["ast_refused"]) == (mode == "hc")    # several call sites resolve at install by the static config; the closure (hc) goes through the trace split
        assert SC.STATS["attn_pre"] >= 3 and SC.STATS["attn_post"] >= 3
        assert SC.STATS["eager_warm"] <= 4                           # the growing mask did not retrace every step
    finally:
        SC.ACTIVE = SC.ATTN = False


def test_host_offset_no_device_read():
    class C:
        offset = 7
    assert _host_offset(C()) == 7
    b = types.SimpleNamespace(offset=mx.array([5]), _idx=5)
    assert _host_offset(b) == 5
    assert _host_offset(None) == 0
