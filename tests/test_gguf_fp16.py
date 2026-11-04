# SPDX-License-Identifier: Apache-2.0
"""GGUF F32 norm weights load in the stream dtype; a KV checkpoint with fp32 overlay rows still serves an fp16 stream."""
import mlx.core as mx
import pytest

from kyojin_mlx import foreign_linear as FL
from kyojin_mlx import tq as TQ


def test_f32_on_stream_names():
    w = mx.ones((8,), mx.float32)
    assert FL._f32_on_stream("model.layers.0.self_attn.q_norm.weight", w)
    assert not FL._f32_on_stream("model.layers.0.linear_attn.norm.weight", w)        # gdn_norm's fused kernel takes it as fp32
    assert not FL._f32_on_stream("x.weight", mx.ones((8, 8), mx.float32))
    assert not FL._f32_on_stream("x.weight", mx.ones((8,), mx.float16))


@pytest.mark.parametrize("rows", [1, 4])
def test_fp32_overlay_decodes_fp16(rows):
    if mx.default_device() != mx.gpu:
        pytest.skip("Metal kernels")
    mx.random.seed(3)
    k = (mx.random.normal((1, 4, 700, 256)) * 0.5).astype(mx.float16)
    v = (mx.random.normal((1, 4, 700, 256)) * 0.5).astype(mx.float16)
    q = (mx.random.normal((1, 24, rows, 256)) * 0.5).astype(mx.float16)
    a, b = TQ.TQKVCache(), TQ.TQKVCache()
    a.update_and_fetch(k, v), b.update_and_fetch(k, v)
    mx.eval(a.kq, a.vq, a.rk, a.rv, a.sk, a.sv, b.kq, b.vq, b.rk, b.rv, b.sk, b.sv)
    for n in ("sk", "sv", "rk", "rv"):
        setattr(b, n, getattr(b, n).astype(mx.float32))
    ref, out = a.attend(q, 256 ** -0.5), b.attend(q, 256 ** -0.5)
    mx.eval(ref, out)
    assert out.dtype == mx.float16 and b.rk.dtype == mx.float16 and b.sk.dtype == mx.float16
    assert mx.array_equal(ref, out).item()
