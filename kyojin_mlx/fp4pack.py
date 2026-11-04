# SPDX-License-Identifier: Apache-2.0
"""Lossless 4-bit storage of the DS4.1 compressed KV pool and index keys.

The vendor model fake-quantizes every pooled row (`_fake_quant_fp4`: blocks of 16 with an e4m3 scale for the KV pool, blocks of 32 with a power-of-two scale
for the index keys) BEFORE it enters the cache, so the cache holds only values `sign * LUT[code] * scale`. Storing the 3-bit code + sign and the one-byte scale
instead of the fp16 value is exact (KL 0.0 by construction) and takes 4.5 bits (pool) / 4.25 bits (index) per value instead of 16: the pool of a 1M-token prompt
shrinks 3.5x. This is the DS4.1 form of "quantized KV by default": TurboQuant on top would round already-rounded values a second time.

pack(x, block, e4m3_scale)  -> (codes uint8 [..., D // 2], scales uint8 [..., D // block])   same rounding as the vendor `_fake_quant_fp4`
unpack(codes, scales, block, e4m3_scale, dtype) -> [..., D] bit-identical to `_fake_quant_fp4(x)` (tests/test_fp4pack.py)
"""
from __future__ import annotations

import mlx.core as mx

_LUT = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=mx.float32)
_THR_GT = (0.25, 1.25, 2.5, 5.0)
_THR_GE = (0.75, 1.75, 3.5)


def _code(mag):
    idx = mx.zeros(mag.shape, dtype=mx.int32)
    for t in _THR_GT:
        idx = idx + (mag > t).astype(mx.int32)
    for t in _THR_GE:
        idx = idx + (mag >= t).astype(mx.int32)
    return idx


def pack(x, block: int, e4m3_scale: bool = False):
    shape = x.shape
    assert shape[-1] % block == 0 and shape[-1] % 2 == 0
    blocks = x.astype(mx.float32).reshape(*shape[:-1], -1, block)
    floor = 6.0 * 2.0 ** -9 if e4m3_scale else 6.0 * 2.0 ** -126
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=-1, keepdims=True), floor)
    if e4m3_scale:
        s8 = mx.to_fp8(amax / 6.0)                                      # uint8 e4m3 bits
        scale = mx.from_fp8(s8, mx.float32)
        sbyte = s8.view(mx.uint8)
    else:
        e = mx.ceil(mx.log2(amax / 6.0))
        scale = mx.power(2.0, e)
        sbyte = (e + 127.0).astype(mx.uint8)                            # exponent byte (ue8m0), 2^-126 floor -> 1
    v = mx.clip(blocks / scale, -6.0, 6.0)
    code = _code(mx.abs(v)).astype(mx.uint8) | ((v < 0).astype(mx.uint8) << 3)   # sign bit set for negatives (a -0 rounds to +0, like the vendor sign * 0)
    code = code.reshape(*shape[:-1], -1, 2)
    packed = code[..., 0] | (code[..., 1] << 4)
    return packed.reshape(*shape[:-1], shape[-1] // 2), sbyte.reshape(*shape[:-1], shape[-1] // block)


def unpack(codes, scales, block: int, e4m3_scale: bool = False, dtype=mx.float16):
    d = codes.shape[-1] * 2
    lo = codes & 0xF
    hi = codes >> 4
    c = mx.stack([lo, hi], axis=-1).reshape(*codes.shape[:-1], d)
    mag = _LUT[(c & 7).astype(mx.int32)]
    val = mx.where((c & 8) > 0, -mag, mag)
    if e4m3_scale:
        sc = mx.from_fp8(scales, mx.float32)
    else:
        sc = mx.power(2.0, scales.astype(mx.float32) - 127.0)
    out = val.reshape(*codes.shape[:-1], -1, block) * sc[..., None]
    return out.reshape(*codes.shape[:-1], d).astype(dtype)
