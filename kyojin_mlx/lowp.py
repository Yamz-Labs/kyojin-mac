# SPDX-License-Identifier: Apache-2.0
"""Narrow copies of small fp32 weights that lose nothing (lever router_w16, KYOJIN_ROUTER_W16).

Weights such as the router gate are held in fp32 because the model upcasts them at load (the checkpoint stores bf16), while every kernel that reads them
upcasts each element to fp32 before the first multiply anyway. When every value is exactly representable in bf16 (or fp16), a narrow copy read through the
same kernel gives the same fp32 operands in the same order, so the product is bit-identical and half the bytes cross the memory bus. The check is
a value test (round trip equal), not a promise about the file, so a weight that really needs fp32 keeps its fp32 read.
"""
from __future__ import annotations

import mlx.core as mx

_cache: dict = {}


def exact_lowp(w: mx.array) -> mx.array | None:
    """A bf16 (else fp16) array with the same values as the fp32 `w`, or None when neither is exact (or the check cannot run). One check per weight object;
    the weight is kept alive so its id is not reused. Needs an eager call (it evaluates); inside a compile trace it returns None."""
    if w.dtype != mx.float32:
        return None
    hit = _cache.get(id(w))
    if hit is not None:
        return hit[1]
    got = None
    try:
        for dt in (mx.bfloat16, mx.float16):
            c = w.astype(dt)
            ok = mx.array_equal(c.astype(mx.float32), w)
            mx.eval(c, ok)
            if bool(ok.item()):
                got = c
                break
    except Exception:                                                             # evaluating inside a function transform: keep fp32, retry never
        got = None
    _cache[id(w)] = (w, got)
    return got


def reset() -> None:
    _cache.clear()
