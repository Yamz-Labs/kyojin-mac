# SPDX-License-Identifier: Apache-2.0
"""Test-only slope probe (env KYOJIN_TEST_DEP_CHAIN=K): after every decoder layer, K chained trivial kernels on the hidden stream.

Each kernel copies its input (value-preserving) and feeds the next one and the following layer, so the GPU must run them in order and
barrier between them: K added dependent dispatches per layer. The tok/s slope against K prices one dependent dispatch inside the
real model (dispatches per token = K x layers)."""
from __future__ import annotations

import os

import mlx.core as mx

K = int(os.environ.get("KYOJIN_TEST_DEP_CHAIN", "0") or 0)

_kern = None


def _copy(x):
    global _kern
    if _kern is None:
        _kern = mx.fast.metal_kernel(name="dep_chain_copy", input_names=["x"], output_names=["y"],
                                     source="uint i = thread_position_in_grid.x; y[i] = x[i];")
    n = x.size
    return _kern(inputs=[x], grid=(n, 1, 1), threadgroup=(min(256, n), 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def chain(h, k: int = K):
    for _ in range(k):
        h = _copy(h)
    return h


def install(layers) -> bool:
    """Wrap the decoder layer class once; a no-op when K is 0."""
    if K <= 0 or not layers:
        return False
    cls = type(layers[0])
    if getattr(cls, "_dep_chain", False):
        return True
    inner = cls.__call__

    def call(self, *a, **kw):
        return chain(inner(self, *a, **kw))

    cls.__call__ = call
    cls._dep_chain = True
    return True
