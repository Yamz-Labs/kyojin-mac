# SPDX-License-Identifier: Apache-2.0
"""Prompt-scoped fp16 weight cache for the dense EXL3 linears (prefill).

A chunked prompt runs every dense linear once per chunk, and each call decodes the trellis to fp16 first (`kernels.decode_wr`). The decoded weight does not depend on
the rows, so inside `prefill_scope` the first chunk keeps it and later chunks reuse it: the same decoded values and the same matmul (bit-identical outputs); only the decode
launches of chunks 2.. disappear. The scope frees the cache when the prefill ends.

Memory rule: the cache needs `need_bytes(model)` (every Exl3Linear except the LM head, fp16). The scope turns the cache off when that exceeds the cap
KYOJIN_PF_WCACHE_GB (GiB; 0 = off; default 10 % of the device memory), so a box with little memory never grows."""
from __future__ import annotations

import contextlib
import os

import mlx.core as mx

from . import kernels as KN

_state: dict | None = None            # active scope: {"w": {id(lin): wr}, "bytes": int}
last: dict = {}                       # stats of the last scope (need, cap, on, hits, misses, bytes)


def cap_bytes() -> int:
    v = os.environ.get("KYOJIN_PF_WCACHE_GB")
    if v is not None:
        return int(float(v) * 2 ** 30)
    try:
        mem = int(mx.device_info()["memory_size"])
    except Exception:
        mem = 0
    return int(0.10 * mem)


def _linears(model) -> list:
    from .exl3_linear import Exl3Linear
    lm = getattr(model, "language_model", model)
    head = getattr(lm, "head", None)
    skip = {id(m) for m in head.modules()} if head is not None and hasattr(head, "modules") else set()
    return [m for m in model.modules() if type(m) is Exl3Linear and id(m) not in skip]


def need_bytes(model) -> int:
    return sum(m.in_features * m.out_features * 2 for m in _linears(model))


@contextlib.contextmanager
def prefill_scope(model, rows_per_chunk: int, n_chunks: int):
    """Cache the decoded dense weights for the chunks run inside the `with`. No-op for a single chunk (nothing to reuse), chunks the fused decode path takes
    (rows <= DENSE_FUSED_MAX: no fp16 weight exists), or a need above the cap."""
    global _state
    from . import exl3_linear as EL
    need, cap = need_bytes(model), cap_bytes()
    on = _state is None and n_chunks > 1 and cap > 0 and need <= cap and rows_per_chunk > EL.DENSE_FUSED_MAX
    last.clear()
    last.update(need=need, cap=cap, on=on, hits=0, misses=0, bytes=0)
    if not on:
        yield
        return
    _state = {"w": {}, "bytes": 0}
    try:
        yield
    finally:
        last["bytes"] = _state["bytes"]
        _state = None


def decode_wr(lin) -> mx.array:
    """`KN.decode_wr` of an Exl3Linear, served from the prompt scope when one is active."""
    st = _state
    if st is None:
        return KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb)
    wr = st["w"].get(id(lin))
    if wr is None:
        wr = st["w"][id(lin)] = KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb)
        st["bytes"] += wr.nbytes
        last["misses"] += 1
    else:
        last["hits"] += 1
    return wr
