# SPDX-License-Identifier: Apache-2.0
"""ablate: probe flag (KYOJIN_ABLATE, flags.py `ablate`, default "" = off) that prices the EXPOSED cost of an op class of the DS4.1 decode step.

The flag is a comma list of classes. Each listed class is replaced, on the decode path (rows [n, 1, ...]: the 1-row plain step and the verify rows),
by the cheapest stand-in that keeps every shape and dtype the rest of the step reads (zeros, the input itself, fixed ids), so the step still runs end to end.
The outputs are garbage: this is a timing probe, never a quality lever. Run the same step with and without a class and subtract the round times: that is the
class's cost with the real overlap (a single-dispatch profile over-prices small kernels unevenly).

    moe_routed  routed experts (rot_gu .. final) -> zeros [rows, hidden] fp32                      patch._routed
    moe_shared  shared expert -> zeros                                                              pf_glue._impl_shared_mlp, patch._routed (sumoe rows)
    router      router logits + top-k -> fixed ids / weights (same shape and dtype)                 rows._router_sqrtsoftplus
    moe_glue    routed GEMVs kept; rot_gu / mid / final -> zero buffers of their output shape        moe.MoeGather.rot_gu / mid / final / final_par (+ sumoe stages)
    mhc         hc mixes + sinkhorn -> constants, pre -> stream mean, expand -> add                   rows._hc_block_mixes, fuse.mhc_pre / mhc_expand / mhc_mix_pre
    norm        RMSNorm modules -> identity                                                         nn.RMSNorm.__call__ (installed on demand)
    attn        attention core (kv write, compressor, indexer, attend) -> zeros, cache untouched    dsv4_core.core, fuse_attn.core
    idx         indexer only -> fixed first-k positions                                             dsv4_core.core (the idx_short form of the selection)
    ablit       decensor projection -> identity                                                     ablit.project

`mhc` and `norm` route the decode block through the stock chain (fuse_layer), because the fused stage paths (decstage, glue_fuse, suglue, the compiled
regions) fold those ops into their own kernels and have no stand-in; ops folded inside other kernels (the kv_chain / qkv_stage norm, the router GEMV of the
launchcut router) stay where the class does not name them. `ACTIVE` is a module attribute (flags.set writes it and bumps flags.GEN, so compiled regions
retrace); the sites read it per call, off costs one attribute test."""
from __future__ import annotations

import os
from collections import Counter

import mlx.core as mx

ENV = "KYOJIN_ABLATE"
CLASSES = ("moe_routed", "moe_shared", "router", "moe_glue", "mhc", "norm", "attn", "idx", "ablit")
_CHAINED = frozenset(("mhc", "norm"))                      # classes whose stand-ins live in the stock chain only: the fused block paths step aside
HITS: Counter = Counter()                                  # stand-in executions per class (python side: counts graph builds, not launches)
_const: dict = {}


def parse(v) -> frozenset:
    """'moe_routed, norm' -> frozenset. Unknown names raise ValueError (a typo must not silently measure nothing)."""
    names = [t.strip().lower() for t in str(v or "").replace(";", ",").split(",") if t.strip()]
    bad = [t for t in names if t not in CLASSES]
    if bad:
        raise ValueError(f"{ENV}: unknown class {bad}; known: {', '.join(CLASSES)}")
    return frozenset(names)


def normalize(v) -> str:
    return ",".join(c for c in CLASSES if c in parse(v))


ACTIVE: frozenset = parse(os.environ.get(ENV, ""))


def configure(v) -> frozenset:
    """flags.set hook: the new class set (also installs the on-demand wrappers). Returned value becomes `ACTIVE`."""
    a = parse(v)
    if "norm" in a:
        _install_norm()
    return a


def on(name: str) -> bool:
    return name in ACTIVE


def hit(name: str) -> None:
    HITS[name] += 1


def chain_only() -> bool:
    """True when the decode block must run the stock chain (a class whose stand-ins sit in the chain's ops is on)."""
    return not ACTIVE.isdisjoint(_CHAINED)


def decode_rows(x) -> bool:
    return x.ndim >= 3 and x.shape[1] == 1


def _cached(key, make):
    a = _const.get(key)
    if a is None:
        if len(_const) >= 256:
            _const.clear()
        a = _const[key] = make()                          # built from host data (never a lazy op): safe to keep across mx.compile traces, no eval needed
    return a


# --------------------------------------------------------------------------------------------------------------- router
def fixed_route(rows: int, k: int, n_experts: int, scale: float):
    """(ids int32 [rows, 1, k], weights fp32 [rows, 1, k]): row r takes experts r*k .. r*k+k-1 (mod n_experts), weight scale / k. Constants (no launch after the first call)."""
    import numpy as np

    def make():
        ids = (np.arange(rows)[:, None] * k + np.arange(k)[None, :]) % max(1, n_experts)
        return mx.array(ids.astype(np.int32)[:, None, :])
    ids = _cached(("rid", rows, k, n_experts), make)
    wts = _cached(("rwt", rows, k, float(scale)), lambda: mx.array(np.full((rows, 1, k), float(scale) / k, dtype=np.float32)))
    return ids, wts


# --------------------------------------------------------------------------------------------------------------- mhc
def mhc_coeffs(rows: int, hc: int = 4):
    """(pre [rows, 1, hc] = 1/hc, post [rows, 1, hc] = 1, comb [rows, 1, hc, hc] = identity) fp32 constants: the mean collapse and the plain-add expand."""
    import numpy as np
    pre = _cached(("pre", rows, hc), lambda: mx.array(np.full((rows, 1, hc), 1.0 / hc, dtype=np.float32)))
    post = _cached(("post", rows, hc), lambda: mx.array(np.ones((rows, 1, hc), dtype=np.float32)))
    comb = _cached(("comb", rows, hc), lambda: mx.array(np.broadcast_to(np.eye(hc, dtype=np.float32), (rows, 1, hc, hc)).copy()))
    return pre, post, comb


def mhc_pre(x):
    """Stream mean of x [B, 1, hc, D] -> [B, 1, D] (the fp16 range clamp the real collapse carries is kept)."""
    hit("mhc")
    y = x.astype(mx.float32).mean(axis=2).astype(x.dtype) if x.dtype != mx.float32 else x.mean(axis=2)
    return y


def mhc_expand(x, residual):
    """x [B, 1, D] + residual [B, 1, hc, D] -> [B, 1, hc, D], saturated at the fp16 range like the real expand."""
    hit("mhc")
    y = x[:, :, None, :] + residual
    return mx.clip(y, -65504, 65504) if y.dtype == mx.float16 else y


# --------------------------------------------------------------------------------------------------------------- moe glue
def zeros_like_shape(shape, dtype):
    """A concrete zero buffer (host built, cached per shape / dtype): a kernel input, so it is not a lazy broadcast the kernel call would first copy."""
    import numpy as np
    npd = {mx.float16: np.float16, mx.float32: np.float32, mx.int32: np.int32}.get(dtype)
    if npd is None:                                                          # bf16 has no host dtype: a lazy zeros (consumed by an elementwise op, no launch of its own)
        return mx.zeros(tuple(shape), dtype=dtype)
    return _cached(("z", tuple(shape), dtype), lambda: mx.array(np.zeros(tuple(shape), dtype=npd)))


# --------------------------------------------------------------------------------------------------------------- norm
_norm_installed = False


def _install_norm() -> None:
    """Wrap nn.RMSNorm.__call__ once (process wide; idle unless 'norm' is active)."""
    global _norm_installed
    if _norm_installed:
        return
    import mlx.nn as nn
    orig = nn.RMSNorm.__call__

    def __call__(self, x):
        if "norm" in ACTIVE and x.ndim == 3 and x.shape[1] == 1:
            hit("norm")
            return x
        return orig(self, x)

    nn.RMSNorm.__call__ = __call__
    _norm_installed = True


if "norm" in ACTIVE:
    _install_norm()
