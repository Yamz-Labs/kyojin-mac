# SPDX-License-Identifier: Apache-2.0
"""Flag attn_fp16 (KYOJIN_ATTN_FP16, default off): the absorbed-MLA attention path of the skeleton in fp16 with fp32 accumulation.

Hy4 runs its hidden stream, q, the latents and the fixed-state KV buffer in fp32 (the loader's activation dtype), so every attention product is
fp32 and the buffer, the pooled-key reads and the fused attend kernel move 2x the bytes. The stored linears already stage x as fp16 and
accumulate in fp32 (castfree / shared-input kernels), so the fp32 results were a cast of an fp16 product. With the flag on:

  - the KV buffer of a new FixedKVCache is fp16 (the row is saturated to +-65504 before the cast: an fp16 stream keeps the saturating clamp);
  - the attention projections (q_b, kv_a, the index projections, the output gate) write fp16 straight from the fp32 accumulators (no cast launch);
  - q, latent, k_pe, the absorbs (k_b / v_b copies in fp16, built once outside the compiled layers), scores (fp32 accumulate), P @ latent and the
    pooled-key / index reads all run on fp16 data; softmax stays fp32 (precise); the fused attend kernel (flag mla_attend_fuse) reads the fp16 buffer.
  - the hidden stream (residual, norms, hyper-connection mix) stays fp32: it is a few KB per row (no bandwidth to win) and is where fp16 overflows.
The path is chosen by the buffer dtype (fp16 buffer = fp16 path), so one cache is never half and half. Generic: any absorbed-MLA skeleton model."""
from __future__ import annotations

import os

import mlx.core as mx

ON = os.environ.get("KYOJIN_ATTN_FP16", "0") not in ("0", "")      # flag attn_fp16 (flags.py rewrites it at runtime)
PARTS = int(os.environ.get("KYOJIN_ATTN_FP16_PARTS", "7"))         # flag attn_fp16_parts: 1 = KV buffer fp16, 2 = attention projections written fp16, 4 = fp16 k_b / v_b absorbs (q cast); 7 = all
F16MAX = 65504.0


def proj16(at) -> bool:
    """projections of this layer in fp16 (PARTS & 2, the fp16 absorb copies exist)"""
    return ON and bool(PARTS & 2) and "_kb16" in at.__dict__


def abs16(at) -> bool:
    """k_b / v_b absorbs in fp16 (PARTS & 4)"""
    return ON and bool(PARTS & 4) and "_kb16" in at.__dict__


def pf_kv32() -> bool:
    """Flag KYOJIN_HY4_PF_FP16_KV32 (with KYOJIN_HY4_PF_FP16): the KV rows of the cache are the fp32 products (kv_a / kpool projections stay fp32); only the q side, the scores and P @ V run in fp16
    (the whole cache is viewed as fp16 for the chunk's attention). Decode then reads KV with no fp16 rounding at all."""
    return os.environ.get("KYOJIN_HY4_PF_FP16_KV32", "0") not in ("0", "")


def pf_on() -> bool:
    """Flag KYOJIN_HY4_PF_FP16 (read per call, default ON since 19/11: the paired 256-token decode NLL delta +0.030 +- 0.039 sits inside the two null perturbations +0.052 / +0.078): the fp16 attention path for PROMPT CHUNKS only (L > 1). Decode (L == 1) and the KV cache stay in the decode dtype: the fp16 rows
    of a chunk are cast back to the hidden dtype before the cache update (exact: they are fp16 values) and the whole cache is viewed as fp16 for the attention of the chunk."""
    return os.environ.get("KYOJIN_HY4_PF_FP16", "1") not in ("0", "")


def sat(x: mx.array) -> mx.array:
    """-> fp16 with the saturating clamp (no inf from an fp32 value past the fp16 range)."""
    return mx.clip(x, -F16MAX, F16MAX).astype(mx.float16)


def kv_dtype(h_dtype):
    """dtype of a new fixed-state KV buffer."""
    return mx.float16 if ON and PARTS & 1 else h_dtype


def _f32(m) -> bool:
    d = getattr(m, "out_dtype", None)
    return d is not None and d == mx.float32


def _mods(m):
    yield m
    for s in getattr(m, "members", ()):                    # SplitLinear
        yield s
    lin = m.__dict__.get("_lin") if hasattr(m, "__dict__") else None      # GroupMember's own-rows linear (built on first use)
    if lin is not None:
        yield lin


def call(m, x: mx.array) -> mx.array:
    """m(x) with the output written as fp16 (a stored linear skips its fp32 cast; any other module's output is clamped and cast)."""
    saved = []
    for s in _mods(m):
        if _f32(s):
            saved.append(s)
            s.out_dtype = mx.float16
    try:
        y = m(x)
    finally:
        for s in saved:
            s.out_dtype = mx.float32
        lin = m.__dict__.get("_lin") if hasattr(m, "__dict__") else None
        if lin is not None and _f32(m):      # built while the dtype was switched
            lin.out_dtype = mx.float32
    return y if y.dtype == mx.float16 else sat(y)


def prepare(model) -> None:
    """Eager, host side (Model.__call__, next to the cache growth): fp16 copies of the k_b / v_b absorb weights of every MLA layer, once. They live outside
    the module's parameters so the compiled layers see two constants, not a cast per call."""
    for layer in model.layers:
        at = getattr(layer, "self_attn", None)
        if at is None or not hasattr(at, "k_b") or "_kb16" in at.__dict__:
            continue
        a, b = at.k_b.astype(mx.float16), at.v_b.astype(mx.float16)
        mx.eval(a, b)
        at.__dict__["_kb16"], at.__dict__["_vb16"] = a, b
