# SPDX-License-Identifier: Apache-2.0
"""V1-VERIFY: cheaper speculative verify rounds (engine level, flags KYOJIN_VERIFY_* / KYOJIN_DRAFT_HEAD_*, all default off, read per call, bit-identical outputs).

Where a d1 round loses to a plain step (G2h + V1 dump, GLM-5.3, 8K ctx): ~930 of the ~965 extra dispatches of the round are not model kernels but glue of the speculative
cache transaction (copies, int32 index arithmetic). Each costs a python op, a Metal encode (~4 us of host CPU) and ~3-20 us of GPU, and the pass is host-bound.

POOLSEL: PoolingCache.commit_speculation (the 11 DSA layers, 2 caches each: indexer keys + gate remainder buffer) selects the remainder window with
`_select_pooling_remainder`: ~22 tiny mx ops (arange, mod, clip, take_along_axis, where ...) although every argument is a python int list. When all rows share the
same (remainder, start, valid) -- always true for a PoolingCache, which requires uniform commit lengths -- the selection is a static slice of the initial buffer, a static
slice of the verifier input and zeros: one concatenate + one pad. Pure data movement, so the result is bit-identical to the original.

HEAD16: the MTP draft head receives the target's hidden state in float32 and therefore runs entirely in float32 (every f16 weight is cast to f32 per call:
13 casts of up to 200 MB = 0.8 ms, plus float32 GEMVs on the up-cast weights). With the flag the normed hidden is rounded to the model dtype before eh_proj, so the
head runs in f16 like the reference. Draft only: the verify pass (and so every emitted token) is untouched, only the proposals can differ in the last bit.
"""
from __future__ import annotations

import os

import mlx.core as mx

POOLSEL = os.environ.get("KYOJIN_VERIFY_POOLSEL", "0") not in ("0", "")
HEAD16 = os.environ.get("KYOJIN_DRAFT_HEAD_F16", "0") not in ("0", "")
STATS = {"fast": 0, "orig": 0}
_ORIG = None


def _fast(initial, values, remainders, starts, valid_lengths, ratio):
    """Static-slice form of mlx_vlm.models.cache._select_pooling_remainder for uniform rows; None when the shortcut does not apply."""
    if not (len(set(remainders)) == 1 and len(set(starts)) == 1 and len(set(valid_lengths)) == 1):
        return None
    r, s, v = int(remainders[0]), int(starts[0]), int(valid_lengths[0])
    batch, lv, d = values.shape
    if initial is not None and (initial.dtype != values.dtype or tuple(initial.shape) != (batch, ratio, d)):
        return None
    if r < 0 or s < 0 or v < 0 or r >= ratio or s + v > lv:
        return None
    total = r + v
    nr = total % ratio                                   # slots 0 .. nr-1 stay filled, the rest is zero
    base = total - nr
    n_init = min(nr, max(0, r - base))                   # slots whose logical position is still inside the old remainder (only when no window completed)
    parts = []
    if n_init > 0:
        parts.append(initial[:, base:base + n_init] if initial is not None else mx.zeros((batch, n_init, d), dtype=values.dtype))   # no buffer yet = zeros, as the original
    if nr > n_init:
        a = s + base + n_init - r
        if a < 0 or a + (nr - n_init) > lv:
            return None
        parts.append(values[:, a:a + nr - n_init])
    if not parts:
        return mx.zeros((batch, ratio, d), dtype=values.dtype)
    sel = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    if nr == ratio:
        return sel
    return mx.pad(sel, [(0, 0), (0, ratio - nr), (0, 0)])


def _select(initial, values, remainders, starts, valid_lengths, ratio):
    if POOLSEL:
        out = _fast(initial, values, remainders, starts, valid_lengths, ratio)
        if out is not None:
            STATS["fast"] += 1
            return out
    STATS["orig"] += 1
    return _ORIG(initial, values, remainders, starts, valid_lengths, ratio)


def install() -> int:
    """Patch mlx_vlm's `_select_pooling_remainder` once per process (PoolingCache.commit_speculation looks the name up in the module at call time)."""
    global _ORIG
    try:
        import mlx_vlm.models.cache as C
    except Exception:
        return 0
    if getattr(C, "_select_pooling_remainder_v1", False):
        return 0
    _ORIG = C._select_pooling_remainder
    C._select_pooling_remainder = _select
    C._select_pooling_remainder_v1 = True
    return 1
