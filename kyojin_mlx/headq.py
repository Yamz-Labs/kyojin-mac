# SPDX-License-Identifier: Apache-2.0
"""Flag head_q (KYOJIN_HEAD_Q, default 0): the trunk's own output head of a 1 .. 16 row pass (plain decode, verify rows) through an 8- or 4-bit affine (group 64)
copy of the dense head matrix, instead of the dense float32 matmul over the stored matrix (Hy4: 129280 x 3072 fp32 = 1.6 GB, 1.3 ms of a 22.9 ms step at the
bandwidth roof; Q8 reads a quarter). Prompt chunks (more rows) keep the dense head. The packed copy is built once per (weight, bits) and shared with the MTP
draft's output projection (hy4_mtp.MtpDraft.set_head_q), so there is one low-bit copy. Logits differ at the 8-bit rounding: gate with the model's noise band."""
from __future__ import annotations

import os

import mlx.core as mx

BITS = int(os.environ.get("KYOJIN_HEAD_Q", "0") or 0)           # flag head_q: 0 off, 8 or 4
MAX_ROWS = 16
GROUP = 64
_PACKS: dict = {}                                               # (id(weight), bits) -> (wq, scales, biases, bits, weight): the weight reference keeps the id valid


def packed(w, bits: int):
    """(wq, scales, biases, bits) of the dense head matrix `w` [V, D], or None when it cannot be packed (not a dense 2-D float matrix, D % 64)."""
    if bits not in (4, 8) or not isinstance(w, mx.array) or w.ndim != 2 or w.dtype not in (mx.float32, mx.bfloat16, mx.float16) or w.shape[-1] % GROUP:
        return None
    key = (id(w), bits)
    hit = _PACKS.get(key)
    if hit is None:
        wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=bits)
        mx.eval(wq, sc, bi)
        hit = _PACKS[key] = (wq, sc, bi, bits, w)
    return hit[:4]


def project(lm, h):
    """Output head `lm` (nn.Linear) on h [B, L, D] through the packed copy, or None when the flag is off / the shape is not eligible (the caller runs the dense head)."""
    bits = BITS
    if not bits or h.shape[-2] > MAX_ROWS:
        return None
    pk = packed(getattr(lm, "weight", None), bits)
    if pk is None:
        return None
    wq, sc, bi, bits = pk
    return mx.quantized_matmul(h, wq, sc, bi, transpose=True, group_size=GROUP, bits=bits)
