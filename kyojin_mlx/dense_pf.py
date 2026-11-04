# SPDX-License-Identifier: Apache-2.0
"""Folded dense prefill weights for the EXL3 linears (NAX dense prefill, large-memory devices).

An EXL3 linear computes y = svh * H(Wr @ H(suh * x)) with H = block-diagonal Walsh-Hadamard over 128 (scaled by 1/sqrt(128)) and Wr the trellis-decoded inner weight. Both Hadamards
and both sign vectors are linear, so they fold into one dense matrix
    W_eff = diag(svh) Hn Wr Hk diag(suh)        y = x @ W_eff^T
and the prefill of a long prompt is then ONE fp16 matmul per linear: no input rotation launch, no finalize launch, no per-call decode. The fold is built once per linear in
fp32 (rounded to fp16 at the end, so the outputs differ from the rotated path at fp16-matmul level, ~1e-3 relative) and costs 2 bytes per weight: it is built only when the
total fits KYOJIN_PF_DENSE_GB (GiB; default 10 % of the device memory, 0 = off) so the 16 GB Air keeps the compressed path and the M5 Ultra (hundreds of GiB) takes it.
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import kernels as KN

MIN_ROWS = int(os.environ.get("KYOJIN_PF_DENSE_MIN_ROWS", "128"))     # below: the compressed paths (multi-user rows keep their bits)
_H: mx.array | None = None


def cap_bytes() -> int:
    v = os.environ.get("KYOJIN_PF_DENSE_GB")                # GiB; 0 = off; default 10 % of the device memory (the Air never folds, an Ultra does)
    if v is not None:
        return int(float(v) * 2 ** 30)
    try:
        return int(0.10 * int(mx.device_info()["memory_size"]))
    except Exception:
        return 0


def _had128() -> mx.array:
    global _H
    if _H is None:
        h = mx.array([[1.0]])
        while h.shape[0] < 128:
            h = mx.concatenate([mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0)
        _H = h / (128 ** 0.5)
    return _H


def fold(lin, rows_block: int = 1024, max_rows: int | None = None) -> mx.array:
    """W_eff [n_out, k_in] fp16 of an Exl3Linear (see the module docstring), built in fp32 slabs of `rows_block` output rows (max_rows: only the first ones, a multiple of 128)."""
    wr = KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb)                 # (n, k) fp16, natural k order
    n, k = wr.shape
    h = _had128()
    suh = lin._suh.astype(mx.float32)
    svh = lin._svh.astype(mx.float32)
    out = []
    n = n if max_rows is None else min(n, max_rows)
    for r0 in range(0, n, rows_block):
        w = wr[r0:min(n, r0 + rows_block)].astype(mx.float32)                               # (b, k)
        b = w.shape[0]
        w = (w.reshape(b, k // 128, 128) @ h).reshape(b, k) * suh                    # Wr Hk diag(suh)   (H symmetric)
        w = (h @ w.reshape(b // 128, 128, k)).reshape(b, k) * svh[r0:r0 + b, None]  # diag(svh) Hn (...)
        out.append(w.astype(mx.float16))
        mx.eval(out[-1])
    return mx.concatenate(out, 0)


def need_bytes(model) -> int:
    from . import wcache as WC
    return WC.need_bytes(model)


def install(model, force: bool = False) -> dict:
    """Fold every dense Exl3Linear (not the LM head) when the fp16 total fits the cap; returns {on, need, cap}. Calls with rows >= MIN_ROWS then run one fp16 matmul."""
    from . import wcache as WC
    need, cap = need_bytes(model), cap_bytes()
    info = {"on": False, "need": need, "cap": cap}
    if not force and (cap <= 0 or need > cap):
        return info
    for m in WC._linears(model):
        m._weff = fold(m)
    info["on"] = True
    return info
