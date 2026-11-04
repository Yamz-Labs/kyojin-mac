# SPDX-License-Identifier: Apache-2.0
"""Prefill GDN recurrence through mx.fast.gated_delta_update (chunked along time, NAX tiles on M5) instead of the step-wise mlx_lm Metal kernel.

mlx_lm layers (qwen3_5, qwen3_next) call `gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask, ...)`. `install()` wraps that name in those modules:
rows T >= min_t, no mask, Dk == Dv == 128, GPU, fp32 state -> mx.fast.gated_delta_update (same state layout [B, Hv, Dv, Dk] fp32, same gate maths, so the state
is carried across calls, sessions and saves unchanged). Everything else (decode T=1, masks, other shapes) goes to the stock function untouched.

Choice (off by default = stock, bit-identical):
  KYOJIN_GDN_FAST=0|1|<min_t>   env wins (1 = min_t 64)
  tuning table entry "gdn": {"min_t": N}   from <pack>/tuning/<device>.json (tuning.py), N = 0 or absent -> off
The chunked kernel sums in another order than the step loop: y / state differ at bf16 rounding level (checked by tests/test_gdn_fast.py and tools/ab_gdn.py)."""
from __future__ import annotations

import os

import mlx.core as mx

DEFAULT_MIN_T = 64
_stock: dict = {}


def min_t() -> int:
    e = os.environ.get("KYOJIN_GDN_FAST")
    if e is not None:
        v = int(e)
        return DEFAULT_MIN_T if v == 1 else v
    from . import tuning
    return int(tuning.TABLE.get("gdn", {}).get("min_t", 0))


def _wrap(stock):
    def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True, lower_bound=None, allow_neg_eigval=False):
        T = q.shape[1]
        m = min_t()
        if (m > 0 and T >= m and mask is None and use_kernel and lower_bound is None and not allow_neg_eigval and mx.default_device() == mx.gpu
                and q.shape[-1] == 128 and v.shape[-1] == 128 and v.shape[2] % q.shape[2] == 0
                and (state is None or state.dtype == mx.float32)):
            from mlx_lm.models.gated_delta import compute_g
            y, st = mx.fast.gated_delta_update(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), state)
            return y.astype(q.dtype), st
        return stock(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel, lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
    gated_delta_update._kyojin_fast = True
    return gated_delta_update


def install() -> None:
    import importlib
    for name in ("qwen3_5", "qwen3_next"):
        try:
            mod = importlib.import_module(f"mlx_lm.models.{name}")
        except ImportError:
            continue
        cur = getattr(mod, "gated_delta_update", None)
        if cur is None or getattr(cur, "_kyojin_fast", False):
            continue
        _stock[name] = cur
        mod.gated_delta_update = _wrap(cur)
