# SPDX-License-Identifier: Apache-2.0
"""Flag KYOJIN_FOREIGN_GDN_FAST (default off): the fused gated-delta decode kernel (gdn_fuse) on a foreign pack's GatedDeltaNet-like linear-attention modules.

The kernel (conv + silu + q / k norm + g / beta + recurrence + gated output norm in ONE launch) is what the q35 backend runs for Qwen3.5 / 3.6. A module qualifies by
structure, never by name: attributes in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / conv1d / A_log / dt_bias / norm / out_proj, head dims 128, conv kernel 4, and a
probe that its `_normalize_qk` and its norm gate are one of the two arithmetics the kernel mirrors:
  stock : q / k = rms_norm(x, eps / 128) scaled, gate silu                           (mlx-lm / mlx-vlm Qwen3.5 / 3.6)
  qw4   : q / k = x * rsqrt(sum(x^2) + 1e-6), q * 128^-0.5, gate sigmoid in fp32       (the qwen4_exp class: L2 norm with the epsilon outside, sigmoid output gate)
Anything else (an unknown norm, a mask, a speculating cache, padding, > MAX_ROWS rows, a batch above MAX_B) runs the module's own __call__. Rows B > 1 launch once per row (rows are independent).
The class __call__ is replaced once; `ACTIVE` switches the fast path at run time (a profile cell A/Bs in one process)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import verify_capture as VC

ACTIVE = os.environ.get("KYOJIN_FOREIGN_GDN_FAST", "0") not in ("0", "")
MAX_ROWS = 8
BATCH = os.environ.get("KYOJIN_GDN_FAST_BATCH", "0") not in ("0", "")      # B > 1: ONE fused launch for all rows (off: one launch per row)
MAX_B = int(os.environ.get("KYOJIN_FOREIGN_GDN_FAST_MAXB", "8"))
STATS = {"fast": 0, "stock": 0}
_ATTRS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "A_log", "dt_bias", "norm", "out_proj", "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim",
          "conv_kernel_size", "conv_dim", "_normalize_qk", "_project_gates")


def _probe_mode(m):
    """'stock' / 'qw4' when the module's q / k norm and gate are the arithmetic of one kernel mode (bit-equal on a probe), else None."""
    import mlx.nn as nn
    dt = mx.bfloat16
    mx.random.seed(7)
    q = (mx.random.normal((1, 2, 2, 128)) * 3).astype(dt)
    k = (mx.random.normal((1, 2, 2, 128)) * 3).astype(dt)
    qo, ko = m._normalize_qk(q, k)
    sc = 128 ** -0.5
    ref_stock = ((sc ** 2) * mx.fast.rms_norm(q, None, 1e-6 / 128), sc * mx.fast.rms_norm(k, None, 1e-6 / 128))
    ref_qw4 = ((q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)) * sc, k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6))
    mode = None
    for name, ref in (("stock", ref_stock), ("qw4", ref_qw4)):
        if mx.array_equal(qo, ref[0]).item() and mx.array_equal(ko, ref[1]).item():
            mode = name
            break
    if mode is None:
        return None
    x = (mx.random.normal((1, 3, 128)) * 2).astype(dt).reshape(1, 3, 1, 128)
    zz = (mx.random.normal((1, 3, 1, 128)) * 2).astype(dt)
    y = m.norm(x, zz)
    w = m.norm.weight
    n = mx.fast.rms_norm(x, w, m.norm.eps)
    if mode == "qw4":
        ref = (n.astype(mx.float32) * mx.sigmoid(zz.astype(mx.float32))).astype(dt)
    else:
        ref = (nn.silu(zz.astype(mx.float32)) * n.astype(mx.float32)).astype(dt)
    if y.shape != ref.shape or mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))).item() > 0.02 * mx.max(mx.abs(ref.astype(mx.float32))).item():
        return None
    return mode


def qualifies(m):
    if not all(hasattr(m, a) for a in _ATTRS):
        return None
    try:
        if not (m.head_k_dim == 128 and m.head_v_dim == 128 and m.conv_kernel_size == 4 and m.num_v_heads % m.num_k_heads == 0
                and tuple(m.conv1d.weight.shape) == (m.conv_dim, 4, 1) and tuple(m.dt_bias.shape) == (m.num_v_heads,) and tuple(m.A_log.shape) == (m.num_v_heads,)
                and tuple(m.norm.weight.shape) == (128,) and m.conv_dim == (2 * m.num_k_heads + m.num_v_heads) * 128):
            return None
        return _probe_mode(m)
    except Exception:
        return None


def _wrap(stock):
    from . import gdn_fuse

    def call(self, inputs, mask=None, cache=None):
        mode = self.__dict__.get("_kfg_mode")
        B, S, _ = inputs.shape
        if (ACTIVE and mode is not None and mask is None and cache is not None and 1 <= S <= MAX_ROWS and B <= MAX_B and getattr(cache, "lengths", None) is None and getattr(cache, "left_padding", None) is None
                and not getattr(cache, "is_speculating", False) and gdn_fuse.ENABLED and mx.default_device() == mx.gpu and inputs.dtype in (mx.float16, mx.bfloat16)
                and self.dt_bias.dtype in (inputs.dtype, mx.float32) and (cache[1] is None or (cache[1].dtype == mx.float32 and cache[1].shape[0] == B))
                and (cache[0] is None or cache[0].shape[0] == B) and not self.training):
            STATS["fast"] += 1
            Hv, Hk, C = self.num_v_heads, self.num_k_heads, self.conv_dim
            qkv, z = self.in_proj_qkv(inputs), self.in_proj_z(inputs)
            b, a = self._project_gates(inputs)
            z = z.reshape(B, S, Hv, 128)
            cs = cache[0] if cache[0] is not None else mx.zeros((B, 3, C), dtype=inputs.dtype)
            st = cache[1] if cache[1] is not None else mx.zeros((B, Hv, 128, 128), dtype=mx.float32)
            sk = VC.SINK
            if sk is not None and S == sk.S and B == 1 and S > 1:      # speculative verify: the state after every row, for a rollback without a replay forward
                o, cn, s2, ss = gdn_fuse.fused_step(qkv, z, a, b, cs, st, self.conv1d.weight, self.A_log, self.dt_bias, self.norm.weight, self.norm.eps, Hk, Hv, qw4=mode == "qw4", steps=True)
                ext = mx.concatenate([cs, qkv], axis=1)
                cache[0], cache[1] = cn, s2
                sk.put(cache, 0, [ext[:, k:k + 3] for k in range(1, S + 1)])
                sk.put(cache, 1, [ss[k:k + 1] for k in range(S - 1)] + [s2])
                if hasattr(cache, "advance"):
                    cache.advance(S)
                return self.out_proj(o)
            if BATCH and B > 1:
                o, cn, s2 = gdn_fuse.fused_step(qkv, z, a, b, cs, st, self.conv1d.weight, self.A_log, self.dt_bias, self.norm.weight, self.norm.eps, Hk, Hv, qw4=mode == "qw4")[:3]
                cache[0], cache[1] = cn, s2
                if hasattr(cache, "advance"):
                    cache.advance(S)
                return self.out_proj(o)
            outs, cns, sts = [], [], []
            for r in range(B):
                sl = slice(r, r + 1)
                o, cn, s2 = gdn_fuse.fused_step(qkv[sl], z[sl], a[sl], b[sl], cs[sl], st[sl], self.conv1d.weight, self.A_log, self.dt_bias, self.norm.weight, self.norm.eps,
                                                Hk, Hv, qw4=mode == "qw4")[:3]
                outs.append(o); cns.append(cn); sts.append(s2)
            cat = (lambda xs: xs[0] if len(xs) == 1 else mx.concatenate(xs, axis=0))
            cache[0], cache[1] = cat(cns), cat(sts)
            if hasattr(cache, "advance"):
                cache.advance(S)
            return self.out_proj(cat(outs))
        STATS["stock"] += 1
        if ACTIVE:                                                   # first failing guard, counted (diagnostic; the stock call below is unchanged)
            why = ("mode" if mode is None else "mask" if mask is not None else "nocache" if cache is None else f"S{S}" if not 1 <= S <= MAX_ROWS
                   else f"B{B}" if B > MAX_B else "lengths" if getattr(cache, "lengths", None) is not None else "left_padding" if getattr(cache, "left_padding", None) is not None
                   else "speculating" if getattr(cache, "is_speculating", False) else "disabled" if not gdn_fuse.ENABLED else "device" if mx.default_device() != mx.gpu
                   else f"in_{inputs.dtype}" if inputs.dtype not in (mx.float16, mx.bfloat16) else f"dtb_{self.dt_bias.dtype}" if self.dt_bias.dtype not in (inputs.dtype, mx.float32)
                   else f"state_{cache[1].dtype}_{cache[1].shape[0]}" if cache[1] is not None and not (cache[1].dtype == mx.float32 and cache[1].shape[0] == B)
                   else "conv_B" if cache[0] is not None and cache[0].shape[0] != B else "training" if self.training else "other")
            W = STATS.setdefault("why", {})
            W[why] = W.get(why, 0) + 1
        return stock(self, inputs, mask, cache)
    call._kyojin_gdn_fast = True
    return call


def install(model, log=print) -> int:
    """Arm every qualifying module of `model` (mode stored on the instance) and replace its class __call__ once. Returns the number armed (0 with the flag unset: nothing touched)."""
    if not ACTIVE:
        return 0
    n, modes = 0, {}
    for _, m in model.named_modules():
        mode = qualifies(m)
        if mode is None:
            continue
        m.__dict__["_kfg_mode"] = mode
        cls = type(m)
        if not getattr(cls.__call__, "_kyojin_gdn_fast", False):
            cls.__call__ = _wrap(cls.__call__)
        n += 1
        modes[mode] = modes.get(mode, 0) + 1
    log(f"[foreign] gdn_fast: {n} gated-delta modules on the fused decode kernel {modes}")
    return n
