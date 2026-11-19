# SPDX-License-Identifier: Apache-2.0
"""Block N2 of the decoder skeleton: hyper-connection streams (hc copies of the residual). Fires on the keys `hyper_connection.{count, epsilon,
magnitude[, sinkhorn_iterations]}` and the tensors `hc_{attn,ffn}_{fn,base,scale}`, `output_hc_{fn,base,scale}`; no architecture name appears here.

Per sub-layer (attention, ffn), on the stream h [B, L, hc, D]:
  mixes   = rms(flatten(h)) @ fn.T                       (no norm weight; f32)
  pre     = sigmoid(mixes[:hc] * scale[0] + base[:hc]) + eps
  post    = magnitude * sigmoid(mixes[hc:2hc] * scale[1] + base[hc:2hc])  (+ eps in the identity form when `post_eps`: M:5414-5415)
  comb    identity form (fn has 2 hc rows): none, the residual passes as is
          Sinkhorn form (fn has (2 + hc) hc rows): softmax rows + eps, then column / row normalisation `iters` times
  x       = sum_i pre_i h_i            -> the sub-layer (its own norm, attention / ffn) -> y
  h'      = post_j y + sum_i comb_ij h_i   (identity: h_j + post_j y)
Head (top): pre = sigmoid(mixes * scale + base) + eps, out = sum_i pre_i h_i.
The op order is the one of the served references (mlx-vlm deepseek_v4 hyper_connection / deepseek_v41 language for the Sinkhorn form, hy_v4
hyper_connection for the identity form), so the tests compare with array_equal.
"""
from __future__ import annotations

import os

import mlx.core as mx

from . import hc_fuse_pf as HF
from .rowinv import skinny_mm
from .step_compile import around as _around

KEYS = {"hyper_connection.count", "hyper_connection.epsilon", "hyper_connection.magnitude", "hyper_connection.sinkhorn_iterations"}


def form_of(kv_sub: dict) -> str:
    """'sinkhorn' when the file has a sinkhorn_iterations key, else 'identity' (HY4_DESIGN.md §3 N2)."""
    return "sinkhorn" if "hyper_connection.sinkhorn_iterations" in kv_sub else "identity"


def mixes(h, fn, norm_eps):
    return skinny_mm(mx.fast.rms_norm(h.flatten(-2).astype(mx.float32), None, norm_eps), fn)     # flag skinny_rows: few rows (chained verify) = one rowdot launch


def split(m, scale, base, hc, eps, magnitude, iters=0, post_eps=False):
    """-> pre [.., hc], post [.., hc], comb [.., hc, hc] or None (identity form)."""
    m, scale, base = m.astype(mx.float32), scale.astype(mx.float32), base.astype(mx.float32)
    pre = mx.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = magnitude * mx.sigmoid(m[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    if m.shape[-1] == 2 * hc:
        return pre, (post + eps if post_eps else post), None
    comb = m[..., 2 * hc:].reshape(*m.shape[:-1], hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def collapse(h, pre):
    return (pre[..., None].astype(mx.float32) * h.astype(mx.float32)).sum(axis=2).astype(h.dtype)


def expand(y, h, post, comb=None):
    """y [B, L, D] sub-layer output, h [B, L, hc, D] the stream before the sub-layer -> the new stream."""
    if comb is None:
        return (h.astype(mx.float32) + post[..., None] * y[:, :, None, :].astype(mx.float32)).astype(h.dtype)
    mixed = (comb[..., None] * h.astype(mx.float32)[..., None, :]).sum(axis=2)
    return (post[..., None] * y.astype(mx.float32)[:, :, None, :] + mixed).astype(y.dtype)


def embed(x, hc):
    """[B, L, D] embeddings -> [B, L, hc, D] stream (each copy = the embedding)."""
    return mx.broadcast_to(x[:, :, None, :], (x.shape[0], x.shape[1], hc, x.shape[2]))


def head(h, fn, base, scale, eps, norm_eps):
    y = h.astype(mx.float32)
    m = skinny_mm(mx.fast.rms_norm(y.flatten(-2), None, norm_eps), fn)
    pre = mx.sigmoid(m * scale + base) + eps
    return (pre[..., None] * y).sum(axis=2).astype(h.dtype)


class HcStream:
    """One sub-layer's hyper-connection: weights fn / base / scale and the file's count / eps / magnitude / iterations."""

    def __init__(self, fn, base, scale, hc, eps, magnitude, norm_eps, iters=0, post_eps=False):
        self.fn, self.base, self.scale = fn, base, scale
        self.hc, self.eps, self.magnitude, self.norm_eps, self.iters, self.post_eps = hc, eps, magnitude, norm_eps, iters, post_eps

    def __call__(self, h, sub):
        """h [B, L, hc, D], sub: [B, L, D] -> [B, L, D] (norm + attention / ffn) -> the new stream."""
        if HF.on() and HF.ok(self, h):                                              # KYOJIN_HY4_HC_FUSE=1: mix + collapse in one launch, expand in one (prompt chunks, identity form)
            x, post = HF.pre(self, h)
            y, (h, post) = _around((h, post), x, sub)
            return HF.expand(y, h, post)
        pre, post, comb = split(mixes(h, self.fn, self.norm_eps), self.scale, self.base, self.hc, self.eps, self.magnitude, self.iters, self.post_eps)
        y, (h, post, comb) = _around((h, post, comb), collapse(h, pre), sub)   # step_compile split point (a plain call outside a split trace)
        return expand(y, h, post, comb)

    def fusable(self, h) -> bool:
        """KYOJIN_HC_ID_FUSE (IDFUSE, read per call): the identity form on a decode-sized row count runs the fused launches (fused())."""
        return (IDFUSE and self.fn.shape[0] == 2 * self.hc and h.ndim == 4 and h.shape[2] == self.hc and h.shape[3] % 4 == 0
                and 0 < h.shape[0] * h.shape[1] <= MAX_ROWS and h.dtype in (mx.float32, mx.float16, mx.bfloat16) and mx.default_device() == mx.gpu)

    def fused(self, h, sub, pre_norm, post_norm):
        """The identity form in 3 launches + the sub-layer: = expand(post_norm(sub(pre_norm(collapse(h, pre)))), h, post) of __call__ with
        sub = core (attention / ffn without its norms). A norm the kernels do not take (layer norm, bias, other weight dtype) runs as its own module.
        mix: gluefuse.mixvec (stock: rms_norm + f32 matmul, another summation order: ~1e-7 rel); split / collapse / expand: the stock op sequence."""
        from . import fuse as F
        from . import gluefuse as GF
        B, L, H, D = h.shape
        R = B * L
        hr = h.reshape(R, H, D)
        m = GF.mixvec(hr.reshape(R, H * D), self.fn, self.norm_eps)
        if m is None:
            STATS["mix_stock"] += 1
            m = mixes(h, self.fn, self.norm_eps).reshape(R, 2 * H).astype(mx.float32)
        f32 = lambda a: a if a.dtype == mx.float32 else a.astype(mx.float32)  # noqa: E731
        fold = _rms_weight(pre_norm, h.dtype, D, post=False)
        x, post = F.mhc_pre_identity(hr, m, f32(self.scale), f32(self.base), self.eps, self.magnitude, self.post_eps,
                                     *((fold, pre_norm.eps) if fold is not None else ()))
        x = x.reshape(B, L, D)
        if fold is None and pre_norm is not None:
            STATS["norm_stock"] += 1
            x = pre_norm(x)
        y = sub(x)
        pw = _rms_weight(post_norm, y.dtype, D, post=True)
        if pw is None and post_norm is not None:
            STATS["norm_stock"] += 1
            y = post_norm(y)
        STATS["fused"] += 1
        out = F.mhc_expand_identity(y.reshape(R, D), hr, post, *((pw, post_norm.eps, bool(post_norm.mul_first)) if pw is not None else ()))
        return out.reshape(B, L, H, D)


IDFUSE = os.environ.get("KYOJIN_HC_ID_FUSE", "0") != "0"     # flags.hc_id_fuse; read per call
MAX_ROWS = 16                                                  # decode / short verify rows; prompt chunks keep the stock ops
STATS = {"fused": 0, "mix_stock": 0, "norm_stock": 0}


def _rms_weight(norm, dtype, D, post):
    """The weight of `norm` when the fused kernels compute it exactly as foreign_skeleton.Norm does: rms kind with a [D] weight of the operand's dtype
    (mul_first only for the post norm), else None (the module runs on its own)."""
    if norm is None or getattr(norm, "kind", None) != "rms" or "bias" in norm or not isinstance(getattr(norm, "eps", None), (int, float)):
        return None
    w = getattr(norm, "weight", None)
    if w is None or w.shape != (D,) or w.dtype != dtype or (getattr(norm, "mul_first", False) and not post):
        return None
    return w
