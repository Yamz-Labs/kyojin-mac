"""Prefill (S > 1) hyper-connection expand / pre for DS4.1: the decode kernels of fuse.py are row-wise, so a prefill chunk is B*S rows of the same kernel.
The stock compiled hc_expand builds a [B,S,4,4,D] fp32 intermediate (89 ms at T=2048); the kernel reads x and the residual once."""
import mlx.core as mx
from . import fuse

import os
ENABLED = os.environ.get("KYOJIN_DS_PF_MHC", "1") != "0"


def _wrap_expand(orig):
    clamp = fuse.sat if not getattr(orig, "saturates", False) else (lambda y: y)    # the decode-row kernels clamp inline: a second clip is two more launches
    def expand(x, residual, post, comb):
        if ENABLED and x.ndim == 3 and x.shape[1] > 1 and residual.ndim == 4 and residual.shape[2] == 4 and post.dtype == mx.float32 \
                and comb.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            B, S, D = x.shape
            from . import pf_glue as PG
            if B == 1 and fuse.ENABLED and fuse.MHC and PG.ENABLED and D % 8 == 0 and S >= fuse.MIN_PROMPT_ROWS:
                fuse._last = (None, None)                                   # the prompt kernel: no fp32 copy store (the row kernel writes one nobody reads)
                return PG.expand(x, residual, post, comb)
            y, y32 = fuse.mhc_expand(x.reshape(B * S, 1, D), residual.reshape(B * S, 1, 4, D), post.reshape(B * S, 1, 4), comb.reshape(B * S, 1, 4, 4))
            return y.reshape(B, S, 4, D)
        return clamp(orig(x, residual, post, comb))
    return expand


def _wrap_pre(orig):
    clamp = fuse.sat if not getattr(orig, "saturates", False) else (lambda y: y)
    def pre(x, pre_mix):
        if ENABLED and x.ndim == 4 and x.shape[1] > 1 and x.shape[2] == 4 and pre_mix.dtype == mx.float32 and x.dtype in (mx.float16, mx.bfloat16):
            B, S, _, D = x.shape
            return fuse.mhc_pre(x.reshape(B * S, 1, 4, D), pre_mix.reshape(B * S, 1, 4)).reshape(B, S, D)
        return clamp(orig(x, pre_mix))
    return pre


def install(mod, block_cls):
    mod.hc_expand = _wrap_expand(mod.hc_expand)
    block_cls.hc_pre = staticmethod(_wrap_pre(block_cls.__dict__["hc_pre"].__func__))
