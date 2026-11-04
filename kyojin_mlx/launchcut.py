# SPDX-License-Identifier: Apache-2.0
"""launchcut: fewer launches and graph nodes per decode token (KYOJIN_LAUNCHCUT=1, default off).

One switch, three groups. Every group is a rewrite of the glue between launches the model already makes; none changes a number the model computes
(the stage kernels keep the arithmetic and reduction layout of the kernels they replace). KYOJIN_LAUNCHCUT_SKIP=chain,comb,router leaves groups out
(bisecting on the Ultra); the groups apply on top of KYOJIN_PORT_SUGLUE=1 (the served config: the stages are suglue's).

  chain   the chained verify (spec.verify: MTP / DSpark rows, R = 1..CHAIN_MAX) runs suglue's stages A / B / E and the q+kv norm launch inside its
          compiled regions, like the solo decode does (fuse_layer._chain_regions sg=True): the ten glue launches of a layer become three, the two
          RMSNorms one. suglue's stage kernels take one grid row per position, so a row keeps its solo bits.
  comb    the routed + shared expert combine ((routed fp32 + shared fp32) -> x dtype, one compiled elementwise launch) moves into the prologue of
          stage E, which then reads the two MoE outputs instead of the rounded sum (suglue.stage_e_comb). One launch less per MoE layer.
  router  the router GEMV (rowdot2) and the top-k / weight normalisation (router_select) in one launch: the last threadgroup of a row group to
          finish its logits selects (fuse_router.py). One launch less per MoE layer.

`on(group)` is read at trace time by the compiled regions: flip it through kyojin_mlx.flags (bumps GEN, the regions retrace)."""
from __future__ import annotations

from . import archs as _A
import os

ENABLED = os.environ.get("KYOJIN_LAUNCHCUT", "0") == "1"
GROUPS = ("chain", "comb", "router")
SKIP = {g for g in os.environ.get("KYOJIN_LAUNCHCUT_SKIP", "").replace(" ", "").split(",") if g}
STATS = {g: 0 for g in GROUPS}                                              # call sites that took the group (trace-time for the compiled regions)


def on(group: str) -> bool:
    assert group in GROUPS, group
    return ENABLED and group not in SKIP


# ---------------------------------------------------------------------------------------------------------------------- comb
_MOE = None                                           # the model's MoE class, resolved on the first call (one model per process)


def comb_ok(ffn, x) -> bool:
    """`ffn` is the stock DS4.1 MoE and `x` one token per row: its tail ((routed fp32 + shared fp32) -> x dtype) can ride in stage E."""
    import mlx.core as mx
    global _MOE
    if _MOE is None:
        _MOE = _A.cls("moe")
    return bool(on("comb") and isinstance(ffn, _MOE) and x.ndim == 3 and x.shape[1] == 1 and x.dtype in (mx.float16, mx.bfloat16))


def moe_parts(ffn, x, image_mask=None):
    """DeepseekV41MoE.__call__ up to the combine: (routed experts summed over the slot axis, fp32 [.., D]; shared expert output, x dtype). The same
    calls, in the same order, as the stock body."""
    import importlib
    import mlx.core as mx
    L = _A.lang()
    inds, scores = ffn.gate(x, image_mask)
    y = ffn.switch_mlp(L.fake_quant_fp8_ue8m0(x), inds, scores)
    return y.astype(mx.float32).sum(-2), ffn.shared_experts(x)


def combine_ref(routed, shared, dtype):
    """The stock combine, the reference of stage E's prologue: (routed + shared.astype(fp32)).astype(x dtype)."""
    import mlx.core as mx
    return (routed + shared.astype(mx.float32)).astype(dtype)


def ffn_stage_e(blk, ffn, yn, image_mask, x2, z_ffn):
    """FFN + stage E of a block: (h', h' as fp32, pre'). With the comb group the MoE combine is stage E's prologue; otherwise `ffn(yn)` then stage E."""
    import mlx.core as mx
    from . import suglue as SG
    from . import ablit as AB
    a = getattr(blk, "__dict__", {}).get("_ablit")
    if a is not None and a.wm != 0.0:
        return SG.stage_e(blk, AB.ffn_out(blk, ffn(yn, image_mask)), x2, z_ffn)
    if comb_ok(ffn, yn):
        routed, shared = moe_parts(ffn, yn, image_mask)
        if shared.dtype == yn.dtype and routed.dtype == mx.float32 and routed.shape == shared.shape:
            STATS["comb"] += 1
            return SG.stage_e_comb(blk, routed, shared, x2, z_ffn)
        return SG.stage_e(blk, combine_ref(routed, shared, yn.dtype), x2, z_ffn)
    return SG.stage_e(blk, ffn(yn, image_mask), x2, z_ffn)
