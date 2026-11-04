# SPDX-License-Identifier: Apache-2.0
"""dec_stage: the served DS4.1 decode block as six explicit stages, each stage owning the written fused kernels (KYOJIN_DEC_STAGE=1, default off).

The block body of one decode layer spends its launches on glue around a few GEMVs: per sub-layer the mHC mix norm + row dot + sinkhorn,
the collapse + RMSNorm, the fp8 round trips of the attention / MoE inputs, q_norm + kv_norm, the q / kv rotary, the inverse rotary and the
expand. This module runs the block as the stages a fused decode kernel would use, in this order, so each stage owns a contiguous slice of the
launch list and can be bisected on its own:

    in    mHC pre mix + attn_norm, then wq_a | wkv (one same-input EXL3 group, the fp8 round trip inside the rot), q_norm and wq_b
          (fp8 inside the rot); kv_norm rides in the attn stage's kv chain
    attn  the batched core with the kv chain (kv_norm + rope + fp8 + ring scatter), the q rope in the attention kernel's q load, the
          inverse rope in the merge epilogue and, up to IDX_FUSE_MAX rows x keys, index score + exact top-k + candidate mask in one launch
    out   wo_a (grouped GEMV, fp8 inside the rot) + wo_b (fp8 inside the rot) + hc expand
    ffn   mHC mix + ffn_norm + sinkhorn (one launch) + router + routed / shared experts + hc expand
    moe   the routed MoE decode launches of the FFN stage (moe.MoeGather, kept where the model has it)
    engram layers keep the chain (their extra embedding write is not part of the stage path)

Two stage kernels (stagek: mix_stage, expand_stage; sub-flag GLUE, on with the stage, self-checked at first use) run the mHC glue of a
sub-layer in one launch each, f32 inside and one rounding: those two are NOT the chain's arithmetic (another summation order, no fp16 round of
the collapse before its norm), they are row independent and closer to an f32 reference (tests/test_dec_stage_k.py); with GLUE off the stage
path below is the chain's arithmetic:
Every other piece is the arithmetic of the op it replaces, in the same order, so the stage path is bit-identical to the chain it replaces (the
fused kernels it uses carry their own bit-equality tests: glue_fuse.mix_norm, suglue.qk_norm, dsv4_core.kv_chain / rope fuse, the EXL3
fp8 fold). Anything outside the covered case (other cache types, hc_mult != 4, non-fp16 activations, prompt rows, Engram layers) returns
None and the caller runs the chain.

Flags (runtime; the lever and the four sub-flags are "0", the sub-flags read only when the lever is on): KYOJIN_DEC_STAGE, _IN, _ATTN, _FFN, _MOE.
The chain it replaces, per layer: glue_fuse's five-op mix chain, two fp8 round trips before wq_a / wkv, q_norm + kv_norm, the q / kv
rotary, the kv scatter, the inverse rotary, two fp8 round trips before wo_a / wo_b, and the FFN mix chain.
"""
from __future__ import annotations

from . import archs as _A
import os

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import ablit as AB
from . import fired as FD
from . import fuse as F
from . import glue_fuse as GN
from . import stagek as SK
from . import stagek2 as SK2

ENABLED = os.environ.get("KYOJIN_DEC_STAGE", "0") not in ("0", "")
IN = os.environ.get("KYOJIN_DEC_STAGE_IN", "0") != "0"
ATTN = os.environ.get("KYOJIN_DEC_STAGE_ATTN", "0") != "0"
FFN = os.environ.get("KYOJIN_DEC_STAGE_FFN", "0") != "0"
MOE = os.environ.get("KYOJIN_DEC_STAGE_MOE", "0") != "0"
GLUE = os.environ.get("KYOJIN_DEC_STAGE_GLUE", "1") != "0"      # the one-launch mix / expand stages (stagek); off: the lane's glue chain around the stage
QKV = os.environ.get("KYOJIN_DEC_QKV", "0") != "0"              # k1-k2: q_norm + the kv chain in one launch (stagek2.qkv_stage); needs IN and the kv chain (ATTN)
IDXQ = os.environ.get("KYOJIN_DEC_IDXQ", "0") != "0"            # k1-k2: the attention wq_b and the indexer wq_b as one same-input group launch; needs IN
MAX_ROWS = int(os.environ.get("KYOJIN_DEC_STAGE_ROWS", "24"))   # the rows a stage pass serves: a plain step or any verify block (the engine's max_spec_rows)
STATS = {"qkv": 0, "idxq": 0, "block": 0, "fallback": 0, "engram": 0, "in": 0, "attn": 0, "out": 0, "ffn": 0, "moe": 0}

_SEEN: dict = {}                                          # (layer, block, members) -> fused_group.Exl3Group, built once per model


def _lang():
    L = _A.lang()
    return L


def rows_cache(cache):
    """The pass's stacked core cache. A plain step reaches the first block before any core call stacked the cache (`rows.step` hands the
    batched cache straight to the model), so stack it here: idempotent, host only, it is what the core does one call later anyway."""
    from . import dsv4_core as DC
    base = _A.cls("batch_cache")
    if not isinstance(cache, base) or not hasattr(cache, "rows"):
        return None
    if DC._rows_cls is not None and type(cache) is DC._rows_cls:
        return cache
    try:
        return DC.stack_rows(cache)
    except Exception:                                                          # a cache shape the splitter does not cover: the chain
        return None


def ok(blk, h, pre_mix, image_mask, cache) -> bool:
    """The stage path covers a pass of 1..MAX_ROWS rows of one block: fp16 activations, 4 hyper-connection streams, plain RMSNorms,
    fp32 mix matrices, the batched (stacked) core cache, no Engram write on this layer."""
    import mlx.nn as nn
    from . import dsv4_core as DC
    if not ENABLED or h.ndim != 4 or h.shape[1] != 1 or h.shape[2] != KG.HC or h.shape[0] > MAX_ROWS \
            or h.dtype not in (mx.float16, mx.bfloat16) or pre_mix.dtype != mx.float32 \
            or pre_mix.shape != (h.shape[0], 1, KG.HC) or getattr(blk, "hc_mult", 0) != KG.HC \
            or h.shape[-1] % 128 or getattr(blk, "engram", None) is not None:
        return False
    if image_mask is not None and (image_mask.dtype != mx.bool_ or image_mask.size != h.shape[0]):
        return False
    if type(cache) is not DC._rows_cls or DC._rows_cls is None:
        return False
    d = h.shape[-1]
    if blk.hc_attn_fn.dtype != mx.float32 or blk.hc_ffn_fn.dtype != mx.float32 \
            or blk.hc_attn_fn.shape != (24, 4 * d) or blk.hc_ffn_fn.shape != (24, 4 * d):
        return False
    if not all(type(n) is nn.RMSNorm for n in (blk.attn_norm, blk.ffn_norm)):
        return False
    if blk.attn_norm.weight.dtype != h.dtype or blk.ffn_norm.weight.dtype != h.dtype:
        return False
    if not hasattr(blk, "hc_sinkhorn_iters") or not hasattr(blk, "hc_eps"):
        return False
    return blk.attn.window_size >= 1 and hasattr(blk.attn, "layer_idx")


def _mix(blk, h, fn, scale, base, pre_mix, norm, fast: bool = True):
    """mHC mix + collapse + RMSNorm -> (x, pre', post, comb, z). fast with GLUE: ONE launch (stagek.mix_stage: the 24 mix rows on 24 threadgroups, the
    collapse + norm on a 25th; f32, one rounding), `post` / `comb` None and the raw mixes `z` returned: the sinkhorn runs inside the expand stage.
    fast without GLUE: mixes_of + glue_fuse.mix_norm only when the lane turned that fusion on (KYOJIN_GLUE_FUSE_NORM: measured slower, so it is never
    forced here), else the chain's five ops. fast=False: the chain's ops (the sub-flag bisect)."""
    if fast and GLUE and glue_ok(blk, h, fn, scale, base, norm) and SK.usable():
        FD.hit("dec_stage_glue")
        hn, z, pre = SK.mix_stage(h, pre_mix, fn, scale, base, norm.weight, blk.norm_eps, norm.eps, blk.hc_eps)
        return hn, pre.reshape(h.shape[0], 1, 4), None, None, z
    if fast and GN.ENABLED:
        mixes = GN.mixes_of(blk, h, fn)
        return (*GN.mix_norm(mixes, h, pre_mix, norm.weight, norm.eps, scale, base, blk.hc_sinkhorn_iters, blk.hc_eps), None)
    pre, post, comb = blk.hc_mixes(h, fn, scale, base)
    return norm(blk.hc_pre(h, pre_mix)), pre, post, comb, None


def glue_ok(blk, h, fn, scale, base, norm) -> bool:
    """mix_stage covers fp16 / bf16 streams of 4 hyper-connections, f32 mix matrices and a plain norm weight of the stream dtype."""
    D = h.shape[-1]
    return (h.dtype in (mx.float16, mx.bfloat16) and h.shape[1:3] == (1, 4) and fn.dtype == mx.float32 and fn.shape == (SK.NMIX, 4 * D)
            and scale.size == 3 and base.size == SK.NMIX and norm.weight.dtype == h.dtype and norm.weight.shape == (D,)
            and hasattr(blk, "norm_eps") and hasattr(blk, "hc_eps"))


def _expand(blk, y, res, post, comb, z, scale, base):
    """hc_expand of the sub-layer output `y` (after the ablit hook) into the residual streams `res`: ONE launch with the sinkhorn inside when the
    mix came from mix_stage (z), else the model's hc_expand."""
    if z is not None:
        FD.hit("dec_stage_glue")
        out, out32 = SK.expand_stage(y, res, z, scale, base, blk.hc_sinkhorn_iters, blk.hc_eps)
        F._last = (out, out32)                                                # the next mix (and any chain fallback) reads the fp32 copy
        return out
    return _lang().hc_expand(y, res, post, comb)


def _lin(mod, x, fq8: bool = False):
    """One projection with the fp8 fake quant of its input folded into the GEMV's rot when the module carries it."""
    from . import exl3_linear as EL
    if fq8 and isinstance(mod, EL.Exl3Linear):
        return mod(x, fq8=True)
    if fq8:
        return mod(_lang().fake_quant_fp8_ue8m0(x))
    return mod(x)


def group_rows_ok(mods: tuple, rows: int) -> bool:
    """True when the chain would run these members as the plain fused GEMV, the only path the same-input group reproduces: the group kernel has no
    packed-fp16 row FMAs (metal_src.HALF_ROWS: the single-linear kernel uses them from 4 rows on, the served default) and no MMA / NAX / row-group
    band (Exl3Linear.__call__ takes those above GEMV_MAX_ROWS), so at those row counts the group would be another arithmetic and another kernel."""
    from . import exl3_linear as EL
    from . import kernels as KN
    from . import metal_src as S
    if rows > min(EL.GEMV_MAX_ROWS, KN.MAX_ROWS) or S.half_on(rows):
        return False
    return not any(m._mma_on(rows) for m in mods)


def _same_input_group(blk, mods: tuple, rows: int = 1):
    """One launch for linears that read the same array (wq_a | wkv) through fused_group (rot + GEMV + finalize). None when the modules are not
    plain Exl3Linear members, were already grouped by S5 (their own group then shares the same launch), or the row count is outside the band the
    chain runs as the same kernel (group_rows_ok): the members then take their own `__call__`, i.e. the chain's routing."""
    from . import exl3_linear as EL
    from . import fused_group as FG
    if not (EL.LINEAR_FUSED and EL.LF_MASK & 2) or not all(type(m) is EL.Exl3Linear and m._bias is None for m in mods) or not group_rows_ok(mods, rows):
        return None
    key = (blk.layer_idx, id(blk), tuple(id(m) for m in mods))
    g = _SEEN.get(key)
    if g is None:
        try:
            g = _SEEN[key] = FG.Exl3Group(list(mods))
        except AssertionError:
            return None
    return g


# ----------------------------------------------------------------------------------------------------------------- stages
def _qkv_args(blk, qa, ka, cache):
    """The arguments of stagek2.qkv_stage for this layer, or None when the kv chain / q_norm are outside what it covers (same gates as the core's
    kvfuse: 512-wide kv, a 64-wide rope tail, fp16 kv and kv_norm weight, plus a plain nn.RMSNorm q_norm)."""
    import mlx.nn as nn
    from . import dsv4_core as DC
    if not QKV:
        return None
    attn = blk.attn
    rows = qa.size // qa.shape[-1]
    if not (SK2.usable() and (ATTN or DC.KV_CHAIN) and getattr(attn.rope, "dims", 0) == 64 and ka.shape[-1] == 512
            and type(attn.q_norm) is nn.RMSNorm and attn.q_norm.weight.dtype == qa.dtype and attn.kv_norm.weight.dtype == mx.float16):
        return None
    qa2, ka2 = qa.reshape(rows, qa.shape[-1]), ka.reshape(rows, ka.shape[-1])
    st = DC.stack_rows(cache)
    plan = DC.plan_of(st)
    if plan.n != rows:
        return None
    W, l = attn.window_size, attn.layer_idx
    base = st.win[l] if st.win[l] is not None else mx.zeros((plan.NS, W, ka2.shape[-1]), dtype=ka.dtype)
    if not SK2.supported(qa2, ka2, base):
        return None
    rope_f, _ = plan.attn_rope(attn.rope)
    return qa2, attn.q_norm.weight, attn.q_norm.eps, ka2, attn.kv_norm.weight, base, plan.sess_arr, plan.wmap(W), rope_f, attn.kv_norm.eps


def stage_in(blk, h, pre_mix, cache=None, ex=None):
    """-> (x, qr, q, kv, (attn_pre, attn_post, attn_comb, z)): the attention input, its norms and the q / kv projections. post / comb are None
    (and z the raw mixes) when the mix came from the one-launch stage. `ex` (dict, with the cache): the k1-k2 levers hand the core what they computed
    ahead of it: "win" (QKV: the window rings, so the core skips its kv chain) and "qi" (IDXQ: the indexer wq_b output)."""
    L = _lang()
    attn = blk.attn
    x, attn_pre, attn_post, attn_comb, z = _mix(blk, h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base, pre_mix, blk.attn_norm,
                                                fast=IN)
    if not IN:                                                                # bisect: the chain's projection glue
        xf = L.fake_quant_fp8_ue8m0(x)
        qr = attn.q_norm(attn.wq_a(xf))
        kv = attn.wkv(xf)
        q = attn.wq_b(L.fake_quant_fp8_ue8m0(qr))
        return x, qr, q, kv, (attn_pre, attn_post, attn_comb, z)
    g = _same_input_group(blk, (attn.wq_a, attn.wkv), x.size // x.shape[-1])
    if g is not None:
        qa, ka = g.run(x, fq8=True)                                           # one launch: fp8 + rot + both GEMVs + finalize
    else:
        qa, ka = _lin(attn.wq_a, x, fq8=True), _lin(attn.wkv, x, fq8=True)
    qkv = _qkv_args(blk, qa, ka, cache) if ex is not None and cache is not None else None
    if qkv is not None:
        qr, ex["win"] = SK2.qkv_stage(*qkv)                                   # q_norm + kv chain, one launch
        qr = qr.reshape(qa.shape)
        STATS["qkv"] += 1
    else:
        qr = attn.q_norm(qa)                                                  # kv_norm rides in the core's kv chain (raw kv out)
    idxr = getattr(attn, "indexer", None)
    g2 = _same_input_group(blk, (attn.wq_b, idxr.wq_b), x.size // x.shape[-1]) if IDXQ and ex is not None and idxr is not None \
        and getattr(attn, "compress_ratio", 0) and qr.dtype == mx.float16 else None
    if g2 is not None:
        q, ex["qi"] = g2.run(qr.reshape(-1, qr.shape[-1]), fq8=True)                                   # wq_b | indexer wq_b: one rot, one GEMV launch (the same fp8 input)
        STATS["idxq"] += 1
    else:
        q = _lin(attn.wq_b, qr, fq8=True)
    STATS["in"] += 1
    return x, qr, q, ka, (attn_pre, attn_post, attn_comb, z)


def stage_attn(blk, x, qr, q, kv, cache, ex=None):
    """The batched core on the hoisted projections: ring / compressor / indexer / attention / merge with the ropes and the fp8 folded."""
    from . import dsv4_core as DC
    STATS["attn"] += 1
    return DC.core(blk.attn, x, cache, pre=(qr, q, kv), stage=ATTN, ex=ex)


def stage_out(blk, o, h, mix, projected: bool = True):
    """wo_a | wo_b (fp8 in the rot) and the mHC expand back into the residual. `o` is the core output [rows, G, 1, (H/G)*D] when
    `projected` (the `attn` stage), else the core's own wo_b output [rows, 1, D] and only the expand is left."""
    attn = blk.attn
    attn_pre, attn_post, attn_comb, z = mix
    if projected:
        ao = attn.wo_a(o)                                                     # Exl3MultiLinear folds the fp8 round trip (fq_in)
        ao = ao.transpose(0, 2, 1, 3).flatten(-2)
        y = _lin(attn.wo_b, ao, fq8=True)
    else:
        y = o
    y = AB.attn_out(blk, y)
    STATS["out"] += 1
    return _expand(blk, y, h, attn_post, attn_comb, z, blk.hc_attn_scale, blk.hc_attn_base)


def stage_ffn(blk, x, attn_pre, image_mask):
    """mHC mix + ffn_norm + sinkhorn, then the FFN (router + routed / shared experts) and the expand: -> (h', ffn_pre)."""
    yn, ffn_pre, ffn_post, ffn_comb, z = _mix(blk, x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base, attn_pre, blk.ffn_norm, fast=FFN)
    if MOE and getattr(blk.ffn, "switch_mlp", None) is not None and hasattr(blk.ffn.switch_mlp, "moe"):
        STATS["moe"] += 1
    y = AB.ffn_out(blk, blk.ffn(yn, image_mask))                              # routed (MoeGather: 5 launches) + shared expert
    STATS["ffn"] += 1
    return _expand(blk, y, x, ffn_post, ffn_comb, z, blk.hc_ffn_scale, blk.hc_ffn_base), ffn_pre


def block(blk, h, pre_mix, image_mask, cache):
    """The six-stage block, or None when the call is outside the covered case (the caller runs the chain)."""
    if not ENABLED or h.shape[0] > MAX_ROWS:                                      # a pass wider than the stage serves: the chain, before any host work
        return None
    cache = rows_cache(cache)
    if cache is None or not ok(blk, h, pre_mix, image_mask, cache):
        STATS["fallback"] += 1
        return None
    STATS["block"] += 1
    ex: dict = {}
    x, qr, q, kv, mix = stage_in(blk, h, pre_mix, cache, ex)
    x2 = stage_out(blk, stage_attn(blk, x, qr, q, kv, cache, ex), h, mix, projected=ATTN)
    return stage_ffn(blk, x2, mix[0], image_mask)

