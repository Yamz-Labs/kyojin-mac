# SPDX-License-Identifier: Apache-2.0
"""Compiled decode regions of the DS4.1 block: the python graph build of a layer is paid once (mx.compile trace), later tokens replay it.

The decode step of one user is ~50-150 launches per layer; building their lazy graph in python costs about as much CPU as encoding them (layer_cpu.py: build 0.18 ms +
encode 0.15 ms for a reuse layer). A block is cut in two regions around the attention core, which is the only part with growing shapes and host state:

    A  mHC mix + collapse + norm of the attention input, q / kv projections (+ the normalised input and q latent the compressor / indexer read)
    .  attention core (fuse_attn.core): compressor / indexer / ring write / attention kernel, python + a few launches, cache side effects
    B  wo_a, wo_b, mHC expand, mHC mix + collapse + norm of the FFN, FFN (router, routed + shared experts), mHC expand

Both regions are pure functions of arrays (the weights are constants of the trace); the fp32 copy the fused mHC expand writes crosses the boundary as an explicit
output / input (fuse.take_f32 side channel). The first call of a block runs the stock code (lazily built kernel parameters evaluate there; a trace may not eval).
Shapes of the regions do not depend on the position, so a block traces once per (dtype, image mask) variant. Anything else (batch > 1, several tokens, local layers,
fuse off) runs the stock `__call__`. Switch: module attribute ENABLED (env KYOJIN_FUSE_COMPILE=0 turns it off); flip it before the first call of a model.

Chained verify (spec.verify: n consecutive positions of ONE sequence, `ChainCache`). The same two regions for n rows, traced once per (n, variant):
    A_n  as A for n rows, plus the projections the compressor and the indexer read (the `rows` hoist table: they run once for all rows, as in
         spec.dsv4_attention_chain), every per-row piece (x, q latent, q, kv, hoisted outputs) emitted as its own array
    .    n attention cores one after the other, row j on the layer slots row j - 1 left (ChainCache.take_layer): fuse_attn.core, python
    B_n  as B for n rows (the cores' outputs are stacked inside the region)
The rows keep the bits of their solo step (row-invariant linears, EXL3 GEMV of <= 4 rows, pinned MoE, one grid row per mHC row): tools/spec_check.py.
Switch: CHAIN (env KYOJIN_FUSE_CHAIN=0). STATS counts the traces of the regions and the blocks that fell back to the stock code (a retrace per position or
a silent fallback shows up there)."""
from __future__ import annotations

from . import archs as _A
import os
from types import SimpleNamespace

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import ablit as AB
from . import fuse as F
from . import fuse_attn as FA
from . import flags as FL
from . import launchcut as LC                                                   # KYOJIN_LAUNCHCUT=1: fewer launches (chain: suglue stages in the chained verify)
from . import hostsync as _HS
from . import suglue as SG                                                       # KYOJIN_PORT_SUGLUE=1: fused hyper-connection stages (3 launches per layer)
from . import glue_fuse as GN                                                     # KYOJIN_GLUE_FUSE=1 + KYOJIN_GLUE_FUSE_NORM=1: collapse + RMSNorm + sinkhorn in one launch (perf/glueroof)

ENABLED = os.environ.get("KYOJIN_FUSE_COMPILE", "1") != "0"
CHAIN = os.environ.get("KYOJIN_FUSE_CHAIN", "1") != "0"
CHAIN_MAX = 8                                                                      # rows of a chained verify the compiled path takes (more run the stock code)
ROWS = os.environ.get("KYOJIN_ROWS_REGIONS", "0") not in ("0", "")      # compiled regions of the served batched rows pass (default OFF; explicit opt-in, independent of KYOJIN_VERIFY_ASYNC)
ROWS_MAX = int(os.environ.get("KYOJIN_ROWS_REGIONS_MAX", "8"))                    # rows of a pass the compiled regions take (more run the stock code)
STATS = {"trace_solo": 0, "trace_chain": 0, "call_solo": 0, "call_chain": 0, "fallback": 0, "core_rows": 0, "glue": 0, "trace_rows": 0, "call_rows": 0}      # trace_*: python bodies run by mx.compile (one per new signature)


def fold_ok(blk) -> bool:
    """The one-launch mix (rows.mix_vector + fuse.mhc_mix_pre) applies to a 4-stream block whose hc_mixes is the rows site, also under stacked CED / spec subclasses."""
    return bool(F.MIXPRE and GN.site_installed(blk) and blk.hc_mult == KG.HC)


def _regions(self, L, with_h32: bool, with_mask: bool, out_done: bool, sg: bool = False):
    """The compiled functions of block `self`. A(h, [h32], pre_mix) -> (x, qr, q, kv, attn_pre, attn_post, attn_comb);
    B(o | attn_out, h, attn_post, attn_comb, attn_pre, [image_mask]) -> (h', h'32, ffn_pre). `out_done`: B's first input is already wo_b's output."""
    attn = self.attn
    fold = fold_ok(self)

    def mix_pre(h, fn, scale, base, pre_mix, norm=None):
        """(hc_pre(h, pre_mix), pre', post, comb of the mix of h, normed hc_pre): one launch when the block's mix is the row-invariant one
        (rows.mix_vector); with KYOJIN_GLUE_FUSE the RMSNorm of the collapsed input rides in that kernel (gluefuse.mixpre_norm) and the caller
        uses the fifth value instead of running its own norm launch. None as the fifth value: the caller runs the stock norm."""
        if norm is not None and GN.ENABLED and GN.block_ok(self, h, pre_mix, None, None):    # glue_fuse.mix_norm: works with or without the fold
            xn, pre, post, comb = GN.mix_norm(GN.mixes_of(self, h, fn), h, pre_mix, norm.weight, norm.eps, scale, base, self.hc_sinkhorn_iters, self.hc_eps)
            return xn, pre, post, comb, xn
        if fold:
            from . import rows as RW
            from . import gluefuse as GF
            mixes = RW.mix_vector(self, h, fn)
            if GF.ENABLED and GF.MIXNORM and norm is not None and GF.mixnorm_ok(h, norm.weight):
                y, pre, post, comb = GF.mixpre_norm(mixes, scale, base, h, pre_mix, norm.weight, self.hc_sinkhorn_iters, self.hc_eps, norm.eps)
                return y, pre, post, comb, y
            return F.mhc_mix_pre(mixes, scale, base, h, pre_mix, self.hc_sinkhorn_iters, self.hc_eps) + (None,)
        pre, post, comb = self.hc_mixes(h, fn, scale, base)
        return self.hc_pre(h, pre_mix), pre, post, comb, None

    def A(*args):
        STATS["trace_solo"] += 1
        h, pre_mix = args[0], args[-1]
        F._last = (h, args[1]) if with_h32 else (None, None)
        if sg:
            x, z_attn = SG.stage_a(self, h, pre_mix)
        else:
            y, attn_pre, attn_post, attn_comb, xn = mix_pre(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix, self.attn_norm)
            x = self.attn_norm(y) if xn is None else xn
        fq = L.fake_quant_fp8_ue8m0
        if sg:
            qa, ka_ = attn.wq_a(fq(x)), attn.wkv(fq(x))
            qr, kv = SG.qk_norm(attn, qa, ka_) if SG.qk_norm_ok(attn, qa, ka_) else (attn.q_norm(qa), attn.kv_norm(ka_))
            q = attn.wq_b(fq(qr))
        else:
            qr = attn.q_norm(attn.wq_a(fq(x)))
            q = attn.wq_b(fq(qr))
            kv = attn.kv_norm(attn.wkv(fq(x)))
        F._last = (None, None)
        if sg:
            return x, qr, q, kv, z_attn
        return x, qr, q, kv, attn_pre, attn_post, attn_comb

    def B_sg(o, h, z_attn, *mask):
        STATS["trace_solo"] += 1
        ao = AB.attn_out(self, o if out_done else FA.project_out(attn, L, o))
        yn, x2, z_ffn = SG.stage_b(self, ao, h, z_attn)
        h2, h2_32, ffn_pre = LC.ffn_stage_e(self, self.ffn, yn, mask[0] if with_mask else None, x2, z_ffn)
        F._last = (None, None)
        return h2, h2_32, ffn_pre

    def B(o, h, attn_post, attn_comb, attn_pre, *mask):
        STATS["trace_solo"] += 1
        ao = AB.attn_out(self, o if out_done else FA.project_out(attn, L, o))
        x = L.hc_expand(ao, h, attn_post, attn_comb)
        y, ffn_pre, ffn_post, ffn_comb, yn = mix_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre, self.ffn_norm)
        y = self.ffn_norm(y) if yn is None else yn
        y = AB.ffn_out(self, self.ffn(y, mask[0] if with_mask else None))
        h2 = L.hc_expand(y, x, ffn_post, ffn_comb)
        h2_32 = F.region_f32(h2)
        F._last = (None, None)
        return h2, h2_32, ffn_pre

    from . import replay as RP                                                         # KYOJIN_REPLAY=1: the two compiled regions are recorded once and replayed as indirect command buffers
    return RP.region(mx.compile(A), f"ds_A{self.layer_idx}"), RP.region(mx.compile(B_sg if sg else B), f"ds_B{self.layer_idx}")


def _chain_regions(self, L, n: int, with_h32: bool, with_mask: bool, out_done: bool, stacked: bool = False, sg: bool = False):
    """The compiled functions of block `self` for a chained verify of n rows. A(h, [h32], pre_mix) -> (attn_pre, attn_post, attn_comb, rows) with
    rows[j] = (x, qr, q, kv, {hoisted name: output}) for row j (each its own array); B(cores, h, attn_post, attn_comb, attn_pre, [image_mask]) -> (h', h'32, ffn_pre).
    `cores`: the n attention-core outputs [1, G, 1, in] (or [1, 1, H] after wo_b when `out_done`), stacked inside the region.
    `sg` (launchcut chain): suglue's stages A / B / E and the q + kv norm launch replace the mix / expand / norm chains, as in the solo regions: A returns
    (z_attn, rows, q, kv), B(cores, h, z_attn, [image_mask]); suglue's stages take one grid row per position, so a row keeps its solo bits."""
    attn = self.attn
    hoist = attn.rows.hoist
    fold = fold_ok(self)

    def mix_pre(h, fn, scale, base, pre_mix, norm=None):
        """As in _regions: the fifth value is the normed hc_pre when gluefuse folded the norm into the mix kernel, else None."""
        if norm is not None and GN.ENABLED and GN.block_ok(self, h, pre_mix, None, None):    # glue_fuse.mix_norm: works with or without the fold
            xn, pre, post, comb = GN.mix_norm(GN.mixes_of(self, h, fn), h, pre_mix, norm.weight, norm.eps, scale, base, self.hc_sinkhorn_iters, self.hc_eps)
            return xn, pre, post, comb, xn
        if fold:
            from . import rows as RW
            from . import gluefuse as GF
            mixes = RW.mix_vector(self, h, fn)
            if GF.ENABLED and GF.MIXNORM and norm is not None and GF.mixnorm_ok(h, norm.weight):
                y, pre, post, comb = GF.mixpre_norm(mixes, scale, base, h, pre_mix, norm.weight, self.hc_sinkhorn_iters, self.hc_eps, norm.eps)
                return y, pre, post, comb, y
            return F.mhc_mix_pre(mixes, scale, base, h, pre_mix, self.hc_sinkhorn_iters, self.hc_eps) + (None,)
        pre, post, comb = self.hc_mixes(h, fn, scale, base)
        return self.hc_pre(h, pre_mix), pre, post, comb, None

    def A(*args):
        STATS["trace_chain"] += 1
        h, pre_mix = args[0], args[-1]
        F._last = (h, args[1]) if with_h32 else (None, None)
        fq = L.fake_quant_fp8_ue8m0
        if sg:
            LC.STATS["chain"] += 1
            x, z_attn = SG.stage_a(self, h, pre_mix)
            qa, ka_ = attn.wq_a(fq(x)), attn.wkv(fq(x))
            qr, kv = SG.qk_norm(attn, qa, ka_) if SG.qk_norm_ok(attn, qa, ka_) else (attn.q_norm(qa), attn.kv_norm(ka_))
        else:
            y, attn_pre, attn_post, attn_comb, xn = mix_pre(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix, self.attn_norm)
            x = self.attn_norm(y) if xn is None else xn
            qr = attn.q_norm(attn.wq_a(fq(x)))
            kv = attn.kv_norm(attn.wkv(fq(x)))
        qrq = fq(qr)
        q = attn.wq_b(qrq)
        pre = {}                                                                   # what the compressor and the indexer would project row by row (rows.Hoist)
        comp, idx = attn.compressor, attn.indexer
        if comp is not None:
            if comp.compress_ratio == 1:
                pre["compressor.wkv"] = hoist["compressor.wkv"].inner(x)
            else:
                xf = x.astype(mx.float32)
                pre["compressor.wkv"], pre["compressor.wgate"] = hoist["compressor.wkv"].inner(xf), hoist["compressor.wgate"].inner(xf)
        if idx is not None:
            pre["indexer.wq_b"], pre["indexer.weights_proj"] = hoist["indexer.wq_b"].inner(qrq), hoist["indexer.weights_proj"].inner(x)
        F._last = (None, None)
        rows = [(x[i:i + 1], qr[i:i + 1], q[i:i + 1], kv[i:i + 1], {k: v[i:i + 1] for k, v in pre.items()}) for i in range(n)]
        if sg:
            return z_attn, rows, q, kv
        return attn_pre, attn_post, attn_comb, rows, q, kv

    def B_sg(cores, h, z_attn, *mask):
        STATS["trace_chain"] += 1
        o = cores[0] if stacked else mx.concatenate(cores, axis=0)
        ao = AB.attn_out(self, o if out_done else FA.project_out(attn, L, o))
        yn, x2, z_ffn = SG.stage_b(self, ao, h, z_attn)
        h2, h2_32, ffn_pre = LC.ffn_stage_e(self, self.ffn, yn, mask[0] if with_mask else None, x2, z_ffn)
        F._last = (None, None)
        return h2, h2_32, ffn_pre

    def B(cores, h, attn_post, attn_comb, attn_pre, *mask):
        STATS["trace_chain"] += 1
        o = cores[0] if stacked else mx.concatenate(cores, axis=0)
        ao = AB.attn_out(self, o if out_done else FA.project_out(attn, L, o))
        x = L.hc_expand(ao, h, attn_post, attn_comb)
        y, ffn_pre, ffn_post, ffn_comb, yn = mix_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre, self.ffn_norm)
        y = self.ffn_norm(y) if yn is None else yn
        y = AB.ffn_out(self, self.ffn(y, mask[0] if with_mask else None))
        h2 = L.hc_expand(y, x, ffn_post, ffn_comb)
        h2_32 = F.region_f32(h2)
        F._last = (None, None)
        return h2, h2_32, ffn_pre

    from . import replay as RP
    tag = "s" if sg else ""
    return RP.region(mx.compile(A), f"ds_cA{tag}{n}_{self.layer_idx}"), RP.region(mx.compile(B_sg if sg else B), f"ds_cB{tag}{n}_{self.layer_idx}")


def _chain_block(orig, L, SP, self, h, pre_mix, image_mask, cache):
    """The compiled chained verify of one block, or None when the call is outside the covered case (the caller then runs the stock code)."""
    n = len(cache.rows)
    if not (CHAIN and ENABLED and FA.ENABLED and F.ENABLED and F.MHC and 1 <= n <= CHAIN_MAX and h.ndim == 4 and h.shape[0] == n and h.shape[1] == 1
            and h.shape[2] == 4 and h.dtype in (mx.float16, mx.bfloat16) and pre_mix.dtype == mx.float32 and (image_mask is None or image_mask.size == n)
            and FA.decode_ok(self.attn, cache.rows[0], 1, 1, h.dtype) and getattr(self.attn, "rows", None) is not None):
        return None
    st = self.__dict__.get("_kfl")
    if st is None:
        st = self._kfl = SimpleNamespace(warm=False, fns={}, warm_n=set(), gen=FL.GEN)
    if st.gen != FL.GEN:                                                           # a lever flag switched (kyojin_mlx.flags): retrace the regions
        st.fns.clear()
        st.gen = FL.GEN
    if n not in st.warm_n:                                                         # lazily built kernel parameters of the n-row kernels evaluate in a stock call
        st.warm_n.add(n)
        return orig(self, h, pre_mix, image_mask, cache)
    STATS["call_chain"] += 1
    h32 = F.take_f32(h)
    with_mask = image_mask is not None
    sg = LC.on("chain") and SG.stage_ok(self, h) and pre_mix.shape == (n, 1, 4)
    ka = ("A", n, h32 is not None, sg)
    if ka not in st.fns:
        st.fns[ka] = _chain_regions(self, L, n, h32 is not None, with_mask, False, False, sg)[0]
    r = st.fns[ka](h, h32, pre_mix) if h32 is not None else st.fns[ka](h, pre_mix)
    if sg:
        z_attn, rows, q_all, kv_all = r
        rest = (z_attn,)
    else:
        attn_pre, attn_post, attn_comb, rows, q_all, kv_all = r
        rest = (attn_post, attn_comb, attn_pre)
    attn, hoist, l = self.attn, self.attn.rows.hoist, self.layer_idx
    cores, is_out = [], []
    o_all = FA.core_rows(attn, L, cache, q_all, kv_all)                            # reuse layer: the n cores in 3 launches
    if o_all is not None:
        STATS["core_rows"] += 1
        kb = ("B", n, with_mask, False, True, sg)
        if kb not in st.fns:
            st.fns[kb] = _chain_regions(self, L, n, False, with_mask, False, True, sg)[1]
        h2, h2_32, ffn_pre = st.fns[kb]([o_all], h, *rest, *((image_mask,) if with_mask else ()))
        F._last = (h2, h2_32)
        return h2, ffn_pre
    try:
        for j, (x, qr, q, kv, pre) in enumerate(rows):
            cache.take_layer(j, l)
            for name, v in pre.items():
                m = hoist[name].st
                m.pre, m.row = v, None
            o, out = FA.core(attn, L, cache.rows[j], x, qr, q, kv)
            cores.append(out if o is None else o)
            is_out.append(o is None)
            for name in pre:
                hoist[name].st.pre = None
    finally:
        for m in hoist.values():
            m.st.pre = None
    if any(is_out) and not all(is_out):                                            # rows disagree (the stock tail for some of them): all rows to the wo_b output
        cores = [c if f else FA.project_out(attn, L, c) for c, f in zip(cores, is_out)]
    out_done = any(is_out)
    kb = ("B", n, with_mask, out_done, False, sg)
    if kb not in st.fns:
        st.fns[kb] = _chain_regions(self, L, n, False, with_mask, out_done, False, sg)[1]
    h2, h2_32, ffn_pre = st.fns[kb](cores, h, *rest, *((image_mask,) if with_mask else ()))
    F._last = (h2, h2_32)
    return h2, ffn_pre


def _rows_cls():
    """The RowsCache class (None until a batched pass stacked a cache)."""
    from . import dsv4_core as DC
    return DC._rows_cls


def _rows_regions(self, L, with_h32: bool, with_mask: bool):
    """Compiled functions of block `self` for a pass of the batched rows cache (any row count: mx.compile keeps one trace per input shape).
    A(h, [h32], pre_mix) -> (x, attn_pre, attn_post, attn_comb): the mHC mix + collapse + norm in front of the attention (the cores and their
    projections stay eager: dsv4_core.core); B(o, h, attn_post, attn_comb, attn_pre, [image_mask]) -> (h', h'32, ffn_pre) with `o` the attention
    output after wo_b. Same pieces, same order as _chain_regions (and as the stock block body), so the bits are those of the eager pass."""
    def mix_pre(h, fn, scale, base, pre_mix, norm):
        if GN.ENABLED and GN.block_ok(self, h, pre_mix, None, None):
            xn, pre, post, comb = GN.mix_norm(GN.mixes_of(self, h, fn), h, pre_mix, norm.weight, norm.eps, scale, base, self.hc_sinkhorn_iters, self.hc_eps)
            return xn, pre, post, comb, xn
        if fold_ok(self):
            from . import rows as RW
            from . import gluefuse as GF
            mixes = RW.mix_vector(self, h, fn)
            if GF.ENABLED and GF.MIXNORM and GF.mixnorm_ok(h, norm.weight):
                y, pre, post, comb = GF.mixpre_norm(mixes, scale, base, h, pre_mix, norm.weight, self.hc_sinkhorn_iters, self.hc_eps, norm.eps)
                return y, pre, post, comb, y
            return F.mhc_mix_pre(mixes, scale, base, h, pre_mix, self.hc_sinkhorn_iters, self.hc_eps) + (None,)
        pre, post, comb = self.hc_mixes(h, fn, scale, base)
        return self.hc_pre(h, pre_mix), pre, post, comb, None

    def A(*args):
        STATS["trace_rows"] += 1
        h, pre_mix = args[0], args[-1]
        F._last = (h, args[1]) if with_h32 else (None, None)
        y, attn_pre, attn_post, attn_comb, xn = mix_pre(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix, self.attn_norm)
        x = self.attn_norm(y) if xn is None else xn
        F._last = (None, None)
        return x, attn_pre, attn_post, attn_comb

    def B(ao, h, attn_post, attn_comb, attn_pre, *mask):
        STATS["trace_rows"] += 1
        x = L.hc_expand(ao, h, attn_post, attn_comb)
        y, ffn_pre, ffn_post, ffn_comb, yn = mix_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre, self.ffn_norm)
        y = self.ffn_norm(y) if yn is None else yn
        y = AB.ffn_out(self, self.ffn(y, mask[0] if with_mask else None))
        h2 = L.hc_expand(y, x, ffn_post, ffn_comb)
        h2_32 = F.region_f32(h2)
        F._last = (None, None)
        return h2, h2_32, ffn_pre

    return mx.compile(A), mx.compile(B)


def _rows_block(orig, L, self, h, pre_mix, image_mask, cache):
    """The served batched pass (RowsCache: one row per session or several per session) of one block with its two pure regions compiled, or None
    when the call is outside the covered case (the caller then runs the code it ran before). The first call per row count runs the stock code
    (lazily built kernel parameters evaluate there; a trace may not eval)."""
    n = h.shape[0]
    if not (ROWS and ENABLED and F.ENABLED and F.MHC and 1 <= n <= ROWS_MAX and h.ndim == 4 and h.shape[1] == 1 and h.shape[2] == 4
            and h.dtype in (mx.float16, mx.bfloat16) and pre_mix.dtype == mx.float32 and (image_mask is None or image_mask.size == n)):
        return None
    st = self.__dict__.get("_krr")
    if st is None:
        st = self._krr = SimpleNamespace(fns={}, warm_n=set(), gen=FL.GEN)
    if st.gen != FL.GEN:
        st.fns.clear()
        st.warm_n.clear()
        st.gen = FL.GEN
    if n not in st.warm_n:
        st.warm_n.add(n)
        return orig(self, h, pre_mix, image_mask, cache)
    STATS["call_rows"] += 1
    h32 = F.take_f32(h)
    with_mask = image_mask is not None
    key = (h32 is not None, with_mask)
    if key not in st.fns:
        st.fns[key] = _rows_regions(self, L, h32 is not None, with_mask)
    fa, fb = st.fns[key]
    x, attn_pre, attn_post, attn_comb = fa(h, h32, pre_mix) if h32 is not None else fa(h, pre_mix)
    ao = AB.attn_out(self, self.attn(x, cache))
    h2, h2_32, ffn_pre = fb(ao, h, attn_post, attn_comb, attn_pre, *((image_mask,) if with_mask else ()))
    F._last = (h2, h2_32)
    return h2, ffn_pre


def _impl_block(orig):
    """Replacement of `DeepseekV41Block.__call__` (see module doc)."""
    import importlib
    L = _A.lang()
    cache_cls = _A.cls("cache")                       # resolved once at install, the per-call test stays one isinstance
    SP = importlib.import_module(f"{__package__}.spec")

    def __call__(self, h, pre_mix, image_mask, cache):
        if type(cache) is SP.ChainCache:
            r = _chain_block(orig, L, SP, self, h, pre_mix, image_mask, cache)
            if r is None:
                if (LC.on("chain") and SG.stage_ok(self, h) and pre_mix.dtype == mx.float32 and pre_mix.shape == (h.shape[0], 1, 4)
                        and (image_mask is None or image_mask.dtype == mx.bool_)):
                    STATS["glue"] += 1                                              # launchcut chain: the local-attention layers of a chained verify get suglue's stages too
                    return SG.block(self, h, pre_mix, image_mask, cache)
                STATS["fallback"] += h.shape[1] == 1
                return orig(self, h, pre_mix, image_mask, cache)
            return r
        if ROWS and type(cache) is _rows_cls():
            r = _rows_block(orig, L, self, h, pre_mix, image_mask, cache)
            if r is not None:
                return r
        from . import decstage as DS                                              # dec_stage: the six-stage block path of the served pass
        r = DS.block(self, h, pre_mix, image_mask, cache)
        if r is not None:
            return r
        if not (ENABLED and FA.ENABLED and F.ENABLED and F.MHC and isinstance(cache, cache_cls) and h.ndim == 4 and h.shape[0] == 1 and h.shape[1] == 1
                and h.shape[2] == 4 and h.dtype in (mx.float16, mx.bfloat16) and pre_mix.dtype == mx.float32
                and (image_mask is None or image_mask.size == 1) and FA.decode_ok(self.attn, cache, 1, 1, h.dtype)):
            if SG.stage_ok(self, h) and pre_mix.dtype == mx.float32 and pre_mix.shape == (h.shape[0], 1, 4) and (image_mask is None or image_mask.dtype == mx.bool_):
                STATS["glue"] += 1                                                  # suglue stages around the stock sublayers (local layers, N-user rows)
                return SG.block(self, h, pre_mix, image_mask, cache)
            if GN.ENABLED and GN.block_ok(self, h, pre_mix, image_mask, cache):    # batched / N-user cache: the stock body with the glue chains fused
                STATS["glue"] += 1
                return GN.block(self, h, pre_mix, image_mask, cache)
            STATS["fallback"] += h.shape[1] == 1 and h.shape[0] == 1
            return orig(self, h, pre_mix, image_mask, cache)
        st = self.__dict__.get("_kfl")
        if st is None:
            st = self._kfl = SimpleNamespace(warm=False, fns={}, warm_n=set(), gen=FL.GEN)
        if st.gen != FL.GEN:                                                       # a lever flag switched (kyojin_mlx.flags): retrace the regions
            st.fns.clear()
            st.gen = FL.GEN
        if not st.warm:                                                           # lazily built kernel parameters evaluate in the first stock call
            st.warm = True
            return orig(self, h, pre_mix, image_mask, cache)
        STATS["call_solo"] += 1
        h32 = F.take_f32(h)
        with_mask = image_mask is not None
        sg = SG.stage_ok(self, h)
        ka = (h32 is not None, sg)
        if ka not in st.fns:
            st.fns[ka] = _regions(self, L, h32 is not None, with_mask, False, sg)[0]
        r = st.fns[ka](h, h32, pre_mix) if h32 is not None else st.fns[ka](h, pre_mix)
        x, qr, q, kv = r[:4]
        o, out = FA.core(self.attn, L, cache, x, qr, q, kv)
        kb = (with_mask, o is None, sg)
        if kb not in st.fns:
            st.fns[kb] = _regions(self, L, False, with_mask, o is None, sg)[1]
        rest = r[4:] if sg else (r[5], r[6], r[4])                                  # (z_attn,) or (attn_post, attn_comb, attn_pre)
        args = (out if o is None else o, h) + rest + ((image_mask,) if with_mask else ())
        h2, h2_32, ffn_pre = st.fns[kb](*args)
        F._last = (h2, h2_32)
        return h2, ffn_pre

    return __call__
