# SPDX-License-Identifier: Apache-2.0
"""MoE decode on the stacked (ragged-u32-v1) layout: five launches per layer instead of ~9 per routed expert.
`MoeGather` is the fused path (kernels in metal_src.py), `MoeUnfused` the per-expert reference path built on `Mul1Gemv`
(same arithmetic, host id read, one rot + GEMV + finalize per expert and projection). Both take the router output
(ids int32 [rows, topk], weights f32 [rows, topk]) and return the routed sum in f32 [rows, hidden]."""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from . import kernels as KN
from . import metal_nax as NX
from . import metal_src as S
from . import fired as FD

MAX_PAIRS = 256
TAIL_GRP = os.environ.get("KYOJIN_PF_TAIL_GRP", "0") == "1"            # prefill: rows between one decode chunk and the S3 floor run as ONE grouped MoE call per layer (one trellis decode per distinct expert) instead of ceil(rows / 4) four-row calls; pf_tail_grp, default 0
TAIL_PAIRS = int(os.environ.get("KYOJIN_PF_TAIL_PAIRS", "1024"))      # pairs of that one call (rows * topk): the wide plan kernel serves up to 1024
TAIL_C4 = os.environ.get("KYOJIN_PF_TAIL_C4", "0") == "1"                  # prefill: a row band the four-row chunks would serve (above the verify / multi-user rows, below the S3 floor) runs as ONE grouped call; pf_tail_c4, default 0
GROUPED = os.environ.get("KYOJIN_ROWS_GROUPED", "1") in ("1", "nax")      # rows >= 2: one trellis decode per distinct expert (grouped_gemv), bitwise equal to the per-row path: ON by default (KYOJIN_ROWS_GROUPED=0 off)
UGROUPED = int(os.environ.get("KYOJIN_ROWS_UGROUPED", "3"))           # rows 2..UGROUPED: the per-pair split-K gather GEMV (the 1-row kernel) instead of the grouped GEMV; an expert with few rows has nothing to share and the grouped body's threadgroup reduction costs more than the k-split. Bit-identical (same pairs, same tgk, same order). 0 = off
MOEGRP = os.environ.get("KYOJIN_MOEGRP", "0") == "1"                  # grouped GEMV: one packed slot range, each slot's body = its member count (no idle rows of the 8-row pass, no idle slot ranges; bit-identical)
VB_SPLIT_K = os.environ.get("KYOJIN_VB_SPLIT_K", "1") != "0"          # rows >= 2 (grouped GEMV and NAX hot path): one launch per bitrate class over all slots (later passes carry earlier results), bit-identical; 0 = one mixed-K launch
SGRED = os.environ.get("KYOJIN_MOE_SGRED", "0") == "1"                # grouped GEMV (rows >= 2): each simdgroup stores its k-split partial straight to the partial buffer (no threadgroup reduction buffer, no barrier), the mid / final kernels sum TGK * 4 partials; fp32 summation order only, opt-in
AFFOLD = os.environ.get("KYOJIN_MOE_AFFOLD", "0") == "1"           # routed decode GEMVs (gather, grouped) of mul1 packs: affine folded out of the per-weight decode (metal_src.af_fold); NOT bit-identical, KL-gated, opt-in
FASTDEC = os.environ.get("KYOJIN_MOE_FASTDEC", "0") == "1"         # routed decode GEMVs (gather, grouped) of mul1 packs: affine folded AND the trellis decode cut to the bare byte sum (metal_src dec_mul1fd), per-pair gather GEMV only; NOT bit-identical, KL-gated, opt-in; wins over KYOJIN_MOE_AFFOLD
ROOF = os.environ.get("KYOJIN_MOEROOF", "1") == "1"                # routed decode GEMVs: cheaper integer decode (metal_src dec_mul1r*, dec_mul1fdr) + next-tile word prefetch in the per-pair gather GEMV; bit-identical to the codebook path it replaces
ROOF_DEC = int(os.environ.get("KYOJIN_MOEROOF_DEC", "1"))          # KYOJIN_MOEROOF decode variant of mul1: 1 = lane sum by imad, 2 = also the byte pairs by a subtract-imad (mul1fd always takes the subtract-imad)
ROOF_PF = os.environ.get("KYOJIN_MOEROOF_PF", "1") != "0"          # KYOJIN_MOEROOF: software prefetch of the next tile's trellis words in the per-pair gather GEMV
HOT_BKS = int(os.environ.get("KYOJIN_HOT_BKS", "0"))                  # k step of the decode-path NAX GEMM (0 = the _BLOCK_CFG default, 64); 32 is faster for 16-row blocks
TGT = int(os.environ.get("KYOJIN_MOE_TGT", "0"))                       # threadgroup target of the routed decode GEMV grid (pick_tgk): 0 = 480; any other value k-splits until blocks * tgk * pairs reach it. Another split adds the f32 partials in another order (1e-7 relative)
KSPLIT = int(os.environ.get("KYOJIN_MOE_KSPLIT", "0"))                 # per-bitrate k-split of the per-pair gather GEMV (rows = 1): 1 = the lower bitrates split less (same grid), 2 = the higher bitrates split more (taller grid); equal bytes per threadgroup; f32 partials added in another order
FINPAR = os.environ.get("KYOJIN_MOE_FINPAR", "0") == "1"               # combine launch (final): one simdgroup per routed pair, the pair terms added in order j by simdgroup 0 instead of one serial loop over the pairs; bit-identical
HOTK = int(os.environ.get("KYOJIN_GROUP_HOTK", "4")) if os.environ.get("KYOJIN_ROWS_GROUPED", "1") == "nax" else 0     # "nax": groups of >= HOTK rows through metal_nax


def _roof(cb: str) -> str:
    """KYOJIN_MOEROOF: the exact-mul1 and mul1fd decodes swap for their cheaper integer forms (same values bit for bit); other codebooks are untouched."""
    if ROOF and cb == "mul1":
        FD.hit("moeroof")                                             # counts codebook lookups (calls of cbf / cbu / cbgx), not launches
        if ROOF_DEC != 1:
            FD.hit("moeroof_dec")                                     # same: lookups that took the non-default r2 decode
        return "mul1r1" if ROOF_DEC == 1 else "mul1r2"
    if ROOF and cb == "mul1fd":
        FD.hit("moeroof")
        return "mul1fdr"
    return cb


SG_WS = 4                                                             # k-splits per threadgroup of the grouped GEMV (metal_src WS): the partials per k-block with SGRED
VIEWS = os.environ.get("KYOJIN_MOE_VIEWS", "1") == "1"           # bind a 256 KiB prefix view of each expert bank instead of the bank: MLX bills the command buffer for the data_size of every bound array, so a bank of 100s of MB forces a commit per kernel; the view is the same MTLBuffer (the kernel indexes the whole bank), so the results are bit-identical
VIEW_WORDS = 1 << 16                                                  # >= 16 KiB: a smaller array takes a slow inline path (measured 2x slower per kernel)
VIEW_MIN_BYTES = 64 << 20                                             # banks below this stay as they are (the bill stays under MLX's default buffer limit)


def bank_view(words: mx.array) -> mx.array:
    """Prefix view of `words` (a flat, row-contiguous bank): same buffer, data_size of VIEW_WORDS elements. Evaluated once; the bank must not be freed while it is used."""
    flat = words.reshape(-1)
    if flat.nbytes < VIEW_MIN_BYTES:
        return words
    v = flat[:VIEW_WORDS]
    mx.eval(v)
    return v


class Stack:
    """One projection of one MoE layer (or the 8 wo_a slices) in the ragged stacked layout, on device."""

    def __init__(self, words: mx.array, offset: mx.array, k: mx.array, suh: mx.array, svh: mx.array, kt: int, nt: int, cb: str = "mul1"):
        self.words, self.offset, self.k, self.suh, self.svh = words, offset, k, suh, svh
        self.kt, self.nt, self.cb = kt, nt, cb
        self.kin, self.nout, self.blocks = kt * 16, nt * 16, nt // 8
        self.experts = int(k.shape[0])
        self._view = None
        self.kh = np.array(k).astype(np.int64)                             # host copy of the per-expert bitrate (prefill plan)
        self.ks = tuple(sorted({int(v) for v in self.kh.tolist()}))
        assert suh.shape == (self.experts, self.kin) and svh.shape == (self.experts, self.nout), (suh.shape, svh.shape)

    @property
    def bound(self) -> mx.array:
        """What a kernel binds for the bank: the bank, or (KYOJIN_MOE_VIEWS) its prefix view, which MLX bills at a few KiB instead of the whole bank."""
        if not VIEWS:
            return self.words
        FD.hit("moe_views")                                           # counts binds of a bank (per kernel launch build), banks under 64 MiB included
        if self._view is None:
            self._view = bank_view(self.words)
        return self._view

    @classmethod
    def from_arrays(cls, d: dict, prefix: str, cb: str = "mul1") -> "Stack":
        kt, nt = (int(v) for v in np.array(d[f"{prefix}.shape"]).tolist())
        return cls(d[f"{prefix}.trellis"], d[f"{prefix}.offset"], d[f"{prefix}.k"], d[f"{prefix}.suh"], d[f"{prefix}.svh"], kt, nt, cb)

    def expert_trellis(self, e: int) -> mx.array:
        a, b = int(self.offset[e].item()), int(self.offset[e + 1].item())
        return self.words[a:b].view(mx.int16).reshape(self.kt, self.nt, 16 * int(self.k[e].item()))


_kc: dict = {}


def _kernel(key, **kw):
    """Shared kernel object per key. `source` may be a callable (called on a miss only). header: override the Metal header (dense decode cuts)."""
    k = _kc.get(key)
    if k is None:
        if callable(kw.get("source")):
            kw["source"] = kw["source"]()
        src = kw.get("source", "")
        hdr = kw.pop("header", None)
        if hdr is None:
            roof = "dec_mul1r" in src or "dec_mul1fdr" in src
            hdr = S.HEADER_MOE + (S.HEADER_MOE_ROOF if roof else "")
        k = _kc[key] = mx.fast.metal_kernel(header=hdr, **kw)
    return k


def pick_tgk(kt: int, blocks: int, pairs_z: int, target: int | None = None) -> int:
    """k-splits across threadgroups until blocks * tgk * pairs_z threadgroups reach ~target (KYOJIN_MOE_TGT, else 480)."""
    tgt0 = target
    target = target or TGT or 480
    tgk = 1
    while blocks * tgk * pairs_z < target and kt // (4 * tgk * 2) >= 8:
        tgk *= 2
    if TGT and not tgt0 and TGT != 480:                               # counts plan builds where the flag's target gives another tgk than 480
        t2 = 1
        while blocks * t2 * pairs_z < 480 and kt // (4 * t2 * 2) >= 8:
            t2 *= 2
        if t2 != tgk:
            FD.hit("moe_tgt")
    return tgk


def ksplit_plan(ks: tuple[int, ...], kt: int, tgk: int, mode: int) -> tuple[int, tuple[int, ...]]:
    """KYOJIN_MOE_KSPLIT: (grid k-blocks, active k-blocks per bitrate of `ks`). A threadgroup reads tiles * K words, so the k-blocks of an expert follow its bitrate:
    mode 1 = tgk * K / Kmax (the lower bitrates split less: fewer, equal-size threadgroups, the grid stays tgk tall), mode 2 = tgk * K / Kmin (the higher ones
    split more: the grid grows to the largest). At least 1, at most kt // 4 (a k-block is 4 simdgroup k-splits of >= 1 tile)."""
    cap = max(1, kt // 4)
    ref = max(ks) if mode == 1 else min(ks)
    act = tuple(max(1, min(cap, int(tgk * k / ref + 0.5))) for k in ks)
    return (tgk if mode == 1 else max(act)), act


class MoeGather:
    """Fused MoE decode for one layer. gate / up / down: Stack. wpre: routing weight applied before the down projection (DS4.1,
    needed with fq) instead of in the final sum (GLM). fq: DS4.1 fp8 e4m3 / ue8m0 block-32 fake quant of the down input."""

    def __init__(self, gate: Stack, up: Stack, down: Stack, topk: int, limit: float = 10.0, wpre: bool = False, fq: bool = False,
                 tgk_gu: int | None = None, tgk_dn: int | None = None, alpha: float = 1.0, beta: float = 0.0,
                 union: bool = False, pin_rows: int | None = None, grouped: bool | None = None, hotk: int | None = None):
        """union: pairs routed to the same expert share one decode of its trellis (multi-user decode, rows >= 2); per-pair
        results are bit-identical to the plain gather path. Measured on the Air (N = 4): a win only when more than about half of the
        routed pairs are shared (x1.1 at 50 %, x1.7 at 100 %), 10-20 % slower with independent users, so it is opt-in. pin_rows: choose the k-split (tgk) as if the call had that many rows,
        so a row's result does not depend on how many users share the step (multi-user: pin_rows=1 equals the solo call)."""
        self.alpha, self.beta = float(alpha), float(beta)
        self.union, self.pin_rows = union, pin_rows
        self.grouped = GROUPED if grouped is None else grouped
        self.hotk = HOTK if hotk is None else hotk                       # groups of >= hotk rows take the NAX GEMM (rounding-class), 0 = GEMV only (bitwise)
        if self.hotk and not NX.nax_available():
            self.hotk = 0
        assert gate.kin == up.kin and gate.nout == up.nout and (gate.kt, gate.nt) == (up.kt, up.nt)
        assert down.kin == gate.nout and down.nout == gate.kin and gate.cb == up.cb == down.cb
        assert not fq or wpre, "fake quant sits after the routing weight: use wpre"
        self.g, self.u, self.d, self.topk, self.limit, self.wpre, self.fq = gate, up, down, topk, float(limit), wpre, fq
        self.cb = gate.cb
        self.cbg = "mul1af" if AFFOLD and self.cb == "mul1" else self.cb
        self.fastdec = None                                           # per-instance pin of KYOJIN_MOE_FASTDEC (None = follow moe.FASTDEC at call time, see cbf)
        self.tgk_gu, self.tgk_dn = tgk_gu, tgk_dn
        self.ktop = topk                                              # pairs per row the k-split is chosen for (MoeSharedFused: the routed top-k, not routed + shared)
        self._mf = None                                               # fused path (moe_fused.MoeFused), built on first use
        self.ks_gu = tuple(sorted(set(gate.ks) | set(up.ks)))
        self.ks_dn = down.ks
        self._p: dict = {}
        self._pf = None
        self._kn: dict = {}                                     # (stage tag, variant) -> (kernel, grid/threadgroup per rows): the source text is built once

    @property
    def cbf(self) -> str:
        """Codebook of the per-pair gather GEMV (rows = 1 and ungrouped), read at call time so KYOJIN_MOE_FASTDEC switches on a loaded model
        (kyojin_mlx.flags): the fast decode costs the grouped (shared decode, several rows per weight) path +1..+20 pct, so it stays out of it."""
        fd = FASTDEC if self.fastdec is None else self.fastdec
        if fd and self.cb == "mul1":
            FD.hit("moe_fastdec")                                     # counts cbf lookups (one per per-pair gemv call)
        return _roof("mul1fd" if fd and self.cb == "mul1" else self.cbg)

    @property
    def cbu(self) -> str:
        """Codebook of the union GEMV (read at call time, KYOJIN_MOEROOF)."""
        return _roof(self.cb)

    @property
    def cbgx(self) -> str:
        """Codebook of the grouped GEMV (read at call time, KYOJIN_MOEROOF)."""
        return _roof(self.cbg)

    def _kern(self, tag, key, name, input_names, output_names, make_source):
        """Kernel of one stage for this layer, built once: the python side of a decode step must not regenerate Metal source text."""
        k = self._kn.get((tag, key))
        if k is None:
            k = self._kn[(tag, key)] = _kernel((tag, key, self.limit, self.wpre, self.fq, self.alpha, self.beta, self.cb), name=name, input_names=input_names,
                                               output_names=output_names, source=make_source)
        return k

    # ---- parameter arrays per (rows)
    def _params(self, rows: int):
        pkey = (rows, TGT, KSPLIT) if (TGT or KSPLIT) else rows                 # the flags switch at run time (kyojin_mlx.flags): their plans are cached apart
        if pkey not in self._p:
            npairs = rows * self.topk
            pp = (self.pin_rows or rows) * self.ktop
            tgu = self.tgk_gu or pick_tgk(self.g.kt, self.g.blocks, 2 * pp)
            tdn = self.tgk_dn or pick_tgk(self.d.kt, self.d.blocks, pp)
            u32 = lambda *v: mx.array(list(v), dtype=mx.uint32)
            ksp = KSPLIT in (1, 2) and not (rows > 1 and (self.grouped or self.union))        # the per-pair gather GEMV only (rows = 1, or ungrouped and not union)
            if ksp:
                FD.hit("moe_ksplit")                                  # counts plan builds (per rows), not calls
                tgu, a_gu = ksplit_plan(self.ks_gu, self.g.kt, tgu, KSPLIT)
                tdn, a_dn = ksplit_plan(self.ks_dn, self.d.kt, tdn, KSPLIT)
            else:
                a_gu = a_dn = ()
            self._p[pkey] = dict(npairs=npairs, tgu=tgu, tdn=tdn,
                                 rot_gu=u32(self.g.kin, self.topk), rot_dn=u32(self.d.kin, 1),
                                 gemv_gu=u32(self.g.kt, self.g.nt, tgu, npairs, *a_gu), gemv_dn=u32(self.d.kt, self.d.nt, tdn, npairs, *a_dn),
                                 plan=u32(npairs, rows),
                                 ggemv_gu=u32(self.g.kt, self.g.nt, tgu, npairs, rows), ggemv_dn=u32(self.d.kt, self.d.nt, tdn, npairs, rows),
                                 mid=u32(tgu, self.g.nout), fin=u32(tdn, self.d.nout, self.topk),
                                 mid_sg=u32(tgu * SG_WS, self.g.nout), fin_sg=u32(tdn * SG_WS, self.d.nout, self.topk))
            mx.eval(*[v for v in self._p[pkey].values() if isinstance(v, mx.array)])
        return self._p[pkey]

    # ---- stages (public: the tests check each against the per-expert path)
    def rot_gu(self, x: mx.array, ids: mx.array, fq8: bool = False, half: bool = False) -> mx.array:
        """fq8: x (fp16) is the input of the fp8 fake quant of the routed experts' input, applied inside the rot."""
        rows, kd = x.shape
        P = self._params(rows)
        ih = x.dtype == mx.float16
        kern = self._kern("rotg", (ih, fq8, half), f"kyojin_rotg2_{'h' if ih else 'f'}{'_fq' if fq8 else ''}{'_hx' if half else ''}", ["x", "ids", "suh0", "suh1", "params"], ["xh"],
                          lambda: (S.rot_gather_half_source if half else S.rot_gather_source)(2, ih, fq8))
        gt = P.get("hg_rot")
        if gt is None:
            gt = P["hg_rot"] = S.had_grid(kd // 128, 2 * P["npairs"])
        grid, tg = gt
        return kern(inputs=[x, ids, self.g.suh, self.u.suh, P["rot_gu"]], grid=grid, threadgroup=tg,
                    output_shapes=[(2 * P["npairs"], kd)], output_dtypes=[mx.float16 if half else mx.float32])[0]

    def gemv(self, st_list: tuple[Stack, ...], xh: mx.array, ids: mx.array, params: mx.array, tgk: int, ks: tuple[int, ...]) -> mx.array:
        nproj = len(st_list)
        s0, s1 = st_list[0], st_list[-1]
        npairs = int(ids.size)
        mm = 1 if npairs <= self.topk else (2 if npairs <= 2 * self.topk else (4 if npairs <= 4 * self.topk else 8))
        union = self.union and 1 < mm and npairs <= 8 * self.topk          # more than 8 rows: plain path (a leader holds at most 8 members)
        cbu, cbf, pf = self.cbu, self.cbf, ROOF and ROOF_PF
        pf2 = bool(pf and S.PF2)                                     # KYOJIN_GEMV_PF2: two tiles of the pair's trellis in flight
        if pf and not union:
            FD.hit("moeroof_pf")                                      # per-pair gather GEMV call with the prefetch kernel
        ksp = not union and int(params.size) == 4 + len(ks)                  # KYOJIN_MOE_KSPLIT: _params appended one active k-block count per bitrate
        key = ("ug", ks, nproj, cbu, mm) if union else ("gg", ks, nproj, cbf, pf, pf2) + (("ks",) if ksp else ())
        kern = self._kn.get(("gemv", key))
        if kern is None:
            names = ["xh", "ids", "tr0"] + (["tr1"] if nproj == 2 else []) + ["off0"] + (["off1"] if nproj == 2 else []) + \
                    ["kk0"] + (["kk1"] if nproj == 2 else []) + ["params"]
            kern = self._kn[("gemv", key)] = _kernel(key, name=f"kyojin_{'ug' if union else 'gg'}_{cbu if union else cbf}_{'_'.join(map(str, ks))}_p{nproj}" + (f"_m{mm}" if union else "") + ("_pf" if pf and not union else "") + ("_p2" if pf2 else "") + ("_ks" if ksp else ""),
                                                     input_names=names, output_names=["part"],
                                                     source=(lambda: S.union_gemv_source(ks, nproj, cbu, mm)) if union else (lambda: S.gather_gemv_source(ks, nproj, cbf, pf, ksp, pf2)))
        ins = [xh, ids, s0.bound] + ([s1.bound] if nproj == 2 else []) + [s0.offset] + ([s1.offset] if nproj == 2 else []) + \
              [s0.k] + ([s1.k] if nproj == 2 else []) + [params]
        return kern(inputs=ins, grid=(s0.blocks * 256, tgk, nproj * npairs), threadgroup=(256, 1, 1),
                    output_shapes=[(nproj * npairs, tgk, s0.nout)], output_dtypes=[mx.float32])[0]

    def plan(self, ids: mx.array, rows: int):
        """Group the pairs by expert on the GPU: (lead, cnt, mem), see metal_src.group_plan_source."""
        P = self._params(rows)
        hk, npairs = self.hotk, P["npairs"]
        cp = MOEGRP and not hk                                             # packed slot range (the hot groups keep the per-class layout)
        names = ["lead", "cnt_", "mem"] + (["hrow", "heids", "roff_gu", "roff_d", "hinfo"] if hk else [])
        kern = _kernel(("gplan", hk, cp), name=f"kyojin_gplan_{hk}" + ("_c" if cp else ""), input_names=["ids", "params"], output_names=names,
                       source=S.group_plan_source(hk, cp))
        shapes = [(sum(S.plan_caps(npairs, hk, cp)),), (npairs,), (npairs * rows,)]
        dts = [mx.uint32] * 3
        if hk:
            gh = S.hot_cap(npairs, hk)
            shapes += [(npairs,), (gh,), (2 * gh + 1,), (gh + 1,), (1,)]
            dts += [mx.uint32, mx.int32, mx.uint32, mx.uint32, mx.uint32]
        return kern(inputs=[ids, P["plan"]], grid=(256, 1, 1), threadgroup=(256, 1, 1), output_shapes=shapes, output_dtypes=dts)

    def ggemv(self, st_list: tuple[Stack, ...], xh: mx.array, ids: mx.array, plan, params: mx.array, tgk: int, ks: tuple[int, ...]) -> mx.array:
        nproj = len(st_list)
        s0, s1 = st_list[0], st_list[-1]
        npairs = int(ids.size)
        tail = [s0.bound] + ([s1.bound] if nproj == 2 else []) + [s0.offset] + ([s1.offset] if nproj == 2 else []) + [s0.k] + ([s1.k] if nproj == 2 else []) + [params]
        ent = ([k for k, _ in self._grp_passes(ks, nproj, self.hotk)], tail, (s0.blocks * 256, tgk, nproj * int(plan[0].size)), [(nproj * npairs, tgk, s0.nout)])
        return self._run_passes(ent, [xh, plan[0], plan[1], plan[2], ids])

    def _gstatic(self, rows: int, fq8: bool, ih: bool, half: bool = False, sg: bool = False):
        """Everything the grouped launches of `rows` rows need that does not depend on the call, built once (the python side of a decode step is on the critical path)."""
        P = self._params(rows)
        cp = MOEGRP
        sg = sg and not S.MMA_ROWS                                         # KYOJIN_MOE_SGRED, asked for by _call_grouped only (its mid / final take the TGK * 4 partials); the MMA body keeps its own reduction
        big = P["npairs"] > MAX_PAIRS                                      # pf_tail_grp: the wide plan kernel (128 rows x top-6 = 768 pairs)
        key = ("gstat", fq8, ih, VB_SPLIT_K, half, self.cbgx, cp, VIEWS, sg, big, S.GPF)
        ent = P.get(key)
        if ent is None:
            npairs = P["npairs"]
            caps = sum(S.plan_caps(npairs, 0, cp))
            pk = _kernel(("gplan", 0, cp, big), name="kyojin_gplan_0" + ("_c" if cp else "") + ("_big" if big else ""),
                         input_names=["ids", "params"], output_names=["lead", "cnt_", "mem"],
                         source=lambda: S.group_plan_source(0, cp, big))
            def gk(st_list, ks, tgk, pname):
                nproj = len(st_list)
                s0, s1 = st_list[0], st_list[-1]
                tail = [s0.bound] + ([s1.bound] if nproj == 2 else []) + [s0.offset] + ([s1.offset] if nproj == 2 else []) + [s0.k] + ([s1.k] if nproj == 2 else []) + [P[pname]]
                return [k for k, _ in self._grp_passes(ks, nproj, 0, half, sg)], tail, (s0.blocks * 256, tgk, nproj * caps), [(nproj * npairs, tgk * (SG_WS if sg else 1), s0.nout)]
            ent = P[key] = dict(sg=sg, plan=pk, caps=caps, shapes=[(caps,), (npairs,), (npairs * rows,)],
                                gu=gk((self.g, self.u), self.ks_gu, P["tgu"], "ggemv_gu"), dn=gk((self.d,), self.ks_dn, P["tdn"], "ggemv_dn"))
        return P, ent

    def _call_grouped(self, x: mx.array, ids: mx.array, wts: mx.array, rows: int, fq8: bool) -> mx.array:
        """Grouped path without the NAX GEMM, same launches and arithmetic as plan / ggemv, with the per-call python work cached."""
        half = S.half_on(rows) and not S.af_fold(self.cbg)                 # packed-fp16 row FMAs (KYOJIN_HALF_ROWS): x staged as half in the run's k order
        P, st = self._gstatic(rows, fq8, x.dtype == mx.float16, half, SGRED)
        if half:
            FD.hit("half_rows")                                       # grouped MoE call with half-staged x
        if st["sg"]:
            FD.hit("moe_sgred")
        if MOEGRP:
            FD.hit("moegrp")
        xh = self.rot_gu(x, ids, fq8, half)
        lead, cnt, mem = st["plan"](inputs=[ids, P["plan"]], grid=(256, 1, 1), threadgroup=(256, 1, 1), output_shapes=st["shapes"], output_dtypes=[mx.uint32] * 3)
        part = self._run_passes(st["gu"], [xh, lead, cnt, mem, ids])
        xh2 = self.mid(part, ids, wts, rows, half, st["sg"])
        part2 = self._run_passes(st["dn"], [xh2, lead, cnt, mem, ids])
        return self.final(part2, ids, wts, rows, st["sg"])

    @staticmethod
    def _run_passes(ent, head):
        kerns, tail, grid, shp = ent
        part = None
        for kern in kerns:
            part = kern(inputs=head + tail + ([part] if part is not None else []), grid=grid, threadgroup=(256, 1, 1), output_shapes=shp, output_dtypes=[mx.float32])[0]
        return part

    def _grp_passes(self, ks, nproj, hotk, half=False, sg=False):
        """Grouped GEMV kernels of one call: one kernel over all bitrates, or with VB_SPLIT_K one per bitrate class (the later ones carry the earlier
        partials through: same bits; a launch whose threadgroups mix K decode bodies runs slower on the M5)."""
        names = ["xh", "lead", "cnt_", "mem", "ids", "tr0"] + (["tr1"] if nproj == 2 else []) + ["off0"] + (["off1"] if nproj == 2 else []) + \
                ["kk0"] + (["kk1"] if nproj == 2 else []) + ["params"]
        cp = MOEGRP and not hotk                                          # packed slots, body by member count (KYOJIN_MOEGRP)
        def one(kc, prev):
            cbx = self.cbgx
            pf = S.GPF if not (half and S.MMA_ROWS and max(kc) >= S.MMA_ROWS) else 0     # the MMA body decodes each tile once into threadgroup memory: its loads are its own
            return _kernel(("grp", kc, nproj, cbx, hotk, S.MMA_ROWS, cp) + (("pv",) if prev else ()) + (("hx",) if half else ()) + (("sg",) if sg else ()) + (("pf", pf),),
                           name=f"kyojin_grp_{cbx}_{'_'.join(map(str, kc))}_p{nproj}_h{hotk}" + ("_c" if cp else "") + ("_pv" if prev else "") + ("_hx" if half else "") + ("_sg" if sg else "") + (f"_pf{pf}" if pf else ""),
                           input_names=names + (["pprev"] if prev else []), output_names=["part"],
                           source=lambda: S.grouped_gemv_source(kc, nproj, cbx, hotk, prev, half, S.MMA_ROWS, cp, sg, pf))
        if not VB_SPLIT_K or len(ks) < 2:
            return [(one(ks, False), ks)]
        return [(one((k,), i > 0), (k,)) for i, k in enumerate(ks)]

    def nax_cfg(self, rows: int):
        b, nmb, bns, bks = NX._BLOCK_CFG[16 if rows <= 16 else 32]
        return (b, nmb, bns, HOT_BKS or bks)

    def rot_hot(self, x: mx.array, ids: mx.array, hrow: mx.array, hinfo: mx.array, rows: int):
        P = self._params(rows)
        ih = x.dtype == mx.float16
        kern = _kernel(("roth", ih), name=f"kyojin_roth_{'h' if ih else 'f'}", input_names=["x", "ids", "suh0", "suh1", "hrow", "hinfo", "params"],
                       output_names=["xh", "A"], source=S.rot_hot_source(ih))
        grid, tg = S.had_grid(self.g.kin // 128, 2 * P["npairs"])
        return kern(inputs=[x, ids, self.g.suh, self.u.suh, hrow, hinfo, P["rot_gu"]], grid=grid, threadgroup=tg,
                    output_shapes=[(2 * P["npairs"], self.g.kin)] * 2, output_dtypes=[mx.float32, mx.float16])

    def mid_hot(self, part: mx.array, yn: mx.array, ids: mx.array, wts: mx.array, hrow: mx.array, hinfo: mx.array, rows: int):
        P = self._params(rows)
        kern = _kernel(("midh", self.limit, self.wpre, self.fq, self.alpha, self.beta), name=f"kyojin_midh_{int(self.wpre)}{int(self.fq)}_{len(_kc)}",
                       input_names=["part", "yn", "ids", "wts", "svh0", "svh1", "suhd", "hrow", "hinfo", "params"], output_names=["xh2", "A2"],
                       source=S.mid_hot_source(self.limit, self.wpre, self.fq, self.alpha, self.beta))
        grid, tg = S.had_grid(self.g.nout // 128, P["npairs"])
        return kern(inputs=[part, yn, ids, wts, self.g.svh, self.u.svh, self.d.suh, hrow, hinfo, P["mid"]], grid=grid, threadgroup=tg,
                    output_shapes=[(P["npairs"], self.g.nout)] * 2, output_dtypes=[mx.float32, mx.float16])

    def final_hot(self, part: mx.array, yd: mx.array, ids: mx.array, wts: mx.array, hrow: mx.array, rows: int) -> mx.array:
        P = self._params(rows)
        kern = _kernel(("finh", not self.wpre), name=f"kyojin_finh_{int(not self.wpre)}", input_names=["part", "yd", "ids", "wts", "svh", "hrow", "params"],
                       output_names=["y"], source=S.final_hot_source(not self.wpre))
        grid, tg = S.had_grid(self.d.blocks, rows)
        return kern(inputs=[part, yd, ids, wts, self.d.svh, hrow, P["fin"]], grid=grid, threadgroup=tg,
                    output_shapes=[(rows, self.d.nout)], output_dtypes=[mx.float32])[0]

    def _hot_gemm(self, stacks, heids, roff, A, rows_out, rows, cfg, ks):
        """NAX GEMM of the hot slots. Mixed-K stacks: one launch per bitrate class (a launch whose threadgroups mix K decode bodies runs ~15-20 % slower on the
        M5 even with the same compiled kernel); a tile of another class is zero in the first pass and copied through by the next ones: same bits as one launch."""
        if not VB_SPLIT_K or len(ks) < 2 or rows < self.hotk:                    # fewer rows than hotk: no hot group can exist, one (empty) launch
            return NX.expert_gemm(stacks, heids, roff, A, rows_out, rows, cfg, out_half=False)
        y = None
        for k in ks:
            y = NX.expert_gemm(stacks, heids, roff, A, rows_out, rows, cfg, out_half=False, ks=(k,), prev=y)
        return y

    def _call_hot(self, x: mx.array, ids: mx.array, wts: mx.array, rows: int) -> mx.array:
        """Grouped path with the NAX GEMM for the groups of hotk or more rows (rounding-class: half activations, tensor-unit accumulation)."""
        P, npairs, cfg = self._params(rows), rows * self.topk, self.nax_cfg(rows)
        plan = self.plan(ids, rows)
        hrow, heids, roff_gu, roff_d, hinfo = plan[3:]
        xh, A = self.rot_hot(x, ids, hrow, hinfo, rows)
        part = self.ggemv((self.g, self.u), xh, ids, plan, P["ggemv_gu"], P["tgu"], self.ks_gu)
        yn = self._hot_gemm((self.g, self.u), heids, roff_gu, A, 2 * npairs, rows, cfg, self.ks_gu)
        xh2, A2 = self.mid_hot(part, yn, ids, wts, hrow, hinfo, rows)
        part2 = self.ggemv((self.d,), xh2, ids, plan, P["ggemv_dn"], P["tdn"], self.ks_dn)
        yd = self._hot_gemm((self.d,), heids, roff_d, A2, npairs, rows, cfg, self.ks_dn)
        return self.final_hot(part2, yd, ids, wts, hrow, rows)

    def gemv_gu(self, xh: mx.array, ids: mx.array, rows: int) -> mx.array:
        P = self._params(rows)
        return self.gemv((self.g, self.u), xh, ids, P["gemv_gu"], P["tgu"], self.ks_gu)

    def mid(self, part: mx.array, ids: mx.array, wts: mx.array, rows: int, half: bool = False, sg: bool = False) -> mx.array:
        P = self._params(rows)
        kern = self._kn.get(("mid", half))
        if kern is None:
            kern = self._kn[("mid", half)] = _kernel(("mid", self.limit, self.wpre, self.fq, self.alpha, self.beta, half), name=f"kyojin_mid_{int(self.wpre)}{int(self.fq)}{'_hx' if half else ''}_{len(_kc)}",
                                                  input_names=["part", "ids", "wts", "svh0", "svh1", "suhd", "params"], output_names=["xh2"],
                                                  source=lambda: (S.mid_half_source if half else S.mid_source)(self.limit, self.wpre, self.fq, self.alpha, self.beta))
        gt = P.get("hg_mid")
        if gt is None:
            gt = P["hg_mid"] = S.had_grid(self.g.nout // 128, P["npairs"])
        grid, tg = gt
        return kern(inputs=[part, ids, wts, self.g.svh, self.u.svh, self.d.suh, P["mid_sg" if sg else "mid"]], grid=grid, threadgroup=tg,
                    output_shapes=[(P["npairs"], self.g.nout)], output_dtypes=[mx.float16 if half else mx.float32])[0]

    def gemv_dn(self, xh2: mx.array, ids: mx.array, rows: int) -> mx.array:
        P = self._params(rows)
        return self.gemv((self.d,), xh2, ids, P["gemv_dn"], P["tdn"], self.ks_dn)

    def final_par(self, part: mx.array, ids: mx.array, wts: mx.array, rows: int, bf16_last: bool = False) -> mx.array | None:
        """KYOJIN_MOE_FINPAR: the combine launch with the pairs side by side (None: more than 32 pairs per row, the serial launch runs)."""
        nj = self.topk
        if nj > 32:
            return None
        P = self._params(rows)
        FD.hit("moe_finpar")                                          # combine launches (routed final and sumoe final_sh)
        kern = self._kern("finp", (nj, bf16_last), f"kyojin_finp_{int(not self.wpre)}_{nj}{'_bf' if bf16_last else ''}", ["part", "ids", "wts", "svh", "params"], ["y"],
                          lambda: S.final_par_source(not self.wpre, nj, bf16_last))
        grid, tg = S.final_par_grid(self.d.blocks, rows, nj)
        return kern(inputs=[part, ids, wts, self.d.svh, P["fin"]], grid=grid, threadgroup=tg,
                    output_shapes=[(rows, self.d.nout)], output_dtypes=[mx.float32])[0]

    def final(self, part: mx.array, ids: mx.array, wts: mx.array, rows: int, sg: bool = False) -> mx.array:
        if FINPAR and not sg:                                   # moe_finpar reads the plain `fin` params; moe_sgred (sg) keeps its own
            y = self.final_par(part, ids, wts, rows)
            if y is not None:
                return y
        P = self._params(rows)
        kern = self._kn.get(("fing", 0))
        if kern is None:
            kern = self._kn[("fing", 0)] = _kernel(("fing", not self.wpre), name=f"kyojin_fing_{int(not self.wpre)}", input_names=["part", "ids", "wts", "svh", "params"],
                                                   output_names=["y"], source=lambda: S.final_gather_source(not self.wpre))
        gt = P.get("hg_fin")
        if gt is None:
            gt = P["hg_fin"] = S.had_grid(self.d.blocks, rows)
        grid, tg = gt
        return kern(inputs=[part, ids, wts, self.d.svh, P["fin_sg" if sg else "fin"]], grid=grid, threadgroup=tg,
                    output_shapes=[(rows, self.d.nout)], output_dtypes=[mx.float32])[0]

    def prefill(self, x: mx.array, ids: mx.array, wts: mx.array) -> mx.array:
        """Many rows (prompt chunk): S3 path of kyojin_mlx.moe_prefill, same contract as __call__."""
        if self._pf is None:
            from .moe_prefill import MoePrefill
            self._pf = MoePrefill(self)
        return self._pf(x, ids, wts)

    def __call__(self, x: mx.array, ids: mx.array, wts: mx.array, fq8: bool = False, max_pairs: int | None = None) -> mx.array:
        """max_pairs: pair cap of this call (default MAX_PAIRS). pf_tail_grp passes TAIL_PAIRS so one call serves a whole CED tail (rows x top-k)."""
        rows = x.shape[0]
        assert rows * self.topk <= (MAX_PAIRS if max_pairs is None else max_pairs) and ids.shape == (rows, self.topk) and wts.shape == (rows, self.topk)
        ids, wts = ids.reshape(-1).astype(mx.int32), wts.reshape(-1).astype(mx.float32)
        from . import moe_fused as MF
        if MF.MOEFUSE and not getattr(self, "_is_fused_inner", False) and not self.hotk and self.cbg == self.cb:
            FD.hit("moefuse")
            if self._mf is None:
                self._mf = MF.MoeFused(self)
            return self._mf(x, ids, wts, fq8)
        if self.grouped and rows > 1 and self.hotk:
            return self._call_hot(x, ids, wts, rows)
        if self.grouped and rows > 1 and not (UGROUPED and rows <= UGROUPED):
            return self._call_grouped(x, ids, wts, rows, fq8)
        if self.grouped and UGROUPED and 1 < rows <= UGROUPED:
            FD.hit("rows_ugrouped")                                   # routed MoE calls that took the per-pair split-K gather for a rows pass
        xh = self.rot_gu(x, ids, fq8)
        part = self.gemv_gu(xh, ids, rows)
        xh2 = self.mid(part, ids, wts, rows)
        part2 = self.gemv_dn(xh2, ids, rows)
        return self.final(part2, ids, wts, rows)


def swiglu_ref(g: mx.array, u: mx.array, limit: float, alpha: float = 1.0, beta: float = 0.0) -> mx.array:
    """fp32, op by op like mlx-vlm's _swiglu (min / clip / silu * up); alpha / beta give swigluoai."""
    g, u = mx.minimum(g, limit), mx.clip(u, -limit, limit)
    if alpha == 1.0 and beta == 0.0:
        return nn.silu(g) * u
    return g * mx.sigmoid(g * alpha) * (u + beta)


class MoeUnfused:
    """Reference / baseline: per routed expert and projection a `Mul1Gemv` (rot, GEMV, finalize; f32 out), swiglu with mlx ops,
    host read of the router ids. Same tgk as the fused path gives bit-identical partials."""

    def __init__(self, gate: Stack, up: Stack, down: Stack, topk: int, limit: float = 10.0, wpre: bool = False, fq: bool = False,
                 tgk_gu: int | None = None, tgk_dn: int | None = None, alpha: float = 1.0, beta: float = 0.0):
        self.g, self.u, self.d, self.topk, self.limit, self.wpre, self.fq = gate, up, down, topk, float(limit), wpre, fq
        self.tgk_gu, self.tgk_dn = tgk_gu, tgk_dn
        self.alpha, self.beta = float(alpha), float(beta)
        self._lin: dict = {}

    def _gemv(self, st: Stack, name: str, e: int, tgk: int | None) -> KN.Mul1Gemv:
        key = (name, e)
        if key not in self._lin:
            self._lin[key] = KN.Mul1Gemv(st.expert_trellis(e), np.array(st.suh[e]), np.array(st.svh[e]), rows=1, out_dtype=mx.float32,
                                         tgk=tgk, cb=st.cb)
        return self._lin[key]

    def __call__(self, x: mx.array, ids: mx.array, wts: mx.array, ids_host=None) -> mx.array:
        rows = x.shape[0]
        npairs = rows * self.topk
        tgu = self.tgk_gu or pick_tgk(self.g.kt, self.g.blocks, 2 * npairs)
        tdn = self.tgk_dn or pick_tgk(self.d.kt, self.d.blocks, npairs)
        ids_h = ids_host if ids_host is not None else np.array(ids).reshape(rows, self.topk).tolist()
        outs = []
        for r in range(rows):
            xr = x[r:r + 1]
            acc = None
            for j in range(self.topk):
                e = ids_h[r][j]
                g = self._gemv(self.g, "g", e, tgu)(xr)
                u = self._gemv(self.u, "u", e, tgu)(xr)
                a = swiglu_ref(g, u, self.limit, self.alpha, self.beta)
                w = wts[r, j]
                if self.wpre:
                    a = a * w
                    if self.fq:
                        from . import archs as _AR
                        a = _AR.fq(a.astype(mx.float16)).astype(mx.float32)
                y = self._gemv(self.d, "d", e, tdn)(a)
                if not self.wpre:
                    y = y * w
                acc = y if acc is None else acc + y
            outs.append(acc)
        return mx.concatenate(outs, axis=0)
