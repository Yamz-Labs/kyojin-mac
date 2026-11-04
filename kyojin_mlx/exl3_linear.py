# SPDX-License-Identifier: Apache-2.0
"""Exl3Linear: an EXL3 trellis linear as an MLX module, plus a loader that patches mlx-lm model linears from an EXL3
pack (model.safetensors + config.json quantization_config). Decode / few rows use the Metal GEMV (up to MAX_ROWS rows per
launch); many rows (prefill) decode the trellis to fp16 once per call and use an MLX matmul between the two Hadamards."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from . import fired as FD, fqfuse as FQF
from . import dense_pf as DP, kernels as KN, metal_nax as NX, tuning as TN, wcache as WC
from . import mma_gemv as MM, metal_nax_sh as NSH

GEMV_MAX_ROWS = int(os.environ.get("KYOJIN_GEMV_MAX_ROWS", "4"))     # rows above this go through decode + matmul
LF_MASK = int(os.environ.get("KYOJIN_LF_MASK", "7"))         # bit 0 single linears, bit 1 same-input groups, bit 2 grouped wo_a
LINEAR_FUSED = os.environ.get("KYOJIN_LINEAR_FUSED", "1") != "0"     # decode linear = ONE launch (rot + GEMV + finalize, kernels.Mul1Gemv.fused), bit-identical to the three
DENSE_FUSED = os.environ.get("KYOJIN_DENSE_FUSED", "auto")           # "1" force, "0" off, "auto" = on when the GPU has neural accelerators
DENSE_FUSED_MAX = int(os.environ.get("KYOJIN_DENSE_FUSED_MAX", "1024"))   # rows up to here: fused trellis decode + NAX matmul (metal_nax; 5-15 % faster than decode + matmul up to 1024 rows, slower above); above: decode + MLX matmul
ROWS_NAX = os.environ.get("KYOJIN_ROWS_NAX", "1") != "0"             # ON by default (owner 24/10; KYOJIN_ROWS_NAX=0 / rows.enable(nax=False) restore the solo bits): rows above the GEMV chunk run the fused NAX kernel, whose cost is flat in rows
DENSE_SH = os.environ.get("KYOJIN_DENSE_SH", "1") != "0"                # fused prefill rows <= fused_max: decode each weight tile once per threadgroup into threadgroup memory (metal_nax_sh), bit-identical to the per-simdgroup decode
FIN_FUSED = os.environ.get("KYOJIN_FIN_FUSED", "1") != "0"           # dense fused prefill: H128 / svh / cast folded into the NAX epilogue (KYOJIN_FIN_FUSED=0: separate finalize launch)
NAX_SMALL_WEIGHTS = 4_000_000       # a linear with fewer weights (kin * nout) needs two GEMV chunks before NAX pays (wkv 2.6 M ties at 8 rows, wq_a 6.6 M wins)
MMA_ROWS = int(os.environ.get("KYOJIN_MMA_ROWS", "-1"))              # rows from here up to MMA_MAX: partial GEMV on simdgroup_matrix (mma_gemv); 0 = off; -1 = auto: the rows above the GEMV chunk when ROWS_NAX is on (the tolerance class), else off
MMA_MAX = int(os.environ["KYOJIN_MMA_MAX"]) if "KYOJIN_MMA_MAX" in os.environ else None     # last row count of the MMA GEMV (one 8-row fragment tile: 9+ rows pay a second pass over the trellis, R=12: 246 us vs 203 NAX); None = tuning entry "mma_max", else the chip default (tuning.chip)
NAX_KS = int(os.environ.get("KYOJIN_NAX_KS", "-1"))       # k-split of the fused NAX matmul: -1 = auto (2, or 4 under nax_ks_few threadgroups; only below nax_ks_tgs threadgroups), 0/1 = off; tuning.py "nax_ks" per shape and row count overrides
NAX_KS_TGS = int(os.environ["KYOJIN_NAX_KS_TGS"]) if "KYOJIN_NAX_KS_TGS" in os.environ else None          # None = chip default (tuning.chip: 128 on the Air, scaled by core count elsewhere)
NAX_KS_WEIGHTS = int(os.environ["KYOJIN_NAX_KS_WEIGHTS"]) if "KYOJIN_NAX_KS_WEIGHTS" in os.environ else None       # a smaller linear is launch-bound there: the third launch of the split costs more than it saves (DS4.1 slice 1280x4096: 30 -> 64 us at 9 rows); None = chip default (16M on the Air)
ROW_SPLIT_MAX = 0      # multi-user decode (rows.enable sets it): GEMV_MAX_ROWS < rows <= ROW_SPLIT_MAX run as GEMV launches of <= MAX_ROWS rows (a row keeps its solo bits)
RG_HALF = os.environ.get("KYOJIN_RG_HALF", "0") not in ("0", "")   # flag rg_half: the many-rows GEMV above GEMV_MAX_ROWS runs packed-fp16 row FMAs (the class the fused launch takes from HALF_ROWS rows; read per call)


class Exl3Linear(nn.Module):
    """y = layer(x) for x (..., in_features); out dtype fp16. Arrays are private (not module parameters)."""

    def __init__(self, trellis: mx.array, suh: mx.array, svh: mx.array, bias: mx.array | None = None, cb: str = "mcg",
                 out_dtype: mx.Dtype = mx.float16, tune: dict | None = None):
        super().__init__()
        kt, nt, w = trellis.shape
        self.in_features, self.out_features, self.K, self.cb = kt * 16, nt * 16, w // 16, cb
        self.tune = TN.lookup(self.in_features, self.out_features, self.K) if tune is None else tune      # per-(device, shape) choices (tuning.py); {} = the constants
        self.fused_max = self.tune.get("fused_max", DENSE_FUSED_MAX if "KYOJIN_DENSE_FUSED_MAX" in os.environ else TN.chip().get("dense_fused_max", DENSE_FUSED_MAX))
        self.out_dtype = out_dtype
        self._kt, self._nt = kt, nt
        self._g = {1: KN.Mul1Gemv(trellis, suh, svh, rows=1, out_dtype=out_dtype, cb=cb, tgk=self.tune.get("tgk"))}
        base = self._g[1]
        self._words, self._suh, self._svh = base.words, base.suh, base.svh
        self._bias = bias.astype(out_dtype) if bias is not None else None
        self._fp = {}
        self._nax = None                                    # (SingleStack, eids) on first use
        self._nax_ks: dict = {}                             # rows -> (rows array, tile, finalize params, grid) of the k-split path
        self._ks = 1
        self._nax_rows: dict = {}                           # rows -> (rowoff, cfg)
        self._rg = {}
        self._weff = None                                   # folded fp16 weight (dense_pf.install): prefill rows >= dense_pf.MIN_ROWS run one fp16 matmul

    def _gemv(self, r: int) -> KN.Mul1Gemv:
        if r not in self._g:
            import copy
            g = copy.copy(self._g[1])
            g.rows = r
            g.dcb = KN.S.dense_cb(g.cb)                      # KYOJIN_DENSE_DEC may have moved since load
            g.fused_init()                                  # own counters for the fused kernel (the copy must not share the base's)
            g.fparams = mx.array([g.tgk, g.n, r], dtype=mx.uint32)
            g.kern = KN._kern(("gemv", g.K, r, g.prefetch, g.dcb),
                              name=f"kyojin_{g.dcb}_gemv_k{g.K}_m{r}_p{int(g.prefetch)}",
                              input_names=["xh", "trellis", "params"], output_names=["part"], header=KN.S.dense_header(g.dcb),
                              source=KN.S.gemv_source(g.K, r, g.prefetch, g.dcb))
            g._kerns[(g.dcb, r)] = g.kern
            self._g[r] = g
        return self._g[r]

    def _nax_rows_ok(self, rows: int) -> bool:
        """Rows in the GEMV chunk band (GEMV_MAX_ROWS < rows <= ROW_SPLIT_MAX) that go to the fused NAX kernel when ROWS_NAX: the GEMV cost steps up
        by one launch per MAX_ROWS rows while the NAX cost stays flat, so NAX pays from the first row of a second chunk (from the third chunk
        for the small linears). NAX rounds like fp16 matmul: a row differs from its solo (GEMV) bits at 1e-3 relative."""
        if not ROWS_NAX or not self._fused_ok(rows):
            return False
        return rows > KN.MAX_ROWS * (2 if self.in_features * self.out_features < NAX_SMALL_WEIGHTS else 1)

    def _mma_on(self, rows: int) -> bool:
        lo = (GEMV_MAX_ROWS + 1 if ROWS_NAX else 0) if MMA_ROWS < 0 else MMA_ROWS
        hi = MMA_MAX if MMA_MAX is not None else self.tune.get("mma_max", TN.chip()["mma_max"])
        return bool(lo) and lo <= rows <= hi and self.K in MM.KS

    def _ks_for(self, rows: int) -> int:
        """k-split of the fused NAX matmul for this shape and row count (1 = none)."""
        ks = TN.ks_for(self.tune, rows)
        if ks is None:
            ks = NAX_KS
            if ks < 0 and (not ROWS_NAX or rows > 16 or self.in_features * self.out_features < (NAX_KS_WEIGHTS or TN.chip()["nax_ks_weights"])):      # auto: the verify band only
                ks = 1
            elif ks < 0:                                       # auto: too few 128-column threadgroups to feed the cores (the head has thousands: no split)
                tgs = self._nt // 8
                c = TN.chip()
                ks = 1 if tgs >= (NAX_KS_TGS or c["nax_ks_tgs"]) else 4 if tgs < c["nax_ks_few"] else 2
        return ks if ks > 1 and self.out_features % 128 == 0 and self._kt % (ks * 4) == 0 else 1

    def _fused_ok(self, rows: int) -> bool:
        if rows > self.fused_max or DENSE_FUSED == "0" or self._nt % 2 or self._kt % 2:
            return False
        return NX.site_on(DENSE_FUSED)

    def _inner_fused(self, x2: mx.array, fin: bool = False, fq8: bool = False) -> mx.array:
        """Rotated input with the k permutation, then the trellis decoded in registers inside the NAX matmul (no fp16 weight in memory)."""
        rows = x2.shape[0]
        if self._nax is None:
            self._nax = (NX.SingleStack(self._words, self.K, self._kt, self._nt, self.cb), mx.array([0], dtype=mx.int32))
        st, eids = self._nax
        ent = self._nax_rows.get(rows)
        if ent is None:
            ent = self._nax_rows[rows] = (mx.array([0, rows], dtype=mx.uint32), TN.cfg_for(self.tune, rows) or NX.pick_cfg([rows]),
                                          mx.array([1, self.out_features, rows], dtype=mx.uint32), KN.S.had_grid(self.out_features // 128, rows))
        xh = KN.rot(x2, self._suh, mx.float16, fq8=fq8, perm=True)
        if fin:
            return NX.expert_gemm((st,), eids, ent[0], xh, rows, rows, ent[1], fin=(self._svh, self.out_dtype))
        return NX.expert_gemm((st,), eids, ent[0], xh, rows, rows, ent[1]).reshape(1, rows, self.out_features)
    def _inner_sh(self, x2: mx.array, xh: mx.array | None = None, fq8: bool = False) -> mx.array:
        rows = (x2 if xh is None else xh).shape[0]
        if self._nax is None:
            self._nax = (NX.SingleStack(self._words, self.K, self._kt, self._nt, self.cb), mx.array([0], dtype=mx.int32))
        ent = self._nax_rows.get(rows)
        if ent is None:
            ent = self._nax_rows[rows] = (mx.array([0, rows], dtype=mx.uint32), TN.cfg_for(self.tune, rows) or NX.pick_cfg([rows]),
                                          mx.array([1, self.out_features, rows], dtype=mx.uint32), KN.S.had_grid(self.out_features // 128, rows))
        if xh is None:
            xh = KN.rot(x2, self._suh, mx.float16, fq8=fq8, perm=True)
        return NSH.dense_sh(self._nax[0], xh, rows, (64, 1), True, "", 2, False, 4).reshape(1, rows, self.out_features)

    def _rg_half(self, rows: int) -> bool:
        """Flag rg_half: the many-rows GEMV (rows > GEMV_MAX_ROWS, the 5..8 row block of a verify pass) FMA in packed fp16, the class the <= MAX_ROWS
        fused launch takes from HALF_ROWS rows on. Same gate as `_half_on`, so the two paths agree on the class of a row; fp32 rows otherwise (today)."""
        return bool(RG_HALF) and rows <= KN.fuse_rows() and KN.S.half_on(rows) and not KN.S.af_fold(KN.S.dense_cb(self.cb))

    def _many(self, x2: mx.array, half: bool = False) -> mx.array:
        """More than MAX_ROWS rows of a multi-user decode step (a row keeps its solo bits): rot, ONE GEMV launch (row groups of MAX_ROWS on the
        grid), finalize. Three launches whatever the number of rows. `half`: packed-fp16 row FMAs (flag rg_half), so `rot` must hand the kernel the
        fp16 it reads as half4."""
        rows = x2.shape[0]
        ent = self._rg.get(rows)
        if ent is None:
            import copy
            g = copy.copy(self._g[1])
            g.rows = rows
            g.fparams = mx.array([g.tgk, g.n, rows], dtype=mx.uint32)
            ent = self._rg[rows] = (g, mx.array([g.kt, g.nt, g.tgk, rows, 1], dtype=mx.uint32))
        g, params = ent
        if half:
            FD.hit("rg_half")
        mr = -(-rows // -(-rows // KN.RG_ROWS))
        return g.finalize(g.partial_rows(KN.rot(x2, g.suh, mx.float16 if half else mx.float32), rows, params, mr, half=half))     # equal row groups (12 rows: 2 x 6, not 8 + 4 padded): no dummy rows in the FMA chain, same bits per row
    def _mma(self, x2: mx.array) -> mx.array:
        """4..16 rows: rot, partial GEMV on simdgroup_matrix fragments (the trellis is decoded straight into them), finalize. Rounds like the GEMV up to fp32 summation order."""
        g = self._g[1]
        return MM.finalize_rows(g, MM.partial_mma(g, KN.rot(x2, g.suh), x2.shape[0]), x2.shape[0])

    def _ksplit(self, x2: mx.array, fq8: bool = False) -> mx.array:
        """Fused NAX matmul split along k over KS threadgroup rows (fp32 partials), summed by the finalize: the few-row verify band has too few
        threadgroups (down: 20) to feed every core, and the split also shortens the serial decode chain of each one."""
        rows, KS = x2.shape[0], self._ks
        if self._nax is None:
            self._nax = (NX.SingleStack(self._words, self.K, self._kt, self._nt, self.cb), mx.array([0], dtype=mx.int32))
        st, eids = self._nax
        ent = self._nax_ks.get(rows)
        if ent is None:
            ent = self._nax_ks[rows] = (mx.array([0, rows], dtype=mx.uint32), TN.cfg_for(self.tune, rows) or NX.pick_cfg([rows]),
                                        mx.array([KS, self.out_features, rows], dtype=mx.uint32), KN.S.had_grid(self.out_features // 128, rows))
        xh = KN.rot(x2, self._suh, mx.float16, fq8=fq8, perm=True)
        part = NX.expert_gemm((st,), eids, ent[0], xh, rows, rows, ent[1], ksplit=KS).reshape(KS, rows, self.out_features)
        return KN.fin_kernel(self.out_dtype, mx.float32)(
            inputs=[part, self._svh, ent[2]], template=[("OT", self.out_dtype)], grid=ent[3][0], threadgroup=ent[3][1],
            output_shapes=[(rows, self.out_features)], output_dtypes=[self.out_dtype])[0]

    _defer = False      # layer_pf: the caller finalizes (the call returns the fp16 matmul product [.., n] before the finalize)

    def dense_mode(self, rows: int) -> str | None:
        """How `_dense` runs these rows: "sh" (decode-once fused NAX kernel, input in the k permutation), "mm" (decode + MLX matmul), None (another path: k-split, finalize in the
        NAX epilogue, plain fused). The fused MLP middle (mlp_pf) needs one of the first two."""
        if self._bias is not None or FQF._pairs or self._weff is not None or self.out_dtype != mx.float16 or rows <= max(GEMV_MAX_ROWS, ROW_SPLIT_MAX) or self._mma_on(rows):
            return None
        if self._fused_ok(rows):
            return "sh" if self._ks_for(rows) == 1 and DENSE_SH and self._kt % 8 == 0 and NX.site_on(DENSE_FUSED) else None
        return "mm"

    def _dense(self, x2: mx.array | None, xh: mx.array | None = None, nofin: bool = False, fq8: bool = False) -> mx.array:
        """x2 [rows, k] -> y; xh: the already rotated input (mlp_pf), in the layout of this path (dense_mode); nofin: return the fp16 matmul product [1, rows, n] before the finalize;
        fq8: x2 (fp16) is the input of an fp8 fake quant, run inside the rot."""
        rows = (x2 if xh is None else xh).shape[0]
        if self._fused_ok(rows):
            self._ks = self._ks_for(rows)
            if self._ks > 1:
                return self._ksplit(x2, fq8).reshape(rows, self.out_features)
            if DENSE_SH and self._kt % 8 == 0 and NX.site_on(DENSE_FUSED):
                inner = self._inner_sh(x2, xh, fq8)
                fp, (grid, tg) = self._nax_rows[rows][2:4]
            elif FIN_FUSED and self.out_features % 128 == 0 and self._nt % 8 == 0:           # finalize inside the NAX epilogue: one launch fewer, same bits
                return self._inner_fused(x2, fin=True, fq8=fq8).reshape(rows, self.out_features)
            else:
                inner = self._inner_fused(x2, fq8=fq8)
                fp, (grid, tg) = self._nax_rows[rows][2:4]
        else:
            if xh is None:
                xh = KN.rot(x2, self._suh, mx.float16, fq8=fq8)
            wr = WC.decode_wr(self)                                                          # (n, k); chunks of one prompt share it inside wcache.prefill_scope
            inner = (xh @ wr.T).reshape(1, rows, self.out_features)                       # fp16 partial, 1 k-block
            fp = mx.array([1, self.out_features, rows], dtype=mx.uint32)
            grid, tg = KN.S.had_grid(self.out_features // 128, rows)
        if nofin:
            return inner
        return KN.fin_kernel(self.out_dtype, mx.float16)(
            inputs=[inner, self._svh, fp], template=[("OT", self.out_dtype)], grid=grid, threadgroup=tg,
            output_shapes=[(rows, self.out_features)], output_dtypes=[self.out_dtype])[0]

    def __call__(self, x: mx.array, fq8: bool = False) -> mx.array:
        """fq8: x is the INPUT of an fp8 fake quant (DS4.1 wo_a); prompt rows of the dense path run it inside the rot, other paths run it first."""
        shape = x.shape
        if fq8 or (FQF._pairs and x.size > max(GEMV_MAX_ROWS, ROW_SPLIT_MAX) * shape[-1]):
            rows = x.size // shape[-1]
            ok = x.dtype == mx.float16 and shape[-1] % 128 == 0 and rows > max(GEMV_MAX_ROWS, ROW_SPLIT_MAX) and self._weff is None and not self._mma_on(rows)
            if fq8 and not ok:
                from . import archs as _AR
                x, fq8 = _AR.fq(x), False
            elif not fq8 and ok:                                    # prompt rows of x = fq(src): the round trip inside the rot of the dense path
                src = FQF.source(x)
                if src is not None:
                    x, fq8 = src, True
        if len(shape) == 2 and not FQF._pairs and self._bias is None and LINEAR_FUSED and LF_MASK & 1 and MMA_ROWS != 1 and shape[0] <= GEMV_MAX_ROWS \
                and shape[0] <= KN.MAX_ROWS and (x.dtype == mx.float16 or x.dtype == mx.float32):
            # decode fast path (no fp8 fold pending, plain rows <= the GEMV chunk): the same call as the general path below with the host checks done once
            g = self._g.get(shape[0]) or self._gemv(shape[0])
            if g.can_fuse(shape[0]):
                return g.fused(x)
        nr = x.size // shape[-1]
        vfuse = KN.MAX_ROWS < nr <= KN.fuse_rows() and LINEAR_FUSED and LF_MASK & 1 and self._bias is None and self._weff is None \
            and self._gemv(nr).can_fuse(nr)                         # KYOJIN_FUSE_ROWS: a verify block of 5 / 6 rows takes the one-launch fused linear of a plain step
        if KN.MAX_ROWS < nr <= KN.fuse_rows() and not vfuse:
            FD.hit("fuse_rows_gated")                           # the flag admits this many rows but a gate (LF_MASK, bias, weff, can_fuse smem / tgk) kept the many-rows path
        src = FQF.source(x) if vfuse or (x.size <= GEMV_MAX_ROWS * shape[-1] and x.size <= KN.MAX_ROWS * shape[-1]) else None     # x = fp8 fake quant of src: fold it into the rot
        x2 = x if src is None else src
        if x2.ndim != 2:
            x2 = x2.reshape(-1, shape[-1])                          # (rows, k) already: no reshape node in the decode graph
        if x2.dtype not in (mx.float16, mx.float32):
            x2 = x2.astype(mx.float16)
        rows = x2.shape[0]
        if vfuse:
            y = self._g[rows].fused(x2, fq8=fq8 or src is not None)
        elif self._weff is not None and rows >= DP.MIN_ROWS and rows > ROW_SPLIT_MAX:
            y = (x2.astype(mx.float16) @ self._weff.T).astype(self.out_dtype)
        elif self._mma_on(rows):
            y = self._mma(x2)
        elif GEMV_MAX_ROWS < rows <= ROW_SPLIT_MAX and not self._nax_rows_ok(rows):
            y = self._many(x2, half=self._rg_half(rows))
        elif rows > GEMV_MAX_ROWS:
            y = self._dense(x2, nofin=self._defer, fq8=fq8)
        elif rows <= KN.MAX_ROWS:
            g = self._gemv(rows)
            y = g.fused(x2, fq8=src is not None) if LINEAR_FUSED and LF_MASK & 1 and g.can_fuse(rows) else g(x2, fq8=src is not None)
        if self._bias is not None:
            y = y + self._bias
        return y if len(shape) == 2 else y.reshape(*shape[:-1], self.out_features)


def load_exl3(path: str, dtype=mx.float16, cb: str | None = None):
    """Return (model, config). Dense tensors load as `dtype`; every `<name>.trellis` linear becomes an Exl3Linear."""
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import _get_classes
    path = Path(path)
    TN.install(path)                                        # <pack>/tuning/<device>.json, if any (no file: the constants above)
    config = json.load(open(path / "config.json"))
    qc = config["quantization_config"]
    assert qc["quant_method"] == "exl3"
    cb = cb or qc.get("codebook", "mul1")
    tc = config.get("text_config", config)
    tc["tie_word_embeddings"] = False                       # EXL3 packs store a separate (quantised) lm_head
    model_class, args_class = _get_classes(config=config)
    model = model_class(args_class.from_dict(config))
    weights = {}
    for f in glob.glob(str(path / "model*.safetensors")):
        weights.update(mx.load(f))
    weights = model.sanitize(weights)
    experts = {}
    if any(".mlp.experts." in k for k in weights):                # MoE pack (Qwen3.5-MoE): one trellis per routed expert -> fused gather (qwen_moe.py)
        from . import qwen_moe as QM
        experts = QM.pop_experts(weights)
    prefixes = sorted({k[: -len(".trellis")] for k in weights if k.endswith(".trellis")})
    mods = []
    for p in prefixes:
        mods.append((p, Exl3Linear(weights.pop(p + ".trellis"), weights.pop(p + ".suh"), weights.pop(p + ".svh"),
                                   weights.pop(p + ".bias", None), cb=cb)))
        weights.pop(p + ".mcg", None)
        weights.pop(p + ".mul1", None)
    model.update_modules(tree_unflatten(mods))
    if experts:
        QM.patch_sparse_blocks(model, experts, cb)
    weights = {k: (v.astype(dtype) if v.dtype == mx.bfloat16 else v) for k, v in weights.items()}
    model.load_weights(list(weights.items()), strict=False)
    model.eval()                                            # training=True makes GDN layers use the slow ops loop
    from . import gdn_norm
    gdn_norm.install()                                      # fused GDN output norm (bit-equal to the stock module, KYOJIN_GDN_NORM=0 off)
    from . import gdn_fast
    gdn_fast.install()                                      # chunked GDN prefill recurrence when the tuning table / KYOJIN_GDN_FAST asks (default: stock)
    from . import gdn_pf
    gdn_pf.install()                                        # fused GDN prefill front end + gated norm (KYOJIN_GDN_PF=0 off); quality at the chunk-size KL floor
    from . import layer_pf, mlp_pf
    mlp_pf.install()
    layer_pf.install()                                      # fused decoder-layer seams (out_proj finalize + add + norm + rotations of gate / up; down finalize + add; KYOJIN_LAYER_PF=0 off)
    mx.eval(model.parameters())
    if not experts:
        from . import step_regions
        step_regions.enable(model)                          # compiled decode regions between the attention cores (KYOJIN_STEP_REGIONS=0 off), bit-equal
    DP.install(model)                                       # KYOJIN_PF_DENSE_GB: folded fp16 prefill weights when they fit
    return model, config
