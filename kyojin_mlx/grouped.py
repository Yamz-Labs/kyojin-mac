# SPDX-License-Identifier: Apache-2.0
"""Grouped linear for DS4.1 `attn.wo_a` (8 slices, one weight per group, x [1, G, S, in] -> [1, G, S, out], S = 1..4 rows per group).

The per-group path runs rot + GEMV + finalize once per group (24 launches over 8 slices of 2.6 MB each). The grouped path runs the three stages
once for all groups (3 launches, grid z = group):
  rot       the MoE rot_gather kernel with pair = (group, row), "expert" id = group: xh[g * S + s] = H128(x * suh_g) / sqrt(128)
  gemv      gemv_source(K, S) (one trellis decode shared by the S rows of a group) with z = group: trellis + g * slice words, xh + g * S rows,
            partial stored at [kb, g * S + s, :]
  finalize  final_source with a per-group svh and the fp16 store: y[g * S + s] = H128(sum_kb part) / sqrt(128) * svh_g
The two derived kernels are the per-group ones with asserted text substitutions, so the arithmetic and the k-split order (same tgk) are those of
the per-group path and the result is bit-identical. Batch B > 1: the caller keeps the per-group path. More than KN.MAX_ROWS rows per group: row groups on the grid (S.gemv_rg_source), still 3 launches."""
from __future__ import annotations

import functools

import numpy as np
import mlx.core as mx

from . import cbill as CB
from . import fired as FD
from . import exl3_linear as EL
from . import fqfuse as FQF
from . import kernels as KN
from . import metal_nax as NX
from . import metal_pf as PF
from . import metal_src as S
from .moe import _kernel


def _sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"template changed, {old!r} found {text.count(old)} times"
    return text.replace(old, new)


@functools.lru_cache(maxsize=None)
def gemv_group_source(K: int, rows: int, cb: str = "mul1", prefetch: bool = False) -> str:
    """gemv_source with grid z = group. params u32 [KT, NT, TGK, G]. Output part [TGK, G * rows, N] f32."""
    s = S.gemv_source(K, rows, prefetch, cb)
    s = _sub(s, "const uint KT = params[0], NT = params[1], TGK = params[2];",
             "const uint KT = params[0], NT = params[1], TGK = params[2], G = params[3];\n    const uint zg = threadgroup_position_in_grid.z;")
    s = _sub(s, "const device uint* tp = trellis + (ulong(ks0) * NT + nt) * TW;",
             "const device uint* tp = trellis + ulong(zg) * (ulong(KT) * NT * TW) + (ulong(ks0) * NT + nt) * TW;")
    s = _sub(s, "(const device float4*)(xh + ks0 * 16);", "(const device float4*)(xh + ulong(zg) * MR * KD + ks0 * 16);")
    s = _sub(s, "part[(ulong(kblock) * MR + r) * N + blk * 128 + c] = v;", "part[((ulong(kblock) * G + zg) * MR + r) * N + blk * 128 + c] = v;")
    return s


@functools.lru_cache(maxsize=None)
def final_group_source(part_half: bool = False) -> str:
    """final_source with a per-group svh. params u32 [TGK, N, G * rows, rows]. part [TGK, G * rows, N], row r = g * rows + s."""
    s = S.final_source(part_half)
    s = _sub(s, "const uint TGK = params[0], N = params[1], MRT = params[2];", "const uint TGK = params[0], N = params[1], MRT = params[2], MR = params[3];")
    s = _sub(s, "(const device half4*)(svh + blk * 128 + lane * 4)", "(const device half4*)(svh + ulong(r / MR) * N + blk * 128 + lane * 4)")
    return s


@functools.lru_cache(maxsize=None)
def fused_group_source(K: int, rows: int, cb: str, in_half: bool, fq8: bool, xsr: int) -> str:
    """The grouped linear as ONE launch (rot + GEMV + finalize, metal_src.fuse_gemv over gemv_group_source). Inputs x [G * rows, in], suh [G, in], svh [G, out], trellis,
    params u32 [KT, NT, TGK, G, counter base], ctr; outputs part, y [G * rows, out] fp16."""
    return S.fuse_gemv(gemv_group_source(K, rows, cb, True), rows, xsr, in_half, fq8,
                       xo="ulong(zg * MR + rr) * KD + bb * 128 + lane * 4", suh="suh + ulong(zg) * KD + bb * 128 + lane * 4", ctri="params[4] + zg * (NT >> 3) + blk",
                       pb="(ulong(zg) * MR + warp) * N + blk * 128 + lane * 4", ks="ulong(G) * MR * N", svh="svh + ulong(zg) * N + blk * 128 + lane * 4",
                       yout="y + ulong(zg * MR + warp) * N + blk * 128 + lane * 4",
                       xp_old="    const device float4* xp = (const device float4*)(xh + ulong(zg) * MR * KD + ks0 * 16);\n")


class GroupedGemv:
    """G slices of one grouped linear. `trellis` int16 [G, kt, nt, 16K], `suh` [G, in], `svh` [G, out]."""

    def __init__(self, trellis: mx.array, suh: mx.array, svh: mx.array, cb: str = "mul1", tgk: int | None = None):
        G, kt, nt, w = trellis.shape
        self.G, self.K, self.cb = G, w // 16, cb
        self.dcb = S.dense_cb(cb)                           # KYOJIN_DENSE_DEC, refreshed per call
        self.words = trellis.view(mx.uint32).reshape(-1)
        self.bview = CB.make_view(self.words)
        self.suh, self.svh = suh.astype(mx.float16), svh.astype(mx.float16)
        self.blocks = nt // 8
        self.in_features, self.out_features = kt * 16, nt * 16
        self.tgk = tgk or KN.pick_tgk(kt, self.blocks)                         # the tgk of the per-group Mul1Gemv: same k-split order
        u32 = lambda *v: mx.array(list(v), dtype=mx.uint32)
        self.p_rot, self.p_gemv = u32(self.in_features, 1), u32(kt, nt, self.tgk, G)
        self.fusable = self.tgk & (self.tgk - 1) == 0
        self.xsr = S.fused_dims(kt, self.tgk)
        self.nsets, self._fnext = KN.Mul1Gemv.FUSED_SETS, 0
        self.ctr = mx.array(np.zeros(self.nsets * G * self.blocks, dtype=np.uint32))
        self.p_fused = KN.param_sets([kt, nt, self.tgk, G], G * self.blocks, self.nsets)
        self._ids: dict = {}
        self._fin: dict = {}
        self._nax: tuple | None = None                                         # (stack for expert_gemm, group ids) on first NAX call
        self._nax_rows: dict = {}
        self._rg: dict = {}
        mx.eval(self.words, self.suh, self.svh, self.p_rot, self.p_gemv, self.ctr, *self.p_fused)

    def _group_ids(self, rows: int) -> mx.array:
        if rows not in self._ids:
            self._ids[rows] = mx.array((np.arange(self.G * rows) // rows).astype(np.int32))
            self._fin[rows] = mx.array([self.tgk, self.out_features, self.G * rows, rows], dtype=mx.uint32)
            mx.eval(self._ids[rows], self._fin[rows])
        return self._ids[rows]

    def fuses(self, R: int) -> bool:
        """The one-launch fused kernel serves R rows per group (R <= KYOJIN_FUSE_ROWS, or 4, and the x buffer fits the threadgroup budget)."""
        return bool(EL.LINEAR_FUSED and EL.LF_MASK & 4 and self.fusable and R <= KN.fuse_rows() and max(R * self.xsr, 512 * R) <= KN.Mul1Gemv.FUSED_SMEM_FLOATS)

    def __call__(self, x: mx.array, fq8: bool = False) -> mx.array:
        """fq8: x is the INPUT of the fp8 fake quant that wo_a applies first; the round trip runs inside the rot (fp16 x, kin % 128 == 0)."""
        B, G, R, kin = x.shape
        assert B == 1 and G == self.G and kin == self.in_features, (x.shape, self.G, self.in_features)
        pairs = G * R
        x2 = x.reshape(pairs, kin)
        if x2.dtype not in (mx.float16, mx.float32):
            x2 = x2.astype(mx.float16)
        ih = x2.dtype == mx.float16
        dcb = self.dcb = S.dense_cb(self.cb)                         # KYOJIN_DENSE_DEC, read per call
        if self.fuses(R):
            if R > KN.MAX_ROWS:
                FD.hit("fuse_rows_wo_a")
            kf1 = _kernel(("grp_fused", self.K, R, dcb, ih, fq8, self.xsr), name=f"kyojin_grp_fused_{dcb}_k{self.K}_m{R}_{'h' if ih else 'f'}{'_fq' if fq8 else ''}_x{self.xsr}",
                          input_names=["x", "suh", "svh", "trellis", "params", "ctr"], output_names=["part", "y"],
                          header=S.dense_header(dcb),
                          source=fused_group_source(self.K, R, dcb, ih, fq8, self.xsr))
            params = self.p_fused[self._fnext]
            self._fnext = (self._fnext + 1) % self.nsets
            y = kf1(inputs=[x2, self.suh, self.svh, CB.bind(self.words, self.bview), params, self.ctr], template=[("BITS", self.K), ("MR", R), ("OT", mx.float16)],
                    grid=(self.blocks * 256, self.tgk, G), threadgroup=(256, 1, 1),
                    output_shapes=[(self.tgk, pairs, self.out_features), (pairs, self.out_features)], output_dtypes=[mx.float32, mx.float16])[1]
            return y.reshape(1, G, R, self.out_features)
        ids = self._group_ids(R)
        kr = _kernel(("grp_rot", ih, fq8), name=f"kyojin_grp_rot_{'h' if ih else 'f'}{'_fq' if fq8 else ''}", input_names=["x", "ids", "suh0", "params"],
                     output_names=["xh"], source=S.rot_gather_source(1, ih, fq8))
        grid, tg = S.had_grid(kin // 128, pairs)
        xh = kr(inputs=[x2, ids, self.suh, self.p_rot], grid=grid, threadgroup=tg, output_shapes=[(pairs, kin)], output_dtypes=[mx.float32])[0]
        if R <= KN.MAX_ROWS:
            kg = _kernel(("grp_gemv", self.K, R, dcb), name=f"kyojin_grp_gemv_{dcb}_k{self.K}_m{R}", input_names=["xh", "trellis", "params"],
                         output_names=["part"], header=S.dense_header(dcb), source=gemv_group_source(self.K, R, dcb))
            part = kg(inputs=[xh, CB.bind(self.words, self.bview), self.p_gemv], template=[("BITS", self.K), ("MR", R)], grid=(self.blocks * 256, self.tgk, G),
                      threadgroup=(256, 1, 1), output_shapes=[(self.tgk, pairs, self.out_features)], output_dtypes=[mx.float32])[0]
        else:                                                                  # more rows per group: row groups of MAX_ROWS on the grid, still one launch
            if R not in self._rg:
                self._rg[R] = mx.array([self.in_features // 16, self.out_features // 16, self.tgk, R, G], dtype=mx.uint32)
            kg = _kernel(("grp_gemvrg", self.K, dcb), name=f"kyojin_grp_gemvrg_{dcb}_k{self.K}", input_names=["xh", "trellis", "params"],
                         output_names=["part"], header=S.dense_header(dcb), source=S.gemv_rg_source(self.K, dcb, KN.MAX_ROWS))
            part = kg(inputs=[xh, CB.bind(self.words, self.bview), self._rg[R]], template=[("BITS", self.K), ("MR", KN.MAX_ROWS)],
                      grid=(self.blocks * 256, self.tgk, G * ((R + KN.MAX_ROWS - 1) // KN.MAX_ROWS)), threadgroup=(256, 1, 1),
                      output_shapes=[(self.tgk, pairs, self.out_features)], output_dtypes=[mx.float32])[0]
        kf = _kernel(("grp_fin",), name="kyojin_grp_fin", input_names=["part", "svh", "params"], output_names=["y"], source=final_group_source())
        grid, tg = S.had_grid(self.blocks, pairs)
        y = kf(inputs=[part, self.svh, self._fin[R]], template=[("OT", mx.float16)], grid=grid, threadgroup=tg,
               output_shapes=[(pairs, self.out_features)], output_dtypes=[mx.float16])[0]
        return y.reshape(1, G, R, self.out_features)

    def nax(self, x: mx.array, fq8: bool = False) -> mx.array:
        """Rows above one GEMV chunk (KN.MAX_ROWS) in one fused trellis-decode + NAX matmul over all groups: 3 launches whatever the rows
        (the GEMV path needs 3 per chunk of 4 rows). x [1, G, R, in] -> [1, G, R, out]; fp16 rounding, not the bits of the GEMV path
        (a row does not depend on the other rows). fq8: x is the INPUT of the fp8 fake quant that wo_a applies first, run inside the rot (fp16 x)."""
        B, G, R, kin = x.shape
        assert B == 1 and G == self.G and kin == self.in_features, (x.shape, self.G, self.in_features)
        pairs = G * R
        x2 = x.reshape(pairs, kin)
        if x2.dtype not in (mx.float16, mx.float32):
            x2 = x2.astype(mx.float16)
        if self._nax is None:
            self._nax = (NX.SingleStack(self.words, self.K, self.in_features // 16, self.out_features // 16, self.cb, self.G),
                         mx.array(np.arange(self.G, dtype=np.int32)))
        st, eids = self._nax
        ent = self._nax_rows.get(R)
        if ent is None:
            idx = np.arange(pairs)
            ent = self._nax_rows[R] = (mx.array(idx.astype(np.uint32)), mx.array((idx // R).astype(np.int32)),
                                       mx.array((np.arange(G + 1) * R).astype(np.uint32)), mx.array([kin, pairs], dtype=mx.uint32),
                                       mx.array([1, self.out_features, pairs, R], dtype=mx.uint32), NX.pick_cfg([R] * G))
            mx.eval(*ent[:5])
        tok, gid, rowoff, prot, pfin, cfg = ent
        ih = x2.dtype == mx.float16
        kr = _kernel(("grp_rot_nax", ih, fq8), name=f"kyojin_grp_rot_nax_{'h' if ih else 'f'}{'_fq' if fq8 else ''}", input_names=["x", "tok", "ids", "suh0", "params"],
                     output_names=["xh"], source=PF.rot_pf_source(1, ih, True, True, fq8))
        grid, tg = S.had_grid(kin // 128, pairs)
        xh = kr(inputs=[x2, tok, gid, self.suh, prot], grid=grid, threadgroup=tg, output_shapes=[(pairs, kin)], output_dtypes=[mx.float16])[0]
        inner = NX.expert_gemm((st,), eids, rowoff, xh, pairs, R, cfg)
        kf = _kernel(("grp_fin_h",), name="kyojin_grp_fin_h", input_names=["part", "svh", "params"], output_names=["y"], source=final_group_source(True))
        grid, tg = S.had_grid(self.blocks, pairs)
        y = kf(inputs=[inner.reshape(1, pairs, self.out_features), self.svh, pfin], template=[("OT", mx.float16)], grid=grid, threadgroup=tg,
               output_shapes=[(pairs, self.out_features)], output_dtypes=[mx.float16])[0]
        return y.reshape(1, G, R, self.out_features)
