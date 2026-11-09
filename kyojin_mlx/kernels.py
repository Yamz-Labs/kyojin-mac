# SPDX-License-Identifier: Apache-2.0
"""MLX wrappers (mx.fast.metal_kernel) around the generated mul1 Metal sources. Needs mlx on Apple Silicon."""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

from . import cbill as CB
from . import fired as FD
from . import metal_src as S
from . import tuning as TN

MAX_ROWS = 4
FUSE_ROWS = int(os.environ.get("KYOJIN_FUSE_ROWS", "8"))      # rows of the one-launch fused dense linears: 0 = MAX_ROWS (today), 5 .. 8 = a verify block takes the plain step's fused launch (fuse_rows(), read per call)
FUSE_ROWS_CAP = 8
FUSED_PREFETCH = bool(int(os.environ.get("KYOJIN_FUSED_PREFETCH", "1")))
ROT2_MIN_BLOCKS = int(os.environ.get("KYOJIN_ROT2_MIN_BLOCKS", str(1 << 30)))   # fused linears with this many column blocks or more: rot as its own launch (Mul1Gemv.fused); off by default, no gain measured on the Air
TAIL_SYNC = S.TAIL_SC          # arrival ordering of the fused tail (metal_src.tail_sync); KYOJIN_TAIL_SYNC=acqrel for the one-op form (both bit-identical)
TAIL_READS = S.TAIL_VEC if os.environ.get("KYOJIN_DENSE_FAST", "1") == "1" else S.TAIL_ATOM   # finalize's partial reads (metal_src.tail_reads): one plain
# float4 per k-split instead of four scalar relaxed atomics; bit-identical, 3-18 % of a small dense linear's launch. KYOJIN_DENSE_FAST=1 turns it on.
_ORSYNC = os.environ.get("KYOJIN_TAIL_SYNC")
if _ORSYNC:
    TAIL_SYNC = _ORSYNC
RG_ROWS = 8          # rows per row group of the many-rows GEMV (a group shares one trellis decode; 8 measured 5-8 % faster than 4 at 16-24 rows, 16 not faster): any of 4, 8, 16 gives the same bits
_cache: dict = {}


def fuse_rows() -> int:
    """Most rows a fused dense launch (Mul1Gemv.fused, same-input groups, grouped wo_a) takes: MAX_ROWS, or KYOJIN_FUSE_ROWS (5 .. FUSE_ROWS_CAP) when set."""
    return max(MAX_ROWS, min(FUSE_ROWS, FUSE_ROWS_CAP))


def to_words(trellis: np.ndarray) -> mx.array:
    """int16 [ti, tj, 16K] -> flat uint32 mx.array (little-endian view; tile (kt, nt) at (kt*tj + nt) * 8K)."""
    return mx.array(np.ascontiguousarray(trellis.astype("<i2")).view("<u4").reshape(-1))


def _kern(key, **kw):
    if key not in _cache:
        kw.setdefault("header", S.HEADER)
        _cache[key] = mx.fast.metal_kernel(**kw)
    return _cache[key]


def _synctag(sync: str) -> str:
    """Kernel-name suffix of a tail ordering (empty for today's seq_cst pair: the default kernels keep their names)."""
    return "" if sync == S.TAIL_SC else "_" + sync


def _readstag(reads: str) -> str:
    """Kernel-name suffix of the finalize's partial reads (empty for today's atomic reads)."""
    return "" if reads == S.TAIL_ATOM else "_" + reads


def decode_wr(words: mx.array, K: int, kt: int, nt: int, cb: str = "mul1") -> mx.array:
    """Wr^T fp16 [n, k] (bit-exact against ref_numpy.decode_tiles(...).T)."""
    kern = _kern(("dec", K, cb, S.DEC_REMAP), name=f"kyojin_{cb}_decode_k{K}_r{int(S.DEC_REMAP)}", input_names=["trellis", "params"],
                 output_names=["w"], source=S.decode_source(K, cb))
    params = mx.array([kt, nt], dtype=mx.uint32)
    return kern(inputs=[words, params], template=[("BITS", K)], grid=(kt * nt * 8, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(nt * 16, kt * 16)], output_dtypes=[mx.float16])[0]


_rot_cache: dict = {}


def rot(x: mx.array, suh: mx.array, out_dtype: mx.Dtype = mx.float32, fq8: bool = False, perm: bool = False) -> mx.array:
    """xh = H128(x * suh) / sqrt(128). x: (rows, k) fp16/fp32, k % 128 == 0; out fp32 (default) or fp16.
    suh must already be fp16. Everything but the launch is cached: python overhead is on the decode critical path.
    fq8 (fp16 x): x is first rounded through the fp8 e4m3 / ue8m0 fake quant (blocks of 32), as `fake_quant_fp8_ue8m0(x)` would give, in the same launch.
    perm: k permuted inside 16-blocks (input layout of the fused EXL3 + NAX GEMM, metal_nax)."""
    rows, k = x.shape
    key = (x.dtype, out_dtype, k, fq8, perm)
    ent = _rot_cache.get(key)
    if ent is None:
        ih, oh = x.dtype == mx.float16, out_dtype == mx.float16
        assert x.dtype in (mx.float16, mx.float32) and out_dtype in (mx.float16, mx.float32)
        kern = _kern(("rot", ih, oh, fq8, perm), name=f"kyojin_rot_{'h' if ih else 'f'}{'h' if oh else 'f'}{'_fq' if fq8 else ''}{'_p' if perm else ''}",
                     input_names=["x", "suh", "params"], output_names=["xh"], source=S.rot_source(ih, oh, fq8, perm))
        ent = _rot_cache[key] = (kern, mx.array([k], dtype=mx.uint32))
    kern, params = ent
    grid, tg = S.had_grid(k // 128, rows)
    return kern(inputs=[x, suh if suh.dtype == mx.float16 else suh.astype(mx.float16), params], grid=grid, threadgroup=tg,
                output_shapes=[(rows, k)], output_dtypes=[out_dtype])[0]


def fin_kernel(out_dtype: mx.Dtype, part_dtype: mx.Dtype = mx.float32):
    """Finalize kernel: y = H128(sum of partials) / sqrt(128) * svh. Partials fp32 (GEMV) or fp16 (dense prefill matmul)."""
    key = ("fin", out_dtype, part_dtype)
    if key not in _cache:
        ph = part_dtype == mx.float16
        _cache[key] = mx.fast.metal_kernel(header=S.HEADER, name=f"kyojin_final_{str(out_dtype).split('.')[-1]}_{'h' if ph else 'f'}",
                                           input_names=["part", "svh", "params"], output_names=["y"], source=S.final_source(ph))
    return _cache[key]


FILL_IDLE_SLACK = int(os.environ.get("KYOJIN_FILL_IDLE_SLACK", "8"))     # a k-split may leave up to 1 / this of the slices without tiles (12.5 % at 8)


def pick_tgk(kt: int, blocks: int, ws: int = 4, target_tgs: int | None = None, min_slices: int | None = None) -> int:
    """k-splits across threadgroups until ~target_tgs threadgroups exist (powers of two), each warp keeping at least `min_slices` 16-element k tiles.
    target_tgs / min_slices default to the device fill (tuning.fill_tgs / fill_min_tiles: 8 threadgroups per GPU core, and the tiles per warp fall with
    the core count), so the small linears of a decode step fill a wide GPU; on the 10-core Air both are the 512 / 8 the constants were measured with,
    and KYOJIN_FILL=0 gives them to any device. A split whose k slices do not tile `kt` exactly is allowed up to FILL_IDLE_SLACK (the empty slices skip
    the weight loop and only run the prologue and the zero partial store). Only the number of k-splits changes: the partials are still summed in k order,
    a rounding at 1e-7 relative."""
    if target_tgs is None:
        target_tgs = TN.fill_tgs()
    if min_slices is None:
        min_slices = TN.fill_min_tiles()
    tgk = 1
    while blocks * tgk < target_tgs:
        s = ws * tgk * 2                                            # k slices of the next split
        chunk = -(-kt // s)                                         # 16-element tiles per slice
        if kt // s < min_slices or s * chunk * FILL_IDLE_SLACK > kt * (FILL_IDLE_SLACK + 1):
            break
        tgk *= 2
    return tgk


def param_sets(head: list, stride: int, n: int) -> list:
    """n parameter vectors head + [i * stride] (the counter offset of set i) as evaluated rows of ONE uint32 array: one buffer, not n."""
    allp = mx.array([list(head) + [i * stride] for i in range(n)], dtype=mx.uint32)
    rows = [allp[i] for i in range(n)]
    mx.eval(*rows)
    return rows


class Mul1Gemv:
    """y = layer(x) for one mul1 linear: rot -> partial GEMV (trellis decode + dot) -> finalize (sum, H128, svh)."""

    def __init__(self, trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, rows: int = 1,
                 out_dtype: mx.Dtype = mx.float32, prefetch: bool = False, tgk: int | None = None, cb: str = "mul1"):
        kt, nt, w = trellis.shape
        self.K, self.kt, self.nt, self.rows = w // 16, kt, nt, rows
        assert rows <= MAX_ROWS and (nt * 16) % 128 == 0 and (kt * 16) % 128 == 0
        self.k, self.n, self.blocks = kt * 16, nt * 16, nt // 8
        self.out_dtype, self.prefetch, self.cb = out_dtype, prefetch, cb
        self.dcb = S.dense_cb(cb)                                  # dense decode cut (KYOJIN_DENSE_DEC), read when the kernel is built
        self.tgk = tgk or pick_tgk(kt, self.blocks)
        self.words = trellis.view(mx.uint32).reshape(-1) if isinstance(trellis, mx.array) else to_words(trellis)
        self.bview = CB.make_view(self.words)                      # KYOJIN_DENSE_VIEWS: the prefix view the decode kernels bind (cbill)
        self.suh, self.svh = mx.array(suh), mx.array(svh)
        self.params = mx.array([kt, nt, self.tgk], dtype=mx.uint32)
        self.fparams = mx.array([self.tgk, self.n, rows], dtype=mx.uint32)
        key = ("gemv", self.K, rows, prefetch, self.dcb, S.PF2)
        self._kerns = {(self.dcb, rows, S.PF2): _kern(key, name=f"kyojin_{self.dcb}_gemv_k{self.K}_m{rows}_p{int(prefetch)}" + ("_p2" if S.PF2 else ""),
                                               input_names=["xh", "trellis", "params"], output_names=["part"], header=S.dense_header(self.dcb),
                                               source=S.gemv_source(self.K, rows, prefetch, self.dcb, pf2=S.PF2))}
        self.kern = self._kerns[(self.dcb, rows, S.PF2)]
        self.fin = fin_kernel(out_dtype)
        self._fsuh = self.suh if self.suh.dtype == mx.float16 else self.suh.astype(mx.float16)
        self._fsvh = self.svh if self.svh.dtype == mx.float16 else self.svh.astype(mx.float16)
        self.fused_init()
        mx.eval(self.words, self.suh, self.svh, self.params, self.fparams, self._fsuh, self._fsvh, self._fctr, *self._fparams)

    def partial(self, xh: mx.array) -> mx.array:
        dcb = self.dcb = S.dense_cb(self.cb)
        if dcb != self.cb:
            FD.hit("dense_dec")
        k = self._kerns.get((dcb, self.rows, S.PF2))
        if k is None:
            k = self._kerns[(dcb, self.rows, S.PF2)] = _kern(("gemv", self.K, self.rows, self.prefetch, dcb, S.PF2), name=f"kyojin_{dcb}_gemv_k{self.K}_m{self.rows}_p{int(self.prefetch)}" + ("_p2" if S.PF2 else ""),
                                                       input_names=["xh", "trellis", "params"], output_names=["part"], header=S.dense_header(dcb),
                                                       source=S.gemv_source(self.K, self.rows, self.prefetch, dcb, pf2=S.PF2))
        self.kern = k
        return self.kern(inputs=[xh, CB.bind(self.words, self.bview), self.params], template=[("BITS", self.K), ("MR", self.rows)],
                         grid=(self.blocks * 256, self.tgk, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(self.tgk, self.rows, self.n)], output_dtypes=[mx.float32])[0]

    def partial_rows(self, xh: mx.array, rows: int, params: mx.array, mr: int = MAX_ROWS, half: bool = False) -> mx.array:
        """All `rows` rows (any number) in ONE launch: row groups of `mr` rows on grid z, the trellis decode shared by the rows of a group
        (metal_src.gemv_rg_source). params [kt, nt, tgk, rows, 1]. `half`: the row groups FMA in packed fp16 (flag rg_half; `xh` is the fp16
        rot output then, fp32 otherwise)."""
        dcb = self.dcb = S.dense_cb(self.cb)
        if dcb != self.cb:
            FD.hit("dense_dec")
        kern = _kern(("gemvrg", self.K, dcb, mr, half), name=f"kyojin_{dcb}_gemvrg_k{self.K}_m{mr}" + ("_h" if half else ""),
                     input_names=["xh", "trellis", "params"], output_names=["part"], header=S.dense_header(dcb),
                     source=S.gemv_rg_source(self.K, dcb, mr, half=half))
        return kern(inputs=[xh, CB.bind(self.words, self.bview), params], template=[("BITS", self.K), ("MR", mr)],
                    grid=(self.blocks * 256, self.tgk, (rows + mr - 1) // mr), threadgroup=(256, 1, 1),
                    output_shapes=[(self.tgk, rows, self.n)], output_dtypes=[mx.float32])[0]

    def finalize(self, part: mx.array) -> mx.array:
        grid, tg = S.had_grid(self.blocks, self.rows)
        return self.fin(inputs=[part, self.svh, self.fparams], template=[("OT", self.out_dtype)], grid=grid, threadgroup=tg,
                        output_shapes=[(self.rows, self.n)], output_dtypes=[self.out_dtype])[0]

    FUSED_SETS = 64                     # counter sets, handed out round robin: calls of one module that may run concurrently must not share one. MLX runs independent launches
                                            # concurrently: 16 row chunks of a 64-row multi-user call, or 40 solo rows in one graph, overran 8 sets (two launches on one counter = wrong sums)
    FUSED_SMEM_FLOATS = 7000            # threadgroup memory budget of the fused kernel (32 KB)

    def can_fuse(self, rows: int) -> bool:
        ok = self._cf.get(rows)
        if ok is None:
            ok = self._cf[rows] = not (self.tgk & (self.tgk - 1)) and \
                max(rows * S.fused_dims(self.kt, self.tgk), 512 * rows) <= self.FUSED_SMEM_FLOATS
        return ok and rows <= fuse_rows()

    def fused_init(self) -> None:
        """Counters (zeros, never reset: the kernel counts modulo TGK) and parameter sets of the fused kernel. Plain constants: safe to call while tracing."""
        self._fctr = mx.array(np.zeros(self.FUSED_SETS * self.blocks, dtype=np.uint32))
        self._fparams = param_sets([self.kt, self.nt, self.tgk], self.blocks, self.FUSED_SETS)
        self._fnext = 0
        self._cf: dict = {}                                 # rows -> can_fuse (static per module)
        self._fc: dict = {}                                 # (rows, fp16 input, fq8) -> everything fused() needs but the input arrays (host cost: the graph build is on the decode critical path)

    def _half_on(self, rows: int) -> bool:
        """Packed-fp16 row FMAs: S.half_on, but never with an affine-folded decode (mul1fd returns the bare byte sum, the affine is applied per output).
        The class follows the row count, not the path: a fused block of up to fuse_rows() rows keeps the FMA class its rows have in an <= MAX_ROWS
        launch (HALF_ROWS 4: rows 4..8 packed fp16), so a verify row's bits do not change with the depth the planner picked."""
        return rows <= fuse_rows() and S.half_on(rows) and not S.af_fold(self.dcb)

    def _fused_entry(self, rows: int, ih: bool, fq8: bool, half: bool, dcb: str) -> tuple:
        xsr = S.fused_dims(self.kt, self.tgk)
        grid, tg = (self.blocks * 256, self.tgk, 1), (256, 1, 1)
        shapes, dts = [(self.tgk, rows, self.n), (rows, self.n)], [mx.float32, self.out_dtype]
        tmpl = [("BITS", self.K), ("MR", rows), ("OT", self.out_dtype)]
        sync = TAIL_SYNC
        reads = TAIL_READS
        if self.blocks >= ROT2_MIN_BLOCKS:
            kern = _kern(("fused2", self.K, rows, dcb, FUSED_PREFETCH, sync, reads), name=f"kyojin_{dcb}_fused2_k{self.K}_m{rows}_p{int(FUSED_PREFETCH)}{_synctag(sync)}{_readstag(reads)}",
                         input_names=["xh", "svh", "trellis", "params", "ctr"], output_names=["part", "y"], header=S.dense_header(dcb),
                         source=S.fused_source(self.K, rows, dcb, ih, fq8, xsr, FUSED_PREFETCH, prologue=False, sync=sync, reads=reads))
            return (True, kern, tmpl, grid, tg, shapes, dts)
        hk = (S.HALF_FLUSH,) if half and self.blocks < ROT2_MIN_BLOCKS else ()
        kern = _kern(("fused", self.K, rows, dcb, ih, fq8, xsr, FUSED_PREFETCH, sync, reads) + hk, name=f"kyojin_{dcb}_fused_k{self.K}_m{rows}_{'h' if ih else 'f'}{'_fq' if fq8 else ''}_x{xsr}_p{int(FUSED_PREFETCH)}{_synctag(sync)}{_readstag(reads)}{'_hf%d' % S.HALF_FLUSH if hk else ''}",
                     input_names=["x", "suh", "svh", "trellis", "params", "ctr"], output_names=["part", "y"], header=S.dense_header(dcb),
                     source=S.fused_source(self.K, rows, dcb, ih, fq8, xsr, FUSED_PREFETCH, half=half, sync=sync, reads=reads))
        return (False, kern, tmpl, grid, tg, shapes, dts)

    def can_fuse_glu(self, rows: int) -> bool:
        """fused_glu applies: the one-launch fused kernel with its prologue (not the ROT2 two-launch form)."""
        return self.can_fuse(rows) and self.blocks < ROT2_MIN_BLOCKS

    def fused_glu(self, g: mx.array, u: mx.array, limit: float) -> mx.array:
        """fused(fq8(act)) with act = silu(min(g, L)) * clip(u, -L, L) cast to fp16, computed in the prologue (no separate activation launch). g, u: (rows, k) fp16.
        Bit-identical to the swiglu launch followed by fused(act, fq8=True). Only when can_fuse_glu(rows)."""
        rows = g.shape[0]
        dcb = self.dcb = S.dense_cb(self.cb)
        half = self._half_on(rows)
        key = (rows, "glu", float(limit), half, dcb, TAIL_READS)
        ent = self._fc.get(key)
        if ent is None:
            xsr = S.fused_dims(self.kt, self.tgk)
            hk = (S.HALF_FLUSH,) if half else ()
            kern = _kern(("fusedglu", self.K, rows, dcb, float(limit), xsr, FUSED_PREFETCH, TAIL_SYNC, TAIL_READS) + hk,
                         name=f"kyojin_{dcb}_fusedglu_k{self.K}_m{rows}_x{xsr}_p{int(FUSED_PREFETCH)}{_synctag(TAIL_SYNC)}{_readstag(TAIL_READS)}{'_hf%d' % S.HALF_FLUSH if hk else ''}",
                         input_names=["x", "u", "suh", "svh", "trellis", "params", "ctr"], output_names=["part", "y"], header=S.dense_header(dcb),
                         source=S.fused_source(self.K, rows, dcb, True, True, xsr, FUSED_PREFETCH, half=half, sync=TAIL_SYNC, reads=TAIL_READS, glu=float(limit)))
            ent = self._fc[key] = (kern, [("BITS", self.K), ("MR", rows), ("OT", self.out_dtype)], (self.blocks * 256, self.tgk, 1), (256, 1, 1),
                                   [(self.tgk, rows, self.n), (rows, self.n)], [mx.float32, self.out_dtype])
        kern, tmpl, grid, tg, shapes, dts = ent
        n = self._fnext
        params = self._fparams[n]
        self._fnext = n + 1 if n + 1 < self.FUSED_SETS else 0
        FD.hit("shared_glu")
        return kern(inputs=[g, u, self._fsuh, self._fsvh, CB.bind(self.words, self.bview), params, self._fctr], template=tmpl, grid=grid, threadgroup=tg,
                    output_shapes=shapes, output_dtypes=dts)[1]

    def fused(self, x: mx.array, fq8: bool = False) -> mx.array:
        """rot + GEMV + finalize in one launch (see metal_src.fused_source); bit-identical to __call__. x: (rows, k) fp16 / fp32, rows == self.rows.
        A linear with ROT2_MIN_BLOCKS or more column blocks (wq_b, wo_b: the prologue would be repeated by every column block) takes two launches:
        the rot kernel, then GEMV + finalize."""
        rows = x.shape[0]
        if rows > MAX_ROWS:
            FD.hit("fuse_rows")                                 # a 5 / 6 row block on the plain step's fused launch (KYOJIN_FUSE_ROWS)
        dcb = self.dcb = S.dense_cb(self.cb)                    # KYOJIN_DENSE_DEC: read per call, the key rebuilds the kernel when it moves
        if dcb != self.cb:
            FD.hit("dense_dec")                                 # fused dense call with a dense decode cut
        if TAIL_READS == S.TAIL_VEC:
            FD.hit("dense_fast")                                # fused dense call with the vector tail reads
        if self._half_on(rows):
            FD.hit("half_rows")                                 # fused dense call with packed-fp16 row FMAs
        key = (rows, x.dtype == mx.float16, fq8, self._half_on(rows), dcb, TAIL_READS)   # TAIL_READS in the key: KYOJIN_DENSE_FAST switches at call time
        ent = self._fc.get(key)
        if ent is None:
            ent = self._fc[key] = self._fused_entry(*key[:4], dcb)
        two, kern, tmpl, grid, tg, shapes, dts = ent
        n = self._fnext
        params = self._fparams[n]
        self._fnext = n + 1 if n + 1 < self.FUSED_SETS else 0
        if two:
            return kern(inputs=[rot(x, self._fsuh, fq8=fq8), self._fsvh, CB.bind(self.words, self.bview), params, self._fctr], template=tmpl, grid=grid, threadgroup=tg,
                        output_shapes=shapes, output_dtypes=dts)[1]
        return kern(inputs=[x, self._fsuh, self._fsvh, CB.bind(self.words, self.bview), params, self._fctr], template=tmpl, grid=grid, threadgroup=tg,
                    output_shapes=shapes, output_dtypes=dts)[1]

    def __call__(self, x: mx.array, fq8: bool = False) -> mx.array:
        return self.finalize(self.partial(rot(x, self.suh, fq8=fq8)))
