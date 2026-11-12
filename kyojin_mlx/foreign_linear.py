# SPDX-License-Identifier: Apache-2.0
"""ForeignLinear: a linear on foreign 4-bit weights (NVFP4 / MXFP4 safetensors, GGUF Q4_0 / Q4_K / IQ4_NL / MXFP4 / NVFP4) read as stored, with the call
shape of Exl3Linear (x (..., in) -> (..., out), fp16 out). ForeignSwitchGLU: the routed experts of an MoE block, a drop-in for mlx-lm's `switch_mlp`.
`load_foreign(dir)` builds an mlx-lm model from a directory in one of those formats. Nothing here runs for an EXL3 pack.

Kernels (foreign_metal.py, launched through kernels._kern like the rest of the repo):
  rows <= GEMV_ROWS (4)      float-FMA GEMV of the gate (k % 512 == 0, n % (8 NBR) == 0), else the MMA kernel
  rows <= MMA_MAX (16)       simdgroup_matrix kernel: the matrix is read once for all rows
  rows <= DENSE_ROWS         chunks of 16 rows through the MMA kernel
  more                       the weight is decoded to fp16 once per call and one MLX matmul runs
Thresholds are environment knobs (KYOJIN_FOREIGN_*), unmeasured defaults: tools/fmt_gate_mac.py --native prints the numbers to set them from."""
from __future__ import annotations

import functools
import glob
import json
import math
import os
import re
import weakref
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from . import foreign_gguf as GF, foreign_kqgmm as KQ, foreign_metal as FM, foreign_pack as FP, foreign_wide as FW
from .foreign_names import ARCHS, gguf_name, gguf_config, is_linear, n_trunk_of
from . import foreign_xform as XF
from . import kernels as KN

GEMV_ROWS = int(os.environ.get("KYOJIN_FOREIGN_GEMV_ROWS", "4"))
MMA_MAX = min(16, int(os.environ.get("KYOJIN_FOREIGN_MMA_MAX", "16")))
DENSE_ROWS = int(os.environ.get("KYOJIN_FOREIGN_DENSE_ROWS", "16"))     # no resident fp16 copy: above 16 rows one decode into a transient buffer + one MLX matmul beats the 16-row MMA chunks (Q4_K 17408x5120 R24-64 0.57-0.64 vs 0.76-0.88 ms)
GEMV8_ROWS = int(os.environ.get("KYOJIN_FOREIGN_GEMV8_ROWS", "8"))       # 8-bit formats (flag fp8stored): stored GEMV up to here (measured: GEMV8 wins to R7, mma8 from R8)
MMA8_SLICE = 32                                                           # rows of one mma8 launch (RB = 4 row blocks); more rows are sliced
DENSE8_ROWS = int(os.environ.get("KYOJIN_FOREIGN_DENSE8_ROWS", "32"))     # mma8 up to here (two 16-row slices cost one weight read each), above it MLX mxfp8 / decode + one fp16 matmul (27B R24: mma8 265-374 vs decode + matmul 175-199 GB/s of stored bytes)
F4_ROWS = int(os.environ.get("KYOJIN_FOREIGN_F4_ROWS", "17"))             # flag f4copy: from here the resident fp16 copy runs one matmul (27B N=4: R16 best, R5 cost 8 % solo)
MQ_MIN = int(os.environ.get("KYOJIN_FOREIGN_MQ_MIN", "12"))              # flag mqcopy: from here the MLX-native quantized copy runs mx.quantized_matmul (fmma serves the rows below)
MQ_MAX = int(os.environ.get("KYOJIN_FOREIGN_MQ_MAX", "127"))             # ... up to here (above it the fp16 copy)
MQ_Q5K = os.environ.get("KYOJIN_FOREIGN_MQ_Q5K", "0") not in ("0", "")   # Q5_K copy: wins per matrix, but the real mixed 27B at N8 lost 5-12 % per round (cause open) and load +46 s
MQ_SMALL = 8 << 20                                                         # under this many weights the fp16 copy / stored kernels win (1024x5120 R33-64: 2.4 vs 1.0 ms per 64)
MQ_QMV_ROWS, MQ_QMV_N, MQ_QMV_K = 32, 10240, 5120                         # MLX's quantized matmul takes a slow path up to 32 rows: it wins there only for n >= 10240, k <= 5120 (17408x5120 R12-32 7 vs 8-12 ms per 64, 11008x4096 3.7 vs 4.3; 5120x17408 24 vs 8, 8192x5120 R12 10.7 vs 4.0, 5120x5120 7.3 vs 2.7); from 33 rows it wins everywhere above MQ_SMALL
ZQ = os.environ.get("KYOJIN_FOREIGN_ZQ", "1") not in ("0", "")           # NVFP4 / MXFP4 / FP8 stored bytes read in place by MLX's quantized matmul (the layouts are MLX's own: no copy); 0 = off
ZQ_LOW = int(os.environ.get("KYOJIN_FOREIGN_ZQ_LOW", "4"))               # nvfp4 / mxfp4: rows 1..ZQ_LOW on MLX's qmv (27B 17408x5120 R2 676 vs fmma 502, R4 592 vs 494 GB/s of stored bytes), R5..16 on fmma (R8 503 vs 380)
ZQ_HIGH = int(os.environ.get("KYOJIN_FOREIGN_ZQ_HIGH", "48"))            # from here MLX's quantized matmul serves every matrix (R48 5120x17408 224 vs decode + matmul 100); below it only the matrices of the good shape class
ZQ_PAD = int(os.environ.get("KYOJIN_FOREIGN_ZQ_PAD", "17"))              # 4-bit, other shapes: from here the quantized matmul runs padded to ZQ_HIGH rows
ZQ8_PAD = int(os.environ.get("KYOJIN_FOREIGN_ZQ8_PAD", "29"))            # fp8: same; below it mma8 (up to 32 rows, DENSE8_ROWS): 5120x6144 R24 258 vs padded 263, R32 223 vs 265 GB/s of stored bytes
ZQ_GOOD = int(os.environ.get("KYOJIN_FOREIGN_ZQ_GOOD", "17"))            # first row count of MLX's quantized matmul on the good shape class (4-bit and fp8)
F16_ROWS = int(os.environ.get("KYOJIN_FOREIGN_F16_ROWS", "16"))           # arm fp8stored=2: from here the resident fp16 copy runs one matmul (R16: 7270 vs mma8 6633 Gw/s)
GEMVK_ROWS = int(os.environ.get("KYOJIN_FOREIGN_GEMVK_ROWS", "8"))       # K formats (flag kstored): stored GEMV up to here (gate: GEMV ahead to R4, tie R5; real 27B solo with mmak from R5: 38-44 vs 47-57 tok/s, so GEMV keeps R5-8), then mmak
GEMVK_COPY_ROWS = int(os.environ.get("KYOJIN_FOREIGN_GEMVK_COPY_ROWS", "8"))   # with a resident fp16 copy: GEMV up to here on big matrices (real 27B Q4_K_M, rows 1 / 8: solo 40-48 / 47-54, N=4 same)
GEMVK_COPY_SMALL = 8 << 20                                                 # below it the copy serves every row (0.6B UD-IQ3_XXS solo 110 stored -> 280 copy)                                                 # under this many weights the fixed dispatch cost rules (0.6B IQ4_XS: GEMV R1 +8 %, at R4 N=4 cost 15-25 %): one row only
MMAK_ROWS = int(os.environ.get("KYOJIN_FOREIGN_MMAK_ROWS", "32"))        # K formats: stored simdgroup-matrix kernel up to here (gate: +30..36 % over decode + matmul at R32, below it at R48), then the copy / decode + matmul; 0 = no mmak
NBRK = int(os.environ.get("KYOJIN_FOREIGN_NBRK", "2"))
NBR = int(os.environ.get("KYOJIN_FOREIGN_NBR", "2"))
NTN = int(os.environ.get("KYOJIN_FOREIGN_NTN", "1"))
_FMMA_MIN_ENV = os.environ.get("KYOJIN_FOREIGN_FMMA_MIN")                      # one value for every format; unset = FMMA_MIN_FMT
# 11/11 retune after the K / IQ wrapper cache (a7341f4, the stored GEMV lost 25-46 us of host per call): the GEMV leads up to R4 on IQ3_XXS / IQ3_S /
# IQ2_XXS / IQ2_XS / IQ1_S (R4 GB/s fmma -> GEMV: 303 -> 341, 289 -> 355, 188 -> 255, 163 -> 263, 96 -> 207; IQ2_XS R2 162 -> 361) and drops from R5 (NBR 1)
FMMA_MIN_FMT = {"nvfp4": 2, "mxfp4": 2, "q4_k": 3, "q4_0": 3, "q5_k": 4, "q6_k": 4, "iq4_xs": 4, "q3_k": 4, "iq3_s": 5, "iq3_xxs": 5, "q2_k": 4, "iq2_xxs": 5, "iq2_xs": 5, "iq2_s": 3, "iq1_s": 5, "iq1_m": 3}               # gate 17408x5120, GB/s GEMV vs fmma: K formats R3 GEMV vs fmma over 3 shapes: q5_k 670/706 728/689 568/547, q6_k 779/744 855/861 663/635, iq4_xs 618/632 678/622 514/482 (min 4); fp4 R2 608/692 R4 314/701; q4_k R2 752/687 R3 667/694; q4_0 R2 773/736 R3 705/745
FMMA_MAX = int(os.environ.get("KYOJIN_FOREIGN_FMMA_MAX", "16"))   # flag foreign_fmma: rows served by the few-row MMA
WIDE_MIN = int(os.environ.get("KYOJIN_FOREIGN_WIDE_MIN", "9"))                # flag foreign_wide: rows served by the register-fed NAX product (foreign_wide.wide_source_reg) on stored bytes
WIDE_MAX_FMT = {"nvfp4": 24, "mxfp4": 24, "q4_0": 64, "q4_k": 64, "q5_k": 64, "q6_k": 64, "iq4_xs": 64, "fp8": 32, "q8_0": 64, "fp8b": 64}   # gate 17408x5120 / 5120x17408 / 5120x5120 (min of 14 interleaved reps): the K types, q4_0 and q8_0 beat decode + matmul to R64 (q8_0 2-3x), fp8 beats mma8 to R32, MLX's in-place qmm wins fp8 / fp4 above that
WIDE_MAX_ENV = os.environ.get("KYOJIN_FOREIGN_WIDE_MAX")
FMMA_NT, FMMA_WS = int(os.environ.get("KYOJIN_FOREIGN_FMMA_NT", "2")), int(os.environ.get("KYOJIN_FOREIGN_FMMA_WS", "4"))
GMM = os.environ.get("KYOJIN_FOREIGN_GMM", "1") not in ("0", "")             # prefill routed experts: sorted gathered matmul over the decoded stack (0 = one launch per expert)
KQ_MT = int(os.environ.get("KYOJIN_KQ_GMM_MT", "32"))                      # flag kqgmm: rows of one expert per tile
GMM_MAX_BYTES = int(float(os.environ.get("KYOJIN_FOREIGN_GMM_MAX_GIB", "3")) * 2 ** 30)   # largest transient fp16 stack
GATHER_MAX = int(os.environ.get("KYOJIN_FOREIGN_GATHER_MAX", "128"))       # routed-expert slots up to here: one gather launch (R = 1 per slot)


def cfmt() -> bool:
    """Flag cfmtspeed (KYOJIN_CFMTSPEED, default 0; read per kernel build, the grouping runs once at load): table decode of E2M1 + same-input groups."""
    return os.environ.get("KYOJIN_CFMTSPEED", "0") not in ("0", "")


def fp8stored() -> bool:
    """Flag fp8stored (KYOJIN_FP8STORED, default 2, read at load): FP8 E4M3 safetensors linears and GGUF Q8_0 tensors stay one byte per weight on the device and are
    read as stored (GEMV for few rows, simdgroup-matrix kernel for 9..56, decode + matmul above). 2 = the same plus a resident fp16 copy that runs the matmul from 16 rows.
    0 = decoded to fp16 at load, as before."""
    return os.environ.get("KYOJIN_FP8STORED", "1") not in ("0", "")


def fp8_f16copy() -> bool:
    return os.environ.get("KYOJIN_FP8STORED", "1") == "2"


def gguf_stream_f16() -> bool:
    """Flag gguf_fp16 (KYOJIN_GGUF_FP16, default 1, read at load): the 1-D F32 tensors of a GGUF (norm weights, biases) are loaded in the model dtype, as the BF16 ones of the
    same model in safetensors are. ggml writes them F32 whatever the file type; kept F32, q_norm and k_norm promote Q and K to fp32, so prefill attention ran 1.4x slower and
    decode missed the fp16 flash kernel. 0 = kept F32, as before."""
    return os.environ.get("KYOJIN_GGUF_FP16", "1") not in ("0", "")


def _f32_on_stream(name: str, v: mx.array) -> bool:
    return v.dtype == mx.float32 and v.ndim == 1 and not name.endswith("linear_attn.norm.weight")   # gdn_norm's fused kernel takes that one as fp32


def kstored() -> bool:
    """Flag kstored (KYOJIN_KSTORED, default 2, read at load): GGUF Q5_K / Q6_K / IQ4_XS matrices stay as stored on the device (GEMV for few rows). 2 = the same plus a
    resident fp16 copy for the rows above (kept only while it fits, _f16copy_fits). 0 = decoded to fp16 at load, as before."""
    return os.environ.get("KYOJIN_KSTORED", "1") not in ("0", "")


_FP8B_CACHE: dict = {}


def _fp8b_pack(ck, c, block):
    """Pack "fp8b" of a block-scaled E4M3 linear, or None when its blocks do not fit the 32-weight units (block k-size not a multiple of 32, k % 32).
    meta half [n, k/32] = the block scale of each unit times a power of two 2^p (the largest one lands in [64, 128): no fp16 subnormal, and x256 in the
    wide kernel stays finite); gs = 2^-p per row, exact. Memoised per weight name (the loader asks twice)."""
    key = c["weight"]
    if key in _FP8B_CACHE:
        return _FP8B_CACHE[key]
    _, shp, w8 = ck.raw(c["weight"])
    n, k = int(shp[0]), int(shp[1])
    g = np.asarray(ck.scale_f32(c["scale"]), np.float32).reshape(-1, 1) if np.ndim(ck.scale_f32(c["scale"])) < 2 else np.asarray(ck.scale_f32(c["scale"]), np.float32)
    bn, bk = (block if block else (-(-n // g.shape[0]), -(-k // g.shape[1])))
    pk = None
    if k % 32 == 0 and bk % 32 == 0 and -(-n // bn) == g.shape[0] and -(-k // bk) == g.shape[1] and np.all(np.isfinite(g)) and np.all(g > 0):
        rows = np.arange(n) // bn
        units = (np.arange(k // 32) * 32) // bk
        p2 = 6 - int(np.floor(np.log2(float(g.max()))))
        meta = (g[rows][:, units] * np.float32(2.0 ** p2)).astype(np.float16)
        pk = FP.Pack("fp8b", n, k, np.ascontiguousarray(w8).view(np.uint8).reshape(n, k), meta, np.float32(2.0 ** -p2))
    _FP8B_CACHE[key] = pk
    return pk


def k_f16copy() -> bool:
    return os.environ.get("KYOJIN_KSTORED", "1") == "2"


PAD_ROWS, PAD_MIN = 16, 8 << 20                                           # flag fp16pad: 8 x 8192^2 fp16 matmul ms plain / padded: R2 2.52/1.50, R8 2.05/1.18, R12 2.82/1.17, R16 1.18
_PAD = os.environ.get("KYOJIN_FP16PAD", "1") not in ("0", "")
_zeros: dict = {}


def _matmul(x2, w):
    """x2 [R, k] @ w.T as fp16. MLX's NAX path starts at 16 rows: below it, a big matrix takes the 16-row product over zero rows and keeps its own rows."""
    R, k = x2.shape[0], x2.shape[1]
    if not _PAD or R >= PAD_ROWS or w.shape[0] * k < PAD_MIN:
        return (x2 @ w.T).astype(mx.float16)
    z = _zeros.get((PAD_ROWS - R, k, x2.dtype))
    if z is None:
        z = _zeros[(PAD_ROWS - R, k, x2.dtype)] = mx.zeros((PAD_ROWS - R, k), x2.dtype)
        mx.eval(z)
    return (mx.concatenate([x2, z], axis=0) @ w.T)[:R].astype(mx.float16)


_ONES8 = {}
_scale_out = mx.compile(lambda y, s: (y.astype(mx.float32) * s).astype(mx.float16), shapeless=True)   # one fused kernel; fp32 scale: separate and grouped linears give the same bits


def f4_copy() -> bool:
    return os.environ.get("KYOJIN_F4COPY", "0") not in ("0", "")


def mq_copy() -> bool:
    return os.environ.get("KYOJIN_FOREIGN_MQCOPY", "0") not in ("0", "")


def _f16copy_fits(nbytes: int) -> bool:
    """The fp16 copy is speed at N sessions, not needed to serve: a model that fills the machine keeps its stored bytes only (mma8 serves those rows)."""
    return mx.get_active_memory() + nbytes <= 0.6 * mx.device_info().get("max_recommended_working_set_size", 0)


# ------------------------------------------------------------------------------------------------------------------------------ kernel launches
# rowgs: gs is one float per output row (a same-input group); lut: the E2M1 decode is a table read. Both False = the kernels of 3d5fd49 (same keys, names, text).
@functools.lru_cache(maxsize=None)
def _gemv_kern(fmt, nbr, gather, rowgs=False, lut=False, glu=False):
    tag = f"{'g' if gather else 's'}" + ("r" if rowgs else "") + ("l" if lut else "") + ("u" if glu else "")
    return KN._kern(("fgemv", fmt, nbr, gather) + ((rowgs, lut) if rowgs or lut else ()) + (("glu",) if glu else ()), name=f"kyojin_fg_gemv_{fmt}_n{nbr}{tag}",
                    input_names=FM.gemv_inputs(gather, glu), output_names=["y"], header=FM.header(fmt, "gemv", lut) + (FM.GLU_HELPER if glu else ""),
                    source=FM.gemv_source(fmt, nbr, gather, rowgs, glu))


DENSE_GLU = os.environ.get("KYOJIN_FOREIGN_DENSE_GLU", "1") not in ("0", "")   # dense gated MLP (dense layers, shared experts): silu(gate) * up inside the down projection's launch
PAIR = os.environ.get("KYOJIN_FOREIGN_PAIR", "1") not in ("0", "")      # gate | up gathers of a routed-expert block in one launch (DevPack.gather_pair)


def _pair_kern(tag, fmt, nbr, lut, names, mk):
    hdr, src = mk()
    src, pnames = FM.pair_gather(src, names)
    return KN._kern(("fpair", tag, fmt, nbr, lut), name=f"kyojin_fg_pair{tag}_{fmt}_n{nbr}", input_names=["x", "xrows", "eids"] + pnames, output_names=["y"], header=hdr, source=src)


@functools.lru_cache(maxsize=None)
def _mma_kern(fmt, ntn, rowgs=False, lut=False):
    return KN._kern(("fmma", fmt, ntn) + ((rowgs, lut) if rowgs or lut else ()), name=f"kyojin_fg_mma_{fmt}_t{ntn}" + ("r" if rowgs else "") + ("l" if lut else ""),
                    input_names=["xp", "qs", "mt", "gs"], output_names=["y"], header=FM.header(fmt, "mma", lut), source=FM.mma_source(fmt, ntn, rowgs))


def wide_on() -> bool:
    """Flag foreign_wide (KYOJIN_FOREIGN_WIDE): the register-fed NAX product for rows WIDE_MIN..WIDE_MAX. Read per call."""
    return os.environ.get("KYOJIN_FOREIGN_WIDE", "1") not in ("0", "")


def fmma_on() -> bool:
    """Flag foreign_fmma (KYOJIN_FOREIGN_FMMA): the few-row MMA of foreign_metal.fmma_source for rows FMMA_MIN..FMMA_MAX of the 4-bit formats. Read per call."""
    return os.environ.get("KYOJIN_FOREIGN_FMMA", "1") not in ("0", "")


@functools.lru_cache(maxsize=None)   # wide_header(fmt) formats the IQ grid tables as source text (28-115 us of host per R16 call)
def _wide_kern(fmt, rowgs):
    from . import foreign_wide as FW
    return KN._kern(("fwide", fmt, rowgs), name=f"kyojin_fg_wide_{fmt}" + ("r" if rowgs else ""), input_names=FW.wide_inputs(fmt), output_names=["y"],
                    header=FW.wide_header(fmt), source=FW.wide_source_reg(fmt, rowgs))


@functools.lru_cache(maxsize=None)   # fmma_header(fmt) formats the IQ grid tables as source text (28-115 us of host per R5-R16 call)
def _fmma_kern(fmt, rowgs=False):
    return KN._kern(("ffmma", fmt, rowgs), name=f"kyojin_fg_fmma_{fmt}" + ("r" if rowgs else ""), input_names=FM.fmma_inputs(fmt), output_names=["y"],
                    header=FM.fmma_header(fmt), source=FM.fmma_source(fmt, rowgs))


@functools.lru_cache(maxsize=None)
def _fmma_prep_kern():
    return KN._kern(("ffmma_prep",), name="kyojin_fg_fmma_prep", input_names=["x", "idx"], output_names=["xt"], source=FM.fmma_prep_source())


@functools.lru_cache(maxsize=None)
def _deq_kern(fmt, rowgs=False, lut=False):
    return KN._kern(("fdeq", fmt) + ((rowgs, lut) if rowgs or lut else ()), name=f"kyojin_fg_deq_{fmt}" + ("r" if rowgs else "") + ("l" if lut else ""),
                    input_names=["qs", "mt", "gs"], output_names=["w"], header=FM.header(fmt, "mma", lut), source=FM.dequant_source(fmt, rowgs))


@functools.lru_cache(maxsize=None)
def _gemv8_kern(fmt, nbr, gather=False, glu=False):
    return KN._kern(("fgemv8", fmt, nbr) + (("g",) if gather else ()) + (("glu",) if glu else ()), name=f"kyojin_fg_gemv8_{fmt}_n{nbr}" + ("g" if gather else "") + ("u" if glu else ""),
                    input_names=FM.gemv8_inputs(fmt, gather, glu), output_names=["y"], header=FM.header8(fmt) + (FM.GLU_HELPER if glu else ""),
                    source=FM.gemv8_source(fmt, nbr, gather, glu))


@functools.lru_cache(maxsize=None)
def _mma8_kern(fmt, ntn):
    return KN._kern(("fmma8", fmt, ntn), name=f"kyojin_fg_mma8_{fmt}_t{ntn}", input_names=["xp"] + FM.gemv8_inputs(fmt)[1:], output_names=["y"], header=FM.header8(fmt),
                    source=FM.mma8_source(fmt, ntn))


@functools.lru_cache(maxsize=None)
def _deq8_kern(fmt):
    return KN._kern(("fdeq8", fmt), name=f"kyojin_fg_deq8_{fmt}", input_names=FM.gemv8_inputs(fmt)[1:], output_names=["w"], header=FM.header8(fmt),
                    source=FM.dequant8_source(fmt))


# cached: the K / IQ headers rebuild their grid tables as source text (IQ3_S ~40 us of host time per call)
@functools.lru_cache(maxsize=None)
def _gemvk_kern(fmt, nbr, gather=False, glu=False):
    return KN._kern(("fgemvk", fmt, nbr) + (("g",) if gather else ()) + (("glu",) if glu else ()),
                    name=f"kyojin_fg_gemvk_{fmt}_n{nbr}" + ("g" if gather else "") + ("u" if glu else ""),
                    input_names=FM.gemvk_inputs(fmt, gather, glu), output_names=["y"], header=FM.hdrk(fmt) + (FM.GLU_HELPER if glu else ""),
                    source=FM.gemvk_source(fmt, nbr, gather, glu))


# cached: the K / IQ headers rebuild their grid tables as source text (IQ3_S ~40 us of host time per call)
@functools.lru_cache(maxsize=None)
def _deqk_kern(fmt):
    return KN._kern(("fdeqk", fmt), name=f"kyojin_fg_deqk_{fmt}", input_names=FM.dequantk_inputs(fmt), output_names=["w"], header=FM.hdrk(fmt), source=FM.dequantk_source(fmt))


# cached: the K / IQ headers rebuild their grid tables as source text (IQ3_S ~40 us of host time per call)
@functools.lru_cache(maxsize=None)
def _mmak_kern(fmt, ntn):
    return KN._kern(("fmmak", fmt, ntn), name=f"kyojin_fg_mmak_{fmt}_t{ntn}", input_names=["xp"] + FM.gemvk_inputs(fmt)[1:], output_names=["y"], header=FM.hdrk(fmt),
                    source=FM.mmak_source(fmt, ntn))


_idx_cache: dict = {}


def _xperm_idx(fmt, k):
    key = (FP.LAYOUT[fmt], k)
    if key not in _idx_cache:
        _idx_cache[key] = mx.array(FP.xperm_index(fmt, k))
    return _idx_cache[key]


def _xpermk_idx(fmt, k):
    key = ("k", FM._LAYOUTK[fmt], k)
    if key not in _idx_cache:
        _idx_cache[key] = mx.array(FM.xpermk_index(fmt, k))
    return _idx_cache[key]


def _xperm8_idx(k):
    key = ("8", k)
    if key not in _idx_cache:
        _idx_cache[key] = mx.array(FM.xperm8_index(k))
    return _idx_cache[key]


# Prepped x of the input object ForeignLinear.__call__ is serving: [x, {(perm key, R, XW): xt}, active]. The slot holds x, so its id cannot be reused while it is
# current; a different input object starts empty. Only apply() calls made inside ForeignLinear.__call__ use it (GroupMember, direct apply do not).
_PREP = [None, {}, False]


class DevPack:
    """The arrays of one weight matrix (or a stack of E expert matrices, leading axis) on the device."""
    _zv = None                                             # cached MLX-layout views of the stored 4-bit arrays
    _mqs = None
    _mqc = None                                            # power-of-two input scale of the nvfp4 quantized copy (set at first use)
    mq = None                                              # flag mqcopy: (wq, scales, biases, mode, group, unit_gs, bits) resident MLX-native quantized copy, 4-bit formats only

    def __init__(self, packs):
        ps = packs if isinstance(packs, list) else [packs]
        p0 = ps[0]
        self.fmt, self.n, self.k, self.E = p0.fmt, p0.n, p0.k, (len(ps) if isinstance(packs, list) else None)
        self.rowgs = False                                   # True: gs holds one scale per output row (a same-input group, see group_same_input)
        assert all((p.fmt, p.n, p.k) == (self.fmt, self.n, self.k) for p in ps), "experts differ in format / shape"
        if self.fmt == "dense":
            self.w = mx.array(np.stack([p.qs for p in ps]) if self.E else p0.qs)
            self.qs = self.meta = self.gs = None
            self.w16 = None
            return
        if self.fmt in FM.FORMATS8:                          # one byte per weight; gs = one float per output row (scale of the tensor / channel, ones for q8_0)
            if self.E:                                       # routed experts: the E matrices stacked as E * n rows, as stored (no fp16 copy)
                self.qs = mx.array(np.concatenate([np.ascontiguousarray(p.qs).view(np.uint32).reshape(p.n, p.k // 4) for p in ps]))
                self.meta = None if p0.meta is None else mx.array(np.concatenate([np.ascontiguousarray(p.meta) for p in ps]))
                self.gs = mx.array(np.concatenate([np.broadcast_to(np.asarray(p.gscale, np.float32).reshape(-1), (p.n,)) for p in ps]))
                self.w, self.rowgs, self._views, self.w16 = None, True, {}, None
                return
            self.qs = mx.array(np.ascontiguousarray(p0.qs).view(np.uint32).reshape(p0.n, p0.k // 4))
            self.meta = None if p0.meta is None else mx.array(np.ascontiguousarray(p0.meta))
            self.gs = mx.array(np.ascontiguousarray(np.broadcast_to(np.asarray(p0.gscale, np.float32).reshape(-1), (p0.n,))))
            self.w, self.rowgs, self._views, self.w16 = None, True, {}, None
            if fp8_f16copy() and _f16copy_fits(p0.n * p0.k * 2):  # arm 2: resident fp16 copy (3x the bytes), the matmul from F16_ROWS rows
                self.w16 = self.dequant()
                mx.eval(self.w16)
            return
        if self.fmt in FM.FORMATSK:                          # GGUF K / IQ blocks as stored: nibbles qs, scale words meta, high bits hi
            cat = (lambda a: np.concatenate(a)) if self.E else (lambda a: a[0])    # experts: E matrices stacked as E * n rows, as stored (no fp16 / mq copy)
            self.qs = mx.array(cat([np.ascontiguousarray(p.qs).view(np.uint32).reshape(p.n, -1) for p in ps]))
            self.meta = mx.array(cat([np.ascontiguousarray(p.meta) for p in ps]))
            self.hi = mx.array(cat([np.ascontiguousarray(p.extra["hi"]) for p in ps])) if "hi" in p0.extra else None
            self.gs, self.w, self.rowgs, self._views, self.w16 = None, None, False, {}, None
            if self.E:
                return
            if mq_copy() and MQ_Q5K and self.fmt == "q5_k" and self.hi is not None and _f16copy_fits(p0.n * p0.k * 7 // 10):   # flag mqcopy: ~1.1x the stored bytes
                wq, sc, bi, mode, g, bits = FP.mq_arrays(p0)
                self.mq = (mx.array(wq), mx.array(sc), mx.array(bi), mode, g, True, bits)
                mx.eval(list(self.mq[:3]))
            if k_f16copy() and _f16copy_fits(p0.n * p0.k * 2):
                self.w16 = self.dequant()
                mx.eval(self.w16)
            return
        st = (lambda a: np.stack(a)) if self.E else (lambda a: a[0])
        self.qs = mx.array(np.ascontiguousarray(st([np.ascontiguousarray(p.qs).view(np.uint32).reshape(p.n, p.k // 8) for p in ps])))
        self.meta = mx.array(np.ascontiguousarray(st([np.ascontiguousarray(p.meta) for p in ps])))
        self.gs = mx.array(np.array([float(p.gscale) for p in ps], np.float32))
        self.w = None
        self.w16 = None
        self._views = {}
        if not self.E and mq_copy() and p0.fmt in ("q4_0", "q4_k", "nvfp4", "mxfp4") and _f16copy_fits(p0.n * p0.k * 5 // 8):   # ~1.1x the stored bytes
            wq, sc, bi, mode, g, bits = FP.mq_arrays(p0)
            self.mq = (mx.array(wq), mx.array(sc), None if bi is None else mx.array(bi), mode, g, mode != "nvfp4" or float(p0.gscale) == 1.0, bits)
            mx.eval([a for a in self.mq[:3] if a is not None])
        if not self.E and f4_copy() and _f16copy_fits(p0.n * p0.k * 2):  # flag f4copy: resident fp16 copy (4x the bytes), the matmul from F4_ROWS rows
            self.w16 = self.dequant()
            mx.eval(self.w16)

    @classmethod
    def grouped(cls, devs):
        """One DevPack over the concatenated rows of same-input DevPacks (the device twin of foreign_pack.concat_rows: the members' bytes one after the other,
        gs = each member's global scale repeated over its rows). The caller drops the members afterwards."""
        d0 = devs[0]
        assert d0.fmt != "dense" and all((d.fmt, d.k, d.E) == (d0.fmt, d0.k, None) for d in devs), "grouped: one non-dense format, one k, no expert stacks"
        d = object.__new__(cls)
        d.fmt, d.k, d.E, d.n, d.w, d.rowgs, d._views = d0.fmt, d0.k, None, sum(m.n for m in devs), None, True, {}
        d.qs = mx.concatenate([m.qs for m in devs], axis=0)
        d.meta = None if d0.meta is None else mx.concatenate([m.meta for m in devs], axis=0)
        d.gs = mx.concatenate([m.gs if m.rowgs else mx.broadcast_to(m.gs.reshape(1), (m.n,)) for m in devs], axis=0)
        d.w16 = mx.concatenate([m.w16 for m in devs], axis=0) if all(m.w16 is not None for m in devs) else None
        d.mq = None
        if d0.mq is not None and all(m.mq is not None and m.mq[3:5] == d0.mq[3:5] and m.mq[6] == d0.mq[6] for m in devs):
            d.mq = (mx.concatenate([m.mq[0] for m in devs], axis=0), mx.concatenate([m.mq[1] for m in devs], axis=0),
                    None if d0.mq[2] is None else mx.concatenate([m.mq[2] for m in devs], axis=0), d0.mq[3], d0.mq[4], all(m.mq[5] for m in devs), d0.mq[6])
        return d

    def rows(self, n):
        """A DevPack over output rows (no expert stack): n = an int (the first n rows) or an int array of row ids (gathered in that order). Every per-row array
        is cut on its leading axis and made resident; the global scale stays."""
        assert not self.E, "rows: one matrix"
        if isinstance(n, int):
            assert 0 < n <= self.n, "rows: 0 < n <= N"
            cut = lambda a: None if a is None else a[:n] + 0
        else:
            idx = mx.array(n, dtype=mx.int32)
            assert idx.ndim == 1 and 0 < idx.size <= self.n, "rows: 1-D index, 0 < len <= N"
            cut = lambda a: None if a is None else mx.take(a, idx, axis=0)
            n = idx.size
        d = object.__new__(type(self))
        d.fmt, d.n, d.k, d.E, d.rowgs, d._views, d.mq = self.fmt, n, self.k, None, self.rowgs, {}, None
        d.w, d.qs, d.meta, d.w16 = cut(self.w), cut(self.qs), cut(self.meta), cut(self.w16)
        d.gs = cut(self.gs) if self.rowgs else self.gs
        if hasattr(self, "hi"):
            d.hi = cut(self.hi)
        if self.mq is not None:
            wq, sc, bi, mode, g, unit, bits = self.mq
            d.mq = (cut(wq), cut(sc), cut(bi), mode, g, unit, bits)
        mx.eval(d.arrays())
        return d

    def arrays(self):
        return [a for a in (self.qs, self.meta, self.gs, self.w, self.w16, getattr(self, "hi", None)) if a is not None] + ([a for a in self.mq[:3] if a is not None] if self.mq else [])

    def nbytes(self):
        return sum(a.nbytes for a in (self.qs, self.meta, self.gs, getattr(self, "hi", None)) if a is not None)

    def expert(self, e):
        v = self._views.get(e) if self.w is None else None
        if v is None:
            v = object.__new__(DevPack)
            v.fmt, v.n, v.k, v.E, v.rowgs, v.w16 = self.fmt, self.n, self.k, None, False, None
            if self.w is not None:
                v.w, v.qs, v.meta, v.gs = self.w[e], None, None, None
            elif self.fmt in FM.FORMATSK or self.fmt in FM.FORMATS8:       # stacked as E * n rows: the expert is a row range
                r = slice(e * self.n, (e + 1) * self.n)
                v.w, v.qs, v.meta, v._views, v.mq = None, self.qs[r], None if self.meta is None else self.meta[r], {}, None
                v.hi = None if getattr(self, "hi", None) is None else self.hi[r]
                v.gs, v.rowgs = (None, False) if self.fmt in FM.FORMATSK else (self.gs[r], True)
                self._views[e] = v
            else:
                # qs[e] on the [E, n, ...] stack is a slice + squeeze that MLX materialises (0.5 MB per expert and projection, 8.6 GB for a 40-layer MoE after one
                # prefill); the row range of the flattened stack stays a view of the one resident copy
                v.w, v.qs, v.meta, v.gs, v._views = None, self.qs.reshape(-1, self.qs.shape[-1])[e * self.n:(e + 1) * self.n], self.meta.reshape(-1, self.meta.shape[-1])[e * self.n:(e + 1) * self.n], self.gs[e:e + 1], {}
                self._views[e] = v
        return v

    # ---- one matrix, x2 [R, k] fp16 -> [R, n] fp16
    def gemv_ok(self, nbr=None):
        return (self.k % 512 == 0 or (self.fmt in FM.FORMATS8 and self.k % 32 == 0)) and self.n % (8 * (nbr or NBR)) == 0

    def apply(self, x2):
        R = x2.shape[0]
        if self.fmt == "dense":
            return _matmul(x2, self.w)
        if self.wide_ok(R):
            return self._wide(x2)
        if self.fmt in FM.FORMATSK:
            cp = self.w16 is not None
            if self.fmma_ok(R):
                return self._fmma(x2)
            if R <= ((GEMVK_COPY_ROWS if self.k * self.n >= GEMVK_COPY_SMALL else 0) if cp else GEMVK_ROWS) and self.gemv_ok(NBRK):    # gemv_ok(2): n % 16 also covers NBR 1
                return self._gemvk(x2)
            if self.mq is not None and self.mq_rows_ok(R):
                return self._mqmm(x2)
            if cp:                                       # real 27B, arm 2: mmak at rows 2..15 cost 10 % solo, N=4 equal; the copy's matmul serves them
                return _matmul(x2, self.w16)
            if R <= MMAK_ROWS and self.fmt in FM.FORMATSK_MMA:
                return self._mmak(x2, 1 if R <= 8 else 2)
            return (x2 @ (self.w16 if cp else self.dequant()).T).astype(mx.float16)
        if self.fmt in FM.FORMATS8:
            if R <= GEMV8_ROWS and self.gemv_ok():
                return self._gemv8(x2)
            if self.w16 is not None and R >= F16_ROWS:
                return _matmul(x2, self.w16)
            if self._zq8_ok(R):
                return self._zq8(x2)
            if R <= DENSE8_ROWS:
                return self._mma8(x2) if R <= MMA8_SLICE else mx.concatenate([self._mma8(x2[i:i + MMA8_SLICE]) for i in range(0, R, MMA8_SLICE)], axis=0)
            return (x2 @ self.dequant().T).astype(mx.float16)
        if self.mq is not None and self.mq_rows_ok(R):
            return self._mqmm(x2)
        if self._zq_route(R):
            return self._zq(x2)
        if self._f2_ok(R):   # two 16-row few-row MMA launches (one weight read each) beat decode + matmul (Q4_K 17408x5120 R24 224 vs 115 GB/s of stored bytes)
            return mx.concatenate([self._fmma(x2[:16], share=False), self._fmma(x2[16:], share=False)], axis=0)   # row slices of one input: no shared prep
        if self.w16 is not None and R >= F4_ROWS:
            return _matmul(x2, self.w16)
        if self.fmma_ok(R):
            return self._fmma(x2)
        if R <= GEMV_ROWS and self.gemv_ok():
            return self._gemv(x2)
        if R <= MMA_MAX:
            return self._mma(x2)
        if R <= DENSE_ROWS:
            return mx.concatenate([self._mma(x2[i:i + 16]) for i in range(0, R, 16)], axis=0)
        return (x2 @ self.dequant().T).astype(mx.float16)

    def apply_split(self, x2, cuts):
        """apply() for a same-input group: the columns [cuts[i], cuts[i+1]) of the result are member i's output. Up to DENSE_ROWS the group is one product (rows are
        independent). Above, the group is decoded once and each member runs the matmul it ran alone, on its own rows of the decoded matrix: the same shapes, the
        same MLX kernel. Returns the list of per-member outputs."""
        R = x2.shape[0]
        if R <= DENSE_ROWS or self.wide_stored(R):
            y = self.apply(x2)
            return [y[:, a:b] for a, b in zip(cuts[:-1], cuts[1:])]
        w = self.w16 if self.w16 is not None else self.dequant()
        return [(x2 @ w[a:b].T).astype(mx.float16) for a, b in zip(cuts[:-1], cuts[1:])]

    def mq_rows_ok(self, R):
        if not MQ_MIN <= R <= MQ_MAX or self.n * self.k < MQ_SMALL:
            return False
        return R > MQ_QMV_ROWS or (self.n >= MQ_QMV_N and self.k <= MQ_QMV_K)

    def _mqmm(self, x2):
        """x2 [R, k] on the MLX-native quantized copy; the nvfp4 global scale (one float, or one per row of a group) multiplies the result."""
        wq, sc, bi, mode, g, unit, bits = self.mq
        if x2.dtype != mx.float16:                     # fp32 inputs (ssm_out, attention outputs) would run MLX's fp32 kernel and leak fp32 downstream: saturate to fp16
            x2 = mx.clip(x2, -65504.0, 65504.0).astype(mx.float16)
        x2 = mx.contiguous(x2)                         # a strided view (a column slice of a fused output) sends MLX's qmm to a slow path: R48 5120x6144 b5 3.50 vs 2.59 ms per 32 with the copy
        if unit:
            return mx.quantized_matmul(x2, wq, sc, bi, transpose=True, group_size=g, bits=bits, mode=mode)
        return self._qmm_gs(x2, wq, sc, g, bits, mode, bi)

    def _good_class(self):
        """The shapes MLX's quantized matmul serves fast at 12..32 rows (wide output, short input); the others take its slow path there."""
        return self.n >= MQ_QMV_N and self.k <= MQ_QMV_K and self.n * self.k >= MQ_SMALL

    def _zq_route(self, R):
        """Rows -> path for the 4-bit block formats read in place by MLX (flag zq): "q" = quantized matmul, None = the usual path."""
        if not ZQ or self.fmt not in ("nvfp4", "mxfp4") or self.E or self.k % 128:
            return None
        if self.n * self.k < MQ_SMALL:
            return None
        if R <= ZQ_LOW:
            return "q"
        if R > MQ_MAX:                                   # prefill chunks: decode + one big MLX matmul
            return None
        if R >= ZQ_HIGH or (R >= ZQ_GOOD and self._good_class()):
            return "q"
        return "q" if R >= ZQ_PAD else None               # the other shapes: MLX's small-row kernel is slow, padded to ZQ_HIGH rows its large-row kernel runs (5120x17408 R20-32 217 vs two fmma 160)

    def _f2_ok(self, R):
        return 16 < R <= 32 and self.w16 is None and self.fmma_ok(16)

    def wide_stored(self, R):
        """Above DENSE_ROWS: the rows the stored bytes still serve in one product (MLX quantized matmul, two few-row MMA launches)."""
        return bool(self.wide_ok(R) or self._zq_route(R) or self._f2_ok(R) or self._zq8_ok(R) or (self.fmt in FM.FORMATS8 and R <= DENSE8_ROWS))

    def _zq_view(self):
        """(codes, scales) in MLX's own layout, views of the stored arrays (the byte order is the same: element 2b in the low nibble, one UE4M3 / E8M0 byte per block)."""
        v = self._zv
        if v is None:
            g = 16 if self.fmt == "nvfp4" else 32
            v = self._zv = (self.qs, mx.view(self.meta, mx.uint8).reshape(self.n, self.k // g))
        return v

    def _in_scale(self):
        """(c, s): power-of-two input scale c and output scale s = gscale / c (fp32, exact: c is a power of two; one per row or one). The block scales carry 1/gscale (nvfp4 ~1e4, fp8 ~1e2), so the raw
        fp16 product is ~1/gscale x the result and overflows (the "!!!!" output of the nvfp4 copy). Scaling the input by ~sqrt(gscale) keeps it ~1/sqrt(gscale) x; exact in fp16."""
        if self._mqc is None:
            c = float(min(1.0, 2.0 ** math.floor(math.log2(max(float(mx.max(self.gs)), 1e-30)) / 2)))
            self._mqs = self.gs / c
            mx.eval(self._mqs)
            self._mqc = c
        return self._mqc, self._mqs

    def _qmm_gs(self, x2, wq, sc, g, bits, mode, bi=None):
        R = x2.shape[0]
        if x2.dtype != mx.float16:
            x2 = mx.clip(x2, -65504.0, 65504.0).astype(mx.float16)
        c, s = self._in_scale()
        x2 = mx.contiguous(x2)
        if c != 1.0:
            x2 = x2 * c
        pad = 16 < R < ZQ_HIGH and not self._good_class()
        if pad:
            z = _zeros.get((ZQ_HIGH - R, self.k, mx.float16))
            if z is None:
                z = _zeros[(ZQ_HIGH - R, self.k, mx.float16)] = mx.zeros((ZQ_HIGH - R, self.k), mx.float16)
                mx.eval(z)
            x2 = mx.concatenate([x2, z], axis=0)
        y = mx.quantized_matmul(x2, wq, sc, bi, transpose=True, group_size=g, bits=bits, mode=mode)
        return _scale_out((y[:R] if pad else y), s)

    def _zq(self, x2):
        wq, sc = self._zq_view()
        return self._qmm_gs(x2, wq, sc, 16 if self.fmt == "nvfp4" else 32, 4, self.fmt)

    def _zq8_ok(self, R):
        if not ZQ or self.fmt != "fp8" or self.meta is not None or self.k % 32 or self.n * self.k < MQ_SMALL:
            return False
        return R <= MQ_MAX and (R >= ZQ_HIGH or R >= ZQ8_PAD or (R >= ZQ_GOOD and self._good_class()))

    def _zq8(self, x2):
        key = (self.n, self.k)
        sc = _ONES8.get(key)
        if sc is None:                                   # E8M0 127 = 1.0: the same constant for every matrix of one shape
            sc = _ONES8[key] = mx.full((self.n, self.k // 32), 127, mx.uint8)
            mx.eval(sc)
        return self._qmm_gs(x2, self.qs, sc, 32, 8, "mxfp8")

    def _gemv(self, x2, nbr=None):
        R, NBR = x2.shape[0], nbr or globals()['NBR']
        return _gemv_kern(self.fmt, NBR, False, self.rowgs, cfmt())(inputs=[x2, self.qs, self.meta, self.gs], template=[("R", R), ("KIN", self.k), ("N", self.n), ("NBR", NBR)],
                                               grid=(self.n // (8 * NBR) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(R, self.n)],
                                               output_dtypes=[mx.float16])[0]

    def _gemvk(self, x2, nbr=None):
        R = x2.shape[0]
        NBR = nbr or (NBRK if R <= 4 else 1)
        ins = [x2, self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
        return _gemvk_kern(self.fmt, NBR)(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NBR", NBR)], grid=(self.n // (8 * NBR) * 128, 1, 1),
                                          threadgroup=(128, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def _mmak(self, x2, ntn=2):
        R = x2.shape[0]
        ins = [mx.take(x2, _xpermk_idx(self.fmt, self.k), axis=1), self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
        return _mmak_kern(self.fmt, ntn)(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NTN", ntn)], grid=(-(-self.n // (32 * ntn)) * 128, 1, 1),
                                         threadgroup=(128, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def _gemv8(self, x2, nbr=None):
        R, NBR = x2.shape[0], nbr or globals()['NBR']
        ins = [x2, self.qs, self.gs] if self.meta is None else [x2, self.qs, self.meta, self.gs]
        return _gemv8_kern(self.fmt, NBR)(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NBR", NBR)], grid=(self.n // (8 * NBR) * 128, 1, 1),
                                          threadgroup=(128, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def glu_dense_ok(self, R) -> bool:
        """apply(gate|up) -> silu * mul -> apply(self) can run as ONE launch for self: the one-row GEMV class of this format takes R rows (same kernel family as apply())."""
        if not DENSE_GLU or self.E or self.fmt == "dense" or self.wide_ok(R):
            return False
        if self.fmt in FM.FORMATSK:
            lim = (GEMVK_COPY_ROWS if self.k * self.n >= GEMVK_COPY_SMALL else 0) if self.w16 is not None else GEMVK_ROWS     # the same test as apply()
            return not self.fmma_ok(R) and R <= lim and self.gemv_ok(NBRK)
        if self.fmt in FM.FORMATS8:
            return R <= GEMV8_ROWS and self.gemv_ok()
        # the 4-bit classes: the same chain of tests as apply() up to its _gemv
        return (not (self.mq is not None and self.mq_rows_ok(R)) and not self._zq_route(R) and not self._f2_ok(R) and not (self.w16 is not None and R >= F4_ROWS)
                and not self.fmma_ok(R) and R <= GEMV_ROWS and self.gemv_ok())

    def apply_glu(self, a, b, boff=0, stride=None):
        """apply() of the down projection of a gated MLP with the input silu(a) * b (half) computed while the input is staged. Rows of a and b are `stride` halves
        apart (default k); b starts boff halves after b's base (a = b = the [R, 2 k] gate | up product: stride 2 k, boff k)."""
        R = a.shape[0]
        st = stride or self.k
        tpl = [("R", R), ("KIN", self.k), ("N", self.n), ("AST", st), ("BST", st), ("BOFF", boff)]
        if self.fmt in FM.FORMATSK:
            NB = NBRK if R <= 4 else 1
            ins = [a, b, self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
            return _gemvk_kern(self.fmt, NB, False, True)(inputs=ins, template=tpl + [("NBR", NB)], grid=(self.n // (8 * NB) * 128, 1, 1), threadgroup=(128, 1, 1),
                                                          output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]
        if self.fmt not in FM.FORMATS8:
            return _gemv_kern(self.fmt, NBR, False, self.rowgs, cfmt(), True)(inputs=[a, b, self.qs, self.meta, self.gs], template=tpl + [("NBR", NBR)],
                                                                             grid=(self.n // (8 * NBR) * 128, 1, 1), threadgroup=(128, 1, 1),
                                                                             output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]
        ins = [a, b, self.qs, self.gs] if self.meta is None else [a, b, self.qs, self.meta, self.gs]
        return _gemv8_kern(self.fmt, NBR, False, True)(inputs=ins, template=tpl + [("NBR", NBR)], grid=(self.n // (8 * NBR) * 128, 1, 1), threadgroup=(128, 1, 1),
                                                       output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def _mma8(self, x2, ntn=2):
        R = x2.shape[0]
        ins = [mx.take(x2, _xperm8_idx(self.k), axis=1), self.qs, self.gs] if self.meta is None else [mx.take(x2, _xperm8_idx(self.k), axis=1), self.qs, self.meta, self.gs]
        return _mma8_kern(self.fmt, ntn)(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NTN", ntn)], grid=(-(-self.n // (32 * ntn)) * 128, 1, 1),
                                         threadgroup=(128, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def wide_ok(self, R):
        hi = int(WIDE_MAX_ENV) if WIDE_MAX_ENV else WIDE_MAX_FMT.get(self.fmt, 32)
        return wide_on() and WIDE_MIN <= R <= hi and not self.E and self.k % 128 == 0 and (self.fmt in FM._FMMA_PAIR or self.fmt in FM.FORMATS8) and self.w16 is None

    def _wide(self, x2, share=True):
        """x2 [R, k] -> [R, n] fp16 on the stored bytes: foreign_wide.wide_source_reg (one weight read for all rows, NAX matmul, fp32 accumulation)."""
        R = x2.shape[0]
        if x2.dtype != mx.float16:                       # fp32 inputs would feed an fp32 kernel input: saturate to fp16
            x2 = mx.clip(x2, -65504.0, 65504.0).astype(mx.float16)
        key = ("w", FW.wide_layout(self.fmt), self.k)
        if key not in _idx_cache:
            _idx_cache[key] = mx.array(FW.wide_xperm_index_reg(self.fmt, self.k))
        pk = (key, R)
        xp = _PREP[1].get(pk) if _PREP[2] and share else None        # siblings of one input share one permuted copy
        if xp is None:
            xp = mx.take(x2, _idx_cache[key], axis=1)
            if _PREP[2] and share:
                _PREP[1][pk] = xp
        MT = FW.mt_for(R)
        ws = 3 if MT >= 48 else 4                          # simdgroups per threadgroup split k (3 with the 48-row tile: registers)
        rc = min(MT, 7168 // (ws * 32))                    # rows per reduction pass: 28 KB of threadgroup memory at most
        names = FW.wide_inputs(self.fmt)
        arrs = {"x": xp, "qs": self.qs, "mt": self.meta, "gs": self.gs, "hi": getattr(self, "hi", None)}
        kern = _wide_kern(self.fmt, self.rowgs)
        return kern(inputs=[arrs[a] for a in names], template=[("MT", MT), ("KIN", self.k), ("N", self.n), ("WS", ws), ("RC", rc)],
                    grid=(-(-self.n // 32) * 32 * ws, -(-R // MT), 1), threadgroup=(32 * ws, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def fmma_ok(self, R):
        lo = int(_FMMA_MIN_ENV) if _FMMA_MIN_ENV else FMMA_MIN_FMT.get(self.fmt, 5)
        return fmma_on() and lo <= R <= FMMA_MAX and not self.E and self.k % 128 == 0 and self.fmt in FM._FMMA_PAIR

    def _fmma(self, x2, nt=None, ws=None, share=True):
        R, NT, WS = x2.shape[0], nt or FMMA_NT, ws or FMMA_WS
        XW = 8 if R <= 8 else 16
        key = ("f", FM._FMMA_LAYOUT[self.fmt], self.k)                # the permutation depends on the layout letter only
        if key not in _idx_cache:
            _idx_cache[key] = mx.array(FM.fmma_xperm_index(self.fmt, self.k))
        pk = (key, R, XW)
        xt = _PREP[1].get(pk) if _PREP[2] and share else None        # siblings of one input (q/k/v, gate/up) share one prep
        if xt is None:
            xt = _fmma_prep_kern()(inputs=[x2, _idx_cache[key]], template=[("R", R), ("KIN", self.k), ("XW", XW)], grid=(self.k, 1, 1), threadgroup=(256, 1, 1),
                                   output_shapes=[(self.k, XW)], output_dtypes=[mx.float16])[0]
            if _PREP[2] and share:
                _PREP[1][pk] = xt
        ntg = -(-self.n // (8 * NT))
        ins = [xt, self.qs, self.meta] + ([self.hi] if "hi" in FM.fmma_inputs(self.fmt) else []) + ([] if self.fmt in FM._FMMA_K else [self.gs])
        return _fmma_kern(self.fmt, self.rowgs)(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NT", NT), ("WS", WS), ("XW", XW)],
                                                grid=(ntg * 32 * WS, 1, 1), threadgroup=(32 * WS, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def _mma(self, x2, ntn=None):
        R, NTN = x2.shape[0], ntn or globals()['NTN']
        xp = mx.take(x2, _xperm_idx(self.fmt, self.k), axis=1)
        ntg = -(-self.n // (8 * NTN * 4))
        return _mma_kern(self.fmt, NTN, self.rowgs, cfmt())(inputs=[xp, self.qs, self.meta, self.gs], template=[("R", R), ("KIN", self.k), ("N", self.n), ("NTN", NTN)],
                                        grid=(ntg * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def dequant(self):
        if self.fmt == "dense":
            return self.w
        if self.fmt in FM.FORMATSK:
            ins = [self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
            return _deqk_kern(self.fmt)(inputs=ins, template=[("KIN", self.k), ("N", self.n)], grid=(-(-self.n * (self.k // 32) // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
                                        output_shapes=[(self.n, self.k)], output_dtypes=[mx.float16])[0]
        if self.fmt in FM.FORMATS8:
            ins = [self.qs, self.gs] if self.meta is None else [self.qs, self.meta, self.gs]
            return _deq8_kern(self.fmt)(inputs=ins, template=[("KIN", self.k), ("N", self.n)], grid=(self.n * self.k // 4, 1, 1), threadgroup=(256, 1, 1),
                                        output_shapes=[(self.n, self.k)], output_dtypes=[mx.float16])[0]
        return _deq_kern(self.fmt, self.rowgs, cfmt())(inputs=[self.qs, self.meta, self.gs], template=[("KIN", self.k), ("N", self.n)],
                                   grid=(self.n * (self.k // 32) * 4, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(self.n, self.k)],
                                   output_dtypes=[mx.float16])[0]

    # ---- routed-expert slots: slot s uses expert eids[s] and row xrows[s] of x (half [T, k]) -> [S, n]
    def gather(self, x, xrows, eids):
        S = int(eids.shape[0])
        assert self.E and self.fmt != "dense" and self.gemv_ok()
        nbr = NBRK if self.fmt in FM.FORMATSK else NBR                   # the rows per simdgroup the one-row product of this class uses
        grid = (self.n // (8 * nbr) * 128, S, 1)
        tpl = [("R", 1), ("KIN", self.k), ("N", self.n), ("NBR", nbr)]
        if self.fmt in FM.FORMATSK:
            ins = [x, xrows, eids, self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
            return _gemvk_kern(self.fmt, nbr, True)(inputs=ins, template=tpl, grid=grid, threadgroup=(128, 1, 1), output_shapes=[(S, self.n)], output_dtypes=[mx.float16])[0]
        if self.fmt in FM.FORMATS8:
            ins = [x, xrows, eids, self.qs, self.gs] if self.meta is None else [x, xrows, eids, self.qs, self.meta, self.gs]
            return _gemv8_kern(self.fmt, nbr, True)(inputs=ins, template=tpl, grid=grid, threadgroup=(128, 1, 1), output_shapes=[(S, self.n)], output_dtypes=[mx.float16])[0]
        return _gemv_kern(self.fmt, nbr, True, False, cfmt())(inputs=[x, xrows, eids, self.qs, self.meta, self.gs], template=tpl, grid=grid, threadgroup=(128, 1, 1),
                                               output_shapes=[(S, self.n)], output_dtypes=[mx.float16])[0]

    def pair_ok(self, o) -> bool:
        """gate | up of one expert block can share a launch: same format, shape, expert count and array dtypes."""
        f = lambda d: (d.fmt, d.k, d.n, d.E, d.qs.dtype, d.qs.shape, None if d.meta is None else (d.meta.dtype, d.meta.shape), getattr(d, "hi", None) is None, getattr(d, "gs", None) is None)
        return PAIR and bool(self.E) and self.fmt != "dense" and self.gemv_ok() and f(self) == f(o)

    def gather_pair(self, o, x, xrows, eids):
        """(self.gather(...), o.gather(...)) in ONE launch (the two matrices are bound as stored; rows [0, S) of the output are self's, [S, 2 S) are o's)."""
        S = int(eids.shape[0])
        nbr = NBRK if self.fmt in FM.FORMATSK else NBR
        tpl = [("R", 1), ("KIN", self.k), ("N", self.n), ("NBR", nbr), ("SP", S)]
        grid = (self.n // (8 * nbr) * 128, 2 * S, 1)
        if self.fmt in FM.FORMATSK:
            names, mk = FM.gemvk_inputs(self.fmt, True)[3:], (lambda: (FM.hdrk(self.fmt), FM.gemvk_source(self.fmt, nbr, True, False)))
            arrs = lambda d: [d.qs, d.meta] + ([d.hi] if d.hi is not None else [])
            tag = "k"
        elif self.fmt in FM.FORMATS8:
            names, mk = FM.gemv8_inputs(self.fmt, True)[3:], (lambda: (FM.header8(self.fmt), FM.gemv8_source(self.fmt, nbr, True)))
            arrs = lambda d: [d.qs, d.gs] if d.meta is None else [d.qs, d.meta, d.gs]
            tag = "8"
        else:
            names, mk = FM.gemv_inputs(True)[3:], (lambda: (FM.header(self.fmt, "gemv", cfmt()), FM.gemv_source(self.fmt, nbr, True, False)))
            arrs = lambda d: [d.qs, d.meta, d.gs]
            tag = "p"
        kern = _pair_kern(tag, self.fmt, nbr, cfmt() if tag == "p" else False, names, mk)
        y = kern(inputs=[x, xrows, eids] + arrs(self) + arrs(o), template=tpl, grid=grid, threadgroup=(128, 1, 1), output_shapes=[(2 * S, self.n)], output_dtypes=[mx.float16])[0]
        return y[:S], y[S:]

    def gather_glu(self, a, b, xrows, eids):
        """gather() of the down projection with the gated-MLP input silu(a) * b (a, b half [S, k], slot s reads row xrows[s]) computed while the input is staged."""
        S = int(eids.shape[0])
        assert self.glu_ok()
        nbr = NBRK
        ins = [a, b, xrows, eids, self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
        return _gemvk_kern(self.fmt, nbr, True, True)(inputs=ins, template=[("R", 1), ("KIN", self.k), ("N", self.n), ("NBR", nbr)], grid=(self.n // (8 * nbr) * 128, S, 1),
                                                      threadgroup=(128, 1, 1), output_shapes=[(S, self.n)], output_dtypes=[mx.float16])[0]

    def glu_ok(self):
        return bool(self.E) and self.fmt in FM.FORMATSK and self.gemv_ok()


class StoredEmbedding(nn.Module):
    """Token embedding kept in its GGUF blocks (one resident copy): the rows of the ids are gathered from the stored arrays and decoded on the fly.
    as_linear (a tied head) runs the stored product. `weight` is a shape-only stand-in for code that reads the table's size."""

    def __init__(self, pack: FP.Pack):
        super().__init__()
        self.dev = DevPack(pack)
        self.num_embeddings, self.dims = pack.n, pack.k
        self.weight = _Shape((pack.n, pack.k))

    def __call__(self, ids: mx.array) -> mx.array:
        d = self.dev
        flat = ids.reshape(-1).astype(mx.int32)
        take = lambda a: None if a is None else mx.take(a, flat, axis=0)
        sub = object.__new__(DevPack)                                       # the rows of the ids, lazily (DevPack.rows evals): every per-row array gathered, the global scale stays
        sub.__dict__.update(d.__dict__)
        sub.n, sub._views = int(flat.shape[0]), {}
        sub.qs, sub.meta, sub.w16 = take(d.qs), take(d.meta), None
        if d.rowgs:
            sub.gs = take(d.gs)
        if hasattr(d, "hi"):
            sub.hi = take(d.hi)
        sub.mq = None
        return sub.dequant().reshape(*ids.shape, self.dims)

    def as_linear(self, x):
        shape = x.shape
        y = self.dev.apply(x.reshape(-1, shape[-1]).astype(mx.float16))
        return y.reshape(*shape[:-1], self.num_embeddings)


class _Shape:
    def __init__(self, shape):
        self.shape, self.dtype = tuple(shape), mx.float16


class ForeignLinear(nn.Module):
    """y = layer(x) for x (..., in_features); out dtype fp16 (bias added when present). Arrays are private (not module parameters)."""

    def __init__(self, pack: FP.Pack, bias: mx.array | None = None, out_dtype: mx.Dtype = mx.float16, in_perm=None):
        super().__init__()
        self.dev = DevPack(pack)
        self._in_perm = None if in_perm is None else mx.array(np.asarray(in_perm).astype(np.int32))      # x_stored = x[..., in_perm] (foreign_xform: out_proj of a V-tiled GDN)
        self._in_inv = None if in_perm is None else mx.array(np.argsort(np.asarray(in_perm)).astype(np.uint32))      # x_stored[inv[c]] = x[c]: lets the producer write the permuted row
        self.in_features, self.out_features, self.fmt = pack.k, pack.n, pack.fmt
        self.out_dtype = out_dtype
        self._bias = bias.astype(out_dtype) if bias is not None else None

    def __call__(self, x: mx.array, permuted: bool = False) -> mx.array:
        """permuted=True: x is already in the stored column order (the producer applied _in_inv)."""
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.dtype != mx.float16:
            x2 = x2.astype(mx.float16)
        if self._in_perm is not None and permuted:
            _PREP[2] = False
        elif self._in_perm is not None:
            x2 = mx.take(x2, self._in_perm, axis=1)                        # a permuted input is not the shared one; take = one Gather (x2[:, idx] adds an Arange and two casts)
        elif _PREP[0] is not x:
            _PREP[0], _PREP[1] = x, {}
        _PREP[2] = self._in_perm is None
        try:
            y = self.dev.apply(x2)
        finally:
            _PREP[2] = False
        if y.dtype != self.out_dtype:
            y = y.astype(self.out_dtype)
        if self._bias is not None:
            y = y + self._bias
        return y.reshape(*shape[:-1], self.out_features)


class ForeignSwitchGLU(nn.Module):
    """mlx-lm SwitchGLU drop-in: (x [.., H], inds [.., K]) -> [.., K, H]; silu(gate x) * (up x) -> down. Slots <= GATHER_MAX: one launch per projection
    (expert id read in the kernel). More slots (prefill): the slots are sorted by expert and each expert's rows go through DevPack.apply."""

    def __init__(self, gate: DevPack, up: DevPack, down: DevPack):
        super().__init__()
        self.g, self.u, self.d = gate, up, down

    def _grouped(self, dp, x, xrows, eids_np):
        order = np.argsort(eids_np, kind="stable")
        xr = np.asarray(xrows)[order]
        es = eids_np[order]
        outs = []
        for e in np.unique(es):
            sel = np.nonzero(es == e)[0]
            xe = x[mx.array(xr[sel].astype(np.int32))]
            outs.append(dp.expert(int(e)).apply(xe))
        y = mx.concatenate(outs, axis=0)
        return y[mx.array(np.argsort(order).astype(np.int32))]

    @staticmethod
    def _stack16(dp):
        """[E, n, k] fp16 decode of a stacked expert pack in one launch (decoded as E * n rows), None for a dense pack or a stack over GMM_MAX_BYTES."""
        if dp.E is None or dp.fmt == "dense" or dp.E * dp.n * dp.k * 2 > GMM_MAX_BYTES:
            return None
        v = object.__new__(DevPack)
        v.__dict__.update(dp.__dict__)
        v.n, v.E = dp.E * dp.n, None
        if dp.fmt not in FM.FORMATSK and dp.fmt not in FM.FORMATS8:       # 4-bit classes: [E, n, ...] words and one scale per expert -> E * n rows with one scale per row
            v.qs, v.meta = dp.qs.reshape(v.n, -1), dp.meta.reshape(v.n, -1)
            v.gs, v.rowgs = mx.repeat(dp.gs, dp.n), True
        return v.dequant().reshape(dp.E, dp.n, dp.k)

    def _sorted_gmm(self, x2, e32, T, K):
        """Prefill: the slots sorted by expert on the device, each projection one gathered matmul over the decoded stack (no host readback, no per-expert launch).
        Returns [S, H] fp16 in slot order, or None when a projection cannot be decoded as one stack."""
        order = mx.argsort(e32)
        es = e32[order]
        xs = x2[order // K][:, None, :]
        outs = []
        for dp in (self.g, self.u):
            w = self._stack16(dp)
            if w is None:
                return None
            outs.append(mx.gather_mm(xs, w.swapaxes(-1, -2), rhs_indices=es, sorted_indices=True)[:, 0])
            del w
        h = (nn.silu(outs[0].astype(mx.float32)) * outs[1].astype(mx.float32)).astype(mx.float16)[:, None, :]
        w = self._stack16(self.d)
        if w is None:
            return None
        y = mx.gather_mm(h, w.swapaxes(-1, -2), rhs_indices=es, sorted_indices=True)[:, 0]
        return y[mx.argsort(order)]

    def __call__(self, x, inds):
        lead, H = x.shape[:-1], x.shape[-1]
        K = inds.shape[-1]
        T = int(np.prod(lead))
        x2 = x.reshape(T, H).astype(mx.float16)
        e32 = inds.reshape(-1).astype(mx.uint32)
        S = T * K
        if GMM and S > GATHER_MAX:
            y = KQ.mlp(self, x2, e32, K, KQ_MT) if KQ.on() else None      # flag kqgmm: grouped NAX GEMM on the stored rows (no fp16 expert stack at all)
            if y is None:
                y = self._sorted_gmm(x2, e32, T, K)
            if y is not None:
                return y.reshape(*lead, K, self.d.n)
        tok = mx.array((np.arange(S) // K).astype(np.uint32))
        gather = S <= GATHER_MAX and all(p.fmt != "dense" and p.gemv_ok() for p in (self.g, self.u, self.d))
        if gather:
            a = self.g.gather(x2, tok, e32)
            b = self.u.gather(x2, tok, e32)
        else:
            en = np.array(e32).astype(np.int64)
            tn = np.arange(S) // K
            a = self._grouped(self.g, x2, tn, en)
            b = self._grouped(self.u, x2, tn, en)
        h = (nn.silu(a.astype(mx.float32)) * b.astype(mx.float32)).astype(mx.float16)
        if gather:
            y = self.d.gather(h, mx.arange(S).astype(mx.uint32), e32)
        else:
            y = self._grouped(self.d, h, np.arange(S), en)
        return y.reshape(*lead, K, self.d.n)


# ------------------------------------------------------------------------------------------------------------------------------ same-input groups (flag cfmtspeed)
class _Group:
    """The concatenated matrix of the linears that read one input, and the last (input, outputs) it computed: the first member called runs the whole group, the
    others return their slice when called with the same input object (fused_group.FusedMember does the same for EXL3)."""

    def __init__(self, dev: DevPack, cuts):
        self.dev, self.cuts = dev, cuts
        self.x = self.ys = None


class GroupMember(nn.Module):
    """Stand-in for one ForeignLinear of a group: same call, same output (fp16 products of its own rows, bias, out dtype)."""

    def __init__(self, grp: _Group, i: int, src: "ForeignLinear"):
        super().__init__()
        self._g, self._i = grp, i
        self.in_features, self.out_features, self.fmt = src.in_features, src.out_features, src.fmt
        self.out_dtype, self._bias = src.out_dtype, src._bias

    def __call__(self, x: mx.array) -> mx.array:
        g, shape = self._g, x.shape
        if g.x is None or g.x() is not x or g.ys[self._i] is None:
            x2 = x.reshape(-1, shape[-1])
            if x2.dtype != mx.float16:
                x2 = x2.astype(mx.float16)
            g.ys = list(g.dev.apply_split(x2, g.cuts))
            g.x = weakref.ref(x)                                 # a strong ref would pin one prefill-chunk activation per group (11 GB at 8192 rows)
        y, g.ys[self._i] = g.ys[self._i], None                   # each member takes its slice once; the group keeps nothing a finished forward owns
        if y.dtype != self.out_dtype:
            y = y.astype(self.out_dtype)
        if self._bias is not None:
            y = y + self._bias
        return y.reshape(*shape[:-1], self.out_features)


def find_groups(calls, names):
    """calls: [(input object id, module id)] of one forward; names: module id -> name. Groups of >= 2 distinct modules that received the same input object,
    each module called once in the whole forward; first-call order, so a group is the same on every trace."""
    from collections import Counter
    seen = Counter(m for _, m in calls)
    by, order = {}, []
    for xi, mi in calls:
        if seen[mi] != 1 or mi not in names:
            continue
        if xi not in by:
            by[xi] = []
            order.append(xi)
        by[xi].append(mi)
    return [[names[m] for m in by[xi]] for xi in order if len(by[xi]) >= 2]


def group_ok(lins) -> bool:
    """The group's output rows must be what each member computes alone: one non-dense format, one k and out dtype, and every member takes the same kernel
    family at every row count (gemv_ok is a property of k and n % 16: all members agree with the total)."""
    l0 = lins[0]
    if any(type(l) is not ForeignLinear or l.dev.fmt == "dense" or l.dev.fmt in FM.FORMATSK or l.dev.E is not None for l in lins):
        return False
    if any((l.fmt, l.in_features, l.out_dtype) != (l0.fmt, l0.in_features, l0.out_dtype) for l in lins):
        return False
    ok = {l.dev.gemv_ok() for l in lins}
    return len(ok) == 1 and ok == {(l0.in_features % 512 == 0) and (sum(l.out_features for l in lins) % (8 * NBR) == 0)}


def trace_calls(model):
    """One 1-row forward on the lazy graph (nothing is evaluated: ForeignLinear / ForeignSwitchGLU return zeros of their output shape) -> [(x id, module id)]."""
    calls, keep = [], []
    orig_l, orig_s = ForeignLinear.__call__, ForeignSwitchGLU.__call__

    def rec_l(self, x):
        calls.append((id(x), id(self)))
        keep.append(x)                                                    # ids stay unique while the trace holds the arrays
        return mx.zeros((*x.shape[:-1], self.out_features), self.out_dtype)

    def rec_s(self, x, inds):
        return mx.zeros((*x.shape[:-1], inds.shape[-1], self.d.n), mx.float16)

    ForeignLinear.__call__, ForeignSwitchGLU.__call__ = rec_l, rec_s
    try:
        model(mx.zeros((1, 1), mx.int32))
    finally:
        ForeignLinear.__call__, ForeignSwitchGLU.__call__ = orig_l, orig_s
    return calls


def group_same_input(model, log=print) -> dict:
    """Flag cfmtspeed: replace the ForeignLinears that read the same input by GroupMembers over one concatenated matrix. Groups are built one at a time and
    the members' arrays dropped before the next, so the peak is the model plus one group. A model whose trace fails is left as it is."""
    from mlx.utils import tree_unflatten
    named = {n: m for n, m in model.named_modules() if type(m) is ForeignLinear}
    try:
        groups = find_groups(trace_calls(model), {id(m): n for n, m in named.items()})
    except Exception as e:                                                # a family whose forward needs more than tokens: stay as served today
        log(f"[foreign] cfmtspeed: no grouping ({type(e).__name__}: {e})")
        return {"groups": 0, "members": 0, "launches_saved": 0}
    mods, saved = [], 0
    parts = []
    for g in groups:                                                      # a same-input set of mixed formats splits into one group per (format, k, out dtype)
        by = {}
        for n in g:
            by.setdefault((named[n].fmt, named[n].in_features, named[n].out_dtype), []).append(n)
        parts += [p for p in by.values() if len(p) >= 2]
    for g in parts:
        lins = [named[n] for n in g]
        if not group_ok(lins):
            continue
        dev = DevPack.grouped([l.dev for l in lins])
        mx.eval(dev.qs, dev.meta, dev.gs)
        cuts = [0]
        for l in lins:
            cuts.append(cuts[-1] + l.out_features)
        grp = _Group(dev, cuts)
        mods += [(n, GroupMember(grp, i, l)) for i, (n, l) in enumerate(zip(g, lins))]
        for l in lins:
            l.dev = None
        saved += len(g) - 1
    if mods:
        model.update_modules(tree_unflatten(mods))
    ng = len({id(m._g) for _, m in mods})
    log(f"[foreign] cfmtspeed: {len(mods)} linears in {ng} same-input groups, {saved} fewer launches per 1-row tick (counted, not timed)")
    return {"groups": ng, "members": len(mods), "launches_saved": saved}


# ------------------------------------------------------------------------------------------------------------------------------ dense gated MLP (flag KYOJIN_FOREIGN_DENSE_GLU)
def _glu_call(self, x):
    """down(silu(gate(x)) * up(x)) for few rows: the silu * mul is done while the down projection stages its input, so the MLP is gate | up + one launch (no
    separate activation launch). Any other case runs the module's own call."""
    dn, g, u = self.down_proj, self.gate_proj, self.up_proj
    lead, H = x.shape[:-1], x.shape[-1]
    R = 1
    for d in lead:
        R *= d
    K = dn.in_features
    if dn.dev is None or R > GEMV8_ROWS or not dn.dev.glu_dense_ok(R):
        return self._stock_call(x)
    x2 = x.reshape(-1, H)
    if x2.dtype != mx.float16:
        x2 = x2.astype(mx.float16)
    if isinstance(g, GroupMember) and isinstance(u, GroupMember):
        grp = g._g
        if u._g is not grp or (g._i, u._i) != (0, 1) or grp.cuts != [0, K, 2 * K]:
            return self._stock_call(x)
        y = grp.dev.apply(x2)
        out = dn.dev.apply_glu(y, y, K, 2 * K)
    else:
        if type(g) is not ForeignLinear or type(u) is not ForeignLinear or g.dev is None or u.dev is None:
            return self._stock_call(x)
        a, b = g(x).reshape(R, K), u(x).reshape(R, K)
        out = dn.dev.apply_glu(a, b)
    return out.reshape(*lead, dn.out_features)


def install_dense_glu(model, log=print) -> int:
    """Swap the call of every gated MLP (gate_proj / up_proj / down_proj) whose down projection is a ForeignLinear of a GEMV-able format. Returns the number."""
    if not DENSE_GLU:
        return 0
    n = 0
    for _, mod in model.named_modules():
        g, u, dn = (getattr(mod, k, None) for k in PROJ)
        if type(dn) is not ForeignLinear or getattr(mod.__class__, "_dense_glu", False):
            continue
        ok = lambda l: (type(l) is ForeignLinear and l.dev is not None) or type(l) is GroupMember
        if not (ok(g) and ok(u)) or dn._bias is not None or dn._in_perm is not None or dn.out_dtype != mx.float16:
            continue
        if any(l.out_dtype != mx.float16 or l._bias is not None or l.out_features != dn.in_features for l in (g, u)):
            continue
        cls = mod.__class__
        mod.__class__ = type("Glu" + cls.__name__, (cls,), {"__call__": _glu_call, "_stock_call": cls.__call__, "_dense_glu": True})
        n += 1
    if n:
        log(f"[foreign] dense_glu: {n} gated MLPs run silu * mul inside the down projection's launch")
    return n


# ------------------------------------------------------------------------------------------------------------------------------ detection
def is_foreign(path) -> str | None:
    """'gguf' | 'safetensors' | None. Header-only; an EXL3 pack or an MLX-quantized pack is never foreign."""
    p = str(path)
    if os.path.isfile(p) and p.endswith(".gguf"):
        return "gguf"
    if not os.path.isdir(p):
        return None
    try:
        cfg = json.load(open(os.path.join(p, "config.json")))
    except Exception:
        cfg = {}
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config") or {}
    if qc.get("quant_method") == "exl3" or "quantization" in cfg:
        return None
    if GF.find_gguf(p) and not glob.glob(os.path.join(p, "*.safetensors")):
        return "gguf"
    if glob.glob(os.path.join(p, "*.safetensors")):
        from . import vendor_input as VI
        ck = VI.Checkpoint(p)
        try:
            rep = VI.detect_checkpoint(ck.info(), ck.config)
        finally:
            ck.close()
        if (any(f in rep["counts"] for f in ("nvfp4", "mxfp4")) or any(f.startswith("fp8_") for f in rep["counts"])) and "mxfp4_blocks" not in rep["counts"]:
            return "safetensors"                                 # NVFP4 / MXFP4 (FP8 beside them) or a plain FP8 release (Qwen-FP8, DeepSeek, GLM, MiniMax layouts)
    return None


# ------------------------------------------------------------------------------------------------------------------------------ loading
_EXP = re.compile(r"^(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.*)$")
PROJ = ("gate_proj", "up_proj", "down_proj")


class _Src:
    """Normalised source: dense tensors (mx arrays) and quantised linears (Packs), under HF names, plus expert Packs."""

    def __init__(self):
        self.dense: dict = {}
        self.packs: dict = {}            # linear prefix -> Pack, or a thunk that builds it (resolved by _build, one at a time)
        self.embed: dict = {}            # embedding prefix -> thunk of a stored Pack (StoredEmbedding)
        self.experts: dict = {}          # mlp prefix -> {proj: [Pack per expert]}
        self.in_perm: dict = {}          # linear prefix -> activation gather (foreign_xform.Xf.cols)
        self.mtp: dict = {}              # draft head: `mtp.*` name -> fp16 / fp32 mx array in the checkpoint's own convention (norms without the +1)
        self.notes: list = []
        self.rope_freqs = None           # llama.cpp `rope_freqs.weight` (Llama 3.x): per-pair frequency divisors, see FreqRoPE


def _read_safetensors(path) -> _Src:
    from . import vendor_input as VI
    ck = VI.Checkpoint(path)
    src = _Src()
    info = ck.info()
    block = VI._config_block(ck.config)
    used = set()
    for p in VI.linear_prefixes(info):
        c = VI.classify_linear(info, p)
        if c is None or c["fmt"].startswith("dense_"):
            continue
        tag8 = ck.raw(c["weight"])[0] if c["fmt"].startswith("fp8_") else None
        if (fp8stored() and tag8 == "F8_E4M3" and c["fmt"] in ("fp8_tensor", "fp8_channel", "fp8_noscale")
                and ck.raw(c["weight"])[1][-1] % 32 == 0 and len(ck.raw(c["weight"])[1]) == 2):    # flag fp8stored: E4M3 bytes as stored, scale per row
            _, shp, w8 = ck.raw(c["weight"])
            sc = np.float32(1.0) if c["fmt"] == "fp8_noscale" else np.asarray(ck.scale_f32(c["scale"]), np.float32).reshape(-1)
            pk = FP.Pack("fp8", shp[0], shp[1], np.ascontiguousarray(w8).view(np.uint8), None, sc if np.ndim(sc) == 0 or sc.size != 1 else np.float32(sc[0]))
        elif (fp8stored() and tag8 == "F8_E4M3" and c["fmt"] == "fp8_block" and len(ck.raw(c["weight"])[1]) == 2
              and _fp8b_pack(ck, c, block) is not None):                # flag fp8stored: block-scaled E4M3 (DeepSeek / Qwen-FP8 / GLM / MiniMax layout) as stored
            pk = _fp8b_pack(ck, c, block)
        elif c["fmt"].startswith("fp8_"):                            # E5M2, blocks that straddle the 32-weight units: fp16 at load
            w = np.asarray(VI.load_linear(ck, p)).astype(np.float16)
            pk = FP.Pack("dense", w.shape[0], w.shape[1], w)
        elif c["fmt"] in ("nvfp4", "mxfp4"):
            pk = FP.pack_from_safetensors(ck, c)
        else:
            raise ValueError(f"{p}: {c['fmt']} ({c['detail']}) is not served as stored (only NVFP4 / MXFP4 / FP8); convert it first")
        used.update(k for k in info if k.startswith(p + ".") and k.rsplit(".", 1)[-1] in
                    ("weight", "weight_packed", "weight_scale", "weight_scale_2", "weight_global_scale", "input_scale", "input_global_scale", "scale"))
        m = _EXP.match(p + ".weight")
        if m:
            src.experts.setdefault(m[1], {}).setdefault(m[3], {})[int(m[2])] = pk
        else:
            src.packs[p] = pk
    for f in sorted(ck.files):
        for k, v in mx.load(os.path.join(path, f)).items():
            if k not in used:
                src.dense[k] = v
    src._ck = ck
    for mp in src.experts:
        for pj in src.experts[mp]:
            d = src.experts[mp][pj]
            assert sorted(d) == list(range(len(d))), f"{mp}.{pj}: expert ids are not contiguous"
            src.experts[mp][pj] = [d[e] for e in range(len(d))]
    return src


def _build(v):
    """A Pack (or list of Packs) from a thunk of _read_gguf, or the value itself."""
    return v() if callable(v) else v


def _dequant_f16(tname, raw, k, slab=1 << 24):
    """float16 [rows, k] from stored rows, decoded `slab` elements at a time."""
    out = np.empty((raw.shape[0], k), np.float16)
    step = max(1, slab // k)
    for i in range(0, raw.shape[0], step):
        out[i:i + step] = GF.dequant(tname, raw[i:i + step], k)
    return out


def embed_stored(hf, ti, arch, g) -> bool:
    """A token embedding that stays in its GGUF blocks: a K / IQ type that has a kernel, with its own output head (a tied table is also the head and
    stays dense), and big enough to matter (>= 256 MiB as fp16). KYOJIN_STORED_EMBED=0 turns it off."""
    if os.environ.get("KYOJIN_STORED_EMBED", "1") == "0" or not hf.endswith("embed_tokens.weight") or len(ti.ne) != 2:
        return False
    if emb_stored_all():                                                    # every stored type with a kernel, tied table included (it is also the head: as_linear runs the stored product)
        ok = (ti.tname in GF.KERNEL_TYPES or (ti.tname in GF.KERNEL_TYPES8 and fp8stored()) or (ti.tname in GF.KERNEL_TYPESK and kstored())) \
            and ti.ne[0] % (32 if ti.tname in GF.KERNEL_TYPES8 or ti.tname in ("Q4_0", "IQ4_NL", "MXFP4") else 256) == 0 and ti.ne[0] * ti.ne[1] * 2 >= (256 << 20)
        return bool(ok)
    if ti.tname not in GF.KERNEL_TYPESK or ti.ne[0] % 256 != 0 or ti.ne[0] * ti.ne[1] * 2 < (256 << 20):
        return False
    return "output.weight" in g.tensors and kstored()


def emb_stored_all() -> bool:
    """Flag KYOJIN_GGUF_EMB_STORED (read at load, default 1; 0 = the old rule: K types with their own head only): the token embedding (and a tied head) of any kernel type stays in its GGUF blocks, no fp16 table."""
    return os.environ.get("KYOJIN_GGUF_EMB_STORED", "1") not in ("0", "")


def _read_gguf(path, arch) -> _Src:
    g = GF.GgufFile(GF.find_gguf(path))
    src = _Src()
    A = ARCHS[arch]
    nt = n_trunk_of(g.kv, arch)
    for name, ti in g.tensors.items():
        if name == "rope_freqs.weight" and len(ti.ne) == 1:            # not a model weight: the rope frequency factors the converter baked from rope_scaling
            src.rope_freqs = mx.array(GF.dequant(ti.tname, g.raw(name), ti.ne[0]).reshape(-1))
            continue
        hf = gguf_name(name, arch, nt)
        if hf is None:
            src.notes.append(f"{name}: draft-head block, not used by the {arch} class (skipped, as llama.cpp does)")
            continue
        xf = XF.plan(name, A.xform, g.kv, ti.ne[-1] if len(ti.ne) == 2 else (ti.ne[0] if len(ti.ne) == 1 else None))
        raw = g.raw(name)
        head = hf.startswith("mtp.")
        rows = xf.rows if (xf is not None and xf.rows is not None and raw.ndim == 2) else None      # rows are independent in every GGUF type: permute the stored blocks, nothing is decoded
        is_w = is_linear(hf, len(ti.ne), arch)
        if embed_stored(hf, ti, arch, g) and not head:                       # token embedding kept as stored (StoredEmbedding): no fp16 table
            src.embed[hf[:-len(".weight")]] = lambda raw=raw, rows=rows, tn=ti.tname, k=ti.ne[0]: GF.to_pack(tn, raw if rows is None else raw[rows], k)
            continue
        if (ti.tname in GF.KERNEL_TYPES or (ti.tname in GF.KERNEL_TYPES8 and fp8stored() and len(ti.ne) in (2, 3))
                or (ti.tname in GF.KERNEL_TYPESK and kstored() and len(ti.ne) in (2, 3) and ti.ne[0] % 256 == 0)) and is_w and not head:
            k = ti.ne[0]
            if len(ti.ne) == 3:                                              # stacked experts [E, n, k]: built when the loader reaches the block (see _build), so the host copy is never whole
                m = re.match(r"(.*\.mlp)\.experts\.(\w+)\.weight", hf)
                src.experts.setdefault(m[1], {})[m[2]] = lambda raw=raw, tn=ti.tname, k=k, E=ti.ne[2]: [GF.to_pack(tn, raw[e], k) for e in range(E)]
            else:
                src.packs[hf[:-len(".weight")]] = lambda raw=raw, rows=rows, tn=ti.tname, k=k: GF.to_pack(tn, raw if rows is None else raw[rows], k)
                if xf is not None and xf.cols is not None:
                    src.in_perm[hf[:-len(".weight")]] = xf.cols
            continue
        if ti.tname not in GF.DEQUANT:
            raise GF.GgufTypeError(f"tensor {name}: GGUF type {ti.tname} is not readable (see foreign_gguf.DEQUANT)")
        if rows is not None:
            raw = raw[rows]
        if raw.ndim == 2 and raw.shape[0] * ti.ne[0] > (1 << 27) and ti.tname not in ("F32", "F16", "BF16") and (xf is None or (xf.rows is None and xf.post is None)):
            w = _dequant_f16(ti.tname, raw, ti.ne[0])                       # big 2-D tensor: decoded in row slabs straight to fp16, the fp32 table (2x the bytes) never exists
        else:
            w = GF.dequant(ti.tname, raw, ti.ne[0]).reshape(ti.shape)
        if xf is not None and xf.rows is not None and w.ndim == 1:
            w = w[xf.rows]
        if xf is not None and xf.post is not None:
            w = XF.POST[xf.post](w)
        if head:                                                             # the draft head stays at the file's precision, fp16 (never re-quantised)
            src.mtp[hf] = mx.array(w.astype(np.float16) if w.ndim > 1 else w.astype(np.float32))
            if ti.tname not in ("F32", "F16", "BF16"):
                src.notes.append(f"{name}: {ti.tname} draft head decoded to fp16")
            continue
        if len(ti.ne) == 3 and is_w:
            m = re.match(r"(.*\.mlp)\.experts\.(\w+)\.weight", hf)
            src.experts.setdefault(m[1], {})[m[2]] = [FP.Pack("dense", w.shape[1], w.shape[2], w[e].astype(np.float16)) for e in range(w.shape[0])]
            src.notes.append(f"{name}: {ti.tname} experts decoded to fp16 (no Metal kernel for this type)")
        elif is_w:
            if xf is not None and xf.cols is not None:
                w = w[:, np.argsort(xf.cols, kind="stable")]                 # dense: undo the column order on the weight itself (exact)
            src.packs[hf[:-len(".weight")]] = FP.Pack("dense", w.shape[0], w.shape[1], w.astype(np.float16))
            src.notes.append(f"{name}: {ti.tname} decoded to fp16 (no Metal kernel for this type)")
        else:
            src.dense[hf] = mx.array(w.astype(np.float16) if w.ndim > 1 and ti.tname != "F32" else w)
    src._g = g
    return src


def take_mtp(model) -> dict | None:
    """The `mtp.*` tensors load_foreign read from the GGUF's nextn blocks (once: the dict leaves the model), or None."""
    return model.__dict__.pop("_gguf_mtp", None)


class FreqRoPE(nn.Module):
    """Rotary embedding with llama.cpp frequency factors: angle_i = pos / (base^(2i/d) * ff_i), as ggml_rope_ext divides theta by freq_factors
    (the same `freqs` form mlx-lm's Llama3RoPE hands mx.fast.rope)."""

    def __init__(self, dims: int, base: float, traditional: bool, ff: mx.array):
        super().__init__()
        self.dims, self.traditional = dims, traditional
        self._freqs = (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)) * ff.astype(mx.float32)

    def __call__(self, x, offset=0):
        return mx.fast.rope(x, self.dims, traditional=self.traditional, base=None, scale=1.0, offset=offset, freqs=self._freqs)


def install_rope_freqs(model, ff: mx.array) -> int:
    """Swap every plain nn.RoPE of the attention layers for FreqRoPE with the file's factors. Refuses what it cannot follow (scaled or partial rope)."""
    n = 0
    for _, m in model.named_modules():
        r = getattr(m, "rope", None)
        if not isinstance(r, nn.RoPE):
            continue
        if r.scale != 1.0 or r.dims != 2 * ff.size:
            raise ValueError(f"rope_freqs: {ff.size} factors for a rope of {r.dims} dims, scale {r.scale}: not followable")
        m.rope = FreqRoPE(r.dims, r.base, r.traditional, ff)
        n += 1
    if n == 0:
        raise ValueError("rope_freqs: the file carries rope frequency factors but the model class has no plain rope to apply them to")
    return n


def load_foreign(path, dtype=mx.float16, log=print):
    """(model, config) for a directory (or .gguf file) in a foreign format. The model class is mlx-lm's for config['model_type'], the quantised linears are
    ForeignLinear, MoE experts ForeignSwitchGLU; non-quantised tensors are kept as stored (bf16 -> `dtype`)."""
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import _get_classes
    path = str(path)
    kind = is_foreign(path)
    if kind is None:
        raise ValueError(f"{path}: neither a GGUF file nor an NVFP4 / MXFP4 safetensors directory")
    d = path if os.path.isdir(path) else os.path.dirname(path) or "."
    if kind == "gguf":
        gf = GF.GgufFile(GF.find_gguf(path))
        arch = gf.kv.get("general.architecture")
        cfg_path = os.path.join(d, "config.json")
        config = json.load(open(cfg_path)) if os.path.exists(cfg_path) else gguf_config(gf)
        gf.close()
        src = _read_gguf(path, arch)
    else:
        config = json.load(open(os.path.join(d, "config.json")))
        src = _read_safetensors(d)
    config.pop("quantization_config", None)
    tc = config.get("text_config", config)
    model_class, args_class = _get_classes(config=config)
    model = model_class(args_class.from_dict(config))
    # sanitize may rename (language_model.* ...): run the dense tensors plus one marker per quantised linear through it and follow the markers
    marks = {p + ".weight": mx.zeros((1,)) for p in {**src.packs, **src.embed}}
    weights = (model.sanitize if hasattr(model, "sanitize") else dict)({**src.dense, **marks})        # glm4 and some others have no sanitize: names are the HF ones already
    by_id = {id(v): k for k, v in weights.items()}
    # a top-level group sanitize drops whole (mtp.* in qwen3_5: the MTP head is served from its own sidecar) is dropped for the quantised linears too
    kept = {id(v) for v in weights.values()}
    groups = {}
    for k, v in {**src.dense, **marks}.items():
        groups.setdefault(k.split(".", 1)[0], []).append(id(v) in kept)
    dropped = {g for g, ks in groups.items() if not any(ks)}
    where = {}
    for k, v in marks.items():
        if id(v) not in by_id:
            if k.split(".", 1)[0] in dropped:
                src.packs.pop(k[:-len(".weight")], None); src.embed.pop(k[:-len(".weight")], None)
                continue
            raise ValueError(f"model.sanitize() rewrote {k}: this architecture's weight renaming is not followable")
        where[k[:-len(".weight")]] = by_id[id(v)][:-len(".weight")]
    mods = []
    fm = {}
    n_packs = len(src.packs)
    for p in list(src.packs):                                     # one host pack at a time: built, copied to the device, dropped (host and device arrays are never both whole)
        pk = _build(src.packs.pop(p))
        fm[pk.fmt] = fm.get(pk.fmt, 0) + 1
        bias = weights.pop(where[p] + ".bias", None)
        mods.append((where[p], ForeignLinear(pk, bias, in_perm=src.in_perm.get(p))))
        weights.pop(where[p] + ".weight", None)
        del pk
    for p in list(src.embed):                                     # token embedding as stored: rows are decoded when looked up
        pk = _build(src.embed.pop(p))
        mods.append((where[p], StoredEmbedding(pk)))
        weights.pop(where[p] + ".weight", None)
        del pk
    model.update_modules(tree_unflatten(mods))
    n_exp = 0
    if src.experts:
        root = model.language_model.model if hasattr(model, "language_model") and hasattr(model.language_model, "model") else (model.model if hasattr(model, "model") else model)
        for mp in list(src.experts):
            projs = src.experts.pop(mp)
            li = int(re.search(r"layers\.(\d+)\.mlp$", mp)[1])
            blk = root.layers[li].mlp
            assert hasattr(blk, "switch_mlp"), f"layer {li}: the model class has no switch_mlp to replace"
            blk.switch_mlp = ForeignSwitchGLU(*[DevPack(_build(projs[pj])) for pj in PROJ])
            n_exp += 1
    weights = {k: (v.astype(dtype) if v.dtype == mx.bfloat16 or (gguf_stream_f16() and _f32_on_stream(k, v)) else v) for k, v in weights.items()}
    from mlx.utils import tree_flatten
    missing = sorted(k for k, _ in tree_flatten(model.parameters()) if k not in weights)
    if missing:                                                   # strict=False below tolerates extra tensors; a parameter nothing filled would keep its random / default init
        raise ValueError(f"{path}: the file has no tensor for {len(missing)} parameter(s) of the {config.get('model_type')} model, e.g. {missing[:4]}: "
                         f"an unsupported or incomplete checkpoint is refused, not served with initial weights")
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    if src.rope_freqs is not None:
        n_rope = install_rope_freqs(model, src.rope_freqs)
        log(f"rope_freqs: {n_rope} attention layers use the file's frequency factors")
    if src.mtp:                                                   # the draft head of a GGUF with nextn blocks: `take_mtp(model)` (not a module attribute: nn.Module would register the dict as parameters)
        model.__dict__["_gguf_mtp"] = src.mtp
    try:
        from . import gdn_norm, gdn_fast, gdn_pf
        gdn_norm.install()
        gdn_fast.install()
        gdn_pf.install(dec=os.environ.get("KYOJIN_GDN_DEC", "1") != "0")      # fused gated-delta front end + gated norm for decode rows too (fp16 rounding differs from the stock ops; -8 % tick on Qwen3.6-35B-A3B)
    except ImportError:
        pass
    from . import decfuse, decrows
    if cfmt() or (decfuse.ENABLED and decfuse.PROJ):                  # decfuse class 1 (same-input projections in one launch): exact, so on with the decfuse switches
        group_same_input(model, log)
    decrows.enable_foreign(model)                                       # decfuse class 2 for any row count: residual add + RMSNorm in one launch
    from . import foreign_moe
    foreign_moe.install(model, log)
    install_dense_glu(model, log)
    mx.eval(model.parameters())
    log(f"[foreign] {kind}: {n_packs} linears {fm}, {n_exp} expert blocks" + "".join("\n[foreign] " + n for n in src.notes[:8]))
    return model, config
