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

from . import loadcache as LC, foreign_gguf as GF, foreign_kqgmm as KQ, foreign_metal as FM, foreign_pack as FP, foreign_wide as FW, fp4tile as FT
from .foreign_names import ARCHS, gguf_name, gguf_config, is_linear, is_lookup_table, mapped_row_table, n_trunk_of
from . import foreign_xform as XF
from . import fired as FD
from . import kernels as KN
from . import moe_union as MU
from . import moe_scale_compact as MC
from . import glm_dense6 as G6
from . import foreign_pfi8 as PI8
from . import fmm8 as F8

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
NBR_SMALL = int(os.environ.get("KYOJIN_FOREIGN_NBR_SMALL", "0") or 0)       # flag foreign_nbr_small: output rows per simdgroup of the 8-bit stored GEMV on matrices under NBR_SMALL_MAX weights (0 = NBR; 1 doubles the threadgroups)
NBR_SMALL_MAX = 8 << 20
NTN = int(os.environ.get("KYOJIN_FOREIGN_NTN", "1"))
_FMMA_MIN_ENV = os.environ.get("KYOJIN_FOREIGN_FMMA_MIN")                      # one value for every format; unset = FMMA_MIN_FMT
# 11/11 retune after the K / IQ wrapper cache (a7341f4, the stored GEMV lost 25-46 us of host per call): the GEMV leads up to R4 on IQ3_XXS / IQ3_S /
# IQ2_XXS / IQ2_XS / IQ1_S (R4 GB/s fmma -> GEMV: 303 -> 341, 289 -> 355, 188 -> 255, 163 -> 263, 96 -> 207; IQ2_XS R2 162 -> 361) and drops from R5 (NBR 1)
FMMA_MIN_FMT = {"q2_0": 4, "nvfp4": 2, "mxfp4": 2, "q4_k": 3, "q4_0": 3, "q5_k": 4, "q6_k": 4, "iq4_xs": 4, "q3_k": 4, "iq3_s": 5, "iq3_xxs": 5, "q2_k": 4, "iq2_xxs": 5, "iq2_xs": 5, "iq2_s": 3, "iq1_s": 5, "iq1_m": 3, "q4_1": 4, "q5_0": 4, "q5_1": 4}               # gate 17408x5120, GB/s GEMV vs fmma: K formats R3 GEMV vs fmma over 3 shapes: q5_k 670/706 728/689 568/547, q6_k 779/744 855/861 663/635, iq4_xs 618/632 678/622 514/482 (min 4); fp4 R2 608/692 R4 314/701; q4_k R2 752/687 R3 667/694; q4_0 R2 773/736 R3 705/745
FMMA_MAX = int(os.environ.get("KYOJIN_FOREIGN_FMMA_MAX", "16"))   # flag foreign_fmma: rows served by the few-row MMA
WIDE_MIN_ENV = os.environ.get("KYOJIN_FOREIGN_WIDE_MIN")                     # one value for every format (KYOJIN_FOREIGN_WIDE_MIN=9 restores the old 8-bit route); unset = WIDE_MIN_FMT
WIDE_MIN = int(WIDE_MIN_ENV or "9")
# 12/11: the 8-bit formats take the wide product from R5 (gate 17408x5120, GB/s of stored bytes, stored GEMV -> wide: Q8_0 R5 725 -> 865, R8 521 -> 867;
# fp8 R5 629 -> 856, R8 456 -> 858; fp8b R5 650 -> 858, R8 458 -> 847; R4 equal), the others from R9
WIDE_MIN_FMT = {"q8_0": 5, "fp8": 5, "fp8b": 5}
# ... on the large dense linears only: served dense 27B FP8 N8 97.9 -> 149.4 (+53 %), N8d 86.8 -> 125.1; the small attention linears of the MoE models lost
# (Qwen3-30B-A3B FP8 N8 318 -> 301, GLM-4.7-Flash Q8_0 188 -> 172 with every 8-bit linear on it), so below WIDE8_MIN_SIZE weights the 8-bit route is unchanged
WIDE8_MIN_SIZE = int(os.environ.get("KYOJIN_FOREIGN_WIDE8_MIN_SIZE", str(16 << 20)))                # flag foreign_wide: rows served by the register-fed NAX product (foreign_wide.wide_source_reg) on stored bytes
WIDE_MAX_FMT = {"nvfp4": 24, "mxfp4": 24, "q4_0": 64, "q4_k": 64, "q5_k": 64, "q6_k": 64, "iq4_xs": 64, "fp8": 32, "q8_0": 64, "fp8b": 64}   # gate 17408x5120 / 5120x17408 / 5120x5120 (min of 14 interleaved reps): the K types, q4_0 and q8_0 beat decode + matmul to R64 (q8_0 2-3x), fp8 beats mma8 to R32, MLX's in-place qmm wins fp8 / fp4 above that
WIDE_MAX_ENV = os.environ.get("KYOJIN_FOREIGN_WIDE_MAX")
FMMA_NT, FMMA_WS = int(os.environ.get("KYOJIN_FOREIGN_FMMA_NT", "2")), int(os.environ.get("KYOJIN_FOREIGN_FMMA_WS", "4"))
GMM = os.environ.get("KYOJIN_FOREIGN_GMM", "1") not in ("0", "")             # prefill routed experts: sorted gathered matmul over the decoded stack (0 = one launch per expert)
KQ_MT = int(os.environ.get("KYOJIN_KQ_GMM_MT", "32"))                      # flag kqgmm: rows of one expert per tile
GMM_MAX_BYTES = int(float(os.environ.get("KYOJIN_FOREIGN_GMM_MAX_GIB", "3")) * 2 ** 30)   # largest transient fp16 stack
TILE = os.environ.get("KYOJIN_GLM_TILE", "0") not in ("0", "")             # flag glmtile (default off): NVFP4 routed-expert stacks are re-laid out once at load into the decode tile layout (fp4tile.py) and replace the stored arrays
TILE_UB, TILE_GLU_NSG, TILE_DOWN_NSG = 2, 8, 8                           # chunk units, simdgroups per threadgroup of the gate|up|glu launch / of the down launch (tools/g2b_layer.py sweep)
GATHER_MAX = int(os.environ.get("KYOJIN_FOREIGN_GATHER_MAX", "128"))       # routed-expert slots up to here: one gather launch (R = 1 per slot)
CASTFREE = os.environ.get("KYOJIN_FOREIGN_CASTFREE", "0") not in ("0", "")     # flag foreign_castfree (default off, read per call; flags.set updates it): one-byte stored linears (fp8 / q8_0 / fp8b) on the
# gemv8 route read a float32 / bfloat16 x and write a float32 y in the GEMV itself (ForeignLinear.__call__): no AsType launch before / after, bit-identical
DENSE_GEMV = os.environ.get("KYOJIN_GLM_DENSE_GEMV", "0") not in ("0", "")          # flag glmdense (default off): fp8 / fp8b linears of 1..DENSE_GEMV_ROWS rows run the G3 GEMV (foreign_metal.gemv8g3_source: arithmetic E4M3 decode, x staged once per threadgroup, reduce-scatter)
DENSE_GEMV_ROWS = int(os.environ.get("KYOJIN_GLM_DENSE_GEMV_ROWS", "4"))
DENSE_Q_ALL = os.environ.get("KYOJIN_GLM_DENSE_Q_ALL", "0") not in ("0", "")        # QUALITY GATE ONLY (with DENSE_Q): every row count reads the 6-bit copy (decoded to fp16 per call + one matmul), so a prefill-side PPL run sees the 6-bit weights
ROWS_CAP = [0]                                                                      # flag glm_vrows: row cap of the G3 / 6-bit GEMV for the ONE call glm_dense_rows.call() is making (0 = DENSE_GEMV_ROWS); never set outside that call
DENSE_Q = os.environ.get("KYOJIN_GLM_DENSE_Q", "0") not in ("0", "")                # flag glmdenseq (default off, needs KYOJIN_GLM_DENSE_GEMV): the fp8 / fp8b linears also get a symmetric 6-bit g32 copy (glm_dense6, 6.5 bits / weight) made at load; the 1..DENSE_GEMV_ROWS row GEMV reads it (-21 % bytes), more rows keep the fp8 bytes (+0.8 B / weight resident)


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
    return (v.dtype == mx.float32 and v.ndim == 1 and not name.endswith("linear_attn.norm.weight")    # gdn_norm's fused kernel takes that one as fp32
            and not re.search(r"(_hc\.|\.hc_)(base|scale)$|hc_(attn|ffn|head)_(base|scale)$", name))           # mHC mixes: fp32 in the fused hc kernels


def act_dtype(default):
    """Flag act_dtype (KYOJIN_FOREIGN_ACT_DTYPE, read at load): None = off, else the mx dtype ('auto' = `default`, the load dtype) of the F32 parameters the forward multiplies into the hidden state."""
    v = os.environ.get("KYOJIN_FOREIGN_ACT_DTYPE", "").strip().lower()
    if v in ("", "0", "off"):
        return None
    return {"bf16": mx.bfloat16, "bfloat16": mx.bfloat16, "fp16": mx.float16, "float16": mx.float16, "auto": default, "1": default}[v]


_KEEP_F32 = re.compile(r"(^|\.)(A_log|dt_bias)$|linear_attn\.norm\.weight$|(_hc\.|\.hc_)(base|scale)$|hc_(attn|ffn|head)_(base|scale)$")      # the class computes these in fp32 on purpose


def _cast_act(weights: dict, adt) -> int:
    n = 0
    for k, v in list(weights.items()):
        if v.dtype == mx.float32 and not _KEEP_F32.search(k):
            weights[k] = v.astype(adt)
            n += 1
    return n


def g4_dense4() -> bool:
    """Flag glm_dense4 (KYOJIN_GLM_DENSE4, default 0, read at load): a checkpoint converted by tools/glm_dense4_convert.py (dense linears NVFP4 instead of FP8) also serves the MTP
    head's NVFP4 linears as stored (ForeignLinear in the draft head: ~0.12 GB per step instead of the 0.37 GB fp16 decode). The trunk needs no flag: its NVFP4 linears are read in place."""
    return os.environ.get("KYOJIN_GLM_DENSE4", "0") not in ("0", "")


legacy5_stored = GF.legacy5_stored      # flag KYOJIN_LEGACY5_STORED (default 0): Q4_1 / Q5_0 / Q5_1 read as stored; the definition sits in foreign_gguf (numpy only, the tools read it)


def gemvk_partial() -> bool:
    """Flag KYOJIN_GEMVK_PARTIAL (default 0, read per call): the K-class GEMV / gather / glu kernels (foreign_metal.gemvk_source) also take a stored matrix whose width k is not a
    multiple of the 512-weight chunk (k % 32 == 0: Q4_1 / Q5_0 / Q5_1 / Q2_0 experts of 640 / 704, K / IQ types of 768, 1280). Off: those matrices decode into a transient
    buffer, as before."""
    return os.environ.get("KYOJIN_GEMVK_PARTIAL", "0") not in ("0", "")


def moe_partialk() -> bool:
    """Flag KYOJIN_MOE_PARTIALK (default off, read per call): routed-expert stacks (E set) take the partial-chunk gather / pair kernels at k % 32 == 0, k % 512 != 0 for the
    legacy 32-blocks (q4_1 / q5_0 / q5_1: the K-class partial text, as gemvk_partial but for experts only) and iq4_nl (gemv_source partial text), and the grouped prefill
    (foreign_kqgmm) reads them as stored. Dense matrices are untouched. KYOJIN_LEGACY5_STORED (load time) still decides whether the q4_1 / q5 files are kept stored."""
    return os.environ.get("KYOJIN_MOE_PARTIALK", "0") not in ("0", "")


def moe_partialk_fp4() -> bool:
    """Flag KYOJIN_MOE_PARTIALK_FP4 (default off, read per call; sibling of KYOJIN_MOE_PARTIALK for the NVFP4 routed experts): stacks with k % 32 == 0, k % 1024 != 0 (Qwen3.8-Flash-Next
    down, k = 640) take the g2a gather / pair kernels in their partial-unit text (units past the row skipped), and the decode MLP gathers instead of the per-expert dequant loop.
    glm_moe / glm_moe_rows keep their own k % 1024 gate. Off: g2a_ok is byte-for-byte what it was."""
    return os.environ.get("KYOJIN_MOE_PARTIALK_FP4", "0") not in ("0", "")


def kstored() -> bool:
    """Flag kstored (KYOJIN_KSTORED, default 2, read at load): GGUF Q5_K / Q6_K / IQ4_XS matrices stay as stored on the device (GEMV for few rows). 2 = the same plus a
    resident fp16 copy for the rows above (kept only while it fits, _f16copy_fits). 0 = decoded to fp16 at load, as before."""
    return os.environ.get("KYOJIN_KSTORED", "1") not in ("0", "")


_FP8B_CACHE: dict = {}


_GGUF_FMTS = tuple(FP.FORMATSK) + tuple(f for f in FP.FORMATS if f != "nvfp4") + ("q8_0",)      # the formats a GGUF reader produces (nvfp4 is the converted packs' own, merged above)


class Split:
    """The members of a sanitize-fused linear whose GGUF types differ, each kept as its stored Pack (no decoded copy); served by SplitLinear."""
    fmt = "split"

    def __init__(self, packs):
        self.packs = list(packs)
        self.n, self.k = sum(p.n for p in self.packs), self.packs[0].k


def _concat_rows_exact(packs):
    """Same-input linears that sanitize concatenated along the output rows, as one Pack. fp8b members (one byte per weight, fp16 block scales with a
    power-of-two per-tensor gscale) get one common gscale: each member's meta is rescaled by a power of two, which fp16 holds exactly (checked)."""
    p0 = packs[0]
    if len(packs) == 1:
        return p0
    if p0.fmt == "nvfp4" and all(p.fmt == "nvfp4" and p.k == p0.k for p in packs):       # G4 converted dense linears: one global scale per fused group (glm_dense4_convert), bytes and block scales concatenate as they are
        if any(float(p.gscale) != float(p0.gscale) for p in packs):
            raise ValueError(f"sanitize concatenated NVFP4 linears with different global scales {[float(p.gscale) for p in packs]}: re-run tools/glm_dense4_convert.py (one global scale per fused group)")
        return FP.Pack("nvfp4", sum(p.n for p in packs), p0.k, np.concatenate([np.ascontiguousarray(p.qs) for p in packs], 0),
                       np.concatenate([np.ascontiguousarray(p.meta) for p in packs], 0), p0.gscale)
    if (p0.fmt in FP.FORMATSK or p0.fmt == "q8_0") and all(p.fmt == p0.fmt and p.k == p0.k and np.ndim(p.gscale) == 0 and float(p.gscale) == float(p0.gscale) for p in packs):      # GGUF K / IQ / Q8_0 blocks read as stored: every array is row-major per output row, so rows concatenate as stored bytes (exact)
        ex = {}
        for key in p0.extra:
            vs = [p.extra.get(key) for p in packs]
            if all(isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == p.n for v, p in zip(vs, packs)):
                ex[key] = np.concatenate([np.ascontiguousarray(v) for v in vs], 0)
            elif all(isinstance(v, np.ndarray) and isinstance(vs[0], np.ndarray) and np.array_equal(v, vs[0]) for v in vs) or all(not isinstance(v, np.ndarray) and v == vs[0] for v in vs):
                ex[key] = vs[0]
            else:
                raise ValueError(f"sanitize concatenated {p0.fmt} linears whose extra {key!r} is not per-row: not merged")
        return FP.Pack(p0.fmt, sum(p.n for p in packs), p0.k, np.concatenate([np.ascontiguousarray(p.qs) for p in packs], 0),
                       None if p0.meta is None else np.concatenate([np.ascontiguousarray(p.meta) for p in packs], 0), p0.gscale, ex)
    if any(p.fmt in _GGUF_FMTS for p in packs):                  # GGUF members of different types (GLM-5.3 MLA qkv_a: attn_q_a Q6_K + attn_kv_a_mqa Q8_0): each member stays stored, outputs concatenated (SplitLinear)
        bad = [(p.fmt, p.k, np.ndim(p.qs)) for p in packs if p.k != p0.k or np.ndim(p.qs) != 2 or not (p.fmt in _GGUF_FMTS or p.fmt == "dense")]
        if bad:
            raise ValueError(f"sanitize concatenated GGUF linears of formats {[p.fmt for p in packs]} that cannot be split into stored members: {bad}")
        return Split(packs)
    if any(p.fmt != "fp8b" or p.k != p0.k for p in packs):
        raise ValueError(f"sanitize concatenated linears of formats {[p.fmt for p in packs]}: only fp8b members (or NVFP4 with one global scale) are merged as stored")
    gs = max(float(p.gscale) for p in packs)
    metas = []
    for p in packs:
        m = (p.meta.astype(np.float32) * np.float32(float(p.gscale) / gs)).astype(np.float16)
        assert np.array_equal(m.astype(np.float32) * np.float32(gs), p.meta.astype(np.float32) * np.float32(p.gscale)), "fp8b rescale not exact"
        metas.append(m)
    return FP.Pack("fp8b", sum(p.n for p in packs), p0.k, np.concatenate([np.ascontiguousarray(p.qs) for p in packs], 0),
                   np.concatenate(metas, 0), np.float32(gs))


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
    if g.ndim > 2:                                                # modelopt FP8_PB_WO: [n/bn, 1, k/bk, 1] block grid
        g = g.reshape([d for d in g.shape if d != 1] if sum(d != 1 for d in g.shape) == 2 else (g.shape[0], -1))
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
def _gemv_kern(fmt, nbr, gather, rowgs=False, lut=False, glu=False, partial=False):
    tag = f"{'g' if gather else 's'}" + ("r" if rowgs else "") + ("l" if lut else "") + ("u" if glu else "") + ("p" if partial else "")
    return KN._kern(("fgemv", fmt, nbr, gather) + ((rowgs, lut) if rowgs or lut else ()) + (("glu",) if glu else ()) + (("part",) if partial else ()), name=f"kyojin_fg_gemv_{fmt}_n{nbr}{tag}",
                    input_names=FM.gemv_inputs(gather, glu), output_names=["y"], header=FM.header(fmt, "gemv", lut) + (FM.GLU_HELPER if glu else ""),
                    source=FM.gemv_source(fmt, nbr, gather, rowgs, glu, partial))


DENSE_GLU = os.environ.get("KYOJIN_FOREIGN_DENSE_GLU", "1") not in ("0", "")   # dense gated MLP (dense layers, shared experts): silu(gate) * up inside the down projection's launch
G2A_NBR, G2A_SGN = int(os.environ.get("KYOJIN_FOREIGN_G2A_NBR", "1")), int(os.environ.get("KYOJIN_FOREIGN_G2A_SGN", "16"))   # 1 row per simdgroup, 16 simdgroups: R1..R4 sweep, 512 threads


def g2a_on() -> bool:
    """Flag g2a (KYOJIN_FOREIGN_G2A, default 1, read per call): the register-fed NVFP4 gather GEMV of foreign_metal.gemv_g2_source for routed-expert slots, and in
    ForeignSwitchGLU the pair launch + one gated-activation launch (instead of 2 gathers + the ~8 elementwise ops of _glu)."""
    return os.environ.get("KYOJIN_FOREIGN_G2A", "1") not in ("0", "")


@functools.lru_cache(maxsize=None)
def _g2a_kern(partial: bool = False):
    if partial:
        return KN._kern(("fg2a", "pk"), name="kyojin_fg_g2a_nvfp4_pk", input_names=FM.gemv_inputs(True), output_names=["y"], header=FM.header("nvfp4", "gemv", False), source=FM.gemv_g2_source(True))
    return KN._kern(("fg2a",), name="kyojin_fg_g2a_nvfp4", input_names=FM.gemv_inputs(True), output_names=["y"], header=FM.header("nvfp4", "gemv", False), source=FM.gemv_g2_source())


GATEUP_PAIR = os.environ.get("KYOJIN_EXPERT_GATEUP_PAIR", "0") not in ("0", "")      # flag expert_gateup_pair (default off; flags.set updates it): ForeignSwitchGLU decode, gate | up in one gather pass + the activation folded into the down launch
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
def _gemv8_kern(fmt, nbr, gather=False, glu=False, xin="h", yout="h"):
    if xin != "h" or yout != "h":                     # flag foreign_castfree: x / y dtype in the kernel (own key and name; the stock kernels are untouched)
        return KN._kern(("fgemv8", fmt, nbr, "xio", xin, yout), name=f"kyojin_fg_gemv8_{fmt}_n{nbr}_x{xin}y{yout}", input_names=FM.gemv8_inputs(fmt), output_names=["y"],
                        header=FM.header8(fmt) + FM.GEMV8_XIO_HELPER, source=FM.gemv8_source(fmt, nbr, False, False, xin, yout))
    return KN._kern(("fgemv8", fmt, nbr) + (("g",) if gather else ()) + (("glu",) if glu else ()), name=f"kyojin_fg_gemv8_{fmt}_n{nbr}" + ("g" if gather else "") + ("u" if glu else ""),
                    input_names=FM.gemv8_inputs(fmt, gather, glu), output_names=["y"], header=FM.header8(fmt) + (FM.GLU_HELPER if glu else ""),
                    source=FM.gemv8_source(fmt, nbr, gather, glu))


@functools.lru_cache(maxsize=None)
def _gemv8g3_kern(r, nr, u, nsg, meta, xmode="h", limit=0.0):
    return KN._kern(("fg3", r, nr, u, nsg, meta, xmode, limit), name=f"kyojin_fg_g3_r{r}_n{nr}_u{u}_s{nsg}" + ("m" if meta else "") + xmode + (f"{limit:g}".replace(".", "p") if xmode == "glu" else ""),
                    input_names=["x", "qs"] + (["mt"] if meta else []) + ["gs"], output_names=["y"], header=FM.G3_HDR, source=FM.gemv8g3_source(r, nr, u, nsg, meta, xmode, limit))


@functools.lru_cache(maxsize=None)
def _mma8_kern(fmt, ntn):
    return KN._kern(("fmma8", fmt, ntn), name=f"kyojin_fg_mma8_{fmt}_t{ntn}", input_names=["xp"] + FM.gemv8_inputs(fmt)[1:], output_names=["y"], header=FM.header8(fmt),
                    source=FM.mma8_source(fmt, ntn))


@functools.lru_cache(maxsize=None)
def _deq8_kern(fmt):
    return KN._kern(("fdeq8", fmt), name=f"kyojin_fg_deq8_{fmt}", input_names=FM.gemv8_inputs(fmt)[1:], output_names=["w"], header=FM.header8(fmt),
                    source=FM.dequant8_source(fmt))


def _gemvk_kern(fmt, nbr, gather=False, glu=False, act=None, partial=False):
    fast = FM.qdec_fast(fmt)                                                              # flag qdec_fast (read per call) is part of the cache key
    if fast:
        FM.PATHS["qdec_fast"] += 1
    return _gemvk_kern_c(fmt, nbr, gather, glu, act, partial, fast)


# cached: the K / IQ headers rebuild their grid tables as source text (IQ3_S ~40 us of host time per call)
@functools.lru_cache(maxsize=None)
def _gemvk_kern_c(fmt, nbr, gather, glu, act, partial, fast):
    return KN._kern(("fgemvk", fmt, nbr) + (("g",) if gather else ()) + (("glu",) if glu else ()) + ((act,) if act else ()) + (("part",) if partial else ()) + (("qf",) if fast else ()),
                    name=f"kyojin_fg_gemvk_{fmt}_n{nbr}" + ("g" if gather else "") + ("u" if glu else "") + ("_situ" if act else "") + ("_p" if partial else "") + ("_qf" if fast else ""),
                    input_names=FM.gemvk_inputs(fmt, gather, glu), output_names=["y"], header=FM.hdrk(fmt) + (FM.glu_helper(act) if glu else ""),
                    source=FM.gemvk_source(fmt, nbr, gather, glu, partial))


_gemvk_kern.cache_clear = _gemvk_kern_c.cache_clear


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
    cm = None                                              # flag KYOJIN_MOE_SCALE_COMPACT: dict of the compact scale arrays of a stacked NVFP4 expert projection (moe_scale_compact.ARRAYS), mx arrays
    _meta = None                                           # the resident expanded scales (None with KYOJIN_MOE_SCALE_COMPACT: then `meta` expands from `cm` / the parent per access, transient)
    _msrc = None                                           # (parent DevPack, expert) of an expert view of a compact stack
    _mdrop = False
    tl = None                                              # flag glmtile: tile-layout array of a stacked NVFP4 expert projection (qs / meta are None then)
    g6 = None                                              # flag glmdenseq: (words uint32 [n, k / 512 * 96], scales half [n, k / 32]) symmetric 6-bit copy of a fp8 / fp8b linear (glm_dense6)
    mq = None                                              # flag mqcopy: (wq, scales, biases, mode, group, unit_gs, bits) resident MLX-native quantized copy, 4-bit formats only

    @property
    def meta(self):
        m = self._meta
        if m is not None:
            return m
        if self._msrc is not None:
            par, e = self._msrc
            return MC.expand_dev(par.cm, par.n, par.k // 16, e, 1)[0]
        if self._mdrop:
            return MC.expand_dev(self.cm, self.n, self.k // 16, 0, self.E)
        return None

    @meta.setter
    def meta(self, v):
        self._meta = v

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
            elif DENSE_Q and DENSE_GEMV and self.fmt in ("fp8", "fp8b") and self.k % 512 == 0:
                self._make_g6()
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
        st = _stack_par if self.E else (lambda a: a[0])
        self.qs = mx.array(np.ascontiguousarray(st([np.ascontiguousarray(p.qs).view(np.uint32).reshape(p.n, p.k // 8) for p in ps])))
        self.meta = mx.array(np.ascontiguousarray(st([np.ascontiguousarray(p.meta) for p in ps])))
        self.gs = mx.array(np.array([float(p.gscale) for p in ps], np.float32))
        self.w = None
        self.w16 = None
        self._views = {}
        if self.E and self.fmt == "nvfp4" and TILE and FT.tile_ok(self.n, self.k, TILE_UB):
            self.tl = FT.to_tile(self.qs, self.meta, TILE_UB)          # one projection stack at a time; the stored arrays are dropped (never both resident)
            mx.eval(self.tl)
            self.qs = self.meta = None
        if self.E and self.fmt == "nvfp4" and self.meta is not None and MC.enabled():          # flag KYOJIN_MOE_SCALE_COMPACT: the compact scale form read by the decode MoE kernels (glm_moe_e / glm_moe_rows)
            self.cm = MC.build_dev(ps)
            if MC.drop_meta():                                       # the expanded scales are not kept: prefill / gather / slice paths expand transiently from `cm`
                mx.eval(MC.cm_list(self))
                self._meta, self._mdrop = None, True
        if not self.E and mq_copy() and p0.fmt in ("q4_0", "q4_k", "nvfp4", "mxfp4") and _f16copy_fits(p0.n * p0.k * 5 // 8):   # ~1.1x the stored bytes
            wq, sc, bi, mode, g, bits = FP.mq_arrays(p0)
            self.mq = (mx.array(wq), mx.array(sc), None if bi is None else mx.array(bi), mode, g, mode != "nvfp4" or float(p0.gscale) == 1.0, bits)
            mx.eval([a for a in self.mq[:3] if a is not None])
        if not self.E and f4_copy() and _f16copy_fits(p0.n * p0.k * 2):  # flag f4copy: resident fp16 copy (4x the bytes), the matmul from F4_ROWS rows
            self.w16 = self.dequant()
            mx.eval(self.w16)

    def _make_g6(self, rows=4096):
        """Flag glmdenseq: the symmetric 6-bit g32 copy (glm_dense6.quantize_mx of the decoded fp8 weights), a few thousand rows at a time (transient fp16 / fp32 stay small)."""
        ws, ss = [], []
        for a in range(0, self.n, rows):
            b = min(self.n, a + rows)
            v = object.__new__(DevPack)
            v.fmt, v.n, v.k, v.E, v.rowgs, v.w16, v.g6 = self.fmt, b - a, self.k, None, True, None, None
            v.qs, v.meta, v.gs = self.qs[a:b], None if self.meta is None else self.meta[a:b], self.gs[a:b]
            wq, sc = G6.quantize_mx(v.dequant())
            mx.eval(wq, sc)
            ws.append(wq)
            ss.append(sc)
        self.g6 = (ws[0] if len(ws) == 1 else mx.concatenate(ws, axis=0), ss[0] if len(ss) == 1 else mx.concatenate(ss, axis=0))
        mx.eval(self.g6)
        mx.clear_cache()

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
        d.g6 = (mx.concatenate([m.g6[0] for m in devs], axis=0), mx.concatenate([m.g6[1] for m in devs], axis=0)) if all(m.g6 is not None for m in devs) else None
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
        d.g6 = None if self.g6 is None else (cut(self.g6[0]), cut(self.g6[1]))
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
        return [a for a in (self.qs, self._meta, self.gs, self.w, self.w16, self.tl, getattr(self, "hi", None)) if a is not None] + (MC.cm_list(self) if self.cm is not None else []) + ([a for a in self.mq[:3] if a is not None] if self.mq else []) + (list(self.g6) if self.g6 is not None else [])

    def nbytes(self):
        return sum(a.nbytes for a in (self.qs, self._meta, self.gs, self.tl, getattr(self, "hi", None)) if a is not None) + (sum(a.nbytes for a in MC.cm_list(self)) if self.cm is not None else 0) + (sum(a.nbytes for a in self.g6) if self.g6 is not None else 0)

    def untiled(self):
        """A transient stored-layout copy of a tile-layout expert stack (prefill / grouped paths read the stored arrays); the caller drops it after the call."""
        assert self.tl is not None
        v = object.__new__(DevPack)
        v.__dict__.update(self.__dict__)
        v.qs, v.meta = FT.from_tile(self.tl, self.n, self.k, TILE_UB)
        v.tl, v._views = None, {}
        v._parent = self                                     # per-weight caches (foreign_pfq8 row_emax) live on the resident pack, not on this transient copy
        return v

    def expert(self, e):
        assert self.tl is None, "expert(): tile-layout stack (use untiled())"
        v = self._views.get(e) if self.w is None else None
        if v is None:
            v = object.__new__(DevPack)
            v.fmt, v.n, v.k, v.E, v.rowgs, v.w16 = self.fmt, self.n, self.k, None, False, None
            if self._mdrop:                                  # compact stack: the view expands its own scales (one expert, transient) from the parent
                v.w, v.qs, v.gs, v._views, v._msrc = None, self.qs.reshape(-1, self.qs.shape[-1])[e * self.n:(e + 1) * self.n], self.gs[e:e + 1], {}, (self, e)
                self._views[e] = v
                return v
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
        return (self.k % 512 == 0 or (self.fmt in FM.FORMATS8 and self.k % 32 == 0) or (self.fmt in FM.FORMATSK and self.k % 32 == 0 and gemvk_partial())
                or (self.k % 32 == 0 and moe_partialk() and bool(getattr(self, "E", 0)) and (self.fmt in FM.FORMATSK or self.fmt == "iq4_nl"))
                or (self.fmt == "nvfp4" and self.k % 1024 != 0 and bool(getattr(self, "E", 0)) and self.g2a_ok(True))) and self.n % (8 * (nbr or NBR)) == 0

    def _part(self) -> bool:
        """The K-class kernel of this matrix is the partial-chunk text (flag gemvk_partial, k % 512 != 0): same source as ever for the widths that had a kernel."""
        return self.k % 512 != 0 and (self.fmt in FM.FORMATSK or self.fmt == "iq4_nl")

    def apply(self, x2):
        R = x2.shape[0]
        if R >= PI8.hook_rows() and not self.E and PI8.on():                           # flag pf_gemm_i8_dense (default off, read per call): int8 x int8 -> int32 prefill GEMM on an int8 copy of the weight
            y = PI8.linear(self, x2)
            if y is not None:
                return y
        if self.fmt == "dense":
            return _matmul(x2, self.w)
        if DENSE_Q_ALL and self.g6 is not None and R > DENSE_GEMV_ROWS:
            return _matmul(x2, self.dequant())
        if DENSE_GEMV:
            if self.g3_ok(R):
                FD.fire("dense_gemv")
                return self._gemv8g3(x2)
            if FD.AUDIT:
                FD.fb("dense_gemv", self.g3_why(R))
        if PI8.native_on():                                                             # flag KYOJIN_PF_GEMM_I8_NATIVE (default off): stored Q8_0 codes x dynamic int8 activations, per-32 scales, no weight re-grid
            y = PI8.native_linear(self, x2) if not self.E else None
            if y is not None:
                return y
        if KQ.dense_ok(self, R):                                                        # flag kqdense (KYOJIN_KQ_DENSE, default off): big-row Q8_0 linear through the grouped kernel (decode shared by KYOJIN_KQ_RT row tiles)
            return KQ.dense(self, x2)
        if self.fmm8_ok(R):                                                             # flag foreign_fmm8 (default off, read per call): few-row simdgroup-matrix product on the stored Q8_0 bytes
            return self._fmm8(x2)
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
        if R >= PI8.hook_rows() and not self.E and PI8.on() and not PI8.why_not(self, R):
            y = PI8.linear(self, x2)
            return [y[:, a:b] for a, b in zip(cuts[:-1], cuts[1:])]
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
        return bool(self.wide_ok(R) or self.fmm8_ok(R) or self._zq_route(R) or self._f2_ok(R) or self._zq8_ok(R) or (self.fmt in FM.FORMATS8 and R <= DENSE8_ROWS))

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
        return _gemvk_kern(self.fmt, NBR, partial=self._part())(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NBR", NBR)], grid=(self.n // (8 * NBR) * 128, 1, 1),
                                          threadgroup=(128, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def _mmak(self, x2, ntn=2):
        R = x2.shape[0]
        ins = [mx.take(x2, _xpermk_idx(self.fmt, self.k), axis=1), self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
        return _mmak_kern(self.fmt, ntn)(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NTN", ntn)], grid=(-(-self.n // (32 * ntn)) * 128, 1, 1),
                                         threadgroup=(128, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def _nbr8(self, nbr=None):
        """Output rows per simdgroup of the 8-bit stored GEMV: the global NBR, or NBR_SMALL on a matrix under NBR_SMALL_MAX weights (flag foreign_nbr_small: more threadgroups on the small projections)."""
        if nbr:
            return nbr
        if NBR_SMALL and self.n * self.k < NBR_SMALL_MAX and self.n % (8 * NBR_SMALL) == 0:
            return NBR_SMALL
        return globals()['NBR']

    def _gemv8(self, x2, nbr=None):
        R, NBR = x2.shape[0], self._nbr8(nbr)
        ins = [x2, self.qs, self.gs] if self.meta is None else [x2, self.qs, self.meta, self.gs]
        return _gemv8_kern(self.fmt, NBR)(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NBR", NBR)], grid=(self.n // (8 * NBR) * 128, 1, 1),
                                          threadgroup=(128, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def cf_ok(self, R) -> bool:
        """Flag foreign_castfree: apply() of R rows would end in _gemv8 (the same chain of tests, in order). Only called with the flag on."""
        if self.fmt not in FM.FORMATS8 or self.E or not 1 <= R <= GEMV8_ROWS:
            return False
        if R >= PI8.hook_rows() and PI8.on():
            return False
        if DENSE_Q_ALL and self.g6 is not None and R > DENSE_GEMV_ROWS:
            return False
        if DENSE_GEMV and self.g3_ok(R):
            return False
        return self.fmm8_ok(R) or (not self.wide_ok(R) and self.gemv_ok())

    def gemv8_cf(self, x2, out_dtype):
        """_gemv8 with x2 half / float32 / bfloat16 read as stored and y in out_dtype (float16 / float32): the values of
        x2.astype(float16) -> _gemv8 -> .astype(out_dtype), bit for bit, in one launch (flag foreign_castfree)."""
        if self.fmm8_ok(x2.shape[0]):                                                   # flag foreign_fmm8: the same cast-free contract (x read as stored, y in out_dtype)
            return F8.run(x2, self.qs, self.meta, self.gs, self.n, self.k, out_dtype)
        R, NBR = x2.shape[0], self._nbr8()
        xin, yout = _CF_X[x2.dtype], _CF_Y[out_dtype]
        ins = [x2, self.qs, self.gs] if self.meta is None else [x2, self.qs, self.meta, self.gs]
        return _gemv8_kern(self.fmt, NBR, False, False, xin, yout)(inputs=ins, template=[("R", R), ("KIN", self.k), ("N", self.n), ("NBR", NBR)],
                                                                  grid=(self.n // (8 * NBR) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(R, self.n)],
                                                                  output_dtypes=[out_dtype])[0]

    def g3_why(self, R) -> str:
        """Flag audit: why `g3_ok` is false for this matrix and row count (the guards in order; only called under fired.ON)."""
        if not 1 <= R <= max(DENSE_GEMV_ROWS, ROWS_CAP[0]):
            return f"rows>{max(DENSE_GEMV_ROWS, ROWS_CAP[0])}" if R > 1 else "rows<1"
        if self.fmt not in ("fp8", "fp8b"):
            return "fmt=" + str(self.fmt)
        if self.E:
            return "experts"
        if not self.rowgs:
            return "no_rowgs"
        if self.w16 is not None:
            return "w16_copy"
        if self.k % 512 != 0:
            return "k%512"
        if self.qs is None:
            return "no_qs"
        return "?"

    def _ones(self):
        o = self.__dict__.get("_g6_ones")
        if o is None:
            o = self._g6_ones = mx.ones((self.n,), mx.float32)
            mx.eval(o)
        return o

    def g3_ok(self, R) -> bool:
        """Flag glmdense: this matrix and row count take the G3 GEMV (one-byte formats of a plain matrix, k % 512 == 0, no resident fp16 copy)."""
        return (DENSE_GEMV and 1 <= R <= max(DENSE_GEMV_ROWS, ROWS_CAP[0]) and self.fmt in ("fp8", "fp8b") and not self.E and self.rowgs and self.w16 is None and self.k % 512 == 0
                and self.qs is not None)

    def _gemv8g3(self, x2, limit=None):
        """x2 [R, k] half / float32 / bfloat16 (the cast to half is done while the kernel stages x), or with limit: the [R, 2 k] half gate | up row of a gated MLP,
        whose silu(min(gate, limit)) * clip(up, -limit, limit) is the input of this (down) projection."""
        R = x2.shape[0]
        if self.g6 is not None:                           # flag glmdenseq: the 6-bit copy (same launch shape, same x staging / glu input)
            return G6.gemv(x2, self.g6[0], self.g6[1], self._ones(), self.n, self.k, limit)
        nr, nsg, u = FM.gemv8g3_config(R, self.n, self.k)
        kc = FM.gemv8g3_kc(R, self.k)
        xm = "glu" if limit is not None else {mx.float16: "h", mx.float32: "f", mx.bfloat16: "b"}[x2.dtype]
        ins = [x2, self.qs, self.gs] if self.meta is None else [x2, self.qs, self.meta, self.gs]
        return _gemv8g3_kern(R, nr, u, nsg, self.meta is not None, xm, float(limit or 0.0))(
            inputs=ins, template=[("KIN", self.k), ("N", self.n), ("KC", kc), ("NPH", -(-self.k // kc))], grid=(-(-self.n // (nr * nsg)) * nsg * 32, 1, 1),
            threadgroup=(nsg * 32, 1, 1), output_shapes=[(R, self.n)], output_dtypes=[mx.float16])[0]

    def glu_dense_ok(self, R) -> bool:
        """apply(gate|up) -> silu * mul -> apply(self) can run as ONE launch for self: the one-row GEMV class of this format takes R rows (same kernel family as apply())."""
        if not DENSE_GLU or self.E or self.fmt == "dense" or self.wide_ok(R) or self.fmm8_ok(R):
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
            return _gemvk_kern(self.fmt, NB, False, True, None, self._part())(inputs=ins, template=tpl + [("NBR", NB)], grid=(self.n // (8 * NB) * 128, 1, 1), threadgroup=(128, 1, 1),
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

    def fmm8_ok(self, R):
        return (bool(F8.MIN_ROWS) and F8.MIN_ROWS <= R <= min(F8.MAX_ROWS, F8.ROWS_CAP) and self.fmt == "q8_0" and not self.E and self.meta is not None
                and self.w16 is None and F8.eligible(R, self.n, self.k))

    def _fmm8(self, x2):
        """x2 [R, k] -> [R, n] fp16 on the stored Q8_0 bytes (fmm8.run: one weight read, K split over the simdgroups of a threadgroup)."""
        return F8.run(x2, self.qs, self.meta, self.gs, self.n, self.k, mx.float16)

    def wide_ok(self, R):
        hi = int(WIDE_MAX_ENV) if WIDE_MAX_ENV else WIDE_MAX_FMT.get(self.fmt, 32)
        lo = int(WIDE_MIN_ENV) if WIDE_MIN_ENV else (WIDE_MIN_FMT.get(self.fmt, WIDE_MIN) if self.n * self.k >= WIDE8_MIN_SIZE else WIDE_MIN)
        return wide_on() and lo <= R <= hi and not self.E and self.k % 128 == 0 and (self.fmt in FM._FMMA_PAIR or self.fmt in FM.FORMATS8) and self.w16 is None

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
        if DENSE_Q_ALL and self.g6 is not None:
            return G6.dequant_mx(self.g6[0], self.g6[1], self.n, self.k)
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
        if self.tl is not None:
            return FT.gather(x, xrows, eids, self.tl, self.gs, self.n, self.k, TILE_UB, TILE_DOWN_NSG)
        nbr = NBRK if self.fmt in FM.FORMATSK else NBR                   # the rows per simdgroup the one-row product of this class uses
        grid = (self.n // (8 * nbr) * 128, S, 1)
        tpl = [("R", 1), ("KIN", self.k), ("N", self.n), ("NBR", nbr)]
        if self.g2a_ok(True):
            pk = self.k % 1024 != 0
            FD.fire("g2a_gather_pk" if pk else "g2a_gather")
            return _g2a_kern(pk)(inputs=[x, xrows, eids, self.qs, self.meta, self.gs], template=[("KIN", self.k), ("N", self.n), ("NBR", G2A_NBR), ("SGN", G2A_SGN)],
                               grid=(self.n // (G2A_SGN * G2A_NBR) * 32 * G2A_SGN, S, 1), threadgroup=(32 * G2A_SGN, 1, 1), output_shapes=[(S, self.n)], output_dtypes=[mx.float16])[0]
        if FD.AUDIT:
            FD.fb("g2a_gather", self.g2a_why())
        if self.fmt in FM.FORMATSK:
            ins = [x, xrows, eids, self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
            if self._part() and moe_partialk():
                FD.fire("moe_partialk_gather_" + self.fmt)
            return _gemvk_kern(self.fmt, nbr, True, False, None, self._part())(inputs=ins, template=tpl, grid=grid, threadgroup=(128, 1, 1), output_shapes=[(S, self.n)], output_dtypes=[mx.float16])[0]
        if self.fmt in FM.FORMATS8:
            ins = [x, xrows, eids, self.qs, self.gs] if self.meta is None else [x, xrows, eids, self.qs, self.meta, self.gs]
            return _gemv8_kern(self.fmt, nbr, True)(inputs=ins, template=tpl, grid=grid, threadgroup=(128, 1, 1), output_shapes=[(S, self.n)], output_dtypes=[mx.float16])[0]
        part = self.k % 512 != 0 and self.fmt == "iq4_nl"
        if part:
            FD.fire("moe_partialk_gather_iq4nl")
        return _gemv_kern(self.fmt, nbr, True, False, cfmt(), False, part)(inputs=[x, xrows, eids, self.qs, self.meta, self.gs], template=tpl, grid=grid, threadgroup=(128, 1, 1),
                                               output_shapes=[(S, self.n)], output_dtypes=[mx.float16])[0]

    def g2a_why(self) -> str:
        """Flag audit: why `g2a_ok` is false (only called under fired.ON)."""
        if not g2a_on():
            return "flag_off"
        if self.fmt != "nvfp4":
            return "fmt=" + str(self.fmt)
        if not self.E:
            return "not_experts"
        if self.k % 1024 != 0 or self.k > 12288:
            return "k" if not moe_partialk_fp4() or self.k % 32 != 0 or self.k > 12288 else "n%sgn"
        return "n%sgn"

    def g2a_ok(self, pk: bool = False) -> bool:
        """pk (the ForeignSwitchGLU gather callers only): also k % 32 == 0, k % 1024 != 0 under flag KYOJIN_MOE_PARTIALK_FP4 (partial-unit kernel)."""
        return (g2a_on() and self.fmt == "nvfp4" and bool(self.E) and self.n % (G2A_SGN * G2A_NBR) == 0 and self.k <= 12288
                and (self.k % 1024 == 0 or (pk and self.k % 32 == 0 and moe_partialk_fp4())))

    def pair_ok(self, o) -> bool:
        """gate | up of one expert block can share a launch: same format, shape, expert count and array dtypes."""
        f = lambda d: (d.fmt, d.k, d.n, d.E, d.qs.dtype, d.qs.shape, None if d.meta is None else (d.meta.dtype, d.meta.shape), getattr(d, "hi", None) is None, getattr(d, "gs", None) is None)
        return PAIR and bool(self.E) and self.tl is None and o.tl is None and self.fmt != "dense" and self.gemv_ok() and f(self) == f(o)

    def gather_pair(self, o, x, xrows, eids):
        """(self.gather(...), o.gather(...)) in ONE launch (the two matrices are bound as stored; rows [0, S) of the output are self's, [S, 2 S) are o's)."""
        S = int(eids.shape[0])
        y = self.gather_pair_raw(o, x, xrows, eids)
        return y[:S], y[S:]

    def gather_pair_raw(self, o, x, xrows, eids):
        S = int(eids.shape[0])
        nbr = NBRK if self.fmt in FM.FORMATSK else NBR
        tpl = [("R", 1), ("KIN", self.k), ("N", self.n), ("NBR", nbr), ("SP", S)]
        grid, tg = (self.n // (8 * nbr) * 128, 2 * S, 1), (128, 1, 1)
        if self.g2a_ok(True):
            pk = self.k % 1024 != 0
            FD.fire("g2a_pair_pk" if pk else "g2a_pair")
            names, mk = FM.gemv_inputs(True)[3:], (lambda: (FM.header("nvfp4", "gemv", False), FM.gemv_g2_source(pk)))
            arrs = lambda d: [d.qs, d.meta, d.gs]
            tag, nbr = "gk" if pk else "g", G2A_NBR
            tpl = [("KIN", self.k), ("N", self.n), ("NBR", G2A_NBR), ("SGN", G2A_SGN), ("SP", S)]
            grid, tg = (self.n // (G2A_SGN * G2A_NBR) * 32 * G2A_SGN, 2 * S, 1), (32 * G2A_SGN, 1, 1)
        elif self.fmt in FM.FORMATSK:
            part = self._part()
            names, mk = FM.gemvk_inputs(self.fmt, True)[3:], (lambda: (FM.hdrk(self.fmt), FM.gemvk_source(self.fmt, nbr, True, False, part)))
            arrs = lambda d: [d.qs, d.meta] + ([d.hi] if d.hi is not None else [])
            tag = ("kp" if part else "k") + ("q" if FM.qdec_fast(self.fmt) else "")         # flag qdec_fast: its own kernel
        elif self.fmt in FM.FORMATS8:
            names, mk = FM.gemv8_inputs(self.fmt, True)[3:], (lambda: (FM.header8(self.fmt), FM.gemv8_source(self.fmt, nbr, True)))
            arrs = lambda d: [d.qs, d.gs] if d.meta is None else [d.qs, d.meta, d.gs]
            tag = "8"
        else:
            part = self._part()
            names, mk = FM.gemv_inputs(True)[3:], (lambda: (FM.header(self.fmt, "gemv", cfmt()), FM.gemv_source(self.fmt, nbr, True, False, False, part)))
            arrs = lambda d: [d.qs, d.meta, d.gs]
            tag = "pp" if part else "p"
        kern = _pair_kern(tag, self.fmt, nbr, cfmt() if tag in ("p", "pp") else False, names, mk)
        y = kern(inputs=[x, xrows, eids] + arrs(self) + arrs(o), template=tpl, grid=grid, threadgroup=tg, output_shapes=[(2 * S, self.n)], output_dtypes=[mx.float16])[0]
        return y

    def gather_glu(self, a, b, xrows, eids, act=None):
        """gather() of the down projection with the gated-MLP input silu(a) * b (a, b half [S, k], slot s reads row xrows[s]) computed while the input is staged.
        act: None = silu; ("situ", scale_gate, scale_lin) = the SiTU form (ForeignSwitchGLU.act_kind)."""
        S = int(eids.shape[0])
        assert self.glu_ok()
        nbr = NBRK
        ins = [a, b, xrows, eids, self.qs, self.meta] + ([self.hi] if self.hi is not None else [])
        return _gemvk_kern(self.fmt, nbr, True, True, act, self._part())(inputs=ins, template=[("R", 1), ("KIN", self.k), ("N", self.n), ("NBR", nbr)], grid=(self.n // (8 * nbr) * 128, S, 1),
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
        sub.cm, sub._mdrop, sub._msrc = None, False, None
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


_CF_X = {mx.float16: "h", mx.float32: "f", mx.bfloat16: "b"}       # flag foreign_castfree: x dtypes the gemv8 staging reads as stored
_CF_Y = {mx.float16: "h", mx.float32: "f"}                          # and y dtypes it writes


class _Shape:
    def __init__(self, shape):
        self.shape, self.dtype = tuple(shape), mx.float16


class ForeignLinear(nn.Module):
    """y = layer(x) for x (..., in_features); out dtype fp16 (bias added when present). Arrays are private (not module parameters)."""

    def __init__(self, pack: FP.Pack, bias: mx.array | None = None, out_dtype: mx.Dtype = mx.float16, in_perm=None):
        super().__init__()
        if isinstance(pack, Split):
            raise TypeError("a Split (mixed GGUF types) is served by SplitLinear: build it with make_linear")
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
        cf = CASTFREE and self._in_perm is None and x2.dtype in _CF_X and self.out_dtype in _CF_Y and self.dev.cf_ok(x2.shape[0])   # flag foreign_castfree
        if x2.dtype != mx.float16 and not cf and not (DENSE_GEMV and x2.dtype in (mx.float32, mx.bfloat16) and self._in_perm is None and self.dev.g3_ok(x2.shape[0])):    # glmdense: the kernel casts while it stages x
            x2 = x2.astype(mx.float16)
        if self._in_perm is not None and permuted:
            _PREP[2] = False
        elif self._in_perm is not None:
            x2 = mx.take(x2, self._in_perm, axis=1)                        # a permuted input is not the shared one; take = one Gather (x2[:, idx] adds an Arange and two casts)
        elif _PREP[0] is not x:
            _PREP[0], _PREP[1] = x, {}
        _PREP[2] = self._in_perm is None
        try:
            y = self.dev.gemv8_cf(x2, self.out_dtype) if cf else self.dev.apply(x2)
        finally:
            _PREP[2] = False
        if y.dtype != self.out_dtype:
            y = y.astype(self.out_dtype)
        if self._bias is not None:
            y = y + self._bias
        return y.reshape(*shape[:-1], self.out_features)



class SplitLinear(nn.Module):
    """A sanitize-fused linear whose members have different GGUF types: one stored ForeignLinear per member, outputs concatenated on the last axis
    (the rows of the fused matrix, in sanitize's order). Same values as one fused matrix: each output row is the same dot product."""

    def __init__(self, split: Split, bias: mx.array | None = None, out_dtype: mx.Dtype = mx.float16, in_perm=None):
        super().__init__()
        if in_perm is not None:
            raise ValueError("SplitLinear: a column permutation over mixed-type members is not supported")
        cuts = [0]
        for p in split.packs:
            cuts.append(cuts[-1] + p.n)
        self.members = [ForeignLinear(p, None if bias is None else bias[a:b], out_dtype) for p, a, b in zip(split.packs, cuts[:-1], cuts[1:])]
        self.in_features, self.out_features, self.fmt, self.out_dtype = split.k, split.n, "split", out_dtype

    def __call__(self, x: mx.array, permuted: bool = False) -> mx.array:
        if permuted:
            raise ValueError("SplitLinear: no permuted input")
        return mx.concatenate([m(x) for m in self.members], axis=-1)


def make_linear(pack, bias: mx.array | None = None, out_dtype: mx.Dtype = mx.float16, in_perm=None):
    """ForeignLinear for a Pack, SplitLinear for a Split (_concat_rows_exact of mixed GGUF types)."""
    return (SplitLinear if isinstance(pack, Split) else ForeignLinear)(pack, bias, out_dtype, in_perm)

_GLU_SRC = r"""
    const uint i = thread_position_in_grid.x;
    float4 a = float4(((device const half4*)y2)[i]);
    float4 b = float4(((device const half4*)y2)[i + uint(NQ)]);
    LIMIT_CLAMP
    const float4 e = exp(-abs(a));                                    // mx.sigmoid: 1 / (1 + exp(-|x|)), 1 - that for x < 0
    const float4 y_ = 1.0f / (1.0f + e);
    const float4 sg = select(y_, 1.0f - y_, a < 0.0f);
    ((device half4*)h)[i] = half4((a * sg) * b);
"""


_GLU_OLD = "((device half4*)h)[i] = half4((a * sg) * b);"


def act_glu_source(src: str, act):
    """Activation kind of an elementwise gate * up kernel source. act None = the silu-gated form already in the source (returned unchanged); act = ("situ", scale_gate, scale_lin):
    half4(((scale_gate * tanh(a / scale_gate)) * sigmoid(a)) * (scale_lin * tanh(b / scale_lin))), no clamp (the tanh bounds both)."""
    if act is None:
        return src
    kind, sg, sl = act
    assert kind == "situ" and sg > 0.0 and sl > 0.0 and src.count(_GLU_OLD) == 1, act
    return src.replace(_GLU_OLD, f"((device half4*)h)[i] = half4((({float(sg)!r}f * metal::precise::tanh(a / {float(sg)!r}f)) * sg) * ({float(sl)!r}f * metal::precise::tanh(b / {float(sl)!r}f)));")


def act_kind(kind: str, mod):
    """ArchSpec.moe.act kind name + the attributes of the model's own switch module `mod` -> the kernel kind of ForeignSwitchGLU.act_kind (None for silu)."""
    if kind in (None, "", "silu", "swiglu"):
        return None
    if kind == "situ":
        return ("situ", float(mod.beta), float(mod.lbeta))
    raise ValueError(f"expert activation {kind!r} has no fused kernel kind (have: silu, situ)")


@functools.lru_cache(maxsize=None)
def _glu_kern(limit, act=None):
    clamp = "" if (limit is None or act) else f"a = min(a, {float(limit)!r}f); b = clamp(b, -{float(limit)!r}f, {float(limit)!r}f);"
    return KN._kern(("fg2a_glu", limit) + ((act,) if act else ()), name="kyojin_fg_glu" + ("_situ" if act else ""), input_names=["y2"], output_names=["h"], header="",
                    source=act_glu_source(_GLU_SRC.replace("LIMIT_CLAMP", clamp), act))


def glu_rows(y2, S, n, limit, act=None):
    """h half [S, n] = fp16(silu(min(a, limit)) * clip(b, -limit, limit)) in f32 (ForeignSwitchGLU._glu), a = rows [0, S) of y2 (gate), b = rows [S, 2 S) (up): one launch.
    act = ("situ", scale_gate, scale_lin): the SiTU form instead (limit unused)."""
    return _glu_kern(limit, act)(inputs=[y2], template=[("NQ", S * n // 4)], grid=(S * n // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(S, n)], output_dtypes=[mx.float16])[0]


_SLOT_ROWS: dict = {}


def _slot_rows(S, K):
    """Constant device arrays for S slots of K per token: (row of x of each slot, slot ids)."""
    v = _SLOT_ROWS.get((S, K))
    if v is None:
        v = _SLOT_ROWS[(S, K)] = (mx.array((np.arange(S) // K).astype(np.uint32)), mx.array(np.arange(S).astype(np.uint32)))
    return v


class ForeignSwitchGLU(nn.Module):
    """mlx-lm SwitchGLU drop-in: (x [.., H], inds [.., K]) -> [.., K, H]; silu(gate x) * (up x) -> down. Slots <= GATHER_MAX: one launch per projection
    (expert id read in the kernel). More slots (prefill): the slots are sorted by expert and each expert's rows go through DevPack.apply."""

    limit = None                                       # clamped SwiGLU (GLM-5.3 / DS4.1 swiglu_limit): gate <= limit, |up| <= limit; None = plain silu(gate) * up
    @property
    def act_kind(self):
        """Activation kind of the fused kernels: None = silu-gated (limit above), ("situ", scale_gate, scale_lin) = SiTU (act_kind() builds it from ArchSpec.moe.act).
        Stored outside the module's parameter tree (nn.Module would take a tuple for one); set with set_act_kind."""
        return self.__dict__.get("_act_kind")

    def set_act_kind(self, kind):
        self.__dict__["_act_kind"] = kind

    def __init__(self, gate: DevPack, up: DevPack, down: DevPack, limit: float | None = None, act_kind=None):
        super().__init__()
        self.g, self.u, self.d = gate, up, down
        self.limit = limit
        self.set_act_kind(act_kind)

    def _glu(self, a, b):
        """silu(a) * b in f32, rounded to fp16 (a = gate rows, b = up rows)."""
        a, b = a.astype(mx.float32), b.astype(mx.float32)
        if self.act_kind is not None:                  # SiTU on the stock ops: the reference of glu_rows / silu_perm / the tile and NAX epilogues
            _, sg, sl = self.act_kind
            return (((sg * mx.tanh(a / sg)) * mx.sigmoid(a)) * (sl * mx.tanh(b / sl))).astype(mx.float16)
        if self.limit is not None:
            a, b = mx.minimum(a, self.limit), mx.clip(b, -self.limit, self.limit)
        return (nn.silu(a) * b).astype(mx.float16)

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
        v.cm, v._mdrop, v._msrc = None, False, None
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
        h = self._glu(outs[0], outs[1])[:, None, :]
        w = self._stack16(self.d)
        if w is None:
            return None
        y = mx.gather_mm(h, w.swapaxes(-1, -2), rhs_indices=es, sorted_indices=True)[:, 0]
        return y[mx.argsort(order)]

    def tile_ok(self) -> bool:
        """All three projections are in the tile layout (flag glmtile)."""
        return self.g.tl is not None and self.u.tl is not None and self.d.tl is not None

    def tile_h(self, x16, xrows, eids):
        """h = glu(gate x, up x) per slot in ONE launch on tile-layout experts (gate | up | silu * up)."""
        return FT.gather_glu(x16, xrows, eids, self.g.tl, self.u.tl, self.g.gs, self.u.gs, self.g.n, self.g.k, self.limit, TILE_UB, TILE_GLU_NSG, act=self.act_kind)

    def _gu_plan(self):
        """Flag expert_gateup_pair: how the decode slots of this block run, decided ONCE (cached; nothing per call but a dict read). Needs gate and up stored alike (pair_ok) on a
        stored gather GEMV (not the nvfp4 g2a pair, which has its own path). "glu_down": plain silu, down takes silu(gate) * up while staging its input (2 launches per block);
        "rows": the gated activation in one launch (clamp / SiTU kinds, n % 4 == 0); "stock": the stock ops. None: today's path. Generic by stored type: no model name."""
        key = g2a_on()
        c = self.__dict__.get("_gu_cache")
        if c is not None and c[0] == key:
            return c[1]
        g, u, d = self.g, self.u, self.d
        plan = None
        if (all(p.fmt != "dense" and p.gemv_ok() for p in (g, u, d)) and g.pair_ok(u) and not g.g2a_ok(True) and not d.g2a_ok(True) and g.n == d.k):
            if self.limit is None and self.act_kind is None and d.glu_ok():
                plan = "glu_down"
            else:
                plan = "rows" if g.n % 4 == 0 else "stock"
        self.__dict__["_gu_cache"] = (key, plan)
        return plan

    def _gateup_pair(self, plan, x2, e32, S, K):
        """gate | up in one launch over the stored bytes (DevPack.gather_pair_raw, rows [0, S) gate, [S, 2 S) up), then the activation and the down gather."""
        tok, ar = _slot_rows(S, K)
        y2 = self.g.gather_pair_raw(self.u, x2, tok, e32)
        if plan == "glu_down":
            return self.d.gather_glu(y2[:S], y2[S:], ar, e32, None)
        h = glu_rows(y2, S, self.g.n, self.limit, self.act_kind) if plan == "rows" else self._glu(y2[:S], y2[S:])
        return self.d.gather(h, ar, e32)

    def __call__(self, x, inds):
        lead, H = x.shape[:-1], x.shape[-1]
        K = inds.shape[-1]
        T = int(np.prod(lead))
        x2 = x.reshape(T, H).astype(mx.float16)
        e32 = inds.reshape(-1).astype(mx.uint32)
        S = T * K
        if self.tile_ok():
            if S <= GATHER_MAX:                                          # decode / verify rows: two launches per layer on the tile layout
                tok = mx.array((np.arange(S) // K).astype(np.uint32))
                h = self.tile_h(x2, tok, e32)
                y = self.d.gather(h, mx.arange(S).astype(mx.uint32), e32)
                return y.reshape(*lead, K, self.d.n)
            tmp = ForeignSwitchGLU(self.g.untiled(), self.u.untiled(), self.d.untiled(), limit=self.limit, act_kind=self.act_kind)   # prefill rows: today's kernels on a transient stored-layout copy
            y = tmp(x, inds)
            del tmp
            return y
        if GMM and S > GATHER_MAX:
            y = None
            if os.environ.get("KYOJIN_PF_GEMM_I8", "0") not in ("0", "") or os.environ.get("KYOJIN_GLM_PF_Q8", "0") not in ("0", ""):   # flag pfq8 (default off): int8 activations (one scale per row and 128-wide K block) x int8 weight tiles decoded from the stored NVFP4
                from . import foreign_pfq8 as Q8
                if self.limit is not None and self.act_kind is None and KQ.on() and KQ.glm_on():
                    y = Q8.mlp(self, x2, e32, K)
                if y is not None:
                    FD.fire("pf_q8")
                elif FD.AUDIT:
                    FD.fb("pf_q8", Q8.why_not(self, S) if self.limit is None or (KQ.on() and KQ.glm_on()) else "kq_gmm_glm_off")
            if y is None:
                y = KQ.mlp(self, x2, e32, K, KQ_MT) if KQ.on() and (self.limit is None or KQ.glm_on()) else None      # flag kqgmm: grouped NAX GEMM on the stored rows (no fp16 expert stack at all)
            if y is None:
                if FD.AUDIT:
                    FD.fb("kq_gmm", KQ.why_not(self))
                y = self._sorted_gmm(x2, e32, T, K)
            else:
                FD.fire("kq_gmm")
            if y is not None:
                return y.reshape(*lead, K, self.d.n)
        if GATEUP_PAIR and S <= GATHER_MAX:
            plan = self._gu_plan()
            if plan is not None:
                up = MU.plan(self.g, self.u, self.d, T, K, NBRK) if MU.UNION else None          # flag moe_union: each distinct expert read once for all the rows
                if up is not None:
                    MU.PATHS["union"] += 1
                    R = MU.rows_per_group()
                    eu, osl = MU.tables(inds, T, K, R)
                    y2 = MU.gate_up(self.g, self.u, up[0], x2, eu, osl, R, S, NBRK, K)
                    if up[2]:
                        return MU.down(self.d, up[1], True, y2[:S], y2[S:], eu, osl, R, S, NBRK).reshape(*lead, K, self.d.n)
                    h = glu_rows(y2, S, self.g.n, self.limit, self.act_kind) if plan == "rows" else self._glu(y2[:S], y2[S:])
                    return MU.down(self.d, up[1], False, h, None, eu, osl, R, S, NBRK).reshape(*lead, K, self.d.n)
                return self._gateup_pair(plan, x2, e32, S, K).reshape(*lead, K, self.d.n)
        gather = S <= GATHER_MAX and all(p.fmt != "dense" and p.gemv_ok() for p in (self.g, self.u, self.d))
        if gather and self.g.g2a_ok(True) and self.d.g2a_ok(True) and self.g.pair_ok(self.u) and self.g.n % 4 == 0:      # flag g2a: gate | up in one launch, the gated activation in one launch, down
            tok, ar = _slot_rows(S, K)
            h = glu_rows(self.g.gather_pair_raw(self.u, x2, tok, e32), S, self.g.n, self.limit, self.act_kind)
            return self.d.gather(h, ar, e32).reshape(*lead, K, self.d.n)
        tok = mx.array((np.arange(S) // K).astype(np.uint32))
        if gather:
            a = self.g.gather(x2, tok, e32)
            b = self.u.gather(x2, tok, e32)
        else:
            en = np.array(e32).astype(np.int64)
            tn = np.arange(S) // K
            a = self._grouped(self.g, x2, tn, en)
            b = self._grouped(self.u, x2, tn, en)
        h = self._glu(a, b)
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
        self._tog = False                                          # flag shared_input_gemv: the group is switchable at run time (a cell flips the flag on one load); False = always grouped
        self._ab = (0, src.out_features)

    def _alone(self, x):
        """Flag shared_input_gemv OFF on a switchable group: the member's own rows as a ForeignLinear of their own (built once, from a row gather of the group's bytes)."""
        lin = self.__dict__.get("_lin")
        if lin is None:
            lin = ForeignLinear.__new__(ForeignLinear)
            nn.Module.__init__(lin)
            lin.dev = self._g.dev.rows(list(range(*self._ab)))
            lin._in_perm = lin._in_inv = None
            lin.in_features, lin.out_features, lin.fmt, lin.out_dtype, lin._bias = self.in_features, self.out_features, self.fmt, self.out_dtype, self._bias
            self.__dict__["_lin"] = lin
        return lin(x)

    def __call__(self, x: mx.array) -> mx.array:
        if self._tog and not SHARED_GEMV:
            return self._alone(x)
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
    orig_l, orig_s, orig_d = ForeignLinear.__call__, ForeignSwitchGLU.__call__, DenseLinear.__call__

    def rec_l(self, x):
        calls.append((id(x), id(self)))
        keep.append(x)                                                    # ids stay unique while the trace holds the arrays
        return mx.zeros((*x.shape[:-1], self.out_features), self.out_dtype)

    def rec_s(self, x, inds):
        return mx.zeros((*x.shape[:-1], inds.shape[-1], self.d.n), mx.float16)

    ForeignLinear.__call__, ForeignSwitchGLU.__call__, DenseLinear.__call__ = rec_l, rec_s, rec_l
    try:
        model(mx.zeros((1, 1), mx.int32))
    finally:
        ForeignLinear.__call__, ForeignSwitchGLU.__call__, DenseLinear.__call__ = orig_l, orig_s, orig_d
    return calls


def group_same_input(model, log=print, toggle=False) -> dict:
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
        gm = [GroupMember(grp, i, l) for i, l in enumerate(lins)]
        for m, a, b in zip(gm, cuts[:-1], cuts[1:]):
            m._tog, m._ab = toggle, (a, b)
        mods += list(zip(g, gm))
        for l in lins:
            l.dev = None
        saved += len(g) - 1
    if mods:
        model.update_modules(tree_unflatten(mods))
    ng = len({id(m._g) for _, m in mods})
    log(f"[foreign] cfmtspeed: {len(mods)} linears in {ng} same-input groups, {saved} fewer launches per 1-row tick (counted, not timed)")
    return {"groups": ng, "members": len(mods), "launches_saved": saved}


# ------------------------------------------------------------------------------------------------------------------------------ dense 16-bit linears kept as stored (flag shared_input_gemv)
SHARED_GEMV = os.environ.get("KYOJIN_SHARED_INPUT_GEMV", "0") not in ("0", "")   # flag shared_input_gemv (default off, read per call; flags.set updates it)
BF_ROWS = int(os.environ.get("KYOJIN_SHARED_GEMV_ROWS", "8"))                    # decode rows served by the 16-bit GEMV; above, the fp32 matmul of the decoded matrix

# One simdgroup per output row: 8 stored 16-bit weights per lane per step (one 16-byte load), the fp32 x row(s) from cache, fp32 FMA, simd_sum. The weights are widened exactly
# (bf16: the bits are the high half of a float32; f16: half -> float), so each product is the one the fp32-resident matrix gave; only the sum's order differs.
_BF_SRC = r"""
    const uint tid = thread_position_in_grid.x;
    const uint row = tid >> 5, lane = tid & 31;
    if (row >= (uint)N) return;
    float acc[R];
    for (uint r = 0; r < (uint)R; r++) acc[r] = 0.0f;
    device const uint4* wr = (device const uint4*)(w + (ulong)row * K);
    for (uint c = lane; c < (uint)K / 8; c += 32) {
        const uint4 v = wr[c];
        const uint ww[4] = {v.x, v.y, v.z, v.w};
        float wf[8];
        for (uint j = 0; j < 4; j++) {
            if (MODE == 0) { wf[2 * j] = as_type<float>(ww[j] << 16); wf[2 * j + 1] = as_type<float>(ww[j] & 0xffff0000u); }
            else { const half2 h = as_type<half2>(ww[j]); wf[2 * j] = float(h.x); wf[2 * j + 1] = float(h.y); }
        }
        for (uint r = 0; r < (uint)R; r++) {
            device const float4* xr = (device const float4*)(x + (ulong)r * K);
            const float4 a = xr[2 * c], b = xr[2 * c + 1];
            acc[r] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z + wf[7] * b.w;
        }
    }
    for (uint r = 0; r < (uint)R; r++) {
        const float s = simd_sum(acc[r]);
        if (lane == 0) y[(ulong)r * N + row] = s;
    }
"""


@functools.lru_cache(maxsize=None)
def _bfgemv_kern():
    return KN._kern(("kyojin_bfgemv",), name="kyojin_bfgemv", input_names=["x", "w"], output_names=["y"], header="", source=_BF_SRC)


def bf_gemv_ok(w, R) -> bool:
    """The 16-bit GEMV takes this matrix and row count: flag on, a GPU default device, bf16 / f16 weights, k % 8 == 0 (16-byte loads), 1..BF_ROWS rows."""
    return SHARED_GEMV and 1 <= R <= BF_ROWS and w.shape[1] % 8 == 0 and w.dtype in (mx.bfloat16, mx.float16) and mx.default_device() == mx.gpu


def bf_gemv(x2, w):
    """x2 [R, k] (any float, read as float32) times the stored 16-bit w [n, k] transposed -> [R, n] float32 in one launch."""
    R, k = x2.shape
    n = w.shape[0]
    if x2.dtype != mx.float32:
        x2 = x2.astype(mx.float32)
    return _bfgemv_kern()(inputs=[x2, w], template=[("K", k), ("N", n), ("R", R), ("MODE", 0 if w.dtype == mx.bfloat16 else 1)], grid=(n * 32, 1, 1), threadgroup=(128, 1, 1),
                          output_shapes=[(R, n)], output_dtypes=[mx.float32])[0]


class _DW:
    """The stored 16-bit matrix of a DenseLinear (a plain object: its array is not a module parameter) and its float32 twin, the matrix the model had before the flag.
    The twin is kept while the flag is off (the OFF arm is the old cost); with the flag on it is made per call (prompt rows), so the 16-bit bytes are all that stays resident."""

    def __init__(self, w, a=0, b=None):
        self.full, self.a, self.b, self._w32 = w, a, (w.shape[0] if b is None else b), None

    @property
    def w(self):
        return self.full if (self.a, self.b) == (0, self.full.shape[0]) else self.full[self.a:self.b]

    def w32(self):
        if self._w32 is not None:
            return self._w32
        w = self.w.astype(mx.float32)
        if not SHARED_GEMV:
            mx.eval(w)
            self._w32 = w
        return w


def _dense_apply(x2, dw: _DW):
    if bf_gemv_ok(dw.full, x2.shape[0]):
        return bf_gemv(x2, dw.w)
    return x2 @ dw.w32().T


class DenseLinear(nn.Module):
    """y = x @ W.T for a BF16 / F16 GGUF matrix kept as stored (half the bytes of the float32 matrix the skeleton decoded it to; the lm_head of a 120k vocabulary is 3072 x 120832).
    Decode rows read the 16-bit bytes in one GEMV (bf_gemv); other rows (and the flag off) run the float32 matmul of the same values."""

    def __init__(self, w: mx.array, out_dtype: mx.Dtype = mx.float32):
        super().__init__()
        self.dev = _DW(w)
        self.in_features, self.out_features, self.fmt, self.out_dtype = w.shape[1], w.shape[0], "bf16" if w.dtype == mx.bfloat16 else "f16", out_dtype

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        y = _dense_apply(x.reshape(-1, shape[-1]), self.dev)
        if y.dtype != self.out_dtype:
            y = y.astype(self.out_dtype)
        return y.reshape(*shape[:-1], self.out_features)


class _DGroup:
    def __init__(self, w, cuts):
        self.w, self.cuts, self.x, self.ys = w, cuts, None, None


class DenseGroupMember(nn.Module):
    """Stand-in for one DenseLinear of a same-input group: the first member called runs the whole concatenated matrix (one pass over x), the others take their slice
    (same protocol as GroupMember). Flag off, or rows the GEMV does not take: its own rows through the float32 matmul, as before."""

    def __init__(self, grp: _DGroup, i: int, src: DenseLinear):
        super().__init__()
        self._g, self._i = grp, i
        self.dev = _DW(grp.w, grp.cuts[i], grp.cuts[i + 1])
        self.in_features, self.out_features, self.fmt, self.out_dtype = src.in_features, src.out_features, src.fmt, src.out_dtype

    def __call__(self, x: mx.array) -> mx.array:
        g, shape = self._g, x.shape
        x2 = x.reshape(-1, shape[-1])
        if not bf_gemv_ok(g.w, x2.shape[0]):
            y = x2 @ self.dev.w32().T
        else:
            if g.x is None or g.x() is not x or g.ys[self._i] is None:
                y = bf_gemv(x2, g.w)
                g.ys = [y[:, a:b] for a, b in zip(g.cuts[:-1], g.cuts[1:])]
                g.x = weakref.ref(x)
            y, g.ys[self._i] = g.ys[self._i], None
        if y.dtype != self.out_dtype:
            y = y.astype(self.out_dtype)
        return y.reshape(*shape[:-1], self.out_features)


def group_dense(model, log=print) -> dict:
    """Flag shared_input_gemv: DenseLinears that read one input object (found by tracing one 1-row forward, as group_same_input) become DenseGroupMembers over one
    concatenated matrix. A model whose trace fails is left as it is."""
    from mlx.utils import tree_unflatten
    named = {n: m for n, m in model.named_modules() if type(m) is DenseLinear}
    if len(named) < 2:
        return {"groups": 0, "members": 0, "launches_saved": 0}
    try:
        groups = find_groups(trace_calls(model), {id(m): n for n, m in named.items()})
    except Exception as e:
        log(f"[foreign] shared_input_gemv: no dense grouping ({type(e).__name__}: {e})")
        return {"groups": 0, "members": 0, "launches_saved": 0}
    mods, saved = [], 0
    for g in groups:
        by = {}
        for n in g:
            l = named[n]
            by.setdefault((l.in_features, l.dev.full.dtype, l.out_dtype), []).append(n)
        for part in by.values():
            if len(part) < 2:
                continue
            lins = [named[n] for n in part]
            cuts = [0]
            for l in lins:
                cuts.append(cuts[-1] + l.out_features)
            w = mx.concatenate([l.dev.w for l in lins], axis=0)
            mx.eval(w)
            grp = _DGroup(w, cuts)
            mods += [(n, DenseGroupMember(grp, i, l)) for i, (n, l) in enumerate(zip(part, lins))]
            for l in lins:
                l.dev = None
            saved += len(part) - 1
    if mods:
        model.update_modules(tree_unflatten(mods))
    ng = len({id(m._g) for _, m in mods})
    log(f"[foreign] shared_input_gemv: {len(mods)} dense linears in {ng} same-input groups, {saved} fewer GEMV launches per 1-row tick (counted, not timed)")
    return {"groups": ng, "members": len(mods), "launches_saved": saved}


def install_shared_input(model, log=print) -> dict:
    """Flag shared_input_gemv, at load: the same-input groups of stored foreign linears (switchable at run time) and of the dense 16-bit ones."""
    a = group_same_input(model, log, toggle=True)
    b = group_dense(model, log)
    return {"foreign": a, "dense": b}


# ------------------------------------------------------------------------------------------------------------------------------ fused gate | up MLP (flag glmdense)
def _g3_mlp_why(gu, dn, R) -> str:
    """Flag audit: why the fused gate_up / down gated MLP did not take the G3 call (only called under fired.ON)."""
    if not DENSE_GEMV:
        return "flag_off"
    if type(gu) is not ForeignLinear or type(dn) is not ForeignLinear:
        return "not_ForeignLinear"
    if R > DENSE_GEMV_ROWS or R < 1:
        return f"rows>{DENSE_GEMV_ROWS}"
    if not gu.dev.g3_ok(R):
        return "gate_up=" + gu.dev.g3_why(R)
    if not dn.dev.g3_ok(R):
        return "down=" + dn.dev.g3_why(R)
    return "dims_dtype_bias"


def _g3_mlp_call(self, x):
    """Glm5NextMLP (fused gate_up_proj): down(limited_swiglu(gate, up)) with the product taken while the down projection stages its input: gate_up | down, no swiglu launch."""
    gu, dn = self.gate_up_proj, self.down_proj
    lead = x.shape[:-1]
    R = 1
    for d in lead:
        R *= d
    if not (DENSE_GEMV and type(gu) is ForeignLinear and type(dn) is ForeignLinear and 1 <= R <= DENSE_GEMV_ROWS and gu.dev.g3_ok(R) and dn.dev.g3_ok(R)
            and gu.out_features == 2 * dn.in_features and gu.out_dtype == mx.float16 and dn.out_dtype == mx.float16 and gu._bias is None and dn._bias is None):
        if FD.AUDIT:
            FD.fb("dense_gemv_mlp", _g3_mlp_why(gu, dn, R))
        return self._stock_call(x)
    FD.fire("dense_gemv_mlp")
    y = gu(x).reshape(R, gu.out_features)
    return dn.dev._gemv8g3(y, limit=float(self.swiglu_limit)).reshape(*lead, dn.out_features)


def install_g3_mlp(model, log=print) -> int:
    """Flag glmdense: swap the call of every fused gate_up / down gated MLP (GLM-5.3 dense layers and shared experts). Returns the number."""
    if not DENSE_GEMV:
        return 0
    n = 0
    for _, mod in model.named_modules():
        gu, dn = getattr(mod, "gate_up_proj", None), getattr(mod, "down_proj", None)
        if type(gu) is not ForeignLinear or type(dn) is not ForeignLinear or getattr(mod, "swiglu_limit", None) is None or getattr(mod.__class__, "_g3_mlp", False):
            continue
        cls = mod.__class__
        mod.__class__ = type("G3" + cls.__name__, (cls,), {"__call__": _g3_mlp_call, "_stock_call": cls.__call__, "_g3_mlp": True})
        n += 1
    if n:
        log(f"[foreign] glmdense: {n} gated MLPs run silu * up inside the down projection's launch")
    return n


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


class _Probe(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def __call__(self, x):
        return self._fn(x)


def silu_gated(mod) -> bool:
    """True when the module's own call is down(silu(gate(x)) * up(x)): probed by running it on stand-in projections (gate = identity, up = ones, down = identity) and comparing with silu.
    A gated MLP with another activation (gelu in the gemma classes) must keep its call: the fused launch computes silu. Any failure of the probe = False."""
    import copy
    try:
        m = copy.copy(mod)
        m.gate_proj, m.up_proj, m.down_proj = _Probe(lambda x: x), _Probe(lambda x: mx.ones_like(x)), _Probe(lambda x: x)
        x = mx.linspace(-4.0, 4.0, 33).reshape(1, 1, 33)
        return bool(mx.allclose(type(mod).__call__(m, x), nn.silu(x), atol=1e-5).item())
    except Exception:
        return False


def install_dense_glu(model, log=print) -> int:
    """Swap the call of every gated MLP (gate_proj / up_proj / down_proj) whose down projection is a ForeignLinear of a GEMV-able format. Returns the number."""
    if not DENSE_GLU:
        return 0
    n = 0
    for _, mod in model.named_modules():
        g, u, dn = (getattr(mod, k, None) for k in PROJ)
        if type(dn) is not ForeignLinear or getattr(mod.__class__, "_dense_glu", False) or not silu_gated(mod):
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
    has_st = bool(glob.glob(os.path.join(p, "*.safetensors")) or glob.glob(os.path.join(p, "tensors", "*.safetensors")))   # shards may sit in tensors/ (Flash-Next FP4-Local; the index names them)
    if GF.find_gguf(p) and not has_st:
        return "gguf"
    if has_st:
        from . import vendor_input as VI
        from .csf_scale import CsfCheckpoint
        ck = CsfCheckpoint(p)                                     # modelopt NVFP4 with csf scale streams (GLM-5.3-Flash-FP4-Local) reads as plain NVFP4
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
        self.mapped_st: dict = {}        # qwen4_exp safetensors pack, flag KYOJIN_MAPPED_ROW_NVFP4: n-gram table prefix -> MappedNvfp4Rows over the mapped shards (foreign_qwen4exp_st)
        self.mapped: dict = {}           # GGUF name -> (uint8 view of the mapped tensor, type name, row width, module suffix): flag mapped_row_table, see foreign_mapped_rows
        self.experts: dict = {}          # mlp prefix -> {proj: [Pack per expert]} (or a thunk building that list)
        self.mtp_experts: dict = {}      # same for the MTP head's layer (modelopt NVFP4 packs: layer index >= num_hidden_layers)
        self.in_perm: dict = {}          # linear prefix -> activation gather (foreign_xform.Xf.cols)
        self.mtp_packs: dict = {}        # flag glm_dense4: the draft head's NVFP4 linears kept as stored Packs (HF prefix -> Pack), see MTP_SERVED
        self.mtp: dict = {}              # draft head: `mtp.*` name -> fp16 / fp32 mx array in the checkpoint's own convention (norms without the +1)
        self.notes: list = []
        self.rope_freqs = None           # llama.cpp `rope_freqs.weight` (Llama 3.x): per-pair frequency divisors, see FreqRoPE


def _layer_of(name):
    m = re.search(r"\blayers\.(\d+)\.", name)
    return int(m[1]) if m else None


_LIN_SFX = frozenset(("weight", "weight_packed", "weight_scale", "weight_scale_2", "weight_global_scale", "input_scale", "input_global_scale", "scale"))


_POOL: list = []


def load_threads() -> int:
    """Worker threads of the loader (KYOJIN_LOAD_THREADS, default min(16, cores); 1 = serial). The expert tensors are mmap views: one thread faults them in at ~1 GiB/s,
    a few threads reach the SSD's 3-6 GiB/s, and the CSF scale decode / the stacking copies release the GIL."""
    v = os.environ.get("KYOJIN_LOAD_THREADS")
    return max(1, int(v)) if v else min(16, os.cpu_count() or 4)


def pmap(fn, items):
    """[fn(x) for x in items], in order, on the loader's thread pool (serial when load_threads() == 1 or there is one item)."""
    items = list(items)
    n = load_threads()
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    if not _POOL or _POOL[0]._max_workers != n:
        from concurrent.futures import ThreadPoolExecutor
        _POOL[:] = [ThreadPoolExecutor(n, thread_name_prefix="kyojin-load")]
    return list(_POOL[0].map(fn, items))


def _stack_par(arrs):
    """np.stack(arrs) with the copies spread over the loader's threads (the sources are mmap views: this is also where the pages are read)."""
    out = np.empty((len(arrs),) + arrs[0].shape, arrs[0].dtype)
    pmap(lambda i: LC.copy_into(out[i], arrs[i]), range(len(arrs)))
    return out


MTP_SERVED = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj", "eh_proj", "gate_proj", "up_proj", "down_proj")      # the draft head's linears glm_nextn can swap for a ForeignLinear
MTP_GROUPS = {"self_attn.qkv_a_proj": ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"), "mlp.shared_experts.gate_up_proj": ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj")}


def _g4_mtp_finish(src) -> None:
    """Flag glm_dense4: a stored 4-bit draft-head linear is served in place only when the module that takes it is complete (the drafter fuses q_a + kv_a and gate + up
    into one matrix); the others are decoded to fp16 as without the flag."""
    tail = lambda q: re.sub(r"^.*?layers\.\d+\.", "", q)
    have = {tail(q) for q in src.mtp_packs}
    keep = {}
    for p, pk in src.mtp_packs.items():
        grp = next((ms for ms in MTP_GROUPS.values() if tail(p) in ms), None)
        mem = [src.mtp_packs[q] for q in src.mtp_packs if grp is not None and tail(q) in grp]
        same = len({q.fmt for q in mem}) == 1
        split = not same and all(q.fmt in _GGUF_FMTS and q.k == mem[0].k and np.ndim(q.qs) == 2 for q in mem)      # GGUF q_a Q6_K + kv_a Q8_0: stored members, SplitLinear (as the trunk)
        if grp is None or (all(m in have for m in grp) and (same or split)):
            keep[p] = pk
        else:
            src.mtp[p + ".weight"] = mx.array(FP.dequant(pk).astype(np.float16))
    src.mtp_packs = keep


def _read_safetensors(path, head=False) -> _Src:
    """head=True: the MTP layer's (index >= num_hidden_layers) non-expert tensors go to src.mtp (HF names; quantised linears decoded to fp16), not into the trunk."""
    mtp_only = head == "mtp"                                      # read_mtp_head: only `mtp.*` tensors (qwen4_exp style head, names `mtp.layers.0...`)
    from . import vendor_input as VI
    from .csf_scale import CsfCheckpoint, csf_stream_names
    ck = CsfCheckpoint(path)
    LC.register(ck)                                               # G1b: expert copies read the shard with pread (F_NOCACHE) and drop the pages they consumed
    src = _Src()
    info = ck.info()
    block = VI._config_block(ck.config)
    used = set(csf_stream_names(info))
    ple_skip: set = set()
    from . import foreign_qwen4exp_st as Q4
    if Q4.mapped_row_nvfp4():                                     # flag mapped_row_nvfp4 (default off): a sharded NVFP4 row-lookup table (rows read by id) stays in the mapped files, no Pack of it is built
        _tabs = Q4.ple_tables(info)                               # structural: any pack whose header carries such a table, whatever the model type
        if _tabs:
            used.update(Q4.table_keys(_tabs))
            ple_skip = {w[:-len(".weight")] for t in _tabs.values() for w, _s in t["shards"]}
            src.mapped_st = Q4.build_tables(ck, _tabs)
    n_trunk = (ck.config.get("text_config") or ck.config).get("num_hidden_layers")
    own: dict = {}                                                # linear prefix -> its weight / scale tensors: one pass over the header (a scan of `info` per linear is quadratic: 36k expert linears x 331k names ~ 8 min)
    for k in info:
        sfx = k.rsplit(".", 1)[-1]
        if sfx in _LIN_SFX:
            a = -1
            while (a := k.find(".", a + 1)) >= 0:                 # every dot-prefix of k, as `k.startswith(p + ".")` selects it
                own.setdefault(k[:a], []).append(k)
    _owned = lambda p: own.get(p, ())
    lazy_exp: dict = {}                                           # mlp prefix -> {proj: {expert: classify_linear dict}} (NVFP4 experts: the Packs are built one projection at a time at load)
    for p in VI.linear_prefixes(info):
        if p in ple_skip:
            continue
        if mtp_only and not p.startswith("mtp."):                     # read_mtp_head: the draft head's tensors only, never the trunk of the pack
            used.update(_owned(p))
            continue
        if mtp_only and not _EXP.match(p + ".weight"):                # the head's dense linears: fp16 (the main pack's own convention for a draft head), its routed experts stay NVFP4 packs below
            c0 = VI.classify_linear(info, p)
            if c0 is not None and not c0["fmt"].startswith("dense_"):
                src.mtp[p + ".weight"] = mx.array(np.asarray(VI.load_linear(ck, p)).astype(np.float16))
                used.update(_owned(p))
                continue
        if p.startswith("model.visual.") or p.startswith("visual."):     # text trunk only; the vision tower is dropped below (its k need not be a multiple of 32: no Pack to build)
            used.update(_owned(p))
            continue
        c = VI.classify_linear(info, p)
        if c is None or c["fmt"].startswith("dense_"):
            continue
        if head and n_trunk is not None and not _EXP.match(p + ".weight") and _layer_of(p) is not None and _layer_of(p) >= n_trunk:
            if g4_dense4() and c["fmt"] == "nvfp4" and p.rsplit(".", 1)[-1] in MTP_SERVED:
                src.mtp_packs[p] = FP.pack_from_safetensors(ck, c)                                    # converted checkpoint: stays 4-bit, served in place (_g4_mtp_finish)
            else:
                src.mtp[p + ".weight"] = mx.array(np.asarray(VI.load_linear(ck, p)).astype(np.float16))   # the draft head's dense linears (~0.25 B params): fp16, nothing quantised to serve
            used.update(_owned(p))
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
            me = _EXP.match(p + ".weight") if c["fmt"] == "nvfp4" else None
            if me:                                                # routed expert: stacked per projection at load (glm_nvfp4), never held as 12k host Packs
                lazy_exp.setdefault(me[1], {}).setdefault(me[3], {})[int(me[2])] = c
                used.update(_owned(p))
                continue
            pk = FP.pack_from_safetensors(ck, c)
        else:
            raise ValueError(f"{p}: {c['fmt']} ({c['detail']}) is not served as stored (only NVFP4 / MXFP4 / FP8); convert it first")
        used.update(_owned(p))
        m = _EXP.match(p + ".weight")
        if m:
            src.experts.setdefault(m[1], {}).setdefault(m[3], {})[int(m[2])] = pk
        else:
            src.packs[p] = pk
    for f in sorted(ck.files):
        for k, v in mx.load(os.path.join(path, f)).items():
            if k not in used:
                if mtp_only:
                    if k.startswith("mtp."):
                        src.mtp[k] = v if v.dtype == mx.float32 else v.astype(mx.float16)
                    continue
                if head and n_trunk is not None and (_layer_of(k) or -1) >= n_trunk:
                    src.mtp[k] = v.astype(mx.float32) if v.dtype == mx.float32 else v.astype(mx.float16)
                else:
                    src.dense[k] = v
    src._ck = ck
    for mp, projs in lazy_exp.items():
        li = int(re.search(r"layers\.(\d+)\.mlp$", mp)[1])
        tgt = src.mtp_experts if (mtp_only or (n_trunk is not None and li >= n_trunk)) else src.experts      # layers past the trunk: the MTP head's experts
        for pj, cs in projs.items():
            assert sorted(cs) == list(range(len(cs))), f"{mp}.{pj}: expert ids are not contiguous"
            tgt.setdefault(mp, {})[pj] = (lambda cs=cs: pmap(lambda e: FP.pack_from_safetensors(ck, cs[e]), range(len(cs))))
    for mp in src.experts:
        for pj in src.experts[mp]:
            d = src.experts[mp][pj]
            if callable(d):
                continue
            assert sorted(d) == list(range(len(d))), f"{mp}.{pj}: expert ids are not contiguous"
            src.experts[mp][pj] = [d[e] for e in range(len(d))]
    if src.mtp_packs:
        _g4_mtp_finish(src)
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
    if os.environ.get("KYOJIN_STORED_EMBED", "1") == "0" or not is_lookup_table(hf) or len(ti.ne) != 2:
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


def _archkit_consumed(arch) -> tuple:
    from . import foreign_archkit as _K
    return _K.vetted(arch, "consumed")


def _concat_parts(packs):
    """Row-concatenation of a fused tensor's parts: plain (decoded fp16) matrices join as they are, stored ones go through _concat_rows_exact."""
    if all(p.fmt == "dense" and p.k == packs[0].k for p in packs):
        return FP.Pack("dense", sum(p.n for p in packs), packs[0].k, np.concatenate([np.ascontiguousarray(p.qs) for p in packs], 0))
    return _concat_rows_exact(packs)


def _apply_concat(src, A):
    """Tensors the model class holds fused (Arch.concat, HF suffix -> suffixes of the file's parts): the parts' packs become one thunk that row-concatenates them in order."""
    for fused, parts in A.concat.items():
        for p0 in [p for p in src.packs if p.endswith(parts[0])]:
            pre = p0[:-len(parts[0])]
            held = [src.packs.pop(pre + q) for q in parts]                       # taken out now: the load loop walks the dict and would build a part on its own first
            src.packs[pre + fused] = lambda held=held: _concat_parts([_build(h) for h in held])


def _stored_type(ti, hf) -> bool:
    """The Metal kernel types of the trunk rule (2-D / 3-D), for the draft head's stored path."""
    k = ti.ne[0]
    return ti.tname in GF.KERNEL_TYPES or (ti.tname in GF.KERNEL_TYPES8 and fp8stored() and len(ti.ne) in (2, 3)) or (ti.tname in GF.KERNEL_TYPESK and kstored() and k % 256 == 0 and len(ti.ne) in (2, 3))


def _read_gguf(path, arch, mtp=True) -> _Src:
    """mtp=False: the draft-head blocks (nextn) are not read at all (they would sit decoded in fp16, ~0.8 GiB on a 27B, with nothing to use them)."""
    g = GF.GgufFile(GF.find_gguf(path))
    src = _Src()
    A = ARCHS[arch]
    nt = n_trunk_of(g.kv, arch)
    for name, ti in g.tensors.items():
        if name == "rope_freqs.weight" and name in _archkit_consumed(arch):       # read by the architecture's config row (gemma4, behind KYOJIN_GEMMA4_ROWS): the class carries its own rope
            continue
        if name == "rope_freqs.weight" and len(ti.ne) == 1 and ti.tname not in GF.ROW_ONLY:            # not a model weight: the rope frequency factors the converter baked from rope_scaling
            src.rope_freqs = mx.array(GF.dequant(ti.tname, g.raw(name), ti.ne[0]).reshape(-1))
            continue
        if name in A.unserved and name in A.mapped_rows and mapped_row_table() and len(ti.ne) == 2:      # flag KYOJIN_MAPPED_ROW_TABLE: the table stays in the mapped file
            if ti.tname not in GF.DEQUANT:
                raise GF.GgufTypeError(f"tensor {name}: GGUF type {ti.tname} has no row decode (foreign_gguf.DEQUANT)")
            src.mapped[name] = (g.raw(name), ti.tname, ti.ne[0], A.mapped_rows[name])
            continue
        if name in A.unserved:
            raise GF.GgufError(f"tensor {name}: no loader path serves it yet for {arch} (Arch.unserved); the file cannot be served")
        hf = gguf_name(name, arch, nt)
        if hf is None:
            src.notes.append(f"{name}: draft-head block, not used by the {arch} class (skipped, as llama.cpp does)")
            continue
        head = hf.startswith("mtp.")
        if head and not mtp:
            continue
        xf = XF.plan(name, A.xform, g.kv, ti.ne[-1] if len(ti.ne) == 2 else (ti.ne[0] if len(ti.ne) == 1 else None))
        raw = g.raw(name)
        rows = xf.rows if (xf is not None and xf.rows is not None and raw.ndim == 2) else None      # rows are independent in every GGUF type: permute the stored blocks, nothing is decoded
        is_w = is_linear(hf, len(ti.ne), arch)
        if embed_stored(hf, ti, arch, g) and not head:                       # token embedding kept as stored (StoredEmbedding): no fp16 table
            src.embed[hf[:-len(".weight")]] = lambda raw=raw, rows=rows, tn=ti.tname, k=ti.ne[0]: GF.to_pack(tn, raw if rows is None else raw[rows], k)
            continue
        if (ti.tname in GF.KERNEL_TYPES or (ti.tname in GF.KERNEL_TYPES8 and fp8stored() and len(ti.ne) in (2, 3))
                or (ti.tname in GF.KERNEL_TYPESK and kstored() and len(ti.ne) in (2, 3) and ti.ne[0] % 256 == 0)
                or (ti.tname in GF.KERNEL_TYPES5 and legacy5_stored() and len(ti.ne) in (2, 3))
                or (ti.tname in GF.KERNEL_TYPES2 and GF.q2_0_stored() and len(ti.ne) in (2, 3))) and is_w and not head:
            k = ti.ne[0]
            if len(ti.ne) == 3:                                              # stacked experts [E, n, k]: built when the loader reaches the block (see _build), so the host copy is never whole
                m = re.match(r"(.*\.layers\.\d+\.\w+)\.experts\.(\w+)\.weight", hf)
                src.experts.setdefault(m[1], {})[m[2]] = lambda raw=raw, tn=ti.tname, k=k, E=ti.ne[2]: [GF.to_pack(tn, raw[e], k) for e in range(E)]
            else:
                src.packs[hf[:-len(".weight")]] = lambda raw=raw, rows=rows, tn=ti.tname, k=k: GF.to_pack(tn, raw if rows is None else raw[rows], k)
                if xf is not None and xf.cols is not None:
                    src.in_perm[hf[:-len(".weight")]] = xf.cols
            continue
        if head and is_w and hf.rsplit(".", 2)[-2] in MTP_SERVED + ("gate_proj", "up_proj", "down_proj") and _stored_type(ti, hf):      # draft head as stored: experts / MTP_SERVED linears are not decoded
            k = ti.ne[0]
            if len(ti.ne) == 3:
                m = re.match(r"(.*\.layers\.\d+\.\w+)\.experts\.(\w+)\.weight", hf)
                src.mtp_experts.setdefault(m[1], {})[m[2]] = lambda raw=raw, tn=ti.tname, k=k, E=ti.ne[2]: [GF.to_pack(tn, raw[e], k) for e in range(E)]
            else:
                src.mtp_packs[hf[:-len(".weight")]] = GF.to_pack(ti.tname, raw, k)
            continue
        if ti.tname not in GF.DEQUANT:
            raise GF.GgufTypeError(f"tensor {name}: GGUF type {ti.tname} is not readable (see foreign_gguf.DEQUANT)")
        if ti.tname in GF.ROW_ONLY:
            raise GF.GgufTypeError(f"tensor {name}: GGUF type {ti.tname} is read by row only (foreign_gguf.ROW_ONLY, mapped row tables); no dense load path takes it")
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
            m = re.match(r"(.*\.layers\.\d+\.\w+)\.experts\.(\w+)\.weight", hf)
            src.experts.setdefault(m[1], {})[m[2]] = [FP.Pack("dense", w.shape[1], w.shape[2], w[e].astype(np.float16)) for e in range(w.shape[0])]
            src.notes.append(f"{name}: {ti.tname} experts decoded to fp16 (no Metal kernel for this type)")
        elif is_w:
            if xf is not None and xf.cols is not None:
                w = w[:, np.argsort(xf.cols, kind="stable")]                 # dense: undo the column order on the weight itself (exact)
            src.packs[hf[:-len(".weight")]] = FP.Pack("dense", w.shape[0], w.shape[1], w.astype(np.float16))
            src.notes.append(f"{name}: {ti.tname} decoded to fp16 (no Metal kernel for this type)")
        else:
            src.dense[hf] = mx.array(w.astype(np.float16) if w.ndim > 1 and ti.tname != "F32" else w)
    if src.mtp_packs:
        _g4_mtp_finish(src)                                                  # a linear whose fused partner is missing is decoded to fp16 as before
    _apply_concat(src, A)
    src._g = g
    return src


def read_mtp_head(path, log=print):
    """The `mtp.*` draft head of a modelopt-style safetensors pack (qwen4_exp FP4-Local): ({`mtp.*` name: fp16 / fp32 array}, ForeignSwitchGLU of its routed experts or None).
    Only the head is read: the pack's trunk is neither decoded nor held. Dense linears are fp16, routed experts stay NVFP4 DevPacks (as the main layers do)."""
    src = _read_safetensors(path, head="mtp")
    sw = None
    for mp in list(src.mtp_experts):
        projs = src.mtp_experts.pop(mp)
        sw = ForeignSwitchGLU(*[DevPack(_build(projs[pj])) for pj in PROJ])
    log(f"[kyojin] mtp head read from {path}: {len(src.mtp)} dense tensors, experts {'NVFP4 packs' if sw is not None else 'none'}")
    return src.mtp, sw


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


from . import foreign_vlm_classes as _VC
VLM_FAMILIES = _VC.FAMILIES                                       # the one table lives in foreign_vlm_classes


def vlm_family(model_type):
    """The mlx-vlm package for `model_type`, or None (the mlx-lm class is used)."""
    return _VC.family(model_type)


def load_foreign(path, dtype=mx.float16, log=print, mtp=True):
    """(model, config) for a directory (or .gguf file) in a foreign format. The model class is mlx-lm's for config['model_type'], the quantised linears are
    ForeignLinear, MoE experts ForeignSwitchGLU; non-quantised tensors are kept as stored (bf16 -> `dtype`)."""
    from mlx.utils import tree_unflatten
    path = str(path)
    kind = is_foreign(path)
    if kind is None:
        raise ValueError(f"{path}: neither a GGUF file nor an NVFP4 / MXFP4 safetensors directory")
    d = path if os.path.isdir(path) else os.path.dirname(path) or "."
    if kind == "gguf":
        gf = GF.GgufFile(GF.find_gguf(path))
        arch = gf.kv.get("general.architecture")
        if arch not in ARCHS:                                              # no hand-written row: build it from the generated llama.cpp tables (foreign_archkit_serve)
            from .foreign_archkit_serve import register
            register(path)
        cfg_path = os.path.join(d, "config.json")
        config = json.load(open(cfg_path)) if os.path.exists(cfg_path) else gguf_config(gf)
        tnames = list(gf.tensors)
        gf.close()
        src = _read_gguf(path, arch, mtp)
    else:
        config = json.load(open(os.path.join(d, "config.json")))
        src = _read_safetensors(d, head=bool(mtp and config.get("model_type") == "glm5_next"))
    config.pop("quantization_config", None)
    tc = config.get("text_config", config)
    fam = vlm_family(config.get("model_type"))
    if fam is not None:                                            # GLM-5.3 and the families of VLM_FAMILIES: the model code is mlx-vlm's (mlx-lm has none); text trunk only, the vision tower is never read
        model_class, args_class, _mt, _src = _VC.resolve([config.get("model_type")], config=config)       # the family step of the one lookup (foreign_vlm_classes.resolve)
        assert _src == "family", _src
        model = model_class(args_class.from_dict(config))
        class _NoVision(nn.Module):
            sanitize = staticmethod(lambda w: {k: v for k, v in w.items() if not k.startswith("vision_tower.")})
        model.vision_tower = _NoVision()
        for part in (src.dense, src.packs, src.embed):                # text trunk only: the vision tower's tensors never reach sanitize (it renames them)
            for k in [k for k in part if k.startswith("model.visual.")]:
                part.pop(k)
    else:
        model_class, args_class, _mt, _src = _VC.resolve([config.get("model_type")], config=config, families=False)       # mlx-lm's class as before; mlx-vlm's text class only with KYOJIN_KIT_VLM_CLASSES (same lookup as day0_inspect)
        model = model_class(args_class.from_dict(config))
        if _src == "mlx-vlm" and kind == "gguf":
            _VC.drop_vision(model, tnames)
    # sanitize may rename (language_model.* ...): run the dense tensors plus one marker per quantised linear through it and follow the markers.
    # Each marker holds its own id, so a projection sanitize concatenates (GLM-5.3: gate+up -> gate_up_proj, q/k/v -> qkv_proj) is found as a 1-D
    # array of member ids and served as one row-concatenated Pack (_concat_rows_exact).
    mkeys = [p + ".weight" for p in {**src.packs, **src.embed}]
    marks = {k: mx.array([float(i + 1)]) for i, k in enumerate(mkeys)}
    weights = (model.sanitize if hasattr(model, "sanitize") else dict)({**src.dense, **marks})        # glm4 and some others have no sanitize: names are the HF ones already
    by_id = {id(v): k for k, v in weights.items()}
    fused = {}
    mark_ids = {id(m) for m in marks.values()}
    for wk, v in weights.items():
        if (id(v) not in mark_ids and wk.endswith(".weight") and isinstance(v, mx.array) and v.ndim == 1
                and 1 < v.size <= 8 and v.dtype == mx.float32):
            ids = [float(x) for x in v.tolist()]
            if all(x == int(x) and 1 <= x <= len(mkeys) for x in ids):
                fused[wk] = [mkeys[int(x) - 1] for x in ids]
    in_fused = {m for ms in fused.values() for m in ms}
    # a top-level group sanitize drops whole (mtp.* in qwen3_5: the MTP head is served from its own sidecar) is dropped for the quantised linears too
    kept = {id(v) for v in weights.values()}
    def _grp(k):                                                  # top-level name, or "...layers.N" (sanitize drops whole layers too: GLM-5.3's MTP layer 45 without mtp)
        m = re.match(r"(.*?layers\.\d+)\.", k)
        return m.group(1) if m else k.split(".", 1)[0]
    groups = {}
    for k, v in {**src.dense, **marks}.items():
        groups.setdefault(_grp(k), []).append(id(v) in kept)
    dropped = {g for g, ks in groups.items() if not any(ks)}
    where, lost = {}, []
    for k, v in marks.items():
        if id(v) not in by_id:
            if k in in_fused:
                continue
            if _grp(k) in dropped:
                src.packs.pop(k[:-len(".weight")], None); src.embed.pop(k[:-len(".weight")], None)
                continue
            lost.append(k)
            continue
        where[k[:-len(".weight")]] = by_id[id(v)][:-len(".weight")]
    if lost:                                                      # all of them at once (a full read costs minutes on a 178 GB pack)
        raise ValueError(f"model.sanitize() rewrote {len(lost)} quantised linears (renaming not followable), e.g. {lost[:6]}")
    if os.environ.get("KYOJIN_FOREIGN_MAPONLY"):                  # dry check of the name mapping (tests / first load of a new family): no Pack is built
        return {"where": where, "fused": fused, "dropped": sorted(dropped), "plan": dict(packs=list(src.packs), embed=list(src.embed), experts=list(src.experts), mapped=list(src.mapped), mapped_st=list(src.mapped_st),
                                                                                         weights=list(weights), model=model, src=src)}, config
    for wk, members in fused.items():                             # one Pack over the members' rows, in sanitize's order
        nk = wk[:-len(".weight")]
        src.packs[nk] = _concat_rows_exact([_build(src.packs.pop(m[:-len(".weight")])) for m in members])
        where[nk] = nk
    mods = []
    fm = {}
    n_packs = len(src.packs)
    for p in list(src.packs):                                     # one host pack at a time: built, copied to the device, dropped (host and device arrays are never both whole)
        pk = _build(src.packs.pop(p))
        fm[pk.fmt] = fm.get(pk.fmt, 0) + 1
        bias = weights.pop(where[p] + ".bias", None)
        mods.append((where[p], make_linear(pk, bias, in_perm=src.in_perm.get(p))))
        weights.pop(where[p] + ".weight", None)
        del pk
    for p in list(src.embed):                                     # token embedding as stored: rows are decoded when looked up
        pk = _build(src.embed.pop(p))
        mods.append((where[p], StoredEmbedding(pk)))
        weights.pop(where[p] + ".weight", None)
        del pk
    if src.mapped:                                                # row-gathered tables (flag mapped_row_table): the class's lookup module is replaced, no parameter of it is loaded
        from .foreign_mapped_rows import MappedRows
        for gname, (raw, tn, width, suffix) in src.mapped.items():
            hits = [p for p, _m in model.named_modules() if p.endswith(suffix)]
            if len(hits) != 1:
                raise ValueError(f"{gname}: {len(hits)} modules of the model class end with {suffix!r} (one row table, one module expected)")
            mods.append((hits[0], MappedRows(raw, tn, width)))
    for pre, tab in src.mapped_st.items():                        # qwen4_exp n-gram tables (flag mapped_row_nvfp4): the class's lookup module of that layer is replaced, no parameter of it is loaded
        li = re.search(r"layers[.](\d+)[.]", pre)[1]
        suffix = f"layers.{li}.ple.ple_embedding.ngram_embedding"
        hits = [p for p, _m in model.named_modules() if p.endswith(suffix)]
        if len(hits) != 1:
            raise ValueError(f"{pre}: {len(hits)} modules of the model class end with {suffix!r} (one row table, one module expected)")
        mods.append((hits[0], tab))
    model.update_modules(tree_unflatten(mods))
    if src.mapped and os.environ.get("KYOJIN_MAPPED_ROW_BATCH", "0") not in ("0", ""):      # flag mapped_row_batch: one host row read for all layers of a step
        from .foreign_mapped_rows import install_batch
        install_batch(model)
    n_exp = 0
    if src.experts or src.mtp_experts:
        root = model.language_model.model if hasattr(model, "language_model") and hasattr(model.language_model, "model") else (model.model if hasattr(model, "model") else model)
        for mp in list(src.experts):
            projs = src.experts.pop(mp)
            mm = re.search(r"layers\.(\d+)\.(\w+)$", mp)
            li = int(mm[1])
            blk = getattr(root.layers[li], mm[2])                  # `mlp`, or the class's own MoE container name
            assert hasattr(blk, "switch_mlp"), f"layer {li}: the model class has no switch_mlp to replace"
            lim = getattr(getattr(blk.switch_mlp, "activation", None), "limit", None)      # LimitedSwiGLU (GLM-5.3 swiglu_limit); None for a plain SwiGLU
            blk.switch_mlp = ForeignSwitchGLU(*[DevPack(_build(projs[pj])) for pj in PROJ], limit=lim)
            n_exp += 1
        for mp in list(src.mtp_experts):                           # the MTP head's experts (W4A16 NVFP4): kept for the draft head, not a module of the trunk
            projs = src.mtp_experts.pop(mp)
            lim = tc.get("swiglu_limit")
            model.__dict__.setdefault("_nvfp4_mtp_switch", {})[mp] = ForeignSwitchGLU(*[DevPack(_build(projs[pj])) for pj in PROJ], limit=lim)
    # F32 1-D tensors go to the model dtype only for a GGUF (ggml writes norms / biases F32 whatever the file type); a safetensors F32 tensor is F32 on
    # purpose: GLM-5.3's HyperConnection base / scale are read as float by mlx-vlm's fused hc kernel (fp16 there = a wrong Sinkhorn comb, garbage text)
    adt = act_dtype(dtype) if kind == "gguf" else None                  # flag act_dtype: A_log / dt_bias stay F32 then (the 1-D rule below would cast them)
    weights = {k: (v.astype(dtype) if v.dtype == mx.bfloat16 or (kind == "gguf" and gguf_stream_f16() and _f32_on_stream(k, v) and not (adt is not None and _KEEP_F32.search(k))) else v) for k, v in weights.items()}
    if adt is not None:
        log(f"[foreign] act_dtype: {_cast_act(weights, adt)} F32 parameters cast to {adt}")
    from mlx.utils import tree_flatten
    missing = sorted(k for k, v in tree_flatten(model.parameters()) if k not in weights and not mx.issubdtype(v.dtype, mx.integer))   # an integer array is a constant the class builds from its config (qwen4_exp PLE hash multipliers / offsets), never an initial weight
    if missing:                                                   # strict=False below tolerates extra tensors; a parameter nothing filled would keep its random / default init
        raise ValueError(f"{path}: the file has no tensor for {len(missing)} parameter(s) of the {config.get('model_type')} model, e.g. {missing[:4]}: "
                         f"an unsupported or incomplete checkpoint is refused, not served with initial weights")
    model.load_weights(list(weights.items()), strict=False)
    model.eval()
    if src.rope_freqs is not None:
        n_rope = install_rope_freqs(model, src.rope_freqs)
        log(f"rope_freqs: {n_rope} attention layers use the file's frequency factors")
    if src.mtp_packs:                                             # flag glm_dense4: the draft head's stored 4-bit linears (glm_nextn.NextN.build swaps them in)
        model.__dict__["_g4_mtp_packs"] = src.mtp_packs
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
    from . import glm_moe
    glm_moe.install(model, log)                                         # flag KYOJIN_GLM_MOE: fused GLM decode MoE block (router + experts + combine)
    install_dense_glu(model, log)
    install_g3_mlp(model, log)
    from . import hc_compile
    hc_compile.install(model, log)                                       # flag KYOJIN_HC_COMPILE: gated hyper-connection module (qwen4_exp) as one compiled call at decode
    from . import hc_fuse
    hc_fuse.install(model, log)                                          # flag KYOJIN_HC_FUSE: the same module as 3 fused kernels around the stock linears (supersedes HC_COMPILE at decode)
    from . import glm_hc
    glm_hc.install(model, log)                                           # flag KYOJIN_GLM_HC_FUSE (G3b): mHC glue in 2 launches per branch at decode
    mx.eval(model.parameters())
    from . import foreign_gdn_fast
    foreign_gdn_fast.install(model, log)                                  # flag KYOJIN_FOREIGN_GDN_FAST (off): fused gated-delta decode kernel on structurally matching linear-attention modules (qwen4_exp)
    from . import foreign_attn_fast
    foreign_attn_fast.install(model, log)                                 # flag KYOJIN_FOREIGN_ATTN_FAST (off): QSA indexer decode mask (qwen4_exp) in one launch
    from . import foreign_qsa_fuse
    foreign_qsa_fuse.install(model, log)                                  # flag KYOJIN_FOREIGN_QSA_FUSE (off): sparse attend over the selected QSA blocks (qwen4_exp) in one launch at decode
    from . import glm_small
    glm_small.install(model, log)                                        # flag KYOJIN_GLM_FUSE_SMALL (G10b): linear-attention decode body in one launch
    from . import step_compile
    step_compile.install(model, log)                                     # flag KYOJIN_STEP_COMPILE (off): fixed-state decoder layers (and the mlp of KV layers) as compiled programs at decode; installed last
    if LC.drop_all():
        log("[foreign] page cache of the shards dropped (loadcache)")
    log(f"[foreign] {kind}: {n_packs} linears {fm}, {n_exp} expert blocks" + "".join("\n[foreign] " + n for n in src.notes[:8]))
    return model, config
