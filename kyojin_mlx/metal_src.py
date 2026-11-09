# SPDX-License-Identifier: Apache-2.0
"""Metal source generators for the EXL3 `mul1` kernels. Pure Python strings: no mlx import, so the window
and index logic can be emulated and tested on any machine (tests/test_emulation.py).

Ideas reused (Apache-2.0) from daig/omlx-exl3 (src/omlx/exl3/kernels.py) and beamivalice/PonyExl3
(ponyexl3/mlx/gemv_metal.py):
  * a lane owns one "run" (32 consecutive trellis positions = tile columns g and g+8, all 16 rows) and builds its
    16-bit states by funnel shifts of the K stream words of the run plus the word before (omlx gen_windows);
  * mul1 decode by a two-lane byte-pair sum and ONE half fma (omlx dec1; Pony uses a byte loop, same value);
  * 8 simdgroups per threadgroup: 2 column halves x 4 k-splits, reduced in a fixed order (omlx);
  * stream words for the next k-slice are loaded before the current slice is decoded (omlx gen_prefetch).
Differences from both: fp32 rotated input, k split across threadgroups goes through a partials buffer plus a
finalize kernel (no atomics, deterministic), one codebook (mul1), no sibling groups.
"""
from __future__ import annotations

import functools

import os
NACC = int(os.environ.get("KYOJIN_NACC", "1"))      # independent accumulator chains per (row, column half)
DENSE_DEC = int(os.environ.get("KYOJIN_DENSE_DEC", "0"))   # dense (non-routed) trellis GEMV decode cut: 0 = dec_mul1 (today), 1 = dec_mul1r1 (lane sum by imad,
# the same fp16 value bit for bit), 2 = dec_mul1fd (bare byte sum + affine folded out, NOT bit-identical; see dense_cb / af_fold). Read per kernel build.

HEADER = r"""
inline half dec_mul1(uint s)
{
    uint h = s * 0x83DCD12Du;
    uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);        // (b0+b1, b2+b3) in two 16-bit lanes
    half r = as_type<half>(ushort(ushort(t) + ushort(t >> 16) + ushort(0x6400)));   // 1024 + bytesum
    return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
}

inline half dec_mul1af(uint s)                                  // dec_mul1 before its affine (r * A + B): the affine-folded GEMVs apply it once per output
{
    uint h = s * 0x83DCD12Du;
    uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
    return as_type<half>(ushort(ushort(t) + ushort(t >> 16) + ushort(0x6400)));
}

inline float fmad(half d, float x, float a) { return fma(float(d), x, a); }     // accumulate one decoded weight (every GEMV emit site)
inline float fmad(half2 d, float x, float a) { return fma(float(d.y), x, fma(float(d.x), x, a)); }     // dec_mul1fd: the two lane sums of the byte sum, no lane add

inline half2 dec_mul1fd(uint s)                                 // fast decode (KYOJIN_MOE_FASTDEC): the byte sum left as two denormal-half lane sums (bits = integer, value S * 2^-24): no lane add, no 1024 bias; affine folded like mul1af
{
    uint h = s * 0x83DCD12Du;
    return as_type<half2>((h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu));
}

inline half dec_mcg(uint s)
{
    uint h = s * 0xCBAC1FEDu;
    half2 v = as_type<half2>((h & 0x8fff8fffu) ^ 0x3b603b60u);
    return v.x + v.y;
}

inline half dec_3inst(uint s)
{
    uint h = s * 89226354u + 64248484u;
    half2 v = as_type<half2>((h & 0x8fff8fffu) ^ 0x3b603b60u);
    return v.x + v.y;
}
"""

RSQRT_HAD = "0.08838834764831845f"      # 1/sqrt(128)


def run_kk(p: int) -> int:
    """Row (k inside the 16-slice) of position p of a run."""
    return 2 * (p // 8) + (p % 2) + 8 * ((p % 4) // 2)


def gen_windows(K: int) -> tuple[list[str], list[str]]:
    """(mid lines, 32 state expressions) over the words w0..w{K-1} and wp of a run."""
    srcs, mids = [], set()
    for p in range(32):
        e = (p + 1) * K
        ih = (e - 1) // 32
        s = (ih + 1) * 32 - e
        if s <= 16:
            srcs.append(f"extract_bits(w{ih}, {s}u, 16u)" if s else f"(w{ih} & 0xffffu)")
        else:
            prev = f"w{ih - 1}" if ih > 0 else "wp"
            mids.add((ih, prev))
            srcs.append(f"extract_bits(m{ih}, {s - 16}u, 16u)")
    mid_lines = [f"uint m{ih} = insert_bits(w{ih} >> 16, {prev}, 16u, 16u);" for ih, prev in sorted(mids)]
    return mid_lines, srcs


def gen_loads(K: int, ptr: str = "tp") -> tuple[list[tuple[str, str, str]], list[str]]:
    """Loads of the run's words from `ptr` (uint*, tile base): [(var, type, expr)] plus the unpack lines.
    Expressions use g and wprev. Vector loads when K allows (the run offset K*g words stays aligned)."""
    if K % 4 == 0:
        loads = [(f"q{v}", "uint4", f"((const device uint4*) {ptr})[{K // 4} * g + {v}]") for v in range(K // 4)]
        unpack = [f"uint w{4 * v + c} = q{v}.{'xyzw'[c]};" for v in range(K // 4) for c in range(4)]
    elif K % 2 == 0:
        loads = [(f"q{v}", "uint2", f"((const device uint2*) {ptr})[{K // 2} * g + {v}]") for v in range(K // 2)]
        unpack = [f"uint w{2 * v + c} = q{v}.{'xy'[c]};" for v in range(K // 2) for c in range(2)]
    else:
        loads = [(f"w{c}", "uint", f"{ptr}[{K} * g + {c}]") for c in range(K)]
        unpack = []
    loads.append(("wp", "uint", f"{ptr}[wprev]"))
    return loads, unpack


AF_A, AF_B = "float(as_type<half>(ushort(0x1eee)))", "float(as_type<half>(ushort(0xc931)))"   # the mul1 affine, w = r * A + B
PF2 = os.environ.get("KYOJIN_GEMV_PF2", "0") != "0"       # two-tile prefetch depth of the decode GEMVs (dense gemv_source and the routed gather GEMV): tile i+2 in flight before tile i decodes; loads reordered only, bit-identical
GPF = int(os.environ.get("KYOJIN_MOE_GPF", "0"))          # grouped (rows >= 2) routed GEMV: k tiles of trellis words in flight ahead of the decode (0 = none, 1 = one, 2 = two); loads reordered only, bit-identical       # two-tile prefetch depth of the decode GEMVs (dense gemv_source and the routed gather GEMV): tile i+2 in flight before tile i decodes; loads reordered only, bit-identical


def af_consts(cb: str) -> tuple[str, str]:
    """(A, B) of the folded affine: w = A * v + B, v the value the decoder returns. mul1fd returns lane sums as denormal halves, value S * 2^-24, so A' = 2^24 * A and B' = B + 1024 * A."""
    if cb in ("mul1fd", "mul1fdr"):
        return f"(16777216.0f * {AF_A})", f"({AF_B} + 1024.0f * {AF_A})"
    return AF_A, AF_B


def af_fold(cb: str) -> bool:
    """Affine-folded mul1 (cb "mul1af", KYOJIN_MOE_AFFOLD): the GEMV accumulates r * x and sum(x), then y = A * acc + B * sum(x). Not bit-identical
    (w is not rounded to half, fp32 sums), KL-gated, opt-in."""
    return cb in ("mul1af", "mul1fd", "mul1fdr")


def run_decode_lines(K: int, emit, cb: str = "mul1") -> list[str]:
    """The 32 decoded values of a run; `emit(p, kk, hi, value_expr)` returns the statement for each."""
    mid, srcs = gen_windows(K)
    out = list(mid)
    for p in range(32):
        out.append(emit(p, run_kk(p), (p % 8) >= 4, f"dec_{cb}({srcs[p]})"))
    return out


# --------------------------------------------------------------------------------------------- rot

ROT_SOURCE_V0 = r"""
    // xh = H128(x * suh) / sqrt(128), fp32 out. One threadgroup of 128 threads per (128-block, row).
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint KD = params[0];
    threadgroup float bv[128];
    bv[tid] = float(x[ulong(r) * KD + blk * 128 + tid]) * float(suh[blk * 128 + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint hs = 1; hs < 128; hs <<= 1) {
        const float a = bv[tid], b = bv[tid ^ hs];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bv[tid] = (tid & hs) ? (b - a) : (a + b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    xh[ulong(r) * KD + blk * 128 + tid] = bv[tid] * RSQRT;
""".replace("RSQRT", RSQRT_HAD)

# ------------------------------------------------------------------------------------------ finalize

FINAL_SOURCE_V0 = r"""
    // y = H128(sum over k-blocks of partials) / sqrt(128) * svh. One threadgroup of 128 threads per (block, row).
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint TGK = params[0], N = params[1], MRT = params[2];
    threadgroup float bv[128];
    float v = 0.0f;
    for (uint kb = 0; kb < TGK; ++kb) v += part[(ulong(kb) * MRT + r) * N + blk * 128 + tid];
    bv[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint hs = 1; hs < 128; hs <<= 1) {
        const float a = bv[tid], b = bv[tid ^ hs];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bv[tid] = (tid & hs) ? (b - a) : (a + b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    y[ulong(r) * N + blk * 128 + tid] = OT(bv[tid] * RSQRT * float(svh[blk * 128 + tid]));
""".replace("RSQRT", RSQRT_HAD)

# ---------------------------------------- simdgroup Hadamard (v1: registers + simd_shuffle_xor, no barriers)

_STAGE = r"""
    { const bool up = (lane & D) != 0;
      const float p0 = simd_shuffle_xor(v0, D), p1 = simd_shuffle_xor(v1, D), p2 = simd_shuffle_xor(v2, D), p3 = simd_shuffle_xor(v3, D);
      v0 = up ? (p0 - v0) : (v0 + p0); v1 = up ? (p1 - v1) : (v1 + p1);
      v2 = up ? (p2 - v2) : (v2 + p2); v3 = up ? (p3 - v3) : (v3 + p3); }          // h = 4 * D
"""

_BUTTERFLY = r"""
    // H128 over one 128-block held by a simdgroup: lane owns v0..v3 = elements 4*lane .. 4*lane+3.
    // Same stages and operand order as the reference (stage h: lower = a + b, upper = a - b with a = lower element).
    { float a0 = v0 + v1, a1 = v0 - v1, a2 = v2 + v3, a3 = v2 - v3; v0 = a0; v1 = a1; v2 = a2; v3 = a3; }          // h = 1
    { float a0 = v0 + v2, a2 = v0 - v2, a1 = v1 + v3, a3 = v1 - v3; v0 = a0; v1 = a1; v2 = a2; v3 = a3; }          // h = 2
STAGES"""

_BUTTERFLY = _BUTTERFLY.replace("STAGES", "".join(_STAGE.replace("D)", f"{d})").replace("D,", f"{d},").replace("* D", f"* {d}") for d in (1, 2, 4, 8, 16)))

_ROT_V1 = r"""
    // xh = H128(x * suh) / sqrt(128). One simdgroup per (128-block, row), 4 blocks per threadgroup of 128 threads.
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y;
    const uint KD = params[0];
    if (blk * 128 >= KD) return;
    const ulong o = ulong(r) * KD + blk * 128 + lane * 4;
    const float4 xv = float4(*((const device XV4*)(x + o)));
    const float4 sv = float4(*((const device half4*)(suh + blk * 128 + lane * 4)));
    float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
BUTTERFLY
    *((device OV4*)(xh + o)) = OV4(float4(v0, v1, v2, v3) * RSQRT);
"""

_FINAL_V1 = r"""
    // y = H128(sum over k-blocks of partials) / sqrt(128) * svh. One simdgroup per (128-block, row).
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y;
    const uint TGK = params[0], N = params[1], MRT = params[2];
    if (blk * 128 >= N) return;
    float4 acc = float4(0.0f);
    for (uint kb = 0; kb < TGK; ++kb) acc += float4(*((const device PV4*)(part + (ulong(kb) * MRT + r) * N + blk * 128 + lane * 4)));
    float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
    const float4 sv = float4(*((const device half4*)(svh + blk * 128 + lane * 4)));
    const float4 o4 = float4(v0, v1, v2, v3) * RSQRT * sv;
    *((device vec<OT, 4>*)(y + ulong(r) * N + blk * 128 + lane * 4)) = vec<OT, 4>(o4);
"""


_XV_LOAD = "const float4 xv = float4(*((const device XV4*)(x + o)));"

_FQ8_BLOCK = r"""
    // fp8 e4m3 (ue8m0 power-of-two scale, blocks of 32 = 8 lanes of 4) round trip of the input, the arithmetic of deepseek_v41_fp8_roundtrip
    // (mlx-vlm fakequant.py) lane by lane: amax is a max (order free), every element is stored as half like that kernel's output, then read back.
    const float4 xr = float4(*((const device half4*)(x + o)));
    float am = metal::max(metal::max(metal::abs(xr.x), metal::abs(xr.y)), metal::max(metal::abs(xr.z), metal::abs(xr.w)));
    am = metal::max(am, simd_shuffle_xor(am, 1));
    am = metal::max(am, simd_shuffle_xor(am, 2));
    am = metal::max(am, simd_shuffle_xor(am, 4));
    am = metal::max(am, 1e-4f);
    const uint amb = as_type<uint>(am * (1.0f / 448.0f));
    const uint fq_e = (amb >> 23) + ((amb & 0x7fffff) != 0 ? 1u : 0u);
    const float fq_scale = as_type<float>(fq_e << 23);
    const float fq_inv = as_type<float>((254u - fq_e) << 23);                  // 1 / fq_scale, exact (a power of two): x * fq_inv == x / fq_scale bit for bit
    float4 xv;
FQ8_ELEMS"""

_FQ8_ELEM = r"""    { const float q = metal::clamp(xr.C * fq_inv, -448.0f, 448.0f);
      const uint qb = as_type<uint>(metal::abs(q));
      const uint se = uint(metal::max(int(qb >> 23) - 3, 118));
      const float step = as_type<float>(se << 23), istep = as_type<float>((254u - se) << 23);      // power of two: q * istep == q / step exactly
      xv.C = float(half((metal::rint(q * istep) * step) * fq_scale)); }
"""


_GLU_XR = r"""
    // shared_glu: xr = half(silu(min(g, L)) * clip(u, -L, L)), the rounding points of the stock chain (fp32 inside, one cast to the stream dtype), g / u = the gate / up outputs.
    float4 xr;
    {
        const float4 gx = float4(*((const device half4*)(x + o))), ux = float4(*((const device half4*)(u + o)));
        const float4 g4 = CLAMPG, u4 = CLAMPU;
        float4 sg;
        for (int i = 0; i < 4; ++i) {                                 // mlx sigmoid: y = 1 / (1 + exp(|x|)); x < 0 ? y : 1 - y
            const float xs = g4[i];
            const float ys = 1.0f / (1.0f + metal::precise::exp(abs(xs)));
            sg[i] = (xs < 0.0f) ? ys : 1.0f - ys;
        }
        xr = float4(half4((g4 * sg) * u4));
    }
"""
_FQ8_LOAD = "const float4 xr = float4(*((const device half4*)(x + o)));"


def with_glu(src: str, limit: float) -> str:
    """`src` (a fused kernel body whose fq8 block loads xr from x, see with_fq8) with xr = the clamped SwiGLU of x (gate) and u (up) instead: the shared expert's
    activation inside the down projection's prologue. Needs the kernel input `u` (same layout as x)."""
    assert src.count(_FQ8_LOAD) == 1, "fq8 block template changed"
    lim = repr(float(limit)) + "f"
    block = _GLU_XR.replace("CLAMPG", f"min(gx, float4({lim}))" if limit > 0 else "gx").replace("CLAMPU", f"clamp(ux, float4(-{lim}), float4({lim}))" if limit > 0 else "ux")
    return src.replace(_FQ8_LOAD, block.strip("\n"))


def with_fq8(src: str) -> str:
    """`src` (a rot / rot_gather kernel body that loads its input with _XV_LOAD, fp16 input) with the fp8 round trip of that input folded into the load:
    one launch fewer than a separate fake-quant kernel in front, same bits."""
    assert src.count(_XV_LOAD.replace("XV4", "half4")) == 1, "rot template changed"
    block = _FQ8_BLOCK.replace("FQ8_ELEMS", "".join(_FQ8_ELEM.replace(".C", f".{c}") for c in "xyzw"))
    return src.replace(_XV_LOAD.replace("XV4", "half4"), block.strip("\n"))


def perm_store(ptr: str, val: str, half: bool) -> str:
    """Store of the lane's float4 `val` (elements 4 * lane .. 4 * lane + 3 of a 128-block that starts at `ptr`) with the k permutation
    inside every 16-block that the fused EXL3 + NAX kernels (metal_nax) expect: stored position 8 * (q & 1) + 4 * (c >> 1) + 2 * (q >> 1) + (c & 1)
    holds element 4 * q + c of the 16-block (q = lane & 3). The values are the same, only their order inside the 16-block changes."""
    ot = "half" if half else "float"
    return (f"{{ const float4 pv = {val}; device {ot}* pd = (device {ot}*)({ptr}) + (lane >> 2) * 16 + 8 * (lane & 1) + 2 * ((lane >> 1) & 1);\n"
            f"      pd[0] = {ot}(pv.x); pd[1] = {ot}(pv.y); pd[4] = {ot}(pv.z); pd[5] = {ot}(pv.w); }}")


@functools.lru_cache(maxsize=None)
def rot_source(in_half: bool, out_half: bool, fq8: bool = False, perm: bool = False) -> str:
    s = _ROT_V1
    if perm:
        old = "*((device OV4*)(xh + o)) = OV4(float4(v0, v1, v2, v3) * RSQRT);"
        assert s.count(old) == 1
        s = s.replace(old, perm_store("xh + ulong(r) * KD + blk * 128", "float4(v0, v1, v2, v3) * RSQRT", out_half))
    src = (s.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD)
           .replace("XV4", "half4" if in_half else "float4").replace("OV4", "half4" if out_half else "float4"))
    assert not fq8 or in_half, "the folded fp8 round trip is for fp16 activations"
    return with_fq8(src) if fq8 else src


@functools.lru_cache(maxsize=None)
def final_source(part_half: bool = False) -> str:
    return (_FINAL_V1.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD).replace("PV4", "half4" if part_half else "float4"))


FINAL_SOURCE = final_source(False)


def had_grid(blocks: int, rows: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(grid, threadgroup) of the simdgroup Hadamard kernels: 4 blocks of 128 per threadgroup."""
    return (((blocks + 3) // 4) * 128, rows, 1), (128, 1, 1)


# --------------------------------------------------------------------------------------------- decode

_DECODE_SOURCE = r"""
    // Wr^T fp16 [n, k] (inner weight before the Hadamards). One thread per (tile, run g).
    constexpr uint TW = 8 * BITS;
    const uint gid = thread_position_in_grid.x;
    const uint KT = params[0], NT = params[1];
    const uint g = gid & 7, t = gid >> 3;
    if (t >= KT * NT) return;
    // Four consecutive threads-of-8 take four consecutive k tiles of one 16-row block: a row's 4 x 32 bytes leave as one 128-byte line (the plain tile order
    // writes 32-byte pieces, 25 % of a line: the kernel is store bound at 60 GB/s). KT % 4 != 0: the plain order.
    const bool q4 = REMAP && (KT & 3u) == 0u;
    const uint kt = q4 ? ((t >> 2) / NT) * 4u + (t & 3u) : t / NT;
    const uint nt = q4 ? (t >> 2) % NT : t % NT;
    const uint tile = kt * NT + nt;
    const uint wprev = (BITS * g + TW - 1) % TW;
    const device uint* tp = trellis + ulong(tile) * TW;
    LOADS
    const ulong KD = ulong(KT) * 16;
    device half* wlo = w + (ulong(nt) * 16 + g) * KD + kt * 16;
    device half* whi = w + (ulong(nt) * 16 + 8 + g) * KD + kt * 16;
    RUN
"""

# Vector-store variant: the 16 values of a row slice go to registers first, then leave as 4 x half4 per row (the scalar form issues 32 two-byte stores per thread at a
# row stride of 2 * KD bytes). The values are the same expressions: bit-identical.
_DECODE_SOURCE_V = _DECODE_SOURCE.replace("    RUN\n", "    half lo[16], hi[16];\n    RUN\n    STORES\n")
_DECODE_STORES = "\n    ".join(f"*((device half4*)(w{r} + {4 * j})) = half4({r2}[{4 * j}], {r2}[{4 * j + 1}], {r2}[{4 * j + 2}], {r2}[{4 * j + 3}]);"
                             for r, r2 in (("lo", "lo"), ("hi", "hi")) for j in range(4))


DEC_REMAP = True      # decode_wr tile order (see _DECODE_SOURCE); a module switch for A/B runs, part of the kernel key in kernels.decode_wr


def decode_source(K: int, cb: str = "mul1", vec: bool = True, remap: bool | None = None) -> str:
    remap = DEC_REMAP if remap is None else remap
    loads, unpack = gen_loads(K)
    ld = [f"{t} {v} = {e};" for v, t, e in loads] + unpack
    if not vec:
        run = run_decode_lines(K, lambda p, kk, hi, e: f"{'whi' if hi else 'wlo'}[{kk}] = {e};", cb)
        return _DECODE_SOURCE.replace("LOADS", "\n    ".join(ld)).replace("RUN", "\n    ".join(run)).replace("REMAP", "true" if remap else "false")
    run = run_decode_lines(K, lambda p, kk, hi, e: f"{'hi' if hi else 'lo'}[{kk}] = {e};", cb)
    return (_DECODE_SOURCE_V.replace("LOADS", "\n    ".join(ld)).replace("RUN", "\n    ".join(run))
            .replace("STORES", _DECODE_STORES).replace("REMAP", "true" if remap else "false"))


# ------------------------------------------------------------------------------------------ GEMV partial

_GEMV_SOURCE = r"""
    constexpr uint TW = 8 * BITS;
    constexpr uint WS = 4;                                  // k-splits per threadgroup (x 2 column halves)
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;        // 128-column output block
    const uint kblock = threadgroup_position_in_grid.y;
    const uint KT = params[0], NT = params[1], TGK = params[2];
    const uint KD = KT * 16, N = NT * 16;
    const uint half_ = warp & 1u, split = warp >> 1;
    const uint g = lane & 7;
    const uint nt = blk * 8 + half_ * 4 + (lane >> 3);

    const uint S = TGK * WS;
    const uint chunk = (KT + S - 1) / S;
    const uint ks0 = (kblock * WS + split) * chunk;
    const uint myn = ks0 < KT ? min(chunk, KT - ks0) : 0u;

    ACC_DECL

    const uint wprev = (BITS * g + TW - 1) % TW;
    const device uint* tp = trellis + (ulong(ks0) * NT + nt) * TW;
    const ulong tstride = ulong(NT) * TW;
    const device float4* xp = (const device float4*)(xh + ks0 * 16);

    PRELOAD
    for (uint i = 0; i < myn; ++i) {
        LOAD_WORDS
        X_DECL
        RUN_CODE
        tp += tstride;
        xp += 4;
    }

    // ---- k-splits of this threadgroup -> 128-column partial (fixed order) ----
    threadgroup float red[WS][2][32][2 * MR];
    STORE_RED
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < MR * 128; idx += 256) {
        const uint r = idx >> 7, c = idx & 127;
        const uint h = c >> 6, l = ((c & 63) >> 4) * 8 + (c & 7), hl = (c >> 3) & 1;
        float v = 0.0f;
        for (uint j = 0; j < WS; ++j) v += red[j][h][l][2 * r + hl];
        part[(ulong(kblock) * MR + r) * N + blk * 128 + c] = v;
    }
"""


HALF_ROWS = int(os.environ.get("KYOJIN_HALF_ROWS", "4"))       # ON at 4 rows (real-text NLL vs fp32 rows +0.002 nats, within noise; +4.5 % served MTP decode); >= 2: the fused decode GEMV of that many rows or more multiply-accumulates in packed fp16 (half4 FMA against x staged as half4 by the prologue, flushed into the fp32 accumulators every HALF_FLUSH tiles); 0 = fp32 FMAs. Row results differ from the fp32 bits at one fp16 ulp of the output (KL-gated)
HALF_FLUSH = int(os.environ.get("KYOJIN_HALF_FLUSH", "4"))
MMA_ROWS = int(os.environ.get("KYOJIN_MMA_ROWS", "0"))     # >= 2 (needs the half-staged x, KYOJIN_HALF_ROWS <= it): grouped GEMV classes of that many rows or more decode each tile once into threadgroup fp16 and run the rows through simdgroup 8x8 MMA (fp32 accumulate, KL-gated like HALF_ROWS); 0 = row FMAs


def half_on(mr: int) -> bool:
    """Packed-fp16 row FMAs for a fused launch of mr rows (HALF_ROWS is a module constant that tools may flip between calls)."""
    return bool(HALF_ROWS) and mr >= HALF_ROWS and NACC == 1


def run_decode_half(K: int, mr: int, cb: str = "mul1") -> list[str]:
    """The 32 decoded weights of a run as 8 half4 groups (positions 4m..4m+3: k rows 2j, 2j+1, 8+2j, 9+2j of the 16-slice, j = m // 2), each FMA'd against every
    row's staged half4 of x (xh{r}_{j}) into a half4 accumulator per (row, lo/hi); the accumulators go into the fp32 ones every HALF_FLUSH tiles."""
    mid, srcs = gen_windows(K)
    out = list(mid)
    for m in range(8):
        hh = "hi" if m % 2 else "lo"
        out.append("{ half4 dv = half4(" + ", ".join(f"dec_{cb}({srcs[4 * m + c]})" for c in range(4)) + "); "
                   + " ".join(f"hacc{r}_{hh} = fma(dv, xh{r}_{m // 2}, hacc{r}_{hh});" for r in range(mr)) + " }")
    fl = " ".join(f"a{r}_{h}0 += (float(hacc{r}_{h}.x) + float(hacc{r}_{h}.y)) + (float(hacc{r}_{h}.z) + float(hacc{r}_{h}.w)); hacc{r}_{h} = half4(0);"
                  for r in range(mr) for h in ("lo", "hi"))
    out.append(f"if ((i % {HALF_FLUSH}u) == {HALF_FLUSH - 1}u || i + 1 == myn) {{ {fl} }}")
    return out


def gemv_source(K: int, mr: int, prefetch: bool = True, cb: str = "mul1", half: bool = False, pf2: bool = False) -> str:
    """half: packed-fp16 row FMAs; only valid under fuse_gemv (its prologue stages x as half4, see _FUSED_PRO_HALF).
    pf2 (KYOJIN_GEMV_PF2): keep TWO k tiles in flight (tile i+2 issued before tile i decodes) instead of one; the loads are only reordered."""
    assert not half or NACC == 1
    loads, unpack = gen_loads(K)
    if prefetch:
        pre = [f"{t} n{v} = {t}(0);" for v, t, _ in loads]
        if pf2:
            pre += [f"{t} z{v} = {t}(0);" for v, t, _ in loads]
        pre.append("if (myn > 0) { " + " ".join(f"n{v} = {e};" for v, _, e in loads) + " }")
        nxt = gen_loads(K, "tn")[0]                                         # same loads on the next tile
        if pf2:
            pre.append("if (myn > 1) { const device uint* tn = tp + tstride; " + " ".join(f"z{v} = {e};" for v, _, e in nxt) + " }")
        top = [f"{t} {v} = n{v};" for v, t, _ in loads]
        if pf2:
            top += [f"n{v} = z{v};" for v, t, _ in loads]
            top.append("if (i + 2 < myn) { const device uint* tn = tp + 2 * tstride; "
                       + " ".join(f"z{v} = {e};" for v, _, e in nxt) + " }")
        else:
            top.append("if (i + 1 < myn) { const device uint* tn = tp + tstride; "
                       + " ".join(f"n{v} = {e};" for v, _, e in nxt) + " }")
        load_lines = top + unpack
    else:
        pre = []
        load_lines = [f"{t} {v} = {e};" for v, t, e in loads] + unpack

    xdecl = []
    for r in range(mr):
        for q in range(0 if half else 4):
            xdecl.append(f"float4 xq{r}_{q} = xp[{q} + {r} * (KD / 4)];")
        if half:
            xdecl += [f"half4 xh{r}_{j} = xhp[{j} + {r} * (KD / 4)];" for j in range(4)]
        else:
            for kk in range(16):
                xdecl.append(f"float x{r}_{kk} = xq{r}_{kk // 4}.{'xyzw'[kk % 4]};")
    nc = NACC
    acc = "\n    ".join(f"float " + ", ".join(f"a{r}_{h}{c} = 0.0f" for h in ("lo", "hi") for c in range(nc)) + ";" for r in range(mr))
    def tot(r, h):
        return " + ".join(f"a{r}_{h}{c}" for c in range(nc))
    store = "\n    ".join(f"red[split][half_][lane][{2 * r}] = {tot(r, 'lo')}; red[split][half_][lane][{2 * r + 1}] = {tot(r, 'hi')};"
                          for r in range(mr))

    if af_fold(cb) and not half:                                     # affine folded out of the per-weight decode (see af_fold): y = A * acc + B * sum(x), per row
        assert nc == 1
        acc += "\n    " + " ".join(f"float4 sx{r} = float4(0.0f);" for r in range(mr))
        xdecl += [f"sx{r} += (xq{r}_0 + xq{r}_1) + (xq{r}_2 + xq{r}_3);" for r in range(mr)]
        A, B = af_consts(cb)
        store = " ".join(f"{{ const float bs = {B} * ((sx{r}.x + sx{r}.y) + (sx{r}.z + sx{r}.w)); a{r}_lo0 = fma(a{r}_lo0, {A}, bs); a{r}_hi0 = fma(a{r}_hi0, {A}, bs); }}"
                         for r in range(mr)) + "\n    " + store

    def emit(p, kk, hi, e):
        h = ("hi" if hi else "lo") + str((p // 8) % nc)
        fm = " ".join(f"a{r}_{h} = fmad(d, x{r}_{kk}, a{r}_{h});" for r in range(mr))
        return f"{{ auto d = {e}; {fm} }}"

    run = run_decode_half(K, mr, cb) if half else run_decode_lines(K, emit, cb)
    if half:
        acc += "\n    " + "\n    ".join(f"half4 hacc{r}_lo = half4(0), hacc{r}_hi = half4(0);" for r in range(mr))
    return (_GEMV_SOURCE.replace("ACC_DECL", acc)
            .replace("PRELOAD", "\n    ".join(pre))
            .replace("LOAD_WORDS", "\n        ".join(load_lines))
            .replace("X_DECL", "\n        ".join(xdecl))
            .replace("RUN_CODE", "\n        ".join(run))
            .replace("STORE_RED", store))



# ------------------------------------------------------------------------------------------ fused linear (one launch)
# rot + GEMV + finalize of one EXL3 linear in ONE kernel (decode, 1..MAX_ROWS rows). Same grid as the GEMV: (blocks x TGK) threadgroups.
#   prologue  the threadgroup rotates the 128-blocks that cover its own k range into threadgroup memory (same butterfly, same arithmetic as the rot
#             kernel; the fp8 fake quant of the input folds into the load as in with_fq8), then the GEMV body reads x from there
#   tail      each threadgroup stores its partial (fp32 bits, relaxed atomics) and bumps the counter of its 128-column block; the arrival that completes a
#             round (TGK is a power of two, the counter wraps harmlessly and is never reset) sums the TGK partials in k order and runs the finalize butterfly:
#             the same operations in the same order as final_source, so the result is bit-identical to the three-launch path
# The counter array is private to the module and shared between calls: calls that can run concurrently need different counter sets (Mul1Gemv.fused rotates them).
# `fuse_gemv` rewrites any of the GEMV templates (single, same-input group, grouped) with asserted substitutions, like the group kernels do.

_FUSED_PRO = r"""
    threadgroup float sm[SMN];
    threadgroup uint last_;
    {
        // ---- prologue: xh = H128(x * suh) / sqrt(128) for the 128-blocks that cover this threadgroup's k range (all rows) ----
        const uint tb = kblock * WS * chunk;
        const uint eb0 = (tb * 16) / 128;
        const uint eb1 = tb < KT ? (min(tb + WS * chunk, KT) * 16 + 127) / 128 : eb0;
        for (uint rr = 0; rr < MR; ++rr) {
            for (uint bb = eb0 + warp; bb < eb1; bb += 8) {
                const ulong o = XO;
                const float4 xv = float4(*((const device XV4*)(x + o)));
                const float4 sv = float4(*((const device half4*)(SUH)));
                float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
BUTTERFLY
                *((threadgroup float4*)(sm + rr * XSR + (bb - eb0) * 128 + lane * 4)) = float4(v0, v1, v2, v3) * RSQRT;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float4* xp = (const threadgroup float4*)(sm + (ks0 * 16 - ((kblock * WS * chunk * 16) / 128) * 128));
"""

_FUSED_TAIL = r"""
    // ---- tail: the last threadgroup of this column block finalizes ----
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit(((device atomic_uint*)(ctr)) + CTRI, 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last_ = ((old + 1u) & (TGK - 1u)) == 0u ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last_ == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (warp < MR) {
TAIL_PRE
        float4 acc = float4(0.0f);
        const device atomic_uint* pb = (const device atomic_uint*)(part + PB);
        const ulong ks = KS;                                                            // stride of one k-split
#define LDP(j) float4(as_type<float>(atomic_load_explicit(pb + (ulong(kb) + (j)) * ks, memory_order_relaxed)), as_type<float>(atomic_load_explicit(pb + (ulong(kb) + (j)) * ks + 1, memory_order_relaxed)), \
                      as_type<float>(atomic_load_explicit(pb + (ulong(kb) + (j)) * ks + 2, memory_order_relaxed)), as_type<float>(atomic_load_explicit(pb + (ulong(kb) + (j)) * ks + 3, memory_order_relaxed)))
        uint kb = 0;
        for (; kb + 4 <= TGK; kb += 4) {                                                // four independent loads in flight, summed in k order
            const float4 p0 = LDP(0), p1 = LDP(1), p2 = LDP(2), p3 = LDP(3);
            acc += p0; acc += p1; acc += p2; acc += p3;
        }
        for (; kb < TGK; ++kb) acc += LDP(0);
#undef LDP
        float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
        const float4 sv = float4(*((const device half4*)(SVH)));
        const float4 o4 = float4(v0, v1, v2, v3) * RSQRT * sv;
        *((device vec<OT, 4>*)(YOUT)) = vec<OT, 4>(o4);
    }
"""


TAIL_SC = "sc"          # the release / acquire pair as two device-scope seq_cst fences around a relaxed counter bump (measured: the dominant part of the tail)
TAIL_ACQREL = "acqrel"  # one acq_rel counter bump: release orders this threadgroup's partial stores, acquire makes the counter's release sequence carry the other
                        # threadgroups' stores to the finalizing threadgroup. Same arithmetic, same reads, so the result is bit-identical (only the ordering of the
                        # arrivals changes, and the finalize only runs on the arrival that completes the round).
TAIL_ATOM = "atom"      # the finalizing warp reads the TGK partials of its 4 columns with four scalar relaxed atomic loads per float4 (visibility did not need a
                        # fence in the read itself: it needed the fence that the arrival block already runs)
TAIL_VEC = "vec"        # same reads as one 16-byte plain load per (k-split, this lane's 4 columns) after the same fence: same values, same k order of the adds, so
                        # bit-identical; the 4x fewer load instructions and 4x fewer atomic-unit trips are worth 3-18 % of a small linear's launch
_SYNC_SC = ("        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);\n"
            "        const uint old = atomic_fetch_add_explicit(((device atomic_uint*)(ctr)) + CTRI, 1u, memory_order_relaxed);\n"
            "        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);\n")
_SYNC_ACQREL = ("        const uint old = atomic_fetch_add_explicit(((device atomic_uint*)(ctr)) + CTRI, 1u, memory_order_acq_rel, mem_flags::mem_device);\n")
_SYNC_POST = "    if (last_ == 0u) return;\n    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);\n"

_LDP_ATOM = """#define LDP(j) float4(as_type<float>(atomic_load_explicit(pb + (ulong(kb) + (j)) * ks, memory_order_relaxed)), as_type<float>(atomic_load_explicit(pb + (ulong(kb) + (j)) * ks + 1, memory_order_relaxed)), \\
                      as_type<float>(atomic_load_explicit(pb + (ulong(kb) + (j)) * ks + 2, memory_order_relaxed)), as_type<float>(atomic_load_explicit(pb + (ulong(kb) + (j)) * ks + 3, memory_order_relaxed)))
        uint kb = 0;
        for (; kb + 4 <= TGK; kb += 4) {                                                // four independent loads in flight, summed in k order
            const float4 p0 = LDP(0), p1 = LDP(1), p2 = LDP(2), p3 = LDP(3);
            acc += p0; acc += p1; acc += p2; acc += p3;
        }
        for (; kb < TGK; ++kb) acc += LDP(0);
#undef LDP
"""

_LDP_VEC = """        const device float4* pf4 = (const device float4*)pb;                            // one 16-byte load per (k-split, lane's 4 columns)
        const ulong ks4 = ks / 4;
        uint kb = 0;
        for (; kb + 4 <= TGK; kb += 4) {                                                // four independent loads in flight, summed in k order
            const float4 p0 = pf4[ulong(kb) * ks4], p1 = pf4[ulong(kb + 1) * ks4], p2 = pf4[ulong(kb + 2) * ks4], p3 = pf4[ulong(kb + 3) * ks4];
            acc += p0; acc += p1; acc += p2; acc += p3;
        }
        for (; kb < TGK; ++kb) acc += pf4[ulong(kb) * ks4];
"""


def tail_reads(tail: str, mode: str = TAIL_ATOM) -> str:
    """The finalize's partial reads: scalar relaxed atomics (default) or one plain float4 load per k-split (same values, same order of the adds)."""
    if mode == TAIL_ATOM:
        return tail
    assert mode == TAIL_VEC and tail.count(_LDP_ATOM) == 1, "fused tail template changed"
    return tail.replace(_LDP_ATOM, _LDP_VEC)


def tail_sync(tail: str, mode: str = TAIL_SC, arrive: str = "TGK") -> str:
    """The arrival block of the fused tail: `mode` picks the memory ordering (the arithmetic of the finalize is untouched), `arrive` is the number of
    threadgroups of one column block whose arrival completes it (the k-splits, or two per k-split when a 128-column block is two threadgroups wide)."""
    if arrive != "TGK":
        old = "        last_ = ((old + 1u) & (TGK - 1u)) == 0u ? 1u : 0u;\n"
        assert tail.count(old) == 1, "fused tail template changed"
        tail = tail.replace(old, f"        last_ = ((old + 1u) & (({arrive}) - 1u)) == 0u ? 1u : 0u;\n")
    if mode == TAIL_SC:
        return tail
    if mode == TAIL_ACQREL:
        assert tail.count(_SYNC_SC) == 1 and tail.count(_SYNC_POST) == 1, "fused tail template changed"
        return tail.replace(_SYNC_SC, _SYNC_ACQREL).replace(_SYNC_POST, "    if (last_ == 0u) return;\n")
    raise ValueError(mode)


_HALF_STORE = """{
                    threadgroup half* smh = (threadgroup half*)sm;
                    threadgroup half* tph = smh + rr * XSR + (bb - eb0) * 128 + (lane >> 2) * 16;
                    const float4 v4 = float4(v0, v1, v2, v3) * RSQRT;
                    for (uint c = 0; c < 4; ++c) {
                        const uint e = 4 * (lane & 3u) + c;                         // element of the 16-slice -> half4 j = (e & 7) / 2, component (e & 1) + 2 (e >> 3)
                        tph[((e & 7u) >> 1) * 4 + (e & 1u) + ((e >> 3) << 1)] = half(v4[c]);
                    }
                }"""


def stage_half(pro: str) -> str:
    """_FUSED_PRO with the x tile stored as half4 vectors (j = (x[2j], x[2j+1], x[8+2j], x[9+2j]) per 16-slice) instead of float4 columns."""
    old = "                *((threadgroup float4*)(sm + rr * XSR + (bb - eb0) * 128 + lane * 4)) = float4(v0, v1, v2, v3) * " + RSQRT_HAD + ";"
    assert pro.count(old) == 1, "fused prologue template changed"
    pro = pro.replace(old, _HALF_STORE.replace("RSQRT", RSQRT_HAD))
    old = "    const threadgroup float4* xp = (const threadgroup float4*)(sm + (ks0 * 16 - ((kblock * WS * chunk * 16) / 128) * 128));"
    assert pro.count(old) == 1, "fused prologue template changed"
    return pro.replace(old, "    const threadgroup half4* xp = (const threadgroup half4*)((threadgroup half*)sm + (ks0 * 16 - ((kblock * WS * chunk * 16) / 128) * 128));")


def fused_dims(kt: int, tgk: int) -> int:
    """Floats per row of the threadgroup x buffer: the widest k range (in whole 128-blocks) over the TGK k-splits (WS = 4 per threadgroup)."""
    chunk = (kt + tgk * 4 - 1) // (tgk * 4)
    w = 0
    for kb in range(tgk):
        tb = kb * 4 * chunk
        if tb < kt:
            w = max(w, (min(tb + 4 * chunk, kt) * 16 + 127) // 128 - (tb * 16) // 128)
    return w * 128


def fuse_gemv(s: str, mr: int, xsr: int, in_half: bool, fq8: bool, *, xo: str, suh: str, ctri: str, pb: str, ks: str, svh: str, yout: str,
              tail_pre: str = "", xp_old: str = "    const device float4* xp = (const device float4*)(xh + ks0 * 16);\n", prologue: bool = True,
              sync: str = TAIL_SC, arrive: str = "TGK", warps: int = 8, reads: str = TAIL_ATOM, glu: float | None = None) -> str:
    """`s` (a gemv_source or a derivative: same-input group, grouped) as a one-launch rot + GEMV + finalize kernel. Expressions (C++ text, in the scope of the
    kernel body): xo = element offset of x for row rr / block bb (rr, bb, lane, KD), suh = pointer of the half4 suh, ctri = counter index, pb = element offset of the
    partial (k-split 0) of row `warp` at this lane's 4 columns, ks = element stride of a k-split, svh = pointer of the half4 svh, yout = pointer of the 4 outputs
    (row `warp`). The kernel also needs the inputs x, suh, svh, ctr and the outputs part, y(s) declared by the caller.
    prologue=False: no rot in the kernel, the GEMV reads the already rotated xh (input `xh`, a separate rot launch) and only the finalize tail is fused;
    the repeated per-column-block prologue is what makes the one-launch form slower on big linears.
    sync / arrive / warps / reads: memory ordering of the arrival block (metal_src.tail_sync), the number of arrivals that complete one column block, the warps of
    the threadgroup (the prologue's 128-block stride), and how the finalize reads the partials (metal_src.tail_reads). All default to today's text."""
    smn = max(mr * xsr, 4 * 2 * 32 * 2 * mr)
    xv4 = "half4" if in_half else "float4"
    pro = _FUSED_PRO.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD).replace("XV4", xv4)
    if warps != 8:
        old = "for (uint bb = eb0 + warp; bb < eb1; bb += 8) {"
        assert pro.count(old) == 1, "fused prologue template changed"
        pro = pro.replace(old, f"for (uint bb = eb0 + warp; bb < eb1; bb += {warps}) {{")
    if "half4 xh0_0" in s:                                                      # packed-fp16 GEMV (gemv_source half=True): x staged as half4 in the run's k order
        pro = stage_half(pro)
        s = s.replace("xhp[", "xp[")
    assert not fq8 or in_half
    assert glu is None or fq8
    if fq8:
        pro = with_fq8(pro)
        if glu is not None:
            pro = with_glu(pro, glu)
    pro = pro.replace("SMN", str(smn)).replace("XSR", str(xsr)).replace("XO", xo).replace("SUH", suh)
    tail = _FUSED_TAIL.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD).replace("TAIL_PRE", tail_pre)
    tail = tail_reads(tail, reads)
    tail = tail_sync(tail, sync, arrive)
    tail = tail.replace("CTRI", ctri).replace("PB", pb).replace("KS", ks).replace("SVH", svh).replace("YOUT", yout)

    def sub(old, new, cnt=1):
        nonlocal s
        assert s.count(old) == cnt, f"gemv template changed: {old!r} x{s.count(old)}"
        s = s.replace(old, new)
    if prologue:
        sub(xp_old, "")
        sub("    for (uint i = 0; i < myn; ++i) {\n", pro + "    for (uint i = 0; i < myn; ++i) {\n")      # after PRELOAD: the first weight loads are in flight during the prologue
        sub("(KD / 4)", f"({xsr} / 4)", mr * 4)
    else:
        sub("    for (uint i = 0; i < myn; ++i) {\n", f"    threadgroup float sm[{smn}];\n    threadgroup uint last_;\n" + "    for (uint i = 0; i < myn; ++i) {\n")
    sub("    threadgroup float red[WS][2][32][2 * MR];\n",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);                       // sm is reused for the k-split reduction\n"
        "    threadgroup float (*red)[2][32][2 * MR] = (threadgroup float (*)[2][32][2 * MR])sm;\n")
    i0 = s.index("        part[")
    i1 = s.index("] = v;\n", i0)
    idx = s[i0 + len("        part["):i1]
    s = (s[:i0] + f"        atomic_store_explicit((device atomic_uint*)(part + ({idx})), as_type<uint>(v), memory_order_relaxed);\n" + s[i1 + len("] = v;\n"):])
    return s + tail


@functools.lru_cache(maxsize=None)
def fused_source(K: int, mr: int, cb: str, in_half: bool, fq8: bool, xsr: int, prefetch: bool = True, prologue: bool = True, half: bool = False,
                 sync: str = TAIL_SC, reads: str = TAIL_ATOM, glu: float | None = None) -> str:
    """The fused single linear. Inputs x, suh (fp16), svh (fp16), trellis, params u32 [KT, NT, TGK, counter base], ctr (u32); outputs part (scratch), y.
    prologue=False: GEMV + finalize only, inputs xh (rotated, fp32) instead of x / suh. sync / reads: ordering of the arrival block and the finalize's partial
    reads (metal_src.tail_sync / tail_reads; both bit-identical to the defaults)."""
    return fuse_gemv(gemv_source(K, mr, prefetch, cb, half=half and prologue and NACC == 1), mr, xsr, in_half, fq8, prologue=prologue, sync=sync, reads=reads, glu=glu,
                     xo="ulong(rr) * KD + bb * 128 + lane * 4", suh="suh + bb * 128 + lane * 4", ctri="params[3] + blk",
                     pb="ulong(warp) * N + blk * 128 + lane * 4", ks="ulong(MR) * N", svh="svh + blk * 128 + lane * 4",
                     yout="y + ulong(warp) * N + blk * 128 + lane * 4")
def _sub(text: str, old: str, new: str, count: int = 1) -> str:
    assert text.count(old) == count, f"template changed, {old!r} found {text.count(old)} times"
    return text.replace(old, new)


@functools.lru_cache(maxsize=None)
def gemv_rg_source(K: int, cb: str = "mul1", mr: int = 4, half: bool = False) -> str:
    """gemv_source(K, mr) for ANY number of rows, one launch: grid z = (group of the weights) * row groups + row group, mr rows per row group.
    Every row group is the mr-row kernel on its own rows, so the arithmetic of a row (and its bits) is that of the mr-row launch; the last row
    group re-reads its final row for the rows it does not have (their results are not stored).
    params u32 [KT, NT, TGK, ROWS, GN]: ROWS rows per weight group, GN weight groups (1 for a plain linear; trellis words of group zg at
    zg * KT * NT * TW, its xh rows at zg * ROWS). Output part [TGK, GN * ROWS, N] f32.

    half (flag rg_half): the rows multiply-accumulate in packed fp16, the class the <= MAX_ROWS fused launch takes from HALF_ROWS rows on. That launch
    stages x as half4 in the k order the decoded half4 weight groups pair with, so gemv_source(half=True) reads `xhp`; here `xh` is the plain fp16 `rot`
    output, so every xh{r}_j (positions 2j, 2j+1, 8+2j, 9+2j of the tile's 16-slice) is built from the four half4 of the tile instead of staged. Same
    grouping, same order, so a row's bits are the fused launch's bits at the same row count (tests/test_rg_half.py)."""
    t = gemv_source(K, mr, False, cb, half=half)
    t = _sub(t, "const uint KT = params[0], NT = params[1], TGK = params[2];",
             "const uint KT = params[0], NT = params[1], TGK = params[2], ROWS = params[3], GN = params[4];\n"
             "    const uint RGN = (ROWS + MR - 1) / MR, zz = threadgroup_position_in_grid.z, zg = zz / RGN, rbase = (zz % RGN) * MR;")
    t = _sub(t, "const device uint* tp = trellis + (ulong(ks0) * NT + nt) * TW;",
             "const device uint* tp = trellis + ulong(zg) * (ulong(KT) * NT * TW) + (ulong(ks0) * NT + nt) * TW;")
    if half:                                                          # x is the fp16 rot output of every row of the group: 4 half4 per 16-slice
        t = _sub(t, "const device float4* xp = (const device float4*)(xh + ks0 * 16);",
                 "const device half4* xq4 = (const device half4*)(xh + ulong(zg) * ROWS * KD + ks0 * 16);")
        for r in range(mr):
            row = f"xq4[min(rbase + {r}u, ROWS - 1u) * (KD / 4)"
            for j in range(4):
                pos = (2 * j, 2 * j + 1, 8 + 2 * j, 9 + 2 * j)
                comp = ", ".join(f"{row} + {p // 4}].{'xyzw'[p % 4]}" for p in pos)
                t = _sub(t, f"half4 xh{r}_{j} = xhp[{j} + {r} * (KD / 4)];", f"half4 xh{r}_{j} = half4({comp});")
        t = _sub(t, "xp += 4;", "xq4 += 4;")
    else:
        t = _sub(t, "(const device float4*)(xh + ks0 * 16);", "(const device float4*)(xh + ulong(zg) * ROWS * KD + ks0 * 16);")
        for r in range(mr):
            t = _sub(t, f" + {r} * (KD / 4)]", f" + min(rbase + {r}u, ROWS - 1u) * (KD / 4)]", 4)
    t = _sub(t, "part[(ulong(kblock) * MR + r) * N + blk * 128 + c] = v;",
             "if (rbase + r < ROWS) part[((ulong(kblock) * GN + zg) * ROWS + rbase + r) * N + blk * 128 + c] = v;")
    return t


# ================================================================================================ MoE gather kernels
# One MoE layer at decode (1..4 rows, top-k experts per row) in five launches. Pair p = row * TOPK + j owns expert ids[p].
#   rot_gather      xh[p*NPROJ+proj]  = H128(x[p / DIV] * suh_proj[e]) / sqrt(128)                    (gate and up share one launch)
#   gemv_gather     part[(p*NPROJ+proj)*TGK + kb]  = trellis partial of expert e (K read from the table, uniform per threadgroup)
#   mid             act = silu(min(gate, L)) * clip(up, -L, L) [* w] [fp8 fake quant]; xh2[p] = H128(act * suh_down[e]) / sqrt(128)
#   gemv_gather     (NPROJ = 1) down projection partials
#   final_gather    y[row] = sum_j [w_j *] H128(sum_kb part)/sqrt(128) * svh_down[e_j]                 (fp32, j in order)
# Stacked layout: convert_stack.py (ragged-u32-v1). Same per-weight arithmetic and order as gemv_source (bit-identical partials).

HEADER_MOE = HEADER + "\n#pragma METAL fp contract(off)\n"

# ---- KYOJIN_MOEROOF decoders (routed MoE only; HEADER_MOE_ROOF is added to the header only when the lever is on) ----
# Same values as dec_mul1 / dec_mul1fd bit for bit (exhaustive over all 65536 states in tests/test_moeroof.py), fewer integer ops:
#   lane sum:   (t * 0x00010001) puts lo + hi in the upper 16 bits (lanes <= 510 each, no carry); the + 0x64000000 builds the half bits 0x6400 + S
#               in the same imad and the half is read from the upper register half (as_type<half2>(..).y): replaces shr, add, add.
#   byte pairs: (h & m) + ((h >> 8) & m) == h - 255 * ((h >> 8) & m)  (h = e + 256 * o per lane): shr, and, imad instead of and, shr, and, add.
HEADER_MOE_ROOF = r"""
inline half dec_mul1r1(uint s)                                  // dec_mul1, lane sum by imad
{
    uint h = s * 0x83DCD12Du;
    uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
    half r = as_type<half2>(t * 0x00010001u + 0x64000000u).y;
    return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
}

inline half dec_mul1r2(uint s)                                  // dec_mul1, lane sum by imad and byte pairs by one subtract-imad
{
    uint h = s * 0x83DCD12Du;
    uint t = h - 255u * ((h >> 8) & 0x00ff00ffu);
    half r = as_type<half2>(t * 0x00010001u + 0x64000000u).y;
    return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
}

inline half2 dec_mul1fdr(uint s)                                // dec_mul1fd with the byte pairs by one subtract-imad
{
    uint h = s * 0x83DCD12Du;
    return as_type<half2>(h - 255u * ((h >> 8) & 0x00ff00ffu));
}
"""

# Dense (non-routed) profile: the same two exact decoders without the MoE header's `#pragma METAL fp contract(off)`
# (the dense kernels keep the default contraction they were calibrated with).
DEC_R1R2 = r"""
inline half dec_mul1r1(uint s)                                  // dec_mul1, lane sum by imad
{
    uint h = s * 0x83DCD12Du;
    uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);
    half r = as_type<half2>(t * 0x00010001u + 0x64000000u).y;
    return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
}

inline half dec_mul1r2(uint s)                                  // dec_mul1, lane sum by imad and byte pairs by one subtract-imad
{
    uint h = s * 0x83DCD12Du;
    uint t = h - 255u * ((h >> 8) & 0x00ff00ffu);
    half r = as_type<half2>(t * 0x00010001u + 0x64000000u).y;
    return fma(r, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
}
"""
HEADER_ROOF = HEADER + DEC_R1R2                 # dense header for the mul1r1 / mul1r2 decode (dec_mul1fd is already in HEADER)
_ROOF_CBS = ("mul1r1", "mul1r2", "mul1fdr")


def dense_cb(cb: str) -> str:
    """Decode codebook of a dense trellis GEMV under DENSE_DEC (KYOJIN_DENSE_DEC, read when the kernel is built).
    Only mul1 packs are touched; every other codebook is returned unchanged. 1 = mul1r1 (bit-identical), 2 = mul1fd (fast, affine folded)."""
    if DENSE_DEC == 1 and cb == "mul1":
        return "mul1r1"
    if DENSE_DEC == 2 and cb in ("mul1", "mul1r1"):
        return "mul1fd"
    return cb


def dense_header(cb: str) -> str:
    return HEADER_ROOF if cb in _ROOF_CBS else HEADER



_ROT_GATHER = r"""
    // xh[p * NPROJ + proj] = H128(x[p / DIV] * suh_proj[ids[p]]) / sqrt(128). One simdgroup per (128-block, pair*proj).
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint zz = threadgroup_position_in_grid.y;
    const uint KD = params[0], DIV = params[1];
    if (blk * 128 >= KD) return;
    const uint p = zz / NPROJ;
    const uint e = uint(ids[p]);
    const device half* sp = SUHSEL + ulong(e) * KD;
    const ulong o = ulong(p / DIV) * KD + blk * 128 + lane * 4;
    const float4 xv = float4(*((const device XV4*)(x + o)));
    const float4 sv = float4(*((const device half4*)(sp + blk * 128 + lane * 4)));
    float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
BUTTERFLY
    *((device float4*)(xh + ulong(zz) * KD + blk * 128 + lane * 4)) = float4(v0, v1, v2, v3) * RSQRT;
"""


@functools.lru_cache(maxsize=None)
def rot_gather_source(nproj: int, in_half: bool, fq8: bool = False) -> str:
    sel = "suh0" if nproj == 1 else "((zz % 2u) ? suh1 : suh0)"
    src = (_ROT_GATHER.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD).replace("NPROJ", str(nproj))
           .replace("SUHSEL", sel).replace("XV4", "half4" if in_half else "float4"))
    assert not fq8 or in_half, "the folded fp8 round trip is for fp16 activations"
    return with_fq8(src) if fq8 else src


@functools.lru_cache(maxsize=None)
def rot_gather_half_source(nproj: int, in_half: bool, fq8: bool = False) -> str:
    """rot_gather with xh stored as half in the run's k order (perm_store: j = (x[2j], x[2j+1], x[8+2j], x[9+2j]) per 16-slice) for the packed-fp16 grouped GEMV."""
    s = rot_gather_source(nproj, in_half, fq8)
    old = "*((device float4*)(xh + ulong(zz) * KD + blk * 128 + lane * 4)) = float4(v0, v1, v2, v3) * " + RSQRT_HAD + ";"
    assert s.count(old) == 1
    return s.replace(old, perm_store("xh + ulong(zz) * KD + blk * 128", "float4(v0, v1, v2, v3) * " + RSQRT_HAD, True))


_GATHER_GEMV = r"""
    constexpr uint WS = 4;                                  // k-splits per threadgroup (x 2 column halves)
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;        // 128-column output block
    const uint kblock = threadgroup_position_in_grid.y;
    const uint zz = threadgroup_position_in_grid.z;         // pair * NPROJ + proj
    const uint KT = params[0], NT = params[1], TGK = params[2];
    const uint KD = KT * 16, N = NT * 16;
    const uint p = zz / NPROJ;
    const uint e = uint(ids[p]);
    TRSEL
    const uint K = uint(kk[e]);
    const device uint* tbase = trellis + off[e];
    const uint half_ = warp & 1u, split = warp >> 1;
    const uint g = lane & 7;
    const uint nt = blk * 8 + half_ * 4 + (lane >> 3);

    const uint S = TGK * WS;
    const uint chunk = (KT + S - 1) / S;
    const uint ks0 = (kblock * WS + split) * chunk;
    const uint myn = ks0 < KT ? min(chunk, KT - ks0) : 0u;

    float a0_lo0 = 0.0f, a0_hi0 = 0.0f;
    float4 sx0 = float4(0.0f);
    const device float4* xp0 = (const device float4*)(xh + ulong(zz) * KD + ks0 * 16);
    BODIES

    // ---- k-splits of this threadgroup -> 128-column partial (fixed order) ----
    threadgroup float red[WS][2][32][2];
    AFSTORE
    red[split][half_][lane][0] = a0_lo0; red[split][half_][lane][1] = a0_hi0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 128; idx += 256) {
        const uint c = idx;
        const uint h = c >> 6, l = ((c & 63) >> 4) * 8 + (c & 7), hl = (c >> 3) & 1;
        float v = 0.0f;
        for (uint j = 0; j < WS; ++j) v += red[j][h][l][hl];
        part[(ulong(zz) * TGK + kblock) * N + blk * 128 + c] = v;
    }
"""

_GATHER_BODY = r"""{
        constexpr uint TW = 8 * KV;
        const uint wprev = (KV * g + TW - 1) % TW;
        const device uint* tp = tbase + (ulong(ks0) * NT + nt) * TW;
        const ulong tstride = ulong(NT) * TW;
        const device float4* xp = xp0;
        for (uint i = 0; i < myn; ++i) {
            LOAD_WORDS
            X_DECL
            RUN_CODE
            tp += tstride;
            xp += 4;
        }
    }"""


_GATHER_BODY_PF = r"""{
        constexpr uint TW = 8 * KV;
        const uint wprev = (KV * g + TW - 1) % TW;
        const device uint* tp = tbase + (ulong(ks0) * NT + nt) * TW;
        const ulong tstride = ulong(NT) * TW;
        const device float4* xp = xp0;
        DECL_NEXT
        if (myn > 0u) { LOAD_NEXT }
        for (uint i = 0; i < myn; ++i) {
            COPY_CUR
            if (i + 1u < myn) { tp += tstride; LOAD_NEXT }                  // KYOJIN_MOEROOF prefetch: the next tile's words are in flight while this one decodes
            X_DECL
            RUN_CODE
            xp += 4;
        }
    }"""


_GATHER_BODY_PF2 = r"""{
        constexpr uint TW = 8 * KV;
        const uint wprev = (KV * g + TW - 1) % TW;
        const device uint* tp = tbase + (ulong(ks0) * NT + nt) * TW;
        const ulong tstride = ulong(NT) * TW;
        const device float4* xp = xp0;
        DECL_NEXT
        DECL_2
        if (myn > 0u) { LOAD_NEXT }
        if (myn > 1u) { const device uint* t2 = tp + tstride; LOAD_2 }
        for (uint i = 0; i < myn; ++i) {
            COPY_CUR
            if (i + 2u < myn) { const device uint* t2 = tp + 2u * tstride; LOAD_2 }     // KYOJIN_GEMV_PF2: two tiles in flight
            X_DECL
            RUN_CODE
            tp += tstride;
            xp += 4;
        }
    }"""


_T128_SUBS = (("const uint blk = threadgroup_position_in_grid.x;", "const uint blk = threadgroup_position_in_grid.x >> 1;"),
              ("const uint half_ = warp & 1u, split = warp >> 1;", "const uint half_ = threadgroup_position_in_grid.x & 1u, split = warp;"),
              ("for (uint idx = tid; idx < 128; idx += 256) {\n        const uint c = idx;", "for (uint idx = tid; idx < 64; idx += 128) {\n        const uint c = half_ * 64 + idx;"))


def gather_half_tg(src: str) -> str:
    """KYOJIN_GATHER_ROOF: the gather GEMV as threadgroups of 128 threads (4 simdgroups = the 4 k-splits of ONE column half) instead of 256 (8 simdgroups, both halves):
    the grid x is (2 * blocks) threadgroups, x >> 1 the 128-column block, x & 1 the half. Every simdgroup runs the very same tile chain and the 4 splits reduce in the same order j = 0..3,
    so the partials are bit for bit those of the 256-thread launch; only the threadgroups are finer (shorter tail wave). Launch with threadgroup (128, 1, 1) and the same grid."""
    for a, b in _T128_SUBS:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    return src


def gather_gemv_source(Ks: tuple[int, ...], nproj: int, cb: str = "mul1", pf: bool = False, ksplit: bool = False, pf2: bool = False, t128: bool = False) -> str:
    """Gather GEMV partial for the bit widths `Ks` (a runtime branch on the expert's K, uniform per threadgroup).
    Inputs: xh [NZ, KD] f32, ids [npairs] i32, trellis{0,1} u32, off{0,1} u32 [E+1], kk{0,1} u8 [E], params [KT, NT, TGK].
    Output part [NZ, TGK, N] f32 (NZ = npairs * nproj).
    pf / pf2 (KYOJIN_GEMV_PF2): one / two k tiles of the pair's trellis in flight ahead of the decode (loads reordered only, same bits).
    ksplit (KYOJIN_MOE_KSPLIT): the k-split of an expert follows its bitrate. params[4 + i] = active k-blocks TGA of the i-th bitrate of `Ks`; a threadgroup with
    kblock >= TGA stores zeros for its slice of `part` and ends (mid / final still add TGK partials, 0 + x is exact), the others split the k range in TGA."""
    if nproj == 1:
        trsel = "const device uint* trellis = tr0; const device uint* off = off0; const device uint8_t* kk = kk0;"
    else:
        trsel = ("const bool pj = (zz % 2u) != 0u; const device uint* trellis = pj ? tr1 : tr0; "
                 "const device uint* off = pj ? off1 : off0; const device uint8_t* kk = pj ? kk1 : kk0;")
    xdecl = [f"float4 xq0_{q} = xp[{q}];" for q in range(4)]
    xdecl += [f"float x0_{kk} = xq0_{kk // 4}.{'xyzw'[kk % 4]};" for kk in range(16)]
    af = af_fold(cb)
    if af:
        xdecl.append("sx0 += (xq0_0 + xq0_1) + (xq0_2 + xq0_3);")
    afstore = (f"{{ const float bs = {af_consts(cb)[1]} * ((sx0.x + sx0.y) + (sx0.z + sx0.w)); a0_lo0 = fma(a0_lo0, {af_consts(cb)[0]}, bs); a0_hi0 = fma(a0_hi0, {af_consts(cb)[0]}, bs); }}"
               if af else "")

    def emit(p, kk, hi, e):
        h = "hi" if hi else "lo"
        return f"{{ auto d = {e}; a0_{h}0 = fmad(d, x0_{kk}, a0_{h}0); }}"

    bodies = []
    for K in Ks:
        loads, unpack = gen_loads(K)
        load_lines = [f"{t} {v} = {e};" for v, t, e in loads] + unpack
        run = run_decode_lines(K, emit, cb)
        if pf:
            decl = " ".join(f"{t} n_{v};" for v, t, _ in loads)
            nxt = " ".join(f"n_{v} = {e};" for v, _, e in loads)
            cur = "\n            ".join(f"{t} {v} = n_{v};" for v, t, _ in loads)
            if pf2:
                decl2 = " ".join(f"{t} z_{v};" for v, t, _ in loads)
                nxt2 = " ".join(f"z_{v} = {e};" for v, _, e in gen_loads(K, "t2")[0])
                cur = "\n            ".join(f"{t} {v} = n_{v};" for v, t, _ in loads)
                cur += "\n            " + "\n            ".join(f"n_{v} = z_{v};" for v, _, _ in loads)
                body = (_GATHER_BODY_PF2.replace("DECL_NEXT", decl).replace("DECL_2", decl2).replace("LOAD_NEXT", nxt)
                        .replace("LOAD_2", nxt2).replace("COPY_CUR", cur + "\n            " + "\n            ".join(unpack)))
            else:
                body = _GATHER_BODY_PF.replace("DECL_NEXT", decl).replace("LOAD_NEXT", nxt).replace("COPY_CUR", cur + "\n            " + "\n            ".join(unpack))
        else:
            body = _GATHER_BODY.replace("LOAD_WORDS", "\n            ".join(load_lines))
        bodies.append(body.replace("X_DECL", "\n            ".join(xdecl)).replace("RUN_CODE", "\n            ".join(run)).replace("KV", str(K)))
    chain = " else ".join(f"if (K == {K}u) {b}" for K, b in zip(Ks, bodies))
    src = _GATHER_GEMV.replace("NPROJ", str(nproj)).replace("TRSEL", trsel).replace("BODIES", chain).replace("AFSTORE", afstore)
    assert not (t128 and ksplit)
    return ksplit_gather(src, Ks) if ksplit else (gather_half_tg(src) if t128 else src)


def ksplit_sel(Ks: tuple[int, ...]) -> str:
    """Metal expression: the active k-block count of the expert's bitrate K (params[4 + i], i = position of K in `Ks`; the last bitrate is the else branch)."""
    sel = f"params[{4 + len(Ks) - 1}]"
    for i in range(len(Ks) - 2, -1, -1):
        sel = f"(K == {Ks[i]}u ? params[{4 + i}] : {sel})"
    return sel


def ksplit_gather(src: str, Ks: tuple[int, ...]) -> str:
    """The gather GEMV with a per-bitrate k-split (see gather_gemv_source)."""
    old = "    const uint S = TGK * WS;\n"
    assert src.count(old) == 1, "gather template changed"
    new = (f"    const uint TGA = {ksplit_sel(Ks)};\n"
           "    if (kblock >= TGA) {                                    // beyond this bitrate's split: a zero partial, the slice is still summed by mid / final\n"
           "        if (tid < 128) part[(ulong(zz) * TGK + kblock) * N + blk * 128 + tid] = 0.0f;\n"
           "        return;\n"
           "    }\n"
           "    const uint S = TGA * WS;\n")
    return src.replace(old, new)


_UNION_GEMV = r"""
    constexpr uint WS = 4;                                  // k-splits per threadgroup (x 2 column halves)
    threadgroup float red[WS][2][32][2];                    // one row at a time: threadgroup memory (occupancy) does not grow with MVMAX
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;        // 128-column output block
    const uint kblock = threadgroup_position_in_grid.y;
    const uint zz = threadgroup_position_in_grid.z;         // pair * NPROJ + proj
    const uint KT = params[0], NT = params[1], TGK = params[2], NPAIRS = params[3];
    const uint KD = KT * 16, N = NT * 16;
    const uint p = zz / NPROJ;
    const uint pj = zz - p * NPROJ;
    const int eid = ids[p];
    // Members = every pair routed to this expert (at most one per row). The lowest pair leads and does all the work: the
    // trellis of the expert is decoded once and applied to the inputs of all members; the other threadgroups end here.
    uint mem[MVMAX];
    mem[0] = p;
    uint m = 1;
    for (uint q = 0; q < NPAIRS; ++q) {
        if (q != p && ids[q] == eid) {
            if (q < p) return;
            if (m < MVMAX) mem[m++] = q;
        }
    }
    TRSEL
    const uint e = uint(eid);
    const uint K = uint(kk[e]);
    const device uint* tbase = trellis + off[e];
    const uint half_ = warp & 1u, split = warp >> 1;
    const uint g = lane & 7;
    const uint nt = blk * 8 + half_ * 4 + (lane >> 3);

    const uint S = TGK * WS;
    const uint chunk = (KT + S - 1) / S;
    const uint ks0 = (kblock * WS + split) * chunk;
    const uint myn = ks0 < KT ? min(chunk, KT - ks0) : 0u;
    BODIES
"""

_UNION_BODY = r"""{
        constexpr uint TW = 8 * KV;
        const uint wprev = (KV * g + TW - 1) % TW;
        const device uint* tp = tbase + (ulong(ks0) * NT + nt) * TW;
        const ulong tstride = ulong(NT) * TW;
        ACC_DECL
        XP_DECL
        for (uint i = 0; i < myn; ++i) {
            LOAD_WORDS
            X_DECL
            RUN_CODE
            tp += tstride;
            XP_ADV
        }
        STORE_RED
    }"""


def union_gemv_source(Ks: tuple[int, ...], nproj: int, cb: str = "mul1", mmax: int = 4) -> str:
    """Union GEMV partial: like gather_gemv_source, but pairs routed to the same expert share one decode of its trellis.
    The leader threadgroup (lowest pair index of the expert) accumulates one row per member (1, 2, 4 or 8 accumulators, picked
    at runtime from the member count); the other threadgroups of the expert return at once. Same inputs and output layout as the
    gather kernel plus params[3] = number of pairs; per-pair partials are bit-identical to it (same arithmetic, same order)."""
    assert NACC == 1 and mmax in (1, 2, 4, 8)
    variants = [v for v in (1, 2, 4, 8) if v <= mmax]
    if nproj == 1:
        trsel = "const device uint* trellis = tr0; const device uint* off = off0; const device uint8_t* kk = kk0;"
    else:
        trsel = ("const bool pj1 = pj != 0u; const device uint* trellis = pj1 ? tr1 : tr0; "
                 "const device uint* off = pj1 ? off1 : off0; const device uint8_t* kk = pj1 ? kk1 : kk0;")

    def body(K: int, mv: int) -> str:
        loads, unpack = gen_loads(K)
        load_lines = [f"{t} {v} = {e};" for v, t, e in loads] + unpack
        acc = "\n        ".join(f"float a{r}_lo0 = 0.0f, a{r}_hi0 = 0.0f;" for r in range(mv))
        xpd = "\n        ".join(f"const device float4* xp{r} = (const device float4*)(xh + ulong((" + (f"{r} < m ? mem[{r}] : mem[0]" if r else "mem[0]")
                                 + f") * NPROJ + pj) * KD + ks0 * 16);" for r in range(mv))
        xdecl = []
        for r in range(mv):
            xdecl += [f"float4 xq{r}_{q} = xp{r}[{q}];" for q in range(4)] if mv == 1 else []
            xdecl += [f"float x{r}_{kk} = xq{r}_{kk // 4}.{'xyzw'[kk % 4]};" for kk in range(16)] if mv == 1 else []

        def emit(p, kk, hi, e):
            h = "hi" if hi else "lo"
            if mv == 1:
                fm = f"a0_{h}0 = fmad(d, x0_{kk}, a0_{h}0);"
            else:                                                # multi-row: load x on demand (fewer live registers)
                fm = " ".join(f"a{r}_{h}0 = fmad(d, xp{r}[{kk // 4}].{'xyzw'[kk % 4]}, a{r}_{h}0);" for r in range(mv))
            return f"{{ auto d = {e}; {fm} }}"

        run = run_decode_lines(K, emit, cb)
        store = "\n        ".join(
            f"if (m > {r}u) {{ red[split][half_][lane][0] = a{r}_lo0; red[split][half_][lane][1] = a{r}_hi0;\n"
            "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "            for (uint idx = tid; idx < 128; idx += 256) {\n"
            "                const uint c = idx, h = c >> 6, l = ((c & 63) >> 4) * 8 + (c & 7), hl = (c >> 3) & 1;\n"
            "                float v = 0.0f;\n"
            "                for (uint j = 0; j < WS; ++j) v += red[j][h][l][hl];\n"
            f"                part[(ulong(mem[{r}] * NPROJ + pj) * TGK + kblock) * N + blk * 128 + c] = v;\n"
            "            }\n"
            "            threadgroup_barrier(mem_flags::mem_threadgroup); }"
            for r in range(mv))
        return (_UNION_BODY.replace("ACC_DECL", acc).replace("XP_DECL", xpd).replace("LOAD_WORDS", "\n            ".join(load_lines))
                .replace("X_DECL", "\n            ".join(xdecl)).replace("RUN_CODE", "\n            ".join(run))
                .replace("XP_ADV", " ".join(f"xp{r} += 4;" for r in range(mv))).replace("STORE_RED", store)
                .replace("MV", str(mv)).replace("KV", str(K)))

    def per_k(K: int) -> str:
        parts = []
        for i, mv in enumerate(variants):
            cond = f"m <= {mv}u" if i < len(variants) - 1 else "true"
            parts.append(f"if ({cond}) {body(K, mv)}")
        return " else ".join(parts)

    chain = " else ".join(f"if (K == {K}u) {{ {per_k(K)} }}" for K in Ks)
    return (_UNION_GEMV.replace("BODIES", chain).replace("NPROJ", str(nproj)).replace("MVMAX", str(variants[-1])).replace("TRSEL", trsel))


# ---------------------------------------------------------------------------------- grouped experts (several rows per expert)
# Pairs (row, expert) are grouped by expert on the GPU: group_plan lists, per expert read, its member pairs (ascending pair index), in five
# classes by member count k (GROUP_CLASSES). grouped_gemv decodes the trellis of each listed expert ONCE per pass and applies it to the
# rows of the class (1, 2, 3, 4 or 8 member rows per pass, x in registers like the dense row-group kernel); k > 8 loops over passes.
# 3:3 has its own class: a 3-member group in the 4-row pass paid for a dead row (served MTP3 +1.3 %, identical text, ab20). A k == 1 group is
# the gather kernel's arithmetic (same fma order), so every row's partial is bit-identical to the per-row path.

GROUP_CLASSES = tuple((int(a), int(b)) for a, b in (c.split(":") for c in os.environ.get("KYOJIN_GCLASSES", "1:1,2:2,3:3,4:4,5:8").split(",")))      # (lowest k, rows per pass; the last class loops over passes for more members): a group of k members runs in the class with the highest lowest-k <= k
def group_classes(hotk: int = 0) -> tuple[tuple[int, int], ...]:
    """The classes the GEMV kernel serves: all of GROUP_CLASSES, or those below `hotk` when groups of hotk or more rows go to the NAX GEMM."""
    return tuple(c for c in GROUP_CLASSES if not hotk or c[0] < hotk)


def plan_caps(npairs: int, hotk: int = 0, compact: bool = False) -> tuple[int, ...]:
    """Slot capacity of each GEMV class (a group of k >= lo members takes lo pairs: at most npairs // lo groups). compact: ONE slot range of
    npairs slots (at most one group per pair), the kernel picks the body by the group's member count: no idle slot range per class."""
    if compact:
        assert not hotk, "compact slots: the hot (NAX) groups keep the per-class layout"
        return (npairs,)
    return tuple(npairs // lo for lo, _ in group_classes(hotk))


def hot_cap(npairs: int, hotk: int) -> int:
    """Slots of the NAX groups (k >= hotk): at most npairs // hotk."""
    return npairs // hotk


_GROUP_PLAN = r"""
    // One threadgroup, thread t owns pair t. Outputs: lead[slot] = leader pair (lowest pair of the expert) of each group of the class
    // (class 0 slots first, then class 1, ...; ~0u = empty), cnt_[leader] = members, mem[leader * ROWS + j] = member pairs ascending.
    // HOTK > 0: groups of HOTK or more members are not listed; they are the rows of a NAX GEMM: hrow[pair] = row of the pair when the hot
    // groups are laid out one after the other (~0u for the others), heids[l] = expert of hot group l, roff_gu / roff_d = row offsets of
    // the groups for the gate + up launch (up rows after gate rows) and the down launch, hinfo[0] = hot rows.
    const uint t = thread_position_in_threadgroup.x;
    const uint NP = params[0], ROWS = params[1];
    threadgroup uint cls_[256];
    threadgroup uint pos_[256];
    threadgroup uint cntv[256];
    threadgroup uint pre_[HOTCAP1];
    uint cnt = 0, rank = 0, first = t;
    int e = -1;
    if (t < NP) {
        e = ids[t];
        bool seen = false;
        for (uint q = 0; q < NP; ++q) if (ids[q] == e) { if (!seen) { first = q; seen = true; } cnt += 1; if (q < t) rank += 1; }
    }
    const bool lead_ = t < NP && rank == 0;
    uint c = NCLS + 1;                                    // not a leader
    if (lead_) {
        c = 0; for (uint k = 1; k < NCLS; ++k) if (cnt >= LOWK[k]) c = k;
        HOTCLASS
    }
    cls_[t] = c;
    cntv[t] = cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pos = 0;
    if (lead_) {
        for (uint q = 0; q < t; ++q) pos += (cls_[q] == c) ? 1u : 0u;
        if (c < NCLS) {
            cnt_[t] = cnt;
            uint j = 0;
            for (uint q = 0; q < NP; ++q) if (ids[q] == e) { mem[t * ROWS + j] = q; j += 1; }
        }
    }
    pos_[t] = pos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0;
    for (uint k = 0; k < NCLS; ++k) {
        const uint cap = NP / LOWK[k];
        if (t < cap) {
            uint v = 0xffffffffu;
            for (uint q = 0; q < NP; ++q) if (cls_[q] == k && pos_[q] == t) v = q;
            lead[base + t] = v;
        }
        base += cap;
    }
    HOTCODE
"""

# Compact grouping (KYOJIN_MOEGRP): the slots of all member counts share ONE range of NP slots, groups packed in ascending leader order, and the
# GEMV picks a slot's body by its member count. The per-class layout (above) reserves NP // lo slots per class for the worst case and the kernel
# picks the body by the slot's range: with 24 pairs (verify R = 4) the five class ranges hold 54 slots of which about 17 are filled, so more than
# half of the launched threadgroups read the plan and return. Same groups, same bodies, same per-member FMAs: bit-identical.
_GROUP_PLAN_COMPACT = r"""
    // One threadgroup, thread t owns pair t. Outputs: lead[slot] = leader pair (lowest pair of its expert) of the slot-th group (~0u = empty),
    // cnt_[leader] = members, mem[leader * ROWS + j] = member pairs ascending.
    const uint t = thread_position_in_threadgroup.x;
    const uint NP = params[0], ROWS = params[1];
    threadgroup uint lf[256];
    uint cnt = 0, rank = 0;
    int e = -1;
    if (t < NP) {
        e = ids[t];
        for (uint q = 0; q < NP; ++q) if (ids[q] == e) { cnt += 1; if (q < t) rank += 1; }
    }
    const bool lead_ = t < NP && rank == 0;
    lf[t] = lead_ ? 1u : 0u;
    if (t < NP) lead[t] = 0xffffffffu;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lead_) {
        uint pg = 0;
        for (uint q = 0; q < t; ++q) pg += lf[q];
        lead[pg] = t;
        cnt_[t] = cnt;
        uint j = 0;
        for (uint q = 0; q < NP; ++q) if (ids[q] == e) { mem[t * ROWS + j] = q; j += 1; }
    }
"""

# Wide plan (pf_tail_grp): the same per-class slot layout as _GROUP_PLAN for up to NPMAX pairs (one CED decoder-tail MoE pass: 128 rows x top-6 =
# 768 pairs), one threadgroup of 256 threads, each owning the pairs tid, tid + 256, ... The passes are the small kernel's (count + rank over all
# pairs, class position, slot fill); only the per-thread pair loop is new, so a pair's slot is the one the 256-pair kernel would give it.
_GROUP_PLAN_BIG = r"""
    // One threadgroup, thread tid owns pairs tid, tid + 256, ... Outputs as _GROUP_PLAN (one class range per member-count class).
    const uint tid = thread_position_in_threadgroup.x;
    const uint NP = params[0], ROWS = params[1];
    threadgroup uint cls_[NPMAX];
    threadgroup uint pos_[NPMAX];
    for (uint p = tid; p < NP; p += 256u) {
        const int e = ids[p];
        uint cnt = 0, rank = 0;
        for (uint q = 0; q < NP; ++q) if (ids[q] == e) { cnt += 1; if (q < p) rank += 1; }
        uint c = NCLS + 1;                                 // not a leader
        if (rank == 0) {
            c = 0; for (uint k = 1; k < NCLS; ++k) if (cnt >= LOWK[k]) c = k;
            cnt_[p] = cnt;
            uint j = 0;
            for (uint q = 0; q < NP; ++q) if (ids[q] == e) mem[p * ROWS + j++] = q;
        }
        cls_[p] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < NP; p += 256u) {
        uint pos = 0;
        if (cls_[p] < NCLS) for (uint q = 0; q < p; ++q) pos += (cls_[q] == cls_[p]) ? 1u : 0u;
        pos_[p] = pos;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0;
    for (uint k = 0; k < NCLS; ++k) {
        const uint cap = NP / LOWK[k];
        for (uint s = tid; s < cap; s += 256u) {
            uint v = 0xffffffffu;
            for (uint q = 0; q < NP; ++q) if (cls_[q] == k && pos_[q] == s) v = q;
            lead[base + s] = v;
        }
        base += cap;
    }
"""

_HOT_PLAN = r"""
    if (t <= GH) {                                        // rows before hot group t
        uint s = 0;
        for (uint q = 0; q < NP; ++q) if (cls_[q] == NCLS && pos_[q] < t) s += cntv[q];
        pre_[t] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < NP) hrow[t] = (cls_[first] == NCLS) ? pre_[pos_[first]] + rank : 0xffffffffu;
    if (t < GH) {
        int he = 0;
        for (uint q = 0; q < NP; ++q) if (cls_[q] == NCLS && pos_[q] == t) he = ids[q];
        heids[t] = he;
        roff_gu[t] = pre_[t];
        roff_gu[GH + t] = pre_[GH] + pre_[t];
        roff_d[t] = pre_[t];
    }
    if (t == 0) { roff_gu[2 * GH] = 2 * pre_[GH]; roff_d[GH] = pre_[GH]; hinfo[0] = pre_[GH]; }
"""


@functools.lru_cache(maxsize=None)
def group_plan_source(hotk: int = 0, compact: bool = False, big: bool = False, np_max: int = 1024) -> str:
    """Inputs: ids [NP] i32, params u32 [NP, ROWS]. Outputs: lead u32 [sum(plan_caps(NP))], cnt_ u32 [NP], mem u32 [NP * ROWS]; with hotk also
    hrow u32 [NP], heids i32 [GH], roff_gu u32 [2 * GH + 1], roff_d u32 [GH + 1], hinfo u32 [1] (GH = hot_cap(NP, hotk)). NP <= 256.
    compact: one slot range of NP slots, groups packed in ascending leader order; the GEMV picks each slot's body by its member count.
    big: NP up to np_max on one threadgroup of 256 threads (each owns NP / 256 pairs), same per-class layout as the 256-pair kernel (pf_tail_grp)."""
    if compact:
        assert not hotk, "compact slots: the hot (NAX) groups keep the per-class layout"
        return _GROUP_PLAN_COMPACT
    cl = group_classes(hotk)
    lowk = "constexpr uint LOWK[" + str(len(cl)) + "] = {" + ", ".join(f"{lo}u" for lo, _ in cl) + "};\n"
    src = (_GROUP_PLAN_BIG if big else _GROUP_PLAN).replace("NCLS", str(len(cl)))
    if hotk:
        assert not big, "the wide plan has no NAX hot groups"
        src = (src.replace("HOTCLASS", f"if (cnt >= {hotk}u) c = {len(cl)}u;").replace("HOTCODE", _HOT_PLAN.replace("NCLS", str(len(cl))))
               .replace("HOTCAP1", str(256 // hotk + 1)))
        src = src.replace("GH", f"GH_").replace("GH_", "(NP / " + str(hotk) + "u)")
    else:
        src = src.replace("HOTCLASS", "").replace("HOTCODE", "").replace("HOTCAP1", "1")
    if big:
        src = src.replace("NPMAX", str(np_max))
    return lowk + src


_GROUPED_GEMV = r"""
    constexpr uint WS = 4;                                  // k-splits per threadgroup (x 2 column halves)
    const uint lane = thread_index_in_simdgroup;
    const uint warp = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;        // 128-column output block
    const uint kblock = threadgroup_position_in_grid.y;
    const uint zz = threadgroup_position_in_grid.z;         // slot * NPROJ + proj
    const uint KT = params[0], NT = params[1], TGK = params[2], NP = params[3], ROWS = params[4];
    const uint KD = KT * 16, N = NT * 16;
    const uint slot = zz / NPROJ;
    const uint pj = zz - slot * NPROJ;
    const uint lp = lead[slot];
    if (lp == 0xffffffffu) return;
    const uint cnt = (slot < NP) ? 1u : cnt_[lp];                         // class 0 (the first NP slots) holds the single-member groups
    const device uint* mm = mem + ulong(lp) * ROWS;
    const int eid = ids[lp];
    TRSEL
    const uint e = uint(eid);
    const uint K = uint(kk[e]);
    PREVCOPY
    const device uint* tbase = trellis + off[e];
    const uint half_ = warp & 1u, split = warp >> 1;
    const uint g = lane & 7;
    const uint nt = blk * 8 + half_ * 4 + (lane >> 3);

    const uint S = TGK * WS;
    const uint chunk = (KT + S - 1) / S;
    const uint ks0 = (kblock * WS + split) * chunk;
    const uint myn = ks0 < KT ? min(chunk, KT - ks0) : 0u;
    RED_DECL
    uint cbase[NCLS + 1];
    CBASE
    BODIES
"""

_GROUPED_BODY = r"""{
        constexpr uint TW = 8 * KV;
        constexpr uint MR = MRV;
        const uint wprev = (KV * g + TW - 1) % TW;
        const device uint* tp0 = tbase + (ulong(ks0) * NT + nt) * TW;
        const ulong tstride = ulong(NT) * TW;
        for (uint c0 = 0; c0 < cnt; c0 += MR) {
            const device uint* tp = tp0;
            XP_DECL
            ACC_DECL
            PRE_LOADS
            for (uint i = 0; i < myn; ++i) {
                LOAD_WORDS
                X_DECL
                RUN_CODE
                tp += tstride;
                XP_ADV
            }
            STORE_RED
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < MR * 128; idx += 256) {
                const uint r = idx >> 7, c = idx & 127;
                if (c0 + r < cnt) {
                    const uint h = c >> 6, l = ((c & 63) >> 4) * 8 + (c & 7), hl = (c >> 3) & 1;
                    float v = 0.0f;
                    for (uint j = 0; j < WS; ++j) v += red[j][h][l][2 * r + hl];
                    part[(ulong(mm[c0 + r] * NPROJ + pj) * TGK + kblock) * N + blk * 128 + c] = v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }"""


_GROUPED_BODY_SG = r"""{
        constexpr uint TW = 8 * KV;
        constexpr uint MR = MRV;
        const uint wprev = (KV * g + TW - 1) % TW;
        const device uint* tp0 = tbase + (ulong(ks0) * NT + nt) * TW;
        const ulong tstride = ulong(NT) * TW;
        for (uint c0 = 0; c0 < cnt; c0 += MR) {
            const device uint* tp = tp0;
            XP_DECL
            ACC_DECL
            PRE_LOADS
            for (uint i = 0; i < myn; ++i) {
                LOAD_WORDS
                X_DECL
                RUN_CODE
                tp += tstride;
                XP_ADV
            }
            AFFINE
            // the simdgroup owns its rows' partial: no threadgroup reduction, no barrier. Split (kblock * WS + split) of the pair at the column the
            // lane would have reduced into: half_ * 64 + (lane >> 3) * 16 + hl * 8 + g, hl 0 = lo, 1 = hi
            SG_STORE
        }
    }"""


def mma_body(K: int, loads, unpack, cb: str) -> str:
    """Grouped GEMV body that decodes every 16 x 64 weight slab of a simdgroup ONCE into threadgroup fp16 (the lane's 32 values as 8 half4, the same decode
    as the FMA path) and runs up to 8 rows through simdgroup 8x8 MMA with fp32 accumulate: C^T[col, row] += W[col, s] * X^T[s, row], s = the staged
    k order of x (half4 groups of positions 4j .. 4j+3, see run_decode_half). The unused row columns repeat the last member (never stored). The weight
    tiles and the partial sums share the red buffer's threadgroup memory."""
    mid, srcs = gen_windows(K)
    load_lines = [f"{t} {v} = {e};" for v, t, e in loads] + unpack + mid
    wr = [f"*((threadgroup half4*)(wt + (wcol + {8 * (m % 2)}u) * 16u + {4 * (m // 2)}u)) = half4("
          + ", ".join(f"dec_{cb}({srcs[4 * m + c]})" for c in range(4)) + ");" for m in range(8)]
    cdecl = " ".join(f"simdgroup_float8x8 C{t} = simdgroup_float8x8(0.0f);" for t in range(8))
    mm = []
    for ks in range(2):
        mm.append(f"simdgroup_half8x8 X{ks}; X{ks}.thread_elements()[0] = xr0[{8 * ks}u + fm]; X{ks}.thread_elements()[1] = xr1[{8 * ks}u + fm];")
        for t in range(8):
            mm.append(f"{{ simdgroup_half8x8 Wm; simdgroup_load(Wm, wt + {t * 8 * 16 + 8 * ks}, 16); simdgroup_multiply_accumulate(C{t}, Wm, X{ks}, C{t}); }}")
    st = " ".join(f"simdgroup_store(C{t}, red2 + ((split * 2u + half_) * 64u + {8 * t}u) * 8u, 8);" for t in range(8))
    return r"""{
        constexpr uint TW = 8 * KV;
        const uint wprev = (KV * g + TW - 1) % TW;
        const device uint* tp0 = tbase + (ulong(ks0) * NT + nt) * TW;
        const ulong tstride = ulong(NT) * TW;
        threadgroup half* wt = (threadgroup half*)smem + warp * 1024u;
        threadgroup float* red2 = (threadgroup float*)smem;
        const uint wcol = (lane >> 3) * 16u + g;
        const uint qid = lane >> 2, fm = (qid & 4u) + ((lane >> 1) & 3u), fn = ((qid & 2u) << 1) + ((lane & 1u) << 1);
        for (uint c0 = 0; c0 < cnt; c0 += 8u) {
            const device half* xr0 = (const device half*)xh + ulong(mm[min(c0 + fn, cnt - 1u)] * NPROJ + pj) * KD + ks0 * 16;
            const device half* xr1 = (const device half*)xh + ulong(mm[min(c0 + fn + 1u, cnt - 1u)] * NPROJ + pj) * KD + ks0 * 16;
            const device uint* tp = tp0;
            CDECL
            for (uint i = 0; i < myn; ++i) {
                LOADS
                WRITES
                simdgroup_barrier(mem_flags::mem_threadgroup);
                MMS
                simdgroup_barrier(mem_flags::mem_threadgroup);
                tp += tstride; xr0 += 16; xr1 += 16;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STORES
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < 8u * 128u; idx += 256u) {
                const uint r = idx >> 7, c = idx & 127u;
                if (c0 + r < cnt) {
                    float v = 0.0f;
                    for (uint j = 0; j < WS; ++j) v += red2[((j * 2u + (c >> 6)) * 64u + (c & 63u)) * 8u + r];
                    part[(ulong(mm[c0 + r] * NPROJ + pj) * TGK + kblock) * N + blk * 128 + c] = v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }""".replace("CDECL", cdecl).replace("LOADS", "\n                ".join(load_lines)).replace("WRITES", "\n                ".join(wr)) \
        .replace("MMS", "\n                ".join(mm)).replace("STORES", st).replace("KV", str(K))


@functools.lru_cache(maxsize=None)
def grouped_gemv_source(Ks: tuple[int, ...], nproj: int, cb: str = "mul1", hotk: int = 0, prev: bool = False, half: bool = False, mma: int = 0,
                        compact: bool = False, sgred: bool = False, pf: int = 0) -> str:
    """Grouped GEMV partial for `Ks` bit widths. Inputs: xh [npairs * nproj, KD] f32, lead / cnt_ / mem (group_plan), ids [NP] i32, trellis{0,1},
    off{0,1}, kk{0,1}, params u32 [KT, NT, TGK, NP, ROWS]. Grid z = (C0 + C1 + C2) * nproj: slots are laid out class by class (plan_caps), a class runs
    its GROUP_CLASSES rows per pass. Output part [npairs * nproj, TGK, N] f32, same layout and bits as gather_gemv_source.
    mma: groups of that many rows or more (with half) run their rows through simdgroup MMA (mma_body); 0 = row FMAs.
    prev: extra input pprev (a part of the same shape); a slot whose expert bitrate is not in Ks copies its members' partials from pprev (per-bitrate
    passes over one plan: pass 1 writes its class, every later pass computes its own class and carries the rest through).
    compact: slots are one packed range (plan_caps compact) and the body of a slot follows the group's member count (1..8 rows per pass), so a group
    of 5..7 members no longer pays the idle rows of the 8-row pass.
    sgred (KYOJIN_MOE_SGRED): each simdgroup stores its own k-split partial straight to `part` (shape [npairs * nproj, TGK * WS, N], split index
    kblock * WS + split), no threadgroup reduction buffer and no barrier; the consumers sum TGK * WS partials (their params carry TGK * WS). Same
    products and per-lane sums, the four-way sum moves from the kernel into the consumer: fp32 summation order only. Not with the MMA body.
    pf (KYOJIN_MOE_GPF): that many k tiles of trellis words are loaded ahead of the tile being decoded (1 or 2), so the HBM latency of a tile is
    covered by the decode of the previous one; the same words are read in the same order, only earlier: bit-identical."""
    assert NACC == 1 and not (half and af_fold(cb))
    assert not (sgred and half and mma)
    if nproj == 1:
        trsel = "const device uint* trellis = tr0; const device uint* off = off0; const device uint8_t* kk = kk0;"
    else:
        trsel = ("const bool pj1 = pj != 0u; const device uint* trellis = pj1 ? tr1 : tr0; "
                 "const device uint* off = pj1 ? off1 : off0; const device uint8_t* kk = pj1 ? kk1 : kk0;")

    def body(K: int, mrv: int) -> str:
        loads, unpack = gen_loads(K)
        if half and mma and mrv >= mma:
            return mma_body(K, loads, unpack, cb)
        pre_lines = []
        if pf and not (half and mma and mrv >= mma):
            nxt = gen_loads(K, "tn")[0]
            pre_lines = [f"{t} n{v} = {t}(0);" for v, t, _ in loads]
            if pf >= 2:
                pre_lines += [f"{t} z{v} = {t}(0);" for v, t, _ in loads]
            pre_lines.append("if (myn > 0) { " + " ".join(f"n{v} = {e};" for v, _, e in loads) + " }")
            if pf >= 2:
                pre_lines.append("if (myn > 1) { const device uint* tn = tp + tstride; " + " ".join(f"z{v} = {e};" for v, _, e in nxt) + " }")
            load_lines = [f"{t} {v} = n{v};" for v, t, _ in loads]
            if pf >= 2:
                load_lines += [f"n{v} = z{v};" for v, t, _ in loads]
                load_lines.append("if (i + 2 < myn) { const device uint* tn = tp + 2 * tstride; "
                                  + " ".join(f"z{v} = {e};" for v, _, e in nxt) + " }")
            else:
                load_lines.append("if (i + 1 < myn) { const device uint* tn = tp + tstride; "
                                  + " ".join(f"n{v} = {e};" for v, _, e in nxt) + " }")
            load_lines += unpack
        else:
            load_lines = [f"{t} {v} = {e};" for v, t, e in loads] + unpack
        xt = "half4" if half else "float4"
        xp = "\n            ".join(
            f"const device {xt}* xp{r} = (const device {xt}*)(xh + ulong(mm[min(c0 + {r}u, cnt - 1u)] * NPROJ + pj) * KD + ks0 * 16);" for r in range(mrv))
        xdecl = []
        hfma = half and half_on(mrv)                                    # half-staged x; classes below HALF_ROWS widen it back and FMA in fp32
        for r in range(mrv):
            if hfma:
                xdecl += [f"half4 xh{r}_{q} = xp{r}[{q}];" for q in range(4)]
                continue
            if half:
                xdecl += [f"half4 xs{r}_{q} = xp{r}[{q}];" for q in range(4)]
                pos = lambda k: 4 * (k >> 1) + (k & 1) if k < 8 else 4 * ((k - 8) >> 1) + 2 + (k & 1)
                xdecl += [f"float x{r}_{k} = float(xs{r}_{pos(k) // 4}.{'xyzw'[pos(k) % 4]});" for k in range(16)]
                continue
            xdecl += [f"float4 xq{r}_{q} = xp{r}[{q}];" for q in range(4)]
            xdecl += [f"float x{r}_{k} = xq{r}_{k // 4}.{'xyzw'[k % 4]};" for k in range(16)]
        acc = "\n            ".join(f"float a{r}_lo0 = 0.0f, a{r}_hi0 = 0.0f;" + (f" half4 hacc{r}_lo = half4(0), hacc{r}_hi = half4(0);" if hfma else "") for r in range(mrv))
        store = "\n            ".join(f"red[split][half_][lane][{2 * r}] = a{r}_lo0; red[split][half_][lane][{2 * r + 1}] = a{r}_hi0;" for r in range(mrv))
        affine = ""
        if af_fold(cb):
            acc += "\n            " + " ".join(f"float4 sx{r} = float4(0.0f);" for r in range(mrv))
            xdecl += [f"sx{r} += (xq{r}_0 + xq{r}_1) + (xq{r}_2 + xq{r}_3);" for r in range(mrv)]
            affine = " ".join(f"{{ const float bs = {af_consts(cb)[1]} * ((sx{r}.x + sx{r}.y) + (sx{r}.z + sx{r}.w)); a{r}_lo0 = fma(a{r}_lo0, {af_consts(cb)[0]}, bs); "
                              f"a{r}_hi0 = fma(a{r}_hi0, {af_consts(cb)[0]}, bs); }}" for r in range(mrv))
            store = affine + "\n            " + store

        def emit(p, k, hi, e):
            h = "hi" if hi else "lo"
            return "{ auto d = " + e + "; " + " ".join(f"a{r}_{h}0 = fmad(d, x{r}_{k}, a{r}_{h}0);" for r in range(mrv)) + " }"

        run = run_decode_half(K, mrv, cb) if hfma else run_decode_lines(K, emit, cb)
        if sgred:
            sgs = " ".join(f"if (c0 + {r}u < cnt) {{ device float* po{r} = part + (ulong(mm[c0 + {r}u] * NPROJ + pj) * (TGK * WS) + kblock * WS + split) * N + blk * 128 + half_ * 64 + (lane >> 3) * 16 + g; "
                            f"po{r}[0] = a{r}_lo0; po{r}[8] = a{r}_hi0; }}" for r in range(mrv))
            return (_GROUPED_BODY_SG.replace("XP_DECL", xp).replace("ACC_DECL", acc).replace("LOAD_WORDS", "\n                ".join(load_lines))
                    .replace("X_DECL", "\n                ".join(xdecl)).replace("RUN_CODE", "\n                ".join(run))
                    .replace("XP_ADV", " ".join(f"xp{r} += 4;" for r in range(mrv))).replace("AFFINE", affine).replace("SG_STORE", sgs).replace("PRE_LOADS", "\n            ".join(pre_lines))
                    .replace("MRV", str(mrv)).replace("KV", str(K)))
        return (_GROUPED_BODY.replace("XP_DECL", xp).replace("ACC_DECL", acc).replace("LOAD_WORDS", "\n                ".join(load_lines))
                .replace("X_DECL", "\n                ".join(xdecl)).replace("RUN_CODE", "\n                ".join(run))
                .replace("XP_ADV", " ".join(f"xp{r} += 4;" for r in range(mrv))).replace("STORE_RED", store).replace("PRE_LOADS", "\n            ".join(pre_lines))
                .replace("MRV", str(mrv)).replace("KV", str(K)))

    def per_class(mrv: int) -> str:
        return " else ".join(f"if (K == {K}u) {body(K, mrv)}" for K in Ks)

    cl = group_classes(hotk)
    mrs = list(range(1, 9)) if compact else [m for _, m in cl]          # compact: one body per member count, the slot's body follows cnt
    use_mma = bool(half and mma and any(m >= mma for m in mrs))
    red_decl = "" if sgred else (f"threadgroup uchar smem[{max(16384, 2 * 4 * 32 * 2 * max(mrs) * 4)}];\n    threadgroup float (*red)[2][32][{2 * max(mrs)}] = (threadgroup float (*)[2][32][{2 * max(mrs)}])smem;"
                if use_mma else f"threadgroup float red[WS][2][32][{2 * max(mrs)}];")
    bounds = " ".join(f"cbase[{c + 1}] = cbase[{c}] + NP / {lo}u;" for c, (lo, _) in enumerate(cl))
    if compact:
        cnt_decl = "const uint cnt = cnt_[lp];"
        chain = " else ".join([f"if (cnt <= {m}u) {{ {per_class(m)} }}" for m in mrs[:-1]] + [f"{{ {per_class(mrs[-1])} }}"])
    else:
        cnt_decl = "const uint cnt = (slot < NP) ? 1u : cnt_[lp];"
        chain = " else ".join([f"if (slot < cbase[{c + 1}]) {{ {per_class(m)} }}" for c, m in enumerate(mrs[:-1])] + [f"{{ {per_class(mrs[-1])} }}"])
    miss = " && ".join(f"K != {K}u" for K in Ks)
    pc = (f"if ({miss}) {{ for (uint idx = tid; idx < cnt * 128u; idx += 256u) {{ const ulong o = (ulong(mm[idx >> 7] * NPROJ + pj) * TGK + kblock) * N + blk * 128 + (idx & 127u); "
          "part[o] = pprev[o]; } return; }") if prev else ""
    if prev and sgred:                                                  # TGK * WS partials per pair: copy the four splits of this k-block
        pc = (f"if ({miss}) {{ for (uint idx = tid; idx < cnt * 128u * WS; idx += 256u) {{ const uint j = idx / (cnt * 128u), q = idx - j * (cnt * 128u); "
              "const ulong o = ((ulong(mm[q >> 7] * NPROJ + pj) * TGK + kblock) * WS + j) * N + blk * 128 + (q & 127u); part[o] = pprev[o]; } return; }")
    return (_GROUPED_GEMV.replace("PREVCOPY", pc).replace("BODIES", chain).replace("NPROJ", str(nproj)).replace("MRMAX", str(max(mrs))).replace("TRSEL", trsel)
            .replace("NCLS", str(len(cl))).replace("CBASE", "cbase[0] = 0u; " + bounds).replace("RED_DECL", red_decl)
            .replace("const uint cnt = (slot < NP) ? 1u : cnt_[lp];", cnt_decl))


# ------------------------------------------------------------- NAX variants of rot / mid / final for the hot groups (k >= hotk members)
# The rows of a hot group go through the fused trellis-decode + NAX GEMM (metal_nax): its activations are half with the k permutation inside the
# 16-blocks (perm_store), rows laid out group after group (plan hrow). Pairs of other groups keep the fp32 GEMV layout.

def perm_store(ptr: str, val: str, half: bool) -> str:
    """Store of the lane's float4 `val` (elements 4 * lane .. 4 * lane + 3 of a 128-block that starts at `ptr`) with the k permutation
    inside every 16-block that the fused EXL3 + NAX kernels (metal_nax) expect: stored position 8 * (q & 1) + 4 * (c >> 1) + 2 * (q >> 1) + (c & 1)
    holds element 4 * q + c of the 16-block (q = lane & 3). The values are the same, only their order inside the 16-block changes."""
    ot = "half" if half else "float"
    return (f"{{ const float4 pv = {val}; device {ot}* pd = (device {ot}*)({ptr}) + (lane >> 2) * 16 + 8 * (lane & 1) + 2 * ((lane >> 1) & 1);\n"
            f"      pd[0] = {ot}(pv.x); pd[1] = {ot}(pv.y); pd[4] = {ot}(pv.z); pd[5] = {ot}(pv.w); }}")


@functools.lru_cache(maxsize=None)
def rot_hot_source(in_half: bool) -> str:
    """rot_gather (gate + up) for the grouped path with hot groups. Inputs: x, ids, suh0, suh1, hrow u32 [NP], hinfo u32 [1], params [KD, DIV].
    Outputs: xh f32 [2 * NP, KD] (pairs outside hot groups), A half [2 * NP, KD] (hot pairs: row pj * hot_rows + hrow, k permuted)."""
    s = _ROT_GATHER
    old = "*((device float4*)(xh + ulong(zz) * KD + blk * 128 + lane * 4)) = float4(v0, v1, v2, v3) * RSQRT;"
    assert s.count(old) == 1
    s = s.replace(old, "const uint hr = hrow[p];\n    if (hr == 0xffffffffu) { " + old + " }\n    else " +
                  perm_store("A + ulong((zz - p * 2u) * hinfo[0] + hr) * KD + blk * 128", "float4(v0, v1, v2, v3) * RSQRT", True))
    return (s.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD).replace("NPROJ", "2")
            .replace("SUHSEL", "((zz % 2u) ? suh1 : suh0)").replace("XV4", "half4" if in_half else "float4"))


@functools.lru_cache(maxsize=None)
def mid_hot_source(limit: float, wpre: bool, fq: bool, alpha: float = 1.0, beta: float = 0.0) -> str:
    """mid for the grouped path with hot groups. Inputs: part (GEMV partials), yn f32 [2 * hot_rows, N] (NAX gate + up), ids, wts, svh0, svh1, suhd,
    hrow, hinfo, params [TGK, N]. Outputs: xh2 f32 [NP, N] (pairs outside hot groups), A2 half [NP, N] (hot pairs at row hrow, k permuted)."""
    s = mid_source(limit, wpre, fq, alpha, beta)
    old_e = "const uint e = uint(ids[p]);"
    assert s.count(old_e) == 1
    s = s.replace(old_e, old_e + "\n    const uint hr = hrow[p];")
    old_a = "for (uint kb = 0; kb < TGK; ++kb) acc += *((const device float4*)(part + (ulong(p * 2 + pj) * TGK + kb) * N + col));"
    assert s.count(old_a) == 1
    s = s.replace(old_a, "if (hr != 0xffffffffu) acc = *((const device float4*)(yn + (ulong(pj) * hinfo[0] + hr) * N + col));\n        else " + old_a)
    old_s = "*((device float4*)(xh2 + ulong(p) * N + col)) = float4(v0, v1, v2, v3) * " + RSQRT_HAD + ";"
    assert s.count(old_s) == 1
    return s.replace(old_s, "if (hr == 0xffffffffu) { " + old_s + " }\n    else " + perm_store("A2 + ulong(hr) * N + blk * 128", "float4(v0, v1, v2, v3) * " + RSQRT_HAD, True))


@functools.lru_cache(maxsize=None)
def final_hot_source(wpost: bool) -> str:
    """final_gather for the grouped path with hot groups: inputs part, yd f32 [hot_rows, N] (NAX down), ids, wts, svh, hrow, params [TGK, N, TOPK]."""
    s = final_gather_source(wpost)
    old = "for (uint kb = 0; kb < TGK; ++kb) acc += *((const device float4*)(part + (ulong(p) * TGK + kb) * N + col));"
    assert s.count(old) == 1
    return s.replace(old, "const uint hr = hrow[p];\n        if (hr != 0xffffffffu) acc = *((const device float4*)(yd + ulong(hr) * N + col));\n        else " + old)


FQ_FP8 = r"""
    // FP8 e4m3 round trip with ue8m0 (power of two) scales, blocks of 32 = 8 lanes x 4 values (DS4.1 activation fake quant).
    { const float4 hq = float4(half4(a4));                       // activations are fp16 at this point
      float m = max(max(abs(hq.x), abs(hq.y)), max(abs(hq.z), abs(hq.w)));
      m = max(m, simd_shuffle_xor(m, 1)); m = max(m, simd_shuffle_xor(m, 2)); m = max(m, simd_shuffle_xor(m, 4));
      const float amax = max(m, 1e-4f);
      const uint sb = as_type<uint>(amax * (1.0f / 448.0f));
      const uint se = (sb >> 23) + (((sb & 0x7fffffu) != 0u) ? 1u : 0u);
      const float scale = as_type<float>(se << 23);
      float ov[4]; const float hv[4] = {hq.x, hq.y, hq.z, hq.w};
      for (int i = 0; i < 4; ++i) {
          const float v = clamp(hv[i] / scale, -448.0f, 448.0f);
          const uint vb = as_type<uint>(abs(v));
          const uint qe = uint(max(int(vb >> 23) - 3, 118));
          const float step = as_type<float>(qe << 23);
          ov[i] = float(half((rint(v / step) * step) * scale));
      }
      a4 = float4(ov[0], ov[1], ov[2], ov[3]); }
"""

_MID = r"""
    // act = silu(min(gate, L)) * clip(up, -L, L) [* w]  [fp8 fake quant]; xh2[p] = H128(act * suh_down[e]) / sqrt(128).
    // gate / up = H128(sum of partials) / sqrt(128) * svh, as in the finalize kernel. One simdgroup per (128-block, pair).
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint p = threadgroup_position_in_grid.y;
    const uint TGK = params[0], N = params[1];
    if (blk * 128 >= N) return;
    const uint e = uint(ids[p]);
    const ulong col = blk * 128 + lane * 4;
    float4 gu[2];
    for (uint pj = 0; pj < 2; ++pj) {
        float4 acc = float4(0.0f);
        for (uint kb = 0; kb < TGK; ++kb) acc += *((const device float4*)(part + (ulong(p * 2 + pj) * TGK + kb) * N + col));
        float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
        const float4 sv = float4(*((const device half4*)((pj ? svh1 : svh0) + ulong(e) * N + col)));
        gu[pj] = float4(v0, v1, v2, v3) * RSQRT * sv;
    }
    float4 g4 = min(gu[0], float4(LIMIT)), u4 = clamp(gu[1], float4(-LIMIT), float4(LIMIT));
    float4 sg;
    for (int i = 0; i < 4; ++i) {                                    // mlx sigmoid: y = 1 / (1 + exp(|x|)); x < 0 ? y : 1 - y
        const float x = g4[i] ALPHAMUL;
        const float y = 1.0f / (1.0f + metal::precise::exp(abs(x)));
        sg[i] = (x < 0.0f) ? y : 1.0f - y;
    }
    float4 a4 = (g4 * sg) * UBETA;
    WPRE
    FQ
    const float4 sd = float4(*((const device half4*)(suhd + ulong(e) * N + col)));
    float v0 = a4.x * sd.x, v1 = a4.y * sd.y, v2 = a4.z * sd.z, v3 = a4.w * sd.w;
BUTTERFLY
    *((device float4*)(xh2 + ulong(p) * N + col)) = float4(v0, v1, v2, v3) * RSQRT;
"""


@functools.lru_cache(maxsize=None)
def mid_source(limit: float, wpre: bool, fq: bool, alpha: float = 1.0, beta: float = 0.0) -> str:
    """act = min(g, L) * sigmoid(alpha * min(g, L)) * (clip(u, -L, L) + beta): silu-gated (alpha 1, beta 0) or swigluoai."""
    return (_MID.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD).replace("LIMIT", repr(float(limit)) + "f")
            .replace("ALPHAMUL", f"* {float(alpha)!r}f" if alpha != 1.0 else "")
            .replace("UBETA", f"(u4 + {float(beta)!r}f)" if beta != 0.0 else "u4")
            .replace("WPRE", "a4 = a4 * wts[p];" if wpre else "").replace("FQ", FQ_FP8 if fq else ""))


@functools.lru_cache(maxsize=None)
def mid_half_source(limit: float, wpre: bool, fq: bool, alpha: float = 1.0, beta: float = 0.0) -> str:
    """mid with xh2 stored as half in the run's k order (see rot_gather_half_source)."""
    s = mid_source(limit, wpre, fq, alpha, beta)
    old = "*((device float4*)(xh2 + ulong(p) * N + col)) = float4(v0, v1, v2, v3) * " + RSQRT_HAD + ";"
    assert s.count(old) == 1
    return s.replace(old, perm_store("xh2 + ulong(p) * N + blk * 128", "float4(v0, v1, v2, v3) * " + RSQRT_HAD, True))


_FINAL_GATHER = r"""
    // y[r] = sum_j [w_j *] H128(sum_kb part[r*TOPK + j]) / sqrt(128) * svh[ids[r*TOPK + j]]  (fp32, j in order)
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y;
    const uint TGK = params[0], N = params[1], TOPK = params[2];
    if (blk * 128 >= N) return;
    const ulong col = blk * 128 + lane * 4;
    float4 out = float4(0.0f);
    for (uint j = 0; j < TOPK; ++j) {
        const uint p = r * TOPK + j;
        const uint e = uint(ids[p]);
        float4 acc = float4(0.0f);
        for (uint kb = 0; kb < TGK; ++kb) acc += *((const device float4*)(part + (ulong(p) * TGK + kb) * N + col));
        float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
        const float4 sv = float4(*((const device half4*)(svh + ulong(e) * N + col)));
        float4 y4 = float4(v0, v1, v2, v3) * RSQRT * sv;
        WPOST
        out = out + y4;
    }
    *((device float4*)(y + ulong(r) * N + col)) = out;
"""


@functools.lru_cache(maxsize=None)
def final_gather_source(wpost: bool) -> str:
    return (_FINAL_GATHER.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD)
            .replace("WPOST", "y4 = y4 * wts[p];" if wpost else ""))


_FINAL_PAR = r"""
    // Same sum as final_gather, the pair terms computed side by side: one simdgroup per pair j of a threadgroup (NJ simdgroups, one 128-block of one row), the terms
    // are added in the order j = 0.. by simdgroup 0 (final_gather's serial loop adds the same terms in the same order: bit-identical).
    const uint lane = thread_index_in_simdgroup;
    const uint j = simdgroup_index_in_threadgroup;
    const uint blk = threadgroup_position_in_grid.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint TGK = params[0], N = params[1], TOPK = params[2];
    threadgroup float4 ys[NJ][32];
    const ulong col = blk * 128 + lane * 4;
    {
        const uint p = r * TOPK + j;
        const uint e = uint(ids[p]);
        float4 acc = float4(0.0f);
        for (uint kb = 0; kb < TGK; ++kb) acc += *((const device float4*)(part + (ulong(p) * TGK + kb) * N + col));
        float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
        const float4 sv = float4(*((const device half4*)(svh + ulong(e) * N + col)));
        float4 y4 = float4(v0, v1, v2, v3) * RSQRT * sv;
        WPOST
        BF16LAST
        ys[j][lane] = y4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0) {
        float4 out = float4(0.0f);
        for (uint q = 0; q < TOPK; ++q) out = out + ys[q][lane];
        *((device float4*)(y + ulong(r) * N + col)) = out;
    }
"""


def final_par_grid(blocks: int, rows: int, nj: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(grid, threadgroup) of final_par_source: one threadgroup of nj simdgroups per 128-block and row."""
    return (blocks * 32 * nj, rows, 1), (32 * nj, 1, 1)


@functools.lru_cache(maxsize=None)
def final_par_source(wpost: bool, nj: int, bf16_last: bool = False) -> str:
    """KYOJIN_MOE_FINPAR: final_gather with the pairs side by side (nj = pairs per row = params[2]); bf16_last: the last slot (shared expert) is rounded to bf16 first, as final_gather_sh."""
    return (_FINAL_PAR.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD).replace("NJ", str(nj))
            .replace("WPOST", "y4 = y4 * wts[p];" if wpost else "").replace("BF16LAST", _BF16_ROUND if bf16_last else ""))


# ---------------------------------------------------------------------------------------------------------------- shared expert in the routed launches (sumoe)
# The shared expert rides the routed MoE launches as one more pair per row: slot TOPK of each row owns the expert ESH (the last expert of the stacks, where the
# shared expert's trellis and Hadamard signs were appended), routing weight 1. The rot launch builds the extended ids / weights (no extra launch). The shared slot
# skips the fp8 fake quant of the down input and its down projection is rounded to bf16 before it joins the fp32 sum, as the stock chain rounds it.
# Idea after Sushi's `moeDecodeClampedSharedFused` (10 launches a layer become 5): Portions adapted from beamivalice/sushi (MIT / Apache-2.0),
# src/exl3/rate_groups.zig@0f4f00b8a166fe8dc67bca05b34c732df7c7d176.
@functools.lru_cache(maxsize=None)
def rot_gather_sh_source(nproj: int, in_half: bool, fq8: bool = False) -> str:
    """rot_gather over the extended pairs (p = row * (TOPK + 1) + j, j == TOPK is the shared slot). params: KD, DIV = TOPK + 1, TOPK, ESH.
    Also writes ids2 / wts2 (the extended routing) for the launches that follow."""
    s = rot_gather_source(nproj, in_half, fq8)
    old = "const uint KD = params[0], DIV = params[1];"
    assert s.count(old) == 1
    s = s.replace(old, "const uint KD = params[0], DIV = params[1], TKR = params[2], ESH = params[3];")
    old = "const uint p = zz / NPROJ;\n    const uint e = uint(ids[p]);".replace("NPROJ", str(nproj))
    assert s.count(old) == 1, "rot template changed"
    new = ("const uint p = zz / NPROJ;\n    const uint rr = p / DIV, jj = p - rr * DIV;\n"
           "    const uint e = (jj == TKR) ? ESH : uint(ids[rr * TKR + jj]);\n"
           "    if (blk == 0 && lane == 0 && (zz % NPROJ) == 0) { ids2[p] = int(e); wts2[p] = (jj == TKR) ? 1.0f : wts[rr * TKR + jj]; }").replace("NPROJ", str(nproj))
    return s.replace(old, new)


@functools.lru_cache(maxsize=None)
def mid_sh_source(limit: float, wpre: bool, fq: bool, alpha: float = 1.0, beta: float = 0.0) -> str:
    """mid with the fp8 fake quant skipped for the shared expert (e == params[2])."""
    s = mid_source(limit, wpre, fq, alpha, beta)
    if not fq:
        return s
    assert s.count(FQ_FP8) == 1, "mid template changed"
    return s.replace(FQ_FP8, "    if (e != params[2]) {\n" + FQ_FP8 + "\n    }\n")


_BF16_ROUND = r"""
        if (j == TOPK - 1) {                                     // shared slot: bf16 output of the shared down projection
            uint4 ub = as_type<uint4>(y4);
            ub = (ub + uint4(0x7FFFu) + ((ub >> 16) & 1u)) & uint4(0xFFFF0000u);
            y4 = as_type<float4>(ub);
        }"""


@functools.lru_cache(maxsize=None)
def final_gather_sh_source(wpost: bool) -> str:
    """final_gather over TOPK = topk + 1 slots, the last one (shared) rounded to bf16 before the sum."""
    s = final_gather_source(wpost)
    old = "        out = out + y4;"
    assert s.count(old) == 1
    return s.replace(old, _BF16_ROUND + "\n" + old)


# ---------------------------------------------------------------------------------------------------------------- moe_chain
# KYOJIN_MOE_CHAIN: the routed decode chain in three launches instead of five. mid rides the tail of the gate/up gather GEMV and final (the combine) rides the
# tail of the down gather GEMV: the last threadgroup to finish a 128-column block (arrival counter, reset by that last arriver so the buffer is zero again for
# the next launch) runs the very statements of the stand-alone mid / final on the partials the others stored. Same arithmetic in the same order: bit-identical.
_CH_LDF4 = r"""
#define LDF4(ptr) float4(as_type<float>(atomic_load_explicit((const device atomic_uint*)((ptr)), memory_order_relaxed)), as_type<float>(atomic_load_explicit((const device atomic_uint*)((ptr)) + 1, memory_order_relaxed)), \
                         as_type<float>(atomic_load_explicit((const device atomic_uint*)((ptr)) + 2, memory_order_relaxed)), as_type<float>(atomic_load_explicit((const device atomic_uint*)((ptr)) + 3, memory_order_relaxed)))
"""

_CH_ARRIVE = r"""
    threadgroup_barrier(mem_flags::mem_device);
    const uint chain_tgk = threadgroups_per_grid.y, chain_blocks = threadgroups_per_grid.x;    // both read from the launch itself: nothing about the grid is baked into the source
    threadgroup uint chain_last;
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit(((device atomic_uint*)(ctr)) + CTRI, 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const bool lst = (old + 1u) == ARRIVE;
        if (lst) atomic_store_explicit(((device atomic_uint*)(ctr)) + CTRI, 0u, memory_order_relaxed);
        chain_last = lst ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chain_last == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
"""

_CH_MID_HEAD = ("    const uint lane = thread_index_in_simdgroup;\n"
                "    const uint blk = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;\n"
                "    const uint p = threadgroup_position_in_grid.y;\n"
                "    const uint TGK = params[0], N = params[1];\n"
                "    if (blk * 128 >= N) return;\n"
                "    const uint e = uint(ids[p]);\n")
_CH_PART_LD = "acc += *((const device float4*)(part + (ulong(p * 2 + pj) * TGK + kb) * N + col));"


def _chain_check(gemv: str, src: str) -> None:
    """Refuse a GEMV template the arrival protocol was not verified on: no early `return` may precede the arrival (a threadgroup that leaves without arriving
    would leave its block's counter short), the only returns are the non-last-arriver one in _CH_ARRIVE and the combine's own, and no grid size is baked."""
    assert "return" not in gemv, "gather GEMV template has an early return before the arrival"
    assert src.count("if (chain_last == 0u) return;") == 1 and src.index("atomic_fetch_add") > src.index(gemv[-40:])
    assert "CTRI" not in src and "ARRIVE" not in src


def chain_gu_source(Ks: tuple[int, ...], cb: str, pf: bool, pf2: bool, blocks: int, limit: float, wpre: bool, fq: bool, alpha: float, beta: float,
                    shared_skip: int | None = None, t128: bool = False) -> str:
    """gather_gemv_source (gate + up) with mid in its tail. Extra inputs after the GEMV's: wts, svh0, svh1, suhd, ctr. Outputs part, xh2 [npairs, N].
    shared_skip: expert id whose down input skips the fp8 fake quant (moe_sumoe's shared slot; 0xFFFFFFFF = none) or None for the plain mid."""
    gemv = gather_gemv_source(Ks, 2, cb, pf, False, pf2, t128)
    mid = mid_source(limit, wpre, fq, alpha, beta) if shared_skip is None else mid_sh_source(limit, wpre, fq, alpha, beta)
    assert mid.count(_CH_MID_HEAD) == 1 and mid.count(_CH_PART_LD) == 1, "mid template changed"
    mid = mid.replace(_CH_MID_HEAD, "").replace(_CH_PART_LD, "acc += LDF4(part + (ulong(p * 2 + pj) * TGK + kb) * N + col);")
    if shared_skip is not None:
        mid = mid.replace("params[2]", f"{int(shared_skip)}u")
    arrive = _CH_ARRIVE.replace("CTRI", "(p * chain_blocks + blk)").replace("ARRIVE", "(4u * chain_tgk)" if t128 else "(2u * chain_tgk)")
    if t128:                                                        # two threadgroups (halves) per 128-column block and (pair, proj): 4 * tgk arrivals, the block count is half the grid x
        assert arrive.count("chain_blocks = threadgroups_per_grid.x;") == 1
        arrive = arrive.replace("chain_blocks = threadgroups_per_grid.x;", "chain_blocks = threadgroups_per_grid.x >> 1;")
    out = _CH_LDF4 + gemv + arrive + "    if (warp == 0) {\n" + mid + "    }\n"
    _chain_check(gemv, out)
    return out


def chain_dn_source(Ks: tuple[int, ...], cb: str, pf: bool, pf2: bool, blocks: int, topk: int, wpost: bool, bf16_last: bool = False) -> str:
    """gather_gemv_source (down) with the combine (final_par's arithmetic) in its tail. Extra inputs after the GEMV's: ids is the GEMV's, then wts, svh, ctr.
    Outputs part, y [rows, N]. One simdgroup per pair of the row computes its term, simdgroup 0 adds them in order j."""
    assert topk <= 8
    gemv = gather_gemv_source(Ks, 1, cb, pf, False, pf2)
    arrive = _CH_ARRIVE.replace("CTRI", f"((zz / {topk}u) * chain_blocks + blk)").replace("ARRIVE", f"({topk}u * chain_tgk)")
    tail = r"""
    threadgroup float4 ys[TOPK][32];
    if (warp < TOPK) {
        const uint j = warp;
        const uint r = zz / TOPK;
        const ulong col = blk * 128 + lane * 4;
        const uint pp = r * TOPK + j;
        const uint e2 = uint(ids[pp]);
        float4 acc = float4(0.0f);
        for (uint kb = 0; kb < TGK; ++kb) acc += LDF4(part + (ulong(pp) * TGK + kb) * N + col);
        float v0 = acc.x, v1 = acc.y, v2 = acc.z, v3 = acc.w;
BUTTERFLY
        const float4 sv = float4(*((const device half4*)(svh + ulong(e2) * N + col)));
        float4 y4 = float4(v0, v1, v2, v3) * RSQRT * sv;
        WPOST
        BF16LAST
        ys[j][lane] = y4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        const uint r = zz / TOPK;
        float4 out = float4(0.0f);
        for (uint q = 0; q < TOPK; ++q) out = out + ys[q][lane];
        *((device float4*)(y + ulong(r) * N + blk * 128 + lane * 4)) = out;
    }
"""
    tail = (tail.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD)
            .replace("WPOST", "y4 = y4 * wts[pp];" if wpost else "").replace("BF16LAST", _BF16_ROUND if bf16_last else "").replace("TOPK", str(topk)))
    out = _CH_LDF4 + gemv + arrive + tail
    _chain_check(gemv, out)
    return out


# KYOJIN_ROUTED_FX: the rot launch folded into the prologue of the gate/up gather GEMV. Every threadgroup rotates (sign, 128-Hadamard, fp8 fake quant: the very statements
# of rot_gather_source) the 128-blocks of the k slice its four k-splits read, into threadgroup memory, and the GEMV reads x from there: one launch fewer per layer, same bits.
_FX_PRO = r"""
    threadgroup float4 rfx[RFX_NB * 32];
    const uint rfx_a = kblock * WS * chunk * 16;                    // first k of this threadgroup's slice (a multiple of 128: chunk is even)
RFX_SPLIT    {
        const device half* rsp = ((zz & 1u) ? suh1 : suh0) + ulong(e) * KD;
        for (uint bi = warp; bi < RFX_NB; bi += 8) {
            const uint kb0 = rfx_a + bi * 128;
            if (kb0 >= KD) break;
            const ulong o = ulong(p / RFX_TOPK) * KD + kb0 + lane * 4;
            RFX_XLOAD
            const float4 sv = float4(*((const device half4*)(rsp + kb0 + lane * 4)));
            float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
BUTTERFLY
            rfx[bi * 32 + lane] = float4(v0, v1, v2, v3) * RSQRT;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""


def fx_ok(kt: int, tgk: int, ws: int = 4, cap: int = 24) -> bool:
    """The folded rot needs the threadgroup's k slice to start on a 128-block (even chunk) and to fit `cap` blocks of threadgroup memory."""
    chunk = (kt + tgk * ws - 1) // (tgk * ws)
    return chunk % 2 == 0 and (ws * chunk * 16 + 127) // 128 <= cap


def chain_gu_fx_source(Ks, cb, pf, pf2, blocks, limit, wpre, fq, alpha, beta, shared_skip, kt: int, tgk: int, topk: int, fq8: bool) -> str:
    """chain_gu_source with the rot in the GEMV prologue. Inputs: xh is the fp16 activation [rows, KD] (not rotated), extras after the chain's: suh0, suh1."""
    assert shared_skip is None and fx_ok(kt, tgk)
    chunk = (kt + tgk * 4 - 1) // (tgk * 4)
    nb = min((4 * chunk * 16 + 127) // 128, kt * 16 // 128)
    src = chain_gu_source(Ks, cb, pf, pf2, blocks, limit, wpre, fq, alpha, beta, shared_skip)
    xl = _XV_LOAD.replace("XV4", "half4")
    xl = with_fq8(xl) if fq8 else xl
    assert xl.count("x + o") >= 1
    xl = xl.replace("(x + o)", "(xh + o)")
    assert "(x + o)" not in xl
    pro = (_FX_PRO.replace("BUTTERFLY", _BUTTERFLY).replace("RSQRT", RSQRT_HAD).replace("RFX_XLOAD", xl)
           .replace("RFX_NB", str(nb)).replace("RFX_TOPK", str(topk) + "u"))
    old = "const uint myn = ks0 < KT ? min(chunk, KT - ks0) : 0u;"
    assert src.count(old) == 1
    head, comp = pro.split("RFX_SPLIT")
    if pf and not pf2:                                              # the first tile's trellis words are issued before the rot: their latency hides behind it
        import re
        pat = re.compile(r"if \(myn > 0u\) \{ [^}]*\}")
        src, n = pat.subn(lambda m: m.group(0) + "\n    " + comp, src)
        assert n == len(Ks)
        src = src.replace(old, old + head)
    else:
        src = src.replace(old, old + head + comp)
    old = "const device float4* xp0 = (const device float4*)(xh + ulong(zz) * KD + ks0 * 16);"
    assert src.count(old) == 1
    src = src.replace(old, "const threadgroup float4* xp0 = rfx + ((ks0 * 16 - rfx_a) >> 2);")
    old = "const device float4* xp = xp0;"
    assert src.count(old) == len(Ks)
    return src.replace(old, "const threadgroup float4* xp = xp0;")


