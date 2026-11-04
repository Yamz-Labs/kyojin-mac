# SPDX-License-Identifier: Apache-2.0
"""Metal sources for the foreign 4-bit linears (foreign_pack.py layouts). Pure strings: nothing here imports mlx; foreign_linear.py compiles them
through kernels._kern (the repo's launch path). NOT compiled or timed where this was written (no Metal device): tools/fmt_gate_mac.py --native
checks every kernel against the float64 reference on the Mac and times it against EXL3.

Two kernels per format, the same A-element decode in both:
  gemv   rows <= 4 (any R <= 16 works): the float-FMA kernel of tools/fmt_gate_metal.py (the fastest of the gate), i.e. 16 B units, a simdgroup owns
         2 * NBR output rows, the R x 512 activation chunk sits in threadgroup memory, next chunk's words are prefetched. Needs k % 512 == 0 and
         n % (8 * NBR) == 0. `gather=True`: one routed-expert slot per grid.y (expert id, x row index), the weights of all experts in one array.
  mma    rows 1..16, k % 32 == 0: C^T = W x^T on simdgroup_matrix 8x8 fp32 fragments (the layout used by mma_gemv.py). A lane decodes its own
         fragment elements straight from one 32-bit word of the unit (4 lanes of a row cover the 16 bytes; no shuffle), the matrix is read ONCE for all
         rows (<= 16 = two 8-row B blocks), the activation is pre-permuted on the host (x_perm) so a lane's B elements of a unit are one half4 per row.
         No threadgroup memory, no barrier, a simdgroup owns NTN * 8 output rows and walks the whole k (no k split).
  dequant  w half [n, k] with the same decode (prefill with many rows: one fp16 matmul)."""
from __future__ import annotations
import functools
import os

UNR = "#define UNR _Pragma(\"clang loop unroll(full)\")\n"

HDR = UNR + r"""
inline float fp4v(uint c) {                                   // E2M1 code -> float: 0,.5,1,1.5,2,3,4,6 and the sign bit
  uint v = c & 7u; float f = as_type<float>((v << 22) + 0x3F000000u);
  f = (v < 2u) ? 0.5f * float(v) : f;
  return as_type<float>(as_type<uint>(f) | ((c & 8u) << 28));
}
inline float e4m3f(uint b) {                                  // unsigned E4M3 scale (sign bit ignored); 0x7F (NaN code) -> 0
  b &= 127u; float n = as_type<float>((b << 20) + (120u << 23));
  float s = float(b & 7u) * 0.001953125f;
  return b == 127u ? 0.0f : ((b >> 3) == 0u ? s : n);
}
inline float e8m0f(uint b) { return b == 0u ? 5.8774717541114375e-39f : as_type<float>(b << 23); }
constant float KVC[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};
inline uint k4b(uint4 m, uint k) { uint wd = k < 4u ? m.x : (k < 8u ? m.y : m.z); return (wd >> ((k & 3u) * 8u)) & 255u; }
inline void q4k_sm(uint4 m, uint j, thread float& sc, thread float& mn) {
  if (j < 4u) { sc = float(k4b(m, j) & 63u); mn = float(k4b(m, j + 4u) & 63u); }
  else { sc = float((k4b(m, j + 4u) & 15u) | ((k4b(m, j - 4u) >> 6) << 4)); mn = float((k4b(m, j + 4u) >> 4) | ((k4b(m, j) >> 6) << 4)); }
}
inline void q4k_scales(uint4 m, uint i, thread float& Ad, thread float& Am, thread float& Bd, thread float& Bm) {
  half2 dd = as_type<half2>(m.w); float d = float(dd.x), dm = float(dd.y);
  uint ja = (i >> 1) * 2u; float sa, ma, sb, mb;
  q4k_sm(m, ja, sa, ma); q4k_sm(m, ja + 1u, sb, mb);
  Ad = d * sa; Am = dm * ma; Bd = d * sb; Bm = dm * mb;
}
inline void dec_hl(uint4 q, float Ad, float Am, float Bd, float Bm, thread float* w) {
  UNR for (uint i = 0; i < 4; ++i) { uint wd = q[i];
    UNR for (uint b = 0; b < 4; ++b) { uint by = (wd >> (8u * b)) & 255u;
      w[i * 4u + b] = float(by & 15u) * Ad - Am; w[16u + i * 4u + b] = float(by >> 4) * Bd - Bm; } }
}
"""
FP4V_BITS = r"""inline float fp4v(uint c) {                                   // E2M1 code -> float: 0,.5,1,1.5,2,3,4,6 and the sign bit
  uint v = c & 7u; float f = as_type<float>((v << 22) + 0x3F000000u);
  f = (v < 2u) ? 0.5f * float(v) : f;
  return as_type<float>(as_type<uint>(f) | ((c & 8u) << 28));
}
"""
FP4V_LUT = r"""constant float E2M1T[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float fp4v(uint c) { return E2M1T[c & 15u]; }                // flag cfmtspeed: one table read, the same 16 floats as the bit trick (tests/test_cfmtspeed.py)
"""
assert FP4V_BITS in HDR
HDR_LUT = HDR.replace(FP4V_BITS, FP4V_LUT)
MMA_HDR = HDR + r"""
inline uint cqH(uint wq, uint t, uint e) { return (wq >> (8u * ((t & 1u) * 2u + e) + 4u * (t >> 1))) & 15u; }
inline uint cqN(uint wq, uint t, uint e) { return (wq >> (8u * t + 4u * e)) & 15u; }
inline uint kUnitH(uint u) { return 32u * u; }
inline uint kUnitQ(uint u) { uint u8 = u & 7u; return 256u * (u >> 3) + 64u * (u8 >> 1) + 16u * (u8 & 1u); }
inline uint kOffN(uint t, uint s) { return 8u * (s >> 1) + 2u * t + (s & 1u); }
inline uint kOffH(uint t, uint s) { return 16u * (t >> 1) + 4u * (s >> 1) + 2u * (t & 1u) + (s & 1u); }
inline uint kOffQ(uint t, uint s) { return 32u * (t >> 1) + 4u * (s >> 1) + 2u * (t & 1u) + (s & 1u); }
"""
MMA_HDR_LUT = MMA_HDR.replace(FP4V_BITS, FP4V_LUT)

# meta ctype-free load expressions (n = row, u = unit); MTP is the (possibly expert-offset) meta pointer
_META = {
    "q4_0": "float(MTP[n * UPR + u])", "iq4_nl": "float(MTP[n * UPR + u])",
    "nvfp4": "uint(MTP[n * UPR + u])", "mxfp4": "uint(MTP[n * UPR + u])",
}
_MT = {"q4_0": "float", "iq4_nl": "float", "nvfp4": "uint", "mxfp4": "uint", "q4_k": "uint4"}
_LAYOUT = {"q4_0": "H", "iq4_nl": "H", "q4_k": "Q", "nvfp4": "N", "mxfp4": "N"}
FORMATS = tuple(_MT)
META_DTYPE = {"q4_0": "float16", "iq4_nl": "float16", "nvfp4": "uint16", "mxfp4": "uint8", "q4_k": "uint32"}


def _mload(fmt, nvar="n", uvar="u"):
    if fmt == "q4_k":
        return f"((device const uint4*)MTP)[{nvar} * (UPR / 8u) + ({uvar} >> 3)]"
    return _META[fmt].replace("n * UPR + u", f"{nvar} * UPR + {uvar}")


# ----------------------------------------------------------------------------------------------------------------------------------- GEMV
def _gemv_dec(fmt):
    """statements decoding unit j (qc[j], mc[j], ul) into w[j * 32 .. +31] (element order of the unit as in tools/fmt_gate_metal.py)."""
    if fmt == "q4_0":
        return "dec_hl(qc[j], mc[j], 8.0f * mc[j], mc[j], 8.0f * mc[j], &w[j * 32u]);"
    if fmt == "q4_k":
        return "float Ad_, Am_, Bd_, Bm_; q4k_scales(mc[j], ul & 7u, Ad_, Am_, Bd_, Bm_); dec_hl(qc[j], Ad_, Am_, Bd_, Bm_, &w[j * 32u]);"
    if fmt == "iq4_nl":
        return ("{ float d_ = mc[j]; UNR for (uint i = 0; i < 4; ++i) { uint wd = qc[j][i]; UNR for (uint b = 0; b < 4; ++b) { uint by = (wd >> (8u * b)) & 255u;"
                " w[j * 32u + i * 4u + b] = KVC[by & 15u] * d_; w[j * 32u + 16u + i * 4u + b] = KVC[by >> 4] * d_; } } }")
    sc = ("float s0 = e4m3f(mc[j] & 255u), s1 = e4m3f((mc[j] >> 8) & 255u);" if fmt == "nvfp4" else "float s0 = e8m0f(mc[j]), s1 = s0;")
    # E2M1 code (sign s, magnitude v) -> half bits (s << 15) | (v << 9) = value / 2^14, exact; nibbles k and k + 4 decode together as one half2, the 2^14 rides in the scale
    return ("{ " + sc + " s0 *= 16384.0f; s1 *= 16384.0f; UNR for (uint i = 0; i < 4; ++i) { uint wd = qc[j][i]; const float sx = i < 2u ? s0 : s1;"
            " UNR for (uint k = 0; k < 4; ++k) { uint m = (wd >> (4u * k)) & 0x000F000Fu; float2 f = float2(as_type<half2>(((m & 0x00070007u) << 9) | ((m & 0x00080008u) << 12))) * sx;"
            " w[j * 32u + i * 8u + k] = f.x; w[j * 32u + i * 8u + 4u + k] = f.y; } } }")


def _gemv_xmap(fmt):
    """(xa expr, xb offset): the two 16-runs of the unit's 32 elements inside the 512-element chunk."""
    if fmt == "q4_k":
        return "((ul >> 3) * 256u + ((ul & 7u) >> 1) * 64u + (ul & 1u) * 16u)", 32
    return "(ul * 32u)", 16


def gemv_source(fmt: str, nbr: int = 2, gather: bool = False, rowgs: bool = False, glu: bool = False) -> str:
    """Kernel body. Template ints: R (rows, 1 when gather), KIN, N, NBR (= nbr). Inputs x (half [rows, KIN]), qs (uint32), mt, gs (float [1], or [E] when gather);
    gather adds xrows (uint32 [S]) and eids (uint32 [S]) before qs and writes y half [S, N]. Output y half [R, N].
    rowgs (flag cfmtspeed, same-input groups): gs is float [N], one scale per output row (the members' global scales repeated over their rows), read as
    gs[row] where the plain kernel reads gs[0]; the multiplication is the same float product, so a row's bits do not depend on the group."""
    assert not (rowgs and gather)
    xa, xboff = _gemv_xmap(fmt)
    mt = _MT[fmt]
    mload = _mload(fmt)
    dec = _gemv_dec(fmt)
    if gather:
        setup = r"""
    const uint slot = threadgroup_position_in_grid.y;
    const uint ex = eids[slot];
    device const half* XP = (device const half*)(x + ulong(xrows[slot]) * uint(KIN));
    device half* YP = y + ulong(slot) * uint(N);
    device const uint* QSP = qs + ulong(ex) * (ulong(N) * uint(KIN) / 8u);
    auto MTP = mt + ulong(ex) * MT_PER_EXPERT;
    const float GS = gs[ex];
"""
    else:
        setup = r"""
    device const half* XP = (device const half*)x;
    device half* YP = y;
    device const uint* QSP = qs;
    auto MTP = mt;
    const float GS = gs[0];
"""
    setup = setup.replace("MT_PER_EXPERT", "(ulong(N) * (uint(KIN) / 256u) * 4u)" if fmt == "q4_k" else "(ulong(N) * (uint(KIN) / 32u))")
    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tgi = threadgroup_position_in_grid.x;
    const uint hh = lane >> 4, ul = lane & 15u;
    const uint rbase = (tgi * 4u + sg) * (2u * NBR) + hh * NBR;
    const uint UPR = uint(KIN) / 32u, NC = uint(KIN) / 512u;
{setup}
    device const uint4* q4 = (device const uint4*)QSP;
    threadgroup uint4 xs_v[R * 80]; threadgroup half* xs = (threadgroup half*)xs_v;
    float w[NBR * 32];
    float acc[NBR * R];
    UNR for (uint i = 0; i < NBR * R; ++i) acc[i] = 0.0f;
    uint4 qc[NBR], qn[NBR]; {mt} mc[NBR], mn_[NBR];
    UNR for (uint j = 0; j < NBR; ++j) {{ const uint n = rbase + j, u = ul; qc[j] = q4[n * UPR + u]; mc[j] = {mload}; }}
    for (uint c = 0; c < NC; ++c) {{
      for (uint i = tid; i < uint(R) * 64u; i += 128u) {{ uint r = i >> 6, j = i & 63u;
        uint4 v = *(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u);
        *(threadgroup uint4*)(xs + r * 640u + j * 8u + (j >> 2) * 8u) = v; }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (c + 1u < NC) {{
        UNR for (uint j = 0; j < NBR; ++j) {{ const uint n = rbase + j, u = (c + 1u) * 16u + ul; qn[j] = q4[n * UPR + u]; mn_[j] = {mload}; }}
      }}
      const uint xa = {xa}, xb = xa + {xboff}u;
      UNR for (uint j = 0; j < NBR; ++j) {{ {dec} }}
      const uint pa = xa + (xa >> 5) * 8u, pb = xb + (xb >> 5) * 8u;
      UNR for (uint r = 0; r < R; ++r) {{
        threadgroup const half* xr = xs + r * 640u;
        float xv[32];
        UNR for (uint t = 0; t < 4; ++t) {{ float4 va = float4(*(threadgroup const half4*)(xr + pa + 4u * t)); float4 vb = float4(*(threadgroup const half4*)(xr + pb + 4u * t));
          UNR for (uint k = 0; k < 4; ++k) {{ xv[4u * t + k] = va[k]; xv[16u + 4u * t + k] = vb[k]; }} }}
        UNR for (uint j = 0; j < NBR; ++j) {{ float s = 0.0f;
          UNR for (uint e = 0; e < 32; ++e) s += w[j * 32u + e] * xv[e];
          acc[j * R + r] += s; }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      UNR for (uint j = 0; j < NBR; ++j) {{ qc[j] = qn[j]; mc[j] = mn_[j]; }}
    }}
    UNR for (uint j = 0; j < NBR; ++j) {{
      UNR for (uint r = 0; r < R; ++r) {{
        float v = acc[j * R + r];
        v += simd_shuffle_xor(v, 8u); v += simd_shuffle_xor(v, 4u); v += simd_shuffle_xor(v, 2u); v += simd_shuffle_xor(v, 1u);
        if (ul == 0u) YP[r * uint(N) + rbase + j] = half(v * GS);
      }}
    }}
"""
    if glu:      # dense down projection of a gated MLP (GLU_STAGE): the staged input is silu(x) * x2
        s_old = "uint4 v = *(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u);"
        assert not gather and src.count(s_old) == 1
        src = src.replace(s_old, GLU_STAGE)
    return _row_gs(src, "half(v * GS)", "half(v * gs[rbase + j])") if rowgs else src


def _row_gs(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"template changed, {old!r} found {src.count(old)} times"
    return src.replace(old, new)


def gemv_inputs(gather=False, glu=False):
    if glu and not gather:
        return ["x", "x2", "qs", "mt", "gs"]
    return ["x", "xrows", "eids", "qs", "mt", "gs"] if gather else ["x", "qs", "mt", "gs"]


# ----------------------------------------------------------------------------------------------------------------------------------- A-element decode (mma, dequant)
def _a_decode(fmt):
    """Declares float a[4][2] = this lane's weights of unit u of row rn (wq: its 32-bit word, c: lane column, UPR, MTP in scope)."""
    L = _LAYOUT[fmt]
    cq = "cqN" if L == "N" else "cqH"
    pre, expr = "", ""
    if fmt == "q4_0":
        pre = f"const float d_ = {_mload(fmt, 'rn')};"
        expr = "(float(q) - 8.0f) * d_"
    elif fmt == "iq4_nl":
        pre = f"const float d_ = {_mload(fmt, 'rn')};"
        expr = "KVC[q] * d_"
    elif fmt == "q4_k":
        pre = (f"const uint4 m_ = {_mload(fmt, 'rn')}; float Ad_, Am_, Bd_, Bm_; q4k_scales(m_, u & 7u, Ad_, Am_, Bd_, Bm_);")
        expr = "(t < 2u) ? (float(q) * Ad_ - Am_) : (float(q) * Bd_ - Bm_)"
    elif fmt == "nvfp4":
        pre = f"const uint m_ = {_mload(fmt, 'rn')}; const float sc_ = e4m3f((m_ >> (8u * (c >> 1))) & 255u);"
        expr = "fp4v(q) * sc_"
    else:
        pre = f"const float sc_ = e8m0f({_mload(fmt, 'rn')});"
        expr = "fp4v(q) * sc_"
    return (f"{pre}\n      float a[4][2];\n      UNR for (uint t = 0; t < 4; ++t) UNR for (uint e = 0; e < 2; ++e) {{ const uint q = {cq}(wq, t, e); a[t][e] = {expr}; }}")


def mma_source(fmt: str, ntn: int = 1, rowgs: bool = False) -> str:
    """Body. Template ints: R (1..16), KIN, N, NTN (= ntn). Inputs xp (half [R, KIN], the x_perm of foreign_pack.xperm_index), qs, mt, gs. Output y half [R, N].
    Grid: ceil(N / (8 NTN * 4)) threadgroups of 128 threads. rowgs: gs float [N], read as gs[nn] (see gemv_source)."""
    dec = _a_decode(fmt)
    setup = "    auto MTP = mt;\n    const float GS = gs[0];\n"
    src = f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n0 = ((threadgroup_position_in_grid.x * 4u + sg) * uint(NTN)) * 8u;
    if (n0 >= uint(N)) return;
    constexpr uint RB = (uint(R) + 7u) / 8u;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    const uint UPR = uint(KIN) / 32u;
{setup}
    uint rows_n[NTN];
    UNR for (uint j = 0; j < uint(NTN); ++j) rows_n[j] = min(n0 + 8u * j + fm, uint(N) - 1u);
    simdgroup_matrix<float, 8, 8> acc[NTN][RB];
    UNR for (uint j = 0; j < uint(NTN); ++j) UNR for (uint rb = 0; rb < RB; ++rb) acc[j][rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    device const uint* q32 = (device const uint*)qs;
    for (uint u = 0; u < UPR; ++u) {{
      simdgroup_matrix<float, 8, 8> Bm[4][RB];
      UNR for (uint rb = 0; rb < RB; ++rb) {{
        const uint r0 = min(8u * rb + fn, uint(R) - 1u), r1 = min(8u * rb + fn + 1u, uint(R) - 1u);
        const half4 b0 = *(device const half4*)(xp + r0 * uint(KIN) + u * 32u + fm * 4u);
        const half4 b1 = *(device const half4*)(xp + r1 * uint(KIN) + u * 32u + fm * 4u);
        UNR for (uint t = 0; t < 4; ++t) {{ Bm[t][rb].thread_elements()[0] = float(b0[t]); Bm[t][rb].thread_elements()[1] = float(b1[t]); }}
      }}
      UNR for (uint j = 0; j < uint(NTN); ++j) {{
        const uint rn = rows_n[j];
        const uint wq = q32[rn * (uint(KIN) / 8u) + u * 4u + c];
        {dec}
        UNR for (uint t = 0; t < 4; ++t) {{
          simdgroup_matrix<float, 8, 8> A;
          A.thread_elements()[0] = a[t][0]; A.thread_elements()[1] = a[t][1];
          UNR for (uint rb = 0; rb < RB; ++rb) simdgroup_multiply_accumulate(acc[j][rb], A, Bm[t][rb], acc[j][rb]);
        }}
      }}
    }}
    UNR for (uint j = 0; j < uint(NTN); ++j) {{
      const uint nn = n0 + 8u * j + fm;
      UNR for (uint rb = 0; rb < RB; ++rb) UNR for (uint el = 0; el < 2; ++el) {{
        const uint row = 8u * rb + fn + el;
        if (row < uint(R) && nn < uint(N)) y[row * uint(N) + nn] = half(acc[j][rb].thread_elements()[el] * GS);
      }}
    }}
"""
    return _row_gs(src, "thread_elements()[el] * GS)", "thread_elements()[el] * gs[nn])") if rowgs else src


def dequant_source(fmt: str, rowgs: bool = False) -> str:
    """Body. Template ints: KIN, N. Inputs qs, mt, gs; output w half [N, KIN]. Grid N * KIN / 32 * 4 threads (lane word c fastest)."""
    dec = _a_decode(fmt)
    L = _LAYOUT[fmt]
    ku = "kUnitQ(u)" if L == "Q" else "kUnitH(u)"
    ko = {"N": "kOffN", "H": "kOffH", "Q": "kOffQ"}[L]
    src = f"""
    const uint tid = thread_position_in_grid.x;
    const uint c = tid & 3u;
    const uint UPR = uint(KIN) / 32u;
    const uint u = (tid >> 2) % UPR;
    const uint rn = (tid >> 2) / UPR;
    if (rn >= uint(N)) return;
    auto MTP = mt;
    const float GS = gs[0];
    device const uint* q32 = (device const uint*)qs;
    const uint wq = q32[rn * (uint(KIN) / 8u) + u * 4u + c];
    {dec}
    UNR for (uint t = 0; t < 4; ++t) UNR for (uint e = 0; e < 2; ++e)
      w[rn * uint(KIN) + {ku} + {ko}(t, 2u * c + e)] = half(a[t][e] * GS);
"""
    return _row_gs(src, "half(a[t][e] * GS)", "half(a[t][e] * gs[rn])") if rowgs else src


def header(fmt: str, kind: str = "gemv", lut: bool = False) -> str:
    """lut (flag cfmtspeed): the E2M1 decode is a table read; the header text is otherwise today's."""
    if lut:
        return HDR_LUT if kind == "gemv" else MMA_HDR_LUT
    return HDR if kind == "gemv" else MMA_HDR


# ----------------------------------------------------------------------------------------------------------------------------------- 8-bit formats (flag fp8stored)
# fp8  : one E4M3 byte per weight, natural order, no per-unit scale; the per-tensor / per-row scale is gs float [N] (one per output row), applied to the row sum.
# q8_0 : GGUF Q8_0 as stored: int8 per weight, natural order, one float16 scale per 32 weights (mt half [N, K/32]); applied to each 32-block sum. gs float [N] (ones).
# Not part of FORMATS / _MT: the 4-bit kernels above are untouched. Both write half(clamp(v, -65504, 65504)): an overflowing row saturates, never inf (fuse.sat).
FORMATS8 = ("fp8", "q8_0")
META_DTYPE8 = {"fp8": None, "q8_0": "float16"}


def _e4m3_table():
    t = []
    for b in range(256):
        s = -1.0 if b & 128 else 1.0
        v = b & 127
        t.append(0.0 if v == 127 else (s * (v & 7) * 2.0 ** -9 if v >> 3 == 0 else s * (1 + (v & 7) / 8.0) * 2.0 ** ((v >> 3) - 7)))
    return t


E4M3_TABLE = _e4m3_table()                                         # 0x7F / 0xFF (the NaN codes) decode to 0 like the 4-bit UE4M3 scales
HDR8 = UNR + "constant float E4M3T[256] = {" + ",".join(f"{x:.10e}f" for x in E4M3_TABLE) + "};\n"
SAT16 = "clamp({v}, -65504.0f, 65504.0f)"


def header8(fmt: str) -> str:
    return HDR8 if fmt == "fp8" else UNR


def gemv8_inputs(fmt, gather=False, glu=False):
    base = ["x", "qs", "gs"] if fmt == "fp8" else ["x", "qs", "mt", "gs"]
    if glu and not gather:
        return ["x", "x2"] + base[1:]
    return ["x", "xrows", "eids"] + base[1:] if gather else base


@functools.lru_cache(maxsize=None)     # the launch wrappers rebuild the source on every call: 18 us of Python per linear without the cache
def gemv8_source(fmt: str, nbr: int = 2, gather: bool = False, glu: bool = False) -> str:
    """Body. Template ints: R (rows, any 1..16), KIN, N, NBR. Inputs x (half [R, KIN]), qs (uint32 [N, KIN/4]), [mt], gs (float [N]); output y half [R, N].
    The structure of gemv_source: a simdgroup owns 2 NBR output rows, a lane owns 32 consecutive weights of each row per 512-chunk (two 16-byte units), the chunk of x
    sits in threadgroup memory, the next chunk's words are prefetched. Needs KIN % 512 == 0 and N % (8 NBR) == 0.
    gather: as gemvk_source (E stacked matrices as E * N rows, gs float [E * N], slot = grid y, y half [S, N])."""
    q8 = fmt == "q8_0"
    if q8:
        dec = ("float4 d0 = float4(as_type<char4>(wa[i])), d1 = float4(as_type<char4>(wb[i]));"
               " w[j * 32u + i * 4u + 0u] = d0.x; w[j * 32u + i * 4u + 1u] = d0.y; w[j * 32u + i * 4u + 2u] = d0.z; w[j * 32u + i * 4u + 3u] = d0.w;"
               " w[j * 32u + 16u + i * 4u + 0u] = d1.x; w[j * 32u + 16u + i * 4u + 1u] = d1.y; w[j * 32u + 16u + i * 4u + 2u] = d1.z; w[j * 32u + 16u + i * 4u + 3u] = d1.w;")
    else:
        dec = ("UNR for (uint b = 0; b < 4; ++b) { w[j * 32u + i * 4u + b] = E4M3T[(wa[i] >> (8u * b)) & 255u];"
               " w[j * 32u + 16u + i * 4u + b] = E4M3T[(wb[i] >> (8u * b)) & 255u]; }")
    ml0 = "mc[j] = float(mt[n * UPR + ul]);" if q8 else ""
    mln = "mn_[j] = float(mt[n * UPR + (c + 1u) * 16u + ul]);" if q8 else ""
    mcd = "float mc[NBR], mn_[NBR];" if q8 else ""
    mcs = "mc[j] = mn_[j];" if q8 else ""
    fin = "acc[j * R + r] += s * mc[j];" if q8 else "acc[j * R + r] += s;"
    src = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tgi = threadgroup_position_in_grid.x;
    const uint hh = lane >> 4, ul = lane & 15u;
    const uint rbase = (tgi * 4u + sg) * (2u * NBR) + hh * NBR;
    const uint RWD = uint(KIN) / 16u, NC = uint(KIN) / 512u, UPR = uint(KIN) / 32u;
    device const half* XP = (device const half*)x;
    device half* YP = y;
    device const uint4* q4 = (device const uint4*)qs;
    threadgroup uint4 xs_v[R * 80]; threadgroup half* xs = (threadgroup half*)xs_v;
    float w[NBR * 32];
    float acc[NBR * R];
    UNR for (uint i = 0; i < NBR * R; ++i) acc[i] = 0.0f;
    uint4 qa[NBR], qb[NBR], na[NBR], nb[NBR];
    @MCD@
    UNR for (uint j = 0; j < NBR; ++j) { const uint n = rbase + j; qa[j] = q4[n * RWD + ul * 2u]; qb[j] = q4[n * RWD + ul * 2u + 1u]; @ML0@ }
    for (uint c = 0; c < NC; ++c) {
      for (uint i = tid; i < uint(R) * 64u; i += 128u) { uint r = i >> 6, j = i & 63u;
        uint4 v = *(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u);
        *(threadgroup uint4*)(xs + r * 640u + j * 8u + (j >> 2) * 8u) = v; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (c + 1u < NC) {
        UNR for (uint j = 0; j < NBR; ++j) { const uint n = rbase + j, u = (c + 1u) * 32u + ul * 2u; na[j] = q4[n * RWD + u]; nb[j] = q4[n * RWD + u + 1u]; @MLN@ }
      }
      const uint xa = ul * 32u, xb = xa + 16u;
      UNR for (uint j = 0; j < NBR; ++j) {
        UNR for (uint i = 0; i < 4; ++i) { @DEC@ }
      }
      const uint pa = xa + (xa >> 5) * 8u, pb = xb + (xb >> 5) * 8u;
      UNR for (uint r = 0; r < R; ++r) {
        threadgroup const half* xr = xs + r * 640u;
        float xv[32];
        UNR for (uint t = 0; t < 4; ++t) { float4 va = float4(*(threadgroup const half4*)(xr + pa + 4u * t)); float4 vb = float4(*(threadgroup const half4*)(xr + pb + 4u * t));
          UNR for (uint k = 0; k < 4; ++k) { xv[4u * t + k] = va[k]; xv[16u + 4u * t + k] = vb[k]; } }
        UNR for (uint j = 0; j < NBR; ++j) { float s = 0.0f;
          UNR for (uint e = 0; e < 32; ++e) s += w[j * 32u + e] * xv[e];
          @FIN@ }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      UNR for (uint j = 0; j < NBR; ++j) { qa[j] = na[j]; qb[j] = nb[j]; @MCS@ }
    }
    UNR for (uint j = 0; j < NBR; ++j) {
      UNR for (uint r = 0; r < R; ++r) {
        float v = acc[j * R + r];
        v += simd_shuffle_xor(v, 8u); v += simd_shuffle_xor(v, 4u); v += simd_shuffle_xor(v, 2u); v += simd_shuffle_xor(v, 1u);
        if (ul == 0u) YP[r * uint(N) + rbase + j] = half(@SAT@);
      }
    }
"""
    # q8_0 decodes inside the i loop from wa[i] / wb[i]: bind them to the words of unit i
    pre = "const uint4 wa_ = qa[j], wb_ = qb[j]; uint wa[4] = {wa_.x, wa_.y, wa_.z, wa_.w}; uint wb[4] = {wb_.x, wb_.y, wb_.z, wb_.w};"
    src = src.replace("UNR for (uint i = 0; i < 4; ++i) { @DEC@ }", pre + " UNR for (uint i = 0; i < 4; ++i) { @DEC@ }")
    for k, v in (("@MCD@", mcd), ("@ML0@", ml0), ("@MLN@", mln), ("@DEC@", dec), ("@FIN@", fin), ("@MCS@", mcs), ("@SAT@", SAT16.format(v="v * gs[rbase + j]"))):
        src = src.replace(k, v)
    if gather:
        for old, new in (("device const half* XP = (device const half*)x;", "const uint slot = threadgroup_position_in_grid.y; const uint ex = eids[slot];\n"
                          "    device const half* XP = (device const half*)(x + ulong(xrows[slot]) * uint(KIN)); const uint EXO = ex * uint(N);"),
                         ("device half* YP = y;", "device half* YP = y + ulong(slot) * uint(N);"),
                         ("const uint n = rbase + j", "const uint n = EXO + rbase + j"), ("gs[rbase + j]", "gs[EXO + rbase + j]")):
            assert old in src, old
            src = src.replace(old, new)
    if glu and not gather:
        s_old = "uint4 v = *(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u);"
        assert src.count(s_old) == 1
        src = src.replace(s_old, GLU_STAGE)
    return src


@functools.lru_cache(maxsize=None)
def dequant8_source(fmt: str) -> str:
    """Body. Template ints: KIN, N. Inputs qs (uint32 [N, KIN/4]), [mt], gs (float [N]); output w half [N, KIN]. One thread per 4 weights."""
    sc = "float(mt[row * (uint(KIN) / 32u) + (col4 >> 3)]) * gs[row]" if fmt == "q8_0" else "gs[row]"
    if fmt == "q8_0":
        body = "float4 d_ = float4(as_type<char4>(wd));"
    else:
        body = "float4 d_ = float4(E4M3T[wd & 255u], E4M3T[(wd >> 8) & 255u], E4M3T[(wd >> 16) & 255u], E4M3T[wd >> 24]);"
    return f"""
    const ulong tid = thread_position_in_grid.x;
    if (tid >= ulong(N) * (uint(KIN) / 4u)) return;
    const uint row = uint(tid / (uint(KIN) / 4u)), col4 = uint(tid % (uint(KIN) / 4u));
    const uint wd = ((device const uint*)qs)[tid];
    {body}
    const float s_ = {sc};
    UNR for (uint b = 0; b < 4; ++b) w[tid * 4u + b] = half({SAT16.format(v="d_[b] * s_")});
"""


def xperm8_index(k: int):
    """Host gather for mma8_source: position u*32 + fm*4 + t of a unit holds x[u*32 + 8*(fm>>1) + 2*t + (fm&1)] (the k of tile t, B row fm; see mma8_source)."""
    import numpy as np
    p = np.arange(32)
    fm, t = p // 4, p % 4
    return (np.arange(k // 32)[:, None] * 32 + (8 * (fm >> 1) + 2 * t + (fm & 1))[None, :]).astype(np.int32).reshape(-1)


@functools.lru_cache(maxsize=None)
def mma8_source(fmt: str, ntn: int = 1, half_a: bool = False) -> str:
    """Simdgroup-matrix kernel of the 8-bit formats. Template ints: R (1..32), KIN, N, NTN. Inputs xp (half [R, KIN], x gathered by xperm8_index), qs (uint32 [N, KIN/4]),
    [mt (q8_0)], gs (float [N]). Output y half [R, N] (saturated). A = 8 weight rows x 8 k, B = 8 k x 8 activation rows. A unit is 32 weights; lane (fm, c) holds the 8
    bytes c*8..c*8+7 of row fm: tile t, element e is byte 2t+e, so tile t contracts over k = 8c + 2t + e (A column 2c+e). half_a: A and B are half (exact for E4M3 and int8),
    the float accumulator is scaled once per unit for q8_0."""
    q8 = fmt == "q8_0"
    T = "half" if half_a else "float"
    if q8:
        dec = (f"{T}2 a_[4]; UNR for (uint t = 0; t < 4; ++t) {{ const uint wv = (t < 2u) ? w2.x : w2.y; const uint sh = 16u * (t & 1u);"
               f" a_[t] = {T}2({T}(char((wv >> sh) & 255u)), {T}(char((wv >> (sh + 8u)) & 255u))); }}")
    else:
        dec = (f"{T}2 a_[4]; UNR for (uint t = 0; t < 4; ++t) {{ const uint wv = (t < 2u) ? w2.x : w2.y; const uint sh = 16u * (t & 1u);"
               f" a_[t] = {T}2({T}(E4M3T[(wv >> sh) & 255u]), {T}(E4M3T[(wv >> (sh + 8u)) & 255u])); }}")
    scale_a = "" if (q8 and half_a) or not q8 else "const float d_ = float(mt[rn * UPR + u]);"
    if q8 and not half_a:
        dec += " UNR for (uint t = 0; t < 4; ++t) a_[t] *= d_;"
    mma = f"""
        UNR for (uint t = 0; t < 4; ++t) {{
          simdgroup_matrix<{T}, 8, 8> A;
          A.thread_elements()[0] = a_[t].x; A.thread_elements()[1] = a_[t].y;
          UNR for (uint rb = 0; rb < RB; ++rb) simdgroup_multiply_accumulate({'tmp[rb]' if (q8 and half_a) else 'acc[j][rb]'}, A, Bm[t][rb], {'tmp[rb]' if (q8 and half_a) else 'acc[j][rb]'});
        }}"""
    if q8 and half_a:
        mma = ("        simdgroup_matrix<float, 8, 8> tmp[RB];\n        UNR for (uint rb = 0; rb < RB; ++rb) tmp[rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);"
               + mma + "\n        const float d_ = float(mt[rn * UPR + u]);\n        UNR for (uint rb = 0; rb < RB; ++rb) { acc[j][rb].thread_elements()[0] += d_ * tmp[rb].thread_elements()[0]; acc[j][rb].thread_elements()[1] += d_ * tmp[rb].thread_elements()[1]; }")
    return f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n0 = ((threadgroup_position_in_grid.x * 4u + sg) * uint(NTN)) * 8u;
    if (n0 >= uint(N)) return;
    constexpr uint RB = (uint(R) + 7u) / 8u;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    const uint UPR = uint(KIN) / 32u;
    uint rows_n[NTN];
    UNR for (uint j = 0; j < uint(NTN); ++j) rows_n[j] = min(n0 + 8u * j + fm, uint(N) - 1u);
    simdgroup_matrix<float, 8, 8> acc[NTN][RB];
    UNR for (uint j = 0; j < uint(NTN); ++j) UNR for (uint rb = 0; rb < RB; ++rb) acc[j][rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    device const uint2* q2 = (device const uint2*)qs;
    for (uint u = 0; u < UPR; ++u) {{
      simdgroup_matrix<{T}, 8, 8> Bm[4][RB];
      UNR for (uint rb = 0; rb < RB; ++rb) {{
        const uint r0 = min(8u * rb + fn, uint(R) - 1u), r1 = min(8u * rb + fn + 1u, uint(R) - 1u);
        const half4 b0 = *(device const half4*)(xp + r0 * uint(KIN) + u * 32u + fm * 4u);
        const half4 b1 = *(device const half4*)(xp + r1 * uint(KIN) + u * 32u + fm * 4u);
        UNR for (uint t = 0; t < 4; ++t) {{ Bm[t][rb].thread_elements()[0] = {T}(b0[t]); Bm[t][rb].thread_elements()[1] = {T}(b1[t]); }}
      }}
      UNR for (uint j = 0; j < uint(NTN); ++j) {{
        const uint rn = rows_n[j];
        const uint2 w2 = q2[rn * (uint(KIN) / 8u) + u * 4u + c];
        {scale_a}
        {dec}
{mma}
      }}
    }}
    UNR for (uint j = 0; j < uint(NTN); ++j) {{
      const uint nn = n0 + 8u * j + fm;
      UNR for (uint rb = 0; rb < RB; ++rb) UNR for (uint el = 0; el < 2; ++el) {{
        const uint row = 8u * rb + fn + el;
        if (row < uint(R) && nn < uint(N)) y[row * uint(N) + nn] = half({SAT16.format(v='acc[j][rb].thread_elements()[el] * gs[nn]')});
      }}
    }}
"""


# ----------------------------------------------------------------------------------------------------------------------------------- K formats (flag kstored)
# q5_k, q6_k, iq4_xs: GGUF blocks as stored (foreign_gguf._to_pack_k: same bytes, bit fields split per 32-unit). A unit = 32 weights = one 16-byte nibble word (qs), plus
#   q5_k  : 1 word of high bits (hi, bit e = element e of the unit's decode order), Q4_K scale block (meta uint4 per 256), qs in Q4_K order
#   q6_k  : 2 words of high 2-bit pairs (hi), meta = 5 words per 256 (16 int8 scales, d as half in the low bits of word 4), qs in H order (byte j = element j | j+16 << 4)
#   iq4_xs: no hi, meta = 2 words per 256 (d | scales_h << 16, scales_l), qs in H order, codebook KVC
# The GEMV is the gemv_source frame (a simdgroup owns 2 NBR rows, a lane one 32-unit per 512-chunk); the unit state is the KU struct so the frame is shared.
FORMATSK = ("q5_k", "q6_k", "iq4_xs", "q3_k", "iq3_s", "iq3_xxs", "q2_k", "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m")
FORMATSK_MMA = ("q5_k", "q6_k", "iq4_xs")         # the stored simdgroup-matrix kernel exists for these only (the 3-bit types are GEMV only; a resident fp16 copy serves the rows above)
_LAYOUTK = {"q5_k": "Q", "q6_k": "H", "iq4_xs": "H", "q3_k": "H", "iq3_s": "H", "iq3_xxs": "H", "q2_k": "H", "iq2_xxs": "H", "iq2_xs": "H", "iq2_s": "H", "iq1_s": "H", "iq1_m": "H"}
_K2 = ("q2_k", "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m")      # 2-bit class: qs = 4 or 8 bytes per 32-unit (see foreign_gguf._to_pack_k2)
_NOHI = ("iq4_xs", "q2_k", "iq2_xs", "iq1_s", "iq1_m")      # K formats without a hi array
_K3 = ("q3_k", "iq3_s", "iq3_xxs")      # 3-bit class: qs = 2 words per 32-unit, hi = 1 word per unit (see foreign_gguf._to_pack_k3)
HDRK = HDR + r"""
struct KU { uint4 q; uint4 m; uint2 h; uint d; };
inline uint kuq(uint u) { uint u8 = u & 7u; return 256u * (u >> 3) + 64u * (u8 >> 1) + 16u * (u8 & 1u); }
inline KU ld_q5k(device const uint4* q4, device const uint4* m4, device const uint* hi, uint n, uint u, uint UPR) {
  KU c; c.q = q4[n * UPR + u]; c.m = m4[n * (UPR / 8u) + (u >> 3)]; c.h = uint2(hi[n * UPR + u], 0u); c.d = 0u; return c; }
inline KU ld_q6k(device const uint4* q4, device const uint* mt, device const uint2* h2, uint n, uint u, uint UPR) {
  KU c; c.q = q4[n * UPR + u]; device const uint* mp = mt + (n * (UPR / 8u) + (u >> 3)) * 5u; c.m = uint4(mp[0], mp[1], mp[2], mp[3]); c.d = mp[4]; c.h = h2[n * UPR + u]; return c; }
inline KU ld_iqxs(device const uint4* q4, device const uint2* m2, uint n, uint u, uint UPR) {
  KU c; c.q = q4[n * UPR + u]; uint2 m = m2[n * (UPR / 8u) + (u >> 3)]; c.m = uint4(m.x, m.y, 0u, 0u); c.h = uint2(0u); c.d = 0u; return c; }
inline void dec_q5k(KU c, uint u, thread float* w) {
  float Ad, Am, Bd, Bm; q4k_scales(c.m, u & 7u, Ad, Am, Bd, Bm);
  UNR for (uint i = 0; i < 4; ++i) { uint wd = c.q[i];
    UNR for (uint b = 0; b < 4; ++b) { uint by = (wd >> (8u * b)) & 255u, e = i * 4u + b;
      w[e] = float((by & 15u) | (((c.h.x >> e) & 1u) << 4)) * Ad - Am;
      w[16u + e] = float((by >> 4) | (((c.h.x >> (16u + e)) & 1u) << 4)) * Bd - Bm; } }
}
inline float q6sc(uint4 m, uint k) { uint wd = k < 4u ? m.x : (k < 8u ? m.y : (k < 12u ? m.z : m.w)); return float(char((wd >> ((k & 3u) * 8u)) & 255u)); }
inline void dec_q6k(KU c, uint u, thread float* w) {
  float d = float(as_type<half2>(c.d).x); uint ib = (u & 7u) * 2u;
  float s0 = d * q6sc(c.m, ib), s1 = d * q6sc(c.m, ib + 1u);
  UNR for (uint i = 0; i < 4; ++i) { uint wd = c.q[i];
    UNR for (uint b = 0; b < 4; ++b) { uint by = (wd >> (8u * b)) & 255u, e = i * 4u + b;
      w[e] = s0 * (float((by & 15u) | (((c.h.x >> (2u * e)) & 3u) << 4)) - 32.0f);
      w[16u + e] = s1 * (float((by >> 4) | (((c.h.y >> (2u * e)) & 3u) << 4)) - 32.0f); } }
}
inline void dec_iqxs(KU c, uint u, thread float* w) {
  uint ib = u & 7u; uint sl = c.m.y, sh = c.m.x >> 16;
  uint ls = ((sl >> (8u * (ib >> 1) + 4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4);
  float dl = float(as_type<half2>(c.m.x).x) * float(int(ls) - 32);
  UNR for (uint i = 0; i < 4; ++i) { uint wd = c.q[i];
    UNR for (uint b = 0; b < 4; ++b) { uint by = (wd >> (8u * b)) & 255u;
      w[i * 4u + b] = KVC[by & 15u] * dl; w[16u + i * 4u + b] = KVC[by >> 4] * dl; } }
}
"""
HDRK3 = r"""
inline KU ld_q3k(device const uint* qs, device const uint* mt, device const uint* hi, uint n, uint u, uint UPR) {
  KU c; uint2 q = ((device const uint2*)qs)[n * UPR + u]; c.q = uint4(q.x, q.y, 0u, 0u); device const uint* mp = mt + (n * (UPR / 8u) + (u >> 3)) * 5u;
  c.m = uint4(mp[0], mp[1], mp[2], mp[3]); c.d = mp[4]; c.h = uint2(hi[n * UPR + u], 0u); return c; }
inline void dec_q3k(KU c, uint u, thread float* w) {
  float d = float(as_type<half2>(c.d).x); uint ib = (u & 7u) * 2u;
  float s0 = d * q6sc(c.m, ib), s1 = d * q6sc(c.m, ib + 1u);
  UNR for (uint e = 0; e < 16; ++e) {
    w[e] = s0 * float(int((c.q.x >> (2u * e)) & 3u) + int(((c.h.x >> e) & 1u) << 2) - 4);
    w[16u + e] = s1 * float(int((c.q.y >> (2u * e)) & 3u) + int(((c.h.x >> (16u + e)) & 1u) << 2) - 4); }
}
"""
HDRK3I = r"""
inline KU ld_iq3x(device const uint* qs, device const uint* mt, device const uint* hi, uint n, uint u, uint UPR) {
  KU c; uint2 q = ((device const uint2*)qs)[n * UPR + u]; c.q = uint4(q.x, q.y, 0u, 0u); c.m = uint4(0u); c.d = mt[n * (UPR / 8u) + (u >> 3)]; c.h = uint2(hi[n * UPR + u], 0u); return c; }
inline KU ld_iq3s(device const uint* qs, device const uint* mt, device const uint* hi, uint n, uint u, uint UPR) {
  KU c; uint2 q = ((device const uint2*)qs)[n * UPR + u]; c.q = uint4(q.x, q.y, 0u, 0u); c.m = ((device const uint4*)mt)[n * (UPR / 8u) + (u >> 3)]; c.d = c.m.x; c.h = uint2(hi[n * UPR + u], 0u); return c; }
inline uint ksg(uint s) { return s | ((popcount(s) & 1u) << 7); }
inline void dec_iq3x(KU c, uint u, thread float* w) {
  float db = float(as_type<half2>(c.d).x) * (0.5f + float(c.h.x >> 28)) * 0.5f;
  UNR for (uint l = 0; l < 4; ++l) {
    uint g1 = IQ3_G[(c.q[l >> 1] >> (16u * (l & 1u))) & 255u], g2 = IQ3_G[(c.q[l >> 1] >> (16u * (l & 1u) + 8u)) & 255u];
    uint sg = ksg((c.h.x >> (7u * l)) & 127u);
    UNR for (uint i = 0; i < 4; ++i) {
      w[8u * l + i] = ((sg >> i) & 1u ? -db : db) * float((g1 >> (8u * i)) & 255u);
      w[8u * l + 4u + i] = ((sg >> (4u + i)) & 1u ? -db : db) * float((g2 >> (8u * i)) & 255u); } }
}
inline void dec_iq3s(KU c, uint u, thread float* w) {
  uint ib = u & 7u; uint qh = ((ib < 4u ? c.m.y : c.m.z) >> (8u * (ib & 3u))) & 255u;
  float db = float(as_type<half2>(c.d).x) * (1.0f + 2.0f * float((c.m.w >> (4u * ib)) & 15u));
  UNR for (uint l = 0; l < 4; ++l) {
    uint a = (c.q[l >> 1] >> (16u * (l & 1u))) & 255u, b = (c.q[l >> 1] >> (16u * (l & 1u) + 8u)) & 255u;
    uint g1 = IQ3_G[a | ((qh << (8u - 2u * l)) & 256u)], g2 = IQ3_G[b | ((qh << (7u - 2u * l)) & 256u)];
    uint sg = (c.h.x >> (8u * l)) & 255u;
    UNR for (uint i = 0; i < 4; ++i) {
      w[8u * l + i] = ((sg >> i) & 1u ? -db : db) * float((g1 >> (8u * i)) & 255u);
      w[8u * l + 4u + i] = ((sg >> (4u + i)) & 1u ? -db : db) * float((g2 >> (8u * i)) & 255u); } }
}
"""

HDRK2 = r"""
inline uint kub(uint4 m, uint k) { uint wd = k < 4u ? m.x : (k < 8u ? m.y : (k < 12u ? m.z : m.w)); return (wd >> ((k & 3u) * 8u)) & 255u; }
inline KU ld_q2k(device const uint* qs, device const uint* mt, uint n, uint u, uint UPR) {
  KU c; uint2 q = ((device const uint2*)qs)[n * UPR + u]; c.q = uint4(q.x, q.y, 0u, 0u); device const uint* mp = mt + (n * (UPR / 8u) + (u >> 3)) * 5u;
  c.m = uint4(mp[0], mp[1], mp[2], mp[3]); c.d = mp[4]; c.h = uint2(0u); return c; }
inline void dec_q2k(KU c, uint u, thread float* w) {
  half2 dd = as_type<half2>(c.d); uint ib = (u & 7u) * 2u; uint a = kub(c.m, ib), b = kub(c.m, ib + 1u);
  float d0 = float(dd.x) * float(a & 15u), m0 = float(dd.y) * float(a >> 4), d1 = float(dd.x) * float(b & 15u), m1 = float(dd.y) * float(b >> 4);
  UNR for (uint e = 0; e < 16; ++e) {
    w[e] = d0 * float((c.q.x >> (2u * e)) & 3u) - m0;
    w[16u + e] = d1 * float((c.q.y >> (2u * e)) & 3u) - m1; }
}
inline KU ld_iq2xxs(device const uint* qs, device const uint* mt, device const uint* hi, uint n, uint u, uint UPR) {
  KU c; c.q = uint4(qs[n * UPR + u], 0u, 0u, 0u); c.m = uint4(0u); c.d = mt[n * (UPR / 8u) + (u >> 3)]; c.h = uint2(hi[n * UPR + u], 0u); return c; }
inline KU ld_iq2xs(device const uint* qs, device const uint* mt, uint n, uint u, uint UPR) {
  KU c; uint2 q = ((device const uint2*)qs)[n * UPR + u]; c.q = uint4(q.x, q.y, 0u, 0u); device const uint* mp = mt + (n * (UPR / 8u) + (u >> 3)) * 3u;
  c.m = uint4(mp[1], mp[2], 0u, 0u); c.d = mp[0]; c.h = uint2(0u); return c; }
inline KU ld_iq2s(device const uint* qs, device const uint* mt, device const uint* hi, uint n, uint u, uint UPR) {
  KU c; c.q = uint4(qs[n * UPR + u], 0u, 0u, 0u); device const uint* mp = mt + (n * (UPR / 8u) + (u >> 3)) * 5u;
  c.m = uint4(mp[1], mp[2], mp[3], mp[4]); c.d = mp[0]; c.h = uint2(hi[n * UPR + u], 0u); return c; }
inline KU ld_iq1s(device const uint* qs, device const uint* mt, uint n, uint u, uint UPR) {
  KU c; c.q = uint4(qs[n * UPR + u], 0u, 0u, 0u); device const uint* mp = mt + (n * (UPR / 8u) + (u >> 3)) * 5u;
  c.m = uint4(mp[1], mp[2], mp[3], mp[4]); c.d = mp[0]; c.h = uint2(0u); return c; }
inline KU ld_iq1m(device const uint* qs, device const uint* mt, uint n, uint u, uint UPR) {
  KU c; c.q = uint4(qs[n * UPR + u], 0u, 0u, 0u); device const uint* mp = mt + (n * (UPR / 8u) + (u >> 3)) * 7u;
  c.m = uint4(mp[3], mp[4], mp[5], mp[6]); c.d = mp[0]; c.h = uint2(mp[1], mp[2]); return c; }
inline void iq2_group(uint g0, uint g1, uint sg, float db, thread float* w) {
  UNR for (uint i = 0; i < 4; ++i) {
    w[i] = ((sg >> i) & 1u ? -db : db) * float((g0 >> (8u * i)) & 255u);
    w[4u + i] = ((sg >> (4u + i)) & 1u ? -db : db) * float((g1 >> (8u * i)) & 255u); }
}
template <typename TP> inline void dec_iq2xxs(KU c, uint u, thread float* w, TP T) {
  float db = float(as_type<half2>(c.d).x) * (0.5f + float(c.h.x >> 28)) * 0.25f;
  UNR for (uint l = 0; l < 4; ++l) { uint g = (c.q.x >> (8u * l)) & 255u;
    iq2_group(T[2u * g], T[2u * g + 1u], ksg((c.h.x >> (7u * l)) & 127u), db, &w[8u * l]); }
}
template <typename TP> inline void dec_iq2xs(KU c, uint u, thread float* w, TP T) {
  uint sb = kub(c.m, u & 7u); float d = float(as_type<half2>(c.d).x);
  UNR for (uint l = 0; l < 4; ++l) { uint v = (c.q[l >> 1] >> (16u * (l & 1u))) & 0xFFFFu, g = v & 511u;
    float db = d * (0.5f + float(l < 2u ? (sb & 15u) : (sb >> 4))) * 0.25f;
    iq2_group(T[2u * g], T[2u * g + 1u], ksg(v >> 9), db, &w[8u * l]); }
}
template <typename TP> inline void dec_iq2s(KU c, uint u, thread float* w, TP T) {
  uint ib = u & 7u, qh = kub(c.m, ib), sb = kub(c.m, 8u + ib); float d = float(as_type<half2>(c.d).x);
  UNR for (uint l = 0; l < 4; ++l) { uint g = ((c.q.x >> (8u * l)) & 255u) | ((qh << (8u - 2u * l)) & 0x300u);
    float db = d * (0.5f + float(l < 2u ? (sb & 15u) : (sb >> 4))) * 0.25f;
    iq2_group(T[2u * g], T[2u * g + 1u], (c.h.x >> (8u * l)) & 255u, db, &w[8u * l]); }
}
template <typename TP> inline void iq1_group(uint g, float dl, float delta, thread float* w, TP T) {
  const uint s = T[g];
  UNR for (uint i = 0; i < 8; ++i) w[i] = dl * (float((s >> (2u * i)) & 3u) - 1.0f + delta);
}
template <typename TP> inline void dec_iq1s(KU c, uint u, thread float* w, TP T) {
  uint ib = u & 7u; uint qh = ((ib < 4u ? (ib < 2u ? c.m.x : c.m.y) : (ib < 6u ? c.m.z : c.m.w)) >> (16u * (ib & 1u))) & 0xFFFFu;
  float dl = float(as_type<half2>(c.d).x) * float(2u * ((qh >> 12) & 7u) + 1u), delta = (qh & 0x8000u) ? -0.125f : 0.125f;
  UNR for (uint l = 0; l < 4; ++l) iq1_group(((c.q.x >> (8u * l)) & 255u) | (((qh >> (3u * l)) & 7u) << 8), dl, delta, &w[8u * l], T);
}
template <typename TP> inline void dec_iq1m(KU c, uint u, thread float* w, TP T) {
  uint ib = u & 7u, j = ib >> 1; uint scb = ((j < 2u ? c.h.x : c.h.y) >> (16u * (j & 1u))) & 0xFFFFu; uint lsh = 6u * (ib & 1u);
  float dm = float(as_type<half2>(c.d).x);
  float dl0 = dm * float(2u * ((scb >> lsh) & 7u) + 1u), dl1 = dm * float(2u * ((scb >> (lsh + 3u)) & 7u) + 1u);
  uint h0 = kub(c.m, 2u * ib), h1 = kub(c.m, 2u * ib + 1u);
  uint gi[4] = { ((c.q.x) & 255u) | ((h0 << 8) & 0x700u), ((c.q.x >> 8) & 255u) | ((h0 << 4) & 0x700u), ((c.q.x >> 16) & 255u) | ((h1 << 8) & 0x700u), ((c.q.x >> 24) & 255u) | ((h1 << 4) & 0x700u) };
  UNR for (uint l = 0; l < 4; ++l) { uint hh = l < 2u ? h0 : h1; iq1_group(gi[l], l < 2u ? dl0 : dl1, (hh & ((l & 1u) ? 0x80u : 0x08u)) ? -0.125f : 0.125f, &w[8u * l], T); }
}
"""

HDRKP = r"""
inline uint sgm(uint sg, uint sh) { return (((sg >> sh) & 5u) * 0x20008000u) & 0x80008000u; }
inline half2 bp(uint g) { return as_type<half2>((g & 0x00FF00FFu) | 0x64006400u) - half2(1024.0h); }
inline half2 h2sat(float v) { return half2(half(clamp(v, -65504.0f, 65504.0f))); }
inline half2 h2cl(half2 v) { return clamp(v, half2(-65504.0h), half2(65504.0h)); }
inline void gp8(uint g0, uint g1, uint sg, half2 db, thread half2* w) {
  w[0] = h2cl(as_type<half2>(as_type<uint>(bp(g0)) ^ sgm(sg, 0u)) * db);
  w[1] = h2cl(as_type<half2>(as_type<uint>(bp(g0 >> 8)) ^ sgm(sg, 1u)) * db);
  w[2] = h2cl(as_type<half2>(as_type<uint>(bp(g1)) ^ sgm(sg, 4u)) * db);
  w[3] = h2cl(as_type<half2>(as_type<uint>(bp(g1 >> 8)) ^ sgm(sg, 5u)) * db);
}
"""
HDRKP3 = r"""
inline void dp_iq3x(KU c, uint u, thread half2* w, int) {
  half2 db = h2sat(float(as_type<half2>(c.d).x) * (0.5f + float(c.h.x >> 28)) * 0.5f);
  UNR for (uint l = 0; l < 4; ++l) {
    gp8(IQ3_G[(c.q[l >> 1] >> (16u * (l & 1u))) & 255u], IQ3_G[(c.q[l >> 1] >> (16u * (l & 1u) + 8u)) & 255u], ksg((c.h.x >> (7u * l)) & 127u), db, &w[4u * l]); }
}
inline void dp_iq3s(KU c, uint u, thread half2* w, int) {
  uint ib = u & 7u; uint qh = ((ib < 4u ? c.m.y : c.m.z) >> (8u * (ib & 3u))) & 255u;
  half2 db = h2sat(float(as_type<half2>(c.d).x) * (1.0f + 2.0f * float((c.m.w >> (4u * ib)) & 15u)));
  UNR for (uint l = 0; l < 4; ++l) {
    uint a = (c.q[l >> 1] >> (16u * (l & 1u))) & 255u, b = (c.q[l >> 1] >> (16u * (l & 1u) + 8u)) & 255u;
    gp8(IQ3_G[a | ((qh << (8u - 2u * l)) & 256u)], IQ3_G[b | ((qh << (7u - 2u * l)) & 256u)], (c.h.x >> (8u * l)) & 255u, db, &w[4u * l]); }
}
"""
HDRKP2 = r"""
template <typename TP> inline void dp_iq2xxs(KU c, uint u, thread half2* w, TP T) {
  half2 db = h2sat(float(as_type<half2>(c.d).x) * (0.5f + float(c.h.x >> 28)) * 0.25f);
  UNR for (uint l = 0; l < 4; ++l) { uint g = (c.q.x >> (8u * l)) & 255u;
    gp8(T[2u * g], T[2u * g + 1u], ksg((c.h.x >> (7u * l)) & 127u), db, &w[4u * l]); }
}
template <typename TP> inline void dp_iq2xs(KU c, uint u, thread half2* w, TP T) {
  uint sb = kub(c.m, u & 7u); float d = float(as_type<half2>(c.d).x);
  UNR for (uint l = 0; l < 4; ++l) { uint v = (c.q[l >> 1] >> (16u * (l & 1u))) & 0xFFFFu, g = v & 511u;
    half2 db = h2sat(d * (0.5f + float(l < 2u ? (sb & 15u) : (sb >> 4))) * 0.25f);
    gp8(T[2u * g], T[2u * g + 1u], ksg(v >> 9), db, &w[4u * l]); }
}
template <typename TP> inline void dp_iq2s(KU c, uint u, thread half2* w, TP T) {
  uint ib = u & 7u, qh = kub(c.m, ib), sb = kub(c.m, 8u + ib); float d = float(as_type<half2>(c.d).x);
  UNR for (uint l = 0; l < 4; ++l) { uint g = ((c.q.x >> (8u * l)) & 255u) | ((qh << (8u - 2u * l)) & 0x300u);
    half2 db = h2sat(d * (0.5f + float(l < 2u ? (sb & 15u) : (sb >> 4))) * 0.25f);
    gp8(T[2u * g], T[2u * g + 1u], (c.h.x >> (8u * l)) & 255u, db, &w[4u * l]); }
}
inline half2 dgt(uint v, uint i) { return as_type<half2>(((v >> (2u * i)) & 0x00030003u) | 0x64006400u) - half2(1024.0h); }
inline void dp_q2k(KU c, uint u, thread half2* w, int) {
  half2 dd = as_type<half2>(c.d); uint ib = (u & 7u) * 2u; uint a = kub(c.m, ib), b = kub(c.m, ib + 1u);
  half2 S0 = h2sat(float(dd.x) * float(a & 15u)), Z0 = h2sat(-float(dd.y) * float(a >> 4)), S1 = h2sat(float(dd.x) * float(b & 15u)), Z1 = h2sat(-float(dd.y) * float(b >> 4));
  UNR for (uint e = 0; e < 8; ++e) { w[e] = h2cl(fma(S0, dgt(c.q.x, e), Z0)); w[8u + e] = h2cl(fma(S1, dgt(c.q.y, e), Z1)); }
}
template <typename TP> inline void iq1_pairs(uint g, float dl, float delta, thread half2* w, TP T) {
  const uint s = T[g]; const uint v = (s & 255u) | ((s >> 8) << 16); half2 S = h2sat(dl), Z = h2sat(dl * (delta - 1.0f));
  UNR for (uint i = 0; i < 4; ++i) w[i] = h2cl(fma(S, dgt(v, i), Z));
}
template <typename TP> inline void dp_iq1s(KU c, uint u, thread half2* w, TP T) {
  uint ib = u & 7u; uint qh = ((ib < 4u ? (ib < 2u ? c.m.x : c.m.y) : (ib < 6u ? c.m.z : c.m.w)) >> (16u * (ib & 1u))) & 0xFFFFu;
  float dl = float(as_type<half2>(c.d).x) * float(2u * ((qh >> 12) & 7u) + 1u), delta = (qh & 0x8000u) ? -0.125f : 0.125f;
  UNR for (uint l = 0; l < 4; ++l) iq1_pairs(((c.q.x >> (8u * l)) & 255u) | (((qh >> (3u * l)) & 7u) << 8), dl, delta, &w[4u * l], T);
}
template <typename TP> inline void dp_iq1m(KU c, uint u, thread half2* w, TP T) {
  uint ib = u & 7u, j = ib >> 1; uint scb = ((j < 2u ? c.h.x : c.h.y) >> (16u * (j & 1u))) & 0xFFFFu; uint lsh = 6u * (ib & 1u);
  float dm = float(as_type<half2>(c.d).x);
  float dl0 = dm * float(2u * ((scb >> lsh) & 7u) + 1u), dl1 = dm * float(2u * ((scb >> (lsh + 3u)) & 7u) + 1u);
  uint h0 = kub(c.m, 2u * ib), h1 = kub(c.m, 2u * ib + 1u);
  uint gi[4] = { ((c.q.x) & 255u) | ((h0 << 8) & 0x700u), ((c.q.x >> 8) & 255u) | ((h0 << 4) & 0x700u), ((c.q.x >> 16) & 255u) | ((h1 << 8) & 0x700u), ((c.q.x >> 24) & 255u) | ((h1 << 4) & 0x700u) };
  UNR for (uint l = 0; l < 4; ++l) { uint hh = l < 2u ? h0 : h1; iq1_pairs(gi[l], l < 2u ? dl0 : dl1, (hh & ((l & 1u) ? 0x80u : 0x08u)) ? -0.125f : 0.125f, &w[4u * l], T); }
}
"""


def _tab(name, vals, ctype="uint"):
    return f"constant {ctype} {name}[{len(vals)}] = {{" + ",".join(str(int(v)) + "u" for v in vals) + "};\n"


def hdrkp(fmt):
    """hdrk plus the half2 decoders (dp_*), for the few-row MMA."""
    return hdrk(fmt) + HDRKP + (HDRKP3 if fmt in ("iq3_s", "iq3_xxs") else HDRKP2 if fmt in _K2 else "")


def hdrk(fmt):
    """Metal header of a K format: the shared one, plus for the 2- and 3-bit classes their decoders and the grid tables (constant memory)."""
    if fmt in _K2:
        from . import foreign_grids as G, foreign_grids_gen as GG, foreign_seeds as S
        h = HDRK + "inline uint ksg(uint s) { return s | ((popcount(s) & 1u) << 7); }\n"
        h += "" if fmt in ("iq2_xxs", "iq2_xs", "iq2_s") else _tab("IQ2_G", [0, 0])         # decoders of the other types stay compiled, their tables are stubs
        h += "" if fmt in ("iq1_s", "iq1_m") else _tab("IQ1_G", [0], "ushort")
        if fmt in ("iq2_xxs", "iq2_xs", "iq2_s"):
            g = GG.iq2_grid({"iq2_xxs": "xxs", "iq2_xs": "xs", "iq2_s": "s"}[fmt]).astype("<u1").reshape(-1, 4).view("<u4").reshape(-1)
            h += _tab("IQ2_G", g)
        if fmt in ("iq1_s", "iq1_m"):
            h += _tab("IQ1_G", S.KGRID_1BIT_2048, "ushort")
        return h + HDRK2
    if fmt not in _K3:
        return HDRK
    if fmt == "q3_k":
        return HDRK + HDRK3
    from . import foreign_grids as G
    g = G.IQ3XXS_GRID if fmt == "iq3_xxs" else G.IQ3S_GRID
    return HDRK + f"constant uint IQ3_G[{len(g)}] = {{" + ",".join(f"0x{v:08x}u" for v in g) + "};\n" + HDRK3I


_KLD = {"q5_k": "ld_q5k(q4, (device const uint4*)mt, hi, {n}, {u}, UPR)", "q6_k": "ld_q6k(q4, mt, (device const uint2*)hi, {n}, {u}, UPR)",
        "iq4_xs": "ld_iqxs(q4, (device const uint2*)mt, {n}, {u}, UPR)",
        "q3_k": "ld_q3k(qs, mt, hi, {n}, {u}, UPR)", "iq3_xxs": "ld_iq3x(qs, mt, hi, {n}, {u}, UPR)", "iq3_s": "ld_iq3s(qs, mt, hi, {n}, {u}, UPR)",
        "q2_k": "ld_q2k(qs, mt, {n}, {u}, UPR)", "iq2_xxs": "ld_iq2xxs(qs, mt, hi, {n}, {u}, UPR)", "iq2_xs": "ld_iq2xs(qs, mt, {n}, {u}, UPR)", "iq2_s": "ld_iq2s(qs, mt, hi, {n}, {u}, UPR)",
        "iq1_s": "ld_iq1s(qs, mt, {n}, {u}, UPR)", "iq1_m": "ld_iq1m(qs, mt, {n}, {u}, UPR)"}
_KTAB = {"iq2_xxs": ("IQ2_G", "uint", 512), "iq2_xs": ("IQ2_G", "uint", 1024), "iq2_s": ("IQ2_G", "uint", 2048), "iq1_s": ("IQ1_G", "ushort", 2048), "iq1_m": ("IQ1_G", "ushort", 2048)}


TGTAB = os.environ.get("KYOJIN_K_TGTAB", "0") == "1"       # experiment: grid tables of the 2-bit class in threadgroup memory


def ktab_decl(fmt, tg):
    """Kernel-body lines defining TABP, the grid table the decoders of fmt read: the constant array, or (tg) a threadgroup copy filled by the whole group. '' for the other formats."""
    if fmt not in _KTAB:
        return ""
    nm, ct, n = _KTAB[fmt]
    if not tg:
        return f"constant {ct}* TABP = {nm};"
    return (f"threadgroup {ct} TABS[{n}]; for (uint i_ = tid; i_ < {n}u; i_ += 128u) TABS[i_] = {nm}[i_]; threadgroup_barrier(mem_flags::mem_threadgroup);"
            f" threadgroup const {ct}* TABP = TABS;")


def ktab_arg(fmt):
    return ", TABP" if fmt in _KTAB else ""


_KDEC = {"q5_k": "dec_q5k", "q6_k": "dec_q6k", "iq4_xs": "dec_iqxs", "q3_k": "dec_q3k", "iq3_xxs": "dec_iq3x", "iq3_s": "dec_iq3s", "q2_k": "dec_q2k",
         "iq2_xxs": "dec_iq2xxs", "iq2_xs": "dec_iq2xs", "iq2_s": "dec_iq2s", "iq1_s": "dec_iq1s", "iq1_m": "dec_iq1m"}


def gemvk_inputs(fmt, gather=False, glu=False):
    base = ["x", "qs", "mt"] if fmt in _NOHI else ["x", "qs", "mt", "hi"]
    if glu and not gather:
        return ["x", "x2"] + base[1:]
    return ["x"] + (["x2"] if glu else []) + ["xrows", "eids"] + base[1:] if gather else base


def dequantk_inputs(fmt):
    return gemvk_inputs(fmt)[1:]


@functools.lru_cache(maxsize=None)
def gemvk_source(fmt: str, nbr: int = 2, gather: bool = False, glu: bool = False) -> str:
    """Body. Template ints: R (1..16), KIN, N, NBR (= nbr). Inputs x (half [R, KIN]), qs (uint32 [N, KIN/8]), mt, [hi] (see gemvk_inputs); output y half [R, N], saturated to +-65504.
    Needs KIN % 512 == 0 and N % (8 NBR) == 0.
    gather (routed experts, R = 1): the arrays hold E stacked matrices as E * N rows; slot = grid y reads row xrows[slot] of x and expert eids[slot] (rows EXO = eid * N on),
    and writes y half [S, N] row slot. Every loader indexes by row, so the expert offset is added to the row number only.
    glu (gather only, down projection of a gated MLP): the staged input of slot s is silu(x[s]) * x2[s], rounded to half per element (GLU_HELPER), so the
    gate / up products need no separate silu-mul launch; x and x2 are [S, KIN] and xrows is the slot id."""
    if gather:
        setup = """const uint slot = threadgroup_position_in_grid.y; const uint ex = eids[slot];
    device const half* XP = (device const half*)(x + ulong(xrows[slot]) * uint(KIN));
    device half* YP = y + ulong(slot) * uint(N);
    const uint EXO = ex * uint(N);"""
    else:
        setup = """device const half* XP = (device const half*)x;
    device half* YP = y;
    const uint EXO = 0u;"""
    xa, xboff = _gemv_xmap("q4_k" if _LAYOUTK[fmt] == "Q" else fmt)
    ld = _KLD[fmt]
    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tgi = threadgroup_position_in_grid.x;
    const uint hh = lane >> 4, ul = lane & 15u;
    const uint rbase = (tgi * 4u + sg) * (2u * NBR) + hh * NBR;
    const uint UPR = uint(KIN) / 32u, NC = uint(KIN) / 512u;
    {setup}
    device const uint4* q4 = (device const uint4*)qs;
    threadgroup uint4 xs_v[R * 80]; threadgroup half* xs = (threadgroup half*)xs_v;
    {ktab_decl(fmt, TGTAB)}
    float w[NBR * 32];
    float acc[NBR * R];
    UNR for (uint i = 0; i < NBR * R; ++i) acc[i] = 0.0f;
    KU cu[NBR], nu[NBR];
    UNR for (uint j = 0; j < NBR; ++j) {{ const uint n = rbase + j + EXO, u = ul; cu[j] = {ld.format(n='n', u='u')}; }}
    for (uint c = 0; c < NC; ++c) {{
      for (uint i = tid; i < uint(R) * 64u; i += 128u) {{ uint r = i >> 6, j = i & 63u;
        uint4 v = *(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u);
        *(threadgroup uint4*)(xs + r * 640u + j * 8u + (j >> 2) * 8u) = v; }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (c + 1u < NC) {{
        UNR for (uint j = 0; j < NBR; ++j) {{ const uint n = rbase + j + EXO, u = (c + 1u) * 16u + ul; nu[j] = {ld.format(n='n', u='u')}; }}
      }}
      const uint xa = {xa}, xb = xa + {xboff}u;
      UNR for (uint j = 0; j < NBR; ++j) {_KDEC[fmt]}(cu[j], ul, &w[j * 32u]{ktab_arg(fmt)});
      const uint pa = xa + (xa >> 5) * 8u, pb = xb + (xb >> 5) * 8u;
      UNR for (uint r = 0; r < R; ++r) {{
        threadgroup const half* xr = xs + r * 640u;
        float xv[32];
        UNR for (uint t = 0; t < 4; ++t) {{ float4 va = float4(*(threadgroup const half4*)(xr + pa + 4u * t)); float4 vb = float4(*(threadgroup const half4*)(xr + pb + 4u * t));
          UNR for (uint k = 0; k < 4; ++k) {{ xv[4u * t + k] = va[k]; xv[16u + 4u * t + k] = vb[k]; }} }}
        UNR for (uint j = 0; j < NBR; ++j) {{ float s = 0.0f;
          UNR for (uint e = 0; e < 32; ++e) s += w[j * 32u + e] * xv[e];
          acc[j * R + r] += s; }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      UNR for (uint j = 0; j < NBR; ++j) cu[j] = nu[j];
    }}
    UNR for (uint j = 0; j < NBR; ++j) {{
      UNR for (uint r = 0; r < R; ++r) {{
        float v = acc[j * R + r];
        v += simd_shuffle_xor(v, 8u); v += simd_shuffle_xor(v, 4u); v += simd_shuffle_xor(v, 2u); v += simd_shuffle_xor(v, 1u);
        if (ul == 0u) YP[r * uint(N) + rbase + j] = half({SAT16.format(v='v')});
      }}
    }}
"""

    if glu and not gather:
        s_old = "uint4 v = *(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u);"
        assert src.count(s_old) == 1
        src = src.replace(s_old, GLU_STAGE)
    elif glu:
        old = "device const half* XP = (device const half*)(x + ulong(xrows[slot]) * uint(KIN));"
        assert old in src and src.count("uint4 v = *(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u);") == 1
        src = src.replace(old, old + "\n    device const half* XP2 = (device const half*)(x2 + ulong(xrows[slot]) * uint(KIN));")
        src = src.replace("uint4 v = *(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u);",
                          "uint4 v = glu8(*(device const uint4*)(XP + r * uint(KIN) + c * 512u + j * 8u), *(device const uint4*)(XP2 + r * uint(KIN) + c * 512u + j * 8u));")
    return src


# Dense (non-gather) glu staging: the input of the down projection is silu(a) * b. a = x rows of stride AST, b = x2 rows of stride BST starting BOFF halves in
# (gate | up of one grouped product: x = x2 = the [R, 2 KIN] output, AST = BST = 2 KIN, BOFF = KIN; two separate products: AST = BST = KIN, BOFF = 0).
GLU_STAGE = ("uint4 v = glu8(*(device const uint4*)((device const half*)x + r * uint(AST) + c * 512u + j * 8u), "
             "*(device const uint4*)((device const half*)x2 + uint(BOFF) + r * uint(BST) + c * 512u + j * 8u));")

GLU_HELPER = r"""
inline uint4 glu8(uint4 a, uint4 b) {
  half h[8]; const thread half* pa = (const thread half*)&a; const thread half* pb = (const thread half*)&b;
  for (uint i = 0; i < 8; ++i) { const float av = float(pa[i]); h[i] = half(av * (1.0f / (1.0f + metal::precise::exp(-av))) * float(pb[i])); }
  return *(thread uint4*)h;
}
"""


@functools.lru_cache(maxsize=None)
def dequantk_source(fmt: str) -> str:
    """Body. Template ints: KIN, N. Inputs qs, mt, [hi]; output w half [N, KIN] (natural order). One thread per 32-unit; grid N * KIN / 32 threads."""
    Q = _LAYOUTK[fmt] == "Q"
    ld = _KLD[fmt]
    return f"""
    const uint tid = thread_position_in_grid.x;
    const uint UPR = uint(KIN) / 32u;
    if (tid >= uint(N) * UPR) return;
    const uint n = tid / UPR, u = tid % UPR;
    device const uint4* q4 = (device const uint4*)qs;
    KU c = {ld.format(n='n', u='u')};
    float v[32];
    {ktab_decl(fmt, False)}
    {_KDEC[fmt]}(c, u, v{ktab_arg(fmt)});
    const uint base = n * uint(KIN) + {'kuq(u)' if Q else '32u * u'};
    UNR for (uint e = 0; e < 16; ++e) {{ w[base + e] = half({SAT16.format(v='v[e]')}); w[base + {'32u' if Q else '16u'} + e] = half({SAT16.format(v='v[16u + e]')}); }}
"""


def xpermk_index(fmt: str, k: int):
    """Host gather for mmak_source. Lane class c (0..3) of a unit holds nibble word c: weights m = 0..3 are group A elements 4c + m, m = 4..7 group B elements 4c + m - 4.
    Position u*32 + fm*4 + t of a unit holds the x of weight m = 2t + (fm & 1) of lane class fm >> 1 (the k of tile t, B row fm)."""
    import numpy as np
    p = np.arange(32)
    fm, t = p // 4, p % 4
    c, m = fm >> 1, 2 * t + (fm & 1)
    el = np.where(m < 4, 4 * c + m, 16 + 4 * c + m - 4)          # decode order element 0..31 of the unit
    u = np.arange(k // 32)[:, None]
    if _LAYOUTK[fmt] == "Q":
        u8 = u & 7
        base = 256 * (u >> 3) + 64 * (u8 >> 1) + 16 * (u8 & 1)
        nat = base + np.where(el < 16, el, 32 + el - 16)[None, :]
    else:
        nat = 32 * u + el[None, :]
    return nat.astype(np.int32).reshape(-1)


_KMMA_DEC = {
    "q5_k": ("""const uint4 m_ = ((device const uint4*)mt)[rn * (UPR / 8u) + (u >> 3)];
        const uint hb_ = hi[rn * UPR + u];
        float Ad, Am, Bd, Bm; q4k_scales(m_, u & 7u, Ad, Am, Bd, Bm);
        UNR for (uint b = 0; b < 4; ++b) { const uint by = (wd >> (8u * b)) & 255u, e = c * 4u + b;
          wv[b] = float((by & 15u) | (((hb_ >> e) & 1u) << 4)) * Ad - Am;
          wv[4u + b] = float((by >> 4) | (((hb_ >> (16u + e)) & 1u) << 4)) * Bd - Bm; }"""),
    "q6_k": ("""device const uint* mp_ = mt + (rn * (UPR / 8u) + (u >> 3)) * 5u;
        const uint4 m_ = uint4(mp_[0], mp_[1], mp_[2], mp_[3]);
        const float d_ = float(as_type<half2>(mp_[4]).x); const uint ib_ = (u & 7u) * 2u;
        const float s0 = d_ * q6sc(m_, ib_), s1 = d_ * q6sc(m_, ib_ + 1u);
        const uint2 h_ = ((device const uint2*)hi)[rn * UPR + u];
        UNR for (uint b = 0; b < 4; ++b) { const uint by = (wd >> (8u * b)) & 255u, e = c * 4u + b;
          wv[b] = s0 * (float((by & 15u) | (((h_.x >> (2u * e)) & 3u) << 4)) - 32.0f);
          wv[4u + b] = s1 * (float((by >> 4) | (((h_.y >> (2u * e)) & 3u) << 4)) - 32.0f); }"""),
    "iq4_xs": ("""const uint2 m_ = ((device const uint2*)mt)[rn * (UPR / 8u) + (u >> 3)];
        const uint ib_ = u & 7u; const uint sl_ = m_.y, sh_ = m_.x >> 16;
        const uint ls_ = ((sl_ >> (8u * (ib_ >> 1) + 4u * (ib_ & 1u))) & 15u) | (((sh_ >> (2u * ib_)) & 3u) << 4);
        const float dl_ = float(as_type<half2>(m_.x).x) * float(int(ls_) - 32);
        UNR for (uint b = 0; b < 4; ++b) { const uint by = (wd >> (8u * b)) & 255u;
          wv[b] = KVC[by & 15u] * dl_; wv[4u + b] = KVC[by >> 4] * dl_; }"""),
}


@functools.lru_cache(maxsize=None)
def mmak_source(fmt: str, ntn: int = 1) -> str:
    """Simdgroup-matrix kernel of the K formats, weights read as stored. Template ints: R (1..32), KIN, N, NTN. Inputs xp (half [R, KIN], x gathered by xpermk_index), qs
    (uint32 [N, KIN/8]), mt, [hi] (see gemvk_inputs); output y half [R, N], saturated. Lane (fm, c) decodes only its nibble word c of the unit (8 weights of row fm: tile t
    holds weights 2t, 2t+1) so no decode is shared across lanes. A and B are half: the same rounding as the fp16 copy of the weights, float accumulator."""
    return f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n0 = ((threadgroup_position_in_grid.x * 4u + sg) * uint(NTN)) * 8u;
    if (n0 >= uint(N)) return;
    constexpr uint RB = (uint(R) + 7u) / 8u;
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    const uint UPR = uint(KIN) / 32u;
    uint rows_n[NTN];
    UNR for (uint j = 0; j < uint(NTN); ++j) rows_n[j] = min(n0 + 8u * j + fm, uint(N) - 1u);
    simdgroup_matrix<float, 8, 8> acc[NTN][RB];
    UNR for (uint j = 0; j < uint(NTN); ++j) UNR for (uint rb = 0; rb < RB; ++rb) acc[j][rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint u = 0; u < UPR; ++u) {{
      simdgroup_matrix<half, 8, 8> Bx[4][RB];
      UNR for (uint rb = 0; rb < RB; ++rb) {{
        const uint r0 = min(8u * rb + fn, uint(R) - 1u), r1 = min(8u * rb + fn + 1u, uint(R) - 1u);
        const half4 b0 = *(device const half4*)(xp + r0 * uint(KIN) + u * 32u + fm * 4u);
        const half4 b1 = *(device const half4*)(xp + r1 * uint(KIN) + u * 32u + fm * 4u);
        UNR for (uint t = 0; t < 4; ++t) {{ Bx[t][rb].thread_elements()[0] = b0[t]; Bx[t][rb].thread_elements()[1] = b1[t]; }}
      }}
      UNR for (uint j = 0; j < uint(NTN); ++j) {{
        const uint rn = rows_n[j];
        const uint wd = qs[(rn * UPR + u) * 4u + c];
        float wv[8];
        {_KMMA_DEC[fmt]}
        UNR for (uint t = 0; t < 4; ++t) {{
          simdgroup_matrix<half, 8, 8> A;
          A.thread_elements()[0] = half({SAT16.format(v='wv[2u * t]')}); A.thread_elements()[1] = half({SAT16.format(v='wv[2u * t + 1u]')});
          UNR for (uint rb = 0; rb < RB; ++rb) simdgroup_multiply_accumulate(acc[j][rb], A, Bx[t][rb], acc[j][rb]);
        }}
      }}
    }}
    UNR for (uint j = 0; j < uint(NTN); ++j) {{
      const uint nn = n0 + 8u * j + fm;
      UNR for (uint rb = 0; rb < RB; ++rb) UNR for (uint el = 0; el < 2; ++el) {{
        const uint row = 8u * rb + fn + el;
        if (row < uint(R) && nn < uint(N)) y[row * uint(N) + nn] = half({SAT16.format(v='acc[j][rb].thread_elements()[el]')});
      }}
    }}
"""


# ----------------------------------------------------------------------------------------------------------------------------------- few-row MMA (flag foreign_fmma)
# The design of q4mma.q4mma3 on the foreign layouts: half A (decoded in packed half2), half B from a shared transposed x (xt half [K, XW], XW 8 or 16 batch rows),
# float accumulators, a simdgroup owns NT tiles of 8 output rows, WS simdgroups split K in blocks of 128 weights and meet in threadgroup memory.
# A lane (fm, c) loads the 16 bytes of unit 4 kb + c of its rows (one uint4) and walks its 4 words; the weight pair (e = 0, 1) of tile t is nibbles t and t + 4 of
# the word (bits 4t and 4t + 16), so one mask feeds a half2 decode with no repack. k-slot s = 2c + e of the 8x8 tile is any fixed weight as long as x_perm follows
# (fmma_xperm_index). Not iq4_nl: its 16 values need a table (constant, threadgroup or simd_shuffle lookup: 444 / 444 / 204 GB/s at R8, the MMA of `mma_source` 530).
_FMMA_PAIR = {"q4_0": "nib", "q4_k": "nib", "nvfp4": "fp4", "mxfp4": "fp4", "q5_k": "nib", "q6_k": "nib", "iq4_xs": "lut"}
# K formats: q5_k = the q4_k nibbles + one high bit per weight (hi, one word per unit); q6_k = nibbles + two high bits (hi, two words per unit); iq4_xs pairs the two
# nibbles of one byte (weights e and e + 16 share the unit scale) and decodes the pair with a 256-entry half2 table in threadgroup memory.
_FMMA_LAYOUT = {**_LAYOUT, "q5_k": "Q", "q6_k": "H", "iq4_xs": "B"}
_FMMA_HI = {"q5_k": "uint", "q6_k": "uint2"}                                     # K formats with a hi array: its element type
_FMMA_KDEC = tuple(f for f in FORMATSK if f not in ("q5_k", "q6_k", "iq4_xs"))   # the 3- and 2-bit classes: the lane decodes its whole unit with the GEMV decoder (dec_*), any weight pairing works
_FMMA_K = ("q5_k", "q6_k", "iq4_xs") + _FMMA_KDEC                              # no global scale input
_FMMA_PAIR.update({f: "kdec" for f in _FMMA_KDEC})
_DECP = {"iq3_xxs": "dp_iq3x", "iq3_s": "dp_iq3s", "iq2_xxs": "dp_iq2xxs", "iq2_xs": "dp_iq2xs", "iq2_s": "dp_iq2s", "q2_k": "dp_q2k", "iq1_s": "dp_iq1s", "iq1_m": "dp_iq1m"}
# half2 decoders (dp_*) fill 16 pairs per unit; pair m = 4 q + t holds the weights below (the lane lays the unit out by what is cheap to unpack, xperm follows)
_FMMA_LAYOUT.update({f: ("KG" if f in ("iq3_xxs", "iq3_s", "iq2_xxs", "iq2_xs", "iq2_s") else "K2" if f == "q2_k" else "K1" if f in ("iq1_s", "iq1_m") else "K") for f in _FMMA_KDEC})


def _pair_elements(L, m):
    """(first, second) weight of pair m of a unit for the half2 decoders: KG 8-weight groups (g0 bytes 0,2 / 1,3, g1 bytes 0,2 / 1,3), K2 (e, e + 8) of each 16 codes, K1 (i, i + 4) of each 8 digits, K sequential."""
    import numpy as np
    m = np.asarray(m)
    if L == "KG":
        l, p = m // 4, m % 4
        a = np.array([0, 1, 4, 5])[p]
        return 8 * l + a, 8 * l + a + 2
    if L == "K2":
        return 16 * (m // 8) + m % 8, 16 * (m // 8) + m % 8 + 8
    if L == "K1":
        return 8 * (m // 4) + m % 4, 8 * (m // 4) + m % 4 + 4
    return 2 * m, 2 * m + 1


def fmma_header(fmt: str) -> str:
    return hdrkp(fmt) if fmt in _FMMA_KDEC else MMA_HDR


def fmma_inputs(fmt: str):
    return ["xt", "qs", "mt"] + (["hi"] if fmt in _FMMA_HI or (fmt in _FMMA_KDEC and fmt not in _NOHI) else []) + ([] if fmt in _FMMA_K else ["gs"])


def fmma_xperm_index(fmt: str, k: int):
    """int32 [k]: position p = (kb * 16 + q * 4 + t) * 8 + s of the transposed x holds natural input idx[p] (k % 128 == 0): weight pair t of word q, k-slot s = 2c + e of unit 4 kb + c."""
    import numpy as np
    assert k % 128 == 0
    p = np.arange(k)
    s, t, q, kb = p % 8, (p // 8) % 4, (p // 32) % 4, p // 128
    u = 4 * kb + (s >> 1)
    e = s & 1
    nj = t + 4 * e
    L = _FMMA_LAYOUT[fmt]
    if L in ("K", "KG", "K2", "K1"):                     # kdec: pair (q, t) of the unit is pair 4 q + t of the lane's decoder
        first, second = _pair_elements(L, 4 * q + t)
        return (32 * u + np.where(e == 0, first, second)).astype(np.int32)
    if L == "B":
        off = 4 * q + t + 16 * e
    else:
        off = 8 * q + nj if L == "N" else 4 * q + (nj >> 1) + (16 if L == "H" else 32) * (nj & 1)
    if L == "Q":
        u8 = u & 7
        unit = 256 * (u >> 3) + 64 * (u8 >> 1) + 16 * (u8 & 1)
    else:
        unit = 32 * u
    return (unit + off).astype(np.int32)


_FMMA_PREP = r"""
    const uint p = thread_position_in_grid.x;
    if (p >= uint(KIN)) return;
    const uint src = idx[p];
    device half4* o = (device half4*)(xt + ulong(p) * XW);
    for (uint r4 = 0; r4 < XW / 4; ++r4) {
        half v[4];
        for (uint i = 0; i < 4; ++i) { const uint r = 4 * r4 + i; v[i] = r < uint(R) ? x[ulong(r) * KIN + src] : half(0.0f); }
        o[r4] = half4(v[0], v[1], v[2], v[3]);
    }
"""


def fmma_prep_source() -> str:
    """Body. Template ints: R, KIN, XW. Inputs x half [R, KIN], idx int32 [KIN] (fmma_xperm_index); output xt half [KIN, XW], zero padded rows. Grid KIN threads."""
    return _FMMA_PREP


def _fmma_scales(fmt: str) -> str:
    """Fills S[j][i] (half2 scale) and Z[j][i] (half2 bias) of tile j for unit uu of row rn."""
    if fmt == "iq4_xs":      # one scale per unit: d * (6-bit ls - 32); meta = 2 words per 8 units (d | scales_h << 16, scales_l)
        return ("const uint2 m_ = ((device const uint2*)MTP)[ulong(rn) * (UPR / 8u) + (uu >> 3)]; const uint ib_ = uu & 7u;"
                " const uint ls_ = ((m_.y >> (8u * (ib_ >> 1) + 4u * (ib_ & 1u))) & 15u) | ((((m_.x >> 16) >> (2u * ib_)) & 3u) << 4);"
                " S[j][0] = half2(half(float(as_type<half2>(m_.x).x) * float(int(ls_) - 32)));")
    if fmt == "q6_k":        # meta = 5 words per 8 units: 16 int8 scales, d (half); units 2 ib and 2 ib + 1 of the low / high nibble weights
        return ("device const uint* mp_ = MTP + (ulong(rn) * (UPR / 8u) + (uu >> 3)) * 5u; const float d_ = float(as_type<half2>(mp_[4]).x);"
                " const uint ib_ = (uu & 7u) * 2u; S[j][0] = half2(half(d_ * float(((device const char*)mp_)[ib_]))); S[j][1] = half2(half(d_ * float(((device const char*)mp_)[ib_ + 1u])));")
    ml = _mload("q4_k" if fmt == "q5_k" else fmt, "rn", "uu")
    if fmt == "q4_0":
        return f"const half d_ = half({ml}); S[j][0] = half2(d_); Z[j][0] = half2(-8.0h * d_);"
    if fmt == "mxfp4":
        return f"S[j][0] = half2(half(e8m0f({ml})));"
    if fmt == "nvfp4":
        return f"const uint m_ = {ml}; S[j][0] = half2(half(e4m3f(m_ & 255u))); S[j][1] = half2(half(e4m3f((m_ >> 8u) & 255u)));"
    return (f"const uint4 m_ = {ml}; float Ad_, Am_, Bd_, Bm_; q4k_scales(m_, uu & 7u, Ad_, Am_, Bd_, Bm_);"
            " S[j][0] = half2(half(Ad_)); Z[j][0] = half2(half(-Am_)); S[j][1] = half2(half(Bd_)); Z[j][1] = half2(half(-Bm_));")


def _fmma_blk_dec(fmt: str):
    """(kd, blk, dec): the statements of fmma_source that load the unit words / scales of NT row tiles (blk) and decode pair (q, t) into half2 w2 (dec)."""
    pair = _FMMA_PAIR[fmt]
    kd = pair == "kdec"
    sel = {"q4_k": "(t & 1u)", "q5_k": "(t & 1u)", "q6_k": "(t & 1u)", "nvfp4": "(q >> 1)"}.get(fmt, "0u")
    if kd and fmt in _DECP:
        dec = "const half2 w2 = wp[j][q * 4u + t];"
    elif kd:
        dec = ("const float2 f2_ = clamp(float2(wf[j][2u * (q * 4u + t)], wf[j][2u * (q * 4u + t) + 1u]), -65504.0f, 65504.0f); const half2 w2 = half2(f2_);")
    elif fmt in _FMMA_HI:     # high bits of the pair (weights t, t + 4 of word q: hi bit of byte (t >> 1) and (t >> 1) + 2 of nibble t & 1) land on bits 4.. of each half
        if fmt == "q5_k":
            hb = "const uint hs_ = hv[j] >> (4u * q + (t >> 1) + 16u * (t & 1u)); const uint hb_ = ((hs_ & 1u) << 4) | ((hs_ & 4u) << 18);"
            mask, off_ = "0x000F000Fu", "1024.0h"
        else:
            hb = "const uint hs_ = (((t & 1u) != 0u) ? hv[j].y : hv[j].x) >> (8u * q + 2u * (t >> 1)); const uint hb_ = ((hs_ & 3u) << 4) | ((hs_ & 0x30u) << 16);"
            mask, off_ = "0x000F000Fu", "1056.0h"
        dec = (f"const uint xs_ = wv[j][q] >> (4u * t); {hb} const half2 hh_ = as_type<half2>((xs_ & {mask}) | hb_ | 0x64006400u) - half2({off_});")
        dec += (f" const half2 w2 = fma(S[j][{sel}], hh_, Z[j][{sel}]);" if fmt == "q5_k" else f" const half2 w2 = hh_ * S[j][{sel}];")
    elif pair == "lut":
        dec = "const half2 w2 = lut[(wv[j][q] >> (8u * t)) & 255u] * S[j][0];"
    elif pair == "nib":
        dec = (f"const uint xs_ = wv[j][q] >> (4u * t); const half2 hh_ = as_type<half2>((xs_ & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);"
               f" const half2 w2 = fma(S[j][{sel}], hh_, Z[j][{sel}]);")
    else:                    # fp4: E2M1 code (sign s, magnitude bits v) -> half bits (s << 15) | (v << 9) = value / 2^14, exact; the 2^14 comes back in one multiply
        dec = (f"const uint xs_ = (wv[j][q] >> (4u * t)) & 0x000F000Fu; const half2 w2 = as_type<half2>(((xs_ & 0x00070007u) << 9u) | ((xs_ & 0x00080008u) << 12u))"
               f" * 16384.0h * S[j][{sel}];")
    hload = {"q5_k": "hv[j] = hi[ulong(rn) * UPR + uu];", "q6_k": "hv[j] = ((device const uint2*)hi)[ulong(rn) * UPR + uu];"}.get(fmt, "")
    hdecl = f"{_FMMA_HI[fmt]} hv[NT];" if fmt in _FMMA_HI else ""
    if kd and fmt in _DECP:
        blk = (f"half2 wp[NT][16];\n      const uint uu = kb * 4u + c;\n      UNR for (uint j = 0; j < uint(NT); ++j) {{ const uint rn = rowt[j];"
               f" KU cu_ = {_KLD[fmt].format(n='rn', u='uu')}; {_DECP[fmt]}(cu_, uu, wp[j]{ktab_arg(fmt) or ', 0'}); }}")
    elif kd:
        blk = (f"float wf[NT][32];\n      const uint uu = kb * 4u + c;\n      UNR for (uint j = 0; j < uint(NT); ++j) {{ const uint rn = rowt[j];"
               f" KU cu_ = {_KLD[fmt].format(n='rn', u='uu')}; {_KDEC[fmt]}(cu_, uu, wf[j]{ktab_arg(fmt)}); }}")
    else:
        blk = f"""uint4 wv[NT]; half2 S[NT][2], Z[NT][2]; {hdecl}
      const uint uu = kb * 4u + c;
      UNR for (uint j = 0; j < uint(NT); ++j) {{
        const uint rn = rowt[j];
        wv[j] = q4[ulong(rn) * UPR + uu];
        {hload}
        S[j][1] = half2(0.0h); Z[j][0] = Z[j][1] = half2(0.0h);
        {_fmma_scales(fmt)}
      }}"""
    return kd, blk, dec


def fmma_source(fmt: str, rowgs: bool = False) -> str:
    """Body. Template ints: R (1..16), KIN (% 128), N, NT, WS, XW (8 if R <= 8 else 16). Inputs xt (fmma_prep_source), qs, mt, gs; output y half [R, N]. Grid ceil(N / (8 NT)) * 32 WS threads, groups of 32 WS."""
    pair = _FMMA_PAIR[fmt]
    kd, blk, dec = _fmma_blk_dec(fmt)
    lutinit = ("threadgroup half2 lut[256];\n    for (uint i = tid; i < 256u; i += 32u * uint(WS)) lut[i] = half2(half(KVC[i & 15u]), half(KVC[i >> 4]));\n"
               "    threadgroup_barrier(mem_flags::mem_threadgroup);") if pair == "lut" else ""
    gsx = "1.0f" if fmt in _FMMA_K else "gs[0]"
    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    {ktab_decl(fmt, False) if kd else ''}
    const uint warp = simdgroup_index_in_threadgroup;
    {lutinit}
    const uint fm = ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
    const uint c = ((lane >> 3) & 1u) * 2u + (lane & 1u);
    const uint fn = c * 2u;
    const uint n0 = threadgroup_position_in_grid.x * (8u * uint(NT));
    constexpr uint RB = uint(XW) / 8u;
    const uint UPR = uint(KIN) / 32u, KB = uint(KIN) / 128u;
    auto MTP = mt;
    const float GS = {gsx};
    device const uint4* q4 = (device const uint4*)qs;
    simdgroup_matrix<float, 8, 8> acc[NT][RB];
    uint rowt[NT];
    UNR for (uint j = 0; j < uint(NT); ++j) {{
      UNR for (uint rb = 0; rb < RB; ++rb) acc[j][rb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      rowt[j] = min(n0 + 8u * j + fm, uint(N) - 1u);
    }}
    for (uint kb = warp; kb < KB; kb += uint(WS)) {{
      {blk}
      const device half* xb = xt + ulong(kb * 128u + fm) * uint(XW) + fn;
      UNR for (uint q = 0; q < 4; ++q) {{
        UNR for (uint t = 0; t < 4; ++t) {{
          simdgroup_matrix<half, 8, 8> B[RB];
          UNR for (uint rb = 0; rb < RB; ++rb) {{
            const half2 bv = *((const device half2*)(xb + ulong(q * 4u + t) * 8u * uint(XW) + 8u * rb));
            B[rb].thread_elements()[0] = bv.x; B[rb].thread_elements()[1] = bv.y;
          }}
          UNR for (uint j = 0; j < uint(NT); ++j) {{
            {dec}
            simdgroup_matrix<half, 8, 8> A;
            A.thread_elements()[0] = w2.x; A.thread_elements()[1] = w2.y;
            UNR for (uint rb = 0; rb < RB; ++rb) simdgroup_multiply_accumulate(acc[j][rb], A, B[rb], acc[j][rb]);
          }}
        }}
      }}
    }}
    threadgroup float red[WS][8 * NT][XW];
    UNR for (uint j = 0; j < uint(NT); ++j) UNR for (uint rb = 0; rb < RB; ++rb) {{
      red[warp][8u * j + fm][8u * rb + fn] = acc[j][rb].thread_elements()[0];
      red[warp][8u * j + fm][8u * rb + fn + 1u] = acc[j][rb].thread_elements()[1];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 8u * uint(NT) * uint(XW); idx += 32u * uint(WS)) {{
      const uint nl = idx / uint(XW), r = idx % uint(XW);
      if (r < uint(R) && n0 + nl < uint(N)) {{
        float v = 0.0f;
        UNR for (uint s = 0; s < uint(WS); ++s) v += red[s][nl][r];
        y[ulong(r) * uint(N) + n0 + nl] = half(clamp(v * GS, -65504.0f, 65504.0f));
      }}
    }}
"""
    return _row_gs(src, "half(clamp(v * GS, -65504.0f, 65504.0f))", "half(clamp(v * gs[n0 + nl], -65504.0f, 65504.0f))") if rowgs else src


# ----------------------------------------------------------------------------------------------------------------------------------- gate | up in one launch
def pair_gather(src: str, rest: list[str]):
    """A gather kernel (routed experts, any format family) as a PAIR: grid y covers 2 S rows, rows [0, S) read the A matrices, rows [S, 2 S) the B matrices (same
    format, shape and dtypes), and the output y is half [2 S, N] (A rows first). Template int SP = S. Returns (source, input names after x / xrows / eids): every
    per-matrix input `n` of `rest` becomes nA, nB and is selected per threadgroup, so the stored arrays are bound as they are (no joint copy)."""
    a = "const uint slot = threadgroup_position_in_grid.y;"
    b = "y + ulong(slot) * uint(N)"
    assert src.count(a) == 1 and src.count(b) == 1, "pair_gather: unexpected gather setup"
    sel = "".join(f" auto {n} = sec_ ? {n}B : {n}A;" for n in rest)
    src = src.replace(a, "const uint s2_ = threadgroup_position_in_grid.y; const bool sec_ = s2_ >= uint(SP); const uint slot = sec_ ? s2_ - uint(SP) : s2_;" + sel)
    return src.replace(b, "y + ulong(s2_) * uint(N)"), [n + "A" for n in rest] + [n + "B" for n in rest]
