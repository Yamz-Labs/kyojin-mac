"""Metal sources for tools/fmt_gate_mac.py: matrix-vector kernels for foreign 4-bit weight formats (NVFP4, Q4_0, Q4_K, IQ4_NL).

NOT compiled or timed where this was written (no Metal device): every kernel here is checked against a CPU reference by fmt_gate_mac.py on the Mac.

Common design (one skeleton, per-format decode pieces):
  * Weights are stored row-major per output row n, in "units" of 16 bytes = 32 weights, so a lane's load is one 16 B vector and the 32 lanes
    of a simdgroup read 512 contiguous bytes of one row (the repo's EXL3 kernels also stream whole rows of words).
  * A threadgroup is 4 simdgroups (128 threads). A simdgroup owns 2*NBR output rows: lanes 0-15 take rows 0..NBR-1, lanes 16-31 rows NBR..2*NBR-1,
    16 lanes walk the 16 units of a 512-element k chunk. A weight unit is decoded ONCE into registers and reused for all R activation rows
    (the batched-rows form of a routed expert: R tokens share one weight read).
  * The R x 512 activation chunk is staged in threadgroup memory (half, padded 16 B per 32 elements so 16 B lane reads do not share banks),
    next chunk's weight words are loaded before the compute of this one (register prefetch).
  * Reduction: simd_shuffle_xor inside each 16-lane half, lane 0 / 16 writes half.
Variants: f = float FMA on decoded weights; i = int8 activations (Q8_0 style: per-32 scale), integer multiply-accumulate on the nibbles, which
is the llama.cpp form (there it uses dp4a / sdot; MSL has no such instruction, so this is a scalar imad loop and is here to be measured, not assumed).
"""

UNR = "#define UNR _Pragma(\"clang loop unroll(full)\")\n"

HDR_COMMON = UNR + r"""
inline void dec_hl(uint4 q, float Ad, float Am, float Bd, float Bm, thread float* w) {
  UNR for (uint i = 0; i < 4; ++i) { uint wd = q[i];
    UNR for (uint b = 0; b < 4; ++b) { uint by = (wd >> (8u * b)) & 255u;
      w[i * 4u + b] = float(by & 15u) * Ad - Am; w[16u + i * 4u + b] = float(by >> 4) * Bd - Bm; } }
}
inline void dec_hl_i(uint4 q, thread int* wl, thread int* wh) {
  UNR for (uint i = 0; i < 4; ++i) { uint wd = q[i];
    UNR for (uint b = 0; b < 4; ++b) { uint by = (wd >> (8u * b)) & 255u; wl[i * 4u + b] = int(by & 15u); wh[i * 4u + b] = int(by >> 4); } }
}
inline float fp4v(uint c) {                                   // E2M1 code -> float without fp16 subnormals: 0,.5,1,1.5,2,3,4,6 and the sign bit
  uint v = c & 7u; float f = as_type<float>((v << 22) + 0x3F000000u);
  f = (v < 2u) ? 0.5f * float(v) : f;
  return as_type<float>(as_type<uint>(f) | ((c & 8u) << 28));
}
constant float FP4LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
constant char KVC[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
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
"""

# per-format pieces. meta type MT, load expr (n = row, ub = unit in row), x offset of the first 16-run (xa, in k units inside a 512 chunk, from ul), offset of the second run.
Q40_XA = "(ul * 32u)"
Q4K_XA = "((ul >> 3) * 256u + ((ul & 7u) >> 1) * 64u + (ul & 1u) * 16u)"
FORMATS = {
    # name: (meta ctype, meta load, xa expr, xb offset, scale statement for (Ad,Am,Bd,Bm) of unit j or None)
    "q4_0": ("float", "float(mt[n * UPR + ub])", Q40_XA, 16, "Ad[j] = mc[j]; Bd[j] = mc[j]; Am[j] = 8.0f * mc[j]; Bm[j] = Am[j];"),
    "q4_k": ("uint4", "((device const uint4*)mt)[n * (UPR / 8u) + (ub >> 3)]", Q4K_XA, 32,
             "q4k_scales(mc[j], ul & 7u, Ad[j], Am[j], Bd[j], Bm[j]);"),
    "nvfp4": ("uint", "uint(mt[n * UPR + ub])", Q40_XA, 16, None),
    "iq4_nl": ("float", "float(mt[n * UPR + ub])", Q40_XA, 16, None),
}

VARIANTS = {"q4_0": ("f", "i"), "q4_k": ("f", "i"), "nvfp4": ("bits", "lut"), "iq4_nl": ("c", "t")}


def _dec_float(fmt, var):
    """statement(s) decoding unit j of the lane into w[j*32 .. j*32+31] (float mode)."""
    if fmt in ("q4_0", "q4_k"):
        return ""                                                    # filled in by gemv_source (scales + dec_hl)
    if fmt == "nvfp4":
        get = "fp4v(cd)" if var == "bits" else "FP4LUT[cd]"
        return ("{ float s0 = as_type<float>((mc[j] & 255u) << 23), s1 = as_type<float>(((mc[j] >> 8) & 255u) << 23);"
                " UNR for (uint i = 0; i < 4; ++i) { uint wd = qc[j][i]; UNR for (uint t = 0; t < 8; ++t) { uint cd = (wd >> (4u * t)) & 15u;"
                f" w[j * 32u + i * 8u + t] = {get} * (i < 2u ? s0 : s1); }} }} }}")
    if fmt == "iq4_nl":
        get = "float(KVC[v])" if var == "c" else "lut[v]"
        return ("{ float d_ = mc[j]; UNR for (uint i = 0; i < 4; ++i) { uint wd = qc[j][i]; UNR for (uint b = 0; b < 4; ++b) { uint by = (wd >> (8u * b)) & 255u;"
                f" uint v = by & 15u; w[j * 32u + i * 4u + b] = {get} * d_; v = by >> 4; w[j * 32u + 16u + i * 4u + b] = {get} * d_; }} }} }}")
    raise ValueError(fmt)


def gemv_source(fmt, var, nbr):
    """Body of the matrix-vector kernel. Template ints: R (rows), KIN, N, NBR=nbr (given again as template). Inputs x/xq, qs, mt; output y (half [R, N])."""
    mt, mload, xa, xboff, _ = FORMATS[fmt]
    ints = var == "i"
    if ints:
        stage = r"""
    for (uint i = tid; i < uint(R) * 32u; i += 128u) { uint r = i >> 5, j = i & 31u;
      uint4 v = *(device const uint4*)(xq + r * uint(KIN) + c * 512u + j * 16u);
      *(threadgroup uint4*)(xq_t + r * 768u + j * 16u + (j >> 1) * 16u) = v; }
    for (uint i = tid; i < uint(R) * 16u; i += 128u) xsc[i] = xsa[(i >> 4) * (uint(KIN) / 32u) + c * 16u + (i & 15u)];
"""
        decl_t = "threadgroup uint4 xq_v[R * 48]; threadgroup char* xq_t = (threadgroup char*)xq_v; threadgroup float xsc[R * 16];"
        sc = FORMATS[fmt][4]
        comp = r"""
      const uint pa = xa + (xa >> 5) * 16u, pb = xb + (xb >> 5) * 16u;
      UNR for (uint r = 0; r < R; ++r) {
        threadgroup const char* xr = xq_t + r * 768u;
        int xva[16], xvb[16]; int sA = 0, sB = 0;
        UNR for (uint t = 0; t < 4; ++t) { char4 va = *(threadgroup const char4*)(xr + pa + 4u * t); char4 vb = *(threadgroup const char4*)(xr + pb + 4u * t);
          UNR for (uint k = 0; k < 4; ++k) { xva[4u * t + k] = int(va[k]); xvb[4u * t + k] = int(vb[k]); sA += xva[4u * t + k]; sB += xvb[4u * t + k]; } }
        const float sxa = xsc[r * 16u + (xa >> 5)], sxb = xsc[r * 16u + (xb >> 5)];
        UNR for (uint j = 0; j < NBR; ++j) { int aA = 0, aB = 0;
          UNR for (uint e = 0; e < 16; ++e) { aA += wl[j * 16u + e] * xva[e]; aB += wh[j * 16u + e] * xvb[e]; }
          acc[j * R + r] += sxa * (Ad[j] * float(aA) - Am[j] * float(sA)) + sxb * (Bd[j] * float(aB) - Bm[j] * float(sB)); }
      }
"""
        pre = "int wl[NBR * 16], wh[NBR * 16]; float Ad[NBR], Am[NBR], Bd[NBR], Bm[NBR];"
        dec = "dec_hl_i(qc[j], &wl[j * 16u], &wh[j * 16u]); " + sc
    else:
        stage = r"""
    for (uint i = tid; i < uint(R) * 64u; i += 128u) { uint r = i >> 6, j = i & 63u;
      uint4 v = *(device const uint4*)(x + r * uint(KIN) + c * 512u + j * 8u);
      *(threadgroup uint4*)(xs + r * 640u + j * 8u + (j >> 2) * 8u) = v; }
"""
        decl_t = "threadgroup uint4 xs_v[R * 80]; threadgroup half* xs = (threadgroup half*)xs_v;"
        pre = "float w[NBR * 32];"
        dec = _dec_float(fmt, var)
        comp = r"""
      const uint pa = xa + (xa >> 5) * 8u, pb = xb + (xb >> 5) * 8u;
      UNR for (uint r = 0; r < R; ++r) {
        threadgroup const half* xr = xs + r * 640u;
        float xv[32];
        UNR for (uint t = 0; t < 4; ++t) { float4 va = float4(*(threadgroup const half4*)(xr + pa + 4u * t)); float4 vb = float4(*(threadgroup const half4*)(xr + pb + 4u * t));
          UNR for (uint k = 0; k < 4; ++k) { xv[4u * t + k] = va[k]; xv[16u + 4u * t + k] = vb[k]; } }
        UNR for (uint j = 0; j < NBR; ++j) { float s = 0.0f;
          UNR for (uint e = 0; e < 32; ++e) s += w[j * 32u + e] * xv[e];
          acc[j * R + r] += s; }
      }
"""
    if not ints and fmt in ("q4_0", "q4_k"):
        # float mode: scales of unit j into locals, then the generic nibble decode
        dec = ("float Ad_, Am_, Bd_, Bm_; " + {
            "q4_0": "Ad_ = mc[j]; Bd_ = mc[j]; Am_ = 8.0f * mc[j]; Bm_ = Am_;",
            "q4_k": "q4k_scales(mc[j], ul & 7u, Ad_, Am_, Bd_, Bm_);"}[fmt] + " dec_hl(qc[j], Ad_, Am_, Bd_, Bm_, &w[j * 32u]);")
    lut_init = "threadgroup float lut[16]; if (tid < 16u) lut[tid] = float(KVC[tid]);" if (fmt == "iq4_nl") else ""
    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tgi = threadgroup_position_in_grid.x;
    const uint hh = lane >> 4, ul = lane & 15u;
    const uint rbase = (tgi * 4u + sg) * (2u * NBR) + hh * NBR;
    const uint UPR = uint(KIN) / 32u, NC = uint(KIN) / 512u;
    device const uint4* q4 = (device const uint4*)qs;
    {decl_t}
    {lut_init}
    {pre}
    float acc[NBR * R];
    UNR for (uint i = 0; i < NBR * R; ++i) acc[i] = 0.0f;
    uint4 qc[NBR], qn[NBR]; {mt} mc[NBR], mn_[NBR];
    UNR for (uint j = 0; j < NBR; ++j) {{ const uint n = rbase + j, ub = ul; qc[j] = q4[n * UPR + ub]; mc[j] = {mload}; }}
    for (uint c = 0; c < NC; ++c) {{
{stage}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (c + 1u < NC) {{
        UNR for (uint j = 0; j < NBR; ++j) {{ const uint n = rbase + j, ub = (c + 1u) * 16u + ul; qn[j] = q4[n * UPR + ub]; mn_[j] = {mload}; }}
      }}
      const uint xa = {xa}, xb = xa + {xboff}u;
      UNR for (uint j = 0; j < NBR; ++j) {{ {dec} }}
{comp}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      UNR for (uint j = 0; j < NBR; ++j) {{ qc[j] = qn[j]; mc[j] = mn_[j]; }}
    }}
    UNR for (uint j = 0; j < NBR; ++j) {{
      UNR for (uint r = 0; r < R; ++r) {{
        float v = acc[j * R + r];
        v += simd_shuffle_xor(v, 8u); v += simd_shuffle_xor(v, 4u); v += simd_shuffle_xor(v, 2u); v += simd_shuffle_xor(v, 1u);
        if (ul == 0u) y[r * uint(N) + rbase + j] = half(v);
      }}
    }}
"""
    return src


def gemv_inputs(fmt, var):
    return ["xq", "xsa", "qs", "mt"] if var == "i" else ["x", "qs", "mt"]


def header(fmt, var):
    return HDR_COMMON


# activation quantiser (Q8_0 style): one thread per 32-element block of a row: scale = amax/127, q = rint(x * 127/amax)
QUANT_SRC = r"""
    const uint t = thread_position_in_grid.x;
    device const half* p = x + t * 32u;
    float am = 0.0f;
    for (uint e = 0; e < 32u; ++e) am = max(am, abs(float(p[e])));
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    for (uint e = 0; e < 32u; ++e) xq[t * 32u + e] = (char)clamp(rint(float(p[e]) * inv), -127.0f, 127.0f);
    xsa[t] = am / 127.0f;
"""
