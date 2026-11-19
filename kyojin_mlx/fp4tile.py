# SPDX-License-Identifier: Apache-2.0
"""Decode-friendly "tile" layout of NVFP4 routed experts (flag KYOJIN_GLM_TILE, default off) and its gathered GEMV.

Stored layout (foreign_pack): qs uint32 [E, N, K/8] (8 e2m1 nibbles per word, element t = nibble t), meta uint16 [E, N, K/32] (two e4m3 scales per 32-value unit).
Tile layout, ONE array per projection (qs and scales share it; the stored arrays are dropped): for expert e, row group g (32 rows), chunk c (UB units = 32 UB
values of every row of the group) one contiguous block of 576 UB bytes:
    [UB * 32 uint4 : qs   unit u, lane l (= row 32 g + l) at (u * 32 + l)    ]  a simdgroup load instruction reads 512 contiguous bytes, lane = row
    [32 * 2 UB bytes: scale codes of lane l: UB units x (first 16 values, last 16 values)]  one 64 / 128 B contiguous read per chunk
Per word the 8 nibbles are bit-sliced so that a half2 of two e2m1 values costs 3-5 integer ops, no table, no float conversion of the scale:
    pair p (values 2p, 2p + 1 of the word, low / high half): magnitude (3 bits) at [11 - 3p : 9 - 3p], sign at bit 12 + p, of each 16-bit half
    half2 bits = ((w << 3p) & 0x0E000E00) | ((w << (3 - p)) & 0x80008000)  ==  (+-code) * 2^-14 as two exact fp16 values
Scale codes: the e4m3 scale (non-negative; 0x7F -> 0) re-coded as an unsigned E5M3 byte c = (e + 10) << 3 | m3 of 2^e (1 + m3 / 8) (exact, subnormals included), so
    as_type<float>(c << 20) = scale * 2^-117 with no branch; code 0 = scale 0. The three power-of-two factors (2^-14 fp16 codes, 2^-117 scales, 2^100 on x) are one
    final multiply by 2^31.
"""
from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from . import kernels as KN

ROWS = 32                                    # output rows per row group = lanes of a simdgroup
XSCALE_LOG2 = 100                            # x is staged as float(x) * 2^100: keeps (2^-14 code) * (2^-117 scale) * x in the normal float range
FINAL_LOG2 = 117 + 14 - XSCALE_LOG2          # 31


def chunk_words(ub: int) -> int:
    return 144 * ub                          # 32 lanes * (16 B qs + 2 ub B scales) per unit... = 576 ub bytes


def tile_ok(n: int, k: int, ub: int = 2) -> bool:
    return n % ROWS == 0 and k % 512 == 0 and k % (32 * ub) == 0 and k * 4 <= 24 * 1024   # chunks per simdgroup even; x staged in float in threadgroup memory


# ------------------------------------------------------------------------------------------------------------------ converter (stored -> tile)
def _scale_code_table() -> np.ndarray:
    """e4m3 byte -> E5M3 code (see module doc), 256 entries (the sign bit is ignored, 0x7F -> 0 as foreign_metal.e4m3f)."""
    t = np.zeros(256, np.uint8)
    for b in range(256):
        b7 = b & 127
        if b7 == 127:
            continue
        E, m = b7 >> 3, b7 & 7
        if E:
            t[b] = ((E - 7 + 10) << 3) | m
        elif m:
            e = m.bit_length() - 1                      # m * 2^-9 = 2^(e - 9) * (m / 2^e)
            m3 = (m - (1 << e)) << (3 - e)
            t[b] = ((e - 9 + 10) << 3) | m3
    return t


_SC_TAB = None


def _recode_words(w):
    """uint32 words of 8 stored nibbles -> bit-sliced words (module doc)."""
    out = mx.zeros(w.shape, mx.uint32)
    for t in range(8):
        c = (w >> (4 * t)) & 15
        p, h = t // 2, t % 2
        out = out | ((c & 7) << (9 - 3 * p + 16 * h)) | ((c >> 3) << (12 + p + 16 * h))
    return out


_recode_c = mx.compile(_recode_words)


def to_tile(qs, meta, ub: int = 2):
    """qs uint32 [E, N, K/8], meta uint16 [E, N, K/32] (stored layout) -> uint32 [E, N/32 * K/(32 ub) * 144 ub] tile layout. Slabs of experts bound the temporaries."""
    global _SC_TAB
    if _SC_TAB is None:
        _SC_TAB = mx.array(_scale_code_table())
    E, N, K8 = qs.shape
    K = K8 * 8
    NG, CH = N // ROWS, K // (32 * ub)
    outs = []
    step = max(1, (256 << 20) // (N * K // 2))                 # ~256 MB of stored qs per slab
    for e0 in range(0, E, step):
        e1 = min(E, e0 + step)
        w = _recode_c(qs[e0:e1])                                # [e, N, K/8]
        w = w.reshape(e1 - e0, NG, ROWS, CH, ub, 4).transpose(0, 1, 3, 4, 2, 5).reshape(e1 - e0, NG, CH, ub * ROWS * 4)
        sc = meta[e0:e1].view(mx.uint8).reshape(e1 - e0, N, K // 16)
        sc = mx.take(_SC_TAB, sc.astype(mx.int32), axis=0)      # [e, N, K/16] codes
        sc = sc.reshape(e1 - e0, NG, ROWS, CH, ub * 2).transpose(0, 1, 3, 2, 4).reshape(e1 - e0, NG, CH, ROWS * ub * 2)   # lane-major, UB * 2 bytes per lane
        sc = sc.view(mx.uint32)                                 # [e, NG, CH, 16 ub]
        outs.append(mx.concatenate([w, sc], axis=-1).reshape(e1 - e0, -1))
        mx.eval(outs[-1])
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


_SC_INV = None


def _unrecode_words(w):
    """bit-sliced words -> the stored nibble words (inverse of _recode_words)."""
    out = mx.zeros(w.shape, mx.uint32)
    for t in range(8):
        p, h = t // 2, t % 2
        c = ((w >> (9 - 3 * p + 16 * h)) & 7) | (((w >> (12 + p + 16 * h)) & 1) << 3)
        out = out | (c << (4 * t))
    return out


_unrecode_c = mx.compile(_unrecode_words)


def from_tile(tl, n: int, k: int, ub: int = 2):
    """tile layout uint32 [E, ...] -> (qs uint32 [E, n, k/8], meta uint16 [E, n, k/32]) in the stored layout (prefill of a tile-layout model: today's wide / fmma kernels read it;
    transient, one layer at a time). Scale codes map back to the e4m3 byte (a NaN 0x7F scale comes back as 0, which decodes the same)."""
    global _SC_INV
    if _SC_INV is None:
        fwd = _scale_code_table()
        inv = np.zeros(256, np.uint8)
        for b in range(127):
            inv[fwd[b]] = b
        _SC_INV = mx.array(inv)
    E = tl.shape[0]
    NG, CH = n // ROWS, k // (32 * ub)
    qo, mo = [], []
    step = max(1, (256 << 20) // (n * k // 2))
    for e0 in range(0, E, step):
        e1 = min(E, e0 + step)
        t = tl[e0:e1].reshape(e1 - e0, NG, CH, 144 * ub)
        w = _unrecode_c(t[..., :128 * ub]).reshape(e1 - e0, NG, CH, ub, ROWS, 4).transpose(0, 1, 4, 2, 3, 5).reshape(e1 - e0, n, k // 8)
        sc = t[..., 128 * ub:].reshape(e1 - e0, NG, CH, 16 * ub).view(mx.uint8).reshape(e1 - e0, NG, CH, ROWS, ub * 2)
        sc = mx.take(_SC_INV, sc.astype(mx.int32), axis=0).astype(mx.uint8)
        sc = sc.transpose(0, 1, 3, 2, 4).reshape(e1 - e0, n, k // 16).view(mx.uint16).reshape(e1 - e0, n, k // 32)
        mx.eval(w, sc)
        qo.append(w); mo.append(sc)
    return (qo[0], mo[0]) if len(qo) == 1 else (mx.concatenate(qo, 0), mx.concatenate(mo, 0))


# ------------------------------------------------------------------------------------------------------------------ reference (numpy, float64)
def tile_ref(qs, meta, gs):
    """float64 [E, N, K] dequant of the STORED arrays (numpy in): the reference both layouts are compared with."""
    qs, meta = np.asarray(qs), np.asarray(meta)
    E, N, K8 = qs.shape
    nib = ((qs[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(E, N, K8 * 8)
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64)
    v = lut[nib & 7] * np.where(nib & 8, -1.0, 1.0)
    sb = meta.view(np.uint8).reshape(E, N, K8 * 8 // 16).astype(np.int64) & 127
    Ex, m = sb >> 3, sb & 7
    sc = np.where(Ex == 0, m * 2.0 ** -9, 2.0 ** (Ex - 7.0) * (1 + m / 8.0))
    sc = np.where(sb == 127, 0.0, sc)
    return v * np.repeat(sc, 16, axis=-1) * np.asarray(gs, np.float64).reshape(E, 1, 1)


# ------------------------------------------------------------------------------------------------------------------ kernels
def _lim_src(limit, act=None):
    """float32 epilogue of a gated MLP: silu(min(a, limit)) * clip(b, -limit, limit) (limit None: plain silu(a) * b), as ForeignSwitchGLU._glu.
    act = ("situ", scale_gate, scale_lin): ((sg tanh(a / sg)) * sigmoid(a)) * (sl tanh(b / sl)), no clamp."""
    if act is not None:
        _, sg, sl = act
        assert act[0] == "situ" and sg > 0.0 and sl > 0.0, act
        return f"const float gl = (({float(sg)!r}f * metal::precise::tanh(a / {float(sg)!r}f)) * (1.0f / (1.0f + metal::precise::exp(-a)))) * ({float(sl)!r}f * metal::precise::tanh(b / {float(sl)!r}f));"
    if limit is None:
        return "const float gl = a * (1.0f / (1.0f + metal::precise::exp(-a))) * b;"
    L = repr(float(limit)) + "f"
    return f"a = min(a, {L}); b = clamp(b, -{L}, {L}); const float gl = a * (1.0f / (1.0f + metal::precise::exp(-a))) * b;"


def gather_source(ub: int = 2, nsg: int = 4, nmat: int = 1, limit=None, mode: str = "", pf: int = 1, act=None) -> str:
    """Gathered GEMV on tile-layout experts (one routed-expert slot per grid.y, R = 1). Template ints KIN, N. Inputs x half [T, K], xrows, eids uint32 [S], tl uint32,
    gs float [E] (nmat 2: tl, tl2, gs, gs2); output y half [S, N].
    A threadgroup = nsg simdgroups on ONE row group of 32 rows; the simdgroups of a matrix split its chunks (c = sgm, sgm + nsm, ...) and add up at the end.
    nmat 2 = gate | up of one expert in one launch: the first nsg / 2 simdgroups stream the gate matrix, the others the up matrix, and the epilogue writes
    h = glu(gate, up) (fp16-rounded gate and up, then the float32 GLU of ForeignSwitchGLU._glu): no silu * up launch.
    pf: ring of pf chunks in registers (pf = 1: the next chunk's loads are issued after the current one is decoded; deeper rings measured slower: register pressure).
    mode: probes only ("mem": loads, no decode; "alu": always chunk 0, i.e. cache-resident; "D": decode without the FMA)."""
    assert ub in (2, 4) and nmat in (1, 2) and nsg % nmat == 0
    nsm = nsg // nmat
    inner = "dacc ^= hb;" if "D" in mode else """const float2 f = float2(as_type<half2>(hb));
              const float2 xv = *(threadgroup const float2*)(xw + 2u * p);
              b0 = fma(f.x, xv.x, b0); b1 = fma(f.y, xv.y, b1);"""
    sw_t = "uint" if ub == 2 else "uint2"
    cb = 36 * ub
    ld = lambda sl, cexpr: (f"{{ device const uint4* cp = base + ulong({'0u' if 'alu' in mode else cexpr}) * {cb}u;\n"
                           f"      UNR for (uint u = 0; u < {ub}u; ++u) q{sl}[u] = cp[u * 32u + lane];\n"
                           f"      sw{sl} = ((device const {sw_t}*)(cp + {ub * 32}u))[lane]; }}")
    decls = "\n    ".join(f"uint4 q{s}[{ub}]; {sw_t} sw{s};" for s in range(pf))
    pre = "\n    ".join(ld(s, f"sgm + {s * nsm}u") for s in range(pf))

    def step(s):
        memonly = (f"UNR for (uint u = 0; u < {ub}u; ++u) acc += as_type<float>((q{s}[u].x ^ q{s}[u].y ^ q{s}[u].z ^ q{s}[u].w) & 0x007FFFFFu) + float(" + (f"sw{s}" if ub == 2 else f"sw{s}.x ^ sw{s}.y") + ");") if "mem" in mode else ""
        comp = "" if "mem" in mode else f"""
        threadgroup const float* xc = xs + c * {32 * ub}u;
        UNR for (uint u = 0; u < {ub}u; ++u) {{
          UNR for (uint blk = 0; blk < 2u; ++blk) {{
            {f"const uint swd = sw{s};" if ub == 2 else f"const uint swd = sw{s}[u >> 1];"}
            const uint sb = ({"u" if ub == 2 else "(u & 1u)"} * 2u + blk) * 8u;
            const float sc = as_type<float>(((swd >> sb) & 255u) << 20);
            float b0 = 0.0f, b1 = 0.0f; uint dacc = 0u;
            UNR for (uint jj = 0; jj < 2u; ++jj) {{
              const uint w = q{s}[u][blk * 2u + jj];
              threadgroup const float* xw = xc + u * 32u + (blk * 2u + jj) * 8u;
              UNR for (uint p = 0; p < 4u; ++p) {{
                const uint hb = ((w << (3u * p)) & 0x0E000E00u) | ((w << (3u - p)) & 0x80008000u);
                {inner}
              }}
            }}
            acc = fma(b0 + b1 + as_type<float>(dacc & 0x007FFFFFu), sc, acc);
          }}
        }}"""
        return f"""{{ const uint c = sgm + (it + {s}u) * {nsm}u;
        {memonly}{comp}
        if (c + {pf * nsm}u < CH) {ld(s, f"c + {pf * nsm}u")}
      }}"""
    steps = "\n      ".join(step(s) for s in range(pf))
    if nmat == 1:
        outs = f"""
      float t = 0.0f;
      UNR for (uint i = 0; i < {nsm}u; ++i) t += red[i * 32u + lane];
      y[ulong(slot) * uint(N) + g * 32u + lane] = half(t * FIN * gs[ex]);"""
        base_sel = "device const uint* tb = tl;"
    else:
        outs = f"""
      float ta = 0.0f, tb2 = 0.0f;
      UNR for (uint i = 0; i < {nsm}u; ++i) {{ ta += red[i * 32u + lane]; tb2 += red[({nsm}u + i) * 32u + lane]; }}
      float a = float(half(ta * FIN * gs[ex])), b = float(half(tb2 * FIN * gs2[ex]));
      {_lim_src(limit, act)}
      y[ulong(slot) * uint(N) + g * 32u + lane] = half(gl);"""
        base_sel = "device const uint* tb = (sg < " + str(nsm) + "u) ? tl : tl2;"
    return f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint sgm = sg % {nsm}u;
    const uint g = threadgroup_position_in_grid.x;
    const uint slot = threadgroup_position_in_grid.y;
    const uint ex = eids[slot];
    const uint NG = uint(N) / 32u, CH = uint(KIN) / {32 * ub}u;
    device const half* XP = (device const half*)(x + ulong(xrows[slot]) * uint(KIN));
    {base_sel}
    device const uint4* base = (device const uint4*)(tb + (ulong(ex) * NG + g) * CH * {144 * ub}u);
    {decls}
    {pre}
    threadgroup float xs[KIN];
    for (uint i = tid; i < uint(KIN) / 8u; i += {32 * nsg}u) {{
      half4 a = *(device const half4*)(XP + i * 8u), b = *(device const half4*)(XP + i * 8u + 4u);
      *(threadgroup float4*)(xs + i * 8u) = float4(a) * XS; *(threadgroup float4*)(xs + i * 8u + 4u) = float4(b) * XS;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint it = 0; it < CH / {nsm}u; it += {pf}u) {{
      {steps}
    }}
    threadgroup float red[{32 * nsg}];
    red[sg * 32u + lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {{{outs}
    }}
"""


def dequant_source() -> str:
    """Tile layout -> float16 [E * N, K] (row major, the global scale applied): the prefill decode of a stacked expert projection (one thread = one 32-value unit).
    Template ints KIN, N, UB, E. Inputs tl, gs float [E]."""
    return """
    const uint t = thread_position_in_grid.x;
    const uint NG = uint(N) / 32u, CH = uint(KIN) / (32u * uint(UB));
    const uint lane = t & 31u;
    ulong r = t >> 5;                                       // unit index over (expert, group, chunk, unit-in-chunk)
    const uint u = uint(r % uint(UB)); r /= uint(UB);
    const uint c = uint(r % CH); r /= CH;
    const uint g = uint(r % NG);
    const uint ex = uint(r / NG);
    device const uint* cb = tl + ((ulong(ex) * NG + g) * CH + c) * (144u * uint(UB));
    const uint4 q = ((device const uint4*)cb)[u * 32u + lane];
    device const uchar* scb = (device const uchar*)(cb + 128u * uint(UB)) + lane * (2u * uint(UB)) + u * 2u;
    const float m = gs[ex] * 2147483648.0f;                 // 2^31
    const float s0 = as_type<float>(uint(scb[0]) << 20) * 1.2676506002282294e30f * m, s1 = as_type<float>(uint(scb[1]) << 20) * 1.2676506002282294e30f * m;   // 2^100 * 2^31 * scale * 2^-117
    device half* wp = w + ((ulong(ex) * uint(N) + g * 32u + lane) * uint(KIN) + (c * uint(UB) + u) * 32u);
    UNR for (uint j = 0; j < 4u; ++j) {
      const uint wd = q[j];
      UNR for (uint p = 0; p < 4u; ++p) {
        const uint hb = ((wd << (3u * p)) & 0x0E000E00u) | ((wd << (3u - p)) & 0x80008000u);
        const float2 f = float2(as_type<half2>(hb)) * (j < 2u ? s0 : s1);
        wp[j * 8u + 2u * p] = half(f.x); wp[j * 8u + 2u * p + 1u] = half(f.y);
      }
    }
"""


HEADER = "#define UNR _Pragma(\"clang loop unroll(full)\")\nconstant constexpr float XS = " + repr(float(2.0 ** XSCALE_LOG2)) + "f;\nconstant constexpr float FIN = " + repr(float(2.0 ** FINAL_LOG2)) + "f;\n"


@functools.lru_cache(maxsize=None)
def _gather_kern(ub, nsg, nmat=1, limit=None, mode="", pf=1, act=None):
    names = ["x", "xrows", "eids", "tl", "gs"] if nmat == 1 else ["x", "xrows", "eids", "tl", "tl2", "gs", "gs2"]
    return KN._kern(("fp4tile_g", ub, nsg, nmat, limit, mode, pf) + ((act,) if act else ()), name=f"kyojin_fp4tile_g{nmat}_u{ub}s{nsg}p{pf}{mode}" + ("_situ" if act else ""), input_names=names, output_names=["y"],
                    header=HEADER, source=gather_source(ub, nsg, nmat, limit, mode, pf, act))


def gather(x, xrows, eids, tl, gs, n: int, k: int, ub: int = 2, nsg: int = 4, mode: str = "", pf: int = 1):
    """y half [S, n]: slot s = expert eids[s] applied to row xrows[s] of x (half [T, k])."""
    S = int(eids.shape[0])
    return _gather_kern(ub, nsg, 1, None, mode, pf)(inputs=[x, xrows, eids, tl, gs], template=[("KIN", k), ("N", n)], grid=(n // ROWS * 32 * nsg, S, 1), threadgroup=(32 * nsg, 1, 1),
                                                    output_shapes=[(S, n)], output_dtypes=[mx.float16])[0]


def gather_glu(x, xrows, eids, tl_g, tl_u, gs_g, gs_u, n: int, k: int, limit=None, ub: int = 2, nsg: int = 8, pf: int = 1, act=None):
    """h half [S, n] = glu(gate(x), up(x)) for each slot, gate and up in one launch (see gather_source nmat 2)."""
    S = int(eids.shape[0])
    return _gather_kern(ub, nsg, 2, None if limit is None else float(limit), "", pf, act)(inputs=[x, xrows, eids, tl_g, tl_u, gs_g, gs_u], template=[("KIN", k), ("N", n)],
                                                                                   grid=(n // ROWS * 32 * nsg, S, 1), threadgroup=(32 * nsg, 1, 1), output_shapes=[(S, n)], output_dtypes=[mx.float16])[0]


@functools.lru_cache(maxsize=None)
def _deq_kern():
    return KN._kern(("fp4tile_deq",), name="kyojin_fp4tile_deq", input_names=["tl", "gs"], output_names=["w"], header=HEADER, source=dequant_source())


def dequant(tl, gs, E: int, n: int, k: int, ub: int = 2):
    """float16 [E, n, k] of a tile-layout stack (prefill: the gathered matmul runs on the decoded stack)."""
    units = E * (n // ROWS) * (k // (32 * ub)) * ub
    return _deq_kern()(inputs=[tl, gs], template=[("KIN", k), ("N", n), ("UB", ub), ("E", E)], grid=(units * 32, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(E * n, k)],
                       output_dtypes=[mx.float16])[0].reshape(E, n, k)
