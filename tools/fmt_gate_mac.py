#!/usr/bin/env python3
"""cfmtgate: can foreign 4-bit weight formats be read faster than our EXL3 kernels on an Apple GPU?  One matrix-vector product, one fixed logical size.

  encodings   EXL3 2 / 3 / 5 bit (the repo's own kernels through Exl3Linear, rotations included)
              NVFP4   4-bit E2M1 codes, one E8M0 (power of two) scale per 16
              Q4_0    unsigned 4-bit minus 8, one fp16 scale per 32 (GGUF Q4_0)
              Q4_K    unsigned 4-bit, 8 sub-blocks of 32 per 256 super-block, 6-bit scale + 6-bit min per sub-block (real GGUF 12-byte packing), fp16 d / dmin
              IQ4_NL  4-bit index into the 16-entry non-linear table of GGUF IQ4_NL, one fp16 scale per 32
  rows        1, 4, 16 activation rows share one weight read (R tokens routed to the same expert).
  output      per encoding and row count: bytes/weight, GB/s of weight bytes read, ms per product, time relative to EXL3 3-bit.

Run on the Mac (needs mlx and this repo):   python3 tools/fmt_gate_mac.py
CPU only (no GPU: encoders, decoders, reference):   python3 tools/fmt_gate_mac.py --cpu-only

Every Metal kernel is first checked against a float64 CPU reference (decoded weights) and is left out of the table if it does not match.
Foreign weights are synthetic N(0,1): one 256-row tile is encoded and tiled down the output dimension; the pool holds POOL distinct buffer
copies that are cycled, so the System Level Cache cannot serve repeated reads. Limits: <= 1 GiB of buffers (accounted before allocation, peak printed),
<= 90 s (deadline checks between stages), one thread (numpy BLAS pinned to 1, no Python threads), every loop bounded.

Metal features (MSL, as used / not available; none of it measured by the author, there was no GPU):
  used     simd_shuffle_xor lane reductions, threadgroup memory (32 KiB per threadgroup), 16-byte vector loads (uint4), half4 / char4 vectors,
           fp16 and fp32 ALUs, as_type bit casts, constant-address tables.
  exist    extract_bits / insert_bits, mul24 / mad24, popcount, simd_sum, simd_prefix ops, simdgroup_matrix: not used here (shifts and masks are the
           baseline; nothing was measured, so whether they would be faster is open).
  absent   no dp4a / sdot: MSL has no packed int8 dot product intrinsic, `dot()` is floating point only; an int8 x 4-bit product is a scalar integer
           multiply-add per element (variant "i" below measures exactly that). No fp4 / E8M0 / 4-bit hardware conversion, no byte-permute (PRMT) or LUT
           instruction: nibbles are expanded with shifts and masks, tables are a constant-memory load (variants "lut", "c") or a threadgroup table ("t").
  maybe    simdgroup_matrix 8x8 fp16 / fp32 multiply-accumulate (all Apple GPUs from A14 / M1 up), and the neural-accelerator tensor ops of newer parts that the
           repo's NAX kernels use: they take fp16 operands, so foreign 4-bit data would first be expanded to fp16 in registers (not tried here).
"""
import argparse, os, sys, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repo root: kyojin_mlx without PYTHONPATH
import fmt_gate_metal as M

TILE = 256                                               # rows encoded once and tiled
GIB = 1 << 30
F32 = np.float32


# ----------------------------------------------------------------------------------------------------------- encoders / decoders (numpy)
# Each format: enc(w [rows, K] f32) -> (qs uint8 [rows, K/2], meta ndarray [device layout]); dec(qs, meta, K) -> f32 [rows, K]
# meta device layout: q4_0 / iq4_nl: fp16 [rows, K/32]; nvfp4: uint16 [rows, K/32] (two E8M0 bytes per 32 weights); q4_k: uint32 [rows, K/256 * 4].

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], F32)
KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], F32)


def _hl_pack(q):                                          # q uint8 [..., 32] codes -> bytes [..., 16]: byte e = q[e] | q[16+e] << 4
    return (q[..., :16] | (q[..., 16:] << 4)).astype(np.uint8)


def _hl_unpack(b):
    return np.concatenate([b & 15, b >> 4], -1)


def q40_enc(w):
    rows, K = w.shape
    b = w.reshape(rows, K // 32, 32).astype(F32)
    idx = np.abs(b).argmax(-1)
    big = np.take_along_axis(b, idx[..., None], -1)[..., 0]
    d = (big / F32(-8.0)).astype(np.float16)
    return q40_assign(b, d)


def q40_assign(b, d):
    df = d.astype(F32)
    inv = np.where(df != 0, F32(1) / np.where(df != 0, df, F32(1)), F32(0)).astype(F32)
    q = np.clip(np.floor(b * inv[..., None] + F32(8.5)), 0, 15).astype(np.uint8)
    return _hl_pack(q).reshape(b.shape[0], -1), d


def q40_dec(qs, d, K):
    rows = qs.shape[0]
    q = _hl_unpack(qs.reshape(rows, K // 32, 16)).astype(F32)
    return ((q - F32(8)) * d.astype(F32)[..., None]).reshape(rows, K)


def iq4_nearest(b, d):
    df = d.astype(F32)
    inv = np.where(df != 0, F32(1) / np.where(df != 0, df, F32(1)), F32(0)).astype(F32)
    return np.abs((b * inv[..., None])[..., None] - KV).argmin(-1).astype(np.uint8)


def iq4_enc(w):
    rows, K = w.shape
    b = w.reshape(rows, K // 32, 32).astype(F32)
    d0 = (np.abs(b).max(-1) / F32(127.0)).astype(np.float16)
    q = iq4_nearest(b, d0)
    kv = KV[q]
    num, den = (b * kv).sum(-1), (kv * kv).sum(-1)
    d = np.where(den > 0, num / np.where(den > 0, den, 1), 0).astype(np.float16)
    d = np.where(d == 0, d0, d).astype(np.float16)
    return iq4_assign(b, d)


def iq4_assign(b, d):
    return _hl_pack(iq4_nearest(b, d)).reshape(b.shape[0], -1), d


def iq4_dec(qs, d, K):
    rows = qs.shape[0]
    q = _hl_unpack(qs.reshape(rows, K // 32, 16))
    return (KV[q] * d.astype(F32)[..., None]).reshape(rows, K)


def nv_assign(g, E):                                      # g [rows, K/16, 16] f32, E uint8 [rows, K/16] -> qs, meta
    sc = np.exp2(E.astype(F32) - 127)
    a = np.abs(g) / sc[..., None]
    mag = np.abs(a[..., None] - FP4).argmin(-1).astype(np.uint8)
    code = (mag | (np.signbit(g).astype(np.uint8) << 3)).reshape(g.shape[0], -1)
    qs = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    e2 = E.reshape(E.shape[0], -1, 2).astype(np.uint16)
    return qs, (e2[..., 0] | (e2[..., 1] << 8)).astype(np.uint16)


def nv_enc(w):
    rows, K = w.shape
    g = w.reshape(rows, K // 16, 16).astype(F32)
    am = np.abs(g).max(-1)
    e = np.where(am > 0, np.ceil(np.log2(np.where(am > 0, am, F32(1)) / F32(6.0))), 0).astype(np.int32)
    E = np.clip(e + 127, 1, 254).astype(np.uint8)
    return nv_assign(g, E)


def nv_dec(qs, meta, K):
    rows = qs.shape[0]
    c = np.empty((rows, K), np.uint8)
    c[:, 0::2], c[:, 1::2] = qs & 15, qs >> 4
    E = np.stack([meta & 255, meta >> 8], -1).reshape(rows, K // 16)
    v = FP4[c & 7].reshape(rows, K // 16, 16) * np.exp2(E.astype(F32) - 127)[..., None]
    return np.where((c >> 3).reshape(rows, K // 16, 16) == 1, -v, v).reshape(rows, K).astype(F32)


def q4k_pack_scales(sc, mn):                              # uint8 [rows, nsb, 8] each (6-bit) -> uint8 [rows, nsb, 12] (llama.cpp layout)
    s = np.zeros(sc.shape[:-1] + (12,), np.uint8)
    s[..., 0:4] = sc[..., 0:4] | ((sc[..., 4:8] >> 4) << 6)
    s[..., 4:8] = mn[..., 0:4] | ((mn[..., 4:8] >> 4) << 6)
    s[..., 8:12] = (sc[..., 4:8] & 15) | ((mn[..., 4:8] & 15) << 4)
    return s


def q4k_unpack_scales(s):                                 # inverse; the same formulas as k4b / q4k_sm in the Metal header
    sc, mn = np.empty(s.shape[:-1] + (8,), np.uint8), np.empty(s.shape[:-1] + (8,), np.uint8)
    sc[..., :4], mn[..., :4] = s[..., 0:4] & 63, s[..., 4:8] & 63
    sc[..., 4:] = (s[..., 8:12] & 15) | ((s[..., 0:4] >> 6) << 4)
    mn[..., 4:] = (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)
    return sc, mn


def q4k_meta(sc, mn, d, dm):                              # -> uint32 [rows, nsb*4]: 3 words of scales, then half2(d, dmin)
    s = q4k_pack_scales(sc, mn)
    w = np.ascontiguousarray(s).view("<u4")               # [rows, nsb, 3]
    dd = d.astype(np.float16).view(np.uint16).astype(np.uint32) | (dm.astype(np.float16).view(np.uint16).astype(np.uint32) << 16)
    return np.concatenate([w, dd[..., None]], -1).reshape(sc.shape[0], -1).astype(np.uint32)


def q4k_enc(w):
    rows, K = w.shape
    sb = w.reshape(rows, K // 256, 8, 32).astype(F32)
    lo = np.minimum(sb.min(-1), 0)
    scale = (sb.max(-1) - lo) / F32(15.0)
    m = -lo
    d = (scale.max(-1) / F32(63.0)).astype(np.float16)
    dm = (m.max(-1) / F32(63.0)).astype(np.float16)
    df, dmf = d.astype(F32)[..., None], dm.astype(F32)[..., None]
    sc = np.clip(np.rint(np.where(df > 0, scale / np.where(df > 0, df, 1), 0)), 0, 63).astype(np.uint8)
    mn = np.clip(np.rint(np.where(dmf > 0, m / np.where(dmf > 0, dmf, 1), 0)), 0, 63).astype(np.uint8)
    return q4k_assign(sb, sc, mn, d, dm)


def q4k_assign(sb, sc, mn, d, dm):
    rows, nsb = sb.shape[:2]
    sd = d.astype(F32)[..., None] * sc.astype(F32)
    off = dm.astype(F32)[..., None] * mn.astype(F32)
    q = np.clip(np.rint(np.where(sd[..., None] > 0, (sb + off[..., None]) / np.where(sd > 0, sd, 1)[..., None], 0)), 0, 15).astype(np.uint8)   # [rows, nsb, 8, 32]
    qg = q.reshape(rows, nsb, 4, 2, 32)
    qs = (qg[:, :, :, 0] | (qg[:, :, :, 1] << 4)).astype(np.uint8)                # [rows, nsb, 4, 32]: byte l of group g = elem 64g+l | elem 64g+32+l << 4
    return qs.reshape(rows, nsb * 128), q4k_meta(sc, mn, d, dm)


def q4k_dec(qs, meta, K):
    rows, nsb = qs.shape[0], K // 256
    q = qs.reshape(rows, nsb, 4, 32)
    q = np.stack([q & 15, q >> 4], 3).reshape(rows, nsb, 8, 32).astype(F32)
    m4 = np.ascontiguousarray(meta.reshape(rows, nsb, 4))
    sc, mn = q4k_unpack_scales(np.ascontiguousarray(m4[..., :3]).view(np.uint8))
    hd = np.ascontiguousarray(m4[..., 3]).view(np.float16).reshape(rows, nsb, 2).astype(F32)
    w = hd[..., 0, None, None] * sc.astype(F32)[..., None] * q - hd[..., 1, None, None] * mn.astype(F32)[..., None]
    return w.reshape(rows, K).astype(F32)


ENC = {"nvfp4": (nv_enc, nv_dec), "q4_0": (q40_enc, q40_dec), "q4_k": (q4k_enc, q4k_dec), "iq4_nl": (iq4_enc, iq4_dec)}
NAMES = {"nvfp4": "NVFP4", "q4_0": "Q4_0", "q4_k": "Q4_K", "iq4_nl": "IQ4_NL"}


def synth_weights(rows, K, seed=0):
    return np.random.default_rng(seed).standard_normal((rows, K)).astype(F32)


def act_quant_ref(x):                                     # x f16 [R, K] -> (xq int [R, K], xs f32 [R, K/32]) same arithmetic as QUANT_SRC (float32)
    R, K = x.shape
    b = x.astype(F32).reshape(R, K // 32, 32)
    am = np.abs(b).max(-1)
    inv = np.where(am > 0, F32(127.0) / np.where(am > 0, am, 1), F32(0)).astype(F32)
    q = np.clip(np.rint(b * inv[..., None]), -127, 127).astype(np.int32)
    return q.reshape(R, K), (am / F32(127.0)).astype(F32)


def reference_y(x, wdec, ints=False):
    """float64 reference y [R, rows] = x @ W.T; with ints the activations are first quantised as the int8 kernels do."""
    xf = x.astype(np.float64)
    if ints:
        q, s = act_quant_ref(x)
        xf = (q.reshape(x.shape[0], -1, 32) * s[..., None].astype(np.float64)).reshape(x.shape)
    return xf @ wdec.astype(np.float64).T


# ----------------------------------------------------------------------------------------------------------- self test (CPU, no mlx)
def selftest(verbose=True):
    """Round trips of every encoder / decoder / packer, and a decode of known codes. Returns a list of failure strings (empty = ok)."""
    bad = []
    rng = np.random.default_rng(1)
    K, rows = 512, 6
    w = rng.standard_normal((rows, K)).astype(F32)
    w[1, :64] = 0                                          # zero blocks
    w[2, 32:64] *= 100                                     # different dynamic range per block
    # Q4_0: bound |err| <= |d| (max element is exactly -8d, others within half a step, +1 step slack for the clip at 15), re-quantising the decode is the identity
    qs, d = q40_enc(w); wd = q40_dec(qs, d, K)
    err = np.abs(wd - w).reshape(rows, -1, 32).max(-1)
    if not (err <= np.abs(d.astype(F32)) * 1.0001 + 1e-6).all(): bad.append("q4_0 error bound")
    qs2, d2 = q40_assign(wd.reshape(rows, -1, 32), d)
    if not (np.array_equal(qs2, qs) and np.array_equal(d2, d)): bad.append("q4_0 requantise not identity")
    # IQ4_NL: decode is the nearest table point for the stored scale; assign(decode) is the identity
    qs, d = iq4_enc(w); wd = iq4_dec(qs, d, K)
    b = w.reshape(rows, -1, 32); dd = d.astype(F32)[..., None]
    best = np.abs(b[..., None] - dd[..., None] * KV).min(-1)
    if not np.allclose(np.abs(wd.reshape(rows, -1, 32) - b), best, atol=1e-5 * (1 + np.abs(b).max())): bad.append("iq4_nl not nearest table point")
    qs2, _ = iq4_assign(wd.reshape(rows, -1, 32), d)
    if not np.array_equal(qs2, qs): bad.append("iq4_nl requantise not identity")
    if not (np.abs(wd - w).reshape(rows, -1, 32).max(-1) <= 24 * np.abs(d.astype(F32)) * 0.5 + 15 * np.abs(d.astype(F32)) * (np.abs(b).max(-1) > 113 * np.abs(d.astype(F32))) + 1e-6).all():
        bad.append("iq4_nl error bound")
    # NVFP4: decoded values are exactly code * 2^(E-127); decode of the encoder output re-encodes to the same bytes
    qs, meta = nv_enc(w); wd = nv_dec(qs, meta, K)
    E = np.stack([meta & 255, meta >> 8], -1).reshape(rows, -1)
    qs2, meta2 = nv_assign(wd.reshape(rows, -1, 16), E.astype(np.uint8))
    if not (np.array_equal(qs2, qs) and np.array_equal(meta2, meta)): bad.append("nvfp4 requantise not identity")
    mags = np.abs(wd.reshape(rows, -1, 16)) / np.exp2(E.astype(F32) - 127)[..., None]
    if not np.isin(mags, FP4).all(): bad.append("nvfp4 decoded value outside E2M1 set")
    if not (np.abs(wd - w).reshape(rows, -1, 16).max(-1) <= 1.0001 * np.exp2(E.astype(F32) - 127) * 1.0 + 1e-6).all(): bad.append("nvfp4 error bound")  # max gap 2 in units of scale -> half-gap 1
    # E2M1 table, all 16 codes (sign bit 3)
    c = np.arange(16, dtype=np.uint8); qsx = (c[0::2] | (c[1::2] << 4)).astype(np.uint8)[None]
    got = nv_dec(np.tile(qsx, (1, 2)), np.full((1, 1), 127 | (127 << 8), np.uint16), 32)[0, :16]
    exp = np.concatenate([FP4, -FP4])
    if not np.array_equal(got, exp): bad.append("nvfp4 code table")
    # Q4_K: scale packing round trip over all 6-bit values, decode from packed arrays, requantise is the identity
    s6 = rng.integers(0, 64, (5, 3, 8), dtype=np.uint8); m6 = rng.integers(0, 64, (5, 3, 8), dtype=np.uint8)
    a, bb = q4k_unpack_scales(q4k_pack_scales(s6, m6))
    if not (np.array_equal(a, s6) and np.array_equal(bb, m6)): bad.append("q4_k scale packing")
    qs, meta = q4k_enc(w); wd = q4k_dec(qs, meta, K)
    m4 = meta.reshape(rows, -1, 4)
    sc, mn = q4k_unpack_scales(np.ascontiguousarray(m4[..., :3]).view(np.uint8))
    hd = np.ascontiguousarray(m4[..., 3]).view(np.float16).reshape(rows, -1, 2)
    sb = w.reshape(rows, -1, 8, 32)
    qs2, meta2 = q4k_assign(wd.reshape(rows, -1, 8, 32), sc, mn, hd[..., 0], hd[..., 1])
    # re-encode of the decode with the same scales: the codes stay (up to the clip of values that were outside the block range)
    if (qs2 != qs).mean() > 0.001 or not np.array_equal(meta2, meta): bad.append("q4_k requantise")
    step = (hd[..., 0].astype(F32)[..., None] * sc.astype(F32))[..., None]
    if not (np.abs(wd.reshape(rows, -1, 8, 32) - sb) <= step * 0.5001 + 1e-5 + np.abs(sb) * 0.0).all():
        # blocks whose max < 0 can clip at 15: allow the bound only where the block range is represented
        ok = (np.abs(wd.reshape(rows, -1, 8, 32) - sb) <= step * 1.5 + 1e-5).all()
        if not ok: bad.append("q4_k error bound")
    # bytes per weight
    for name, (e, _) in ENC.items():
        q, mt = e(w)
        bpw = (q.nbytes + mt.nbytes) / w.size
        if abs(bpw - 0.5625) > 1e-9: bad.append(f"{name} bytes/weight {bpw}")
    # int8 activation quantiser: round trip error <= scale / 2
    x = rng.standard_normal((3, 256)).astype(np.float16); q, s = act_quant_ref(x)
    xd = (q.reshape(3, -1, 32) * s[..., None]).reshape(3, 256)
    if not (np.abs(xd - x.astype(F32)) <= (s.repeat(32, 1) * 0.5001 + 1e-7)).all(): bad.append("act quant")
    # kernel logic (unit decode + x mapping, python emulation) == numpy decoder, all four formats
    Ke = 1024
    we = rng.standard_normal((3, Ke)).astype(F32); xe = rng.standard_normal((4, Ke)).astype(np.float16)
    for name, (e, dcd) in ENC.items():
        q, mt = e(we)
        ye, yr = emulate_gemv(name, q, mt, xe, Ke), reference_y(xe, dcd(q, mt, Ke))
        if np.abs(ye - yr).max() > 1e-4 * np.abs(yr).max():
            bad.append(f"{name} kernel-logic emulation differs from decoder ({np.abs(ye - yr).max():.2e})")
    if verbose:
        print("selftest:", "OK" if not bad else "FAIL " + "; ".join(bad))
    return bad


# ----------------------------------------------------------------------------------------------------------- CPU emulation of the kernels' unit decode and x mapping
def _fp4v(c):                                             # the bit trick of fp4v() in the Metal header, in scalar form
    v = c & 7
    f = np.array([(v << 22) + 0x3F000000], np.uint32).view(F32)[0]
    f = F32(0.5 * v) if v < 2 else f
    return np.array([int(np.array([f], F32).view(np.uint32)[0]) | ((c & 8) << 28)], np.uint32).view(F32)[0]


def emulate_gemv(fmt, qs, meta, x, K):
    """y [R, rows] computed unit by unit exactly as gemv_source does: 16-byte units, lane -> (chunk, ul) -> x offsets xa / xb, the header's decode formulas.
    Checks the storage layout / index arithmetic of the Metal code against the numpy decoder on the CPU (slow python loops: small shapes only)."""
    rows, R = qs.shape[0], x.shape[0]
    UPR = K // 32
    q32 = np.ascontiguousarray(qs).view("<u4").reshape(rows, UPR, 4)
    y = np.zeros((R, rows), np.float64)
    xf = x.astype(np.float64)
    for n in range(rows):
        for u in range(UPR):
            ul, base = u & 15, (u >> 4) * 512
            wd = [int(v) for v in q32[n, u]]
            by = [(wd[i] >> (8 * b)) & 255 for i in range(4) for b in range(4)]          # byte e = word e/4 byte e%4
            if fmt in ("q4_0", "iq4_nl", "q4_k"):
                if fmt == "q4_k":
                    m = [int(v) for v in meta.reshape(rows, -1, 4)[n, u >> 3]]
                    k4b = lambda k: ((m[0] if k < 4 else (m[1] if k < 8 else m[2])) >> ((k & 3) * 8)) & 255
                    def sm(j):
                        if j < 4: return float(k4b(j) & 63), float(k4b(j + 4) & 63)
                        return float((k4b(j + 4) & 15) | ((k4b(j - 4) >> 6) << 4)), float((k4b(j + 4) >> 4) | ((k4b(j) >> 6) << 4))
                    hd = np.array([m[3]], np.uint32).view(np.float16).astype(F32)
                    d, dm = hd[0], hd[1]
                    ja = ((ul & 7) >> 1) * 2
                    (sa, ma), (sb_, mb) = sm(ja), sm(ja + 1)
                    Ad, Am, Bd, Bm = d * sa, dm * ma, d * sb_, dm * mb
                    xa = base + (ul >> 3) * 256 + ((ul & 7) >> 1) * 64 + (ul & 1) * 16; xb = xa + 32
                else:
                    d = F32(meta[n, u]); Ad = Bd = d; Am = Bm = F32(8) * d if fmt == "q4_0" else F32(0)
                    xa = base + ul * 32; xb = xa + 16
                if fmt == "iq4_nl":
                    w = np.array([KV[b & 15] * d for b in by] + [KV[b >> 4] * d for b in by], F32)
                else:
                    w = np.array([F32(b & 15) * Ad - Am for b in by] + [F32(b >> 4) * Bd - Bm for b in by], F32)
            else:                                                                           # nvfp4
                e2 = int(meta[n, u])
                s0, s1 = np.exp2(F32((e2 & 255) - 127)), np.exp2(F32(((e2 >> 8) & 255) - 127))
                w = np.array([_fp4v((wd[i] >> (4 * t)) & 15) * (s0 if i < 2 else s1) for i in range(4) for t in range(8)], F32)
                xa = base + ul * 32; xb = xa + 16
            xv = np.concatenate([xf[:, xa:xa + 16], xf[:, xb:xb + 16]], 1)
            y[:, n] += xv @ w.astype(np.float64)
    return y


# ----------------------------------------------------------------------------------------------------------- timing helpers
class Clock:
    def __init__(self, budget):
        self.t0 = time.perf_counter(); self.budget = budget

    def left(self):
        return self.budget - (time.perf_counter() - self.t0)

    def elapsed(self):
        return time.perf_counter() - self.t0


def synth_exl3(EL, mx, kin, nout, K, seed):
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16)
    f16 = lambda n: (rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16)
    suh, svh = f16(kin), f16(nout)
    return EL.Exl3Linear(mx.array(tr), mx.array(suh), mx.array(svh), cb="mul1"), (tr, suh, svh)


_CLOCK = None


def chk_mem(mx, limit=0.93 * GIB):
    g = getattr(mx, "get_active_memory", None) or getattr(mx.metal, "get_active_memory", None)
    if g and g() > limit:
        raise RuntimeError(f"active GPU memory {g() / 2**20:.0f} MiB over the budget")


def time_calls(mx, fns, reps, chain):
    """fns: list of thunks (each one product with its own weight buffer). Warm-up, then `reps` evals of `chain` calls cycling through fns, graph built
    outside the timer. Returns (min_s, median_s) per product."""
    mx.eval([f() for f in fns[:2]])
    ts = []
    for _ in range(reps):
        if _CLOCK is not None and _CLOCK.left() < 1.0:
            if not ts:
                raise TimeoutError("time budget")
            break
        outs = [fns[i % len(fns)]() for i in range(chain)]
        t0 = time.perf_counter()
        mx.eval(outs)
        ts.append((time.perf_counter() - t0) / chain)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def errtxt(e, n=1500):
    """Whole exception text (MLX puts the compiler's `error:` lines after the first line), capped."""
    t = f"{type(e).__name__}: {e}"
    ls = [l for l in t.splitlines() if "error" in l.lower()] or t.splitlines()[:3]
    return " | ".join(ls)[:n]


class HardStop(BaseException):
    pass


def fmt_row(label, bpw, gbs, ms, rel, note=""):
    return f"  {label:<22s} {bpw:7.4f} {gbs:9.1f} {ms:10.4f} {rel:8.3f}  {note}"


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="kin", type=int, default=4096, help="input features (multiple of 512)")
    ap.add_argument("--out", type=int, default=12288, help="output rows (multiple of 256)")
    ap.add_argument("--rows", default="1,4,16")
    ap.add_argument("--pool", type=int, default=8, help="distinct weight buffers cycled (reduced automatically to fit the memory limit)")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--chain", type=int, default=32, help="products per timed eval")
    ap.add_argument("--budget", type=float, default=80.0, help="seconds; stages are skipped when exceeded (hard limit is 90)")
    ap.add_argument("--cpu-only", action="store_true", help="only run the CPU round-trip tests and the reference (no mlx needed)")
    ap.add_argument("--no-exl3", action="store_true")
    ap.add_argument("--native", action="store_true", help="also time the kernels of kyojin_mlx/foreign_linear.py (real NVFP4 = UE4M3 scale + global scale, MXFP4, GGUF-layout Q4_0 / Q4_K / IQ4_NL): gemv nbr1/2 and the simdgroup_matrix kernel ntn1/2, "
                    "checked against float64 first; prints the best variant per row count, which sets KYOJIN_FOREIGN_GEMV_ROWS / _MMA_MAX / _NBR / _NTN")
    ap.add_argument("--native-only", action="store_true", help="with --native: skip the legacy gate kernels (E8M0 NVFP4, int8 variants)")
    ap.add_argument("--dump-msl", default="", help="directory: write every generated kernel body (header + source) there and exit, for xcrun metal")
    a = ap.parse_args()
    global _CLOCK
    clock = _CLOCK = Clock(a.budget)
    bad = selftest()
    if a.cpu_only:
        return 1 if bad else 0
    if a.dump_msl:
        os.makedirs(a.dump_msl, exist_ok=True)
        for f, vs in M.VARIANTS.items():
            for v in vs:
                with open(os.path.join(a.dump_msl, f"{f}_{v}.metal.txt"), "w") as fh:
                    fh.write("// header\n" + M.header(f, v) + "\n// body (template ints R, KIN, N, NBR)\n" + M.gemv_source(f, v, 2))
        print("written to", a.dump_msl); return 0
    if bad:
        print("CPU encoder tests failed: not timing anything"); return 1
    KIN, N = a.kin, a.out
    Rs = [int(v) for v in a.rows.split(",")]
    assert KIN % 512 == 0 and N % TILE == 0 and N % 16 == 0 and max(Rs) <= 16 and len(Rs) <= 4 and a.pool >= 2 and a.reps <= 50 and a.chain <= 128

    import mlx.core as mx
    try:
        mx.set_cache_limit(128 << 20)
    except Exception:
        pass
    getpeak = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
    print(f"matrix {N} x {KIN} = {N * KIN / 1e6:.1f} M weights, rows {Rs}, pool<= {a.pool}, reps {a.reps}, chain {a.chain}")
    try:
        di = mx.metal.device_info()
        print("device:", {k: di[k] for k in ("device_name", "architecture", "memory_size", "max_threadgroup_memory_length") if k in di})
    except Exception as e:
        print("device info unavailable:", e)

    xr = np.random.default_rng(7).standard_normal((max(Rs), KIN)).astype(np.float16)
    xs = {R: mx.array(xr[:R]) for R in Rs}
    results = {}                                           # (label) -> dict(bpw, bytes, t={R: (min, med)}, note)
    notes = []
    peak_bytes = 0

    def pool_size(per_bytes, factor):
        P = min(a.pool, int(0.6 * GIB / (per_bytes * factor)))
        if P < 2:
            raise RuntimeError("matrix too large for the 1 GiB buffer limit")
        return P

    # ---- EXL3 (the repo's kernels)
    def run_exl3(K):
        nonlocal peak_bytes
        import kyojin_mlx.exl3_linear as EL
        from kyojin_mlx import ref_numpy as RN
        label = f"EXL3 {K}-bit"
        per = N * KIN * K // 8 + 2 * (N + KIN)
        P = pool_size(per, 2.0)
        # numeric check at a small shape against the repo's numpy reference
        try:
            lc, (tr, suh, svh) = synth_exl3(EL, mx, 512, 512, K, 99)
            worst = 0.0
            for R in Rs:
                xc = xr[:R, :512]
                y = np.array(lc(mx.array(xc))).astype(np.float64).reshape(R, -1)
                yr = RN.gemv(xc, tr, suh, svh)
                worst = max(worst, np.abs(y - yr).max() / (np.abs(yr).max() + 1e-12))
            ok = worst < 3e-2
            print(f"[{label}] check vs ref_numpy.gemv (512x512, rows {Rs}): max rel err {worst:.2e} {'ok' if ok else 'MISMATCH'}")
            if not ok:
                notes.append(f"{label}: numeric check MISMATCH {worst:.2e}")
        except Exception as e:
            print(f"[{label}] check skipped: {type(e).__name__}: {e}"); notes.append(f"{label}: check error {type(e).__name__}")
        ls = [synth_exl3(EL, mx, KIN, N, K, 10 + i)[0] for i in range(P)]
        mx.eval(*[l._words for l in ls])
        chk_mem(mx)
        wb = int(ls[0]._g[1].words.size * 4 + ls[0]._suh.size * 2 + ls[0]._svh.size * 2)
        t = {}
        for R in Rs:
            t[R] = time_calls(mx, [(lambda l=l, R=R: l(xs[R])) for l in ls], a.reps, a.chain)
        results[label] = dict(bpw=wb / (N * KIN), bytes=wb, t=t, note=f"pool {P}")
        if getpeak: peak_bytes = max(peak_bytes, getpeak())
        del ls
        mx.clear_cache()

    # ---- foreign formats
    def run_foreign(fmt):
        nonlocal peak_bytes
        label = NAMES[fmt]
        enc, dec = ENC[fmt]
        wt = synth_weights(TILE, KIN, seed=3)
        qs, meta = enc(wt)
        wdec = dec(qs, meta, KIN)
        qs_full = np.ascontiguousarray(np.tile(qs, (N // TILE, 1))).view(np.uint32)
        meta_full = np.ascontiguousarray(np.tile(meta, (N // TILE, 1)))
        wb = qs_full.nbytes + meta_full.nbytes
        P = pool_size(wb, 1.2)
        qd = [mx.array(qs_full) for _ in range(P)]
        md = [mx.array(meta_full) for _ in range(P)]
        mx.eval(*qd, *md)
        chk_mem(mx)
        del qs_full, meta_full
        quant = mx.fast.metal_kernel(name="fg_actq", input_names=["x"], output_names=["xq", "xsa"], source=M.QUANT_SRC)
        refs = {False: reference_y(xr, wdec, False), True: reference_y(xr, wdec, True)}
        good = {}                                            # (var, nbr) -> {R: (min, med)}
        for var in M.VARIANTS[fmt]:
            ints = var == "i"
            for nbr in (2, 1):
                if clock.left() < 8:
                    notes.append(f"{label}/{var}/nbr{nbr}: skipped (time budget)"); continue
                try:
                    kern = mx.fast.metal_kernel(name=f"fg_{fmt}_{var}_{nbr}", input_names=M.gemv_inputs(fmt, var), output_names=["y"],
                                                source=M.gemv_source(fmt, var, nbr), header=M.header(fmt, var))
                    ntg = N // (8 * nbr)

                    def call(i, R, kern=kern, ints=ints, nbr=nbr, ntg=ntg):
                        if ints:
                            xq, xsa = quant(inputs=[xs[R]], grid=(R * KIN // 32, 1, 1), threadgroup=(64, 1, 1),
                                            output_shapes=[(R, KIN), (R, KIN // 32)], output_dtypes=[mx.int8, mx.float32])
                            ins = [xq, xsa, qd[i], md[i]]
                        else:
                            ins = [xs[R], qd[i], md[i]]
                        return kern(inputs=ins, template=[("R", R), ("KIN", KIN), ("N", N), ("NBR", nbr)], grid=(ntg * 128, 1, 1),
                                    threadgroup=(128, 1, 1), output_shapes=[(R, N)], output_dtypes=[mx.float16])[0]

                    tt = {}
                    okall = True
                    for R in Rs:
                        y = np.array(call(0, R)).astype(np.float64)
                        yr = refs[ints][:R]
                        err = np.abs(y.reshape(R, N // TILE, TILE) - yr[:, None, :]).max() / (np.abs(yr).max() + 1e-12)
                        tol = 8e-3 if ints else 3e-3
                        if not (err < tol):
                            okall = False
                            notes.append(f"{label}/{var}/nbr{nbr} R={R}: MISMATCH rel err {err:.2e} (tol {tol}) - excluded")
                            break
                    if not okall:
                        continue
                    for R in Rs:
                        tt[R] = time_calls(mx, [(lambda i=i, R=R: call(i, R)) for i in range(P)], a.reps, a.chain)
                    good[(var, nbr)] = tt
                except Exception as e:
                    notes.append(f"{label}/{var}/nbr{nbr}: {errtxt(e)}")
        for (var, nbr), tt in good.items():
            results[f"{label} [{var},nbr{nbr}]"] = dict(bpw=wb / (N * KIN), bytes=wb, t=tt, note="variant", fmt=label)
        if good:
            best = {R: min(good, key=lambda k: good[k][R][0]) for R in Rs}
            results[label] = dict(bpw=wb / (N * KIN), bytes=wb, t={R: good[best[R]][R] for R in Rs}, best=best, note=f"pool {P}", fmt=label)
        if getpeak: peak_bytes = max(peak_bytes, getpeak())
        del qd, md
        mx.clear_cache()

    native_res = {}

    def run_native(t):
        """kyojin_mlx.foreign_linear kernels on a synthetic GGUF-layout (or safetensors-layout) matrix, N x KIN tiled from 256 random rows."""
        nonlocal peak_bytes
        from kyojin_mlx import foreign_linear as FL, foreign_gguf as GF, foreign_pack as FP, foreign_synth as SY
        label = "native " + t
        rng = np.random.default_rng(5)
        raw = SY.random_blocks(t, TILE, KIN, rng)
        pk = GF.to_pack(t, raw, KIN)
        wdec = FP.dequant(pk).astype(np.float64)
        pk_full = FP.Pack(pk.fmt, N, KIN, np.ascontiguousarray(np.tile(pk.qs, (N // TILE, 1))), np.ascontiguousarray(np.tile(pk.meta, (N // TILE, 1))), pk.gscale)
        P = pool_size(pk_full.nbytes, 1.2)
        dps = [FL.DevPack(pk_full) for _ in range(P)]
        mx.eval(*[arr for d in dps for arr in d.arrays()])
        chk_mem(mx)
        ref = {R: xr[:R].astype(np.float64) @ wdec.T for R in Rs}
        good = {}
        for var, nb in (("gemv", 1), ("gemv", 2), ("mma", 1), ("mma", 2)):
            if clock.left() < 6:
                notes.append(f"{label}/{var}{nb}: skipped (time budget)"); continue
            try:
                if var == "gemv" and not dps[0].gemv_ok(nb):
                    continue
                call = (lambda d, R, nb=nb: d._gemv(xs[R], nb)) if var == "gemv" else (lambda d, R, nb=nb: d._mma(xs[R], nb))
                ok = True
                for R in Rs:
                    y = np.array(call(dps[0], R)).astype(np.float64)
                    err = np.abs(y.reshape(R, N // TILE, TILE) - ref[R][:, None, :]).max() / (np.abs(ref[R]).max() + 1e-12)
                    if not err < 3e-3:
                        notes.append(f"{label}/{var}{nb} R={R}: MISMATCH rel err {err:.2e} - excluded"); ok = False; break
                if ok:
                    good[f"{var}{nb}"] = {R: time_calls(mx, [(lambda d=d, R=R: call(d, R)) for d in dps], a.reps, a.chain) for R in Rs}
            except Exception as e:
                notes.append(f"{label}/{var}{nb}: {errtxt(e)}")
        # dequant + matmul reference path (what rows > KYOJIN_FOREIGN_DENSE_ROWS pay)
        try:
            if clock.left() > 6:
                good["deq+mm"] = {R: time_calls(mx, [(lambda d=d, R=R: (xs[R] @ d.dequant().T)) for d in dps], a.reps, 8) for R in Rs}
        except Exception as e:
            notes.append(f"{label}/deq+mm: {errtxt(e)}")
        native_res[t] = (pk_full.nbytes / (N * KIN), good)
        if getpeak: peak_bytes = max(peak_bytes, getpeak())
        del dps
        mx.clear_cache()

    stages = []
    if not a.no_exl3:
        stages.append(("EXL3 3-bit", lambda: run_exl3(3)))
    if a.native:
        stages += [("native " + t, (lambda t=t: run_native(t))) for t in ("NVFP4", "Q4_0", "Q4_K", "IQ4_NL", "MXFP4")]
    if not a.native_only:
        stages += [(NAMES[f], (lambda f=f: run_foreign(f))) for f in ("nvfp4", "q4_0", "q4_k", "iq4_nl")]
    if not a.no_exl3:
        stages += [("EXL3 2-bit", lambda: run_exl3(2)), ("EXL3 5-bit", lambda: run_exl3(5))]
    import signal

    def on_alarm(signum, frame):
        raise HardStop()
    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(int(a.budget) + 8)                       # hard stop (no thread): the partial table is still printed
    dur = {}
    try:
        for name, fn in stages:
            kind = "exl3" if name.startswith("EXL3") else "foreign"
            if clock.left() < max(6.0, 1.3 * dur.get(kind, 0.0)):
                notes.append(f"{name}: skipped (time budget, left {clock.left():.0f} s)"); continue
            t_st = clock.elapsed()
            try:
                fn()
            except Exception as e:
                notes.append(f"{name}: {errtxt(e)}")
            dur[kind] = max(dur.get(kind, 0.0), clock.elapsed() - t_st)
    except HardStop:
        notes.append("HARD STOP: 90 s alarm fired, table below is partial")
    signal.alarm(0)

    # ---- report
    base = results.get("EXL3 3-bit")
    order = ([] if a.no_exl3 else ["EXL3 2-bit", "EXL3 3-bit", "EXL3 5-bit"]) + ([] if a.native_only else [NAMES[f] for f in ("nvfp4", "q4_0", "q4_k", "iq4_nl")])      # rows of stages that were not asked for are not "no result"
    for R in Rs:
        print(f"\n=== {R} row{'s' if R > 1 else ''}:  {N} x {KIN} matrix-vector product  (rel = time / EXL3 3-bit time at {R} rows; < 1 is faster) ===")
        print(f"  {'encoding':<22s} {'B/weight':>7s} {'GB/s':>9s} {'ms/product':>10s} {'rel':>8s}  best variant")
        for lab in order:
            r = results.get(lab)
            if not r:
                print(f"  {lab:<22s} (no result)"); continue
            tmin = r["t"][R][0]
            rel = tmin / base["t"][R][0] if base else float("nan")
            bv = r.get("best", {}).get(R)
            print(fmt_row(lab, r["bpw"], r["bytes"] / tmin / 1e9, tmin * 1e3, rel, f"{bv[0]},nbr{bv[1]}" if bv else r["note"]))
    if native_res:
        print("\n=== native kernels (kyojin_mlx.foreign_linear): ms/product, rel = time / EXL3 3-bit at the same rows; the best variant per row count sets the KYOJIN_FOREIGN_* knobs ===")
        for R in Rs:
            b3 = base["t"][R][0] if base else None
            for t, (bpw, good) in native_res.items():
                cells = "  ".join(f"{k}: {v[R][0] * 1e3:8.4f}" for k, v in good.items())
                best = min(good, key=lambda k: good[k][R][0]) if good else "-"
                rel = f"{good[best][R][0] / b3:6.3f}" if good and b3 else "   n/a"
                print(f"  R={R:<3d} {t:<7s} {bpw:6.4f} B/w  best {best:<7s} rel {rel}   {cells}")
    print("\n--- all checked foreign variants (ms/product, min of reps) ---")
    for lab, r in results.items():
        if r["note"] == "variant":
            print(f"  {lab:<28s} " + "  ".join(f"R{R}: {r['t'][R][0] * 1e3:8.4f}" for R in Rs))
    print("\nEXL3 time includes its input / output Hadamard rotations (the foreign kernels do none); int8 variants include the activation quantiser launch.")
    for n in notes:
        print("NOTE:", n)
    if getpeak:
        pk = max(peak_bytes, getpeak())
        print(f"peak GPU memory {pk / 2**20:.0f} MiB (limit 1024)" + ("  LIMIT VIOLATION" if pk > GIB else ""))
    el = clock.elapsed()
    print(f"total {el:.1f} s" + ("  OVER 90 s" if el > 90 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
