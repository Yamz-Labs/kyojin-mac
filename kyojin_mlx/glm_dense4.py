# SPDX-License-Identifier: Apache-2.0
"""G4: dense MXFP8 -> 4-bit for the GLM-5.3-Flash FP4-Local checkpoint (attention / shared-expert / dense-MLP / lm_head linears, 1.0625 B per weight on the device today).

Target format: modelopt NVFP4, the layout the routed experts already have (E2M1 codes two per byte, one UE4M3 scale per 16 inputs, one fp32 per-tensor scale
`weight_scale_2`). foreign_linear reads that layout in place (Pack "nvfp4": MLX's quantized matmul for few rows, the fmma kernels above), so a converted
checkpoint needs no loader change and no resident copy: 0.5625 B per weight (-47 %). MXFP4 (one E8M0 scale per 32) is also writable (0.531 B per weight).

The quantizer is round-to-nearest with an exhaustive search of the block scale (a few E4M3 codes around amax / 6, squared-error minimum): the scale of a
4-bit block is the one free parameter that matters, and the grid has 8 magnitudes only. Everything runs through mlx (GPU or CPU stream), one chunk of rows at a time.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np
import mlx.core as mx

from . import foreign_pack as FP

E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)                   # midpoints of the magnitude grid
E4M3_POS = FP.E4M3_TAB[:127].copy()                                     # UE4M3 magnitudes, byte 0..126 (127 = NaN), monotonic in the byte
E4M3_MAX = float(E4M3_POS[-1])                                          # 448
NV_SEARCH = (-3, -2, -1, 0, 1, 2, 3)                                    # E4M3 codes around the amax scale tried per block (nvfp4)
NV_SEARCH_WIDE = tuple(range(-8, 9))                                    # G4b: -4.5 % rel RMSE (-9 % error energy) vs NV_SEARCH on every dense linear, same stored format (+-14 is no better)
MX_SEARCH = (-1, 0, 1)                                                  # E8M0 exponents around ceil(log2(amax / 6)) tried per block (mxfp4)
_GRID = None
_E4 = None


def _tabs():
    global _GRID, _E4
    if _GRID is None:
        _GRID, _E4 = mx.array(E2M1_POS), mx.array(E4M3_POS)
    return _GRID, _E4


# ---------------------------------------------------------------------------------------------------------------------------------- source side
def load_f32(ck, prefix: str) -> np.ndarray:
    """float32 [n, k] of one linear of the source checkpoint: E4M3 x (E8M0 per 32 inputs | fp32 128x128 block grid [n/bn, 1, k/bk, 1] | row | tensor), or a
    BF16 / F16 / F32 matrix as is. Same values as the loader's fp8b pack (checked by tools/g4_cutchk.py), without the 1-ULP fp16 rounding of its block scales."""
    tag, shp, w = ck.raw(prefix + ".weight")
    w = np.asarray(w)
    if tag == "BF16":
        return (w.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shp)
    if tag in ("F16", "F32"):
        return np.asarray(w, np.float32).reshape(shp)
    assert tag == "F8_E4M3", f"{prefix}: {tag}"
    n, k = shp
    W = FP.E4M3_TAB[w.view(np.uint8).reshape(n, k)]
    st, ss, sv = ck.raw(prefix + ".weight_scale")
    sv = np.asarray(sv)
    sc = np.exp2(sv.view(np.uint8).astype(np.float32) - 127) if st in ("U8", "F8_E8M0") else np.asarray(ck.scale_f32(prefix + ".weight_scale"), np.float32)
    sc = np.squeeze(sc.reshape(ss) if sc.size == int(np.prod(ss)) else sc)
    if sc.ndim == 0:
        return W * sc
    if sc.ndim == 1:
        return W * sc[:, None]
    return W * np.repeat(np.repeat(sc, n // sc.shape[0], 0), k // sc.shape[1], 1)[:n, :k]


def load_f32_rows(ck, prefix: str, r0: int, r1: int) -> np.ndarray:
    """Rows [r0, r1) of load_f32 (same values) without materialising the whole matrix: the converter and the CPU analysis run under a small RSS budget."""
    tag, shp, w = ck.raw(prefix + ".weight")
    n, k = shp
    w = np.asarray(w)
    if tag == "BF16":
        return (w.view(np.uint16).reshape(n, k)[r0:r1].astype(np.uint32) << 16).view(np.float32)
    if tag in ("F16", "F32"):
        return np.asarray(w.reshape(n, k)[r0:r1], np.float32)
    assert tag == "F8_E4M3", f"{prefix}: {tag}"
    W = FP.E4M3_TAB[w.view(np.uint8).reshape(n, k)[r0:r1]]
    st, ss, sv = ck.raw(prefix + ".weight_scale")
    sv = np.asarray(sv)
    sc = np.exp2(sv.view(np.uint8).astype(np.float32) - 127) if st in ("U8", "F8_E8M0") else np.asarray(ck.scale_f32(prefix + ".weight_scale"), np.float32)
    sc = np.squeeze(sc.reshape(ss) if sc.size == int(np.prod(ss)) else sc)
    if sc.ndim == 0:
        return W * sc
    if sc.ndim == 1:
        return W * sc[r0:r1, None]
    bn = n // sc.shape[0]
    rows = np.arange(r0, r1) // bn
    return W * np.repeat(sc[rows], k // sc.shape[1], 1)[:, :k]


# ---------------------------------------------------------------------------------------------------------------------------------- E2M1 rounding
def _round_e2m1(v):
    """float32 scaled values -> (E2M1 4-bit codes uint8 with the sign in bit 3, dequantised magnitudes*sign float32). Nearest, ties go to the smaller magnitude."""
    grid, _ = _tabs()
    a = mx.abs(v)
    idx = (a > E2M1_BOUNDS[0]).astype(mx.uint8)
    for b in E2M1_BOUNDS[1:]:
        idx = idx + (a > b).astype(mx.uint8)
    neg = v < 0
    deq = mx.take(grid, idx.astype(mx.int32))
    code = idx | (mx.where(neg & (idx > 0), 8, 0)).astype(mx.uint8)
    return code, mx.where(neg, -deq, deq)


def pack_nibbles(codes: mx.array) -> mx.array:
    """codes uint8 [r, k] (0..15) -> bytes [r, k/2], element 2b in the low nibble, 2b+1 in the high one (the stored layout)."""
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


# ---------------------------------------------------------------------------------------------------------------------------------- NVFP4
def global_scale(amax: float) -> float:
    """weight_scale_2: the largest block scale (amax / 6) lands on the largest E4M3 magnitude."""
    return float(amax) / (6.0 * E4M3_MAX) if amax > 0 else 1.0


def e4m3_nearest_byte(t: mx.array) -> mx.array:
    """UE4M3 byte (int32, 0..127) whose magnitude is nearest to t >= 0 (closed form: no [.., 127] distance table, which cost 0.9 GB of temporaries per 4M weights).
    Normal range (>= 2^-6): exponent field floor(log2 t) + 7, mantissa round((t / 2^e - 1) * 8), a carry rolls into the next exponent; below it: round(t * 512)."""
    e = mx.floor(mx.log2(mx.maximum(t, 1e-30)))
    m = mx.round((t / mx.power(2.0, e) - 1.0) * 8.0)
    normal = (e + 7.0) * 8.0 + m
    sub = mx.round(t * 512.0)
    return mx.clip(mx.where(t >= 2.0 ** -6, normal, sub), 0, 127).astype(mx.int32)


def quant_nvfp4_rows(w: mx.array, g: float, search=NV_SEARCH, h=None):
    """w float32 [r, k] -> (qs uint8 [r, k/2], scale bytes uint8 [r, k/16] (UE4M3), dequantised float32 [r, k] without the global scale g).
    h: optional float32 [k] input-channel weights (mean squared activation of the input channel, G4b): the block scale is then the one that minimises sum_j h_j (w_j - q_j)^2,
    the output error for uncorrelated inputs, instead of the plain squared error (the rounding of each element to the grid is unchanged: nearest is optimal for any h > 0)."""
    r, k = w.shape
    _, e4 = _tabs()
    wb = w.reshape(r, k // 16, 16)
    hb = None if h is None else mx.array(np.asarray(h, np.float32)).reshape(1, k // 16, 16)
    amax = mx.max(mx.abs(wb), axis=-1)
    t = amax / (6.0 * g)
    b0 = e4m3_nearest_byte(t)
    best_err = best_code = best_deq = best_byte = None
    for d in search:
        byte = mx.clip(b0 + d, 1, 126)
        s = mx.take(e4, byte) * g
        code, deq = _round_e2m1(wb / s[..., None])
        deq = deq * s[..., None]
        err = mx.sum((wb - deq) ** 2 if hb is None else hb * (wb - deq) ** 2, axis=-1)
        if best_err is None:
            best_err, best_code, best_deq, best_byte = err, code, deq, byte
        else:
            m = err < best_err
            best_err = mx.where(m, err, best_err)
            best_code = mx.where(m[..., None], code, best_code)
            best_deq = mx.where(m[..., None], deq, best_deq)
            best_byte = mx.where(m, byte, best_byte)
        mx.eval(best_err, best_code, best_deq, best_byte)                   # one candidate's temporaries alive at a time
    return pack_nibbles(best_code.reshape(r, k)), best_byte.astype(mx.uint8), best_deq.reshape(r, k)


def dequant_nvfp4(qs, sc, g: float) -> mx.array:
    """The stored layout back to float32 (reference): qs uint8 [n, k/2], sc UE4M3 bytes [n, k/16], g the per-tensor scale."""
    grid, e4 = _tabs()
    qs, sc = mx.array(qs), mx.array(sc)
    n = qs.shape[0]
    c = mx.stack([qs & 15, qs >> 4], axis=-1).reshape(n, -1).astype(mx.int32)
    v = mx.take(grid, c & 7) * mx.where(c >= 8, -1.0, 1.0)
    s = mx.take(e4, sc.astype(mx.int32)) * g
    return (v.reshape(n, -1, 16) * s[..., None]).reshape(n, -1)


# ---------------------------------------------------------------------------------------------------------------------------------- MXFP4
def quant_mxfp4_rows(w: mx.array, search=MX_SEARCH):
    """w float32 [r, k] -> (qs uint8 [r, k/2], E8M0 bytes uint8 [r, k/32], dequantised float32 [r, k])."""
    r, k = w.shape
    wb = w.reshape(r, k // 32, 32)
    amax = mx.max(mx.abs(wb), axis=-1)
    e0 = mx.ceil(mx.log2(mx.maximum(amax, 1e-30) / 6.0))
    best_err = best_code = best_deq = best_byte = None
    for d in search:
        byte = mx.clip(e0 + d + 127.0, 1.0, 254.0)
        s = mx.power(2.0, byte - 127.0)
        code, deq = _round_e2m1(wb / s[..., None])
        deq = deq * s[..., None]
        err = mx.sum((wb - deq) ** 2, axis=-1)
        if best_err is None:
            best_err, best_code, best_deq, best_byte = err, code, deq, byte
        else:
            m = err < best_err
            best_err = mx.where(m, err, best_err)
            best_code = mx.where(m[..., None], code, best_code)
            best_deq = mx.where(m[..., None], deq, best_deq)
            best_byte = mx.where(m, byte, best_byte)
        mx.eval(best_err, best_code, best_deq, best_byte)                   # one candidate's temporaries alive at a time
    return pack_nibbles(best_code.reshape(r, k)), best_byte.astype(mx.uint8), best_deq.reshape(r, k)


# ---------------------------------------------------------------------------------------------------------------------------------- whole matrix
def quantize_matrix(w: mx.array, fmt: str = "nvfp4", chunk_elems: int = 1 << 25, g: float | None = None):
    """w float32 [n, k] (k % 32 == 0) -> dict(qs, scale, scale2 | None, rel_rmse) for `fmt` in nvfp4 / mxfp4. g: the per-tensor scale to use (members of a fused group
    share one, so the loader can concatenate them as stored); default = global_scale(amax of w). The relative error is computed on the fly."""
    n, k = w.shape
    assert k % 32 == 0, f"k={k} is not a multiple of 32"
    rows = max(1, chunk_elems // k)
    if fmt == "nvfp4":
        g = global_scale(float(mx.max(mx.abs(w)))) if g is None else g
    else:
        g = 1.0
    qs, sc, num, den = [], [], 0.0, 0.0
    for a in range(0, n, rows):
        wc = w[a:a + rows]
        if fmt == "nvfp4":
            q, s, deq = quant_nvfp4_rows(wc, g)
            deq = deq
        else:
            q, s, deq = quant_mxfp4_rows(wc)
        e = mx.sum((wc - deq) ** 2)
        d = mx.sum(wc ** 2)
        mx.eval(q, s, e, d)
        qs.append(q)
        sc.append(s)
        num += float(e)
        den += float(d)
    return dict(qs=mx.concatenate(qs, axis=0) if len(qs) > 1 else qs[0], scale=mx.concatenate(sc, axis=0) if len(sc) > 1 else sc[0],
                scale2=np.float32(g) if fmt == "nvfp4" else None, rel_rmse=float(np.sqrt(num / max(den, 1e-30))))


def amax_stream(ck, prefix: str, chunk_elems: int = 1 << 22) -> float:
    """max |w| of a source linear, row chunks (small RSS)."""
    n, k = ck.raw(prefix + ".weight")[1]
    rows = max(1, chunk_elems // k)
    return max(float(np.abs(load_f32_rows(ck, prefix, a, min(n, a + rows))).max()) for a in range(0, n, rows))


def quantize_stream(ck, prefix: str, g: float | None = None, search=NV_SEARCH, chunk_elems: int = 1 << 22, sink=None, h=None) -> dict:
    """quantize_matrix(nvfp4) of a source linear read and quantised in row chunks: the same bytes as quantize_matrix(load_f32) for the same g and search (every op is
    row-local), but at most ~chunk_elems floats alive. sink(qs_chunk np.uint8) receives each chunk of the weight bytes in row order instead of keeping them (lm_head:
    317 MB); -> dict(qs (None with a sink), scale np.uint8 [n, k/16], scale2, rel_rmse, amax)."""
    n, k = ck.raw(prefix + ".weight")[1]
    assert k % 32 == 0, f"{prefix}: k={k}"
    am = amax_stream(ck, prefix, chunk_elems) if g is None else None
    g = global_scale(am) if g is None else g
    rows = max(1, chunk_elems // k)
    qs, sc, num, den, numh, denh = [], [], 0.0, 0.0, 0.0, 0.0
    hv = None if h is None else mx.array(np.asarray(h, np.float32))
    for a in range(0, n, rows):
        wc = mx.array(load_f32_rows(ck, prefix, a, min(n, a + rows)))
        q, s, deq = quant_nvfp4_rows(wc, g, search, h)
        e, d = mx.sum((wc - deq) ** 2), mx.sum(wc ** 2)
        eh, dh = (mx.sum(hv * (wc - deq) ** 2), mx.sum(hv * wc ** 2)) if hv is not None else (e, d)
        mx.eval(q, s, e, d, eh, dh)
        q, s = np.array(q), np.array(s)
        if sink is not None:
            sink(q)
        else:
            qs.append(q)
        sc.append(s)
        num += float(e)
        den += float(d)
        numh += float(eh)
        denh += float(dh)
        del wc, deq
    return dict(qs=np.concatenate(qs, 0) if qs else None, scale=np.concatenate(sc, 0), scale2=np.float32(g), rel_rmse=float(np.sqrt(num / max(den, 1e-30))),
                rel_rmse_h=float(np.sqrt(numh / max(denh, 1e-30))), amax=am)


def affine_rel_rmse(w: mx.array, gs: int, bits: int = 4) -> float:
    """MLX affine quantization (min / max per group) relative RMSE, the reference alternative."""
    q = mx.quantize(w.astype(mx.float16), group_size=gs, bits=bits)
    d = mx.dequantize(*q, group_size=gs, bits=bits).astype(mx.float32)
    return float(mx.sqrt(mx.sum((w - d) ** 2) / mx.sum(w ** 2)))


# ---------------------------------------------------------------------------------------------------------------------------------- safetensors writer
_DT = {"U8": 1, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1}


class StWriter:
    """Streams a safetensors file: header from (name, dtype, shape) first, then each tensor's bytes in order (no whole-file buffer)."""

    def __init__(self, path, plan):
        self.path, self.plan = path, list(plan)
        off, hdr = 0, {}
        for name, dt, shape in self.plan:
            nb = int(np.prod(shape, dtype=np.int64)) * _DT[dt]
            hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
            off += nb
        hdr["__metadata__"] = {"format": "pt"}
        raw = json.dumps(hdr, separators=(",", ":")).encode()
        raw += b" " * ((8 - len(raw) % 8) % 8)
        self.f = open(path, "wb")
        self.f.write(struct.pack("<Q", len(raw)))
        self.f.write(raw)
        self.i = 0

    def put(self, name, arr: np.ndarray):
        n, dt, shape = self.plan[self.i]
        assert n == name and tuple(arr.shape) == tuple(shape), f"{name}: plan {self.plan[self.i]} vs {arr.shape}"
        self.f.write(np.ascontiguousarray(arr).tobytes())
        self.i += 1

    def sink(self, name):
        """The next planned tensor written chunk by chunk (row order): returns a callable taking row chunks; call its done() after the last one."""
        n, dt, shape = self.plan[self.i]
        assert n == name, f"{name}: plan {self.plan[self.i]}"
        return _RowSink(self, int(np.prod(shape, dtype=np.int64)) * _DT[dt])

    def close(self):
        assert self.i == len(self.plan), "writer closed before every planned tensor was written"
        self.f.close()


class _RowSink:
    """Callable that appends row chunks of the planned tensor to the writer; `done()` checks the byte count and moves to the next tensor."""

    def __init__(self, w, nbytes):
        self.w, self.nbytes, self.got = w, nbytes, 0

    def __call__(self, arr):
        b = np.ascontiguousarray(arr).tobytes()
        self.w.f.write(b)
        self.got += len(b)

    def done(self):
        assert self.got == self.nbytes, f"row sink wrote {self.got} of {self.nbytes} bytes"
        self.w.i += 1
