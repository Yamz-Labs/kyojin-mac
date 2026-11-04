"""Python emulation of the generated Metal sources (windows, loads, thread mapping, reductions).
This is the only kernel check that runs without a Mac: it executes the SAME generated expression strings
(extract_bits / insert_bits / word loads) and mirrors the thread/lane/k-split indexing of metal_src.gemv_source."""
from __future__ import annotations

import re

import numpy as np

from . import metal_src as S

M32 = 0xFFFFFFFF


def extract_bits(x, off, bits):
    return (x >> off) & ((1 << bits) - 1)


def insert_bits(base, ins, off, bits):
    mask = ((1 << bits) - 1) << off
    return ((base & ~mask) | ((ins << off) & mask)) & M32


def dec_mul1(s: int) -> np.float16:
    """Line-by-line port of the Metal dec_mul1 (uint32 / ushort wraparound, half fma = exact then round)."""
    h = (s * 0x83DCD12D) & M32
    t = ((h & 0x00FF00FF) + ((h >> 8) & 0x00FF00FF)) & M32
    r = (((t & 0xFFFF) + ((t >> 16) & 0xFFFF)) + 0x6400) & 0xFFFF       # ushort(ushort(t) + ushort(t>>16) + 0x6400)
    rh = np.array([r], np.uint16).view(np.float16)[0].astype(np.float64)
    ki = np.array([0x1EEE], np.uint16).view(np.float16)[0].astype(np.float64)
    kb = np.array([0xC931], np.uint16).view(np.float16)[0].astype(np.float64)
    return np.float16(rh * ki + kb)


_U = np.uint32
_M8 = _U(0x00FF00FF)
_A16 = np.array([0x1EEE], np.uint16).view(np.float16)[0].astype(np.float64)
_B16 = np.array([0xC931], np.uint16).view(np.float16)[0].astype(np.float64)


def _h(s) -> np.ndarray:
    return np.asarray(s, np.uint32) * _U(0x83DCD12D)                    # uint32 wraparound


def byte_pairs(s, roof: bool = False) -> np.ndarray:
    """(b0 + b1, b2 + b3) in two 16-bit lanes of uint32 (dec_mul1 / dec_mul1fd), roof: the dec_mul1fdr / dec_mul1r2 form h - 255 * ((h >> 8) & m)."""
    h = _h(s)
    o = (h >> _U(8)) & _M8
    return (h - _U(255) * o) if roof else ((h & _M8) + o)


def mul1r_values(s, level: int = 2) -> np.ndarray:
    """Vector port of dec_mul1r1 / dec_mul1r2 (metal_src.HEADER_MOE_ROOF): lane sum by t * 0x00010001 + 0x64000000, half bits read from the upper 16 bits."""
    t = byte_pairs(s, roof=level == 2)
    u = t * _U(0x00010001) + _U(0x64000000)
    r = (u >> _U(16)).astype(np.uint16).view(np.float16).astype(np.float64)
    return (r * _A16 + _B16).astype(np.float16)                         # half fma: exact then one rounding


def mul1_values(s) -> np.ndarray:
    """Vector port of dec_mul1 (the stock ushort lane sum)."""
    t = byte_pairs(s)
    r = (((t & _U(0xFFFF)) + (t >> _U(16))) + _U(0x6400)).astype(np.uint16).view(np.float16).astype(np.float64)
    return (r * _A16 + _B16).astype(np.float16)


def dec_mul1r(s: int, level: int = 2) -> np.float16:
    return mul1r_values(np.array([s], np.uint32), level)[0]


def dec_mcg(s: int, three_inst: bool = False) -> np.float16:
    h = ((s * 89226354 + 64248484) if three_inst else (s * 0xCBAC1FED)) & M32
    v = np.array([(h & 0x8FFF8FFF) ^ 0x3B603B60], np.uint32).view(np.float16)
    return np.float16(v[0] + v[1])


DECODERS = {"mul1": dec_mul1, "mcg": dec_mcg, "3inst": lambda s: dec_mcg(s, True), "mul1r1": lambda s: dec_mul1r(s, 1), "mul1r2": lambda s: dec_mul1r(s, 2)}


def _py(expr: str) -> str:
    return re.sub(r"\b(0x[0-9a-fA-F]+|\d+)u\b", r"\1", expr)


def lane_values(K: int, words_tile: np.ndarray, g: int, cb: str = "mul1") -> list[np.float16]:
    """The 32 decoded values of run g of one tile, from the generated window expressions."""
    TW = 8 * K
    env = {"extract_bits": extract_bits, "insert_bits": insert_bits}
    for c in range(K):
        env[f"w{c}"] = int(words_tile[K * g + c])
    env["wp"] = int(words_tile[(K * g + TW - 1) % TW])
    mids, srcs = S.gen_windows(K)
    for line in mids:
        m = re.match(r"uint (\w+) = (.*);", line)
        env[m.group(1)] = eval(_py(m.group(2)), env)
    return [DECODERS[cb](eval(_py(e), env) & 0xFFFF) for e in srcs]


def decode_wr_t(words: np.ndarray, K: int, kt: int, nt: int) -> np.ndarray:
    return decode_wr_t_cb(words, K, kt, nt, "mul1")


def decode_wr_t_cb(words: np.ndarray, K: int, kt: int, nt: int, cb: str) -> np.ndarray:
    """Emulates the decode kernel: Wr^T fp16 [n, k]."""
    w = np.zeros((nt * 16, kt * 16), np.float16)
    for tile in range(kt * nt):
        k_, n_ = divmod(tile, nt)
        for g in range(8):
            vals = lane_values(K, words[tile], g, cb)
            for p in range(32):
                kk, hi = S.run_kk(p), int((p % 8) >= 4)
                w[n_ * 16 + g + 8 * hi, k_ * 16 + kk] = vals[p]
    return w


def gemv_partial(words: np.ndarray, K: int, kt: int, nt: int, xh: np.ndarray, tgk: int, cb: str = "mul1", act: int | None = None) -> np.ndarray:
    """Emulates gemv_source (no prefetch difference: it only reorders loads). Returns part [tgk, rows, n] fp32."""
    rows, KD = xh.shape
    N, WS = nt * 16, 4
    blocks = nt // 8
    part = np.zeros((tgk, rows, N), np.float32)
    f = np.float32
    tga = tgk if act is None else act                                    # KYOJIN_MOE_KSPLIT: active k-blocks of this bitrate, the others store zeros
    for kblock in range(tgk):
        if kblock >= tga:
            continue
        for blk in range(blocks):
            red = np.zeros((WS, 2, 32, 2 * rows), np.float32)
            S_ = tga * WS
            chunk = (kt + S_ - 1) // S_
            for warp in range(8):
                half_, split = warp & 1, warp >> 1
                ks0 = (kblock * WS + split) * chunk
                myn = min(chunk, kt - ks0) if ks0 < kt else 0
                for lane in range(32):
                    g = lane & 7
                    nt_ = blk * 8 + half_ * 4 + (lane >> 3)
                    acc = np.zeros((rows, 2), np.float32)
                    for i in range(myn):
                        tile = (ks0 + i) * nt + nt_
                        vals = lane_values(K, words[tile], g, cb)
                        for p in range(32):
                            kk, hi = S.run_kk(p), int((p % 8) >= 4)
                            for r in range(rows):
                                acc[r, hi] = f(np.float32(vals[p]) * xh[r, (ks0 + i) * 16 + kk] + acc[r, hi])
                    for r in range(rows):
                        red[split, half_, lane, 2 * r] = acc[r, 0]
                        red[split, half_, lane, 2 * r + 1] = acc[r, 1]
            for idx in range(rows * 128):
                r, c = idx >> 7, idx & 127
                h, l, hl = c >> 6, ((c & 63) >> 4) * 8 + (c & 7), (c >> 3) & 1
                v = f(0)
                for j in range(WS):
                    v = f(v + red[j, h, l, 2 * r + hl])
                part[kblock, r, blk * 128 + c] = v
    return part


def finalize(part: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """Emulates FINAL_SOURCE (butterfly with the tid ^ hs partner rule)."""
    tgk, rows, N = part.shape
    y = np.zeros((rows, N), np.float32)
    for blk in range(N // 128):
        for r in range(rows):
            v = np.zeros(128, np.float32)
            for kb in range(tgk):
                v = v + part[kb, r, blk * 128:(blk + 1) * 128]
            hs = 1
            while hs < 128:
                a, b = v.copy(), v[np.arange(128) ^ hs]
                v = np.where((np.arange(128) & hs) != 0, b - a, a + b).astype(np.float32)
                hs <<= 1
            y[r, blk * 128:(blk + 1) * 128] = v * np.float32(0.08838834764831845) * svh[blk * 128:(blk + 1) * 128].astype(np.float32)
    return y
