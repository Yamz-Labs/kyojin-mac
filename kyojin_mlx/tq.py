# SPDX-License-Identifier: Apache-2.0
"""TurboQuant KV cache (tq6w / tq5w) on MLX, the port of the HIP / CUDA design (exllamav3 cache/tq.py, doc/kv_tq.md).

Per KV head vector (head_dim D, a power of two <= 256): y = R x (R = Hadamard x random signs / sqrt(D), orthonormal), n = ||y||, code = nearest
Lloyd-Max centroid (5 / 6 bit, tq_codebooks.py) of y / n, one fp16 scale per head vector s = n / ||centroid vector|| (norm correction), so a
b-bit pool costs b + 16 / D bits per value. The pool, the fp16 sink / window overlay and the attention all live in the ROTATED domain
(q . k = Rq . Rk, o = R^T sum p Rv): q is rotated on the way in, the output on the way out.

Pool layout = the engine's: per 32-value group the codes are stored as power-of-two bit planes (6 bit: a 4-bit plane of 4 words, then a 2-bit
plane of 2 words; 5 bit: 4 + 1), words uint32, [B, H, cap, D / 32 * bits]; scales fp16 [B, H, cap]. The sink (first tokens) and the window
(last tokens) are kept fp16 exactly as in kvwin (TQKVCache is a WinKVCache whose pool is TurboQuant).

    KYOJIN_TQ=kbits:vbits:sink:window       e.g. 6:6:4:256 (tq6w) or 5:5:4:256 (tq5w); unset / "0" / "off": no TurboQuant
    KYOJIN_TQ_DECODE=0                      decode through the generic dequantize path instead of the fused Metal kernel (A/B, debugging)

Metal kernels: tq_write (rotated fp32 vectors -> words + scale, one simdgroup per vector), tq_dequant (pool rows -> fp16, bit-exact with the numpy
reference), tq_decode (T = 1, B = 1 flash-decode over pool + sink + ring, split-K, partials combined with a few MLX ops), verify rows (T 2..8) in
fused kernels. Prefill chunks (T >= KYOJIN_TQ_PF_MIN) dequantize the pool once per layer and run ONE stock fused SDPA call over sink | pool | fp16
ring + chunk (KYOJIN_TQ_PF, TQKVCache._prefill); KYOJIN_TQ_PF=0 keeps kvwin's per-row window score path."""
from __future__ import annotations

import math
import os
from functools import lru_cache

import mlx.core as mx
import numpy as np

from .kvwin import WinKVCache
from .tq_codebooks import CODEBOOKS, SIGNS

ENV = "KYOJIN_TQ"
NSPLIT = int(os.environ.get("KYOJIN_TQ_NSPLIT", "0"))                # 0 = adaptive (_nsplit)
SPLIT_TGS_PER_CORE = 12.8                                           # threadgroups per GPU core that saturate the decode kernel (Air M5: 128 TGs on 10 cores)
VU = int(os.environ.get("KYOJIN_TQ_VU", "2"))                      # keys per step of the pool V loop (loads of all issued before the FMAs; 4+ spills registers on the Air)
VX = int(os.environ.get("KYOJIN_TQ_VX", "1"))                      # 1 = pool V words loaded and extracted with constant shifts (6-bit, head_dim 256); 0 = generic tq_codes (was the only path before)
VH = int(os.environ.get("KYOJIN_TQ_VH", "1"))                      # 2 = V loop in two passes over halves of the lane dims (fewer live temporaries)
OV4 = int(os.environ.get("KYOJIN_TQ_OV4", "1"))                      # 1 = batched fp16-part V loop for decode
OVU = int(os.environ.get("KYOJIN_TQ_OVU", "4"))
CF = int(os.environ.get("KYOJIN_TQ_CF", "0"))                      # 1 = query-head chunks fastest in the decode grid
ROWS = int(os.environ.get("KYOJIN_TQ_ROWS", "1"))                   # 1 = verify rows (T > 1) in the one-pass kernel (_DEC2_SRC; 2 = simdgroup 8x8 variant _DEC3_SRC, slower from 12 rows) when it takes more rows per threadgroup than the plain kernel
NAX = int(os.environ.get("KYOJIN_TQ_NAX", "1"))                     # 1 = verify rows on the matrix units (_DECN_SRC) where the GPU has them (sdpa_dec.nax_ok, M5+); 0 = ROWS kernels everywhere
NAX_WN = int(os.environ.get("KYOJIN_TQ_NAX_WN", "0"))               # head-dim groups of that kernel (0 = 2 when the score exchange fits, else 1)
NAX_PAD = int(os.environ.get("KYOJIN_TQ_NAX_PAD", "8"))             # fp16 row padding of the decoded K / V block of that kernel
NAX_RING = int(os.environ.get("KYOJIN_TQ_NAX_RING", "2"))           # N = the fp16 part (sink + ring) of the verify rows runs as one extra split of the NAX kernel (one dispatch less); (N threadgroups); 0 = separate plain-kernel pass
NAX_NSX = int(os.environ.get("KYOJIN_TQ_NAX_NSX", "1"))             # split multiplier of that kernel over _nsplit
MARGIN = int(os.environ.get("KYOJIN_TQ_MARGIN", "80"))                  # spare fp16 ring rows past the window: speculative rollback stays exact, and sessions whose lengths differ by up to MARGIN - rows share the pool pass (attend_shared)
ROWS_MIN = int(os.environ.get("KYOJIN_TQ_ROWS_MIN", "2048"))        # pool keys below which the plain kernel stays (barriers of the one-pass kernel do not pay on short contexts)
QHALF = int(os.environ.get("KYOJIN_TQ_QHALF", "1"))                # 1 = the rotated query rows sit in threadgroup memory as fp16 in that kernel (twice the rows per threadgroup)
GA_ENV = int(os.environ.get("KYOJIN_TQ_GA", "0"))                  # experiment: V dims per lane of that kernel (0 = accumulator budget: 4 up to 8 rows, 2 up to 20)
RCV_MAX = int(os.environ.get("KYOJIN_TQ_RCV", "0"))                # experiment: rows per threadgroup of that kernel (0 = threadgroup memory budget)
RC_MAX = int(os.environ.get("KYOJIN_TQ_RC", "0"))                  # 0 = largest that fits (_rc); a smaller cap is an experiment knob
PF = int(os.environ.get("KYOJIN_TQ_PF", "1"))                      # prefill chunks through one fused SDPA call over sink | dequantized pool | fp16 tail (TQKVCache._prefill): 1 = causal, ring + chunk at fp16; 2 = per-row window mask (decode-faithful); 0 = kvwin per-row score path
PF_MIN = int(os.environ.get("KYOJIN_TQ_PF_MIN", "16"))              # chunk rows from which _prefill is used (shorter chunks keep the score path)
_PFMASK = {}                                                        # PF=2 chunk mask per (L, T, window, segments): built once per chunk, shared by every attention layer
SPLIT_MIN_KEYS = 512                                               # fewer keys than this per split costs more in prologue + merge than it saves


# ---------------------------------------------------------------------------------------------- constants (numpy = the reference)
def supported(dim: int, bits: int) -> bool:
    return (dim, bits) in CODEBOOKS and dim <= 256 and dim <= len(SIGNS)


@lru_cache(maxsize=None)
def codebook(dim: int, bits: int) -> np.ndarray:
    assert (dim, bits) in CODEBOOKS, f"no TurboQuant codebook for head_dim {dim}, {bits} bits (have {sorted(CODEBOOKS)})"
    return np.array(CODEBOOKS[dim, bits], dtype=np.float32)


@lru_cache(maxsize=None)
def midpoints(dim: int, bits: int) -> np.ndarray:
    cb = codebook(dim, bits)
    return ((cb[1:] + cb[:-1]) / 2).astype(np.float32)


@lru_cache(maxsize=None)
def rotation(dim: int) -> np.ndarray:
    """M with y = x @ M.T = H diag(s) x / sqrt(dim): fp32, orthonormal."""
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < dim:
        h = np.block([[h, h], [h, -h]])
    s = np.array([1.0 if c == "1" else -1.0 for c in SIGNS[:dim]])
    return ((h / math.sqrt(dim)) * s).astype(np.float32)


@lru_cache(maxsize=None)
def _rot_mx(dim: int):
    m = rotation(dim)
    return mx.array(m.T.copy()), mx.array(m)            # x @ M.T = R x ; y @ M = R^T y


def rotate(x, dim):
    return (x.astype(mx.float32) @ _rot_mx(dim)[0])


def unrotate(y, dim):
    return (y.astype(mx.float32) @ _rot_mx(dim)[1])


def _planes(bits: int):
    """[(width, shift)] high to low, the engine's power-of-two bit planes."""
    out, rem = [], bits
    for w in (8, 4, 2, 1):
        if bits & w:
            rem -= w
            out.append((w, rem))
    return out


def encode_ref(y: np.ndarray, bits: int):
    """Reference writer. y [N, D] rotated fp32 -> (words uint32 [N, D / 32 * bits], scale fp16 [N], codes int64 [N, D])."""
    n, d = y.shape
    cb, mid = codebook(d, bits), midpoints(d, bits)
    y = y.astype(np.float32)
    nrm = np.maximum(np.sqrt((y * y).sum(-1, keepdims=True, dtype=np.float32)), np.float32(1e-12))
    codes = np.searchsorted(mid, y / nrm, side="left")                 # torch.bucketize(right=False): number of midpoints < value
    cn = np.maximum(np.sqrt((cb[codes] ** 2).sum(-1, keepdims=True, dtype=np.float32)), np.float32(1e-12))
    words = pack_ref(codes.reshape(n, d // 32, 32), bits).reshape(n, d // 32 * bits)
    return words, (nrm / cn)[:, 0].astype(np.float16), codes


def pack_ref(codes: np.ndarray, bits: int) -> np.ndarray:
    """codes [..., 32] -> uint32 [..., bits]: per plane (width w), word i holds the codes [i * 32/w, (i + 1) * 32/w) at shifts j * w."""
    planes = []
    for w, rem in _planes(bits):
        vpw = 32 // w
        f = ((codes >> rem) & ((1 << w) - 1)).astype(np.uint64).reshape(*codes.shape[:-1], w, vpw)
        planes.append((f << (np.arange(vpw, dtype=np.uint64) * w)).sum(-1))
    return np.concatenate(planes, axis=-1).astype(np.uint32)


def unpack_ref(words: np.ndarray, bits: int) -> np.ndarray:
    """uint32 [..., bits] -> codes int64 [..., 32] (inverse of pack_ref)."""
    codes = np.zeros((*words.shape[:-1], 32), dtype=np.int64)
    base = 0
    for w, rem in _planes(bits):
        vpw = 32 // w
        ww = words[..., base:base + w].astype(np.uint64)
        f = (ww[..., :, None] >> (np.arange(vpw, dtype=np.uint64) * w)) & ((1 << w) - 1)
        codes |= f.reshape(*words.shape[:-1], 32).astype(np.int64) << rem
        base += w
    return codes


def decode_ref(words: np.ndarray, scale: np.ndarray, bits: int, d: int) -> np.ndarray:
    """-> fp16 [N, D] rotated domain (centroid * scale, one fp32 multiply, one rounding)."""
    n = words.shape[0]
    codes = unpack_ref(words.reshape(n, d // 32, bits), bits).reshape(n, d)
    return (codebook(d, bits)[codes] * scale.astype(np.float32)[:, None]).astype(np.float16)


# ---------------------------------------------------------------------------------------------- Metal kernels
_HEADER = """
// lane l owns the DPL = D / 32 dims [DPL * l, DPL * l + DPL): in the plane layout those sit in ONE word per plane (lane = group * LPG + q, LPG = 32 / DPL)
template <int BITS, int DPL>
inline void tq_codes(const device uint* w, uint lane, thread uint* c) {
    constexpr int LPG = 32 / DPL;
    const device uint* p = w + (lane / LPG) * BITS;
    uint q = lane % LPG;
    for (int j = 0; j < DPL; j++) c[j] = 0;
    int rem = BITS, wo = 0;
    for (int W = 4; W >= 1; W >>= 1) {
        if (!(BITS & W)) continue;
        rem -= W;
        int vpw = 32 / W;
        uint word = p[wo + (DPL * q) / vpw];
        uint off = ((DPL * q) % vpw) * W;
        for (int j = 0; j < DPL; j++) c[j] |= extract_bits(word, off + j * W, W) << rem;
        wo += W;
    }
}

// GA contiguous dims [d0, d0 + GA) of a row (GA | 32, d0 a multiple of GA): the one-pass verify kernel gives every lane GA dims of the V row
template <int BITS, int GA>
inline void tq_codes_a(const device uint* w, int d0, thread uint* c) {
    const device uint* p = w + (d0 >> 5) * BITS;
    int dd = d0 & 31;
    for (int j = 0; j < GA; j++) c[j] = 0;
    int rem = BITS, wo = 0;
    for (int W = 4; W >= 1; W >>= 1) {
        if (!(BITS & W)) continue;
        rem -= W;
        int vpw = 32 / W;
        uint word = p[wo + dd / vpw];
        uint off = (dd % vpw) * W;
        for (int j = 0; j < GA; j++) c[j] |= extract_bits(word, off + j * W, W) << rem;
        wo += W;
    }
}

template <int BITS>
inline uint tq_code(const device uint* w, uint lane) {
    uint code = 0; int rem = BITS; int wo = 0;
    for (int W = 4; W >= 1; W >>= 1) {
        if (!(BITS & W)) continue;
        rem -= W;
        int vpw = 32 / W;
        uint word = w[wo + lane / vpw];
        code |= ((word >> ((lane % vpw) * W)) & ((1u << W) - 1u)) << rem;
        wo += W;
    }
    return code;
}

// K pool, tiled layout: word w of row j sits at tile(j) + w * 32 + (j % 32) (32 consecutive rows interleave word by word), so a word has stride 32
template <int BITS>
inline uint tq_code_s(const device uint* w, uint lane) {
    uint code = 0; int rem = BITS; int wo = 0;
    for (int W = 4; W >= 1; W >>= 1) {
        if (!(BITS & W)) continue;
        rem -= W;
        int vpw = 32 / W;
        uint word = w[(wo + lane / vpw) * 32];
        code |= ((word >> ((lane % vpw) * W)) & ((1u << W) - 1u)) << rem;
        wo += W;
    }
    return code;
}
"""

_WRITE_SRC = """
    uint gid = thread_position_in_grid.x;
    uint n = gid / 32;
    uint lane = thread_index_in_simdgroup;
    constexpr int G = D / 32;
    constexpr int NM = (1 << BITS) - 1;
    float v[G];
    float ss = 0.0f;
    for (int g = 0; g < G; g++) { v[g] = y[n * D + g * 32 + lane]; ss += v[g] * v[g]; }
    float nrm = max(sqrt(simd_sum(ss)), 1e-12f);
    uint code[G];
    float cn = 0.0f;
    for (int g = 0; g < G; g++) {
        float t = metal::precise::divide(v[g], nrm);
        int lo = 0, hi = NM;
        while (lo < hi) { int m = (lo + hi) >> 1; if (mid[m] < t) lo = m + 1; else hi = m; }
        code[g] = (uint)lo;
        float c = cb[lo];
        cn += c * c;
    }
    cn = max(sqrt(simd_sum(cn)), 1e-12f);
    if (lane == 0) scale[n] = (half)metal::precise::divide(nrm, cn);
    int rem = BITS, wo = 0;
    for (int W = 4; W >= 1; W >>= 1) {
        if (!(BITS & W)) continue;
        rem -= W;
        int vpw = 32 / W;
        for (int g = 0; g < G; g++) {
            uint field = (code[g] >> rem) & ((1u << W) - 1u);
            uint word = 0;
            for (int j = 0; j < vpw; j++) word |= simd_shuffle(field, (lane * vpw + j) & 31) << (j * W);
            if ((int)lane < W) words[n * NW + g * BITS + wo + lane] = word;
        }
        wo += W;
    }
"""

_DEQ_SRC = """
    uint gid = thread_position_in_grid.x;
    uint n = gid / 32;
    uint lane = thread_index_in_simdgroup;
    constexpr int G = D / 32;
    float s = (float)scale[n];
    for (int g = 0; g < G; g++) {
        uint code = tq_code<BITS>(words + n * NW + g * BITS, lane);
        out[n * D + g * 32 + lane] = (T)(cb[code] * s);
    }
"""

# K pool dequant from the tiled layout: grid (32 * n, B * H); info = [row0, n, tiles per head, cap]
_DEQT_SRC = """
    uint i = thread_position_in_grid.x / 32;
    uint bh = thread_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    constexpr int G = D / 32;
    int r0 = info[0], n = info[1], NT = info[2], cap = info[3];
    int j = r0 + (int)i;
    const device uint* wp = words + ((size_t)bh * NT + (j >> 5)) * (G * BITS * 32) + (j & 31);
    float s = (float)scale[(size_t)bh * cap + j];
    for (int g = 0; g < G; g++) {
        uint code = tq_code_s<BITS>(wp + g * BITS * 32, lane);
        out[((size_t)bh * n + i) * D + g * 32 + lane] = (T)(cb[code] * s);
    }
"""

# T = 1 flash-decode, B = 1. Grid (256 * (NSPLIT + 1), Hk * REP / RC): threadgroup (split, hk, query-head chunk), 8 simdgroups. The pool is cut into NSPLIT slices of
# whole 32-key tiles; split NSPLIT is the fp16 part (sink + ring). K: lane = key of the tile (the tiled K layout makes every word load a 128 B line and the q.k
# dot needs no cross-lane reduce); V: lane = G = D / 32 dims of every key (plane layout, one word per plane). q is the raw query [Hq, D]; it is rotated and
# scaled in the threadgroup. info = [L, S, T0, NR, cap, NS]. Partials: po [Hk, NSPLIT + 1, REP, D], pml [.., 2].
_KDOT = '''            for (int g = 0; g < G; g++) {
                uint wk[KB];
                for (int i = 0; i < KB; i++) wk[i] = @KP[(g * KB + i) * 32];
                for (int q4 = 0; q4 < 8; q4++) {
                    float4 kv;
                    for (int e = 0; e < 4; e++) {
                        constexpr int WL = KB - 4, VPL = WL > 0 ? 32 / WL : 32;                // planes: 4 bits (KB - 4 <= 2 for 4..6 bits), then the KB - 4 low bits
                        int i = q4 * 4 + e;
                        uint code = (wk[i >> 3] >> ((i & 7) * 4)) & 15u;
                        if (WL > 0) code = (code << WL) | ((wk[4 + i / VPL] >> ((i % VPL) * WL)) & ((1u << WL) - 1u));
#if ABL & 16
                        kv[e] = (float)code * 0.01f;
#else
                        kv[e] = cbk_s[code];
#endif
                    }
#if ABL & 32
                    for (int r = 0; r < 1; r++) @S[r] += QDOT(qsh + r * D + g * 32 + q4 * 4, kv);
#else
                    for (int r = 0; r < RC; r++) @S[r] += QDOT(qsh + r * D + g * 32 + q4 * 4, kv);
#endif
                }
            }
'''

_DEC_SRC = """
    constexpr int G = D / 32;
    constexpr int NCH = REP / RC;
    uint split = CF ? threadgroup_position_in_grid.x / NCH : threadgroup_position_in_grid.x;       // CF: the query-head chunks of one split are neighbours in the launch order (they read the same K / V)
    uint hk = CF ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.y / NCH;
    int r0 = (int)(CF ? threadgroup_position_in_grid.x % NCH : threadgroup_position_in_grid.y % NCH) * RC;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint tid = sg * 32 + lane;
    int L = info[0], S = info[1], T0 = info[2], NR = info[3], cap = info[4], NS = info[5], WIN = info[6];
    int NT = cap / 32;
    threadgroup float cbk_s[1 << KB];
    threadgroup float cbv_s[1 << VB];
    threadgroup float sh[8 * D];
    threadgroup float sml[16];
    if (tid < (1u << KB)) cbk_s[tid] = cbk[tid];
    if (tid < (1u << VB)) cbv_s[tid] = cbv[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float qsh[RC * D];                                  // q -> H diag(s) q * scale / sqrt(D) (fast Walsh-Hadamard in the threadgroup)
    for (int i = (int)tid; i < RC * D; i += 256) qsh[i] = (float)q[((int)hk * REP + r0) * D + i] * sgn[i % D];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if !(ABL & 128)
    for (int len = 1; len < D; len <<= 1) {
        for (int idx = (int)tid; idx < RC * D / 2; idx += 256) {
            int rr = idx / (D / 2), k = idx % (D / 2);
            int a = rr * D + (k / len) * 2 * len + (k % len);
            float x0 = qsh[a], x1 = qsh[a + len];
            qsh[a] = x0 + x1; qsh[a + len] = x0 - x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif
    float qscl = scl[0];
    for (int i = (int)tid; i < RC * D; i += 256) qsh[i] *= qscl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if ABL & 256
    if (tid < 256) { if (tid < RC * D) po[((int)(hk * (NSPLIT + 1) + split) * REP + r0) * D + tid] = qsh[tid]; return; }
#endif
    float acc[RC][G];
    for (int r = 0; r < RC; r++) for (int g = 0; g < G; g++) acc[r][g] = 0.0f;
    float m_l[RC], l_l[RC];
    for (int r = 0; r < RC; r++) { m_l[r] = -INFINITY; l_l[r] = 0.0f; }
    bool ov = (int)split == NSPLIT;
    int Pa = S, Pb = T0;                                            // pool keys [S, T0); sink [0, S) and ring [T0, L) are fp16
    int j0, j1, ja;
    if (ov) { j0 = 0; j1 = S + (L - T0); ja = 0; }
    else {
        int base = Pa & ~31;                                        // splits start on a tile boundary and must cover [base, Pb)
        int per = ((Pb - base + NSPLIT - 1) / NSPLIT + 31) / 32 * 32;
        j0 = base + (int)split * per; j1 = min(Pb, j0 + per); ja = Pa;
    }
    for (int tb = j0 + (int)sg * 32; tb < j1; tb += 256) {
        int j = tb + (int)lane;
        bool ok = j >= ja && j < j1;
        int jc = min(max(j, ja), j1 - 1);
        float s[RC];
        for (int r = 0; r < RC; r++) s[r] = 0.0f;
        float kscl = 1.0f, vscl = 1.0f;
        int jabs = 0; bool isring = false;
        if (!ov) {
            const device uint* kp = kw + ((size_t)((int)hk * NT + (tb >> 5)) * (G * KB)) * 32 + lane;
#if !(ABL & 1)
@KDOT(kp, s)
#endif
            int bb = (int)hk * cap + jc; kscl = (float)ks[bb]; vscl = (float)vs[bb];
        } else {
            isring = jc >= S;
            jabs = isring ? T0 + (jc - S) : jc;                     // absolute position of this lane's key
            const device T* kp = isring ? rk + ((int)hk * NR + (jabs - (L - NR))) * D : sk + ((int)hk * NS + jc) * D;
#if !(ABL & 64)
            for (int d4 = 0; d4 < D / 4; d4++) {
                float4 kv = float4(*(const device vec<T, 4>*)(kp + d4 * 4));
                for (int r = 0; r < RC; r++) s[r] += dot(*(threadgroup float4*)(qsh + r * D + d4 * 4), kv);
            }
#endif
            if (isring && jabs <= L - 1 - WIN && jabs >= S) {         // verify chunks: rows whose window no longer reaches this key read its pool version
                float sp[RC];
                for (int r = 0; r < RC; r++) sp[r] = 0.0f;
                const device uint* kq = kw + ((size_t)((int)hk * NT + (jabs >> 5)) * (G * KB)) * 32 + (jabs & 31);
@KDOT(kq, sp)
                int bb = (int)hk * cap + jabs; kscl = (float)ks[bb]; vscl = (float)vs[bb];
                for (int r = 0; r < RC; r++) { int pr = L - QT + ((r0 + r) % QT); if (jabs <= pr - WIN) s[r] = sp[r] * kscl; }
            }
        }
        float p[RC];
        for (int r = 0; r < RC; r++) {
            bool vis = true;
            if (ov) { int pr = L - QT + ((r0 + r) % QT); vis = jabs <= pr; }                  // row position pr sees keys <= pr
            float sc = (!ok || !vis) ? -INFINITY : (ov ? s[r] : s[r] * kscl);
            float mn = max(m_l[r], simd_max(sc));
            float al = (mn == -INFINITY) ? 1.0f : exp(m_l[r] - mn);
            p[r] = (sc == -INFINITY) ? 0.0f : exp(sc - mn);
            l_l[r] = l_l[r] * al + p[r];
            m_l[r] = mn;
            for (int g = 0; g < G; g++) acc[r][g] *= al;
            if (!ov) p[r] *= vscl;
        }
#if !(ABL & 2)
        if (!ov) {
            if (VB == 6 && G == 8 && VX && VH == 3) {                    // single pass: only the loaded words stay live, codes are extracted inside the FMA loop of each key
                uint qq = lane % 4, o2 = (qq & 1u) * 16u;
                for (int kk0 = 0; kk0 < 32; kk0 += VU) {
                    uint wa[VU], wb[VU];
                    _Pragma("unroll")
                    for (int u = 0; u < VU; u++) {
                        int jk = min(max(tb + kk0 + u, ja), j1 - 1);
                        const device uint* pv = vw + ((size_t)((int)hk * cap + jk)) * (G * VB) + (lane / 4) * VB;
                        wa[u] = pv[qq]; wb[u] = pv[4 + (qq >> 1)];
                    }
                    _Pragma("unroll")
                    for (int u = 0; u < VU; u++) {
                        float vf[G];
                        for (int g = 0; g < G; g++) vf[g] = cbv_s[(extract_bits(wa[u], g * 4, 4) << 2) | extract_bits(wb[u], o2 + g * 2, 2)];
                        for (int r = 0; r < RC; r++) {
                            float pk = simd_shuffle(p[r], (ushort)(kk0 + u));
                            for (int g = 0; g < G; g++) acc[r][g] += pk * vf[g];
                        }
                    }
                }
            } else
            if (VB == 6 && G == 8 && VX && VH == 2) {                    // same loop on two halves of the lane's dims: half the V temporaries live at once, so more keys per step without spills
                uint qq = lane % 4, o2 = (qq & 1u) * 16u;
                _Pragma("unroll")
                for (int h = 0; h < 2; h++) {
                    for (int kk0 = 0; kk0 < 32; kk0 += VU) {
                        uint wa[VU], wb[VU];
                        _Pragma("unroll")
                        for (int u = 0; u < VU; u++) {
                            int jk = min(max(tb + kk0 + u, ja), j1 - 1);
                            const device uint* pv = vw + ((size_t)((int)hk * cap + jk)) * (G * VB) + (lane / 4) * VB;
                            wa[u] = pv[qq]; wb[u] = pv[4 + (qq >> 1)];
                        }
                        _Pragma("unroll")
                        for (int u = 0; u < VU; u++) {
                            float vf[4];
                            for (int g = 0; g < 4; g++) vf[g] = cbv_s[(extract_bits(wa[u], (h * 4 + g) * 4, 4) << 2) | extract_bits(wb[u], o2 + (h * 4 + g) * 2, 2)];
                            for (int r = 0; r < RC; r++) {
                                float pk = simd_shuffle(p[r], (ushort)(kk0 + u));
                                for (int g = 0; g < 4; g++) acc[r][h * 4 + g] += pk * vf[g];
                            }
                        }
                    }
                }
            } else
            for (int kk0 = 0; kk0 < 32; kk0 += VU) {
                uint c[VU][G];
                if (VB == 6 && G == 8 && VX) {                         // 6 bits, 8 dims per lane: one 4-bit-plane word and one 2-bit-plane word per key, scalar loads and constant-shift extracts
                    uint wa[VU], wb[VU];
                    uint qq = lane % 4, o2 = (qq & 1u) * 16u;
                    _Pragma("unroll")
                    for (int u = 0; u < VU; u++) {
                        int jk = min(max(tb + kk0 + u, ja), j1 - 1);
                        const device uint* pv = vw + ((size_t)((int)hk * cap + jk)) * (G * VB) + (lane / 4) * VB;
                        wa[u] = pv[qq]; wb[u] = pv[4 + (qq >> 1)];
                    }
                    _Pragma("unroll")
                    for (int u = 0; u < VU; u++)
                        for (int g = 0; g < G; g++) c[u][g] = (extract_bits(wa[u], g * 4, 4) << 2) | extract_bits(wb[u], o2 + g * 2, 2);
                } else {
                    _Pragma("unroll")
                    for (int u = 0; u < VU; u++) {
                        int jk = min(max(tb + kk0 + u, ja), j1 - 1);
                        tq_codes<VB, G>(vw + ((size_t)((int)hk * cap + jk)) * (G * VB), lane, c[u]);
                    }
                }
                _Pragma("unroll")
                for (int u = 0; u < VU; u++) {
                    float vf[G];
#if ABL & 8
                    for (int g = 0; g < G; g++) vf[g] = (float)c[u][g] * 0.01f;
#else
                    for (int g = 0; g < G; g++) vf[g] = cbv_s[c[u][g]];
#endif
#if ABL & 4
                    for (int r = 0; r < 1; r++) {
#else
                    for (int r = 0; r < RC; r++) {
#endif
                        float pk = simd_shuffle(p[r], (ushort)(kk0 + u));
                        for (int g = 0; g < G; g++) acc[r][g] += pk * vf[g];
                    }
                }
            }
        } else if (QT == 1 && OV4) {                                  // decode fp16 part (sink + ring): OVU keys per step, all loads issued before the FMAs (latency chain of the critical threadgroup)
            for (int kk0 = 0; kk0 < 32; kk0 += OVU) {
                float vf[OVU][G];
                _Pragma("unroll")
                for (int u = 0; u < OVU; u++) {
                    int jk = min(max(tb + kk0 + u, ja), j1 - 1);
                    const device T* vp = (jk < S) ? sv + ((int)hk * NS + jk) * D : rv + ((int)hk * NR + (jk - S + T0 - (L - NR))) * D;
                    vp += lane * G;
                    if (G % 4 == 0) {
                        for (int g4 = 0; g4 < G / 4; g4++) {
                            float4 x = float4(*(const device vec<T, 4>*)(vp + g4 * 4));
                            for (int e = 0; e < 4; e++) vf[u][g4 * 4 + e] = x[e];
                        }
                    } else {
                        for (int g = 0; g < G; g++) vf[u][g] = (float)vp[g];
                    }
                }
                _Pragma("unroll")
                for (int u = 0; u < OVU; u++)
                    for (int r = 0; r < RC; r++) {
                        float pk = simd_shuffle(p[r], (ushort)(kk0 + u));
                        for (int g = 0; g < G; g++) acc[r][g] += pk * vf[u][g];
                    }
            }
        } else
        for (int kk = 0; kk < 32; kk++) {
            int jk = min(max(tb + kk, ja), j1 - 1);
            float vf[G];
            if (!ov) {
                uint c[G]; tq_codes<VB, G>(vw + ((size_t)((int)hk * cap + jk)) * (G * VB), lane, c);
                for (int g = 0; g < G; g++) vf[g] = cbv_s[c[g]];
            } else {
                const device T* vp;
                if (jk < S) vp = sv + ((int)hk * NS + jk) * D; else vp = rv + ((int)hk * NR + (jk - S + T0 - (L - NR))) * D;
                for (int g = 0; g < G; g++) vf[g] = (float)vp[lane * G + g];
            }
            if (!ov) {
                for (int r = 0; r < RC; r++) {
                    float pk = simd_shuffle(p[r], (ushort)kk);
                    for (int g = 0; g < G; g++) acc[r][g] += pk * vf[g];
                }
            } else {
                int jka = jk < S ? jk : T0 + (jk - S);
                bool needpool = jk >= S && jka <= L - 1 - WIN && jka >= S;                  // boundary key of a verify chunk: pool version for the rows whose window no longer reaches it
                float vsk = needpool ? simd_shuffle(vscl, (ushort)kk) : 1.0f;
                float vq[G];
                if (needpool) {
                    uint c[G]; tq_codes<VB, G>(vw + ((size_t)((int)hk * cap + jka)) * (G * VB), lane, c);
                    for (int g = 0; g < G; g++) vq[g] = cbv_s[c[g]];
                }
                for (int r = 0; r < RC; r++) {
                    float pk = simd_shuffle(p[r], (ushort)kk);
                    if (needpool && jka <= L - QT + ((r0 + r) % QT) - WIN) { pk *= vsk; for (int g = 0; g < G; g++) acc[r][g] += pk * vq[g]; }
                    else for (int g = 0; g < G; g++) acc[r][g] += pk * vf[g];
                }
            }
        }
#endif
    }
    for (int r = 0; r < RC; r++) {
        float lsum = simd_sum(l_l[r]);
        if (lane == 0) { sml[sg] = m_l[r]; sml[8 + sg] = lsum; }
        for (int g = 0; g < G; g++) sh[sg * D + lane * G + g] = acc[r][g];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mt = -INFINITY;
        for (int s = 0; s < 8; s++) mt = max(mt, sml[s]);
        float lt = 0.0f, o = 0.0f;
        for (int s = 0; s < 8; s++) {
            float w = (sml[s] == -INFINITY) ? 0.0f : exp(sml[s] - mt);
            lt += w * sml[8 + s];
            if (tid < (uint)D) o += w * sh[s * D + tid];
        }
        int row = ((int)hk * (NSPLIT + 1) + (int)split) * REP + r0 + r;
        if (tid < (uint)D) po[row * D + tid] = o;
        if (tid == 0) { pml[row * 2] = mt; pml[row * 2 + 1] = lt; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
import re as _re
_DEC_SRC = _re.sub(r"@KDOT\((\w+), (\w+)\)\n", lambda m: _KDOT.replace("@KP", m.group(1)).replace("@S", m.group(2)), _DEC_SRC)

# Verify rows, pool part. One threadgroup = (split, kv head, chunk of RC virtual query heads): K and V are dequantized ONCE per key for all RC rows. Per round of 256 keys: every simdgroup
# scores its tile of 32 keys (lane = key, as above) -> scores to threadgroup memory -> online softmax over the 256 keys (one running max per row for the whole threadgroup, rows spread
# over the simdgroups) -> P.V of the 256 keys, the lanes holding GA contiguous dims of the V row each (D / GA lanes per key, the other simdgroups take the other keys; KS partial sums
# merged at the end). Partials like the plain kernel (po [Hk, NSPLIT + 1, REP, D], pml); the fp16 part (sink + ring) stays in the plain kernel.
_DEC2_SRC = """
    constexpr int G = D / 32;
    constexpr int NLN = D / GA;                                     // lanes per V row: NLN / 32 simdgroups cover a key, KS key subsets run side by side
    constexpr int KS = 8 * 32 / NLN;
    constexpr int RQ = (RC + 3) / 4;                                // score block layout: [row quad][key][4 rows] (float4 reads in the P.V loop, float4 writes by consecutive lanes)
    constexpr int NCH = REP / RC;
    uint split = threadgroup_position_in_grid.x;
    uint hk = threadgroup_position_in_grid.y / NCH;
    int r0 = (int)(threadgroup_position_in_grid.y % NCH) * RC;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint tid = sg * 32 + lane;
    int S = info[1], T0 = info[2], cap = info[4];
    int NT = cap / 32;
    threadgroup float cbk_s[1 << KB];
    threadgroup float cbv_s[1 << VB];
    threadgroup QT_ qsh[RC * D];
    threadgroup float xs[(1024 * RQ > KS * D) ? 1024 * RQ : KS * D];
    threadgroup float vsx[256];
    threadgroup float mrun[RC], lrun[RC], alr[RC];
    if (tid < (1u << KB)) cbk_s[tid] = cbk[tid];
    if (tid < (1u << VB)) cbv_s[tid] = cbv[tid];
    if (tid < RC) { mrun[tid] = -INFINITY; lrun[tid] = 0.0f; }
    threadgroup float* qf = xs;                                     // q -> H diag(s) q * scale / sqrt(D) in the (still idle) score block, then to qsh
    for (int i = (int)tid; i < RC * D; i += 256) qf[i] = (float)q[((int)hk * REP + r0) * D + i] * sgn[i % D];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int len = 1; len < D; len <<= 1) {
        for (int idx = (int)tid; idx < RC * D / 2; idx += 256) {
            int rr = idx / (D / 2), k = idx % (D / 2);
            int a = rr * D + (k / len) * 2 * len + (k % len);
            float x0 = qf[a], x1 = qf[a + len];
            qf[a] = x0 + x1; qf[a + len] = x0 - x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float qscl = scl[0];
    for (int i = (int)tid; i < RC * D; i += 256) qsh[i] = (QT_)(qf[i] * qscl);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[RC][GA];
    for (int r = 0; r < RC; r++) for (int g = 0; g < GA; g++) acc[r][g] = 0.0f;
    int base0 = S & ~31;
    int per = ((T0 - base0 + NSPLIT - 1) / NSPLIT + 31) / 32 * 32;
    int j0 = base0 + (int)split * per, j1 = min(T0, j0 + per), ja = S;
    int NIT = j1 > j0 ? (j1 - j0 + 255) / 256 : 0;
    int ksi = (int)sg / (NLN / 32);
    int d0 = (((int)sg % (NLN / 32)) * 32 + (int)lane) * GA;
    for (int it = 0; it < NIT; it++) {
        int base = j0 + it * 256;
        int tb = base + (int)sg * 32;
        int j = tb + (int)lane;
        bool ok = j >= ja && j < j1;
        int jc = min(max(j, ja), j1 - 1);
        float s[RC];
        for (int r = 0; r < RC; r++) s[r] = 0.0f;
        float kscl = 0.0f, vscl = 0.0f;
        if (tb < j1) {
            const device uint* kp = kw + ((size_t)((int)hk * NT + (tb >> 5)) * (G * KB)) * 32 + lane;
#if !(ABL & 4096)
@KDOT(kp, s)
#endif
            int bb = (int)hk * cap + jc; kscl = (float)ks[bb]; vscl = (float)vs[bb];
        }
        for (int r = 0; r < RC; r++) xs[(((r >> 2) * 256 + (int)tid) << 2) + (r & 3)] = ok ? s[r] * kscl : -INFINITY;
        vsx[tid] = ok ? vscl : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if !(ABL & 2048)
        for (int r = (int)sg; r < RC; r += 8) {
            float sc[8];
            float mx = -INFINITY;
            for (int i = 0; i < 8; i++) { sc[i] = xs[(((r >> 2) * 256 + (int)lane + 32 * i) << 2) + (r & 3)]; mx = max(mx, sc[i]); }
            mx = simd_max(mx);
            float mo = mrun[r];
            float mn = max(mo, mx);
            float al = (mn == -INFINITY) ? 1.0f : exp(mo - mn);
            float sum = 0.0f;
            for (int i = 0; i < 8; i++) {
                float e = (sc[i] == -INFINITY) ? 0.0f : exp(sc[i] - mn);
                sum += e;
                xs[(((r >> 2) * 256 + (int)lane + 32 * i) << 2) + (r & 3)] = e * vsx[(int)lane + 32 * i];
            }
            sum = simd_sum(sum);
            if (lane == 0) { lrun[r] = lrun[r] * al + sum; mrun[r] = mn; alr[r] = al; }
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < RC; r++) for (int g = 0; g < GA; g++) acc[r][g] *= alr[r];
#if !(ABL & 1024)
        for (int kb = 0; kb < 256 && base + kb < j1; kb += KS * VU) {
            uint cd[VU][GA];
            _Pragma("unroll")
            for (int u = 0; u < VU; u++) {
                int jk = min(max(base + kb + ksi + u * KS, ja), j1 - 1);
                tq_codes_a<VB, GA>(vw + ((size_t)((int)hk * cap + jk)) * (G * VB), d0, cd[u]);
            }
            _Pragma("unroll")
            for (int u = 0; u < VU; u++) {
                float v[GA];
                for (int g = 0; g < GA; g++) v[g] = cbv_s[cd[u][g]];
                int kk = kb + ksi + u * KS;
                for (int rq = 0; rq < RQ; rq++) {
                    float4 pv = *(threadgroup float4*)(xs + ((rq * 256 + kk) << 2));
                    for (int e = 0; e < 4; e++) if (rq * 4 + e < RC) for (int g = 0; g < GA; g++) acc[rq * 4 + e][g] += pv[e] * v[g];
                }
            }
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < RC; r++) {
        int row = ((int)hk * (NSPLIT + 1) + (int)split) * REP + r0 + r;
        if (KS == 1) { for (int g = 0; g < GA; g++) po[row * D + d0 + g] = acc[r][g]; }
        else {
            for (int g = 0; g < GA; g++) xs[ksi * D + d0 + g] = acc[r][g];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < (uint)D) { float o = 0.0f; for (int k = 0; k < KS; k++) o += xs[k * D + tid]; po[row * D + tid] = o; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) { pml[row * 2] = mrun[r]; pml[row * 2 + 1] = lrun[r]; }
    }
"""
_DEC2_SRC = _re.sub(r"@KDOT\((\w+), (\w+)\)\n", lambda m: _KDOT.replace("@KP", m.group(1)).replace("@S", m.group(2)), _DEC2_SRC)

# verify rows on the GPU's matrix units (NAX, M5+): the pool K / V codes of a 32-key block are decoded ONCE into threadgroup fp16 (centroid only; the per-key
# scales stay fp32: ks on the scores, vs on P), then S = Q K^T and O += P V run as 16x16 NAX tiles (sdpa_dec._NAX_SRC layout: WM row groups of 16 rows x WN
# head-dim groups, partial scores summed through threadgroup memory). Same contract as _DEC2_SRC: pool keys [S, T0) of every row, NSPLIT splits of whole
# 32-key tiles, partials po [B * Hk, NSPLIT + 1, REP, D] (unnormalized, rotated domain) and pml [.., 2] (natural-log max, sum); the fp16 part stays in "dec".
_DECN_HDR = """
template <int BITS, int STR>
inline void tq_ld32(const device uint* p, thread uint* w) {
    for (int i = 0; i < BITS; i++) w[i] = p[i * STR];
}

// position of virtual fp16 key v of the verify ring split: sink keys [0, S), then ring keys [T0, L); -1 = past the end
inline int tq_fpos(int v, int nv, int S, int T0) { return v >= nv ? -1 : (v < S ? v : T0 + (v - S)); }

template <int BITS>
inline void tq_dec32(thread const uint* w, const threadgroup half* cb, threadgroup half* dst) {
    _Pragma("unroll")
    for (int j = 0; j < 32; j += 2) {
        uint c[2];
        for (int e = 0; e < 2; e++) {
            c[e] = 0; int rem = BITS, wo = 0;
            for (int W = 4; W >= 1; W >>= 1) {
                if (!(BITS & W)) continue;
                rem -= W;
                const int vpw = 32 / W, jj = j + e;
                c[e] |= ((w[wo + jj / vpw] >> ((jj % vpw) * W)) & ((1u << W) - 1u)) << rem;
                wo += W;
            }
        }
        *(threadgroup half2*)(dst + j) = half2(cb[c[0]], cb[c[1]]);
    }
}
"""

_DECN_SRC = """
    using namespace mlx::steel;
    const int L = info[0], NR = info[3], NS = info[5], WIN = info[6];
    constexpr int G = D / 32;
    constexpr short kU = 16;
    constexpr int TD = D / kU, TK = BK / kU, TDh = TD / WN, BDh = D / WN, NTH = WM * WN * 32;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const uint tid = sgi * 32 + lane;
    const uint split = threadgroup_position_in_grid.x;
    const uint bh = threadgroup_position_in_grid.y;
    const int S = info[1], T0 = info[2], cap = info[4];
    const int NT = cap / 32;
    const short row_group = sgi / WN, d_group = sgi % WN;
    const short tm = kU * row_group;
    using otile_t = NAXTile<float, 1, TDh>;
    using stile_t = NAXTile<float, 1, TK>;
    constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
    threadgroup half cbk_s[1 << KB];
    threadgroup half cbv_s[1 << VB];
    threadgroup float ksc[BK], vsc[BK];
    threadgroup int kab[BK];
    threadgroup float smf[SMF];                                     // q rotation scratch (16 rows), then the decoded K / V block (fp16 BK x LDK) + the score exchange
    threadgroup half* kvb = (threadgroup half*)smf;
    threadgroup float* xch = smf + (BK * LDK) / 2;
    for (uint i = tid; i < (1u << KB); i += NTH) cbk_s[i] = (half)cbk[i];
    for (uint i = tid; i < (1u << VB); i += NTH) cbv_s[i] = (half)cbv[i];
    const float qscl = scl[0] * 1.44269504089f;                     // scores in the log2 domain (exp2 softmax), max stored back in natural log
    NAXTile<half, 1, 1> Qtiles[TDh];
    const short2 sc0 = otile_t::NAXFrag_t::get_coord();
    const short sm = sc0.y, sn = sc0.x;
    for (int rg = 0; rg < WM; rg++) {                               // q -> H diag(s) q * scale, one row group (16 rows) at a time
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = (int)tid; i < 16 * D; i += NTH) {
            int r = rg * 16 + i / D;
            smf[i] = r < REP ? (float)q[((int)bh * REP + r) * D + i % D] * sgn[i % D] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int len = 1; len < D; len <<= 1) {
            for (int idx = (int)tid; idx < 16 * D / 2; idx += NTH) {
                int rr = idx / (D / 2), k = idx % (D / 2);
                int a = rr * D + (k / len) * 2 * len + (k % len);
                float x0 = smf[a], x1 = smf[a + len];
                smf[a] = x0 + x1; smf[a + len] = x0 - x1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row_group == rg) {
            STEEL_PRAGMA_UNROLL for (short id = 0; id < TDh; id++) {
                thread auto& fq = Qtiles[id].frag_at(0, 0);
                STEEL_PRAGMA_UNROLL for (short i = 0; i < 2; i++) STEEL_PRAGMA_UNROLL for (short j = 0; j < 4; j++)
                    fq[i * 4 + j] = (half)(smf[(sm + i * 8) * D + d_group * BDh + id * kU + sn + j] * qscl);
            }
        }
    }
    otile_t Otile;
    Otile.clear();
    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<float, kRowsPT> max_score, sum_score{0};
    STEEL_PRAGMA_UNROLL for (short i = 0; i < kRowsPT; ++i) max_score[i] = -1e30f;
    const int base0 = S & ~31;
    const int per = ((T0 - base0 + NSPLIT - 1) / NSPLIT + 31) / 32 * 32;
    const int j0 = base0 + (int)split * per, j1 = min(T0, j0 + per), ja = S;
    const int key = (int)tid % BK;
    // RING: the extra split NSPLIT runs the fp16 part (sink + ring keys, per-row visibility) and the few pool keys that rows past the window still read at pool precision
    const bool rs = RING > 0 && (int)split >= NSPLIT;
    const int nv = S + (L - T0), nfp = (nv + BK - 1) / BK;
    const int sa = max(S, T0), sl = L - WIN, tb0 = sa & ~31;
    const int nst = (sa < sl) ? (sl - tb0 + 31) / 32 : 0;
    const int tot = nfp + nst, rsi = (int)split - NSPLIT;           // RING threadgroups share the tot ring-part blocks
    const int it0 = rs ? rsi * tot / RING : 0;
    const int nit = rs ? (rsi + 1) * tot / RING : (j1 > j0 ? (j1 - j0 + BK - 1) / BK : 0);
#if ABL & 32
    if (j0 < 0)
#endif
    for (int it = it0; it < nit; it++) {
        const bool fp = rs && it < nfp;
        const int kb = rs ? (fp ? it * BK : tb0 + (it - nfp) * 32) : j0 + it * BK;   // pool: a multiple of 32 = one K tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device uint* kp = kw + ((size_t)((int)bh * NT + (kb >> 5)) * (G * KB)) * 32 + key;
        if (fp) {
            for (int i = (int)tid; i < BK * D; i += NTH) {
                const int c = i / D, d = i % D, a = tq_fpos(kb + c, nv, S, T0);
                float x = 0.0f;
                if (a >= 0) x = a < S ? (float)sk[((int)bh * NS + a) * D + d] : (float)rk[((int)bh * NR + (a - (L - NR))) * D + d];
                kvb[c * LDK + d] = (half)x;
            }
        } else
#if !(ABL & 1)
        for (int g = (int)tid / BK; g < G; g += NTH / BK) {           // thread = (key, 32-dim groups)
            uint w[KB];
            tq_ld32<KB, 32>(kp + g * KB * 32, w);
            tq_dec32<KB>(w, cbk_s, kvb + key * LDK + g * 32);
        }
#endif
        if (tid < (uint)BK) {
            const int j = kb + (int)tid; const int bb = (int)bh * cap + j;
            if (fp) {
                const int a = tq_fpos(j, nv, S, T0);
                kab[tid] = a; ksc[tid] = a >= 0 ? 1.0f : 0.0f; vsc[tid] = ksc[tid];
            } else if (rs) {
                const bool ok = j >= sa && j < sl;
                kab[tid] = ok ? j : -1; ksc[tid] = ok ? (float)ks[bb] : 0.0f; vsc[tid] = ok ? (float)vs[bb] : 0.0f;
            } else {
                const bool ok = j >= ja && j < j1;
                ksc[tid] = ok ? (float)ks[bb] : 0.0f; vsc[tid] = ok ? (float)vs[bb] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stile_t Stile;
        Stile.clear();
#if !(ABL & 4)
        STEEL_PRAGMA_UNROLL for (short ik = 0; ik < TK; ik += 2) {
            STEEL_PRAGMA_UNROLL for (short id = 0; id < TDh; id++) {
                NAXTile<half, 2, 1> Ktile;
                Ktile.template load<half, LDK, 1>(kvb + (ik * kU) * LDK + d_group * BDh + id * kU);
                stile_t::NAXFrag_t::mma(Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id].frag_at(0, 0), metal::false_type{},
                                        Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
            }
        }
#endif
        if (WN > 1) {
            constexpr short NE = TK * kEPF;
            threadgroup float* slot = xch + (row_group * WN + d_group) * (NE * 32);
            const short base = short(lane) * NE;
            STEEL_PRAGMA_UNROLL for (short ii = 0; ii < NE; ii++) slot[base + ii] = Stile.elems()[ii];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL for (short ii = 0; ii < NE; ii++) {
                float s = 0.0f;
                STEEL_PRAGMA_UNROLL for (short pg = 0; pg < WN; pg++) s += xch[(row_group * WN + pg) * (NE * 32) + base + ii];
                Stile.elems()[ii] = s;
            }
        }
#if !(ABL & 16)
        const bool edge = kb < ja || kb + BK > j1;
        STEEL_PRAGMA_UNROLL for (short ik = 0; ik < TK; ik++) {     // per-key K scale, keys outside [ja, j1) out
            thread auto& fg = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL for (short ii = 0; ii < stile_t::kFragThrRows; ii++) STEEL_PRAGMA_UNROLL for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                const int c = ik * kU + jj + sn;
                const auto loc = ii * stile_t::kFragThrCols + jj;
                if (rs) {
                    const int pr = L - QT + ((tm + ii * stile_t::kFragRowsJump + sm) % QT);   // absolute position of this score row
                    const int a = kab[c];
                    const bool vis = a >= 0 && (fp ? (a <= pr && (a < S || a > pr - WIN)) : a <= pr - WIN);
                    fg[loc] = vis ? fg[loc] * ksc[c] : -INFINITY;
                } else {
                    fg[loc] = (!edge || (kb + c >= ja && kb + c < j1)) ? fg[loc] * ksc[c] : -1e30f;
                }
            }
        }
        metal::vec<float, kRowsPT> new_max, factor;
        STEEL_PRAGMA_UNROLL for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
        Stile.template row_reduce<KMaxOp>(new_max);
        Stile.template row_bin_op<KExpSubOp>(new_max);
        STEEL_PRAGMA_UNROLL for (short i = 0; i < kRowsPT; ++i) { factor[i] = fast::exp2(max_score[i] - new_max[i]); max_score[i] = new_max[i]; sum_score[i] *= factor[i]; }
        Stile.template row_reduce<KSumOp>(sum_score);
        Otile.template row_bin_op<KMulOp>(factor);
        STEEL_PRAGMA_UNROLL for (short ik = 0; ik < TK; ik++) {     // per-key V scale on P
            thread auto& fg = Stile.frag_at(0, ik);
            STEEL_PRAGMA_UNROLL for (short ii = 0; ii < stile_t::kFragThrRows; ii++) STEEL_PRAGMA_UNROLL for (short jj = 0; jj < stile_t::kFragThrCols; jj++)
                fg[ii * stile_t::kFragThrCols + jj] *= vsc[ik * kU + jj + sn];
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (fp) {
            for (int i = (int)tid; i < BK * D; i += NTH) {
                const int c = i / D, d = i % D, a = tq_fpos(kb + c, nv, S, T0);
                float x = 0.0f;
                if (a >= 0) x = a < S ? (float)sv[((int)bh * NS + a) * D + d] : (float)rv[((int)bh * NR + (a - (L - NR))) * D + d];
                kvb[c * LDK + d] = (half)x;
            }
        } else
#if !(ABL & 2)
        for (int g = (int)tid / BK; g < G; g += NTH / BK) {
            uint w[VB];
            tq_ld32<VB, 1>(vw + ((size_t)((int)bh * cap + kb + key)) * (G * VB) + g * VB, w);
            tq_dec32<VB>(w, cbv_s, kvb + key * LDK + g * 32);
        }
#else
        {}
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if !(ABL & 8)
        STEEL_PRAGMA_UNROLL for (short id = 0; id < TDh; id += 2) {
            STEEL_PRAGMA_UNROLL for (short ik = 0; ik < TK; ik++) {
                NAXTile<half, 1, 2> Vtile;
                Vtile.template load<half, LDK, 1>(kvb + (ik * kU) * LDK + d_group * BDh + id * kU);
                otile_t::NAXFrag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{},
                                        Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
            }
        }
#endif
    }
    const int prow = ((int)bh * (NSPLIT + (RING > 0 ? RING : 1)) + (int)split) * REP;
    STEEL_PRAGMA_UNROLL for (short id = 0; id < TDh; id++) {
        thread auto& fo = Otile.frag_at(0, id);
        STEEL_PRAGMA_UNROLL for (short i = 0; i < 2; i++) {
            const int r = tm + sm + i * 8;
            if (r < REP) STEEL_PRAGMA_UNROLL for (short j = 0; j < 4; j++) po[(prow + r) * D + d_group * BDh + id * kU + sn + j] = fo[i * 4 + j];
        }
    }
    if (d_group == 0 && sn == 0) {
        STEEL_PRAGMA_UNROLL for (short i = 0; i < kRowsPT; ++i) {
            const int r = tm + (i / stile_t::kFragThrRows) * kU + (i % stile_t::kFragThrRows) * stile_t::kFragRowsJump + sm;
            if (r < REP) { pml[(prow + r) * 2] = nit > it0 ? max_score[i] * M_LN2_F : -INFINITY; pml[(prow + r) * 2 + 1] = sum_score[i]; }
        }
    }
"""

_DEC3_SRC = """
    // verify rows on simdgroup 8x8 matrices: S = Q K^T and O += P V per 64-key block (fp16 operands, fp32 accumulate); K / V codes decoded ONCE per key
    // per threadgroup into a per-simdgroup fp16 stage, shared by all RC virtual heads (the scalar kernel spent one threadgroup q load per 4 FMAs)
    constexpr int G = D / 32;
    constexpr int RP = (RC + 7) / 8 * 8;
    constexpr int RT = RP / 8;
    constexpr int NCH = REP / RC;
    constexpr int DPL = D / 32;
    constexpr int DV = D / 8;                                       // V dims per simdgroup
    constexpr int CT = DV / 8;
    constexpr int NV = D / 16;                                      // V codes per lane per 16-key chunk
    uint split = threadgroup_position_in_grid.x;
    uint hk = threadgroup_position_in_grid.y / NCH;
    int r0 = (int)(threadgroup_position_in_grid.y % NCH) * RC;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint tid = sg * 32 + lane;
    int S = info[1], T0 = info[2], cap = info[4];
    int NT = cap / 32;
    threadgroup float cbk_s[1 << KB];
    threadgroup float cbv_s[1 << VB];
    threadgroup half qsh[RP * D];
    threadgroup half stg[8 * 512];
    threadgroup float ss[RP * 64];
    threadgroup half ph[RP * 64];
    threadgroup float ksx[64], vsx[64];
    threadgroup float mrun[RP], lrun[RP], alr[RP];
    if (tid < (1u << KB)) cbk_s[tid] = cbk[tid];
    if (tid < (1u << VB)) cbv_s[tid] = cbv[tid];
    if (tid < (uint)RP) { mrun[tid] = -INFINITY; lrun[tid] = 0.0f; alr[tid] = 1.0f; }
    for (int i = (int)tid; i < RP * 64; i += 256) ph[i] = 0.0h;
    float qscl = scl[0];
    for (int r = (int)sg; r < RP; r += 8) {                         // q -> H diag(s) q * scale / sqrt(D): one row per simdgroup, butterflies in registers + lane shuffles
        float x[DPL];
        if (r < RC) {
            _Pragma("unroll") for (int i = 0; i < DPL; i++) { int d = (int)lane * DPL + i; x[i] = (float)q[((int)hk * REP + r0 + r) * D + d] * sgn[d]; }
            for (int len = 1; len < DPL; len <<= 1)
                _Pragma("unroll") for (int i = 0; i < DPL; i++) if (!(i & len)) { float a = x[i], b = x[i + len]; x[i] = a + b; x[i + len] = a - b; }
            for (ushort m = 1; m < 32; m <<= 1)
                _Pragma("unroll") for (int i = 0; i < DPL; i++) { float v = simd_shuffle_xor(x[i], m); x[i] = (lane & m) ? v - x[i] : x[i] + v; }
        } else {
            _Pragma("unroll") for (int i = 0; i < DPL; i++) x[i] = 0.0f;
        }
        _Pragma("unroll") for (int i = 0; i < DPL; i++) qsh[r * D + (int)lane * DPL + i] = (half)(x[i] * qscl);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const short qid = (short)lane / 4;
    const short fm = (qid & 4) + (((short)lane / 2) % 4);           // simdgroup 8x8 fragment: this lane holds (fm, fn) and (fm, fn + 1)
    const short fn = (qid & 2) * 2 + ((short)lane % 2) * 2;
    simdgroup_matrix<float, 8, 8> oacc[RT][CT];
    _Pragma("unroll") for (int t = 0; t < RT; t++) _Pragma("unroll") for (int c = 0; c < CT; c++) oacc[t][c] = simdgroup_matrix<float, 8, 8>(0.0f);
    int base0 = S & ~31;
    int per = ((T0 - base0 + NSPLIT - 1) / NSPLIT + 63) / 64 * 64;
    int j0 = base0 + (int)split * per, j1 = min(T0, j0 + per), ja = S;
    int NIT = j1 > j0 ? (j1 - j0 + 63) / 64 : 0;
    threadgroup half* st = stg + sg * 512;
    for (int it = 0; it < NIT; it++) {
        int base = j0 + it * 64;
        simdgroup_matrix<float, 8, 8> sacc[RT];
        _Pragma("unroll") for (int t = 0; t < RT; t++) sacc[t] = simdgroup_matrix<float, 8, 8>(0.0f);
        {
            int kl = (int)lane / 4, kq = ((int)lane % 4) * 16;      // simdgroup sg scores keys base + 8 sg .. + 8
            int jk = min(max(base + (int)sg * 8 + kl, ja), j1 - 1);
            const device uint* kp = kw + ((size_t)((int)hk * NT + (jk >> 5)) * (G * KB)) * 32 + (jk & 31);
            _Pragma("unroll") for (int dc = 0; dc < D / 64; dc++) {
                uint cd[16];
                tq_codes_n<KB, 16, 32>(kp, dc * 64 + kq, cd);
                _Pragma("unroll") for (int i = 0; i < 16; i++) st[kl * 64 + kq + i] = (half)cbk_s[cd[i]];
                simdgroup_barrier(mem_flags::mem_threadgroup);
                _Pragma("unroll") for (int kk = 0; kk < 8; kk++) {
                    simdgroup_matrix<half, 8, 8> b;
                    simdgroup_load(b, st + kk * 8, 64, ulong2(0, 0), true);
                    _Pragma("unroll") for (int t = 0; t < RT; t++) {
                        simdgroup_matrix<half, 8, 8> a;
                        simdgroup_load(a, qsh + t * 8 * D + dc * 64 + kk * 8, D);
                        simdgroup_multiply_accumulate(sacc[t], a, b, sacc[t]);
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        _Pragma("unroll") for (int t = 0; t < RT; t++) simdgroup_store(sacc[t], ss + t * 8 * 64 + sg * 8, 64);
        if (tid < 64) {
            int j = base + (int)tid;
            bool ok = j >= ja && j < j1;
            int bb = (int)hk * cap + min(max(j, ja), j1 - 1);
            ksx[tid] = ok ? (float)ks[bb] : NAN; vsx[tid] = ok ? (float)vs[bb] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = (int)sg; r < RC; r += 8) {
            float k0 = ksx[lane], k1 = ksx[lane + 32];
            float s0 = isnan(k0) ? -INFINITY : ss[r * 64 + (int)lane] * k0;
            float s1 = isnan(k1) ? -INFINITY : ss[r * 64 + (int)lane + 32] * k1;
            float mx = simd_max(max(s0, s1));
            float mo = mrun[r];
            float mn = max(mo, mx);
            float al = (mn == -INFINITY) ? 1.0f : exp(mo - mn);
            float e0 = (s0 == -INFINITY) ? 0.0f : exp(s0 - mn), e1 = (s1 == -INFINITY) ? 0.0f : exp(s1 - mn);
            float sum = simd_sum(e0 + e1);
            ph[r * 64 + (int)lane] = (half)(e0 * vsx[lane]);
            ph[r * 64 + (int)lane + 32] = (half)(e1 * vsx[lane + 32]);
            if (lane == 0) { lrun[r] = lrun[r] * al + sum; mrun[r] = mn; alr[r] = al; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("unroll") for (int t = 0; t < RT; t++) {
            float a0 = alr[t * 8 + fm];
            _Pragma("unroll") for (int c = 0; c < CT; c++) { thread auto& e = oacc[t][c].thread_elements(); e[0] *= a0; e[1] *= a0; }
        }
        _Pragma("unroll") for (int kc = 0; kc < 4; kc++) {
            if (base + kc * 16 >= j1) break;
            int kl = (int)lane / 2, kd = ((int)lane % 2) * NV;
            int jk = min(max(base + kc * 16 + kl, ja), j1 - 1);
            uint cd[NV];
            tq_codes_n<VB, NV, 1>(vw + ((size_t)((int)hk * cap + jk)) * (G * VB), (int)sg * DV + kd, cd);
            _Pragma("unroll") for (int i = 0; i < NV; i++) st[kl * DV + kd + i] = (half)cbv_s[cd[i]];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            _Pragma("unroll") for (int k8 = 0; k8 < 2; k8++) {
                simdgroup_matrix<half, 8, 8> a[RT];
                _Pragma("unroll") for (int t = 0; t < RT; t++) simdgroup_load(a[t], ph + t * 8 * 64 + kc * 16 + k8 * 8, 64);
                _Pragma("unroll") for (int c = 0; c < CT; c++) {
                    simdgroup_matrix<half, 8, 8> b;
                    simdgroup_load(b, st + k8 * 8 * DV + c * 8, DV);
                    _Pragma("unroll") for (int t = 0; t < RT; t++) simdgroup_multiply_accumulate(oacc[t][c], a[t], b, oacc[t][c]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    _Pragma("unroll") for (int t = 0; t < RT; t++) {
        int r = t * 8 + fm;
        if (r >= RC) continue;
        int row = ((int)hk * (NSPLIT + 1) + (int)split) * REP + r0 + r;
        _Pragma("unroll") for (int c = 0; c < CT; c++) {
            thread auto& e = oacc[t][c].thread_elements();
            int col = (int)sg * DV + c * 8 + fn;
            po[row * D + col] = e[0]; po[row * D + col + 1] = e[1];
        }
    }
    if (tid < (uint)RC) { int row = ((int)hk * (NSPLIT + 1) + (int)split) * REP + r0 + (int)tid; pml[row * 2] = mrun[tid]; pml[row * 2 + 1] = lrun[tid]; }
"""

_HEADER3 = """
#include <metal_simdgroup_matrix>
// N codes of dims [d0, d0 + N) of one row (N in 4, 8, 16; d0 a multiple of N); STR = word stride (1 = row layout, 32 = tiled K layout)
template <int BITS, int N, int STR>
inline void tq_codes_n(const device uint* p, int d0, thread uint* c) {
    const device uint* w = p + (d0 >> 5) * BITS * STR;
    int dd = d0 & 31;
    constexpr int NW4 = (N + 7) / 8;
    uint w4[NW4];
    _Pragma("unroll") for (int i = 0; i < NW4; i++) w4[i] = w[(dd / 8 + i) * STR];
    _Pragma("unroll") for (int j = 0; j < N; j++) { int x = (dd % 8) + j; c[j] = extract_bits(w4[x / 8], (x % 8) * 4, 4); }
    if (BITS & 2) { uint w2 = w[(4 + dd / 16) * STR]; _Pragma("unroll") for (int j = 0; j < N; j++) c[j] = (c[j] << 2) | extract_bits(w2, ((dd % 16) + j) * 2, 2); }
    if (BITS & 1) { uint w1 = w[(4 + ((BITS & 2) ? 2 : 0)) * STR]; _Pragma("unroll") for (int j = 0; j < N; j++) c[j] = (c[j] << 1) | extract_bits(w1, dd + j, 1); }
}
"""


# merge of the NSPLIT (+ the fp16 split) partials of one query head + inverse rotation (out = diag(s) H y / sqrt(D)); one threadgroup per head
_MRG_SRC = """
    uint h = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    int hk = (int)h / REP, r = (int)h % REP;
    threadgroup float xs[D];
    float mt = -INFINITY;
    for (int s = 0; s < NUSE; s++) mt = max(mt, pml[((hk * NSPLIT + s) * REP + r) * 2]);
    if (OV) mt = max(mt, pml2[(hk * REP + r) * 2]);
    float lt = 0.0f, o = 0.0f;
    for (int s = 0; s < NUSE; s++) {
        int row = (hk * NSPLIT + s) * REP + r;
        float ms = pml[row * 2];
        float w = (ms == -INFINITY) ? 0.0f : exp(ms - mt);
        lt += w * pml[row * 2 + 1];
        if (tid < (uint)D) o += w * po[row * D + tid];
    }
    if (OV) {                                                       // the separate fp16-part partials (verify rows)
        int row = hk * REP + r;
        float ms = pml2[row * 2];
        float w = (ms == -INFINITY) ? 0.0f : exp(ms - mt);
        lt += w * pml2[row * 2 + 1];
        if (tid < (uint)D) o += w * po2[row * D + tid];
    }
    if (tid < (uint)D) xs[tid] = o / lt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int len = 1; len < D; len <<= 1) {
        if (tid < (uint)(D / 2)) {
            int k = (int)tid;
            int a = (k / len) * 2 * len + (k % len);
            float x0 = xs[a], x1 = xs[a + len];
            xs[a] = x0 + x1; xs[a + len] = x0 - x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < (uint)D) out[h * D + tid] = (T)(xs[tid] * sgn[tid] * scl[0]);
"""

_kernels = {}
_INFO = {}
STATS = {}


ABL2 = 0                                                            # experiment (tools/abl102.py): ablation bits of the verify-rows kernel, 0 = the real kernel


def _kernel(name):
    if name in ("dec2", "naxq") and ABL2:
        name = name + ":%d" % ABL2
    if name not in _kernels:
        if name == "write":
            _kernels[name] = mx.fast.metal_kernel("tq_write", ["y", "mid", "cb"], ["words", "scale"], _WRITE_SRC, header=_HEADER)
        elif name == "deq":
            _kernels[name] = mx.fast.metal_kernel("tq_dequant", ["words", "scale", "cb"], ["out"], _DEQ_SRC, header=_HEADER)
        elif name == "deqt":
            _kernels[name] = mx.fast.metal_kernel("tq_dequant_tiled", ["words", "scale", "cb", "info"], ["out"], _DEQT_SRC, header=_HEADER)
        elif name.startswith("dec2"):
            _kernels[name] = mx.fast.metal_kernel(
                "tq_decode_rows" + ("_a%d" % ABL2 if ABL2 else ""), ["q", "kw", "ks", "vw", "vs", "sk", "sv", "rk", "rv", "cbk", "cbv", "info", "sgn", "scl"], ["po", "pml"], _DEC2_SRC,
                header=("#define ABL %d\n" % ABL2) + "#define QT_ %s\n#define QDOT(p, kv) %s\n" % (("half", "dot(float4(*(threadgroup half4*)(p)), kv)") if QHALF == 1 else ("half", "float(dot(*(threadgroup half4*)(p), half4(kv)))") if QHALF == 2 else ("float", "dot(*(threadgroup float4*)(p), kv)")) + _HEADER)
        elif name.startswith("naxq"):
            from .sdpa_dec import _nax_header
            _kernels[name] = mx.fast.metal_kernel(
                "tq_decode_rows_nax" + ("_a%d" % ABL2 if ABL2 else ""), ["q", "kw", "ks", "vw", "vs", "sk", "sv", "rk", "rv", "cbk", "cbv", "info", "sgn", "scl"], ["po", "pml"], _DECN_SRC,
                header=("#define ABL %d\n" % ABL2) + _nax_header() + _DECN_HDR)
        elif name == "dec3":
            _kernels[name] = mx.fast.metal_kernel(
                "tq_decode_rows_mma", ["q", "kw", "ks", "vw", "vs", "sk", "sv", "rk", "rv", "cbk", "cbv", "info", "sgn", "scl"], ["po", "pml"], _DEC3_SRC,
                header=_HEADER + _HEADER3)
        elif name == "mrg":
            _kernels[name] = mx.fast.metal_kernel("tq_merge", ["po", "pml", "po2", "pml2", "sgn", "scl"], ["out"], _MRG_SRC)
        else:
            _kernels[name] = mx.fast.metal_kernel(
                "tq_decode_tiled", ["q", "kw", "ks", "vw", "vs", "sk", "sv", "rk", "rv", "cbk", "cbv", "info", "sgn", "scl"], ["po", "pml"], _DEC_SRC, header="#define ABL %d\n#define QDOT(p, kv) dot(*(threadgroup float4*)(p), kv)\n" % int(os.environ.get("KYOJIN_TQ_ABL", "0")) + _HEADER)
    return _kernels[name]


@lru_cache(maxsize=None)
def _sgn_mx(d):
    return mx.array(np.array([1.0 if c == "1" else -1.0 for c in SIGNS[:d]], dtype=np.float32))


_SCL = {}


def _scl_mx(scale, d, inv):
    """scale / sqrt(d) (query side) or 1 / sqrt(d) (output side) as a 1-element device array."""
    key = (scale, d, inv)
    if key not in _SCL:
        _SCL[key] = mx.array([(1.0 if inv else scale) / math.sqrt(d)], dtype=mx.float32)
    return _SCL[key]


@lru_cache(maxsize=None)
def _cb_mx(d, bits):
    return mx.array(codebook(d, bits)), mx.array(midpoints(d, bits))


def tq_write(y, bits):
    """y fp32 [N, D] rotated -> (words uint32 [N, D / 32 * bits], scale fp16 [N])."""
    n, d = y.shape
    cb, mid = _cb_mx(d, bits)
    nw = d // 32 * bits
    return _kernel("write")(inputs=[y, mid, cb], template=[("D", d), ("BITS", bits), ("NW", nw)], grid=(n * 32, 1, 1), threadgroup=(128, 1, 1),
                            output_shapes=[(n, nw), (n,)], output_dtypes=[mx.uint32, mx.float16])


def tq_dequant(words, scale, bits, d, dtype=mx.float16):
    """words [N, D / 32 * bits], scale [N] -> [N, D] (rotated domain)."""
    n = words.shape[0]
    cb, _ = _cb_mx(d, bits)
    return _kernel("deq")(inputs=[words, scale, cb], template=[("D", d), ("BITS", bits), ("NW", d // 32 * bits), ("T", dtype)], grid=(n * 32, 1, 1),
                          threadgroup=(128, 1, 1), output_shapes=[(n, d)], output_dtypes=[dtype])[0]


def rows_to_tiles(w):
    """K words, row layout [B, H, n, NW] (n % 32 == 0) -> tiled layout [B, H, n / 32, NW, 32] (the runtime layout of the K pool)."""
    B, H, n, nw = w.shape
    return mx.contiguous(w.reshape(B, H, n // 32, 32, nw).transpose(0, 1, 2, 4, 3))      # a transposed view would be copied again by every kernel launch


def tiles_to_rows(t):
    B, H, nt, nw, _ = t.shape
    return t.transpose(0, 1, 2, 4, 3).reshape(B, H, nt * 32, nw)


def tq_dequant_tiled(words, scale, bits, d, lo, hi, dtype=mx.float16):
    """words [B, H, cap / 32, D / 32 * bits, 32] (tiled K pool), scale [B, H, cap] -> rows [lo, hi) as [B, H, hi - lo, D] (rotated domain)."""
    B, H, nt = words.shape[:3]
    cb, _ = _cb_mx(d, bits)
    n = hi - lo
    info = mx.array([lo, n, nt, scale.shape[-1]], dtype=mx.int32)
    out = _kernel("deqt")(inputs=[words, scale, cb, info], template=[("D", d), ("BITS", bits), ("T", dtype)], grid=(n * 32, B * H, 1), threadgroup=(128, 1, 1),
                          output_shapes=[(B * H, n, d)], output_dtypes=[dtype])[0]
    return out.reshape(B, H, n, d)


@lru_cache(maxsize=None)
def _cores() -> int:
    from .tuning import gpu_cores
    return gpu_cores()


def _nsplit(keys: int, groups: int) -> int:
    """Key splits of the decode kernel: about SPLIT_TGS_PER_CORE threadgroups per GPU core over `groups` = B * Hk * query-head chunks, at least SPLIT_MIN_KEYS keys
    each, a power of two (few kernel variants). Air M5 (10 cores): Hk 4 -> 32 splits from 16K keys on, 8 at 4K; a bigger chip splits further."""
    if NSPLIT > 0:
        return NSPLIT
    want = min(SPLIT_TGS_PER_CORE * _cores() / groups, keys / SPLIT_MIN_KEYS)
    return max(4, 1 << max(int(want), 1).bit_length() - 1) if want >= 1 else 4


def _naxq_ok(d: int, repv: int) -> bool:
    """The NAX verify-rows kernel: matrix units present, head dim a multiple of 64 up to 256 (16 rows of fp32 rotation scratch fit), up to 64 rows."""
    if not NAX or d % 64 or d > 256 or repv > 64:
        return False
    from .sdpa_dec import nax_ok
    return nax_ok()


def _rcv(repv: int, d: int, kb: int = 6, vb: int = 6) -> int:
    """Rows per threadgroup of the one-pass verify kernel: the largest divisor of repv whose q rows (fp16) + 256-key score block + tables fit the 32 KiB of threadgroup memory."""
    def lds(r):
        return r * d * (2 if QHALF else 4) + 4096 * ((r + 3) // 4) + 1024 + 4 * ((1 << kb) + (1 << vb)) + 12 * r + 256
    top = RCV_MAX or max(r for r in range(1, 33) if lds(r) <= 32768)
    return max(c for c in range(1, min(repv, top) + 1) if repv % c == 0)


def _rcv3(repv: int, d: int, kb: int = 6, vb: int = 6) -> int:
    """Rows per threadgroup of the simdgroup-matrix verify kernel (_DEC3_SRC): fp16 q rows padded to 8 + 8 KiB code stage + 64-key score / P blocks within 32 KiB."""
    def lds(r):
        rp = (r + 7) // 8 * 8
        return rp * d * 2 + 8192 + rp * 64 * 6 + 4 * ((1 << kb) + (1 << vb)) + 512 + 12 * rp
    top = RCV_MAX or max(r for r in range(1, 33) if lds(r) <= 32768)
    return max(c for c in range(1, min(repv, top) + 1) if repv % c == 0)


def _rc(rep: int, d: int, rows: int = 1) -> int:
    """Query heads of a kv group per threadgroup: the largest divisor of rep that fits the registers (8 spills at head_dim 256, measured on verify chunks)."""
    top = RC_MAX or (8 if d <= 128 else 4)
    return max(c for c in range(1, min(rep, top) + 1) if rep % c == 0)


# ---------------------------------------------------------------------------------------------- cache
class TQKVCache(WinKVCache):
    """WinKVCache with a TurboQuant pool. kq = (words, scale) for K, vq likewise for V.

    LAYOUT 2 (runtime state): the K words are TILED, [B, H, cap / 32, D / 32 * bits, 32] (32 consecutive keys interleave word by word) so the fused decode
    kernel gives one key per lane (coalesced loads, no cross-lane reduce); V words stay in rows [B, H, cap, D / 32 * bits], scales [B, H, cap]. Same bits
    and codebook as before (bytes per token unchanged), only the order differs. `state` (prefix / disk checkpoints) is always the row layout, LAYOUT 1: the
    setter tiles it, so old checkpoints load unchanged and new ones stay readable by the row-layout code."""
    LAYOUT = 2
    CKPT_LAYOUT = 1
    NPOOL = 4
    SLOTS = ("kq0", "kq1", "vq0", "vq1", "sk", "sv", "rk", "rv")

    def __init__(self, kbits: int = 6, vbits: int = 6, sink: int = 4, window: int = 256, margin: int = MARGIN, f32: bool = False):
        super().__init__(kbits, vbits, 0, sink, window, margin, f32)
        self.group_size = 0
        self.fused = os.environ.get("KYOJIN_TQ_DECODE", "1") != "0"

    # -------------------------------------------------------------- write
    def _alloc(self, B, H, kd, vd, dtype, n):
        assert supported(kd, self.kbits) and supported(vd, self.vbits), f"tq: head_dim {kd}/{vd} with {self.kbits}/{self.vbits} bits is not supported"
        steps = (self.step + n - 1) // self.step * self.step

        def new(dim, bits, tiled):
            w = mx.zeros((B, H, steps // 32, dim // 32 * bits, 32) if tiled else (B, H, steps, dim // 32 * bits), dtype=mx.uint32)
            return (w, mx.zeros((B, H, steps), dtype=mx.float16))

        def cut(t, tiled, rows):
            return (t[0][:, :, :rows // 32] if tiled else t[0][:, :, :rows], t[1][:, :, :rows])

        if self.kq is None:
            self.kq, self.vq = new(kd, self.kbits, True), new(vd, self.vbits, False)
        else:
            if self.offset % self.step != 0:                    # all four arrays share one capacity, a multiple of 32 (whole K tiles)
                rows = (self.offset + 31) // 32 * 32
                self.kq, self.vq = cut(self.kq, True, rows), cut(self.vq, False, rows)
            self.kq = tuple(mx.concatenate([a, b], axis=2) for a, b in zip(self.kq, new(kd, self.kbits, True)))
            self.vq = tuple(mx.concatenate([a, b], axis=2) for a, b in zip(self.vq, new(vd, self.vbits, False)))
        from .tq_step import mark_owned
        mark_owned(self)                                        # fresh buffers: the one-dispatch decode step may write them in place

    def _cap(self):
        return self.kq[1].shape[-1]

    def update_and_fetch(self, keys, values):
        dt = keys.dtype
        H = keys.shape[1]
        kv32 = rotate(mx.concatenate([keys, values], axis=1), keys.shape[-1])         # one matmul for K and V (same head_dim)
        k32, v32 = kv32[:, :H], kv32[:, H:]
        self._src = (k32, v32)                                  # the pool quantizes the fp32 rotation, the overlay keeps the rotated rows in the model dtype
        return super().update_and_fetch(k32.astype(dt), v32.astype(dt))

    @staticmethod
    def update_many(caches, keys, values):
        """update_and_fetch of g sessions (keys / values [g, H, n, D], one per cache) with ONE rotation matmul and ONE quantize launch per K / V for the whole
        group; each cache then runs its own pool write (slice assignments), ring and sink bookkeeping. Same ops per row as update_and_fetch."""
        g, H, n, d = keys.shape
        dt = keys.dtype
        kv32 = rotate(mx.concatenate([keys, values], axis=1), d)
        k32, v32 = kv32[:, :H], kv32[:, H:]
        pre = []
        for x, bits in ((k32, caches[0].kbits), (v32, caches[0].vbits)):
            w, sc = tq_write(x.astype(mx.float32).reshape(-1, d), bits)
            pre.append((w.reshape(g, H, n, -1), sc.reshape(g, H, n)))
        outs = []
        for j, c in enumerate(caches):
            c._src = (k32[j:j + 1], v32[j:j + 1])
            c._pre = tuple((w[j:j + 1], sc[j:j + 1]) for w, sc in pre)
            try:
                outs.append(WinKVCache.update_and_fetch(c, k32[j:j + 1].astype(dt), v32[j:j + 1].astype(dt)))
            finally:
                c._pre = None
        return outs

    def _write_pool(self, prev, keys, values):
        B, H, n, _ = keys.shape
        keys, values = self._src
        pre = getattr(self, "_pre", None)
        for ix, (dst, x, bits, tiled) in enumerate(((self.kq, keys, self.kbits, True), (self.vq, values, self.vbits, False))):
            d = x.shape[-1]
            if pre is not None:
                w, s = pre[ix]
                w = w.reshape(B, H, n, -1)
            else:
                w, s = tq_write(x.astype(mx.float32).reshape(-1, d), bits)
                w = w.reshape(B, H, n, -1)
            if not tiled:
                dst[0][:, :, prev:self.offset] = w
            elif n == 1:                                        # decode: one strided word column of one tile
                dst[0][:, :, prev >> 5, :, prev & 31] = w[:, :, 0]
            elif n <= 64:                                       # verify chunk: strided columns of the (at most 3) tiles it touches, written in place
                i = 0                                           # (the rows-view patch below reads the pool it overwrites, so MLX copies the whole pool)
                while i < n:
                    p = prev + i
                    m = min(n - i, 32 - (p & 31))
                    dst[0][:, :, p >> 5, :, (p & 31):(p & 31) + m] = w[:, :, i:i + m].swapaxes(-1, -2)
                    i += m
            else:                                              # chunk: rewrite the tiles it touches (rows view, patch, back to tiles)
                t0, t1, a = prev >> 5, (self.offset + 31) >> 5, prev & 31
                blk = tiles_to_rows(dst[0][:, :, t0:t1])
                blk[:, :, a:a + n] = w
                dst[0][:, :, t0:t1] = rows_to_tiles(blk)
            dst[1][:, :, prev:self.offset] = s.reshape(B, H, n)

    # -------------------------------------------------------------- read
    def _dq(self, pool, bits, L, d, dtype):
        w, s = pool
        B, H = w.shape[0], w.shape[1]
        return tq_dequant(w[:, :, :L].reshape(-1, w.shape[-1]), s[:, :, :L].reshape(-1), bits, d, dtype).reshape(B, H, L, d)

    def _dq_k(self, lo, hi, d, dtype):
        return tq_dequant_tiled(self.kq[0], self.kq[1], self.kbits, d, lo, hi, dtype)

    def _pool_scores(self, q, L, rep, ex):
        k = self._dq_k(0, L, q.shape[-1], mx.float16)
        k = mx.expand_dims(k, -3) if rep > 1 else k
        return q @ k.astype(q.dtype).swapaxes(-1, -2)

    def _pool_pv(self, p, L, rep, ex):
        v = self._dq(self.vq, self.vbits, L, self.vq[0].shape[-1] // self.vbits * 32, mx.float16)
        v = mx.expand_dims(v, -3) if rep > 1 else v
        return p @ v.astype(p.dtype)

    def _deq_rows(self, lo, hi):
        kd = self.kq[0].shape[-2] // self.kbits * 32
        vd = self.vq[0].shape[-1] // self.vbits * 32
        dt = self.sk.dtype if self.sk is not None else mx.float16
        return (self._dq_k(lo, hi, kd, dt),
                self._dq((self.vq[0][:, :, lo:hi], self.vq[1][:, :, lo:hi]), self.vbits, hi - lo, vd, dt))

    def _overlay_as(self, dt):
        """Sink and ring rows in the model dtype: a disk checkpoint written by a run with another stream dtype (a GGUF loaded with gguf_fp16=0) restores them as stored,
        and the decode kernels take them in the query dtype."""
        self.sk, self.sv, self.rk, self.rv = (x if x is None else x.astype(dt) for x in (self.sk, self.sv, self.rk, self.rv))

    def attend(self, queries, scale: float):
        B, Hq, T, D = queries.shape
        if self.rk is not None and self.rk.dtype != queries.dtype:
            self._overlay_as(queries.dtype)
        if self.fused and D in (64, 128, 256) and (T == 1 or self._verify_ok(B, T)):
            return self._decode(queries, scale)
        qr = rotate(queries, D).astype(queries.dtype)
        out = self._prefill(qr, scale) if PF and T >= PF_MIN and self._prev is not None else super().attend(qr, scale)
        return unrotate(out, D).astype(queries.dtype)

    def _prefill(self, q, scale):
        """Chunk rows (rotated queries [B, Hq, T, D]) through ONE fused flash-attention call (no [Hq, T, L] score tensor): keys = fp16 sinks [0, S) |
        pool dequantized | fp16 tail (ring + chunk).
        PF=1: the tail is the whole ring + chunk [c0, L) and the mask is plain causal (the stock kernel skips the masked blocks): row p reads keys
          [c0, p] at fp16, a superset of its window (c0 <= p - window + 1), so every row is at least as precise as on the decode-faithful path.
        PF=2: decode-faithful (kvwin.attend semantics): pool [S, L - window) + tail with a boolean mask, row p reads key a at fp16 iff p - window < a."""
        B, Hq, T, D = q.shape
        L, S, win, dt = self.offset, min(self.sink, self.offset), self.window, q.dtype
        c0, fk = L, None
        if win > 0:
            rk, rv, _ = self._prev
            ck, cv = self._chunk
            fk = ck if rk is None else mx.concatenate([rk, ck], axis=-2)
            fv = cv if rv is None else mx.concatenate([rv, cv], axis=-2)
            c0 = max(S, L - fk.shape[-2])
        Pe = (max(S, L - win) if win > 0 else L) if PF == 2 else c0
        parts_k, parts_v, segs = [], [], []
        if S > 0:
            parts_k.append(self.sk[..., :S, :].astype(dt)); parts_v.append(self.sv[..., :S, :].astype(dt)); segs.append((0, 0, S))
        if Pe > S:
            dk, dv = self._deq_rows(S, Pe)
            parts_k.append(dk.astype(dt)); parts_v.append(dv.astype(dt)); segs.append((1, S, Pe))
        if c0 < L:
            parts_k.append(fk[..., c0 - L:, :].astype(dt)); parts_v.append(fv[..., c0 - L:, :].astype(dt)); segs.append((2, c0, L))
        k = mx.concatenate(parts_k, axis=-2) if len(parts_k) > 1 else parts_k[0]
        v = mx.concatenate(parts_v, axis=-2) if len(parts_v) > 1 else parts_v[0]
        if PF != 2:
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        key = (L, T, win, tuple(segs))
        m = _PFMASK.get(key)
        if m is None:
            if len(_PFMASK) > 4:
                _PFMASK.clear()
            pos = mx.arange(L - T, L)[:, None]
            ms = []
            for kind, a, b in segs:
                col = mx.arange(a, b)[None, :]
                ms.append(col <= pos if kind == 0 else (col <= pos - win if kind == 1 else (col <= pos) & (col > pos - win)))
            m = _PFMASK[key] = mx.concatenate(ms, axis=-1) if len(ms) > 1 else ms[0]
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m)

    def _verify_ok(self, B, T):
        """A chunk of 2..8 rows (MTP verify) can use the fused kernel when the fp16 ring still holds every key the rows read at fp16 (L - Tb of them)."""
        if B != 1 or T > 8 or self.window <= 0 or self.rk is None:
            return False
        L = self.offset
        S = min(self.sink, L)
        return self.rk.shape[2] >= L - max(S, L - self.window - T + 1)

    def _decode(self, queries, scale, gate=None):
        """Raw (unrotated) queries [B, Hq, 1, D] -> attention output [B, Hq, 1, D] in the model dtype: two dispatches (flash-decode over the splits, merge + inverse rotation)."""
        B, Hq, QT, D = queries.shape                                  # QT query rows per head (1 = decode, 2..8 = verify chunk)
        L = self.offset
        Hk = self.kq[0].shape[1]
        rep = Hq // Hk
        repv = rep * QT                                         # virtual heads per kv head: (rep, row), row fastest
        rc = _rc(repv, D, QT)
        S = min(self.sink, L)
        T0 = max(S, L - self.window - QT + 1) if self.window > 0 else L        # keys [S, T0) are pool for every row
        nsplit = _nsplit(max(T0 - S, 1), B * Hk * (repv // rc))
        z = mx.zeros((B, Hk, 1, D), dtype=queries.dtype)
        sk, sv = (self.sk, self.sv) if S > 0 else (z, z)
        rk, rv = (self.rk, self.rv) if T0 < L else (z, z)
        NR = rk.shape[2] if T0 < L else 1
        NS = sk.shape[2] if S > 0 else 1
        key = (L, S, T0, NR, self._cap(), NS, self.window)                  # the same for every layer of a step: one host -> device copy per step, not per layer
        info = _INFO.get(key)
        if info is None:
            if len(_INFO) > 8:
                _INFO.clear()
            info = _INFO[key] = mx.array(list(key), dtype=mx.int32)
        ins = [queries.reshape(B * Hq * QT, D), self.kq[0], self.kq[1], self.vq[0], self.vq[1], sk, sv, rk, rv, _cb_mx(D, self.kbits)[0], _cb_mx(D, self.vbits)[0], info,
               _sgn_mx(D), _scl_mx(scale, D, False)]
        tpl = lambda rc_, ns_: [("D", D), ("KB", self.kbits), ("VB", self.vbits), ("REP", repv), ("RC", rc_), ("QT", QT), ("NSPLIT", ns_), ("GA", ga), ("VU", VU), ("VX", VX), ("VH", VH), ("CF", CF), ("OV4", OV4), ("OVU", OVU), ("T", queries.dtype)]
        rcv = (_rcv3 if ROWS == 2 else _rcv)(repv, D, self.kbits, self.vbits) if QT > 1 and ROWS else rc
        ga = GA_ENV or (4 if rcv <= 12 else 2)
        ga = max(ga, D // 128)                                          # a lane covers at most 4 groups of ... (NLN = D / ga lanes must be <= 256)
        if QT > 1 and T0 - S >= ROWS_MIN and _naxq_ok(D, repv):
            # verify rows on the matrix units: K / V codes decoded once per 32-key block for all repv rows; the fp16 part keeps the plain kernel
            wm = -(-repv // 16)
            lds = lambda w: 4 * max(16 * D, 16 * (D + NAX_PAD) + (wm * w * 2 * 8 * 32 if w > 1 else 0)) + 1024
            wn = NAX_WN or (2 if lds(2) <= 32768 else 1)                    # head-dim groups: 2 (scores summed through threadgroup memory) when it fits
            nsplit = _nsplit(max(T0 - S, 1), B * Hk) * NAX_NSX
            smf = (lds(wn) - 1024) // 4
            po, pml = _kernel("naxq")(inputs=ins, template=[("D", D), ("KB", self.kbits), ("VB", self.vbits), ("REP", repv), ("NSPLIT", nsplit), ("BK", 32), ("WM", wm),
                                                            ("WN", wn), ("LDK", D + NAX_PAD), ("SMF", smf), ("RING", NAX_RING), ("QT", QT)],
                                      grid=(32 * wm * wn * (nsplit + NAX_RING), B * Hk, 1), threadgroup=(32 * wm * wn, 1, 1),
                                      output_shapes=[(B * Hk, nsplit + max(NAX_RING, 1), repv, D), (B * Hk, nsplit + max(NAX_RING, 1), repv, 2)], output_dtypes=[mx.float32, mx.float32])
            if NAX_RING:                                                # the fp16 part ran as split NSPLIT of the same dispatch
                mt = [("D", D), ("REP", repv), ("NSPLIT", nsplit + NAX_RING), ("NUSE", nsplit + NAX_RING), ("OV", 0), ("T", queries.dtype)]
                mi = [po, pml, po, pml]
            else:
                po2, pml2 = _kernel("dec")(inputs=ins, template=tpl(rc, 0), grid=(256, B * Hk * (repv // rc), 1), threadgroup=(256, 1, 1),
                                           output_shapes=[(B * Hk, 1, repv, D), (B * Hk, 1, repv, 2)], output_dtypes=[mx.float32, mx.float32])
                mt = [("D", D), ("REP", repv), ("NSPLIT", nsplit + 1), ("NUSE", nsplit), ("OV", 1), ("T", queries.dtype)]
                mi = [po, pml, po2, pml2]
        elif rcv > rc and T0 - S >= ROWS_MIN:
            # verify rows: the pool is read ONCE per chunk of rcv rows (instead of once per rc rows); the fp16 part (sink + ring) keeps the plain kernel; both partial sets meet in the merge
            nch = repv // rcv
            nsplit = _nsplit(max(T0 - S, 1), B * Hk * nch)
            po, pml = _kernel("dec3" if ROWS == 2 else "dec2")(inputs=ins, template=tpl(rcv, nsplit), grid=(256 * nsplit, B * Hk * nch, 1), threadgroup=(256, 1, 1),
                                      output_shapes=[(B * Hk, nsplit + 1, repv, D), (B * Hk, nsplit + 1, repv, 2)], output_dtypes=[mx.float32, mx.float32])
            po2, pml2 = _kernel("dec")(inputs=ins, template=tpl(rc, 0), grid=(256, B * Hk * (repv // rc), 1), threadgroup=(256, 1, 1),
                                       output_shapes=[(B * Hk, 1, repv, D), (B * Hk, 1, repv, 2)], output_dtypes=[mx.float32, mx.float32])
            mt = [("D", D), ("REP", repv), ("NSPLIT", nsplit + 1), ("NUSE", nsplit), ("OV", 1), ("T", queries.dtype)]
            mi = [po, pml, po2, pml2]
        else:
            po, pml = _kernel("dec")(inputs=ins, template=tpl(rc, nsplit),
                                     grid=((256 * (nsplit + 1) * (repv // rc), B * Hk, 1) if CF else (256 * (nsplit + 1), B * Hk * (repv // rc), 1)), threadgroup=(256, 1, 1),
                                     output_shapes=[(B * Hk, nsplit + 1, repv, D), (B * Hk, nsplit + 1, repv, 2)], output_dtypes=[mx.float32, mx.float32])
            mt = [("D", D), ("REP", repv), ("NSPLIT", nsplit + 1), ("NUSE", nsplit + 1), ("OV", 0), ("T", queries.dtype)]
            mi = [po, pml, po, pml]
        if gate is not None:                                          # decode only: the output gate (sigmoid of the q projection's second half) is applied by the merge
            from .tq_step import merge_gated_kernel
            out = merge_gated_kernel()(inputs=mi + [_sgn_mx(D), _scl_mx(scale, D, True), gate], template=mt, grid=(256 * B * Hq * QT, 1, 1), threadgroup=(256, 1, 1),
                                       output_shapes=[(B * Hq * QT, D)], output_dtypes=[queries.dtype])[0]
            return out.reshape(B, 1, Hq * D)
        out = _kernel("mrg")(inputs=mi + [_sgn_mx(D), _scl_mx(scale, D, True)], template=mt,
                             grid=(256 * B * Hq * QT, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(B * Hq * QT, D)], output_dtypes=[queries.dtype])[0]
        return out.reshape(B, Hq, QT, D)

    def _fp16_part(self, queries, scale, T0):
        """The fp16 part (sink + ring keys [T0, L), per-row visibility) of a verify chunk as partials (po2, pml2) for the merge: the separate plain-kernel pass of
        `_decode`, with the ring start moved down to T0 (<= the natural start), so that a pool pass over [S, T0) done elsewhere (attend_shared) meets it."""
        B, Hq, QT, D = queries.shape
        if self.rk is not None and self.rk.dtype != queries.dtype:
            self._overlay_as(queries.dtype)
        L = self.offset
        Hk = self.kq[0].shape[1]
        rep = Hq // Hk
        repv = rep * QT
        rc = _rc(repv, D, QT)
        S = min(self.sink, L)
        z = mx.zeros((B, Hk, 1, D), dtype=queries.dtype)
        sk, sv = (self.sk, self.sv) if S > 0 else (z, z)
        rk, rv = (self.rk, self.rv) if T0 < L else (z, z)
        NR = rk.shape[2] if T0 < L else 1
        NS = sk.shape[2] if S > 0 else 1
        key = (L, S, T0, NR, self._cap(), NS, self.window)
        info = _INFO.get(key)
        if info is None:
            if len(_INFO) > 8:
                _INFO.clear()
            info = _INFO[key] = mx.array(list(key), dtype=mx.int32)
        ins = [queries.reshape(B * Hq * QT, D), self.kq[0], self.kq[1], self.vq[0], self.vq[1], sk, sv, rk, rv, _cb_mx(D, self.kbits)[0], _cb_mx(D, self.vbits)[0], info,
               _sgn_mx(D), _scl_mx(scale, D, False)]
        rcv = (_rcv3 if ROWS == 2 else _rcv)(repv, D, self.kbits, self.vbits) if QT > 1 and ROWS else rc
        ga = GA_ENV or (4 if rcv <= 12 else 2)
        ga = max(ga, D // 128)
        tpl = [("D", D), ("KB", self.kbits), ("VB", self.vbits), ("REP", repv), ("RC", rc), ("QT", QT), ("NSPLIT", 0), ("GA", ga), ("VU", VU), ("VX", VX), ("VH", VH), ("CF", CF), ("OV4", OV4), ("OVU", OVU), ("T", queries.dtype)]
        return _kernel("dec")(inputs=ins, template=tpl, grid=(256, B * Hk * (repv // rc), 1), threadgroup=(256, 1, 1),
                              output_shapes=[(B * Hk, 1, repv, D), (B * Hk, 1, repv, 2)], output_dtypes=[mx.float32, mx.float32])

    @staticmethod
    def attend_shared(caches, queries, scale: float, P: int):
        """Verify chunks of several sessions that share the first P keys of their pool (forks of one prompt cache): the pool keys [sink, Pq) are read ONCE for
        the query rows of all sessions (the NAX verify kernel with the sessions' rows folded into its row axis, one pool = the first session's), each session's own
        sink + ring keys [Pq, L) run through the plain kernel pass, one merge launch finishes all sessions. Pq <= min(P, every session's natural pool end), so
        every session sees the same keys, at the same precision, as its own `attend` (row visibility is the kernel's own). queries [g, Hq, T, D] raw, one per
        cache (offsets already advanced). Returns [g, Hq, T, D], or None when it does not apply (the caller attends per session)."""
        g, Hq, QT, D = queries.shape
        c0 = caches[0]
        if (QT < 2 or QT > 8 or P <= 0 or not NAX or any(not (type(c) is TQKVCache and c.fused and c.rk is not None and c.window > 0) for c in caches)
                or D not in (64, 128, 256)):
            return None
        Hk = c0.kq[0].shape[1]
        rep = Hq // Hk
        gmax = 64 // (rep * QT)
        if gmax < 2 or not _naxq_ok(D, rep * QT * 2):
            return None
        sink = min(c0.sink, min(c.offset for c in caches))
        t0s = [max(min(c.sink, c.offset), c.offset - c.window - QT + 1) for c in caches]
        Pq = min(P, min(t0s))
        if Pq - sink < ROWS_MIN or any(c.offset - Pq > c.rk.shape[2] or c.kbits != c0.kbits or c.vbits != c0.vbits for c in caches):
            return None
        STATS['shared'] = STATS.get('shared', 0) + 1
        po2s, pml2s = [], []
        for j, c in enumerate(caches):
            a, b = c._fp16_part(queries[j:j + 1], scale, Pq)
            po2s.append(a); pml2s.append(b)
        outs = []
        for lo in range(0, g, gmax):
            hi = min(g, lo + gmax)
            n = hi - lo
            repv = n * rep * QT
            qa = queries[lo:hi].reshape(n, Hk, rep, QT, D).transpose(1, 0, 2, 3, 4).reshape(Hk * repv, D)
            wm = -(-repv // 16)
            lds = lambda w: 4 * max(16 * D, 16 * (D + NAX_PAD) + (wm * w * 2 * 8 * 32 if w > 1 else 0)) + 1024
            wn = NAX_WN or (2 if lds(2) <= 32768 else 1)
            smf = (lds(wn) - 1024) // 4
            nsplit = _nsplit(max(Pq - sink, 1), Hk) * NAX_NSX
            cap = c0._cap()
            L0 = Pq + c0.window + QT                                  # every row sees the pool keys [sink, Pq) at pool precision (position - window >= Pq)
            key = (L0, sink, Pq, 1, cap, 1, c0.window)
            info = _INFO.get(key)
            if info is None:
                if len(_INFO) > 8:
                    _INFO.clear()
                info = _INFO[key] = mx.array(list(key), dtype=mx.int32)
            z = mx.zeros((1, Hk, 1, D), dtype=queries.dtype)
            ins = [qa, c0.kq[0], c0.kq[1], c0.vq[0], c0.vq[1], z, z, z, z, _cb_mx(D, c0.kbits)[0], _cb_mx(D, c0.vbits)[0], info, _sgn_mx(D), _scl_mx(scale, D, False)]
            po, pml = _kernel("naxq")(inputs=ins, template=[("D", D), ("KB", c0.kbits), ("VB", c0.vbits), ("REP", repv), ("NSPLIT", nsplit), ("BK", 32), ("WM", wm),
                                                            ("WN", wn), ("LDK", D + NAX_PAD), ("SMF", smf), ("RING", 0), ("QT", QT)],
                                      grid=(32 * wm * wn * nsplit, Hk, 1), threadgroup=(32 * wm * wn, 1, 1),
                                      output_shapes=[(Hk, nsplit + 1, repv, D), (Hk, nsplit + 1, repv, 2)], output_dtypes=[mx.float32, mx.float32])
            p2 = po2s[lo] if n == 1 else mx.concatenate(po2s[lo:hi], axis=2)
            m2 = pml2s[lo] if n == 1 else mx.concatenate(pml2s[lo:hi], axis=2)
            out = _kernel("mrg")(inputs=[po, pml, p2, m2, _sgn_mx(D), _scl_mx(scale, D, True)],
                                 template=[("D", D), ("REP", repv), ("NSPLIT", nsplit + 1), ("NUSE", nsplit), ("OV", 1), ("T", queries.dtype)],
                                 grid=(256 * Hk * repv, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(Hk * repv, D)], output_dtypes=[queries.dtype])[0]
            outs.append(out.reshape(Hk, n, rep, QT, D).transpose(1, 0, 2, 3, 4).reshape(n, Hq, QT, D))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)

    # -------------------------------------------------------------- cache protocol
    def fork(self) -> "TQKVCache":
        c = TQKVCache(self.kbits, self.vbits, self.sink, self.window, self.margin, self.f32)
        cp = lambda t: None if t is None else (tuple(x[...] for x in t) if isinstance(t, tuple) else t[...])
        c.kq, c.vq, c.sk, c.sv, c.rk, c.rv = (cp(x) for x in (self.kq, self.vq, self.sk, self.sv, self.rk, self.rv))
        c.offset = self.offset
        self._own_ids = None                                    # the copy aliases these buffers: the next in-place step copies the pool first
        return c

    @property
    def state(self):
        if self.kq is None:
            return []
        self._own_ids = None                                    # the snapshot holds views of the pool: no in-place write until it is copied
        return [tiles_to_rows(self.kq[0])[:, :, :self.offset], self.kq[1][:, :, :self.offset], *(x[:, :, :self.offset] for x in self.vq), self.sk, self.sv, self.rk, self.rv]

    @state.setter
    def state(self, v):
        if not v:
            self.kq = self.vq = self.sk = self.sv = self.rk = self.rv = None
            self.offset = 0
            return
        n = v[0].shape[2]                                       # checkpoint rows (layout 1) -> runtime layout 2: pad to whole tiles, tile K
        pad = lambda x: x if x.shape[2] % 32 == 0 else mx.concatenate([x, mx.zeros((*x.shape[:2], (-n) % 32, *x.shape[3:]), dtype=x.dtype)], axis=2)
        self.kq, self.vq = (rows_to_tiles(pad(v[0])), pad(v[1])), (pad(v[2]), pad(v[3]))
        self.sk, self.sv, self.rk, self.rv = v[4:8]
        self.offset = n
        self._prev = self._chunk = None


FROM = int(os.environ.get("KYOJIN_TQ_FROM", "4096"))        # > 0: a session whose prompt + reply fits in FROM tokens keeps a plain fp16 cache (faster at short
                                                            # depth); longer ones get TurboQuant at admission (serve.backends.Q35Backend). 0 = TurboQuant for all. Default 4096: Qwen3-8B crossover 4-8K (11/11).


def from_kv(c, cfg: dict) -> "TQKVCache":
    """A plain KVCache (keys after rope) -> a TurboQuant cache holding the same rows: the pool quantizes [sink, L - window), the ring keeps the rest fp16, as if
    the rows had been prefilled into it."""
    t = TQKVCache(**cfg)
    if c.keys is not None and c.offset > 0:
        n = int(c.offset)
        t.update_and_fetch(c.keys[:, :, :n], c.values[:, :, :n])
    return t


def config():
    """Parse KYOJIN_TQ -> dict(kbits, vbits, sink, window) or None (off)."""
    v = os.environ.get(ENV, "").strip().lower()
    if v in ("", "0", "off", "false"):
        return None
    f = [int(x) for x in v.split(":")]
    if len(f) != 4:
        raise ValueError(f"{ENV}={v!r}: expected kbits:vbits:sink:window")
    return dict(kbits=f[0], vbits=f[1], sink=f[2], window=f[3])


def bytes_per_token(D: int, kbits: int, vbits: int, sink: int, window: int, ctx: int, kvh: int = 1) -> float:
    """Pool + amortized overlay bytes per token per layer for kvh heads (fp16 scale per head vector)."""
    return kvh * ((kbits + vbits) * D / 8 + 4 + 2 * D * 2 * min(sink + window, ctx) / ctx)
