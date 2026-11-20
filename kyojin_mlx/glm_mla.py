# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x MLA decode glue of the DSA layers (flag `KYOJIN_GLM_MLA_FUSE`, G10d).

After the indexer (G10c / G3d) the served DSA attention of a decode call is, per layer and per (session, row):
    embed_q   a batched fp16 GEMV  [64 heads] x (512 x 256)                       1 launch (MLX gemv, ~50 us: far below the weight stream)
    SDPA      D = 512 is outside MLX's fused sdpa: mul, matmul, select, softmax, split-K matmul + accumulate = 6-7 launches per (session, row)
    unembed   a batched fp16 GEMV  [64 heads] x (256 x 512)                       1-2 launches (one copy), ~90 us
Here, for B sessions x L rows in the whole call:
    `embed_q`            one launch, all rows (`proj`): a simdgroup per 4 output rows of one head reads its weight rows ONCE and serves every query row
    `gather`             the dense-latent gather with block padding (`gather_pad`, rows >= the selection read as zero and invalid)
    `attention`          the G3c flash-decoding kernel (`glm_dsa._ATT`: 8x8 simdgroup matrices, fp32 accumulate, online softmax, P split in a high and a low part) with a
                         (session, row) axis on the grid
    `merge + unembed`    ONE launch: the split merge (`glm_dsa._ATT_MERGE` arithmetic) of one head, rounded to the model dtype like the stock path, then that head's
                         256 x 512 projection; the output is written in the [B, L, 64, 256] layout o_proj reads
-> 4 launches per layer for any (B, L) (stock: 3 + 8 per row and session).

Row invariance: every (session, row) is reduced on its own, in an order that does not depend on how many rows / sessions share the launch. The arithmetic differs from the
stock matmul / softmax only in rounding (fp32 accumulation everywhere, the stock SDPA fallback rounds the scores and P to the model dtype)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_GLM_MLA_FUSE", "0") not in ("0", "")
MAX_U = int(os.environ.get("KYOJIN_GLM_MLA_MAX_ROWS", "16"))        # (session, row) pairs per launch
CPS = 3                                                             # 32-row chunks per attention split (glm_dsa.attend)
STATS = {"fused": 0, "stock": 0}

_k: dict = {}

_HDR = """
template <typename T, int N>
inline void ld(const device T* p, thread float* o) {
    for (int i = 0; i < N; ++i) { o[i] = float(p[i]); }
}
"""

# ------------------------------------------------------------------------------------------------------------------------------ per-head projection
# out[b, h, j, n] = sum_k W[h, n, k] * x[b, h, j, k]; one simdgroup = RPS consecutive rows n of one head (the weight rows are read once for all U = B * L query rows).
# lane l owns k in [l * KPL, (l + 1) * KPL): one 16-byte load per row for K = 256 (KPL = 8), two for K = 512.
_PROJ = r"""
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint h = threadgroup_position_in_grid.y;
    const uint n0 = (threadgroup_position_in_grid.x * 8 + sg) * RPS;
    constexpr uint KPL = K / 32;
    float wv[RPS][KPL];
    for (uint r = 0; r < RPS; ++r) {
        const device T* wp = w + ((size_t)h * N + n0 + r) * K + lane * KPL;
        for (uint i = 0; i < KPL; ++i) { wv[r][i] = float(wp[i]); }
    }
    for (uint u = 0; u < U; ++u) {
        const uint b = u / L, j = u % L;
        const device T* xp = x + (((size_t)b * 64 + h) * L + j) * K + lane * KPL;
        float xv[KPL];
        for (uint i = 0; i < KPL; ++i) { xv[i] = float(xp[i]); }
        for (uint r = 0; r < RPS; ++r) {
            float a = 0.0f;
            for (uint i = 0; i < KPL; ++i) { a = metal::fma(wv[r][i], xv[i], a); }
            a = simd_sum(a);
            if (lane == 0) { out[(((size_t)b * 64 + h) * L + j) * N + n0 + r] = static_cast<T>(a); }
        }
    }
"""


def proj_ok(x, w, transpose: bool = True) -> bool:
    """x [B, 64, L, K], w [64, N, K] (MultiLinear weight, y = x @ w^T) in fp16 / bf16."""
    if not transpose or w.ndim != 3 or x.ndim != 4 or w.shape[0] != 64 or x.shape[1] != 64:
        return False
    if x.dtype not in (mx.float16, mx.bfloat16) or w.dtype != x.dtype:
        return False
    N, K = int(w.shape[1]), int(w.shape[2])
    return K in (256, 512) and x.shape[3] == K and N % 32 == 0 and x.shape[0] * x.shape[2] <= MAX_U


def proj(x, w):
    """x [B, 64, L, K] @ w[h]^T -> [B, 64, L, N]; row-invariant (the result of a row does not depend on the other rows)."""
    B, _, L, K = (int(v) for v in x.shape)
    N = int(w.shape[1])
    rps = 4
    key = ("proj", str(x.dtype), N, K, B * L, L, rps)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_mla_proj", input_names=["x", "w"], output_names=["out"], source=_PROJ)
    ntg = N // (8 * rps)
    return k(inputs=[x, w], template=[("T", x.dtype), ("N", N), ("K", K), ("U", B * L), ("L", L), ("RPS", rps)], grid=(ntg * 256, 64, 1), threadgroup=(256, 1, 1),
             output_shapes=[(B, 64, L, N)], output_dtypes=[x.dtype])[0]


# ------------------------------------------------------------------------------------------------------------------------------ padded dense gather
# row = u * KS + k of the output, u = b * NQ + j: the selected position topk[u * K + k] (k < K) of the session b of buf [B, 1, CAP, 512]; k in [K, KS) is zero and invalid.
_GATHER_PAD = r"""
    const uint t = thread_position_in_grid.x;
    const uint row = t / 64, c = t % 64;
    const uint u = row / KS, kk = row % KS;
    const bool inb = (u < U) && (kk < K);
    const int idx = inb ? topk[u * K + kk] : -1;
    const bool ok = (idx >= 0) && (idx < ip[0]);
    if (c == 0) { vm[row] = ok; }
    device T* o = out + (size_t)row * 512 + c * 8;
    if (ok) {
        const device T* sp = lat + ((size_t)(u / NQ) * ip[1] + (size_t)idx) * 512 + c * 8;
        for (int e = 0; e < 8; ++e) { o[e] = sp[e]; }
    } else {
        for (int e = 0; e < 8; ++e) { o[e] = (T)0; }
    }
"""


def ks_of(K: int) -> int:
    return -(-K // 32) * 32


def gather_pad(buf, topk, B: int, NQ: int, K: int, nt: int):
    """buf [B, 1, CAP, 512], topk int32 [B * NQ * K] -> (rows [B * NQ * KS, 512], valid bool [same]); KS = K rounded up to 32."""
    KS = ks_of(K)
    U = B * NQ
    key = ("gpad", str(buf.dtype), K, NQ)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_mla_gather", input_names=["lat", "topk", "ip"], output_names=["out", "vm"], source=_GATHER_PAD)
    n = U * KS
    return k(inputs=[buf, topk, mx.array([nt, int(buf.shape[2])], dtype=mx.int32)], template=[("T", buf.dtype), ("K", K), ("KS", KS), ("U", U), ("NQ", NQ)],
             grid=(n * 64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(n, 512), (n,)], output_dtypes=[buf.dtype, mx.bool_])


# ------------------------------------------------------------------------------------------------------------------------------ attention + merge + unembed
def _att_src(nsp: int, nch: int):
    from . import glm_dsa as G
    s = G._ATT
    s = s.replace("const uint tg = threadgroup_position_in_grid.x;", "const uint tg = threadgroup_position_in_grid.x;\n    const uint u = threadgroup_position_in_grid.y;\n"
                  "    const uint ub = u / NQ, uj = u % NQ;", 1)
    s = s.replace("q[hg * 8 * 512 + i]", "q[(((size_t)ub * 64 + hg * 8 + i / 512) * NQ + uj) * 512 + i % 512]", 1)
    s = s.replace("const device T* kp = rows + (r0 + t * 8) * 512;", "const device T* kp = rows + ((size_t)u * KS + r0 + t * 8) * 512;", 1)
    s = s.replace("simdgroup_load(v, rows + (r0 + t * 8) * 512 + cb * 8, 512);", "simdgroup_load(v, rows + ((size_t)u * KS + r0 + t * 8) * 512 + cb * 8, 512);", 1)
    s = s.replace("if (vm[r0 + r])", "if (vm[u * KS + r0 + r])")
    s = s.replace("po[(tg * 8) * 512 + i]", "po[((size_t)(u * NTG + tg) * 8) * 512 + i]", 1)
    s = s.replace("pml[tg * 16 + lane]", "pml[(u * NTG + tg) * 16 + lane]", 1).replace("pml[tg * 16 + 8 + lane]", "pml[(u * NTG + tg) * 16 + 8 + lane]", 1)
    s = s.replace("NSP", str(nsp)).replace("CPS", str(CPS)).replace("NCH", str(nch))
    assert "u * NTG + tg" in s and s.count("u * KS") == 4 and "ub * 64" in s and s.count("(u * NTG + tg) * 16") == 2, "glm_dsa._ATT changed"
    return s


# merge of the splits of one head (glm_dsa._ATT_MERGE arithmetic) -> o (model dtype) -> out[u, h, n0 .. n0 + 128) = W[h, n, :] . o
_UNEMB = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint tg = threadgroup_position_in_grid.x, u = threadgroup_position_in_grid.y;
    const uint h = tg / 2, nh = tg % 2;
    const uint hg = h / 8, hh = h % 8;
    threadgroup T os[512];
    float M = -INFINITY;
    for (int s = 0; s < NSP; ++s) { M = metal::max(M, pml[((u * NTG) + hg * NSP + s) * 16 + hh]); }
    float Ls = 0.0f, a0 = 0.0f, a1 = 0.0f;
    for (int s = 0; s < NSP; ++s) {
        const uint t = u * NTG + hg * NSP + s;
        const float ms = pml[t * 16 + hh];
        const float e = (ms == -INFINITY) ? 0.0f : metal::exp(ms - M);
        Ls += pml[t * 16 + 8 + hh] * e;
        a0 += po[((size_t)t * 8 + hh) * 512 + tid] * e;
        a1 += po[((size_t)t * 8 + hh) * 512 + 256 + tid] * e;
    }
    os[tid] = static_cast<T>(a0 / Ls);
    os[256 + tid] = static_cast<T>(a1 / Ls);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ov[16];
    for (uint i = 0; i < 16; ++i) { ov[i] = float(os[lane * 16 + i]); }
    for (uint g = 0; g < 4; ++g) {
        float wv[4][16];
        for (uint r = 0; r < 4; ++r) {
            const uint n = nh * 128 + sg * 16 + g * 4 + r;
            const device T* wp = w + ((size_t)h * 256 + n) * 512 + lane * 16;
            for (uint i = 0; i < 16; ++i) { wv[r][i] = float(wp[i]); }
        }
        for (uint r = 0; r < 4; ++r) {
            float a = 0.0f;
            for (uint i = 0; i < 16; ++i) { a = metal::fma(wv[r][i], ov[i], a); }
            a = simd_sum(a);
            if (lane == 0) { out[((size_t)u * 64 + h) * 256 + nh * 128 + sg * 16 + g * 4 + r] = static_cast<T>(a); }
        }
    }
"""


def attend(q_lat, rows, vm, w_unembed, B: int, NQ: int, K: int, scale: float):
    """q_lat [B, 64, NQ, 512] (embed_q output), rows / vm from `gather_pad`, w_unembed [64, 256, 512] -> [B, NQ, 64 * 256] (the o_proj input layout)."""
    dt = q_lat.dtype
    KS = ks_of(K)
    nch = KS // 32
    nsp = -(-nch // CPS)
    ntg = 8 * nsp
    U = B * NQ
    key = ("att", str(dt), K, NQ, nsp)
    k = _k.get(key)
    if k is None:
        src = _att_src(nsp, nch).replace("KS", str(KS)).replace("NQ", str(NQ)).replace("NTG", str(ntg))
        k1 = mx.fast.metal_kernel(name="kyojin_glm_mla_att", input_names=["q", "rows", "vm", "scale"], output_names=["po", "pml"], header=_att_header(), source=src)
        usrc = _UNEMB.replace("NSP", str(nsp)).replace("NTG", str(ntg))
        k2 = mx.fast.metal_kernel(name="kyojin_glm_mla_unembed", input_names=["po", "pml", "w"], output_names=["out"], source=usrc)
        k = _k[key] = (k1, k2)
    k1, k2 = k
    po, pml = k1(inputs=[q_lat, rows, vm, mx.array([scale], dtype=mx.float32)], template=[("T", dt)], grid=(ntg * 32, U, 1), threadgroup=(32, 1, 1),
                 output_shapes=[(U * ntg * 8, 512), (U * ntg, 16)], output_dtypes=[mx.float32, mx.float32])
    return k2(inputs=[po, pml, w_unembed], template=[("T", dt)], grid=(128 * 256, U, 1), threadgroup=(256, 1, 1), output_shapes=[(B, NQ, 64 * 256)], output_dtypes=[dt])[0]


def _att_header():
    from . import dsa_decode as D
    return D._HEADER


def usable(self, q, topk, kv_cache, w_embed, w_unembed) -> bool:
    return bool(ENABLED and q.shape[1] == 64 and q.shape[3] == 256 and w_embed.shape == (64, 512, 256) and w_unembed.shape == (64, 256, 512)
                and w_embed.dtype == q.dtype == w_unembed.dtype and q.dtype in (mx.float16, mx.bfloat16) and q.shape[0] * q.shape[2] <= MAX_U)


def attend_batch(self, q, topk, kv_cache, start: int, last_only: bool):
    """Fused counterpart of the SDPA loop of `glm_fuse_dsa.attend_batch`: q [B, 64, Lr, 256], topk [B, Lr, K] -> (o [B, nq, 64 * 256], K rows used)."""
    B, _, Lr, _ = q.shape
    nq = Lr - start
    K = int(topk.shape[-1])
    q_lat = proj(q[:, :, start:], self.embed_q.weight)
    rows, vm = gather_pad(kv_cache.keys, topk[:, start:].reshape(-1), B, nq, K, int(kv_cache._idx))
    STATS["fused"] += 1
    return attend(q_lat, rows, vm, self.unembed_out.weight, B, nq, K, self.scale)


def attend_rows(self, q, topk, kv_cache, start: int):
    """The same for ONE session whose latent is a LatentTQCache (the rows path of `glm_fuse_dsa`): q [1, 64, Lr, 256], topk [1, Lr, K]."""
    from . import glm_dsa as G
    nq = int(q.shape[2]) - start
    K = int(topk.shape[-1])
    KS = ks_of(K)
    tk = mx.pad(topk[0, start:], [(0, 0), (0, KS - K)], constant_values=-1) if KS != K else topk[0, start:]
    rows, vm = G.gather_rows(kv_cache, tk.reshape(-1), q.dtype)
    q_lat = proj(q[:, :, start:], self.embed_q.weight)
    STATS["fused"] += 1
    return attend(q_lat, rows, vm, self.unembed_out.weight, 1, nq, K, self.scale)
