# SPDX-License-Identifier: Apache-2.0
"""Adapted in part (ideas of the pool-compress and select + expand dispatches) from oMLX (https://github.com/jundot/omlx), Apache-2.0,
omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py; the Metal sources here are our own.

GLM-5.x DSA decode, fused (flag `KYOJIN_GLM_DSA_FAST`): the indexer of one decode query and the gather + dequant of the selected latent rows.

Indexer (`Glm5NextIndexer.__call__`, batch 1, one query, plain `KVCache` valid-state, `PoolingCache`, no padding mask, more pools than `select_k`):
the stock path is ~80 launches (score matmul in fp32, where, argpartition, argsort, gathers, cumsum / argmax over the whole key-valid state for the
tail ...). Here: the stock key projection and pool cache update (they keep the caches exact), the stock query projection, one scoring launch
(`dsa_decode.scores`, fp32 like the stock path), one radix select (ascending, ties to the lower index) and ONE launch that expands the selected pools
to token positions and appends the partial tail. No work scales with the context but the scoring launch (P = ctx / 4 pooled keys, 256 B each).
With no padding every pool is a candidate, the key-valid state is all true and the tail start is `4 * P`, so the whole bookkeeping of the stock code is
known on the host. Anything else (prefill, batches, padding, speculation, few pools) takes the stock code.

Gather (`LatentTQCache.gather_fused`): the selected rows of the TurboQuant latent cache (pool rows dequantized, sink / window rows exact), the inverse
rotation done as an in-register Walsh-Hadamard transform, and the validity mask, in one launch (stock: ~15 launches and a 256 x 256 matmul)."""
from __future__ import annotations

import os

import mlx.core as mx

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_GLM_DSA_FAST", "0") == "1"
ATTN_ENABLED = os.environ.get("KYOJIN_GLM_DSA_ATTN", "0") == "1"
POOLC_ENABLED = os.environ.get("KYOJIN_GLM_DSA_POOLC", "1") == "1"
GATHER_ENABLED = os.environ.get("KYOJIN_GLM_DSA_GATHER", "1") == "1"
IDX_SCALE_FUSE = os.environ.get("KYOJIN_GLM_IDX_SCALE_FUSE", "0") == "1"      # BP-TAIL: same no-op-astype drop as glm_fuse_dsa.indexer_batch, one-row path

_k: dict = {}
_c: dict = {}

_EXPAND = """
    uint i = thread_position_in_grid.x;
    int n_sel = SELN;
    int ratio = RATIO;
    int body = n_sel * ratio;
    if (i < (uint)body) {
        out[i] = sel[i / ratio] * ratio + (int)(i % ratio);
    } else if (i < (uint)(body + ratio - 1)) {
        int j = (int)i - body;
        out[i] = j < meta[1] ? meta[0] + j : -1;
    }
"""

_SELX = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint NT = 1024;
    constexpr uint NS = 32;
    const uint nn = n[0];
    const uint Kk = uint(K);
    threadgroup atomic_uint hist[2048];
    threadgroup uint red[NS];
    threadgroup uint sh[2];
    uint kv[CPT];
    for (uint j = 0; j < CPT; ++j) { uint i = tid * CPT + j; kv[j] = i < nn ? skey(sc[i]) : 0u; }
    uint pre = 0, above = 0;
    for (uint p = 0; p < 3; ++p) {
        const uint bits = p < 2 ? 11u : 10u;
        const uint shift = p == 0 ? 21u : (p == 1 ? 10u : 0u);
        const uint NB = 1u << bits;
        const uint m = NB / NT;
        for (uint b = tid; b < NB; b += NT) { atomic_store_explicit(&hist[b], 0u, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < CPT; ++j) {
            const uint i = tid * CPT + j;
            if (i < nn) {
                const uint k = kv[j];
                if (p == 0 || (k >> (shift + bits)) == pre) { atomic_fetch_add_explicit(&hist[(k >> shift) & (NB - 1u)], 1u, memory_order_relaxed); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint c0 = atomic_load_explicit(&hist[tid * m], memory_order_relaxed);
        const uint c1 = (m == 2u) ? atomic_load_explicit(&hist[tid * m + 1u], memory_order_relaxed) : 0u;
        const uint ts = c0 + c1;
        const uint pfx = simd_prefix_exclusive_sum(ts);
        const uint ssum = simd_sum(ts);
        if (lane == 0) { red[sg] = ssum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint hi_sg = 0;
        for (uint s = sg + 1; s < NS; ++s) { hi_sg += red[s]; }
        const uint suf = hi_sg + (ssum - pfx - ts);
        const uint need = Kk - above;
        if (suf < need && need <= suf + ts) {
            uint cum = suf, b = tid * m;
            if (m == 2u) { if (cum + c1 >= need) { b = tid * m + 1u; } else { cum += c1; } }
            sh[0] = b; sh[1] = above + cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pre = (p == 0 ? 0u : (pre << bits)) | sh[0];
        above = sh[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint T0 = pre;
    const uint needTie = Kk - above;
    uint gt = 0, eq = 0;
    for (uint j = 0; j < CPT; ++j) { const uint i = tid * CPT + j; if (i < nn) { gt += (kv[j] > T0); eq += (kv[j] == T0); } }
    uint tot;
    const uint eqPref = tg_excl_scan(eq, red, sg, lane, NS, tot);
    const uint tieTake = (eqPref >= needTie) ? 0u : metal::min(eq, needTie - eqPref);
    uint total2;
    uint pos = tg_excl_scan(gt + tieTake, red, sg, lane, NS, total2);
    uint seen = 0;
    for (uint j = 0; j < CPT; ++j) {
        const uint i = tid * CPT + j;
        if (i >= nn) { break; }
        const uint k = kv[j];
        bool take = k > T0;
        if (k == T0) { take = seen < tieTake; seen++; }
        if (take) {
            for (int e = 0; e < RATIO; ++e) { out[pos * RATIO + e] = int(i) * RATIO + e; }
            ++pos;
        }
    }
    if (tid < uint(RATIO - 1)) { out[Kk * RATIO + tid] = int(tid) < meta[1] ? meta[0] + int(tid) : -1; }
"""

# one simdgroup per (selected row, 256-wide half): lane l owns dims g * 32 + l (g = 0..7), the plane layout of tq.tq_write; the inverse rotation
# R^T y = s * (H y) / sqrt(256) is a butterfly: 3 stages over g in registers, 5 over the lanes with simd_shuffle_xor
_HEADER = """
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
"""

_GATHER = """
    uint item = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    const int K = KROWS;
    int kk = (int)(item >> 1);
    int h = (int)(item & 1);
    int idx = kk < K ? topk[kk] : -1;
    int n = info[0], s = info[1], t0 = info[2], r = info[3];
    bool ok = (idx >= 0) && (idx < n);
    if (h == 0 && lane == 0) vm[kk] = ok;
    device T* o = out + (size_t)kk * 512 + h * 256;
    float v[8];
    if (!ok) {
        for (int g = 0; g < 8; g++) o[g * 32 + lane] = (T)0;
        return;
    }
    if (idx < s) {
        for (int g = 0; g < 8; g++) o[g * 32 + lane] = sk[(size_t)idx * 512 + h * 256 + g * 32 + lane];
        return;
    }
    if (idx >= t0) {
        int ri = idx - (n - r);
        for (int g = 0; g < 8; g++) o[g * 32 + lane] = rk[(size_t)ri * 512 + h * 256 + g * 32 + lane];
        return;
    }
    const device uint* wp = words + ((size_t)idx * 2 + h) * NW;
    float sc = (float)scales[(size_t)idx * 2 + h];
    for (int g = 0; g < 8; g++) v[g] = cb[tq_code<BITS>(wp + g * BITS, lane)] * sc;
    for (int m = 1; m < 8; m <<= 1) {
        for (int g = 0; g < 8; g++) {
            if (g & m) continue;
            float a = v[g], b = v[g | m];
            v[g] = a + b; v[g | m] = a - b;
        }
    }
    for (int m = 1; m < 32; m <<= 1) {
        bool up = (lane & m) != 0;
        for (int g = 0; g < 8; g++) {
            float p = simd_shuffle_xor(v[g], (ushort)m);
            v[g] = up ? p - v[g] : v[g] + p;
        }
    }
    for (int g = 0; g < 8; g++) o[g * 32 + lane] = (T)(v[g] * sg[g * 32 + lane] * 0.0625f);
"""


def _arr(key, make):
    a = _c.get(key)
    if a is None:
        a = _c[key] = make()
        mx.eval(a)
    return a


# Pool compress of a completed window (stock: 14 casts + softmax pieces + multiply + sum): one thread per column. The fused-kernel idea (one dispatch for the
# window softmax + weighted sum instead of ~8 composed ops) is from oMLX's `pool_compress`; the arithmetic here replays OUR stock `_compress_pools`
# (fp32 logits and softmax, probabilities, products and the running sum rounded to the key dtype, summed row by row).
_POOLC = r"""
    const uint d = thread_position_in_grid.x;
    float lg[KP];
    float top = -INFINITY;
    for (int j = 0; j < KP; ++j) {
        lg[j] = float(gate[j * HD + d]) + ape[j * HD + d];
        top = metal::max(top, lg[j]);
    }
    float tot = 0.0f;
    for (int j = 0; j < KP; ++j) { lg[j] = metal::precise::exp(lg[j] - top); tot += lg[j]; }
    const float inv = 1.0f / tot;
    T acc = static_cast<T>(0.0f);
    for (int j = 0; j < KP; ++j) {
        const T pr = static_cast<T>(lg[j] * inv);
        const T prod = static_cast<T>(float(pr) * float(keys[j * HD + d]));
        acc = j == 0 ? prod : static_cast<T>(float(acc) + float(prod));
    }
    out[d] = acc;
"""


def pool_compress(keys, gate, ape):
    """keys / gate [1, KP, HD] (one completed window, key dtype), ape fp32 [KP, HD] -> [1, 1, HD]."""
    kp, hd = int(keys.shape[1]), int(keys.shape[2])
    key = ("poolc", str(keys.dtype), kp, hd)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_pool_compress", input_names=["keys", "gate", "ape"], output_names=["out"], source=_POOLC)
    return k(inputs=[keys, gate, ape], template=[("T", keys.dtype), ("KP", kp), ("HD", hd)], grid=(hd, 1, 1), threadgroup=(hd, 1, 1), output_shapes=[(1, 1, hd)], output_dtypes=[keys.dtype])[0]


def expand(sel: mx.array, ratio: int, base: int, tail: int) -> mx.array:
    """sel int32 [S] (ascending pools) -> int32 [S * ratio + ratio - 1]: the tokens of the pools, then base + j for j < tail, -1 after."""
    key = ("expand", int(sel.shape[0]), ratio)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_expand", input_names=["sel", "meta"], output_names=["out"], source=_EXPAND.replace("SELN", str(sel.shape[0])).replace("RATIO", str(ratio)))
    total = sel.shape[0] * ratio + ratio - 1
    return k(inputs=[sel, mx.array([base, tail], dtype=mx.int32)], grid=(-(-total // 256) * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(total,)], output_dtypes=[mx.int32])[0]


SC_NSG = int(os.environ.get("KYOJIN_GLM_DSA_NSG", "4"))
SC_TPT = int(os.environ.get("KYOJIN_GLM_DSA_TPT", "2"))


def scores(q, w, keys, n, n_host: int):
    """`dsa_decode.scores(f32=True)` (same kernel, same values) with a finer split of the keys over threadgroups: the stock 512 keys per threadgroup leave
    4 of 80 cores busy at 8K context."""
    from . import dsa_decode as D
    dt = q.dtype
    key = ("score_t", str(dt))
    k = D._k.get(key)
    if k is None:
        k = D._k[key] = mx.fast.metal_kernel(name="kyojin_dsa_score", input_names=["q", "w", "km", "n", "cand"], output_names=["out"], source=D._SCORE, header=D._HEADER)
    nsg, tpt = SC_NSG, SC_TPT
    per = nsg * tpt * 8
    ntg = -(-n_host // per)
    return k(inputs=[q, w, keys, n, D._nomask()], template=[("T", dt), ("NSG", nsg), ("TPT", tpt), ("GT", tpt), ("HASMASK", False), ("F32", True), ("OT", mx.float32)],
             grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1), output_shapes=[(ntg * per,)], output_dtypes=[mx.float32])[0]


def select_expand(sc, n, n_host: int, topk: int, ratio: int, base: int, tail: int):
    """Radix select of the `topk` largest fp32 scores of the first n_host (ascending index, ties to the lower index: `dsa_decode.select`) and the expansion to
    token positions + partial tail (`expand`) in ONE launch. Three passes (11 / 11 / 10 bits) over keys held in registers, parallel prefix over the histogram."""
    from . import dsa_decode as D
    cpt = 1
    while cpt * 1024 < n_host:
        cpt *= 2
    if cpt > 32 or n_host <= topk:
        return expand(D.select(sc, n, n_host, topk), ratio, base, tail)
    key = ("selx", cpt, topk, ratio)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_select_expand", input_names=["sc", "n", "meta"], output_names=["out"], header=D._HEADER,
                                           source=_SELX.replace("CPT", str(cpt)).replace("RATIO", str(ratio)).replace("uint(K)", str(topk)))
    total = topk * ratio + ratio - 1
    return k(inputs=[sc, n, mx.array([base, tail], dtype=mx.int32)], grid=(1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(total,)], output_dtypes=[mx.int32])[0]


# one simdgroup per (group of 8 query heads, split of CPS chunks of 32 selected rows): S = q.K^T on 8x8 matrix tiles (fp32 accumulate), online softmax in
# fp32, P.V with P split into a bf16 high and low part (about 16 mantissa bits); the partial (O, m, l) of every split is merged by `_ATT_MERGE`.
_ATT = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint hg = tg / NSP, sp = tg % NSP;
    threadgroup T qs[8 * 512];
    threadgroup float O[8 * 512];
    threadgroup float scr[32 * 8];
    threadgroup T Ph[8 * 32];
    threadgroup T Pl[8 * 32];
    threadgroup float alp[8];
    for (uint i = lane; i < 8 * 512; i += 32) { qs[i] = q[hg * 8 * 512 + i]; O[i] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float sc = scale[0];
    for (int c = 0; c < CPS; ++c) {
        const int ch = int(sp) * CPS + c;
        if (ch >= NCH) { break; }
        const int r0 = ch * 32;
        for (int t = 0; t < 4; ++t) {
            simdgroup_matrix<float, 8, 8> acc(0.0f);
            const device T* kp = rows + (r0 + t * 8) * 512;
            for (int ks = 0; ks < 64; ++ks) {
                simdgroup_matrix<T, 8, 8> a, b;
                simdgroup_load(a, kp + ks * 8, 512);
                simdgroup_load(b, &qs[ks * 8], 512, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, &scr[t * 64], 8);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8) {
            float mxv = -INFINITY;
            for (int r = 0; r < 32; ++r) { if (vm[r0 + r]) { mxv = metal::max(mxv, scr[r * 8 + lane] * sc); } }
            const float mn = metal::max(m, mxv);
            const float alpha = (mn == -INFINITY) ? 1.0f : metal::exp(m - mn);
            float ls = 0.0f;
            for (int r = 0; r < 32; ++r) {
                float p = 0.0f;
                if (vm[r0 + r]) { p = metal::exp(scr[r * 8 + lane] * sc - mn); }
                const T hi = static_cast<T>(p);
                Ph[lane * 32 + r] = hi;
                Pl[lane * 32 + r] = static_cast<T>(p - float(hi));
                ls += p;
            }
            l = l * alpha + ls;
            m = mn;
            alp[lane] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lane; i < 8 * 512; i += 32) { O[i] *= alp[i / 512]; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int cb = 0; cb < 64; ++cb) {
            simdgroup_matrix<float, 8, 8> acc;
            simdgroup_load(acc, &O[cb * 8], 512);
            for (int t = 0; t < 4; ++t) {
                simdgroup_matrix<T, 8, 8> ph, pl, v;
                simdgroup_load(ph, &Ph[t * 8], 32);
                simdgroup_load(pl, &Pl[t * 8], 32);
                simdgroup_load(v, rows + (r0 + t * 8) * 512 + cb * 8, 512);
                simdgroup_multiply_accumulate(acc, ph, v, acc);
                simdgroup_multiply_accumulate(acc, pl, v, acc);
            }
            simdgroup_store(acc, &O[cb * 8], 512);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = lane; i < 8 * 512; i += 32) { po[(tg * 8) * 512 + i] = O[i]; }
    if (lane < 8) { pml[tg * 16 + lane] = m; pml[tg * 16 + 8 + lane] = l; }
"""

_ATT_MERGE = r"""
    const uint gid = thread_position_in_grid.x;
    const uint h = gid / 512, d = gid % 512;
    const uint hg = h / 8, hh = h % 8;
    float M = -INFINITY;
    for (int s = 0; s < NSP; ++s) { M = metal::max(M, pml[(hg * NSP + s) * 16 + hh]); }
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < NSP; ++s) {
        const uint t = hg * NSP + s;
        const float ms = pml[t * 16 + hh];
        const float e = (ms == -INFINITY) ? 0.0f : metal::exp(ms - M);
        L += pml[t * 16 + 8 + hh] * e;
        acc += po[(t * 8 + hh) * 512 + d] * e;
    }
    out[gid] = static_cast<T>(acc / L);
"""


def attend(q, rows, vm, scale: float, k_valid: int = 0):
    """Latent attention of one decode query over the gathered rows: q [1, 64, 1, 512], rows [KP, 512] (KP a multiple of 32, rows >= the selection zero), vm bool [KP]
    -> [1, 64, 1, 512] (the dtype of q). K = V = rows (MQA), softmax scale `scale`; masked rows contribute exactly nothing."""
    from . import dsa_decode as D
    dt = q.dtype
    KP = int(rows.shape[0])
    nch = KP // 32
    cps = 3
    nsp = -(-nch // cps)
    key = ("att", str(dt), KP)
    k = _k.get(key)
    if k is None:
        k1 = mx.fast.metal_kernel(name="kyojin_glm_attn_part", input_names=["q", "rows", "vm", "scale"], output_names=["po", "pml"], header=D._HEADER,
                                  source=_ATT.replace("NSP", str(nsp)).replace("CPS", str(cps)).replace("NCH", str(nch)))
        k2 = mx.fast.metal_kernel(name="kyojin_glm_attn_merge", input_names=["po", "pml"], output_names=["out"], source=_ATT_MERGE.replace("NSP", str(nsp)))
        k = _k[key] = (k1, k2)
    k1, k2 = k
    ntg = 8 * nsp
    po, pml = k1(inputs=[q, rows, vm, mx.array([scale], dtype=mx.float32)], template=[("T", dt)], grid=(ntg * 32, 1, 1), threadgroup=(32, 1, 1),
                 output_shapes=[(ntg * 8, 512), (ntg, 16)], output_dtypes=[mx.float32, mx.float32])
    return k2(inputs=[po, pml], template=[("T", dt)], grid=(64 * 512, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(1, 64, 1, 512)], output_dtypes=[dt])[0]


def gather_rows(cache, topk: mx.array, dtype, pad: int = 1):
    """cache LatentTQCache (B = H = 1, 512-wide rows, 6-bit), topk int32 [K] (-1 / >= offset invalid) -> (rows [K, 512] dtype, valid bool [K])."""
    from . import tq
    K = int(topk.shape[0])
    KP = -(-K // pad) * pad
    key = ("gather", str(dtype), K, cache.qbits)
    k = _k.get(key)
    if k is None:
        k = _k[key] = mx.fast.metal_kernel(name="kyojin_glm_gather", input_names=["words", "scales", "sk", "rk", "topk", "cb", "sg", "info"],
                                           output_names=["out", "vm"], source=_GATHER.replace("KROWS", str(K)), header=_HEADER)
    cb, _ = tq._cb_mx(256, cache.qbits)
    sg = _arr(("sg",), lambda: mx.array([1.0 if c == "1" else -1.0 for c in tq.SIGNS[:256]], dtype=mx.float32))
    n = cache.offset
    s = min(cache.sk.shape[2], n)
    r = cache.rk.shape[2]
    t0 = max(s, n - r)
    info = mx.array([n, s, t0, r], dtype=mx.int32)
    out, vm = k(inputs=[cache.words, cache.scales, cache.sk.astype(dtype), cache.rk.astype(dtype), topk, cb, sg, info],
                template=[("T", dtype), ("BITS", cache.qbits), ("NW", 8 * cache.qbits)], grid=(32 * 2 * KP, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[(KP, 512), (KP,)], output_dtypes=[dtype, mx.bool_])
    return out, vm


def gather_usable(cache, q, topk, new_latent) -> bool:
    """The fused gather serves one decode query of one row on a 6-bit LatentTQCache with 512-wide rows and both fp16 parts present."""
    return (ENABLED and GATHER_ENABLED and q.shape[0] == 1 and q.shape[2] == 1 and topk.shape[0] == 1 and topk.shape[1] == 1 and topk.dtype == mx.int32
            and cache.qbits == 6 and cache.dim == 512 and cache.words.shape[:2] == (1, 1) and cache.sk is not None and cache.rk is not None
            and new_latent.dtype in (mx.float16, mx.bfloat16))


SINK_ATTRS = ("sinks", "sink", "attn_sink", "attn_sinks", "attention_sink")


def plain_latent_attn(attn) -> bool:
    """The fused latent attention is softmax(q . rows * scale) . rows over rows of kv_lora_rank dims: no rotary tail on the keys and no sink logit. An attention module that
    declares a rope tail (qk_rope_head_dim > 0) or carries a sink would get a wrong result from it, so the shape tests below are not enough."""
    if int(getattr(attn, "qk_rope_head_dim", 0) or 0) != 0:
        return False
    return all(getattr(attn, n, None) is None for n in SINK_ATTRS)


def attend_usable(attn, q) -> bool:
    """The fused attention serves 64 query heads of 512 dims over one latent head (GLM-5.x MLA decode) in fp16 / bf16."""
    return ATTN_ENABLED and plain_latent_attn(attn) and q.shape[0] == 1 and q.shape[1] == 64 and attn.num_heads == 64 and attn.kv_lora_rank == 512 and q.dtype in (mx.float16, mx.bfloat16)


HITS: dict = {}                                             # served-path census (G3d): "fast" / "stock:<reason>" per indexer call; `hits_reset()` between phases


def _dump_loop(path: str) -> None:
    import json, threading, time
    def run():
        while True:
            time.sleep(2.0)
            try:
                from . import glm_fuse_dsa as F
                fs = dict(F.STATS)
                fs["why"] = dict(F.WHY)
            except Exception:
                fs = {}
            try:
                with open(path + ".tmp", "w") as f:
                    json.dump({"glm_dsa": dict(HITS), "fuse_dsa": fs}, f)
                os.replace(path + ".tmp", path)
            except Exception:
                pass
    threading.Thread(target=run, daemon=True).start()


if os.environ.get("KYOJIN_GLM_HITS_FILE"):                  # served-path census of a live server (G3d): a json of the counters every 2 s
    _dump_loop(os.environ["KYOJIN_GLM_HITS_FILE"])


def hits_reset() -> dict:
    out = dict(HITS)
    HITS.clear()
    return out


def keys_ok(k, pool_cache) -> bool:
    """fp16 / bf16 keys on both sides of the kernels. Known BEFORE the caches are written: afterwards the stock call can no longer run (its cache update would apply twice).
    The MTP head (layer 45) may hold fp32 indexer weights (fp32 keys, fp32 pool): that call takes the stock indexer (G3e)."""
    return k.dtype in (mx.float16, mx.bfloat16) and (pool_cache.pooled is None or pool_cache.pooled.dtype == k.dtype)


def _why(self, x, padding_mask, cache, pool_cache):
    """None when the fused indexer serves this call, else the reason it does not (the guards of `_fast_ok`, in the same order; the reason is the flag-audit key)."""
    from mlx_vlm.models.cache import KVCache, PoolingCache
    if self.training:
        return "training"
    if padding_mask is not None:
        return "padding_mask"
    if cache is None or pool_cache is None:
        return "no_cache"
    if x.shape[0] != 1 or x.shape[1] != 1:
        return f"shape B{x.shape[0]}xL{x.shape[1]}" if x.shape[0] != 1 else f"rows L{x.shape[1]}"
    if type(cache) is not KVCache:
        return "cache=" + type(cache).__name__
    if type(pool_cache) is not PoolingCache:
        return "pool_cache=" + type(pool_cache).__name__
    if pool_cache._speculation is not None:
        return "speculation"
    if pool_cache.pooled is None or pool_cache.pooled.shape[1] * self.index_kpool + pool_cache.remainder != cache.offset:
        return "pools_not_1to1"                             # the pools do not cover the history 1:1 (restored / trimmed state): stock bookkeeping
    if self.n_heads != 32 or self.head_dim != 128 or not self.always_select_tail or self.index_kpool < 2:
        return "indexer_dims"
    return None


def _fast_ok(self, x, padding_mask, cache, pool_cache) -> bool:
    return _why(self, x, padding_mask, cache, pool_cache) is None


def install() -> bool:
    if not ENABLED:
        return False
    from . import archs as _AR
    L = _AR.lang(_AR.GLM5)
    cls = L.Glm5NextIndexer
    if getattr(cls, "_kyojin_glm_dsa", False):
        return True
    stock = cls.__call__
    from . import dsa_decode as D

    def call(self, x, q_resid, padding_mask=None, cache=None, pool_cache=None, offset=0):
        why = _why(self, x, padding_mask, cache, pool_cache)
        if why is not None:
            FD.fb("dsa_fast", why)
            HITS["stock:" + why] = HITS.get("stock:" + why, 0) + 1                          # flag audit (KYOJIN_FIRED): one counter per guard reason, a global read when off
            return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
        sel_k = self.index_topk // self.index_kpool
        pc = pool_cache.pooled.shape[1]
        if pc + (1 if pool_cache.remainder + 1 == self.index_kpool else 0) <= sel_k:
            FD.fb("dsa_fast", "few_pools")
            HITS["stock:few-pools"] = HITS.get("stock:few-pools", 0) + 1
            return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
        FD.fire("dsa_fast")
        k, gate = self._project_keys(x)
        if not keys_ok(k, pool_cache):
            w = "stock:pool-dtype-" + str(k.dtype).split(".")[-1]
            HITS[w] = HITS.get(w, 0) + 1
            FD.fb("dsa_fast", "pool_dtype_" + str(k.dtype).split(".")[-1])
            return stock(self, x, q_resid, padding_mask, cache, pool_cache, offset)
        HITS["fast"] = HITS.get("fast", 0) + 1
        cache.update_and_fetch(_arr(("one",), lambda: mx.ones((1, 1, 1, 1), dtype=mx.bool_)), _arr(("zero",), lambda: mx.zeros((1, 1, 1, 0), dtype=mx.bool_)))
        ready_k, ready_gate, _ = pool_cache.accumulate_windows(k, gate.astype(k.dtype), offset)
        if ready_k.shape[1] == self.index_kpool and ready_k.dtype in (mx.float16, mx.bfloat16) and ready_gate.dtype == ready_k.dtype and POOLC_ENABLED:
            new_pool = pool_compress(ready_k, ready_gate, self.index_kpool_compress_ape)
        else:
            new_pool = self._compress_pools(ready_k, ready_gate)
        pool_keys = pool_cache.update_and_fetch(new_pool)
        P = pool_keys.shape[1]
        kv_len = cache.offset
        q, weights = self._project_queries(x, q_resid)
        if pool_keys.dtype not in (mx.float16, mx.bfloat16) or P <= sel_k:
            raise RuntimeError(f"glm_dsa: unexpected pool dtype / count after the cache update: pool {pool_keys.dtype} P {P}")
        q = q.astype(pool_keys.dtype)                       # model dtype in the engine (a no-op); the kernel wants q and the pooled keys alike
        n = mx.array([P], dtype=mx.int32)
        if IDX_SCALE_FUSE:
            w = weights[0, 0] * self.softmax_scale          # weights is already float32 (see glm_fuse_dsa's IDX_SCALE_FUSE docstring); the astype below was a no-op
        else:
            w = weights[0, 0].astype(mx.float32) * self.softmax_scale
        sc = scores(q[0, 0], w, pool_keys[0], n, P)
        r = self.index_kpool
        topk = select_expand(sc, n, P, sel_k, r, r * P, kv_len - r * P)
        return topk[None, None, :self.index_topk + r - 1]

    cls._kyojin_stock_call = stock
    cls.__call__ = call
    cls._kyojin_glm_dsa = True
    return True
