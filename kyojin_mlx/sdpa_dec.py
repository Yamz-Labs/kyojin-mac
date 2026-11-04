"""Split-K flash decode for fp16 / bf16 KV caches on the Neural Accelerator tiles (M5 / A19 and newer): T = 1..8 query rows (plain decode and MTP /
speculative verify chunks) over N keys, any GQA ratio, head dim 64 / 128 / 256. Op class: attention + KV read in decode; every model with
full-attention layers goes through it (mlx-lm's scaled_dot_product_attention).

Why: MLX's decode paths leave the K/V read under the roof at depth. One row (sdpa_vector_2pass) reaches 78-86 % of the read roof on the M5 Air; verify rows
are worse: the vector kernel re-reads K/V per query row (ALU bound, 22-58 GB/s), and the blocked kernel (the macdec89 fold) has one query block, so its
grid is Hk threadgroups (4 on Qwen3.5) with no split along the keys (88-90 GB/s of 142).

Kernel: the blocked body is MLX's attention_nax_dsplit (steel_attention_nax.h, MIT, Copyright (c) 2024-26 Apple Inc.) cut down to one query block of
WM x 16 folded rows (all R = G * T rows of a kv head: GQA heads x verify rows, row r = g * T + t; WM = ceil(R / 16), only the row groups that hold rows
exist) and given a key range per threadgroup: grid (key split, kv head, batch), so a threadgroup reads one contiguous key chunk of one kv head ONCE for
all R rows. The head dim is split over WN simdgroups (partial Q K^T exchanged through threadgroup memory, each group owns its P V slice). The causal
tail of the verify rows is computed in the kernel (key j visible to row t iff j <= N - T + t; no mask array). K / V are read through their strides, so
the cache views (slices of the preallocated buffer) are never copied. A second launch merges the split partials (unnormalised O, max, sum).

Per-chip choices (tuning.chip(): chip_defaults.json, else the Air's values): `sdpa_tpc` threadgroups per GPU core (split count = tpc * cores / (B * Hk),
chunks of at least MIN_KEYS keys; KYOJIN_SDPA_NSPLIT forces it), `sdpa_min_n1` {head dim: smallest key count} routed here for one row, `sdpa_min_n` for
verify rows (below them, or for a head dim not listed, the stock kernels win or tie). KYOJIN_SDPA_DEC=0 = stock path, bit-identical to before."""
from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

ENABLE = os.environ.get("KYOJIN_SDPA_DEC", "1") == "1"
NSPLIT = int(os.environ.get("KYOJIN_SDPA_NSPLIT", "0"))
MIN_KEYS = int(os.environ.get("KYOJIN_SDPA_MINK", "256"))
MAX_ROWS = 64                       # R = G * T rows per threadgroup (4 row groups of 16)
NAX_BK = int(os.environ.get("KYOJIN_SDPA_NAX_BK", "32"))     # keys per block (BK 64 measured slower on the Air)


# one row: head dim -> smallest key count routed here (Air: D 256 stock 2-pass 124.5 / 113 GB/s at 32K / 64K vs 131 / 132 here, tie at 16K; D 128
# Hq 32 Hk 8, serial chain (decode order, tools/sdpa91.py CHAIN=1): stock 0.180 / 1.078 / 2.271 ms per layer at 4K / 32K / 64K -> 0.161 / 1.034 / 2.055
# (+4-11 %: the stock 2-pass kernel already reads at 118-125 GB/s, split-K at 130), a tie at 1K)
SDPA_MIN_N1 = {"128": 4096, "256": 24576}
# verify rows: head dim -> smallest key count (unlisted head dims: 8192). D 128 T=5 1K / 4K best of stock / fold 0.125 / 0.436 -> 0.063 / 0.164 ms per layer;
# D 256 4K fold 0.202 -> 0.169 (serial chain, tools/sdpa91.py CHAIN=1).
SDPA_MIN_N = {"128": 1024, "256": 4096}


@lru_cache(maxsize=None)
def _chip() -> dict:
    from .tuning import chip
    c = chip()
    e = os.environ.get
    n1 = {int(d): int(n) for d, n in c.get("sdpa_min_n1", SDPA_MIN_N1).items()}
    if e("KYOJIN_SDPA_MIN_N1"):
        n1 = {d: int(e("KYOJIN_SDPA_MIN_N1")) for d in (64, 128, 256)}
    mn = c.get("sdpa_min_n", SDPA_MIN_N)
    mn = {int(d): int(n) for d, n in mn.items()} if isinstance(mn, dict) else {d: int(mn) for d in (64, 128, 256)}
    if e("KYOJIN_SDPA_MIN_N"):
        mn = {d: int(e("KYOJIN_SDPA_MIN_N")) for d in (64, 128, 256)}
    return dict(cores=c["cores"], tpc=float(e("KYOJIN_SDPA_TPC") or c.get("sdpa_tpc", 2)), min_n1=n1, min_n=mn)


# info = [N, T, G, R, C (keys per split), NS]
# merge of the NS split partials; grid (D, R, B * Hk), out [B * Hk, R, D] = [B, Hq, T, D] (row r = g * T + t)
_MRG = r"""
    const uint d = thread_position_in_grid.x, r = thread_position_in_grid.y, bh = thread_position_in_grid.z;
    const int R = info[3], NS = info[5];
    float M = -1e30f;
    for (int s = 0; s < NS; s++) M = max(M, pml[((bh * NS + s) * RP + r) * 2]);
    float acc = 0.0f, L = 0.0f;
    for (int s = 0; s < NS; s++) {
        int row = (bh * NS + s) * RP + r;
        float w = fast::exp(pml[row * 2] - M);
        acc += w * po[row * D + d];
        L += w * pml[row * 2 + 1];
    }
    out[(bh * R + r) * D + d] = (T_)(acc / L);
"""



_kernels: dict = {}


def nsplit(n: int, groups: int, tpc: float | None = None, up: bool = False) -> int:
    """Key splits per kv head: about `tpc` threadgroups per GPU core over `groups` = B * Hk, at least MIN_KEYS keys each. up (two row groups, R > 16:
    verify rows of 4+ GQA heads): at least `tpc` per core, rounded up (Air, serial chain, D 128 Hk 8 T=5 2 -> 3 splits: 4K / 64K 0.193 / 2.759 ->
    0.164 / 2.215 ms per layer; one row group ties: 64K T=1 2.032 / 2.009)."""
    if NSPLIT > 0:
        return max(1, min(NSPLIT, n // NAX_BK))
    c = _chip()
    x = (tpc or c["tpc"]) * c["cores"] / groups
    want = -int(-x // 1) if up else round(x)
    return max(1, min(want, n // MIN_KEYS))


@lru_cache(maxsize=None)
def nax_ok() -> bool:
    try:
        from .metal_nax import nax_available
        return bool(nax_available())
    except Exception:
        return False


def supported(q, k, v) -> bool:
    B, Hq, T, D = q.shape
    Hk, N = k.shape[1], k.shape[2]
    if not (1 <= T <= 8 and N >= max(T, NAX_BK) and N >= (_chip()["min_n1"].get(D, 1 << 62) if T == 1 else _chip()["min_n"].get(D, 8192))):
        return False
    return (q.dtype in (mx.float16, mx.bfloat16) and k.dtype == q.dtype and v.dtype == q.dtype and D in (64, 128, 256) and v.shape[-1] == D
            and Hq % Hk == 0 and (Hq // Hk) * T <= MAX_ROWS and v.shape[2] == N and k.shape[0] == B and v.shape[0] == B and nax_ok())


# NAX tile ops of the kernel body (mlx steel's are not reachable from the metal_kernel JIT header)
_NAX_SRC = r"""
    using namespace mlx::steel;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const uint split = threadgroup_position_in_grid.x;
    const uint hk = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const int Hk = (int)threadgroups_per_grid.y;
    const int N = info[0], T = info[1], G = info[2], R = info[3], C = info[4], NS = info[5];
    constexpr short kU = 16;
    constexpr int TD = D / kU, TK = BK / kU, TDh = TD / WN, BDh = D / WN, RP = WM * TQ * kU;
    const float scale2 = scale[0] * 1.44269504089f;
    const short row_group = sgi / WN, d_group = sgi % WN;
    const short tm = TQ * kU * row_group;
    // q is [B, Hq, T, D] row contiguous: the G * T rows of kv head hk are consecutive, stride D
    const device T_* Q = q + (ulong(b) * Hk + hk) * ulong(R) * D + tm * D + d_group * BDh;
    const device T_* K = k + b * k_strides[0] + hk * k_strides[1] + d_group * BDh;
    const device T_* V = v + b * v_strides[0] + hk * v_strides[1] + d_group * BDh;
    const int ks = k_strides[2], vs = v_strides[2];
    const int k0 = split * C, k1 = (int(split) == NS - 1) ? N : min(k0 + C, N);
    const short lim_rows_q = R - tm;

    using otile_t = NAXTile<float, TQ, TDh>;
    otile_t Otile;
    Otile.clear();
    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<float, kRowsPT> max_score, sum_score{0};
    STEEL_PRAGMA_UNROLL for (short i = 0; i < kRowsPT; ++i) max_score[i] = -1e30f;

    using stile_t = NAXTile<float, TQ, TK>;
    constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;
    threadgroup float s_xchg[WM][WN][TQ * TK * kEPF * 32];
    NAXTile<T_, TQ, 1> Qtiles[TDh];
    STEEL_PRAGMA_UNROLL for (short id = 0; id < TDh; id++) {
        if (lim_rows_q < TQ * kU) Qtiles[id].load_rows(Q + id * kU, D, lim_rows_q);
        else Qtiles[id].load(Q + id * kU, D);
    }
    const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
    const short sm = simd_coord.y, sn = simd_coord.x;
    const int vis0 = N - T;                     // row r sees keys c <= vis0 + r % T

    for (int kb = k0; kb < k1; kb += BK) {
        const short lim_k = (short)min(BK, k1 - kb);
        const device T_* Kb = K + ulong(kb) * ks;
        const device T_* Vb = V + ulong(kb) * vs;
        stile_t Stile;
        Stile.clear();
        STEEL_PRAGMA_UNROLL for (short ik = 0; ik < TK; ik += 2) {
            STEEL_PRAGMA_UNROLL for (short id = 0; id < TDh; id++) {
                NAXTile<T_, 2, 1> Ktile;
                const int off = ik * kU * ks + id * kU;
                if (lim_k < BK) Ktile.load_rows(Kb + off, ks, lim_k - ik * kU);
                else Ktile.load(Kb + off, ks);
                STEEL_PRAGMA_UNROLL for (short iq = 0; iq < TQ; iq++)
                    stile_t::NAXFrag_t::mma(Stile.frag_at(iq, ik), Stile.frag_at(iq, ik + 1), Qtiles[id].frag_at(iq, 0), metal::false_type{},
                                            Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
            }
        }
        if (WN > 1) {
            threadgroup float* slot = s_xchg[row_group][d_group];
            constexpr short NE = TQ * TK * kEPF;
            const short base = short(lane) * NE;
            STEEL_PRAGMA_UNROLL for (short ii = 0; ii < NE; ii++) slot[base + ii] = Stile.elems()[ii];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL for (short ii = 0; ii < NE; ii++) {
                float s = s_xchg[row_group][0][base + ii];
                STEEL_PRAGMA_UNROLL for (short pg = 1; pg < WN; pg++) s += s_xchg[row_group][pg][base + ii];
                Stile.elems()[ii] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        STEEL_PRAGMA_UNROLL for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) Stile.elems()[ii] *= scale2;
        if (kb + BK > vis0) {                   // key tail of the range and causal tail of the verify rows
            STEEL_PRAGMA_UNROLL for (short iq = 0; iq < TQ; iq++) STEEL_PRAGMA_UNROLL for (short ik = 0; ik < TK; ik++) {
                thread auto& fg = Stile.frag_at(iq, ik);
                STEEL_PRAGMA_UNROLL for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
                    const int r = tm + iq * kU + ii * stile_t::kFragRowsJump + sm;
                    const int lim = min(k1 - 1, vis0 + r % T);
                    STEEL_PRAGMA_UNROLL for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                        const int c = kb + ik * kU + jj + sn;
                        const auto loc = ii * stile_t::kFragThrCols + jj;
                        fg[loc] = (c > lim) ? -1e30f : fg[loc];
                    }
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
        simdgroup_barrier(mem_flags::mem_none);
        STEEL_PRAGMA_UNROLL for (short id = 0; id < TDh; id += 2) {
            STEEL_PRAGMA_UNROLL for (short ik = 0; ik < TK; ik++) {
                NAXTile<T_, 1, 2> Vtile;
                const int off = ik * kU * vs + id * kU;
                if (lim_k < BK) Vtile.load_rows(Vb + off, vs, lim_k - ik * kU);
                else Vtile.load(Vb + off, vs);
                STEEL_PRAGMA_UNROLL for (short iq = 0; iq < TQ; iq++)
                    otile_t::NAXFrag_t::mma(Otile.frag_at(iq, id), Otile.frag_at(iq, id + 1), Stile.frag_at(iq, ik), metal::false_type{},
                                            Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
            }
        }
    }
    const ulong prow = (ulong(b * Hk + hk) * NS + split) * RP;
    Otile.store(po + (prow + tm) * D + d_group * BDh, D);
    if (d_group == 0 && sn == 0) {
        STEEL_PRAGMA_UNROLL for (short i = 0; i < kRowsPT; ++i) {
            const int r = tm + (i / stile_t::kFragThrRows) * kU + (i % stile_t::kFragThrRows) * stile_t::kFragRowsJump + sm;
            pml[(prow + r) * 2] = max_score[i] * M_LN2_F;
            pml[(prow + r) * 2 + 1] = sum_score[i];
        }
    }
"""

_NAX_OPS = r"""
struct KMaxOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return metal::max(x, y); } };
struct KSumOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x + y; } };
struct KMulOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x * y; } };
struct KExpSubOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return fast::exp2(x - y); } };
"""


@lru_cache(maxsize=None)
def _nax_header() -> str:
    """mlx's steel/attn/nax.h with its mlx includes inlined (the metal_kernel JIT has no include path to the installed headers)."""
    import re
    root = os.path.join(os.path.dirname(mx.__file__), "include")
    seen: set = set()

    def inline(rel: str) -> str:
        if rel in seen:
            return ""
        seen.add(rel)
        out = []
        for line in open(os.path.join(root, rel)).read().splitlines():
            m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
            if m:
                out.append(inline(m.group(1)))
            elif not line.strip().startswith("#pragma once"):
                out.append(line)
        return "\n".join(out)
    return inline("mlx/backend/metal/kernels/steel/attn/nax.h") + "\n" + _NAX_OPS


# array-mask variant (bool mask broadcast to [B, Hq, T, N] through its strides: GLM / DSA top-k rows with -1 padding, any mask of the model): the mask replaces
# the causal tail (mlx sdpa semantics) and is applied on every key block; row r = g * T + t is query head hk * G + g, verify row t
_NAX_MASK_SRC = _NAX_SRC.replace(
    "        if (kb + BK > vis0) {                   // key tail of the range and causal tail of the verify rows",
    "        {                                       // key tail of the range and the array mask").replace(
    "                    const int lim = min(k1 - 1, vis0 + r % T);",
    "                    const int lim = k1 - 1;\n"
    "                    const device bool* mrow = mask + b * mask_strides[0] + (hk * G + min(r, R - 1) / T) * mask_strides[1] + (min(r, R - 1) % T) * mask_strides[2];").replace(
    "                        fg[loc] = (c > lim) ? -1e30f : fg[loc];",
    "                        fg[loc] = (c > lim || !mrow[min(c, lim) * mask_strides[3]]) ? -1e30f : fg[loc];")
assert _NAX_MASK_SRC.count("mrow") == 2


def _nax_kernel(masked: bool = False):
    key = "naxm" if masked else "nax"
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel("kyojin_sdpa_nax_mask" if masked else "kyojin_sdpa_nax", ["q", "k", "v", "info", "scale"] + (["mask"] if masked else []),
                                             ["po", "pml"], _NAX_MASK_SRC if masked else _NAX_SRC, header=_nax_header(), ensure_row_contiguous=False)
    return _kernels[key]


def _mrg_kernel():
    if "mrg" not in _kernels:
        _kernels["mrg"] = mx.fast.metal_kernel("kyojin_sdpa_mrg", ["po", "pml", "info"], ["out"], _MRG)
    return _kernels["mrg"]


def sdpa_decode(q, k, v, scale: float, ns: int | None = None, bk: int | None = None, mask=None):
    """Causal-tail attention of T query rows (the last T of the N keys) for q [B, Hq, T, D], k / v [B, Hk, N, D] (any strides along B / H / N); with a
    bool `mask` (broadcastable to [B, Hq, T, N]) the mask replaces the causal tail."""
    B, Hq, T, D = q.shape
    Hk, N = k.shape[1], k.shape[2]
    G = Hq // Hk
    R = G * T
    TQ = int(os.environ.get("KYOJIN_SDPA_NAX_TQ", "1"))
    WM = -(-R // (16 * TQ))
    bk = bk or NAX_BK
    # head-dim split: D / 32 (one row group) or D / 64 simdgroups, capped by the score exchange (WM * WN * TQ * BK / 16 KiB <= 32 KiB of threadgroup memory)
    WN = int(os.environ.get("KYOJIN_SDPA_NAX_WN", "0")) or max(2, min(D // (32 if R <= 16 else 64), 32 // (WM * TQ * (bk // 16))))
    ns = ns or nsplit(N, B * Hk, up=WM > 1)
    C = max(bk, (N // ns) // bk * bk)        # last split takes the remainder (>= C keys: every verify row sees one key at least)
    ns = max(1, N // C)
    q = mx.contiguous(q)
    info = mx.array([N, T, G, R, C, ns], dtype=mx.int32)
    sc = mx.array([scale], dtype=mx.float32)
    RP = WM * TQ * 16
    extra = [] if mask is None else [mx.broadcast_to(mask, (B, Hq, T, N))]
    po, pml = _nax_kernel(mask is not None)(inputs=[q, k, v, info, sc] + extra, template=[("D", D), ("BK", bk), ("TQ", TQ), ("WM", WM), ("WN", WN), ("T_", q.dtype)],
                            grid=(ns * WM * WN * 32, Hk, B), threadgroup=(WM * WN * 32, 1, 1),
                            output_shapes=[(B * Hk * ns * RP, D), (B * Hk * ns * RP, 2)], output_dtypes=[mx.float32, mx.float32])
    out = _mrg_kernel()(inputs=[po, pml, info], template=[("D", D), ("RP", RP), ("T_", q.dtype)],
                        grid=(D, R, B * Hk), threadgroup=(min(D, 256), 1, 1), output_shapes=[(B * Hk * R, D)], output_dtypes=[q.dtype])[0]
    return out.reshape(B, Hq, T, D)


sdpa_decode_nax = sdpa_decode


def _mask_ok(mask, q, k) -> bool:
    """A bool array mask broadcastable to [B, Hq, T, N] (padding / window / top-k validity masks of batched or sparse decode); additive masks -> stock."""
    if not isinstance(mask, mx.array) or mask.dtype != mx.bool_ or not 1 <= mask.ndim <= 4:
        return False
    want = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    return all(m in (1, w) for m, w in zip(mask.shape[::-1], want[::-1])) and mask.shape[-1] == want[-1]


_TR = {}
TRACE = os.environ.get("KYOJIN_SDPA_TRACE") == "1"         # stderr: first / 100th / 1000th call of each (T, D, routed, mask, cache) kind


def route(queries, keys, values, cache, scale, mask, sinks=None):
    out = _route(queries, keys, values, cache, scale, mask, sinks)
    if TRACE:
        m = None if mask is None else (mask if isinstance(mask, str) else (str(mask.dtype), tuple(mask.shape)))
        key = (queries.shape[2], queries.shape[3], out is not None, str(m)[:60], type(cache).__name__, sinks is not None)
        _TR[key] = _TR.get(key, 0) + 1
        if _TR[key] in (1, 100, 1000):
            import sys
            print("SDPATRACE", key, tuple(keys.shape), _TR[key], file=sys.stderr, flush=True)
    return out


def _route(queries, keys, values, cache, scale, mask, sinks=None):
    """The split-K decode attention when the call is a decode / verify step it covers (fp16 / bf16 KV, no quantized or TurboQuant cache, no sinks;
    no mask, the causal tail, or a bool array mask), else None."""
    if (not ENABLE or sinks is not None or hasattr(cache, "bits") or "TurboQuant" in type(cache).__name__
            or not (mask is None or (isinstance(mask, str) and mask == "causal") or _mask_ok(mask, queries, keys))
            or (mask is None and queries.shape[2] != 1) or not supported(queries, keys, values)):
        return None
    return sdpa_decode(queries, keys, values, scale, mask=mask if isinstance(mask, mx.array) else None)


_orig = None


def install() -> int:
    """Route mlx-lm's scaled_dot_product_attention (base and every loaded model module bound to it) through `route` (idempotent). Returns the
    number of modules patched."""
    global _orig
    import sys
    from mlx_lm.models import base
    if _orig is None:
        _orig = base.scaled_dot_product_attention

        def scaled_dot_product_attention(queries, keys, values, cache, scale, mask, sinks=None):
            out = route(queries, keys, values, cache, scale, mask, sinks)
            return out if out is not None else _orig(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

        scaled_dot_product_attention._kyojin_sdpa_dec = True
        base.scaled_dot_product_attention = scaled_dot_product_attention
    new = base.scaled_dot_product_attention
    n = 0
    for name, m in list(sys.modules.items()):
        if name.startswith("mlx_lm.models.") and getattr(m, "scaled_dot_product_attention", None) is _orig:
            m.scaled_dot_product_attention = new
            n += 1
    return n
