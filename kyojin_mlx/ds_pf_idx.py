# SPDX-License-Identifier: Apache-2.0
"""DS4.1 indexer for prefill (S > 1, batch 1): the query rotary + FP4 round trip as one kernel, the head-weighted score reduction as one kernel.

Stock chain (per layer owning an indexer, prefill): `_apply_index_rotary` (complex64 view, several passes) then `fake_quant_fp4_ue8m0` (about 20 elementwise passes
over [T, 32, 128]) then `_index_scores` (a [T * 32, Tc] matmul written to memory, relu, weight, head sum: three more passes over T x 32 x Tc values).

`fq4_rot`: one simdgroup per (t, head) row, 4 values per lane, bit-identical to the stock chain (the rotary result is rounded to half before the FP4 step, as the
stock chain does). `scores_mm`: MLX matmul per block of queries, then ONE kernel for relu * head weight, head sum and the causal mask.

`fscore` (E3, KYOJIN_PF_IDX_FSCORE): the same scores with the product never leaving the threadgroup: one query row per threadgroup, a 32-head x 32-key matmul2d
tile per simdgroup, the product rounded to half (as the materialised half product is), relu * head weight and the head sum in the `_POST` order, causal mask in the
same kernel. It replaces the matmul + post pair of `scores_range` and drops the [T * 32, Tc] half product (the largest buffer of the old chain).

Switch: module attribute ENABLED (env KYOJIN_DS_PF_IDX=0)."""
from __future__ import annotations

from . import archs as _A
import os

import mlx.core as mx
from .modeldesc import KERNEL as KG

from . import fired as FD
from . import idx_topk as IT
from . import longctx as LC
from . import metal_nax as NX
from . import cpackscan as CP
from . import pf_bound as PB
from . import pf_cluster as PC
from . import pf_flat as PF
from . import pfroof2 as PR

METAL_OPT = (mx.metal.is_available() and mx.default_device() == mx.gpu)

ENABLED = os.environ.get("KYOJIN_DS_PF_IDX", "1") != "0"
KTILE_ENV = "KYOJIN_PF_IDX_KTILE"       # E2: key-tiled score loop (read per call; default 1 = flags.py)
CAND_ENV = "KYOJIN_PF_IDX_CAND"         # E1: candidate-consumer layers score only the kept key blocks (read per call; default 1 = flags.py)
FSCORE = os.environ.get("KYOJIN_PF_IDX_FSCORE", "1") not in ("0", "")     # E3: fused score kernel (read per call)
TOPK_MT = os.environ.get("KYOJIN_PF_IDX_TOPK_MT", "1") not in ("0", "")   # E4: exact radix top-k of the block scores (read per call)
KEY_TILE = 4096                         # the vendor key tile width (language.py INDEX_SCORE_TILE)
FSCORE_TPT = 8                          # 128-key tiles a fused-score threadgroup walks (one query row per threadgroup)

_FQ4_ROT = """
    // q [R, 128] half rows (R = T * n_heads), cos / sin [T, rd/2] float (row t = position), rd = rope dim (64). Row r belongs to position t = r / NH.
    const uint lane = thread_index_in_simdgroup;
    const uint r = thread_position_in_grid.y;
    const uint NH = prm[0], RD = prm[1];
    const uint t = r / NH;
    const uint e0 = lane * 4;                                  // 4 consecutive elements of the 128
    float v[4];
    for (uint j = 0; j < 4; ++j) v[j] = float(q[ulong(r) * 128 + e0 + j]);
    if (e0 >= 128 - RD) {                                      // rotary pairs (2i, 2i+1) inside the last RD entries
        const uint p0 = (e0 - (128 - RD)) / 2;
        for (uint j = 0; j < 2; ++j) {
            const float c = cs[ulong(t) * (RD / 2) + p0 + j], s = sn[ulong(t) * (RD / 2) + p0 + j];
            const float a = v[2 * j], b = v[2 * j + 1];
            v[2 * j] = float(half(a * c - b * s));             // rounded to half like the stock chain (astype(dtype) after the rotary)
            v[2 * j + 1] = float(half(a * s + b * c));
        }
    }
    // FP4 round trip, blocks of 32 = 8 lanes
    float amax = max(max(abs(v[0]), abs(v[1])), max(abs(v[2]), abs(v[3])));
    amax = max(amax, simd_shuffle_xor(amax, 1));
    amax = max(amax, simd_shuffle_xor(amax, 2));
    amax = max(amax, simd_shuffle_xor(amax, 4));
    amax = max(amax, 6.0f * exp2(-126.0f));
    const uint bits = as_type<uint>(amax * (1.0f / 6.0f));
    const uint ex = (bits >> 23) + ((bits & 0x7fffff) != 0 ? 1u : 0u);
    const float scale = as_type<float>(ex << 23);
    for (uint j = 0; j < 4; ++j) {
        const float x = clamp(v[j] / scale, -6.0f, 6.0f);
        const float m = abs(x);
        uint idx = (m > 0.25f) + (m > 1.25f) + (m > 2.5f) + (m > 5.0f) + (m >= 0.75f) + (m >= 1.75f) + (m >= 3.5f);
        const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        const float sg = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        out[ulong(r) * 128 + e0 + j] = half((sg * lut[idx]) * scale);
    }
"""
_kern: dict = {}


def fq4_rot(q, cos, sin, rope_dim: int):
    """q [1, T, NH, 128] half; cos / sin [T, rope_dim // 2] float32 -> same shape, rotary on the last `rope_dim` entries then FP4 (ue8m0, blocks of 32) round trip."""
    _, T, NH, D = q.shape
    assert D == 128 and q.dtype == mx.float16
    k = _kern.get("q")
    if k is None:
        k = _kern["q"] = mx.fast.metal_kernel(name="kyojin_dsidx_fq4rot", input_names=["q", "cs", "sn", "prm"], output_names=["out"], source=_FQ4_ROT)
    prm = mx.array([NH, rope_dim] + [0] * 14, dtype=mx.uint32)
    return k(inputs=[q.reshape(-1, 128), cos.astype(mx.float32), sin.astype(mx.float32), prm], grid=(32, T * NH, 1), threadgroup=(32, 1, 1),
             output_shapes=[(T * NH, 128)], output_dtypes=[mx.float16])[0].reshape(q.shape)


_POST = """
    // M [BT, 32, Tc] half (q . k), w [BT, 32] half -> S [BT, Tc] float32 = sum_h w * relu(M); causal: n < (start + t0 + t + 1) / ratio else -inf
    const uint n = thread_position_in_grid.x, t = thread_position_in_grid.y;
    const uint Tc = prm[0], t0 = prm[1], start = prm[2], ratio = prm[3], n0 = prm[4];   // n0: first key column of this slice of the past (0 = the whole past)
    if (n >= Tc) return;
    const uint clen = (start + t0 + t + 1) / ratio;
    float acc = 0.0f;
    const device half* mp = M + ulong(t) * 32 * Tc + n;
    for (uint h = 0; h < 32; ++h) acc += max(float(mp[ulong(h) * Tc]), 0.0f) * float(w[ulong(t) * 32 + h]);
    S[ulong(t) * Tc + n] = n + n0 < clen ? acc : -INFINITY;
"""


QMM_ENV = "KYOJIN_PF_IDX_QMM"                 # index score GEMM on the MXFP8 path (read per call; default 0)
QMM_MIN = 4096                                # keys from which the path is used (below: the fused fp16 kernel)


def qmm_on() -> bool:
    return os.environ.get(QMM_ENV, "0") not in ("0", "")


def qmm_pack(keys):
    """`keys` [1, Tc, 128] half -> (packed uint32 [Tc, D / 4], scales uint8 [Tc, D / 32]) for `mx.quantized_matmul(mode="mxfp8")`.
    Lossless for index keys that are already fp4 / ue8m0 quantized: every value is a code (0, 0.5, 1, 1.5, 2, 3, 4, 6) times a power of two,
    all of them exact e4m3 numbers on the same 32-value block, and e8m0 scales are powers of two, so the round trip returns the same values."""
    return mx.quantize(keys[0], group_size=32, bits=8, mode="mxfp8")


def qmm_ok(q, keys, kq) -> bool:
    """The quantized score path needs the DS4.1 indexer shapes, half operands and keys long enough to pay for the pack."""
    return bool(kq is not None and METAL_OPT and q.shape[3] == 128 and keys.shape[1] >= QMM_MIN)


def scores_qmm(q, keys, w, start: int, ratio: int, t0: int, n: int, n0: int, kan):
    """`scores_range` with the q . k product on the MXFP8 path: same values (the index keys are e4m3-exact, the product is rounded to half,
    relu * half head weight, heads summed in the order 0..31, same causal mask), the GEMM reads a 4-bit-per-value-shorter key row and runs the
    vendor's quantized matmul instead of the per-row fused fp16 tile kernel."""
    NH, D = q.shape[2], q.shape[3]
    Tc = keys.shape[1]
    kp, ks = kan
    k = _kern.get("p")
    if k is None:
        k = _kern["p"] = mx.fast.metal_kernel(name="kyojin_dsidx_post", input_names=["M", "w", "prm"], output_names=["S"], source=_POST)
    kw = KEY_TILE if Tc > KEY_TILE else Tc
    bt = max(min(n, 4), min(n, max_elems_bt(NH, kw)))
    outs = []
    for s0 in range(t0, t0 + n, bt):
        rows_n = min(bt, t0 + n - s0)
        qb = q[0, s0:s0 + rows_n].reshape(-1, D)
        tiles = []
        for a in range(0, (Tc // kw) * kw, kw):
            wd = Tc - a if a + 2 * kw > Tc else kw
            m = mx.quantized_matmul(qb, kp[a:a + wd], ks[a:a + wd], None, transpose=True, group_size=32, bits=8, mode="mxfp8")
            prm = mx.array([wd, s0, start, ratio, n0 + a] + [0] * 11, dtype=mx.uint32)
            tiles.append(k(inputs=[m, w[0, s0:s0 + rows_n], prm], grid=(wd, rows_n, 1), threadgroup=(min(wd, 256), 1, 1),
                           output_shapes=[(rows_n, wd)], output_dtypes=[mx.float32])[0])
        outs.append(tiles[0] if len(tiles) == 1 else mx.concatenate(tiles, axis=1))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


def max_elems_bt(nh: int, wd: int, max_elems: int = 1 << 26) -> int:
    """Rows of one quantized score tile: keep the half product [rows * nh, wd] under `max_elems` elements."""
    return max(4, max_elems // (nh * wd))


def scores_range(q, keys, w, start: int, ratio: int, t0: int, n: int, max_elems: int = 1 << 26, n0: int = 0, kq=None):
    """Rows [t0, t0 + n) of the score matrix: [n, Tc] float32, the values `scores_mm` gives for those rows. Split again over queries by `max_elems` (the matmul
    product [n * 32, Tc] half is the largest single buffer here); every row is computed on its own, so where the splits fall changes no value.
    n0: keys are the past columns n0 .. n0 + Tc - 1 (a slice; the causal mask uses the absolute column). kq: the packed MXFP8 keys (see `qmm_pack`)."""
    if qmm_ok(q, keys, kq):
        return scores_qmm(q, keys, w, start, ratio, t0, n, n0, kq)
    if fscore_ok(q, keys, w):
        FD.hit("pf_idx_fscore")
        if os.environ.get(KTILE_ENV, "1") not in ("0", ""):
            FD.hit("pf_idx_ktile_shadowed")
        return fscore(q, keys, w, start, ratio, t0, n, n0)
    NH, D = q.shape[2], q.shape[3]
    Tc = keys.shape[1]
    k = _kern.get("p")
    if k is None:
        k = _kern["p"] = mx.fast.metal_kernel(name="kyojin_dsidx_post", input_names=["M", "w", "prm"], output_names=["S"], source=_POST)
    kt = keys[0].T
    bt = max(1, min(n, max_elems // (NH * Tc)))
    kw = KEY_TILE if (bt < n and Tc > KEY_TILE and os.environ.get(KTILE_ENV, "1") not in ("0", "")) else Tc     # E2: tile the keys so the row micro-block stays large
    if kw != Tc:
        FD.hit("pf_idx_ktile")
        bt = max(min(n, 4), min(n, max_elems // (NH * kw)))                              # >= 4 rows: the matmul bits of 1-3 row blocks differ from the larger ones
    outs = []
    for s0 in range(t0, t0 + n, bt):
        rows_n = min(bt, t0 + n - s0)
        qb = q[0, s0:s0 + rows_n].reshape(-1, D)
        tiles = []
        for a in range(0, (Tc // kw) * kw, kw):                                             # the last tile takes the remainder (a 1-column tile would run another matmul kernel)
            wd = Tc - a if a + 2 * kw > Tc else kw
            m = qb @ (kt if kw == Tc else kt[:, a:a + wd])                                  # [rows * 32, wd] half
            rows = m.shape[0] // NH
            prm = mx.array([wd, s0, start, ratio, n0 + a] + [0] * 11, dtype=mx.uint32)
            tiles.append(k(inputs=[m, w[0, s0:s0 + rows], prm], grid=(wd, rows, 1), threadgroup=(min(wd, 256), 1, 1),
                           output_shapes=[(rows, wd)], output_dtypes=[mx.float32])[0])
        outs.append(tiles[0] if len(tiles) == 1 else mx.concatenate(tiles, axis=1))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


_FSCORE = """
    // E3: one query row per threadgroup (grid.y: 128-key tiles), 32 heads x 32 keys per simdgroup tile. The product stays in the threadgroup; keys are read
    // straight from device memory. Values are the ones the matmul + _POST pair give: the product rounded to half, relu, times the half head weight, in fp32,
    // heads summed in the order 0..31; the causal mask is applied here.
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.x;
    const uint Tc = prm[0], start = prm[1], ratio = prm[2], t0 = prm[3], n0 = prm[4], TPT = prm[5];
    threadgroup half Qs[32 * 128];
    threadgroup float Wh[32];
    threadgroup float scr[4][32 * 33];                                  // one [32 keys, 32 heads] tile per simdgroup, 33 floats per key (no bank conflicts)
    for (uint i = tid; i < 32u * 128u; i += 128u) Qs[i] = Q[ulong(t) * 32 * 128 + i];
    if (tid < 32u) Wh[tid] = float(w[ulong(t) * 32 + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint clen = (start + t0 + t + 1) / ratio;
    constexpr auto desc = matmul2d_descriptor(32, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tQ((threadgroup half*)Qs, dextents<int32_t, 2>(128, 32));
    for (uint j = 0; j < TPT; ++j) {
        const uint k0 = (threadgroup_position_in_grid.y * TPT + j) * 128 + sg * 32;
        if (k0 >= Tc) break;
        const uint ks = k0 + 32u > Tc ? Tc - 32u : k0;              // a tail tile reads the last 32 keys (a few recomputed, never out of bounds)
        const uint rem = min(32u, Tc - k0);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tK((device half*)(K + ulong(ks) * 128), dextents<int32_t, 2>(128, 32));
        auto cT = op.get_destination_cooperative_tensor<decltype(tK), decltype(tQ), float>();
        op.run(tK, tQ, cT);
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                const auto ix = cT.get_multidimensional_index(i);
                scr[sg][uint(ix[1]) * 33 + uint(ix[0])] = float(half(cT[i]));
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < rem) {
            float s = 0.0f;
            const uint lk = k0 + lane - ks;
            for (uint h = 0; h < 32; ++h) s += metal::max(scr[sg][lk * 33 + h], 0.0f) * Wh[h];
            const uint n = k0 + lane;
            S[ulong(t) * Tc + n] = n + n0 < clen ? s : -INFINITY;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def fscore_ok(q, keys, w) -> bool:
    """The fused score kernel needs the DS4.1 indexer shapes, half operands and a GPU with matmul2d (the NAX probe)."""
    return bool(FSCORE and METAL_OPT and keys.ndim == 3 and q.dtype == mx.float16 and w.dtype == mx.float16 and keys.dtype == mx.float16
                and q.shape[2] == KG.IDX_HEADS and q.shape[3] == KG.IDX_DIM and keys.shape[1] >= 128 and NX.nax_available())


def fscore(q, keys, w, start: int, ratio: int, t0: int, n: int, n0: int = 0):
    """[n, Tc] float32: `scores_range` for rows [t0, t0 + n) in one launch, without the [n * 32, Tc] half product (same values, see _FSCORE)."""
    Tc = keys.shape[1]
    k = _kern.get("f")
    if k is None:
        k = _kern["f"] = mx.fast.metal_kernel(name="kyojin_dsidx_fscore", header=NX.HEADER_NAX, input_names=["Q", "K", "w", "prm"], output_names=["S"],
                                             source=_FSCORE)
    prm = mx.array([Tc, start, ratio, t0, n0, FSCORE_TPT] + [0] * 10, dtype=mx.uint32)
    return k(inputs=[q[0][t0:t0 + n].reshape(-1, 128), keys[0], w[0][t0:t0 + n], prm], grid=(n * 128, -(-Tc // (128 * FSCORE_TPT)), 1),
             threadgroup=(128, 1, 1), output_shapes=[(n, Tc)], output_dtypes=[mx.float32])[0]


def topk_block(sc, k: int):
    """`idx_topk.topk` on one block of scores: sort(argsort(-sc)[:, :k]); with TOPK_MT the exact multi-threadgroup radix select (`idxfast.topk_mt`, the same
    indices: value desc, lower index first on ties) instead of the O(Tc log Tc) sort. E4."""
    if TOPK_MT and 0 < k < sc.shape[1] and METAL_OPT:
        from . import idxfast as IX
        FD.hit("pf_idx_topk_mt")
        return IX.topk_mt(sc, mx.full((sc.shape[0],), sc.shape[1], dtype=mx.int32), k).astype(mx.int32)
    return IT.topk(sc, k)


_POST_G = """
    // M [R, 32, C] half (q . gathered keys), w [R, 32] half, pos [R, C] uint (absolute key column, 0xFFFFFFFF = not a candidate) -> S [R, C] float32,
    // the same sum_h w * relu(M) as _POST in the same order; causal: pos < (start + t0 + t + 1) / ratio else -inf
    const uint n = thread_position_in_grid.x, t = thread_position_in_grid.y;
    const uint C = prm[0], t0 = prm[1], start = prm[2], ratio = prm[3];
    if (n >= C) return;
    const uint clen = (start + t0 + t + 1) / ratio;
    float acc = 0.0f;
    const device half* mp = M + ulong(t) * 32 * C + n;
    for (uint h = 0; h < 32; ++h) acc += max(float(mp[ulong(h) * C]), 0.0f) * float(w[ulong(t) * 32 + h]);
    S[ulong(t) * C + n] = pos[ulong(t) * C + n] < clen ? acc : -INFINITY;
"""


def cand_enabled() -> bool:
    return os.environ.get(CAND_ENV, "1") not in ("0", "")


def cand_topk(q, keys, w, start: int, ratio: int, t0: int, n: int, cand, bs: int, nb: int, topk: int, max_elems: int = 1 << 26, with_scores: bool = False):
    """E1: top-k of rows [t0, t0 + n) of a candidate-consumer indexer layer, scoring only the kept key blocks of each row.
    cand [1, T, Tc] bool is block-granular (bs keys per block, `select_candidate_blocks`), at most nb blocks kept per row. Per row the kept blocks are gathered in
    ascending order (stable sort), scored with the same matmul + relu / weight / head-sum order as `scores_range`, and the compacted columns (ascending absolute
    position, the unkept padding last at -inf) go through the stock `sort(argsort(-s)[:topk])`: argsort is stable, so ties resolve to the lower absolute index as in
    the stock full-width path. Returns int32 [n, topk] absolute key positions ascending (with_scores: also the float32 scores at them).
    Differs from the stock path only for a row with fewer than `topk` finite scores (its -inf filler picks other positions); the caller keeps nb * bs >= topk."""
    NH, D = q.shape[2], q.shape[3]
    Tc = keys.shape[1]
    nblk = -(-Tc // bs)
    C = nb * bs
    k = _kern.get("g")
    if k is None:
        k = _kern["g"] = mx.fast.metal_kernel(name="kyojin_dsidx_post_g", input_names=["M", "w", "pos", "prm"], output_names=["S"], source=_POST_G)
    kb = mx.pad(keys[0], [(0, nblk * bs - Tc), (0, 0)]).reshape(nblk, bs, D)
    bt = max(1, min(n, max_elems // (NH * C), (1 << 26) // C))                                # also bounds the gathered keys [bt, C, 128]
    offs = mx.arange(bs, dtype=mx.uint32)
    idxs, vals = [], []
    for s0 in range(t0, t0 + n, bt):
        r = min(bt, t0 + n - s0)
        flag = cand[0, s0:s0 + r, ::bs]                                                       # [r, nblk] kept block
        ids = mx.argsort(mx.logical_not(flag).astype(mx.uint8), axis=-1)[:, :nb]              # kept blocks first, ascending
        ok = mx.take_along_axis(flag, ids, axis=-1)
        pos = mx.where(ok[..., None], ids.astype(mx.uint32)[..., None] * bs + offs, mx.array(0xFFFFFFFF, dtype=mx.uint32)).reshape(r, C)
        kg = kb[ids].reshape(r, C, D)
        m = q[0, s0:s0 + r].reshape(r, NH, D) @ kg.swapaxes(1, 2)                           # [r, 32, C] half
        prm = mx.array([C, s0, start, ratio] + [0] * 12, dtype=mx.uint32)
        sc = k(inputs=[m, w[0, s0:s0 + r], pos, prm], grid=(C, r, 1), threadgroup=(min(C, 256), 1, 1), output_shapes=[(r, C)], output_dtypes=[mx.float32])[0]
        sel = mx.argsort(-sc, axis=-1)[:, :topk]
        ab = mx.take_along_axis(pos, sel, axis=-1).astype(mx.int32)
        o = mx.argsort(ab, axis=-1)
        idxs.append(mx.take_along_axis(ab, o, axis=-1))
        if with_scores:
            vals.append(mx.take_along_axis(mx.take_along_axis(sc, sel, axis=-1), o, axis=-1))
        mx.eval(idxs[-1]) if not with_scores else mx.eval(idxs[-1], vals[-1])
    cat = lambda xs: xs[0] if len(xs) == 1 else mx.concatenate(xs, axis=0)
    return (cat(idxs), cat(vals)) if with_scores else cat(idxs)



def scores_mm(q, keys, w, start: int, ratio: int, max_elems: int = 1 << 26, n0: int = 0):
    """Same result as `scores` (up to the half rounding of q . k, as in the stock path): MLX matmul per block of queries + one fused post kernel (relu, head weights,
    head sum, causal mask in one read of the product instead of the stock chain of passes). q [1, T, 32, 128], keys [1, Tc, 128], w [1, T, 32] -> [T, Tc] float32."""
    return scores_range(q, keys, w, start, ratio, 0, q.shape[1], max_elems, n0)


def score_blocks(q, keys, w, start: int, ratio: int, qblock: int | None = None, max_elems: int = 1 << 26):
    """(t0, n, scores [n, Tc]) for every reduction block of the chunk: the rows are handed out in blocks of `qblock` (0/None: `idx_topk.qblock(T)`), so the caller
    can reduce a block and drop its score tile before the next one is built."""
    T = q.shape[1]
    for t0, n in IT.plan(T, IT.qblock(T) if qblock is None else qblock):
        yield t0, n, scores_range(q, keys, w, start, ratio, t0, n, max_elems)


def _impl_indexer(orig):
    import importlib
    L = _A.lang()

    def __call__(self, x, qr, latent, start_pos, offset, cache):
        batch, seqlen = x.shape[0], x.shape[1]
        if not (ENABLED and batch == 1 and seqlen >= 32 and x.dtype == mx.float16 and self.index_head_dim == 128 and self.n_heads == 32):
            return orig(self, x, qr, latent, start_pos, offset, cache)
        ratio, end_pos = self.compress_ratio, start_pos + seqlen
        cp_ent = None                                      # KYOJIN_PF_IDX_CPACK: the packed key entry this plain layer scans (no dense mirror)
        if self.owns_k:
            cp_on = CP.layer_ok(self, start_pos, seqlen)
            if latent is not None:
                with CP.deferred(cache, cp_on):
                    self._publish_keys(latent, start_pos, cache)
            cp_ent = CP.packed_entry(cache, self.layer_idx) if cp_on else None
            if cp_ent is not None and cp_ent[1].n < end_pos // ratio:
                cp_ent = None
            if cp_ent is None:
                cache.index_k = cache.keys[self.layer_idx]
        elif getattr(cache, "index_k", None) is None and getattr(cache, "cp_owner", None) is not None:
            ent = CP.packed_entry(cache, cache.cp_owner)       # the owner scanned its packed keys: a layer that reads its keys gets a transient decode
            if ent is not None:
                cache.index_k = CP.decode(ent[0], ent[1], ent[1].n)
        cos, sin = L._index_cos_sin(end_pos, self.rope_head_dim, self.rope_theta, self.yarn)
        q = self.wq_b(L.fake_quant_fp8_ue8m0(qr)).reshape(batch, seqlen, self.n_heads, self.index_head_dim)
        q = fq4_rot(q, cos[start_pos:end_pos], sin[start_pos:end_pos], self.rope_head_dim)
        weights = self.weights_proj(x) * (self.scale * self.n_heads ** -0.5)
        compress_lens = (mx.arange(start_pos + 1, end_pos + 1) // ratio)[:, None]
        if cp_ent is not None and not CP.applies(q, CP.shape_of(end_pos // ratio), weights, self.index_topk, start_pos, ratio):
            cache.index_k = cache.keys[self.layer_idx]     # shapes / device / flags say the packed path cannot serve this chunk: the stock read, nothing decoded or dropped
            cp_ent = None
        if cp_ent is not None:
            keys = CP.decode(cp_ent[0], cp_ent[1], end_pos // ratio)
            CP.drop_mirror(cp_ent[1])
            cache.cp_owner = self.layer_idx
            got = CP.try_plain(q, keys, weights, start_pos, ratio, self.index_topk, seqlen, key=self.layer_idx) \
                if CP.applies(q, keys, weights, self.index_topk, start_pos, ratio) else None
            if got is not None:
                cache.index_k = None
                return mx.where(got[None] < compress_lens, got[None] + offset, -1).astype(mx.int32)
            cache.index_k = keys                           # not served: the normal chain below runs on the transient decode
        index_k = cache.index_k[:batch, : end_pos // ratio]
        Tc = index_k.shape[1]
        cand_ids = PF.fetch(cache, start_pos, seqlen, Tc) if self.uses_candidates else None      # block ids left by a block-max / cand_ids source layer
        if LC.PFTOPK and not self.is_candidate_source and Tc >= max(LC.PF_MIN, 2 * self.index_topk):      # long past: tiled exact top-k, no [T, Tc] scores
            cm = cache.candidates if self.uses_candidates else None
            if self.uses_candidates and cm is None:        # the source layer kept ids, not a mask: expand them (same mask the stock source would have written)
                cm = PF.mask_from_ids(cand_ids, -(-Tc // PF.BS), Tc)[None]
            cand = cm.reshape(-1, cm.shape[-1]) if cm is not None else None
            idxs = LC.pf_topk(q, index_k, weights, start_pos, ratio, self.index_topk, cand[None] if cand is not None else None)[None]
            return mx.where(idxs < compress_lens, idxs + offset, -1).astype(mx.int32)
        topk = min(self.index_topk, end_pos // ratio)
        cand = cache.candidates if self.uses_candidates else None
        kq = qmm_pack(index_k) if (qmm_on() and index_k.dtype == mx.float16 and index_k.shape[-1] == 128
                                  and index_k.shape[1] >= QMM_MIN and mx.metal.is_available() and mx.default_device() == mx.gpu) else None

        def block(t0, n):
            s = scores_range(q, index_k, weights, start_pos, ratio, t0, n, kq=kq)
            return s if cand is None else mx.where(cand[0, t0:t0 + n], s, -mx.inf)

        if self.uses_candidates and cand is None and cand_ids is None:
            raise RuntimeError("candidate consumer layer: the source layer left neither a candidate mask nor block ids for this chunk")
        idxs = None
        if PB.ENABLED and PB.applies(q, index_k, weights, self.index_topk, start_pos, ratio, not self.is_candidate_source and not self.uses_candidates):
            idxs = PB.try_plain(q, index_k, weights, start_pos, ratio, self.index_topk, seqlen, key=self.layer_idx)       # KYOJIN_PF_IDX_BOUND: exact superblock bound; None = stock path
        if idxs is not None:
            pass
        elif PC.applies(q, index_k, weights, Tc, self.index_topk, start_pos, ratio, not self.is_candidate_source and not self.uses_candidates):
            idxs = PC.plain_topk(q, index_k, weights, start_pos, ratio, self.index_topk, seqlen)          # KYOJIN_PF_IDX_CLUSTER: one candidate block set per query block
        elif self.is_candidate_source and self.candidate_block_size == PF.BS and PF.bmax_ok(q, index_k, weights, Tc, self.index_topk, start_pos, ratio, True,
                                                                                             self.candidate_topk_blocks):
            idxs, ids = PF.source_topk(q, index_k, weights, start_pos, ratio, self.index_topk, seqlen, self.candidate_topk_blocks,
                                       key=self.layer_idx)                                                                               # KYOJIN_PF_IDX_BMAX (+ GSCORE)
            PF.stash(cache, ids, start_pos, seqlen, Tc)
            cache.candidates = None                        # the consumers of this chunk read the block ids; nothing else reads the mask after the forward
        elif cand_ids is not None:                         # KYOJIN_PF_CAND_IDS / BMAX: the kept blocks come from the source layer, no mask read, no sort over Tc / 8
            idxs = PF.consumer_topk(q, index_k, weights, start_pos, ratio, topk, seqlen, cand_ids)
        elif not self.is_candidate_source and not self.uses_candidates and self.candidate_block_size == PF.BS and PF.bmax_ok(
                q, index_k, weights, Tc, self.index_topk, start_pos, ratio, False):
            idxs = PF.plain_topk(q, index_k, weights, start_pos, ratio, self.index_topk, seqlen, key=self.layer_idx)                     # KYOJIN_PF_IDX_BMAX
        elif self.is_candidate_source:                     # the same score tile serves the block maxima of level one and the top-k of level two
            fused = PF.SELFUSE and PF.METAL_OPT and self.candidate_block_size == PF.BS and (start_pos + 1) // ratio >= topk == self.index_topk

            def both(t0, s):
                if fused:                                  # KYOJIN_PF_IDX_SELFUSE: top-k and the candidate mask in one exact pass (idxfast.select_fused)
                    from . import idxfast as IX
                    PF.HITS["selfuse"] += 1
                    idx, m = IX.select_fused(s, compress_lens[t0:t0 + s.shape[0], 0].astype(mx.int32), topk, self.candidate_topk_blocks, PF.BS)
                    return m, idx
                m = L.select_candidate_blocks(s[None], compress_lens[t0:t0 + s.shape[0]], self.candidate_topk_blocks, self.candidate_block_size)
                return m[0], topk_block(s, topk)
            masks, idxs = IT.scan(block, seqlen, IT.qblock(seqlen), both)
            cache.candidates = masks[None]
            PF.clear(cache)
            if PF.CAND_IDS and self.candidate_block_size == PF.BS and cand_enabled() and Tc >= 5 * self.candidate_topk_blocks * self.candidate_block_size:
                PF.stash(cache, PF.make_ids(masks, self.candidate_topk_blocks), start_pos, seqlen, Tc)      # the consumers' gate (E1) is the same width test
        elif PR.enabled() and topk >= 1:                   # KYOJIN_PFROOF2: each block scores only the columns its last row can see (identical output, see pfroof2)
            FD.hit("pfroof2")
            def block_w(t0, n):
                w = PR.causal_width(start_pos, t0, n, ratio, Tc, topk)
                kw = (kq[0][:w], kq[1][:w]) if kq is not None else None
                s = scores_range(q, index_k[:, :w], weights, start_pos, ratio, t0, n, kq=kw)
                return s if cand is None else mx.where(cand[0, t0:t0 + n, :w], s, -mx.inf)
            idxs = IT.scan(block_w, seqlen, IT.qblock(seqlen), lambda t0, s: PR.select_topk(s, topk))
        elif cand is not None and cand_enabled() and topk >= 1 and self.candidate_topk_blocks * self.candidate_block_size >= max(topk, 1) \
                and Tc >= 5 * self.candidate_topk_blocks * self.candidate_block_size:        # E1: score only the kept blocks; the per-row key gather is ~flat, the full width wins below ~80K keys (bench 30/10)
            FD.hit("pf_idx_cand")
            idxs = IT.scan(lambda t0, n: (t0, n), seqlen, IT.qblock(seqlen),
                           lambda t0, tn: cand_topk(q, index_k, weights, start_pos, ratio, tn[0], tn[1], cand, self.candidate_block_size,
                                                    self.candidate_topk_blocks, topk, with_scores=False))
        else:
            idxs = IT.scan(block, seqlen, IT.qblock(seqlen), lambda t0, s: topk_block(s, topk))
        idxs = idxs[None]
        return mx.where(idxs < compress_lens, idxs + offset, -1).astype(mx.int32)

    return __call__


def _impl_v32_indexer(orig):
    """deepseek_v32 Indexer (GLM-5.x shares it): the stock prefill builds [b, 32, s, T] scores, relu, weights, head sum (dense, quadratic memory). Same result with the
    per-block matmul + one reduction kernel; the top-k is the stock argpartition on the same masked scores."""
    def __call__(self, x, qr, mask, cache=None):
        b, s, _ = x.shape
        if not (ENABLED and b == 1 and s >= 32 and x.dtype == mx.float16 and self.head_dim == 128 and self.n_heads == 32):
            return orig(self, x, qr, mask, cache)
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = mx.reshape(self.k_norm(self.wk(x)), (b, 1, s, self.head_dim))
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))
        if k.shape[2] <= self.index_topk:
            return None
        w = self.weights_proj(x) * (self.n_heads ** -0.5 * self.softmax_scale)
        topk = self.index_topk

        def block(t0, n):
            sc = scores_range(q.swapaxes(1, 2), k[:, 0], w, offset, 1, t0, n)
            return sc if mask is None else mx.where(mask[..., t0:t0 + n, :].reshape(n, -1), sc, -float("inf"))

        idx = IT.scan(block, s, IT.qblock(s), lambda t0, sc: IT.partition(sc, topk))
        return idx[None, None]
    return __call__
