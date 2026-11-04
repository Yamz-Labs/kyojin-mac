# SPDX-License-Identifier: Apache-2.0
"""Prefill flatness past 128K (cloud/opflat-2): the terms of one prompt chunk that grow with the past and are not the indexer's own arithmetic.

Model of one 8192-row chunk (DS4.1: 3 plain indexer layers at index ratio 2, the candidate source at ratio 1, four candidate consumers at ratio 1; Tc = keys):
  indexer FLOPs      2 x 32 x 128 x T x Tc per layer, ~0.83 s per 1M keys at the 83 TFLOP/s fscore roof   grows with Tc, exact arithmetic: NOT removable here
  score traffic      [rows, Tc] fp32 scores written once, read ~5x (select_candidate_blocks, radix top-k)   grows with Tc: levers `bmax` / `selfuse`
  consumer gather    per layer: strided read of the [T, Tc] candidate mask (8 GB at 1M), argsort over Tc / 8 blocks per row   grows with Tc: lever `cand_ids`
  consumer rescoring gather of the kept 16K keys per row into [rows, 16K, 128] and a batched matmul: constant in Tc, ~100 ms per layer: lever `gscore`

Levers (each behind its own flag, default off, all exact: same indices as the stock chain, see tests/test_pf_flat.py):

  bmax     (KYOJIN_PF_IDX_BMAX, KYOJIN_PF_IDX_BMAX_MIN keys)  a plain indexer layer, and with `gscore` the candidate source, never builds the [rows, Tc] scores.
           One fused kernel scores like `fscore` but writes only the max of each 8-key block ([rows, Tc / 8], 1/32 of the bytes). The best blocks come from the
           exact radix select on those maxima (idxfast.topk_mt), the picked blocks are rescored (gather + the same score order) and the exact top-k of that
           small set is the answer. Exactness is the argument of idxblk: if a top-k key lay in an unpicked block B, the k picked blocks would hold k keys
           ranked before B's best, hence before it. The scores of the rescored keys are the ones the full-width path computes (bitwise, as `cand_topk` already relies on).
           The candidate source needs more than the top-k: its candidate blocks are the top `candidate_topk_blocks` block maxima with the last visible block always
           kept, ties to the lower block, only blocks with a finite maximum (select_candidate_blocks); the same ranking gives them from the same maxima. The top-k of the
           whole row lies inside the top-k blocks, hence inside the candidate blocks, so the source rescoring them yields its top-k as well.
  gscore   (KYOJIN_PF_IDX_GSCORE)  rescoring of picked blocks in one kernel: the 8-key blocks are staged straight from the key array into threadgroup memory and
           fed to the same matmul2d as `fscore`; the [rows, 8 x blocks, 128] gathered copy and its batched matmul disappear.
  cand_ids (KYOJIN_PF_CAND_IDS)  the candidate source hands the kept block ids of each row ([T, K] int32, ascending) to its consumers through `cache.cand_ids`
           instead of the [T, Tc] bool mask they each re-read (strided) and argsort over Tc / 8 blocks: one cumsum + scatter in the source layer (or nothing, with `bmax`).
  selfuse  (KYOJIN_PF_IDX_SELFUSE)  candidate source on the stock path: `idxfast.select_fused` (top-k and the candidate mask in one exact pass) instead of the vendor
           select_candidate_blocks plus a second radix top-k over the same scores.

`HITS` counts the calls of each path. Scoring kernels need Metal; the glue (ranking, compaction, selection, tie-breaking) is plain MLX and runs on the CPU in the tests
with numpy scoring (`METAL_OPT` False)."""
from __future__ import annotations

import os
import sys

import mlx.core as mx
import numpy as np

from . import idx_topk as IT
from . import metal_nax as NX

METAL_OPT = (mx.metal.is_available() and mx.default_device() == mx.gpu)

BMAX = os.environ.get("KYOJIN_PF_IDX_BMAX", "1") != "0"
BMAX_MIN = int(os.environ.get("KYOJIN_PF_IDX_BMAX_MIN", "65536"))     # keys from which a plain / source layer takes the block-max path
GSCORE = os.environ.get("KYOJIN_PF_IDX_GSCORE", "1") != "0"
CAND_IDS = os.environ.get("KYOJIN_PF_CAND_IDS", "1") != "0"
SELFUSE = os.environ.get("KYOJIN_PF_IDX_SELFUSE", "1") != "0"
I8 = os.environ.get("KYOJIN_PF_IDX_I8", "1") not in ("0", "")                # plain layers: first pass on int8 operands (`i8pack`), 2x the half matmul2d rate; rescoring stays exact
I8_MARGIN = int(os.environ.get("KYOJIN_PF_IDX_I8_MARGIN", "32"))             # blocks picked beyond topk, so the rare rounded (row, head) cannot push a true top-k block out
I8_SRC = os.environ.get("KYOJIN_PF_IDX_I8_SRC", "1") not in ("0", "")        # candidate source: the same int8 first pass, the kept blocks ranked again on the exact maxima of the picked ones
I8_SRC_MARGIN = int(os.environ.get("KYOJIN_PF_IDX_I8_SRC_MARGIN", "128"))     # blocks picked beyond the candidate count and rescored exactly
CAPTURE = os.environ.get("KYOJIN_PF_IDX_CAPTURE", "0") not in ("0", "")      # offline analysis of real keys; nothing reads the dump in the served path
CAPTURE_DIR = os.environ.get("KYOJIN_PF_IDX_CAPTURE_DIR", "/tmp/idxcap")
CAPTURE_MIN = 400000
CAPTURE_MAX_FILES = 12                  # 3 layers x 4 chunks; keys are written once per layer

BS = 8                                  # keys per block (the vendor candidate block size; the kernels are written for 8)
INVALID = 0x7FFFFFFF                    # position handed back for a column that holds no key (never < a causal length)
I8_MAX_OFFGRID = 0.01                                                          # keys off the int8 grid above this share: the half first pass stays
_I8_GRID = None
I8_REG = os.environ.get("KYOJIN_PF_IDX_I8_REG", "1") not in ("0", "")          # int8 first pass: the head sum in registers (`i8_fast_source`)
I8_TPT = int(os.environ.get("KYOJIN_PF_IDX_I8_TPT", "32"))                     # 128-key tiles per threadgroup of the int8 first pass (more: fewer query copies)
HEADS = int(os.environ.get("KYOJIN_PF_IDX_HEADS", "8"))                        # plain / source layer first pass: query heads per row (0 = all 32 = today; 4, 8 or 16 = a subset)
HEADS_M = int(os.environ.get("KYOJIN_PF_IDX_HEADS_M", "1024"))                 # blocks that subset keeps per row, all rescored exactly (0 = the exact path's own count)
HEADS_CHECK = os.environ.get("KYOJIN_PF_IDX_HEADS_CHECK", "0") not in ("0", "")  # also run the exact all-head pass and count the top-k recall per layer
_I8_MAP = False                                                                # False = not probed, None = layout not supported
HITS = {"i8_plain": 0, "i8_source": 0, "i8_offgrid": 0, "bmax_plain": 0, "bmax_source": 0, "gscore": 0, "gscore_fallback": 0, "cand_ids_make": 0, "cand_ids_use": 0, "selfuse": 0,
        "heads_plain": 0, "heads_source": 0, "heads_i8off": 0, "heads_recall_hit": 0, "heads_recall_total": 0}
_kern: dict = {}
_failed: set = set()                    # kernel names that did not build on this device (the caller falls back to the unfused path)


# ----------------------------------------------------------------------------------------------------------------- block-max score kernel
_BM_OLD = """        if (lane < rem) {
            float s = 0.0f;
            const uint lk = k0 + lane - ks;
            for (uint h = 0; h < 32; ++h) s += metal::max(scr[sg][lk * 33 + h], 0.0f) * Wh[h];
            const uint n = k0 + lane;
            S[ulong(t) * Tc + n] = n + n0 < clen ? s : -INFINITY;
        }
"""
_BM_NEW = """        float s = -INFINITY;
        if (lane < rem) {
            float acc = 0.0f;
            const uint lk = k0 + lane - ks;
            for (uint h = 0; h < 32; ++h) acc += metal::max(scr[sg][lk * 33 + h], 0.0f) * Wh[h];
            const uint n = k0 + lane;
            s = n + n0 < clen ? acc : -INFINITY;
        }
        s = metal::max(s, simd_shuffle_xor(s, 1u));              // max over the 8 lanes (= 8 keys) of a block; k0 is a multiple of 32, so blocks never straddle tiles
        s = metal::max(s, simd_shuffle_xor(s, 2u));
        s = metal::max(s, simd_shuffle_xor(s, 4u));
        if ((lane & 7u) == 0u && k0 + lane < Tc) S[ulong(t) * ((Tc + 7u) >> 3) + ((k0 + lane) >> 3)] = s;
"""


def bmax_source() -> str:
    """The `fscore` kernel body with its write replaced by the block maximum (same products, same head order, same causal mask)."""
    from . import ds_pf_idx as PI
    assert _BM_OLD in PI._FSCORE, "ds_pf_idx._FSCORE changed: update pf_flat._BM_OLD"
    return PI._FSCORE.replace(_BM_OLD, _BM_NEW)


def fscore_ok(q, keys, w) -> bool:
    from . import ds_pf_idx as PI
    return PI.fscore_ok(q, keys, w)


def bmax_scores(q, keys, w, start: int, ratio: int, t0: int, n: int, sel=None):
    """[n, ceil(Tc / 8)] float32: for rows [t0, t0 + n) the maximum over each 8-key block of the indexer scores (causal -inf included), i.e.
    `scores_range(...).reshape(blocks, 8).max(-1)` without the [n, Tc] scores in memory. Without Metal: numpy scoring (tests).
    `sel` (numpy [T, nh] int32) scores only the nh heads of each row: the CPU mirror of the packed int8 kernel. The Metal kernel takes no subset (its
    head tile is the whole 32), so a call that reaches it with `sel` scores every head: an exact ranking, only the candidate set is wider."""
    from . import ds_pf_idx as PI
    from . import pfroof2 as PR
    Tc = keys.shape[1]
    nblk = -(-Tc // BS)
    if not (METAL_OPT and fscore_ok(q, keys, w)):
        qq, kk, ww = np.array(q[0, t0:t0 + n]), np.array(keys[0]), np.array(w[0, t0:t0 + n])
        if sel is not None:
            s = np.asarray(sel)[t0:t0 + n]
            qq = np.take_along_axis(qq, s[..., None], axis=1)
            ww = np.take_along_axis(ww, s, axis=1)
        s = PR.ref_scores(qq, kk, ww, start, ratio, t0)
        s = np.pad(s, [(0, 0), (0, nblk * BS - Tc)], constant_values=-np.inf)
        return mx.array(s.reshape(n, nblk, BS).max(-1))
    k = _kern.get("bm")
    if k is None:
        k = _kern["bm"] = mx.fast.metal_kernel(name="kyojin_pfflat_bmax", header=NX.HEADER_NAX, input_names=["Q", "K", "w", "prm"], output_names=["S"],
                                              source=bmax_source())
    prm = mx.array([Tc, start, ratio, t0, 0, PI.FSCORE_TPT] + [0] * 10, dtype=mx.uint32)
    return k(inputs=[q[0][t0:t0 + n].reshape(-1, 128), keys[0], w[0][t0:t0 + n], prm], grid=(n * 128, -(-Tc // (128 * PI.FSCORE_TPT)), 1), threadgroup=(128, 1, 1),
             output_shapes=[(n, nblk)], output_dtypes=[mx.float32])[0]


# ----------------------------------------------------------------------------------------------------------------- int8 first pass
# matmul2d takes int8 x int8 -> int32 at twice the half x half rate (135 against 67 TFLOP/s, M5 Ultra, real keys). The FP4 grid fits: each value is m * 2^s with few
# significant bits, so one power of two per key (per query head) maps a row to int8 without loss in nearly all cases (real keys >= 99.9 %, query heads >= 99.3 %). Those
# powers of two fold into the head weight and the key's score, the int32 sum is the exact fp32 sum of the half path, and only the half rounding of the head product goes.
# The first pass only ranks blocks: the picked blocks (topk + I8_MARGIN) are rescored by the exact half kernel, so a returned score is always the exact one.
_I8PACK = """
    // A [R, 128] half -> Y [R, 128] int8, E [R] float: Y[r] * E[r] == A[r] when A[r] lies on the int8 grid of its own power of two (else rounded). One row per simdgroup.
    const uint r = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    float4 v;
    for (uint j = 0; j < 4; ++j) v[j] = float(A[ulong(r) * 128 + lane * 4 + j]);
    const float m = simd_max(metal::max(metal::max(fabs(v.x), fabs(v.y)), metal::max(fabs(v.z), fabs(v.w))));
    const float e = m > 0.0f ? ceil(log2(m / 127.0f)) : 0.0f;
    const float up = exp2(e), dn = exp2(-e);
    for (uint j = 0; j < 4; ++j) Y[ulong(r) * 128 + lane * 4 + j] = char(clamp(rint(v[j] * dn), -127.0f, 127.0f));
    if (lane == 0u) E[r] = up;
"""


def i8pack(a):
    """a [R, 128] half -> (int8 [R, 128], float32 [R] power of two): a[r] == codes[r] * scale[r] on the int8 grid. Without Metal: numpy (tests)."""
    if not METAL_OPT:
        x = np.array(a).astype(np.float32)
        m = np.abs(x).max(-1)
        e = np.where(m > 0, np.ceil(np.log2(np.where(m > 0, m, 1.0) / 127.0)), 0.0)
        return mx.array(np.clip(np.rint(x * np.exp2(-e)[:, None]), -127, 127).astype(np.int8)), mx.array(np.exp2(e).astype(np.float32))
    k = _kern.get("i8p")
    if k is None:
        k = _kern["i8p"] = mx.fast.metal_kernel(name="kyojin_pfflat_i8pack", input_names=["A"], output_names=["Y", "E"], source=_I8PACK)
    R = a.shape[0]
    return k(inputs=[a], grid=(R * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(R, 128), (R,)], output_dtypes=[mx.int8, mx.float32])


def i8_on_grid(keys) -> bool:
    """Once per process, on 4096 keys: a pack off the int8 grid would only rank by rounded values, so the half first pass stays (no silent recall loss on another quant format)."""
    global _I8_GRID
    if _I8_GRID is None:
        a = keys[0, :4096]
        c, s = i8pack(a)
        bad = mx.any(c.astype(mx.float32) * s[:, None] != a.astype(mx.float32), axis=1)
        _I8_GRID = float(mx.mean(bad.astype(mx.float32))) <= I8_MAX_OFFGRID
        HITS["i8_offgrid"] += not _I8_GRID
    return _I8_GRID


def i8_source() -> str:
    """The block-max kernel on int8 operands: the int32 product goes straight to fp32 (no half rounding), head scales are in w, the key's scale multiplies its sum."""
    s = bmax_source()
    for old, new in (("threadgroup half Qs[32 * 128];", "threadgroup int8_t Qs[32 * 128];"),
                     ("if (tid < 32u) Wh[tid] = float(w[ulong(t) * 32 + tid]);", "if (tid < 32u) Wh[tid] = w[ulong(t) * 32 + tid];"),
                     ("tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tQ((threadgroup half*)Qs,", "tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ((threadgroup int8_t*)Qs,"),
                     ("tensor<device half, dextents<int32_t, 2>, tensor_inline> tK((device half*)(K + ulong(ks) * 128),", "tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t*)(K + ulong(ks) * 128),"),
                     ("decltype(tQ), float>()", "decltype(tQ), int32_t>()"),
                     ("= float(half(cT[i]));", "= float(cT[i]);"),
                     ("s = n + n0 < clen ? acc : -INFINITY;", "s = n + n0 < clen ? acc * E[n] : -INFINITY;")):
        assert old in s, old
        s = s.replace(old, new)
    return s


# The copy of the product to threadgroup memory and the scalar head sum cost about 30 % of the int8 first pass on 16 fewer query copies (TPT 8 -> 32: 259 -> 196 ms per layer at 983K keys; a kernel reading one constant cell is elided, 5 ms per block means nothing). matmul2d itself is the rest, ~180 TFLOP/s. A lane of the
# cooperative tensor holds a regular [4 keys, 8 heads] piece, so the sum stays in registers: 8 head weights per lane, one float4 per lane, two xor shuffles join the heads of a
# key, three more give the block maxima (4 blocks per tile, written by lane 0). The piece layout comes from a probe at first use (`i8_tile_map`); another layout keeps the
# slow post. A tail tile (keys not aligned on blocks) always takes the slow post.
_I8_PROBE = """
    const uint lane = thread_index_in_simdgroup;
    threadgroup int8_t Qs[32 * 128];
    for (uint i = thread_index_in_threadgroup; i < 32u * 128u; i += 128u) Qs[i] = Q[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = matmul2d_descriptor(32, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ((threadgroup int8_t*)Qs, dextents<int32_t, 2>(128, 32));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t*)K, dextents<int32_t, 2>(128, 32));
    auto cT = op.get_destination_cooperative_tensor<decltype(tK), decltype(tQ), int32_t>();
    op.run(tK, tQ, cT);
    if (simdgroup_index_in_threadgroup == 0) {
        const uint cap = cT.get_capacity();
        S[lane * 66] = int(cap);
        for (uint16_t i = 0; i < min(cap, 32u); ++i) {
            const auto ix = cT.get_multidimensional_index(i);
            S[lane * 66 + 1 + i * 2] = cT.is_valid_element(i) ? int(ix[0]) : -1;
            S[lane * 66 + 2 + i * 2] = int(ix[1]);
        }
    }
"""


def i8_tile_map():
    """{"hb": (lane bits of head base +4, +8), "kr": (lane bits of key residue +1, +2, +4)} when each lane of a [32 keys, 32 heads] tile holds the regular [4 keys, 8 heads] piece, else None."""
    global _I8_MAP
    if _I8_MAP is not False:
        return _I8_MAP
    _I8_MAP = None
    k = mx.fast.metal_kernel(name="kyojin_pfflat_i8probe", header=NX.HEADER_NAX, input_names=["Q", "K"], output_names=["S"], source=_I8_PROBE)
    z = mx.zeros((32, 128), dtype=mx.int8)
    o = np.array(k(inputs=[z, z], grid=(128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(32, 66)], output_dtypes=[mx.int32])[0])
    hb, kr = [], []
    for lane in range(32):
        if o[lane, 0] != 32:
            return None
        heads, keys = o[lane, 1::2], o[lane, 2::2]
        hb.append(int(heads[0]))
        kr.append(int(keys[0]))
        for i in range(32):
            if heads[i] != hb[lane] + (i & 3) + 16 * ((i >> 3) & 1) or keys[i] != kr[lane] + 8 * ((i >> 2) & 1) + 16 * ((i >> 4) & 1):
                return None
    bits = {}
    for b in range(5):
        dh = {hb[la ^ (1 << b)] - hb[la] for la in range(32) if not la & (1 << b)}
        dk = {kr[la ^ (1 << b)] - kr[la] for la in range(32) if not la & (1 << b)}
        if len(dh) != 1 or len(dk) != 1 or (dh == {0}) == (dk == {0}):
            return None
        bits[dh.pop() or 100 * dk.pop()] = b
    if sorted(bits) != [4, 8, 100, 200, 400]:
        return None
    _I8_MAP = {"hb": (bits[4], bits[8]), "kr": (bits[100], bits[200], bits[400])}
    return _I8_MAP


def i8_fast_source(m) -> str:
    """`i8_source` with the register post on the aligned tiles, m from `i8_tile_map`."""
    h4, h8 = m["hb"]
    r1, r2, r4 = m["kr"]
    s = i8_source()
    head = f"""    const uint hb = ((lane >> {h4}u) & 1u) * 4u + ((lane >> {h8}u) & 1u) * 8u;
    const uint kr = ((lane >> {r1}u) & 1u) + ((lane >> {r2}u) & 1u) * 2u + ((lane >> {r4}u) & 1u) * 4u;
    const float4 w0(Wh[hb], Wh[hb + 1u], Wh[hb + 2u], Wh[hb + 3u]), w1(Wh[hb + 16u], Wh[hb + 17u], Wh[hb + 18u], Wh[hb + 19u]);
    for (uint j = 0; j < TPT; ++j) {{"""
    body = f"""        op.run(tK, tQ, cT);
        if (ks == k0) {{
            float4 a;
            for (uint c = 0; c < 4u; ++c) {{
                const uint16_t ib = uint16_t(((c & 1u) << 2) | ((c & 2u) << 3));
                float acc = 0.0f;
                for (uint16_t h = 0; h < 4u; ++h) acc += metal::max(float(cT[ib + h]), 0.0f) * w0[h] + metal::max(float(cT[ib + 8u + h]), 0.0f) * w1[h];
                a[c] = acc;
            }}
            a += simd_shuffle_xor(a, ushort({1 << h4}));
            a += simd_shuffle_xor(a, ushort({1 << h8}));
            const uint4 n = k0 + kr + uint4(0u, 8u, 16u, 24u);
            a *= float4(E[n.x], E[n.y], E[n.z], E[n.w]);
            a = select(float4(-INFINITY), a, n + n0 < clen);
            a = metal::max(a, simd_shuffle_xor(a, ushort({1 << r1})));
            a = metal::max(a, simd_shuffle_xor(a, ushort({1 << r2})));
            a = metal::max(a, simd_shuffle_xor(a, ushort({1 << r4})));
            if (lane == 0u) for (uint c = 0; c < 4u; ++c) S[ulong(t) * ((Tc + 7u) >> 3) + (k0 >> 3) + c] = a[c];
            continue;
        }}"""
    for old, new_ in (("    for (uint j = 0; j < TPT; ++j) {", head), ("        op.run(tK, tQ, cT);", body)):
        assert s.count(old) == 1, old
        s = s.replace(old, new_)
    return s


def bmax_i8(qi, ki, w8, ke, start: int, ratio: int, t0: int, n: int):
    """[n, ceil(Tc / 8)] float32 block maxima from the int8 operands: qi [T * 32, 128], ki [Tc, 128], w8 [T, 32] float32 (head weight * head scale), ke [Tc]."""
    from . import ds_pf_idx as PI
    Tc = ki.shape[0]
    kk = "bmi8r" if I8_REG else "bmi8"
    k = _kern.get(kk)
    if k is None:
        m = i8_tile_map() if I8_REG else None
        k = _kern[kk] = mx.fast.metal_kernel(name="kyojin_pfflat_bmax_i8" + ("r" if m else ""), header=NX.HEADER_NAX, input_names=["Q", "K", "w", "E", "prm"], output_names=["S"],
                                                 source=i8_fast_source(m) if m else i8_source())
    tpt = max(I8_TPT, 1)
    prm = mx.array([Tc, start, ratio, t0, 0, tpt] + [0] * 10, dtype=mx.uint32)
    return k(inputs=[qi[t0 * 32:(t0 + n) * 32], ki, w8[t0:t0 + n], ke, prm], grid=(n * 128, -(-Tc // (128 * tpt)), 1), threadgroup=(128, 1, 1),
             output_shapes=[(n, -(-Tc // BS))], output_dtypes=[mx.float32])[0]


# ----------------------------------------------------------------------------------------------------------------- first pass on a query-head subset
# The int8 block-max pass is one matmul2d tile per query row: [32 keys x 32 head slots x 128] per op.run, 257 TFLOP/s on the M5 (17.1 ms for 8192 rows x
# 65536 keys, no post), 24.7 ms for the whole kernel. A NARROWER tile is not cheaper: with 16 or 8 head slots the same tile still costs 16.5 ms (134 / 67
# TFLOP/s), because matmul2d prices the tile, not the flop. What pays is FILLING the 32 slots with 32 / N query rows of the same key tile: N = 16 puts two
# rows per tile (15.8 ms, 64 % of today), N = 8 four (8.1 ms, 33 %), the keys read once for all of them.
# The head sum of a row covers only its own N slots, so the block maxima rank blocks on a partial score; the blocks picked (topk + margin, or pf_idx_heads_m)
# are rescored by the exact all-head half kernel (`select_from_blocks`), so a returned score is always the exact one and only the candidate set is
# approximate. The heads of a row are the N of largest |w| * ||q_h|| (idxan3's metric: the offline study whose recall the flags quote), row-local, so the
# packed tile keeps the per-row choice; the instantiated descriptor keeps 32 head slots (32 keys, 32 slots, 128), the layout `i8_source` writes.

def heads_n() -> int:
    """pf_idx_heads as the first pass can read it: 4, 8 or 16 heads per row (32 / N query rows share a 32-slot head tile), 32 = every head through the
    same path (today's kernel, the exact first pass), 0 or any other value = today, nothing widened."""
    return HEADS if HEADS in (4, 8, 16, 32) else 0


def heads_sub(nh: int) -> bool:
    """True when nh is a real subset: fewer heads scored per row, 32 / nh rows packed into one tile."""
    return nh in (4, 8, 16)


def head_sel(q, w, nh: int):
    """[T, nh] int32: for each row the nh heads it scores in the first pass, the largest |w| * ||q_h|| first (row-local, so a 32-slot head tile can carry
    32 / nh rows). Computed in row blocks of 2048 to keep the fp32 square transient small."""
    T = q.shape[1]
    out = []
    for a in range(0, T, 2048):
        x = q[0, a:min(a + 2048, T)].astype(mx.float32)
        met = mx.abs(w[0, a:min(a + 2048, T)].astype(mx.float32)) * mx.sqrt(mx.sum(x * x, axis=-1))
        out.append(mx.argsort(-met, axis=-1)[:, :nh])
    return out[0] if len(out) == 1 else mx.concatenate(out, axis=0)


def head_rows(qi, w8, sel, nh: int):
    """(qs [T * nh, 128] int8, ws [T * nh] float32): the per-row head subset in the order a 32-slot head tile reads (32 / nh consecutive rows fill one tile).
    The tail group repeats the last row; `bmax_i8_rows` drops those rows from the block maxima."""
    T = sel.shape[0]
    K = 32 // nh
    qs = mx.take_along_axis(qi.reshape(T, 32, qi.shape[-1]), sel[..., None], axis=1).reshape(T * nh, qi.shape[-1])
    ws = mx.take_along_axis(w8, sel, axis=1).reshape(T * nh)
    pad = (-T) % K
    if pad:                                                     # the last group holds fewer than K rows: the missing ones repeat the last row and are sliced off
        qs = mx.concatenate([qs] + [qs[-nh:]] * pad, axis=0)
        ws = mx.concatenate([ws] + [ws[-nh:]] * pad, axis=0)
    return qs, ws


def i8_rows_source(nh: int) -> str:
    """`i8_source` with 32 / nh query rows per 32-slot head tile: the same Q / key tile, the same matmul2d, the same int32 product, one head sum per row
    (heads slot * nh .. slot * nh + nh - 1), one causal length and one block-max reduction per row."""
    K = 32 // nh
    s = i8_source()
    old = "const uint clen = (start + t0 + t + 1) / ratio;"
    assert s.count(old) == 1, old
    s = s.replace(old, "const uint row0 = t0 + t * %du;" % K)
    head = s.split("        float s = -INFINITY;")
    assert len(head) == 2, "ds_pf_idx._FSCORE post changed: update pf_flat.i8_rows_source"
    return head[0] + f"""
        float v[{K}];
        if (lane < rem) {{
            const uint lk = k0 + lane - ks;
            const uint n = k0 + lane;
            for (uint sl = 0; sl < {K}u; ++sl) {{
                float acc = 0.0f;
                for (uint h = 0; h < {nh}u; ++h) acc += metal::max(scr[sg][lk * 33 + sl * {nh}u + h], 0.0f) * Wh[sl * {nh}u + h];
                v[sl] = n + n0 < (start + row0 + sl + 1) / ratio ? acc * E[n] : -INFINITY;
            }}
        }} else {{
            for (uint sl = 0; sl < {K}u; ++sl) v[sl] = -INFINITY;
        }}
        for (uint sl = 0; sl < {K}u; ++sl) {{                  // max over the 8 lanes (= 8 keys) of a block, per row; blocks never straddle tiles
            float b = metal::max(v[sl], simd_shuffle_xor(v[sl], 1u));
            b = metal::max(b, simd_shuffle_xor(b, 2u));
            b = metal::max(b, simd_shuffle_xor(b, 4u));
            if ((lane & 7u) == 0u && k0 + lane < Tc) S[ulong(t * {K}u + sl) * ((Tc + 7u) >> 3) + ((k0 + lane) >> 3)] = b;
        }}
    }}
"""


def bmax_i8_rows(qs, ki, ws, ke, start: int, ratio: int, t0: int, n: int, nh: int):
    """[n, ceil(Tc / 8)] block maxima of rows [t0, t0 + n) on nh heads per row: qs [T * nh, 128] int8 and ws [T * nh] float32 from `head_rows` (T a multiple
    of K = 32 // nh rows), ki [Tc, 128], ke [Tc]. K rows per threadgroup, one tile read of the keys for all of them."""
    Tc = ki.shape[0]
    K = 32 // nh
    g0, g1 = t0 // K, -(-(t0 + n) // K)
    kk = "bmi8x%d" % nh
    k = _kern.get(kk)
    if k is None:
        k = _kern[kk] = mx.fast.metal_kernel(name="kyojin_pfflat_bmax_i8x%d" % nh, header=NX.HEADER_NAX, input_names=["Q", "K", "w", "E", "prm"], output_names=["S"],
                                             source=i8_rows_source(nh))
    tpt = max(I8_TPT, 1)
    prm = mx.array([Tc, start, ratio, g0 * K, 0, tpt] + [0] * 10, dtype=mx.uint32)
    out = k(inputs=[qs[g0 * 32:g1 * 32], ki, ws[g0 * 32:g1 * 32], ke, prm], grid=((g1 - g0) * 128, -(-Tc // (128 * tpt)), 1), threadgroup=(128, 1, 1),
            output_shapes=[((g1 - g0) * K, -(-Tc // BS))], output_dtypes=[mx.float32])[0]
    return out[t0 - g0 * K:t0 - g0 * K + n]


def heads_pick(pick: int) -> int:
    """Blocks the approximate first pass keeps: pf_idx_heads_m when that is wider than the exact path's own count (0 = keep the exact path's count)."""
    return max(pick, HEADS_M) if HEADS_M > 0 else pick


# ----------------------------------------------------------------------------------------------------------------- ranking of the block maxima
def _blk_topk(bm, lens, k: int):
    """Exact top-k columns t < lens[u] of bm [N, nb], int32 [N, k] ascending with a -1 tail (value desc, lower index first on ties): idxfast.topk_mt on the GPU,
    the stock sort(argsort(-bm)) chain without it."""
    if METAL_OPT:
        from . import idxfast as IX
        return IX.topk_mt(bm, lens.astype(mx.int32), k)
    cols = mx.arange(bm.shape[1])[None, :]
    sc = mx.where(cols < lens[:, None], bm, -mx.inf)
    ids = mx.sort(mx.argsort(-sc, axis=-1)[:, :k], axis=-1).astype(mx.int32)
    return mx.where(ids < lens[:, None], ids, -1)


def pick_blocks(bm, clen, k: int, force_last: bool):
    """The k best 8-key blocks of each row from their maxima bm [N, nb]; clen [N] = causal key count of the row. Returns int32 [N, k] ascending block ids, -1 = none.
    force_last: the candidate-source ranking (select_candidate_blocks): the last visible block ranks first whatever its maximum, and a block whose maximum is -inf
    is not kept."""
    nb = bm.shape[1]
    blens = mx.minimum((clen + (BS - 1)) // BS, nb).astype(mx.int32)
    if force_last:
        last = ((clen - 1) // BS)[:, None]
        bm = mx.where(mx.arange(nb)[None, :] == last, mx.inf, bm)
    ids = _blk_topk(bm, blens, k)
    if force_last:
        fin = mx.take_along_axis(bm, mx.maximum(ids, 0), axis=1) > -mx.inf
        ids = mx.where((ids >= 0) & fin, ids, -1)
    return ids


def ids_from_mask(flag, k: int):
    """Kept block ids of each row from a block flag [r, nblk] bool: ascending, at most k per row (the first k, as `argsort(~flag)[:k]` keeps), -1 padding.
    One cumsum and one scatter instead of a sort of every row."""
    r, nblk = flag.shape
    rank = mx.cumsum(flag.astype(mx.int32), axis=1) - 1
    dest = mx.where(flag & (rank < k), rank, k)                                  # everything else lands in the spill column k
    out = mx.full((r, k + 1), -1, dtype=mx.int32)
    out = mx.put_along_axis(out, dest, mx.broadcast_to(mx.arange(nblk, dtype=mx.int32), (r, nblk)), axis=1)
    return out[:, :k]


def mask_from_ids(ids, nblk: int, Tc: int):
    """bool [r, Tc] key mask of block ids [r, K] (-1 = none): what the stock candidate mask holds (the fallback when a consumer cannot use the ids)."""
    r, K = ids.shape
    flag = mx.zeros((r, nblk + 1), dtype=mx.bool_)
    flag = mx.put_along_axis(flag, mx.where(ids >= 0, ids, nblk), mx.ones((r, K), dtype=mx.bool_), axis=1)[:, :nblk]
    return mx.repeat(flag, BS, axis=1)[:, :Tc]


# ----------------------------------------------------------------------------------------------------------------- rescoring of picked blocks
_GATHERED_POST = """
    // M [R, 32, C] half (q . gathered keys), w [R, 32] half, pos [R, C] uint (absolute key column, 0xFFFFFFFF = none) -> S [R, C] float32: sum_h w * relu(M) in order, causal pos < clen
    const uint n = thread_position_in_grid.x, t = thread_position_in_grid.y;
    const uint C = prm[0], t0 = prm[1], start = prm[2], ratio = prm[3];
    if (n >= C) return;
    const uint clen = (start + t0 + t + 1) / ratio;
    float acc = 0.0f;
    const device half* mp = M + ulong(t) * 32 * C + n;
    for (uint h = 0; h < 32; ++h) acc += max(float(mp[ulong(h) * C]), 0.0f) * float(w[ulong(t) * 32 + h]);
    S[ulong(t) * C + n] = pos[ulong(t) * C + n] < clen ? acc : -INFINITY;
"""

_GSCORE = """
    // rescoring of picked 8-key blocks: one query row per threadgroup (NSG simdgroups), 4 blocks (32 keys) per simdgroup tile, staged from the key array into
    // threadgroup memory, then the same 32 heads x 32 keys matmul2d / half rounding / head order / causal mask as the fused score kernel.
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.x;
    const uint Tc = prm[0], start = prm[1], ratio = prm[2], t0 = prm[3], KB = prm[4], TPT = prm[5];
    threadgroup half Qs[32 * 128];
    threadgroup float Wh[32];
    threadgroup half stage[NSG][32 * 128];                              // a simdgroup's 32 staged keys; the same bytes later hold its [32 keys, 33] float score tile
    for (uint i = tid; i < 32u * 128u; i += NSG * 32u) Qs[i] = Q[ulong(t) * 32 * 128 + i];
    if (tid < 32u) Wh[tid] = float(w[ulong(t) * 32 + tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint clen = (start + t0 + t + 1) / ratio;
    constexpr auto desc = matmul2d_descriptor(32, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tQ((threadgroup half*)Qs, dextents<int32_t, 2>(128, 32));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tK((threadgroup half*)stage[sg], dextents<int32_t, 2>(128, 32));
    threadgroup float* scr = (threadgroup float*)stage[sg];
    for (uint j = 0; j < TPT; ++j) {
        const uint tile = (threadgroup_position_in_grid.y * NSG + sg) * TPT + j;      // 4 blocks per tile
        if (tile * 4u >= KB) break;
        const uint b4 = tile * 4u + (lane >> 3);
        const int id = b4 < KB ? ids[ulong(t) * KB + b4] : -1;
        const uint pos = uint(max(id, 0)) * 8u + (lane & 7u);
        const bool ok = id >= 0 && pos < Tc;
        const device uint4* src = (const device uint4*)(K + ulong(ok ? pos : 0u) * 128);
        threadgroup uint4* dst = (threadgroup uint4*)(stage[sg] + lane * 128);
        for (uint c = 0; c < 16; ++c) dst[c] = ok ? src[c] : uint4(0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto cT = op.get_destination_cooperative_tensor<decltype(tK), decltype(tQ), float>();
        op.run(tK, tQ, cT);
        simdgroup_barrier(mem_flags::mem_threadgroup);                                // every lane has read the staged keys before the tile overwrites them
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                const auto ix = cT.get_multidimensional_index(i);
                scr[uint(ix[1]) * 33 + uint(ix[0])] = float(half(cT[i]));
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f;
        for (uint h = 0; h < 32; ++h) s += metal::max(scr[lane * 33 + h], 0.0f) * Wh[h];
        S[ulong(t) * KB * 8u + tile * 32u + lane] = (ok && pos < clen) ? s : -INFINITY;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
GS_NSG = 2
GS_TPT = 4


def _positions(ids, bs: int = BS):
    """uint32 [r, K * bs] absolute key column of every gathered slot (0xFFFFFFFF where the block id is -1)."""
    r, K = ids.shape
    offs = mx.arange(bs, dtype=mx.uint32)
    return mx.where((ids >= 0)[..., None], ids.astype(mx.uint32)[..., None] * bs + offs, mx.array(0xFFFFFFFF, dtype=mx.uint32)).reshape(r, K * bs)


def _gathered_ref(q, keys, w, ids, start: int, ratio: int, t0: int):
    """numpy scores of the gathered slots in the order of the fused kernels (tests, no Metal): [r, K * 8] float32."""
    r, K = ids.shape
    qq, kk, ww = np.array(q[0, t0:t0 + r]).astype(np.float32), np.array(keys[0]), np.array(w[0, t0:t0 + r])
    Tc = kk.shape[0]
    pos = np.array(_positions(ids)).astype(np.int64)
    ok = pos < Tc
    kg = kk[np.where(ok, pos, 0)].astype(np.float32)                                # [r, C, 128]
    prod = np.einsum("rhd,rcd->rhc", qq, kg).astype(np.float16).astype(np.float32)
    acc = np.zeros((r, pos.shape[1]), dtype=np.float32)
    for h in range(qq.shape[1]):
        acc = (acc + np.maximum(prod[:, h], 0.0) * ww[:, h:h + 1].astype(np.float32)).astype(np.float32)
    clen = ((start + t0 + np.arange(r) + 1) // ratio)[:, None]
    return mx.array(np.where(ok & (pos < clen), acc, -np.inf).astype(np.float32))


def _gscore_kernel(q, keys, w, ids, start: int, ratio: int, t0: int):
    from . import ds_pf_idx as PI
    r, K = ids.shape
    if K % 4:
        ids = mx.pad(ids, [(0, 0), (0, 4 - K % 4)], constant_values=-1)
        K = ids.shape[1]
    k = _kern.get("gs")
    if k is None:
        k = _kern["gs"] = mx.fast.metal_kernel(name="kyojin_pfflat_gscore", header=NX.HEADER_NAX, input_names=["Q", "K", "w", "ids", "prm"], output_names=["S"],
                                              source=_GSCORE.replace("NSG", str(GS_NSG)))
    prm = mx.array([keys.shape[1], start, ratio, t0, K, GS_TPT] + [0] * 10, dtype=mx.uint32)
    return k(inputs=[q[0][t0:t0 + r].reshape(-1, 128), keys[0], w[0][t0:t0 + r], ids, prm],
             grid=(r * 32 * GS_NSG, -(-(K // 4) // (GS_NSG * GS_TPT)), 1), threadgroup=(32 * GS_NSG, 1, 1), output_shapes=[(r, K * 8)], output_dtypes=[mx.float32])[0]


def gathered_scores(q, keys, kb, w, ids, start: int, ratio: int, t0: int, max_elems: int = 1 << 26):
    """[r, K * 8] float32 scores of the keys of the blocks `ids` [r, K] of each row (rows t0 .. t0 + r), -inf for none / not yet visible; column j is the key
    8 * ids[j // 8] + j % 8. kb: keys [nblk, 8, 128] padded to whole blocks (`key_blocks`). Same values as the full-width scores at those columns."""
    r, K = ids.shape
    if not METAL_OPT:
        return _gathered_ref(q, keys, w, ids, start, ratio, t0)
    if GSCORE and "gs" not in _failed and fscore_ok(q, keys, w):
        try:
            out = _gscore_kernel(q, keys, w, ids, start, ratio, t0)
            HITS["gscore"] += 1
            return out[:, :K * BS]
        except Exception:                                    # the kernel did not build here: the gathered copy below gives the same scores
            _failed.add("gs")
            HITS["gscore_fallback"] += 1
    from . import ds_pf_idx as PI
    NH, D = q.shape[2], q.shape[3]
    C = K * BS
    k = PI._kern.get("g")
    if k is None:
        k = PI._kern["g"] = mx.fast.metal_kernel(name="kyojin_dsidx_post_g", input_names=["M", "w", "pos", "prm"], output_names=["S"], source=PI._POST_G)
    pos = _positions(ids)
    bt = max(1, min(r, max_elems // (NH * C), (1 << 26) // C))
    outs = []
    for a in range(0, r, bt):
        rr = min(bt, r - a)
        kg = kb[mx.maximum(ids[a:a + rr], 0)].reshape(rr, C, D)
        m = q[0, t0 + a:t0 + a + rr].reshape(rr, NH, D) @ kg.swapaxes(1, 2)
        prm = mx.array([C, t0 + a, start, ratio] + [0] * 12, dtype=mx.uint32)
        outs.append(k(inputs=[m, w[0, t0 + a:t0 + a + rr], pos[a:a + rr], prm], grid=(C, rr, 1), threadgroup=(min(C, 256), 1, 1),
                      output_shapes=[(rr, C)], output_dtypes=[mx.float32])[0])
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


def key_blocks(keys):
    """keys [1, Tc, 128] -> [nblk, 8, 128] (padded with zero rows to whole blocks; a view when Tc is a multiple of 8). Built once per layer, not per row block."""
    Tc = keys.shape[1]
    nblk = -(-Tc // BS)
    kk = keys[0] if Tc == nblk * BS else mx.pad(keys[0], [(0, nblk * BS - Tc), (0, 0)])
    return kk.reshape(nblk, BS, keys.shape[2])


def select_from_blocks(q, keys, kb, w, ids, start: int, ratio: int, t0: int, topk: int):
    """Exact top-k key positions of rows t0 .. t0 + r from the keys of the blocks `ids`: int32 [r, topk] ascending absolute positions. Ties go to the lower position
    (the gathered columns ascend and the argsort is stable). A column without a key never wins while the row has topk finite scores."""
    return _top_of_gathered(gathered_scores(q, keys, kb, w, ids, start, ratio, t0), ids, topk)


def _top_of_gathered(sc, ids, topk: int):
    pos = mx.minimum(_positions(ids), INVALID).astype(mx.int32)
    sel = mx.argsort(-sc, axis=-1)[:, :topk]
    ab = mx.take_along_axis(pos, sel, axis=-1)
    return mx.sort(ab, axis=-1)


# ----------------------------------------------------------------------------------------------------------------- entry points (called from ds_pf_idx._impl_indexer)
def bmax_ok(q, keys, w, Tc: int, topk: int, start: int, ratio: int, source: bool, cand_blocks: int = 0) -> bool:
    """The block-max path serves this layer: flag on, enough keys, every row sees at least `topk` keys (so no row has -inf filler picks), and for the source the
    fused gathered rescoring (a gathered copy of 16K keys per row costs about what the score traffic it removes does)."""
    if not (BMAX and Tc >= max(BMAX_MIN, 1) and Tc >= 8 * max(topk, cand_blocks if source else 0) and topk >= 1 and (start + 1) // ratio >= topk):
        return False
    if source and not (GSCORE and "gs" not in _failed and cand_blocks > topk):      # the top-k of the row lies in its top-k blocks; the forced last block takes one of the candidate slots
        return False
    return fscore_ok(q, keys, w) if METAL_OPT else True


def plain_topk(q, keys, w, start: int, ratio: int, topk: int, seqlen: int, key=None):
    """Top-k indices [seqlen, topk] int32 (ascending absolute key positions) of a plain indexer layer without any [rows, Tc] scores.
    pf_idx_heads: the first pass scores `heads_n()` heads of each row and the kept blocks are rescored on all 32 (exact scores, approximate candidate set)."""
    HITS["bmax_plain"] += 1
    nh = heads_n()
    out = _plain_topk(q, keys, w, start, ratio, topk, seqlen, nh)
    if nh and HEADS_CHECK:
        _recall(out, _plain_topk(q, keys, w, start, ratio, topk, seqlen, 0), "plain", key)
    return out


def _plain_topk(q, keys, w, start: int, ratio: int, topk: int, seqlen: int, nh: int = 0):
    """`plain_topk` with the head count explicit: nh = 0 is the exact all-head first pass (and what pf_idx_heads_check compares against)."""
    kb = key_blocks(keys)
    pick = topk
    sub = heads_sub(nh)
    sel = head_sel(q, w, nh) if sub else None                   # 32 heads stay in their natural order (same sum, same bits as the exact pass)
    if sub:
        HITS["heads_plain"] += 1
    if I8 and METAL_OPT and i8_on_grid(keys):                                    # packed per call: no state to keep in step with the keys (one launch per layer, the chunk's transient)
        HITS["i8_plain"] += 1
        ki, ke = i8pack(keys[0])
        qi, qe = i8pack(q[0].reshape(-1, q.shape[3]))
        w8 = w[0].astype(mx.float32) * qe.reshape(seqlen, q.shape[2])
        pick = topk + I8_MARGIN
        if sub:
            qs, ws = head_rows(qi, w8, sel, nh)
            pick = heads_pick(pick)

            def make(t0, n):
                return bmax_i8_rows(qs, ki, ws, ke, start, ratio, t0, n, nh)
        else:
            if nh:
                pick = heads_pick(pick)

            def make(t0, n):
                return bmax_i8(qi, ki, w8, ke, start, ratio, t0, n)
    else:
        if nh:
            HITS["heads_i8off"] += 1                                             # the packed subset kernel is the int8 first pass: this pass scores all 32 heads
            pick = heads_pick(pick)

        def make(t0, n):
            return bmax_scores(q, keys, w, start, ratio, t0, n, np.array(sel) if sub else None)

    def red(t0, bm):
        n = bm.shape[0]
        clen = (start + t0 + mx.arange(n) + 1) // ratio
        ids = pick_blocks(bm, clen, pick, False)
        return select_from_blocks(q, keys, kb, w, ids, start, ratio, t0, topk)

    out = IT.scan(make, seqlen, IT.qblock(seqlen), red)
    if CAPTURE and keys.shape[1] >= CAPTURE_MIN and seqlen >= 4096:
        _capture(q, keys, w, start, ratio, topk, out)
    return out


def _recall(got, ref, tag: str, key) -> None:
    """Top-k recall of an approximate pass: the share of the exact top-k positions the served indices contain, per layer under HITS heads_hit_l<layer> /
    heads_total_l<layer> and summed, plus one stderr line (what one served run reports on real text)."""
    g, r = np.array(got), np.array(ref)
    hit = 0
    for a in range(0, r.shape[0], 64):
        b = min(a + 64, r.shape[0])
        hit += int((g[a:b, :, None] == r[a:b, None, :]).any(-1).sum())
    tot = int(r.size)
    HITS["heads_recall_hit"] += hit
    HITS["heads_recall_total"] += tot
    if key is not None:
        HITS["heads_hit_l%s" % key] = HITS.get("heads_hit_l%s" % key, 0) + hit
        HITS["heads_total_l%s" % key] = HITS.get("heads_total_l%s" % key, 0) + tot
    print("[kyojin] pf_idx_heads: %s layer %s top-%d recall %.2f %% (%d rows x topk, %d heads, M=%d)"
          % (tag, key, r.shape[1], 100.0 * hit / tot if tot else 0.0, r.shape[0], heads_n(), HEADS_M), file=sys.stderr, flush=True)


def heads_recall_pct() -> float:
    """Recall of the sampled chunks in percent of topk (0 when pf_idx_heads_check is off or nothing ran)."""
    tot = HITS.get("heads_recall_total", 0)
    return 100.0 * HITS.get("heads_recall_hit", 0) / tot if tot else 0.0


def _capture(q, keys, w, start: int, ratio: int, topk: int, idxs) -> None:
    """Write one plain layer chunk for offline analysis: float16 q [T, 32, 128], w [T, 32], int32 top-k [T, topk], float16 keys [Tc, 128]."""
    d = CAPTURE_DIR
    os.makedirs(d, exist_ok=True)
    n = sum(f.startswith("idx_") for f in os.listdir(d))
    if n >= CAPTURE_MAX_FILES:
        return
    tag = f"T{keys.shape[1]}_s{start}_r{ratio}_{n:02d}"
    np.savez(os.path.join(d, f"idx_{tag}.npz"), q=np.array(q[0].astype(mx.float16)), w=np.array(w[0].astype(mx.float16)), idx=np.array(idxs), start=start, ratio=ratio, topk=topk)
    np.save(os.path.join(d, f"keys_{tag}.npy"), np.array(keys[0].astype(mx.float16)))


def source_topk(q, keys, w, start: int, ratio: int, topk: int, seqlen: int, cand_blocks: int, key=None):
    """Candidate source layer: (top-k indices [seqlen, topk] int32, kept block ids [seqlen, cand_blocks] int32) without any [rows, Tc] scores or mask.
    pf_idx_heads: the first pass scores `heads_n()` heads of each row, the kept blocks (and the top-k) come from their exact all-head scores."""
    HITS["bmax_source"] += 1
    nh = heads_n()
    out = _source_topk(q, keys, w, start, ratio, topk, seqlen, cand_blocks, nh)
    if nh and HEADS_CHECK:
        _recall(out[0], _source_topk(q, keys, w, start, ratio, topk, seqlen, cand_blocks, 0)[0], "source", key)
    return out


def _source_topk(q, keys, w, start: int, ratio: int, topk: int, seqlen: int, cand_blocks: int, nh: int = 0):
    """`source_topk` with the head count explicit: nh = 0 is the exact all-head first pass."""
    kb = key_blocks(keys)
    sub = heads_sub(nh)
    sel = head_sel(q, w, nh) if sub else None
    if sub:
        HITS["heads_source"] += 1

    def red(t0, bm):
        n = bm.shape[0]
        clen = (start + t0 + mx.arange(n) + 1) // ratio
        pick = cand_blocks if not nh else heads_pick(cand_blocks)
        ids = pick_blocks(bm, clen, pick, True)
        if pick == cand_blocks:                                 # the exact ranking: the picked blocks are the kept blocks
            return select_from_blocks(q, keys, kb, w, ids, start, ratio, t0, topk), ids
        sc = gathered_scores(q, keys, kb, w, ids, start, ratio, t0)   # the same two steps as the int8 source pass: exact scores of the wide set
        return _top_of_gathered(sc, ids, topk), rerank_blocks(sc, ids, clen, cand_blocks)

    if I8 and I8_SRC and METAL_OPT and i8_on_grid(keys):
        HITS["i8_source"] += 1
        make, red = _i8_source_pass(q, keys, kb, w, start, ratio, topk, seqlen, cand_blocks, nh, sel)
    else:
        if nh:
            HITS["heads_i8off"] += 1

        def make(t0, n):
            return bmax_scores(q, keys, w, start, ratio, t0, n, np.array(sel) if sub else None)
    out = IT.scan(make, seqlen, IT.qblock(seqlen), red)
    if CAPTURE and keys.shape[1] >= CAPTURE_MIN and seqlen >= 4096:
        _capture(q, keys, w, start, ratio, topk, out[0])
    return out



def rerank_blocks(sc, ids, clen, k: int):
    """The candidate blocks from the exact scores of more blocks than needed: sc [r, P * 8] (gathered_scores), ids [r, P] ascending, -1 = none. Returns int32 [r, k] ascending,
    -1 = none, by the ranking of `pick_blocks(force_last=True)`: the last visible block first, then the block maximum, ties to the lower block, none with a -inf maximum."""
    r, P = ids.shape
    bm = mx.max(sc.reshape(r, P, BS), axis=2)
    bm = mx.where(ids == ((clen - 1) // BS)[:, None], mx.inf, bm)
    cols = _blk_topk(bm, mx.full((r,), P, dtype=mx.int32), k)
    fin = mx.take_along_axis(bm, mx.maximum(cols, 0), axis=1) > -mx.inf
    return mx.where((cols >= 0) & fin, mx.take_along_axis(ids, mx.maximum(cols, 0), axis=1), -1)


def _i8_source_pass(q, keys, kb, w, start: int, ratio: int, topk: int, seqlen: int, cand_blocks: int, nh: int = 0, sel=None):
    """(make, red) of the candidate source on int8 operands. The first pass only names `cand_blocks + I8_SRC_MARGIN` blocks (pf_idx_heads: at least
    pf_idx_heads_m of them): the kept ones and the top-k come from their exact scores, so a kept set differs from the half path only when a rounded
    (row, head) pushes a block past the margin."""
    ki, ke = i8pack(keys[0])
    qi, qe = i8pack(q[0].reshape(-1, q.shape[3]))
    w8 = w[0].astype(mx.float32) * qe.reshape(seqlen, q.shape[2])
    pick = cand_blocks + I8_SRC_MARGIN
    if nh:
        pick = heads_pick(pick)
    if heads_sub(nh):
        qs, ws = head_rows(qi, w8, sel, nh)

        def make(t0, n):
            return bmax_i8_rows(qs, ki, ws, ke, start, ratio, t0, n, nh)
    else:
        def make(t0, n):
            return bmax_i8(qi, ki, w8, ke, start, ratio, t0, n)

    def red(t0, bm):
        clen = (start + t0 + mx.arange(bm.shape[0]) + 1) // ratio
        ids = pick_blocks(bm, clen, pick, True)
        sc = gathered_scores(q, keys, kb, w, ids, start, ratio, t0)
        return _top_of_gathered(sc, ids, topk), rerank_blocks(sc, ids, clen, cand_blocks)

    return make, red


def consumer_topk(q, keys, w, start: int, ratio: int, topk: int, seqlen: int, ids):
    """Candidate consumer layer: top-k indices [seqlen, topk] over the blocks the source kept (ids [seqlen, K], -1 = none)."""
    HITS["cand_ids_use"] += 1
    kb = key_blocks(keys)
    return IT.scan(lambda t0, n: (t0, n), seqlen, IT.qblock(seqlen),
                   lambda t0, tn: select_from_blocks(q, keys, kb, w, ids[tn[0]:tn[0] + tn[1]], start, ratio, tn[0], topk))


def make_ids(masks, k: int):
    """Kept block ids [T, k] from the stock candidate mask [T, Tc] bool (block granular): read once in the source layer so no consumer has to."""
    HITS["cand_ids_make"] += 1
    flag = masks[:, ::BS]
    rows = []
    for a in range(0, flag.shape[0], 1024):
        rows.append(ids_from_mask(flag[a:a + 1024], k))
    return rows[0] if len(rows) == 1 else mx.concatenate(rows, axis=0)


def stash(cache, ids, start: int, seqlen: int, Tc: int) -> None:
    cache.cand_ids = (ids, start, seqlen, Tc)


def fetch(cache, start: int, seqlen: int, Tc: int):
    """The ids the source layer of THIS chunk left on the cache (None when it left none or they belong to another chunk)."""
    e = getattr(cache, "cand_ids", None)
    return e[0] if e is not None and e[1:] == (start, seqlen, Tc) else None


def clear(cache) -> None:
    if getattr(cache, "cand_ids", None) is not None:
        cache.cand_ids = None
