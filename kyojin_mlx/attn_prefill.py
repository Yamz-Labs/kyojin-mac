# SPDX-License-Identifier: Apache-2.0
"""Prefill attention over gathered keys (DS4.1 sparse pooled attention: sliding window + top-k compressed positions, one shared K = V stream).

The stock `_sparse_pooled_attention` gathers [T, topk, D] pooled rows per chunk and scores the window against all T keys with a dense fp32 mask:
O(T^2) memory and flops (T = 2048: 10 GiB peak, ~500 ms per layer on the Air). Here every query has one key list of `W + topk` rows of a
table (window kv rows then pool rows, -1 = masked) and three kernels run per chunk of queries:
  K1 scores   S[t, h, j] = scale * q[h, t] . tab[ids[t, j]]     (q read in place, strided over heads) keys of a query gathered into threadgroup memory, NAX matmul2d (fp32 accumulate)
  K2 softmax  P[t, h, j] = exp(S - logaddexp(logsumexp_j S, sink[h]))   one simdgroup per (t, h) row, fp32
  K3 values   O[h, t, :] = sum_j P[t, h, j] tab[ids[t, j], :]    same gather, NAX matmul2d
The result matches the stock function up to the fp16 rounding of P and the order of the fp32 accumulation (rel 5e-4 on DS4.1 data).

E5 (KYOJIN_ATTN_PF_POOL): the table is not built. K1/K3 take the window rows and the pool rows as two inputs and pick the row of a key id from
`id < Tw` (the ids already separate them), so a chunk no longer copies the whole pool once per layer.
E6 (KYOJIN_ATTN_PF_LAZYMASK): the [T, T + W] window mask of the stock `_window_part` is not built when the gathered-key path runs (it is read only by
the dense fallback, which builds it on demand): the mask is 68 MB per layer at 8192 rows."""
from __future__ import annotations

from . import archs as _A
import functools
import itertools
import os

import mlx.core as mx

from . import fired as FD
from . import metal_nax as NX
from . import pf_ring as _RING
from . import pfnomirror as PNF

ENABLED = os.environ.get("KYOJIN_ATTN_PF", "auto")           # "1" force, "0" off, "auto" = on when the GPU has neural accelerators
MIN_ROWS = int(os.environ.get("KYOJIN_ATTN_PF_MIN_ROWS", "64"))     # queries in one call from which the kernels run (fewer: the stock code)
PIECES = os.environ.get("KYOJIN_ATTN_PF_PIECES", "1") != "0"        # wo_a reads the launch groups of the attention output in place (no concatenation)
CHUNK = int(os.environ.get("KYOJIN_ATTN_PF_CHUNK", "512"))          # queries per kernel launch group (S buffer = chunk * H * NK * 4 bytes)
POOLIN = os.environ.get("KYOJIN_ATTN_PF_POOL", "1") != "0"          # E5: window and pool as two kernel inputs (no concatenated table per layer)
LAZY_MASK = os.environ.get("KYOJIN_ATTN_PF_LAZYMASK", "1") != "0"   # E6: no [T, Tw] window mask when the gathered-key path runs
# pf_attn_online: K2 (softmax) folded into K1. K1 keeps the scores of its 128-key tile in threadgroup memory, takes the per-head
# max over those keys, writes P = exp(S - m_tile) fp16 (the S fp32 buffer and K2's read of it disappear) plus the per-tile m and
# l = sum exp(S - m_tile). K3 reads P and rescales each block by exp(m_tile - M) / Z, M = max(max_tile m, sink) and
# Z = sum_tile l exp(m_tile - M) + exp(sink - M), both computed from the small arrays at the top of the launch group. Reduction
# order differs from K2 (tile max instead of row max, per-thread sum instead of a simdgroup sum): same math, not bit-identical.
ONLINE = os.environ.get("KYOJIN_ATTN_PF_ONLINE", "1") != "0"

# E5: the row of a key id in the table: ids < Tw are window rows, the rest pool rows (the kernels already treat them apart). tab = concat([window, pool]).
_ROWEXPR = ("tab + ulong(id) * DD",
            "(id < 0 ? win : (id < int(TWEXP) ? win + ulong(id) * DD : pl + ulong(id - int(TWEXP)) * DD))")

_K1 = r"""
    const uint sg = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y;
    const uint nbase = threadgroup_position_in_grid.x * 128 + sg * 32;
    const int qrow = int(params[1]) * DD;                       // elements between two heads of q [H, T, D]
    constexpr auto desc = matmul2d_descriptor(32, 32, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(Q + ulong(params[0] + t) * DD), dextents<int32_t, 2>(DD, HH), array<int32_t, 2>{1, qrow});
    auto tA0 = tA.slice(0, 0);
    auto rB = op.get_right_input_cooperative_tensor<half, half, float>();
    using RA = mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>;
    using RB = mpp::tensor_ops::__tensor_ops_detail::__remove_addrspace_t<decltype(rB)>;
DECL
    // keys go straight from the table into the right-input cooperative tensor (no threadgroup staging, no barrier): the lane holds k' = 4 * G + c
    // of every 16-block for the key rows nb + 16 * tile + 8 * hi (element 4 * g + c, g = 4 * block + 2 * tile + hi; layout as in metal_nax.coop_source)
    const auto ix0 = rB.get_multidimensional_index(ushort(0));
    const uint G = uint(ix0[0]) >> 2, nb = uint(ix0[1]);
    int kid[4];
    for (uint s = 0; s < 4; ++s) { const uint n = nbase + nb + 16 * (s >> 1) + 8 * (s & 1); kid[s] = n < NK ? ids[ulong(t) * NK + n] : -1; }
    for (uint kt = 0; kt < DD; kt += 64) {
        for (uint g = 0; g < 16; ++g) {
            const int id = kid[g & 3];
            const device half* rp = ROWEXPR;
            const uint2 v = id >= 0 ? *((const device uint2*)(rp + kt + 16 * (g >> 2) + 4 * G)) : uint2(0);
            rB[4 * g + 0] = as_type<half2>(v.x).x; rB[4 * g + 1] = as_type<half2>(v.x).y; rB[4 * g + 2] = as_type<half2>(v.y).x; rB[4 * g + 3] = as_type<half2>(v.y).y;
        }
RUN
    }
STORE
"""


def _k1_source(H: int, D: int, NK: int, scale: float, poolin: bool = False) -> str:
    nmb = H // 32
    decl = "\n".join(f"    auto cT{m} = op.get_destination_cooperative_tensor<RA, RB, float>();\n"
                     f"    for (uint16_t i = 0; i < cT{m}.get_capacity(); ++i) if (cT{m}.is_valid_element(i)) cT{m}[i] = 0.0f;" for m in range(nmb))
    run = "\n".join(f"        {{ auto sA = tA.slice(int(kt), {32 * m}); op.run(sA, rB, cT{m}); }}" for m in range(nmb))
    store = "\n".join(f"    for (uint16_t i = 0; i < cT{m}.get_capacity(); ++i) if (cT{m}.is_valid_element(i)) {{\n"
                      f"        const auto ix = cT{m}.get_multidimensional_index(i);\n        const uint n = nbase + uint(ix[0]);\n"
                      f"        if (n < NK) S[(ulong(t) * HH + {32 * m}u + uint(ix[1])) * NK + n] = cT{m}[i] * SCALE;\n    }}" for m in range(nmb))
    return (_K1.replace("DECL", decl).replace("RUN", run).replace("STORE", store).replace("ROWEXPR", _ROWEXPR[poolin]).replace("TWEXP", "params[2]")
            .replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK)).replace("SCALE", repr(float(scale)) + "f"))


# pf_no_mirror: the row read of K1 / K3 (`_ROWEXPR[1]`) with the pool rows decoded from their fp4 codes in the same copy (pfnomirror.HEAD). The window
# rows stay dense (the ring is not packed); the pool row is `id - Tw` of the codes, guarded by the valid count `params[3]`. K1 holds four key ids per lane
# (`kid[0..3]`, one per 16-block), so the four block scales of each are read once per 64-dim step instead of once per read.
_LOAD4 = ("    for (uint kt = 0; kt < DD; kt += 64) {\n"
          "        for (uint g = 0; g < 16; ++g) {\n"
          "            const int id = kid[g & 3];\n"
          "            const device half* rp = " + _ROWEXPR[1] + ";\n"
          "            const uint2 v = id >= 0 ? *((const device uint2*)(rp + kt + 16 * (g >> 2) + 4 * G)) : uint2(0);")
_CODE4 = r"""    uint prow[4];
    bool pok[4];
    for (uint s = 0; s < 4; ++s) { const int id = kid[s]; pok[s] = id >= int(TWEXP) && ulong(id - int(TWEXP)) < ulong(@PN@); prow[s] = pok[s] ? uint(id - int(TWEXP)) : 0u; }
    for (uint kt = 0; kt < DD; kt += 64) {
        float psc[4][4];
        for (uint s = 0; s < 4; ++s) if (pok[s]) { const ulong sb = ulong(prow[s]) * (DD / 16) + (kt >> 4); for (uint b = 0; b < 4; ++b) psc[s][b] = pnm_e4m3(uint(poolsc[sb + b])); }
        for (uint g = 0; g < 16; ++g) {
            const int id = kid[g & 3];
            uint2 v = uint2(0);
            if (id >= 0) {
                const uint off = kt + 16 * (g >> 2) + 4 * G;
                if (id < int(TWEXP)) v = *((const device uint2*)(win + ulong(id) * DD + off));
                else if (pok[g & 3]) v = pnm_row4s(poolc, prow[g & 3], off, DD, psc[g & 3][g >> 2]);
            }"""
_LOAD8 = ("            const device half* rp = " + _ROWEXPR[1] + ";\n"
          "            *((threadgroup uint4*)(stage + r * 128 + o)) = id >= 0 ? *((const device uint4*)(rp + dbase + o)) : uint4(0);")
_CODE8 = r"""            if (id < 0) *((threadgroup uint4*)(stage + r * 128 + o)) = uint4(0);
            else if (id < int(TWEXP)) *((threadgroup uint4*)(stage + r * 128 + o)) = *((const device uint4*)(win + ulong(id) * DD + dbase + o));
            else { const ulong pr = ulong(id - int(TWEXP)); *((threadgroup uint4*)(stage + r * 128 + o)) = pr < ulong(@PN@) ? pnm_row8(poolc, poolsc, pr, dbase + o, DD) : uint4(0); }"""


def _codes(src: str, d: int) -> str:
    """`src` with its dense key-row read (K1's 4 halves or K3's 8 dims) replaced by the packed decode of the pool rows. The window row `Tw` is `params[1]` in
    K3 and `params[2]` in K1 / K3 online; the valid pool count is the param after it. Raises when the template moved."""
    for tw_tok, pn_tok in (("params[1]", "params[2]"), ("params[2]", "params[3]")):
        for dpl, cpl in ((_LOAD4, _CODE4), (_LOAD8, _CODE8)):
            dpl = dpl.replace("DD", str(d)).replace("TWEXP", tw_tok)
            if dpl in src:
                return src.replace(dpl, cpl.replace("DD", str(d)).replace("TWEXP", tw_tok).replace("@PN@", pn_tok))
    raise AssertionError("attn_prefill: the row read of _K1 / _K3 changed, update pf_no_mirror's snippets")


# pf_attn_online: the K1 epilogue that replaces the S store. One 16.5 KB threadgroup buffer (padded +1 float a row: a head's 128 keys
# then sit in different banks) holds the 128-key tile of one 32-head group; the four simdgroups of the threadgroup wrote their 32-key
# slices of it straight from the cooperative tensors, so the per-head max over the tile is a plain threadgroup reduction (4 threads a
# head, 32 keys each) with no second score pass and no S buffer. P is written coalesced (a thread owns 32 consecutive keys of one head).
_K1O_HEAD = r"""
    const uint tid = thread_index_in_threadgroup;
    threadgroup float sc[32 * 129];
    threadgroup float pmax[32 * 4];
    threadgroup float psum[32 * 4];
    const uint hh = tid & 31u, qq = tid >> 5;
    const uint k0 = threadgroup_position_in_grid.x * 128u + qq * 32u;
"""

_K1O_EPI = r"""
    {
        for (uint16_t i = 0; i < cT@M@.get_capacity(); ++i) if (cT@M@.is_valid_element(i)) {
            const auto ix = cT@M@.get_multidimensional_index(i);
            sc[uint(ix[1]) * 129u + simdgroup_index_in_threadgroup * 32u + uint(ix[0])] = cT@M@[i] * SCALE;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mm = -INFINITY;
        for (uint i = 0; i < 32u; ++i) {
            const uint kk = k0 + i;
            const int id = kk < @NK@u ? ids[ulong(t) * @NK@u + kk] : -1;
            if (id >= 0) mm = max(mm, sc[hh * 129u + qq * 32u + i]);
        }
        pmax[hh * 4u + qq] = mm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm = max(max(pmax[hh * 4u], pmax[hh * 4u + 1u]), max(pmax[hh * 4u + 2u], pmax[hh * 4u + 3u]));
        float ss = 0.0f;
        device half* pp = P + ulong(t) * @H@u * @NK@u + ulong(@OFF@u + hh) * @NK@u + k0;
        for (uint i = 0; i < 32u; ++i) {
            const uint kk = k0 + i;
            const int id = kk < @NK@u ? ids[ulong(t) * @NK@u + kk] : -1;
            const float e = id >= 0 ? metal::precise::exp(sc[hh * 129u + qq * 32u + i] - mm) : 0.0f;
            if (kk < @NK@u) pp[i] = half(e);
            ss += e;
        }
        psum[hh * 4u + qq] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qq == 0u) {
            const ulong row = ulong(t) * @H@u + @OFF@u + hh;
            MX[row * @NTL@u + threadgroup_position_in_grid.x] = mm;
            LX[row * @NTL@u + threadgroup_position_in_grid.x] = psum[hh * 4u] + psum[hh * 4u + 1u] + psum[hh * 4u + 2u] + psum[hh * 4u + 3u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def _k1o_source(H: int, D: int, NK: int, scale: float, poolin: bool = False) -> str:
    nmb = H // 32
    ntl = -(-NK // 128)
    decl = "\n".join(f"    auto cT{m} = op.get_destination_cooperative_tensor<RA, RB, float>();\n"
                     f"    for (uint16_t i = 0; i < cT{m}.get_capacity(); ++i) if (cT{m}.is_valid_element(i)) cT{m}[i] = 0.0f;" for m in range(nmb))
    run = "\n".join(f"        {{ auto sA = tA.slice(int(kt), {32 * m}); op.run(sA, rB, cT{m}); }}" for m in range(nmb))
    store = _K1O_HEAD + "".join(_K1O_EPI.replace("@M@", str(m)).replace("@OFF@", str(32 * m)).replace("@NTL@", str(ntl)).replace("@NK@", str(NK))
                                .replace("@H@", str(H)) for m in range(nmb))
    return (_K1.replace("DECL", decl).replace("RUN", run).replace("STORE", store).replace("ROWEXPR", _ROWEXPR[poolin]).replace("TWEXP", "params[2]")
            .replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK)).replace("SCALE", repr(float(scale)) + "f"))


# pf_attn_online: K3 over P written by the online K1. The scores of a tile are already exp(S - m_tile), so a block's contribution is
# exp(m_tile - M) / Z times P @ V with M and Z over the tiles of the row (plus the sink). P is staged into threadgroup memory
# (32 keys x H heads) with that factor applied; matmul2d takes its left input from threadgroup memory (same tile, same k order).
_K3O = r"""
    const uint sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y, TC = params[0], NTL = @NTL@u;
    const uint dbase = threadgroup_position_in_grid.x * 128;
    threadgroup half stage[32 * 128];
    threadgroup half astage[32 * HH];
    threadgroup int sid[NK];
    threadgroup float fac[HH * @NTL@];
    for (uint j = tid; j < NK; j += 128) sid[j] = ids[ulong(t) * NK + j];
    if (tid < HH) {
        float M = sinks[tid];
        for (uint j = 0; j < NTL; ++j) M = max(M, MX[(ulong(t) * HH + tid) * NTL + j]);
        float Z = metal::precise::exp(sinks[tid] - M);
        for (uint j = 0; j < NTL; ++j) Z += LX[(ulong(t) * HH + tid) * NTL + j] * metal::precise::exp(MX[(ulong(t) * HH + tid) * NTL + j] - M);
        for (uint j = 0; j < NTL; ++j) fac[tid * NTL + j] = metal::precise::exp(MX[(ulong(t) * HH + tid) * NTL + j] - M) / Z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = matmul2d_descriptor(HH, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tAa((threadgroup half*)astage, dextents<int32_t, 2>(32, HH));
    auto sA0 = tAa.slice(0, 0);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tBa((threadgroup half*)stage, dextents<int32_t, 2>(128, 32));
    auto sB0 = tBa.slice(sg * 32, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kt = 0; kt < NK; kt += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 4; ++i) {
            const uint c = tid + 128 * i, r = c >> 4, o = (c & 15u) * 8u;
            const int id = sid[kt + r];
            const device half* rp = ROWEXPR;
            *((threadgroup uint4*)(stage + r * 128 + o)) = id >= 0 ? *((const device uint4*)(rp + dbase + o)) : uint4(0);
        }
        const uint jt = kt >> 7;
        const uint hd = tid >> 2u, q8 = tid & 3u;
        for (uint r = 0; r < HH / 32u; ++r) {
            const uint h2 = hd + 32u * r;
            const uint4 v = *((const device uint4*)(P + (ulong(t) * HH + h2) * NK + kt + q8 * 8u));
            const float f = fac[h2 * NTL + jt];
            const half2* s2 = (const half2*)&v;
            threadgroup half2* d2 = (threadgroup half2*)(astage + h2 * 32u + q8 * 8u);
            for (uint j = 0; j < 4; ++j) d2[j] = half2(float(s2[j].x) * f, float(s2[j].y) * f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(sA0, sB0, cT);
    }
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
        const auto ix = cT.get_multidimensional_index(i);
        O[(ulong(uint(ix[1])) * TC + t) * DD + dbase + sg * 32 + uint(ix[0])] = half(cT[i]);
    }
"""



_K2 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    const uint t = row / HH, h = row % HH;
    const device float* sp = S + ulong(row) * NK;
    const device int* ip = ids + ulong(t) * NK;
    float v[NK / 32];
    float m = -INFINITY;
    for (uint i = 0; i < NK / 32; ++i) { const uint j = lane + 32 * i; v[i] = ip[j] >= 0 ? sp[j] : -INFINITY; m = max(m, v[i]); }
    m = simd_max(m);
    float sum = 0.0f;
    for (uint i = 0; i < NK / 32; ++i) sum += metal::precise::exp(v[i] - m);
    sum = simd_sum(sum);
    const float lse = m + metal::precise::log(sum), sk = sinks[h];
    const float norm = max(lse, sk) + metal::precise::log(1.0f + metal::precise::exp(-fabs(lse - sk)));
    device half* pp = P + ulong(row) * NK;
    for (uint i = 0; i < NK / 32; ++i) pp[lane + 32 * i] = half(metal::precise::exp(v[i] - norm));
"""

_K3 = r"""
    const uint sg = simdgroup_index_in_threadgroup, tid = thread_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y, TC = params[0];
    const uint dbase = threadgroup_position_in_grid.x * 128;
    threadgroup half stage[32 * 128];
    threadgroup int sid[NK];
    for (uint j = tid; j < NK; j += 128) sid[j] = ids[ulong(t) * NK + j];
    constexpr auto desc = matmul2d_descriptor(HH, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)(P + ulong(t) * HH * NK), dextents<int32_t, 2>(NK, HH));
    auto sA0 = tA.slice(0, 0);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tBa((threadgroup half*)stage, dextents<int32_t, 2>(128, 32));
    auto sB0 = tBa.slice(sg * 32, 0);
    auto cT = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sB0), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kt = 0; kt < NK; kt += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 4; ++i) {
            const uint c = tid + 128 * i, r = c >> 4, o = (c & 15u) * 8u;
            const int id = sid[kt + r];
            const device half* rp = ROWEXPR;
            *((threadgroup uint4*)(stage + r * 128 + o)) = id >= 0 ? *((const device uint4*)(rp + dbase + o)) : uint4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(int(kt), 0);
        op.run(sA, sB0, cT);
    }
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) {
        const auto ix = cT.get_multidimensional_index(i);
        O[(ulong(uint(ix[1])) * TC + t) * DD + dbase + sg * 32 + uint(ix[0])] = half(cT[i]);
    }
"""


@functools.lru_cache(maxsize=None)
def _kernels(H: int, D: int, NK: int, scale: float, poolin: bool = False, packed: bool = False):
    def sub(s):
        return (s.replace("ROWEXPR", _ROWEXPR[poolin]).replace("TWEXP", "params[1]")
                .replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK)).replace("SCALE", repr(float(scale)) + "f"))
    uid = f"{H}_{D}_{NK}" + ("_pi" if poolin else "") + ("_pc" if packed else "")
    head = NX.HEADER_NAX + PNF.HEAD if packed else NX.HEADER_NAX
    if packed:
        names, n3 = ["Q", "win", "poolc", "poolsc", "ids", "params"], ["P", "win", "poolc", "poolsc", "ids", "params"]
    elif poolin:
        names, n3 = ["Q", "win", "pl", "ids", "params"], ["P", "win", "pl", "ids", "params"]
    else:
        names, n3 = ["Q", "tab", "ids", "params"], ["P", "tab", "ids", "params"]
    k1 = mx.fast.metal_kernel(name=f"kyojin_ap_s_{uid}", header=head, input_names=names, output_names=["S"],
                              source=_codes(_k1_source(H, D, NK, scale, poolin), D) if packed else _k1_source(H, D, NK, scale, poolin))
    k2 = mx.fast.metal_kernel(name=f"kyojin_ap_p_{uid}", header=NX.HEADER_NAX, input_names=["S", "ids", "sinks"], output_names=["P"], source=sub(_K2))
    k3 = mx.fast.metal_kernel(name=f"kyojin_ap_o_{uid}", header=head, input_names=n3, output_names=["O"],
                              source=_codes(sub(_K3), D) if packed else sub(_K3))
    return k1, k2, k3


@functools.lru_cache(maxsize=None)
def _kernels_online(H: int, D: int, NK: int, scale: float, poolin: bool = False, packed: bool = False):
    """pf_attn_online: K1 writes P (and the per-tile max / sum), K3 rescales; K2 does not exist."""
    def sub(s):
        return (s.replace("ROWEXPR", _ROWEXPR[poolin]).replace("TWEXP", "params[2]").replace("@NTL@", str(-(-NK // 128)))
                .replace("HH", str(H)).replace("DD", str(D)).replace("NK", str(NK)).replace("SCALE", repr(float(scale)) + "f"))
    uid = f"{H}_{D}_{NK}" + ("_pi" if poolin else "") + ("_pc" if packed else "")
    head = NX.HEADER_NAX + PNF.HEAD if packed else NX.HEADER_NAX
    k1 = mx.fast.metal_kernel(name=f"kyojin_ap_s_{uid}_on", header=head,
                              input_names=["Q", "win", "poolc", "poolsc", "ids", "params"] if packed else (["Q", "win", "pl", "ids", "params"] if poolin else ["Q", "tab", "ids", "params"]),
                              output_names=["P", "MX", "LX"],
                              source=_codes(_k1o_source(H, D, NK, scale, poolin), D) if packed else _k1o_source(H, D, NK, scale, poolin))
    k3 = mx.fast.metal_kernel(name=f"kyojin_ap_o_{uid}_on", header=head,
                              input_names=["P", "MX", "LX", "sinks"] + (["win", "poolc", "poolsc", "ids", "params"] if packed else
                                                                        (["win", "pl", "ids", "params"] if poolin else ["tab", "ids", "params"])),
                              output_names=["O"], source=_codes(sub(_K3O), D) if packed else sub(_K3O))
    return k1, k3


def available() -> bool:
    return NX.site_on(ENABLED)


def supported(q: mx.array, window: mx.array, pool: mx.array) -> bool:
    B, H, T, D = q.shape
    return (B == 1 and q.dtype == mx.float16 and window.dtype == mx.float16 and pool.dtype == mx.float16 and H in (32, 64) and D % 128 == 0
            and T >= MIN_ROWS and window.shape[0] == 1 and available())


def key_ids(T: int, Tw: int, idxs: mx.array, W: int) -> mx.array:
    """int32 [T, NK] rows of the table [window kv (Tw rows, the last T are this chunk's); pool]: the W window keys of every query, then its top-k pooled
    keys; -1 = no key. NK is padded to a multiple of 32."""
    hist = Tw - T
    r = hist + mx.arange(T)[:, None] - (W - 1) + mx.arange(W)[None, :]
    loc = mx.where(r >= 0, r, -1)
    pl = mx.where(idxs[0] >= 0, idxs[0] + Tw, -1)
    ids = mx.concatenate([loc, pl], axis=-1).astype(mx.int32)
    pad = -ids.shape[-1] % 32
    return mx.concatenate([ids, mx.full((T, pad), -1, dtype=mx.int32)], axis=-1) if pad else ids


_ids_memo: list = []          # (idxs, T, Tw, W, ids) of the last calls: layers that reuse the top-k of a source layer share its key table


def win_mask(win: int, start_pos: int, seqlen: int, Tw: int):
    """The vendor `_window_part` mask [1, 1, T, Tw]: window row j (absolute position max(0, start_pos - win + 1) + j) is visible to query i (absolute
    start_pos + i) when it is at or before it and inside the window."""
    positions = mx.arange(Tw) + max(0, start_pos - win + 1)
    queries = mx.arange(start_pos, start_pos + seqlen)
    return ((positions[None, :] <= queries[:, None]) & (positions[None, :] > queries[:, None] - win))[None, None]


def window_part(self, x, start_pos: int, cache, need_mask: bool = True):
    """E6: the vendor `DeepseekV41Attention._window_part` with the [T, Tw] mask built only when the caller reads it (the gathered-key kernels take the window
    rows of every query from its key list: the mask is 68 MB per layer at 8192 rows and never read). Same ring buffer update and same `part`."""
    import importlib
    L = _A.lang()
    batch, seqlen = x.shape[0], x.shape[1]
    win = self.window_size
    kv = self.kv_norm(self.wkv(L.fake_quant_fp8_ue8m0(x)))
    kv = self.rope(kv[:, None], start_pos).reshape(batch, seqlen, self.head_dim)
    kv = L.fake_quant_fp8_ue8m0(kv)
    buffer = cache.window[self.layer_idx]
    if buffer is None:
        buffer = mx.zeros((batch, win, self.head_dim), dtype=kv.dtype)
    elif buffer.shape[0] < batch:
        buffer = mx.pad(buffer, [(0, batch - buffer.shape[0]), (0, 0), (0, 0)])
    base = max(0, start_pos - win + 1)
    end_pos = start_pos + seqlen
    retained = min(seqlen, win)
    if _RING.ENABLED:                                                       # pf_win_slice: slices instead of a Gather and a Scatter, same rows
        part = mx.concatenate([_RING.read(buffer, batch, base, start_pos, win), kv], axis=1) if start_pos else kv
        buffer = _RING.write(buffer, batch, kv[:, -retained:], end_pos, win)
    else:
        if start_pos:
            history = mx.take(buffer[:batch], mx.arange(base, start_pos) % win, axis=1)
            part = mx.concatenate([history, kv], axis=1)
        else:
            part = kv
        buffer[:batch, mx.arange(end_pos - retained, end_pos) % win] = kv[:, -retained:]
    cache.window[self.layer_idx] = buffer
    return part, (win_mask(win, start_pos, seqlen, part.shape[1]) if need_mask else None)


def window_for(attn, x, start_pos: int, cache):
    """(window kv, window mask or None) of a prompt chunk: E6's mask-free `window_part`, unless the cache says the positions before `start_pos` were skipped (CED tail:
    ced._window_part restarts the window at the tail through the overridden method, and `window_part` would read the ring's zero or stale history as valid keys)."""
    if LAZY_MASK and not getattr(cache, "_ced_nohist", False):
        FD.hit("pf_attn_lazymask")
        return window_part(attn, x, start_pos, cache, need_mask=False)
    if LAZY_MASK:
        FD.hit("pf_attn_lazymask_ced")
    return attn._window_part(x, start_pos, cache)


def sparse_attention(q: mx.array, window: mx.array, pool: mx.array, idxs: mx.array, scale: float, sinks: mx.array, W: int, chunks: bool = False):
    """q [1, H, T, D] fp16, window [1, Tw, D], pool [1, Np, D] (fp16), idxs [1, T, topk] (-1 = none), sinks [H] -> [1, H, T, D] fp16.
    chunks: with more than one launch group return the list of the groups' outputs ([H, rows, D] each, at most 8) instead of their concatenation (a 268 MB copy at T = 4096);
    the consumer (pf_glue.wo_a_prompt) reads them in place."""
    _, H, T, D = q.shape
    Tw = window.shape[1]
    ids = None
    for e in _ids_memo:
        if e[0] is idxs and e[1:4] == (T, Tw, W):
            ids = e[4]
    if ids is None:
        ids = key_ids(T, Tw, idxs, W)
        _ids_memo.append((idxs, T, Tw, W, ids))
        del _ids_memo[:-2]
    NK = ids.shape[-1]
    q0 = mx.contiguous(q[0])                                          # [H, T, D]
    tw = window.shape[1]
    if POOLIN:
        FD.hit("pf_attn_pool")
    pc = PNF.pool_codes(pool) if POOLIN else None                     # pf_no_mirror: the pool rows of its codes, no dense table
    if pc is not None:
        tab = [window[0], pc[0], pc[1]]
        PNF.HITS["attention"] += 1
    else:
        tab = [window[0], pool[0]] if POOLIN else [mx.concatenate([window[0], pool[0]], axis=0)]     # E5: no copy of the pool per layer
    sk = sinks.astype(mx.float32)
    outs = []
    cs = max(CHUNK, -(-T // 8)) if chunks else CHUNK
    ntl = -(-NK // 128)
    if ONLINE:
        k1, k3 = _kernels_online(H, D, NK, scale, POOLIN, pc is not None)
        FD.hit("pf_attn_online")
        for a in range(0, T, cs):
            tc = min(cs, T - a)
            ic = ids[a:a + tc]
            prm = mx.array([a, T, tw] + ([pc[2]] if pc is not None else []), dtype=mx.uint32)
            P, MX, LX = k1(inputs=[q0] + tab + [ic, prm], grid=(-(-NK // 128) * 128, tc, 1), threadgroup=(128, 1, 1),
                           output_shapes=[(tc * H, NK), (tc * H, ntl), (tc * H, ntl)], output_dtypes=[mx.float16, mx.float32, mx.float32])
            prm3 = mx.array([tc, ntl, tw] + ([pc[2]] if pc is not None else []), dtype=mx.uint32)
            outs.append(k3(inputs=[P, MX, LX, sk] + tab + [ic, prm3], grid=(D // 128 * 128, tc, 1), threadgroup=(128, 1, 1),
                           output_shapes=[(H, tc, D)], output_dtypes=[mx.float16])[0])
    else:
        k1, k2, k3 = _kernels(H, D, NK, scale, POOLIN, pc is not None)
        for a in range(0, T, cs):
            tc = min(cs, T - a)
            ic = ids[a:a + tc]
            prm = mx.array([a, T, tw] + ([pc[2]] if pc is not None else []), dtype=mx.uint32)
            S = k1(inputs=[q0] + tab + [ic, prm], grid=(-(-NK // 128) * 128, tc, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float32])[0]
            P = k2(inputs=[S, ic, sk], grid=(tc * H * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(tc * H, NK)], output_dtypes=[mx.float16])[0]
            prm3 = mx.array([tc, tw] + ([pc[2]] if pc is not None else []), dtype=mx.uint32)
            outs.append(k3(inputs=[P] + tab + [ic, prm3], grid=(D // 128 * 128, tc, 1), threadgroup=(128, 1, 1),
                           output_shapes=[(H, tc, D)], output_dtypes=[mx.float16])[0])
    if len(outs) > 1 and chunks:
        return outs
    return (outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1))[None]


def _compress_part_packed(self, x, qr, start_pos, cache):
    """`_compress_part` of a prompt chunk under pf_no_mirror: the pool layer of the cache reads back as its own codes (`PackedSlots.lazy`), an append takes no
    read-back (`defer_dense`) and a mirror an earlier chunk built is dropped, so the chunk builds no dense pool. Flag off / an unpacked layer: the plain call."""
    with PNF.prefill_pool(cache, self.layer_idx):
        return self._compress_part(x, qr, start_pos, cache)


def _impl_sparse(orig):
    """Replacement of `DeepseekV41Attention._sparse_attention(q, window, pool, indices, window_mask)`: prompts (>= MIN_ROWS queries) run the kernels above."""
    def _sparse_attention(self, q, window, pool, indices, window_mask):
        if not supported(q, window, pool) or indices.ndim != 3 or indices.shape[1] != q.shape[2] or window.shape[1] < q.shape[2]:
            return orig(self, q, window, PNF.dense_of(pool), indices, window_mask)
        return sparse_attention(q, window, pool, indices, self.scale, self.attn_sink, self.window_size)
    return _sparse_attention


def _impl_attention_local(orig):
    """Replacement of `DeepseekV41Attention._attention` for prompts (>= MIN_ROWS rows, batch 1). Window-only layers (compress_ratio 0, 5 of the 43 DS4.1 blocks):
    the stock method scores the window with `scaled_dot_product_attention` over a [T, T] mask (T = 2048: 280 ms and 2 GiB of scores per layer), here the gathered-key
    kernels see the 128 window rows of each query. All layers: the activation copies between the GEMMs are single passes (pf_glue: q rope writing [H, T, D], inverse
    rope writing the wo_a input, wo_a output transposed and fp8-rounded in one pass). Same module calls in the same order as the stock method; anything else runs the stock code."""
    import importlib
    from . import pf_glue as PG
    L = _A.lang()

    def _attention(self, cache, x):
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < MIN_ROWS or not available():
            return orig(self, cache, x)
        start_pos, seqlen = cache.offset, x.shape[1]
        fq = L.fake_quant_fp8_ue8m0
        H, D = self.n_heads, self.head_dim
        qr = self.q_norm(self.wq_a(fq(x)))
        qin = fq(qr)
        wq_b, wo_a = PG.unwrap(self.wq_b), PG.unwrap(self.wo_a)             # the fused prompt kernels read the EXL3 linear under the rows wrappers
        if PG.wq_b_ok(wq_b, qin, H, D) and PG.supported(qin, D, self.rope.dims):
            glue, q, qb = True, PG.wq_b_rope(wq_b, qin, self.rope, start_pos, H, D), qin
        else:
            qb = self.wq_b(qin)
            glue = PG.ENABLED and PG.supported(qb, D, self.rope.dims) and qb.shape == (1, seqlen, H * D)
            q = PG.q_rope(qb, self.rope, start_pos, H, D) if glue else self.rope(qb.reshape(1, seqlen, H, D).transpose(0, 2, 1, 3), start_pos)
        window_kv, window_mask = window_for(self, x, start_pos, cache)

        def wm():                                                            # E6: the [T, Tw] mask, only built when a fallback reads it
            nonlocal window_mask
            if window_mask is None:
                window_mask = win_mask(self.window_size, start_pos, seqlen, window_kv.shape[1])
            return window_mask

        out = None
        pieces = PIECES and glue and PG.wo_a_ok(wo_a, q)                    # the attention output goes to wo_a: per launch group, read in place
        if self.compress_ratio:
            pool, idxs = _compress_part_packed(self, x, qr, start_pos, cache)  # pf_no_mirror: no dense pool, the codes serve the gather
            if pool.shape[1]:
                if pieces and supported(q, window_kv, pool) and idxs.ndim == 3 and idxs.shape[1] == seqlen and window_kv.shape[1] >= seqlen:
                    out = sparse_attention(q, window_kv, pool, idxs, self.scale, self.attn_sink, self.window_size, chunks=True)
                else:
                    out = self._sparse_attention(q, window_kv, PNF.dense_of(pool), idxs, wm())
        elif supported(q, window_kv, window_kv) and window_kv.shape[1] >= seqlen:
            out = sparse_attention(q, window_kv, window_kv[:, :1], mx.zeros((1, seqlen, 0), dtype=mx.int32), self.scale, self.attn_sink, self.window_size, chunks=pieces)
        if out is None:
            kv = window_kv[:, None].astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(q, kv, kv, scale=self.scale, mask=wm(), sinks=self.attn_sink.astype(q.dtype))
        if isinstance(out, list):
            return self.wo_b(PG.wo_a_prompt(wo_a, out, self.rope, start_pos))
        if glue and out.shape == (1, H, seqlen, D) and PG.wo_a_ok(wo_a, out):
            return self.wo_b(PG.wo_a_prompt(wo_a, out, self.rope, start_pos))
        if glue and out.shape == (1, H, seqlen, D) and out.dtype == qb.dtype:
            out = PG.o_rope_inv(out, self.rope, start_pos, self.o_groups)
        else:
            out = self.rope(out, start_pos, inverse=True)
            out = out.reshape(1, self.o_groups, -1, seqlen, D)
            out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        if glue and out.ndim == 4 and out.dtype == qb.dtype and out.shape[-1] % 256 == 0:
            return self.wo_b(PG.fq_t(out))
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(fq(out))

    return _attention
