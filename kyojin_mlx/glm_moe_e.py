# SPDX-License-Identifier: Apache-2.0
"""G2e: second backend of the GLM-5.3 decode / verify routed experts (flag KYOJIN_GLM_MOE_E, default off), written for our NVFP4 expert layout
(16-value blocks, e4m3 scales, one global scale per expert, G2a register-fed units).

Adapted from oMLX (https://github.com/jundot/omlx), Apache-2.0:
  omlx/patches/fp4_expert_rows.py                              dedupe of the (row, expert) pairs inside the kernel (the threadgroup of the first pair that picked an expert
                                                               computes every pair that picked it, the others leave; no group launch) and, for the variant "omx", the lane layout
                                                               (a lane owns 16 inputs of every 512-wide block, 8 weight bytes + 1 scale byte, 4 rows per simdgroup, 2 simdgroups)
  omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py      moe_down_combine: the routing-weighted sum over the K slots (and the shared-expert add) inside the down launch,
                                                               so the [S, H] expert outputs never go back to memory and the combine launch disappears
  omlx/patches/moe_verify_gather.py                            (same idea as the first item: rows that picked one expert read it once)
No oMLX source text is copied: the Metal code below is ours (NVFP4 decode, fp16 roundings of glm_moe.combine).

Winner (flag KYOJIN_GLM_MOE_E, used by glm_moe._call): "g2m" = G2c's lane mapping and arithmetic (bit-identical outputs) in oMLX's launch structure.

Variants (module level `pair_glu(kind, ...)`, `down(kind, ...)`, `down_combine`):
    inl   G2c's register-fed unit (glm_moe_rows._HDR.unit_acc: 16 B + 2 scales per lane, all loads issued first) with the dedupe done in the kernel (no group launch)
    omx   oMLX's lane layout (uint2 + 1 scale byte per lane per 512-block, RPS rows per simdgroup), same dedupe
    dc    down + combine in one launch: a simdgroup per (output row, slot position), the K products of a token are summed from threadgroup memory in slot order,
          fp16 products / f32 sum / fp16 round / + shared as glm_moe.combine does (bit-identical to down (G2c) + combine)
"""
from __future__ import annotations

import functools
import os

import mlx.core as mx

from . import glm_moe_rows as R
from . import moe_scale_compact as MC
from . import kernels as KN

def cfg(T):
    """g2m (output rows per simdgroup, simdgroups per threadgroup): bench R1 1x4 108 us vs 2x2 114; R2-R4 2x2 = 2x4 = 1x4 within 2 % (2x2 best at R4). Env KYOJIN_GLM_MOE_E_CFG=RxN overrides."""
    v = os.environ.get("KYOJIN_GLM_MOE_E_CFG", "")
    if v:
        a, b = v.split("x")
        return int(a), int(b)
    return (1, 4) if T == 1 else (2, 2)

def down_cfg(T):
    """BP-MOE2: (rps, nsg, xv, iu) of the g2m down launch. Env KYOJIN_GLM_MOE_E_DOWN="RxN[+xv][+iu]" (default off = the shared cfg, plain loads); only for T == 1 (the shape it was measured at).
    Measured (synthetic chain, default buffering, span per block, 3-4 interleaved rounds): 2x4 / 2x2 / 1x4+iu -1.2 us of 124, every variant bit-identical; noise level +-1 us."""
    rps, nsg = cfg(T)
    v = os.environ.get("KYOJIN_GLM_MOE_E_DOWN", "")
    if not v or T != 1:
        return rps, nsg, 0, 0
    parts = v.split("+")
    a, b = parts[0].split("x")
    return int(a), int(b), int("xv" in parts[1:]), int("iu" in parts[1:])


ROWS2_UNROLL = os.environ.get("KYOJIN_MOE_ROWS2_UNROLL", "0") not in ("0", "")      # V2-VERIFY: g2m pair loops unrolled over TMAX for T > 1 rows (registers, not private memory); bit-identical; read per call


def rows_unroll(T: int) -> int:
    return int(ROWS2_UNROLL and T > 1)


TMAX = 4                                                # rows per block of the in-kernel dedupe (an expert is picked at most once per row)
S_MAX = 32                                              # slots one ballot covers (one lane per slot); also the simdgroups of the down_combine threadgroup (1024 threads)
S_MAX_WIDE = 64                                         # slots of the two-ballot variant (_wide): T * K above S_MAX, up to this


def ok(g, d, T, K) -> bool:
    """g2m eligibility: <= TMAX rows, T * K <= S_MAX_WIDE slots (one ballot lane per slot, two ballots above S_MAX), NVFP4 units of 1024 inputs, rows split evenly."""
    return T <= TMAX and T * K <= S_MAX_WIDE and g.k % 1024 == 0 and d.k % 1024 == 0 and g.n % 4 == 0 and d.n % 4 == 0


def on() -> bool:
    """Flag KYOJIN_GLM_MOE_E (read per call, only read when KYOJIN_GLM_MOE is on): the G2e expert backend (inl pair_glu + dc down_combine)."""
    return os.environ.get("KYOJIN_GLM_MOE_E", "0") not in ("0", "")


def _mt(d, scm):
    """The scale input(s) of a stacked expert DevPack: its expanded `meta`, or (scm, flag KYOJIN_MOE_SCALE_COMPACT at load) the seven compact arrays."""
    return MC.cm_list(d) if scm else [d.meta]


def _kern_s(kind, scm, key, **kw):
    """KN._kern; with scm the scale reads of the source go through the compact helper (new kernel name / key / inputs / header; scm = 0 leaves everything as it was)."""
    if scm:
        key = key + ("scm",)
        kw["name"] += "_scm"
        kw["input_names"] = MC.scm_names(kw["input_names"])
        kw["header"] = kw["header"] + MC.CSC_HDR
        kw["source"] = MC.scm_source(kw["source"], kind)
    return KN._kern(key, **kw)


_PRO = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tgi = threadgroup_position_in_grid.x;
    const uint p = threadgroup_position_in_grid.y;
    uint mm = 1u << p;                                                 // pairs of this expert at or after p (bit q = slot q)
    uint ex;
    if (SING) {
      ex = eids[p];
    } else {
      const uint mine = lane < uint(S) ? eids[lane] : 0xFFFFFFFFu;     // one load per lane, no serial chain of device reads
      ex = simd_broadcast(mine, ushort(p));
      const uint all = uint(ulong(simd_ballot(mine == ex)));
      if ((all & ((1u << p) - 1u)) != 0u) return;                      // the threadgroup of the first pair of an expert computes all its pairs
      mm = all & ~((1u << p) - 1u);
    }
    const uint cnt = SING ? 1u : popcount(mm);
"""

_WIDE_OLD = ("    uint mm = 1u << p;", "      const uint mine = lane < uint(S) ? eids[lane] : 0xFFFFFFFFu;", "      ex = simd_broadcast(mine, ushort(p));",
             "      const uint all = uint(ulong(simd_ballot(mine == ex)));", "      if ((all & ((1u << p) - 1u)) != 0u) return;", "      mm = all & ~((1u << p) - 1u);")


def _wide(src: str) -> str:
    """The dedupe of _PRO for S above 32 slots: the pair mask is 64 bits wide, slot s >= 32 sits in lane s - 32 of a second ballot. Everything after the prologue (popcount / ctz /
    `mm &= mm - 1u`) is width-agnostic. The S <= 32 sources are not touched (this function is not applied to them)."""
    for o in _WIDE_OLD:
        assert src.count(o) == 1, "prologue template changed"
    for o, n in zip(_WIDE_OLD, ("    ulong mm = 1ul << p;",
                                "      const uint mine = lane < uint(S) ? eids[lane] : 0xFFFFFFFFu;\n      const uint mine2 = lane + 32u < uint(S) ? eids[lane + 32u] : 0xFFFFFFFFu;",
                                "      ex = p < 32u ? simd_broadcast(mine, ushort(p)) : simd_broadcast(mine2, ushort(p - 32u));",
                                "      const ulong all = ulong(uint(ulong(simd_ballot(mine == ex)))) | (ulong(uint(ulong(simd_ballot(mine2 == ex)))) << 32);",
                                "      if ((all & ((1ul << p) - 1ul)) != 0ul) return;",
                                "      mm = all & ~((1ul << p) - 1ul);")):
        src = src.replace(o, n)
    return src


_INL = _PRO + r"""
    constexpr uint UPR = uint(KIN) / 32u, NU = uint(KIN) / 1024u, NT = 32u * uint(SGN);
    constexpr uint NBU = PG ? NU : 1u;
    const uint rbase = tgi * uint(SGN) + sg;
    device const uint4* qa = (device const uint4*)(qsA + ulong(ex) * (ulong(N) * uint(KIN) / 8u));
    device const ushort* ma = (device const ushort*)(mtA + ulong(ex) * (ulong(N) * (uint(KIN) / 32u)));
    device const uint4* qb = (device const uint4*)(qsB + ulong(ex) * (ulong(N) * uint(KIN) / 8u));
    device const ushort* mb = (device const ushort*)(mtB + ulong(ex) * (ulong(N) * (uint(KIN) / 32u)));
    const float gsa = gsA[ex];
    const float gsb = PG ? gsB[ex] : 1.0f;
    uint4 qva[NU]; ushort mva[NU]; uint4 qvb[NBU]; ushort mvb[NBU];
    UNR for (uint i = 0; i < NU; ++i) {
      const ulong o = ulong(rbase) * UPR + lane + 32u * i;
      qva[i] = qa[o]; mva[i] = ma[o];
    }
    if (PG) {
      UNR for (uint i = 0; i < NU; ++i) {
        const ulong o = ulong(rbase) * UPR + lane + 32u * i;
        qvb[i] = qb[o]; mvb[i] = mb[o];
      }
    }
    threadgroup half xs[(uint(KIN) / 32u) * 40u];
    for (uint r = 0; r < cnt; ++r) {
      const uint slot = ctz(mm);
      mm &= mm - 1u;
      device const half* XP = (device const half*)(x + ulong(slot / uint(XDIV)) * uint(KIN));
      if (r > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint j = tid; j < uint(KIN) / 8u; j += NT) *(threadgroup uint4*)(xs + j * 8u + (j >> 2) * 8u) = *(device const uint4*)(XP + j * 8u);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float acca = 0.0f, accb = 0.0f;
      UNR for (uint i = 0; i < NU; ++i) {
        threadgroup const half4* xp = (threadgroup const half4*)(xs + (lane + 32u * i) * 40u);
        float4 xf[8];
        UNR for (uint t = 0; t < 8; ++t) xf[t] = float4(xp[t]);
        acca = unit_acc(qva[i], mva[i], xf, acca);
        if (PG) accb = unit_acc(qvb[i], mvb[i], xf, accb);
      }
      const float va = simd_sum(acca);
      const float vb = simd_sum(accb);
      if (lane == 0u) {
        if (PG) {
          float a = float(half(va * gsa)), b = float(half(vb * gsb));
          LIMIT_CLAMP
          const float e = exp(-abs(a));
          const float y_ = 1.0f / (1.0f + e);
          const float sg_ = a < 0.0f ? 1.0f - y_ : y_;
          y[ulong(slot) * uint(N) + rbase] = half((a * sg_) * b);
        } else {
          y[ulong(slot) * uint(N) + rbase] = half(va * gsa);
        }
      }
    }
"""

# oMLX lane layout: lane l owns the 16 inputs [512 b + 16 l, +16) of block b: one uint2 (8 weight bytes) and one scale byte of each of RPS rows.
_OMX_HDR = R._HDR + r"""
inline float omx_qdot16(uint2 w, ushort mb, thread const float* xt) {
  // the scale byte of this lane's half unit (low byte: values 0..15 of the unit, high byte: 16..31) already selected by the caller
  const half sh = as_type<half>(ushort((uint(mb) & 0x7Fu) << 7));
  const float sc = float(sh) * 4194304.0f;
  float p = 0.0f;
  UNR for (uint wi = 0; wi < 2; ++wi) {
    const uint wd = wi == 0 ? w.x : w.y;
    UNR for (uint t = 0; t < 4; ++t) {
      const uint mg = (t < 3u ? (wd << (9u - 4u * t)) : (wd >> 3)) & 0x0E000E00u;
      const uint sn = (wd << (12u - 4u * t)) & 0x80008000u;
      const float2 f = float2(as_type<half2>(mg | sn));
      p = fma(f.x, xt[8u * wi + t], p);
      p = fma(f.y, xt[8u * wi + 4u + t], p);
    }
  }
  return p * sc;
}
"""

_OMX = _PRO + r"""
    constexpr uint KU2 = uint(KIN) / 16u;                              // uint2 per row
    constexpr uint KB = uint(KIN) / 512u;
    const uint n0 = (tgi * uint(NSGO) + sg) * uint(RPS);
    device const uint2* qa = (device const uint2*)(qsA + ulong(ex) * (ulong(N) * uint(KIN) / 8u)) + ulong(n0) * KU2 + lane;
    device const uint2* qb = (device const uint2*)(qsB + ulong(ex) * (ulong(N) * uint(KIN) / 8u)) + ulong(n0) * KU2 + lane;
    device const uchar* ma = (device const uchar*)(mtA + ulong(ex) * (ulong(N) * (uint(KIN) / 32u))) + ulong(n0) * (uint(KIN) / 16u) + lane;
    device const uchar* mb = (device const uchar*)(mtB + ulong(ex) * (ulong(N) * (uint(KIN) / 32u))) + ulong(n0) * (uint(KIN) / 16u) + lane;
    float acca[TMAX][RPS], accb[TMAX][RPS];
    for (uint j = 0; j < uint(TMAX); ++j) for (uint r = 0; r < uint(RPS); ++r) { acca[j][r] = 0.0f; accb[j][r] = 0.0f; }
    uint mem[TMAX];
    for (uint j = 0; j < cnt; ++j) { mem[j] = ctz(mm); mm &= mm - 1u; }
    for (uint b = 0; b < KB; ++b) {
      uint2 wa[RPS], wb[RPS]; ushort sa[RPS], sb[RPS];
      for (uint r = 0; r < uint(RPS); ++r) {
        wa[r] = qa[r * KU2 + b * 32u]; sa[r] = ma[r * (uint(KIN) / 16u) + b * 32u];
        if (PG) { wb[r] = qb[r * KU2 + b * 32u]; sb[r] = mb[r * (uint(KIN) / 16u) + b * 32u]; }
      }
      for (uint j = 0; j < cnt; ++j) {
        device const half* xp = (device const half*)(x + ulong(mem[j] / uint(XDIV)) * uint(KIN)) + b * 512u + lane * 16u;
        float xt[16];
        for (uint i = 0; i < 16u; ++i) xt[i] = float(xp[i]);
        for (uint r = 0; r < uint(RPS); ++r) {
          acca[j][r] += omx_qdot16(wa[r], sa[r], xt);
          if (PG) accb[j][r] += omx_qdot16(wb[r], sb[r], xt);
        }
      }
    }
    const float gsa = gsA[ex];
    const float gsb = PG ? gsB[ex] : 1.0f;
    for (uint j = 0; j < cnt; ++j) {
      const uint slot = mem[j];
      for (uint r = 0; r < uint(RPS); ++r) {
        const float va = simd_sum(acca[j][r]);
        const float vb = simd_sum(accb[j][r]);
        if (lane == 0u) {
          if (PG) {
            float a = float(half(va * gsa)), b = float(half(vb * gsb));
            LIMIT_CLAMP
            const float e = exp(-abs(a));
            const float y_ = 1.0f / (1.0f + e);
            const float sg_ = a < 0.0f ? 1.0f - y_ : y_;
            y[ulong(slot) * uint(N) + n0 + r] = half((a * sg_) * b);
          } else {
            y[ulong(slot) * uint(N) + n0 + r] = half(va * gsa);
          }
        }
      }
    }
"""

# down + combine: tg = RP output rows x P slot positions (one simdgroup each, 1 row of 1 expert, G2a register-fed units, x read from the device).
_DC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint tgi = threadgroup_position_in_grid.x;
    const uint pos = sg % uint(P);
    const uint rr = sg / uint(P);
    const uint n = tgi * uint(RP) + rr;
    constexpr uint UPR = uint(KIN) / 32u, NU = uint(KIN) / 1024u;
    threadgroup half prod[RP * S];
    const uint mine = lane < uint(S) ? eids[lane] : 0xFFFFFFFFu;
    const uint ex = simd_broadcast(mine, ushort(pos));
    const uint all = TT > 1 ? uint(ulong(simd_ballot(mine == ex))) : (1u << pos);
    const bool lead = (all & ((1u << pos) - 1u)) == 0u;
    uint mm = all & ~((1u << pos) - 1u);
    if (lead) {
      device const uint4* q4 = (device const uint4*)(qs + ulong(ex) * (ulong(N) * uint(KIN) / 8u));
      device const ushort* MTP = (device const ushort*)(mt + ulong(ex) * (ulong(N) * (uint(KIN) / 32u)));
      const float GS = gs[ex];
      uint4 qv[NU]; ushort mv[NU];
      UNR for (uint i = 0; i < NU; ++i) {
        const ulong o = ulong(n) * UPR + lane + 32u * i;
        qv[i] = q4[o]; mv[i] = MTP[o];
      }
      while (mm != 0u) {
        const uint slot = ctz(mm);
        mm &= mm - 1u;
        device const half4* xp4 = (device const half4*)(h + ulong(slot) * uint(KIN));
        float acc = 0.0f;
        UNR for (uint i = 0; i < NU; ++i) {
          device const half4* xp = xp4 + (lane + 32u * i) * 8u;
          float4 xf[8];
          UNR for (uint t = 0; t < 8; ++t) xf[t] = float4(xp[t]);
          acc = unit_acc(qv[i], mv[i], xf, acc);
        }
        const float v = simd_sum(acc);
        if (lane == 0u) {
          const float w = float(half(wts[slot]));
          prod[rr * uint(S) + slot] = half(float(half(v * GS)) * w);     // routed * weights.astype(fp16): the product rounded to fp16
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(RP * TT)) {
      const uint r2 = tid / uint(TT), t = tid % uint(TT);
      float acc = 0.0f;
      for (uint k = 0; k < uint(K); ++k) acc += float(prod[r2 * uint(S) + t * uint(K) + k]);
      float r = float(half(acc));
      const uint nn = tgi * uint(RP) + r2;
      if (HAS_S) r += float(sh[t * uint(N) + nn]);
      o[t * uint(N) + nn] = half(r);
    }
"""


# G2c lane mapping (lane l owns unit l + 32 i of a 1024-block: 16 B + 2 scales, unit_acc: the sums are those of G2a / G2c, bit for bit) in oMLX's launch structure:
# 32 * NSGO threads per threadgroup, RPS rows per simdgroup, x read from the device (no staging, no barrier), in-kernel dedupe.
_G2M = _PRO + r"""
    constexpr uint UPR = uint(KIN) / 32u, NU = uint(KIN) / 1024u;
    const uint n0 = (tgi * uint(NSGO) + sg) * uint(RPS);
    device const uint4* qa = (device const uint4*)(qsA + ulong(ex) * (ulong(N) * uint(KIN) / 8u)) + ulong(n0) * UPR + lane;
    device const uint4* qb = (device const uint4*)(qsB + ulong(ex) * (ulong(N) * uint(KIN) / 8u)) + ulong(n0) * UPR + lane;
    device const ushort* ma = (device const ushort*)(mtA + ulong(ex) * (ulong(N) * (uint(KIN) / 32u))) + ulong(n0) * UPR + lane;
    device const ushort* mb = (device const ushort*)(mtB + ulong(ex) * (ulong(N) * (uint(KIN) / 32u))) + ulong(n0) * UPR + lane;
    uint mem[TMAX];
    for (uint j = 0; j < cnt; ++j) { mem[j] = ctz(mm); mm &= mm - 1u; }
    float acca[TMAX][RPS], accb[TMAX][RPS];
    for (uint j = 0; j < uint(TMAX); ++j) for (uint r = 0; r < uint(RPS); ++r) { acca[j][r] = 0.0f; accb[j][r] = 0.0f; }
    for (uint i = 0; i < NU; ++i) {
      uint4 wa[RPS], wb[RPS]; ushort sa[RPS], sb[RPS];
      UNR for (uint r = 0; r < uint(RPS); ++r) {
        wa[r] = qa[r * UPR + 32u * i]; sa[r] = ma[r * UPR + 32u * i];
        if (PG) { wb[r] = qb[r * UPR + 32u * i]; sb[r] = mb[r * UPR + 32u * i]; }
      }
      for (uint j = 0; j < cnt; ++j) {
        device const half4* xp = (device const half4*)(x + ulong(mem[j] / uint(XDIV)) * uint(KIN)) + (lane + 32u * i) * 8u;
        float4 xf[8];
        UNR for (uint t = 0; t < 8; ++t) xf[t] = float4(xp[t]);
        UNR for (uint r = 0; r < uint(RPS); ++r) {
          acca[j][r] = unit_acc(wa[r], sa[r], xf, acca[j][r]);
          if (PG) accb[j][r] = unit_acc(wb[r], sb[r], xf, accb[j][r]);
        }
      }
    }
    const float gsa = gsA[ex];
    const float gsb = PG ? gsB[ex] : 1.0f;
    for (uint j = 0; j < cnt; ++j) {
      const uint slot = mem[j];
      for (uint r = 0; r < uint(RPS); ++r) {
        const float va = simd_sum(acca[j][r]);
        const float vb = simd_sum(accb[j][r]);
        if (lane == 0u) {
          if (PG) {
            float a = float(half(va * gsa)), b = float(half(vb * gsb));
            LIMIT_CLAMP
            const float e = exp(-abs(a));
            const float y_ = 1.0f / (1.0f + e);
            const float sg_ = a < 0.0f ? 1.0f - y_ : y_;
            y[ulong(slot) * uint(N) + n0 + r] = half((a * sg_) * b);
          } else {
            y[ulong(slot) * uint(N) + n0 + r] = half(va * gsa);
          }
        }
      }
    }
"""

_G2MDC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tile = threadgroup_position_in_grid.x;
    const uint t = threadgroup_position_in_grid.y;
    constexpr uint UPR = uint(KIN) / 32u, NU = uint(KIN) / 1024u;
    const uint n0 = (tile * uint(NSGO) + sg) * uint(RPS);
    float acc[RPS];
    for (uint r = 0; r < uint(RPS); ++r) acc[r] = 0.0f;
    for (uint k = 0; k < uint(K); ++k) {
      const uint slot = t * uint(K) + k;
      const uint ex = eids[slot];
      device const uint4* qa = (device const uint4*)(qs + ulong(ex) * (ulong(N) * uint(KIN) / 8u)) + ulong(n0) * UPR + lane;
      device const ushort* ma = (device const ushort*)(mt + ulong(ex) * (ulong(N) * (uint(KIN) / 32u))) + ulong(n0) * UPR + lane;
      float a[RPS];
      for (uint r = 0; r < uint(RPS); ++r) a[r] = 0.0f;
      for (uint i = 0; i < NU; ++i) {
        device const half4* xp = (device const half4*)(h + ulong(slot) * uint(KIN)) + (lane + 32u * i) * 8u;
        float4 xf[8];
        UNR for (uint u8 = 0; u8 < 8; ++u8) xf[u8] = float4(xp[u8]);
        UNR for (uint r = 0; r < uint(RPS); ++r) a[r] = unit_acc(qa[r * UPR + 32u * i], ma[r * UPR + 32u * i], xf, a[r]);
      }
      const float GS = gs[ex];
      const float w = float(half(wts[slot]));
      UNR for (uint r = 0; r < uint(RPS); ++r) acc[r] += float(half(float(half(simd_sum(a[r]) * GS)) * w));
    }
    if (lane < uint(RPS)) {
      float v = 0.0f;
      for (uint r = 0; r < uint(RPS); ++r) v = r == lane ? acc[r] : v;
      float r2 = float(half(v));
      if (HAS_S) r2 += float(sh[t * uint(N) + n0 + lane]);
      o[t * uint(N) + n0 + lane] = half(r2);
    }
"""

# down + combine, oMLX moe_down_combine structure: a simdgroup owns RPS output rows of ONE token and walks its K routed experts in slot order (weighted sum in registers);
# the grid is slot-major (tokens vary fastest) so the rows that share an expert run back to back (L2 hits instead of an in-kernel dedupe).
_OMXDC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tile = threadgroup_position_in_grid.x;
    const uint t = threadgroup_position_in_grid.y;
    constexpr uint KU2 = uint(KIN) / 16u, KB = uint(KIN) / 512u;
    const uint n0 = (tile * uint(NSGO) + sg) * uint(RPS);
    float acc[RPS];
    for (uint r = 0; r < uint(RPS); ++r) acc[r] = 0.0f;
    for (uint k = 0; k < uint(K); ++k) {
      const uint slot = t * uint(K) + k;
      const uint ex = eids[slot];
      device const uint2* qa = (device const uint2*)(qs + ulong(ex) * (ulong(N) * uint(KIN) / 8u)) + ulong(n0) * KU2 + lane;
      device const uchar* ma = (device const uchar*)(mt + ulong(ex) * (ulong(N) * (uint(KIN) / 32u))) + ulong(n0) * (uint(KIN) / 16u) + lane;
      device const half* hp = h + ulong(slot) * uint(KIN) + lane * 16u;
      float a[RPS];
      for (uint r = 0; r < uint(RPS); ++r) a[r] = 0.0f;
      for (uint b = 0; b < KB; ++b) {
        float xt[16];
        for (uint i = 0; i < 16u; ++i) xt[i] = float(hp[b * 512u + i]);
        for (uint r = 0; r < uint(RPS); ++r) a[r] += omx_qdot16(qa[r * KU2 + b * 32u], ma[r * (uint(KIN) / 16u) + b * 32u], xt);
      }
      const float GS = gs[ex];
      const float w = float(half(wts[slot]));
      for (uint r = 0; r < uint(RPS); ++r) acc[r] += float(half(float(half(simd_sum(a[r]) * GS)) * w));
    }
    if (lane < uint(RPS)) {
      float v = 0.0f;
      for (uint r = 0; r < uint(RPS); ++r) v = r == lane ? acc[r] : v;
      float r2 = float(half(v));
      if (HAS_S) r2 += float(sh[t * uint(N) + n0 + lane]);
      o[t * uint(N) + n0 + lane] = half(r2);
    }
"""


_XLOAD8 = "UNR for (uint t = 0; t < 8; ++t) xf[t] = float4(xp[t]);"
_XLOAD16 = """device const uint4* xq = (device const uint4*)xp;                              // the same 64 bytes as four 16-byte loads (as_type: bit-exact)
        UNR for (uint t = 0; t < 4; ++t) { const uint4 xw = xq[t]; xf[2u * t] = float4(as_type<half4>(xw.xy)); xf[2u * t + 1u] = float4(as_type<half4>(xw.zw)); }"""
_ILOOP = "for (uint i = 0; i < NU; ++i) {"


_JLOOP = "for (uint j = 0; j < cnt; ++j)"


def _g2m_src(xv: int, iu: int, ju: int = 0) -> str:
    """G2M source. xv: x segment of a lane read as 4 x 16 B instead of 8 x 8 B (same bits, half the load instructions); iu: the unit loop fully unrolled (the loads of all units issue before the first FMA);
    ju (V2-VERIFY, KYOJIN_MOE_ROWS2_UNROLL): the pair loops `for j < cnt` (cnt = rows that picked the expert, dynamic) become `UNR for j < TMAX: if (j < cnt)`, so mem[] / acca[][] / accb[][] are indexed
    by compile-time constants (registers) instead of a runtime index (private memory); the same operations in the same order for every j < cnt: bit-identical."""
    s = _G2M
    assert _XLOAD8 in s and _ILOOP in s
    if ju:
        assert s.count(_JLOOP) == 3, "G2M changed"
        s = s.replace(_JLOOP, "UNR for (uint j = 0; j < uint(TMAX); ++j) if (j < cnt)")
    if xv:
        s = s.replace(_XLOAD8, _XLOAD16)
    if iu:
        s = s.replace(_ILOOP, "UNR " + _ILOOP)
    return s


@functools.lru_cache(maxsize=None)
def _g2m_kern(pg: bool, limit, xv: int = 0, iu: int = 0, ju: int = 0, wide: int = 0, act=None, scm: int = 0):
    clamp = "" if (limit is None or act) else f"a = min(a, {float(limit)!r}f); b = clamp(b, -{float(limit)!r}f, {float(limit)!r}f);"
    return _kern_s("g2m", scm, ("g2e_g2m", pg, limit) + ((xv, iu) if (xv or iu) else ()) + (("ju",) if ju else ()) + (("wide",) if wide else ()) + ((act,) if act else ()), name="kyojin_g2e_g2m_" + ("pair_glu" if pg else "down") + (f"_x{xv}u{iu}" if (xv or iu) else "") + ("_ju" if ju else "") + ("_w" if wide else "") + ("_situ" if act else ""),
                    input_names=["x", "eids", "qsA", "mtA", "gsA", "qsB", "mtB", "gsB"], output_names=["y"], header=R._HDR,
                    source=R.act_source((_wide if wide else str)(_g2m_src(xv, iu, ju)).replace("LIMIT_CLAMP", clamp), act))


@functools.lru_cache(maxsize=None)
def _g2mdc_kern(scm: int = 0):
    return _kern_s("g2mdc", scm, ("g2e_g2mdc",), name="kyojin_g2e_g2m_down_combine", input_names=["h", "eids", "wts", "qs", "mt", "gs", "sh"], output_names=["o"], header=R._HDR, source=_G2MDC)


@functools.lru_cache(maxsize=None)
def _omxdc_kern(scm: int = 0):
    return _kern_s("omxdc", scm, ("g2e_omxdc",), name="kyojin_g2e_omx_down_combine", input_names=["h", "eids", "wts", "qs", "mt", "gs", "sh"], output_names=["o"], header=_OMX_HDR, source=_OMXDC)


def down_combine_seq(d, h, inds, wts, sh, T, K, rps=2, nsg=2, kind="omx"):
    """oMLX's sequential walk over the K experts of a token. kind g2m: bit-identical to down_combine (G2c lane mapping); omx: oMLX lane mapping (<= 1 ulp per product)."""
    scm = MC.usable(d)
    return (_g2mdc_kern(scm) if kind == "g2m" else _omxdc_kern(scm))(inputs=[h, inds, wts, d.qs, *_mt(d, scm), d.gs, sh if sh is not None else h],
                         template=[("KIN", d.k), ("N", d.n), ("K", K), ("RPS", rps), ("NSGO", nsg), ("HAS_S", sh is not None)],
                         grid=(d.n // (rps * nsg) * 32, T * nsg, 1), threadgroup=(32, nsg, 1), output_shapes=[(T, d.n)], output_dtypes=[mx.float16])[0]


@functools.lru_cache(maxsize=None)
def _inl_kern(pg: bool, limit, wide: int = 0, act=None, scm: int = 0):
    clamp = "" if (limit is None or act) else f"a = min(a, {float(limit)!r}f); b = clamp(b, -{float(limit)!r}f, {float(limit)!r}f);"
    return _kern_s("inl", scm, ("g2e_inl", pg, limit) + (("wide",) if wide else ()) + ((act,) if act else ()), name="kyojin_g2e_inl_" + ("pair_glu" if pg else "down") + ("_w" if wide else "") + ("_situ" if act else ""),
                    input_names=["x", "eids", "qsA", "mtA", "gsA", "qsB", "mtB", "gsB"], output_names=["y"], header=R._HDR, source=R.act_source((_wide if wide else str)(_INL).replace("LIMIT_CLAMP", clamp), act))


@functools.lru_cache(maxsize=None)
def _omx_kern(pg: bool, limit, unb: int = 0, wide: int = 0, act=None, scm: int = 0):
    clamp = "" if (limit is None or act) else f"a = min(a, {float(limit)!r}f); b = clamp(b, -{float(limit)!r}f, {float(limit)!r}f);"
    return _kern_s("omx", scm, ("g2e_omx", pg, limit, unb) + (("wide",) if wide else ()) + ((act,) if act else ()), name="kyojin_g2e_omx_" + ("pair_glu" if pg else "down") + ("_u" if unb else "") + ("_w" if wide else "") + ("_situ" if act else ""),
                    input_names=["x", "eids", "qsA", "mtA", "gsA", "qsB", "mtB", "gsB"], output_names=["y"], header=_OMX_HDR, source=R.act_source((_wide if wide else str)(_OMX).replace("LIMIT_CLAMP", clamp).replace("for (uint b = 0; b < KB; ++b) {", "UNR for (uint b = 0; b < KB; ++b) {" if unb else "for (uint b = 0; b < KB; ++b) {"), act))


@functools.lru_cache(maxsize=None)
def _dc_kern(scm: int = 0):
    return _kern_s("dc", scm, ("g2e_dc",), name="kyojin_g2e_down_combine", input_names=["h", "eids", "wts", "qs", "mt", "gs", "sh"], output_names=["o"], header=R._HDR, source=_DC)


def pair_glu(kind, g, u, x2, inds, T, K, limit, rps=4, nsg=2, unb=0, xv=0, iu=0, ju=0, act=None):
    """h half [S, g.n]: clamped silu(gate_e(x[slot // K])) * up_e(...), the pairs of one expert in one threadgroup column (no group table)."""
    from . import foreign_linear as FL
    S = T * K
    scm = MC.usable(g, u)
    if kind == "inl":
        SGN = FL.G2A_SGN
        return _inl_kern(True, limit, int(S > S_MAX), act, scm)(inputs=[x2, inds, g.qs, *_mt(g, scm), g.gs, u.qs, *_mt(u, scm), u.gs], template=[("KIN", g.k), ("N", g.n), ("SGN", SGN), ("S", S), ("SING", int(T == 1)), ("PG", 1), ("XDIV", K)],
                                      grid=(g.n // SGN * 32 * SGN, S, 1), threadgroup=(32 * SGN, 1, 1), output_shapes=[(S, g.n)], output_dtypes=[mx.float16])[0]
    assert g.n % (rps * nsg) == 0 and g.k % 512 == 0
    if kind == "g2m":
        return _g2m_kern(True, limit, xv, iu, ju if T > 1 else 0, int(S > S_MAX), act, scm)(inputs=[x2, inds, g.qs, *_mt(g, scm), g.gs, u.qs, *_mt(u, scm), u.gs],
                                      template=[("KIN", g.k), ("N", g.n), ("S", S), ("SING", int(T == 1)), ("PG", 1), ("XDIV", K), ("TMAX", T), ("RPS", rps), ("NSGO", nsg)],
                                      grid=(g.n // (rps * nsg) * 32, S * nsg, 1), threadgroup=(32, nsg, 1), output_shapes=[(S, g.n)], output_dtypes=[mx.float16])[0]
    return _omx_kern(True, limit, unb, int(S > S_MAX), act, scm)(inputs=[x2, inds, g.qs, *_mt(g, scm), g.gs, u.qs, *_mt(u, scm), u.gs],
                                  template=[("KIN", g.k), ("N", g.n), ("S", S), ("SING", int(T == 1)), ("PG", 1), ("XDIV", K), ("TMAX", T), ("RPS", rps), ("NSGO", nsg)],
                                  grid=(g.n // (rps * nsg) * 32, S * nsg, 1), threadgroup=(32, nsg, 1), output_shapes=[(S, g.n)], output_dtypes=[mx.float16])[0]


def down(kind, d, h, inds, T, K, rps=4, nsg=2, unb=0, xv=0, iu=0, ju=0):
    """y half [S, d.n] = down_e(h[slot])."""
    from . import foreign_linear as FL
    S = T * K
    scm = MC.usable(d)
    if kind == "inl":
        SGN = FL.G2A_SGN
        return _inl_kern(False, None, int(S > S_MAX), None, scm)(inputs=[h, inds, d.qs, *_mt(d, scm), d.gs, d.qs, *_mt(d, scm), d.gs], template=[("KIN", d.k), ("N", d.n), ("SGN", SGN), ("S", S), ("SING", int(T == 1)), ("PG", 0), ("XDIV", 1)],
                                      grid=(d.n // SGN * 32 * SGN, S, 1), threadgroup=(32 * SGN, 1, 1), output_shapes=[(S, d.n)], output_dtypes=[mx.float16])[0]
    if kind == "g2m":
        return _g2m_kern(False, None, xv, iu, ju if T > 1 else 0, int(S > S_MAX), None, scm)(inputs=[h, inds, d.qs, *_mt(d, scm), d.gs, d.qs, *_mt(d, scm), d.gs],
                                      template=[("KIN", d.k), ("N", d.n), ("S", S), ("SING", int(T == 1)), ("PG", 0), ("XDIV", 1), ("TMAX", T), ("RPS", rps), ("NSGO", nsg)],
                                      grid=(d.n // (rps * nsg) * 32, S * nsg, 1), threadgroup=(32, nsg, 1), output_shapes=[(S, d.n)], output_dtypes=[mx.float16])[0]
    return _omx_kern(False, None, unb, int(S > S_MAX), None, scm)(inputs=[h, inds, d.qs, *_mt(d, scm), d.gs, d.qs, *_mt(d, scm), d.gs],
                                  template=[("KIN", d.k), ("N", d.n), ("S", S), ("SING", int(T == 1)), ("PG", 0), ("XDIV", 1), ("TMAX", T), ("RPS", rps), ("NSGO", nsg)],
                                  grid=(d.n // (rps * nsg) * 32, S * nsg, 1), threadgroup=(32, nsg, 1), output_shapes=[(S, d.n)], output_dtypes=[mx.float16])[0]


def dc_ok(d, T, K) -> bool:
    return T <= TMAX and T * K <= S_MAX and d.k % 1024 == 0 and d.n % 4 == 0


def down_combine(d, h, inds, wts, sh, T, K):
    """out half [T, d.n] = sum_k half(half(down_e(h[t K + k])) * half(w[t K + k])) (f32 sum, one fp16 rounding) + sh (fp16), one launch."""
    S = T * K
    scm = MC.usable(d)
    rp = max(1, 16 // S)
    while d.n % rp:
        rp -= 1
    ns = S * rp
    return _dc_kern(scm)(inputs=[h, inds, wts, d.qs, *_mt(d, scm), d.gs, sh if sh is not None else h], template=[("KIN", d.k), ("N", d.n), ("S", S), ("P", S), ("RP", rp), ("TT", T), ("K", K), ("HAS_S", sh is not None)],
                      grid=(d.n // rp * 32 * ns, 1, 1), threadgroup=(32 * ns, 1, 1), output_shapes=[(T, d.n)], output_dtypes=[mx.float16])[0]
