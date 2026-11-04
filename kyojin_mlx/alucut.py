# SPDX-License-Identifier: Apache-2.0
"""Exact ALU / latency cuts of the 1-row decode step (cplain). Pure Python strings and switches: no mlx import, so every rewrite is
checked on any machine (tests/test_alucut.py) by asserting that the rewrite applied exactly once and by emulating the old and the new
algorithm in numpy.

Every cut keeps the arithmetic of the kernel it rewrites value for value (same operands, same operation order, same reduction trees, same
roundings); only the barrier / load / store structure around the arithmetic changes. All default OFF, one switch each (read per kernel build,
so the flags table can flip them in one process and the compiled regions retrace on the flags generation):

  KYOJIN_ALU_MIXBAR   gluefuse.mixvec (the mHC mix vector, 2 per layer): the row reduction of the 1024-thread threadgroup ends in ONE barrier
                      instead of three. All 32 simdgroups write their partial (the pre-zeroing of `ls` and its barrier were only needed for fewer
                      than 32 simdgroups), then every simdgroup runs the same `simd_sum(ls[lane])` simdgroup 0 ran, so `inv` is in every thread's
                      register and the broadcast through threadgroup memory (and its barrier) goes away.
  KYOJIN_ALU_MIXPF    gluefuse.mixvec: the weight row of the output is loaded one 8-wide step ahead of its use, the first step before the
                      row reduction starts, so the load latency of `hc_fn` (fp32, 1.97 MB per mix, 157 MB per step) hides behind the reduction.
                      Loads are reordered only; the FMA chain per lane is unchanged.
  KYOJIN_ALU_ROUTER1  fuse.router_select (1 per layer): each selection round ends in ONE barrier instead of two, and the merge of the four
                      simdgroup winners runs in every thread (registers) instead of thread 0 (threadgroup memory + a second barrier). The
                      selected expert's key is retired by the thread that owns it, which is also the only thread that reads it again.
  KYOJIN_ALU_NOY32    fuse.mhc_expand (2 per layer): the fp32 copy of the expanded streams (`y32`, 80 KB per call) is not written. Nobody reads it
                      under the default flags (gluefuse.mixvec reads x in its own dtype); where somebody does (the stock mix vector) the
                      cast is taken from the fp16 array, which is the same value (y32 IS float(y))."""
from __future__ import annotations

import os


def _on(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "")


def mixbar() -> bool:
    return _on("KYOJIN_ALU_MIXBAR")


def mixpf() -> bool:
    return _on("KYOJIN_ALU_MIXPF")


def router1() -> bool:
    return _on("KYOJIN_ALU_ROUTER1")


def noy32() -> bool:
    return _on("KYOJIN_ALU_NOY32")


def _sub1(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"alucut: template changed, expected one occurrence of {old[:60]!r}, found {src.count(old)}"
    return src.replace(old, new)


# ------------------------------------------------------------------------------------------------------------------- mixvec (gluefuse._MIXVEC)

_MIXVEC_RED_OLD = """    acc = simd_sum(acc);
    if (sg == 0) { ls[lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { ls[sg] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(ls[lane]);
        if (lane == 0) { inv_s[0] = metal::precise::rsqrt(acc / float(K) + eps[0]); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_s[0];
"""
_MIXVEC_RED_NEW = """    acc = simd_sum(acc);
    if (lane == 0) { ls[sg] = acc; }                                       // the threadgroup is 1024 threads = 32 simdgroups: every slot of `ls` is written
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(simd_sum(ls[lane]) / float(K) + eps[0]);     // the simd_sum simdgroup 0 ran, in every simdgroup: same 32 values, same lanes
"""

_MIXVEC_DOT_OLD = """    if (sg < KS) {
        const device W* wp = w + ulong(o) * K;
        float d = 0.0f;
        for (uint kk = (sg * 32 + lane) * 8; kk < K; kk += 256 * KS) {
            float wv[8];
            for (uint j = 0; j < 8; ++j) { wv[j] = (kk + j < K) ? float(wp[kk + j]) : 0.0f; }
            for (uint j = 0; j < 8; ++j) {
"""
_MIXVEC_DOT_NEW = """    if (sg < KS) {
        const device W* wp = w + ulong(o) * K;
        float d = 0.0f;
        for (uint kk = (sg * 32 + lane) * 8; kk < K; kk += 256 * KS) {
            float wv[8];
            for (uint j = 0; j < 8; ++j) { wv[j] = wn[j]; }
            {                                                                // the next step's weights are in flight while this one runs
                const uint kn = kk + 256 * KS;
                for (uint j = 0; j < 8; ++j) { wn[j] = (kn + j < K) ? float(wp[kn + j]) : 0.0f; }
            }
            for (uint j = 0; j < 8; ++j) {
"""
_MIXVEC_HEAD_OLD = """    const device T* xr = x + ulong(b) * K;
    float acc = 0.0f;
"""
_MIXVEC_HEAD_NEW = """    const device T* xr = x + ulong(b) * K;
    float wn[8];                                                             // weights of the first step of this simdgroup's k range, loaded before the row reduction
    if (sg < KS) {
        const device W* wp0 = w + ulong(o) * K;
        const uint k0 = (sg * 32 + lane) * 8;
        for (uint j = 0; j < 8; ++j) { wn[j] = (k0 + j < K) ? float(wp0[k0 + j]) : 0.0f; }
    }
    float acc = 0.0f;
"""


def mixvec_source(src: str, bar: bool, pf: bool) -> str:
    """gluefuse._MIXVEC with the one-barrier row reduction (bar) and / or the weight prefetch (pf)."""
    if bar:
        src = _sub1(src, _MIXVEC_RED_OLD, _MIXVEC_RED_NEW)
    if pf:
        src = _sub1(src, _MIXVEC_HEAD_OLD, _MIXVEC_HEAD_NEW)
        src = _sub1(src, _MIXVEC_DOT_OLD, _MIXVEC_DOT_NEW)
    return src


def mixvec_dot_steps(K: int, KS: int, sg: int, lane: int, prefetch: bool) -> list[int]:
    """The k offsets whose weights one lane multiplies, in order. With the prefetch the weights of step i + 1 are loaded during step i (wn), the first
    one before the row reduction (the head load); the multiplied sequence must not change (the emulation tests/test_alucut.py compares)."""
    k0 = (sg * 32 + lane) * 8
    if not prefetch:
        return list(range(k0, K, 256 * KS))
    used, wn = [], k0                                        # head load: weights of the first step
    for kk in range(k0, K, 256 * KS):
        wv, wn = wn, kk + 256 * KS                           # wv <- wn; wn <- loads of the next step (past K: zeros, never multiplied)
        used.append(wv)
    return used


# ------------------------------------------------------------------------------------------------------------------- router_select (fuse._ROUTER)

_ROUTER_TG_OLD = """    threadgroup float rk[4];
    threadgroup uint ri[4];
    threadgroup uint chosen[K];
"""
_ROUTER_TG_NEW = """    threadgroup float rk[2][4];                                     // double buffered by round parity: one barrier per round
    threadgroup uint ri[2][4];
    uint chosen[K];                                                 // every thread tracks the same winners in registers
"""
_ROUTER_LOOP_OLD = """        if (lane == 0) { rk[sg] = mk; ri[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            float fk = rk[0];
            uint fi = ri[0];
            for (uint g = 1; g < 4; ++g) {
                if (rk[g] < fk || (rk[g] == fk && ri[g] < fi)) { fk = rk[g]; fi = ri[g]; }
            }
            chosen[j] = fi;
            skey[fi] = inf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
_ROUTER_LOOP_NEW = """        if (lane == 0) { rk[j & 1][sg] = mk; ri[j & 1][sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float fk = rk[j & 1][0];
        uint fi = ri[j & 1][0];
        for (uint g = 1; g < 4; ++g) {
            if (rk[j & 1][g] < fk || (rk[j & 1][g] == fk && ri[j & 1][g] < fi)) { fk = rk[j & 1][g]; fi = ri[j & 1][g]; }
        }
        chosen[j] = fi;
        if (fi < E && (fi & 127u) == t) { skey[fi] = inf; }         // the owner (e = t + 128 m) retires it and is the only thread that scans it again
    }
"""


def router_source(src: str) -> str:
    """fuse._ROUTER with one barrier per selection round."""
    src = _sub1(src, _ROUTER_TG_OLD, _ROUTER_TG_NEW)
    return _sub1(src, _ROUTER_LOOP_OLD, _ROUTER_LOOP_NEW)


# ------------------------------------------------------------------------------------------------------------------- mhc_expand (fuse._EXPAND)

_EXPAND_Y32 = "        y32[(ulong(b) * 4 + j) * D + d] = float(o);\n"


def expand_source_no_y32(src: str) -> str:
    """fuse._EXPAND without the store of the fp32 copy (the kernel is built with the single output `y`)."""
    return _sub1(src, _EXPAND_Y32, "")
