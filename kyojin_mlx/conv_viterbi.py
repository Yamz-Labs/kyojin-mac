# SPDX-License-Identifier: Apache-2.0
"""EXL3 trellis quantizer (tail-biting Viterbi over 2^16 states, mul1 codebook) on Metal via mx.fast.metal_kernel.

Port of exllamav3 quantize_tiles_kernel.cuh (cb = 2): same fp16 arithmetic (hsub, hmul, fma), same strict-< branch tie
order, same argmin tie order, so (tiles -> idx) is meant to be bit-equal to the CUDA / ROCm kernel.
The history stores the branch number k (K bits, one byte) instead of the 16-bit predecessor edge: pred = ((k << (16-K)) | out) >> K.
One threadgroup per tile; Metal has 32 KB of threadgroup memory so the two cost rows live in device memory."""
from __future__ import annotations

import numpy as np
import mlx.core as mx

MUL1 = 0x83DCD12D
L = 256
NT = 1024

HEADER = """
#include <metal_stdlib>
using namespace metal;
constant uint MUL1 = 0x83DCD12Du;
inline half dec_mul1(uint x) {
    uint p = x * MUL1;
    uint s = (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24) + 0x6400u;
    return fma(as_type<half>((ushort) s), as_type<half>((ushort) 0x1eee), as_type<half>((ushort) 0xc931));
}
inline uint brev5(uint v) { return ((v & 1u) << 4) | ((v & 2u) << 2) | (v & 4u) | ((v & 8u) >> 2) | ((v & 16u) >> 4); }
"""


def _source(K: int, NT: int = NT, table: bool = False, exp: int = 0) -> str:
    Kr = 16 - K
    edges = 65536 >> K
    tl = "as_type<half2>(tab2[(uint)(X) >> 1])" if table else "half2(dec_mul1((uint)(X)), dec_mul1((uint)(X) + 1u))"
    return f"""
    #define TABLE_LOAD(X) ({tl})
    constexpr int K = {K}; constexpr int Kr = {Kr}; constexpr int EDGES = {edges}; constexpr int MAXQ = {1 << K};
    constexpr int L = {L}; constexpr int NT = {NT};
    uint tile = threadgroup_position_in_grid.x;
    uint thr = thread_position_in_threadgroup.x;
    uint lane = thr & 31u, wid = thr >> 5;
    threadgroup half sh_tile[L];
    threadgroup uint sh_key[32];
    threadgroup int sh_edge;
    device half* costA = costs + (size_t) tile * 2 * EDGES;
    device half* costB = costA + EDGES;
    device uchar* H = hist + (size_t) tile * L * EDGES;
    for (uint i = thr; i < L; i += NT) sh_tile[i] = (half) tiles[(size_t) tile * L + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half* cur = costA; device half* prv = costB;
    int end_state = 0;
    for (int pass = 0; pass < 2; ++pass) {{
        const int roll = pass == 0 ? L / 2 : 0;
        const int pre_state = pass == 0 ? -1 : end_state;
        for (int i = 0; i < L; ++i) {{
            int ri = i + roll; if (ri >= L) ri -= L;
            device half* t = cur; cur = prv; prv = t;          // prv = costs of the previous step, cur = this step
            const half w = sh_tile[ri];
            for (int oe = 2 * (int) thr; oe < EDGES; oe += 2 * NT) {{
                half2 dd = TABLE_LOAD(oe);
                half d0 = dd.x, d1 = dd.y;
                int ie = oe >> K;
                half e0, e1;
                {{
                    half a = d0 - w, b = d1 - w;
                    if (i == 0) {{
                        e0 = a * a; e1 = b * b;
                        if (pre_state >= 0 && ie != pre_state) {{ e0 = INFINITY; e1 = INFINITY; }}
                    }} else {{
                        half c = prv[ie];
                        e0 = fma(a, a, c); e1 = fma(b, b, c);
                    }}
                }}
                uchar k0 = 0, k1 = 0;
                for (int k = 1; k < MAXQ; ++k) {{
                    int s0 = (k << Kr) | oe;
                    ie = s0 >> K;
                    half2 dk = TABLE_LOAD(s0);
                    half a = dk.x - w, b = dk.y - w;
                    half f0, f1;
                    if (i == 0) {{
                        f0 = a * a; f1 = b * b;
                        if (pre_state >= 0 && ie != pre_state) {{ f0 = INFINITY; f1 = INFINITY; }}
                    }} else {{
                        half c = prv[ie];
                        f0 = fma(a, a, c); f1 = fma(b, b, c);
                    }}
                    if (f0 < e0) {{ e0 = f0; k0 = (uchar) k; }}
                    if (f1 < e1) {{ e1 = f1; k1 = (uchar) k; }}
                }}
                cur[oe] = e0; cur[oe + 1] = e1;
                if (!({exp} & 2) && (pre_state >= 0 || ri < L / 2)) {{
                    ((device ushort*) H)[((size_t) ri * EDGES + oe) >> 1] = (ushort) k0 | ((ushort) k1 << 8);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_device);
        }}
        // forward leaves the last step's costs in cur
        int edge;
        if (pass == 0) {{
            uint best = 0x7c00ffffu;
            for (int e = (int) thr; e < EDGES; e += NT) {{
                uint v = (uint) e & 1023u;
                uint rank = (brev5(v >> 5) << 10) | (brev5(v & 31u) << 5) | ((uint) e >> 10);
                uint key = ((uint) as_type<ushort>(cur[e]) << 16) | rank;
                best = min(best, key);
            }}
            best = simd_min(best);
            if (lane == 0) sh_key[wid] = best;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thr == 0) {{
                uint b = 0x7c00ffffu;
                for (int j = 0; j < NT / 32; ++j) b = min(b, sh_key[j]);
                uint rank = b & 65535u;
                uint v = (brev5(rank >> 10) << 5) | brev5((rank >> 5) & 31u);
                edge = b >= 0x7c000000u ? 0 : (int) (((rank & 31u) << 10) | v);
                for (int i = L - 1; i >= 0 && !({exp} & 1); --i) {{
                    int ri = i + roll; if (ri >= L) ri -= L;
                    int kk = (int) H[(size_t) ri * EDGES + edge];
                    edge = ((kk << Kr) | edge) >> K;
                    if (ri == 0) break;
                }}
                sh_edge = edge;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            end_state = sh_edge;
        }} else {{
            if (thr == 0) {{
                edge = end_state;
                for (int i = L - 1; i >= 0 && !({exp} & 1); --i) {{
                    int kk = (int) H[(size_t) i * EDGES + edge];
                    int prev = ((kk << Kr) | edge) >> K;
                    int enc = (prev << K) | edge;
                    edge = prev;
                    idx[(size_t) tile * L + i] = (ushort) enc;
                    wq[(size_t) tile * L + i] = (float) dec_mul1((uint) enc);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_device);
    }}
    """


def _source2(K: int, NT: int, exp: int = 0, S: int = 0) -> str:
    """Port of the optimized CUDA structure: the 2^K candidate codebook values of every output pair stay in registers
    for all 512 steps (no decode, no table load), the cost row lives in threadgroup memory (single row: next costs are
    staged in registers, barrier, then stored), the argmin reads the registers. Same fp16 arithmetic and strict-< tie order.
    A thread owns NG runs of S consecutive output pairs; the S pairs of a run share their predecessor cost (out >> K), so one
    threadgroup load serves S pairs (S <= 2^(K-1))."""
    Kr = 16 - K
    edges = 65536 >> K
    NC = 1 << K
    NP = edges // (2 * NT)
    S = S or {2: 2, 3: 2}.get(K, 1)  # measured on the Air M5: more pairs per run help K2 / K3 only (registers)
    NG = NP // S
    assert NP >= 1 and 2 * NT * NP == edges and S * NG == NP and S <= (1 << (K - 1))
    o = []
    for g in range(NG):
        for u in range(S):
            o.append(f"const int out{g}_{u} = 2 * (((int) thr + {g} * NT) * {S} + {u});")
            for c in range(NC):
                o.append(f"half2 v{g}_{u}_{c} = half2(dec_mul1((uint) ((({c}) << Kr) | out{g}_{u})), dec_mul1((uint) ((({c}) << Kr) | out{g}_{u}) + 1u));")
            o.append(f"half2 n{g}_{u};")
    decl = "\n    ".join(o)

    def step(first):
        t = []
        for g in range(NG):
            t.append("{")
            for u in range(S):
                t.append(f"half2 best{u}; uchar k0_{u} = 0, k1_{u} = 0;")
            for c in range(NC):
                ie = f"((({c}) << (Kr - K)) | (out{g}_0 >> K))"
                if not first:
                    t.append(f"const half cc{c} = row[{ie}];")
                for u in range(S):
                    d = f"v{g}_{u}_{c} - w2"
                    if first:
                        mask = f"if (pre_state >= 0 && {ie} != pre_state) err = half2(INFINITY);"
                        err = f"half2 err = {d}; err = err * err; {mask}"
                    else:
                        err = f"half2 dd = {d}; half2 err = fma(dd, dd, half2(cc{c}));"
                    if c == 0:
                        t.append(f"{{ {err} best{u} = err; }}")
                    else:
                        t.append(f"{{ {err} if (err.x < best{u}.x) {{ best{u}.x = err.x; k0_{u} = {c}; }} if (err.y < best{u}.y) {{ best{u}.y = err.y; k1_{u} = {c}; }} }}")
            for u in range(S):
                t.append(f"n{g}_{u} = best{u};")
            t.append(f"if (!({exp} & 2) && (pre_state >= 0 || ri < L / 2)) {{")
            for u in range(0, S, 2):
                if S == 1:
                    t.append(f"((device ushort*) H)[((size_t) ri * EDGES + out{g}_0) >> 1] = (ushort) k0_0 | ((ushort) k1_0 << 8);")
                else:
                    t.append(f"((device uint*) H)[((size_t) ri * EDGES + out{g}_{u}) >> 2] = (uint) k0_{u} | ((uint) k1_{u} << 8) | ((uint) k0_{u+1} << 16) | ((uint) k1_{u+1} << 24);")
            t.append("}")
            t.append("}")
        t.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
        for g in range(NG):
            for u in range(S):
                t.append(f"((threadgroup half2*) row)[(thr + {g} * NT) * {S} + {u}] = n{g}_{u};")
        t.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
        return "\n            ".join(t)

    def argmin():
        t = ["uint best = 0x7c00ffffu;"]
        for g in range(NG):
            for u in range(S):
                for h, comp in ((0, "x"), (1, "y")):
                    t.append(f"{{ uint e = (uint) out{g}_{u} + {h}u; uint v = e & 1023u; uint rank = (brev5(v >> 5) << 10) | (brev5(v & 31u) << 5) | (e >> 10);"
                             f" best = min(best, ((uint) as_type<ushort>(n{g}_{u}.{comp}) << 16) | rank); }}")
        return "\n            ".join(t)

    return f"""
    constexpr int K = {K}; constexpr int Kr = {Kr}; constexpr int EDGES = {edges};
    constexpr int L = {L}; constexpr int NT = {NT};
    uint tile = threadgroup_position_in_grid.x;
    uint thr = thread_position_in_threadgroup.x;
    uint lane = thr & 31u, wid = thr >> 5;
    threadgroup half row[EDGES];
    threadgroup uint* rowu = (threadgroup uint*) row;
    device uchar* H = hist + (size_t) tile * L * EDGES;
    {decl}
    int end_state = 0;
    for (int pass = 0; pass < 2; ++pass) {{
        const int roll = pass == 0 ? L / 2 : 0;
        const int pre_state = pass == 0 ? -1 : end_state;
        {{
            const int ri = roll;
            const half2 w2 = half2((half) tiles[(size_t) tile * L + ri]);
            {step(True)}
        }}
        for (int i = 1; i < L; ++i) {{
            int ri = i + roll; if (ri >= L) ri -= L;
            const half2 w2 = half2((half) tiles[(size_t) tile * L + ri]);
            {step(False)}
        }}
        int edge;
        if (pass == 0) {{
            {argmin()}
            best = simd_min(best);
            if (lane == 0) rowu[wid] = best;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (thr == 0) {{
                uint b = 0x7c00ffffu;
                for (int j = 0; j < NT / 32; ++j) b = min(b, rowu[j]);
                uint rank = b & 65535u;
                uint v = (brev5(rank >> 10) << 5) | brev5((rank >> 5) & 31u);
                edge = b >= 0x7c000000u ? 0 : (int) (((rank & 31u) << 10) | v);
                for (int i = L - 1; i >= 0 && !({exp} & 1); --i) {{
                    int ri = i + roll; if (ri >= L) ri -= L;
                    int kk = (int) H[(size_t) ri * EDGES + edge];
                    edge = ((kk << Kr) | edge) >> K;
                    if (ri == 0) break;
                }}
                rowu[32] = (uint) edge;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            end_state = (int) rowu[32];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }} else {{
            if (thr == 0) {{
                edge = end_state;
                for (int i = L - 1; i >= 0 && !({exp} & 1); --i) {{
                    int kk = (int) H[(size_t) i * EDGES + edge];
                    int prev = ((kk << Kr) | edge) >> K;
                    int enc = (prev << K) | edge;
                    edge = prev;
                    idx[(size_t) tile * L + i] = (ushort) enc;
                    wq[(size_t) tile * L + i] = (float) dec_mul1((uint) enc);
                }}
            }}
        }}
    }}
    """


_kern: dict = {}
_tab = []


def table2():
    """mul1 codebook as half2 pairs (state 2j, 2j+1) packed in a uint32 [32768]; only read when CFG['table']."""
    if not _tab:
        st = np.arange(65536, dtype=np.uint64)
        p = (st * np.uint64(MUL1)) & np.uint64(0xFFFFFFFF)
        bsum = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24)
        r = (np.uint16(0x6400) + bsum.astype(np.uint16)).view(np.float16).astype(np.float64)
        kinv = np.array([0x1EEE], np.uint16).view(np.float16)[0].astype(np.float64)
        kb = np.array([0xC931], np.uint16).view(np.float16)[0].astype(np.float64)
        v = (r * kinv + kb).astype(np.float16)
        _tab.append(mx.array(np.ascontiguousarray(v.reshape(-1, 2)).view(np.uint32).reshape(-1)))
    return _tab[0]
CFG = {"NT": 256, "table": True, "exp": 0, "impl": 2, "NT2": 512, "S2": 0}


def quantize_tiles(tiles: np.ndarray | mx.array, K: int, chunk: int | None = None, NT: int | None = None):
    """tiles: float32 [N, 256] in the kernel's (tensor-core-permuted) element order.
    Returns (wq float32 [N, 256], idx uint16 [N, 256]) as mx arrays."""
    v2 = CFG["impl"] == 2 and 2 <= K <= CFG.get("v2max", 5)
    NT = NT or (min(CFG["NT2"], 32768 >> K) if v2 else CFG["NT"])
    key = (K, NT, CFG["table"], CFG["exp"], v2, CFG["S2"])
    if key not in _kern:
        src = _source2(K, NT, CFG["exp"], CFG["S2"]) if v2 else _source(K, NT, CFG["table"], CFG["exp"])
        _kern[key] = mx.fast.metal_kernel(
            name=f"kyojin_conv_vit_k{K}_t{NT}_{int(CFG['table'])}_{CFG['exp']}_{int(v2)}_{CFG['S2']}", header=HEADER, input_names=["tiles", "tab2"],
            output_names=["idx", "wq", "costs", "hist"], source=src)
    tiles = mx.array(tiles) if isinstance(tiles, np.ndarray) else tiles
    n = tiles.shape[0]
    edges = 65536 >> K
    chunk = chunk or max(1, min(256 if K < 6 else 512, (192 << 20) // (L * edges)))      # K6 (head): 512 tiles per dispatch is 5 % faster
    outs_w, outs_i = [], []
    for s in range(0, n, chunk):
        t = tiles[s:s + chunk]
        m = t.shape[0]
        idx, wq, _, _ = _kern[key](
            inputs=[t, table2()], grid=(m * NT, 1, 1), threadgroup=(NT, 1, 1),
            output_shapes=[(m, L), (m, L), (m * 2 * edges,), (m * L * edges,)],
            output_dtypes=[mx.uint16, mx.float32, mx.float16, mx.uint8])
        mx.eval(idx, wq)
        outs_w.append(wq); outs_i.append(idx)
    return (mx.concatenate(outs_w) if len(outs_w) > 1 else outs_w[0],
            mx.concatenate(outs_i) if len(outs_i) > 1 else outs_i[0])
