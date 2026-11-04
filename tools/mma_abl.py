#!/usr/bin/env python3
"""MMA-GEMV partial kernel ablations on the dependent up->down chain (kernel only): full, nomem (weights re-read from one tile row: L1/L2 hits),
nodec (dec_null), nomma (no simdgroup_multiply_accumulate; keeps decode), pf (register prefetch of the next k-step words).
usage: mma_abl.py [--rows 5] [--variants full,nomem,nodec,nomma]"""
import argparse, statistics, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import kernels as KN, mma_gemv as MM, metal_src as S
from exl3_roof import synth

S.HEADER += "\ninline half dec_null(uint s) { return as_type<half>(ushort(s)); }\n"

def src_for(name, K, cb):
    fl = name.split("_")
    s = MM.mma_gemv_source(K, "null" if "nodec" in fl else cb)
    if "nomem" in fl:
        s = s.replace("tp += tstride;", "")
        s = s.replace("const device uint* tp = trellis + (ulong(ks0) * NT + nt0) * TW;", "const device uint* tp = trellis + (ulong(blk & 7) * 8 * TW);")
    if "nomma" in fl:
        s = s.replace("simdgroup_multiply_accumulate(acc[t][0][r], A, B[0][r], acc[t][0][r])", "acc[t][0][r].thread_elements()[0] += A.thread_elements()[0]")
        import re
        s = re.sub(r"simdgroup_multiply_accumulate\(acc\[t\]\[(\d)\]\[r\], A, B\[(\d)\]\[r\], acc\[t\]\[\d\]\[r\]\)", r"acc[t][\1][r].thread_elements()[0] += A.thread_elements()[0] + A.thread_elements()[1]", s)
    if "unr2" in fl:
        a_ = "for (uint i = 0; i < myn; ++i) {"
        assert a_ in s
        s = s.replace(a_, '_Pragma("clang loop unroll_count(2)") ' + a_)
    if "pf" in fl:
        s = pf_source(s, K)
    if "half" in fl:
        for a_, b_ in (("simdgroup_matrix<float, 8, 8> B[2][RB];", "simdgroup_matrix<half, 8, 8> B[2][RB];"), ("simdgroup_matrix<float, 8, 8> A;", "simdgroup_matrix<half, 8, 8> A;"),
                       ("= float(d0);", "= d0;"), ("= float(d1);", "= d1;"), ("const device float* xk", "const device half* xk")):
            assert a_ in s, a_
            s = s.replace(a_, b_)
    return s

def pf_source(s, K):
    """register prefetch: the 4 tiles' words (u0..u{nu-1}) of step i+1 are loaded before step i is decoded"""
    import re
    nvw = ((7 * K + 15) >> 5) + 1; nu = nvw + 1
    # split the LOAD part off: lines 'const uint u<j> = tt[...];' inside the t loop
    m = re.findall(r"( +const uint u(\d+) = (tt\[[^;]*\]);)", s)
    assert len(m) == nu, (len(m), nu)
    for full, j, ex in m:
        s = s.replace(full, f"            const uint u{j} = pw[t][{j}];")
    decl = "uint pw[4][%d];\n    auto LDW = [&](const device uint* base) {\n" % nu
    # load code for next step, as a macro-like block
    blk = "".join(f"        {{ const device uint* tt = base + t * TW; _Pragma(\"clang loop unroll(full)\") for (uint t_ = 0; t_ < 0; ++t_) {{}} }}\n" for _ in range(0))
    loads = []
    for j in range(nu):
        ex = "tt[i0]" if j == 0 else f"tt[min(uint(ir + {j}), ilast)]"
        loads.append(f"pw[t][{j}] = {ex};")
    ld = "_Pragma(\"clang loop unroll(full)\") for (uint t = 0; t < 4; ++t) { const device uint* tt = BASE + t * TW; " + " ".join(loads) + " }"
    pre = "uint pw[4][%d];\n    { const device uint* BASEP = tp; " % nu + ld.replace("BASE", "BASEP") + " }\n"
    s = s.replace("    const device float* xk = xh + ks0 * 16 + fm;\n", "    const device float* xk = xh + ks0 * 16 + fm;\n    " + pre, 1)
    nxt = "uint pn[4][%d];\n        if (i + 1 < myn) { const device uint* BASEN = tp + tstride; " % nu + ld.replace("BASE", "BASEN").replace("pw[t]", "pn[t]") + " }\n"
    s = s.replace("        _Pragma(\"clang loop unroll(full)\") for (uint t = 0; t < 4; ++t) {\n            const device uint* tt = tp + t * TW;", "        " + nxt + "        _Pragma(\"clang loop unroll(full)\") for (uint t = 0; t < 4; ++t) {\n            const device uint* tt = tp + t * TW;", 1)
    s = s.replace("        tp += tstride;\n", "        if (i + 1 < myn) { for (uint t = 0; t < 4; ++t) for (uint j = 0; j < %d; ++j) pw[t][j] = pn[t][j]; }\n        tp += tstride;\n" % nu, 1)
    return s

def partial(name, g, xh, R, ws=4):
    kern = KN._kern(("mmaabl", name, g.K, g.cb), name=f"kyojin_mmaabl_{name}_k{g.K}", input_names=["xh", "trellis", "params"], output_names=["part"], source=src_for(name, g.K, g.cb))
    return kern(inputs=[xh, g.words, g.params], template=[("BITS", g.K), ("MR", R), ("WSP", ws)], grid=(g.blocks * 64 * ws, g.tgk * 4 // ws, 1), threadgroup=(64 * ws, 1, 1),
                output_shapes=[(g.tgk * 4 // ws, R, g.n)], output_dtypes=[mx.float32])[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5); ap.add_argument("--pool", type=int, default=12); ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--chain", type=int, default=12); ap.add_argument("--variants", default="full,nomem,nodec,nomma")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    R = a.rows
    ups = [synth(2560, 9216, 4, 10 + i) for i in range(a.pool)]; dns = [synth(9216, 2560, 4, 100 + i) for i in range(a.pool)]
    mx.eval(*[l._words for l in ups + dns])
    x0 = mx.array(np.random.default_rng(5).standard_normal((R, 2560)).astype(np.float32)); mx.eval(x0)
    names = a.variants.split(",")
    def f(n, l, x):
        g = l._g[1]
        if x.shape[1] < g.kt * 16: x = mx.pad(x, [(0, 0), (0, g.kt * 16 - x.shape[1])])
        x = x.astype(mx.float16 if "half" in n else mx.float32)
        return partial(n, g, x, R)[0]
    def once(n, k):
        x = x0
        for i in range(k):
            h = f(n, ups[i % a.pool], x)
            y = f(n, dns[i % a.pool], h)
            x = y[:, :2560]
        return x
    for n in names: mx.eval(once(n, 2))
    T = {n: [] for n in names}
    for rep in range(a.reps):
        for n in names:
            t0 = time.perf_counter(); mx.eval(once(n, a.chain)); T[n].append((time.perf_counter() - t0) / a.chain * 1e6 / 2)
    print(f"R={R}")
    for n in names: print(f"  {n:7s} min {min(T[n]):7.1f} med {statistics.median(T[n]):7.1f} us per linear", flush=True)
main()
