#!/usr/bin/env python3
"""EXL3 fused-GEMV vs the bandwidth roof on the Air: per-op table with ablations. Distinct weights per call (pool, no SLC reuse).
usage: exl3_roof.py [--rows 1] [--variants lin,full,dec1,load] [--pool 16] [--reps 9] [--dep] [--sustain 0]
 --dep      dependent chain mlp_up(2560->9216) -> mlp_down(9216->2560): every kernel waits for the previous one (what decode does)
 variants   source transforms of the fused kernel (metal_src.fused_source), all timed in ONE process, interleaved ABAB:
            roof = stream-read of the same bytes (independent calls only); lin = the engine path lin(x); full = the fused kernel launched here;
            dec1 = decode replaced by a 1-op stand-in; load = loads only (no decode, no FMA)
 --sustain  also run each variant for N seconds back to back and print the time at the start and the end (thermal / power throttle)"""
import argparse, statistics, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import kernels as KN, metal_src as S, exl3_linear as EL

_orig_rdl = S.run_decode_lines
HDR_EXTRA = "\ninline half dec_null(uint s) { return as_type<half>(ushort(s)); }\n"


def synth(kin, nout, K, seed, cb="mul1"):
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    return EL.Exl3Linear(tr, f16(kin), f16(nout), cb=cb)


def make_source(variant, K, R, xsr):
    def rdl(K_, emit, cb="mul1"):
        lines = _orig_rdl(K_, emit, "null" if variant == "dec1" else cb)
        if variant == "load":
            return ["a0_lo0 += float(" + " ^ ".join(f"w{i}" for i in range(K_)) + ");"]
        return lines
    S.run_decode_lines = rdl
    try:
        return S.fused_source.__wrapped__(K, R, "mul1", True, False, xsr, True)
    finally:
        S.run_decode_lines = _orig_rdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1); ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--reps", type=int, default=9); ap.add_argument("--chain", type=int, default=16); ap.add_argument("--variants", default="lin,full,dec1,load")
    ap.add_argument("--dep", action="store_true"); ap.add_argument("--sustain", type=float, default=0); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--tgk", type=int, default=0)
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    R, K = a.rows, a.K
    ups = [synth(2560, 9216, K, 10 + i) for i in range(a.pool)]
    dns = [synth(9216, 2560, K, 100 + i) for i in range(a.pool)]
    mx.eval(*[l._words for l in ups + dns])
    nbytes = ups[0]._g[1].words.size * 4
    rng = np.random.default_rng(5)
    x0 = mx.array(rng.standard_normal((R, 2560)).astype(np.float16)); mx.eval(x0)
    tg = lambda l: a.tgk or l._g[1].tgk
    print(f"K={K} rows={R} bytes/linear={nbytes/1e6:.2f} MB tgk up={tg(ups[0])} down={tg(dns[0])} pool={a.pool} dep={a.dep}", flush=True)

    def mkfun(name):
        if name == "roof":
            n4 = nbytes // 16
            rd = mx.fast.metal_kernel(name="roof_rd", input_names=["w"], output_names=["o"],
                                      source="uint t = thread_position_in_grid.x; uint nt = threads_per_grid.x; device const uint4* p = (device const uint4*)w; uint4 acc = uint4(0);"
                                             "for (uint i = t; i < N4; i += nt) acc ^= p[i]; o[t] = acc.x ^ acc.y ^ acc.z ^ acc.w;")
            return lambda l, x: rd(inputs=[l._g[1].words], template=[("N4", n4)], grid=(32768, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(32768,)], output_dtypes=[mx.uint32])[0]
        if name == "lin":
            return lambda l, x: l(x)
        if name == "gf":                                 # Mul1Gemv.fused directly (no Exl3Linear.__call__ around it)
            return lambda l, x: l._g[1].fused(x)
        if name == "gf0":                                # same, one counter set (what the ablation kernels use)
            def f(l, x):
                l._g[1]._fnext = 0
                return l._g[1].fused(x)
            return f
        if name in ("lin1", "lin2"):                     # engine path with the rot fused into the GEMV launch (lin1) / a separate rot launch (lin2)
            def f(l, x):
                old = KN.ROT2_MIN_BLOCKS
                KN.ROT2_MIN_BLOCKS = 10 ** 6 if name == "lin1" else 1
                try: return l(x)
                finally: KN.ROT2_MIN_BLOCKS = old
            return f
        cache = {}
        def f(l, x):
            g = l._g[1]
            key = (g.kt, g.nt)
            t = tg(l)
            if key not in cache:
                src = make_source(name, K, R, S.fused_dims(g.kt, t))
                cache[key] = mx.fast.metal_kernel(name=f"rv_{name}_{g.kt}_{g.nt}_{K}_{R}_{t}", header=S.HEADER + HDR_EXTRA,
                                                  input_names=["x", "suh", "svh", "trellis", "params", "ctr"], output_names=["part", "y"], source=src)
            params = mx.array([g.kt, g.nt, t, 0], dtype=mx.uint32)
            return cache[key](inputs=[x, g._fsuh, g._fsvh, g.words, params, g._fctr], template=[("BITS", K), ("MR", R), ("OT", mx.float16)],
                              grid=(g.blocks * 256, t, 1), threadgroup=(256, 1, 1), output_shapes=[(t, R, g.n), (R, g.n)], output_dtypes=[mx.float32, mx.float16])[1]
        return f

    names = [n for n in (["roof"] if not a.dep else []) + a.variants.split(",") if n != "roof" or not a.dep]
    names = list(dict.fromkeys(names))
    fs = {n: mkfun(n) for n in names}

    def once(f, n):
        if a.dep:
            x = x0
            for i in range(n):
                x = f(dns[i % a.pool], f(ups[i % a.pool], x))
            return x
        return [f(ups[i % a.pool], x0) for i in range(n)]

    per = 2 if a.dep else 1
    for n in names:
        mx.eval(once(fs[n], 2))
    T = {n: [] for n in names}
    for rep in range(a.reps):
        for n in names:
            t0 = time.perf_counter()
            mx.eval(once(fs[n], a.chain))
            T[n].append((time.perf_counter() - t0) / a.chain * 1e6 / per)
    for n in names:
        print(f"  {n:6s} min {min(T[n]):7.1f} med {statistics.median(T[n]):7.1f} us per linear", flush=True)
    if "roof" in T:
        r = min(T["roof"])
        for n in names:
            print(f"  {n:6s} {100 * r / min(T[n]):.0f} % of read roof ({nbytes / min(T[n]) / 1e3:.1f} GB/s at min)")
    if a.sustain:
        for n in names:
            ts = []
            t_end = time.perf_counter() + a.sustain
            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                mx.eval(once(fs[n], a.chain))
                ts.append((time.perf_counter() - t0) / a.chain * 1e6 / per)
            k = max(1, len(ts) // 5)
            print(f"  sustain {n:6s} first {statistics.median(ts[:k]):7.1f} last {statistics.median(ts[-k:]):7.1f} us ({len(ts)} blocks)", flush=True)


if __name__ == "__main__":
    main()
