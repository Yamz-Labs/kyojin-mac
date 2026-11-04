#!/usr/bin/env python3
"""Per-shape k-split (tgk) sweep of the fused EXL3 1-row GEMV: dependent chain of a pair (kin->nout, nout->kin), distinct weights per call, interleaved.
usage: exl3_tgk_sweep.py --pair 2560,9216 [--K 4] [--tgks 1,2,4,8,16,32] [--pool 12]
Prints us per linear (pair/2) per (tgk_fwd, tgk_back) variant; default = engine pick_tgk. Roof = stream read of the same bytes (121 GB/s on the Air)."""
import argparse, statistics, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import kernels as KN, metal_src as S
from exl3_roof import synth, make_source, HDR_EXTRA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="2560,9216"); ap.add_argument("--K", type=int, default=4); ap.add_argument("--tgks", default="1,2,4,8,16,32")
    ap.add_argument("--pool", type=int, default=12); ap.add_argument("--reps", type=int, default=7); ap.add_argument("--chain", type=int, default=12)
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    kin, nout = (int(v) for v in a.pair.split(","))
    K = a.K
    fw = [synth(kin, nout, K, 10 + i) for i in range(a.pool)]; bk = [synth(nout, kin, K, 100 + i) for i in range(a.pool)]
    mx.eval(*[l._words for l in fw + bk])
    x0 = mx.array(np.random.default_rng(5).standard_normal((1, kin)).astype(np.float16)); mx.eval(x0)
    cache = {}
    def call(l, x, t):
        g = l._g[1]
        key = (g.kt, g.nt, t)
        if key not in cache:
            cache[key] = mx.fast.metal_kernel(name=f"sw_{g.kt}_{g.nt}_{K}_{t}", header=S.HEADER, input_names=["x", "suh", "svh", "trellis", "params", "ctr"],
                                              output_names=["part", "y"], source=S.fused_source.__wrapped__(K, 1, "mul1", True, False, S.fused_dims(g.kt, t), True))
        params = mx.array([g.kt, g.nt, t, 0], dtype=mx.uint32)
        return cache[key](inputs=[x, g._fsuh, g._fsvh, g.words, params, g._fctr], template=[("BITS", K), ("MR", 1), ("OT", mx.float16)],
                          grid=(g.blocks * 256, t, 1), threadgroup=(256, 1, 1), output_shapes=[(t, 1, g.n), (1, g.n)], output_dtypes=[mx.float32, mx.float16])[1]
    def legal(l, t):
        g = l._g[1]
        return t & (t - 1) == 0 and g.can_fuse(1) if t == g.tgk else (t & (t - 1) == 0 and max(S.fused_dims(g.kt, t), 512) <= 7000 and g.kt >= 4 * t)
    d_f, d_b = fw[0]._g[1].tgk, bk[0]._g[1].tgk
    var = {"default": (d_f, d_b)}
    for t in (int(v) for v in a.tgks.split(",")):
        if legal(fw[0], t) and t != d_f: var[f"fw{t}"] = (t, d_b)
        if legal(bk[0], t) and t != d_b: var[f"bk{t}"] = (d_f, t)
    nb = fw[0]._g[1].words.size * 4
    rd = mx.fast.metal_kernel(name="roof_rd2", input_names=["w"], output_names=["o"],
                              source="uint t = thread_position_in_grid.x; uint nt = threads_per_grid.x; device const uint4* p = (device const uint4*)w; uint4 acc = uint4(0);"
                                     "for (uint i = t; i < N4; i += nt) acc ^= p[i]; o[t] = acc.x ^ acc.y ^ acc.z ^ acc.w;")
    def roof_once(n):
        return [rd(inputs=[(fw if i % 2 == 0 else bk)[(i // 2) % a.pool]._g[1].words], template=[("N4", nb // 16)], grid=(32768, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(32768,)], output_dtypes=[mx.uint32])[0] for i in range(n * 2)]
    def once(v, n):
        x = x0
        for i in range(n):
            x = call(bk[i % a.pool], call(fw[i % a.pool], x, v[0]), v[1])
        return x
    for v in var.values(): mx.eval(once(v, 2))
    mx.eval(roof_once(2))
    T = {n: [] for n in var}; T["roof"] = []
    for rep in range(a.reps):
        for n, v in var.items():
            t0 = time.perf_counter(); mx.eval(once(v, a.chain)); T[n].append((time.perf_counter() - t0) / a.chain * 1e6 / 2)
        t0 = time.perf_counter(); mx.eval(roof_once(a.chain)); T["roof"].append((time.perf_counter() - t0) / a.chain * 1e6 / 2)
    r = min(T["roof"])
    print(f"pair {kin}x{nout} K={K} bytes {nb/1e6:.2f} MB, default tgk fw {d_f} bk {d_b}; read roof {r:.1f} us/linear ({nb/r/1e3:.0f} GB/s)")
    for n in sorted(T, key=lambda k: min(T[k])):
        print(f"  {n:8s} min {min(T[n]):7.1f} med {statistics.median(T[n]):7.1f}  {100*r/min(T[n]):.0f} % of roof")
if __name__ == "__main__":
    main()
