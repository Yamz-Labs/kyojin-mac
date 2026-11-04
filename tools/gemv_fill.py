#!/usr/bin/env python3
"""Threadgroup fill of the 1-row decode GEMV: what the k-split picks, and what a filled split costs.

The decode GEMV (kernels.Mul1Gemv, fused_group members, grouped wo_a, moe.MoeGather) k-splits each 128-column block across `tgk` threadgroups, so one
launch has `blocks * tgk` threadgroups (blocks = nout / 128). The old default filled to 512 threadgroups but kept >= 8 k-tiles per warp, which caps tgk at
kt / 8: on a wide GPU the small linears of a decode step then launch 16-160 threadgroups and their bandwidth follows the threadgroup count (64 cores,
4-bit, dependent chain, measured by the lead: 32 threadgroups 126 GB/s, 128 345, 512 897 of a ~1000 GB/s roof).

usage:
  gemv_fill.py --host                  # per shape: tgk / threadgroups of the old constants vs the device fill (no GPU, safe anywhere)
  gemv_fill.py --host --cores 64       # the same table for another GPU
  gemv_fill.py --bench                 # Air: per shape, the tgk sweep, 8 dependent calls or 8 in flight, vs the read roof
  gemv_fill.py --bench --shapes wkv,wq_a --dep 0
"""
import argparse
import json
import time

# DS4.1 dense shapes (hidden 5120, q_lora 1280, 64 x 512 heads, 8 x 1024 o, moe_inter 2304, index 32 x 128) plus the two shapes of the Ultra's
# dependent-chain probe (4096x4096, 4096x1024).
SHAPES = [
    ("wq_a", 5120, 1280),
    ("wkv", 5120, 512),
    ("wq_b", 1280, 32768),
    ("wo_b", 8192, 5120),
    ("wo_a_g", 4096, 1024),
    ("sh_gate", 5120, 2304),
    ("sh_down", 2304, 5120),
    ("idx_q", 1280, 4096),
    ("head", 5120, 129280),
    ("probe4096", 4096, 4096),
    ("probe1024", 4096, 1024),
]
OLD = (512, 8)          # the heuristic's constants, measured on the 10-core Air (KYOJIN_FILL=0 restores them)

# the dense launches of one DS4.1 decoder layer: (name, [(kin, nout, members in this launch), ...])
LAUNCHES = [
    ("wq_a + wkv (one launch)", [(5120, 1280), (5120, 512)]),
    ("wq_b", [(1280, 32768)]),
    ("wo_a (8 groups, one launch)", [(4096, 1024, 8)]),
    ("wo_b", [(8192, 5120)]),
    ("shared gate + up (one launch)", [(5120, 2304), (5120, 2304)]),
    ("shared down", [(2304, 5120)]),
    ("indexer q", [(1280, 4096)]),
]


def _picks(kin: int, nout: int):
    from kyojin_mlx import kernels as KN
    b, kt = nout // 128, kin // 16
    return KN.pick_tgk(kt, b, 4, OLD[0], OLD[1]), KN.pick_tgk(kt, b, 4)


def host(cores, per_core, tiles) -> None:
    from kyojin_mlx import kernels as KN, tuning as TN
    if per_core is not None:
        TN.FILL_PER_CORE = per_core
    if tiles is not None:
        TN.FILL_MIN_TILES = str(tiles)
    keep, keep_fn = TN._chip, TN.gpu_cores
    if cores is not None:
        TN._chip = None
        TN.gpu_cores = lambda: cores                       # emulate another GPU's table
    try:
        print(f"# {TN.gpu_cores()} GPU cores, KYOJIN_FILL={int(TN.FILL)}, fill target {TN.fill_tgs()} threadgroups, >= {TN.fill_min_tiles()} k-tiles/warp")
        print("| launch (threadgroups per launch) | shapes | old tgk / TGs | fill tgk / TGs | x |")
        print("|---|---|---|---|---|")
        # DS4.1: wq_a + wkv share one launch (fused_group), the shared gate / up too, wo_a runs its 8 groups in one grid
        for name, members in LAUNCHES:
            o = f = 0
            os_, fs_ = [], []
            for m in members:
                kin, nout, n = m[0], m[1], (m[2] if len(m) > 2 else 1)
                t0 = KN.pick_tgk(kin // 16, nout // 128, 4, OLD[0], OLD[1])
                t1 = KN.pick_tgk(kin // 16, nout // 128, 4, max(1, TN.fill_tgs() // n), TN.fill_min_tiles())
                o += n * (nout // 128) * t0
                f += n * (nout // 128) * t1
                os_.append(f"{kin}x{nout}" + (f" x{n}" if n > 1 else ""))
                fs_.append(f"{t0}->{t1}" if n == 1 else f"{t0}->{t1} x{n}")
            print(f"| {name} | {', '.join(os_)} | {o} | {f} | {f / o:.1f} |")
        print(f"\n| tgk chosen per shape ({TN.gpu_cores()} cores) | kt | blocks | tgk | TGs | tiles/warp |")
        print("|---|---|---|---|---|---|")
        for n, kin, nout in SHAPES:
            kt, b = kin // 16, nout // 128
            t = KN.pick_tgk(kt, b, 4, None, None)
            print(f"| {n} {kin}x{nout} | {kt} | {b} | {t} | {b * t} | {-(-kt // (4 * t))} |")
    finally:
        TN._chip, TN.gpu_cores = keep, keep_fn


def bench(shapes, K: int, tgks, reps: int, pool: int, dep: bool, jsonf) -> None:
    import numpy as np
    import mlx.core as mx
    from kyojin_mlx import kernels as KN, metal_src as S, tuning as TN
    from exl3_roof import synth

    mx.set_cache_limit(512 << 20)
    print(f"# mlx {mx.__version__} | {TN.gpu_cores()} GPU cores, fill {TN.fill_tgs()} TGs, >= {TN.fill_min_tiles()} tiles | K={K}, pool {pool}, reps {reps}")
    out = {"cores": TN.gpu_cores(), "fill_tgs": TN.fill_tgs(), "min_tiles": TN.fill_min_tiles(), "K": K, "rows": []}
    kcache: dict = {}

    def call_of(lin, t, x):
        g = lin._g[1]
        key = (g.kt, g.nt, K, t)
        if key not in kcache:
            kcache[key] = mx.fast.metal_kernel(
                name=f"fill_{g.kt}_{g.nt}_{K}_{t}", header=S.HEADER, input_names=["x", "suh", "svh", "trellis", "params", "ctr"],
                output_names=["part", "y"], source=S.fused_source.__wrapped__(K, 1, "mul1", True, False, S.fused_dims(g.kt, t), True))
        params = mx.array([g.kt, g.nt, t, 0], dtype=mx.uint32)
        return kcache[key](inputs=[x, g._fsuh, g._fsvh, g.words, params, g._fctr], template=[("BITS", K), ("MR", 1), ("OT", mx.float16)],
                           grid=(g.blocks * 256, t, 1), threadgroup=(256, 1, 1), output_shapes=[(t, 1, g.n), (1, g.n)],
                           output_dtypes=[mx.float32, mx.float16])[1]

    def legal(t, kt):
        return t & (t - 1) == 0 and t <= kt // 4 and max(S.fused_dims(kt, t), 512) <= KN.Mul1Gemv.FUSED_SMEM_FLOATS

    for name, kin, nout in shapes:
        nb = kin * nout * K / 8
        p = max(1, min(pool, int(400e6 // max(nb, 1))))
        fw = [synth(kin, nout, K, 10 + i) for i in range(p)]
        bwd = [synth(nout, kin, K, 100 + i) for i in range(p)] if (dep and nb * p < 64e6) else None
        mx.eval(*[l._words for l in fw + (bwd or [])])
        x0 = mx.array(np.random.default_rng(5).standard_normal((1, kin)).astype(np.float16))
        mx.eval(x0)
        old_t, fill_t = _picks(kin, nout)
        b = nout // 128
        var = sorted({t for t in tgks if legal(t, kin // 16)} | {old_t, fill_t})
        # the read roof of the same bytes: one launch, no decode, 8 calls
        nbytes = fw[0]._g[1].words.size * 4
        rd = mx.fast.metal_kernel(name=f"fill_roof_{nbytes}", input_names=["w"], output_names=["o"],
                                  source="uint t = thread_position_in_grid.x; device const uint4* p = (device const uint4*)w; uint4 a = uint4(0);"
                                         "for (uint i = t; i < N4; i += threads_per_grid.x) a ^= p[i]; o[t] = a.x ^ a.y ^ a.z ^ a.w;")

        def roof():
            return [rd(inputs=[fw[i % p]._g[1].words], template=[("N4", nbytes // 16)], grid=(32768, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(32768,)], output_dtypes=[mx.uint32])[0] for i in range(8)]

        def one(t, n=8):
            y = None
            for i in range(n):
                y = call_of(fw[i % p], t, x0)
            return y

        def chained(t, n=8):
            x = x0
            for i in range(n):
                x = call_of(bwd[i % p], t, call_of(fw[i % p], t, x))
            return x

        fn = chained if bwd else one
        for t in var:
            mx.eval(fn(t, 1))
        mx.eval(roof())
        T = {t: [] for t in var}
        R = []
        for _ in range(reps):
            for t in var:
                t0 = time.perf_counter()
                mx.eval(fn(t))
                T[t].append((time.perf_counter() - t0) / (8 * (2 if bwd else 1)) * 1e6)     # one linear (the chain runs fw + its reverse)
            t0 = time.perf_counter()
            mx.eval(roof())
            R.append((time.perf_counter() - t0) / 8 * 1e6)
        r = min(R)
        us = {t: min(v) for t, v in T.items()}
        row = dict(shape=name, kin=kin, nout=nout, MB=round(nb / 1e6, 2), blocks=b, roof_us=round(r, 1), roof_GBs=round(nb / r / 1e3),
                   old_tgk=old_t, old_tgs=b * old_t, fill_tgk=fill_t, fill_tgs=b * fill_t, variants={})
        print(f"\n## {name} {kin}x{nout} K={K} {nb / 1e6:.2f} MB, {b} blocks | read roof {r:.1f} us ({nb / r / 1e3:.0f} GB/s) | "
              f"old tgk {old_t} ({b * old_t} TGs) -> fill tgk {fill_t} ({b * fill_t} TGs) | {'dependent chain of 8' if bwd else '8 in flight'}")
        print("| tgk | TGs | us/linear | GB/s | % roof | vs old |")
        print("|---|---|---|---|---|---|")
        for t in var:
            v = us[t]
            row["variants"][t] = dict(tgs=b * t, us=round(v, 1), GBs=round(nb / v / 1e3), pct=round(100 * nb / v / 1e3 / row["roof_GBs"]),
                                      rel=round(v / us[old_t], 3))
            tag = " old" if t == old_t else (" fill" if t == fill_t else "")
            print(f"| {t}{tag} | {b * t} | {v:.1f} | {nb / v / 1e3:.0f} | {100 * nb / v / 1e3 / row['roof_GBs']:.0f} | {v / us[old_t]:.2f} |")
        out["rows"].append(row)
        del fw, bwd, x0
        mx.clear_cache()
    if jsonf:
        json.dump(out, open(jsonf, "w"), indent=1)
        print(f"\nwrote {jsonf}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="store_true", help="tgk / threadgroup table only (no GPU)")
    ap.add_argument("--bench", action="store_true", help="Metal: sweep tgk per shape (Air only)")
    ap.add_argument("--shapes", default="", help="comma list of names (default: all), extra kinxnout pairs allowed")
    ap.add_argument("--tgks", default="1,2,4,8,16,32,64")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--pool", type=int, default=3)
    ap.add_argument("--dep", type=int, default=1, help="dependent chain of 8 (1) or 8 calls in flight (0)")
    ap.add_argument("--cores", type=int, default=None, help="--host: the core count to emulate")
    ap.add_argument("--per-core", type=int, default=None)
    ap.add_argument("--tiles", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    shapes = SHAPES
    if a.shapes:
        want = [s.strip() for s in a.shapes.split(",") if s.strip()]
        shapes = [s for s in SHAPES if s[0] in want] + [(s, *map(int, s.split("x"))) for s in want if "x" in s]
    if a.host or not a.bench:
        host(a.cores, a.per_core, a.tiles)
    if a.bench:
        bench(shapes, a.K, [int(v) for v in a.tgks.split(",")], a.reps, a.pool, bool(a.dep), a.json)


if __name__ == "__main__":
    main()
