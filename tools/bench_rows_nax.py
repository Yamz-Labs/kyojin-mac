#!/usr/bin/env python3
"""D4: cost of one DS4.1 dense EXL3 linear at 1..32 rows, today's GEMV launches (<= 4 rows each, concatenated) vs the fused trellis-decode +
NAX matmul (metal_nax), on synthetic trellises of the real shapes and bitrates (timing does not depend on the weights).
Each timing is a chain of CHAIN independent calls inside one mx.eval (decode steps queue their launches), min of REPS, ms per call.
   PYTHONPATH=$PWD:$PWD/bench:$PWD/tools python tools/bench_rows_nax.py [--shapes wq_a,head] [--rows 1,2,4,8,16,32] [--cfgs auto|all]"""
import argparse, json, sys, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN, metal_nax as NX

# (name, kin, nout, K) : DS4.1 tensors from the ds2m header (hidden 5120, q_lora 1280, 64 heads x 512, vocab 129280)
SHAPES = [("wq_a", 5120, 1280, 5), ("wq_b", 1280, 32768, 5), ("wkv", 5120, 512, 5), ("wo_b", 8192, 5120, 5), ("idx.wq_b", 1280, 4096, 5),
          ("shared.w1", 5120, 2304, 5), ("shared.w2", 2304, 5120, 5), ("head", 5120, 129280, 6), ("main_proj", 15360, 5120, 4)]
CFGS = [(16, 1, 32, 32), (16, 1, 32, 64), (32, 1, 32, 32), (32, 1, 32, 64), (32, 2, 32, 64)]
CAP_GB = 2.8


def synth_linear(kin, nout, K, cb="mul1", seed=1) -> EL.Exl3Linear:
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    return EL.Exl3Linear(tr, f16(kin), f16(nout), cb=cb)


_aux: dict = {}


def nax_call(lin: EL.Exl3Linear, x2: mx.array, cfg, out_half=True):
    """rot (k permutation) -> fused decode + matmul2d -> finalize, the `Exl3Linear._dense` fused path with a chosen cfg / partial dtype."""
    rows = x2.shape[0]
    if lin._nax is None:
        lin._nax = (NX.SingleStack(lin._words, lin.K, lin._kt, lin._nt, lin.cb), mx.array([0], dtype=mx.int32))
    st, eids = lin._nax
    key = (id(lin), rows)
    ent = _aux.get(key)
    if ent is None:
        ent = _aux[key] = (mx.array([0, rows], dtype=mx.uint32), mx.array([1, lin.out_features, rows], dtype=mx.uint32),
                           KN.S.had_grid(lin.out_features // 128, rows))
    rowoff, fp, (grid, tg) = ent
    xh = KN.rot(x2, lin._suh, mx.float16, perm=True)
    inner = NX.expert_gemm((st,), eids, rowoff, xh, rows, rows, cfg, out_half).reshape(1, rows, lin.out_features)
    return KN.fin_kernel(lin.out_dtype, mx.float16 if out_half else mx.float32)(
        inputs=[inner, lin._svh, fp], template=[("OT", lin.out_dtype)], grid=grid, threadgroup=tg,
        output_shapes=[(rows, lin.out_features)], output_dtypes=[lin.out_dtype])[0]


def gemv_call(lin, x2):
    EL.ROW_SPLIT_MAX = 32
    EL.GEMV_MAX_ROWS = 4
    return lin(x2)


def timeit(fn, xs, reps, chain):
    mx.eval(*[fn(x) for x in xs[:2]])
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(*[fn(x) for x in xs[:chain]])
        best = min(best, (time.perf_counter() - t0) / chain)
    return best * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=""); ap.add_argument("--rows", default="1,2,4,8,16,32")
    ap.add_argument("--cfgs", default="all"); ap.add_argument("--reps", type=int, default=7); ap.add_argument("--chain", type=int, default=12)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    names = a.shapes.split(",") if a.shapes else [s[0] for s in SHAPES]
    rows_l = [int(v) for v in a.rows.split(",")]
    out = []
    for (name, kin, nout, K) in SHAPES:
        if name not in names:
            continue
        lin = synth_linear(kin, nout, K)
        mx.eval(lin._words)
        rng = np.random.default_rng(5)
        wbytes = lin._words.size * 4
        print(f"== {name}: {kin} x {nout} K={K}  ({wbytes/1e6:.1f} MB, {kin*nout/1e6:.1f} Mw)  GEMV roof @125 GB/s {wbytes/125e9*1e3:.3f} ms, ALU roof @270 Gw/s {kin*nout/270e9*1e3:.3f} ms", flush=True)
        for R in rows_l:
            xs = [mx.array(rng.standard_normal((R, kin)).astype(np.float16)) for _ in range(a.chain)]
            mx.eval(*xs)
            tg_ = timeit(lambda x: gemv_call(lin, x), xs, a.reps, a.chain)
            ref = np.array(gemv_call(lin, xs[0])).astype(np.float32)
            res = {"shape": name, "rows": R, "gemv_ms": tg_}
            cfgs = CFGS if a.cfgs == "all" else [NX.pick_cfg([R])]
            best = None
            line = f"  R={R:2d} GEMV {tg_:7.3f} ms |"
            for cfg in cfgs:
                if cfg[2] // 16 > 2 and lin._nt % (cfg[2] // 16):
                    continue
                t = timeit(lambda x: nax_call(lin, x, cfg, True), xs, a.reps, a.chain)
                res[f"nax_{cfg}"] = t
                if best is None or t < best[0]:
                    best = (t, cfg)
                line += f" {cfg[0]},{cfg[3]}:{t:7.3f}"
            t, cfg = best
            y = np.array(nax_call(lin, xs[0], cfg, True)).astype(np.float32)
            y32 = np.array(nax_call(lin, xs[0], cfg, False)).astype(np.float32)
            rel = lambda v: float(np.linalg.norm(v - ref) / np.linalg.norm(ref))
            t32 = timeit(lambda x: nax_call(lin, x, cfg, False), xs, a.reps, a.chain)
            res.update(nax_best_ms=t, nax_best_cfg=cfg, nax_f32_ms=t32, rel_f16=rel(y), rel_f32=rel(y32), speedup=tg_ / t)
            print(f"{line}  best {cfg} {t:7.3f} ms (x{tg_/t:4.2f}) f32-partial {t32:7.3f}  rel vs GEMV {rel(y):.1e} / f32 {rel(y32):.1e}", flush=True)
            out.append(res)
            pk = mx.get_peak_memory() / 1e9
            if pk > CAP_GB:
                print(f"peak {pk:.2f} GB over cap, abort"); sys.exit(3)
        print(f"  peak {mx.get_peak_memory()/1e9:.2f} GB", flush=True)
        del lin
        mx.clear_cache()
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


main()
