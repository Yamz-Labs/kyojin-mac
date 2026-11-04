"""Per-stage GPU time and exactness of the MoE prefill layer: S3 (decode-to-fp16 + gather_mm) vs the fused NAX path (registers), on a real layer slice.
   python tools/prof_moe_pf.py --dir models/ds_l3 --arch ds [--tokens 128,512,2048] [--reps 3] [--json out.json]
Stage times sync after every stage (sum is a little above the unsynced layer time). Routing is uniform random (worst case: all experts active)."""
import argparse, json, os, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from kyojin_mlx import moe as M, moe_prefill as MP
from bench_moe import ARCH, load, routing

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--arch", choices=list(ARCH), required=True)
ap.add_argument("--tokens", default="128,512,2048"); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--json")
a = ap.parse_args()
A = ARCH[a.arch]
g, u, d = load(a.dir, A["proj"])
E, topk, H = g.experts, A["topk"], g.kin
dec = M.MoeGather(g, u, d, topk, wpre=A["wpre"], fq=A["fq"])
old, new = MP.MoePrefill(dec, nax=False), MP.MoePrefill(dec, nax=True)
assert new.nax, "no NAX on this GPU"
rng = np.random.default_rng(1)
res = []
for T in [int(v) for v in a.tokens.split(",")]:
    ids, w = routing(T, E, topk, 11 + T)
    x = mx.array((rng.standard_normal((T, H)) * 2).astype(np.float16))
    mx.eval(x, ids, w)
    row = {"T": T}
    outs = {}
    for name, pf in (("s3", old), ("nax", new)):
        y = pf(x, ids, w); mx.eval(y)
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); y = pf(x, ids, w); mx.eval(y); ts.append(time.perf_counter() - t0)
        outs[name] = np.array(y)
        pf.prof = {}
        pf(x, ids, w)
        row[name] = dict(layer_ms=1e3 * min(ts), stages_ms={k: 1e3 * v for k, v in pf.prof.items()})
        pf.prof = None
    t0 = time.perf_counter(); yd = mx.concatenate([dec(x[i:i + 4], ids[i:i + 4], w[i:i + 4]) for i in range(0, T, 4)]); mx.eval(yd)
    row["dec4"] = dict(layer_ms=1e3 * (time.perf_counter() - t0))
    print(f"T={T:5d} dec4 layer {row['dec4']['layer_ms']:8.1f} ms (decode-path chunks of 4 rows, production below the S3 crossover)", flush=True)
    ref, got = outs["s3"], outs["nax"]
    row["rel_l2_nax_vs_s3"] = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    res.append(row)
    for name in ("s3", "nax"):
        r = row[name]
        print(f"T={T:5d} {name:4s} layer {r['layer_ms']:8.1f} ms  " + "  ".join(f"{k} {v:.1f}" for k, v in r["stages_ms"].items()), flush=True)
    print(f"T={T:5d} speedup x{row['s3']['layer_ms'] / row['nax']['layer_ms']:.2f}  rel L2 nax vs s3 {row['rel_l2_nax_vs_s3']:.2e}", flush=True)
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
