"""Where the fused NAX expert GEMM spends its time: full vs no decode (fake weights) vs no matmul, gate+up of one MoE layer at T tokens.
   python tools/nax_split.py --dir models/glm_l10 --arch glm [--tokens 128,512,2048]"""
import argparse, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from kyojin_mlx import metal_nax as NX
from bench_moe import ARCH, load, routing

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--arch", choices=list(ARCH), required=True)
ap.add_argument("--tokens", default="128,512,2048"); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
A = ARCH[a.arch]
g, u, d = load(a.dir, A["proj"])
E, topk = g.experts, A["topk"]
for T in [int(v) for v in a.tokens.split(",")]:
    ids, _ = routing(T, E, topk, 11 + T)
    flat = np.array(ids).reshape(-1)
    counts = np.bincount(flat, minlength=E); active = np.nonzero(counts)[0]; c = counts[active]
    n = T * topk
    lc = np.concatenate([[0], np.cumsum(c)]).astype(np.uint32)
    roff = mx.array(np.concatenate([lc[:-1], n + lc]).astype(np.uint32)); eids = mx.array(active.astype(np.int32))
    Ain = mx.array(np.random.default_rng(0).standard_normal((2 * n, g.kin)).astype(np.float16))
    mx.eval(Ain)
    cfg = NX.pick_cfg(c)
    out = []
    for mode in ("full", "nodec", "nomm"):
        f = lambda: NX.expert_gemm((g, u), eids, roff, Ain, 2 * n, int(c.max()), cfg, out_half=False, mode=mode)
        mx.eval(f())
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
        out.append(f"{mode} {1e3 * min(ts):7.1f} ms")
    fl = 2 * n * 2 * g.kin * g.nout
    print(f"T={T:5d} experts {len(active)} rows/expert {c.mean():.1f} cfg {cfg}: " + " | ".join(out) + f" | full = {fl / 1e12 / (float(out[0].split()[1]) / 1e3):.2f} TFLOP/s", flush=True)
