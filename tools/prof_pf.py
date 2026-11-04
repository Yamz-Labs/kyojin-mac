"""Stage times of one S3 group on a real stacked layer (each stage evaluated alone): where the prefill time goes.
   python tools/prof_pf.py --dir models/ds_l3 --arch ds --tokens 512 [--experts 30] [--reps 3]"""
import argparse, os, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from kyojin_mlx import moe as M, moe_prefill as MP, metal_pf as PF, metal_src as S
from bench_moe import ARCH, load, routing


def t(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); mx.eval(r); ts.append(time.perf_counter() - t0)
    return min(ts), r


ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--arch", choices=list(ARCH), required=True)
ap.add_argument("--tokens", type=int, default=512); ap.add_argument("--experts", type=int, default=30); ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
A = ARCH[a.arch]
g, u, d = load(a.dir, A["proj"])
E, topk, H = g.experts, A["topk"], g.kin
G = a.experts
n = a.tokens * topk // E * G              # pairs of a group of G experts at uniform routing
rng = np.random.default_rng(0)
x = mx.array((rng.standard_normal((a.tokens, H)) * 2).astype(np.float16))
eids = mx.array(np.arange(G, dtype=np.int32))
per = a.tokens * topk // E
lidx = mx.array(np.repeat(np.arange(G, dtype=np.uint32), per))
n = int(lidx.shape[0])
ids_s = mx.array(np.repeat(np.arange(G, dtype=np.int32), per))
tok_s = mx.array(rng.integers(0, a.tokens, n).astype(np.uint32))
wts_s = mx.array(rng.uniform(0.1, 1, n).astype(np.float32))
mx.eval(x, eids, lidx, ids_s, tok_s, wts_s)
gb = lambda arr: arr.nbytes / 2 ** 30
print(f"group of {G} experts, {a.tokens} tokens, {per} tokens per expert, {n} pairs")
tm, wgu = t(lambda: MP.decode_group((g, u), eids, g.cb), a.reps); print(f"decode gate+up  {tm*1e3:8.1f} ms  writes {gb(wgu):.2f} GiB -> {gb(wgu)/tm:6.1f} GiB/s, {wgu.size/tm/1e9:6.1f} Gw/s")
tm, wd = t(lambda: MP.decode_group((d,), eids, d.cb), a.reps); print(f"decode down     {tm*1e3:8.1f} ms  writes {gb(wd):.2f} GiB -> {gb(wd)/tm:6.1f} GiB/s")
kr = mx.fast.metal_kernel(header=PF.HEADER_PF, name="prof_rot", input_names=["x", "tok", "ids", "suh0", "suh1", "params"], output_names=["xh"], source=PF.rot_pf_source(2, True, True))
grid, tg = S.had_grid(g.kin // 128, 2 * n)
tm, xh = t(lambda: kr(inputs=[x, tok_s, ids_s, g.suh, u.suh, mx.array([g.kin, n], dtype=mx.uint32)], grid=grid, threadgroup=tg, output_shapes=[(2 * n, g.kin)], output_dtypes=[mx.float16])[0], a.reps)
print(f"rot_pf          {tm*1e3:8.1f} ms")
ridx = mx.concatenate([lidx, lidx + G])
fl = 2 * n * g.kin * g.nout * 2
tm, inner = t(lambda: mx.gather_mm(xh.reshape(2 * n, 1, g.kin), wgu.swapaxes(-1, -2), rhs_indices=ridx, sorted_indices=True), a.reps)
print(f"gather_mm g+u   {tm*1e3:8.1f} ms  {fl/tm/1e12:6.2f} TFLOPs, reads weights {gb(wgu)/tm:6.1f} GiB/s")
def loop():
    outs = []
    for e in range(2 * G):
        s = (e % G) * per + (e // G) * n
        outs.append(xh[s:s + per] @ wgu[e].T)
    return mx.concatenate(outs)
tm, _ = t(loop, a.reps); print(f"per-expert matmul loop g+u {tm*1e3:8.1f} ms")
tm, _ = t(lambda: mx.matmul(xh[:per].reshape(1, per, g.kin), wgu[:G].swapaxes(-1, -2)), a.reps); print(f"batched matmul g {tm*1e3:8.1f} ms (same x for all experts: {per} rows each)")
