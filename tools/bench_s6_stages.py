"""Stage times of the fused MoePrefill chunk (host plan, rot, gate+up gemm, mid, down gemm, final). Each stage is eval'd alone: the sum is an upper bound."""
import argparse, glob, os, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import moe as M, moe_prefill as MP, metal_nax as NX
from nax_common import mini_stack
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l3"))
ap.add_argument("--esub", type=int, default=32); ap.add_argument("--T", type=int, default=4096)
ap.add_argument("--experts", type=int, default=384); ap.add_argument("--topk", type=int, default=6); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
T_sub = max(1, round(a.T * a.topk / a.experts * a.esub / a.topk))
rng = np.random.default_rng(1)
exps = sorted(rng.choice(a.experts, a.esub, replace=False).tolist())
g, u, d = (mini_stack(a.dir, p, exps) for p in ("w1", "w3", "w2"))
mx.eval(g.words, u.words, d.words)
ids = np.stack([rng.choice(a.esub, a.topk, replace=False) for _ in range(T_sub)]).astype(np.int32)
wts = rng.uniform(0.1, 1.0, (T_sub, a.topk)).astype(np.float32); wts = wts / wts.sum(-1, keepdims=True) * 1.5
x = mx.array((rng.standard_normal((T_sub, g.kin)) * 2).astype(np.float16))
dec = M.MoeGather(g, u, d, a.topk, wpre=True, fq=False)
pf = MP.MoePrefill(dec, fused=True)
T, topk = ids.shape; P = T * topk
def tm(f, chain=8):
    """ms per call, `chain` independent calls per eval (the sync of one eval is 0.2-0.4 ms: it would dominate the small kernels)."""
    best = 1e9
    for _ in range(a.reps):
        t0 = time.perf_counter(); rs = [f() for _ in range(chain)]; mx.eval(rs); best = min(best, (time.perf_counter() - t0) / chain)
    return best * 1e3, rs[0]
t0 = time.perf_counter(); order, proc, cnt, cum, inv, nhot = pf._plan(ids); t_plan = (time.perf_counter() - t0) * 1e3
flat_e = ids.reshape(-1)[order].astype(np.int32); flat_w = wts.reshape(-1)[order]; tok = (order // topk).astype(np.uint32)
n = P
ids_s, wts_s, tok_s = mx.array(flat_e), mx.array(flat_w), mx.array(tok); eids = mx.array(proc.astype(np.int32))
cl = cum.astype(np.uint32); cfg, mr = NX.pick_cfg(cnt), int(cnt.max())
ro = mx.array(np.concatenate([cl[:-1], n + cl[:-1] * 0 + cl]).astype(np.uint32)) if False else mx.array(np.concatenate([cl[:-1], n + cl]).astype(np.uint32))
t_rot, xh = tm(lambda: pf._rot(mx.array(x), tok_s, ids_s, n, True, True, True))
t_gu, inner = tm(lambda: NX.expert_gemm((g, u), eids, ro, xh, 2 * n, mr, cfg))
t_mid, xh2 = tm(lambda: pf._mid(inner, ids_s, wts_s, n, True, True))
t_d, part = tm(lambda: NX.expert_gemm((d,), eids, mx.array(cl), xh2, n, mr, cfg))
from kyojin_mlx import metal_pf as PF
def fin():
    kf = MP._kernel(("fin", not dec.wpre, True), "kyojin_finpf", input_names=["part", "inv", "ids", "wts", "svh", "params"], output_names=["y"], source=PF.final_pf_source(not dec.wpre, True))
    grid, tgp = MP._had(d.blocks, T)
    return kf(inputs=[part, mx.array(inv.astype(np.uint32)), mx.array(ids.reshape(-1).astype(np.int32)), mx.array(wts.reshape(-1)), d.svh, mx.array([1, d.nout, topk], dtype=mx.uint32)], grid=grid, threadgroup=tgp, output_shapes=[(T, d.nout)], output_dtypes=[mx.float32])[0]
t_fin, _ = tm(fin)
t_all, _ = tm(lambda: pf(x, mx.array(ids), mx.array(wts)))
fl_gu = 2 * 2 * P * g.kin * g.nout; fl_d = 2 * P * d.kin * d.nout
print(f"cfg {cfg} rows/expert {P/a.esub:.0f}  esub {a.esub}")
print(f"plan(host) {t_plan:.2f}  rot {t_rot:.2f}  gate+up {t_gu:.2f} ({fl_gu/t_gu/1e9:.1f} TFLOPS)  mid {t_mid:.2f}  down {t_d:.2f} ({fl_d/t_d/1e9:.1f} TFLOPS)  final {t_fin:.2f}  | sum {t_plan+t_rot+t_gu+t_mid+t_d+t_fin:.2f}  whole call {t_all:.2f} ms")
