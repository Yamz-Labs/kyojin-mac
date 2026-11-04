"""Interleaved A/B of one MoE layer through MoePrefill: S3 (decode_pf + gather_mm) vs S6 (fused EXL3 decode + NAX), optional variants of S6.
A subset of --esub experts of a real pack (partial reads) with tokens routed inside it so that each expert gets the rows it gets in the full layer
(rows/expert = T * topk / experts). Arms run round-robin; min / p25 / median per arm, time scaled to the full layer (x experts / esub), tok/s.
   lockf -k ~/kyojin-mlx/.gpu.lock python tools/bench_s6_ab.py --dir models/ds_l3 --T 512,1024,2048,4096 --arms s3,s6"""
import argparse, os, re, subprocess, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import moe as M, moe_prefill as MP, metal_nax as NX
from nax_common import mini_stack

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l3"))
ap.add_argument("--T", default="512,1024,2048,4096"); ap.add_argument("--esub", type=int, default=32)
ap.add_argument("--experts", type=int, default=384); ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--projs", default="w1,w3,w2"); ap.add_argument("--wpre", type=int, default=1)
ap.add_argument("--skew", type=float, default=0.0); ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--arms", default="s3,s6", help="s3 | s6 | s6noskip | s6c16 | s6c32 | s6c64 (forced row block)")
ap.add_argument("--quiet-wait", type=float, default=20.0, help="seconds to wait for an idle GPU before each round")
ap.add_argument("--group-gib", type=float, default=1.0); ap.add_argument("--check", action="store_true")
a = ap.parse_args()
rng = np.random.default_rng(1)
exps = sorted(rng.choice(a.experts, a.esub, replace=False).tolist())
g, u, d = (mini_stack(a.dir, p, exps) for p in a.projs.split(","))
mx.eval(g.words, u.words, d.words)
dec = M.MoeGather(g, u, d, a.topk, wpre=bool(a.wpre), fq=False)
print(f"{a.dir}: kin {g.kin} nout {g.nout} K {g.ks}; {a.esub}/{a.experts} experts; flops/pair {6*g.kin*g.nout/1e6:.1f} MFLOP")

def gpu_busy() -> int:
    """Device utilization % reported by the GPU driver right now (other agents share the GPU; a rep started while it is busy is not an A/B sample)."""
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"], capture_output=True, text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', out)
    return int(m.group(1)) if m else 0


def wait_quiet(limit: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < limit:
        if gpu_busy() <= 3: return True
        time.sleep(0.15)
    return False


def make(arm):
    if arm == "s3": return MP.MoePrefill(dec, fused=False, group_gib=a.group_gib), (lambda: None)
    name, _, opt = arm.partition(":")
    pf = MP.MoePrefill(dec, fused=True, group_gib=a.group_gib)
    pick = NX.pick_cfg
    def setup():
        NX.SKIP_TAIL = name != "s6noskip"
        NX.pick_cfg = (lambda c, r=int(name[3:]): NX._BLOCK_CFG[r]) if name[:3] == "s6c" else pick     # s6c16 / s6c32 / s6c64: force the row block
    return pf, setup

arms = [(x, *make(x)) for x in a.arms.split(",")]
for T in [int(v) for v in a.T.split(",")]:
    rpe = T * a.topk / a.experts
    T_sub = max(1, round(rpe * a.esub / a.topk)); scale = a.experts / a.esub
    pop = 1.0 / np.arange(1, a.esub + 1) ** a.skew if a.skew else np.ones(a.esub)
    pop = pop[rng.permutation(a.esub)]; pop /= pop.sum()
    ids = np.stack([rng.choice(a.esub, a.topk, replace=False, p=pop) for _ in range(T_sub)]).astype(np.int32)
    wts = rng.uniform(0.1, 1.0, (T_sub, a.topk)).astype(np.float32); wts = wts / wts.sum(-1, keepdims=True) * 1.5
    x = mx.array((rng.standard_normal((T_sub, g.kin)) * 2).astype(np.float16)); idm, wm = mx.array(ids), mx.array(wts)
    cnt = np.bincount(ids.reshape(-1), minlength=a.esub)
    ts = {n: [] for n, _, _ in arms}; ys = {}; peak = {}; nq = 0
    for n, pf, setup in arms:                                         # warm-up (compile) + peak memory of the arm
        setup(); mx.reset_peak_memory(); base = mx.get_active_memory(); y = pf(x, idm, wm); mx.eval(y)
        peak[n] = (mx.get_peak_memory() - base) / 2 ** 20; ys[n] = np.array(y)
    for r in range(a.reps):
        quiet = wait_quiet(a.quiet_wait)
        for n, pf, setup in arms[r % len(arms):] + arms[:r % len(arms)]:         # rotate the order: the first arm after an idle wait runs at a low clock
            setup(); t0 = time.perf_counter(); y = pf(x, idm, wm); mx.eval(y); ts[n].append(time.perf_counter() - t0)
        nq += quiet
    print(f"T={T} ({nq}/{a.reps} rounds started on an idle GPU) rows/expert min {cnt.min()} med {int(np.median(cnt))} max {cnt.max()} (T_sub {T_sub}, x{scale:.1f})")
    ref = arms[0][0]
    for n, _, _ in arms:
        t = np.array(ts[n]) * scale * 1e3
        msg = f"  {n:9s} layer ms min {t.min():7.1f} p25 {np.percentile(t,25):7.1f} med {np.median(t):7.1f} | tok/s min {T/t.min()*1e3:7.0f} p25 {T/np.percentile(t,25)*1e3:7.0f} | extra mem {peak[n]:6.0f} MiB"
        if n != ref:
            r = ys[n] - ys[ref]; msg += f" | x{np.array(ts[ref]).min()/np.array(ts[n]).min():.2f} (min/min) rel vs {ref} {np.linalg.norm(r)/np.linalg.norm(ys[ref]):.1e}"
        print(msg)
