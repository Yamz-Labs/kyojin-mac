"""Dense EXL3 linears of a DS4.1 layer at prefill row counts: decode+matmul (today above DENSE_FUSED_MAX) vs fused S6 with several tile configs.
usage: bench_dense_s6.py PACK [--T 512,1024,2048,4096] [--rounds 15] [--cfgs "32,2,32,64;32,4,32,64"]   (run under lockf)"""
import argparse, time, statistics
import numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, exl3_linear as EL, metal_nax as NX
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--T", default="512,1024,2048,4096"); ap.add_argument("--rounds", type=int, default=15)
ap.add_argument("--cfgs", default="32,2,32,64;32,4,32,64;32,2,64,64;32,4,64,64;16,1,32,64"); ap.add_argument("--only", default="")
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg)
L = model.language_model.layers[0]
mods = {"wq_a": L.attn.wq_a, "wq_b": L.attn.wq_b, "wkv": L.attn.wkv, "wo_b": L.attn.wo_b, "wo_a.g0": L.attn.wo_a.groups[0]}
sh = L.ffn.shared_experts
for n in ("gate_proj", "up_proj", "down_proj"):
    if hasattr(sh, n): mods["sh." + n] = getattr(sh, n)
if a.only: mods = {k: v for k, v in mods.items() if k in a.only.split(",")}
cfgs = [tuple(int(v) for v in c.split(",")) for c in a.cfgs.split(";")]
orig_pick = NX.pick_cfg
rng = np.random.default_rng(0)
def run(m, x, arm):
    m._nax_rows.clear()                                  # the row entry caches the tile config
    if arm == "today": EL.DENSE_FUSED_MAX = 0
    else:
        EL.DENSE_FUSED_MAX = 1 << 30; NX.pick_cfg = lambda counts, c=arm: c
    return m(x)
for name, m in mods.items():
    for T in map(int, a.T.split(",")):
        x = mx.array((rng.standard_normal((1, T, m.in_features))).astype(np.float16))
        arms = ["today"] + cfgs
        outs = {}
        for arm in arms:
            m._nax_rows.clear(); outs[str(arm)] = run(m, x, arm); mx.eval(outs[str(arm)])
        ts = {str(arm): [] for arm in arms}
        for r in range(a.rounds):
            order = arms[r % len(arms):] + arms[:r % len(arms)]
            for arm in order:
                t0 = time.perf_counter(); o = run(m, x, arm); mx.eval(o); ts[str(arm)].append(time.perf_counter() - t0)
        fl = 2 * T * m.in_features * m.out_features
        ref = outs["today"].astype(mx.float32)
        line = f"{name:8s} T={T:5d} " + f"today {min(ts['today'])*1e3:7.2f}ms {fl/min(ts['today'])/1e12:4.1f}TF |"
        for arm in cfgs:
            t = min(ts[str(arm)]); rel = (mx.linalg.norm(outs[str(arm)].astype(mx.float32) - ref) / mx.linalg.norm(ref)).item()
            line += f" {','.join(map(str, arm))}: {t*1e3:6.2f}ms {fl/t/1e12:4.1f}TF ({rel:.0e})"
        print(line, flush=True)
