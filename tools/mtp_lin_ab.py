"""Per-linear-group chain timing (dependent chain over all layers), stock vs q4mma, alternating. usage: mtp_lin_ab.py PACK [--rows 2,4] [--passes 4]"""
import argparse, time, statistics
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rows", default="2,4"); ap.add_argument("--passes", type=int, default=4); a = ap.parse_args()
from kyojin_mlx import mlxpack, q4mma as QM
model, tok = mlxpack.load(a.pack); lm = model.language_model
QM.install(model); L = lm.model.layers
groups = {"gdn.qkv": [l.linear_attn.in_proj_qkv for l in L if l.is_linear], "gdn.z": [l.linear_attn.in_proj_z for l in L if l.is_linear], "gdn.b": [l.linear_attn.in_proj_b for l in L if l.is_linear],
          "gdn.out": [l.linear_attn.out_proj for l in L if l.is_linear], "att.q": [l.self_attn.q_proj for l in L if not l.is_linear], "att.k": [l.self_attn.k_proj for l in L if not l.is_linear],
          "att.o": [l.self_attn.o_proj for l in L if not l.is_linear], "mlp.gate": [l.mlp.gate_proj for l in L], "mlp.down": [l.mlp.down_proj for l in L]}
def bench(fn, n=2, reps=4):
    ts = []
    for r in range(reps + 1):
        t0 = time.perf_counter(); mx.eval([fn() for _ in range(n)]); dt = time.perf_counter() - t0
        if r: ts.append(dt / n * 1e3)
    return statistics.median(ts)
res = {}
for ps in range(a.passes):
    for R in [int(r) for r in a.rows.split(",")]:
        for g, mods in groups.items():
            K = mods[0]._K
            x0 = mx.random.normal((1, R, K)).astype(mx.bfloat16)
            def chain():
                x = x0; y = None
                for m in mods:
                    y = m(x); x = x0 + y[..., :1] * 0
                return y
            for v in ("stock", "mma"):
                QM.MIN_ROWS = 2 if v == "mma" else 99
                res.setdefault((R, g, v), []).append(bench(chain))
for R in [int(r) for r in a.rows.split(",")]:
    tot = {"stock": 0, "mma": 0}
    for g in groups:
        s, m = statistics.median(res[(R, g, "stock")]), statistics.median(res[(R, g, "mma")]); tot["stock"] += s; tot["mma"] += m
        print(f"R={R} {g:9s} stock {s:6.2f} ms  mma {m:6.2f} ms  ({len(groups[g])} layers, per layer {s/len(groups[g])*1e3:.0f} vs {m/len(groups[g])*1e3:.0f} us)")
    print(f"R={R} total {tot['stock']:.1f} vs {tot['mma']:.1f}")
