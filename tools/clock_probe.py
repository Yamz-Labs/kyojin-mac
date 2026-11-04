"""Sustained-load clock probe: mlp.gate of every layer (distinct weights) at S rows, blocks of ~4 s of the same op, ms per full pass per block, in order, so a clock drop shows as a rising time.
usage: clock_probe.py PACK [--plan 1,5,1,5,8,1] [--block 4]   (S=5/8 use q4mma, S=1 the stock qmv; --stock5 runs S=5 on MLX qmm)"""
import argparse, time, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--plan", default="1,5,1,5,8,1"); ap.add_argument("--block", type=float, default=4); ap.add_argument("--stock5", action="store_true"); ap.add_argument("--alu", type=int, default=0, help="ms of an ALU-only load (fp16 matmul) per pass, to see a pure-ALU clock drop"); a = ap.parse_args()
from kyojin_mlx import mlxpack, q4mma as QM
model, tok = mlxpack.load(a.pack); lm = model.language_model
QM.install(model)
mods = [l.mlp.gate_proj for l in lm.model.layers]; K = lm.args.hidden_size
z = mx.random.normal((2048, 2048)).astype(mx.float16)
for S in [int(v) for v in a.plan.split(",")]:
    QM.MIN_ROWS = 99 if (S == 1 or a.stock5) else 1
    x = mx.random.normal((1, S, K)).astype(mx.float16); mx.eval(x)
    mx.eval([m(x) for m in mods]); n = 0; ts = []; t0 = time.perf_counter(); tb = t0
    while time.perf_counter() - t0 < a.block:
        s = time.perf_counter(); mx.eval([m(x) for m in mods]); ts.append((time.perf_counter() - s) * 1e3)
    h = len(ts) // 2
    print(f"S={S} {'stock' if QM.MIN_ROWS == 99 else 'q4mma'}: passes {len(ts)}  first-half {sum(ts[:h])/h:.2f} ms  second-half {sum(ts[h:])/(len(ts)-h):.2f} ms", flush=True)
