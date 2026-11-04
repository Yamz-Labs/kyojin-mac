"""End-to-end prefill of a patched slice (ds2 = 2 DS4.1 layers at full dims, index-source + reuse layer), arm OFF = stock attention + stock mHC rows, arm ON = prefill kernels.
Interleaved rounds, arm order rotated, min / p25. usage: bench_prefill_e2e.py PACK T,T,.. [rounds] [off_max_T]"""
import sys, time, statistics, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, attn_prefill as AP, fuse as F
pack, Ts = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 8; off_max = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model
def arm(on):
    AP.ENABLED = "1" if on else "0"; F.MHC = bool(on)
def run(T, ids):
    cache = lm.make_cache(); t0 = time.perf_counter(); mx.eval(lm(ids, cache=cache).logits); return time.perf_counter() - t0
for T in Ts:
    ids = mx.array([np.random.default_rng(3).integers(1, 4000, T).tolist()])
    arms = [True] + ([False] if T <= off_max else [])
    for a in arms: arm(a); run(T, ids)                      # warm-up
    res = {a: [] for a in arms}
    for r in range(rounds):
        for a in (arms[r % len(arms):] + arms[:r % len(arms)]):
            arm(a); res[a].append(run(T, ids))
    line = f"T={T:5d}"
    for a in arms:
        v = sorted(res[a]); line += f" | {'ON ' if a else 'OFF'} min {v[0]*1e3:7.1f} ms {T/v[0]:7.0f} tok/s p25 {v[len(v)//4]*1e3:7.1f}"
    if len(arms) == 2: line += f" | x{min(res[False])/min(res[True]):.2f}"
    print(line + f" | peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)
