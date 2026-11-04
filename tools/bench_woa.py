"""DS4.1 wo_a at prompt rows: 8 per-group Exl3Linear calls + stack (today above the GEMV rows) vs one grouped fused NAX launch set (grouped.nax). Interleaved, idle-started, min.
usage: bench_woa.py PACK T,T,.. [rounds]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, exl3_linear as EL
pack, Ts = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]; rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 8
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
w = model.language_model.layers[0].attn.wo_a
rng = np.random.default_rng(0)
for T in Ts:
    x = mx.array((rng.standard_normal((1, 8, T, 4096))).astype(np.float16)); mx.eval(x)
    def per_group():
        EL.DENSE_FUSED_MAX = 1024
        return mx.stack([w.groups[g](x[:, g]) for g in range(8)], axis=1)
    def per_group_dec():
        EL.DENSE_FUSED_MAX = 0
        return mx.stack([w.groups[g](x[:, g]) for g in range(8)], axis=1)
    def nax():
        return w.grouped.nax(x)
    arms = {"decode+matmul x8": per_group_dec, "fused x8": per_group, "grouped nax": nax}
    names = list(arms); res = {n: [] for n in names}; outs = {n: arms[n]() for n in names}
    mx.eval(*outs.values())
    for r in range(rounds):
        for n in names[r % 3:] + names[:r % 3]:
            time.sleep(1.0); t0 = time.perf_counter(); mx.eval(arms[n]()); res[n].append(time.perf_counter() - t0)
    ref = outs["decode+matmul x8"].astype(mx.float32)
    print(f"T={T:5d} " + " | ".join(f"{n}: {min(res[n])*1e3:6.1f} ms (rel {(mx.linalg.norm(outs[n].astype(mx.float32)-ref)/mx.linalg.norm(ref)).item():.0e})" for n in names), flush=True)
