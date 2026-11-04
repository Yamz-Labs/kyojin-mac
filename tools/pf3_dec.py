"""Decode kernel throughput per linear shape (isolated eval, min of reps). usage: pf3_dec.py MODEL_DIR [reps]"""
import sys, time, collections, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
seen = collections.OrderedDict()
for _, m in be.model.named_modules():
    if isinstance(m, EL.Exl3Linear) and m.out_features < 100000:
        seen.setdefault((m.in_features, m.out_features, m.K), []).append(m)
tot = 0; totb = 0
for (k, n, K), ls in seen.items():
    m = ls[0]; f = lambda: KN.decode_wr(m._words, m.K, m._kt, m._nt, m.cb)
    mx.eval(f()); ts = []
    for _ in range(reps):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f(), f(), f(), f()); ts.append((time.perf_counter() - t0) / 4)
    t = min(ts); wb = n * k * 2; rb = n * k * K / 8
    tot += t * len(ls); totb += wb * len(ls)
    print(f"  {k}x{n} K={K} x{len(ls)}: {t*1e3:.3f} ms, write {wb/t/1e9:.0f} GB/s, +read {(wb+rb)/t/1e9:.0f} GB/s")
print(f"total decode (isolated) {tot*1e3:.1f} ms, {totb/1e9:.2f} GB written")
