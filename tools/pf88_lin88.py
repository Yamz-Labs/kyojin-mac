"""Per-linear prefill micro (EXL3 4B): for each distinct (in,out,K) shape and rows R: fused NAX path (Exl3Linear._dense), decode_wr alone, fp16 matmul alone, decode+matmul, qmm4 g64 reference, vs 13.7 TF roof.
usage: lin88.py MODEL_DIR rows,rows"""
import sys, time, collections, mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import wcache as WC, kernels as KN, exl3_linear as EL
be, _ = load_backend(sys.argv[1]); model = be.model
rows_l = [int(r) for r in sys.argv[2].split(",")]
mx.set_cache_limit(256 * 2 ** 20)
cnt = collections.Counter(); rep = {}
for m in WC._linears(model):
    k = (m.in_features, m.out_features, m.K); cnt[k] += 1; rep.setdefault(k, m)
head = model.language_model.lm_head if hasattr(model.language_model, "lm_head") else None
def T(f, n=6):
    f(); mx.synchronize(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[0] * 1e3, ts[len(ts)//4] * 1e3
tot = collections.defaultdict(float)
print(f"{'shape(in,out,K)':22s} n  R    fused  dec    mm   dec+mm  qmm4  roofms | TF: fused dec+mm qmm4")
for R in rows_l:
    for (k, n_, K), lin in sorted(cnt.items(), key=lambda kv: -kv[1]*kv[0][0]*kv[0][1]) and [(a, rep[a]) for a in sorted(cnt, key=lambda a: -cnt[a]*a[0]*a[1])]:
        pass
    for key in sorted(cnt, key=lambda a: -cnt[a]*a[0]*a[1]):
        lin = rep[key]; kin, nout, K = key
        x = (mx.random.normal((R, kin)) * 0.5).astype(mx.float16); mx.eval(x)
        fl = 2 * R * kin * nout
        fused = T(lambda: lin._dense(x)) [0] if lin._fused_ok(R) else float("nan")
        dec = T(lambda: KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb))[0]
        wr = KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb); mx.eval(wr)
        mm = T(lambda: x @ wr.T)[0]
        dm = T(lambda: x @ KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb).T)[0]
        w = mx.random.normal((nout, kin)).astype(mx.float16); wq, sc, bi = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, sc, bi)
        q = T(lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=4))[0]
        roof = fl / 13.7e12 * 1e3
        tf = lambda ms: fl / (ms * 1e-3) / 1e12
        print(f"{str(key):22s} {cnt[key]:2d} {R:5d} {fused:6.2f} {dec:6.2f} {mm:6.2f} {dm:6.2f} {q:6.2f} {roof:6.2f} | {tf(fused):5.1f} {tf(dm):5.1f} {tf(q):5.1f}", flush=True)
        tot[(R, "fused")] += cnt[key] * fused; tot[(R, "dec+mm")] += cnt[key] * dm; tot[(R, "qmm4")] += cnt[key] * q; tot[(R, "roof")] += cnt[key] * roof; tot[(R, "dec")] += cnt[key] * dec; tot[(R, "mm")] += cnt[key] * mm
for R in rows_l:
    print(f"R={R} sum over layers (no head): " + "  ".join(f"{n}={tot[(R, n)]:.0f}ms" for n in ("fused", "dec", "mm", "dec+mm", "qmm4", "roof")))
