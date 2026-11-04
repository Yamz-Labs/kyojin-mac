"""Scheduling gap test: the 5 dense linears of a layer (+ wo_a) chained in one eval vs the sum of separate evals, interleaved, idle before each round.
Also a raw matmul reference in the same process to track the clock state. usage: exp_dense_chain.py PACK T"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
pack, T = sys.argv[1], int(sys.argv[2])
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
a = model.language_model.layers[0].attn
rng = np.random.default_rng(0)
def rnd(*s): return mx.array(rng.standard_normal(s).astype(np.float16))
calls = [("wq_a", a.wq_a, rnd(1, T, 5120)), ("wq_b", a.wq_b, rnd(1, T, 1280)), ("wkv", a.wkv, rnd(1, T, 5120)),
         ("wo_a", a.wo_a, rnd(1, 8, T, 4096)), ("wo_b", a.wo_b, rnd(1, T, 8192))]
A = rnd(4096, 4096); B = rnd(4096, 4096)
mx.eval([c[2] for c in calls] + [A, B])
def sep(): return sum(t for t in [timeit(lambda m=m, x=x: m(x)) for _, m, x in calls])
def timeit(fn):
    t0 = time.perf_counter(); mx.eval(fn()); return time.perf_counter() - t0
def chain():
    t0 = time.perf_counter(); mx.eval([m(x) for _, m, x in calls]); return time.perf_counter() - t0
def ref(): return timeit(lambda: [A @ B for _ in range(4)])   # 4 x 137 GF
for _, m, x in calls: mx.eval(m(x))
res = {"sep": [], "chain": [], "ref": []}
for r in range(10):
    for k, f in (("sep", sep), ("chain", chain), ("ref", ref)) if r % 2 == 0 else (("ref", ref), ("chain", chain), ("sep", sep)):
        time.sleep(2.0); res[k].append(f())
for k, v in res.items(): print(k, "min %.1f ms  p25 %.1f  all %s" % (min(v) * 1e3, sorted(v)[len(v)//4] * 1e3, [round(x * 1e3) for x in v]))
print("ref TF (min):", 4 * 2 * 4096 ** 3 / min(res["ref"]) / 1e12)
