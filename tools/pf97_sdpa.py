"""Cool-GPU probe: stock SDPA TF vs key length (q chunk L at the end of N keys, causal) and fp16 GEMM TF, interleaved, short bursts.
usage: pf97_sdpa.py [L=2048] [Ns=2048,8192,32768] [Hq=16] [Hk=4] [hd=256] [rounds=3]"""
import sys, time, mlx.core as mx
L = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
Ns = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2048,8192,32768").split(",")]
Hq, Hk, hd = [int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((3, 16), (4, 4), (5, 256))]
RD = int(sys.argv[6]) if len(sys.argv) > 6 else 3

def t1(fn, it):
    mx.eval(fn()); mx.synchronize(); t0 = time.perf_counter(); mx.eval(*[fn() for _ in range(it)]); return (time.perf_counter() - t0) / it

q = mx.random.normal((1, Hq, L, hd)).astype(mx.float16)
kv = {N: (mx.random.normal((1, Hk, N, hd)).astype(mx.float16), mx.random.normal((1, Hk, N, hd)).astype(mx.float16)) for N in Ns}
x = mx.random.normal((L, 2560)).astype(mx.float16); w = mx.random.normal((9216, 2560)).astype(mx.float16)
mx.eval(q, x, w, *[a for p in kv.values() for a in p])
res = {N: [] for N in Ns}; res["gemm"] = []; res["nomask"] = []
for r in range(RD):
    res["gemm"].append(2 * L * 2560 * 9216 / t1(lambda: x @ w.T, 6) / 1e12)
    for N in Ns:
        k, v = kv[N]; fl = 4.0 * Hq * hd * L * (N - L / 2)
        res[N].append(fl / t1(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask="causal"), max(1, int(8 * 2048 / N))) / 1e12)
    k, v = kv[Ns[-1]]
    res["nomask"].append(4.0 * Hq * hd * L * Ns[-1] / t1(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5), 1) / 1e12)
for key, v in res.items():
    v = sorted(v); print(f"{'sdpa causal N=' + str(key) if isinstance(key, int) else key + (' N=' + str(Ns[-1]) if key == 'nomask' else ' 2048x2560x9216'):28s} best {v[-1]:.2f} TF  med {v[len(v)//2]:.2f} TF")
