"""chain96.py: cost of the dependency chain between quantized GEMVs (1 row, q4 g64): N distinct weights run as a chain (output -> next input)
vs the same N GEMVs independent, at decode shapes. Per-link extra = (chain - independent) / N: the ramp + tail a GEMV pays when it waits for its input."""
import json, statistics, sys, time
import mlx.core as mx
N = 48


def tbest(f, n=9):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return min(ts[1:])


res = {}
for H, I in ((2560, 9216), (4096, 12288)):
    up, dn = [], []
    for i in range(N):
        for (n_, k_), lst in (((I, H), up), ((H, I), dn)):
            w = mx.random.normal((n_, k_)).astype(mx.float16) * 0.02
            q, s, b = mx.quantize(w, group_size=64, bits=4); lst.append((q, s.astype(mx.bfloat16), b.astype(mx.bfloat16))); mx.eval(*lst[-1])
    x = mx.random.normal((1, H)).astype(mx.bfloat16); xi = mx.random.normal((1, I)).astype(mx.bfloat16); mx.eval(x, xi)
    qm = lambda a, t: mx.quantized_matmul(a, *t, transpose=True, group_size=64, bits=4)
    by = sum(a.nbytes for t in up + dn for a in t)

    def chain():
        y = x
        for i in range(N):
            y = qm(qm(y, up[i]), dn[i])
        return y

    def chain_norm():                     # + one RMSNorm-like tiny kernel per link (dependency through a small op)
        y = x
        for i in range(N):
            y = qm(mx.fast.rms_norm(qm(mx.fast.rms_norm(y, None, 1e-6), up[i]), None, 1e-6), dn[i])
        return y

    def indep():
        return [qm(x, up[i]) for i in range(N)] + [qm(xi, dn[i]) for i in range(N)]
    tc, tn, ti = tbest(chain), tbest(chain_norm), tbest(indep)
    res[f"H{H}"] = dict(GBs_chain=round(by / tc / 1e9, 1), GBs_indep=round(by / ti / 1e9, 1), us_per_link=round((tc - ti) / (2 * N) * 1e6, 2),
                        us_per_tiny_dep=round((tn - tc) / (2 * N) * 1e6, 2), ms_chain=round(tc * 1e3, 3), ms_indep=round(ti * 1e3, 3))
print("CHAIN96", json.dumps(res), flush=True)
