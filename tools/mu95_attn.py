"""Decode attention of N sessions sharing a P-token prefix (Qwen3.5 4B/9B attention shape Hq 16, Hk 4, D 256, fp16): per-session split-K kernel
over private full KV (multiuser RowsKV path) vs mu_attn pooled (prefix read once). One full-attention layer, ms per call (8 calls x 8 layers chained).
usage: mu95_attn.py P[,P..] N[,N..] [TAIL]"""
import sys, time
import mlx.core as mx
from kyojin_mlx import sdpa_dec as SD, mu_attn as MA

Ps = [int(v) for v in sys.argv[1].split(",")]
Ns = [int(v) for v in sys.argv[2].split(",")]
TAIL = int(sys.argv[3]) if len(sys.argv) > 3 else 40
Hq, Hk, D, L = 16, 4, 256, 8
sc = D ** -0.5
mx.random.seed(0)


def ref(q, k, v):
    k, v = mx.repeat(k, Hq // Hk, axis=1), mx.repeat(v, Hq // Hk, axis=1)
    s = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * sc
    return mx.softmax(s, axis=-1) @ v.astype(mx.float32)


def bench(fn, reps=6):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3 / L


for P in Ps:
    pk = mx.random.normal((1, Hk, P, D)).astype(mx.float16)
    pv = mx.random.normal((1, Hk, P, D)).astype(mx.float16)
    for N in Ns:
        lens = [TAIL + 3 * u for u in range(N)]
        cap = -(-(max(lens) + 256) // 256) * 256
        tk = mx.random.normal((N, Hk, cap, D)).astype(mx.float16)
        tv = mx.random.normal((N, Hk, cap, D)).astype(mx.float16)
        q = mx.random.normal((N, Hq, 1, D)).astype(mx.float16)
        full = [(mx.concatenate([pk, tk[u:u + 1, :, :lens[u]]], 2), mx.concatenate([pv, tv[u:u + 1, :, :lens[u]]], 2)) for u in range(N)]
        mx.eval(full)
        la = mx.array(lens, dtype=mx.int32)
        pooled = lambda: [MA.pooled_attention(q, pk, pv, tk, tv, la, sc) for _ in range(L)]
        per = lambda: [mx.concatenate([SD.sdpa_decode(q[u:u + 1], *full[u], sc) for u in range(N)]) for _ in range(L)]
        o1 = MA.pooled_attention(q, pk, pv, tk, tv, la, sc).astype(mx.float32)
        o2 = mx.concatenate([SD.sdpa_decode(q[u:u + 1], *full[u], sc) for u in range(N)]).astype(mx.float32)
        r = mx.concatenate([ref(q[u:u + 1], *full[u]) for u in range(N)])
        e1, e2 = mx.abs(o1 - r).max().item(), mx.abs(o2 - r).max().item()
        tp, tu = bench(pooled), bench(per)
        roof_u = N * (P + max(lens)) * Hk * D * 4 / 139.3e9 * 1e3
        roof_p = (P + N * max(lens)) * Hk * D * 4 / 139.3e9 * 1e3
        print(f"P={P} N={N} per-user {tu:.3f} ms (roof {roof_u:.3f}) pooled {tp:.3f} ms (roof {roof_p:.3f}) x{tu / tp:.2f}  maxerr pooled {e1:.2e} per-user {e2:.2e}", flush=True)
