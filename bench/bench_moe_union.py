"""Multi-user MoE decode on real stacked layers: plain per-pair gather path vs the union path (one decode per distinct expert).
   python bench/bench_moe_union.py --dir models/glm_l10 --arch glm [--trace results_mac/macair4/glm_trace_l10_l29.npz --layer 10]
          [--users 1,2,4] [--samples 10] [--chain 6] [--shares 0,0.25,0.5,1] [--json out.json]
Both paths use the k-split of the 1-row call (pin_rows=1), so a user's result does not depend on the others (bitwise, tested).
Routing sources: uniform random per row, real decode traces (distinct conversations, random positions), and controlled overlap
`share=f` (every expert of the following rows equals an expert of row 0 with probability f). Reported per source and N:
distinct experts / (N * topk), ms per layer for each path, aggregate speedup S(N) = N * t(1) / t(N)."""
import argparse, json, os, statistics, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_moe import ARCH, load
from kyojin_mlx import moe as M

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--arch", choices=list(ARCH), required=True)
ap.add_argument("--trace"); ap.add_argument("--layer", type=int)
ap.add_argument("--users", default="1,2,4"); ap.add_argument("--samples", type=int, default=10); ap.add_argument("--chain", type=int, default=6)
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--shares", default="0,0.25,0.5,1"); ap.add_argument("--json")
ap.add_argument("--cache-mb", type=int, default=256); ap.add_argument("--no-trace-skip", action="store_true")
a = ap.parse_args()
mx.set_cache_limit(a.cache_mb * 2 ** 20)
A = ARCH[a.arch]
g, u, d = load(a.dir, A["proj"])
E, topk, H = g.experts, A["topk"], g.kin
mk = lambda union: M.MoeGather(g, u, d, topk, wpre=A["wpre"], fq=A["fq"], union=union, pin_rows=1)
plain, uni = mk(False), mk(True)
auto = M.MoeGather(g, u, d, topk, wpre=A["wpre"], fq=A["fq"])          # k-split chosen for the actual row count (not batch invariant)
ones = mx.ones((H,), mx.float16)
rng = np.random.default_rng(11)
Ns = [int(v) for v in a.users.split(",")]


def t_layer(fn, x0, ids, w):
    def run():
        x = x0
        for _ in range(a.chain):
            x = mx.fast.rms_norm(fn(x, ids, w).astype(mx.float16), ones, 1e-6)
        mx.eval(x)
    run()
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter(); run(); ts.append((time.perf_counter() - t0) / a.chain)
    return statistics.median(ts)


def route_uniform(N):
    return np.stack([rng.choice(E, topk, replace=False) for _ in range(N)])


def route_share(N, f):
    base = rng.choice(E, topk, replace=False)
    rows = [base]
    for _ in range(1, N):
        row = []
        for e in base:
            if rng.random() < f:
                row.append(e)
            else:
                while True:
                    c = int(rng.integers(E))
                    if c not in row and c not in base:
                        break
                row.append(c)
        rows.append(np.array(row))
    return np.stack([rng.permutation(r) for r in rows])


traces = None
if a.trace:
    z = np.load(a.trace)
    traces = [z[k].astype(np.int64) for k in sorted(z.files) if k.endswith(f"_layer{a.layer}")]

def route_trace(N):
    cs = rng.choice(len(traces), N, replace=False)
    return np.stack([traces[c][rng.integers(len(traces[c]))] for c in cs])

def route_consec(N):
    """N consecutive decode tokens of one conversation: what the verify step of a draft (rows = draft + 1) routes."""
    t = traces[int(rng.integers(len(traces)))]
    s = int(rng.integers(len(t) - N))
    return t[s:s + N]

sources = [("uniform", route_uniform)] + ([("trace", route_trace), ("trace-consec", route_consec)] if traces else []) + [(f"share={f}", (lambda N, f=f: route_share(N, f))) for f in (float(v) for v in a.shares.split(",") if v)]
res = {"dir": a.dir, "experts": E, "topk": topk, "chain": a.chain, "sources": {}}
print(f"# {a.dir}: {E} experts top-{topk}, hidden {H}, inter {g.nout}, K {dict((int(k), int((np.array(g.k) == k).sum())) for k in g.ks)}, wpre {A['wpre']} fq {A['fq']}", flush=True)
for name, fn in sources:
    ent = {}
    for N in Ns:
        us, tp, tu, ta = [], [], [], []
        for s in range(a.samples if N > 1 or name != "uniform" else 3):
            ids = fn(N).astype(np.int32)
            w = rng.uniform(0.1, 1.0, ids.shape).astype(np.float32); w = w / w.sum(-1, keepdims=True) * 1.5
            x0 = mx.array(rng.standard_normal((N, H)).astype(np.float16)); ids_m, w_m = mx.array(ids), mx.array(w)
            us.append(len(set(ids.reshape(-1).tolist())) / ids.size)
            tp.append(t_layer(plain, x0, ids_m, w_m)); tu.append(t_layer(uni, x0, ids_m, w_m)); ta.append(t_layer(auto, x0, ids_m, w_m))
        ent[N] = dict(unique_frac=float(np.mean(us)), plain_ms=float(np.mean(tp) * 1e3), union_ms=float(np.mean(tu) * 1e3), auto_ms=float(np.mean(ta) * 1e3))
    base_p, base_u = ent[Ns[0]]["plain_ms"], ent[Ns[0]]["union_ms"]
    for N in Ns:
        e = ent[N]
        e["S_plain"] = N * base_p / e["plain_ms"]; e["S_union"] = N * base_u / e["union_ms"]; e["union_speedup"] = e["plain_ms"] / e["union_ms"]
        print(f"{name:12s} N={N}: distinct/(N*topk) {e['unique_frac']:.2f} | plain {e['plain_ms']:.3f} ms/layer (S {e['S_plain']:.2f}) | union {e['union_ms']:.3f} ms (S {e['S_union']:.2f}) | union gain x{e['union_speedup']:.2f} | k-split by row count {e['auto_ms']:.3f} ms", flush=True)
    res["sources"][name] = ent
res["peak_gb"] = mx.get_peak_memory() / 2 ** 30
print(f"mlx peak {res['peak_gb']:.2f} GB")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
