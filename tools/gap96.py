"""gap96.py PACK [CTX]: where the 1-row decode step loses time against the bytes / probe-BW roof.

A. Achievable q4 GEMV time vs weight size, in-stream: a chain of distinct square N x N q4 g64 weights (output -> next input, the step's real
   dependency) and the same launches independent. Per size: us per kernel, GB/s; fit t_dep = a + bytes / b (a = ramp + tail + barrier).
B. The step's own GEMVs (decfuse layout: fused q|k|v, gate|up, qkv|z|b|a; out / down / o; head): bytes each, predicted in-stream time from the
   curve (log interpolation), sum vs bytes / probe BW.
C. Host side: graph build time and async_eval return time of one step vs the synced step and the pipelined ms/token."""
import json, math, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx

PACK = sys.argv[1]; CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
PROBE = 131.9
qm = lambda a, t: mx.quantized_matmul(a, *t, transpose=True, group_size=64, bits=4)


def tbest(f, n=7):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return min(ts[1:])


curve = []
for N in (512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288):
    w = mx.random.normal((N, N)).astype(mx.float16) * 0.02
    q, s, b = mx.quantize(w, group_size=64, bits=4); one = (q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)); mx.eval(*one); del w
    by = sum(a.nbytes for a in one)
    C = max(8, min(400, int(320e6 // by)))
    ws = [one] + [tuple(a + 0 if a.dtype != mx.uint32 else a ^ mx.array(i, mx.uint32) for a in one) for i in range(1, C)]; mx.eval(ws)
    x = (mx.random.normal((1, N)) * 0.1).astype(mx.bfloat16); mx.eval(x)

    def chain():
        y = x
        for t in ws:
            y = qm(y, t)
        return y
    tc = tbest(chain); ti = tbest(lambda: [qm(x, t) for t in ws])
    curve.append(dict(N=N, MB=round(by / 1e6, 3), copies=C, dep_us=round(tc / C * 1e6, 2), ind_us=round(ti / C * 1e6, 2),
                      dep_GBs=round(by * C / tc / 1e9, 1), ind_GBs=round(by * C / ti / 1e9, 1)))
    print("CURVE", json.dumps(curve[-1]), flush=True)
    del ws, one; mx.clear_cache()
# least squares t_dep = a + bytes / b on all points
X = [c["MB"] * 1e6 for c in curve]; Y = [c["dep_us"] * 1e-6 for c in curve]
mxx, myy = statistics.mean(X), statistics.mean(Y)
slope = sum((x - mxx) * (y - myy) for x, y in zip(X, Y)) / sum((x - mxx) ** 2 for x in X)
fit = dict(a_us=round((myy - slope * mxx) * 1e6, 2), b_GBs=round(1 / slope / 1e9, 1))
print("FIT", json.dumps(fit), flush=True)


def t_of(by):
    lx = [math.log(c["MB"] * 1e6) for c in curve]; ly = [math.log(c["dep_us"]) for c in curve]; l = math.log(by)
    if l <= lx[0]:
        return curve[0]["dep_us"]
    if l >= lx[-1]:
        return curve[-1]["dep_us"] * by / (curve[-1]["MB"] * 1e6)
    i = max(j for j in range(len(lx)) if lx[j] <= l)
    f = (l - lx[i]) / (lx[i + 1] - lx[i])
    return math.exp(ly[i] + f * (ly[i + 1] - ly[i]))


from kyojin_mlx import decfuse as DF
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(PACK, mtp=False, log=lambda *a: None)
tm = be.lm.model
DF.prepare(tm)
nb = lambda m: sum(getattr(m, k).nbytes for k in ("weight", "scales", "biases") if k in m)
gem = {}
for l in tm.layers:
    la, sa, mlp = getattr(l, "linear_attn", None), getattr(l, "self_attn", None), l.mlp
    ops = []
    if la is not None:
        ops += [("gdn.in_fused", la._kr_fused or la.in_proj_qkv), ("gdn.out", la.out_proj)]
    if sa is not None:
        ops += [("attn.qkv", sa._kr_fused or sa.q_proj), ("attn.o", sa.o_proj)]
    ops += [("mlp.gate_up", mlp._kr_fused or mlp.gate_proj), ("mlp.down", mlp.down_proj)]
    for k, m in ops:
        g = gem.setdefault(k, [0, 0.0, 0.0]); by = nb(m); g[0] += 1; g[1] += by; g[2] += t_of(by)
hm = getattr(be.lm, "lm_head", None) or tm.embed_tokens
by = nb(hm); gem["head"] = [1, by, t_of(by)]
tot_b = sum(g[1] for g in gem.values()); tot_t = sum(g[2] for g in gem.values())
rows = {k: dict(n=g[0], MB=round(g[1] / 1e6, 1), MB_each=round(g[1] / g[0] / 1e6, 2), pred_ms=round(g[2] / 1e3, 3),
                roof_ms=round(g[1] / PROBE / 1e6, 3)) for k, g in gem.items()}
print("STEPGEMV", json.dumps(dict(rows=rows, total_MB=round(tot_b / 1e6, 1), n=sum(g[0] for g in gem.values()), pred_ms=round(tot_t / 1e3, 3),
                                  roof_ms=round(tot_b / PROBE / 1e6, 3))), flush=True)

# B2. replay: the step's GEMVs with the real weights in step order, output -> next input (slice / zero-pad to the next K), GEMVs only
seq = []
for l in tm.layers:
    la, sa, mlp = getattr(l, "linear_attn", None), getattr(l, "self_attn", None), l.mlp
    seq += ([la._kr_fused or la.in_proj_qkv, la.out_proj] if la is not None else [sa._kr_fused or sa.q_proj, sa.o_proj])
    seq += [mlp._kr_fused or mlp.gate_proj, mlp.down_proj]
seq.append(hm)
H = tm.layers[0].mlp.down_proj.weight.shape[0]
x0 = (mx.random.normal((1, H)) * 0.1).astype(mx.bfloat16); mx.eval(x0)
K_of = lambda m: m.weight.shape[1] * 32 // m.bits


ks = [K_of(m) for m in seq]


def replay():
    y = x0
    for i, m in enumerate(seq):
        if y.shape[-1] != ks[i]:
            y = y[:, :ks[i]] if y.shape[-1] > ks[i] else mx.pad(y, [(0, 0), (0, ks[i] - y.shape[-1])])
        y = m.as_linear(y) if hasattr(m, "as_linear") else m(y)
    return y


xs = {k: (mx.random.normal((1, k)) * 0.1).astype(mx.bfloat16) for k in set(ks)}; mx.eval(list(xs.values()))
t_chain = tbest(replay); t_ind = tbest(lambda: [m.as_linear(xs[k]) if hasattr(m, "as_linear") else m(xs[k]) for m, k in zip(seq, ks)])
print("REPLAY", json.dumps(dict(n=len(seq), pads=sum(1 for i in range(1, len(seq)) if ks[i] > seq[i - 1].weight.shape[0]),
                                chain_ms=round(t_chain * 1e3, 3), ind_ms=round(t_ind * 1e3, 3), chain_GBs=round(tot_b / t_chain / 1e9, 1),
                                ind_GBs=round(tot_b / t_ind / 1e9, 1), roof_ms=round(tot_b / PROBE / 1e6, 3))), flush=True)

# C. host side on the real step at CTX
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json")))[:256]
cache = be.make_cache(); be.prefill_chunk(cache, ids)
ca = [c for c in cache if hasattr(c, "keys") and c.keys is not None]
_, HK, _, DK = ca[0].keys.shape; DV = ca[0].values.shape[3]; A = -(-(CTX + 64) // 256) * 256
for c in ca:
    dt = c.keys.dtype; c.keys = mx.random.normal((1, HK, A, DK)).astype(dt); c.values = mx.random.normal((1, HK, A, DV)).astype(dt); c.offset = CTX
mx.eval(*[a for c in ca for a in (c.keys, c.values)])
TOK = mx.array([[1000]], dtype=mx.int32)
step = lambda t: mx.argmax(be._head(tm(t, cache)[:, -1:]).astype(mx.float32)[0, -1], -1)
for _ in range(3):
    mx.eval(step(TOK)); [c.trim(1) for c in ca]
bt, at, st = [], [], []
for _ in range(12):
    t0 = time.perf_counter(); y = step(TOK); t1 = time.perf_counter(); mx.async_eval(y); t2 = time.perf_counter(); y.item(); t3 = time.perf_counter()
    bt.append((t1 - t0) * 1e3); at.append((t2 - t1) * 1e3); st.append((t3 - t0) * 1e3); [c.trim(1) for c in ca]
y = step(TOK); mx.async_eval(y); t0 = time.perf_counter(); n = 24
for _ in range(n):
    yn = step(y.reshape(1, 1).astype(mx.int32)); mx.async_eval(yn); y.item(); y = yn
y.item(); pipe = (time.perf_counter() - t0) * 1e3 / n; [c.trim(n + 1) for c in ca]
host = dict(build_ms=round(min(bt), 3), async_eval_ms=round(min(at), 3), step_ms=round(min(st), 3), step_med=round(statistics.median(st), 3), pipe_ms=round(pipe, 3),
            max_ops_per_buffer=os.environ.get("MLX_MAX_OPS_PER_BUFFER", "default"))
print("HOST", json.dumps(host), flush=True)
