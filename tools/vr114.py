"""vr114.py PACK [--len 65536] [--reps 9]: served-path verify (rows.verify on the batched cache) of n = 1..5 rows vs rows.step vs the solo chain spec.verify, at a long context.
Prints median ms (wall of async_eval + eval); --abl NAME[,NAME] ablates an op class (iscore, topk, attend, moe) by a cheap stand-in to get a per-op delta."""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx, numpy as np
from kyojin_mlx.serve.backends import load_backend
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--len", type=int, default=65536); ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--rows", type=int, default=5); ap.add_argument("--abl", default="")
a = ap.parse_args()
be, cfg = load_backend(a.pack, mtp=True, ced=False, log=lambda *x: None)
R, S, model = be.R, be.S, be.model
V = cfg.get("text_config", cfg)["vocab_size"]
rng = np.random.default_rng(1)
prompt = rng.integers(16, V - 16, a.len).tolist()
cache, lg, taps, first = S.prefill(model, prompt, be.drafter.tap_ids, chunk=4096)
toks = rng.integers(16, V - 16, 16).tolist()
from kyojin_mlx import dsv4_core as DC
abl = set(x for x in a.abl.split(",") if x)
if "iscore" in abl:
    o = DC.index_scores
    def f(q, keys, wts, lens, t_max, cand=None, sess=None):
        r = o(q[:1], keys, wts[:1], lens[:1], t_max, None if cand is None else cand[:1], None if sess is None else sess[:1])
        return mx.broadcast_to(r, (q.shape[0], r.shape[1]))
    DC.index_scores = f
bc = R.batch([R.fork(cache)])
def t_verify(n):
    t = time.perf_counter(); p = R.verify(model, bc, [toks[:n]]); mx.eval(p.logits); d = time.perf_counter() - t; p.commit([0]); be.settle(bc); return d
def t_verify_lazy(n):
    t = time.perf_counter(); p = R.verify(model, bc, [toks[:n]]); t1 = time.perf_counter(); mx.async_eval(p.logits); t2 = time.perf_counter(); mx.eval(p.logits); r = (t1 - t, t2 - t1, time.perf_counter() - t2); p.commit([0]); be.settle(bc); return r
def t_step():
    t = time.perf_counter(); l = R.step(model, bc, mx.array([toks[0]])); mx.eval(l); return time.perf_counter() - t
res = {}
for rep in range(a.reps + 2):
    for n in range(1, a.rows + 1):
        d = t_verify(n)
        if rep >= 2: res.setdefault(f"verify{n}", []).append(d)
    c = None
    d = t_step()
    if rep >= 2: res.setdefault("step", []).append(d)
    for n in (2, 4):
        b, e, w = t_verify_lazy(n)
        if rep >= 2: res.setdefault(f"build{n}", []).append(b); res.setdefault(f"enc{n}", []).append(e); res.setdefault(f"gpu{n}", []).append(w)
print(f"len {a.len} abl {sorted(abl)}")
for k, v in res.items(): print(f"{k:9s} {1e3 * statistics.median(v):7.2f} ms  (min {1e3 * min(v):.2f})")
