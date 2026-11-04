"""prof114.py PACK [--len 65536]: per-module GPU time of rows.verify at n rows (synced after every module: perturbs overlap, gives the op table)."""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx, numpy as np
from kyojin_mlx.serve.backends import load_backend
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--len", type=int, default=65536); ap.add_argument("--rows", default="1,2,3,5"); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
be, cfg = load_backend(a.pack, mtp=True, ced=False, log=lambda *x: None)
R, S, model = be.R, be.S, be.model
V = cfg.get("text_config", cfg)["vocab_size"]
rng = np.random.default_rng(1)
cache, lg, taps, first = S.prefill(model, rng.integers(16, V - 16, a.len).tolist(), be.drafter.tap_ids, chunk=4096)
toks = rng.integers(16, V - 16, 16).tolist()
lm = model.language_model
acc = {}
on = [False]
def wrap(obj, name, label):
    cls = type(getattr(obj, name))
    if getattr(cls, "_p114", False): return
    orig = cls.__call__
    def f(self, *x, **k):
        if not on[0]: return orig(self, *x, **k)
        mx.eval(*[v for v in x if isinstance(v, mx.array)])
        t = time.perf_counter(); out = orig(self, *x, **k)
        o = out[0] if isinstance(out, tuple) else out
        mx.eval(o); acc[label] = acc.get(label, 0) + time.perf_counter() - t
        return out
    cls.__call__ = f; cls._p114 = True
L0 = lm.layers[0]
print([n for n in dir(L0) if not n.startswith("_")][:40], type(L0).__name__)
for nm in ("attn", "self_attn", "ffn", "mlp", "mixer"):
    if hasattr(L0, nm): wrap(L0, nm, nm)
for nm in ("gate", "switch_mlp", "shared_experts", "experts"):
    m = getattr(L0, "ffn", None) or getattr(L0, "mlp", None)
    if m is not None and hasattr(m, nm): wrap(m, nm, "moe." + nm)
bc = R.batch([R.fork(cache)])
def run(n):
    p = R.verify(model, bc, [toks[:n]]); mx.eval(p.logits); p.commit([0]); be.settle(bc)
for n in [int(x) for x in a.rows.split(",")]:
    for _ in range(3): run(n)
    on[0] = True; tot = []
    acc.clear()
    for _ in range(a.reps):
        t = time.perf_counter(); run(n); tot.append(time.perf_counter() - t)
    on[0] = False
    print(f"n={n} total {1e3 * statistics.median(tot):.2f} ms | " + " ".join(f"{k} {1e3 * v / a.reps:.2f}" for k, v in acc.items()), flush=True)
