"""Per-group GPU profile of a chained verify of n rows (or a batched step of N users): same synced-group method as prof_groups.py (every group evaluates its
inputs on entry and its output on exit; group time = its own, children excluded; the per-eval floor is subtracted), run on the UNCOMPILED block path
(KYOJIN_FUSE_COMPILE=0 so every module call is visible; the kernels are the same as in the compiled regions).
usage: prof_chain.py PACK [--rows 3] [--users 0] [--len 135] [--reps 10] [--json F] [--count]"""
import argparse, collections, json, os, statistics, sys, time
os.environ["KYOJIN_FUSE_COMPILE"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import prof_groups as PG
from prof_groups import Prof, instrument, wrap_fn
from kyojin_mlx import patch as P, rows as R, spec as S, fuse_attn as FA

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--rows", type=int, default=3); ap.add_argument("--users", type=int, default=0)
ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--json"); ap.add_argument("--count", action="store_true")
ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--first", type=int, default=0)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack, {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.set)})
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
S.install(model, spec)
if a.first:
    model.language_model.layers = model.language_model.layers[:a.first]
V = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(1)
cache, lg, _, _ = S.prefill(model, rng.integers(16, V - 16, a.len).tolist())
toks = rng.integers(16, V - 16, 32).tolist()
mode = "count" if a.count else "sync"
prof = Prof(mode)
instrument(model, prof)
for nm, name in (("kv_ring", "attn.kv_ring"), ("attend", "attn.attend")):
    setattr(FA, nm, wrap_fn(prof, name, getattr(FA, nm)))
from kyojin_mlx import fuse as F
for nm, name in (("mhc_expand", "mhc.expand_k"), ("mhc_pre", "mhc.pre_k"), ("mhc_mix_pre", "mhc.mixpre_k"), ("router_select", "moe.router_k")):
    setattr(F, nm, wrap_fn(prof, name, getattr(F, nm)))
from kyojin_mlx import moe as M
for st in ("rot_gu", "gemv_gu", "mid", "gemv_dn", "final"):
    setattr(M.MoeGather, st, (lambda f, n: lambda self, *aa, **kk: PG.wrap_fn(prof, "moe." + n, lambda *x, **y: f(self, *x, **y))(*aa, **kk))(getattr(M.MoeGather, st), st))
x0 = mx.ones((1, 5120), mx.float16); mx.eval(x0)
floor = 0.0
if mode == "sync":
    prof.reset(); prof.mark = time.perf_counter()
    for _ in range(200):
        prof.enter("floor", (), {}); y = x0 * 1.0001; prof.exit(y); x0 = y
    floor = prof.t["floor"] / prof.n["floor"]
    print(f"eval floor per group: {1e6 * floor:.1f} us")


def step():
    if a.users:
        return R.step(model, R.batch([R.fork(cache) for _ in range(a.users)]), mx.array(toks[:a.users]))
    return S.verify(model, cache, toks[:a.rows])[0]


runs = []
for rep in range(a.reps + 3):
    prof.reset(); prof.stack = []; prof.mark = time.perf_counter()
    t0 = time.perf_counter()
    out = step()
    prof.enter("<end>", (), {}); mx.eval(out); prof.exit(None)
    tot = time.perf_counter() - t0
    if rep >= 3:
        runs.append(dict(t=dict(prof.t), n=dict(prof.n), launch=dict(prof.launch), total=tot))
    if mode == "count" and rep >= 3:
        break
names = sorted({k for r in runs for k in r["t"]})
agg = {}
for nm in names:
    ts = [r["t"].get(nm, 0.0) for r in runs]
    n = max(r["n"].get(nm, 0) for r in runs)
    agg[nm] = dict(ms=1e3 * min(ts), n=n, ms_net=1e3 * max(0.0, min(ts) - floor * n), launches=runs[0]["launch"].get(nm, 0))
key = "ms_net" if mode == "sync" else "launches"
print(f"mode {mode}: {'users ' + str(a.users) if a.users else 'chain rows ' + str(a.rows)}, step {1e3 * min(r['total'] for r in runs):.2f} ms (synced)")
for nm, d in sorted(agg.items(), key=lambda kv: -kv[1][key]):
    print(f"{nm:20s} calls {d['n']:3d}  net {d['ms_net']:7.3f} ms  launches {d['launches']:4d}")
print(f"sum of groups {sum(d[key] for d in agg.values()):.2f}")
if a.json:
    json.dump(dict(step_ms=1e3 * min(r['total'] for r in runs), groups=agg), open(a.json, "w"), indent=1)
