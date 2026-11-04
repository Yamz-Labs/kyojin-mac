"""Find the first module whose output for row i of an N-user decode step differs from the solo decode step of user i.
usage: mu_trace.py PACK [--users 4] [--lens 37,64,90,131] [--set key=int ...] [--steps 1]
Every module of the language model records its output (execution order); the batched run and each solo run (forks of the same prefill caches)
record the same sequence, row i of the batched output is compared bit for bit with the solo output."""
import argparse, json
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--users", type=int, default=4); ap.add_argument("--lens", default="37,64,90,131")
ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--steps", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--show", type=int, default=12)
a = ap.parse_args()
over = {k: ([int(t) for t in v.split(":")] if ":" in v else int(v)) for k, v in (s.split("=", 1) for s in a.set)}   # key=7 or key=4:4 (list)
model, spec, cfg = P.build_model(a.pack, over)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
lm = model.language_model
vocab = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(a.seed)
lens = [int(v) for v in a.lens.split(",")][:a.users]
prompts = [rng.integers(16, vocab - 16, n).tolist() for n in lens]
users = [R.prefill(model, ids) for ids in prompts]
first = [int(mx.argmax(lg, axis=-1).item()) for _, lg in users]
store = []
for name, m in lm.named_modules():
    cls = type(m)

    def __call__(self, *args, _n=name, _c=cls, **kw):
        out = _c.__call__(self, *args, **kw)
        store.append((_n, out))
        return out
    m.__class__ = type(cls.__name__, (cls,), {"__call__": __call__, "__module__": cls.__module__})


def flat(o):
    if isinstance(o, mx.array):
        return [o]
    if isinstance(o, (tuple, list)):
        return [x for v in o for x in flat(v)]
    return []


def run(caches, toks):
    store.clear()
    lg = R.step(model, caches, mx.array(toks))
    rec = [(n, [np.array(x) for x in flat(o)]) for n, o in store]
    mx.eval(lg)
    return rec, np.array(lg)


N = len(users)
solo_caches = [R.fork(c) for c, _ in users]
bat_caches = R.batch([R.fork(c) for c, _ in users])
tok = list(first)
for s in range(a.steps):
    solo = [run(solo_caches[i], [tok[i]]) for i in range(N)]
    rec_b, lg_b = run(bat_caches, tok)
    bad = []
    def keyed(rec):
        cnt, d = {}, {}
        for j, (name, arrs) in enumerate(rec):
            k = cnt.get(name, 0); cnt[name] = k + 1
            d[(name, k)] = (j, arrs)
        return d
    kb = keyed(rec_b)
    ks = [keyed(r[0]) for r in solo]
    for key, (j, arrs) in kb.items():
        for i in range(N):
            if key not in ks[i]:
                continue                                              # module only called by the batched path (hoisted projection)
            for k, (xb, xs) in enumerate(zip(arrs, ks[i][key][1])):
                if xb.shape[:1] == (N,) and xs.shape[:1] == (1,) and xb.shape[1:] == xs.shape[1:]:
                    xi = xb[i:i + 1]
                    if xi.dtype.kind == 'f' and not np.array_equal(xi.view(np.uint8), xs.view(np.uint8)):
                        d = np.abs(xi.astype(np.float64) - xs.astype(np.float64)).max()
                        bad.append((j, key[0], i, f"out{k} {xi.dtype} max|d|={d:.3g}"))
    print(f"step {s}: {len(rec_b)} module calls, {len(bad)} (call,row) outputs differ from solo")
    seen = set()
    for j, name, i, msg in bad:
        if (j, name) in seen:
            continue
        seen.add((j, name))
        if len(seen) <= a.show:
            print(f"  call #{j} {name} row {i}: {msg}")
    tok = [int(solo[i][1][0].argmax()) for i in range(N)]
