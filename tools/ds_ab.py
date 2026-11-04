import os, sys, runpy, time, statistics, io, contextlib
import numpy as np, mlx.core as mx
from kyojin_mlx import exl3_linear as EL
seen = {}
def mk(f):
    def g(self, x2, *a, **k):
        seen.setdefault((self.K, self.in_features, self.out_features), self)
        return f(self, x2, *a, **k)
    return g
for n in ("_mma", "_ksplit", "_dense"):
    setattr(EL.Exl3Linear, n, mk(getattr(EL.Exl3Linear, n)))
sys.argv = ["spec_check.py"] + sys.argv[1:]
with contextlib.redirect_stdout(io.StringIO()):
    try: runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_check.py"), run_name="__main__")
    except SystemExit: pass
def setv(v):
    EL.MMA_ROWS, EL.NAX_KS = (0, 0) if v == "old" else (-1, -1)
for R in (5, 6, 9, 12):
    tot = {"old": 0.0, "new": 0.0}
    rows = []
    for (K, kin, nout), l in sorted(seen.items()):
        if nout * kin < 200_000: continue
        x = mx.array(np.random.default_rng(1).standard_normal((R, kin)).astype(np.float16))
        T = {"old": [], "new": []}
        for rep in range(7):
            for v in T:
                setv(v); mx.eval(l(x)); t0 = time.perf_counter()
                for _ in range(8): y = l(x)
                mx.eval(y); T[v].append((time.perf_counter() - t0) / 8 * 1e6)
        o, n_ = min(T["old"]), min(T["new"])
        rows.append(f"K{K} {kin}x{nout}: {o:.0f}->{n_:.0f}us")
    print(f"R={R}: " + "; ".join(rows), flush=True)
