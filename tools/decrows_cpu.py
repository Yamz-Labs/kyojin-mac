import os as _os; _os.environ.setdefault("KYOJIN_DECROWS", "1")
import os, sys, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in ("..", ".")]
import mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1], mtp=False, log=lambda *a: None)
tm = be.lm.model
base = be.make_cache(); be.prefill_chunk(base, list(range(1000, 1161)))
for U in (1, 4):
    toks = mx.array([2000 + i for i in range(U)])[:, None]
    for name, fn in (("stock", tm._kr_stock), ("decrows", lambda t, c: tm(t, c))):
        ts = []
        for _ in range(12):
            bc = be.batch([be.fork(base) for _ in range(U)]) if U > 1 else be.fork(base)
            if U == 1: toks1 = toks
            t0 = time.perf_counter(); h = fn(toks, bc); y = be._head(h); ts.append((time.perf_counter() - t0) * 1e3); mx.eval(y)
        print(f"users {U} {name} CPU graph build {min(ts):.2f} ms (med {sorted(ts)[len(ts)//2]:.2f})")
