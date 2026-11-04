"""spec114_round.py PACK [--ctx 65536] [--n 48] [--depths 0,1,2,3]: served drafted-round cost split (draft / verify build+sample / commit) at long context,
forced depth per run (depth_override), engine stats deltas per round. Prompt is a repeated filler (as ds41_bar --long)."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve.engine import Engine
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--ctx", type=int, default=65536); ap.add_argument("--n", type=int, default=48)
ap.add_argument("--depths", default="0,1,2,3"); ap.add_argument("--reps", type=int, default=2)
a = ap.parse_args()
backend, cfg = load_backend(a.pack, mtp=True, ced=False, log=lambda *x: None)
vocab = cfg.get("text_config", cfg)["vocab_size"]
eng = Engine(backend, sessions=1, ctx=a.ctx + a.n + 64, chunk=4096, block=4096, prefix_bytes=0, mtp_depth=3, kv_budget=None, row_cost=0.4, draft_cost=0.15, log=lambda *x: None)
import threading
import numpy as np
rng = np.random.default_rng(3)
base = rng.integers(16, vocab - 16, 997).tolist()
def ids(n, salt): return [16 + salt] + (base * (n // len(base) + 1))[:n - 1]
def run(depth, salt, n):
    eng.depth_override = depth
    s0 = dict(eng.stats)
    h = eng.submit(ids(a.ctx, salt), max_new=n, stop_ids=())
    toks, tf, tl = [], None, None
    while True:
        ev = h.events.get()
        if ev[0] != "tok": break
        tl = time.perf_counter(); tf = tf or tl; toks.append(ev[1])
    d = {k: eng.stats.get(k, 0) - s0.get(k, 0) for k in eng.stats}
    return toks, (len(toks) - 1) / max(tl - tf, 1e-9), d
def client():
    try:
        run(0, 1, 8)
        for dep in [int(x) for x in a.depths.split(",")]:
            for r in range(a.reps):
                toks, tps, d = run(dep, 2 + r, a.n)
                R = d.get("spec_rounds", 0)
                ms = lambda k: 1e3 * d.get(k, 0) / max(R, 1)
                print(f"depth {dep} rep {r}: {tps:.1f} tok/s rounds {R} draft {ms('spec_t_draft'):.1f} verify {ms('spec_t_verify'):.1f} commit {ms('spec_t_commit'):.1f} ms/round rows/round {d.get('spec_rows',0)/max(R,1):.1f}", flush=True)
    finally:
        eng.stop()
threading.Thread(target=client, daemon=True).start()
eng.run()
