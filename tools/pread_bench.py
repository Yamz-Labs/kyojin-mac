"""Micro-bench of EngramTables.read_rows_into: threads vs rows (hot page cache). Run from the worktree with PYTHONPATH set."""
import sys, time, statistics, json, os
import numpy as np
from kyojin_mlx import engram_rows as ER
pack = sys.argv[1]
cfg = json.load(open(os.path.join(pack, "config.json")))
es = ER.EngramSpec.from_config(cfg)
for threads in (1, 4, 8, 16):
    t = ER.EngramTables(pack, es, threads=threads) if "threads" in ER.EngramTables.__init__.__code__.co_varnames else ER.EngramTables(pack, es)
    L = len(es.layer_ids)
    rows = [t.rows(p) if hasattr(t, "rows") else 3000 for p in range(L)]
    rng = np.random.default_rng(0)
    for ncols in (24,):
        reqs = []
        for p in range(L):
            r = rng.integers(0, 3000, ncols)
            reqs.append((p, r, np.zeros((ncols, es.head_dim), np.uint8), np.zeros((ncols, es.head_dim // 32), np.uint8)))
        ts = []
        for _ in range(400):
            t0 = time.perf_counter(); t.read_rows_into(reqs); ts.append((time.perf_counter() - t0) * 1e3)
        ts = ts[20:]
        print(f"threads {threads:2d} layers {L} cols {ncols} -> {L*ncols} rows: min {min(ts):.3f} med {statistics.median(ts):.3f} p90 {sorted(ts)[int(len(ts)*.9)]:.3f} ms")
    t.close()
