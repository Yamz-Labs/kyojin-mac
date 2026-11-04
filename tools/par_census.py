import sys, os
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000")
import numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, rows as R
from count_launches import graph_census, VIEWS
pack = sys.argv[1]
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg); R.enable(model, spec)
V = cfg["text_config"]["vocab_size"]
ids = np.random.default_rng(5).integers(16, V - 16, 136).tolist()
cache, lg = R.prefill(model, ids)
res = {}
for i in range(6):
    c2 = R.fork(cache) if i == 0 else None
    o = R.step(model, cache, mx.array([5 + i]))
    c = graph_census(o)
    mx.eval(o)
    res[i] = c
    print(i, "len", 136 + i, "dispatches", c["dispatches"])
for i in (2, 3):
    print(i, {k: v for k, v in res[i]["by_type"].items() if k not in VIEWS})
