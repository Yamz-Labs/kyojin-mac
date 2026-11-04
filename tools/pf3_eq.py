"""Bit-exactness of the layer seams: logits and cache state of one prefill chunk, seams off (mlp_pf on) vs fused vs split, plus all fusions off. usage: pf3_eq.py MODEL_DIR rows [rows...]"""
import sys, mlx.core as mx
from kyojin_mlx import mlp_pf as MP, layer_pf as LP
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); mx.set_cache_limit(256 * 2 ** 20)
def flat(x):
    if isinstance(x, mx.array): return [x]
    if isinstance(x, (list, tuple)): return [a for y in x for a in flat(y)]
    return []
def setv(v):
    MP.ENABLED = v != "none"
    LP.ENABLED = v in ("full", "split")
    LP.MAX_ROWS, LP.SPLIT_MAX_ROWS = (1 << 20, 0) if v == "full" else ((0, 1 << 20) if v == "split" else (0, 0))
for R in map(int, sys.argv[2:]):
    ids = [(i * 7919) % 5000 + 100 for i in range(R)]; out = {}
    for v in ("none", "stock", "full", "split"):
        setv(v); c = be.make_cache(); lg = be.prefill_chunk(c, ids); mx.eval(lg)
        st = flat([getattr(ci, "state", []) for ci in c]); mx.eval(*st); out[v] = (lg, st)
    for v in ("stock", "full", "split"):
        same = bool(mx.array_equal(out[v][0], out["none"][0]).item()) and all(bool(mx.array_equal(a, b).item()) for a, b in zip(out[v][1], out["none"][1]))
        print(f"rows {R}: {v:6s} vs all-off bit-identical (logits + cache): {same}", flush=True)
