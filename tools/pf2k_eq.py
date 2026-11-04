"""Bit-exactness of the fused MLP middle: logits of a prefill chunk with mlp_pf on vs off (same process). usage: pf2k_eq.py MODEL_DIR rows [rows...]"""
import sys, time, mlx.core as mx
from kyojin_mlx import mlp_pf as MP
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); mx.set_cache_limit(256 * 2 ** 20)
for R in map(int, sys.argv[2:]):
    ids = [(i * 7919) % 5000 + 100 for i in range(R)]
    out = {}
    for en in (False, True):
        MP.ENABLED = en
        c = be.make_cache(); lg = be.prefill_chunk(c, ids); mx.eval(lg); out[en] = lg
        c = be.make_cache(); lg = be.prefill_chunk(c, ids); mx.eval(lg)
    a, b = out[False], out[True]
    print(R, "logits bitexact", bool(mx.array_equal(a, b).item()), "maxdiff", mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item(), flush=True)
