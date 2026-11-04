"""Bit-exactness of the fused layer seams: logits with layer_pf on vs off, and mlp_pf off (same process). usage: pf2k_eq2.py MODEL_DIR rows..."""
import sys, mlx.core as mx
from kyojin_mlx import mlp_pf as MP, layer_pf as LP
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); mx.set_cache_limit(256 * 2 ** 20)
for R in map(int, sys.argv[2:]):
    ids = [(i * 7919) % 5000 + 100 for i in range(R)]
    out = {}
    for name, (mp, lp) in {"stock": (False, False), "mlp": (True, False), "layer": (True, True)}.items():
        MP.ENABLED, LP.ENABLED = mp, lp
        c = be.make_cache(); lg = be.prefill_chunk(c, ids); mx.eval(lg); out[name] = lg
    for k in ("mlp", "layer"):
        print(R, k, "vs stock bitexact", bool(mx.array_equal(out["stock"], out[k]).item()), "maxdiff", mx.abs(out["stock"].astype(mx.float32) - out[k].astype(mx.float32)).max().item(), flush=True)
