"""launch89.py MODEL_DIR: dispatch census of one decode step (mlx_lm model, TQ6 cache): by primitive type (views excluded), per-layer-kind counts."""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kmlx-macdisp", "tools"))
import mlx.core as mx
from mlx_lm import load
import kvwin_kl as K
model, tok = load(sys.argv[1]); lm = model.language_model; hd = K.head(lm)
cache = K.cache_for(model, [6, 6, 0, 4, 256])
ids = K.tokens(tok, 1024, 0)
h = lm.model(mx.array(ids)[None], cache); mx.eval(h); mx.clear_cache()
y = mx.argmax(hd(h[:, -1:])[0], axis=-1)
out = mx.argmax(hd(lm.model(y[None], cache))[0], axis=-1)
import count_launches as CL
r = CL.graph_census(out)
print("dispatches", r["dispatches"], "nodes", r["nodes"], "views", r["views"])
for k, v in r["by_type"].items():
    if k not in CL.VIEWS: print("  %-26s %5d" % (k, v))
