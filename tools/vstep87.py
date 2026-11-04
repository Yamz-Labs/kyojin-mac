"""vstep87.py KIND MODEL N [T=5] [REPS=8]: whole-model forward of a T-token verify chunk at depth N (synthetic: 1K prefill tiled), TQ6, then rollback; ms per chunk, and per single-token step for reference."""
import sys, os, time, json
import mlx.core as mx
from kyojin_mlx import tq
import kvwin_kl as K
from mlx_lm import load
mdir, N = sys.argv[1], int(sys.argv[2]); T = int(sys.argv[3]) if len(sys.argv) > 3 else 5; REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
model, tok = load(mdir); lm = model.language_model; hd = K.head(lm)
ids = K.tokens(tok, 1024, 0); cache = K.cache_for(model, [6, 6, 0, 4, 256])
h = lm.model(mx.array(ids)[None], cache); mx.eval(h)
r = -(-N // 1024)
for c in cache:
    if isinstance(c, tq.TQKVCache):
        st = c.state; c.state = [mx.tile(x[:, :, :1024], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
mx.eval([c.state for c in cache]); mx.clear_cache()
toks = mx.array(ids[:T])[None]
def snap():
    return [c.state if not isinstance(c, tq.TQKVCache) else None for c in cache]
saved = [list(c.state) if not isinstance(c, tq.TQKVCache) and c.state else None for c in cache]
def rollback():
    for c, s in zip(cache, saved):
        if isinstance(c, tq.TQKVCache): c.trim(T)
        elif s is not None: c.state = list(s)
def chunk(t):
    out = hd(lm.model(t, cache)); mx.eval(out); rollback()
res = {}
for name, t in (("chunk%d" % T, toks), ("single", toks[:, :1])):
    saved = [list(c.state) if not isinstance(c, tq.TQKVCache) and c.state else None for c in cache]
    ts = []
    for _ in range(REPS + 2):
        t0 = time.perf_counter(); out = hd(lm.model(t, cache)); mx.eval(out); ts.append((time.perf_counter() - t0) * 1e3)
        for c, s in zip(cache, saved):
            if isinstance(c, tq.TQKVCache): c.trim(t.shape[1])
            elif s is not None: c.state = list(s)
    res[name] = round(sorted(ts[2:])[0], 1)
print(json.dumps(dict(src=tq.__file__.split("/")[-3], N=N, T=T, **res)), flush=True)
