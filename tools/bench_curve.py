"""Noise-robust decode curve: one process, one model, one KV cache per mode prefilled to N, then timed blocks of decode steps INTERLEAVED across the modes
(the Air is shared: ambient load moves everything together, the ratio between modes survives). Reports min and p25 tok/s per mode over the blocks.
usage: bench_curve.py MODEL N BLOCKS STEPS_PER_BLOCK OUT_JSONL CFGS_JSON"""
import sys, time, os, json
import mlx.core as mx
from mlx_lm import load
import kvwin_kl as K
mdir, N, B, S = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); out = open(sys.argv[5], "a"); cfgs = json.load(open(sys.argv[6]))
model, tok = load(mdir); lm = model.language_model; hd = K.head(lm)
ids = K.tokens(tok, N, 0)
st = {}
for c in cfgs:
    cache = K.cache_for(model, c)
    for s in range(0, N, 512):
        h = lm.model(mx.array(ids[s:s + 512])[None], cache); mx.eval(h); mx.clear_cache()
    st[K.name(c)] = [cache, mx.argmax(hd(h[:, -1:])[0], axis=-1), []]
    print("prefilled", K.name(c), f"peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)
def block(n):
    cache, y, _ = st[n]
    t0 = time.perf_counter()
    for i in range(S):
        h = lm.model(y[None], cache); y = mx.argmax(hd(h)[0], axis=-1); mx.eval(y)
    dt = time.perf_counter() - t0; st[n][1] = y; return S / dt
for r in range(B + 1):
    for c in cfgs:
        v = block(K.name(c))
        if r: st[K.name(c)][2].append(v)
for c in cfgs:
    v = sorted(st[K.name(c)][2]); rec = dict(cfg=K.name(c), c=c, N=N, tok_s_max=v[-1], tok_s_p25=v[len(v) // 4], tok_s_med=v[len(v) // 2], blocks=B, steps=S, peak_gib=mx.get_peak_memory() / 2**30)
    print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
