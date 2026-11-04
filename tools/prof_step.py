"""Real-model decode step at depth N with a TQ cache: full step vs attention skipped vs fused kernel off. usage: prof_step.py MODEL N [steps]"""
import sys, time, os
import mlx.core as mx
from mlx_lm import load
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from kyojin_mlx import kvwin as KW, tq
import kvwin_kl as K
mdir, N = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
model, tok = load(mdir); lm = model.language_model; hd = K.head(lm)
ids = K.tokens(tok, N, 0)
cache = K.cache_for(model, [6, 6, 0, 4, 256])
for s in range(0, N, 512):
    h = lm.model(mx.array(ids[s:s + 512])[None], cache); mx.eval(h); mx.clear_cache()
y = mx.argmax(hd(h[:, -1:])[0], axis=-1)
def run(label, n=steps):
    global y
    ts = []
    for i in range(n + 3):
        t0 = time.perf_counter(); h = lm.model(y[None], cache); y = mx.argmax(hd(h)[0], axis=-1); mx.eval(y); ts.append(time.perf_counter() - t0)
    ts = sorted(ts[3:]); print(f"{label}: N={N} min {ts[0]*1e3:.1f} p25 {ts[len(ts)//4]*1e3:.1f} ms/step ({1/ts[len(ts)//4]:.1f} tok/s)", flush=True)
orig = tq.TQKVCache.attend
for rnd in range(2):
    run("tq fused")
    tq.TQKVCache.attend = lambda self, queries, scale: queries
    run("attend skipped")
    tq.TQKVCache.attend = orig
    os.environ["KYOJIN_TQ_DECODE"] = "0"
    for c in cache:
        if hasattr(c, "fused"): c.fused = False
    run("generic dequant path", 5)
    for c in cache:
        if hasattr(c, "fused"): c.fused = True
