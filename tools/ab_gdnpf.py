"""A/B of one GatedDeltaNet layer, stock vs gdn_pf, on the real layer weights. usage: ab_gdnpf.py MODEL_DIR [rows=2048] [rounds=6]
Parity: output + conv state + recurrent state vs stock from the same cache; time: interleaved, min / p25 (ms per layer call, sustained back to back)."""
import sys, time, mlx.core as mx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mlx_lm import load
from mlx_lm.models import qwen3_5 as Q5
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx import gdn_pf
path = sys.argv[1]; S = int(sys.argv[2]) if len(sys.argv) > 2 else 2048; R = int(sys.argv[3]) if len(sys.argv) > 3 else 6
model, _ = load(path); lm = model.language_model
li = next(i for i, l in enumerate(lm.model.layers) if l.is_linear); layer = lm.model.layers[li].linear_attn
cache = make_prompt_cache(model)[li]
mx.random.seed(1); x = (mx.random.normal((1, S, lm.args.hidden_size)) * 0.7).astype(mx.bfloat16)
stock = Q5.GatedDeltaNet.__call__
gdn_pf.install(); fused = Q5.GatedDeltaNet.__call__; Q5.GatedDeltaNet.__call__ = stock
def run(fn, c):
    Q5.GatedDeltaNet.__call__ = fn; y = layer(x, cache=c); Q5.GatedDeltaNet.__call__ = stock; return y
# parity from a non-trivial carried state (two chunks)
def fresh():
    c = make_prompt_cache(model)[li]; run(stock, c); mx.eval(c.state); return c
c0, c1 = fresh(), fresh()
y0 = run(stock, c0); y1 = run(fused, c1); mx.eval(y0, y1, c0.state, c1.state)
d = lambda p, q: float(mx.max(mx.abs(p.astype(mx.float32) - q.astype(mx.float32))))
print(f"parity S={S}: out max|d| {d(y0, y1):.4g} (|y|max {float(mx.max(mx.abs(y0.astype(mx.float32)))):.3g}), conv state {d(c0[0], c1[0]):.4g}, rec state {d(c0[1], c1[1]):.4g} (|s|max {float(mx.max(mx.abs(c0[1]))):.3g}), "
      f"out bit-equal frac {float(mx.mean((y0 == y1).astype(mx.float32))):.4f}")
res = {"stock": [], "fused": []}
for fn, nm in ((stock, "stock"), (fused, "fused")): run(fn, c0); mx.eval(c0.state)
for r in range(R):
    for nm, fn in (("stock", stock), ("fused", fused)) if r % 2 == 0 else (("fused", fused), ("stock", stock)):
        mx.synchronize(); t0 = time.perf_counter(); ys = [run(fn, c0) for _ in range(4)]; mx.eval(ys, c0.state); res[nm].append((time.perf_counter() - t0) / 4 * 1e3)
for nm, v in res.items(): v.sort(); print(f"{nm:6s} min {v[0]:.3f} ms  p25 {v[len(v)//4]:.3f} ms")
print(f"speedup min {res['stock'][0]/res['fused'][0]:.2f}x  p25 {res['stock'][len(res['stock'])//4]/res['fused'][len(res['fused'])//4]:.2f}x  (layer incl. its 5 GEMMs)")
