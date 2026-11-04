"""cProfile of the python graph build of one decode step (not the encode): usage prof_py.py PACK [--steps 20] [--top 40] [--sort tottime|cumtime]"""
import argparse, cProfile, os, pstats, sys, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--top", type=int, default=40); ap.add_argument("--sort", default="tottime")
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
lm = model.language_model
ids = np.random.default_rng(0).integers(16, cfg["text_config"]["vocab_size"] - 16, a.len).tolist()
cache = lm.make_cache()
lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg)
tok = mx.argmax(lg, axis=-1)
for _ in range(3):
    lg = R.step(model, cache, tok); mx.eval(lg); tok = mx.argmax(lg, axis=-1)
pr = cProfile.Profile()
for _ in range(a.steps):
    pr.enable(); lg = R.step(model, cache, tok); pr.disable()
    mx.eval(lg); tok = mx.argmax(lg, axis=-1)
st = pstats.Stats(pr); st.sort_stats(a.sort)
s = io.StringIO(); st.stream = s; st.print_stats(a.top)
print("\n".join(l[:170] for l in s.getvalue().splitlines() if l.strip())[:9000])
