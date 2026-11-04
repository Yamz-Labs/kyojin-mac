"""Host gap of one decode token. usage: host_gap.py exl3|mlx MODEL_DIR [--reps 40] [--ctx 512]
Per token (medians/mins over reps): build (python + C++ graph creation, no eval), encode (async_eval return), wait (rest until the token is ready),
sync step (build+eval), pipelined step (mlx-lm style async: next graph built while the GPU runs the current one).
GPU busy ~ pipelined step when the host is hidden; host gap = sync step - pipelined step (what async already hides) and pipelined - GPU floor."""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
if os.environ.get("M10_LIGHT"):
    import light
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("path"); ap.add_argument("--reps", type=int, default=40); ap.add_argument("--ctx", type=int, default=512); ap.add_argument("--regions", action="store_true")
a = ap.parse_args()
if a.kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(a.path)
else:
    from mlx_lm import load
    model, _ = load(a.path)
if a.regions:
    from kyojin_mlx import step_regions as SR
    print("regions", SR.enable(model))
text = "The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 100
ids = AutoTokenizer.from_pretrained(a.path).encode(text)[:a.ctx]
cache = make_prompt_cache(model)
lg = model(mx.array(ids)[None], cache=cache)[:, -1]; t = mx.argmax(lg, axis=-1); mx.eval(t)
step = lambda y: mx.argmax(model(y[:, None], cache=cache)[:, -1], axis=-1)
for _ in range(8): t = step(t); mx.eval(t)
B, E, W = [], [], []
for _ in range(a.reps):
    t0 = time.perf_counter(); nt = step(t); t1 = time.perf_counter(); mx.async_eval(nt); t2 = time.perf_counter(); mx.eval(nt); t3 = time.perf_counter()
    t = nt; B.append(t1 - t0); E.append(t2 - t1); W.append(t3 - t2)
S = []
for _ in range(a.reps):
    t0 = time.perf_counter(); t = step(t); mx.eval(t); S.append(time.perf_counter() - t0)
P = []
for _ in range(3):
    n = 64; t0 = time.perf_counter(); nxt = step(t); mx.async_eval(nxt)
    for i in range(n):
        cur = nxt
        if i + 1 < n: nxt = step(cur); mx.async_eval(nxt)
        cur.item()
    P.append((time.perf_counter() - t0) / n); t = cur
f = lambda v: f"med {1e3*statistics.median(v):6.2f} min {1e3*min(v):6.2f}"
print(f"{a.kind} {os.path.basename(a.path)} ctx {a.ctx}")
print("build  ", f(B)); print("encode ", f(E)); print("wait   ", f(W)); print("sync   ", f(S)); print("pipe   ", f(P), f"-> {1/min(P):.1f} tok/s best, {1/statistics.median(P):.1f} median")
