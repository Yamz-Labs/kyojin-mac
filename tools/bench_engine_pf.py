"""Served-path 4K prefill without HTTP: Engine(sessions=1), ttft of max_new=1. usage: bench_engine_pf.py MODEL_DIR [--chunk 256] [--block 512] [--reps 4] [--prefix-mb 1024]"""
import argparse, threading, time, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--T", type=int, default=4096)
ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--block", type=int, default=512)
ap.add_argument("--reps", type=int, default=4); ap.add_argument("--prefix-mb", type=int, default=1024); ap.add_argument("--cache-mb", type=int, default=256)
a = ap.parse_args()
mx.set_cache_limit(a.cache_mb * 2 ** 20)
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve.engine import Engine
tok = AutoTokenizer.from_pretrained(a.model)
base = tok.encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 600)
backend, _ = load_backend(a.model)
eng = Engine(backend, sessions=1, ctx=32768, chunk=a.chunk, block=a.block, prefix_bytes=a.prefix_mb * 2 ** 20, log=lambda *x: None)
res = []
def client():
    for r in range(a.reps + 1):
        ids = [1000 + r] + base[:a.T - 1]                       # distinct first token: no prefix hit
        t0 = time.perf_counter(); h = eng.submit(ids, max_new=1)
        while True:
            ev = h.events.get()
            if ev[0] != "tok": break
        res.append(a.T / (time.perf_counter() - t0))
    eng.stop()
th = threading.Thread(target=client); th.start(); eng.run(); th.join()
r = sorted(res[1:], reverse=True)
print(f"engine T={a.T} chunk={a.chunk} block={a.block} prefix={a.prefix_mb}MB: best {r[0]:.0f} p25 {r[len(r)//4]:.0f} tok/s", flush=True)
