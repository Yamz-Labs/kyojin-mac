"""Memory returned by released regions: records many regions of a function with ~40 MB of scratch under a tiny budget (every build is released at once) and prints the
footprint (top) every 10 records. KYOJIN_REPLAY=1 KYOJIN_REPLAY_MB=1 python tools/replay_leak.py"""
import os, subprocess, sys
os.environ.setdefault("KYOJIN_REPLAY", "1")
import mlx.core as mx
from kyojin_mlx import replay as RP
w = mx.random.normal((1024, 1024)).astype(mx.float32); mx.eval(w)
def fn(x):
    h = x
    for _ in range(8):
        h = mx.tanh(h @ w)
    return h
f = RP.region(fn, "leak")
def foot():
    out = subprocess.run(["top", "-l", "1", "-pid", str(os.getpid()), "-stats", "mem"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return out
for i in range(80):
    rows = 2048 + i                              # a new shape: a new entry every 4 calls
    x = mx.random.normal((rows, 1024)); mx.eval(x)
    for _ in range(5):
        mx.eval(f(x))
    if i % 10 == 9:
        print(i + 1, "shapes | footprint", foot(), "| MLX active", f"{mx.get_active_memory() / 1e6:.0f} MB", RP.STATS, flush=True)
