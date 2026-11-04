"""flat92_real.py CTX [COOL_S]: plain decode ms/token right after a real CTX-token prefill (engine-like 2048 chunks, the last two prefix forks kept
alive like the prefix cache), then after COOL_S seconds idle, then with the same cache refilled with random K/V (flat92 synthetic state).
Separates real-state / prefill-heat effects from the step's own depth cost."""
import os, sys, json, time, statistics
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", s) for s in ("tests", "")]
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend, Q35Backend
CTX = int(sys.argv[1]); COOL = float(sys.argv[2]) if len(sys.argv) > 2 else 45
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + CTX - 37]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
cache = be.make_cache()
ca = [c for l, c in zip(be.lm.model.layers, cache) if not l.is_linear]
keep = []
t0 = time.perf_counter()
if os.environ.get("FLAT92_SYN") == "1":                      # synthetic state: 256 real tokens, then random K / V up to CTX (no prefill heat)
    be.prefill_chunk(cache, pids[:256]); A = -(-(CTX + 64) // 256) * 256
    for c in ca:
        c.keys = mx.random.normal((1, c.keys.shape[1], A, c.keys.shape[3])).astype(c.keys.dtype); c.values = mx.random.normal((1, c.values.shape[1], A, c.values.shape[3])).astype(c.values.dtype); c.offset = CTX - 37
    mx.eval(*[a for c in ca for a in (c.keys, c.values)]); pids = []
for i in range(0, len(pids), 2048):
    be.prefill_chunk(cache, pids[i:i + 2048])
    if (i + 2048) % 2048 == 0 and i + 2048 <= len(pids):
        keep = (keep + [be.fork(cache)])[-2:]
BURN = float(os.environ.get("FLAT92_BURN", "0"))           # seconds of fp16 matmul load before the decode loop (prefill-like heat, no state change)
if BURN:
    a = mx.random.normal((4096, 4096)).astype(mx.float16); tb = time.perf_counter()
    while time.perf_counter() - tb < BURN:
        mx.eval(*[a @ a for _ in range(8)])
print("prefill_s", round(time.perf_counter() - t0, 1), "alloc", ca[0].keys.shape, "offset", ca[0].offset, flush=True)
def trim(n):
    for c in ca: c.trim(n)
def pipe(n=16):
    bc = be.batch([cache]); y = mx.array([1000], dtype=mx.int32); mx.async_eval(y)
    t = time.perf_counter()
    for _ in range(n):
        yn = mx.argmax(Q35Backend.step(be, bc, y), -1).astype(mx.int32); mx.async_eval(yn); y.tolist(); y = yn
    y.tolist(); r = (time.perf_counter() - t) * 1e3 / n; trim(n); return r
def run(tag, reps=6):
    v = [pipe() for _ in range(reps)]
    print("REAL92", os.path.basename(T.PACK), CTX, tag, round(min(v), 2), round(statistics.median(v), 2), [round(x, 1) for x in v], flush=True)
import subprocess
def vm():
    o = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    g = {l.split(":")[0]: int(l.split(":")[1].strip().rstrip(".")) for l in o.splitlines()[1:] if ":" in l and l.split(":")[1].strip().rstrip(".").isdigit()}
    return g.get("Swapouts", 0), g.get("Pageouts", 0), g.get("Compressions", 0), g.get("Pages free", 0)
print("mem_GB active", round(mx.get_active_memory() / 2**30, 2), "cache", round(mx.get_cache_memory() / 2**30, 2), "peak", round(mx.get_peak_memory() / 2**30, 2), flush=True)
if os.environ.get("FLAT92_CLEAR") == "1":
    mx.clear_cache(); print("cleared cache", flush=True)
def others():
    o = subprocess.run(["ps", "-Ao", "pid,pcpu,command"], capture_output=True, text=True).stdout.splitlines()
    return [l.strip()[:70] for l in o if ("python" in l or "omlx" in l) and str(os.getpid()) not in l.split()[0] and float(l.split()[1].replace(",", ".")) > 5]
print("OTHERS", others(), flush=True)
v0 = vm(); t0 = time.perf_counter()
while time.perf_counter() - t0 < COOL:
    r = [pipe() for _ in range(3)]; v = vm()
    print("T92", round(time.perf_counter() - t0, 1), [round(x, 1) for x in r], "swapout/pageout/compress/free", [b - a for a, b in zip(v0, v)][:3], v[3], flush=True)
print("OTHERS_END", others(), flush=True)
