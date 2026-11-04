"""ops114.py [T=32768]: per-op ms of the DS4.1 verify attention ops (index_scores, argsort top-k, sort, attend) for n = 1..5 rows of one session at T compressed keys."""
import sys, time, os, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
from kyojin_mlx import dsv4_core as DC
T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
dt = mx.bfloat16
NH, DI, H, D, W, K = 32, 128, 64, 512, 128, 512
keys = mx.random.normal((1, T + 8, DI)).astype(dt)
pool = mx.random.normal((1, T + 8, D)).astype(dt)
def bench(f, reps=15):
    for _ in range(3): mx.eval(f())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)
print(f"T={T} compressed keys; ms per op (median of 15), layers: multiply by 2 for the slice")
print("n  iscore  argsort+slice  sort+where  attend")
for n in range(1, 7):
    q = mx.random.normal((n, NH, DI)).astype(dt); w = mx.random.normal((n, NH)).astype(dt)
    lens = mx.full((n,), T, dtype=mx.int32); sess = mx.zeros((n,), dtype=mx.uint32)
    sc = DC.index_scores(q, keys, w, lens, T, None, sess)[:, None, :]; mx.eval(sc)
    a = bench(lambda: DC.index_scores(q, keys, w, lens, T, None, sess))
    b = bench(lambda: mx.argsort(-sc, axis=-1)[..., :K])
    ids = mx.argsort(-sc, axis=-1)[..., :K]; mx.eval(ids)
    c = bench(lambda: mx.where((s := mx.sort(ids, axis=-1).astype(mx.int32)) < lens[:, None, None], s, -1))
    qq = mx.random.normal((n, H, D)).astype(dt); win = mx.random.normal((n, W, D)).astype(dt)
    idx = mx.broadcast_to(mx.arange(K, dtype=mx.int32)[None] * 7, (n, K)); pos = mx.full((n,), T * 2, dtype=mx.int32)
    sink = mx.zeros((H,), dtype=mx.float32)
    d = bench(lambda: DC.attend(qq, win, pool, idx, pos, sink, 0.044, K, sess))
    print(f"{n}  {a:6.3f}  {b:6.3f}  {c:6.3f}  {d:6.3f}", flush=True)
