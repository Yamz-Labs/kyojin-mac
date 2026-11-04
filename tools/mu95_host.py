"""Host cost of one batched multi-user decode step (graph build + async_eval encode), cProfile top functions, N users on a fp16 4K state.
usage: mu95_host.py MODEL_DIR N[,N..] [CTX=4096] [TOP=25]"""
import sys, os, time, cProfile, pstats, io
import mlx.core as mx
from kyojin_mlx import mlxpack, q4mma as QM, multiuser as MU, step_regions as SR
from mlx_lm.models.cache import KVCache
mdir = sys.argv[1]; Ns = [int(v) for v in sys.argv[2].split(",")]; CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
TOP = int(sys.argv[4]) if len(sys.argv) > 4 else 25
model, tok = mlxpack.load(mdir)
if os.environ.get("KYOJIN_Q4MMA", "1") != "0": QM.install(model)
MU.install(model)
ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 120)[:1024]
base, lg = MU.prefill(model, ids)
def tiled():
    c2 = MU.fork_cache(base)
    for c in c2:
        if isinstance(c, KVCache):
            r = -(-CTX // c.offset)
            c.keys = mx.contiguous(mx.tile(c.keys[:, :, :c.offset], (1, 1, r, 1))[:, :, :CTX]); c.values = mx.contiguous(mx.tile(c.values[:, :, :c.offset], (1, 1, r, 1))[:, :, :CTX]); c.offset = CTX
    mx.eval([c.state for c in c2]); return c2
y0 = mx.argmax(lg, axis=-1)
for N in Ns:
    bc = MU.batch_caches([tiled() for _ in range(N)], prefix_len=int(os.environ.get("MU95_PREFIX", "0"))); y = mx.concatenate([y0] * N)
    step = lambda y: mx.argmax(MU.logits_last(model, y[:, None], bc), axis=-1)
    for _ in range(3): y = step(y); mx.eval(y)
    s0 = dict(SR.STATS)
    bt, et, tt = [], [], []
    for _ in range(8):
        t0 = time.perf_counter(); r = step(y); t1 = time.perf_counter(); mx.async_eval(r); t2 = time.perf_counter(); mx.eval(r); t3 = time.perf_counter(); y = r
        bt.append((t1 - t0) * 1e3); et.append((t2 - t1) * 1e3); tt.append((t3 - t0) * 1e3)
    md = lambda v: sorted(v)[len(v) // 2]
    print(f"N={N}: build {md(bt):.1f} ms  encode {md(et):.1f} ms  total {md(tt):.1f} ms  regions fast/fallback +{SR.STATS['fast']-s0['fast']}/+{SR.STATS['fallback']-s0['fallback']}", flush=True)
    pr = cProfile.Profile(); pr.enable()
    for _ in range(4): r = step(y); mx.async_eval(r); mx.eval(r); y = r
    pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(TOP); 
    print("\n".join(l for l in s.getvalue().splitlines()[6:] if l.strip())[:4000], flush=True)
