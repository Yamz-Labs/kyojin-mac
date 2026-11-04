"""K6 head Viterbi throughput: impl 1 (NT / table) vs impl 2 forced to K6, and the quantize_tiles chunk size. usage: head_bench.py [tiles]"""
import sys, time, itertools, numpy as np, mlx.core as mx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kyojin_mlx import conv_viterbi as V
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
rng = np.random.default_rng(0)
tiles = mx.array((rng.standard_normal((N, 256)) * 0.9).astype(np.float32)); mx.eval(tiles)
V.CFG["impl"] = 1
ref_w, ref_i = V.quantize_tiles(tiles, 6); mx.eval(ref_w, ref_i); ref_i = np.array(ref_i)
def run(name, reps=3, **kw):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); w, i = V.quantize_tiles(tiles, 6, **kw); mx.eval(w, i); ts.append(time.perf_counter() - t0)
    same = bool((np.array(i) == ref_i).all())
    print(f"{name}: min {N / min(ts):7.0f} tiles/s  p25 {N / sorted(ts)[len(ts) // 4]:7.0f}  same_as_impl1_default={same}", flush=True)
V.CFG["v2max"] = 6
for impl, nt, tab in [(1, 256, True), (1, 256, False), (1, 512, True), (1, 1024, True), (1, 1024, False)]:
    V.CFG.update(impl=impl, NT=nt, table=tab)
    try: run(f"impl1 NT{nt} table={tab}")
    except Exception as e: print("impl1", nt, tab, "ERR", repr(e)[:150])
for nt2 in (512, 256, 128):
    V.CFG.update(impl=2, NT2=nt2, S2=0)
    try: run(f"impl2 NT2={nt2}")
    except Exception as e: print("impl2", nt2, "ERR", repr(e)[:150])
V.CFG.update(impl=1, NT=256, table=True)
for ch in (64, 128, 256, 512, 1024):
    run(f"impl1 default chunk={ch}", chunk=ch)
