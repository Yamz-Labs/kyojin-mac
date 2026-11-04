"""Where does the prefill time of an Exl3Linear go? usage: prof_prefill.py MODEL_DIR [rows=512]"""
import sys, time, json
import mlx.core as mx
from kyojin_mlx import kernels as KN
from kyojin_mlx.exl3_linear import Exl3Linear
d = sys.argv[1]; rows = int(sys.argv[2]) if len(sys.argv) > 2 else 512
cb = json.load(open(f"{d}/config.json"))["quantization_config"]["codebook"]
w = mx.load(f"{d}/model.safetensors")
def timeit(fn, it=20):
    for _ in range(3): mx.eval(fn())
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(it): mx.eval(fn())
    return (time.perf_counter() - t0) / it * 1e3
seen = set()
for p in sorted(k[:-8] for k in w if k.endswith(".trellis")):
    t = w[p + ".trellis"]; key = (t.shape, )
    if key in seen: continue
    seen.add(key)
    lin = Exl3Linear(t, w[p + ".suh"], w[p + ".svh"], cb=cb)
    x = (mx.random.normal((rows, lin.in_features)) * 0.7).astype(mx.float16)
    xh = KN.rot(x, lin._suh, mx.float16)
    wr = KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, cb); mx.eval(wr, xh)
    tdec = timeit(lambda: KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, cb))
    tmm = timeit(lambda: xh @ wr.T)
    trot = timeit(lambda: KN.rot(x, lin._suh))
    tall = timeit(lambda: lin(x))
    gflop = 2 * rows * lin.in_features * lin.out_features / 1e9
    print(f"{p[-34:]:34s} K={lin.K} {lin.in_features}x{lin.out_features}: decode {tdec:6.2f} ms ({lin.in_features*lin.out_features/tdec/1e6:5.0f} Gw/s)  "
          f"matmul {tmm:6.2f} ms ({gflop/tmm:5.1f} TFLOPs)  rot {trot:5.2f}  total {tall:6.2f} ms", flush=True)
