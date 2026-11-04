"""Per-shape decode profile of the Qwen3.5-4B pack: Exl3Linear (1 row) vs mlx 4-bit g64 quantized_matmul on the same shape.
   python tools/prof_dec4b.py [--iters 100] [--tgk shape=tgk,..]   (random weights; time/bytes only)"""
import argparse, time
import numpy as np
import mlx.core as mx
from kyojin_mlx.exl3_linear import Exl3Linear

# (in, out, K, count per step, label)
SHAPES = [(2560, 1024, 4, 16, "k/v_proj"), (2560, 4096, 4, 24, "gdn z"), (2560, 8192, 4, 24 + 8, "gdn qkv + q_proj"), (2560, 9216, 4, 64, "gate/up"),
          (4096, 2560, 4, 32, "out/o_proj"), (9216, 2560, 4, 32, "down"), (2560, 248320, 6, 1, "lm_head")]

def timeit(fn, iters):
    for _ in range(5): mx.eval(fn())
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); outs = [fn() for _ in range(iters)]; mx.eval(outs); best = min(best, (time.perf_counter() - t0) / iters)
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=50); ap.add_argument("--only", default=None); ap.add_argument("--tgk", type=int, default=None); a = ap.parse_args()
    rng = np.random.default_rng(0); tot = {"exl3": 0.0, "mlx4": 0.0}
    for kin, nout, K, cnt, lab in SHAPES:
        if a.only and a.only not in lab: continue
        # a pool of distinct weights so the run is not served from cache
        n = max(1, min(8, int(300e6 // (kin * nout * K / 8))))
        lins = [Exl3Linear(mx.array(rng.integers(-32768, 32767, (kin // 16, nout // 16, 16 * K), dtype=np.int16)), mx.array(np.ones(kin, np.float16)), mx.array(np.ones(nout, np.float16)), cb="mul1", tune={"tgk": a.tgk} if a.tgk else None) for _ in range(n)]
        x = mx.array(rng.standard_normal((1, kin)).astype(np.float16)); mx.eval(x)
        i = [0]
        def f():
            i[0] += 1; return lins[i[0] % n](x)
        t = timeit(f, a.iters)
        by = kin * nout * K / 8
        qs = [mx.quantize(mx.array(rng.standard_normal((nout, kin)).astype(np.float16)), group_size=64, bits=4) for _ in range(min(n, 2) if nout < 100000 else 1)]
        mx.eval(qs)
        j = [0]
        def g():
            j[0] += 1; w, s, b = qs[j[0] % len(qs)]; return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
        t2 = timeit(g, a.iters)
        by2 = kin * nout * 4.5 / 8
        tot["exl3"] += t * cnt; tot["mlx4"] += t2 * cnt
        print(f"{lab:<18}{kin}x{nout} K{K} x{cnt:<3} exl3 {t*1e6:8.1f} us {by/t/1e9:6.1f} GB/s | mlx4 {t2*1e6:8.1f} us {by2/t2/1e9:6.1f} GB/s | ratio {t/t2:4.2f}  step-ms {t*cnt*1e3:5.2f} vs {t2*cnt*1e3:5.2f}", flush=True)
    print(f"TOTAL linears+head per step: exl3 {tot['exl3']*1e3:.2f} ms  mlx4 {tot['mlx4']*1e3:.2f} ms")
main()
