"""Prefill vs the matmul peak, in process (no server). One fresh prompt per length, served chunk grid.
  peak:  bench_pf_roof.py peak            fp16 matmul peak (large shapes, interleaved, min)
  run:   bench_pf_roof.py run MODEL T[,T..] [--chunk 2048] [--peak TF] [--params N]
Active params (2 x params x tokens = matmul FLOPs, attention FLOPs not counted) come from the GGUF metadata, or --params for a pack dir.
Prints tok/s, TFLOPS and % of peak per length; texts of the last-row argmax are printed so arms compare."""
from hashlib import md5
import argparse, os, sys, time
import numpy as np, mlx.core as mx

def peak():
    best = 0.0
    for M, K, N in [(8192, 4096, 4096), (8192, 5120, 17408), (4096, 8192, 8192), (8192, 2048, 6144)]:
        x = mx.random.normal((M, K)).astype(mx.float16); w = mx.random.normal((N, K)).astype(mx.float16); mx.eval(x, w)
        for _ in range(3): mx.eval(x @ w.T)
        v = []
        for _ in range(20):
            mx.synchronize(); t = time.perf_counter(); mx.eval(x @ w.T); v.append(time.perf_counter() - t)
        tf = 2 * M * K * N / min(v) / 1e12; best = max(best, tf); print(f"fp16 {M}x{K}x{N}: {tf:.1f} TFLOPS")
    print(f"PEAK {best:.1f} TFLOPS"); return best

def active_params(path):
    from kyojin_mlx.foreign_gguf import GgufFile, find_gguf
    g = GgufFile(find_gguf(path)); tot = 0.0
    ne = next((v for k, v in g.kv.items() if k.endswith(".expert_count")), 0); nu = next((v for k, v in g.kv.items() if k.endswith(".expert_used_count")), 0)
    for n, t in g.tensors.items():
        if "token_embd" in n or len(t.ne) < 2: continue
        e = float(np.prod(t.ne))
        tot += e * nu / ne if len(t.ne) == 3 and "_exps" in n else e
    return tot

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("model", nargs="?"); ap.add_argument("T", nargs="?")
    ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--peak", type=float, default=0); ap.add_argument("--params", type=float, default=0)
    a = ap.parse_args()
    if a.mode == "peak": peak(); sys.exit()
    from kyojin_mlx.serve.backends import load_backend
    be, cfg = load_backend(a.model)
    params = a.params or active_params(a.model)
    print(f"active params {params/1e9:.2f} B", flush=True)
    for T in [int(v) for v in a.T.split(",")]:
        ids = np.random.default_rng(T).integers(1000, 100000, T).tolist()     # fresh prompt per length
        mx.reset_peak_memory(); cache = be.make_cache(); t0 = time.perf_counter()
        for s in range(0, T, a.chunk): lg = be.prefill_chunk(cache, ids[s:s + a.chunk])
        dt = time.perf_counter() - t0; tf = 2 * params * T / dt / 1e12
        print(f"T={T} chunk={a.chunk}: {dt:.2f} s {T/dt:.0f} tok/s {tf:.1f} TFLOPS" + (f" {100*tf/a.peak:.0f} % of peak" if a.peak else "") + f" top5 {np.argsort(-np.array(lg.reshape(-1)))[:5].tolist()} lgmd5 {md5(np.array(lg.reshape(-1)).tobytes()).hexdigest()[:8]} peak {mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
