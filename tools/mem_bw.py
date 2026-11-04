"""Real memory bandwidth of the GPU: stream-read GB/s and copy GB/s with Metal kernels (mx.fast.metal_kernel), large buffer, warm-up, min/median of N runs.
usage: mem_bw.py [--gb 8] [--runs 10] [--warmup 3] [--threads 1048576] [--tg 256] [--json F] [--allow-small]

read : every thread strides through the buffer with 16-byte loads (uint4) and XORs them; one word per thread is written back so the loads cannot be removed.
       GB/s = buffer bytes / time (GB = 1e9 bytes).
copy : the same stride, a 16-byte load and a 16-byte store of the output buffer. Reported as copy GB/s = bytes moved in total (read + write = 2 x buffer) / time, and
       as the per-direction rate (buffer / time), the figure most vendors quote for a copy.
Each run is timed alone (mx.eval + synchronize around it); best = fastest run (the bandwidth the hardware can deliver), median = typical, worst. The buffer is filled with
non-zero pseudo-random words. Peak memory: 1 x buffer for read, 2 x for copy (+ the fill temporary: 2 x buffer at the start). --gb below 8 is refused unless --allow-small
(an Ultra has caches far smaller than 8 GB, but a small buffer on the Air is a dry run only). Run it on a quiet machine (the Ultra: nothing else; the Air: wrap in lockf)."""
import argparse, json, statistics, time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--gb", type=float, default=8.0); ap.add_argument("--runs", type=int, default=10); ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--threads", type=int, default=1 << 20, help="threads in the grid (each strides through the buffer)"); ap.add_argument("--tg", type=int, default=256)
ap.add_argument("--json"); ap.add_argument("--allow-small", action="store_true")
a = ap.parse_args()
if a.gb < 8 and not a.allow_small:
    raise SystemExit("mem_bw: --gb < 8 refused (caches and a short run would flatter the figure); --allow-small for a dry run")
nbytes = int(a.gb * 1e9) // 16 * 16
n4 = nbytes // 16
dev = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
print(f"device: {dev.get('device_name')}  memory {dev.get('memory_size', 0) / 2 ** 30:.0f} GiB  max recommended working set {dev.get('max_recommended_working_set_size', 0) / 2 ** 30:.0f} GiB", flush=True)
need = nbytes * 2 / 2 ** 30
if need > dev.get("max_recommended_working_set_size", 1 << 62) / 2 ** 30 * 0.8:
    raise SystemExit(f"mem_bw: copy needs 2 x {nbytes / 2 ** 30:.1f} GiB, more than 80 % of the recommended working set: lower --gb (or raise iogpu.wired_limit_mb)")

HDR = "uint tid = thread_position_in_grid.x; uint nt = threads_per_grid.x; device const uint4* p = (device const uint4*)inp;\n"
read_k = mx.fast.metal_kernel(
    name="bw_read", input_names=["inp"], output_names=["out"],
    source=HDR + "uint4 acc = uint4(0); for (ulong i = tid; i < N4; i += nt) { acc ^= p[i]; } out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;")
copy_k = mx.fast.metal_kernel(
    name="bw_copy", input_names=["inp"], output_names=["out"],
    source=HDR + "device uint4* q = (device uint4*)out; for (ulong i = tid; i < N4; i += nt) { q[i] = p[i]; }")

# the buffer: pseudo-random words (a multiplicative hash of the index), built in 1 GB pieces so the temporaries stay small
buf = mx.zeros((nbytes // 4,), dtype=mx.uint32)
piece = 1 << 28
parts = []
for s in range(0, nbytes // 4, piece):
    e = min(s + piece, nbytes // 4)
    parts.append((mx.arange(s, e, dtype=mx.uint32) * mx.array(2654435761, dtype=mx.uint32)) ^ mx.array(0x9E3779B9, dtype=mx.uint32))
    mx.eval(parts[-1])
buf = mx.concatenate(parts)
del parts
mx.eval(buf)
mx.clear_cache()
print(f"buffer {nbytes / 1e9:.2f} GB ({nbytes / 2 ** 30:.2f} GiB), {a.threads} threads x {a.tg}/threadgroup, {a.warmup} warm-up + {a.runs} timed runs each", flush=True)


def run_read():
    return read_k(inputs=[buf], template=[("N4", n4)], grid=(a.threads, 1, 1), threadgroup=(a.tg, 1, 1), output_shapes=[(a.threads,)], output_dtypes=[mx.uint32])[0]


def run_copy():
    return copy_k(inputs=[buf], template=[("N4", n4)], grid=(a.threads, 1, 1), threadgroup=(a.tg, 1, 1), output_shapes=[buf.shape], output_dtypes=[mx.uint32])[0]


def timed(fn):
    for _ in range(a.warmup):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(a.runs):
        mx.synchronize()
        t = time.perf_counter()
        o = fn()
        mx.eval(o)
        mx.synchronize()
        ts.append(time.perf_counter() - t)
        del o
    return ts


rep = dict(device=dev.get("device_name"), buffer_gb=nbytes / 1e9, runs=a.runs)
tr = timed(run_read)
rep["read_gbs"] = dict(best=nbytes / 1e9 / min(tr), median=nbytes / 1e9 / statistics.median(tr), worst=nbytes / 1e9 / max(tr))
chk = run_copy()                                                   # the copy must be a copy
mx.eval(chk)
assert bool(mx.array_equal(chk[:1 << 20], buf[:1 << 20]).item()) and bool(mx.array_equal(chk[-(1 << 20):], buf[-(1 << 20):]).item()), "copy kernel wrote a wrong buffer"
del chk
tc = timed(run_copy)
rep["copy_gbs_rw"] = dict(best=2 * nbytes / 1e9 / min(tc), median=2 * nbytes / 1e9 / statistics.median(tc), worst=2 * nbytes / 1e9 / max(tc))
rep["copy_gbs_per_direction"] = dict(best=nbytes / 1e9 / min(tc), median=nbytes / 1e9 / statistics.median(tc), worst=nbytes / 1e9 / max(tc))
rep["peak_gib"] = mx.get_peak_memory() / 2 ** 30
r, c, d = rep["read_gbs"], rep["copy_gbs_rw"], rep["copy_gbs_per_direction"]
print(f"read  GB/s: best {r['best']:.1f}  median {r['median']:.1f}  worst {r['worst']:.1f}   (min time {min(tr) * 1e3:.2f} ms, median {statistics.median(tr) * 1e3:.2f} ms)")
print(f"copy  GB/s (read + write traffic): best {c['best']:.1f}  median {c['median']:.1f}  worst {c['worst']:.1f}   per direction: best {d['best']:.1f}  median {d['median']:.1f}   "
      f"(min time {min(tc) * 1e3:.2f} ms)")
print(f"peak memory {rep['peak_gib']:.2f} GiB")
if a.json:
    json.dump(rep, open(a.json, "w"), indent=1)
