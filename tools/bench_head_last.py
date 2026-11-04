"""Prefill of a patched slice in chunks, LM head on every row (arm OFF) vs on the last row only (arm ON, headlast.last_row), interleaved, min / p25 + peak memory.
usage: bench_head_last.py PACK [chunk=4096] [n_chunks=4] [rounds=6]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, spec as S, headlast as HL
pack = sys.argv[1]; chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
nch = int(sys.argv[3]) if len(sys.argv) > 3 else 4; rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 6
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
ids = np.random.default_rng(3).integers(1, 4000, chunk * nch).tolist()
NAME = {True: "last-row", False: "all-rows"}
def run(on):
    HL.ENABLED = on; mx.synchronize(); t0 = time.perf_counter()
    S.prefill(model, ids, (), chunk); mx.synchronize(); return time.perf_counter() - t0
def peak(on):
    mx.reset_peak_memory(); run(on); return mx.get_peak_memory() / 2**30
for a in (True, False): run(a)                                  # warm-up
res = {True: [], False: []}
for r in range(rounds):
    for a in ((True, False) if r % 2 == 0 else (False, True)): res[a].append(run(a))
for a in (False, True):
    v = sorted(res[a]); print(f"{NAME[a]} min {v[0]*1e3:8.1f} ms p25 {v[len(v)//4]*1e3:8.1f} ms  {chunk*nch/v[0]:7.0f} tok/s", flush=True)
print(f"gain min {1-min(res[True])/min(res[False]):+.1%} p25 {1-sorted(res[True])[rounds//4]/sorted(res[False])[rounds//4]:+.1%}")
for a in (False, True): print(f"peak {NAME[a]} {peak(a):.2f} GiB")
