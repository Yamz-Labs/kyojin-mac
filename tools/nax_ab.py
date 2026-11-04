"""NAX capability probe + NAX on/off A/B of the dense EXL3 linear (Exl3Linear), interleaved, in one process (the switch is metal_nax.MODE).
   python tools/nax_ab.py [--rows 8,64,512,1024] [--shapes 8192x5120x3,5120x8192x3] [--rounds 5]
Prints the probe line (arch, matmul2d compile/run result), then per shape and row count the min / p25 ms of both arms (KYOJIN_NAX=0 vs auto) and the
max-rel difference of the outputs (fp16 rounding class). Day 1 on the Ultra: NAX must win at min AND p25 for the rows it serves, else KYOJIN_NAX=0."""
import argparse
import os
import time

import numpy as np
import mlx.core as mx

from kyojin_mlx import exl3_linear as EL, metal_nax as NX

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="8,64,512,1024")
ap.add_argument("--shapes", default="8192x5120x3,5120x8192x3")
ap.add_argument("--rounds", type=int, default=5)
a = ap.parse_args()
base = NX.MODE


def arm(on: bool):
    NX.MODE = base if on else "0"
    NX._status = None
    NX._forced = None


arm(True)
st = NX.nax_status()
print("probe:", st)
os.environ["KYOJIN_NAX_LOG"] = "0"                    # one probe line, not one per arm switch
if not st["on"]:
    print("NAX unavailable here: both arms run the non-NAX kernels")
rng = np.random.default_rng(3)
for sh in a.shapes.split(","):
    kin, nout, K = (int(v) for v in sh.split("x"))
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    lin = EL.Exl3Linear(mx.array(tr), mx.array(suh), mx.array(svh), cb="mul1")
    for rows in (int(v) for v in a.rows.split(",")):
        x = mx.array((rng.standard_normal((rows, kin)) * 2).astype(np.float16))
        ts, ys = {True: [], False: []}, {}
        for on in (True, False):                                        # warm both arms (kernel compile, weight caches)
            arm(on); ys[on] = np.array(lin(x)).astype(np.float32)
        for _ in range(a.rounds):
            for on in (True, False):
                arm(on)
                t0 = time.perf_counter(); y = lin(x); mx.eval(y); ts[on].append((time.perf_counter() - t0) * 1e3)
        rel = np.abs(ys[True] - ys[False]).max() / (np.abs(ys[False]).max() + 1e-9)
        f = lambda v: f"min {min(v):7.3f} p25 {np.percentile(v, 25):7.3f}"
        print(f"{sh} rows {rows:5d}  NAX(auto) {f(ts[True])} ms | NAX off {f(ts[False])} ms | x{min(ts[False]) / min(ts[True]):.2f}  rel {rel:.1e}")
