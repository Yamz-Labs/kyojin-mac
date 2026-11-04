"""Gate: GGUF K / IQ blocks read as stored (foreign_metal.gemvk_source) against the same matrix as resident fp16 and as dequant kernel + matmul, per row count.
Synthetic 8 x 8192^2 matrices of random valid blocks, bounded, GPU only, no model. Args: type (Q5_K | Q6_K | IQ4_XS), then row counts. Run under /tmp/lane_ab.lock."""
import os, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mlx.core as mx
from kyojin_mlx import foreign_linear as FL, foreign_gguf as GF, foreign_pack as FP, foreign_synth as SY

KIN, N, COUNT, REPS = 8192, 8192, 8, 7
T = sys.argv[1] if len(sys.argv) > 1 else "Q5_K"
ROWS = tuple(int(a) for a in sys.argv[2:]) or (1, 2, 4, 8)
NBRS = tuple(int(a) for a in os.environ.get("NBRS", "2").split(","))
rng = np.random.default_rng(1)
raw0 = SY.random_blocks(T, N, KIN, rng)
_, be, bb = GF.TYPES_BY_NAME[T]
b3 = raw0.reshape(N, KIN // be, bb)
for off in SY._FP16_AT[T]:                                # small scales: row sums stay far from the fp16 range, so the error column measures the kernel
    b3[..., off:off + 2] = rng.uniform(1e-4, 5e-4, (N, KIN // be)).astype("<f2").view(np.uint8).reshape(N, KIN // be, 2)
p0 = GF.to_pack(T, raw0, KIN)
d0 = FL.DevPack(p0)
W16 = d0.dequant(); mx.eval(W16)
devs = [d0]
for m in range(1, COUNT):                                 # different bytes per matrix (rolled rows), as many matrices as the cache cannot hold
    p = GF.to_pack(T, np.ascontiguousarray(np.roll(raw0, m * 37, axis=0)), KIN)
    devs.append(FL.DevPack(p))
W16s = [W16] + [d.dequant() for d in devs[1:]]
mx.eval([a for d in devs for a in d.arrays()], W16s)
print(T, "resident GiB stored", round(sum(d.nbytes() for d in devs) / 2 ** 30, 2), "fp16", round(COUNT * N * KIN * 2 / 2 ** 30, 2), flush=True)
ref16 = np.array(W16, np.float32)
for R in ROWS:
    x = mx.array(rng.normal(0, 1, (R, KIN)).astype(np.float16))
    ref = mx.array(np.array(x, np.float32) @ ref16.T)
    mx.eval(x, ref)
    arms = {"fp16": lambda m: (x @ W16s[m].T).astype(mx.float16), "deq_mm": lambda m: (x @ devs[m].dequant().T).astype(mx.float16)}
    for nbr in NBRS:
        arms[f"gemv_n{nbr}"] = (lambda nbr: lambda m: devs[m]._gemvk(x, nbr))(nbr)
    for a, f in arms.items():
        if a == "fp16":
            continue
        e = f(0); mx.eval(e)
        print(f"R{R} {a} rel err {float(mx.max(mx.abs(e.astype(mx.float32) - ref))) / float(mx.max(mx.abs(ref))):.5f}", flush=True)
    res = {a: [] for a in arms}
    for rep in range(REPS):
        for a, f in arms.items():
            t = time.perf_counter()
            mx.eval([f(m) for m in range(COUNT)])
            res[a].append(time.perf_counter() - t)
    w = COUNT * N * KIN * R
    by = sum(d.nbytes() for d in devs)
    print(f"R{R} " + " | ".join(f"{a} {w / st.median(v[1:]) / 1e9:.0f} Gw/s" + (f" ({by / st.median(v[1:]) / 1e9:.0f} GB/s)" if a.startswith("gemv") else "") for a, v in res.items()), flush=True)
