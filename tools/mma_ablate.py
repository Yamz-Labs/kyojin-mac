"""Ablations of the EXL3 MMA GEMV kernel (partial only, us, chained): which part of the 165 us floor is decode, B loads, MMA, reduction. usage: mma_ablate.py [--shape mlp_up] [--rows 5]"""
import argparse, numpy as np, mlx.core as mx
from kyojin_mlx import kernels as KN, mma_gemv as MM
from mma_bench import synth, SHAPES, timeit
ap = argparse.ArgumentParser(); ap.add_argument("--shape", default="mlp_up"); ap.add_argument("--rows", type=int, default=5); a = ap.parse_args()
kin, nout, K = SHAPES[a.shape]; lin = synth(kin, nout, K); mx.eval(lin._words); g = lin._g[1]; R = a.rows
base = MM.mma_gemv_source(K, g.cb)
half = base.replace("simdgroup_matrix<float, 8, 8> B[2][RB]", "simdgroup_matrix<half, 8, 8> B[2][RB]").replace("B[kh][r].thread_elements()[0] = xk[r0 * KD + 8 * kh];", "B[kh][r].thread_elements()[0] = half(xk[r0 * KD + 8 * kh]);").replace("B[kh][r].thread_elements()[1] = xk[r1 * KD + 8 * kh];", "B[kh][r].thread_elements()[1] = half(xk[r1 * KD + 8 * kh]);").replace("simdgroup_matrix<float, 8, 8> A;", "simdgroup_matrix<half, 8, 8> A;").replace("A.thread_elements()[0] = float(d0); A.thread_elements()[1] = float(d1);", "A.thread_elements()[0] = d0; A.thread_elements()[1] = d1;")
V = {"full": base, "half": half,
     "no_decode": base.replace("A.thread_elements()[0] = float(d0); A.thread_elements()[1] = float(d1);", "A.thread_elements()[0] = 1.0f; A.thread_elements()[1] = 0.5f;"),
     "no_bload": base.replace("B[kh][r].thread_elements()[0] = xk[r0 * KD + 8 * kh];", "B[kh][r].thread_elements()[0] = 1.0f;").replace("B[kh][r].thread_elements()[1] = xk[r1 * KD + 8 * kh];", "B[kh][r].thread_elements()[1] = 0.5f;"),
     "no_mma": base.replace("simdgroup_multiply_accumulate(acc[t][0][r], A, B[0][r], acc[t][0][r]);", "acc[t][0][r].thread_elements()[0] += A.thread_elements()[0];")
                   .replace("simdgroup_multiply_accumulate(acc[t][1][r], A, B[0][r], acc[t][1][r]);", "acc[t][1][r].thread_elements()[0] += A.thread_elements()[0];")
                   .replace("simdgroup_multiply_accumulate(acc[t][0][r], A, B[1][r], acc[t][0][r]);", "acc[t][0][r].thread_elements()[0] += A.thread_elements()[0];")
                   .replace("simdgroup_multiply_accumulate(acc[t][1][r], A, B[1][r], acc[t][1][r]);", "acc[t][1][r].thread_elements()[0] += A.thread_elements()[0];"),
     "no_reduce": base.replace("for (uint j = 0; j < WS; ++j) v += red[j][cc][rr];", "v = red[0][cc][rr];"),
     "no_stream_ld": base.replace("const uint u{0}".format(0), "const uint u0"),
     }
rng = np.random.default_rng(5)
xs = [mx.array(rng.standard_normal((R, kin)).astype(np.float16)) for _ in range(12)]; mx.eval(*xs)
xh = [KN.rot(x, g.suh) for x in xs]; mx.eval(*xh)
def run(name, src):
    kern = KN._kern(("mmaabl", name, g.K, g.cb), name=f"abl_{name}", input_names=["xh", "trellis", "params"], output_names=["part"], source=src)
    f = lambda h: kern(inputs=[h, g.words, g.params], template=[("BITS", g.K), ("MR", R)], grid=(g.blocks * 256, g.tgk, 1), threadgroup=(256, 1, 1), output_shapes=[(g.tgk, R, g.n)], output_dtypes=[mx.float32])[0]
    return timeit(f, xh, 7, 12)[1]
import statistics
T = {n: [] for n in V}
for rep in range(8):
    for n, s_ in V.items(): T[n].append(run(n, s_))
print(" ".join(f"{n} min {min(t):.0f} med {statistics.median(t):.0f}" for n, t in T.items()), flush=True)
