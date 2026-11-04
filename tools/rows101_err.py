"""Error of the few-row q4 kernels vs an fp32 reference (dequantized weights in fp32, fp32 activations) at 1..32 rows: MLX qmm, q4mma (8x8
simdgroup MMA, up to 16 rows), q4nax. Activations: real-like heavy tail (normal x lognormal channel scale), bf16. rel = |y - ref| / max|ref|.
usage: rows101_err.py NxK[,..] [ROWS=1,8,9,16,32]"""
import sys
import mlx.core as mx
from kyojin_mlx import q4mma as Q

shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1].split(",")]
Rs = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,8,9,16,32").split(",")]
mx.random.seed(0)
for N, K in shapes:
    w, s, b = mx.quantize(mx.random.normal((N, K)) * 0.02, group_size=64, bits=4)
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    wf = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=4)
    chan = mx.exp(mx.random.normal((K,)) * 1.0)
    for R in Rs:
        x = (mx.random.normal((R, K)) * chan).astype(mx.bfloat16)
        ref = x.astype(mx.float32) @ wf.T
        out = {"mlx": mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)}
        if R <= 16: out["q3"] = Q.q4mma3(Q.prep_x(x), w, s, b, R, Q.NT, Q.KS)
        out["nax"] = Q.q4nax(Q.prep_xn(x, 2), w, s, b, R, 2, 2, 1)
        m = mx.abs(ref).max()
        print(f"{N}x{K} R{R:2d} " + "  ".join(f"{k} max {(mx.abs(v.astype(mx.float32) - ref).max() / m).item():.2e} mean {(mx.abs(v.astype(mx.float32) - ref).mean() / m).item():.2e}" for k, v in out.items()), flush=True)
