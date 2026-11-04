"""ablate layer seams at one chunk. usage: pf3_abl.py EXL3 MLX T C VARIANT  (VARIANT: full | s2stock | s1stock | none)"""
import sys, os
import mlx.core as mx
from kyojin_mlx import layer_pf as LP, kernels as KN
var = sys.argv[5]
def s2stock(inner, svh, res):
    rows, N = inner.shape
    fp = mx.array([1, N, rows], dtype=mx.uint32); grid, tg = KN.S.had_grid(N // 128, rows)
    y = KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner, svh, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg, output_shapes=[(rows, N)], output_dtypes=[mx.float16])[0]
    return res + y
def s1stock(inner, svh, x, nw, eps, sg, su, pg, pu):
    rows, K = inner.shape
    fp = mx.array([1, K, rows], dtype=mx.uint32); grid, tg = KN.S.had_grid(K // 128, rows)
    y = KN.fin_kernel(mx.float16, mx.float16)(inputs=[inner, svh, fp], template=[("OT", mx.float16)], grid=grid, threadgroup=tg, output_shapes=[(rows, K)], output_dtypes=[mx.float16])[0]
    h = x + y
    n = mx.fast.rms_norm(h, nw, eps)
    return h, KN.rot(n, sg, mx.float16, perm=pg), KN.rot(n, su, mx.float16, perm=pu)
if var == "s2stock": LP.seam2 = s2stock
if var == "s1stock": LP.seam1 = s1stock
if var == "none": LP.MAX_ROWS = 0
sys.argv = ["pf2k_ab.py"] + sys.argv[1:5] + ["5"]
exec(open(os.path.expanduser("~/kmlx-macpf3/tools/pf2k_ab.py")).read())
