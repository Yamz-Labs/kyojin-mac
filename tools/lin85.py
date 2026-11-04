"""Per-linear cost vs rows (1..4) on the real model, every layer's instance, one evaluated graph per (linear kind, R) (independent calls), us per call, interleaved.
usage: lin85.py KIND(mlx|exl3) MODEL [PASSES]"""
import sys, time, statistics
import mlx.core as mx
kind, mdir = sys.argv[1], sys.argv[2]; PASS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
if kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(mdir)
else:
    from mlx_lm import load
    model, _ = load(mdir)
lm = model.language_model; Ls = lm.model.layers
H = lm.args.hidden_size
groups = {
    "gdn.qkv": [l.linear_attn.in_proj_qkv for l in Ls if l.is_linear], "gdn.z": [l.linear_attn.in_proj_z for l in Ls if l.is_linear],
    "gdn.b": [l.linear_attn.in_proj_b for l in Ls if l.is_linear], "gdn.a": [l.linear_attn.in_proj_a for l in Ls if l.is_linear],
    "gdn.out": [l.linear_attn.out_proj for l in Ls if l.is_linear],
    "attn.q": [l.self_attn.q_proj for l in Ls if not l.is_linear], "attn.k": [l.self_attn.k_proj for l in Ls if not l.is_linear],
    "attn.v": [l.self_attn.v_proj for l in Ls if not l.is_linear], "attn.o": [l.self_attn.o_proj for l in Ls if not l.is_linear],
    "mlp.gate": [l.mlp.gate_proj for l in Ls], "mlp.up": [l.mlp.up_proj for l in Ls], "mlp.down": [l.mlp.down_proj for l in Ls],
}
Rs = [1, 2, 3, 4]
def kin(m):
    return getattr(m, "in_features", None) or (m.weight.shape[1] * (32 // m.bits) if hasattr(m, "bits") else m.weight.shape[1])
X = {}
def xin(m, R):
    k = kin(m)
    if (k, R) not in X: X[(k, R)] = mx.random.normal((R, 1, k)).astype(mx.float16 if kind == "exl3" else mx.bfloat16); mx.eval(X[(k, R)])
    return X[(k, R)]
import os
HALF = int(os.environ.get("AB_HALF", "0"))            # >0: interleave metal_src.HALF_ROWS = 0 / AB_HALF inside this process
modes = [0, HALF] if HALF else [None]
if HALF:
    from kyojin_mlx import metal_src as S
def setmode(md):
    if md is not None: S.HALF_ROWS = md
def mk(ms, R, md):
    def f():
        setmode(md); return [m(xin(m, R)) for m in ms]
    return f
fns = {(g, R, md): mk(ms, R, md) for g, ms in groups.items() for R in Rs for md in modes}
res = {k: [] for k in fns}
for k, f in fns.items(): mx.eval(f()); mx.eval(f())
for p in range(PASS):
    for k, f in fns.items():
        t0 = time.perf_counter(); mx.eval(f()); res[k].append((time.perf_counter() - t0) / len(groups[k[0]]) * 1e6)
for md in modes:
    tot = {R: 0.0 for R in Rs}
    print("--- HALF_ROWS =", md)
    for g, ms in groups.items():
        row = [statistics.median(res[(g, R, md)]) for R in Rs]
        print(f"{g:9s} x{len(ms):2d}  us/call  R1 {row[0]:6.0f}  R2 {row[1]:6.0f}  R3 {row[2]:6.0f}  R4 {row[3]:6.0f}   (R4-R1)*count = {(row[3]-row[0])*len(ms)/1e3:5.2f} ms")
        for R, v in zip(Rs, row): tot[R] += v * len(ms) / 1e3
    print("sum over all layer linears (ms): " + "  ".join(f"R{R} {tot[R]:.2f}" for R in Rs))
