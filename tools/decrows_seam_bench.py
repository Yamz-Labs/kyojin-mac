"""decrows_seam_bench.py PACK: per-seam microbench, stock vs fused, rows 1..8 (isolated, N independent copies in one eval; GEMVs rotate over all layers so weights are read cold).
Prints us per call per layer and the saving."""
import os as _os; _os.environ.setdefault("KYOJIN_DECROWS", "1")
import os, sys, time, json
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in ("..", ".")]
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import decfuse as DF, decrows as DR

be, _ = load_backend(sys.argv[1], mtp=False, log=lambda *a: None)
tm = be.lm.model
L = tm.layers
dt = mx.bfloat16
H = L[0].input_layernorm.weight.shape[0]
mx.random.seed(0)


def timeit(fn, reps=5, inner=3):
    """graph built outside the timer; the timed part is mx.eval (encode + GPU), as in a served step where the graph is replayed / built ahead"""
    mx.eval(fn())
    best = 1e9
    for _ in range(reps):
        for _ in range(inner):
            g = fn()
            t0 = time.perf_counter()
            mx.eval(g)
            best = min(best, time.perf_counter() - t0)
    return best


def per_layer_us(fn_list):
    """fn_list: one callable per layer returning arrays; one eval over all layers"""
    t = timeit(lambda: [f() for f in fn_list])
    return t / len(fn_list) * 1e6


attn = [l for l in L if not l.is_linear]
gdn = [l for l in L if l.is_linear]
res = {}
for R in range(1, 9):
    x = mx.random.normal((1, R, H)).astype(dt); mx.eval(x)
    r = {}
    r["qkv sep"] = per_layer_us([(lambda a=a: [a.self_attn.q_proj(x), a.self_attn.k_proj(x), a.self_attn.v_proj(x)]) for a in attn])
    r["qkv fused"] = per_layer_us([(lambda a=a: a.self_attn._kr_fused(x)) for a in attn])
    r["gate_up sep"] = per_layer_us([(lambda l=l: swiglu(l.mlp.gate_proj(x), l.mlp.up_proj(x))) for l in L])
    r["gate_up fused"] = per_layer_us([(lambda l=l: (lambda g, u: swiglu(g, u))(*l.mlp._kr_fused.parts(x))) for l in L])
    r["gdn_proj sep"] = per_layer_us([(lambda l=l: [l.linear_attn.in_proj_qkv(x), l.linear_attn.in_proj_z(x), l.linear_attn.in_proj_b(x), l.linear_attn.in_proj_a(x)]) for l in gdn])
    r["gdn_proj fused"] = per_layer_us([(lambda l=l: l.linear_attn._kr_fused.parts(x)) for l in gdn])
    res[R] = r
    # elementwise seams: 200 independent copies
    xs = [mx.random.normal((1, R, H)).astype(dt) for _ in range(4)]; mx.eval(xs)
    nrm = L[0].post_attention_layernorm
    N = 200
    r["add+norm sep"] = timeit(lambda: [nrm(xs[i % 4] + xs[(i + 1) % 4]) for i in range(N)]) / N * 1e6
    r["add+norm fused"] = timeit(lambda: [DF.add_rms(xs[i % 4], xs[(i + 1) % 4], nrm)[1] for i in range(N)]) / N * 1e6
    a0 = attn[0].self_attn
    y = DF.mx.random.normal((1, R, a0._kr_fused.weight.shape[0])).astype(dt); mx.eval(y)
    B_, L_ = 1, R
    HQ, HK, D = a0.num_attention_heads, a0.num_key_value_heads, a0.head_dim
    def qk_stock():
        qo = y[..., :HQ * 2 * D].reshape(B_, L_, HQ, 2 * D)
        q, gate = mx.split(qo, 2, axis=-1); gate = gate.reshape(B_, L_, -1)
        k = y[..., HQ * 2 * D:HQ * 2 * D + HK * D]; v = y[..., HQ * 2 * D + HK * D:]
        q = a0.q_norm(q).transpose(0, 2, 1, 3)
        k = a0.k_norm(k.reshape(B_, L_, HK, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B_, L_, HK, -1).transpose(0, 2, 1, 3)
        q = a0.rope(q, offset=1000); k = a0.rope(k, offset=1000)
        return [q, k, mx.contiguous(v), mx.contiguous(gate)]
    qk_fused = lambda: DR.qk_prep(y, a0.q_norm.weight, a0.k_norm.weight, a0.q_norm.eps, HQ, HK, D, a0.rope.dims, a0.rope.base, a0.rope.scale, 1000, B_, L_)
    r["qk front sep"] = timeit(lambda: [qk_stock() for _ in range(N // 4)]) / (N // 4) * 1e6
    r["qk front fused"] = timeit(lambda: [qk_fused() for _ in range(N // 4)]) / (N // 4) * 1e6
print("DECROWS_SEAM " + json.dumps(res))
names = ["qkv", "gate_up", "gdn_proj", "add+norm", "qk front"]
print("| seam (us per call per layer) | " + " | ".join(f"{R}r" for R in range(1, 9)) + " |\n|---|" + "---|" * 8)
for n in names:
    for v in ("sep", "fused"):
        print(f"| {n} {v} | " + " | ".join(f"{res[R][n + ' ' + v]:.1f}" for R in range(1, 9)) + " |")
    print(f"| **{n} saved** | " + " | ".join(f"{res[R][n + ' sep'] - res[R][n + ' fused']:+.1f}" for R in range(1, 9)) + " |")
