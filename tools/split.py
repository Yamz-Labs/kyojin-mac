import sys, os, time, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, kernels as KN, metal_nax as NX
rows = int(sys.argv[1]); classes = sys.argv[2].split(",")
pack = os.path.expanduser("~/rivals/models/Qwen3.5-4B-exl3-4.0"); L = "model.language_model.layers."
CL = {"qkv": L + "9.linear_attn.in_proj_qkv", "o": L + "8.linear_attn.out_proj", "gate": L + "9.mlp.gate_proj", "down": L + "9.mlp.down_proj"}
w = mx.load(f"{pack}/model.safetensors")
def tm(fn):
    mx.eval(fn()); mx.synchronize(); t0 = time.perf_counter(); mx.eval([fn() for _ in range(8)]); return (time.perf_counter() - t0) / 8 * 1e3
for c in classes:
    p = CL[c]; lin = EL.Exl3Linear(w[p + ".trellis"], w[p + ".suh"], w[p + ".svh"], cb="mul1", tune={})
    x = (mx.random.normal((rows, lin.in_features)) * 0.7).astype(mx.float16); mx.eval(x)
    lin(x); mx.eval(lin(x))
    st, eids = lin._nax; ent = lin._nax_rows[rows]; xh = KN.rot(x, lin._suh, mx.float16, perm=True); mx.eval(xh)
    def g(mode, fin=True): return lambda: NX.expert_gemm((st,), eids, ent[0], xh, rows, rows, ent[1], mode=mode, fin=(lin._svh, mx.float16) if fin else None)
    arms = {"total": lambda: lin(x), "rot": lambda: KN.rot(x, lin._suh, mx.float16, perm=True), "gemm_full": g("full"), "gemm_nodec": g("nodec"), "gemm_nomm": g("nomm")}
    best = {k: 9e9 for k in arms}
    for r in range(7):
        for k, f in arms.items(): best[k] = min(best[k], tm(f))
    print(c, lin.in_features, lin.out_features, rows, "  ".join(f"{k} {v:.3f}" for k, v in best.items()), flush=True)
