"""Tile sweep of the fused dense linear at fixed rows: python tools/sweep_cfg.py rows classes cfgs"""
import sys, json, os, time
import mlx.core as mx
from kyojin_mlx import exl3_linear as EL
rows = int(sys.argv[1]); classes = sys.argv[2].split(","); cfgs = [tuple(map(int, c.split(","))) for c in sys.argv[3].split(";")]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 7
pack = os.path.expanduser("~/rivals/models/Qwen3.5-4B-exl3-4.0")
L = "model.language_model.layers."
CL = {"qkv": L + "9.linear_attn.in_proj_qkv", "z": L + "9.linear_attn.in_proj_z", "o": L + "8.linear_attn.out_proj", "v": L + "7.self_attn.v_proj",
      "gate": L + "9.mlp.gate_proj", "up": L + "9.mlp.up_proj", "down": L + "9.mlp.down_proj"}
w = mx.load(f"{pack}/model.safetensors")
def tm(fn):
    mx.eval(fn()); mx.synchronize(); t0 = time.perf_counter(); mx.eval([fn() for _ in range(8)]); return (time.perf_counter() - t0) / 8 * 1e3
for c in classes:
    p = CL[c]; lin = EL.Exl3Linear(w[p + ".trellis"], w[p + ".suh"], w[p + ".svh"], cb="mul1", tune={})
    x = (mx.random.normal((rows, lin.in_features)) * 0.7).astype(mx.float16); mx.eval(x)
    ref = None; best = {}
    ok = []
    for cfg in cfgs:
        lin.tune = {"nax_cfg": {str(rows): list(cfg)}}; lin._nax_rows = {}
        try:
            y = lin(x); mx.eval(y)
        except Exception as e:
            print(c, cfg, "FAIL", str(e).splitlines()[0][:80]); continue
        if ref is None: ref = y
        d = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        ok.append((cfg, d))
    for r in range(reps):
        for cfg, d in ok:
            lin.tune = {"nax_cfg": {str(rows): list(cfg)}}; lin._nax_rows = {}
            v = tm(lambda: lin(x)); best[cfg] = min(best.get(cfg, 9e9), v)
    fl = 2 * rows * lin.in_features * lin.out_features / 1e9
    base = EL.NX.pick_cfg([rows])
    print(f"{c} {lin.in_features}x{lin.out_features} rows {rows} default {base}")
    for cfg, d in sorted(ok, key=lambda t: best[t[0]]):
        print(f"   {str(cfg):18s} {best[cfg]:7.3f} ms {fl/best[cfg]:5.1f} TF  maxdiff vs first {d:.2g}")
