"""Dump the outputs of a few real EXL3 linears at 1..4 rows (fixed inputs) so two processes (KYOJIN_HALF_ROWS off/on) can be compared.
usage: half85.py MODEL OUT.npz      then:  half85.py --cmp A.npz B.npz"""
import sys, numpy as np
if sys.argv[1] == "--cmp":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for k in a.files:
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        print(f"{k:22s} max|d|/rms {np.abs(x - y).max() / np.sqrt((x ** 2).mean()):.2e}   rms(d)/rms {np.sqrt(((x - y) ** 2).mean() / (x ** 2).mean()):.2e}")
    sys.exit(0)
import mlx.core as mx
from kyojin_mlx.exl3_linear import load_exl3
model, _ = load_exl3(sys.argv[1]); lm = model.language_model; Ls = lm.model.layers
mods = {"mlp.gate.0": Ls[0].mlp.gate_proj, "mlp.up.9": Ls[9].mlp.up_proj, "mlp.down.20": Ls[20].mlp.down_proj, "gdn.qkv.1": Ls[1].linear_attn.in_proj_qkv,
        "gdn.out.2": Ls[2].linear_attn.out_proj, "attn.q.3": Ls[3].self_attn.q_proj, "attn.o.7": Ls[7].self_attn.o_proj}
out = {}
rng = np.random.RandomState(0)
for n, m in mods.items():
    for R in (1, 2, 3, 4):
        x = mx.array(rng.randn(R, 1, m.in_features).astype(np.float16))
        y = m(x); mx.eval(y); out[f"{n}.R{R}"] = np.array(y.astype(mx.float32))
np.savez(sys.argv[2], **out); print("saved", len(out))
