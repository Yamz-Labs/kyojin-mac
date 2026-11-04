"""Capture the real MoE inputs of decode steps: x (post-norm hidden), router ids and weights per layer, for several independent users.
usage: mu_route.py PACK OUT.npz [--users 8] [--steps 40] [--lens ...] [--seed 0]
Prompts are structured token streams (repeated phrases mapped into the pack's vocabulary), each user a different one, then greedy solo decode.
Arrays: L{layer}_x [users, steps, H] f16, L{layer}_ids [users, steps, topk] i32, L{layer}_w [users, steps, topk] f32."""
import argparse
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R, moe as M

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("out"); ap.add_argument("--users", type=int, default=8); ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--len", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack, {})
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
vocab = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(a.seed)
layer_of, rec = {}, {}
orig = M.MoeGather.__call__


def hook(self, x, ids, wts, **kw):
    if x.shape[0] == 1:
        li = layer_of.setdefault(id(self), len(layer_of))
        xr = x
        if kw.get("fq8"):                                                         # DS4.1: the call carries the input of the fp8 fake quant, which the kernels fold in
            from mlx_vlm.models.deepseek_v41.language import fake_quant_fp8_ue8m0
            xr = fake_quant_fp8_ue8m0(x)
        mx.eval(xr, ids, wts)
        rec.setdefault(li, []).append((np.array(xr.astype(mx.float16))[0], np.array(ids)[0], np.array(wts)[0]))
    return orig(self, x, ids, wts, **kw)


M.MoeGather.__call__ = hook
out = {}
for u in range(a.users):
    base = rng.integers(16, vocab - 16, int(rng.integers(6, 14)))             # phrase repeated with noise: text-like locality
    prompt = np.concatenate([np.where(rng.random(len(base)) < 0.8, base, rng.integers(16, vocab - 16, len(base))) for _ in range(a.len // len(base) + 1)])[:a.len].tolist()
    cache, lg = R.prefill(model, prompt)
    rec.clear(); layer_of.clear() if False else None
    c = R.batch([cache]); tok = int(mx.argmax(lg, axis=-1).item())
    for s in range(a.steps):
        l = R.step(model, c, mx.array([tok])); mx.eval(l); tok = int(mx.argmax(l[0]).item())
    for li, v in rec.items():
        v = v[-a.steps:]
        for nm, k in (("x", 0), ("ids", 1), ("w", 2)):
            out.setdefault(f"L{li}_{nm}", []).append(np.stack([t[k] for t in v]))
    rec.clear()
    print("user", u, "peak GiB", round(mx.get_peak_memory() / 2**30, 2), flush=True)
np.savez(a.out, **{k: np.stack(v) for k, v in out.items()})
print({k: np.stack(v).shape for k, v in out.items()})
