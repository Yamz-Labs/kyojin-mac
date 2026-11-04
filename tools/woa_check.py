"""Attention output -> wo_a, patched model vs the checkpoint's model.py arithmetic (fp8 act_quant of the attention output before the wo_a einsum).
  python tools/woa_check.py dump PACK OUT.npz [--tokens 40] [--no-fq]   (Mac, mlx) real slice, real wo_a: records input / output of every wo_a call
                                                                          (prefill of n-1 tokens, then one decode step) and the dense fp32 wo_a weights
  python tools/woa_check.py ref  IN.npz [--json F]                      (any machine, torch CPU) recomputes the output with the reference recipe
                                                                          (ds41-ref/inference/model.py Attention.forward: act_quant(o, 32, ue8m0, inplace) then einsum with wo_a)
The reference recipe in torch: per 32-block amax (floored at 1e-4), scale = 2^ceil(log2(amax / 448)) via the IEEE bits, clamp +-448, cast to float8_e4m3fn
(round to nearest even), multiply back. Verdict: the patched output must match the quantized recipe at fp16 noise and be clearly farther from the
recipe WITHOUT the quantization (negative control: the unquantized einsum)."""
import argparse, json, sys

import numpy as np


def dump(a):
    import mlx.core as mx
    from kyojin_mlx import patch as P, ref_numpy as RN
    model, spec, cfg = P.build_model(a.pack)
    t = P.load_pack(a.pack)
    if a.no_fq:
        import dataclasses
        spec = dataclasses.replace(spec, fq_grouped_in=False)
    P.patch_model(model, a.pack, spec, cfg)
    lm = model.language_model
    store = []
    for li, layer in enumerate(lm.layers):
        mod = layer.attn.wo_a
        cls = type(mod)

        def call(self, x, *args, _cls=cls, _li=li, **k):
            y = _cls.__call__(self, x, *args, **k)
            store.append((_li, x, y))
            return y
        mod.__class__ = type(cls.__name__, (cls,), {"__call__": call, "__module__": cls.__module__})
    ids = np.random.default_rng(1).integers(16, cfg["text_config"]["vocab_size"] - 16, a.tokens).tolist()
    cache = lm.make_cache()
    out = {}
    for step, sl in (("prefill", slice(0, len(ids) - 1)), ("decode", slice(len(ids) - 1, len(ids)))):
        store.clear()
        lg = lm(mx.array([ids[sl]]), cache=cache).logits
        mx.eval(lg, *[v for _, x, y in store for v in (x, y)])
        for li, x, y in store:
            out[f"{step}.L{li}.x"] = np.array(x.astype(mx.float32))
            out[f"{step}.L{li}.y"] = np.array(y.astype(mx.float32))
    for key in sorted(t):
        if key.endswith("wo_a.slices.trellis"):
            p = key[:-len(".trellis")]
            li = int(p.split("layers.")[1].split(".")[0])
            cb = P._cb(t, p, "mul1")
            tr, suh, svh = np.array(t[p + ".trellis"]), np.array(t[p + ".suh"]), np.array(t[p + ".svh"])
            out[f"W.L{li}"] = np.stack([RN.reconstruct(tr[g], suh[g], svh[g], cb) for g in range(tr.shape[0])])      # [G, in, out]
    np.savez(a.out, **out)
    print("dumped", len(out), "arrays ->", a.out)


def act_quant_ref(o):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(o)).float()
    n = x.shape[-1]
    xb = x.reshape(*x.shape[:-1], n // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    s = ((e + 127) << 23).view(torch.float32)
    q = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return q.reshape(x.shape).numpy()


def ref(a):
    import torch
    d = np.load(a.inp)
    res = {}
    print(f"{'call':16s} {'rel L2 patched vs recipe':>26s} {'fp16 noise floor':>17s} {'rel L2 unquantized vs recipe':>30s}")
    ok = True
    for k in sorted(d.files):
        if not k.endswith(".x"):
            continue
        step, L = k.split(".")[0], k.split(".")[1]
        x, y = d[k], d[k.replace(".x", ".y")]                 # x [1, G, S, in], y [1, G, S, out]
        W = d[f"W.{L}"]                                        # [G, in, out]
        xq = act_quant_ref(x)
        yq = np.einsum("bgsi,gio->bgso", xq.astype(np.float64), W.astype(np.float64))
        yu = np.einsum("bgsi,gio->bgso", x.astype(np.float64), W.astype(np.float64))
        rl2 = lambda p, q: float(np.linalg.norm(p - q) / np.linalg.norm(q))
        noise = rl2(yq.astype(np.float16).astype(np.float64), yq)
        m = dict(patched_vs_recipe=rl2(y, yq), noise=noise, unquantized_vs_recipe=rl2(yu, yq))
        res[f"{step}.{L}"] = m
        good = m["patched_vs_recipe"] < 4 * max(noise, 5e-4) and m["unquantized_vs_recipe"] > 3 * m["patched_vs_recipe"]
        ok &= good
        print(f"{step + '.' + L:16s} {m['patched_vs_recipe']:26.2e} {noise:17.2e} {m['unquantized_vs_recipe']:30.2e}  {'ok' if good else 'FAIL'}")
    print("VERDICT:", "PASS (patched wo_a input is quantized like the checkpoint recipe)" if ok else "FAIL")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("pack"); d.add_argument("out"); d.add_argument("--tokens", type=int, default=40); d.add_argument("--no-fq", action="store_true")
    r = sub.add_parser("ref"); r.add_argument("inp"); r.add_argument("--json")
    a = ap.parse_args()
    sys.exit(dump(a) if a.cmd == "dump" else ref(a))
