"""Layer-level gate (S4 lite): Kyojin-patched model vs the same mlx-vlm model with dense fp32 weights from the numpy reconstruct.
  python tools/layer_gate.py run   PACK OUT.npz [--mode patched|ref] [--ids PATCHED.npz] [--tokens 1,100,2000,3500,77]
  python tools/layer_gate.py compare PATCHED.npz REF.npz [--json out.json]
Both runs: prefill of the first n-1 tokens (one forward, S rows), then one decode step with the cache (1 row). Per layer the attention
output, the FFN / MoE output and the layer output are recorded, plus the router choice and the logits.
`patched` = patch_model (fp16 activations, Exl3 kernels, fused MoE). `ref` = same classes, every quantized linear replaced by its
numpy-reconstructed fp32 weight, fp32 activations; routed experts: only the experts the patched run selected are reconstructed (their
ids come from --ids; the router is remapped onto them) and the reference must select the same ids.
Run `patched` and `ref` in separate processes (memory). mlx-vlm on PYTHONPATH (the Mac venv has it in vlm_pkg)."""
import argparse, json, sys, time

import numpy as np
import mlx.core as mx

from kyojin_mlx import moe as M, patch as P, ref_numpy as R
from kyojin_mlx.archs import FUSED, FUSED_ORDER, target

PARTS = {"glm5_next": ("self_attn", "mlp"), "deepseek_v41": ("attn", "ffn")}


def hook(obj, store, key, pick=lambda out: out):
    cls = obj.__class__

    def __call__(self, *a, **k):
        out = cls.__call__(self, *a, **k)
        store.append((key, pick(out)))
        return out

    obj.__class__ = type(cls.__name__, (cls,), {"__call__": __call__})


def install_probes(model, arch, store, inverse=None):
    attn, ffn = PARTS[arch]
    for i, layer in enumerate(model.language_model.layers):
        hook(layer, store, f"L{i}.out", lambda o: o[0])
        hook(getattr(layer, attn), store, f"L{i}.attn", lambda o: o[0] if isinstance(o, tuple) else o)
        mod = getattr(layer, ffn)
        hook(mod, store, f"L{i}.ffn")
        if hasattr(mod, "gate"):
            inv = mx.array(np.array(inverse[i], np.int32)) if inverse else None
            hook(mod.gate, store, f"L{i}.router", (lambda o, inv=inv: inv[o[0]]) if inv is not None else (lambda o: o[0]))


def forward(model, arch, tokens, inverse=None):
    lm = model.language_model
    store: list = []
    install_probes(model, arch, store, inverse)
    cache = lm.make_cache()
    out = {}
    ids = mx.array([tokens])
    for step, sl in (("prefill", slice(0, len(tokens) - 1)), ("decode", slice(len(tokens) - 1, len(tokens)))):
        store.clear()
        logits = lm(ids[:, sl], cache=cache).logits
        rec = {f"{step}.{k}": v for k, v in store}
        rec[f"{step}.logits"] = logits
        mx.eval(*rec.values())
        out.update({k: np.array(v.astype(mx.float32) if v.dtype in (mx.float16, mx.bfloat16) else v) for k, v in rec.items()})
    return out


def fill_reference(model, plan, touched):
    spec, t = plan.spec, plan.t
    Wd = lambda p, g=None: R.reconstruct(np.array(t[p + ".trellis"] if g is None else t[p + ".trellis"][g]),
                                         np.array(t[p + ".suh"] if g is None else t[p + ".suh"][g]),
                                         np.array(t[p + ".svh"] if g is None else t[p + ".svh"][g]), P._cb(t, p, plan.default_cb)).T.copy()
    fused = {}
    for p in plan.linears:
        q = target(spec, p)
        if P._has(model, q):
            P._get(model, q).weight = mx.array(Wd(p))
            plan.loaded.add(q + ".weight")
        else:
            base, leaf = q.rsplit(".", 1)
            fused.setdefault((base, FUSED[leaf]), {})[leaf] = Wd(p)
    for (base, fname), mem in fused.items():
        P._get(model, f"{base}.{fname}").weight = mx.array(np.concatenate([mem[n] for n in FUSED_ORDER[fname]], axis=0))
        plan.loaded.add(f"{base}.{fname}.weight")
    for p in plan.slices:
        old = P._get(model, target(spec, p))
        old.weight = mx.array(np.stack([Wd(p, g) for g in range(old.weight.shape[0])]))
        plan.loaded.add(target(spec, p) + ".weight")
    groups = {}
    for p in plan.stacks:
        parent, proj = target(spec, p).rsplit(".", 1)
        groups.setdefault(parent, {})[proj] = p
    for parent, projs in groups.items():
        layer = int(__import__("re").search(r"layers\.(\d+)\.", parent)[1])
        ids = touched[layer]
        old = P._get(model, parent)
        for proj, p in projs.items():
            st = M.Stack.from_arrays(t, p, P._cb(t, p, plan.default_cb))
            ws = []
            for e in ids:
                ws.append(R.reconstruct(np.array(st.expert_trellis(e)), np.array(st.suh[e]), np.array(st.svh[e]), st.cb).T.copy())
            getattr(old, proj).weight = mx.array(np.stack(ws))
            plan.loaded.add(f"{parent}.{proj}.weight")
        table = np.full(plan.config_experts(parent), -1, np.int32)
        table[np.array(ids)] = np.arange(len(ids), dtype=np.int32)
        gate = P._get(model, parent.rsplit(".", 1)[0]).gate
        cls, tab = gate.__class__, mx.array(table)

        def remapped(self, *a, _cls=cls, _tab=tab, **k):
            inds, w = _cls.__call__(self, *a, **k)
            return _tab[inds], w

        gate.__class__ = type(cls.__name__, (cls,), {"__call__": remapped})


def cmp_arrays(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = np.linalg.norm(a - b)
    return dict(rel_l2=float(d / max(np.linalg.norm(b), 1e-30)), cos=float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
                max_abs_over_max_ref=float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("pack"); r.add_argument("out"); r.add_argument("--mode", choices=["patched", "ref"], default="patched")
    r.add_argument("--ids"); r.add_argument("--tokens", default="1,100,2000,3500,77")
    r.add_argument("--no-fq", action="store_true", help="DS4.1: disable the fp8/fp4 fake quant everywhere (isolates the Kyojin kernels from fq bin flips)")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b"); c.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "run":
        tokens = [int(v) for v in a.tokens.split(",")]
        model, spec, cfg = P.build_model(a.pack)
        if a.no_fq:
            import importlib, dataclasses
            importlib.import_module(f"mlx_vlm.models.{spec.name}.fakequant").DISABLE = True
            if spec.moe is not None:
                spec = dataclasses.replace(spec, moe=dataclasses.replace(spec.moe, fake_quant_fp8=False))
        t0 = time.time()
        if a.mode == "patched":
            rep = P.patch_model(model, a.pack, spec, cfg)
            print("patched:", rep, f"{time.time() - t0:.1f}s", flush=True)
        else:
            touched = {}
            ref = np.load(a.ids)
            for k in ref.files:
                if k.endswith(".router"):
                    layer = int(k.split(".")[1][1:])
                    touched[layer] = sorted(set(touched.get(layer, [])) | set(np.unique(ref[k]).tolist()))
            plan = P.load_dense(model, a.pack, spec, cfg, dense_dtype=mx.float32)
            tc = cfg["text_config"]
            plan.config_experts = lambda parent: int(tc["n_routed_experts"])
            fill_reference(model, plan, touched)
            P.check_loaded(model, plan)
            model.eval()
            mx.eval(model.language_model.parameters())
            print("reference built", f"{time.time() - t0:.1f}s, experts per layer {({k: len(v) for k, v in touched.items()})}", flush=True)
        out = forward(model, spec.name, tokens, None if a.mode == "patched" else touched)
        np.savez(a.out, **out)
        print("peak GiB", round(mx.get_peak_memory() / 2 ** 30, 2), "->", a.out)
    else:
        A, B = np.load(a.a), np.load(a.b)
        res = {}
        print(f"{'tensor':28s} {'rel L2':>10s} {'cosine':>12s} {'max|d|/max|ref|':>16s}")
        for k in A.files:
            if k.endswith(".router"):
                print(f"{k:28s} router ids equal: {bool(np.array_equal(np.sort(A[k], -1), np.sort(B[k], -1)))}")
                res[k] = dict(equal=bool(np.array_equal(np.sort(A[k], -1), np.sort(B[k], -1))))
                continue
            m = cmp_arrays(A[k], B[k])
            res[k] = m
            print(f"{k:28s} {m['rel_l2']:10.2e} {m['cos']:12.8f} {m['max_abs_over_max_ref']:16.2e}")
        la, lb = A["decode.logits"].reshape(-1), B["decode.logits"].reshape(-1)
        res["decode_argmax_equal"] = bool(la.argmax() == lb.argmax())
        print("decode argmax equal:", res["decode_argmax_equal"], "| prefill last-row argmax equal:",
              bool(A["prefill.logits"][0, -1].argmax() == B["prefill.logits"][0, -1].argmax()))
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
