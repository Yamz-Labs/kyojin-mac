"""Pack-side measurements on MLX, run with the venv python (PYTHONPATH = the engine checkout + tools). CLI only; stages parse its JSON.
  mlx_eval.py parity PACK DOCS.json REFDIR OUT.json          KL (renormalised top-256) / top-1 / dNLL per doc against the BF16 reference (refs/qwen35.py)
  mlx_eval.py kv     PACK DOCS.json OUT.json N_EVAL B1 B2 .. KL / top-1 of the quantized KV cache (KYOJIN_KVWIN pool + fp16 sink/window overlay) against the
                     fp16 cache of the SAME pack, and against k8v8, on the last N_EVAL positions of the long doc (every position decode-faithful)
The parity maths is macair41's eval_kl.py (conv_work, feat/conv-mlx); the KV forward and the full-vocab KL are macair37/43's tools/kvwin_kl.py."""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_pack(pack):
    if "quantization" in json.loads((Path(pack) / "config.json").read_text()):    # stock mlx-lm pack (MLX 4-bit rival), not EXL3
        from mlx_lm.utils import load_model
        return load_model(Path(pack), lazy=False)[0]
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(pack)
    return model


def head_of(lm):
    return (lambda h: lm.model.embed_tokens.as_linear(h)) if getattr(lm, "lm_head", None) is None else lm.lm_head


def parity(pack, docs, ref, out):
    d = json.loads(Path(docs).read_text())
    model = load_pack(pack)
    lm = model.language_model
    hd = head_of(lm)
    res = {}
    for n, ids in zip(d["names"], d["ids"]):
        r = np.load(Path(ref) / f"ref_{n}.npz")
        h = lm.model(mx.array(ids)[None])[0]
        kl = ag = nl = rn = 0.0
        T = len(ids) - 1
        for s in range(0, T, 256):
            e = min(s + 256, T)
            lg = hd(h[s:e]).astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(lp, mx.array(ids[s + 1:e + 1])[:, None], axis=-1)[:, 0]
            ti = mx.array(r["idx"][s:e].astype(np.int32)); tl = mx.array(r["lp"][s:e])
            p = mx.exp(tl)
            klv = mx.sum(p * (tl - mx.take_along_axis(lp, ti, axis=-1)), axis=-1) / mx.sum(p, axis=-1)     # KL(teacher || pack) on the teacher's top-256
            am = mx.argmax(lp, axis=-1)
            mx.eval(klv, am, nll)
            kl += float(mx.sum(klv)); nl += float(mx.sum(nll)); ag += int(np.sum(np.array(am) == r["top1"][s:e]))
            mx.clear_cache()
        rn = float(r["nll"][:T].sum())
        res[n] = dict(kl=kl / T, top1=ag / T, dnll=(nl - rn) / T, tokens=T)
        print(n, res[n], flush=True)
    res["_peak_gib"] = mx.get_peak_memory() / 2**30
    Path(out).write_text(json.dumps(res, indent=1))


def kv(pack, docs, out, n_eval, bits):
    import kvwin_kl as K                                  # tools/kvwin_kl.py: cache_for / run_logits / kl_top1
    d = json.loads(Path(docs).read_text())
    ids = d["ids"][0]
    N = len(ids)
    model = load_pack(pack)
    cfg = lambda b: [b, b, 64, 4, 128, 0]                 # kbits, vbits, group, sink, window, f32=0: the values the server's KYOJIN_KVWIN uses (pack.KVWIN_*)
    t0 = time.time()
    ref = K.run_logits(model, ids, K.cache_for(model, [0] * 5), n_eval); mx.eval(ref)
    res, lgs = {}, {}
    for b in [8] + [x for x in bits if x != 8]:
        t1 = time.time()
        got = K.run_logits(model, ids, K.cache_for(model, cfg(b)), n_eval); mx.eval(got)
        kl, top, p99 = K.kl_top1(ref, got)
        res[str(b)] = dict(kl_ref=kl, top1_ref=top, kl_p99=p99, sec=round(time.time() - t1, 1))
        lgs[b] = got
        if b != 8:
            k2, t2, _ = K.kl_top1(lgs[8], got)
            res[str(b)].update(kl_vs_k8=k2, top1_vs_k8=t2)
        print(f"k{b}v{b}", res[str(b)], flush=True)
        if b != 8:
            del lgs[b]
        mx.clear_cache()
    res["_meta"] = dict(N=N, n_eval=n_eval, peak_gib=mx.get_peak_memory() / 2**30, sec=round(time.time() - t0, 1))
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "parity":
        parity(*sys.argv[2:6])
    elif cmd == "kv":
        kv(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), [int(x) for x in sys.argv[6:]])
