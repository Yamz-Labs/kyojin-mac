"""usage: conv_eval_kl.py ref MODEL_DIR OUT.npz | q MODEL_DIR REF.npz   (rows: results_mac/conv/eval_16x1024.npy)  -- top-256 KL vs the BF16 teacher, top-1 agreement, NLL on held-out rows."""
import sys, time, json, numpy as np, mlx.core as mx
from pathlib import Path
mode, mdir, third = sys.argv[1], sys.argv[2], sys.argv[3]
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ids = np.load(ROOT / "results_mac" / "conv" / "eval_16x1024.npy")
if mode == "ref":
    from mlx_lm.utils import load_model
    model, _ = load_model(Path(mdir), lazy=False)
else:
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(mdir)
lm = model.language_model
def head(h):
    return lm.model.embed_tokens.as_linear(h) if getattr(lm, "lm_head", None) is None else lm.lm_head(h)
TOP = 256
out = {"idx": [], "lp": [], "nll": [], "top1": [], "tail": []}
if mode == "q":
    ref = np.load(third)
kl, agree, nlls, n = 0.0, 0, 0.0, 0
t0 = time.time()
for r in range(ids.shape[0]):
    x = mx.array(ids[r:r + 1])
    h = lm.model(x)[0]                                  # [1024, hidden] after final norm
    for s in range(0, 1023, 256):
        e = min(s + 256, 1023)
        lg = head(h[s:e]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tgt = mx.array(ids[r, s + 1:e + 1])
        nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        if mode == "ref":
            ti = mx.argpartition(-lp, TOP, axis=-1)[:, :TOP]
            tl = mx.take_along_axis(lp, ti, axis=-1)
            mx.eval(ti, tl, nll)
            out["idx"].append(np.array(ti)); out["lp"].append(np.array(tl)); out["nll"].append(np.array(nll))
            out["top1"].append(np.array(mx.argmax(lp, axis=-1)))
        else:
            k = r * 1023 + s
            ti = mx.array(ref["idx"][k:k + (e - s)].astype(np.int32)); tl = mx.array(ref["lp"][k:k + (e - s)])
            lq = mx.take_along_axis(lp, ti, axis=-1)
            p = mx.exp(tl)
            klv = mx.sum(p * (tl - lq), axis=-1) / mx.sum(p, axis=-1)      # renormalised top-256 KL(teacher || model)
            am = mx.argmax(lp, axis=-1)
            mx.eval(klv, am, nll)
            kl += float(mx.sum(klv)); nlls += float(mx.sum(nll)); n += e - s
            agree += int(np.sum(np.array(am) == ref["top1"][k:k + (e - s)]))
if mode == "ref":
    np.savez(third, **{k: np.concatenate(v) for k, v in out.items() if v})
    nl = np.concatenate(out["nll"]); print(json.dumps({"mode": "ref", "nll": float(nl.mean()), "ppl": float(np.exp(nl.mean())), "n": int(nl.size)}))
else:
    print(json.dumps({"mode": "q", "dir": mdir, "KL_top256": kl / n, "top1_agree": agree / n, "nll": nlls / n, "ppl": float(np.exp(nlls / n)), "n": n, "s": round(time.time() - t0, 1)}))
