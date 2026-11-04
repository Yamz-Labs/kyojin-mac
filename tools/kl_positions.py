"""Where does the KL of one parity doc sit: per-position KL vs the BF16 reference. usage: kl_positions.py PACK DOCS.json REFDIR DOC [TOP]"""
import json, sys, numpy as np, mlx.core as mx
from pathlib import Path
from kyojin_mlx.exl3_linear import load_exl3
pack, docs, ref, name = sys.argv[1:5]; top = int(sys.argv[5]) if len(sys.argv) > 5 else 15
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(pack)
d = json.loads(Path(docs).read_text()); ids = d["ids"][d["names"].index(name)]
r = np.load(Path(ref) / f"ref_{name}.npz")
lm = load_exl3(pack)[0].language_model
hd = (lambda h: lm.model.embed_tokens.as_linear(h)) if getattr(lm, "lm_head", None) is None else lm.lm_head
h = lm.model(mx.array(ids)[None])[0]
T = len(ids) - 1; kl = np.zeros(T); ok = np.zeros(T)
for s in range(0, T, 256):
    e = min(s + 256, T)
    lg = hd(h[s:e]).astype(mx.float32); lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    ti = mx.array(r["idx"][s:e].astype(np.int32)); tl = mx.array(r["lp"][s:e]); p = mx.exp(tl)
    k = mx.sum(p * (tl - mx.take_along_axis(lp, ti, axis=-1)), axis=-1) / mx.sum(p, axis=-1)
    mx.eval(k); kl[s:e] = np.array(k); ok[s:e] = np.array(mx.argmax(lp, axis=-1)) == r["top1"][s:e]
print(f"{name}: T={T} mean KL {kl.mean():.3f}  median {np.median(kl):.4f}  KL sum share of the top 5 positions {np.sort(kl)[-5:].sum() / kl.sum():.2f}  top1 {ok.mean():.3f}")
for i in np.argsort(-kl)[:top]:
    print(f"pos {i:4d} kl {kl[i]:7.3f} ref top1 {tok.decode([int(r['top1'][i])])!r:14} ctx ...{tok.decode(ids[max(0, i - 6):i + 1])!r}")
