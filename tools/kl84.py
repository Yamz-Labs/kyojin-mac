"""Quality gate on real text: prefill T tokens (calib_text.jsonl, chunk 2048), logits of the last P positions, stock vs lever(s), plus the noise floor stock chunk 2048 vs 512.
usage: kl84.py MODEL_DIR [T=4096] [P=256] [levers=gdnpf]   variants: comma list of pre,pre+norm,pre+fast,pre+norm+fast
Reports mean / max KL(stock || variant) over the P positions, top-1 agreement, and the logit max|d|."""
import os
import sys, json, time, mlx.core as mx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mlx_lm import load
from mlx_lm.models import qwen3_5 as Q5
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
path = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 4096; P = int(sys.argv[3]) if len(sys.argv) > 3 else 256
variants = [v.split("+") for v in (sys.argv[4] if len(sys.argv) > 4 else "pre,pre+norm,pre+fast,pre+norm+fast").split(",")]   # pre = fused front end; +norm fused gated norm; +fast chunked core
tok = AutoTokenizer.from_pretrained(path); model, _ = load(path); lm = model.language_model
txt = ""
for l in open(os.path.expanduser("~/kyojin-mlx/.worktrees/pf84/data/calib_text.jsonl")):
    d = json.loads(l)
    if d.get("domain") in ("en", "tech", "fr", "code") and "text" in d: txt += d["text"] + "\n\n"
    if len(txt) > 6 * T: break
ids = tok.encode(txt)[:T]; assert len(ids) == T, len(ids)
stock_call = Q5.GatedDeltaNet.__call__
from kyojin_mlx import gdn_pf
gdn_pf.install(); fused_call = Q5.GatedDeltaNet.__call__; Q5.GatedDeltaNet.__call__ = stock_call
def logits(chunk, on):
    Q5.GatedDeltaNet.__call__ = fused_call if on else stock_call
    if on: gdn_pf.CORE = "fast16" if "fast16" in on else "fast32" if "fast32" in on else "nax" if "nax" in on else "fast" if "fast" in on else "step"; gdn_pf.NORM = "norm" in on
    c = make_prompt_cache(model); out = None
    for s in range(0, T, chunk):
        x = mx.array(ids[s:s + chunk])[None]; h = lm.model(x, c)
        if s + chunk >= T:
            h = h[:, -P:]; out = (lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)).astype(mx.float32)
        mx.eval(out if out is not None else h, *[st for ci in c for st in ci.state if st is not None]) if out is not None else mx.eval(h, *[st for ci in c for st in ci.state if st is not None])
    Q5.GatedDeltaNet.__call__ = stock_call
    return out[0]
def lp(l): return l - mx.logsumexp(l, -1, keepdims=True)
ref = lp(logits(2048, [])); mx.eval(ref)
def cmp(name, l):
    b = lp(l); kl = mx.sum(mx.exp(ref) * (ref - b), -1); top = mx.mean((mx.argmax(ref, -1) == mx.argmax(b, -1)).astype(mx.float32))
    mx.eval(kl, top); print(f"{name:22s} KL mean {kl.mean().item():.3e} max {kl.max().item():.3e}  top1 {top.item()*100:.1f} %  max|dlogit| {mx.max(mx.abs(ref - b)).item():.3f}", flush=True)
cmp("stock chunk512 (floor)", logits(512, []))
cmp("stock chunk2048 again", logits(2048, []))
for v in variants: cmp("gdnpf " + "+".join(v), logits(2048, v))
