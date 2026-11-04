"""First diverging op between a S-row forward and S solo steps (Qwen3.5 EXL3). usage: qspec_bisect.py PACK [--S 3]"""
import argparse
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx import exl3_linear as EL
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--S", type=int, default=3); ap.add_argument("--plen", type=int, default=300); a = ap.parse_args()
EL.ROW_SPLIT_MAX = 16
model, _ = EL.load_exl3(a.pack); lm = model.language_model
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.pack)
ids = tok.encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. Rome grew from a small town on the Tiber into a vast empire. " * 40)[:a.plen]
REC = {}
# simpler: patch instance call through nn.Module subclass
class Tap(nn.Module):
    def __init__(self, m, name): super().__init__(); self.m, self.name = m, name
    def __call__(self, *a, **k):
        y = self.m(*a, **k); REC.setdefault(self.name, []).append(y); return y
layers = lm.model.layers
for i, l in enumerate(layers):
    if l.is_linear:
        la = l.linear_attn
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "conv1d"): setattr(la, n, Tap(getattr(la, n), f"{i}.{n}"))
        la.norm = Tap(la.norm, f"{i}.gnorm")
    else:
        sa = l.self_attn
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"): setattr(sa, n, Tap(getattr(sa, n), f"{i}.{n}"))
    l.mlp = Tap(l.mlp, f"{i}.mlp"); l.input_layernorm = Tap(l.input_layernorm, f"{i}.iln")
    layers[i] = Tap(l, f"{i}.layer") if False else l
def run(x_list, cache):
    out = []
    for x in x_list:
        h = lm.model(mx.array(x)[None], cache); lg = lm.lm_head(h); mx.eval(lg); out.append(lg)
    return out
def fresh():
    c = make_prompt_cache(model); lm.model(mx.array(ids)[None], c); mx.eval([x for cc in c for x in cc.state]); return c
c = fresh(); REC.clear(); lg = lm.lm_head(lm.model(mx.array(ids[-1:])[None], c)); 
c = fresh(); t = int(mx.argmax(lm.lm_head(lm.model(mx.array(ids[-1:])[None], c))[0, -1]).item())
toks = [t]
c = fresh(); REC.clear()
solo = []
for j in range(a.S):
    lg = lm.lm_head(lm.model(mx.array([toks[-1]])[None], c)); mx.eval(lg); toks.append(int(mx.argmax(lg[0, -1]).item()))
solo = {k: [v for v in vs] for k, vs in REC.items()}
c = fresh(); REC.clear()
lg = lm.lm_head(lm.model(mx.array(toks[:a.S])[None], c)); mx.eval(lg)
multi = {k: vs[0] for k, vs in REC.items()}
order = sorted(solo, key=lambda k: (int(k.split(".")[0]), ["iln", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d", "gnorm", "out_proj", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj", "mlp"].index(k.split(".")[1])))
shown = 0
for k in order:
    vs = solo[k]
    # solo: S calls (rows 0..S-1) but call 0 of the REC includes the first single prefill? (cache fresh before, REC cleared) -> S entries
    if len(vs) != a.S: continue
    m = multi[k]
    bad = []
    for j in range(a.S):
        s = vs[j]
        if k.endswith("conv1d"):   # conv input has 3 extra leading frames in multi: compare output frame j
            mj = m[:, j:j + 1]
        else:
            mj = m[:, j:j + 1]
        if mj.shape != s.shape: bad = ["shape"]; break
        if not bool(mx.array_equal(mj, s)): bad.append(j)
    if bad:
        print("DIFF", k, bad); shown += 1
        if shown >= 6: break
print("done; compared", len(order))
