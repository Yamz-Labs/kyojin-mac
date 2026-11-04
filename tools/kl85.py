"""KL / top-1 gate of a GEMV numerics switch on the batched multi-user decode: N users forked twice from the same prefill; step with the baseline (A) and with the switch (B)
fed the same tokens (A's greedy). usage: kl85.py KIND MODEL N STEPS SWITCH(half4|half2|mma4|none) [PROMPT_TOKENS=300]"""
import sys, json, numpy as np
import mlx.core as mx
from kyojin_mlx import multiuser as MU
import kvwin_kl as K
kind, mdir, N, STEPS, sw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
PT = int(sys.argv[6]) if len(sys.argv) > 6 else 300
from kyojin_mlx.exl3_linear import load_exl3
from transformers import AutoTokenizer
from kyojin_mlx import exl3_linear as EL, metal_src as MS
model, _ = load_exl3(mdir); tok = AutoTokenizer.from_pretrained(mdir)
MU.install(model)
TOG = {"half4": [(MS, "HALF_ROWS", 4)], "half2": [(MS, "HALF_ROWS", 2)], "mma4": [(EL, "GEMV_MAX_ROWS", 3), (EL, "MMA_ROWS", 4)], "none": []}[sw]
base = [(m, a, getattr(m, a)) for m, a, _ in TOG]
def on():
    for m, a, v in TOG: setattr(m, a, v)
def off():
    for m, a, v in base: setattr(m, a, v)
users = []
for u in range(N):
    ids = K.tokens(tok, PT, 700 * u + 11)
    c, lg = MU.prefill(model, ids)
    users.append((c, lg))
bcA = MU.batch_caches([MU.fork_cache(c) for c, _ in users]); bcB = MU.batch_caches([MU.fork_cache(c) for c, _ in users])
tokA = mx.concatenate([mx.argmax(lg, axis=-1) for _, lg in users]); mx.eval(tokA)
kls, agree, rel = [], 0, []
tot = 0
for s in range(STEPS):
    off(); la = MU.logits_last(model, tokA[:, None], bcA).astype(mx.float32); mx.eval(la)
    on(); lb = MU.logits_last(model, tokA[:, None], bcB).astype(mx.float32); mx.eval(lb); off()
    pa = mx.softmax(la, axis=-1); lpa = mx.log(pa + 1e-30); lpb = mx.log(mx.softmax(lb, axis=-1) + 1e-30)
    kl = mx.sum(pa * (lpa - lpb), axis=-1); mx.eval(kl)
    kls += np.array(kl).tolist()
    ta = mx.argmax(la, axis=-1); tb = mx.argmax(lb, axis=-1); mx.eval(ta, tb)
    agree += int(mx.sum(ta == tb).item()); tot += N
    rel.append(float(mx.sqrt(mx.mean((la - lb) ** 2) / mx.mean(la ** 2)).item()))
    tokA = ta
print(json.dumps(dict(kind=kind, switch=sw, N=N, steps=STEPS, kl_mean=float(np.mean(kls)), kl_max=float(np.max(kls)), top1=agree / tot, logit_rel_rms=float(np.mean(rel)))))
