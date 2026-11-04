"""greedy86.py KIND MODEL OUT: 2K real prefill (512-row chunks, TQ6 cache + fp16 cache) + 64 greedy decode steps; prints md5 of the token ids per cache and appends to OUT."""
import sys, hashlib, json
import mlx.core as mx
import kvwin_kl as K
import kyojin_mlx
kind, mdir, out = sys.argv[1:4]
if kind == "exl3":
    from kyojin_mlx.exl3_linear import load_exl3
    from transformers import AutoTokenizer
    model, _ = load_exl3(mdir); tok = AutoTokenizer.from_pretrained(mdir)
else:
    from mlx_lm import load
    model, tok = load(mdir)
import os
if os.environ.get("GDNPF"):
    from kyojin_mlx import gdn_pf; print("gdn_pf", gdn_pf.install())
lm = model.language_model; hd = K.head(lm); ids = K.tokens(tok, 2048, 0)
for name, cfg in (("tq6", [6, 6, 0, 4, 256]), ("fp16", [0, 0, 0, 0, 0])):
    cache = K.cache_for(model, cfg)
    for s in range(0, 2048, 512):
        h = lm.model(mx.array(ids[s:s + 512])[None], cache); mx.eval(h)
    y = mx.argmax(hd(h[:, -1:])[0], axis=-1); seq = []
    for _ in range(64):
        h = lm.model(y[None], cache); y = mx.argmax(hd(h)[0], axis=-1); mx.eval(y); seq.append(y.item())
    r = dict(src=kyojin_mlx.__file__, kind=kind, model=mdir.split("/")[-1], cache=name, md5=hashlib.md5(json.dumps(seq).encode()).hexdigest()[:10], seq=seq)
    print(r["src"].split("/")[-3], kind, r["model"], name, r["md5"], flush=True); open(out, "a").write(json.dumps(r) + "\n")
