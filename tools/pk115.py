"""pk115.py PACK [--tokens N] [--chunk 512] [--decode 24]: DS4.1 decode with the fp4-packed cache read directly by the attention kernel (fuse_attn.PACKED_ATTEND) vs unpacked per layer
vs the stock dense cache: logits bitwise at every step, ms per token. Prefill runs once per arm (the three arms share the prompt)."""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tools")); sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
import mlx.core as mx
import ced_gate as G
from kyojin_mlx import fp4pool, fuse_attn
from kyojin_mlx.serve.backends import arrays_of

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--tokens", type=int, default=4096); ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--decode", type=int, default=24); ap.add_argument("--arms", default="dense,unpack,packed"); a = ap.parse_args()
model, spec, cfg, tables = G.load(a.pack, None)
lm = model.language_model
from kyojin_mlx import fuse
print('fuse installed', fuse.install(spec), flush=True)
vocab = cfg.get("text_config", cfg)["vocab_size"]
ids = [int(x) for x in np.random.default_rng(0).integers(3, vocab, a.tokens)]


def run(arm):
    fuse_attn.PACKED_ATTEND = arm == "packed"
    cache = lm.make_cache()
    if arm != "dense":
        cache = fp4pool.wrap_cache(cache)
    for s in range(0, a.tokens, a.chunk):
        out = lm(mx.array([ids[s:s + a.chunk]]), cache=cache)
        mx.eval(out.logits, *arrays_of(cache))
    lg = out.logits[:, -1].astype(mx.float32)
    logs = [lg]
    t = int(mx.argmax(lg, -1).item())
    t0 = None
    for i in range(a.decode):
        if i == 4: t0 = time.time()
        lg = lm(mx.array([[t]]), cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg, *arrays_of(cache))
        logs.append(lg); t = int(mx.argmax(lg, -1).item())
    return logs, (time.time() - t0) / (a.decode - 4) * 1000


def ab(rounds=3, seg=8):
    """One prefill, then segments of `seg` decode steps alternating unpacked / packed on the same cache (ABAB), ms per token of each."""
    cache = fp4pool.wrap_cache(lm.make_cache())
    for s in range(0, a.tokens, a.chunk):
        out = lm(mx.array([ids[s:s + a.chunk]]), cache=cache)
        mx.eval(out.logits, *arrays_of(cache))
    t = int(mx.argmax(out.logits[:, -1], -1).item())
    ms = {"unpack": [], "packed": []}
    for r in range(rounds):
        for arm in (("unpack", "packed") if r % 2 == 0 else ("packed", "unpack")):
            fuse_attn.PACKED_ATTEND = arm == "packed"
            for i in range(seg + 2):
                if i == 2: t0 = time.time()
                lg = lm(mx.array([[t]]), cache=cache).logits[:, -1].astype(mx.float32)
                mx.eval(lg, *arrays_of(cache)); t = int(mx.argmax(lg, -1).item())
            ms[arm].append(round((time.time() - t0) / seg * 1000, 2))
    print("PK115AB", json.dumps(dict(tokens=a.tokens, **ms)), flush=True)


if a.arms == "ab":
    ab(4, 12); sys.exit(0)
res = {}
for arm in a.arms.split(","):
    res[arm] = run(arm)
    print("arm", arm, round(res[arm][1], 2), "ms/tok", flush=True)
ref = res[a.arms.split(",")[0]][0]
out = dict(tokens=a.tokens, **{f"ms_{k}": round(v[1], 2) for k, v in res.items()}, **{f"bitwise_{k}": all(bool(mx.array_equal(x, y)) for x, y in zip(ref, v[0])) for k, v in res.items()})
print("PK115", json.dumps(out), flush=True)
