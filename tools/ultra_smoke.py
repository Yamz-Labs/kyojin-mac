"""Whole-model smoke: greedy text (64 tokens), pp512 and tg128 of an exl3 Qwen pack. usage: ultra_smoke.py PACK [--json out]. Run from the worktree to test."""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.getcwd())
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
from kyojin_mlx.exl3_linear import load_exl3
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--json"); a = ap.parse_args()
model, _ = load_exl3(a.pack)
tok = AutoTokenizer.from_pretrained(a.pack)
def gen(ids, n):
    cache = make_prompt_cache(model)
    t0 = time.perf_counter(); lg = model(mx.array(ids)[None], cache=cache)[:, -1]; t = mx.argmax(lg, axis=-1); mx.eval(t); pf = time.perf_counter() - t0
    out, t1 = [int(t.item())], time.perf_counter()
    for _ in range(n - 1):
        lg = model(t[:, None], cache=cache)[:, -1]; t = mx.argmax(lg, axis=-1); mx.eval(t); out.append(int(t.item()))
    return out, pf, time.perf_counter() - t1
text = tok.encode("Explain in three sentences how a gated delta network differs from softmax attention.")
toks, _, _ = gen(text, 64)
long = tok.encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 100)[:512]
gen(long[:64], 4)
pf = min(gen(long, 1)[1] for _ in range(2))
_, _, dt = gen(long[:32], 128)
r = dict(md5=hashlib.md5(str(toks).encode()).hexdigest()[:12], toks=toks, pp512_toks=512 / pf, tg128_toks=127 / dt, text=tok.decode(toks))
print(f"md5 {r['md5']} pp512 {r['pp512_toks']:.1f} tok/s tg128 {r['tg128_toks']:.1f} tok/s")
if a.json: json.dump(r, open(a.json, "w"))
