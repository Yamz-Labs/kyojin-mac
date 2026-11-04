"""Is decode ALU-bound by the trellis codebook? ms/token of the same Qwen3.5-4B recipe with the mul1 vs the mcg codebook, plain decode after a --ctx prefill.
Both packs resident in ONE process; blocks of --n tokens interleaved ABAB / BABA for --rounds rounds (3 by default), min / p25 / median reported.
usage: codebook_alu.py PACK_MUL1 PACK_MCG [--ctx 4096] [--n 8] [--rounds 3]
PACK_* = a local dir, or host:dir (rsync -aL into --cache, skipped when already there)."""
import argparse, os, subprocess, sys, time, statistics
sys.path.insert(0, os.getcwd())
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer
from kyojin_mlx.exl3_linear import load_exl3

ap = argparse.ArgumentParser(); ap.add_argument("mul1"); ap.add_argument("mcg")
ap.add_argument("--ctx", type=int, default=4096); ap.add_argument("--n", type=int, default=8); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--chunk", type=int, default=1024); ap.add_argument("--cache", default=os.path.expanduser("~/day1_run/packs/models/mcgmac")); o = ap.parse_args()


def fetch(p):
    if ":" not in p or os.path.isdir(p):
        return p
    host, d = p.split(":", 1)
    dst = os.path.join(o.cache, os.path.basename(d.rstrip("/")))
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["rsync", "-aL", "--partial", f"{host}:{d.rstrip('/')}/", dst + "/"], check=True)
    return dst


P = {"mul1": fetch(o.mul1), "mcg": fetch(o.mcg)}
ids = AutoTokenizer.from_pretrained(P["mul1"]).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 400)[:o.ctx]
S = {}
for k, p in P.items():
    m, _ = load_exl3(p); c = make_prompt_cache(m)
    for i in range(0, len(ids), o.chunk):
        lg = m(mx.array(ids[i:i + o.chunk])[None], cache=c); mx.eval(lg)
    t = mx.argmax(lg[:, -1], axis=-1); mx.eval(t); S[k] = [m, c, t]


def run(k, n):
    m, c, t = S[k]; step = lambda y: mx.argmax(m(y[:, None], cache=c)[:, -1], axis=-1)
    t0 = time.perf_counter(); nxt = step(t); mx.async_eval(nxt)
    for i in range(n):
        cur = nxt
        if i + 1 < n: nxt = step(cur); mx.async_eval(nxt)
        cur.item()
    S[k][2] = cur; return (time.perf_counter() - t0) / n * 1e3


for k in P: run(k, 4)
res = {k: [] for k in P}
for r in range(o.rounds * 4):                       # 4 blocks per pack per ABAB round, 3 rounds
    for k in (("mul1", "mcg") if r % 2 == 0 else ("mcg", "mul1")): res[k].append(run(k, o.n))
st = {}
for k in P:
    v = sorted(res[k]); st[k] = (v[0], v[len(v) // 4], statistics.median(v))
    print(f"{k:5s} {P[k].split('/')[-1]:14s} min {v[0]:6.2f} p25 {v[len(v)//4]:6.2f} median {st[k][2]:6.2f} ms/token  (n={len(v)} blocks of {o.n}, ctx {len(ids)})")
r_min, r_p25 = st["mcg"][0] / st["mul1"][0], st["mcg"][1] / st["mul1"][1]
pair = statistics.median([b / a for a, b in zip(res["mul1"], res["mcg"])])
verdict = "ALU-bound like the Air" if r_p25 < 0.9 else ("bandwidth-bound" if r_p25 > 0.95 else "in between")
print(f"REPORT m.codebook_alu: ratio mcg/mul1 min {r_min:.3f} p25 {r_p25:.3f} paired-median {pair:.3f} => {verdict}  (ratio mcg/mul1: < 0.9 => ALU-bound like the Air; ~1.0 => bandwidth-bound)")
