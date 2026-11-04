"""Interleaved prefill A/B in ONE process (robust to a shared GPU): exl3 fused NAX decode, exl3 decode + MLX matmul, and mlx-lm 4-bit, same tokens.
usage: bench_pf_ab.py EXL3_DIR MLX_DIR [--T 512,2048] [--reps 4] [--chunk N]; prints best/median tok/s per arm and ratios vs mlx."""
import argparse, time, statistics, mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("exl3"); ap.add_argument("mlx")
ap.add_argument("--T", default="512,2048"); ap.add_argument("--reps", type=int, default=4); ap.add_argument("--chunk", type=int, default=0)
a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
from transformers import AutoTokenizer
ids = AutoTokenizer.from_pretrained(a.exl3).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 600)
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.exl3_linear import Exl3Linear
from kyojin_mlx import exl3_linear as EL, wcache as WC, dense_pf as DP
be, _ = load_backend(a.exl3)
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
m = lm = None
if a.mlx != "none": m, _ = load(a.mlx); lm = m.language_model
LINS = [q for q in be.model.modules() if type(q) is Exl3Linear]
WEFF = {}
for q in WC._linears(be.model): WEFF[id(q)] = DP.fold(q)
mx.eval(list(WEFF.values()))
def run_mlx(c, x):
    h = lm.model(mx.array(x)[None], c)[:, -1:]
    lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    mx.eval(lg, *[s for ci in c for s in ci.state if s is not None]); return lg
def fm(v, w=False):
    def run(c, x):
        for q in LINS: q.fused_max = v; q._weff = WEFF.get(id(q)) if w else None
        EL.DENSE_FUSED_MAX = v
        return be.prefill_chunk(c, x)
    return run
arms = {"fused": (be.make_cache, fm(1 << 20)), "dec+mm": (be.make_cache, fm(0)), "weff": (be.make_cache, fm(0, True))}
if m is not None: arms["mlx"] = (lambda: make_prompt_cache(m), run_mlx)
for T in map(int, a.T.split(",")):
    ch = a.chunk or T
    def go(k):
        mk, run = arms[k]; c = mk(); mx.synchronize(); t0 = time.perf_counter()
        if k == "mlx":
            for i in range(0, T, ch): run(c, ids[i:i + ch][:min(ch, T - i)])
        else:
            EL.DENSE_FUSED_MAX = 1 << 20 if k == "fused" else 0
            if k == "weff": pass
            with WC.prefill_scope(be.model, ch, -(-T // ch)):      # wcache: decoded dense weights shared by the chunks of one prompt (as the server does)
                for i in range(0, T, ch): run(c, ids[i:i + ch][:min(ch, T - i)])
        return T / (time.perf_counter() - t0)
    for k in arms: go(k)
    r = {k: [] for k in arms}
    for _ in range(a.reps):
        for k in arms: r[k].append(go(k))
    s = "  ".join(f"{k} {max(v):.0f}/{statistics.median(v):.0f}" for k, v in r.items())
    print(f"T={T} chunk={ch} best/med tok/s: {s}"+("" if "mlx" not in r else f"  fused/mlx {max(r['fused'])/max(r['mlx']):.3f}  dec+mm/mlx {max(r['dec+mm'])/max(r['mlx']):.3f}  weff/mlx {max(r['weff'])/max(r['mlx']):.3f}"), flush=True)
