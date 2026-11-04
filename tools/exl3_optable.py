"""Per-op time of one EXL3 Qwen3.5 trunk pass at S rows: mixers (GDN + attention), MLPs, head, rest; ms and GB/s of weight bytes. Hot: runs --warm seconds first.
usage: exl3_optable.py PACK [--rows 1,5,8] [--mma 5] [--warm 20]"""
import argparse, time, statistics
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rows", default="1,5,8"); ap.add_argument("--mma", type=int, default=5)
ap.add_argument("--warm", type=float, default=20); ap.add_argument("--passes", type=int, default=3); a = ap.parse_args()
from kyojin_mlx import exl3_linear as EL
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
model, cfg = EL.load_exl3(a.pack); lm = model.language_model; L = lm.model.layers
head = lm.__dict__.get("lm_head") or lm.lm_head
cache = make_prompt_cache(model)
mx.eval(lm.model(mx.arange(200)[None] % 1000 + 100, cache))
def nb(m): return sum(l._words.nbytes + l._suh.nbytes + l._svh.nbytes for _, l in nn.Module.named_modules(m) if isinstance(l, EL.Exl3Linear))
mix_b = sum(nb(l.linear_attn if l.is_linear else l.self_attn) for l in L); mlp_b = sum(nb(l.mlp) for l in L); head_b = nb(head)
def bench(fn, n=4, reps=5):
    ts = []
    for r in range(reps + 1):
        t0 = time.perf_counter(); mx.eval(fn()); dt = time.perf_counter() - t0
        if r: ts.append(dt * 1e3)
    return statistics.median(ts)
t0 = time.time()
while time.time() - t0 < a.warm: bench(lambda: lm.model(mx.array([[5] * 4]), cache), reps=1)
rows = [int(x) for x in a.rows.split(",")]
res = {}
for ps in range(a.passes):
    for S in rows:
        for m in ([0, a.mma] if S >= a.mma else [0]):
            EL.MMA_ROWS = m
            x = mx.random.normal((1, S, cfg["text_config"]["hidden_size"] if "text_config" in cfg else cfg["hidden_size"])).astype(mx.float16)
            r = res.setdefault((S, m), {})
            tot = bench(lambda: head(lm.model(mx.array([[5] * S]), cache)))
            mixer = bench(lambda: [l.linear_attn(x, None, None) if l.is_linear else l.self_attn(x, None, None) for l in L])
            mlp = bench(lambda: [l.mlp(x) for l in L]); hd = bench(lambda: head(x))
            lins = [(n, mod) for n, mod in nn.Module.named_modules(lm.model) if isinstance(mod, EL.Exl3Linear)]
            xs = {}
            for n, mod in lins: xs.setdefault(mod.in_features, mx.random.normal((1, S, mod.in_features)).astype(mx.float16))
            mlin = [(n, mod) for n, mod in nn.Module.named_modules(lm.model) if isinstance(mod, EL.Exl3Linear) and ".mlp." not in n]
            lin_m = bench(lambda: [mod(xs[mod.in_features]) for _, mod in mlin])
            lin = bench(lambda: [mod(xs[mod.in_features]) for _, mod in lins])
            for k, v in (("linears", lin), ("lin_m", lin_m), ("total", tot), ("mixers", mixer), ("mlps", mlp), ("head", hd)): r.setdefault(k, []).append(v)
print(f"weight bytes: mixers {mix_b/1e6:.0f} MB  mlps {mlp_b/1e6:.0f} MB  head {head_b/1e6:.0f} MB; roof 134 GB/s -> {(mix_b+mlp_b+head_b)/134e6:.1f} ms")
for (S, m), r in sorted(res.items()):
    med = {k: statistics.median(v) for k, v in r.items()}
    f = lambda k, b: f"{med[k]:.1f}" + (f" ({b / med[k] / 1e6:.0f} GB/s)" if b else "")
    print(f"S={S} mma={m}: total {f('total', 0)} | mixers {f('mixers', mix_b)} | mlps {f('mlps', mlp_b)} | head {f('head', head_b)} | linears: mixers {f('lin_m', 0)} mlps {med['linears']-med['lin_m']:.1f} | non-linear: mixers {med['mixers']-med['lin_m']:.1f} mlps {med['mlps']-(med['linears']-med['lin_m']):.1f} rest {med['total']-med['mixers']-med['mlps']-med['head']:.1f}", flush=True)
