"""Prefill chunk sweep, ours (EXL3, weights decoded per chunk) interleaved, optional mlx-lm at its default 2048 step in the same process.
usage: pf97_chunk.py EXL3 T C1,C2,.. REPS [MLX4]   (C list "0": mlx-lm only; env MFIRST=1: mlx-lm runs first in each round)"""
import os
import sys, time, mlx.core as mx
from transformers import AutoTokenizer
ep, T, Cs, R = sys.argv[1], int(sys.argv[2]), [int(c) for c in sys.argv[3].split(",") if int(c) > 0], int(sys.argv[4]); mp = sys.argv[5] if len(sys.argv) > 5 else None
mx.set_cache_limit(256 * 2 ** 20)
ids = AutoTokenizer.from_pretrained(ep).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 4000)
ids = (ids * (T // len(ids) + 1))[:T]
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import wcache as WC
be, _ = load_backend(ep) if Cs else (None, None)
def run_e(C, n=T):
    c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter()
    with WC.prefill_scope(be.model, C, -(-n // C)):
        for s in range(0, n, C): mx.eval(be.prefill_chunk(c, ids[s:min(s + C, n)]))
    mx.synchronize(); return n / (time.perf_counter() - t0), mx.get_peak_memory() / 2 ** 30
if mp:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, _ = load(mp); lm = model.language_model
    def run_m(n=T, C=2048):
        c = make_prompt_cache(model); mx.synchronize(); t0 = time.perf_counter()
        for s in range(0, n, C):
            h = lm.model(mx.array(ids[s:min(s + C, n)])[None], c)[:, -1:]
            lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
            mx.eval(lg, *[st for ci in c for st in ci.state if st is not None])
        mx.synchronize(); return n / (time.perf_counter() - t0)
for C in Cs: run_e(C, min(C, T))                                       # warm: one chunk each
if mp: run_m(2048)
res = {C: [] for C in Cs}; m = []; peak = {}
for r in range(R):
    mf = bool(os.environ.get("MFIRST")) != (r % 2 == 1)
    if mp and mf: m.append(run_m())
    for C in (Cs if r % 2 == 0 else Cs[::-1]):
        mx.reset_peak_memory(); tps, pk = run_e(C); res[C].append(tps); peak[C] = pk
    if mp and not mf: m.append(run_m())
print(f"T={T}" + (f"  mlx-lm C=2048 best {max(m):.0f} med {sorted(m)[len(m)//2]:.0f}" if m else ""))
for C in Cs:
    x = sorted(res[C]); print(f"  ours C={C:6d} best {x[-1]:.0f} med {x[len(x)//2]:.0f} tok/s  peak {peak[C]:.2f} GiB")
