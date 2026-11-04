"""Interleaved whole-model prefill A/B in one process: exl3 (served prefill_chunk inside wcache.prefill_scope) vs mlx-lm. usage: pf2k_ab.py EXL3_DIR MLX_DIR T CHUNK [reps=5]"""
import sys, time, mlx.core as mx
from transformers import AutoTokenizer
ep, mp, T, C = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); R = int(sys.argv[5]) if len(sys.argv) > 5 else 5
mx.set_cache_limit(256 * 2 ** 20)
ids = AutoTokenizer.from_pretrained(ep).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 4000)[:T]
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import wcache as WC
be, _ = load_backend(ep)
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model, _ = load(mp); lm = model.language_model
def run_e():
    c = be.make_cache(); mx.synchronize(); t0 = time.perf_counter()
    with WC.prefill_scope(be.model, C, -(-T // C)):
        for s in range(0, T, C):
            mx.eval(be.prefill_chunk(c, ids[s:s + C]))
    return T / (time.perf_counter() - t0)
def run_m():
    c = make_prompt_cache(model); mx.synchronize(); t0 = time.perf_counter()
    for s in range(0, T, C):
        h = lm.model(mx.array(ids[s:s + C])[None], c)[:, -1:]
        lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
        mx.eval(lg, *[st for ci in c for st in ci.state if st is not None])
    return T / (time.perf_counter() - t0)
run_e(); run_m()
e, m = [], []
for r in range(R):
    for f, o in ((run_e, e), (run_m, m)) if r % 2 == 0 else ((run_m, m), (run_e, e)):
        o.append(f())
e.sort(); m.sort()
print(f"T={T} chunk={C}: exl3 max {e[-1]:.0f} med {e[len(e)//2]:.0f} | mlx-lm max {m[-1]:.0f} med {m[len(m)//2]:.0f} | ratio max {e[-1]/m[-1]:.3f} med {e[len(e)//2]/m[len(m)//2]:.3f}", flush=True)
