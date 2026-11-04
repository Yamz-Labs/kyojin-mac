"""One-chunk A/B at a given context offset (KV of the full-attention layers tiled to o rows), interleaved, min / p25 / median. usage: ab_chunk84.py mlx|exl3 MODEL_DIR O LEVERS [rounds=8]
levers (comma): gdnpf, sdpa. The stock arm has none of them."""
import sys, os, time, mlx.core as mx
os.environ.setdefault('KYOJIN_GDN_PF', '0')
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from transformers import AutoTokenizer
from mlx_lm.models import qwen3_5 as Q5
from mlx_lm.models.cache import KVCache
kind, path, O, levers = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(","); R = int(sys.argv[5]) if len(sys.argv) > 5 else 8
mx.set_cache_limit(256 * 2 ** 20); CH = 2048
ids = AutoTokenizer.from_pretrained(path).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 4000)[:O + CH]
if kind == "exl3":
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(path); model = be.model; mk = be.make_cache
else:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, _ = load(path); mk = lambda: make_prompt_cache(model)
lm = model.language_model
stock_gdn = Q5.GatedDeltaNet.__call__; pf_gdn = None
if "gdnpf" in levers:
    from kyojin_mlx import gdn_pf; gdn_pf.MIN_T = 64; gdn_pf.install(); pf_gdn = Q5.GatedDeltaNet.__call__; Q5.GatedDeltaNet.__call__ = stock_gdn
if "sdpa" in levers:
    from kyojin_mlx import sdpa_pf; sdpa_pf.install()
cache = mk()
def run(x, c):
    h = lm.model(mx.array(x)[None], c)[:, -1:]
    lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    mx.eval(lg, *[s for ci in c for s in ci.state if s is not None])
run(ids[:CH], cache)
kvl = [c for c in cache if isinstance(c, KVCache)]
reps = O // CH + 1
master = [(mx.tile(c.keys[:, :, :CH], (1, 1, reps, 1)), mx.tile(c.values[:, :, :CH], (1, 1, reps, 1))) for c in kvl]; mx.eval(master)
def at():
    for c, (k, v) in zip(kvl, master): c.keys = k[:, :, :O] + 0; c.values = v[:, :, :O] + 0; c.offset = O
    mx.eval([x for c in kvl for x in (c.keys, c.values)]); mx.synchronize()
def arm(on):
    Q5.GatedDeltaNet.__call__ = pf_gdn if (on and pf_gdn) else stock_gdn
    if "sdpa" in levers: sdpa_pf.set(on)
def t_one(on):
    arm(on); at(); t0 = time.perf_counter(); run(ids[O:O + CH], cache); return (time.perf_counter() - t0) * 1e3
for on in (False, True): t_one(on)
res = {False: [], True: []}
for r in range(R):
    for on in ((False, True) if r % 2 == 0 else (True, False)): res[on].append(t_one(on))
for on, v in res.items(): v.sort(); print(f"{'lever' if on else 'stock':6s} o={O} chunk {CH}: min {v[0]:.1f} p25 {v[len(v)//4]:.1f} median {v[len(v)//2]:.1f} ms")
s, l = sorted(res[False]), sorted(res[True]); print(f"ratio min {s[0]/l[0]:.3f} p25 {s[len(s)//4]/l[len(l)//4]:.3f} median {s[len(s)//2]/l[len(l)//2]:.3f}")
