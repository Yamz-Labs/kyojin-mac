"""Whole-model prefill A/B, interleaved, chunk 2048: stock vs levers. usage: ab_pf84.py mlx|exl3 MODEL_DIR T LEVERS [rounds=6]  (LEVERS comma list: gdnpf, sdpa ...; each round runs stock then lever, order alternated)
Prints min / p25 tok/s per arm. Same ids, same chunking; the lever toggles are module-level switches (arms differ by nothing else)."""
import sys, os, time, mlx.core as mx
os.environ.setdefault('KYOJIN_GDN_PF', '0')                      # the exl3 loader installs gdn_pf by default: the stock arm must not have it
sys.path.insert(0, __file__.rsplit("/", 1)[0]); sys.path.insert(0, __file__.rsplit("/", 2)[0])
from transformers import AutoTokenizer
from mlx_lm.models import qwen3_5 as Q5
kind, path, T, levers = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(","); R = int(sys.argv[5]) if len(sys.argv) > 5 else 6
mx.set_cache_limit(256 * 2 ** 20)
ids = AutoTokenizer.from_pretrained(path).encode("The history of the Roman Empire begins with the founding of Rome in 753 BC. " * 4000)[:T]
if kind == "exl3":
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(path); model = be.model; mk = be.make_cache
else:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, _ = load(path); mk = lambda: make_prompt_cache(model)
lm = model.language_model
stock_gdn = Q5.GatedDeltaNet.__call__
arms = {"stock": {}, }
if "gdnpf" in levers:
    from kyojin_mlx import gdn_pf; gdn_pf.MIN_T = 64; gdn_pf.install(); pf_gdn = Q5.GatedDeltaNet.__call__; Q5.GatedDeltaNet.__call__ = stock_gdn if stock_gdn is not pf_gdn else stock_gdn
    arms["lever"] = {"gdn": pf_gdn}
if "sdpa" in levers:
    import sdpa_pf; arms.setdefault("lever", {})["sdpa"] = True
stock_gdn_fn = getattr(stock_gdn, "__wrapped__", stock_gdn)
def setarm(name):
    cfg = arms[name]; Q5.GatedDeltaNet.__call__ = cfg.get("gdn", stock_gdn)
    if "sdpa" in levers: sdpa_pf.set(bool(cfg.get("sdpa")))
def run():
    c = mk(); mx.synchronize(); t0 = time.perf_counter()
    for s in range(0, T, 2048):
        h = lm.model(mx.array(ids[s:s + 2048])[None], c)[:, -1:]
        lg = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
        mx.eval(lg, *[st for ci in c for st in ci.state if st is not None])
    return T / (time.perf_counter() - t0)
for n in arms: setarm(n); run()
res = {n: [] for n in arms}
for r in range(R):
    for n in (list(arms) if r % 2 == 0 else list(arms)[::-1]): setarm(n); res[n].append(run())
for n, v in res.items():
    v.sort(reverse=True); print(f"{n:6s} T={T}: best {v[0]:.0f} p25 {v[len(v)//4]:.0f} median {v[len(v)//2]:.0f} tok/s  all {[round(x) for x in v]}")
print(f"best ratio {res['lever'][0]/res['stock'][0]:.3f}  p25 ratio {sorted(res['lever'],reverse=True)[len(res['lever'])//4]/sorted(res['stock'],reverse=True)[len(res['stock'])//4]:.3f}  median ratio {sorted(res['lever'],reverse=True)[len(res['lever'])//2]/sorted(res['stock'],reverse=True)[len(res['stock'])//2]:.3f}")
