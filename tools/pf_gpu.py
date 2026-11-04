import os, sys, ctypes, time, collections
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
import numpy as np, mlx.core as mx
lib = ctypes.CDLL(os.path.expanduser("~/kyojin-mlx/tools/cbcount/libcbcount.dylib"))
lib.cb_install(); lib.cb_gpu_us.restype = ctypes.c_double
from kyojin_mlx import patch as P, rows as R, wcache as WC
import mlx_vlm.models.deepseek_v41.language as L
import json
T = int(sys.argv[1]); CH = int(sys.argv[2])
pack = os.path.expanduser("~/ds41/model")
model, spec, cfg = P.build_model(pack, {}); P.patch_model(model, pack, spec, cfg)
try:
    from kyojin_mlx import engram_rows as ER, engram_mlx as EM
    es = ER.EngramSpec.from_config(cfg); tables = ER.EngramTables(pack, es); EM.install_parallel_gather(model, tables)
except ImportError: pass
R.enable(model, spec)
lm = model.language_model
acc = collections.defaultdict(float); cnt = collections.Counter(); ON = [False]
def gpu(f, tag):
    def w(self, *a, **k):
        if not ON[0]: return f(self, *a, **k)
        mx.eval([x for x in a if isinstance(x, mx.array)]); mx.synchronize(); time.sleep(0.001); g0 = lib.cb_gpu_us()
        r = f(self, *a, **k); mx.eval(r); mx.synchronize(); time.sleep(0.001)
        acc[tag] += lib.cb_gpu_us() - g0; cnt[tag] += 1; return r
    return w
blk = lm.layers[3]
Blk = type(blk); Ffn = type(blk.ffn); Sh = type(blk.ffn.shared_experts); Sw = type(blk.ffn.switch_mlp)
Blk.__call__ = gpu(Blk.__call__, "block"); Ffn.__call__ = gpu(Ffn.__call__, "ffn")
Sh.__call__ = gpu(Sh.__call__, "shared"); Sw.__call__ = gpu(Sw.__call__, "switch(moe)")
a = blk.attn
done = set()
for nm in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
    c = type(getattr(a, nm))
    if c not in done:
        done.add(c); c.__call__ = gpu(c.__call__, "dense:" + c.__name__)
Att = type(a); Att.__call__ = gpu(Att.__call__, "attn(total)")
import types, importlib
def gfn(f, tag):
    def w(*a, **k):
        if not ON[0]: return f(*a, **k)
        mx.eval([x for x in a if isinstance(x, mx.array)]); mx.synchronize(); time.sleep(0.001); g0 = lib.cb_gpu_us()
        r = f(*a, **k); mx.eval(r); mx.synchronize(); time.sleep(0.001)
        acc[tag] += lib.cb_gpu_us() - g0; cnt[tag] += 1; return r
    return w
for mn in os.environ.get("MODS", "ds_pf_attn,ds_pf_idx,ds_pf_mhc,compressor_pf,attn_prefill,pf_glue").split(","):
    mod = importlib.import_module("kyojin_mlx." + mn)
    for nm, f in list(vars(mod).items()):
        if isinstance(f, types.FunctionType) and f.__module__ == mod.__name__ and not nm.startswith("_") and nm not in ("install", "enable"):
            setattr(mod, nm, gfn(f, f"{mn}.{nm}"))
base = [t for r in json.load(open(os.path.expanduser("~/kyojin-mlx/results_mac/ref_ds41/ref.json")))['prompts'] for t in r['prompt_ids']]
ids = (base * (T // len(base) + 1))[:T]
for rep in range(3):
    ON[0] = rep == 2; acc.clear(); cnt.clear()
    cache = lm.make_cache(); t = time.perf_counter()
    for s in range(0, T, CH):
        mx.eval(lm(mx.array(ids[s:s + CH])[None], cache=cache).logits[:, -1:]) if False else mx.eval(lm.model(mx.array(ids[s:s + CH])[None], cache) if hasattr(lm, "model") else lm(mx.array(ids[s:s + CH])[None], cache=cache).logits[:, -1:])
    print(f"rep {rep}: {time.perf_counter() - t:.2f}s", flush=True)
    del cache; mx.clear_cache()
for k in sorted(acc):
    print(f"{k:12s} {acc[k]/1e6:6.2f} s GPU over {cnt[k]} calls ({acc[k]/max(cnt[k],1)/1e3:.1f} ms/call)")
print("attention+hc ~ block - ffn: %.2f s" % ((acc["block"] - acc["ffn"]) / 1e6))
