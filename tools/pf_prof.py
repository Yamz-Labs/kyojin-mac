import sys, time, os, collections
import numpy as np, mlx.core as mx
from kyojin_mlx import headlast as HL, patch as P, rows as R, wcache as WC
import kyojin_mlx.moe_prefill as MPF
pack, T, chunk, reps = os.path.expanduser('~/ds41/model'), int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
MG = []
_oi = MPF.MoePrefill.__init__
def _ni(self, *a, **k):
    _oi(self, *a, **k); MG.append(self)
    if os.environ.get('MPROF'): self.prof = {}
MPF.MoePrefill.__init__ = _ni
HOT = collections.defaultdict(float)
def _tw(mod, name, label):
    o = getattr(mod, name)
    def w(*a, **k):
        t = time.perf_counter(); r = o(*a, **k); mx.eval(r); HOT[label] += time.perf_counter() - t; return r
    return w
if os.environ.get('MPROF'):
    MPF.decode_group = _tw(MPF, 'decode_group', 'hot_decode')
    MPF.MoePrefill._mm = (lambda o: (lambda self, *a, **k: (lambda t, r: (mx.eval(r), HOT.__setitem__('hot_mm', HOT['hot_mm'] + time.perf_counter() - t), r)[2])(time.perf_counter(), o(self, *a, **k))))(MPF.MoePrefill._mm)
    _op = MPF.MoePrefill._plan
    MPF.MoePrefill._plan = lambda self, ids_np: (lambda t, r: (HOT.__setitem__('plan', HOT['plan'] + time.perf_counter() - t), r)[1])(time.perf_counter(), _op(self, ids_np))
model, spec, cfg = P.build_model(pack, {}); P.patch_model(model, pack, spec, cfg)
try:
    from kyojin_mlx import engram_rows as ER, engram_mlx as EM
    es = ER.EngramSpec.from_config(cfg)
    if es is not None:
        tables = ER.EngramTables(pack, es); EM.install_parallel_gather(model, tables)
except ImportError: pass
R.enable(model, spec)
lm = model.language_model
TT = collections.defaultdict(float); CN = collections.Counter()
blk = lm.layers[3]
def wrap(cls, name):
    o = cls.__call__
    def w(self, *a, **k):
        mx.eval(a[0] if a and hasattr(a[0], 'shape') else [])
        t = time.perf_counter(); r = o(self, *a, **k); mx.eval(r); TT[name] += time.perf_counter() - t; CN[name] += 1; return r
    cls.__call__ = w
if os.environ.get('PROF'):
    wrap(type(blk.ffn), 'ffn')
    for nm in ('gate','shared_experts','switch_mlp'):
        c = getattr(blk.ffn, nm, None)
        if c is not None: print('wrap', nm, type(c)); wrap(type(c), nm)
import json
if os.environ.get('REF'):
    base = [t for r in json.load(open(os.environ['REF']))['prompts'] for t in r['prompt_ids']]
    ids = (base * (T // len(base) + 1))[:T]
else:
    ids = np.random.default_rng(0).integers(16, 128000, T).tolist()
for r in range(reps):
    mx.reset_peak_memory()
    for m in MG:
        if os.environ.get('MPROF'): m.prof = {}
    cache = lm.make_cache(); t0 = time.perf_counter(); steps = -(-T // chunk)
    with WC.prefill_scope(model, chunk, steps):
        for s in range(0, T, chunk):
            with HL.last_row():
                lg = lm(mx.array(ids[s:s+chunk])[None], cache=cache).logits[:, -1].astype(mx.float32)
            mx.eval(lg)
    dt = time.perf_counter() - t0
    print(f'rep {r}: T={T} chunk={chunk} {dt:.2f}s {T/dt:.0f} tok/s peak {mx.get_peak_memory()/2**30:.1f} GiB', flush=True)
    agg = {}
    for m in MG:
        for k, v in (m.prof or {}).items(): agg[k] = agg.get(k, 0) + v
    print('MPROF', {k: round(v, 2) for k, v in agg.items()}, len(MG), MG[5].last if len(MG) > 5 else None)
    print('HOT', {k: round(v, 2) for k, v in HOT.items()}); HOT.clear()
    print({k: (round(v, 2), CN[k]) for k, v in TT.items()}); TT.clear(); CN.clear()
    del cache; mx.clear_cache()
