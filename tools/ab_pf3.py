"""A/B of the pf3 prompt glue per layer (ds2 slice, routed experts stubbed, 3 s idle before each layer, arms interleaved, rotated, min/p25 over rounds) + bit equality of the logits.
usage: ab_pf3.py PACK T [rounds]; arms = (name, {module: {attr: value}}) below"""
import sys, time, os, importlib, subprocess, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P
pack, T = sys.argv[1], int(sys.argv[2]); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
COOL = float(os.environ.get("KYOJIN_BENCH_COOL", "3"))
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
H = cfg["text_config"]["hidden_size"]
P._moe_rows = lambda moe, x2, ids2, w2, xsrc=None: mx.zeros((x2.shape[0], H), dtype=mx.float32)
lm = model.language_model
SW = [s for s in os.environ.get("AB_SWITCHES", "kyojin_mlx.pf_glue:ENABLED").split(",")]       # module:attr, bool; arm NEW = all True, OLD = all False
times = {}
for i, l in enumerate(lm.layers):
    cls = l.__class__; orig = cls.__call__
    def f(self, h, pre_mix, image_mask, cache, _i=i, _o=orig):
        mx.eval(h, pre_mix); time.sleep(COOL); t0 = time.perf_counter(); o = _o(self, h, pre_mix, image_mask, cache); mx.eval(o); times[_i] = time.perf_counter() - t0; return o
    l.__class__ = type(cls.__name__, (cls,), {"__call__": f})
ME = os.getpid()
def others():
    out = subprocess.run(["ps", "-axo", "pid,pcpu,command"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if ("python" in l or "pytest" in l) and str(ME) not in l.split()[:1] and "lockf" not in l and "ps -axo" not in l and float(l.split()[1].replace(",", ".")) > 3.0]
def quiet(wait=90):
    t0 = time.time()
    while others():
        if time.time() - t0 > wait: return False
        time.sleep(1.5)
    return True
def arm(on):
    for s in SW:
        m, a = s.split(":"); setattr(importlib.import_module(m), a, on)
def run(ids):
    q0 = quiet(); cache = lm.make_cache(); times.clear(); lg = lm(ids, cache=cache).logits; mx.eval(lg); clean = q0 and not others(); return dict(times, clean=clean), lg
ids = mx.array([np.random.default_rng(3).integers(1, 4000, T).tolist()])
out = {}
for a in (True, False): arm(a); _, out[a] = run(ids)
print("logits bit-equal NEW vs OLD:", bool(mx.array_equal(out[True], out[False]).item()), "max abs diff", float(mx.abs(out[True].astype(mx.float32) - out[False].astype(mx.float32)).max().item()), flush=True)
res = {True: [], False: []}
for r in range(rounds):
    for a in ((True, False) if r % 2 == 0 else (False, True)):
        arm(a); res[a].append(run(ids)[0])
for a, nm in ((False, "OLD"), (True, "NEW")):
    cl = [x for x in res[a] if x["clean"]]
    print(f"T={T} {nm}: all-runs " + "  ".join(f"L{i}: min {min(x[i] for x in res[a])*1e3:7.1f} p25 {np.percentile([x[i] for x in res[a]], 25)*1e3:7.1f}" for i in res[a][0] if i != "clean") + f" | clean runs {len(cl)}/{len(res[a])}: " + ("  ".join(f"L{i}: min {min(x[i] for x in cl)*1e3:7.1f} med {np.median([x[i] for x in cl])*1e3:7.1f}" for i in cl[0] if i != "clean") if cl else "none"), flush=True)
