"""A/B of the prompt-scoped weight cache (pf5): ds2 slice, NCH chunks of T rows through lm(...) with a continuing cache, arms interleaved (cache off / on), 2 s idle before each run.
Reports per-chunk ms (min / p25 over rounds), per-layer ms of chunks 1 and 2.., bit equality of the last-chunk logits, peak memory.
usage: ab_wc.py PACK T NCH [rounds]"""
import sys, time, os, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, wcache as WC
pack, T, NCH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 6
COOL = float(os.environ.get("KYOJIN_BENCH_COOL", "2"))
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model
if os.environ.get('STUB_MOE','1')=='1':
    H = cfg['text_config']['hidden_size']; P._moe_rows = lambda moe, x2, ids2, w2, xsrc=None: mx.zeros((x2.shape[0], H), dtype=mx.float32)
lt = {}
for i, l in enumerate(lm.layers):
    cls = l.__class__; orig = cls.__call__
    def f(self, h, pre_mix, image_mask, cache, _i=i, _o=orig):
        mx.eval(h, pre_mix); time.sleep(COOL); t0 = time.perf_counter(); o = _o(self, h, pre_mix, image_mask, cache); mx.eval(o); lt[_i] = lt.get(_i, 0.0) + time.perf_counter() - t0; return o
    l.__class__ = type(cls.__name__, (cls,), {"__call__": f})
ids = np.random.default_rng(3).integers(1, 4000, T * NCH).tolist()
def run(on):
    os.environ["KYOJIN_PF_WCACHE_GB"] = "64" if on else "0"
    time.sleep(COOL); mx.clear_cache(); mx.reset_peak_memory()
    cache = lm.make_cache(); ch, lay = [], []
    with WC.prefill_scope(model, T, NCH):
        for c in range(NCH):
            lt.clear(); t0 = time.perf_counter(); lg = lm(mx.array(ids[c * T:(c + 1) * T])[None], cache=cache).logits; mx.eval(lg); ch.append(time.perf_counter() - t0); lay.append(dict(lt))
    st = dict(WC.last)
    return ch, lay, lg[:, -1].astype(mx.float32), mx.get_peak_memory() / 2 ** 30, st
o1 = run(False); o2 = run(True)
print("scope stats on:", o2[4], "| last-chunk logits bit-equal:", bool(mx.array_equal(o1[2], o2[2]).item()), flush=True)
res = {False: [], True: []}
for r in range(rounds):
    for a in ((False, True) if r % 2 == 0 else (True, False)):
        ch, lay, _, pk, _ = run(a); res[a].append((ch, lay, pk))
for a, nm in ((False, "OFF"), (True, "ON ")):
    R = res[a]
    for c in range(NCH):
        v = [x[0][c] * 1e3 for x in R]
        print(f"{nm} chunk{c+1} T={T}: min {min(v):7.1f} p25 {np.percentile(v, 25):7.1f} ms   layers " + " ".join(f"L{i}:{min(x[1][c][i] for x in R)*1e3:6.1f}" for i in R[0][1][c]), flush=True)
    print(f"{nm} sum chunks: min {min(sum(x[0]) for x in R)*1e3:7.1f} ms | peak {max(x[2] for x in R):.2f} GiB", flush=True)
