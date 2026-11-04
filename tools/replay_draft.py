"""Gate + timing of the replayed draft head: the compiled draft forward of kyojin_mlx.dspark through a replay region (KYOJIN_REPLAY=1; the ring append stays the plain compiled function) against the plain
compiled function (a second mx.compile of the same methods), same inputs, bitwise: tokens, confidences and every stage ring, over many rounds with feed sizes 1..block.
Timing, interleaved per round: CPU of the draft call (graph build + async_eval) and wall (until the tokens are on the host), replay vs compiled.
usage: KYOJIN_REPLAY=1 python tools/replay_draft.py PACK [--rounds 40] [--len 135]"""
import argparse, ctypes, os, statistics, sys, time
os.environ.setdefault("KYOJIN_REPLAY", "1")
if os.environ.get("M10_LIGHT"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import light                                              # custom-kernel grids divided by F: the GPU is never the limit (outputs are garbage: KYOJIN_REPLAY_TRIAL=0 CHECK=0 AUDIT=0)
import mlx.core as mx
import numpy as np
from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S, replay as RP

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rounds", type=int, default=40); ap.add_argument("--len", type=int, default=135); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
tc = cfg["text_config"]
dr = D.build_drafter(model, a.pack, spec, cfg, conf_threshold=0.0)
S.install(model, spec, tap_layers=set(dr.tap_ids))
assert isinstance(dr._cfwd, RP.Region), "KYOJIN_REPLAY=1 expected"
ref_fwd, ref_app = mx.compile(dr._forward), mx.compile(dr._append_rows)
rng = np.random.default_rng(a.seed)
lib = RP._load(); lib.rp_time.argtypes = [ctypes.c_int]; lib.rp_gpu_times.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]; lib.rp_time(1)
GB = (ctypes.c_double * 64)(); GW = (ctypes.c_double * 64)()
def gpu():
    time.sleep(0.003); k = lib.rp_gpu_times(GB, GW, 64); return sum(GB[j] for j in range(k))
H = tc["hidden_size"]; V = tc["vocab_size"]; W = dr.window; L = dr.block
prompt = rng.integers(16, V - 16, a.len).tolist()
cache, lg, taps, first = S.prefill(model, prompt, dr.tap_ids)
dr.append([taps[l] for l in dr.tap_ids], first)
ref_rows = list(dr.ctx.rows)
mx.eval(*dr.ctx.rows)
bad = 0; cmp_n = 0
tc_rep, tc_ref, wl_rep, wl_ref, g_rep, g_ref = [], [], [], [], [], []
p0 = a.len
def cmp(x, y, what):
    global bad, cmp_n
    cmp_n += 1
    if not RP.same(x, y):
        bad += 1; print("DIFFERS:", what, flush=True)
for r in range(a.rounds):
    pending = int(rng.integers(16, V - 16)); ids = mx.array([[pending] + [dr.noise] * (L - 1)]); pp = mx.array(p0, dtype=mx.int32)
    # draft: replay (through the drafter) vs the compiled function
    rows = dr.ctx.rows; mx.eval(*rows, *ref_rows)
    gpu(); t0 = time.perf_counter(); toks, conf = dr._cfwd(ids, pp, *rows); mx.async_eval(toks, conf); t1 = time.perf_counter(); mx.eval(toks, conf); toks.tolist(); t2 = time.perf_counter(); gr = gpu()
    t3 = time.perf_counter(); rt, rc = ref_fwd(ids, pp, *ref_rows); mx.async_eval(rt, rc); t4 = time.perf_counter(); mx.eval(rt, rc); rt.tolist(); t5 = time.perf_counter(); gf = gpu()
    cmp(toks, rt, f"round {r} tokens"); cmp(conf, rc, f"round {r} conf")
    if r >= 4: tc_rep.append(t1 - t0); wl_rep.append(t2 - t0); tc_ref.append(t4 - t3); wl_ref.append(t5 - t3); g_rep.append(gr * 1e-6); g_ref.append(gf * 1e-6)
    # append a feed of 1..L rows
    n = int(rng.integers(1, L + 1)); tp = [mx.array(rng.standard_normal((n, H)).astype(np.float16)) for _ in dr.tap_ids]; st = mx.array(p0 + 1, dtype=mx.int32); mx.eval(*tp)
    new = dr._capp(st, *tp, *dr.ctx.rows); refn = ref_app(st, *tp, *ref_rows); mx.eval(*new, *refn)
    for i, (x, y) in enumerate(zip(new, refn)): cmp(x, y, f"round {r} append n={n} stage {i}")
    dr.ctx.rows, ref_rows = list(new), list(refn); p0 += n
print(f"compared {cmp_n} arrays, mismatches {bad} | replay stats {RP.STATS}")
us = lambda v: f"min {1e6*min(v):.0f} p25 {1e6*sorted(v)[len(v)//4]:.0f} med {1e6*statistics.median(v):.0f}"
print(f"draft CPU us (build + async_eval): replay {us(tc_rep)} | compiled {us(tc_ref)}")
print(f"draft wall us (to tokens on host)  : replay {us(wl_rep)} | compiled {us(wl_ref)}")
print(f"draft GPU us (command buffers): replay {us(g_rep)} | compiled {us(g_ref)}")
