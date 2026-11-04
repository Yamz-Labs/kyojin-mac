"""Gate + timing of the replayed multi-session draft (the serving path): Drafter.draft_many through a replay region (append_many stays eager) (KYOJIN_REPLAY=1) against the same
functions run eagerly (no region), bitwise: proposals, confidences and every ring, over rounds of 1..3 sessions with random feed sizes; and, with OLD=<file>, against the
pre-replay implementation of dspark.py (git show c6e79a8:kyojin_mlx/dspark.py > kyojin_mlx/dspark_old.py, untracked).
usage: KYOJIN_REPLAY=1 python tools/replay_rows.py PACK [--rounds 60] [--sessions 3]"""
import argparse, ctypes, importlib, os, statistics, sys, time
os.environ.setdefault("KYOJIN_REPLAY", "1")  # KYOJIN_REPLAY=0: the new code against the old one with regions off
if os.environ.get("M10_LIGHT"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import light                                              # custom-kernel grids divided by F: the GPU is never the limit (outputs are garbage: KYOJIN_REPLAY_TRIAL=0 CHECK=0 AUDIT=0)
import mlx.core as mx
import numpy as np
from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S, replay as RP

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rounds", type=int, default=60); ap.add_argument("--sessions", type=int, default=3); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack)
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
tc = cfg["text_config"]
dr = D.build_drafter(model, a.pack, spec, cfg, conf_threshold=0.0)
R.enable_draft(dr.lm, spec)
S.install(model, spec, tap_layers=set(dr.tap_ids))
old = None
if os.path.exists(os.path.join(os.path.dirname(D.__file__), "dspark_old.py")):
    O = importlib.import_module("kyojin_mlx.dspark_old")
    old = O.build_drafter(model, a.pack, spec, cfg, conf_threshold=0.0)
    R.enable_draft(old.lm, spec)
rng = np.random.default_rng(a.seed)
if os.environ.get('WARM'):
    t_w = time.perf_counter(); nw = dr.warm_replay(); print(f'warm_replay: {nw} calls, {time.perf_counter() - t_w:.1f} s, STATS {RP.STATS}', flush=True)
H = tc["hidden_size"]; V = tc["vocab_size"]; L = dr.block; W = dr.window; ns = dr.n_stages; nt = len(dr.tap_ids)
lib = RP._load(); lib.rp_time.argtypes = [ctypes.c_int]; lib.rp_gpu_times.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]; lib.rp_time(1)
GB = (ctypes.c_double * 64)(); GW = (ctypes.c_double * 64)()
def gpu():
    time.sleep(0.003); k = lib.rp_gpu_times(GB, GW, 64); return sum(GB[j] for j in range(k))
NS = a.sessions
NS_T = int(os.environ.get('TIME_SESSIONS', a.sessions))      # time only rounds with this many sessions
def make_ctx(n):
    cache, lg, taps, first = S.prefill(model, rng.integers(16, V - 16, n).tolist(), dr.tap_ids)
    c = dr.new_ctx(); dr.ctx = c; dr.append([taps[l] for l in dr.tap_ids], first); mx.eval(*c.rows); return c, n
base = [make_ctx(n) for n in (141, 150, 133, 160)[:NS]]
def clone(c):
    n = dr.new_ctx(); n.rows = list(c.rows); return n
A = [clone(c) for c, _ in base]; B = [clone(c) for c, _ in base]; Cc = [clone(c) for c, _ in base]
pos = [n for _, n in base]
bad = cmp_n = 0
def cmp(x, y, what):
    global bad, cmp_n
    cmp_n += 1
    if not RP.same(x, y): bad += 1; print("DIFFERS:", what, flush=True)
tA, tB, cA, cB, gA, gB, apA, apB = [], [], [], [], [], [], [], []
def timed(fn):
    """(result, CPU seconds = graph build + async_eval encode, wall seconds) of fn(), which ends with one mx.eval"""
    mark = {}; real = mx.eval
    def hook(*xs):
        mark['b'] = time.perf_counter(); mx.async_eval(*xs); mark['e'] = time.perf_counter(); real(*xs)
    mx.eval = hook
    t = time.perf_counter()
    try: r = fn()
    finally: mx.eval = real
    return r, (mark['b'] - t) + (mark['e'] - mark['b']), time.perf_counter() - t
for r in range(a.rounds):
    k = int(rng.integers(1, NS + 1)); sel = sorted(rng.choice(NS, k, replace=False).tolist())
    pend = [int(rng.integers(16, V - 16)) for _ in sel]; p0 = [pos[i] for i in sel]; depth = [int(rng.integers(1, L)) for _ in sel]
    mx.eval(*[x for i in sel for x in A[i].rows], *[x for i in sel for x in B[i].rows])
    gpu(); ra, ca, wa = timed(lambda: dr.draft_many([A[i] for i in sel], pend, p0, depth)); ga = gpu()
    def ref_call():
        toks, conf, _ = dr._rows_core([B[i] for i in sel], pend, p0); mx.eval(toks, conf); return toks.tolist(), conf.tolist()
    (rt, rc), cb, wb_ = timed(ref_call); gb = gpu()
    if rt != dr.last_raw[0] or rc != dr.last_raw[1]: bad += 1; print("DIFFERS: draft_many round", r, flush=True)
    cmp_n += 1
    if old is not None:
        rold = old.draft_many([Cc[i] for i in sel], pend, p0, depth); cmp_n += 1
        if old.last_raw != dr.last_raw: bad += 1; print("DIFFERS vs old draft_many round", r, flush=True)
    if r >= 4 and len(sel) == NS_T: tA.append(wa); tB.append(wb_); cA.append(ca); cB.append(cb); gA.append(ga * 1e-6); gB.append(gb * 1e-6)
    # feed: random counts per selected session
    items_a, items_b, items_c = [], [], []
    cnt = [int(rng.integers(1, L + 1)) for _ in sel]
    for j, i in enumerate(sel):
        tp = [mx.array(rng.standard_normal((cnt[j], H)).astype(np.float16)) for _ in dr.tap_ids]; mx.eval(*tp)
        items_a.append((A[i], tp, pos[i])); items_b.append((B[i], tp, pos[i])); items_c.append((Cc[i], tp, pos[i]))
    dr.append_many(items_a)
    dr.append_many(items_b)                                    # append_many is not a region (its CPU is 0.15-0.9 ms: a trigger op with a dozen inputs costs the same); the rings feed the next draft_many
    if old is not None: old.append_many(items_c)
    if os.environ.get('DBG'): print('round', r, 'sel', sel, 'cnt', cnt, 'draft', [len(x) for x in ra], flush=True)
    mx.eval(*[x for i in sel for x in A[i].rows], *[x for i in sel for x in B[i].rows])
    for i in sel:
        for s in range(ns):
            cmp(A[i].rows[s], B[i].rows[s], f"round {r} ring session {i} stage {s} vs eager")
            if old is not None: cmp(A[i].rows[s], Cc[i].rows[s], f"round {r} ring session {i} stage {s} vs old code")
        pos[i] += cnt[sel.index(i)]
print(f"compared {cmp_n} items, mismatches {bad} | replay stats {RP.STATS}")
us = lambda v: f"min {1e6*min(v):.0f} p25 {1e6*sorted(v)[len(v)//4]:.0f} med {1e6*statistics.median(v):.0f}"
print(f"draft_many with {NS_T} sessions ({len(tA)} timed rounds)")
print(f"  CPU us (build + async_eval): replay {us(cA)} | eager {us(cB)}")
print(f"  wall us (to tokens on host): replay {us(tA)} | eager {us(tB)}")
print(f"  GPU us (command buffers)   : replay {us(gA)} | eager {us(gB)}")
for n in ("_rows_region",):
    if not isinstance(getattr(dr, n), RP.Region): continue
    st = getattr(dr, n).stats(); print(n, {k[:60] if False else i: (v['state'], v['replays']) for i, (k, v) in enumerate(st.items())})
