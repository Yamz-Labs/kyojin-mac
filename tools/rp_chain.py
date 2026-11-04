"""Step replay prototype v2: a teacher-forced chain of K consecutive decode steps recorded once (one ICB + constant arena per step, one shared table of private state
buffers), replayed from one-launch trigger ops with the state advancing from step to step. Each replayed step must equal the eager step bit for bit, on every pass.
usage: PYTHONPATH=.:../../vlm_pkg python tools/rp_chain.py models/ds2 [--len 136] [--k 6] [--passes 20]"""
import argparse, ctypes, os, resource, statistics, sys, time
here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
lib = ctypes.CDLL(os.path.join(here, "..", "replay", "build", "librp.dylib"))
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--len", type=int, default=136); ap.add_argument("--k", type=int, default=6); ap.add_argument("--passes", type=int, default=20)
a = ap.parse_args()
C = ctypes
lib.rp_build_chain.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]; lib.rp_build_chain.restype = C.c_int
lib.rp_set_pins.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
for f in ("rp_commits", "rp_dbg", "rp_nrefresh", "rp_clone_bytes", "rp_nclones", "rp_bound_bytes", "rp_nbound", "rp_replays", "rp_chain_pos"): getattr(lib, f).restype = C.c_long
lib.rp_dbg.argtypes = [C.c_int]
lib.rp_gpu_times.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
lib.rp_install(1)
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R

def addr(x): return ctypes.addressof(ctypes.c_char.from_buffer(memoryview(x)))
def pins_of(root):
    seen, out = set(), []
    def walk(o, depth=0):
        if id(o) in seen or depth > 12: return
        seen.add(id(o))
        if isinstance(o, mx.array):
            mx.eval(o)
            if o.nbytes: out.append((addr(o), o.nbytes))
        elif isinstance(o, (list, tuple)):
            for v in o: walk(v, depth + 1)
        elif isinstance(o, dict):
            for v in o.values(): walk(v, depth + 1)
        elif hasattr(o, "__dict__") and not isinstance(o, type):
            for v in vars(o).values(): walk(v, depth + 1)
    walk(root); return out
def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
lm = model.language_model; V = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(0)
ids = rng.integers(16, V - 16, a.len).tolist()
cache = lm.make_cache()
lg = lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32); mx.eval(lg)
tok = mx.argmax(lg, axis=-1); mx.eval(tok)
for _ in range(4):
    out = R.step(model, cache, tok); tok = mx.argmax(out, axis=-1); mx.eval(tok)
mx.synchronize()
K = a.k; tks = rng.integers(16, V - 16, K).tolist()
# reference: K eager steps on a fork, teacher-forced
cr = R.fork(cache); refs = []
for t in tks:
    o = R.step(model, cr, mx.array([t])); mx.eval(o); refs.append(np.array(o))
pk0 = mx.get_peak_memory() / 1e6; mx.reset_peak_memory()
print(f'peak Metal memory: model load + prefill + warm + K eager reference steps {pk0:.0f} MB; active now {mx.get_active_memory() / 1e6:.0f} MB, cache {mx.get_cache_memory() / 1e6:.0f} MB')
# record: the same K steps on another fork, each under its own recording slot; everything bound is retained so no buffer is recycled between the steps
pins = pins_of(model); pp = (C.c_size_t * len(pins))(*[p for p, _ in pins]); pl = (C.c_size_t * len(pins))(*[l for _, l in pins]); lib.rp_set_pins(pp, pl, len(pins))
print(f"pinned arrays {len(pins)}, {sum(n for _, n in pins) / 1e6:.0f} MB; rss {rss_mb():.0f} MB")
lib.rp_snap_reset()
c2 = R.fork(cache); outs = []; toks = []; ncmd = []
for i, t in enumerate(tks):
    ti = mx.array([t]); mx.eval(ti); toks.append(ti); mx.synchronize()
    lib.rp_rec_begin(i)
    o = R.step(model, c2, ti); mx.eval(o); mx.synchronize()
    ncmd.append(lib.rp_rec_end()); outs.append(o)
    if os.environ.get('RP_DUMP'): os.makedirs(os.environ['RP_DUMP'], exist_ok=True); lib.rp_dump(i, f"{os.environ['RP_DUMP']}/step{i}.txt".encode())
    assert np.array_equal(np.array(o), refs[i]), f"recorded step {i} differs from the reference"
print("recorded dispatches per step", ncmd, "| rss", f"{rss_mb():.0f} MB")
pk1 = mx.get_peak_memory() / 1e6; mx.reset_peak_memory()
print(f'peak Metal memory while recording K steps (everything retained): {pk1:.0f} MB')
trig = mx.fast.metal_kernel(name="kyojin_replay_trigger", input_names=["x"], output_names=["out"], source="out[thread_position_in_grid.x] = 0.0f;")
shape = outs[0].shape
def call(t): return trig(inputs=[t.astype(mx.int32)], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[shape], output_dtypes=[mx.float32])[0]
dummy = mx.array([0]); mx.eval(call(dummy)); mx.synchronize()
slots = (C.c_int * K)(*range(K)); optr = (C.c_size_t * K)(*[addr(o) for o in outs])
t0 = time.perf_counter(); h = lib.rp_build_chain(K, slots, optr, outs[0].nbytes, 1); tb = time.perf_counter() - t0
assert h >= 0, h
print(f"chain {h}: built in {tb * 1e3:.0f} ms; private copies {lib.rp_nclones(h)} ({lib.rp_clone_bytes(h) / 1e6:.1f} MB), bound in place {lib.rp_nbound(h)} ({lib.rp_bound_bytes(h) / 1e6:.0f} MB); per-step input refreshes {lib.rp_nrefresh(h)}; rss {rss_mb():.0f} MB")
del outs, toks
lib.rp_activate(h)
bad = 0; fails = []
for p in range(a.passes):
    lib.rp_chain_rewind(h)
    for i in range(K):
        r0 = lib.rp_replays(); o = call(dummy); mx.eval(o); same = np.array_equal(np.array(o), refs[i]); bad += not same
        if lib.rp_replays() != r0 + 1: print('   trigger did not run the replay: pass', p, 'step', i, 'dbg', [lib.rp_dbg(k) for k in range(5)])
        if not same:
            d = np.array(o); fails.append((p, i, float(np.abs(d - refs[i]).max()), int((d == 0).sum()), int(np.isnan(d).sum())))
print(f"chain passes {a.passes} x {K} steps: mismatching steps {bad} of {a.passes * K} | rss {rss_mb():.0f} MB")
print("   failures (pass, step, maxabs, zeros, nan):", fails[:12])
print("   failing steps histogram", {i: sum(1 for f in fails if f[1] == i) for i in range(K)}, "| failing passes", sorted({f[0] for f in fails})[:20])
# eager steps interleaved with the chain must not disturb it (private state), and the chain must not disturb the live cache
lib.rp_chain_rewind(h); bad2 = 0
for i in range(K):
    e = R.step(model, R.fork(cache), mx.array([tks[0]])); mx.eval(e); bad2 += not np.array_equal(np.array(e), refs[0])
    o = call(dummy); mx.eval(o); bad2 += not np.array_equal(np.array(o), refs[i])
print("interleaved with eager steps on the live cache: bad", bad2)
if os.environ.get('RP_NOTIME'): sys.exit(0)
# timing: CPU to issue one step (graph build + async_eval) and GPU per command buffer; eager on a fresh fork vs replay, interleaved
lib.rp_time(1); buf = (C.c_double * 64)(); wb = (C.c_double * 64)()
ncm = []
def timed(fn):
    lib.rp_gpu_times(buf, wb, 64); c0 = lib.rp_commits(); t0 = time.perf_counter(); o = fn(); t1 = time.perf_counter(); c1 = lib.rp_commits(); mx.async_eval(o); t2 = time.perf_counter(); c2 = lib.rp_commits(); mx.eval(o); time.sleep(0.003); k = lib.rp_gpu_times(buf, wb, 64)
    ncm.append((c1 - c0, c2 - c1))
    return t1 - t0, t2 - t1, sum(buf[j] for j in range(k)), k
et, rt = [], []
for r in range(a.passes * 3):
    lib.rp_chain_rewind(h)
    for i in range(K):
        mt = mx.array([tks[i]]); mx.eval(mt); f = R.fork(cache); mx.eval(*[x for x in [mt]])
        et.append(timed(lambda: R.step(model, f, mt)))
        rt.append(timed(lambda: call(dummy)))
def rep(n, v):
    tot = sorted(x[0] + x[1] for x in v); g = sorted(x[2] for x in v if x[2] > 0); q = lambda a, f: a[int(len(a) * f)]
    print(f"{n:7s}: CPU us per step (build + async_eval)  min {1e6 * tot[0]:7.1f}  p25 {1e6 * q(tot, .25):7.1f}  med {1e6 * q(tot, .5):7.1f} | GPU us per step (cmd buffers)  min {g[0]:7.1f}  p25 {q(g, .25):7.1f}  med {q(g, .5):7.1f}  ({statistics.median(x[3] for x in v):.0f} buffers)")
    return tot[0], q(tot, .25), g[0], q(g, .25)
e = rep("eager", et); r = rep("replay", rt)
nl = int(os.environ.get("RP_LAYERS", "2"))
print(f"per layer ({nl} layers in the slice, whole step / {nl}): CPU eager min {1e6 * e[0] / nl:.0f} p25 {1e6 * e[1] / nl:.0f} us -> replay min {1e6 * r[0] / nl:.1f} p25 {1e6 * r[1] / nl:.1f} us | GPU min {e[2]:.0f} -> {r[2]:.0f} us ({100 * (r[2] / e[2] - 1):+.1f} %), p25 {e[3]:.0f} -> {r[3]:.0f} us ({100 * (r[3] / e[3] - 1):+.1f} %)")
print("command buffers committed during the graph build / during async_eval, eager (median):", statistics.median(x[0] for x in ncm[0::2]), statistics.median(x[1] for x in ncm[0::2]), "| replay:", statistics.median(x[0] for x in ncm[1::2]), statistics.median(x[1] for x in ncm[1::2]))
print("peak Metal memory after the chain was built (replay + eager timing loops): %.0f MB" % (mx.get_peak_memory() / 1e6))
print("replays", lib.rp_replays(), "| peak rss", f"{rss_mb():.0f} MB")
