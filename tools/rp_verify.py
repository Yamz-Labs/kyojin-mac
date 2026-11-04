"""Verify-pass replay tape: the served verify forward (rows.verify over a batched cache, N sessions x (D+1) rows) of K different token lists recorded once from the
same cache state (a verify leaves the cache untouched until commit), replayed from the rp trigger op. Gate: every replayed logits array equals the eager pass bit for bit.
Timing: CPU per verify (graph build + async_eval) eager vs replay, ds2m slice; extrapolate by the replay constant.
usage: PYTHONPATH=.:../../vlm_pkg python tools/rp_verify.py models/ds2m [--users 1] [--depth 3] [--k 3] [--passes 10]"""
import argparse, ctypes, os, statistics, sys, time
here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
lib = ctypes.CDLL(os.path.join(here, "..", "replay", "build", "librp.dylib"))
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--users", type=int, default=1); ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--k", type=int, default=3); ap.add_argument("--passes", type=int, default=10); ap.add_argument("--len", type=int, default=140)
a = ap.parse_args()
C = ctypes
lib.rp_build_chain.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]; lib.rp_build_chain.restype = C.c_int
lib.rp_set_pins.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
for f in ("rp_commits", "rp_nrefresh", "rp_clone_bytes", "rp_nclones", "rp_bound_bytes", "rp_nbound", "rp_replays"): getattr(lib, f).restype = C.c_long
lib.rp_gpu_times.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
lib.rp_install(1)
import numpy as np
import mlx.core as mx
from kyojin_mlx.serve.backends import load_backend

def addr(x): return ctypes.addressof(ctypes.c_char.from_buffer(memoryview(x)))
def pins_of(*roots):
    seen, out = set(), []
    def walk(o, depth=0):
        if id(o) in seen or depth > 14: return
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
    for r in roots: walk(r)
    return out

be, cfg = load_backend(a.pack, mtp=True, ced=False, log=lambda *x: None)
V = cfg.get("text_config", cfg)["vocab_size"]; N, D = a.users, a.depth
rng = np.random.default_rng(7); base = rng.integers(16, V - 16, 61).tolist()
caches = []
for u in range(N):
    c = be.make_cache(); d = be.new_draft(); be.prefill_chunk(c, [16 + u] + (base * 4)[:a.len - 1], draft=d); caches.append(c)
bc = be.batch(caches)
tl = [[[int(t) for t in rng.integers(16, V - 16, D + 1)] for _ in range(N)] for _ in range(a.k)]
def vpass(i):
    ps = be.spec_verify_bc(bc, tl[i]); return ps
def discard(ps):
    st = ps.p.entry if hasattr(ps, "p") else None
    for c in bc:
        if hasattr(c, "_counts"): c._counts = None; c.pend = {}
# eager reference (state untouched: no commit)
refs = []
for i in range(a.k):
    ps = vpass(i); mx.eval(ps.logits); refs.append(np.array(ps.logits)); discard(ps)
mx.synchronize()
pins = pins_of(be.model, bc); pp = (C.c_size_t * len(pins))(*[p for p, _ in pins]); pl = (C.c_size_t * len(pins))(*[l for _, l in pins]); lib.rp_set_pins(pp, pl, len(pins))
print(f"pins {len(pins)}, {sum(n for _, n in pins) / 1e6:.0f} MB")
lib.rp_snap_reset(); outs = []; ncmd = []
for i in range(a.k):
    mx.synchronize(); mx.clear_cache(); lib.rp_rec_begin(i)
    ps = vpass(i); o = ps.logits; mx.eval(o); mx.synchronize(); ncmd.append(lib.rp_rec_end()); outs.append(o); discard(ps)
    assert np.array_equal(np.array(o), refs[i]), f"recorded pass {i} differs"
print("recorded dispatches per pass", ncmd)
trig = mx.fast.metal_kernel(name="kyojin_replay_trigger", input_names=["x"], output_names=["out"], source="out[thread_position_in_grid.x] = 0.0f;")
shape = outs[0].shape
def call(t): return trig(inputs=[t.astype(mx.int32)], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[shape], output_dtypes=[mx.float32])[0]
dummy = mx.array([0]); mx.eval(call(dummy)); mx.synchronize()
K = a.k; slots = (C.c_int * K)(*range(K)); optr = (C.c_size_t * K)(*[addr(o) for o in outs])
h = lib.rp_build_chain(K, slots, optr, outs[0].nbytes, 1); assert h >= 0, h
print(f"chain {h}: private copies {lib.rp_nclones(h)} ({lib.rp_clone_bytes(h) / 1e6:.1f} MB), bound in place {lib.rp_nbound(h)} ({lib.rp_bound_bytes(h) / 1e6:.0f} MB), per-pass refreshes {lib.rp_nrefresh(h)}")
del outs
lib.rp_activate(h); bad = 0
for p in range(a.passes):
    lib.rp_chain_rewind(h)
    for i in range(K):
        o = call(dummy); mx.eval(o); bad += not np.array_equal(np.array(o), refs[i])
print(f"replay passes {a.passes} x {K}: mismatching {bad} of {a.passes * K}")
lib.rp_time(1); buf = (C.c_double * 64)(); wb = (C.c_double * 64)()
def timed(fn):
    lib.rp_gpu_times(buf, wb, 64); t0 = time.perf_counter(); o = fn(); t1 = time.perf_counter(); mx.async_eval(o); t2 = time.perf_counter(); mx.eval(o); time.sleep(0.003); k = lib.rp_gpu_times(buf, wb, 64)
    return t1 - t0, t2 - t1, sum(buf[j] for j in range(k))
et, rt = [], []
for r in range(a.passes * 3):
    lib.rp_chain_rewind(h)
    for i in range(K):
        def eager():
            ps = vpass(i); return ps.logits
        et.append(timed(eager)); discard(None)
        rt.append(timed(lambda: call(dummy)))
def rep(n, v):
    cpu = sorted(x[0] + x[1] for x in v); g = sorted(x[2] for x in v); b = sorted(x[0] for x in v); e = sorted(x[1] for x in v)
    print(f"{n:7s}: CPU us min {1e6 * cpu[0]:7.1f} med {1e6 * cpu[len(cpu) // 2]:7.1f} (build med {1e6 * b[len(b) // 2]:.0f} encode med {1e6 * e[len(e) // 2]:.0f}) | GPU us min {g[0]:.0f} med {g[len(g) // 2]:.0f}")
rep("eager", et); rep("replay", rt)
