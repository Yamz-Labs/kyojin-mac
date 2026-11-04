"""Parametrised region test (T1/T2): the served verify forward (rows.verify over the batched cache, state NOT committed, tokens fixed) recorded at M consecutive positions p, fitted
(periodic-affine per field), built as one region, replayed at positions inside AND beyond the samples; every replayed logits array must equal the eager pass at that position bit for bit.
usage: PYTHONPATH=.:../../vlm_pkg python tools/rp_param_test.py models/ds2m [--m 8] [--extra 24] [--depth 3]"""
import argparse, ctypes, os, time
here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
lib = ctypes.CDLL(os.path.join(here, "..", "replay", "build", "librp.dylib"))
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--m", type=int, default=8); ap.add_argument("--extra", type=int, default=24); ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--len", type=int, default=140); ap.add_argument("--maxl", type=int, default=8)
a = ap.parse_args()
C = ctypes
lib.rp_region_new.restype = C.c_int
lib.rp_region_build.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_int]
lib.rp_region_fit.argtypes = [C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_int, C.c_int]
lib.rp_region_nfields.restype = C.c_long
lib.rp_install(1)
import numpy as np
import mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import dsv4_core as DC

def addr(x): return ctypes.addressof(ctypes.c_char.from_buffer(memoryview(x)))
be, cfg = load_backend(a.pack, mtp=True, ced=False, log=lambda *x: None)
V = cfg.get("text_config", cfg)["vocab_size"]; D = a.depth
rng = np.random.default_rng(7); base = rng.integers(16, V - 16, 61).tolist()
c = be.make_cache(); d = be.new_draft(); be.prefill_chunk(c, [16] + (base * 4)[:a.len - 1], draft=d); bc = be.batch([c])
toks = [[int(t) for t in rng.integers(16, V - 16, D + 1)]]
st = DC.stack_rows(bc[0]); P0 = st.rows[0].offset
def discard():
    st._counts = None; st.pend = {}
def fwd(p):
    st.rows[0].offset = p
    ps = be.spec_verify_bc(bc, toks); return ps.logits
def eager(p):
    o = fwd(p); mx.eval(o); r = np.array(o); discard(); return r
M, E = a.m, a.extra
refs = {p: eager(p) for p in range(P0, P0 + M + E)}
mx.synchronize()
slots = list(range(M)); outs = []
for k in range(M):
    mx.synchronize(); mx.set_cache_limit(0); mx.clear_cache(); lib.rp_rec_begin(k); lib.rp_rec_snap_born(k, 1)
    o = fwd(P0 + k); mx.eval(o); mx.synchronize(); n = lib.rp_rec_end(); discard(); outs.append(o)
    assert np.array_equal(np.array(o), refs[P0 + k])
print("recorded", M, "samples,", n, "dispatches each")
sl = (C.c_int * M)(*slots); ps = (C.c_int * M)(*range(P0, P0 + M))
L = lib.rp_region_fit(0, M, sl, ps, 0, a.maxl) if False else None
h = lib.rp_region_new()
L = lib.rp_region_fit(h, M, sl, ps, 0, a.maxl)
print("fit period L =", L, "fields", lib.rp_region_nfields(h))
assert L > 0
par = mx.array([P0], dtype=mx.int32); mx.eval(par)
sa = [mx.contiguous(x) for x in st.stacked_arrays()]; mx.eval(*sa)                  # the state arrays are region inputs (rebound at every call)
st_vals = None
ins = [par] + sa
ip = (C.c_size_t * len(ins))(*[addr(x) for x in ins]); il = (C.c_size_t * len(ins))(*[x.nbytes for x in ins])
o0 = outs[0]; op = (C.c_size_t * 1)(addr(o0)); ol = (C.c_size_t * 1)(o0.nbytes)
rc = lib.rp_region_build(h, 0, len(ins), ip, il, 1, op, ol, 3); print("build rc", rc); assert rc == 0
lib.rp_region_selfcheck.argtypes = [C.c_int, C.c_int, C.c_int]
print("selfcheck (table vs recording) differing fields per sample:", [lib.rp_region_selfcheck(h, k, P0 + k) for k in range(1, M)])
trig = mx.fast.metal_kernel(name=f"kyojin_region_trigger_{h}", input_names=[f"i{k}" for k in range(len(ins))], output_names=["o0"], source="if (thread_position_in_grid.x == 0) { o0[0] = o0[0]; }")
def call(p):
    return trig(inputs=[mx.array([p], dtype=mx.int32)] + list(st.stacked_arrays()), grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[o0.shape], output_dtypes=[mx.float32])[0]
q = P0 + 2
e1, e2 = eager(q), eager(q); o1 = call(q); mx.eval(o1); o1 = np.array(o1); o2 = call(q); mx.eval(o2); o2 = np.array(o2)
print("DETERMINISM eager-eager", np.array_equal(e1, e2), "replay-replay", np.array_equal(o1, o2), "replay-eager row maxdiff", np.abs(o1 - e1).max(-1))
bad = 0
for p in range(P0, P0 + M + E):
    ref = eager(p)                                        # eager first (same cmp / keys rows written in place), then the replay at the same position
    o = call(p); mx.eval(o); same = np.array_equal(np.array(o), ref); bad += not same
    if not same and p - P0 < 6: print("   nearest eager ref:", min(((float(np.abs(np.array(o) - refs[q]).max()), q - P0) for q in refs)), "top1 agree w/ eager(p):", int((np.array(o).argmax(-1) == ref.argmax(-1)).sum()), "/", ref.shape[0], "nan", int(np.isnan(np.array(o)).sum()))
    if not same: print("  mismatch at p0+%d (%s)" % (p - P0, "sample" if p < P0 + M else "extrapolated"), float(np.abs(np.array(o) - ref).max()))
print(f"replay at {M + E} positions ({M} sampled, {E} extrapolated): mismatching {bad}")
# timing: CPU per verify, eager vs replay
et, rt = [], []
for r in range(30):
    p = P0 + (r % (M + E))
    t0 = time.perf_counter(); o = fwd(p); t1 = time.perf_counter(); mx.async_eval(o); t2 = time.perf_counter(); mx.eval(o); discard(); et.append(t2 - t0)
    t0 = time.perf_counter(); o = call(p); t1 = time.perf_counter(); mx.async_eval(o); t2 = time.perf_counter(); mx.eval(o); rt.append(t2 - t0)
q = lambda v: (1e6 * min(v), 1e6 * sorted(v)[len(v) // 2])
print("CPU us per verify (min, med): eager %.0f %.0f | replay %.0f %.0f" % (*q(et), *q(rt)))
lib.rp_region_stat.restype = C.c_long; lib.rp_region_stat.argtypes = [C.c_int, C.c_int]
print("region stats: replays", lib.rp_region_stat(h, 0), "rebinds", lib.rp_region_stat(h, 1), "private MB", lib.rp_region_stat(h, 3) / 1e6, "| inputs", len(ins))
