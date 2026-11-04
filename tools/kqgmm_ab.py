"""kqgmm A/B: prefill tok/s of the MoE pack with the grouped stored-row GEMM off / on, chunk 2048 / 4096, interleaved in one process (KYOJIN_KQ_GMM is read per call).
usage: kqgmm_ab.py PACK TEXT_FILE NTOK [--arms 0:2048,1:2048,0:4096,1:4096] [--reps 2]   (box cap ~33 GiB beside the DS4.1 lane: NTOK <= 16384)"""
import argparse, os, time
import numpy as np
import mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("text"); ap.add_argument("ntok", type=int)
ap.add_argument("--arms", default="0:2048,1:2048,0:4096,1:4096"); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--skip", type=int, default=0); ap.add_argument("--flag", default="KYOJIN_KQ_GMM"); a = ap.parse_args()
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import multiuser as MU
from kyojin_mlx.foreign_names import gguf_serve_dir
a.pack = gguf_serve_dir(a.pack)                              # a GGUF without config / tokenizer files: built from the metadata
tok = AutoTokenizer.from_pretrained(a.pack)
ids = tok.encode(open(a.text).read())[a.skip:a.skip + a.ntok + 1]
assert len(ids) == a.ntok + 1, len(ids)
import threading, traceback, sys as _s, ctypes
_lc = ctypes.CDLL("/usr/lib/libproc.dylib")
class _R(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in "user system pkg_idle interrupt pageins wired resident footprint start exit".split()]
def _fp():
    r = _R(); _lc.proc_pid_rusage(os.getpid(), 0, ctypes.byref(r)); return r.footprint / 2**30
def _sampler():                                              # footprint trace: where the main thread is when it jumps by > 1.5 GiB between two samples
    last = 0
    while True:
        f = _fp()
        if f - last > 1.5:
            fr = _s._current_frames()[threading.main_thread().ident]
            print(f"[fp] {last:.1f} -> {f:.1f} GiB at " + " < ".join(f"{x.name}:{x.lineno}" for x in traceback.extract_stack(fr)[-4:][::-1]), flush=True)
        last = f; time.sleep(0.25)
threading.Thread(target=_sampler, daemon=True).start()
if os.environ.get('SG_CACHE_MB'): mx.set_cache_limit(int(os.environ['SG_CACHE_MB']) << 20)                                   # load: freed host buffers go back to the OS (the box leakguard sums all footprints)
be, _ = load_backend(a.pack, log=lambda *x: None)
mx.clear_cache()
m = be.model; lm = m.language_model if "language_model" in m else m; trunk = MU._trunk(lm)

def plan(sz, n):
    return [(p, min(p + sz, n)) for p in range(0, n, sz)]

def prefill(chunk, n):
    import gc; gc.collect(); mx.clear_cache(); cache = be.make_cache(); mx.reset_peak_memory(); fpmax = 0.0
    t0 = time.perf_counter()
    for lo, hi in plan(chunk, n):
        t = mx.array(ids[lo:hi])[None]
        h = (trunk(t, cache=cache) if MU._kw_cache(type(trunk)) else trunk(t, cache))[0]
        mx.eval(h, *be.state(cache)); fpmax = max(fpmax, _fp())
    dt = time.perf_counter() - t0
    pk = mx.get_peak_memory() / 2**30; del cache, h; gc.collect(); mx.clear_cache()
    return dt, pk, fpmax

N = a.ntok
print(f"model resident (mlx active) {mx.get_active_memory() / 2**30:.1f} GiB, footprint {_fp():.1f} GiB", flush=True)
prefill(2048, 2048 * 2)                                     # warm: compiles both paths
os.environ[a.flag] = "1"; prefill(2048, 2048 * 2); os.environ[a.flag] = "0"
res = {}
for rep in range(a.reps):
    for arm in a.arms.split(","):
        fl, ch = arm.split(":"); os.environ[a.flag] = fl
        dt, pk, fpm = prefill(int(ch), N)
        res.setdefault(arm, []).append((N / dt, pk, fpm))
        print(f"  rep {rep} kq={fl} chunk={ch}: {N / dt:.0f} tok/s ({dt:.2f} s) mlx peak {pk:.1f} GiB footprint max {fpm:.1f} GiB", flush=True)
print("SUMMARY N=%d" % N)
for arm, v in res.items():
    print(f"  kq:chunk {arm}: best {max(x[0] for x in v):.0f} tok/s  all {[round(x[0]) for x in v]}  mlx peak {max(x[1] for x in v):.1f} GiB  footprint max {max(x[2] for x in v):.1f} GiB")
