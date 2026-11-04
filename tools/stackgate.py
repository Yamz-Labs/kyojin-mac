"""stackgate: teacher-forced gate of the prefill chunk size (KYOJIN_STACK_CHUNK) on a stacked-expert GGUF/NVFP4 pack.
usage: stackgate.py PACK TEXT_FILE NTOK --ref PLAN --save H.npy   then   ... --vs H.npy --cand PLAN[;PLAN...]   (two processes: one prefill each, the box cap leaves ~33 GiB)
       stackgate.py PACK TEXT_FILE NTOK [--ref 2048] [--cand 4096]   (each plan: chunk sizes, comma list, cycled; e.g. 2032,16)
One loaded model, two caches: the ref cache is fed REF-token chunks, the cand cache CAND-token chunks, the same text. Per position (target = the next real token) it compares full-vocab
log-probs: dNLL (cand - ref), KLD(ref||cand), top-1 agreement, ref top-1 in cand top-5, max |dlogprob|. Positions are bucketed by their offset inside a cand chunk
(first REF tokens = same start state as ref; rest = after one ref chunk boundary) to name the cause. The last --tail positions are reported apart."""
import argparse, os, time
import numpy as np
import mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("text"); ap.add_argument("ntok", type=int)
ap.add_argument("--ref", default="2048"); ap.add_argument("--cand", default="4096"); ap.add_argument("--skip", type=int, default=0); ap.add_argument("--save"); ap.add_argument("--vs"); a = ap.parse_args()
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
mx.set_cache_limit(int(os.environ.get('SG_CACHE_MB', '1024')) << 20)                                   # load: freed host buffers go back to the OS (the box leakguard sums all footprints)
be, _ = load_backend(a.pack, log=lambda *x: None)
mx.clear_cache()                                              # the buffer cache stays at 1 GiB: its idle buffers count in the footprint
m = be.model; lm = m.language_model if "language_model" in m else m; trunk = MU._trunk(lm)
cr, cc = be.make_cache(), be.make_cache()

def plan(spec, n):                                        # chunk (lo, hi) list over n positions, sizes cycled
    sz = [int(x) for x in spec.split(",")]; out = []; p = 0; i = 0
    while p < n:
        out.append((p, min(p + sz[i % len(sz)], n))); p = out[-1][1]; i += 1
    return out

def hidden(spec, n):                                      # trunk output [n, H] fp16 of positions 0..n-1, fed in the chunks of the plan
    import gc; gc.collect(); mx.clear_cache(); cache = be.make_cache(); hs = []; mx.reset_peak_memory()
    for lo, hi in plan(spec, n):
        t = mx.array(ids[lo:hi])[None]
        h = (trunk(t, cache=cache) if MU._kw_cache(type(trunk)) else trunk(t, cache))[0]
        mx.eval(h, *be.state(cache)); hs.append(h)
    out = mx.concatenate(hs, axis=0); mx.eval(out); del hs, cache; gc.collect(); mx.clear_cache()
    print(f"  plan {spec}: mlx peak {mx.get_peak_memory() / 2**30:.1f} GiB, fp {_fp():.1f} GiB", flush=True)
    return out

def lp(h):                                                # full-vocab log-probs fp32 of a few hidden rows
    lg = MU.head(lm, h[None])[0].astype(mx.float32)
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

N = a.ntok; SL = 128; t0 = time.perf_counter()
if a.save:
    np.save(a.save, np.array(hidden(a.ref, N).astype(mx.float32))); raise SystemExit(0)
rh = mx.array(np.load(a.vs)).astype(mx.float16) if a.vs else hidden(a.ref, N)
for cand in a.cand.split(';'):
    ch = hidden(cand, N)
    print(f"  prefill {time.perf_counter() - t0:.0f}s; hidden max|d| {float(mx.max(mx.abs(rh.astype(mx.float32) - ch.astype(mx.float32)))):.4f}", flush=True)
    csz = np.zeros(N, dtype=np.int64)
    for lo, hi in plan(cand, N): csz[lo:hi] = hi - lo
    S = {k: [] for k in "dn n kl t1 t5 dm".split()}
    for s0 in range(0, N, SL):
        s1 = min(s0 + SL, N)
        rl, cs = lp(rh[s0:s1]), lp(ch[s0:s1])
        tg = mx.array(ids[s0 + 1:s1 + 1], dtype=mx.int32)[:, None]
        nr = -mx.take_along_axis(rl, tg, axis=-1)[:, 0]; nc = -mx.take_along_axis(cs, tg, axis=-1)[:, 0]
        kl = mx.sum(mx.exp(rl) * (rl - cs), axis=-1)
        rt = mx.argmax(rl, -1); ct5 = mx.argpartition(-cs, 4, axis=-1)[:, :5]
        t1 = rt == mx.argmax(cs, -1); t5 = mx.any(ct5 == rt[:, None], axis=-1)
        dm = mx.max(mx.abs(rl - cs), axis=-1)
        mx.eval(nr, nc, kl, t1, t5, dm)
        for k, v in (("dn", np.array(nc) - np.array(nr)), ("kl", np.array(kl)), ("t1", np.array(t1)), ("t5", np.array(t5)), ("dm", np.array(dm)), ("n", np.array(nr))):
            S[k].append(v)
    cat = {k: np.concatenate(v) for k, v in S.items()}
    def rep(name, sel):
        n = int(sel.sum())
        if n: print(f"{name:26s} n={n:6d} dNLL {cat['dn'][sel].mean():+.5f} (se {cat['dn'][sel].std() / np.sqrt(n):.5f})  NLLref {cat['n'][sel].mean():.4f}  KLD {cat['kl'][sel].mean():.5f}  top1 {100 * cat['t1'][sel].mean():.2f}%  ref-top1-in-cand-top5 {100 * cat['t5'][sel].mean():.2f}%  max|dlogp| {cat['dm'][sel].max():.3f}")
    print(f"stackgate ref={a.ref} cand={cand} N={N} text={os.path.basename(a.text)}")
    rep("ALL", np.ones(N, dtype=bool))
    for z in sorted(set(csz.tolist())): rep(f"in cand chunk of {z}", csz[:N] == z)
