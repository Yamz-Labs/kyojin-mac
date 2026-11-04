"""hostprof_ds41.py PACK [STEPS=64]: is a DS4.1 decode step host- or GPU-bound? Runs the engine's pipelined plain loop (launch step n+1 from the
lazy tokens of step n, then read step n) and teacher-forced 4-row verify passes with a host busy-wait of 0 / 4 / 8 ms per step (SPIN lines).
cProfile cannot see nanobind calls (mx.async_eval time lands in its caller), so the split comes from the spin test, not a profile."""
import glob, os, sys, time
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend

A = sys.argv + [None] * 3
PACK = A[1]; N = int(A[2] or 64)
tok = AutoTokenizer.from_pretrained(PACK)
be, _ = load_backend(PACK, log=lambda *a: None)
pids = tok.encode(open(sorted(glob.glob(os.path.expanduser("docs/*.md")))[0]).read()[:6000], add_special_tokens=False)


def fresh():
    c = be.make_cache()
    for i in range(0, len(pids), 2048): lg = be.prefill_chunk(c, pids[i:i + 2048])
    return be.batch([c]), int(mx.argmax(lg[-1]))


def launch(bc, toks):
    y = mx.argmax(be.step(bc, toks), axis=-1).astype(mx.int32).reshape(1)
    mx.async_eval(y)
    return y


def plain(bc, t, n, text):
    y = launch(bc, mx.array([t], dtype=mx.int32))
    for _ in range(n):
        y2 = launch(bc, y); text.append(y.item()); y = y2
    text.append(y.item())


SPIN = [0.0]
_ae = mx.async_eval
AE = []                                                              # wall of each mx.async_eval of the current step, by submit index


def _timed_ae(*a):
    t = time.perf_counter(); _ae(*a); AE.append(time.perf_counter() - t)


def spin():
    t = time.perf_counter() + SPIN[0]
    while time.perf_counter() < t: pass


def arms(tag, run, n):
    """Host spin test: a busy-wait of s ms added to each step's host work. Wall + s per step => host-bound; flat wall => GPU-bound (the host has
    that much slack); partial => co-bound. The spin-0 arm also times every mx.async_eval by submit index (blocking inside a submit = GPU backpressure)."""
    base = None
    for ms in (0, 4, 8):
        SPIN[0] = ms / 1e3; per = []
        mx.async_eval = _timed_ae if ms == 0 else _ae
        t0 = time.perf_counter(); run(n, per); w = 1e3 * (time.perf_counter() - t0) / n
        mx.async_eval = _ae
        base = w if base is None else base
        line = f"SPIN {tag} spin {ms} ms wall {w:.2f} ms/step (+{w - base:.2f})"
        if ms == 0 and per:
            m = np.mean(np.array([p_[:min(map(len, per))] for p_ in per]), axis=0) * 1e3
            line += f" async_eval sum {m.sum():.2f} ms by submit [{' '.join(f'{v:.2f}' for v in m)}]"
        print(line, flush=True)


bc, t = fresh(); text = []
plain(bc, t, 16, text)                                               # warm: kernels compiled, regions traced
t0 = time.perf_counter(); plain(bc, text[-1], N, text); w0 = time.perf_counter() - t0
print(f"HOSTPROF plain unprofiled wall {1e3 * w0 / N:.2f} ms/step", flush=True)


def run_plain(n, per):
    y = launch(bc, mx.array([text[-1]], dtype=mx.int32))
    for _ in range(n):
        spin(); AE.clear(); y2 = launch(bc, y); per.append(list(AE)); text.append(y.item()); y = y2
    text.append(y.item())


arms("plain", run_plain, N)
vb, _ = fresh(); vt = list(text); vi = [0]


def run_verify(n, per):
    for _ in range(n):
        i = vi[0]; vi[0] += 4
        spin(); AE.clear()
        ps = be.R.verify(be.model, vb, [vt[i:i + 4]]); v = mx.argmax(ps.logits, axis=-1); v.tolist(); ps.commit([3]); per.append(list(AE))


run_verify(4, [])
arms("verify", run_verify, (len(vt) - 16) // 12)
