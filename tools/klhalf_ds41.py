"""klhalf_ds41.py PACK [N=60] [RS=4,5] [TEXTS=8]: teacher-forced KL of R-row served verify blocks (the MTP verify row count) with packed-fp16 row FMAs (KYOJIN_HALF_ROWS=R)
and without (0) against the one-row greedy plain steps, on 4 real texts. The R-row fp32 line is the rows-vs-plain floor the half line is judged against.
Prints KLSTAT lines (mean / max KL, top-1 agreement) per arm and the half vs fp32 rows KL, all positions and block 0 alone (identical cache
at its start: KL there is one pass's error, growth after it is cache drift). R=5 rows run the default ROWS_NAX class: the tool's calibration.
PROBE0 checks a plain step on token id 0 after a normal prompt (the NaN seen after a generated BOS)."""
import glob, os, sys
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx import metal_src as S
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import kernels as KN, moe as MO


class _NoHalf:                                                        # a module's view of metal_src with the half rows path off (bisect dense / MoE)
    def __getattr__(self, k):
        return (lambda mr: False) if k == "half_on" else getattr(S, k)

A = sys.argv + [None] * 4
PACK = A[1]; N = int(A[2] or 60); RS = [int(r) for r in (A[3] or "4,5").split(",")]; NT = int(A[4] or 8)
tok = AutoTokenizer.from_pretrained(PACK)
files = sorted(glob.glob(os.path.expanduser("docs/*.md")))[:NT]
be, _ = load_backend(PACK, log=lambda *a: None)


def lsm(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def kl(p, q):
    lp, lq = lsm(p), lsm(q)
    return mx.sum(mx.where(mx.isinf(lp), 0.0, mx.exp(lp) * (lp - lq)), axis=-1)      # masked vocab rows are -inf on both sides


def rows(pids, text, R, half, only=None):
    c = be.make_cache()
    for i in range(0, len(pids), 2048): be.prefill_chunk(c, pids[i:i + 2048])
    S.HALF_ROWS = half                                                # after the prefill: every arm starts from the same cache
    KN.S, MO.S = (S, _NoHalf()) if only == "dense" else (_NoHalf(), S) if only == "moe" else (S, S)
    bc, out = be.batch([c]), []
    for i in range(0, len(text), R):                                  # the served verify pass (as replay): R rows of one session, all kept
        grp = text[i:i + R]
        ps = be.R.verify(be.model, bc, [grp])
        lg = ps.logits.reshape(-1, ps.logits.shape[-1]).astype(mx.float32)
        ps.commit([len(grp) - 1]); mx.eval(lg, *bc[0].stacked_arrays()); out.append(lg)
    S.HALF_ROWS = 0; KN.S, MO.S = S, S
    return mx.concatenate(out, axis=0)


ARMS = [f"{k}{r}" for r in RS for k in ("rows", "half")]
KEYS = ARMS + [f"half_vs_rows{r}" for r in RS]
acc = {k: [] for k in KEYS}; top = {k: [] for k in KEYS}; acc0 = {k: [] for k in KEYS}; top0 = {k: [] for k in KEYS}


def add(k, p, q, r):
    v = np.array(kl(p, q)); t = np.array(mx.argmax(p, -1)) == np.array(mx.argmax(q, -1))
    acc[k].append(v); top[k].append(t); acc0[k].append(v[:r]); top0[k].append(t[:r])


probe = True
for f in files:
    pids = tok.encode(open(f).read()[:6000], add_special_tokens=False)
    c = be.make_cache()
    for i in range(0, len(pids), 2048): lg = be.prefill_chunk(c, pids[i:i + 2048])
    bc, t, text, ref = be.batch([c]), int(mx.argmax(lg[-1])), [], []
    for _ in range(N):
        text.append(t); lg = be.step(bc, mx.array([t], dtype=mx.int32))[0].astype(mx.float32); mx.eval(lg); ref.append(lg); t = int(mx.argmax(lg))
    ref = mx.stack(ref)
    bad = int(mx.isnan(ref).any(axis=-1).sum())
    if bad:                                                           # a NaN plain run is a separate bug: report it, keep the text out of the gate
        print(f"NANREF {os.path.basename(f)} rows {bad}/{N} prompt {len(pids)} first {text[:4]}", flush=True); continue
    if probe:                                                         # same cache, one more plain step on id 0, then one on a normal id
        probe = False
        a = be.step(bc, mx.array([0], dtype=mx.int32))[0]; b = be.step(bc, mx.array([t], dtype=mx.int32))[0]; mx.eval(a, b)
        print(f"PROBE0 {os.path.basename(f)} id0_nan {bool(mx.isnan(a).any())} next_nan {bool(mx.isnan(b).any())}", flush=True)
    line = []
    for r in RS:
        g = {"rows": rows(pids, text, r, 0), "half": rows(pids, text, r, r)}
        for k in ("rows", "half"):
            add(f"{k}{r}", ref, g[k], r); line.append(f"{k}{r} {float(np.mean(acc[f'{k}{r}'][-1])):.2e}/{float(np.mean(top[f'{k}{r}'][-1])):.3f}")
        add(f"half_vs_rows{r}", g["rows"], g["half"], r)
    print(f"TEXT {os.path.basename(f)} " + " ".join(line), flush=True)
for k in KEYS:
    for nm, a_, t_ in (("all", acc, top), ("blk0", acc0, top0)):
        if not a_[k]: continue
        v = np.concatenate(a_[k]); print(f"KLSTAT {k} {nm} mean {v.mean():.2e} max {v.max():.2e} top1 {np.mean(np.concatenate(t_[k])):.3f} n {v.size}", flush=True)
