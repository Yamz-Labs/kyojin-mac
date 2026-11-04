"""nllhalf_ds41.py PACK [N=128] [RS=4] [TEXTS=8]: real-text quality of packed-fp16 row FMAs (KYOJIN_HALF_ROWS=R) in the served R-row verify.
Each text: the first 1500 tokens are the prompt, the next N real tokens are teacher-forced through one-row plain steps, R-row fp32 verify
blocks and R-row half verify blocks. Prints the mean NLL of the real next token and top-1 hit rate per arm (NLLSTAT), and the paired
half - rows NLL delta with its standard error. The rows - plain delta is the noise floor of the accepted ROWS_NAX default."""
import glob, os, sys
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx import metal_src as S
from kyojin_mlx.serve.backends import load_backend

A = sys.argv + [None] * 4
PACK = A[1]; N = int(A[2] or 128); RS = [int(r) for r in (A[3] or "4").split(",")]; NT = int(A[4] or 8); P = 1500
tok = AutoTokenizer.from_pretrained(PACK)
files = sorted(glob.glob(os.path.expanduser("docs/*.md")))
be, _ = load_backend(PACK, log=lambda *a: None)


def nll(lg, tgt):                                                     # lg [n, V] predicts tgt [n]
    lg = lg.astype(mx.float32); lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    v = -mx.take_along_axis(lp, mx.array(tgt, dtype=mx.int32)[:, None], axis=-1)[:, 0]
    return np.array(v), np.array(mx.argmax(lg, -1)) == np.array(tgt)


def prefilled(pids):
    c = be.make_cache()
    for i in range(0, len(pids), 2048): be.prefill_chunk(c, pids[i:i + 2048])
    return c


def plain(pids, text):
    bc, out = be.batch([prefilled(pids)]), []
    for t in text[:-1]:
        lg = be.step(bc, mx.array([t], dtype=mx.int32))[0]; mx.eval(lg); out.append(lg)
    return mx.stack(out)


def rows(pids, text, R, half):
    c = prefilled(pids); S.HALF_ROWS = half; bc, out = be.batch([c]), []   # half set after the prefill: same cache for every arm
    for i in range(0, len(text) - 1, R):
        grp = text[i:min(i + R, len(text) - 1)]
        ps = be.R.verify(be.model, bc, [grp])
        lg = ps.logits.reshape(-1, ps.logits.shape[-1])
        ps.commit([len(grp) - 1]); mx.eval(lg, *bc[0].stacked_arrays()); out.append(lg)
    S.HALF_ROWS = 0
    return mx.concatenate(out, axis=0)


ARMS = ["plain"] + [f"{k}{r}" for r in RS for k in ("rows", "half")]
L = {k: [] for k in ARMS}; H = {k: [] for k in ARMS}; done = 0
for f in files:
    ids = tok.encode(open(f).read(), add_special_tokens=False)
    if len(ids) < P + N: continue
    pids, text = ids[:P], ids[P:P + N]; tgt = text[1:]
    g = {"plain": plain(pids, text)}
    for r in RS: g[f"rows{r}"], g[f"half{r}"] = rows(pids, text, r, 0), rows(pids, text, r, r)
    res = {k: nll(g[k], tgt) for k in ARMS}
    bad = [k for k in ARMS if np.isnan(res[k][0]).any()]
    if bad: print(f"NAN {os.path.basename(f)} {bad}", flush=True); continue
    for k in ARMS: L[k].append(res[k][0]); H[k].append(res[k][1])
    print(f"TEXT {os.path.basename(f)} " + " ".join(f"{k} {res[k][0].mean():.4f}" for k in ARMS), flush=True); done += 1
    if done >= NT: break
for k in ARMS:
    v = np.concatenate(L[k]); print(f"NLLSTAT {k} nll {v.mean():.4f} ppl {np.exp(v.mean()):.3f} top1 {np.mean(np.concatenate(H[k])):.4f} n {v.size}", flush=True)
for r in RS:
    for a, b in ((f"half{r}", f"rows{r}"), (f"rows{r}", "plain"), (f"half{r}", "plain")):
        d = np.concatenate(L[a]) - np.concatenate(L[b])
        print(f"DELTA {a}-{b} {d.mean():+.5f} se {d.std() / np.sqrt(d.size):.5f}", flush=True)
