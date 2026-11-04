"""moeunion_ds41.py PACK [N=48] [R=4] [TEXTS=2]: distinct routed experts per MoE layer in the R-row served verify pass (teacher-forced on the
greedy plain text) against R x topk pairs, and the plain step's 1 x topk. union / topk ~ the expert bytes a verify pass must read in units of one
plain step: grouped MoE time / plain MoE time should track it if the grouped path reads each union expert once.
NAN0 then runs a plain step on token id 0 and prints, per MoE call, whether its input / output holds a NaN (the id-0 NaN bug bisect)."""
import glob, os, sys
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import moe as MO

A = sys.argv + [None] * 4
PACK = A[1]; N = int(A[2] or 48); R = int(A[3] or 4); NT = int(A[4] or 2)
tok = AutoTokenizer.from_pretrained(PACK)
be, _ = load_backend(PACK, log=lambda *a: None)

REC = None                                                            # None = off; list = one entry per MoE call of the pass
_orig = MO.MoeGather.__call__


def _hook(self, x, ids, wts, fq8=False):
    y = _orig(self, x, ids, wts, fq8)
    if REC is not None:
        mx.eval(x, ids, y)
        REC.append((int(np.unique(np.array(ids)).size), int(ids.size), bool(mx.isnan(x).any()), bool(mx.isnan(y).any())))
    return y


MO.MoeGather.__call__ = _hook
uni, pairs, plain = [], [], []
for f in sorted(glob.glob(os.path.expanduser("docs/*.md")))[:NT]:
    pids = tok.encode(open(f).read()[:6000], add_special_tokens=False)
    c = be.make_cache()
    for i in range(0, len(pids), 2048): lg = be.prefill_chunk(c, pids[i:i + 2048])
    bc, t, text = be.batch([c]), int(mx.argmax(lg[-1])), []
    for _ in range(N):
        REC = []
        text.append(t); lg = be.step(bc, mx.array([t], dtype=mx.int32))[0]; mx.eval(lg); t = int(mx.argmax(lg))
        plain += [u for u, _, _, _ in REC]; REC = None
    c = be.make_cache()
    for i in range(0, len(pids), 2048): be.prefill_chunk(c, pids[i:i + 2048])
    bc = be.batch([c])
    for i in range(0, len(text) - R + 1, R):
        REC = []
        ps = be.R.verify(be.model, bc, [text[i:i + R]]); mx.eval(ps.logits); ps.commit([R - 1]); mx.eval(*bc[0].stacked_arrays())
        uni += [u for u, _, _, _ in REC]; pairs += [p for _, p, _, _ in REC]; REC = None
    REC = []
    a = be.step(bc, mx.array([0], dtype=mx.int32))[0]; mx.eval(a)
    first = next((k for k, r in enumerate(REC) if r[2] or r[3]), -1)
    print(f"NAN0 {os.path.basename(f)} logits_nan {bool(mx.isnan(a).any())} moe_calls {len(REC)} first_nan_call {first}"
          + (f" in_nan {REC[first][2]} out_nan {REC[first][3]}" if first >= 0 else ""), flush=True)
    REC = None
u, p, q = np.array(uni), np.array(pairs), np.array(plain)
if not u.size or not q.size: sys.exit(f"UNION no MoeGather calls recorded (verify {u.size}, plain {q.size})")
print(f"UNION R {R} calls {u.size} pairs/call {p.mean():.1f} union mean {u.mean():.2f} min {u.min()} max {u.max()} plain mean {q.mean():.2f} "
      f"ratio union/plain {u.mean() / q.mean():.2f}", flush=True)
