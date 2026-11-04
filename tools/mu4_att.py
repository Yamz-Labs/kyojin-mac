"""Sub-stage ms of the full-attention core of the N-session verify (update_and_fetch / attend_shared / fallback), sync each.
usage: mu4_att.py MODEL N D P"""
import sys, os, time
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
exec(open(os.path.join(os.path.dirname(__file__), "mu4_ops.py")).read().split("lm = be.lm.model")[0])
lm = be.lm.model
from kyojin_mlx import tq as TQ
bc = be.batch_spec([be.fork(c) for c in users]); mx.eval(*[a for c in bc.caches for a in be.state(c)])
li = lm.fa_idx
mod = lm.layers[li].self_attn
cs = [c[li] for c in bc.caches]
print("cache type", type(cs[0]).__name__, "fused", getattr(cs[0], "fused", None), "window", getattr(cs[0], "window", None), "offset", cs[0].offset, "shared P", be.MU.shared_prefix(cs))
Hq, Hk = mod.num_attention_heads, mod.num_key_value_heads
g, S = N, D + 1
import mlx.core as mx
q = mx.random.normal((g, Hq, S, 256)).astype(mx.bfloat16); k = mx.random.normal((g, Hk, S, 256)).astype(mx.bfloat16); v = mx.random.normal((g, Hk, S, 256)).astype(mx.bfloat16)
mx.eval(q, k, v)
def tm(f, reps=8):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = f(); mx.eval(r); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[reps // 2] * 1e3
floor = tm(lambda: q + 1)
print("sync floor %.2f ms" % floor)
snap = [(c.offset) for c in cs]
def upd():
    outs = []
    for j, c in enumerate(cs):
        c.offset = snap[j]
        outs.append(c.update_and_fetch(k[j:j+1], v[j:j+1]))
    return [x for o in outs for x in o if hasattr(x, "shape")]
print("update_and_fetch x%d: %.2f ms" % (g, tm(upd)))
for j, c in enumerate(cs): c.offset = snap[j] + S
P = be.MU.shared_prefix(cs)
print("attend_shared: %.2f ms" % tm(lambda: TQ.TQKVCache.attend_shared(cs, q, mod.scale, P)))
print("attend per session:", end=" ")
def per():
    return [c.attend(q[j:j+1], mod.scale) if hasattr(c, "attend") else q for j, c in enumerate(cs)]
try: print("%.2f ms" % tm(per))
except Exception as e: print("n/a", e)
Pq = min(P, 4096)
print("fp16_part x%d: %.2f ms" % (g, tm(lambda: [x for j, c in enumerate(cs) for x in c._fp16_part(q[j:j+1], mod.scale, Pq)])))
print("fp16_part x1: %.2f ms" % tm(lambda: list(cs[0]._fp16_part(q[0:1], mod.scale, Pq))))
