"""Ablation of _attn_group (one full-attention layer, N sessions x (1+D) rows): total, minus update_and_fetch, minus attend_shared, minus rope/norm.
usage: mu4_att2.py MODEL N D P"""
import sys, os, time
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
exec(open(os.path.join(os.path.dirname(__file__), "mu4_ops.py")).read().split("lm = be.lm.model")[0])
lm = be.lm.model
from kyojin_mlx import tq as TQ
from mlx_lm.models.base import create_attention_mask
bc = be.batch_spec([be.fork(c) for c in users]); mx.eval(*[a for c in bc.caches for a in be.state(c)])
fa = lm.fa_idx; li = fa
mod = lm.layers[li].self_attn
g, S = N, D + 1
x = mx.random.normal((1, g * S, lm.layers[0].input_layernorm.weight.shape[0])).astype(mx.bfloat16)
ys = {n: getattr(mod, n)(x) for n in QM._ATT_IN}; mx.eval(*ys.values())
cs = [c[li] for c in bc.caches]
snap = [c.offset for c in cs]; snr = [(c.rk, c.rv, c.sk, c.sv) for c in cs]
def run():
    for j, c in enumerate(cs): c.offset = snap[j]; c.rk, c.rv, c.sk, c.sv = snr[j]
    return be._attn_group(mod, li, ys, 0, g, S, x.dtype, bc.caches, create_attention_mask, fa)
def tm(reps=9):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = run(); t1 = time.perf_counter(); mx.eval(r); ts.append(((time.perf_counter() - t1) * 1e3, (t1 - t0) * 1e3))
    ts.sort(); return ts[reps // 2]
print("full: eval %.2f ms  build %.2f ms" % tm())
orig_u = TQ.TQKVCache.update_and_fetch; orig_a = TQ.TQKVCache.attend_shared
TQ.TQKVCache.update_and_fetch = lambda self, k, v: (setattr(self, "offset", self.offset + k.shape[2]), (k, v))[1]
print("no update_and_fetch: eval %.2f build %.2f" % tm())
TQ.TQKVCache.update_and_fetch = orig_u
TQ.TQKVCache.attend_shared = staticmethod(lambda cs_, q, sc, P: mx.zeros(q.shape, q.dtype))
print("no attend_shared (zeros): eval %.2f build %.2f" % tm())
TQ.TQKVCache.update_and_fetch = lambda self, k, v: (setattr(self, "offset", self.offset + k.shape[2]), (k, v))[1]
print("neither: eval %.2f build %.2f" % tm())
TQ.TQKVCache.update_and_fetch = orig_u; TQ.TQKVCache.attend_shared = orig_a
res = {}
for flag in (False, True, False, True):
    QM._KV_MANY = flag
    print("KV_MANY=%s:" % flag, "eval %.2f build %.2f" % tm())
    o = run(); mx.eval(o)
    pools = [mx.array(c.kq[0][:, :, :, :, :]) if False else c.kq[1][:, :, :c.offset] for c in cs]
    vp = [c.vq[0][:, :, :c.offset] for c in cs]; kt = [c.kq[0] for c in cs]
    mx.eval(*pools, *vp, *kt)
    res[flag] = (o, pools, vp, kt)
a, b = res[False], res[True]
print("out equal", bool(mx.array_equal(a[0], b[0]).item()), "k scales equal", all(bool(mx.array_equal(x, y).item()) for x, y in zip(a[1], b[1])),
      "v words equal", all(bool(mx.array_equal(x, y).item()) for x, y in zip(a[2], b[2])), "k words equal", all(bool(mx.array_equal(x, y).item()) for x, y in zip(a[3], b[3])))
QM._KV_MANY = False; o1 = run(); mx.eval(o1); o2 = run(); mx.eval(o2)
print("False vs False equal", bool(mx.array_equal(o1, o2).item()))
print("max abs diff old/new", float(mx.max(mx.abs(a[0].astype(mx.float32) - b[0].astype(mx.float32))).item()), "ref max", float(mx.max(mx.abs(a[0].astype(mx.float32))).item()))
