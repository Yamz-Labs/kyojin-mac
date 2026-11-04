"""Ablation of _gdn_group (one GDN layer, N sessions x (1+D) rows). usage: mu4_gdn2.py MODEL N D P"""
import sys, os, time
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
exec(open(os.path.join(os.path.dirname(__file__), "mu4_ops.py")).read().split("lm = be.lm.model")[0])
lm = be.lm.model
bc = be.batch_spec([be.fork(c) for c in users]); mx.eval(*[a for c in bc.caches for a in be.state(c)])
li = next(i for i, l in enumerate(lm.layers) if l.is_linear)
mod = lm.layers[li].linear_attn
g, S = N, D + 1
x = mx.random.normal((1, g * S, lm.layers[0].input_layernorm.weight.shape[0])).astype(mx.bfloat16)
ys = {n: getattr(mod, n)(x) for n in QM._GDN_IN}; mx.eval(*ys.values())
cs = [c[li] for c in bc.caches]
snap = [(c[0], c[1], c.offset if hasattr(c, "offset") else None) for c in cs]
def run():
    for c, s in zip(cs, snap): c[0], c[1] = s[0], s[1]
    return be._gdn_group(mod, li, ys, 0, g, S, bc.caches, [[] for _ in range(g)])
def tm(reps=11):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = run(); t1 = time.perf_counter(); mx.eval(r, *[a for c in cs for a in (c[0], c[1])]); ts.append(((time.perf_counter() - t1) * 1e3, (t1 - t0) * 1e3))
    ts.sort(); return "eval %.2f ms build %.2f ms" % ts[reps // 2]
for f in (False, True, False, True):
    QM._GDN_ONECOPY = f; print("onecopy", f, tm())
import mlx_lm.models.gated_delta as GD
orig = GD.gated_delta_update
GD.gated_delta_update = lambda q, k, v, a, b, A, dt, st, m, use_kernel=True: (v, st)
print("no recurrence kernel:", tm())
GD.gated_delta_update = orig
def t2(f, reps=11):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = f(); mx.eval(r); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return ts[reps // 2]
print("state shape", cs[0][1].shape, cs[0][1].dtype, "conv", cs[0][0].shape)
print("floor %.2f" % t2(lambda: x + 1))
print("concat states: %.2f ms" % t2(lambda: mx.concatenate([c[1] for c in cs], axis=0)))
print("concat convs: %.2f ms" % t2(lambda: mx.concatenate([c[0] for c in cs], axis=0)))
st = mx.concatenate([c[1] for c in cs], axis=0); mx.eval(st)
print("8 slices contiguous (conv-like): %.2f ms" % t2(lambda: [mx.contiguous(st[j:j+1, -3:, :1]) for j in range(8)]))
