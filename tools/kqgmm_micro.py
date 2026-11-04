"""kqgmm micro: one layer of routed experts as stored (GGUF), fused grouped GEMM (foreign_kqgmm) vs today's decode-stack + gather_mm (ForeignSwitchGLU._sorted_gmm).
usage: kqgmm_micro.py GGUF_DIR [--layer 0] [--rows 2048] [--hist real|zipf|uniform|FILE.npy] [--mt 32] [--reps 7]
Prints time per chunk of the MLP (min of reps, warm), max relative diff (max|a-b| / max|b|) and the routing histogram summary."""
import argparse, time, sys
import numpy as np
import mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("gguf"); ap.add_argument("--layer", type=int, default=0); ap.add_argument("--rows", type=int, default=2048)
ap.add_argument("--hist", default="real"); ap.add_argument("--mt", type=int, default=32); ap.add_argument("--reps", type=int, default=7); ap.add_argument("--text", default=None)
a = ap.parse_args()
from kyojin_mlx import foreign_gguf as GF, foreign_linear as FL, foreign_kqgmm as KQ
g = GF.GgufFile(GF.find_gguf(a.gguf))
def dev(name):
    ti = g.tensors[name]; raw = g.raw(name); k = ti.ne[0]
    d = FL.DevPack([GF.to_pack(ti.tname, raw[e], k) for e in range(ti.ne[2])]); mx.eval(d.arrays()); return d, ti.tname
L = a.layer
dg, tg = dev(f"blk.{L}.ffn_gate_exps.weight"); du, tu = dev(f"blk.{L}.ffn_up_exps.weight"); dd, td = dev(f"blk.{L}.ffn_down_exps.weight")
glu = FL.ForeignSwitchGLU(dg, du, dd)
E, H, K = dg.E, dg.k, 8
print(f"layer {L}: gate {tg} up {tu} down {td}  E={E} H={H} ff={dg.n}  rows={a.rows} K={K} MT={a.mt}")
T = a.rows; rng = np.random.default_rng(0)
x = (rng.standard_normal((T, H)) * 1.0).astype(np.float16)
if a.hist == "real":                                  # token embeddings of real text through the router of this layer (RMS-normed): a real-token routing histogram
    rt = g.tensors[f"blk.{L}.ffn_gate_inp.weight"]; rw = GF.dequant(rt.tname, g.raw(f"blk.{L}.ffn_gate_inp.weight"), rt.ne[0]).reshape(rt.shape).astype(np.float32)
    et = g.tensors["token_embd.weight"]; eraw = g.raw("token_embd.weight")
    from transformers import AutoTokenizer
    from kyojin_mlx.foreign_names import gguf_serve_dir
    tok = AutoTokenizer.from_pretrained(gguf_serve_dir(a.gguf))
    txt = open(a.text).read() if a.text else open(__file__).read() * 40 + open(KQ.__file__).read() * 40
    ids = np.array(tok.encode(txt)[:T]); assert len(ids) == T, len(ids)
    emb = GF.dequant(et.tname, eraw[ids], et.ne[0]).reshape(T, -1).astype(np.float32)
    xn = emb / np.sqrt((emb ** 2).mean(-1, keepdims=True) + 1e-6)
    x = xn.astype(np.float16)
    lg = xn @ rw.T
elif a.hist == "zipf":
    p = 1.0 / np.arange(1, E + 1) ** 0.8; p /= p.sum(); lg = np.log(p)[None] + rng.gumbel(size=(T, E))
elif a.hist == "uniform":
    lg = rng.standard_normal((T, E))
else:
    inds = np.load(a.hist)[:T * K].reshape(-1, K); lg = None
if a.hist.endswith(".npy"):
    e_np = np.load(a.hist).reshape(-1)[:T * K].astype(np.uint32)
else:
    e_np = np.argpartition(-lg, K, axis=1)[:, :K].reshape(-1).astype(np.uint32)
cnt = np.bincount(e_np, minlength=E)
print(f"routing: experts used {int((cnt>0).sum())}/{E}, rows/expert mean {cnt.mean():.1f} max {cnt.max()} p90 {np.percentile(cnt,90):.0f}  tiles(MT={a.mt}) {int(np.ceil(cnt/a.mt).sum())}")
x2 = mx.array(x); e32 = mx.array(e_np)
def old(): return glu._sorted_gmm(x2, e32, T, K)
def new(): return KQ.mlp(glu, x2, e32, K, a.mt)
def bench(f, n=None):
    for _ in range(2): mx.eval(f())
    ts = []
    for _ in range(n or a.reps):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3, sorted(ts)[len(ts) // 2] * 1e3
yo = old(); yn = new(); mx.eval(yo, yn)
o = np.array(yo).astype(np.float32); n_ = np.array(yn).astype(np.float32)
d = np.abs(o - n_); print(f"max|old| {np.abs(o).max():.3f}  max abs diff {d.max():.4f}  max rel diff (max diff / max|old|) {d.max()/np.abs(o).max():.5f}  rms rel {np.sqrt((d**2).mean())/np.sqrt((o**2).mean()):.5f}  nan {int(np.isnan(n_).sum())}")
for _ in range(30): mx.eval(old()); mx.eval(new())           # GPU clocks up before the timed reps
to_, tn_ = [], []
for _ in range(a.reps):                                        # interleaved: both paths see the same clock / cache state
    for f, acc in ((old, to_), (new, tn_)):
        t = time.perf_counter(); mx.eval(f()); acc.append(time.perf_counter() - t)
to = (min(to_) * 1e3, sorted(to_)[len(to_) // 2] * 1e3); tn = (min(tn_) * 1e3, sorted(tn_)[len(tn_) // 2] * 1e3)
print(f"old (decode stack + gather_mm): min {to[0]:.1f} ms med {to[1]:.1f}   new (kqgmm): min {tn[0]:.1f} ms med {tn[1]:.1f}   drop {100*(1-tn[0]/to[0]):.1f} % (med {100*(1-tn[1]/to[1]):.1f} %)")
MT = a.mt; S = T * K
order = mx.argsort(e32).astype(mx.uint32); es = e32[order]; tb, off = KQ.plan(es, E, MT); ix = KQ.xperm(dg.fmt, H); xg = KQ.gather_perm(x2, order, K, ix)
ha = mx.array(rng.standard_normal((S, dg.n)).astype(np.float16)); idd = KQ.xperm(dd.fmt, dd.k); hp = KQ.silu_perm(ha, ha, idd); mx.eval(order, es, tb, off, xg, ha, hp)
parts = {"sort+plan": lambda: KQ.plan(e32[mx.argsort(e32).astype(mx.uint32)], E, MT), "gather_perm": lambda: KQ.gather_perm(x2, order, K, ix),
         "gate gmm": lambda: KQ.gmm(dg, xg, tb, off, S, MT), "up gmm": lambda: KQ.gmm(du, xg, tb, off, S, MT), "down gmm (scatter)": lambda: KQ.gmm(dd, hp, tb, off, S, MT, orow=order),
         "silu_perm": lambda: KQ.silu_perm(ha, ha, idd)}
for k_, f in parts.items():
    print(f"  part {k_}: {bench(f)[0]:.2f} ms")
