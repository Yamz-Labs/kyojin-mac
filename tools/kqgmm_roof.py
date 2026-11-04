"""kqgmm roofline: per projection of one routed layer (real-token routing), achieved TFLOPS of the grouped stored-row GEMM vs the 87 TFLOPS roof,
variants that split dequant / MMA / tail cost, and the fp16 dense grouped GEMM (gather_mm) as the upper bound.
usage: kqgmm_roof.py GGUF_DIR [--layer 3] [--rows 2048] [--mt 32] [--reps 9] [--text FILE]"""
import argparse, time
import numpy as np
import mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("gguf"); ap.add_argument("--layer", type=int, default=3); ap.add_argument("--rows", type=int, default=2048)
ap.add_argument("--mt", type=int, default=32); ap.add_argument("--reps", type=int, default=9); ap.add_argument("--text", default=None); ap.add_argument("--variants", default="full,nodec,nomma")
a = ap.parse_args()
from kyojin_mlx import foreign_gguf as GF, foreign_linear as FL, foreign_kqgmm as KQ, foreign_wide as FW, kernels as KN
g = GF.GgufFile(GF.find_gguf(a.gguf)); L = a.layer
def dev(name):
    ti = g.tensors[name]; raw = g.raw(name); k = ti.ne[0]
    d = FL.DevPack([GF.to_pack(ti.tname, raw[e], k) for e in range(ti.ne[2])]); mx.eval(d.arrays()); return d
dg, du, dd = (dev(f"blk.{L}.ffn_{n}_exps.weight") for n in ("gate", "up", "down"))
E, H, K, T, MT = dg.E, dg.k, 8, a.rows, a.mt
rt = g.tensors[f"blk.{L}.ffn_gate_inp.weight"]; rw = GF.dequant(rt.tname, g.raw(f"blk.{L}.ffn_gate_inp.weight"), rt.ne[0]).reshape(rt.shape).astype(np.float32)
et = g.tensors["token_embd.weight"]; eraw = g.raw("token_embd.weight")
from transformers import AutoTokenizer
from kyojin_mlx.foreign_names import gguf_serve_dir
tok = AutoTokenizer.from_pretrained(gguf_serve_dir(a.gguf))
txt = open(a.text).read() if a.text else open(KQ.__file__).read() * 40 + open(FL.__file__).read() * 5
ids = np.array(tok.encode(txt)[:T]); assert len(ids) == T, len(ids)
emb = GF.dequant(et.tname, eraw[ids], et.ne[0]).reshape(T, -1).astype(np.float32)
xn = emb / np.sqrt((emb ** 2).mean(-1, keepdims=True) + 1e-6)
e_np = np.argpartition(-(xn @ rw.T), K, axis=1)[:, :K].reshape(-1).astype(np.uint32)
cnt = np.bincount(e_np, minlength=E); S = T * K
tiles = int(np.ceil(cnt / MT).sum())
print(f"layer {L} E={E} H={H} ff={dg.n} S={S} MT={MT}: experts used {(cnt>0).sum()}, mean {cnt.mean():.1f} max {cnt.max()}, tiles {tiles}, pad efficiency {S/(tiles*MT):.3f}")
for lo, hi in ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 99999)):
    m = (cnt >= lo) & (cnt <= hi); print(f"  experts with {lo}-{hi} rows: {int(m.sum())} experts, {int(cnt[m].sum())} rows, {int(np.ceil(cnt[m]/MT).sum())} tiles")
rng = np.random.default_rng(0)
x2 = mx.array((rng.standard_normal((T, H))).astype(np.float16)); e32 = mx.array(e_np)
order = mx.argsort(e32).astype(mx.uint32); es = e32[order]; tb, off = KQ.plan(es, E, MT)
xg = KQ.gather_perm(x2, order, K, KQ.xperm(dg.fmt, H))
hp = mx.array(rng.standard_normal((S, dd.k)).astype(np.float16)); mx.eval(order, es, tb, off, xg, hp)
VAR = {"full": lambda s: s,
       "nodec": lambda s: s.replace("{DEC}", ""),                      # filled below
       }
def variant(dp, v, orow):
    src = KQ.source(dp.fmt, orow is not None)
    if v == "nodec":                                                  # decoder replaced by one xor: loads + MMA kept
        import re
        m = re.search(r"const uint xs_ = wv\[j\]\[q\].*?const half2 w2 = [^;]*;", src, re.S); assert m, "dec not found"
        src = src.replace(m.group(0), "const half2 w2 = as_type<half2>(wv[j][q] ^ (t * 0x11111u));", 1)
    if v == "nomma":                                                  # MMA removed, but the decoded registers still consumed
        a_ = "if (step == 0) op.run(sA, rB0, cT); else op.run(sA, rB1, cT);"
        assert a_ in src
        src = src.replace(a_, "{ auto& rb = (step == 0) ? rB0 : rB1; float acc_ = 0.0f; for (uint16_t i = 0; i < rb.get_capacity(); ++i) acc_ += float(rb[i]); cT[0] += acc_; }")
    return src
def run(dp, xin, v, orow=None):
    v, _, wsv = v.partition(":"); E_ = dp.E; ws = int(wsv) if wsv else (3 if MT >= 48 else 4); rc = min(MT, 7168 // (ws * 32))
    nm = KQ.names(dp.fmt, orow is not None)
    arrs = {"x": xin, "qs": dp.qs, "mt": dp.meta, "gs": dp.gs, "hi": getattr(dp, "hi", None), "tb": tb, "off": off, "orow": orow}
    kern = KN._kern(("kqroof", dp.fmt, v, orow is not None), name=f"kqroof_{dp.fmt}_{v}", input_names=nm, output_names=["y"], header=FW.wide_header(dp.fmt), source=variant(dp, v, orow))
    return kern(inputs=[arrs[n] for n in nm], template=[("MT", MT), ("KIN", dp.k), ("N", dp.n), ("WS", ws), ("RC", rc), ("E", E_)],
                grid=(-(-dp.n // 32) * 32 * ws, S // MT + E_, 1), threadgroup=(32 * ws, 1, 1), output_shapes=[(S, dp.n)], output_dtypes=[mx.float16])[0]
def bench(f):
    for _ in range(3): mx.eval(f())
    for _ in range(20): mx.eval(f())
    ts = []
    for _ in range(a.reps):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3
sb = sum(int(p.qs.nbytes) + int(p.meta.nbytes) + (int(p.hi.nbytes) if getattr(p, "hi", None) is not None else 0) for p in (dg, du, dd))
print(f"stored bytes of the layer's 3 stacks: {sb/1e6:.0f} MB; bytes read per tile-set (tiles x one expert x 3) ~ {tiles*sb/E/1e6:.0f} MB")
res = {}
for name, dp, xin, orow in (("gate", dg, xg, None), ("up", du, xg, None), ("down", dd, hp, order)):
    fl = 2.0 * S * dp.k * dp.n
    row = []
    for v in a.variants.split(","):
        try: t = bench(lambda: run(dp, xin, v, orow))
        except Exception as ex: print(f"  {name} {v}: failed {str(ex)[:120]}"); continue
        row.append(f"{v} {t:.3f} ms ({fl/t/1e9:.1f} TF)"); res[(name, v)] = t
    # fp16 dense grouped GEMM upper bound on the same routing (random fp16 stack; mx.gather_mm sorted)
    w = mx.array(rng.standard_normal((E, dp.n, dp.k)).astype(np.float16)); xs = xin[:, None, :]; mx.eval(w)
    td = bench(lambda: mx.gather_mm(xs, w.swapaxes(-1, -2), rhs_indices=es, sorted_indices=True)[:, 0]); del w
    print(f"{name} {dp.fmt} {dp.n}x{dp.k}: " + "; ".join(row) + f"; fp16 gather_mm {td:.3f} ms ({fl/td/1e9:.1f} TF); dense@87TF ideal {fl/87e12*1e3:.3f} ms; useful/padded flops {S/(tiles*MT):.2f}")
