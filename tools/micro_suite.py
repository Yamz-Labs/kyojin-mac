"""Isolated A/B of every op chain the pf3 commits replaced (ds2 real modules, T rows), old vs new interleaved, 1 s idle before each timing, waits for a quiet GPU; per-layer saving = sum(count * (old - new)).
usage: micro_suite.py PACK T [reps]"""
import sys, time, os, subprocess, numpy as np, mlx.core as mx, mlx.nn as nn
from kyojin_mlx import patch as P, pf_glue as PG, fuse as F, attn_prefill as AP
from mlx_vlm.models.deepseek_v41 import language as LM
pack, T = sys.argv[1], int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
L1 = model.language_model.layers[1]; a = L1.attn
H, D = 5120, 512; rng = np.random.default_rng(0)
ME = os.getpid()
def others():
    out = subprocess.run(["ps", "-axo", "pid,pcpu,command"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if ("python" in l or "pytest" in l) and str(ME) != l.split()[0] and "lockf" not in l and float(l.split()[1].replace(",", ".")) > 25.0]
def quiet(wait=0):
    t0 = time.time()
    while others():
        if time.time() - t0 > wait: return False
        time.sleep(1.0)
    return True
def rnd(*s, dt=mx.float16, sc=1.0): return mx.array((rng.standard_normal(s) * sc).astype(np.float32)).astype(dt)
fq = LM.fake_quant_fp8_ue8m0
h4 = rnd(1, T, 4, H); pre = mx.array(rng.uniform(0, 1, (1, T, 4)).astype(np.float32)); xs = rnd(1, T, H); post = mx.array(rng.uniform(0, 2, (1, T, 4)).astype(np.float32)); comb = mx.array(rng.uniform(0, 1, (1, T, 4, 4)).astype(np.float32))
yr = rnd(1, T, 1, H, dt=mx.float32); sh = rnd(1, T, H); nrm = L1.attn_norm
qin = rnd(1, T, a.wq_b.in_features, sc=2.0)
o = rnd(1, 64, T, D, sc=2.0); cs = max(AP.CHUNK, -(-T // 8)); pieces = [o[0, :, i:i + cs] + mx.zeros((1, 1, 1), dtype=mx.float16) for i in range(0, T, cs)]
mx.eval(h4, pre, xs, post, comb, yr, sh, qin, o, *pieces); PG.FA.rope_range(a.rope, 0, T)
def old_pn():
    xn = nrm(F.mhc_pre(h4.reshape(T, 1, 4, H), pre.reshape(T, 1, 4)).reshape(1, T, H)); return [xn, fq(xn)]
def old_q():
    qb = a.wq_b(qin); return a.rope(qb.reshape(1, T, 64, D).transpose(0, 2, 1, 3), 0)
def old_o():
    oc = mx.concatenate(pieces, axis=1)[None] if len(pieces) > 1 else pieces[0][None]
    out = a.rope(oc, 0, inverse=True).reshape(1, 8, -1, T, D).transpose(0, 1, 3, 2, 4).flatten(-2)
    out = a.wo_a(out).transpose(0, 2, 1, 3).flatten(-2); return fq(out)
cases = [
 ("hc_pre + norm + fq", 2, old_pn, lambda: list(PG.pre_norm_fq(h4, pre, nrm.weight, 1e-6))),
 ("hc_expand (prompt rows)", 2, lambda: F.mhc_expand(xs.reshape(T, 1, H), h4.reshape(T, 1, 4, H), post.reshape(T, 1, 4), comb.reshape(T, 1, 4, 4))[0], lambda: PG.expand(xs, h4, post, comb)),
 ("ffn combine", 1, lambda: (yr.astype(mx.float32).sum(-2) + sh.astype(mx.float32)).astype(sh.dtype), lambda: PG.combine(yr, sh)),
 ("second fq pass of the same x (memo)", 2, lambda: fq.__wrapped__(xs), lambda: xs[:, :1] + 0),
 ("wq_b + rope -> q [H,T,D]", 1, old_q, lambda: PG.wq_b_rope(a.wq_b, qin, a.rope, 0, 64, D)),
 ("attn out -> inv rope -> wo_a -> wo_b input", 1, old_o, lambda: PG.wo_a_prompt(a.wo_a, pieces, a.rope, 0)),
]
tot = 0.0
for nm, cnt, fo, fn in cases:
    mx.eval(fo(), fn()); to, tn = [], []
    for i in range(reps):
        for f, l in ((fo, to), (fn, tn)) if i % 2 == 0 else ((fn, tn), (fo, to)):
            q0 = quiet(); time.sleep(1.0); t0 = time.perf_counter(); mx.eval(f()); dt = (time.perf_counter() - t0) * 1e3
            l.append((dt, q0 and not others()))
    def st(l):
        c = [x for x, ok in l if ok] or [x for x, _ in l]
        return min(c), float(np.percentile(c, 10)), len(c)
    (omin, omed, on), (nmin, nmed, nn_) = st(to), st(tn)
    tot += cnt * (omin - nmin)
    print(f"{nm:44s} x{cnt}  old min {omin:7.2f} p10 {omed:7.2f} ({on}) | new min {nmin:7.2f} p10 {nmed:7.2f} ({nn_}) | saving/layer (min) {cnt*(omin-nmin):6.2f} ms (p10 {cnt*(omed-nmed):6.2f})", flush=True)
print(f"T={T}: per reuse layer saving (sum of min deltas) {tot:.1f} ms")
