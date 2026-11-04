"""Per-op exclusive GPU time of one prefill chunk of R rows after D tokens of past (synced around every wrapped op: upper bound, launch overhead included).
usage: pf3_table.py MODEL_DIR R D"""
import sys, time, collections, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, mlp_pf as MP, layer_pf as LP, gdn_pf as GP, gdn_nax as GN, wcache as WC, kernels as KN, metal_nax_sh as SH
from kyojin_mlx.serve.backends import load_backend
be, _ = load_backend(sys.argv[1]); R = int(sys.argv[2]); D = int(sys.argv[3]); mx.set_cache_limit(256 * 2 ** 20)
ids = [(i * 7919) % 5000 + 100 for i in range(R)]
acc = collections.defaultdict(float); cnt = collections.Counter(); on = [False]; stack = []
def arrs(x):
    if isinstance(x, mx.array): return [x]
    if isinstance(x, (list, tuple)): return [a for y in x for a in arrs(y)]
    return []
def mk(f, name):
    def w(*a, **k):
        if not on[0]: return f(*a, **k)
        ins = arrs(a) + arrs(list(k.values()))
        if ins: mx.eval(ins)
        mx.synchronize(); stack.append(0.0); t0 = time.perf_counter()
        out = f(*a, **k); o = arrs(out)
        if o: mx.eval(o)
        mx.synchronize(); dt = time.perf_counter() - t0; ch = stack.pop()
        acc[name] += dt - ch; cnt[name] += 1
        if stack: stack[-1] += dt
        return out
    return w
def wrap(obj, attr, name): setattr(obj, attr, mk(getattr(obj, attr), name))
wrap(EL.Exl3Linear, "_dense", "linear (rot+matmul+fin, excl decode)")
wrap(WC, "decode_wr", "weight decode")
wrap(EL.Exl3Linear, "_inner_sh", "linear SH decode-once kernel")
wrap(KN, "rot", "rot (Hadamard)")
wrap(MP, "swiglu_rot", "mlp middle fused")
wrap(LP, "seam1", "seam1"); wrap(LP, "seam2", "seam2")
wrap(GP, "pre", "GDN pre (conv/qkv prep)"); wrap(GP, "gated_norm", "GDN gated norm"); wrap(GN, "gated_delta_nax", "GDN core (chunked NAX)")
wrap(mx.fast, "scaled_dot_product_attention", "attention SDPA")
wrap(mx.fast, "rms_norm", "rms_norm")
def go(prof):
    c = be.make_cache(); mx.synchronize(); on[0] = False
    with WC.prefill_scope(be.model, R, -(-(D + R) // R)):
        for s in range(0, D, R): mx.eval(be.prefill_chunk(c, ids[:R]))
        mx.synchronize(); acc.clear(); cnt.clear(); stack.clear(); on[0] = prof
        t0 = time.perf_counter(); mx.eval(be.prefill_chunk(c, ids)); tot = time.perf_counter() - t0
    on[0] = False; return tot
unp = min(go(False) for _ in range(3))
tot = go(True)
print(f"R={R} D={D}: unprofiled chunk {unp*1e3:.1f} ms ({R/unp:.0f} tok/s); synced total {tot*1e3:.1f} ms")
s = 0
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:42s} {v*1e3:8.1f} ms  n={cnt[k]}"); s += v
print(f"  {'other (norm-free glue, conv, residual, ..)':42s} {(tot-s)*1e3:8.1f} ms")
