"""Per-op prefill roof table: one chunk of R rows after D tokens of past, every op against an achievable roof measured on THIS chip in the same process.
usage: pf97_table.py PACK R D [reps=3]
- The past is prefilled once; the timed chunk is re-run at the same offset (trimmable KV caches are trimmed by R, recurrent states are overwritten: same work).
- Per-op time = exclusive GPU time with a sync around every wrapped op (upper bound: the sync bubble is in it; 'sync inflation' = synced total - unprofiled).
- Work: linears 2*rows*K*N FLOPs; attention causal 4*Hq*hd*L*(N - L/2); GDN core (chunked, C=16) Hv*T*(6*Dk*Dv + 2*C*(Dk+Dv)); glue ops bytes = inputs + outputs.
- Roofs (probed here, same shapes): fp16 GEMM (MLX matmul at every linear shape, summed), stock SDPA at the same q/k shapes, GEMM TF for attention and GDN
  (FLOP ceiling), streaming bandwidth (best of fp16 a+b and a*2) for glue and weight decode.
Shapes come from the loaded modules, so the same command runs on any pack / chip (day-1 cell m.pf_roof)."""
import sys, time, collections, mlx.core as mx
from kyojin_mlx import exl3_linear as EL, mlp_pf as MP, layer_pf as LP, gdn_pf as GP, gdn_nax as GN, wcache as WC, kernels as KN
from kyojin_mlx.serve.backends import load_backend

pack, R, D = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
be, _ = load_backend(pack); mx.set_cache_limit(256 * 2 ** 20)
ids = [(i * 7919) % 5000 + 100 for i in range(R)]
acc = collections.defaultdict(float); cnt = collections.Counter(); work = collections.defaultdict(float); on = [False]; stack = []
lin_shapes = collections.Counter(); lin_t = collections.defaultdict(float); lin_probe = {}; att_shapes = collections.Counter(); gdn_shapes = collections.Counter()

def arrs(x):
    if isinstance(x, mx.array): return [x]
    if isinstance(x, (list, tuple)): return [a for y in x for a in arrs(y)]
    return []

def nb(xs): return sum(a.nbytes for a in xs)

def w_lin(a, k, out):
    lin = a[0]; x = a[1] if len(a) > 1 and a[1] is not None else k.get("xh", a[2] if len(a) > 2 else None)
    rows = x.shape[0]; lin_shapes[(rows, lin.in_features, lin.out_features)] += 1
    return 2.0 * rows * lin.in_features * lin.out_features

def w_att(a, k, out):
    q, kk = a[0], a[1]; B, H, L, hd = q.shape; N = kk.shape[2]
    causal = isinstance(k.get("mask"), str) or (len(a) > 4 and isinstance(a[4], str)) or k.get("mask") is not None
    att_shapes[(B, H, L, kk.shape[1], N, hd, causal)] += 1
    return 4.0 * B * H * hd * L * ((N - L / 2) if causal else N)

def w_gdn(a, k, out):
    q, v = a[0], a[2]; B, T, Hk, Dk = q.shape; Hv, Dv = v.shape[2], v.shape[3]
    gdn_shapes[(B, T, Hk, Hv, Dk, Dv)] += 1
    return B * T * Hv * (6.0 * Dk * Dv + 2 * 16 * (Dk + Dv))

def w_bytes(a, k, out): return float(nb(arrs(list(a)) + arrs(list(k.values())) + arrs(out)))

KIND = {}
def mk(f, name, wf, kind):
    KIND[name] = kind
    def w(*a, **k):
        if not on[0]: return f(*a, **k)
        ins = arrs(a) + arrs(list(k.values()))
        if ins: mx.eval(ins)
        mx.synchronize(); stack.append(0.0); t0 = time.perf_counter()
        out = f(*a, **k); o = arrs(out)
        if o: mx.eval(o)
        mx.synchronize(); dt = time.perf_counter() - t0; ch = stack.pop()
        acc[name] += dt - ch; cnt[name] += 1; work[name] += wf(a, k, out)
        if kind == "gemm": lin_t[(a[0].in_features, a[0].out_features)] += dt - ch
        if stack: stack[-1] += dt
        return out
    return w

def wrap(obj, attr, name, wf=w_bytes, kind="bw"): setattr(obj, attr, mk(getattr(obj, attr), name, wf, kind))

wrap(EL.Exl3Linear, "_dense", "linears (rot+matmul+fin)", w_lin, "gemm")
wrap(EL.Exl3Linear, "_inner_sh", "linears fused decode+NAX", w_lin, "gemm")
wrap(WC, "decode_wr", "weight decode (trellis->fp16)")
wrap(KN, "rot", "rot (Hadamard)")
wrap(MP, "swiglu_rot", "mlp middle fused")
wrap(LP, "seam1", "seam1"); wrap(LP, "seam2", "seam2")
wrap(GP, "pre", "GDN pre (conv/qkv prep)"); wrap(GP, "gated_norm", "GDN gated norm")
wrap(GN, "gated_delta_nax", "GDN core (chunked NAX)", w_gdn, "gdn")
wrap(mx.fast, "scaled_dot_product_attention", "attention SDPA", w_att, "sdpa")
wrap(mx.fast, "rms_norm", "rms_norm")

# ---------------- roof probes (same process, at the shapes recorded by a profiled chunk, run before and after the timed chunks; the best of the two is the achievable roof)
def tmin(fn, it=4, rep=5):
    mx.eval(fn()); best = 1e9
    for _ in range(rep):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(*[fn() for _ in range(it)]); best = min(best, (time.perf_counter() - t0) / it)
    return best

def probes():
    g_t = g_f = 0.0
    for (rows, k, n), c in lin_shapes.items():
        x = mx.random.normal((rows, k)).astype(mx.float16); w = mx.random.normal((n, k)).astype(mx.float16); wt = mx.random.normal((k, n)).astype(mx.float16); mx.eval(x, w, wt)
        t = min(tmin(lambda: x @ w.T), tmin(lambda: x @ wt)); g_t += t * c; g_f += 2.0 * rows * k * n * c; lin_probe[(k, n)] = max(lin_probe.get((k, n), 0), 2.0 * rows * k * n / t / 1e12)   # both weight layouts, the faster is the roof
        del x, w, wt
    GEMM_TF = g_f / g_t / 1e12 if g_t else 0.0
    s_t = s_f = 0.0
    for (B, H, L, Hk, N, hd, causal), c in att_shapes.items():
        q = mx.random.normal((B, H, L, hd)).astype(mx.float16); k = mx.random.normal((B, Hk, N, hd)).astype(mx.float16); v = mx.random.normal((B, Hk, N, hd)).astype(mx.float16); mx.eval(q, k, v)
        t = tmin(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask="causal" if causal else None), it=2, rep=3)
        s_t += t * c; s_f += 4.0 * B * H * hd * L * ((N - L / 2) if causal else N) * c
        del q, k, v
    SDPA_TF = s_f / s_t / 1e12 if s_t else 0.0
    a = mx.random.normal((64 * 2 ** 20,)).astype(mx.float16); b = mx.random.normal((64 * 2 ** 20,)).astype(mx.float16); mx.eval(a, b)
    BW = max(3 * a.nbytes / tmin(lambda: a + b, it=8), 2 * a.nbytes / tmin(lambda: a * 2, it=8)) / 1e9   # streaming roof: best of a+b and a*2
    del a, b
    sq = mx.random.normal((4096, 4096)).astype(mx.float16); mx.eval(sq)
    PEAK = 2 * 4096 ** 3 / tmin(lambda: sq @ sq, it=8) / 1e12

    return GEMM_TF, SDPA_TF, BW, PEAK

cache = be.make_cache()
with WC.prefill_scope(be.model, R, -(-(D + R) // R)):
    t0 = time.perf_counter()
    for s in range(0, D, R): mx.eval(be.prefill_chunk(cache, ids[:R]))
    past_s = time.perf_counter() - t0
    def chunk(prof):
        mx.synchronize(); acc.clear(); cnt.clear(); work.clear(); stack.clear(); lin_shapes.clear(); lin_t.clear(); att_shapes.clear(); gdn_shapes.clear(); on[0] = prof
        t0 = time.perf_counter(); mx.eval(be.prefill_chunk(cache, ids)); mx.synchronize(); dt = time.perf_counter() - t0; on[0] = False
        for c in cache:
            if hasattr(c, "is_trimmable") and c.is_trimmable(): c.trim(R)
        return dt
    chunk(True)                                                     # warm + records the shapes the probes need
    P0 = probes()
    ts = sorted(chunk(False) for _ in range(REPS)); unp, unp_med = ts[0], ts[len(ts) // 2]
    tot = chunk(True)
    wc = dict(WC.last)
P1 = probes(); GEMM_TF, SDPA_TF, BW, PEAK = [max(x, y) for x, y in zip(P0, P1)]
print(f"PACK {pack.rstrip('/').split('/')[-1]}  R={R} D={D}: past {D} rows {past_s:.1f} s ({D / max(past_s, 1e-9):.0f} tok/s); chunk unprofiled best {unp*1e3:.1f} med {unp_med*1e3:.1f} ms "
      f"({R/unp:.0f} tok/s); synced {tot*1e3:.1f} ms (sync inflation {(tot-unp)*1e3:.1f} ms); wcache on={wc.get('on')} need {wc.get('need', 0)/2**30:.2f} GiB cap {wc.get('cap', 0)/2**30:.2f} GiB")

print(f"ROOFS probed: fp16 GEMM at the model's linear shapes {GEMM_TF:.2f} TF (4096^3 peak {PEAK:.2f} TF), stock SDPA same shapes {SDPA_TF:.2f} TF, stream {BW:.0f} GB/s")

for (rows, k, n), c in sorted(lin_shapes.items()):
    print(f"  linear K={k:6d} N={n:6d} x{c:3d}: in-model {2.0*rows*k*n*c/lin_t[(k, n)]/1e12:6.2f} TF (excl. decode/rot), probe {lin_probe.get((k, n), 0):6.2f} TF")
print(f"{'op':34s} {'ms':>8s} {'n':>5s} {'work':>10s} {'rate':>10s} {'roof':>10s} {'%roof':>6s} {'ms>roof':>8s}")
over = 0.0; s = 0.0
for name, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    kind = KIND[name]; wk = work[name]; s += v
    if kind == "gemm": rate, roof, unit, wu = wk / v / 1e12, GEMM_TF, "TF", f"{wk/1e12:.2f}TF"
    elif kind == "sdpa": rate, roof, unit, wu = wk / v / 1e12, GEMM_TF, "TF", f"{wk/1e12:.2f}TF"
    elif kind == "gdn": rate, roof, unit, wu = wk / v / 1e12, GEMM_TF, "TF", f"{wk/1e12:.3f}TF"
    else: rate, roof, unit, wu = wk / v / 1e9, BW, "GB/s", f"{wk/1e9:.2f}GB"
    r_ms = v * rate / roof if roof else 0.0; over += v - r_ms
    print(f"{name:34s} {v*1e3:8.2f} {cnt[name]:5d} {wu:>10s} {rate:7.2f}{unit:>3s} {roof:7.2f}{unit:>3s} {100*rate/roof if roof else 0:5.0f}% {(v-r_ms)*1e3:8.2f}")
    if kind == "sdpa": print(f"{'  (vs stock SDPA probe)':34s} {'':8s} {'':5s} {'':10s} {rate:7.2f} TF {SDPA_TF:7.2f} TF {100*rate/SDPA_TF:5.0f}% {(v - v*rate/SDPA_TF)*1e3:8.2f}")
oth = tot - s
print(f"{'other (unwrapped glue + sync)':34s} {oth*1e3:8.2f} {'':5s} {'':10s} {'':10s} {'':10s} {'':6s} {oth*1e3:8.2f}")
roof_ms = (s - over) * 1e3
print(f"ROOF TOTAL (sum of work / roof over the wrapped ops; unwrapped glue at 0) {roof_ms:.1f} ms vs unprofiled chunk {unp*1e3:.1f} ms: chunk at {100*roof_ms/(unp*1e3):.0f} % of the roof")
print(f"TOTAL synced {tot*1e3:.1f} ms, at-roof sum {(s-over)*1e3:.1f} ms, over roof {(over+oth)*1e3:.1f} ms ({100*(over+oth)/tot:.0f} %); unprofiled chunk {unp*1e3:.1f} ms")
