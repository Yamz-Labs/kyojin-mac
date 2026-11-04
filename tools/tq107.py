"""tq107.py N,N,.. T,T,.. [SHAPES] [VARIANTS]: per-op TQ attention vs fp16 attention at depth, one serial chain of L layers (decode order: layer l + 1
takes the output of layer l as its query, no overlap of independent launches). ms per layer (best / median of REP), GB/s of the bytes the op must read
(TQ: K + V codes + scales; fp16: K + V) and % of the read roof (mx.sum over 1 GiB).
SHAPES Hq:Hk:D (default 16:4:256 = Qwen3.5 4B / 9B). VARIANTS (default fp16,tq): fp16 = best of stock mx.fast.sdpa and the split-K NAX kernel
(sdpa_dec, what route() picks); stock / nax alone; tq = TQKVCache.attend (tq6w); tq@A = the T = 1 kernel built with KYOJIN_TQ_ABL=A (ablation bits,
tq.py _DEC_SRC: 1 no pool K dot, 2 no pool V, 8 no V LUT, 16 no K LUT, 64 no fp16 K, 128 no q rotation); tq%K=V = the tq module knob K set to V
(e.g. tq%VU=4). Env L (layers, 8), REP (7), IT (3 chain passes per timing)."""
import os, sys, time, json, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
from kyojin_mlx import tq
from kyojin_mlx.sdpa_dec import sdpa_decode

Ns = [int(v) for v in sys.argv[1].split(",")]
Ts = [int(v) for v in sys.argv[2].split(",")]
SH = [tuple(int(x) for x in s.split(":")) for s in (sys.argv[3] if len(sys.argv) > 3 else "16:4:256").split(",")]
VAR = (sys.argv[4] if len(sys.argv) > 4 else "fp16,tq").split(",")
L = int(os.environ.get("L", "8")); REP = int(os.environ.get("REP", "7")); IT = int(os.environ.get("IT", "3"))
SDPA = mx.fast.scaled_dot_product_attention
P0 = 1024


def bench(f):
    mx.eval(f(1)); xs = []
    for _ in range(REP):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f(IT)); xs.append((time.perf_counter() - t0) / IT)
    return min(xs), statistics.median(xs)


def roof():
    x = mx.random.normal((1 << 29,)).astype(mx.float16); mx.eval(x)
    b, _ = bench(lambda n: [mx.sum(x.reshape(-1, 1024), axis=1) for _ in range(n)])
    del x; mx.clear_cache(); return (1 << 30) / b / 1e9


RF = roof()
print(f"read roof {RF:.1f} GB/s", flush=True)
knob0 = {}
for Hq, Hk, D in SH:
    for N in Ns:
        mx.random.seed(2)
        mk = lambda n: mx.random.normal((1, Hk, n, D)).astype(mx.float16)
        base = tq.TQKVCache(6, 6, 4, 256); base.update_and_fetch(mk(P0), mk(P0)); st = base.state; r = -(-N // P0)
        tqc = []
        for i in range(L):
            c = tq.TQKVCache(6, 6, 4, 256)
            c.state = [mx.tile(x[:, :, :P0], (1, 1, r) + (1,) * (x.ndim - 3))[:, :, :N] for x in st[:4]] + list(st[4:])
            tqc.append(c)
        mx.eval([c.state for c in tqc]); del base, st
        fkv = [(mk(N), mk(N)) for _ in range(L)]; mx.eval(fkv); mx.clear_cache()
        tqb = L * 2 * Hk * N * (D * 6 // 8 + 2); fpb = L * 2 * Hk * N * D * 2
        for T in Ts:
            if T > 1:
                for c in tqc: c.update_and_fetch(mk(T), mk(T))
                mx.eval([c.state for c in tqc])
            q0 = mx.random.normal((1, Hq, T, D)).astype(mx.float16); mx.eval(q0); sc = D ** -0.5
            def chain_tq(n):
                x = q0
                for _ in range(n):
                    for c in tqc: x = c.attend(x, sc)
                return x
            def chain_fp(f):
                def g(n):
                    x = q0
                    for _ in range(n):
                        for k, v in fkv: x = f(x, k, v)
                    return x
                return g
            stock = lambda q, k, v: SDPA(q, k, v, scale=sc, mask="causal" if T > 1 else None)
            nax = lambda q, k, v: sdpa_decode(q, k, v, sc)
            row = {}
            for var in VAR:
                if var in ("fp16", "stock", "nax"):
                    ts = [bench(chain_fp(f)) for nm, f in (("stock", stock), ("nax", nax)) if var in (nm, "fp16")]
                    b, m = min(ts); nb = fpb
                else:
                    abl, knobs = "0", {}
                    for part in var.split("%")[1:]:
                        k_, v_ = part.split("="); knobs[k_] = int(v_)
                    if "@" in var.split("%")[0]:
                        abl = var.split("%")[0].split("@")[1]
                    os.environ["KYOJIN_TQ_ABL"] = abl; tq._kernels.clear()
                    for k_, v_ in knobs.items():
                        knob0.setdefault(k_, getattr(tq, k_)); setattr(tq, k_, v_)
                    b, m = bench(chain_tq); nb = tqb
                    for k_ in knobs: setattr(tq, k_, knob0[k_])
                    os.environ["KYOJIN_TQ_ABL"] = "0"; tq._kernels.clear()
                ms = b / L * 1e3
                row[var] = dict(ms=round(ms, 3), med=round(m / L * 1e3, 3), gbs=round(nb / b / 1e9, 1), roof_ms=round(nb / L / RF / 1e6, 3), pct=round(100 * nb / b / 1e9 / RF))
                print(f"Hq{Hq} Hk{Hk} D{D} N={N} T={T} {var:14s} {ms:7.3f} ms/layer (med {m / L * 1e3:.3f})  {nb / b / 1e9:6.1f} GB/s  {100 * nb / b / 1e9 / RF:3.0f} % roof  (roof {nb / L / RF / 1e6:.3f} ms)", flush=True)
            if "fp16" in row and "tq" in row:
                print(f"  -> tq/fp16 {row['tq']['ms'] / row['fp16']['ms']:.2f}", flush=True)
            print("TQ107", json.dumps(dict(Hq=Hq, Hk=Hk, D=D, N=N, T=T, **row)), flush=True)
        del tqc, fkv; mx.clear_cache()
