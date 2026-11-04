"""Decode SDPA at depth, fp16 KV: GB/s of the K/V read vs the read roof. usage: sdpa91.py [N,N,..] [T,T,..] [shapes] [variants]
shapes: Hq:Hk:D comma list (default 16:4:256 = Qwen3.5 4B/9B). L layers of distinct K/V are chained in one eval (launch cost amortised, no cache reuse).
variants: stock (mx.fast.sdpa, causal tail for T > 1), fold (macdec89 fold_sdpa, T > 1), b<k> (stock with MLX_SDPA_BLOCKS=k), nax (kyojin split-K NAX
kernel, sdpa_dec.py, split count from tuning.chip() sdpa_tpc), n<k> (same with k splits per kv head). Env: L layers, IT, REP, PAD (strided cache views),
CHAIN=1: layer l + 1 takes the output of layer l as its query and the IT passes run as ONE serial chain (the decode step order: no overlap of
independent launches; without it, IT independent chains in one eval run side by side and share each layer's K / V through the cache, which reads
as up to ~190 GB/s on the Air for a kernel that leaves cores idle)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
Ns = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "8192,32768,65536").split(",")]
Ts = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,5").split(",")]
SH = [tuple(int(x) for x in s.split(":")) for s in (sys.argv[3] if len(sys.argv) > 3 else "16:4:256").split(",")]
VAR = (sys.argv[4] if len(sys.argv) > 4 else "stock,fold,nax").split(",")
CHAIN = os.environ.get("CHAIN", "0") == "1"
L = int(os.environ.get("L", "4")); IT = int(os.environ.get("IT", "3")); REP = int(os.environ.get("REP", "5"))
S = mx.fast.scaled_dot_product_attention


def bench(fn, serial=None):
    """Best seconds per pass; serial(reps) = one dependent chain of reps passes (CHAIN=1)."""
    mx.eval(fn() if serial is None else serial(1)); best = 1e9
    for _ in range(REP):
        mx.synchronize(); t0 = time.perf_counter()
        mx.eval([fn() for _ in range(IT)] if serial is None else serial(IT))
        best = min(best, (time.perf_counter() - t0) / IT)
    return best


def chain(f, x, kv):
    for k, v in kv:
        x = f(x, k, v)
    return x


def roof(n_bytes):
    x = mx.random.normal((n_bytes // 2,)).astype(mx.float16); mx.eval(x)
    t = bench(lambda: mx.sum(x.reshape(-1, 1024), axis=1))
    del x; return n_bytes / t / 1e9


print(f"read roof (mx.sum fp16 1 GiB): {roof(1 << 30):.1f} GB/s", flush=True)
for Hq, Hk, D in SH:
    for N in Ns:
        cap = N + int(os.environ.get("PAD", "0"))          # PAD > 0: K / V are views of a longer buffer (the serving cache layout)
        kv = [(mx.random.normal((1, Hk, cap, D)).astype(mx.float16), mx.random.normal((1, Hk, cap, D)).astype(mx.float16)) for _ in range(L)]
        if cap > N:
            kv = [(k[:, :, :N], v[:, :, :N]) for k, v in kv]
        mx.eval(kv); nb = L * 2 * Hk * N * D * 2
        for T in Ts:
            q = mx.random.normal((1, Hq, T, D)).astype(mx.float16); mx.eval(q); sc = D ** -0.5; ref = None
            for var in VAR:
                if var == "fold" and T == 1:
                    continue
                os.environ.pop("MLX_SDPA_BLOCKS", None)
                if var == "stock" or var.startswith("b"):
                    if var.startswith("b"):
                        os.environ["MLX_SDPA_BLOCKS"] = var[1:]
                    f = lambda q, k, v: S(q, k, v, scale=sc, mask="causal" if T > 1 else None)
                elif var == "fold":
                    from kyojin_mlx.multiuser import fold_sdpa
                    f = lambda q, k, v: fold_sdpa(q, k, v, sc)
                elif var.startswith("n"):
                    from kyojin_mlx.sdpa_dec import sdpa_decode
                    ns = int(var[1:]) if var != "nax" else None
                    f = lambda q, k, v, ns=ns: sdpa_decode(q, k, v, sc, ns=ns)
                else:
                    raise ValueError(var)
                t = bench(lambda: [f(q, k, v) for k, v in kv], serial=(lambda reps: chain(f, q, kv * reps)) if CHAIN else None)
                o = f(q, *kv[0]).astype(mx.float32)
                if ref is None:
                    ref = o
                err = mx.abs(o - ref).max().item()
                print(f"{'chain ' if CHAIN else ''}Hq{Hq} Hk{Hk} D{D} N={N:6d} T={T} {var:7s}: {t / L * 1e3:7.3f} ms/layer  {nb / t / 1e9:6.1f} GB/s  maxdiff {err:.2e}", flush=True)
        del kv
os.environ.pop("MLX_SDPA_BLOCKS", None)
