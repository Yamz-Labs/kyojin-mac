"""pool102.py CTX [LAYERS]: synthetic served-round memory probe for the windowed KV caches (no model, < 4 GiB).
LAYERS caches at the Qwen3.5 attention shape (B=1, Hk=4, Hq=16, D=256), prefilled to CTX, then 48 MTP-like rounds: update_and_fetch(5 rows) +
attend(5 rows) + eval + trim(1..4). Prints the per-round transient (MLX peak - active, MB) and ms/round for: TQ (as is), TQ with a prefix
snapshot still alive (the served prefix cache), and fp16 KVCache. A full pool copy per round shows as a transient ~ pool bytes."""
import os, sys, time, statistics
KYO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, KYO)
import mlx.core as mx
from kyojin_mlx.tq import TQKVCache
from mlx_lm.models.cache import KVCache

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
NL = int(sys.argv[2]) if len(sys.argv) > 2 else 2
B, HK, HQ, D, T = 1, 4, 16, 256, 5
mx.random.seed(0)


def fp_attend(c, q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")


def run(kind, snap=False, mode=""):
    mk = (lambda: TQKVCache(6, 6, 4, 256)) if kind == "tq" else KVCache
    cs = [mk() for _ in range(NL)]
    for c in cs:
        for p in range(0, CTX, 2048):
            k = mx.random.normal((B, HK, 2048, D)).astype(mx.float16); v = mx.random.normal((B, HK, 2048, D)).astype(mx.float16)
            c.update_and_fetch(k, v)
            mx.eval(*[a for a in (c.kq + c.vq if kind == "tq" else (c.keys, c.values))])
    keep = [c.state for c in cs] if snap else None                  # the prefix snapshot the engine stores during prefill
    if keep: mx.eval(*[a for s in keep for a in s if a is not None])
    if mode == "fork":
        cs = [c.fork() for c in cs]
    if mode == "restore":
        st = [c.state for c in cs]; cs2 = [mk() for _ in range(NL)]
        for c, s in zip(cs2, st): c.state = s
        cs = cs2
    pk, ms = [], []
    for r in range(48):
        mx.eval([c.kq + c.vq for c in cs] if kind == "tq" else [(c.keys, c.values) for c in cs])
        mx.reset_peak_memory(); t0 = time.perf_counter()
        q = mx.random.normal((B, HQ, T, D)).astype(mx.float16)
        outs = []
        for c in cs:
            k = mx.random.normal((B, HK, T, D)).astype(mx.float16); v = mx.random.normal((B, HK, T, D)).astype(mx.float16)
            if kind == "tq":
                c.update_and_fetch(k, v); outs.append(c.attend(q, D ** -0.5))
            else:
                kk, vv = c.update_and_fetch(k, v); outs.append(fp_attend(c, q, kk, vv))
        mx.eval(outs)
        for c in cs: c.trim(1 + r % 4)
        ms.append(time.perf_counter() - t0); pk.append((mx.get_peak_memory() - mx.get_active_memory()) / 2**20)
    pool = sum(a.nbytes for c in cs for a in ((c.kq + c.vq) if kind == "tq" else (c.keys, c.values))) / 2**20
    print(f"{kind:3s} snap={int(snap)} {mode or '-':7s} L={NL} ctx={CTX}: pool {pool:.0f} MB, round transient MB first {pk[0]:.0f} med {statistics.median(pk[4:]):.0f} max {max(pk[4:]):.0f}, "
          f"ms/round med {1e3 * statistics.median(ms[4:]):.2f}, active {mx.get_active_memory() / 2**30:.2f} GB", flush=True)
    del cs, keep; mx.clear_cache()


for spec in (os.environ.get("P102", "tq,tq+snap,tq+snap+restore,fp")).split(","):
    p = spec.split("+")
    run(p[0], "snap" in p, next((x for x in p[1:] if x in ("fork", "restore")), ""))
