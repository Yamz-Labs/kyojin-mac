"""Multi-user batched decode on the Mac: exactness vs solo, aggregate / per-user tok/s for N users, shared-prefix fork.
usage: bench_multiuser.py MODEL_DIR PROMPTS.json [--users 1,2,4] [--n 96] [--reps 5] [--check 64] [--shared] [--json OUT]
Every user decodes the same number of tokens (no EOS stop). Rounds interleave the user counts so GPU contention from other
jobs on the machine hits all counts alike. Peak memory is printed at the end (run under /usr/bin/time -l for the footprint)."""
import argparse, json, os, statistics, time
import numpy as np
import mlx.core as mx
from kyojin_mlx.exl3_linear import load_exl3
from kyojin_mlx import multiuser as MU

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("prompts")
ap.add_argument("--users", default="1,2,4"); ap.add_argument("--n", type=int, default=96); ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--check", type=int, default=64); ap.add_argument("--shared", action="store_true"); ap.add_argument("--json")
ap.add_argument("--mode", default=None, help="exact | mma | default (multiuser.configure; env KYOJIN_MU_MODE)")
ap.add_argument("--exact-n", default="2,4,8"); ap.add_argument("--cache-mb", type=int, default=256)
a = ap.parse_args()
MU.configure(a.mode)
Ns = [int(v) for v in a.users.split(",")]
mx.set_cache_limit(a.cache_mb * 2 ** 20)
P = json.load(open(a.prompts))
model, _ = load_exl3(a.model)
out = {"load": os.getloadavg(), "n": a.n, "exact": {}, "speed": {}}
# stock model (before the row-invariant swap): greedy tokens of each prompt, to check the canonical decode path against it
stock_ref = []
for ids in P["prompts"][:4]:
    c, lg = MU.prefill(model, ids); t = mx.argmax(lg, axis=-1); mx.eval(t)
    stock_ref.append(MU.decode_solo(model, c, t, a.check))
print('row-invariant linears:', MU.install(model))


def first_of(lg):
    t = mx.argmax(lg, axis=-1); mx.eval(t); return t


def users_from(prompts):
    cs, firsts, lgs = [], [], []
    for ids in prompts:
        c, lg = MU.prefill(model, ids)
        cs.append(c); lgs.append(lg); firsts.append(first_of(lg))
    return cs, mx.concatenate(firsts), lgs


# ---------------------------------------------------------------- canonical decode path vs stock mlx matmul path (solo)
cano = []
for ids in P["prompts"][:4]:
    c, lg = MU.prefill(model, ids); t = mx.argmax(lg, axis=-1); mx.eval(t)
    cano.append(MU.decode_solo(model, c, t, a.check))
agree = [sum(int(x == y) for x, y in zip(s, c)) for s, c in zip(stock_ref, cano)]
out["canonical_vs_stock"] = {"tokens": a.check, "matching_prefix_or_count": agree}
print(f"canonical solo vs stock solo greedy ({a.check} tokens, token-wise matches per prompt): {agree}", flush=True)

# ---------------------------------------------------------------- exactness: N users with different lengths vs each solo
for N in [int(v) for v in a.exact_n.split(",")]:
    if N > len(P["prompts"]):
        continue
    prompts = P["prompts"][:N]
    cs, first, _ = users_from(prompts)
    ref_cs = [MU.fork_cache(c) for c in cs]                      # solo reference runs on forks of the same prefill state
    bat_cs = [MU.fork_cache(c) for c in cs]
    # first-step logits, bitwise
    bc = MU.batch_caches(bat_cs)
    lb = np.array(MU.logits_last(model, first[:, None], bc)); mx.eval()
    ls = [np.array(MU.logits_last(model, first[i:i + 1, None], ref_cs[i])) for i in range(N)]
    bitwise = all(np.array_equal(lb[i].view(np.uint16), ls[i][0].view(np.uint16)) for i in range(N))
    # full greedy, batched vs solo (fresh forks)
    ref = [MU.decode_solo(model, MU.fork_cache(cs[i]), first[i:i + 1], a.check) for i in range(N)]
    got = MU.decode_batched(model, MU.batch_caches([MU.fork_cache(c) for c in cs]), first, a.check)
    same = [got[i] == ref[i] for i in range(N)]
    out["exact"][N] = {"lens": [len(p) for p in prompts], "first_logits_bitwise": bool(bitwise), "tokens_equal": same,
                       "n_tokens": a.check}
    print(f"exact N={N} lens={[len(p) for p in prompts]}: first-step logits bitwise {bitwise}; greedy {a.check} tokens equal to solo per row: {same}", flush=True)

# ---------------------------------------------------------------- speed
def run_speed(N):
    cs, first, _ = users_from(P["prompts"][:N])
    bc = MU.batch_caches(cs)
    mx.synchronize()
    t0 = time.perf_counter()
    MU.decode_batched(model, bc, first, a.n)
    return time.perf_counter() - t0

def run_stock(N=1):
    cs, first, _ = users_from(P["prompts"][:1])
    mx.synchronize(); t0 = time.perf_counter()
    MU.decode_solo(model, cs[0], first, a.n, async_=True)
    return time.perf_counter() - t0

for N in Ns:
    run_speed(N)                                                  # warm-up (kernel compile for this row count)
run_stock()
res = {N: [] for N in Ns}; stock = []
for r in range(a.reps):
    for N in Ns:
        res[N].append(N * (a.n - 1) / run_speed(N))
    stock.append((a.n - 1) / run_stock())
base = statistics.median(res[Ns[0]]) if Ns[0] == 1 else None
print(f"\nload avg at start {out['load'][0]:.2f}; decode {a.n} tokens per user, {a.reps} interleaved rounds")
print(f"stock solo (no RowsKV)       : median {statistics.median(stock):7.1f} tok/s   (min {min(stock):.1f} max {max(stock):.1f})")
for N in Ns:
    m = statistics.median(res[N])
    out["speed"][N] = {"agg_median": m, "per_user": m / N, "reps": res[N], "x_vs_N1": (m / base) if base else None}
    print(f"N={N}: aggregate median {m:7.1f} tok/s  per-user {m / N:7.1f}   x{(m / base) if base else float('nan'):.2f} vs N=1   (min {min(res[N]):.1f} max {max(res[N]):.1f})", flush=True)
out["stock"] = stock

# ---------------------------------------------------------------- shared prefix fork (sub-agents)
if a.shared:
    N = max(Ns)
    t0 = time.perf_counter(); base_cache, _ = MU.prefill(model, P["prefix"]); mx.synchronize(); t_prefix = time.perf_counter() - t0
    t0 = time.perf_counter(); forks = [MU.fork_cache(base_cache) for _ in range(N)]; mx.eval([c.keys for f in forks for c in f if hasattr(c, "keys")]); t_fork = time.perf_counter() - t0
    t0 = time.perf_counter(); cs, lgs = [], []
    for f, s in zip(forks, P["suffixes"][:N]):
        c, lg = MU.prefill(model, s, cache=f); cs.append(c); lgs.append(lg)
    mx.synchronize(); t_suf = time.perf_counter() - t0
    t0 = time.perf_counter()
    for s in P["suffixes"][:N]:
        MU.prefill(model, P["prefix"] + s)
    mx.synchronize(); t_full = time.perf_counter() - t0
    first = mx.concatenate([first_of(l) for l in lgs])
    ref = [MU.decode_solo(model, MU.fork_cache(cs[i]), first[i:i + 1], a.check) for i in range(N)]
    got = MU.decode_batched(model, MU.batch_caches([MU.fork_cache(c) for c in cs], prefix_len=0), first, a.check)     # bitwise path (RowsKV)
    out["shared"] = {"N": N, "prefix": len(P["prefix"]), "suffix_lens": [len(s) for s in P["suffixes"][:N]], "prefix_prefill_s": t_prefix,
                     "fork_s": t_fork, "suffix_prefill_s": t_suf, "recompute_all_s": t_full, "tokens_equal": [got[i] == ref[i] for i in range(N)]}
    print(f"\nshared prefix {len(P['prefix'])} tokens, {N} users: prefix prefill {t_prefix * 1e3:.0f} ms once + fork {t_fork * 1e3:.1f} ms + suffixes {t_suf * 1e3:.0f} ms"
          f" = {(t_prefix + t_fork + t_suf) * 1e3:.0f} ms; recompute prefix per user: {t_full * 1e3:.0f} ms; decode rows equal solo: {out['shared']['tokens_equal']}")

out["peak_gb"] = mx.get_peak_memory() / 2 ** 30
print(f"mlx peak memory {out['peak_gb']:.2f} GB")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1, default=str)
