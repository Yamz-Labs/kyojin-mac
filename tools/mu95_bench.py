"""Multi-agent decode with a shared prefix: N sessions = one P-token prefix (prefilled once, forked) + a private SUF-token suffix each, greedy decode of
NEW tokens, aggregate and per-session generation tok/s. ENGINE kmlx = the served mlx4 path (mlxpack + q4mma + multiuser batched step; variants
pool = mu_attn.PoolKV shared prefix, rows = per-user RowsKV, suffix _nonax = q4nax off + tuning json $MU95_TUNING_B, suffix _old = q4mma capped at 8 rows and the tied head on MLX qmm as before macmu100), mlxlm = mlx-lm batch_generate on copies of the same prefix cache (its BatchKVCache).
Prints one JSON line per (variant, N, rep).   usage: mu95_bench.py ENGINE MODEL [P=4096] [Ns=1,2,4,8] [REPS=3] [NEW=64] [SUF=24]"""
import sys, os, time, json, copy
import mlx.core as mx
import kvwin_kl as K

eng, mdir = sys.argv[1], sys.argv[2]
P = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
Ns = [int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "1,2,4,8").split(",")]
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
NEW = int(sys.argv[6]) if len(sys.argv) > 6 else 64
SUF = int(sys.argv[7]) if len(sys.argv) > 7 else 24


def emit(**kw):
    print(json.dumps(dict(engine=eng, model=os.path.basename(mdir), P=P, new=NEW, **kw)), flush=True)


if eng == "kmlx":
    from kyojin_mlx import mlxpack, q4mma as QM, multiuser as MU
    model, tok = mlxpack.load(mdir)
    QM.install(model)
    MU.install(model)
else:
    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load(mdir)
ids = K.tokens(tok, P + max(Ns) * SUF, 0)
pre, sufs = ids[:P], [ids[P + u * SUF:P + (u + 1) * SUF] for u in range(max(Ns))]

if eng == "kmlx":
    base, _ = MU.prefill(model, pre[:-1])                     # the last prefix token is the first fed token of every suffix prefill
    users = []
    for s in sufs:
        c, lg = MU.prefill(model, [pre[-1]] + s, MU.fork_cache(base))
        users.append((c, mx.argmax(lg, axis=-1)))
    variants = os.environ.get("MU95_VARIANTS", "pool,rows").split(",")
    OLD_ENV = QM._ENV_MAX

    NAX0, TUNE_A, TUNE_B = QM.nax_on(), os.environ.get("KYOJIN_Q4MMA_TUNING"), os.environ.get("MU95_TUNING_B")

    def run(N, v):
        QM._ENV_MAX, MU.HEAD_ROUTE = ("8", False) if v.endswith("_old") else (OLD_ENV, True)   # *_old: pre-macmu100 q4mma 8-row cap + head bypass
        QM._NAX_ON[:] = [NAX0 and not v.endswith("_nonax")]                                     # *_nonax: q4nax off (macmu100 routing with MU95_TUNING_B)
        QM.load_tuning(TUNE_B if v.endswith("_nonax") and TUNE_B else TUNE_A)
        bc = MU.batch_caches([MU.fork_cache(c) for c, _ in users[:N]], prefix_len=None if v.startswith("pool") else 0)
        first = mx.concatenate([f for _, f in users[:N]])
        MU.decode_batched(model, bc, first, 4)                # warm the shapes
        t0 = time.perf_counter(); MU.decode_batched(model, bc, first, NEW); dt = time.perf_counter() - t0
        return N * (NEW - 1) / dt
else:
    base = make_prompt_cache(model)
    for s0 in range(0, P - 1, 2048):
        model(mx.array(pre[s0:min(s0 + 2048, P - 1)])[None], cache=base); mx.eval([c.state for c in base])
    variants = ["mlxlm"]

    def run(N, v):
        r = batch_generate(model, tok, [[pre[-1]] + s for s in sufs[:N]], prompt_caches=[copy.deepcopy(base) for _ in range(N)], max_tokens=NEW)
        return r.stats.generation_tps
for v in variants:
    run(2 if 2 in Ns else Ns[0], v)
for rep in range(REPS):
    for N in (Ns if rep % 2 == 0 else Ns[::-1]):
        for v in variants:
            a = run(N, v)
            emit(variant=v, N=N, rep=rep, agg=a, per_session=a / N)
            mx.clear_cache()
