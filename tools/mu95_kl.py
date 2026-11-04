"""KL gate of the pooled shared-prefix attention (mu_attn.PoolKV) vs the per-user path (RowsKV, = solo attention per row) on a real model.
N users = one P-token prefix (fork_cache) + a private TAIL-token suffix each; teacher-forced batched decode of S steps on the RowsKV tokens.
usage: mu95_kl.py MODEL P N[,N..] [S] [TAIL]   (served mlx4 path: mlxpack + q4mma)"""
import sys, json
import mlx.core as mx
from kyojin_mlx import mlxpack, q4mma as QM, multiuser as MU, sdpa_dec as SD
import kvwin_kl as K

mdir, P = sys.argv[1], int(sys.argv[2])
Ns = [int(v) for v in sys.argv[3].split(",")]
S = int(sys.argv[4]) if len(sys.argv) > 4 else 32
TAIL = int(sys.argv[5]) if len(sys.argv) > 5 else 24
model, tok = mlxpack.load(mdir)
QM.install(model)
MU.install(model)
ids = K.tokens(tok, P + max(Ns) * TAIL, 0)
base, _ = MU.prefill(model, ids[:P])
users, firsts = [], []
for u in range(max(Ns)):
    c, lg = MU.prefill(model, ids[P + u * TAIL:P + (u + 1) * TAIL], MU.fork_cache(base))
    users.append(c); firsts.append(mx.argmax(lg, axis=-1))


def run(bc, toks):
    """Teacher-forced (toks given) or greedy (toks None) batched decode; returns (tokens per step, log-probs per step)."""
    out_t, out_l, y = [], [], mx.concatenate(firsts[:len(bc_users)])
    for s in range(S):
        lp = MU.logits_last(model, y[:, None], bc).astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        mx.eval(lp)
        out_l.append(lp)
        nxt = mx.argmax(lp, axis=-1)
        out_t.append(nxt)
        y = toks[s] if toks is not None else nxt
    return out_t, out_l


res = {}
for N in Ns:
    bc_users = users[:N]
    t_ref, l_ref = run(MU.batch_caches([MU.fork_cache(c) for c in bc_users], prefix_len=0), None)
    bc = MU.batch_caches([MU.fork_cache(c) for c in bc_users])          # shared prefix from the fork records (auto)
    npool = sum(isinstance(c, MU.MA.PoolKV) for c in bc)
    t_p, l_p = run(bc, t_ref)
    kl = mx.stack([(mx.exp(a) * (a - b)).sum(-1) for a, b in zip(l_ref, l_p)])     # [S, N]
    agree = mx.stack([(a == b) for a, b in zip(t_ref, t_p)]).astype(mx.float32).mean().item()
    r0 = SD.route                                    # noise floor: same per-user path, split-K kernel forced on every decode call (two correct kernels)
    SD.route = lambda q, k, v, cache, scale, mask, sinks=None: SD.sdpa_decode(q, k, v, scale) if q.shape[2] == 1 and isinstance(k, mx.array) else None
    _, l_f = run(MU.batch_caches([MU.fork_cache(c) for c in bc_users], prefix_len=0), t_ref)
    SD.route = r0
    klf = mx.stack([(mx.exp(a) * (a - b)).sum(-1) for a, b in zip(l_ref, l_f)])
    res[N] = dict(pool_layers=npool, kl_mean=kl.mean().item(), kl_max=kl.max().item(), argmax_agree=agree, floor_kl_mean=klf.mean().item(),
                  floor_kl_max=klf.max().item())
    print(f"P={P} N={N} pool_layers={npool} KL mean {res[N]['kl_mean']:.2e} max {res[N]['kl_max']:.2e} argmax agree {agree:.3f} | "
          f"floor (stock vs split-K per user) mean {res[N]['floor_kl_mean']:.2e} max {res[N]['floor_kl_max']:.2e}", flush=True)
print(json.dumps(dict(model=mdir, P=P, S=S, TAIL=TAIL, res=res)))
