"""KL gate of the current few-row routing (q4nax window on NAX chips, q4mma window, q4mma tied head on the multi-user path) vs the
pre-macmu100 routing (q4mma capped at 8 rows, tied head on MLX quantized_matmul), both on the pooled shared-prefix batch.
N users = one P-token prefix + a private TAIL-token suffix each, teacher-forced on the old-routing greedy tokens for S steps.
Floor = the macmu95 gate pair under the old routing (pool vs per-user RowsKV attention): two correct paths.
usage: mu100_kl.py MODEL P N[,N..] [S] [TAIL]"""
import sys, json
import mlx.core as mx
from kyojin_mlx import mlxpack, q4mma as QM, multiuser as MU
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
NEW, NAX = QM._ENV_MAX, QM.nax_on()


def route(new):   # new = the current routing (q4nax window first where the chip has it, q4mma window, q4mma head); old = pre-macmu100
    QM._ENV_MAX, MU.HEAD_ROUTE = (NEW, True) if new else ("8", False)
    QM._NAX_ON[:] = [NAX and new]


def run(N, prefix_len, toks):
    bc = MU.batch_caches([MU.fork_cache(c) for c in users[:N]], prefix_len=prefix_len)
    out_t, out_l, y = [], [], mx.concatenate(firsts[:N])
    for s in range(S):
        lp = MU.logits_last(model, y[:, None], bc).astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        mx.eval(lp)
        out_l.append(lp); nxt = mx.argmax(lp, axis=-1); out_t.append(nxt)
        y = toks[s] if toks is not None else nxt
    return out_t, out_l


kl = lambda A, B: mx.stack([(mx.exp(a) * (a - b)).sum(-1) for a, b in zip(A, B)])
res = {}
for N in Ns:
    route(False); t_ref, l_ref = run(N, None, None)
    _, l_rows = run(N, 0, t_ref)
    route(True); t_new, l_new = run(N, None, t_ref)
    k, kf = kl(l_ref, l_new), kl(l_ref, l_rows)
    agree = mx.stack([(a == b) for a, b in zip(t_ref, t_new)]).astype(mx.float32).mean().item()
    res[N] = dict(kl_mean=k.mean().item(), kl_max=k.max().item(), argmax_agree=agree, floor_kl_mean=kf.mean().item(), floor_kl_max=kf.max().item())
    print(f"P={P} N={N} new vs old routing KL mean {res[N]['kl_mean']:.2e} max {res[N]['kl_max']:.2e} argmax agree {agree:.3f} | "
          f"floor (old: pool vs per-user) mean {res[N]['floor_kl_mean']:.2e} max {res[N]['floor_kl_max']:.2e}", flush=True)
print(json.dumps(dict(model=mdir, P=P, S=S, TAIL=TAIL, res=res)))
