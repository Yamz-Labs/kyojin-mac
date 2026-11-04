"""Per-stage GPU ms of the batched verify round (Qwen3.5 mlx4), N sessions x (1+D) rows, sync after each stage; weight-byte roof per stage.
usage: mu4_ops.py MODEL [N] [D] [P]"""
import sys, os, time, collections
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve import q35mlx as QM
from mlx_lm.models.base import create_attention_mask
mdir, N, D, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
be, cfg = load_backend(mdir)
tok = AutoTokenizer.from_pretrained(mdir)
ids = tok.encode("The history of the Roman Empire begins with the founding of Rome and the long struggle between patricians and plebeians. " * 200)[:P]
base = be.make_cache(); d0 = be.new_draft()
for s in range(0, P, 256):
    be.prefill_chunk(base, ids[s:s + 256], draft=d0)
sufs = [tok.encode(" Question number %d: explain how the legions were organised." % (i * 7)) for i in range(N)]
users = []
for sf in sufs:
    c = be.fork(base); be.prefill_chunk(c, sf[:-1], draft=be.draft_restore(be.draft_snapshot(d0))); users.append(c)
mx.eval(*[a for c in users for a in be.state(c)])
import random; random.seed(1)
toks = [[sf[-1]] + [random.randrange(1000, 20000) for _ in range(D)] for sf in sufs]
xs = [mx.array(t, dtype=mx.int32) for t in toks]
lm = be.lm.model
def nb(m):
    return sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["x"]).tree_flatten(m.parameters()))
T = collections.defaultdict(float); B = collections.defaultdict(int); C = collections.Counter()
SYNC = os.environ.get("SYNC", "1") == "1"
SKIP = set(os.environ.get("SKIP", "").split(","))
def stage(name, f, nbytes=0, like=None):
    if name in SKIP and like is not None:
        return like
    t0 = time.perf_counter(); r = f()
    if SYNC: mx.eval(r)
    T[name] += time.perf_counter() - t0; B[name] += nbytes; C[name] += 1
    return r
def run():
    bc = be.batch_spec([be.fork(c) for c in users]); mx.eval(*[a for c in bc.caches for a in be.state(c)])
    t0 = time.perf_counter()
    sizes = [int(x.shape[0]) for x in xs]; S = sizes[0]; g = len(xs); caches = bc.caches; caps = [[] for _ in xs]
    h = lm.embed_tokens(mx.concatenate(xs)[None]); fa = lm.fa_idx
    for li, layer in enumerate(lm.layers):
        x = stage("norm_in", lambda: layer.input_layernorm(h))
        mod = layer.linear_attn if layer.is_linear else layer.self_attn
        names = QM._GDN_IN if layer.is_linear else QM._ATT_IN
        ys = {}
        for n in names:
            ys[n] = stage("lin_in_" + ("gdn" if layer.is_linear else "att"), (lambda n=n: getattr(mod, n)(x)), nb(getattr(mod, n)))
        outm = getattr(mod, QM._OUT[layer.is_linear])
        if layer.is_linear:
            o = stage("core_gdn", lambda: be._gdn_group(mod, li, ys, 0, g, S, caches, caps), like=ys["in_proj_z"])
        else:
            o = stage("core_att", lambda: be._attn_group(mod, li, ys, 0, g, S, x.dtype, caches, create_attention_mask, fa), like=ys["k_proj"].repeat(1,1)[..., :1].repeat(1, 1) if False else mx.zeros((1, g * S, mod.num_attention_heads * 256), dtype=x.dtype))
        o = stage("lin_out", lambda: outm(o), nb(outm))
        h = stage("resid1", lambda: h + o)
        m = stage("norm_post", lambda: layer.post_attention_layernorm(h))
        mm = stage("mlp", lambda: layer.mlp(m), nb(layer.mlp), like=m)
        h = stage("resid2", lambda: h + mm)
    h = stage("norm_f", lambda: lm.norm(h))
    r = stage("head", lambda: be._head(h)[0].astype(mx.float32), 0, like=None)
    mx.eval(r)
    return time.perf_counter() - t0
run(); T.clear(); B.clear(); C.clear()
R = 7
ts = sorted(run() for _ in range(R)); tot = ts[R // 2] * 1e3
print("N=%d D=%d rows=%d sync=%s skip=%s total(median) %.1f ms" % (N, D, N * (D + 1), SYNC, sorted(SKIP), tot))
if SYNC:
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
        print("STAGE %-14s calls %3d  %6.2f ms" % (k, C[k] // R, v / R * 1e3))
