"""sr107.py [NTOK]: step_regions fast path on the served solo step (multiuser RowsKV of one cache) and on TurboQuant caches (mactq107):
greedy decode NTOK tokens after a 700-token prefill through backend.step(backend.batch([cache])) with KYOJIN_STEP_REGIONS_WIDE on vs off,
fp16 KV and TQ6 KV (tq6w 6:6:4:256), pack of KYOJIN_Q35MLX. Prints the fast / fallback counts and the max |logit diff| (0 = bit-identical).
SR107_CTX=C: the prefill is C corpus tokens (~/rivals/corpus_ids.json) instead of 700 random ids, the WIDE-on run is teacher-forced on the WIDE-off
greedy tokens, and the mean / max KL(off || on) and top-1 agreement are printed, next to the floor KL(fp16 || TQ6) of the WIDE-off runs."""
import os, sys
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", s) for s in ("tests", "")]
os.environ.pop("KYOJIN_TQ", None)
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import step_regions as SR

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
CTX = int(os.environ.get("SR107_CTX", "0"))
if CTX:
    import json
    j = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = (j["ids"] if isinstance(j, dict) else j)[300000:300000 + CTX]
else:
    ids = [int(x) for x in mx.random.randint(1000, 30000, (700,), key=mx.random.key(7)).tolist()]
bad = 0; forced = None; ref = {}


def kl(a, b):
    la, lb = (x.astype(mx.float32) - mx.logsumexp(x.astype(mx.float32), axis=-1, keepdims=True) for x in (a, b))
    k = (mx.exp(la) * (la - lb)).sum(axis=-1)
    return k.mean().item(), k.max().item(), (mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)).astype(mx.float32).mean().item()
for kv in ("fp16", "6:6:4:256"):
    runs = {}
    for wide in (False, True):
        if kv != "fp16":
            os.environ["KYOJIN_TQ"] = kv
        c = be.make_cache(); os.environ.pop("KYOJIN_TQ", None)
        lg = be.prefill_chunk(c, ids); bc = be.batch([c]); SR.WIDE = wide; s0 = dict(SR.STATS); out = []
        for i in range(N):
            t = mx.argmax(lg, axis=-1) if forced is None else forced[i]; lg = be.step(bc, t); mx.eval(lg); out.append(lg)
            if CTX and not wide and kv == "fp16":
                ref.setdefault("tok", []).append(t)
        if CTX and not wide and kv == "fp16":
            forced = ref["tok"]
        runs[wide] = (mx.concatenate(out), {k: SR.STATS[k] - s0[k] for k in ("fast", "fallback")}, type(c[[i for i, l in enumerate(be.lm.model.layers) if not l.is_linear][0]]).__name__)
    d = mx.abs(runs[True][0] - runs[False][0]).max().item(); bad += d != 0
    print(f"SR107 kv={kv} cache={runs[True][2]} wide_off={runs[False][1]} wide_on={runs[True][1]} maxdiff={d}", flush=True)
    if CTX:
        ref[kv] = runs[False][0]
        print(f"SR107KL kv={kv} ctx={CTX} n={N} wide_on_vs_off mean/max/top1 %.2e %.2e %.4f" % kl(runs[False][0], runs[True][0]), flush=True)
if CTX:
    print(f"SR107KL floor fp16_vs_tq6 (WIDE off) mean/max/top1 %.2e %.2e %.4f" % kl(ref["fp16"], ref["6:6:4:256"]), flush=True)
print("SR107", "OK bit-identical" if not bad else "DIFF")
