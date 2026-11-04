"""cost103.py CTX [REPS] [--write]: per-round cost table of the served MTP path at CTX tokens of context, one KV mode per process (KYOJIN_TQ set =
TurboQuant, else fp16), interleaved reps, ms best + median:
  round<d>   one engine spec round at depth d = 0..DMAX (draft_lazy + spec_verify_lazy + argmax + commit + settle, as Engine._spec_tick), accepted
             drafts per round from a fixed pattern clipped to d (round 0 = the plain step of the spec path)
  verify<T>  the T-row trunk pass alone (synced, attention caches trimmed back)          draft<d>  d chained MTP draft steps alone (synced, state restored)
--write stores {"round": [median ms of round0..roundDMAX], "verify": [...], "draft": [...]} under "spec" -> mode -> CTX of <pack>/tuning/<device>.json
(kyojin_mlx.tuning.write_spec): the engine's depth policy reads it (Engine spec_costs, serve: tuning.spec_costs)."""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import tuning as TN, tq as TQM
args = [a for a in sys.argv[1:] if not a.startswith("--")]
CTX = int(args[0]); REPS = int(args[1]) if len(args) > 1 else 5
DMAX = int(os.environ.get("COST103_DMAX", "5"))
MODE = "tq" if TQM.config() is not None else "fp16"
ACC = (1, 2, 2, 1, 3, 0, 2, 1)                              # accepted drafts per round (clipped to d)
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + CTX]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
L = be.lm.model.layers
cache = be.make_cache(); d = be.new_draft()
for i in range(0, len(pids), 2048): lg = be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
t0 = int(mx.argmax(lg[0]).item())
S = dict(cache=cache, d=d, bc=be.batch_spec([cache]), att=[c for l, c in zip(L, cache) if not l.is_linear], last=t0, out=[t0], n=0)
mx.eval(be.state(cache)); mx.clear_cache()
time.sleep(float(os.environ.get("COST103_IDLE", "45")))   # the fanless Air throttles right after a long prefill


def round_(s, depth):
    jobs = [(s["d"], s["last"], CTX + len(s["out"]) - 1, depth, (pids, s["out"]))]
    lh = be.draft_lazy(jobs)
    res = be.spec_verify_lazy(s["bc"], lh, [s["last"]])
    vals = mx.argmax(res.logits, axis=-1).tolist()
    drafts = lh.tolist()[0]
    a = min(ACC[s["n"] % len(ACC)], len(drafts)); s["n"] += 1
    res.commit([a], [s["d"]], [CTX + len(s["out"]) - 1])
    be.settle(s["bc"], [s["d"]])
    s["out"] += drafts[:a] + [vals[a]]; s["last"] = s["out"][-1]


def t_round(depth):
    def f(s):
        round_(s, depth)                                        # catch-up pairs of a previous depth-0 round land here, not in the timed rounds
        mx.eval(be.state(s["cache"]), be.draft_arrays(s["d"]))
        t0 = time.perf_counter()
        for _ in range(4): round_(s, depth)
        mx.eval(be.state(s["cache"]), be.draft_arrays(s["d"]))
        return (time.perf_counter() - t0) * 1e3 / 4
    return f


def t_verify(n):
    def f(s):
        t0 = time.perf_counter()
        p = be._verify(s["bc"], [mx.array([s["last"]] + [1000 + i for i in range(n - 1)], dtype=mx.int32)]); mx.eval(p.logits)
        dt = (time.perf_counter() - t0) * 1e3
        for c in s["att"]: c.trim(n)
        be.QS.CAP = None
        return dt
    return f


def t_draft(depth):
    def f(s):
        round_(s, 1)                                            # pending pairs back to one row
        mx.eval(be.state(s["cache"]), be.draft_arrays(s["d"]))
        jobs = [(s["d"], s["last"], CTX + len(s["out"]) - 1, depth, (pids, s["out"]))]
        snap = (s["d"].mc.offset, s["d"].mv, list(s["d"].pend))
        t0 = time.perf_counter(); lh = be.draft_lazy(jobs); mx.eval(*lh.arrays.values()); dt = (time.perf_counter() - t0) * 1e3
        s["d"].mc.trim(s["d"].mc.offset - snap[0]); s["d"].mv, s["d"].pend = snap[1], snap[2]
        return dt
    return f


OPS = [(f"round{k}", t_round(k)) for k in range(DMAX + 1)] + [(f"verify{k}", t_verify(k)) for k in range(1, DMAX + 2)] + \
      [(f"draft{k}", t_draft(k)) for k in range(1, DMAX + 1)]
for _ in range(3): round_(S, 3)
res = {}
for r in range(REPS + 1):
    for tag, f in (OPS if r % 2 else OPS[::-1]):
        v = f(S)
        if r: res.setdefault(tag, []).append(v)
out = {k: [round(min(v), 2), round(statistics.median(v), 2)] for k, v in res.items()}
print("COST103", MODE, CTX, json.dumps(out), flush=True)
if "--write" in sys.argv:
    med = lambda p, n, a: [out[f"{p}{k}"][1] for k in range(a, n)]
    f = TN.write_spec(T.PACK, MODE, CTX, {"round": med("round", DMAX + 1, 0), "verify": med("verify", DMAX + 2, 1), "draft": med("draft", DMAX + 1, 1)})
    print("wrote", f)
