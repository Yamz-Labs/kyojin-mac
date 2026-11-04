"""(mu2_table.py: arms plain / before / now interleaved in one process, same thermal state.)
Served multi-agent cell: shared P-token prefix + N different questions through the real Engine (TQ default KV, real MTP drafter when the pack has one).
usage: mu117.py MODEL_DIR [--P 4096] [--Ns 1,2,4] [--new 64] [--depth D|-1 auto] [--reps 1] [--tokens-out F.json] [--ref F.json]
Prints per N: aggregate tok/s (all streams' generated tokens / wall from first to last token), mean per-stream tok/s, mean TTFT. --tokens-out saves the
greedy tokens per (N, stream); --ref compares them with a saved run (solo equality check)."""
import argparse, json, os, threading, time
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve.engine import Engine

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--P", type=int, default=4096); ap.add_argument("--Ns", default="1,2,4")
ap.add_argument("--new", type=int, default=64); ap.add_argument("--depth", type=int, default=-1); ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--force", action="store_true", help="depth_override = --depth (no depth policy)"); ap.add_argument("--tokens-out"); ap.add_argument("--ref"); ap.add_argument("--no-prime", action="store_true"); ap.add_argument("--arms", default="plain,before,now")
a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
backend, cfg = load_backend(a.model)
tok = AutoTokenizer.from_pretrained(a.model)
depth = (backend.default_mtp_depth if backend.supports_spec else 0) if a.depth < 0 else a.depth
if depth > 0 and not backend.supports_spec:
    depth = 0
base = ("The history of the Roman Empire begins with the founding of Rome in 753 BC and the long struggle between patricians and plebeians. "
        "Over the centuries the republic expanded across Italy, then the Mediterranean, absorbing Carthage, Greece and Egypt. ")
prefix = (tok.encode(base * 200))[:a.P]
Q = ["Question: summarise the causes of the fall of the Republic in five sentences.", "Question: list the main Punic War battles with dates.",
     "Question: explain how the Roman legions were organised.", "Question: write a short poem about the Appian Way.",
     "Question: who were the Gracchi brothers and why do they matter?", "Question: describe daily life of a Roman farmer.",
     "Question: compare Caesar and Augustus in a table.", "Question: what is the Pax Romana? Answer briefly.",
     "Question: explain Roman concrete.", "Question: how did the Senate work?", "Question: describe the Colosseum games.",
     "Question: why did Carthage lose?", "Question: write a haiku on Pompeii.", "Question: what were the aqueducts for?",
     "Question: name five Roman emperors and one fact each.", "Question: how did Rome tax its provinces?"]
Ns = [int(v) for v in a.Ns.split(",")]
ref = json.load(open(a.ref)) if a.ref else {}
saved = {}



from kyojin_mlx.serve import q35mlx as QM
auto_depth = depth
def run_cell(N):
    eng = Engine(backend, sessions=max(N, 1), ctx=8192, chunk=256, block=512, prefix_bytes=1 << 30, mtp_depth=cur_depth, depth_override=None, log=lambda *x: None)
    res = {}

    def worker(i, ids):
        t_sub = time.perf_counter()
        h = eng.submit(ids, max_new=a.new, stop_ids=())
        toks, times = [], []
        while True:
            ev = h.events.get()
            if ev[0] == "tok":
                toks.append(ev[1]); times.append(time.perf_counter())
            else:
                res[i] = dict(toks=toks, times=times, t_sub=t_sub, st=ev[2]); return

    def clients():
        if not a.no_prime:                                           # the shared prefix is prefilled once and cached (a prior agent turn)
            w = threading.Thread(target=worker, args=(-1, prefix + tok.encode(" Prime."))); w.start(); w.join()
            res.pop(-1)
        ths = [threading.Thread(target=worker, args=(i, prefix + tok.encode(" " + Q[i]))) for i in range(N)]
        for t in ths: t.start()
        for t in ths: t.join()
        eng.stop()
    d = threading.Thread(target=clients); d.start(); eng.run(); d.join()
    print("  engine:", {k: v for k, v in eng.status().items() if k in ("epochs", "decode_ticks", "spec_rounds", "spec_drafted", "spec_accepted", "plain_rounds") or "spec" in k or k == "tuner_ms" or "draft" in k or "plain" in k}, flush=True)
    t_first = min(r["times"][0] for r in res.values()); t_last = max(r["times"][-1] for r in res.values())
    ntok = sum(len(r["toks"]) - 1 for r in res.values())
    per = [(len(r["toks"]) - 1) / (r["times"][-1] - r["times"][0]) for r in res.values()]
    ttft = [r["times"][0] - r["t_sub"] for r in res.values()]
    return dict(N=N, depth=depth, agg=(ntok + N) / (t_last - min(r["t_sub"] for r in res.values())), agg_dec=ntok / (t_last - t_first), per=sum(per) / N, ttft=sum(ttft) / N,
                st=list(res.values())[0]["st"]), {i: r["toks"] for i, r in res.items()}




def arm(name):
    global cur_depth
    be = backend
    if name == "plain":
        cur_depth = 0
    elif name == "before":                                        # the previous commit: fixed depth 2 for 2-4 sessions, per-session attention, 16 rows
        cur_depth = auto_depth
        QM._ATT_SHARED = QM._ATT_GROUP = False
        QM._MAX_SPEC_SESSIONS = 4
        be.multi_auto, be.multi_depth, be.max_spec_rows = False, 2, 16
    else:
        cur_depth = auto_depth
        QM._ATT_SHARED = QM._ATT_GROUP = True
        QM._MAX_SPEC_SESSIONS = 16
        be.multi_auto, be.multi_depth, be.max_spec_rows = True, None, 32


cur_depth = 0
arms = a.arms.split(",")
for rep in range(a.reps):
    for N in Ns:
        line = []
        for nm in arms:
            arm(nm)
            r, toks = run_cell(N)
            line.append("%s %.1f/%.1f" % (nm, r["agg"], r["per"]))
        print("N=%d agg/per: %s" % (N, "  ".join(line)), flush=True)
