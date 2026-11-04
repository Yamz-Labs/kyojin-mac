"""served102.py CTX MODES: the served MTP decode path in ONE process (the server's Engine thread, no HTTP), to split the served round time.
One prime request (CTX-token document + question, 4 tokens), PAUSE seconds idle, then one cached request (prefix snapshot + tail prefill +
96 greedy tokens at depth 4) per mode, in the order given (comma list):
  a        as served
  clear    mx.clear_cache() before the request (the buffer cache the prefill left behind)
  cNNN     MLX buffer cache limit NNN MB from this request on (the server's --cache-limit-mb, default 256)
  settle1  the round's rollback + draft state evaluated every round instead of every 16
Per request: rounds, ms/round (best quarter mean / median of the tick wall time), spec_t_* split, MLX active / cache / peak GB, OS pageins /
pageouts and swap during the decode. KV mode from the server flags: TQ=6:6:4:256 env (default fp16). D102_PACK = the pack."""
import os, sys, json, time, statistics, subprocess, threading
KYO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, KYO)
HOME = os.path.expanduser("~")
CTX = int(sys.argv[1]); MODES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["a"]
PACK = os.environ.get("D102_PACK", HOME + "/rivals/models/Qwen3.5-9B-mlx4"); PAUSE = float(os.environ.get("PAUSE", "45"))
if os.environ.get("TQ"): os.environ["KYOJIN_TQ"] = os.environ["TQ"]
import mlx.core as mx
from transformers import AutoTokenizer
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve.engine import Engine, default_kv_budget

def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {l.split(":")[0].strip(): int(l.split(":")[1].strip().rstrip(".")) for l in out.splitlines()[1:] if ":" in l}
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()
    return d.get("Pageins", 0), d.get("Pageouts", 0), float(sw[5].rstrip("M").replace(",", "."))

tok = AutoTokenizer.from_pretrained(PACK)
ids = json.load(open(HOME + "/rivals/corpus_ids.json")); ids = ids["ids"] if isinstance(ids, dict) else ids
doc = tok.decode(ids[300000:300000 + CTX - 120])
Q = "Explain how a hash map handles collisions and compare chaining with open addressing."
pids = tok.apply_chat_template([{"role": "user", "content": doc + "\n\n---\nIgnore the text above. " + Q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
pids = tok.encode(pids, add_special_tokens=False)
if os.environ.get("D102_CACHE_MB"): mx.set_cache_limit(int(os.environ["D102_CACHE_MB"]) << 20)   # the server sets --cache-limit-mb before the load
be, cfg = load_backend(PACK, log=lambda *a: None)
print(f"loaded peak {mx.get_peak_memory() / 2**30:.2f} GB, KYOJIN_TQ={os.environ.get('KYOJIN_TQ')}, prompt {len(pids)}", flush=True)
eng = Engine(be, sessions=2, ctx=CTX + 400, chunk=2048, block=2048, prefix_bytes=2048 << 20, mtp_depth=4, kv_budget=default_kv_budget(),
             row_cost=getattr(be, "row_cost", 0.4) or 0.4, draft_cost=getattr(be, "draft_cost", 0.15) or 0.15, log=print)
TICKS = []
_st = eng._spec_tick
PK = []
def timed():
    mx.reset_peak_memory(); t0 = time.perf_counter(); _st(); TICKS.append(time.perf_counter() - t0)
    PK.append((mx.get_peak_memory() - mx.get_active_memory()) / 2**20)     # transient MB of the round above what stays live
eng._spec_tick = timed
SETTLE1 = [False]
_se = be.settle
def settle(bc, drafts=()):
    if SETTLE1[0]:
        mx.eval(*[a for c in bc.caches for a in be.state(c)], *[a for d in drafts for a in be.draft_arrays(d)])
    return _se(bc, drafts)
be.settle = settle
SPLIT = {"wait": 0.0, "draft": 0.0, "dbuild": 0.0, "vbuild": 0.0}             # D102_SPLIT=1: host time of draft_lazy / spec_verify_lazy, and the GPU wait carried into the round
if os.environ.get("D102_SPLIT"):
    _dl, _vl = be.draft_lazy, be.spec_verify_lazy
    def draft_lazy(jobs):
        t0 = time.perf_counter(); mx.synchronize(); t1 = time.perf_counter(); r = _dl(jobs); t2 = time.perf_counter()
        SPLIT["wait"] += t1 - t0; SPLIT["draft"] += t2 - t1; return r
    def verify_lazy(*a):
        t0 = time.perf_counter(); r = _vl(*a); SPLIT["vbuild"] += time.perf_counter() - t0; return r
    be.draft_lazy, be.spec_verify_lazy = draft_lazy, verify_lazy
    _da = be._draft_arrays
    def draft_arrays(jobs):
        t0 = time.perf_counter(); r = _da(jobs); SPLIT["dbuild"] += time.perf_counter() - t0; return r
    be._draft_arrays = draft_arrays

def run(max_new):
    h = eng.submit(pids, max_new=max_new)
    while True:
        ev = h.events.get()
        if ev[0] == "done": return ev[2]

def client():                                     # the server's layout: requests on another thread, the engine loop on the thread that loaded the model
    t0 = time.time(); st = run(4); print(f"prime {time.time() - t0:.1f}s ttft {st['ttft_s']:.2f}", flush=True)
    time.sleep(PAUSE)
    for m in MODES:
        if m == "clear": mx.clear_cache()
        if m.startswith("c") and m[1:].isdigit(): mx.set_cache_limit(int(m[1:]) << 20); mx.clear_cache()   # cNNN: the MLX buffer cache limit (MB) for this request
        SETTLE1[0] = m == "settle1"
        TICKS.clear(); PK.clear(); [SPLIT.__setitem__(k, 0.0) for k in SPLIT]; s0 = dict(eng.stats); v0 = vm()
        st = run(96); print("  stats", {k: v for k, v in eng.stats.items() if not k.startswith("spec_t")}, flush=True) if os.environ.get("D102_DBG") else None
        v1 = vm(); s1 = eng.stats
        n = len(TICKS); tk = sorted(TICKS)
        d = {k: (s1[k] - s0.get(k, 0)) for k in ("spec_rounds", "spec_drafted", "spec_accepted", "spec_t_draft", "spec_t_verify", "spec_t_commit")}
        r = d["spec_rounds"] or 1
        row = (f"{m:8s} rounds {n} acc {d['spec_accepted']}/{d['spec_drafted']} ms/round best-q {1e3 * statistics.mean(tk[:max(1, n // 4)]):.1f} med {1e3 * statistics.median(tk):.1f} "
               f"draft {1e3 * d['spec_t_draft'] / r:.1f} verify {1e3 * d['spec_t_verify'] / r:.1f} commit {1e3 * d['spec_t_commit'] / r:.1f} ttft {st['ttft_s']:.2f}s "
               f"tok/s {95 / max(st['total_s'] - st['ttft_s'], 1e-9):.1f} | active {mx.get_active_memory() / 2**30:.2f} cache {mx.get_cache_memory() / 2**30:.2f} round transient max {max(PK):.0f} MB "
               f"pagein {v1[0] - v0[0]} pageout {v1[1] - v0[1]} swap {v0[2]:.0f}->{v1[2]:.0f}M")
        if os.environ.get("D102_SPLIT"): row += " | host split ms/round " + " ".join(f"{k} {1e3 * v / r:.2f}" for k, v in SPLIT.items())
        print(row, flush=True); rows.append(row)
        if os.environ.get("D102_TICKS"): print("  ticks ms", [round(1e3 * t) for t in TICKS], "\n  transient MB", [round(x) for x in PK], flush=True)
    eng.stop()
rows = []
threading.Thread(target=client, daemon=True).start()
eng.run()
print("SUMMARY", os.environ.get("KYOJIN_TQ") or "fp16", CTX); [print(" ", r) for r in rows]
