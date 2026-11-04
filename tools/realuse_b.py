"""REAL-USE scenario B (multi-agent) on the Mac server: one main agent + 4 sub-agents of the same model sharing system + tools + parent context (a long prefix) plus
an own 1-3K task, then tool-style turns. Black box over /v1/completions, numpy + stdlib only. The server is started as in DAY1_RUNBOOK step 10 (sessions 5).
usage: realuse_b.py [--port 8080] [--prefix 16384] [--task 1500] [--decode 1500] [--sub 200] [--turns 3] [--seed 1] [--arms spawn,race,cold] [--json OUT]
                    [--vocab 4096 | --text]   (slice packs have a 4096 vocabulary: random token ids; the real DS4.1 pack: --text, prompts are English/code text)
arms (a never-seen prefix each, so the cache is cold):
  spawn  the main agent prefills prefix + its task and decodes --decode tokens; once it decodes, the 4 subs arrive at once (prefix cached: only the delta is paid).
         Then each sub alone on the hot prefix (reference: outputs, decode tok/s), then --turns - 1 tool turns (previous prompt + 300 new tokens), 4 at once vs alone.
  race   the main agent starts, the subs arrive 0.5 s later while it still prefills the same prefix (they must wait for it, not prefill it again).
  cold   4 subs at once on a never-seen prefix, no main (one leader prefills, the others reuse).
reports per request: prompt, cached tokens, server ttft (client chunks are late when a byte-level piece waits for its UTF-8 tail), decode tok/s (server), and the
main agent's largest inter-chunk gap after the spawn; subs aggregate = sum of the per-agent rates while they decode together; outputs vs solo (equal / agreement
before the first difference: bitwise with KYOJIN_ROWS_NAX=0, a fp16 rounding class with NAX on).
"""
import argparse, http.client, json, statistics, threading, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080); ap.add_argument("--prefix", type=int, default=12000); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--decode", type=int, default=300, help="main agent tokens"); ap.add_argument("--sub", type=int, default=96, help="tokens per sub turn")
ap.add_argument("--task", type=int, default=1500); ap.add_argument("--turns", type=int, default=3); ap.add_argument("--json")
ap.add_argument("--arms", default="spawn"); ap.add_argument("--vocab", type=int, default=4096); ap.add_argument("--text", action="store_true", help="text prompts instead of token ids")
ap.add_argument("--subs", type=int, default=4, help="number of sub-agents (the day-1 runner sweeps 1..4)")
a = ap.parse_args()
V = a.vocab; NS = a.subs
TOPICS = ["The history of the Roman Empire begins with the founding of Rome in 753 BC and the long struggle between patricians and plebeians. ",
          "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n    mid = len(items) // 2\n    return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))\n\n",
          "Le petit village de montagne se reveillait doucement sous la brume, tandis que les cloches de l'eglise sonnaient sept heures du matin. ",
          "Photosynthesis converts light energy into chemical energy stored in glucose, using carbon dioxide and water, and releases oxygen. "]


def text_of(g, n):
    """about n tokens (4 characters each) of text that differs for every call: numbered sections cycling through TOPICS"""
    out, k = "\n", 0
    while len(out) < 4 * n:
        out += f"[{int(g.integers(10 ** 6))}] " + TOPICS[int(g.integers(4))]; k += 1
    return out + "\n"


def get(path):
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30); c.request("GET", path); return json.loads(c.getresponse().read())


def stream(prompt, n):
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=1800)
    t0 = time.perf_counter()
    c.request("POST", "/v1/completions", json.dumps(dict(prompt=prompt, max_tokens=n, temperature=0, ignore_eos=True, stream=True,
                                                         stream_options={"include_usage": True})), {"Content-Type": "application/json"})
    r = c.getresponse(); res = dict(text="", times=[], t0=t0, usage=None, status=r.status)
    if r.status != 200:
        res["error"] = r.read().decode(); return res
    while True:
        line = r.readline()
        if not line: break
        line = line.strip()
        if not line.startswith(b"data:"): continue
        d = line[5:].strip()
        if d == b"[DONE]": break
        j = json.loads(d)
        if j.get("usage"): res["usage"] = j["usage"]
        if j.get("timings"): res["timings"] = j["timings"]
        for ch in j.get("choices", []):
            if ch.get("text"): res["text"] += ch["text"]; res["times"].append(time.perf_counter())
    c.close(); res["end"] = time.perf_counter(); res["ttft"] = res["times"][0] - t0 if res["times"] else None
    return res


def summ(r):
    u = r["usage"] or {}
    n = u.get("completion_tokens", 0); ts = r["times"]
    tps = (n - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None
    gaps = [y - x for x, y in zip(ts, ts[1:])]
    tm = r.get("timings") or {}      # server-side ttft / decode rate: the client chunks are late when a byte-level piece waits for its UTF-8 tail (random ids)
    return dict(prompt=u.get("prompt_tokens"), cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), ttft=tm.get("ttft_s", r["ttft"]), ttft_client=r["ttft"],
                queue=tm.get("queue_s"), tokens=n, tps=tm.get("predicted_per_second", tps), start=r["t0"] + tm.get("ttft_s", 0), end=r["end"],
                max_gap=max(gaps) if gaps else None, wall=r["end"] - r["t0"])


def agree(x, y):
    """fraction of the solo text reproduced before the first difference (1.0 = identical)"""
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]: n += 1
    return round(n / max(len(y), 1), 3) if x != y else 1.0


def fmt(s): return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}


def make(seed):
    g = np.random.default_rng(seed)
    r = (lambda n: text_of(g, n)) if a.text else (lambda n: g.integers(16, V - 16, n).tolist())  # noqa: E731
    P = r(a.prefix)                                   # shared system + tools + parent context
    return dict(P=P, main=P + r(400), subs=[P + r(a.task + 300 * i) for i in range(NS)], r=r)


def par(fns):
    out = [None] * len(fns)
    def w(i): out[i] = fns[i]()
    ths = [threading.Thread(target=w, args=(i,)) for i in range(len(fns))]
    [t.start() for t in ths]; [t.join() for t in ths]
    return out


OUT = dict(prefix=a.prefix, health=get("/health"))
print("server", {k: OUT["health"][k] for k in ("model", "sessions", "mtp_depth")}, "prefix", a.prefix, flush=True)

# baseline: prefill + decode of the main request alone on a never-seen prefix
W = make(a.seed * 100)
r = stream(W["main"], a.decode); OUT["alone_main"] = summ(r); print("main alone (cold prefix):", fmt(OUT["alone_main"]), flush=True)


def spawn(tag, seed, race):
    S = make(seed); res = {}
    mres = []
    def run_main(): mres.append(stream(S["main"], a.decode))
    mt = threading.Thread(target=run_main); mt.start()
    if race: time.sleep(0.5)
    else:
        while not mres:                                   # wait for the main agent's first token (prefix prefilled, decoding)
            time.sleep(0.05)
            if get("/health")["decoding"] >= 1: break
        time.sleep(1.0)
    t_spawn = time.perf_counter()
    subs = par([lambda i=i: stream(S["subs"][i], a.sub) for i in range(NS)])
    mt.join(); m = mres[0]
    ss = [summ(r) for r in subs]
    lo = max(x["start"] for x in ss); hi = min(x["end"] for x in ss)
    agg_subs = sum(x["tps"] for x in ss) if hi > lo else None      # all subs decode together in [lo, hi]: the sum of the per-agent rates
    mt_after = [t for t in m["times"] if t >= t_spawn]
    gaps = [y - x for x, y in zip(m["times"], m["times"][1:])]
    gaps_after = [y - x for x, y in zip(mt_after, mt_after[1:])]
    cand = [((y - x), x - t_spawn) for x, y in zip(m["times"], m["times"][1:]) if y >= t_spawn]
    big = max(cand) if cand else None
    ms = summ(m)
    wall = max(r["end"] for r in subs) - t_spawn
    # main rate while the subs run
    dur = subs and (max(r["end"] for r in subs))
    inwin = [t for t in m["times"] if t_spawn <= t <= dur]
    main_rate = (len(inwin) - 1) / (inwin[-1] - inwin[0]) if len(inwin) > 1 else None
    d = dict(main=ms, subs=ss, spawn_wall=wall, agg_subs_tps=agg_subs, main_max_gap_after_spawn=big[0] if big else None, main_gap_at=big[1] if big else None,
             main_tps_while_subs=main_rate, main_steady_gap_median=statistics.median(gaps) if gaps else None, health=get("/health"))
    print(f"[{tag}] main: {fmt(ms)}", flush=True)
    for i, s in enumerate(ss): print(f"[{tag}] sub{i}: {fmt(s)}", flush=True)
    print(f"[{tag}] spawn wall {wall:.1f}s; subs aggregate {agg_subs and round(agg_subs, 1)} tok/s (window where all subs decode); main {main_rate and round(main_rate, 1)} tok/s while subs run; "
          f"main max gap after spawn {big and round(big[0], 2)}s (at +{big and round(big[1], 1)}s), steady median gap {d['main_steady_gap_median'] and round(d['main_steady_gap_median'], 3)}s", flush=True)
    return S, subs, d


if "spawn" in a.arms:
    S, subs, d = spawn("spawn", a.seed * 100 + 1, False)
    OUT["spawn"] = d
    # solo reference on the same (now hot) prefix, one after the other
    solo = [stream(S["subs"][i], a.sub) for i in range(NS)]
    OUT["solo_hot"] = [summ(r) for r in solo]
    for i, r in enumerate(solo): print(f"solo hot sub{i}: {fmt(summ(r))}", flush=True)
    eq = [subs[i]["text"] == solo[i]["text"] for i in range(NS)]
    OUT["equal_vs_solo"] = eq; OUT["agree_vs_solo"] = [agree(subs[i]["text"], solo[i]["text"]) for i in range(NS)]
    base = statistics.median([x["tps"] for x in OUT["solo_hot"]])
    OUT["solo_tps"] = base
    print(f"SPAWN equal to solo: {eq} (agreement before first difference {OUT['agree_vs_solo']}); solo decode {base:.1f} tok/s; subs aggregate {d['agg_subs_tps']:.1f} = x{d['agg_subs_tps'] / base:.2f}; per-agent "
          f"{[round(s['tps'], 1) for s in d['subs']]}", flush=True)
    # tool-style turns: each sub appends a tool result (300 tokens) to its previous prompt, 4 at once
    prev = [p if a.text else list(p) for p in S["subs"]]
    OUT["turns"] = []
    for t in range(2, a.turns + 1):
        new = [p + S["r"](300) for p in prev]
        rs = par([lambda i=i: stream(new[i], a.sub) for i in range(NS)])
        ts = [summ(r) for r in rs]; OUT["turns"].append(ts)
        sol = [stream(new[i], a.sub) for i in range(NS)]
        e = [rs[i]["text"] == sol[i]["text"] for i in range(NS)]
        print(f"turn {t}: cached {[x['cached'] for x in ts]} of {[x['prompt'] for x in ts]}, ttft {[round(x['ttft'], 2) for x in ts]}, equal-vs-solo {e} agree {[agree(rs[i]['text'], sol[i]['text']) for i in range(NS)]}", flush=True)
        OUT["turns"][-1] = dict(subs=ts, equal=e, agree=[agree(rs[i]["text"], sol[i]["text"]) for i in range(NS)]); prev = new

if "race" in a.arms:
    S, subs, d = spawn("race", a.seed * 100 + 2, True)
    OUT["race"] = d

if "cold" in a.arms:
    S = make(a.seed * 100 + 3)                           # 4 subs arrive at once on a never-seen prefix, no main
    t0 = time.perf_counter()
    rs = par([lambda i=i: stream(S["subs"][i], a.sub) for i in range(NS)])
    ss = [summ(r) for r in rs]; wall = time.perf_counter() - t0
    OUT["cold"] = dict(subs=ss, wall=wall)
    print(f"[cold] wall {wall:.1f}s cached {[x['cached'] for x in ss]} of {[x['prompt'] for x in ss]} ttft {[round(x['ttft'], 2) for x in ss]}", flush=True)

if a.json: json.dump(OUT, open(a.json, "w"), indent=1, default=str)
