"""Black-box checks of the multi-session server (tools/mac_serve.sh) over HTTP, stdlib only.
usage: serve_check.py [--port 8080] [--model-dir DIR] PHASE ...
phases:
  speed   1 stream alone (x4, one after the other) vs 4 parallel streams: aggregate decode tok/s in the window where all 4 decode, TTFT, every
          parallel output equal to its solo output (greedy, ignore_eos)
  burst   4 chat requests at once that share a 2K-token system prompt: cached_tokens, TTFT, outputs (--save FILE / --compare FILE: outputs
          of a server without prefix cache must be equal)
  tool    a tool-call round trip: the model calls get_weather, the tool answer comes back, the model answers
  hol     head of line: 3 streams decode while a 4K-token prompt arrives; largest inter-token gap of the 3, TTFT of the big one
  stream  streamed vs not streamed, stop strings, usage, max_tokens absent, bad requests
"""
import argparse, http.client, json, statistics, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("phases", nargs="+"); ap.add_argument("--port", type=int, default=8080); ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--n", type=int, default=160, help="tokens per stream in the speed phase")
ap.add_argument("--save"); ap.add_argument("--compare"); ap.add_argument("--json")
a = ap.parse_args()
OUT = {}

TOPICS = ["The history of the Roman Empire begins with the founding of Rome in 753 BC and the long struggle between patricians and plebeians. ",
          "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n    mid = len(items) // 2\n    return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))\n\n",
          "Le petit village de montagne se reveillait doucement sous la brume, tandis que les cloches de l'eglise sonnaient sept heures du matin. ",
          "Photosynthesis converts light energy into chemical energy stored in glucose, using carbon dioxide and water, and releases oxygen. "]


def post(path, body, timeout=600):
    c = http.client.HTTPConnection(a.host, a.port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, (json.loads(data) if data else None)


def get(path):
    c = http.client.HTTPConnection(a.host, a.port, timeout=30)
    c.request("GET", path)
    r = c.getresponse()
    return json.loads(r.read())


def stream(path, body, rec=None):
    """POST a streaming request. Returns dict(text, times=[t of each text chunk], t0, ttft, usage, finish, timings, chunks)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    c = http.client.HTTPConnection(a.host, a.port, timeout=900)
    t0 = time.perf_counter()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    res = dict(text="", reasoning="", times=[], t0=t0, usage=None, finish=None, timings=None, calls={}, status=r.status)
    if r.status != 200:
        res["error"] = r.read().decode()
        return res
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        d = line[5:].strip()
        if d == b"[DONE]":
            break
        j = json.loads(d)
        if j.get("usage"):
            res["usage"] = j["usage"]
        for ch in j.get("choices", []):
            if ch.get("finish_reason"):
                res["finish"] = ch["finish_reason"]
                res["timings"] = j.get("timings")
            delta = ch.get("delta") or {}
            piece = delta.get("content") if "delta" in ch else ch.get("text")
            if piece:
                res["text"] += piece
                res["times"].append(time.perf_counter())
            if delta.get("reasoning_content"):
                res["reasoning"] += delta["reasoning_content"]
            for tc in delta.get("tool_calls", []):
                cur = res["calls"].setdefault(tc["index"], {"name": "", "arguments": ""})
                f = tc.get("function", {})
                cur["name"] += f.get("name", "") or ""
                cur["arguments"] += f.get("arguments", "") or ""
    c.close()
    res["end"] = time.perf_counter()
    res["ttft"] = (res["times"][0] - t0) if res["times"] else None
    return res


def prompt_text(i, approx_tokens):
    return (TOPICS[i % 4] * 400)[: approx_tokens * 4]


def ntok(text):
    st, j = post("/v1/tokenize", {"prompt": text})
    return j["count"]


def wait_ready():
    for _ in range(120):
        try:
            return get("/health")
        except Exception:
            time.sleep(1)
    sys.exit("server not reachable")


h = wait_ready()
print("server:", {k: h[k] for k in ("model", "family", "sessions", "peak_gb")}, flush=True)


# ------------------------------------------------------------------------------------------------ speed
def phase_speed():
    n = a.n
    prompts = [prompt_text(i, 500 + 40 * i) for i in range(4)]
    body = lambda i: dict(prompt=prompts[i], max_tokens=n, temperature=0, ignore_eos=True)      # noqa: E731
    stream("/v1/completions", dict(body(0), max_tokens=8))                                      # warm-up
    solo = []
    for i in range(4):
        r = stream("/v1/completions", body(i))
        solo.append(r)
    rates1 = [(len(r["times"]) - 1) / (r["times"][-1] - r["times"][0]) for r in solo]
    print(f"N=1: decode {statistics.median(rates1):.1f} tok/s (per stream {[round(x, 1) for x in rates1]}), "
          f"ttft {[round(r['ttft'], 2) for r in solo]} s, prompt tokens {[r['usage']['prompt_tokens'] for r in solo]}", flush=True)
    res = [None] * 4

    def worker(i):
        res[i] = stream("/v1/completions", body(i))
    rounds = []
    for rd in range(3):
        ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        [t.start() for t in ths]; [t.join() for t in ths]
        lo = max(r["times"][0] for r in res)                  # all 4 are decoding from here
        hi = min(r["times"][-1] for r in res)                  # until the first one ends
        toks = sum(sum(1 for t in r["times"] if lo <= t <= hi) for r in res)
        agg = toks / (hi - lo)
        same = [res[i]["text"] == solo[i]["text"] for i in range(4)]
        rounds.append(dict(agg=agg, same=same, ttft=[r["ttft"] for r in res], window=hi - lo))
        print(f"N=4 round {rd}: aggregate {agg:.1f} tok/s over a {hi - lo:.1f} s window ({agg / 4:.1f}/user), ttft {[round(r['ttft'], 2) for r in res]} s, "
              f"outputs equal to solo {same}", flush=True)
    base = statistics.median(rates1)
    best = statistics.median([r["agg"] for r in rounds])
    print(f"SPEED: N=1 {base:.1f} tok/s; N=4 aggregate {best:.1f} tok/s = x{best / base:.2f}; all outputs equal to solo: {all(all(r['same']) for r in rounds)}")
    OUT["speed"] = dict(n1=base, n4_aggregate=best, ratio=best / base, rounds=rounds, equal=all(all(r["same"]) for r in rounds))


# ------------------------------------------------------------------------------------------------ burst
def phase_burst():
    sysmsg = "You are a careful research assistant working in a team. Shared context follows.\n" + prompt_text(0, 2000)
    cnt = ntok(sysmsg)
    qs = ["Summarise the context in one sentence.", "List two names that appear in the context.", "Translate the first sentence into French.",
          "What is the topic of the context? Answer with one word."]
    print(f"shared system prompt: {cnt} tokens", flush=True)
    res = [None] * 4

    def worker(i):
        res[i] = stream("/v1/chat/completions", dict(messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": qs[i]}],
                                                     max_tokens=24, temperature=0, chat_template_kwargs={"enable_thinking": False}))
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    t0 = time.perf_counter()
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t0
    for i, r in enumerate(res):
        u = r["usage"]
        print(f"  req {i}: prompt {u['prompt_tokens']} cached {u['prompt_tokens_details']['cached_tokens']} ttft {r['ttft']:.2f}s total {r['end'] - r['t0']:.2f}s "
              f"text {r['text'][:60]!r}", flush=True)
    outs = [r["text"] for r in res]
    st = get("/v1/stats")
    print(f"BURST: wall {wall:.2f}s; cached tokens {[r['usage']['prompt_tokens_details']['cached_tokens'] for r in res]}; prefix hits {st['prefix_hits']} "
          f"stored {st['prefix_entries']} entries ({st['prefix_bytes'] / 2 ** 20:.0f} MB)")
    OUT["burst"] = dict(cached=[r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in res], ttft=[r["ttft"] for r in res], wall=wall, outputs=outs,
                        prompt=[r["usage"]["prompt_tokens"] for r in res])
    if a.save:
        json.dump(outs, open(a.save, "w"))
    if a.compare:
        ref = json.load(open(a.compare))
        print("BURST outputs equal to the reference run:", ref == outs)
        OUT["burst"]["equal_ref"] = ref == outs


# ------------------------------------------------------------------------------------------------ tool
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather of a city.",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                                                           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                                                          "required": ["city"]}}}]


def phase_tool():
    msgs = [{"role": "user", "content": "Call the get_weather function with city=Paris. Reply only with the function call."}]
    kw = dict(tools=TOOLS, temperature=0, max_tokens=150, chat_template_kwargs={"enable_thinking": False})
    st, j = post("/v1/chat/completions", dict(messages=msgs, **kw))
    m = j["choices"][0]["message"]
    print("round 1 finish:", j["choices"][0]["finish_reason"], "| content:", repr(m.get("content")), "| tool_calls:", json.dumps(m.get("tool_calls")), flush=True)
    s = stream("/v1/chat/completions", dict(messages=msgs, **kw))
    print("round 1 streamed: finish", s["finish"], "calls", s["calls"], "text", repr(s["text"]), flush=True)
    ok1 = bool(m.get("tool_calls")) and m["tool_calls"][0]["function"]["name"] == "get_weather"
    ok_stream = bool(s["calls"]) and s["calls"][0]["name"] == "get_weather" and json.loads(s["calls"][0]["arguments"]) == json.loads(m["tool_calls"][0]["function"]["arguments"])
    ok2 = False
    if ok1:
        call = m["tool_calls"][0]
        msgs2 = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": [call]},
                        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"city": "Paris", "temperature": 18, "unit": "celsius", "sky": "sunny"})},
                        {"role": "user", "content": "Thanks. Now tell me the weather in one sentence."}]    # the 2B model repeats the call without this nudge
        st, j2 = post("/v1/chat/completions", dict(messages=msgs2, **dict(kw, tools=TOOLS)))
        m2 = j2["choices"][0]["message"]
        print("round 2:", j2["choices"][0]["finish_reason"], repr(m2.get("content")), flush=True)
        ok2 = "18" in (m2.get("content") or "") and not m2.get("tool_calls")
    print(f"TOOL: call parsed {ok1}; streamed call equal {ok_stream}; final answer uses the tool result {ok2}")
    OUT["tool"] = dict(call=ok1, stream=ok_stream, answer=ok2, message=m)


# ------------------------------------------------------------------------------------------------ hol
def phase_hol():
    n = 1000
    res = [None] * 3

    def worker(i):
        res[i] = stream("/v1/completions", dict(prompt=prompt_text(i, 300), max_tokens=n, temperature=0, ignore_eos=True))
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    [t.start() for t in ths]
    time.sleep(3.0)                                             # the 3 are decoding
    big = f"[{time.time()}] " + prompt_text(3, 6000)                  # unique head: no prefix hit
    t_big = time.perf_counter()
    rb = stream("/v1/completions", dict(prompt=big, max_tokens=8, temperature=0, ignore_eos=True))
    t_big_end = time.perf_counter()
    [t.join() for t in ths]
    gaps_during, gaps_all = [], []
    for r in res:
        ts = r["times"]
        for x, y in zip(ts, ts[1:]):
            gaps_all.append(y - x)
            if x >= t_big and y <= t_big_end + 0.5:
                gaps_during.append(y - x)
    steady = statistics.median(gaps_all)
    worst = sorted(((y - x, x - t_big) for r in res for x, y in zip(r["times"], r["times"][1:])), reverse=True)[:6]
    print("  largest gaps (ms, seconds after the big prompt arrived):", [(round(g * 1e3), round(t, 1)) for g, t in worst], f"big prefill took {rb['ttft']:.2f}s")
    print(f"HOL: 4K-token prompt ({rb['usage']['prompt_tokens']} tokens) ttft {rb['ttft']:.2f}s while 3 streams decode; steady inter-token {steady * 1e3:.0f} ms (median), "
          f"during the big prefill: median {statistics.median(gaps_during) * 1e3:.0f} ms, max {max(gaps_during) * 1e3:.0f} ms ({len(gaps_during)} gaps); max overall {max(gaps_all) * 1e3:.0f} ms")
    OUT["hol"] = dict(big_prompt=rb["usage"]["prompt_tokens"], big_ttft=rb["ttft"], steady_ms=steady * 1e3, max_gap_ms=max(gaps_during) * 1e3 if gaps_during else None,
                      median_gap_during_ms=statistics.median(gaps_during) * 1e3 if gaps_during else None)


# ------------------------------------------------------------------------------------------------ stream / API
def phase_stream():
    msgs = [{"role": "user", "content": "Write the numbers from one to twenty in words, comma separated."}]
    kw = dict(messages=msgs, temperature=0, max_tokens=60, chat_template_kwargs={"enable_thinking": False})
    st, j = post("/v1/chat/completions", kw)
    s = stream("/v1/chat/completions", kw)
    ok = j["choices"][0]["message"]["content"] == s["text"] and j["usage"]["completion_tokens"] == s["usage"]["completion_tokens"]
    print(f"non-stream == stream: {ok} ({j['usage']['completion_tokens']} tokens, finish {j['choices'][0]['finish_reason']}/{s['finish']}) usage {j['usage']}")
    full = j["choices"][0]["message"]["content"]
    marker = "5, "
    st, js = post("/v1/chat/completions", dict(kw, stop=[marker]))
    cut = js["choices"][0]["message"]["content"]
    ss = stream("/v1/chat/completions", dict(kw, stop=[marker]))
    ok_stop = (marker in full and marker not in cut and js["choices"][0]["finish_reason"] == "stop" and cut == full.split(marker)[0].rstrip()
               and ss["text"] == cut and ss["finish"] == "stop")
    print(f"stop string {marker!r}: {ok_stop} -> {cut!r} (stream equal {ss['text'] == cut})")
    st, jl = post("/v1/chat/completions", dict(kw, max_tokens=5))
    ok_len = jl["choices"][0]["finish_reason"] == "length" and jl["usage"]["completion_tokens"] == 5
    print(f"max_tokens=5: finish {jl['choices'][0]['finish_reason']} completion_tokens {jl['usage']['completion_tokens']} {ok_len}")
    st, jn = post("/v1/chat/completions", dict(messages=[{"role": "user", "content": "Reply with the single word: yes"}], temperature=0, chat_template_kwargs={"enable_thinking": False}))
    print(f"no max_tokens: finish {jn['choices'][0]['finish_reason']} completion_tokens {jn['usage']['completion_tokens']} -> {jn['choices'][0]['message']['content']!r}")
    bad = [post("/v1/chat/completions", {"messages": []})[0], post("/v1/chat/completions", {"messages": msgs, "n": 2})[0],
           post("/v1/chat/completions", {"messages": msgs, "temperature": "x"})[0], post("/v1/nope", {})[0],
           post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x " * 40000}]})[0]]
    print("bad requests -> status", bad, "(expect 400 400 400 404 400)")
    models = get("/v1/models")
    print("models:", models["data"][0]["id"], "ctx", models["data"][0]["max_model_len"])
    OUT["stream"] = dict(same=ok, stop=ok_stop, length=ok_len, bad=bad)


for ph in a.phases:
    print(f"=== {ph}", flush=True)
    {"speed": phase_speed, "burst": phase_burst, "tool": phase_tool, "hol": phase_hol, "stream": phase_stream}[ph]()
st = get("/v1/stats")
print("final stats:", {k: st[k] for k in ("peak_gb", "active_gb", "prefix_hits", "prefix_misses", "max_decode_tick_ms")})
OUT["stats"] = st
if a.json:
    json.dump(OUT, open(a.json, "w"), indent=1, default=str)
