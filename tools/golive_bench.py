"""Served A/B for real Hermes-style use: start the Kyojin server from a source tree, send chat requests WITHOUT temperature (the server defaults
apply, as for Hermes), 4 prose + 4 code prompts, report decode tok/s per request. usage: golive_bench.py TREE PACK PORT [--n 256] [--depth 3]"""
import argparse, hashlib, json, os, signal, statistics, subprocess, sys, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("tree"); ap.add_argument("pack"); ap.add_argument("port", type=int)
ap.add_argument("--n", type=int, default=256); ap.add_argument("--depth", type=int, default=3); ap.add_argument("--vlm", default=os.environ.get("KYOJIN_VLM_PKG", ""))
ap.add_argument("--temperature", type=float, default=None, help="send an explicit temperature (default: none, as Hermes)"); a = ap.parse_args()
PROSE = ["Write a long, detailed essay about the history of the Roman Empire, its institutions, its economy and its decline.",
         "Explain in detail how the immune system fights a viral infection, from first contact to long-term memory.",
         "Describe the causes, the course and the consequences of the French Revolution as a narrative for a curious reader.",
         "Write a thoughtful essay on how cities should plan for climate change over the next fifty years."]
CODE = ["Write a complete Python implementation of a red-black tree with insert, delete and search, with comments and tests.",
        "Write a Python module that implements an LRU cache, a thread-safe variant, and unit tests for both.",
        "Write a Python implementation of Dijkstra and A* on a weighted grid graph, with a small command line and tests.",
        "Write a Python class for a B-tree with insert, search and delete, with docstrings and a test suite."]
env = {**os.environ, "PYTHONPATH": a.tree + (":" + a.vlm if a.vlm else ""), "PYTHONUNBUFFERED": "1"}
log = open(f"/tmp/golive_{a.port}.log", "w")
cmd = [sys.executable, "-m", "kyojin_mlx.serve.server", a.pack, "--port", str(a.port), "--sessions", "2", "--ctx", "8192", "--mtp-depth", str(a.depth),
       "--prefill-chunk", "2048", "--kv-disk", f"/tmp/golive_kvdisk_{a.port}"]
pr = subprocess.Popen(cmd, cwd=a.tree, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("server pid", pr.pid, flush=True)
def chat(q):
    body = {"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": a.n, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    if a.temperature is not None: body["temperature"] = a.temperature
    r = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); d = json.load(urllib.request.urlopen(r, timeout=900)); dt = time.perf_counter() - t0
    chat.last = d["choices"][0]["message"]["content"]
    return d["usage"]["completion_tokens"], dt
try:
    t0 = time.time()
    while True:
        try: urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=2); break
        except Exception:
            if pr.poll() is not None: print(open(f"/tmp/golive_{a.port}.log").read()[-3000:]); raise SystemExit("server died")
            time.sleep(2)
    print(f"server up after {time.time() - t0:.0f} s", flush=True)
    chat("Say hello in one sentence.")                                  # warm
    def stats():
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/stats", timeout=5))
        return {k: v for k, v in d.items() if k.startswith("spec_") and not k.startswith("spec_t") and isinstance(v, (int, float))}
    res = {"prose": [], "code": []}
    for kind, qs in (("prose", PROSE), ("code", CODE)):
        s0 = stats()
        for q in qs:
            n, dt = chat(q); res[kind].append(n / dt); print(f"{kind}: {n} tok in {dt:.2f} s = {n / dt:.1f} tok/s", flush=True)
        s1 = stats(); print(f"  {kind} spec stats:", {k: s1[k] - s0.get(k, 0) for k in s1 if s1[k] - s0.get(k, 0)}, flush=True)
    if os.environ.get("GOLIVE_GREEDY"):                                  # same prompts with temperature 0 (greedy MTP reference on the same server)
        a.temperature = 0.0
        for kind, qs in (("prose", PROSE[:2]), ("code", CODE[:2])):
            s0 = stats(); v = []
            for q in qs:
                n, dt = chat(q); v.append(n / dt)
            s1 = stats(); print(f"GREEDY {kind}: {[round(x, 1) for x in v]} tok/s", {k: s1[k] - s0.get(k, 0) for k in s1 if s1[k] - s0.get(k, 0)}, flush=True)
    if os.environ.get("GOLIVE_MD5"):                                     # 4 greedy replies of 128 tokens: their md5 must not move with a speed lever
        a.temperature, a.n, hs = 0.0, 128, []
        for i, q in enumerate(PROSE[:2] + CODE[:2]):
            chat(q); hs.append(hashlib.md5(chat.last.encode()).hexdigest()); print(f"MD5 {i} {hs[-1]}", flush=True)
        print("MD5ALL", hashlib.md5("".join(hs).encode()).hexdigest(), flush=True)
    for kind, v in res.items(): print(f"MEDIAN {kind}: {statistics.median(v):.1f} tok/s (mean {statistics.mean(v):.1f})")
    print(subprocess.run(["grep", "-i", "gate\\|depth policy", f"/tmp/golive_{a.port}.log"], capture_output=True, text=True).stdout[-600:])
finally:
    try: os.killpg(pr.pid, signal.SIGTERM)
    except Exception: pass
    try: os.kill(pr.pid, signal.SIGTERM)
    except Exception: pass
