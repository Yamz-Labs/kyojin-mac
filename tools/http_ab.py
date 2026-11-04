"""HTTP A/B of the Kyojin server: plain (--mtp-depth 0) vs MTP on the same chat prompts, interleaved rounds, medians.
usage: http_ab.py PACK [--rounds 2] [--n 96] [--prompts 4] [--depth 3] [--port 8391] [--extra "..."]"""
import argparse, json, os, signal, statistics, subprocess, sys, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--n", type=int, default=96)
ap.add_argument("--prompts", type=int, default=4); ap.add_argument("--depth", type=int, default=3); ap.add_argument("--port", type=int, default=8391)
ap.add_argument("--extra", default=""); ap.add_argument("--variants", default="", help="name:depth[:ENV=V,ENV=V];... replaces plain/mtp (two or more servers, interleaved)"); ap.add_argument("--think", action="store_true"); ap.add_argument("--conc", type=int, default=1, help="N simultaneous requests per measure: aggregate tok/s"); a = ap.parse_args()
Q = ["Explain how a hash map handles collisions and compare chaining with open addressing.", "Write a Python function that merges overlapping intervals, then explain its complexity.",
     "Summarize the causes of the French Revolution in five bullet points.", "What is the difference between TCP and UDP? Give examples of when to use each."][:a.prompts]
PY = sys.executable
import atexit
_procs = []
def _kill_tree(pid):
    kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    for k in kids: _kill_tree(int(k))
    try: os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError): pass
def _cleanup():
    """stop every server this script started (process group first, then the pid tree: macrun wrappers can refuse killpg), also on error and on SIGTERM / SIGINT"""
    while _procs:
        pr = _procs.pop()
        try: os.killpg(pr.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError): pass
        _kill_tree(pr.pid)
atexit.register(_cleanup)
for _sg in (signal.SIGTERM, signal.SIGINT): signal.signal(_sg, lambda *_: sys.exit(1))
def start(depth, port, env=None):
    cmd = [PY, "-m", "kyojin_mlx.serve.server", a.pack, "--port", str(port), "--sessions", str(max(2, a.conc)), "--ctx", "4096", "--mtp-depth", str(depth), "--prefix-cache-mb", "0", *a.extra.split()]
    if os.environ.get("HTTP_AB_MACRUN"):                                    # one server per macrun job (big packs: the guard counts each heavy process)
        cmd = [os.environ["HTTP_AB_MACRUN"], *cmd]
    pr = subprocess.Popen(cmd, env={**os.environ, **(env or {})}, stdout=open(f"/tmp/http_ab_{port}.log", "w"), stderr=subprocess.STDOUT, text=True, start_new_session=True)
    _procs.append(pr)
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1); return pr
        except Exception:
            if pr.poll() is not None: print(open(f"/tmp/http_ab_{port}.log").read()[-2000:]); raise SystemExit("server died")
            time.sleep(1)
    raise SystemExit("server start timeout")
def chat(port, q):
    body = {"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": a.n, "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": a.think}}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); d = json.load(urllib.request.urlopen(r, timeout=300)); dt = time.perf_counter() - t0
    return d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"], dt, d.get("timings", {})
V = []
for v in (a.variants.split(";") if a.variants else ["plain:0", f"mtp:{a.depth}"]):
    p = v.split(":"); V.append((p[0], int(p[1]), dict(kv.split("=") for kv in p[2].split(",")) if len(p) > 2 else {}))
names = [v[0] for v in V]
servers = {n: start(d, a.port + i, e) for i, (n, d, e) in enumerate(V)}
ports = {n: a.port + i for i, n in enumerate(names)}
try:
    for k in servers: chat(ports[k], Q[0])                      # warm
    R = {k: [] for k in names}; txt = {k: [] for k in names}; extra = []
    if a.conc > 1:
        import threading
        for r in range(a.rounds):
            for k in (names if r % 2 == 0 else names[::-1]):
                outs = [None] * a.conc
                def one(i):
                    outs[i] = chat(ports[k], Q[i % len(Q)] + (" " if i >= len(Q) else ""))
                th = [threading.Thread(target=one, args=(i,)) for i in range(a.conc)]
                t0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; dt = time.perf_counter() - t0
                R[k].append(sum(o[1] for o in outs) / dt); txt[k].append("|".join(o[0] for o in outs))
        base = names[0]
        for k in names:
            same = sum(x == y for x, y in zip(txt[base], txt[k]))
            print(f"HTTP conc={a.conc} {k}: aggregate median {statistics.median(R[k]):.1f} tok/s (x{statistics.median(R[k])/statistics.median(R[base]):.2f} vs {base}); same texts as {base} {same}/{len(txt[base])}  {[round(x, 1) for x in R[k]]}")
        raise SystemExit(0)
    for r in range(a.rounds):
        for i, q in enumerate(Q):
            for k in (names if (r + i) % 2 == 0 else names[::-1]):
                t, n, dt, tm = chat(ports[k], q)
                R[k].append(tm.get("predicted_per_second") or n / dt); txt[k].append(t)
    base = names[0]
    for k in names:
        same = sum(x == y for x, y in zip(txt[base], txt[k]))
        print(f"HTTP {k}: median {statistics.median(R[k]):.1f} tok/s (x{statistics.median(R[k])/statistics.median(R[base]):.2f} vs {base}); same text as {base} {same}/{len(txt[base])}  {[round(x, 1) for x in R[k]]}")
    print("stats", urllib.request.urlopen(f"http://127.0.0.1:{ports[names[-1]]}/v1/stats").read().decode()[:700])
finally:
    _cleanup()
