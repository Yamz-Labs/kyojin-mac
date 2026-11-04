#!/usr/bin/env python3
"""gdnfuse_ab PORT: greedy decode on one server: solo short, solo 8K, 4 concurrent 8K. Prints tok/s and md5 of every text (compare md5 between flag runs)."""
import hashlib, json, random, statistics, sys, threading, urllib.request
port = sys.argv[1]
def call(body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
W = "river stone lamp orbit salt violin market engine cloud paper glass forest signal harbor copper".split()
def prompt(n, seed):
    r = random.Random(seed); return " ".join(r.choice(W) + str(r.randint(0, 99)) for _ in range(n))
def one(p, n=128):
    o = call({"prompt": p, "max_tokens": n, "temperature": 0, "ignore_eos": True})
    return o["timings"]["predicted_per_second"], hashlib.md5(o["choices"][0]["text"].encode()).hexdigest()[:10]
out = {}
for name, n in (("short", 300), ("8k", 3200)):
    p = prompt(n, 7); one(p, 16); rs = [one(p) for _ in range(3)]
    out[name] = {"med": round(statistics.median(r[0] for r in rs), 2), "all": [round(r[0], 1) for r in rs], "md5": rs[0][1]}
ps = [prompt(3200, 20 + i) for i in range(4)]
for _ in range(2):
    res = [None] * 4
    def w(i): res[i] = one(ps[i])
    ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]; [t.start() for t in ts]; [t.join() for t in ts]
out["n4"] = {"agg": round(sum(r[0] for r in res), 1), "each": [round(r[0], 1) for r in res], "md5": [r[1] for r in res]}
print(json.dumps(out))
