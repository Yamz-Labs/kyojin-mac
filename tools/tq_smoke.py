"""3-session smoke of `--kv-mode tq` on a Qwen3.5 EXL3 pack: one server, three different ~1500-token natural-text prompts answered all at once
(--sessions 3, cold), then each solo, then once more (prefix cache hit). Greedy texts must be equal together vs solo vs repeat; also reports agreement with a
default-KV server.
usage: tq_smoke.py MODEL_DIR WORKDIR [--k-bits 6]"""
import http.client, json, os, signal, subprocess, sys, threading, time
import numpy as np

model, work = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
port = 8098
import pydoc_data.topics as T
_txt = "\n\n".join(T.topics[k] for k in sorted(T.topics))
prompts = [_txt[a:a + 6000] + "\n\nSummary of the text above:" for a in (0, 90000, 180000)]      # ~1500 natural-text tokens each, three different prefixes


def serve(args):
    cmd = [sys.executable, "-m", "kyojin_mlx.serve.server", model, "--port", str(port), "--sessions", "3", "--ctx", "4096", "--prefill-chunk", "256",
           "--prefix-block", "256", "--prefix-cache-mb", "512"] + args
    log = open(os.path.join(work, "smoke_server_%d.log" % time.time()), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5); c.request("GET", "/health")
            if c.getresponse().status == 200:
                return p, log.name
        except OSError:
            pass
        time.sleep(2)
    p.send_signal(signal.SIGTERM); raise SystemExit("server did not come up")


def ask(prompt):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", "/v1/completions", json.dumps(dict(prompt=prompt, max_tokens=32, temperature=0, ignore_eos=True)), {"Content-Type": "application/json"})
    j = json.loads(c.getresponse().read())
    return j["choices"][0]["text"], (j["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")


def stop(p):
    p.send_signal(signal.SIGTERM); p.wait(60)


p, logf = serve(["--kv-mode", "tq"] + extra)
res = [None] * 3
ths = [threading.Thread(target=lambda i=i: res.__setitem__(i, ask(prompts[i]))) for i in range(3)]
t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; tt = time.time() - t0          # cold, all three at once
solo = [ask(x) for x in prompts]
rep = [ask(x) for x in prompts]
stop(p)
txt = open(logf).read()
assert "TurboQuant pool" in txt, "tq not active"
p, _ = serve([])
base = [ask(x)[0] for x in prompts]; stop(p)
agree = [sum(a == b for a, b in zip(s[0], r)) / max(len(r), 1) for s, r in zip(solo, base)]
print(json.dumps(dict(extra=extra, concurrent_s=round(tt, 1), eq_solo_vs_concurrent=[s[0] == r[0] for s, r in zip(solo, res)],
                      eq_solo_vs_repeat=[s[0] == r[0] for s, r in zip(solo, rep)], cached_repeat=[r[1] for r in rep],
                      char_agree_with_default_kv=[round(a, 2) for a in agree], sample=solo[0][0][:50])))
assert all(s[0] == r[0] for s, r in zip(solo, res)) and all(s[0] == r[0] for s, r in zip(solo, rep))
print("OK")
