"""depth89.py CTX PASSES [CONFIGS]: decode tok/s over HTTP at CTX tokens of context, 4 chat questions after one shared document, n=96 greedy, thinking off.
Kyojin (this worktree) plain / MTP d3 / d4 with fp16 or TurboQuant KV vs oMLX plain / Lightning MTP d3 (q4 head). Servers run one at a time (passes alternate the order);
decode tok/s = (n-1) / (last chunk - first chunk). A config may carry server env: kfp-plain+KYOJIN_SDPA_DEC=0. Run it as ONE macrun job (the guard counts the server children as the same job)."""
import sys, os, json, time, statistics, subprocess, signal, socket, http.client, urllib.request, shutil
HOME = os.path.expanduser("~"); RIV = HOME + "/rivals"; KYO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); KPY = HOME + "/kyojin-mlx/.venv/bin/python"
CTX = int(sys.argv[1]); PASSES = int(sys.argv[2])
PACK = os.environ.get("D89_PACK", RIV + "/models/Qwen3.5-4B-mlx4"); OMLX_DIR = HOME + "/macfair/omlx_q4"; OMLX_NAME = "Qwen3.5-4B-mlx4-mtp-q4"
CFG = {"kfp-plain": ("k", ["--mtp-depth", "0"]), "kfp-d3": ("k", ["--mtp-depth", "3"]), "kfp-d4": ("k", ["--mtp-depth", "4"]), "ktq-plain": ("k", ["--kv-mode", "tq", "--mtp-depth", "0"]),
       "ktq-d3": ("k", ["--kv-mode", "tq", "--mtp-depth", "3"]), "ktq-d4": ("k", ["--kv-mode", "tq", "--mtp-depth", "4"]), "ktq-d4-npc": ("k", ["--kv-mode", "tq", "--mtp-depth", "4", "--prefix-cache-mb", "0"]), "ktq-plain-npc": ("k", ["--kv-mode", "tq", "--mtp-depth", "0", "--prefix-cache-mb", "0"]), "ktq-d4-old": ("k", ["--kv-mode", "tq", "--mtp-depth", "4"]), "ktq-d3-old": ("k", ["--kv-mode", "tq", "--mtp-depth", "3"]), "omlx-plain": ("o", 0), "omlx-mtp2": ("o", 2), "omlx-mtp3": ("o", 3), "omlx-mtp4": ("o", 4),
       "brt-plain": ("b", ["--speculate", "off"]), "brt-mtp": ("b", ["--speculate", "mtp-head"]), "brt-mtp3": ("b", ["--speculate", "mtp-head", "--spec-draft-tokens", "3", "--no-spec-adaptive"]),
       "brt-mtp4": ("b", ["--speculate", "mtp-head", "--spec-draft-tokens", "4", "--no-spec-adaptive"])}
BRT = RIV + "/baseRT-bin/v0.4.1/x"; BRT_MODEL = os.environ.get("D89_BRT_MODEL", RIV + "/baseRT-bin/v0.4.1/models/q35-4b-q4.base")   # BaseRT served by superfluid (its MTP head spec lives there)
names = sys.argv[3].split(",") if len(sys.argv) > 3 else [c for c in CFG if "npc" not in c and not c.endswith("-old")]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(RIV + "/models/Qwen3.5-4B-bf16")
ids = json.load(open(RIV + "/corpus_ids.json")); ids = ids["ids"] if isinstance(ids, dict) else ids
doc = tok.decode(ids[300000:300000 + CTX - 120])
QS = ["Explain how a hash map handles collisions and compare chaining with open addressing.", "Write a Python function that merges overlapping intervals, then explain its complexity.",
      "Summarize the causes of the French Revolution in five bullet points.", "What is the difference between TCP and UDP? Give examples of when to use each."]
def msgs(q): return [{"role": "user", "content": doc + "\n\n---\nIgnore the text above. " + q}]
def prompt(q): return tok.apply_chat_template([{"role": "user", "content": doc + "\n\n---\nIgnore the text above. " + q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
def start(name):
    base_name, *kv = name.split("+")          # "<config>+VAR=VAL+...": the same config with extra server env (A/B of an engine switch in one job)
    kind, x = CFG[base_name]; port = free_port(); log = open(f"/tmp/depth89_{name}.log", "w"); env = dict(os.environ, PYTHONPATH=KYO, PYTHONUNBUFFERED="1", **({"KYOJIN_TQ_ROWS": "0"} if base_name.endswith("-old") else {}), **dict(e.split("=", 1) for e in kv)); cache = None
    if kind == "k":
        argv = [KPY, "-m", "kyojin_mlx.serve.server", PACK, "--port", str(port), "--ctx", str(CTX + 400), "--sessions", "2", "--prefill-chunk", "2048", "--prefix-block", "2048", "--prefix-cache-mb", "2048", *x]
        health, mid = "/health", "x"
    elif kind == "b":
        base = f"/tmp/brt_d89_{os.getpid()}_{name}"; os.makedirs(base + "/home", exist_ok=True); cache = base; env.update(SUPERFLUID_HOME=base + "/home", SUPERFLUID_NO_UPDATE_CHECK="1")
        argv = [BRT + "/superfluid/bin/superfluid", "serve", "--model", BRT_MODEL, "--runtime", "basert", "--sessions", base + "/s", "--host", "127.0.0.1", "--port", str(port), "--max-batch", "1",
                "--offline", "--no-park", "--sessions-cache", "--no-tui", "--no-http-qos-header", "--basert-lib", BRT, "--max-context", str(CTX + 400), *os.environ.get("D89_BRT_ARGS", "").split(), *x]
        health, mid = "/v1/models", None
    else:
        base = HOME + f"/macfair/omlxbase_d89_{os.getpid()}_{name}"; os.makedirs(base, exist_ok=True); cache = base
        ms = {"version": 1, "models": {OMLX_NAME: ({"mtp_enabled": True, "mtp_fixed_depth": x} if x else {})}}; json.dump(ms, open(base + "/model_settings.json", "w"))
        argv = [RIV + "/venv-omlx/bin/omlx", "serve", "--base-path", base, "--model-dir", OMLX_DIR, "--port", str(port), "--paged-ssd-cache-dir", base + "/ssd"]
        health, mid = "/v1/models", OMLX_NAME
    p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=KYO if kind == "k" else RIV)
    t0 = time.time()
    while True:
        if p.poll() is not None: raise SystemExit(name + " died: " + open(log.name).read()[-600:])
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=3); c.request("GET", health); ok = c.getresponse().status == 200; c.close()
            if ok: break
        except Exception: pass
        if time.time() - t0 > 600: raise SystemExit(name + " timeout")
        time.sleep(1)
    if mid is None:                                   # BaseRT: the served model id
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10); c.request("GET", "/v1/models"); mid = json.loads(c.getresponse().read())["data"][0]["id"]; c.close()
    return p, port, mid, cache
def stop(p, cache):
    try: os.killpg(p.pid, signal.SIGTERM)
    except Exception: pass
    for _ in range(40):
        if p.poll() is not None: break
        time.sleep(0.25)
    if p.poll() is None:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    if cache: shutil.rmtree(cache, ignore_errors=True)
def stream(port, mid, text, n):
    chat = isinstance(text, list)                     # BaseRT: chat messages (its /v1/completions stream flushes a few large chunks, the chat stream is per token)
    body = {"model": mid, "max_tokens": n, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    body.update({"messages": text, "chat_template_kwargs": {"enable_thinking": False}} if chat else {"prompt": text})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600); t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions" if chat else "/v1/completions", json.dumps(body), {"Content-Type": "application/json"}); r = c.getresponse()
    if r.status != 200: raise RuntimeError(r.read()[:300])
    ts, out, usage = [], [], None
    while True:
        l = r.readline()
        if not l: break
        l = l.strip()
        if not l.startswith(b"data:"): continue
        d = l[5:].strip()
        if d == b"[DONE]": break
        j = json.loads(d)
        if j.get("usage"): usage = j["usage"]
        for ch in j.get("choices") or []:
            t = ch.get("text") or (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content")
            if t: ts.append(time.perf_counter() - t0); out.append(t)
    t_end = time.perf_counter() - t0; c.close(); n_out = (usage or {}).get("completion_tokens") or len(ts)
    return dict(chunks=len(ts), t_end=t_end, n=n_out, ttft=ts[0] if ts else None, tps=(n_out - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None, pt=(usage or {}).get("prompt_tokens"), text="".join(out))
R = {n: [] for n in names}; TXT = {n: [] for n in names}
for ps in range(PASSES):
    for name in (names if ps % 2 == 0 else names[::-1]):
        p, port, mid, cache = start(name)
        try:
            pf = msgs if CFG[name.split("+")[0]][0] == "b" else prompt; t0 = time.time(); w = stream(port, mid, pf(QS[0]), 4); prime = time.time() - t0
            time.sleep(float(dict(e.split("=", 1) for e in name.split("+")[1:]).get("PAUSE", os.environ.get("DEPTH89_PAUSE", "0"))))   # idle after the long prefill (the fanless Air throttles right after it): +PAUSE=S per config, else DEPTH89_PAUSE
            def _st():
                try: return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/stats", timeout=5).read())
                except Exception: return {}
            rs, perq = [], []
            for q in QS:                                         # per question: rounds, drafted, accepted, rounds per planned depth (Kyojin)
                s0 = _st() if name.startswith("k") else {}; rs.append(stream(port, mid, pf(q), 96)); s1 = _st() if name.startswith("k") else {}
                perq.append({k[5:] if k.startswith("spec_") else k: s1[k] - s0.get(k, 0) for k in s1 if k.startswith("spec_") and not k.startswith("spec_t") and isinstance(s1[k], (int, float)) and s1[k] != s0.get(k, 0)})
            if name.startswith("k"): print("perq", name, json.dumps(perq), flush=True)
            if name.startswith("k"):
                try:
                    st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/stats", timeout=5).read()); print("stats", name, json.dumps({k: v for k, v in st.items() if "spec" in k or "accept" in k or "draft" in k})[:700], flush=True)
                except Exception as e: print("stats failed", e)
            print('chunks', [r['chunks'] for r in rs], [r['n'] for r in rs], [round(r['ttft'] or -1, 2) for r in rs], [round(r['t_end'], 2) for r in rs], flush=True); R[name].append([round(r["tps"] or 0, 1) for r in rs]); TXT[name].append([r["text"] for r in rs])
            if CFG[name.split("+")[0]][0] == "k":                     # spec counters of the 5 requests (prime + 4): rounds, drafted, accepted -> ms per round
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10); c.request("GET", "/v1/stats"); st = json.loads(c.getresponse().read()); c.close()
                dec = sum(r["t_end"] - (r["ttft"] or 0) for r in rs); print("spec", {k: st.get(k) for k in ("spec_rounds", "spec_drafted", "spec_accepted", "decode_ticks")}, "decode_s", round(dec, 2), flush=True)
            print(f"CTX {CTX} pass {ps} {name}: " + " ".join(f"{r['tps']:.1f}" for r in rs) + f"  prompt_tokens {rs[0]['pt']} prime {prime:.0f}s ttft(cached) {statistics.median(r['ttft'] for r in rs):.2f}s", flush=True)
        finally:
            stop(p, cache)
json.dump(TXT, open(os.environ.get("D89_TEXTS") or os.environ.get("DEPTH89_TEXTS_OUT") or os.path.expanduser("~/macdec89/texts_%d.json" % CTX), "w"))
print("SUMMARY ctx", CTX)
ref = TXT.get("kfp-plain") or TXT.get("ktq-plain") or TXT.get("ktq-plain-npc")
for n in names:
    allv = [x for row in R[n] for x in row]
    best = max(allv); med = statistics.median(allv)
    same = ""
    if n.startswith("k") and ref and n not in ("kfp-plain",):
        base = TXT.get("ktq-plain-npc") if n.endswith("npc") and "ktq-plain-npc" in TXT else (TXT["ktq-plain"] if n.startswith("ktq") and "ktq-plain" in TXT else TXT.get("kfp-plain"))
        if base: print('   first diff (chars) vs plain:', [[next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if a == b else min(len(a), len(b))) for a, b in zip(ra, rb)] for ra, rb in zip(base, TXT[n])])
        if base: same = " text==plain %d/%d" % (sum(a == b for ra, rb in zip(base, TXT[n]) for a, b in zip(ra, rb)), sum(len(x) for x in TXT[n]))
    print(f"  {n:11s} best {best:.1f} median {med:.1f}{same}")
