"""oMLX prefill TTFT over HTTP: one fresh server, warm-up, then unique-slice prompts of each length, max_tokens=1. usage: pf3_omlx.py LENS REPS [ane]"""
import sys, os, json, time, subprocess, signal, http.client, socket, urllib.request, shutil
RIV = os.path.expanduser("~/rivals"); lens = [int(x) for x in sys.argv[1].split(",")]; reps = int(sys.argv[2]); ane = len(sys.argv) > 3 and sys.argv[3] == "ane"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{RIV}/models/Qwen3.5-4B-mlx4")
ids = json.load(open(f"{RIV}/corpus_ids.json")); ids = ids["ids"] if isinstance(ids, dict) else ids
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
cache = f"{RIV}/omlx_cache/pf3_{os.getpid()}"
d = f"{RIV}/omlx_models/q4b"
env = dict(os.environ)
extra = []
if ane:
    bp = f"/tmp/pf3_base_{os.getpid()}"; os.makedirs(bp, exist_ok=True)
    shutil.copy(os.path.expanduser("~/.omlx/settings.json"), bp + "/settings.json")
    ms = json.load(open(os.path.expanduser("~/.omlx/model_settings.json"))); ent = dict(ms["models"]["Qwen3.5-0.8B-mlx4"]); ent["qwen35_ane_prefill_enabled"] = True
    ms["models"] = {"Qwen3.5-4B-mlx4": ent}; json.dump(ms, open(bp + "/model_settings.json", "w")); extra = ["--base-path", bp]
pr = subprocess.Popen([f"{RIV}/venv-omlx/bin/omlx", "serve", "--model-dir", d, "--port", str(port), "--paged-ssd-cache-dir", cache, *extra], env=env, stdout=open("/tmp/pf3_omlx.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
try:
    for _ in range(120):
        try:
            mid = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2))["data"][0]["id"]; break
        except Exception: time.sleep(1)
    def ttft(n, seed):
        txt = tok.decode(ids[seed:seed + n])
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600); t0 = time.perf_counter()
        c.request("POST", "/v1/completions", json.dumps({"model": mid, "prompt": txt, "max_tokens": 1, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}), {"Content-Type": "application/json"})
        r = c.getresponse(); tf = None; pt = None
        while True:
            l = r.readline()
            if not l: break
            l = l.strip()
            if l.startswith(b"data:") and l[5:].strip() != b"[DONE]":
                j = json.loads(l[5:])
                if j.get("usage"): pt = j["usage"].get("prompt_tokens")
                for ch in j.get("choices") or []:
                    if ch.get("text") and tf is None: tf = time.perf_counter() - t0
        c.close(); return pt, tf
    ttft(300, 50000); ttft(300, 51000)
    for n in lens:
        res = []
        for i in range(reps):
            pt, tf = ttft(n, 1000 + 20000 * (i + 1) + n); res.append(pt / tf)
        print(f"omlx{'+ane' if ane else ''} pp{n} (prompt_tokens {pt}): " + " ".join(f"{x:.0f}" for x in res) + f" best {max(res):.0f} tok/s", flush=True)
finally:
    try: os.killpg(pr.pid, signal.SIGTERM)
    except Exception: pass
    shutil.rmtree(cache, ignore_errors=True)
