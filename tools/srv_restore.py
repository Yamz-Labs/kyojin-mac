"""Server-level kvwin restore check: start serve.server on a Qwen3.5 EXL3 pack with --kv-disk, send a long token prompt, stop (SIGTERM, by pid),
start a NEW server on the same disk dir, send prompt + tail: tokens must equal a server with no disk, cached_tokens must show the disk hit.
usage: srv_restore.py MODEL_DIR WORKDIR   (KYOJIN_KVWIN from the environment, SRV_EXTRA = extra server args e.g. '--kv-mode tq'; one server at a time)"""
import http.client, json, os, signal, subprocess, sys, time
import numpy as np

model, work = sys.argv[1], sys.argv[2]
port = 8097
ids = [int(x) for x in np.random.default_rng(5).integers(100, 5000, 1300)]
tail = ids + [21, 22, 23]


def serve(disk):
    cmd = [sys.executable, "-m", "kyojin_mlx.serve.server", model, "--port", str(port), "--sessions", "2", "--ctx", "4096", "--prefill-chunk", "256",
           "--prefix-block", "256", "--prefix-cache-mb", "512"] + (["--kv-disk", disk, "--kv-disk-block", "512"] if disk else []) + os.environ.get("SRV_EXTRA", "").split()
    log = open(os.path.join(work, "server_%d.log" % time.time()), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5); c.request("GET", "/health")
            if c.getresponse().status == 200:
                return p
        except OSError:
            pass
        time.sleep(2)
    p.send_signal(signal.SIGTERM); raise SystemExit("server did not come up")


def ask(prompt):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/completions", json.dumps(dict(prompt=prompt, max_tokens=24, temperature=0, ignore_eos=True)), {"Content-Type": "application/json"})
    j = json.loads(c.getresponse().read())
    return j["choices"][0]["text"], (j["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")


def stop(p):
    p.send_signal(signal.SIGTERM)
    p.wait(60)


disk = os.path.join(work, "kvdisk")
p = serve(None); ref, _ = ask(tail); stop(p)
p = serve(disk); a, ca = ask(ids); stop(p)
n_files = len([f for f in os.listdir(os.path.join(disk, os.listdir(disk)[0])) if f.endswith(".kv.safetensors")])
p = serve(disk); b, cb = ask(tail); stop(p)
print(json.dumps(dict(kvwin=os.environ.get("KYOJIN_KVWIN"), extra=os.environ.get("SRV_EXTRA"), files=n_files, cached_first=ca, cached_after_restart=cb, equal_to_no_disk=(b == ref), text=b[:60])))
assert b == ref and cb and cb >= 1024, "restore failed"
print("OK")
