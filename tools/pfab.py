"""Live A/B of prefill flags on the running lane, one prompt length, per-chunk times.

usage: pfab.py LENS 'tag:{json} tag:{json} ...' [PORT] [REPS]
  e.g. pfab.py 524288 'base:{} fscore0:{"pf_idx_fscore":"0"}' 8292 1
Every arm: POST /admin/flags (drops the memory prefix cache, so each arm re-prefills the same prompt),
one completion (max_tokens=NTOK), then the per-chunk ms from the t_pf_chunk/n_pf_chunk deltas and the
prefill tok/s from the first streamed token. Same prompt text for every arm of a length; a fresh seed
per (length, rep) so no arm starts from a warm prefix.
Waits for the pfsmall agent to exit and holds /tmp/lane_ab.lock for the whole measure.
"""
import fcntl
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time

from transformers import PreTrainedTokenizerFast

TOK = PreTrainedTokenizerFast(tokenizer_file=os.path.expanduser("~/ds41/model/tokenizer.json"))
WORDS = ("empire senate legion trade grain tax road province emperor coin harbor city law army border revolt temple market " +
         "province consul villa slave citizen frontier aqueduct forum treaty siege plague famine reform dynasty").split()
LOCK = "/tmp/lane_ab.lock"


def text(n, seed):
    rnd, ids = random.Random(seed), []
    while len(ids) < n:
        s = "Section %d. " % rnd.randint(0, 10 ** 6) + " ".join(rnd.choice(WORDS) for _ in range(60)) + ".\n"
        ids += TOK.encode(s)
    return ids[:n - 40] + TOK.encode("\nNow write a long, detailed essay that summarizes all the sections above, in order.\n")


def call(port, path, body=None, timeout=30):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST" if body is not None else "GET", path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
    return json.loads(c.getresponse().read())


def stats(port):
    try:
        return call(port, "/v1/stats")
    except Exception:
        return {}


def set_flags(port, js):
    r = call(port, "/admin/flags", js, timeout=60)
    time.sleep(1.0)
    return r


def wait_pfsmall(limit=1800):
    """The pfsmall agent session; the bracket keeps the check from matching its own `sh -c` command line."""
    t0 = time.time()
    while time.time() - t0 < limit:
        if subprocess.run(["pgrep", "-f", "--", "--title[ ]pfsmall"], capture_output=True).returncode != 0:
            return True
        time.sleep(15)
    return False


def one(port, ids, ntok):
    body = {"model": "x", "prompt": ids, "max_tokens": ntok, "temperature": 0, "stream": True, "ignore_eos": True}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    samples, txt, t = [], [], []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            s = stats(port)
            if "n_pf_chunk" in s:
                samples.append((s["n_pf_chunk"], s["t_pf_chunk"]))
            stop.wait(0.05)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.time()
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    first = None
    for line in c.getresponse():
        if line.startswith(b"data:") and line.strip() != b"data: [DONE]":
            if first is None:
                first = time.time()
            t.append(time.time())
            try:
                d = json.loads(line[5:])["choices"][0].get("text")
                if d:
                    txt.append(d)
            except Exception:
                pass
    stop.set()
    th.join(2)
    seen, prev = {}, None
    for nch, tpf in samples:
        if prev is not None and nch > prev[0]:
            seen[nch] = 1e3 * (tpf - prev[1]) / (nch - prev[0])
        prev = (nch, tpf)
    return dict(pf=len(ids) / max((first or t0) - t0, 1e-9), ttft=(first or t0) - t0,
                chunks=dict(sorted(seen.items())), out="".join(txt)[:60].replace("\n", "\\n"))


def main():
    lens = [int(x) for x in sys.argv[1].split(",")]
    arms = []
    for a in sys.argv[2].split():
        tag, js = a.split(":", 1)
        arms.append((tag, json.loads(js)))
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8292
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    ntok = int(os.environ.get("NTOK", "1"))
    lk = open(LOCK, "w")
    fcntl.flock(lk, fcntl.LOCK_EX)
    print("flock held, waiting for pfsmall", flush=True)
    assert wait_pfsmall(), "pfsmall still running"
    print("lane free", flush=True)
    for rep in range(reps):
        for L in lens:
            ids = text(L, L * 13 + 7 + rep * 100003)
            for tag, js in arms:
                set_flags(port, js)
                r = one(port, ids, ntok)
                print("ARM L=%d rep=%d %-12s prefill %.0f tok/s ttft %.2fs out=%r chunks %s" %
                      (L, rep, tag, r["pf"], r["ttft"], r["out"], " ".join("%d:%.0f" % (k * 8192, v) for k, v in r["chunks"].items())), flush=True)
    set_flags(port, {"pf_idx_fscore": "1", "pf_idx_topk_mt": "1", "idx_qblock": "512", "pf_idx_ktile": "1"})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
