"""Per-chunk prefill time against position in the context, measured through the running lane only.

usage: pfchunk.py [LENS] [PORT]      (LENS = comma list of prompt token counts, default 32768,131072,262144,524288)
Prints, per prompt length, the prefill tok/s (prompt / TTFT) and one line per prefill chunk of the engine
(position at the end of the chunk -> ms of that chunk, from the delta of the /v1/stats counters t_pf_chunk /
n_pf_chunk sampled while the request runs; chunk size is the lane's --prefill-chunk, 8192).

No model is loaded here: one HTTP request, one poller. Prompt text is fresh per length (a seed per length), so
no prefix-cache hit hides the prefill.
"""
import http.client
import json
import os
import random
import sys
import threading
import time

from transformers import PreTrainedTokenizerFast

TOK = PreTrainedTokenizerFast(tokenizer_file=os.path.expanduser("~/ds41/model/tokenizer.json"))
WORDS = ("empire senate legion trade grain tax road province emperor coin harbor city law army border revolt temple market " +
         "province consul villa slave citizen frontier aqueduct forum treaty siege plague famine reform dynasty").split()


def text(n: int, seed: int) -> list:
    rnd = random.Random(seed)
    ids = []
    while len(ids) < n:
        s = "Section %d. " % rnd.randint(0, 10 ** 6) + " ".join(rnd.choice(WORDS) for _ in range(60)) + ".\n"
        ids += TOK.encode(s)
    return ids[:n - 40] + TOK.encode("\nNow write a long, detailed essay that summarizes all the sections above, in order.\n")


def stats(port: int) -> dict:
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", "/v1/stats")
        return json.loads(c.getresponse().read())
    except Exception:
        return {}


KEYS = ("t_pf_chunk", "n_pf_chunk", "t_pf_lookup", "t_pf_after", "t_pf_eng0", "t_pf_eng1", "prefill_tokens", "n_pf_chunk")


def run(port: int, ids: list, ntok: int):
    body = {"model": "x", "prompt": ids, "max_tokens": ntok, "temperature": 0, "stream": True, "ignore_eos": True}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    samples = []                                               # (wall, wc, n_pf_chunk, t_pf_chunk)
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            s = stats(port)
            if "n_pf_chunk" in s:
                samples.append((time.time(), s["n_pf_chunk"], s["t_pf_chunk"]))
            stop.wait(0.05)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.time()
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    t, first, txt = [], None, []
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
    th.join(timeout=2)
    return t0, first or time.time(), t, samples, "".join(txt).replace("\n", "\\n")[:80]


def main():
    lens = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768,131072,262144,524288").split(",")]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8292
    print("pfchunk port %d lens %s" % (port, lens), flush=True)
    for n in lens:
        seed = n * 7919 + int(os.environ.get("PFCHUNK_SEED") or time.time()) % 100000
        ids = text(n, seed)
        s0 = stats(port)
        t0, first, tt, samples, txt = run(port, ids, int(os.environ.get("NTOK", "1")))
        s1 = stats(port)
        pf = len(ids) / max(first - t0, 1e-9)
        print("L %d prefill %.0f tok/s ttft %.2f s (chunks %d) out=%r" % (len(ids), pf, first - t0, s1.get("n_pf_chunk", 0) - s0.get("n_pf_chunk", 0), txt), flush=True)
        d = {k: round(s1.get(k, 0) - s0.get(k, 0), 3) for k in dict.fromkeys(KEYS)}
        print("   stats %s" % d, flush=True)
        # per chunk: first sample with a new n_pf_chunk after the request start
        prev = None
        rows = []
        for (w, nch, tpf) in samples:
            if w < t0:
                continue
            if prev is None:
                prev = (nch, tpf, w)
                continue
            if nch > prev[0]:
                ms = 1e3 * (tpf - prev[1]) / max(nch - prev[0], 1)
                rows.append((nch, ms))
                prev = (nch, tpf, w)
        # keep the last sample of each chunk id (a chunk ends once)
        seen = {}
        for nch, ms in rows:
            seen[nch] = ms
        line = " ".join("%d:%.0f" % (nch * int(os.environ.get("CHUNK", "8192")), ms) for nch, ms in sorted(seen.items()))
        print("   chunk_ms %s" % line, flush=True)


if __name__ == "__main__":
    main()
