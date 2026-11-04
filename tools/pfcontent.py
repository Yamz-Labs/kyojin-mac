"""Does the prefill chunk cost follow the token content? Measured through the running lane only (one HTTP client, no model, a few MB).

usage: pfcontent.py [PORT] [N] [KINDS] [DOC_TOKENIZER_JSON] [DOC_TEXT_FILE]
  N      prompt tokens (default 8192 = one chunk); KINDS comma list of repeat,random,real (default repeat,random,real; real needs a text file)
Per kind, a fresh prompt (fresh seed, so no prefix-cache hit), then one line per prefill chunk: ms, tok/s, and the deltas of the lane counters over
that chunk: distinct Engram rows (pf_eng_rows_uniq) of the positions x columns hashed (pf_eng_rows_total), seconds of the row jobs, warm_first hits.
Read: if ms and tok/s follow rows_uniq (repeat < real < random) the cause is the Engram row gather (distinct rows, cold page cache); if they do not,
it is not. Run the lane twice, KYOJIN_PF_WARM0=0 and =1 (same command): chunk 0 should lose its cold reads only if warm_first > 0 and rows_uniq is large.
Counters are cumulative lane stats (/v1/stats); stop other traffic first."""
import http.client
import json
import random
import sys
import threading
import time


def stats(port):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", "/v1/stats")
        return json.loads(c.getresponse().read())
    except Exception:
        return {}


KEYS = ("t_pf_chunk", "pf_eng_rows_uniq", "pf_eng_rows_total", "pf_eng_job_ms", "pf_eng_warm_first")


def prompt(kind, n, seed, doc_ids):
    rnd = random.Random(seed)
    if kind == "repeat":
        base = [rnd.randint(100, 120000) for _ in range(241)]
        return (base * (n // 241 + 1))[:n]
    if kind == "random":
        return [rnd.randint(100, 120000) for _ in range(n)]
    off = rnd.randint(0, max(0, len(doc_ids) - n))                      # real: a random window of the document (fresh start each time)
    return doc_ids[off:off + n]


def run(port, ids):
    body = {"model": "x", "prompt": ids, "max_tokens": 1, "temperature": 0, "stream": True}
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            s = stats(port)
            if "n_pf_chunk" in s:
                samples.append((s["n_pf_chunk"], {k: s.get(k, 0) for k in KEYS}))
            stop.wait(0.02)
    th = threading.Thread(target=poll, daemon=True)
    base = stats(port)
    th.start()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    c.getresponse().read()
    stop.set()
    th.join()
    last = {k: base.get(k, 0) for k in KEYS}
    n0 = base.get("n_pf_chunk", 0)
    seen = {}
    for n, v in samples:
        seen[n] = v                                                     # last sample per chunk count
    for n in sorted(seen):
        if n <= n0:
            continue
        v = seen[n]
        d = {k: v[k] - last[k] for k in KEYS}
        last = v
        yield n - n0, d


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    kinds = (sys.argv[3] if len(sys.argv) > 3 else "repeat,random,real").split(",")
    doc = []
    if "real" in kinds:
        if len(sys.argv) < 6:
            sys.exit("kind real needs DOC_TOKENIZER_JSON and DOC_TEXT_FILE")
        from transformers import PreTrainedTokenizerFast
        doc = PreTrainedTokenizerFast(tokenizer_file=sys.argv[4]).encode(open(sys.argv[5]).read())
    for kind in kinds:
        ids = prompt(kind, n, int(time.time()) ^ hash(kind) & 0xffff, doc)
        print(f"== {kind}: {len(ids)} tokens, distinct ids {len(set(ids))}")
        for i, d in run(port, ids):
            ms = d["t_pf_chunk"] * 1e3
            print(f"chunk {i}: {ms:8.1f} ms {n / max(d['t_pf_chunk'], 1e-9):7.0f} tok/s | rows_uniq {d['pf_eng_rows_uniq']:.0f} of {d['pf_eng_rows_total']:.0f} "
                  f"| row jobs {d['pf_eng_job_ms']:.0f} ms | warm_first {d['pf_eng_warm_first']:.0f}")


if __name__ == "__main__":
    main()
