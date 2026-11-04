#!/usr/bin/env python3
# Same-prompt bench: the decode, prefill, predictable and novel cases of llmprobe (Apache-2.0, bench/index.ts), re-implemented
# against our lane so our figures compare with a published llmprobe figure. Same prompts, 192 tokens, temperature 0, warmup + K runs,
# cache bust tag, rates: prefill = prompt tokens / first-text time, decode = window delivery rate (deliveryRate).
# usage: python3 -I tools/same_bench.py [--k 3] [--cases standard,prefill,predictable,novel] [--no-think] [--json out.json]
# The caller holds the lane lock.
import argparse, http.client, json, os, statistics, time

PASSAGE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "same_bench_passage.json")))["passage"]
DECODE = ("Write a TypeScript function `retryWithBackoff<T>(fn, options)` that retries an async operation with exponential backoff "
          "and jitter. Include the type signature, JSDoc, and inline comments explaining the backoff maths.")
NOVEL = "Invent an original, surprising paragraph of stream-of-consciousness prose. Avoid common phrases."
NORMALISER = "You are a text normaliser. Output the user's text unchanged."
DECODE_TOKENS, PREFILL_TOKENS, PREFILL_BYTES = 192, 8, 40960


def long_prefix(seed, nbytes):
    out, n, i = [], 0, 0
    while n < nbytes:
        piece = "[%d] %s " % (i, seed)
        out.append(piece)
        n += len(piece)
        i += 1
    return "".join(out)[:nbytes]


PREFILL = long_prefix("The archive records the following entry.", PREFILL_BYTES) + "\n\nReply with just the word: OK."
TAG = "%x" % int(time.time())
seq = 0


def bust(text):
    global seq
    seq += 1
    return "[probe %s-%x] %s" % (TAG, seq, text)


def run(a, user, n, system=None, exact=False):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = {"model": a.model, "messages": msgs, "max_tokens": n, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if exact:
        body.update(ignore_eos=True, min_tokens=n)
    if a.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    c = http.client.HTTPConnection("localhost", a.port, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"error": "HTTP %d" % r.status}
    frames, usage, texts = [], None, []
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        d = json.loads(line[5:])
        usage = d.get("usage") or usage
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                frames.append((time.perf_counter(), len(piece)))
                texts.append(delta.get("content") or "")
    c.close()
    if not frames or not usage:
        return {"error": "no frames or usage"}
    total = sum(f[1] for f in frames)
    span = frames[-1][0] - frames[0][0]
    out_tok, in_tok = usage.get("completion_tokens"), usage.get("prompt_tokens")
    rate = out_tok * (1 - frames[0][1] / total) / span if span > 0 else None
    return {"decode": rate, "prefill": in_tok / (frames[0][0] - t0), "out": out_tok, "in": in_tok, "frames": len(frames), "text": "".join(texts)}


def case(a, name, fn, metric):
    warm = fn()
    runs = [fn() for _ in range(a.k)]
    ok = [r for r in runs if "error" not in r]
    vals = [r[metric] for r in ok if r.get(metric)]
    res = {"case": name, "median": statistics.median(vals) if vals else None, "values": vals, "warm": warm.get(metric), "out": [r["out"] for r in ok],
           "in": [r["in"] for r in ok], "frames": [r["frames"] for r in ok], "errors": [r["error"] for r in runs if "error" in r]}
    if name == "predictable":
        opening = " ".join(PASSAGE.split(" ")[:6])
        kept = [r for r in ok if r["text"].lstrip().lstrip("\"'“”*_").startswith(opening)]
        res["echoed"] = len(kept)
        vals = [r[metric] for r in kept if r.get(metric)]
        res["median"] = statistics.median(vals) if vals else None
    print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items() if k != "values"} | {"values": [round(v, 1) for v in vals]}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--port", type=int, default=8292)
    ap.add_argument("--model", default="ds41-ultra")
    ap.add_argument("--cases", default="standard,prefill,predictable,novel")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    table = {"standard": (lambda: run(a, bust(DECODE), DECODE_TOKENS, exact=True), "decode"),
             "prefill": (lambda: run(a, bust(PREFILL), PREFILL_TOKENS), "prefill"),
             "predictable": (lambda: run(a, PASSAGE, DECODE_TOKENS, system=bust(NORMALISER)), "decode"),
             "novel": (lambda: run(a, bust(NOVEL), DECODE_TOKENS), "decode")}
    res = [case(a, n, *table[n]) for n in a.cases.split(",")]
    if a.json:
        json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
    main()
