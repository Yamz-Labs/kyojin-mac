"""Context-curve points against any OpenAI-compatible endpoint (our server or llama-server): prefill tok/s, decode tok/s at depth, passkey retrieval.
One JSON per context length, same prompts on every engine (real text filler from --filler files, fixed needle). Standard library only, no machine names.

  python curve_points.py --endpoint http://127.0.0.1:8312 --model NAME --filler a.md b.md --points 1024,4096,16384 --out DIR [--tag mtp5]
        [--depths 0.1,0.5,0.9] [--decode-tokens 128] [--plain-max 131072] [--extra-json '{"reasoning_effort": "no_think"}'] [--timeout 7200]

Per point L (prompt length in tokens, calibrated on the server's own prompt_tokens):
  * passkey: one request per depth (needle at 10 / 50 / 90 % of the filler), greedy, max_tokens 32; exact match of the 6-digit key in the answer; the request's timings give
    prefill tok/s at that length (prompt_per_second) and, from `predicted_per_second`, a short decode sample.
  * decode: for L <= --plain-max one more request with the filler only and --decode-tokens generated tokens (ignore_eos when the server accepts it): decode tok/s at depth L.
    Above --plain-max the decode figure is taken from the passkey requests (their 32 tokens) to keep the 1M points affordable.
Timings come from the response `timings` (llama.cpp keys: prompt_per_second, predicted_per_second); when absent the client wall time is used and flagged."""
import argparse, hashlib, json, os, statistics, sys, time, urllib.request, urllib.error, random, zlib

NEEDLE = "The pass key is {key}. Remember it. {key} is the pass key."
QUESTION = "\n\nWhat is the pass key mentioned in the text above? Answer with the number only."


def post(base, body, timeout):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["_wall_s"] = time.time() - t0
    return out


def filler_text(files):
    parts = []
    for f in files:
        parts.append(open(f, encoding="utf-8", errors="replace").read())
    return "\n\n".join(parts)


def rep_salt(salt):
    """CURVE_SALT set: one fixed salt for every decode request (repeat protocol: the same prompt in every arm and every rep); unset: the per-point salt."""
    return os.environ.get("CURVE_SALT") or salt


def make_text(corpus, n_chars, salt):
    """n_chars of filler starting at a salted offset (unique prompt: no prefix-cache hit between points); wraps with a changing header when the corpus is short."""
    if not corpus:
        raise SystemExit("empty filler")
    out, total, k = [], 0, 0
    start = (zlib.crc32(salt.encode()) % max(1, len(corpus) // 2)) if len(corpus) > 1000 else 0
    while total < n_chars:
        piece = corpus[start:] if k == 0 else f"\n\n[part {k} {salt}]\n" + corpus
        out.append(piece); total += len(piece); k += 1
    return "".join(out)[:n_chars]


def build_prompt(corpus, n_chars, depth, key, salt):
    body = make_text(corpus, n_chars, salt)
    cut = int(len(body) * depth)
    cut = body.rfind("\n", 0, cut) + 1 or cut                       # insert at a line boundary
    return body[:cut] + "\n" + NEEDLE.format(key=key) + "\n" + body[cut:] + QUESTION


def io_sample():
    """(disk MB read, mmap page-in MB) since boot: `iostat -Id` summed over disks, `vm_stat` Pageins x page size. None off macOS. The delta over one request says what a cold prompt reads
    (pread rows such as the Engram table show in the disk figure only; mmap-backed weights show in both)."""
    import subprocess, re
    try:
        io = subprocess.run(["iostat", "-Id"], capture_output=True, text=True, timeout=5).stdout.split("\n")
        mb = sum(float(l.split()[2 + 3 * k]) for l in io[2:3] for k in range(len(l.split()) // 3)) if len(io) > 2 else None
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        ps = int(re.search(r"page size of (\d+)", vm).group(1)); pi = int(re.search(r"Pageins:\s+(\d+)", vm).group(1))
        return mb, pi * ps / 1e6
    except Exception:
        return None


def ask(a, prompt, max_tokens, ignore_eos=False):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, **json.loads(a.extra_json or "{}")}
    if ignore_eos:
        body["ignore_eos"] = True
    io0 = io_sample()
    out = post(a.endpoint, body, a.timeout)
    io1 = io_sample()
    ch0 = (out.get("choices") or [{}])[0]
    msg = ch0.get("message") or {}
    tm = out.get("timings") or {}
    use = out.get("usage") or {}
    r = {"content": (msg.get("content") or "")[:200], "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:300], "finish": ch0.get("finish_reason"), "prompt_tokens": use.get("prompt_tokens"), "completion_tokens": use.get("completion_tokens"), "wall_s": round(out["_wall_s"], 2),
         "prompt_md5": hashlib.md5(prompt.encode()).hexdigest(), "text_md5": hashlib.md5(((msg.get("content") or "") + "\x00" + (msg.get("reasoning_content") or msg.get("reasoning") or "")).encode()).hexdigest()}
    for k in ("tau", "draft_n", "draft_n_accepted"):                  # speculative replies: equal prompt_md5 + text_md5 + tau between two arms = the same state
        if k in (out.get("timings") or {}): r[k] = out["timings"][k]
    if io0 and io1:
        r["io_disk_mb"] = round(io1[0] - io0[0], 1); r["io_pagein_mb"] = round(io1[1] - io0[1], 1)
    if tm.get("prompt_per_second") is not None:
        r["prefill_tok_s"] = round(tm["prompt_per_second"], 2); r["timings_source"] = "server"
        if tm.get("predicted_per_second") is not None:
            r["decode_tok_s"] = round(tm["predicted_per_second"], 2)
        r["cached"] = tm.get("cache_n")
    else:
        r["timings_source"] = "client_wall (no server timings: prefill / decode not separable)"
    return r


def calibrate(a, corpus):
    """chars per token of this filler on this server's tokenizer."""
    n = 6000
    r = ask(a, make_text(corpus, n, "cal") + "\nReply OK.", 4)
    return n / max(100, (r["prompt_tokens"] or n // 3) - 40)


def fit(a, st, L, build, max_tokens, ignore_eos=False, rep=0):
    """One request of about L prompt tokens: builder(n_chars) -> prompt. The chars-per-token estimate is re-fitted from every answer; a prompt the server refuses as too long
    (HTTP 400 / 413 / 500) is retried shorter (x 0.85, twice). The prompt is capped at ctx - (max_tokens + 512) when --ctx is given."""
    cap = (a.ctx - max_tokens - 512) if a.ctx else L
    for attempt in range(3):
        tgt = min(L, cap) if attempt == 0 else int(min(L, cap) * 0.85 ** attempt)
        prompt = build(int((tgt - 60) * st["cpt"]))
        try:
            r = ask(a, prompt, max_tokens, ignore_eos)
        except urllib.error.HTTPError as e:
            if e.code in (400, 413, 500) and attempt < 2:
                print(f"curve_points: HTTP {e.code} at ~{tgt} tokens, retry shorter", flush=True); continue
            raise
        if r.get("prompt_tokens"):
            st["cpt"] = len(prompt) / r["prompt_tokens"]
        r["target_tokens"] = tgt
        for _ in range(rep if max_tokens > 8 else 0):           # --pf-rep N: the identical prompt again (8 tokens): every chunk shape is warm, so its prefill speed is the steady state (the first request also pays the first-seen shapes)
            r2 = ask(a, prompt, 8)
            r["prefill_cold_tok_s"] = r.get("prefill_cold_tok_s", r.get("prefill_tok_s"))
            r["prefill_tok_s"] = r2.get("prefill_tok_s"); r["prefill_cached"] = r2.get("cached")
            r["io_cold_disk_mb"], r["io_cold_pagein_mb"] = r.get("io_disk_mb"), r.get("io_pagein_mb"); r["io_disk_mb"], r["io_pagein_mb"] = r2.get("io_disk_mb"), r2.get("io_pagein_mb")
        return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True); p.add_argument("--model", default="model"); p.add_argument("--filler", nargs="+", required=True)
    p.add_argument("--points", default="1024,4096,16384,32768,65536,131072,262144,524288,1048576"); p.add_argument("--depths", default="0.1,0.5,0.9")
    p.add_argument("--out", required=True); p.add_argument("--tag", default="run"); p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--plain-max", type=int, default=131072); p.add_argument("--extra-json", default=""); p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--pk-tokens", type=int, default=256, help="max_tokens of the passkey request (the model may think first)"); p.add_argument("--warm", type=int, default=0, help="throwaway decode requests (other filler) before each measured one: first-seen chunk shapes compile on the server")
    p.add_argument("--pf-rep", type=int, default=0, help="repeat the decode point's prompt N times (8 tokens) and report that prefill as the warm figure (prefill_cold_tok_s keeps the first)")
    p.add_argument("--reps", type=int, default=1, help="decode requests per point (each on its own text offset; the first one is the same text as with --reps 1): the point reports the median decode and prefill speed, the single requests stay in decode_reps")
    p.add_argument("--drop-cold", action="store_true", help="the median over the reps leaves the first one out (cold prefill / first-seen shapes)")
    p.add_argument("--seed", type=int, default=7); p.add_argument("--ctx", type=int, default=0, help="server context: prompts are capped at ctx - (max_tokens + 512)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    corpus = filler_text(a.filler)
    fsha = hashlib.sha256(corpus.encode("utf-8")).hexdigest()[:16]               # text id: the same files give the same prompts for the same tag
    st = {"cpt": calibrate(a, corpus)}
    rnd = random.Random(a.seed)
    print(f"curve_points: filler {len(corpus)} chars (sha {fsha}), {st['cpt']:.3f} chars / token", flush=True)
    for L in [int(x) for x in a.points.split(",")]:
        res = {"L": L, "tag": a.tag, "filler_sha": fsha, "filler_chars": len(corpus), "passkey": [], "t0": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            if L <= a.plain_max:
                for w in range(a.warm):
                    fit(a, st, L, lambda n: make_text(corpus, n, rep_salt(f"warm{w}{L}{a.tag}")) + "\nContinue the text with a long story.", 8)
                res["decode"] = fit(a, st, L, lambda n: make_text(corpus, n, rep_salt(f"dec{L}{a.tag}")) + "\nContinue the text with a long story.", a.decode_tokens, ignore_eos=True, rep=getattr(a, "pf_rep", 0))
                if a.reps > 1:
                    res["decode_reps"] = [res["decode"]]
                    for k in range(1, a.reps):
                        res["decode_reps"].append(fit(a, st, L, lambda n: make_text(corpus, n, rep_salt(f"dec{L}{a.tag}r{k}")) + "\nContinue the text with a long story.", a.decode_tokens, ignore_eos=True, rep=getattr(a, "pf_rep", 0)))
                    for key in ("prefill_tok_s", "decode_tok_s"):
                        v = [r[key] for r in (res["decode_reps"][1:] if a.drop_cold else res["decode_reps"]) if r.get(key)]
                        if v: res["decode"][key + "_median"] = statistics.median(v)
                print(f"RESULTS curve {a.tag} L={L} decode: prompt {res['decode'].get('prompt_tokens')} prefill {res['decode'].get('prefill_tok_s')} decode {res['decode'].get('decode_tok_s')} tok/s, cold prefill {res['decode'].get('prefill_cold_tok_s')}, io disk MB cold/warm {res['decode'].get('io_cold_disk_mb')}/{res['decode'].get('io_disk_mb')}, pagein MB cold/warm {res['decode'].get('io_cold_pagein_mb')}/{res['decode'].get('io_pagein_mb')}", flush=True)
            for d in [float(x) for x in a.depths.split(",") if x.strip()]:
                key = "%06d" % rnd.randrange(10 ** 6)
                r = fit(a, st, L, lambda n: build_prompt(corpus, n, d, key, f"pk{L}{d}{a.tag}"), a.pk_tokens)
                # Hy4 in no_think still opens its own think block: the server parses that into reasoning_content, so a short budget leaves content empty. The key is random:
                # finding it in the content OR the reasoning proves retrieval; ok_content is the strict reading.
                hit = lambda t: key in t or key in "".join(c for c in t if c.isdigit())
                r.update(depth=d, key=key, ok_content=hit(r["content"]), ok=hit(r["content"]) or hit(r.get("reasoning") or ""))
                res["passkey"].append(r)
                print(f"RESULTS curve {a.tag} L={L} passkey depth {d}: {'OK' if r['ok'] else 'MISS'} (prompt {r.get('prompt_tokens')}, prefill {r.get('prefill_tok_s')} tok/s, decode sample {r.get('decode_tok_s')}, finish {r.get('finish')}, answer {r['content'][:40]!r}, reasoning {(r.get('reasoning') or '')[:60]!r})", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            res["error"] = repr(e)[:300]
            print(f"RESULTS curve {a.tag} L={L} ERROR {e!r}"[:300], flush=True)
        res["chars_per_token"] = round(st["cpt"], 3)
        res["passkey_ok"] = sum(1 for r in res["passkey"] if r["ok"]); res["passkey_ok_content"] = sum(1 for r in res["passkey"] if r["ok_content"]); res["passkey_n"] = len(res["passkey"])
        json.dump(res, open(os.path.join(a.out, f"point_{a.tag}_{L}.json"), "w"), indent=1)      # a failed point is recorded, the next one still runs (a dead server fails each in seconds)


if __name__ == "__main__":
    main()
