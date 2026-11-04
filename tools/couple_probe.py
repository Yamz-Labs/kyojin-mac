#!/usr/bin/env python3
"""spec_couple probe (key-coupled drafting for sampled sessions). Bounded: no model is loaded here, client memory < 1 GiB, GPU memory < 0.2 GiB.

  python3 -I tools/couple_probe.py selfcheck            # Mac, Metal: the premise on a 129280-wide vocabulary, and the acceptance model. Seconds.
  python3 -I tools/couple_probe.py lane --port 8080     # Mac, on the resident lane (the caller holds the lane lock): ABBA of spec_couple 0 / 100 / 50.

selfcheck (the part that decides if the lever can work at all):
  1. categorical(lp / T, key) == argmax(lp / T + gumbel(key)) on THIS device at the pack's vocabulary width (--vocab, default 129280) (the engine's draft noise must be the sampler's noise, bit for bit).
  2. For logits p and a draft q = p + skew x N(0, 1): acceptance of the argmax proposal (today) against the coupled proposal, and TV of the emitted token
     against softmax(p / T) (must stay at the sampling-noise floor: the lever changes how often a draft is accepted, never what is emitted).
lane (served numbers; the lever's only real test):
  Arms run at the lane's own sampler (temperature 1, top_p 0.95 unless --temperature / --top-p), same prompts for every arm, thinking off, `--tokens` each,
  ABBA order, `--reps` rounds. Per arm: decode tok/s from usage and wall, spec_accepted / spec_drafted and tokens per round from /v1/stats deltas.
  Control: at temperature 0 the flag must change nothing (same text). Prints a table and writes --json. Nothing in the lane is left changed: the flag is
  put back to its start value. The lane is NOT restarted, no model is loaded here.
No claim is made by this file: it prints what it measured."""
import argparse
import http.client
import json
import math
import statistics
import sys
import time

PROMPTS = [
    "Invent an original, surprising paragraph of stream-of-consciousness prose. Avoid common phrases.",
    "Write a short story opening about a lighthouse keeper who discovers the sea has stopped moving.",
    "Describe a city that exists only at night, in the voice of someone who has never seen daylight.",
    "Explain, as a rambling monologue, why a library might dream.",
    "Write a TypeScript function `retryWithBackoff<T>(fn, options)` that retries an async operation with exponential backoff and jitter. Include the type signature, JSDoc, and inline comments.",
    "Repeat the following sentence back to me exactly, ten times, one per line: The quick brown fox jumps over the lazy dog while the clock strikes thirteen.",
]


def selfcheck(a):
    import mlx.core as mx
    V, T = a.vocab, a.temperature
    if hasattr(mx, "set_memory_limit"):
        mx.set_memory_limit(1 << 30)
    bad = 0
    for s in range(a.trials):
        lp = mx.random.normal((V,), key=mx.random.key(s)) * 2.0
        k = mx.random.key(10_000 + s)
        bad += int(mx.random.categorical(lp * (1.0 / T), key=k).item() != mx.argmax(lp * (1.0 / T) + mx.random.gumbel((V,), key=k)).item())
    print(f"1. categorical == argmax(lp/T + gumbel) at V={V} on {mx.default_device()}: {a.trials - bad}/{a.trials} equal  {'OK' if bad == 0 else 'BROKEN: spec_couple would break nothing but gain nothing'}")
    n_ctx = a.trials
    rows = []
    for skew, flat in ((0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (2.0, 1.0), (0.0, 0.5), (0.0, 0.25), (0.5, 0.5)):
        acc0 = acc1 = acc5 = 0
        for c in range(n_ctx):
            p = mx.random.normal((V,), key=mx.random.key(500 + c)) * a.spread
            q = flat * p + skew * mx.random.normal((V,), key=mx.random.key(900 + c))            # flat < 1: a draft FLATTER than the target (the case where coupling can lose)
            k = mx.random.key(77_000 + c)
            g = mx.random.gumbel((V,), key=k)
            x = int(mx.argmax(p / T + g).item())                                     # the verify's sample (the emitted token)
            y0 = int(mx.argmax(q).item())                                             # the argmax proposal of today
            y1 = int(mx.argmax(q + T * g).item())                                     # the coupled proposal (lambda = 1)
            y5 = int(mx.argmax(q + 0.5 * T * g).item())                               # lambda = 0.5
            acc0 += x == y0
            acc1 += x == y1
            acc5 += x == y5
        rows.append((skew, flat, acc0 / n_ctx, acc5 / n_ctx, acc1 / n_ctx))
    print("2. acceptance of the first draft, one fresh random context per trial (p = N(0, spread) logits, q = p + skew x N(0, 1)):")
    print("   (synthetic N(0, spread) logits over a huge vocabulary: flat, nothing like text; read the ORDER of the columns, not the level)")
    print("   skew  q-scale   argmax(today)   lambda=0.5   lambda=1.0")
    for r in rows:
        print("   %4.1f  %7.2f   %12.3f   %10.3f   %10.3f" % r)
    peaked = []
    for flat in (1.0, 0.6, 0.3):                                                           # a PEAKED target (code / echo): two-token world p = (.9, .1)
        a0 = a1 = 0
        for c in range(n_ctx * 5):
            g = mx.random.gumbel((2,), key=mx.random.key(300_000 + c))
            lp = mx.log(mx.array([0.9, 0.1]))
            x = int(mx.argmax(lp / T + g).item())
            lq = flat * lp
            a0 += x == int(mx.argmax(lq).item())
            a1 += x == int(mx.argmax(lq + T * g).item())
        peaked.append((flat, a0 / (n_ctx * 5), a1 / (n_ctx * 5)))
    print("   peaked 2-token target p=(.9,.1): q = flat x log p ->  argmax(today) vs coupled:", ", ".join("flat %.1f: %.3f vs %.3f" % r for r in peaked))
    # emitted-token law: x does not depend on the proposal at all; check it against softmax(p / T) on a small vocabulary with many keys
    Vs, N = 12, a.trials * 40
    p = mx.random.normal((Vs,), key=mx.random.key(1)) * 1.5
    ref = mx.softmax(p / T).tolist()
    cnt = [0] * Vs
    for i in range(N):
        g = mx.random.gumbel((Vs,), key=mx.random.key(2_000_000 + i))
        cnt[int(mx.argmax(p / T + g).item())] += 1
    tv = 0.5 * sum(abs(c / N - r) for c, r in zip(cnt, ref))
    print(f"3. emitted token = argmax(p/T + gumbel(key)) over {N} keys: TV to softmax(p/T) = {tv:.4f} (sampling floor ~ {math.sqrt(Vs / N):.4f}); the proposal never enters this line")
    return 0


class Lane:
    def __init__(self, port):
        self.port = port

    def call(self, method, path, body=None, timeout=900):
        c = http.client.HTTPConnection("localhost", self.port, timeout=timeout)
        c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        if r.status != 200:
            raise SystemExit(f"{path}: HTTP {r.status} {data[:300]!r}")
        return json.loads(data)

    def flag(self, value):
        value = str(value).rstrip("b")
        for _ in range(30):                                                       # 409 while the engine still counts the last request
            try:
                got = self.call("POST", "/admin/flags", {"spec_couple": str(value)})
                break
            except SystemExit as e:
                if "409" not in str(e):
                    raise
                time.sleep(0.5)
        else:
            raise SystemExit("/admin/flags busy")
        assert got["flags"]["spec_couple"]["value"] == str(value), got["flags"]["spec_couple"]

    def stats(self):
        s = self.call("GET", "/v1/stats")
        miss = [k for k in ("spec_drafted", "spec_accepted", "spec_rounds") if k not in s]
        if miss:
            raise SystemExit(f"/v1/stats has no {miss}: the lane does not report speculation counters, nothing to compare (keys: {sorted(s)[:40]})")
        return {k: s[k] for k in ("spec_drafted", "spec_accepted", "spec_rounds")}

    def chat(self, prompt, n, temperature, top_p):
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "stream": False, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True, "min_tokens": n}
        if temperature > 0:
            body["top_p"] = top_p
        t0 = time.perf_counter()
        d = self.call("POST", "/v1/chat/completions", body)
        return d["choices"][0]["message"].get("content") or "", d["usage"]["completion_tokens"], time.perf_counter() - t0


def lane(a):
    L = Lane(a.port)
    start = L.call("GET", "/admin/flags")["flags"]["spec_couple"]["value"]
    arms = ["0", str(a.lam), "0b"] if not a.arms else a.arms.split(",")           # "0b" = a second flag-0 arm: the base-vs-base2 noise floor (value 0)
    order = []
    for rep in range(a.reps):
        order += arms if rep % 2 == 0 else arms[::-1]                             # ABBA
    out = {}
    try:
        for temp in (a.temperature, 0.0):                                         # sampled arms, then the T=0 control
            texts = {}
            for arm in dict.fromkeys(order):
                out[(temp, arm)] = {"tok": 0, "wall": 0.0, "drafted": 0, "accepted": 0, "rounds": 0, "runs": []}
            for arm in order:
                L.flag(arm)
                L.chat("Say hello.", 8, 0.0, 1.0)                                  # warm after the switch
                for pi, pr in enumerate(PROMPTS[:a.prompts]):
                    s0 = L.stats()
                    txt, n, wall = L.chat(pr, a.tokens, temp, a.top_p)
                    s1 = L.stats()
                    o = out[(temp, arm)]
                    o["tok"] += n
                    o["wall"] += wall
                    for k, kk in (("drafted", "spec_drafted"), ("accepted", "spec_accepted"), ("rounds", "spec_rounds")):
                        o[k] += s1[kk] - s0[kk]
                    o["runs"].append(round(n / wall, 2))
                    if temp == 0.0:
                        texts.setdefault(pi, {}).setdefault(arm, txt)
            if temp == 0.0:
                ref = {pi: per.get("0") for pi, per in texts.items()}
                same = all(per.get(arm) == ref[pi] for pi, per in texts.items() for arm in per if arm not in ("0", "0b"))
                floor = all(per.get("0b") in (None, ref[pi]) for pi, per in texts.items())
                print(f"control T=0: text of the flag arms == flag 0: {'YES' if same else 'NO'}; flag 0 repeated == flag 0: {'YES' if floor else 'NO'}"
                      " (ledger section 3: served md5 can vary across requests with no lever change, so a NO here is advisory unless the repeat is YES)")
    finally:
        L.flag(start)
    print("\nT      arm   tok/s   accepted/drafted   tok/round   runs")
    for (temp, arm), o in sorted(out.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        tps = o["tok"] / o["wall"] if o["wall"] else 0.0
        acc = o["accepted"] / o["drafted"] if o["drafted"] else float("nan")
        tpr = o["tok"] / o["rounds"] if o["rounds"] else float("nan")
        print(f"{temp:<5} {arm:>4}  {tps:6.2f}   {acc:16.3f}   {tpr:9.2f}   {o['runs']}")
    if a.json:
        json.dump({f"{t}/{arm}": v for (t, arm), v in out.items()}, open(a.json, "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selfcheck")
    s.add_argument("--trials", type=int, default=400)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--spread", type=float, default=2.0, help="std of the synthetic target logits")
    s.add_argument("--vocab", type=int, default=129280, help="vocabulary width of the check (the pack's own; DS4.1 = 129280)")
    l = sub.add_parser("lane")
    l.add_argument("--port", type=int, default=8080)
    l.add_argument("--temperature", type=float, default=1.0)
    l.add_argument("--top-p", type=float, default=0.95)
    l.add_argument("--tokens", type=int, default=192)
    l.add_argument("--prompts", type=int, default=6, help="how many of PROMPTS (4 prose, 1 code, 1 echo)")
    l.add_argument("--reps", type=int, default=2, help="ABBA rounds; 2 = A B A, B A B per arm pair")
    l.add_argument("--lam", type=int, default=100, help="spec_couple value of the arm under test")
    l.add_argument("--arms", default="", help="explicit comma list of spec_couple values, e.g. 0,100,50")
    l.add_argument("--json", default="")
    a = ap.parse_args()
    return selfcheck(a) if a.cmd == "selfcheck" else lane(a)


if __name__ == "__main__":
    sys.exit(main())
