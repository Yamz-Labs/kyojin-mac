"""cmudec on the Mac: N users send a 33K prompt at once to the served lane; arms off / pf_srpt / pf_floor (flags switched through /admin/flags).
Per arm it prints per-user TTFT and mean, wall, aggregate prefill (N * prompt / last first token), each user's decode tok/s WHILE another user still prefills
(streamed token timestamps: from its first token to the last first token of the group), and the engine counters that tell where the wall went
(t_pf_chunk, t_pf_inline_decode, mix_slice_rounds, floor_bursts, decode_ticks deltas). Loads no model: an HTTP client (stdlib + mu2_ab helpers).

  python tools/cmudec_mac.py --url http://127.0.0.1:8080 --n 4 --tokens 33000 --max-new 144 --reps 2 --json cmudec.json
Arms (--arms, ';' separated, k=v,k=v on top of the base): off | pf_srpt | pf_floor | e.g. "pf_floor=20,pf_floor_max=70". Run it on an idle server with --sessions >= N and no kv-disk.
Nothing here is a result until you have run it; the simulation in tests/cmudec_sim.py is a model, not a measurement."""
import argparse
import hashlib
import http.client
import json
import os
import statistics
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mu2_ab as M                                              # noqa: E402  (idle, apply, get_json, ids, barrier_run, Abort, guard)

BASE = {"pf_fair": 1, "mix_slice_ms": 250, "mu_tuner": 1, "pf_srpt": 0, "pf_floor": 0}
ARMS = {"off": {}, "pf_srpt": {"pf_srpt": 1}, "pf_floor": {"pf_floor": 15, "pf_floor_max": 60, "pf_floor_period_ms": 1000}}


def arm_flags(name):
    f = dict(BASE)
    if name in ARMS:
        f.update(ARMS[name])
        return f
    for part in name.split(","):
        k, _, v = part.partition("=")
        if not k or not v:
            raise M.Abort(f"arm {name!r}: expected off | pf_srpt | pf_floor | k=v,k=v")
        f[k.strip()] = int(v)
    return f


def stream_one(url, prompt, max_new, a):
    """One streamed completion: (list of token arrival times, text, submit time)."""
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=a.max_seconds)
    body = {"model": "served", "prompt": prompt, "max_tokens": max_new, "temperature": 0, "ignore_eos": True, "stream": True}
    t_sub = time.perf_counter()
    c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise M.Abort(f"/v1/completions -> {r.status} {r.read()[:200]!r}")
    ts, text = [], []
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        d = line[5:].strip()
        if d == b"[DONE]":
            break
        ch = (json.loads(d).get("choices") or [{}])[0].get("text", "")
        if ch:
            ts.append(time.perf_counter())
            text.append(ch)
    c.close()
    return ts, "".join(text), t_sub


def counters(url):
    s = M.get_json(url, "/v1/stats")
    tm = s.get("timing", {})
    return {"prefill_tokens": s.get("prefill_tokens", 0), "t_pf_chunk": tm.get("t_pf_chunk", 0.0), "t_pf_inline_decode": tm.get("t_pf_inline_decode", s.get("t_pf_inline_decode", 0.0)),
            "mix_slice_rounds": s.get("mix_slice_rounds", 0), "floor_bursts": s.get("floor_bursts", 0), "floor_burst_rounds": s.get("floor_burst_rounds", 0),
            "decode_ticks": s.get("decode_ticks", 0), "max_decode_tick_ms": s.get("max_decode_tick_ms")}


def decode_rates(res, last_first, active=False, min_tokens=8, min_s=2.0):
    """Per-user decode rate while another user still prefills. res: [(token times, text, t_submit)].
    active=False (the original window): tokens up to the last first token over (first token -> last first token). A user whose max_new tokens
    are all out before that point keeps the long denominator, so the rate is capped at max_new / window (144 / ~60 s = 2.4 tok/s at N=4 33K whatever
    the engine does). active=True: the window ends at the user's own last token when that comes earlier, i.e. the rate over the time the user
    was decoding; windows under min_s or with fewer than min_tokens are dropped (a user that gets its first token last has no window)."""
    out = []
    for ts, _, _ in res:
        if not ts or ts[0] >= last_first:
            continue
        end = min(ts[-1], last_first) if active else last_first
        n = sum(1 for x in ts if x <= end) - 1
        if active and (end - ts[0] < min_s or n < min_tokens):
            continue
        out.append(n / (end - ts[0]))
    return out


def run_arm(a, name, prompts):
    M.idle(a.url, a)
    M.apply(a.url, a, arm_flags(name))
    c0 = counters(a.url)
    t0 = time.perf_counter()
    res = M.barrier_run(lambda i: stream_one(a.url, prompts[i], a.max_new, a), a.n)
    wall = time.perf_counter() - t0
    c1 = counters(a.url)
    ttft = [r[0][0] - r[2] if r[0] else float("nan") for r in res]
    last_first = max(r[0][0] for r in res if r[0])
    rates = decode_rates(res, last_first)
    active = decode_rates(res, last_first, active=True)
    return {"arm": name, "ttft_s": [round(x, 1) for x in ttft], "mean_ttft_s": round(statistics.fmean(ttft), 1), "wall_s": round(wall, 1),
            "agg_prefill_tok_s": round(a.n * a.tokens / (last_first - t0)), "decode_while_prefill_tok_s": [round(x, 1) for x in rates],
            "decode_floor_min": round(min(rates), 1) if rates else None,
            "decode_active_tok_s": [round(x, 1) for x in active], "decode_active_floor_min": round(min(active), 1) if active else None,
            "delta": {k: (round(c1[k] - c0[k], 3) if isinstance(c1[k], (int, float)) and isinstance(c0[k], (int, float)) else c1[k]) for k in c0},
            "md5": hashlib.md5("|".join(r[1] for r in res).encode()).hexdigest()[:8]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=33000)
    ap.add_argument("--max-new", type=int, default=144)
    ap.add_argument("--arms", default="off;pf_srpt;pf_floor")
    ap.add_argument("--reps", type=int, default=2, help="arms are interleaved (reversed every other rep); the first rep after a switch is the warm-up only if --reps > 1 is not enough: read all reps")
    ap.add_argument("--max-seconds", type=float, default=1800)
    ap.add_argument("--rss-gib", type=float, default=3.5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    arms = [x.strip() for x in a.arms.split(";") if x.strip()]
    if a.dry_run:
        for x in arms:
            print(x, arm_flags(x))
        print(f"plan: {len(arms)} arms x {a.reps} reps, N={a.n}, {a.tokens}-token prompts, {a.max_new} new tokens each")
        return 0
    if (urllib.parse.urlparse(a.url).port or 80) == 8292 and not os.environ.get("MU2_ON_LANE"):
        raise SystemExit("refusing :8292 unless MU2_ON_LANE=1")
    M.T0 = time.perf_counter()
    h = M.idle(a.url, a)
    print("server:", json.dumps(M.session_of(h)), file=sys.stderr)
    rows = []
    nonce = int(time.time()) % 100000                                                # a run never repeats the prompts of an earlier run on the same lane
    for rep in range(a.reps):
        for k, name in enumerate(arms if rep % 2 == 0 else list(reversed(arms))):
            prompts = [M.ids(a.tokens, nonce * 100000 + 1000 * rep + 100 * (k + 1) + i + 1) for i in range(a.n)]   # new prompts per arm: a shared prompt hits the prefix cache
            r = run_arm(a, name, prompts)
            r["rep"] = rep
            rows.append(r)
            print(json.dumps(r), flush=True)
    print("\narm        rep  mean_ttft  wall   agg_prefill  decode_floor_min  md5", file=sys.stderr)
    for r in rows:
        print(f"{r['arm']:<10} {r['rep']:>3}  {r['mean_ttft_s']:>8}  {r['wall_s']:>5}  {r['agg_prefill_tok_s']:>10}  {r['decode_floor_min']!s:>16}  active_min={r['decode_active_floor_min']!s}  {r['md5']}", file=sys.stderr)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    M.apply(a.url, a, {"pf_srpt": 0, "pf_floor": 0})
    return 0


if __name__ == "__main__":
    sys.exit(main())
