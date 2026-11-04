"""mu2: several sessions on one lane. Aggregate decode rate at N = 1, 2, 3, 4, the worst time to first token of those N, and the first token of a
small request that arrives while another session prefills 100K+ tokens. Flags off against on, interleaved, bounded.

This script loads no model: it is an HTTP client (stdlib only), so its own footprint is a few MB. It aborts above --rss-gib and above --max-seconds.
The server is the process that holds the model. Nothing here is a result: it prints what it saw.

usage:  python tools/mu2_ab.py [--url http://127.0.0.1:8080] [--arms off,on] [--n 1,2,3,4] [--reps 2]
                               [--agg-tokens 200] [--max-new 96] [--pf-tokens 100000] [--short-tokens 32]
                               [--json OUT] [--dry-run] [--max-seconds 2400] [--rss-gib 3.5]

Set the flag kv_disk_off=1, or run the server without --kv-disk (a disk tier answers the repeated prompts and hides the prefill), on an idle machine, with --sessions >= 4 (the
N arms ask for N at once; a smaller --sessions N is a queue, not a starvation). Switching an arm = POST /admin/flags (409 while a request runs: the
script waits for idle) and drops the in-memory prefix cache. The first run after a switch is dropped. Arms are interleaved (off on off on) so drift
hits all of them, and the prompts of one rep are shared by its arms so the texts can be compared.

Arms:
  off        mix_slice_ms=0  pf_fair=0  mu_tuner=0  stack_dedupe=0  bound_dec=0   (ttft_stages stays on: timers only)
  on         mix_slice_ms=250 pf_fair=1 mu_tuner=1  stack_dedupe=1
  on_dec     `on` + bound_dec=1                                     (mu3: the slice/yield lever now reaches the decoder half)
  k=v,...    any explicit set of admin flag names (e.g. pf_fair=1, or bound_dec=1 alone to price the extra bounds); an arm is always applied
             on top of `off`.

Scenarios:
  agg N      N requests at once, prompt --agg-tokens ids, --max-new tokens each, temperature 0. Reports per request its server-side ttft, and the
             aggregate completion tokens per second of the wall the N shared, plus the max ttft of the N (= queue wait included).
  prefill    a 100K request starts; ~--spawn-after s later a --short-tokens request is sent. Reports the short one's ttft (server-side, and its
             stage split when ttft_stages is on), the 100K prefill's own rate over the window (prefill_tokens / t_pf_chunk deltas), and the
             server's max decode tick in the window.
"""
import argparse
import http.client
import json
import os
import random
import resource
import statistics
import sys
import threading
import time
import urllib.parse

T0 = time.perf_counter()

OFF = {"mix_slice_ms": 0, "pf_fair": 0, "mu_tuner": 0, "stack_dedupe": 0, "bound_dec": 0, "ttft_stages": 1}
ON = {"mix_slice_ms": 250, "pf_fair": 1, "mu_tuner": 1, "stack_dedupe": 1, "ttft_stages": 1}
ON_DEC = {**ON, "bound_dec": 1}                                    # mu3: the combined arm, decoder-half bounds and yields included


class Abort(RuntimeError):
    pass


def guard(a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024                      # bytes on macOS, KiB on Linux
    if rss > a.rss_gib * 2 ** 30:
        raise Abort(f"own RSS {rss / 2 ** 30:.2f} GiB above --rss-gib {a.rss_gib}")
    if time.perf_counter() - T0 > a.max_seconds:
        raise Abort(f"--max-seconds {a.max_seconds} reached")


# ------------------------------------------------------------------------------------------------ http
def conn(url, timeout=3600):
    u = urllib.parse.urlparse(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def get_json(url, path):
    c = conn(url, 60)
    c.request("GET", path)
    r = c.getresponse()
    out = json.loads(r.read())
    c.close()
    return out


def post_json(url, path, body, timeout=3600):
    c = conn(url, timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data or b"{}")


def idle(url, a, quiet_s=0.3, timeout=600):
    """Wait until the engine holds nothing (a flag switch is refused while a request runs)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        guard(a)
        try:
            h = get_json(url, "/health")
        except OSError:
            time.sleep(0.5)
            continue
        if not (h.get("waiting") or h.get("prefilling") or h.get("decoding")):
            time.sleep(quiet_s)
            return h
        time.sleep(0.2)
    raise Abort("the server never went idle")


def session_of(h):
    return {k: h.get(k) for k in ("sessions", "kv_budget_gb", "mtp_depth", "prefix_entries")}


def arm_flags(name):
    if name == "off":
        return dict(OFF)
    if name == "on":
        return dict(ON)
    if name == "on_dec":
        return dict(ON_DEC)
    out = dict(OFF)
    for part in name.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        if not k or not v:
            raise Abort(f"arm {name!r}: expected flag=value pairs, got {part!r}")
        out[k.strip()] = int(v)
    return out


def apply(url, a, flags):
    st, body = post_json(url, "/admin/flags", flags, timeout=120)
    if st != 200:
        raise Abort(f"/admin/flags {flags} -> {st} {str(body)[:200]}")
    return body


# ------------------------------------------------------------------------------------------------ prompts
def ids(n, seed):
    """Token ids in the vocabulary band the packs share, deterministic per (n, seed): the same prompt for every arm of a rep."""
    rnd = random.Random(seed * 1000003 + n)
    return [rnd.randrange(100, 120000) for _ in range(n)]


def ask(url, prompt, max_new, a):
    """One completion, no stream: the server's own timings (ttft with queue wait, completion tokens, total)."""
    body = {"model": "served", "prompt": prompt, "max_tokens": max_new, "temperature": 0,
            "ignore_eos": True, "stream": False}
    t0 = time.perf_counter()
    st, resp = post_json(url, "/v1/completions", body, timeout=a.max_seconds)
    wall = time.perf_counter() - t0
    if st != 200:
        raise Abort(f"/v1/completions -> {st} {str(resp)[:200]}")
    tim = resp.get("timings") or {}
    return {"ttft_s": tim.get("ttft_s", wall), "total_s": tim.get("total_s", wall), "wall_s": wall,
            "n": (resp.get("usage") or {}).get("completion_tokens", 0), "stages": tim.get("ttft_stages"),
            "text": (resp.get("choices") or [{}])[0].get("text", "")}


def barrier_run(fn, count):
    """Run fn(i) in `count` threads, all released at the same instant; returns their results."""
    out = [None] * count
    err = []
    gate = threading.Barrier(count)

    def one(i):
        try:
            gate.wait(timeout=120)
            out[i] = fn(i)
        except BaseException as e:                                   # noqa: BLE001  (a thread must not hang the run)
            err.append(e)

    ts = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(count)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if err:
        raise Abort(f"a request failed: {err[0]!r}")
    return out


# ------------------------------------------------------------------------------------------------ scenarios
def scen_agg(url, a, n, prompt, max_new):
    """N requests at once: aggregate tok/s over the wall they shared, max ttft of the N."""
    guard(a)
    t0 = time.perf_counter()
    res = barrier_run(lambda i: ask(url, prompt, max_new, a), n)
    wall = time.perf_counter() - t0                                       # from the release to the last reply
    toks = sum(r["n"] for r in res)
    return {"scen": "agg", "n": n, "tok_s": round(toks / max(wall, 1e-9), 2), "tokens": toks, "wall_s": round(wall, 2),
            "ttft_max_s": round(max(r["ttft_s"] for r in res), 3), "ttft_mean_s": round(statistics.fmean(r["ttft_s"] for r in res), 3),
            "per_s": [round(r["n"] / max(r["total_s"], 1e-9), 2) for r in res],
            "md5": __import__("hashlib").md5("|".join(r["text"] for r in res).encode()).hexdigest()[:8]}


def scen_prefill(url, a, long_prompt, short_prompt, max_new, spawn_after):
    """A long prefill runs; a small request arrives in the middle of it: how long does its first token take?"""
    guard(a)
    before = get_json(url, "/v1/stats")
    box = {}

    def long_one():
        box["long"] = ask(url, long_prompt, 4, a)

    th = threading.Thread(target=long_one, daemon=True)
    th.start()
    time.sleep(max(spawn_after, 0.0))
    st = get_json(url, "/v1/stats")
    mid = {"prefill_tokens": st.get("prefill_tokens", 0), "n_pf_chunk": st.get("n_pf_chunk", 0)}
    short = ask(url, short_prompt, max_new, a)                             # sent while the long prefill runs
    guard(a)
    th.join(timeout=a.max_seconds)
    after = get_json(url, "/v1/stats")
    dtok = after.get("prefill_tokens", 0) - before.get("prefill_tokens", 0)
    dchunk = after.get("n_pf_chunk", 0) - before.get("n_pf_chunk", 0)
    dchunk_s = after.get("timing", {}).get("t_pf_chunk", 0.0) - before.get("timing", {}).get("t_pf_chunk", 0.0)
    return {"scen": "prefill", "n": 1, "short_ttft_s": round(short["ttft_s"], 3), "short_ttft_stages": short["stages"],
            "short_text": short["text"], "long_tok_s": round(dtok / dchunk_s, 1) if dchunk_s > 0 else None,
            "prefill_tokens": dtok, "chunks": dchunk,
            "max_decode_tick_ms": after.get("max_decode_tick_ms"),
            "mix_slice_rounds": after.get("mix_slice_rounds", 0) - before.get("mix_slice_rounds", 0),
            "share_hits": after.get("share_hits", 0) - before.get("share_hits", 0),
            "share_held": after.get("share_held", 0) - before.get("share_held", 0),
            "ttft_first_chunk_ms": after.get("timing", {}).get("t_ttft_first_chunk_max"),
            "mid_after_s": round(mid["prefill_tokens"] / max(dtok / max(dchunk_s, 1e-9), 1e-9), 2) if dtok else None}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--arms", default="off,on", help="comma separated: off, on, or flag=value,... (see the doc)")
    ap.add_argument("--n", default="1,2,3,4", help="session counts of the agg scenario")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--agg-tokens", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--pf-tokens", type=int, default=100000)
    ap.add_argument("--short-tokens", type=int, default=32)
    ap.add_argument("--spawn-after", type=float, default=0.5, help="seconds between the long prefill's submit and the small request")
    ap.add_argument("--max-seconds", type=float, default=2400)
    ap.add_argument("--rss-gib", type=float, default=3.5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    port = urllib.parse.urlparse(a.url).port or 80
    if port == 8292 and not os.environ.get("MU2_ON_LANE"):
        raise SystemExit("refusing :8292 unless MU2_ON_LANE=1: that is the lane's own port")
    arms = [x.strip() for x in a.arms.split(";" if ";" in a.arms else ",") if x.strip()]   # ";" separates arms when an arm is k=v,k=v
    ns = [int(x) for x in a.n.split(",") if x.strip()]
    if any(n < 1 or n > 8 for n in ns):
        raise SystemExit("--n must be in 1..8")

    if a.dry_run:
        for arm in arms:
            print(f"arm {arm}: {arm_flags(arm)}")
        print(f"plan: {len(arms)} arms x {a.reps} reps: agg N={ns} ({a.max_new} tokens each, {a.agg_tokens}-token prompt), "
              f"then a {a.short_tokens}-token request {a.spawn_after}s into a {a.pf_tokens}-token prefill. "
              f"Bounds: {a.rss_gib} GiB RSS, {a.max_seconds}s.")
        return 0

    h = idle(a.url, a)
    print("server:", json.dumps(session_of(h)), file=sys.stderr)
    rows, prompts = [], {}
    for rep in range(a.reps):
        order = arms if rep % 2 == 0 else list(reversed(arms))
        for arm in order:
            flags = arm_flags(arm)
            apply(a.url, a, flags)
            idle(a.url, a)
            guard(a)
            long_prompt = ids(a.pf_tokens, rep)
            short_prompt = ids(a.short_tokens, rep + 100)
            for n in ns:
                p = prompts.setdefault((rep, n), ids(a.agg_tokens, rep * 10 + n))
                r = scen_agg(a.url, a, n, p, a.max_new)
                r.update(rep=rep, arm=arm)
                rows.append(r)
                print(json.dumps(r), flush=True)
            r = scen_prefill(a.url, a, long_prompt, short_prompt, min(a.max_new, 8), a.spawn_after)
            r.update(rep=rep, arm=arm)
            rows.append(r)
            print(json.dumps(r), flush=True)

    print("\narm      n   tok/s  ttft_max_s  short_ttft_s  long_tok_s  md5", file=sys.stderr)
    for r in rows:
        if r["scen"] == "agg":
            print(f"{r['arm']:<8} {r['n']}  {r['tok_s']:>7.2f}  {r['ttft_max_s']:>10.3f}  {'-':>12}  {'-':>10}  {r['md5']}", file=sys.stderr)
        else:
            print(f"{r['arm']:<8} pf  {'-':>7}  {'-':>10}  {r['short_ttft_s']:>12.3f}  {str(r['long_tok_s']):>10}", file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"url": a.url, "arms": {k: arm_flags(k) for k in arms}, "rows": rows}, f, indent=1)
        print("wrote", a.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Abort as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
