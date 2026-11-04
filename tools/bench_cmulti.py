"""Multi-session scheduling on the Mac, measured through a RUNNING server (cloud/cmulti-2). This script loads no model: it is an HTTP client (stdlib only),
so its own footprint is a few MB; it aborts above --rss-gib (default 3.5) and above --max-seconds. The server is the process that holds the model.

usage:  python tools/bench_cmulti.py [--url http://127.0.0.1:8080] [--scenarios gap,children,memory] [--arms ...] [--reps 3] [--json OUT] [--dry-run]

Run the server on the Mac WITHOUT --kv-disk (a disk tier would answer the repeated prompts and hide the prefill), with the stock flags, on an idle machine.
Switching an arm = POST /admin/flags (loopback only, 409 while a request runs: the script waits for idle). Every switch drops the in-memory prefix cache.
The first run after each switch is dropped. Arms are interleaved (A B A B ...) so that drift hits all of them. Prompts are token-id lists drawn from a
seeded RNG over [--id-lo, --id-hi): the same ids for every arm of one rep (so texts can be compared), new ids for the next rep (no stale hit).

Scenarios (the targets of the task):
  gap       a session decodes (stream); another request of --pf-tokens (32768) prompt tokens arrives. Reports the max / p50 / p99 gap between two tokens of the
            decoder inside B's prefill window, the decoder's tokens in the window, B's client prefill rate, and the same B alone (rate loss = 1 - with / alone).
            Arms: off | mix50 | slice | mix50+slice  (mix_decode, mix_slice_ms)
  children  a parent of --prefix-tokens plus --children children (prefix + --suffix-tokens own tokens) all sent at once. Reports each child's time to first
            token, the prefill tokens the server did (/v1/stats delta), the cached tokens it reports, and the md5 of every child's text (T=0, ignore_eos) so
            flag on can be compared with flag off. --prefix-sweep L1,L2,..: children time to first token against prefix length (the first write into a shared
            KV buffer copies it once per holder: this is the slope of that copy).  Arms: off | pf_share
  memory    sessions of --mem-tokens tokens at once, their number derived from the calibrated bytes per token so that the demand is --mem-target-x times the
            server's KV budget (--mem-sessions N overrides). Start the server for this scenario with a SMALL explicit --kv-budget (a few GiB): the off arm is today's
            behaviour and admits everything, so the demand is what the machine has to hold. The script aborts when that demand (bytes per token x tokens x sessions)
            is above --mem-max-gib (default 6, absolute), above --mem-over x the budget, or cannot be reached with the server's session count.
            Reports kv_budget, the peak KV bytes, drops / rebuilds / parks / refusals, the server's active and peak memory, finish reason and time to first token of
            every session.  Arms: kv_admit, then off (the server's kv_peak_bytes is a lifetime maximum: the second arm's peak is only a floor).

Nothing here is a result: it prints what it saw. The ledger lines at the end are copied by hand after reading the numbers."""
import argparse
import hashlib
import http.client
import json
import random
import resource
import statistics
import sys
import threading
import time
import urllib.parse

ARMS = {
    "gap": {"off": {"mix_decode": 0, "mix_slice_ms": 0}, "mix50": {"mix_decode": 50, "mix_slice_ms": 0},
            "slice": {"mix_decode": 0, "mix_slice_ms": 250}, "mix50+slice": {"mix_decode": 50, "mix_slice_ms": 250}},
    "children": {"off": {"pf_share": 0}, "pf_share": {"pf_share": 1}},
    "memory": {"kv_admit": {"kv_admit": 1}, "off": {"kv_admit": 0}},      # kv_peak_bytes is the server's lifetime maximum: the bounded arm runs first so its peak is its own
}
T0 = time.perf_counter()


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


def post_json(url, path, body):
    c = conn(url, 60)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, (json.loads(data) if data else {})


def wait_idle(url, timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        s = get_json(url, "/health")
        if not (s.get("waiting") or s.get("prefilling") or s.get("decoding")):
            return s
        time.sleep(0.2)
    raise Abort("server did not go idle")


def set_arm(url, flags):
    for _ in range(50):
        wait_idle(url)
        code, body = post_json(url, "/admin/flags", flags)
        if code == 200:
            return body
        if code != 409:
            raise Abort(f"/admin/flags {code}: {body}")
        time.sleep(0.2)
    raise Abort("flags: server stayed busy")


def stream(url, ids, max_tokens, rec, stop=None):
    """POST a streaming completion of token ids, T=0, no EOS stop. rec gets: t_send, t_first, t_done, times (one per streamed text delta), text, usage, finish, error."""
    rec.update(t_send=time.perf_counter(), t_first=None, times=[], text=[], usage=None, finish=None, error=None)
    body = {"prompt": ids, "max_tokens": max_tokens, "temperature": 0, "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    try:
        c = conn(url)
        c.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            rec["error"] = f"{r.status} {r.read()[:300]!r}"
            return
        for raw in r:
            if stop is not None and stop.is_set():
                break
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            now = time.perf_counter()
            if ev.get("usage"):
                rec["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("text"):
                    rec["times"].append(now)
                    rec["text"].append(ch["text"])
                    rec["t_first"] = rec["t_first"] or now
                if ch.get("finish_reason"):
                    rec["finish"] = ch["finish_reason"]
        c.close()
    except Exception as e:                                                   # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["t_done"] = time.perf_counter()


def go(url, ids, max_tokens, stop=None):
    rec = {}
    th = threading.Thread(target=stream, args=(url, ids, max_tokens, rec, stop), daemon=True)
    th.start()
    return th, rec


def ids_for(seed, n, a):
    r = random.Random(seed)
    return [r.randrange(a.id_lo, a.id_hi) for _ in range(n)]


def md5(rec):
    return hashlib.md5("".join(rec["text"]).encode()).hexdigest()[:12]


def pct(v, q):
    return sorted(v)[min(len(v) - 1, int(q * len(v)))] if v else None


# ------------------------------------------------------------------------------------------------ scenario: gap
def run_gap(a, arm_flags, seed, with_decoder):
    url = a.url
    wait_idle(url)
    stop = threading.Event()
    out = {}
    if with_decoder:
        th_a, ra = go(a.url, ids_for(seed + 1, 64, a), a.a_tokens, stop)
        t = time.time()
        while len(ra["times"]) < 10:
            if ra.get("error") or time.time() - t > 120:
                raise Abort(f"decoder did not start: {ra.get('error')}")
            time.sleep(0.05)
    rb = {}
    s0 = stat(a, ["mix_slice_rounds", "t_pf_inline_decode", "prefill_tokens"])
    th_b = threading.Thread(target=stream, args=(url, ids_for(seed, a.pf_tokens, a), 1, rb), daemon=True)
    th_b.start()
    th_b.join()
    if rb.get("error") or not rb["times"]:
        raise Abort(f"long prompt failed: {rb.get('error')} {rb.get('finish')}")
    n_pf = (rb["usage"] or {}).get("prompt_tokens", a.pf_tokens)
    dur = rb["t_first"] - rb["t_send"]
    out["pf_rate"] = n_pf / dur
    out["b_ttft_s"] = dur
    s1 = stat(a, ["mix_slice_rounds", "t_pf_inline_decode", "prefill_tokens"])
    out["slice_rounds"] = s1["mix_slice_rounds"] - s0["mix_slice_rounds"]                 # 0 with a slice arm = the hook never fired (read it as 'no hook', not as 'no effect')
    out["inline_decode_s"] = s1["t_pf_inline_decode"] - s0["t_pf_inline_decode"]
    out["server_prefill_tokens"] = s1["prefill_tokens"] - s0["prefill_tokens"]
    if with_decoder:
        win = [t for t in ra["times"] if rb["t_send"] <= t <= rb["t_first"]]
        edges = [t for t in ra["times"] if t <= rb["t_first"]]
        before = [t for t in edges if t <= rb["t_send"]][-1:]
        seq = before + win + [min([t for t in ra["times"] if t > rb["t_first"]] or [rb["t_first"]])]
        gaps = [(y - x) * 1000 for x, y in zip(seq, seq[1:])]
        out.update(a_tokens_in_window=len(win), a_max_gap_ms=max(gaps), a_p50_gap_ms=statistics.median(gaps), a_p99_gap_ms=pct(gaps, 0.99),
                   a_finish_early=ra.get("finish"))
        stop.set()
        th_a.join(timeout=30)
        wait_idle(url)
    return out


def scenario_gap(a, res):
    ARMS["gap"] = {"off": {"mix_decode": 0, "mix_slice_ms": 0}, "mix50": {"mix_decode": a.mix, "mix_slice_ms": 0},
                   "slice": {"mix_decode": 0, "mix_slice_ms": a.slice_ms}, "mix50+slice": {"mix_decode": a.mix, "mix_slice_ms": a.slice_ms}}
    arms = [x for x in a.arms.split(",") if x in ARMS["gap"]] if a.arms else list(ARMS["gap"])
    rows = {k: [] for k in arms}
    alone = {k: [] for k in arms}
    for rep in range(a.reps + 1):                                             # rep 0 = the dropped first run of each arm
        order = arms if rep % 2 == 0 else arms[::-1]
        for k in order:
            guard(a)
            set_arm(a.url, ARMS["gap"][k])
            seed = 1000 + rep
            w = run_gap(a, ARMS["gap"][k], seed, True)
            n = run_gap(a, ARMS["gap"][k], seed + 500, False)
            print(f"gap rep {rep} {k:12s} max gap {w['a_max_gap_ms']:9.1f} ms  p50 {w['a_p50_gap_ms']:8.1f}  decoder tokens in window {w['a_tokens_in_window']:5d}  "
                  f"pf rate with decoder {w['pf_rate']:8.1f}  alone {n['pf_rate']:8.1f} tok/s  slice rounds {w['slice_rounds']}", flush=True)
            if rep:
                rows[k].append(w)
                alone[k].append(n)
    res["gap"] = {}
    for k in arms:
        r, al = rows[k], alone[k]
        res["gap"][k] = {"n": len(r), "max_gap_ms": [x["a_max_gap_ms"] for x in r], "p50_gap_ms": [x["a_p50_gap_ms"] for x in r],
                         "decoder_tokens_in_window": [x["a_tokens_in_window"] for x in r], "slice_rounds": [x["slice_rounds"] for x in r], "inline_decode_s": [x["inline_decode_s"] for x in r], "pf_rate_with": [x["pf_rate"] for x in r], "pf_rate_alone": [x["pf_rate"] for x in al],
                         "rate_loss_pct": [100 * (1 - w["pf_rate"] / n["pf_rate"]) for w, n in zip(r, al)]}


# ------------------------------------------------------------------------------------------------ scenario: children
def stat(a, keys):
    s = get_json(a.url, "/v1/stats")
    return {k: s.get(k, 0) for k in keys}


def run_children(a, seed, P):
    wait_idle(a.url)
    pre = ids_for(seed, P, a)
    kids = [pre + ids_for(seed + 10 + i, a.suffix_tokens, a) for i in range(a.children)]
    s0 = stat(a, ["prefill_tokens", "share_hits", "share_tokens", "prefix_hits"])
    ths = []
    t_go = time.perf_counter()
    th_p, rp = go(a.url, pre, 8)
    for k in kids:
        ths.append(go(a.url, k, a.check_tokens))
    th_p.join()
    for th, _ in ths:
        th.join()
    s1 = stat(a, ["prefill_tokens", "share_hits", "share_tokens", "prefix_hits"])
    bad = [r.get("error") for _, r in ths if r.get("error") or not r["times"]]
    if rp.get("error") or bad:
        raise Abort(f"children run failed: {rp.get('error')} {bad[:2]}")
    return {"parent_ttft_s": rp["t_first"] - t_go, "child_ttft_s": [r["t_first"] - t_go for _, r in ths],
            "server_prefill_tokens": s1["prefill_tokens"] - s0["prefill_tokens"], "share_hits": s1["share_hits"] - s0["share_hits"],
            "cached_tokens": [(r["usage"] or {}).get("prompt_tokens_details", {}).get("cached_tokens") for _, r in ths],
            "md5": [md5(r) for _, r in ths], "prompt_tokens": P + a.suffix_tokens}


def scenario_children(a, res):
    arms = [x for x in a.arms.split(",") if x in ARMS["children"]] if a.arms else list(ARMS["children"])
    sweep = [int(x) for x in a.prefix_sweep.split(",")] if a.prefix_sweep else [a.prefix_tokens]
    res["children"] = {}
    for P in sweep:
        rows = {k: [] for k in arms}
        for rep in range(a.reps + 1):
            for k in (arms if rep % 2 == 0 else arms[::-1]):
                guard(a)
                set_arm(a.url, ARMS["children"][k])
                r = run_children(a, 2000 + rep, P)
                print(f"children P={P} rep {rep} {k:9s} child ttft max {max(r['child_ttft_s']):7.2f} s  parent {r['parent_ttft_s']:7.2f} s  server prefill tokens {r['server_prefill_tokens']:7d}  "
                      f"share hits {r['share_hits']}  md5 {r['md5'][0]}...", flush=True)
                if rep:
                    rows[k].append(r)
        res["children"][str(P)] = {k: v for k, v in rows.items()}
        if len(arms) == 2:
            off, on = rows[arms[0]], rows[arms[1]]
            same = [x["md5"] == y["md5"] for x, y in zip(off, on)]
            print(f"children P={P}: texts equal between {arms[0]} and {arms[1]} in {sum(same)}/{len(same)} reps", flush=True)
            res["children"][str(P)]["texts_equal"] = same


# ------------------------------------------------------------------------------------------------ scenario: memory
MEM_KEYS = ["kv_budget_gb", "kv_peak_bytes", "kv_drops", "rebuilds", "kv_parks", "kv_refused", "kv_over_budget", "peak_gb", "active_gb"]


def scenario_memory(a, res):
    s = get_json(a.url, "/v1/stats")
    if s.get("kv_budget_gb") is None:
        raise Abort("the server has no kv budget (free memory unknown): start it with --kv-budget GIB")
    budget = s["kv_budget_gb"] * 2 ** 30
    if budget <= 0:
        raise Abort("the server's kv budget is 0")
    set_arm(a.url, ARMS["memory"]["off"])
    wait_idle(a.url)
    th, r = go(a.url, ids_for(77, a.mem_tokens, a), 4)
    th.join()
    s2 = get_json(a.url, "/v1/stats")
    bpt = s2["kv_peak_bytes"] / max(a.mem_tokens, 1)
    per = bpt * (a.mem_tokens + a.mem_new)
    n_srv = int(get_json(a.url, "/health").get("sessions") or 1)
    if a.mem_sessions <= 0:
        a.mem_sessions = max(2, -(-int(a.mem_target_x * budget) // max(int(per), 1)))
        print(f"memory: {a.mem_sessions} sessions derived for {a.mem_target_x} x the budget", flush=True)
    if a.mem_sessions > n_srv:
        raise Abort(f"{a.mem_sessions} sessions needed for the demand, the server runs {n_srv}: lower --mem-target-x or raise --mem-tokens (or start the server with a smaller --kv-budget)")
    need = per * a.mem_sessions
    if need > a.mem_max_gib * 2 ** 30:
        raise Abort(f"the demand is {need / 2 ** 30:.1f} GiB, above --mem-max-gib {a.mem_max_gib}: start the server with a smaller --kv-budget")
    print(f"memory: kv budget {budget / 2 ** 30:.2f} GiB, calibration {bpt:.0f} bytes per token, {a.mem_sessions} x {a.mem_tokens} tokens need ~{need / 2 ** 30:.2f} GiB ({need / budget:.2f} x budget)", flush=True)
    if need > a.mem_over * budget:
        raise Abort(f"the sessions would need {need / budget:.2f} x the budget, above --mem-over {a.mem_over}: lower --mem-sessions / --mem-tokens")
    res["memory"] = {"budget_bytes": budget, "bytes_per_token": bpt, "need_over_budget": need / budget, "arms": {}}
    arms = [x for x in a.arms.split(",") if x in ARMS["memory"]] if a.arms else list(ARMS["memory"])
    for k in arms:
        guard(a)
        set_arm(a.url, ARMS["memory"][k])
        before = stat(a, MEM_KEYS)
        t_go = time.perf_counter()
        ths = [go(a.url, ids_for(3000 + i, a.mem_tokens, a), a.mem_new) for i in range(a.mem_sessions)]
        peak_active = 0.0
        while any(th.is_alive() for th, _ in ths):
            peak_active = max(peak_active, get_json(a.url, "/v1/stats").get("active_gb", 0))
            time.sleep(0.5)
            guard(a)
        after = stat(a, MEM_KEYS)
        out = {"finish": [r.get("finish") or r.get("error") for _, r in ths], "ttft_s": [(r["t_first"] - t_go) if r.get("t_first") else None for _, r in ths],
               "kv_peak_bytes": after["kv_peak_bytes"], "server_peak_gb": after["peak_gb"], "sampled_active_gb": peak_active,
               **{x: after[x] - before[x] for x in ("kv_drops", "rebuilds", "kv_parks", "kv_refused", "kv_over_budget")}}
        res["memory"]["arms"][k] = out
        print(f"memory {k:9s} finish {out['finish']}  kv peak {out['kv_peak_bytes'] / 2 ** 30:.2f} GiB (budget {budget / 2 ** 30:.2f})  server peak {out['server_peak_gb']} GiB  "
              f"drops {out['kv_drops']} parks {out['kv_parks']} refused {out['kv_refused']} over-budget {out['kv_over_budget']}", flush=True)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--scenarios", default="gap,children,memory")
    ap.add_argument("--arms", default="", help="comma list of arm names to run (default: all of the scenario)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rss-gib", type=float, default=3.5)
    ap.add_argument("--max-seconds", type=float, default=7200)
    ap.add_argument("--id-lo", type=int, default=1000)
    ap.add_argument("--id-hi", type=int, default=30000)
    ap.add_argument("--pf-tokens", type=int, default=32768)
    ap.add_argument("--mix", type=int, default=50, help="gap: the mix_decode percent of the mix50 arms")
    ap.add_argument("--slice-ms", type=int, default=250, help="gap: the mix_slice_ms of the slice arms")
    ap.add_argument("--a-tokens", type=int, default=3000, help="gap: reply length of the decoding session (it must outlast B's prefill; it is cancelled after)")
    ap.add_argument("--children", type=int, default=4)
    ap.add_argument("--prefix-tokens", type=int, default=16384)
    ap.add_argument("--prefix-sweep", default="")
    ap.add_argument("--suffix-tokens", type=int, default=64)
    ap.add_argument("--check-tokens", type=int, default=24)
    ap.add_argument("--mem-sessions", type=int, default=0, help="0 = derive from the calibrated bytes per token and --mem-target-x")
    ap.add_argument("--mem-target-x", type=float, default=1.6, help="memory: demand as a multiple of the KV budget")
    ap.add_argument("--mem-max-gib", type=float, default=6.0, help="memory: absolute cap on the demand (bytes per token x tokens x sessions)")
    ap.add_argument("--mem-tokens", type=int, default=16384)
    ap.add_argument("--mem-new", type=int, default=64)
    ap.add_argument("--mem-over", type=float, default=2.5, help="memory: refuse to run when the sessions need more than this multiple of the KV budget")
    a = ap.parse_args(argv)
    sc = [x for x in a.scenarios.split(",") if x]
    if a.dry_run:
        print("plan:", {"scenarios": sc, "arms": {k: list(ARMS[k]) for k in sc}, "reps": a.reps, "note": "rep 0 of every arm is dropped"})
        return 0
    res = {"args": vars(a), "started": time.time()}
    try:
        for s in sc:
            {"gap": scenario_gap, "children": scenario_children, "memory": scenario_memory}[s](a, res)
        flag_back = {"mix_decode": 0, "mix_slice_ms": 0, "pf_share": 0, "kv_admit": 0}
        set_arm(a.url, flag_back)
    except Abort as e:
        res["aborted"] = str(e)
        print("ABORTED:", e, flush=True)
    res["rss_max_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    print(f"own max RSS {res['rss_max_mib']:.0f} MiB")
    return 1 if "aborted" in res else 0


if __name__ == "__main__":
    sys.exit(main())
