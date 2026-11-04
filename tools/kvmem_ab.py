#!/usr/bin/env python3
"""kvmem A/B on the served lane: KV, active memory and the text, flags off against on, from 100K to 1M tokens.

The engine's /health `kv_gb` after a request is the PREFIX STORE (the session cache is dropped when the reply ends), so the live KV of a request is read
while it runs: this tool polls /health and /v1/stats in a side thread and keeps the largest reading of the prefill and of the decode. Each length is a fresh
deterministic prompt (same seed for every arm), greedy, `--decode` tokens out; the md5 of the reply text must be equal across arms of the same length.

    python3 tools/kvmem_ab.py                                   # 100K..1M, 6 arms, one session
    python3 tools/kvmem_ab.py --lengths 100000 300000 --arms base kvseg kvseg+handoff
    python3 tools/kvmem_ab.py --users 2 --arms base kvseg+stack   # stackpack needs 2+ sessions
    python3 tools/kvmem_ab.py --arm-free "mine=kvseg:8192,kv_handoff_free:1,cdecpack:1"

Arm `kvseg+cdec` is the decode-mirror arm: `cdecpack` now hooks the READ (`stack_rows` keeps the session's own codes, `_put` packs into them, the
attention gathers the top set), so the engagement line must read `cdecpack.hits compact/read/put > 0` and `roles.mirror` ~0 while the request decodes.
A `cdecpack.hits` line of zeros means the flag never reached the served path and the arm's numbers belong to something else.

Bounded: --minutes (wall clock, checked before every request), --max-tokens (sum of prompt tokens over concurrent users), --max-gb (the poll thread stops the
run above it). The lane must be idle between arms: /admin/flags answers 409 while a request runs, and the tool retries that for --flag-wait seconds.
Nothing is timed as a claim: prefill / decode seconds are printed so the owner can price the levers, but only an n >= 3 run on the lane decides.
"""
import argparse
import hashlib
import http.client
import json
import os
import random
import sys
import threading
import time

BASE_FLAGS = {"kvseg": 0, "prefix_compact": 0, "kv_handoff_free": 0, "kv_seg_once": 0, "cdecpack": 0, "stackpack": 0}
ARMS = {
    "base": {},
    "kvseg": {"kvseg": 8192, "prefix_compact": 1},
    "compact": {"prefix_compact": 1},
    "handoff": {"kv_handoff_free": 1, "kv_seg_once": 1},
    "kvseg+handoff": {"kvseg": 8192, "prefix_compact": 1, "kv_handoff_free": 1, "kv_seg_once": 1},
    "kvseg+cdec": {"kvseg": 8192, "prefix_compact": 1, "kv_handoff_free": 1, "kv_seg_once": 1, "cdecpack": 1},
    "kvseg+cdec+stack": {"kvseg": 8192, "prefix_compact": 1, "kv_handoff_free": 1, "kv_seg_once": 1, "cdecpack": 1, "stackpack": 1},
    "kvseg+stack": {"kvseg": 8192, "prefix_compact": 1, "kv_handoff_free": 1, "kv_seg_once": 1, "stackpack": 1},
}


def get(url, path, timeout=20):
    c = http.client.HTTPConnection(url[0], url[1], timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, (json.loads(body) if body else {})


def post(url, path, obj, timeout=600):
    c = http.client.HTTPConnection(url[0], url[1], timeout=timeout)
    c.request("POST", path, json.dumps(obj), {"Content-Type": "application/json"})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, (json.loads(body) if body else {})


class Poller(threading.Thread):
    """Samples /health and /v1/stats while a request runs; keeps the largest live reading of each field."""

    def __init__(self, url, max_gb, every=0.1):
        super().__init__(daemon=True)
        self.url, self.max_gb, self.every = url, max_gb, every
        self.stop = threading.Event()
        self.samples = 0
        self.hi = {}                       # field -> largest value seen
        self.roles = None                  # last kv_roles (the largest live cache, by role)
        self.engage = {}                   # last /v1/stats engage (module HITS: cdecpack.compact / read / put / drop, stackpack, depthflat)
        self.over = False

    def _keep(self, d):
        for k in ("kv_gb", "kv_bytes", "active_gb", "peak_gb", "prefix_bytes"):
            v = d.get(k)
            if isinstance(v, (int, float)) and (k not in self.hi or v > self.hi[k]):
                self.hi[k] = v
        if self.max_gb is not None and d.get("active_gb", 0) > self.max_gb:
            self.over = True

    def run(self):
        while not self.stop.is_set():
            try:
                st, h = get(self.url, "/health", timeout=5)
                if st == 200:
                    self._keep(h)
                    self.samples += 1
                st, s = get(self.url, "/v1/stats", timeout=5)
                if st == 200:
                    self._keep(s)
                    r = s.get("kv_roles")
                    if r:
                        self.roles = r
                    e = s.get("engage")
                    if e:
                        self.engage = e
            except (OSError, http.client.HTTPException, ValueError):
                pass
            self.stop.wait(self.every)


def wait_idle(url, seconds):
    end = time.time() + seconds
    while time.time() < end:
        st, h = get(url, "/health")
        if st == 200 and not (h.get("waiting") or h.get("prefilling") or h.get("decoding")):
            return True
        time.sleep(0.5)
    return False


def set_flags(url, values, seconds):
    want = dict(BASE_FLAGS)
    want.update(values)
    end = time.time() + seconds
    while time.time() < end:
        st, body = post(url, "/admin/flags", want)
        if st == 200:
            off = [k for k, v in want.items() if v in (0, "0") or (k == "kvseg" and int(v) == 0)]
            on = {k: v for k, v in want.items() if k not in off}
            return on, off
        if st != 409:
            sys.exit(f"admin/flags {st}: {body}")
        time.sleep(1)
    sys.exit("admin/flags stayed busy: stop the lane first")


def prompt_ids(length, seed):
    rng = random.Random(seed)
    return rng.choices(range(1000, 100000), k=length)


def one(url, ids, decode, timeout, poller):
    """(text, record). One deterministic completion: the whole prompt in one request (the lane chunks it itself)."""
    body = {"prompt": ids, "max_tokens": decode, "temperature": 0, "ignore_eos": True}
    t0 = time.perf_counter()
    try:
        st, r = post(url, "/v1/completions", body, timeout=timeout)
    except (OSError, http.client.HTTPException) as e:
        return None, {"error": f"{type(e).__name__}: {e}"}
    dt = time.perf_counter() - t0
    if st != 200:
        return None, {"error": f"http {st}: {str(r)[:200]}"}
    text = r["choices"][0]["text"]
    u = r.get("usage") or {}
    n_out = u.get("completion_tokens", decode)
    return text, {"s": round(dt, 2), "md5": hashlib.md5(text.encode()).hexdigest()[:8], "out": n_out,
                  "tok_s": round(n_out / dt, 1) if dt > 0 else None,
                  "pred_tok_s": u.get("predicted_per_second")}


def run_arm(url, arm, off, lengths, decode, args, out):
    """One flag set over every length; --users prompts are sent together (the batch is what stackpack packs)."""
    rows = []
    for n in lengths:
        if time.time() > args._end:
            print("wall clock: stopping before", n, flush=True)
            break
        total = n * args.users
        if total > args.max_tokens:
            print("token bound: skipping", n, "x", args.users, flush=True)
            continue
        seeds = [args.seed + 1000 * u for u in range(args.users)]
        ids = [prompt_ids(n, s) for s in seeds]
        poller = Poller(url, args.max_gb)
        poller.start()
        texts, recs = [], []
        threads = []

        def work(i):
            t, rec = one(url, ids[i], decode, args.timeout, poller)
            texts.append((i, t)); recs.append((i, rec))

        for i in range(args.users):
            th = threading.Thread(target=work, args=(i,))
            th.start(); threads.append(th)
        for th in threads:
            th.join()
        poller.stop.set(); poller.join(timeout=5)
        recs.sort(); texts.sort(key=lambda x: x[0])
        err = next((r["error"] for _, r in recs if "error" in r), None)
        flags = {k: v for k, v in {**BASE_FLAGS, **ARMS.get(arm, {})}.items() if v not in (0, "0")}
        hits = {k: v for k, v in poller.engage.items() if k.split(".")[0] in ("cdecpack", "stackpack", "depthflat")}
        row = {"arm": arm, "on": sorted(flags), "flags": flags,
               "n": n, "users": args.users, "md5": [t and hashlib.md5(t.encode()).hexdigest()[:8] for _, t in texts],
               "poll": poller.hi, "samples": poller.samples, "roles": poller.roles, "engage": hits, "error": err,
               "run": [{k: r.get(k) for k in ("s", "tok_s", "out", "pred_tok_s")} for _, r in recs]}
        if err:
            print("ERROR", arm, n, err, flush=True)
        print(json.dumps(row, sort_keys=True), flush=True)
        if flags.get("cdecpack") or flags.get("stackpack"):
            c = hits.get("cdecpack.hits", {})
            print(f"engage n={n} {arm}: cdecpack {c or 'none'} mirror {((poller.roles or {}).get('mirror', 0) + (poller.roles or {}).get('memo', 0)) / 2**30:.3f} GiB"
                  f" counted {(poller.roles or {}).get('counted', 0) / 2**30:.3f} GiB", flush=True)
        out.write(json.dumps(row) + "\n"); out.flush()
        rows.append(row)
        if poller.over:
            print(f"over --max-gb {args.max_gb}: stopping this arm", flush=True)
            break
        if not wait_idle(url, args.flag_wait):
            print("lane not idle after the request; continuing", flush=True)
    return rows


def compare(rows):
    """Per length, every arm's reply md5 must equal the base arm's: the text check, printed once."""
    by = {}
    for r in rows:
        by.setdefault(r["n"], []).append(r)
    ok = True
    for n, rs in sorted(by.items()):
        base = next((r for r in rs if r["arm"] == "base"), None)
        if base is None or base.get("error"):
            continue
        for r in rs:
            same = r.get("error") is None and r["md5"] == base["md5"]
            ok &= same
            print(f"md5 n={n} {r['arm']}: {'equal' if same else 'DIFFERS'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="127.0.0.1:8292")
    ap.add_argument("--lengths", type=int, nargs="+", default=[100000, 200000, 300000, 400000, 500000, 750000, 1000000])
    ap.add_argument("--arms", nargs="+", default=["base", "kvseg", "kvseg+handoff", "kvseg+cdec", "kvseg+stack"])
    ap.add_argument("--arm-free", default=None, help="a custom arm 'name=flag:value,flag:value' (e.g. mine=kvseg:8192,kv_handoff_free:1)")
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--max-tokens", type=int, default=1200000)
    ap.add_argument("--max-gb", type=float, default=45.0)
    ap.add_argument("--no-base", action="store_true", help="with --arm-free: run only that arm")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--flag-wait", type=float, default=120.0)
    ap.add_argument("--out", default="scratch/kvmem_ab.jsonl")
    args = ap.parse_args()
    url = tuple(args.url.split(":"))
    url = (url[0], int(url[1]))
    args._end = time.time() + args.minutes * 60
    if args.arm_free:
        name, spec = args.arm_free.split("=", 1)
        ARMS[name] = {kv.split(":")[0].strip(): int(kv.split(":")[1]) for kv in spec.split(",")}
        args.arms = [name] if args.no_base else [a for a in args.arms if a != "base"] + ["base", name]
    st, h = get(url, "/health")
    if st != 200:
        sys.exit(f"lane /health {st}")
    print("lane:", {k: h.get(k) for k in ("kv_gb", "prefix_bytes", "active_gb", "peak_gb")}, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    with open(args.out, "a") as out:
        for arm in args.arms:
            on, off = set_flags(url, ARMS.get(arm, {}), args.flag_wait)
            print(f"arm {arm}: on={on} off={off}", flush=True)
            rows += run_arm(url, arm, off, args.lengths, args.decode, args, out)
    if not compare(rows):
        sys.exit(1)


if __name__ == "__main__":
    import os
    main()
