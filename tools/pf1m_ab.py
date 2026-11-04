#!/usr/bin/env python3
"""Served A/B for pf1m (pf_cluster, the prefill index candidate block set): prefill the same prompt at several depths with the flag off and on, one request
per arm per depth, interleaved, and report tok/s, the md5 of the answer and the lever's counters (union width, fallbacks, recall sample).

The lane must be up on --port (the owner's lane, :8292). Every flag switch drops the memory prefix cache, so each arm re-prefills its depth from
scratch: that is the number measured. Bounded by --budget seconds and by one request timeout; exits 3 when the budget ran out (partial table still
printed), 1 when --recall-min is not met, 0 otherwise.

  tools/pf1m_ab.py --depths 200000,600000,1000000 --recall            # off/on, recall counters on
  tools/pf1m_ab.py --depths 1000000 --knobs probes=32,mult=8 --json out.json

Corpus: a JSON list of token ids (default ~/kyojin-work/hermes-test/corpus1m.json, refilled by repeating it when --depths asks for more)."""
import argparse
import hashlib
import http.client
import json
import os
import sys
import time

CORPUS = os.path.expanduser("~/kyojin-work/hermes-test/corpus1m.json")
FLAGS_ON = {"pf_idx_cluster": 1}
FLAGS_OFF = {"pf_idx_cluster": 0}


def call(port, path, body=None, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        if body is None:
            c.request("GET", path)
        else:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        return json.load(c.getresponse())
    finally:
        c.close()


def set_flags(port, values):
    """Admin flag switch; 409 while a request runs, so retry a little."""
    for _ in range(30):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            c.request("POST", "/admin/flags", json.dumps(values), {"Content-Type": "application/json"})
            r = c.getresponse()
            body = r.read()
            if r.status == 409:
                time.sleep(2)
                continue
            if r.status != 200:
                raise SystemExit(f"admin flags {values}: HTTP {r.status} {body[:200]!r}")
            return
        finally:
            c.close()
    raise SystemExit(f"admin flags {values}: the lane stayed busy")


def engage(port):
    st = call(port, "/v1/stats", timeout=30)
    return {k: v for k, v in (st.get("engage") or {}).items() if k.startswith("pf_cluster")}


def prefill(port, ids, max_tokens, timeout):
    t0 = time.time()
    r = call(port, "/v1/completions", {"prompt": ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}, timeout=timeout)
    wall = time.time() - t0
    text = r["choices"][0].get("text", "")
    return wall, hashlib.md5(text.encode()).hexdigest()[:8], r.get("usage") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8292)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--depths", default="200000,400000,600000,800000,1000000")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--budget", type=float, default=2400, help="seconds for the whole run")
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--knobs", default="", help="extra pf_cluster knobs for the ON arm, e.g. probes=32,mult=8")
    ap.add_argument("--recall", action="store_true", help="turn the lever's recall sample on (pf_idx_cluster_recall=1, fired=1)")
    ap.add_argument("--recall-min", type=float, default=0.0, help="fail when the measured recall is below this percent of topk")
    ap.add_argument("--same-md5", action="store_true", help="require both arms to produce the same text (they do not: the lever is approximate)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    ids = json.load(open(args.corpus))
    if not isinstance(ids, list) or not ids:
        raise SystemExit(f"{args.corpus}: expected a JSON list of token ids")
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    on = dict(FLAGS_ON)
    for kv in filter(None, args.knobs.split(",")):
        k, v = kv.split("=")
        on["pf_idx_cluster_" + k.strip()] = int(v)
    if args.recall:
        on["pf_idx_cluster_recall"] = 1
    off = {**on, "pf_idx_cluster": 0}                                             # the knobs stay where the ON arm puts them: one variable
    end = time.time() + args.budget
    rows, out_of_time, worst = [], False, None
    print(f"# pf1m_ab: port {args.port}, corpus {args.corpus} ({len(ids)} ids), depths {depths}, on={on}", flush=True)

    for depth in depths:
        ids_d = (ids * (1 + depth // len(ids)))[:depth]
        for arm, values in (("off", off), ("on", on)):
            if time.time() > end:
                out_of_time = True
                break
            set_flags(args.port, values)
            before = engage(args.port)
            wall, md5, usage = prefill(args.port, ids_d, args.max_tokens, args.request_timeout)
            after = engage(args.port)
            hits = {}
            for name, d in after.items():
                b = before.get(name, {})
                hits.update({k: v - b.get(k, 0) for k, v in d.items()})
            tot = hits.get("recall_total", 0)
            rec = 100.0 * hits.get("recall_hit", 0) / tot if tot else None
            blocks = max(hits.get("blocks", 0), 1)
            row = {"depth": depth, "arm": arm, "s": round(wall, 2), "tok_s": round(depth / wall),
                   "md5": md5, "kv_gb": call(args.port, "/health", timeout=30).get("kv_gb"),
                   "union_keys_per_block": round(hits.get("union_keys", 0) / blocks),
                   "fallback": hits.get("fallback", 0), "recall_pct": None if rec is None else round(rec, 1)}
            rows.append(row)
            if rec is not None and (worst is None or rec < worst):
                worst = rec
            print(json.dumps(row), flush=True)
        if out_of_time:
            print("# budget exhausted: remaining depths skipped", flush=True)
            break

    print("\n| depth | arm | s | tok/s | md5 | keys scored per block | fallback | recall % |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['depth']} | {r['arm']} | {r['s']} | {r['tok_s']} | {r['md5']} | {r['union_keys_per_block']} | {r['fallback']} | "
              f"{'-' if r['recall_pct'] is None else r['recall_pct']} |")
    if args.json:
        json.dump({"rows": rows, "on": on, "budget_hit": out_of_time}, open(args.json, "w"), indent=1)
    if args.same_md5:
        for depth in depths:
            got = {r["md5"] for r in rows if r["depth"] == depth}
            if len(got) > 1:
                print(f"# md5 differs at {depth} (expected: the lever is approximate): {sorted(got)}")
                return 1
    if worst is not None and worst < args.recall_min:
        print(f"# recall {worst:.1f} % < {args.recall_min} %: the arm is not usable")
        return 1
    return 3 if out_of_time else 0


if __name__ == "__main__":
    sys.exit(main())
