#!/usr/bin/env python3
"""decspeed2_ab: the served A/B of this track's decode levers, on the lane the owner runs it on.

Goal: decode speed at short context (target 150 tok/s; prose was 43-50 tok/s). This is the one harness that turns the three levers of the track into numbers:
each arm is a flag set on the SAME loaded model (POST /admin/flags), so no restart, no pack load, and the prompts are fixed. Nothing here is loaded or
measured by the agent that wrote it: the owner runs it.

Levers (flag -> off is every one of them at its default):

    spec_ladder=1   verify width from a measured cost ladder (ladder.py; the branch that carries the ladder2 fixes)
    cdecpack=1      decode attention gathers the selected top set from the packed pool instead of the dense mirror
    rg_half=1       the many-rows dense GEMV FMAs in packed fp16 (the class the <= fuse_rows() fused launch already takes)

Arms: `base` (all three at their default) + one arm per lever + `--all` (every lever at the value given). Interleaved ABBA (`--reps`, the first rep after a
switch is a warm-up and is dropped), one fixed prose prompt and one fixed code prompt, greedy, `ignore_eos` with a fixed `--max-tokens` so both arms
emit the same length.

Per request it reads: tok/s (`timings.predicted_per_second`, decode only, TTFT excluded), `usage.completion_tokens`, and the GPU busy of the request from
/v1/stats (`gpu_busy_r*` / `gpu_n_r*` deltas, probe flag `gpu_span`; it is switched on if it is off, and the tool says so when the counters stay empty).
The text of every arm is compared against `base`: md5, token count and the first differing token, via /v1/tokenize. A speed win with different text is
not a win (the report says TEXT-CHANGED).

Bounded: --max-tokens <= 512, --reps <= 5, --budget-s (default 900) of wall clock; it stops and reports what it has. It refuses a non-loopback host
(the admin flag endpoint answers on loopback only).

    python tools/decspeed2_ab.py --base http://127.0.0.1:8292 --reps 3 --max-tokens 160
    python tools/decspeed2_ab.py --arms spec_ladder=1 --depth 5 --tag ladder_fixed5
    python tools/decspeed2_ab.py --json scratch/ab/decspeed2_ab.json

Exit 0: every arm ran, text compared. 1: a text difference on an arm (reported, not judged). 2: no admin / not reachable / refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

PROSE = (
    "The harbour at first light is a study in grey. Gulls turn over the mole, the water barely moves, and the fishing boats lie still against the "
    "wall as if they had been left there overnight by someone in a hurry. A man in a blue coat walks the length of the quay, stops at the end of it, "
    "and looks out at the channel for a long time without doing anything else. Later the light comes up properly, the engines start one after "
    "another, and the whole place becomes ordinary again, which is what most people see when they arrive."
)
CODE = (
    "Write a Python function `merge_ranges(ranges)` that takes a list of (start, end) integer pairs (inclusive, start <= end) and returns a new list "
    "of the same shape: the overlapping or touching ranges merged, sorted by start, with no two entries overlapping. State the time complexity in one "
    "sentence, then give the function and three short assertion examples.\n"
)
LEVERS = {"spec_ladder": "1", "cdecpack": "1", "rg_half": "1"}
TIMEOUT = 900.0


# ---------------------------------------------------------------------------------------------------------------- HTTP
def call(base: str, path: str, body: dict | None = None, timeout: float = 600.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def set_flags(base: str, values: dict, tries: int = 20) -> dict:
    """POST /admin/flags (the endpoint answers 409 while a request runs): the owner's switch, no restart."""
    for i in range(tries):
        try:
            return call(base, "/admin/flags", values)
        except urllib.error.HTTPError as exc:
            if exc.code != 409 or i == tries - 1:
                raise
            time.sleep(0.5)
    return {}


def gpu_now(base: str) -> dict:
    """The gpu_span counters of /v1/stats (GPU busy ms and rounds per row count), or {} when the probe is off."""
    try:
        return {k: v for k, v in (call(base, "/v1/stats").get("timing") or {}).items() if k.startswith("gpu_")}
    except (urllib.error.URLError, KeyError, ValueError):
        return {}


def gpu_delta(a: dict, b: dict) -> dict:
    return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in set(a) | set(b)}


def tokens_of(base: str, text: str) -> list:
    try:
        return call(base, "/v1/tokenize", {"prompt": text}).get("tokens") or []
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return []


# ---------------------------------------------------------------------------------------------------------------- one request
def one(base: str, prompt: str, max_tokens: int) -> dict:
    """One greedy completion. tok/s is decode only (the server's own timings dict)."""
    g0 = gpu_now(base)
    t0 = time.perf_counter()
    r = call(base, "/v1/completions", {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
                                       "ignore_eos": True, "stream": False}, timeout=600.0)
    wall = time.perf_counter() - t0
    g1 = gpu_now(base)
    text = (r.get("choices") or [{}])[0].get("text") or ""
    tm = r.get("timings") or {}
    d = gpu_delta(g0, g1)
    busy = sum(v for k, v in d.items() if k.startswith("gpu_busy_r"))
    rounds = sum(v for k, v in d.items() if k.startswith("gpu_n_r"))
    return {"text": text, "md5": hashlib.md5(text.encode()).hexdigest()[:8], "n": int((r.get("usage") or {}).get("completion_tokens") or 0),
            "tps": float(tm.get("predicted_per_second") or 0.0), "ttft": float(tm.get("ttft_s") or 0.0), "wall": wall,
            "gpu_busy": busy, "gpu_rounds": rounds,
            "gpu_ms_round": (busy / rounds) if rounds else 0.0, "gpu_ms_tok": (busy / max(1, int((r.get("usage") or {}).get("completion_tokens") or 0)))}


# ---------------------------------------------------------------------------------------------------------------- the run
def run(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8292")
    ap.add_argument("--arms", nargs="*", default=[f"{k}={v}" for k, v in LEVERS.items()],
                    help="flag=value arms (default: one arm per lever of this track)")
    ap.add_argument("--set", nargs="*", default=[], help="flags every arm carries (e.g. half_rows=2)")
    ap.add_argument("--all", action="store_true", help="add an arm with every --arms flag at once")
    ap.add_argument("--reps", type=int, default=3, help="ABBA pairs per (class, arm); the first after a switch is dropped")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--depth", type=int, default=None, help="admin mtp_fixed: pin the draft depth for every greedy session (same rows in both arms)")
    ap.add_argument("--budget-s", type=float, default=900.0)
    ap.add_argument("--tag", default="", help="label written into the JSON")
    ap.add_argument("--json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratch", "decspeed2_ab.json"))
    a = ap.parse_args(argv)
    def refuse(msg: str) -> int:
        print(f"refused: {msg}")
        return 2

    if not (1 <= a.reps <= 5 and 16 <= a.max_tokens <= 512):
        return refuse("--reps <= 5, 16 <= --max-tokens <= 512")
    if not any(h in a.base for h in ("127.0.0.1", "localhost", "[::1]")):
        return refuse("the admin flag endpoint answers on loopback only")
    t_start = time.perf_counter()

    def out_of_time() -> bool:
        return time.perf_counter() - t_start > a.budget_s

    arms = {"base": {}}
    for spec in a.arms:
        k, _, v = spec.partition("=")
        if k not in LEVERS:
            return refuse(f"{k!r} is not a lever of this track ({', '.join(LEVERS)})")
        arms[spec] = {k: v}
    if a.all:
        arms["all"] = {s.partition("=")[0]: s.partition("=")[2] for s in a.arms}
    common = {s.partition("=")[0]: s.partition("=")[2] for s in a.set}

    # probe + a reachable server + the engine's own model id
    try:
        call(a.base, "/health")
        cur = (call(a.base, "/admin/flags") or {}).get("flags") or {}
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        return refuse(f"{a.base} answers no admin flags ({exc})")
    if not cur:
        return refuse("/admin/flags returned no flags (admin off, or this is not the kyojin serve)")
    missing = [k for k in list(LEVERS) + [s.partition("=")[0] for s in a.set] if k not in cur]
    if missing:
        return refuse(f"this build has no flag {', '.join(missing)} (the lever is not in this tree)")
    if a.depth is not None:
        set_flags(a.base, {"mtp_fixed": a.depth})
    set_flags(a.base, {"gpu_span": 1})
    probes = {k: cur[k] for k in ("half_rows", "fuse_rows", "gemv_max_rows") if k in cur}

    prompts = {"prose": PROSE, "code": CODE}
    res: dict = {}
    tokens: dict = {}                                    # (class, arm) -> token ids of the last rep, for the sameness check
    for cls, prompt in prompts.items():
        for rep in range(a.reps):
            order = list(arms) if (rep % 2 == 0) else list(reversed(list(arms)))       # ABBA
            for name in order:
                if out_of_time():
                    break
                flags = {k: "0" for k in LEVERS}
                flags.update(common)
                flags.update(arms[name])
                set_flags(a.base, flags)
                if not res.get((cls, name)):                                    # the arm's first visit: a throwaway (cold graph, dropped prefix cache)
                    one(a.base, prompt, a.max_tokens)
                r = one(a.base, prompt, a.max_tokens)                            # every rep keeps its own text: a lever that flaps shows it
                res.setdefault((cls, name), []).append(r)
                if rep == a.reps - 1:
                    tokens[(cls, name)] = tokens_of(a.base, r["text"])
        if out_of_time():
            break

    # leave the lane as it was found: the track's levers back to the values read at the start, the common settings kept only if asked
    restore = {k: cur.get(k, "0") for k in LEVERS}
    restore["gpu_span"] = cur.get("gpu_span", "0")                     # a probe: off again unless it was on
    try:
        set_flags(a.base, restore)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"warning: could not restore the flags ({exc})")

    # -------------------------------------------------------------------------------------------------- report
    def med(rows, key):
        v = [r[key] for r in rows if r[key]]
        return statistics.median(v) if v else 0.0

    report, rc, worst = {}, 0, {}
    for cls in prompts:
        base_rows = res.get((cls, "base")) or []
        b_tps = med(base_rows, "tps")
        report[cls] = {}
        for name in arms:
            rows = res.get((cls, name)) or []
            if not rows:
                continue
            tps = med(rows, "tps")
            same_md5 = bool(base_rows) and rows[0]["md5"] == base_rows[0]["md5"]
            steady = len({r["md5"] for r in rows}) == 1                 # the arm's own reps agree with each other
            bt, rt = tokens.get((cls, "base")) or [], tokens.get((cls, name)) or []
            first = next((i for i, (x, y) in enumerate(zip(bt, rt)) if x != y), None) if bt and rt else None
            same_tok = bool(bt) and bt == rt
            row = {"n": len(rows), "tps": round(tps, 2), "tps_min": round(min(r["tps"] for r in rows), 2),
                   "tps_max": round(max(r["tps"] for r in rows), 2), "vs_base_pct": round(100.0 * (tps / b_tps - 1.0), 2) if b_tps else 0.0,
                   "tokens": rows[0]["n"], "gpu_ms_round": round(med(rows, "gpu_ms_round"), 2),
                   "gpu_ms_tok": round(med(rows, "gpu_ms_tok"), 2), "gpu_rounds_per_req": round(med(rows, "gpu_rounds"), 1),
                   "md5": rows[0]["md5"], "text_same": bool(same_md5), "steady": steady, "token_same": same_tok,
                   "first_diff_token": first, "n_tokens": len(rt)}
            report[cls][name] = row
            if name != "base" and (not same_md5 or not steady or (bt and rt and not same_tok)):
                rc = 1
                worst.setdefault(name, cls)

    print(f"decspeed2_ab  tag={a.tag or '-'}  base={a.base}  reps={a.reps}  max_tokens={a.max_tokens}  depth={a.depth}")
    gpu_seen = any(r["gpu_busy"] > 0 for rows in res.values() for r in rows)
    print(f"model flags: {probes}   gpu_span: {'on' if gpu_seen else 'NO (counters empty: the GPU columns below are 0, read tok/s only)'}")
    for cls in prompts:
        print(f"\n[{cls}]  {'arm':<22}{'tok/s':>8}{'vs base':>9}{'min':>8}{'max':>8}{'tok':>5}{'GPU ms/round':>14}{'GPU ms/tok':>11}  text")
        for name, row in report.get(cls, {}).items():
            if row["text_same"]:
                text = "same" + ("" if row["token_same"] or not row["n_tokens"] else f" (ids {row['n_tokens']} vs base, first diff {row['first_diff_token']})")
            else:
                text = "DIFFERS"
            print(f"      {name:<22}{row['tps']:>8.1f}{row['vs_base_pct']:>8.1f}%{row['tps_min']:>8.1f}{row['tps_max']:>8.1f}{row['tokens']:>5d}"
                  f"{row['gpu_ms_round']:>14.2f}{row['gpu_ms_tok']:>11.2f}  {text}")

    # verdict: the ledger's bar is >= 3 % and the same tokens
    verdict = "NO-WIN (no arm reached +3 % with the same text)"
    for cls in prompts:
        for name, row in report.get(cls, {}).items():
            if name == "base":
                continue
            if row["vs_base_pct"] >= 3.0 and row["text_same"] and row["token_same"]:
                verdict = f"WIN candidate: {name} +{row['vs_base_pct']:.1f} % on {cls}, same tokens (MODEL until the owner reads both classes)"
    if rc:
        verdict = f"TEXT-CHANGED: {', '.join(f'{k} on {v}' for k, v in worst.items())} (a speed number there is not a win)"
    print(f"\nverdict: {verdict}")
    out = {"tag": a.tag, "base": a.base, "reps": a.reps, "max_tokens": a.max_tokens, "depth": a.depth, "flags": probes,
           "arms": {f"{k[0]}|{k[1]}": v for k, v in res.items()}, "report": report, "verdict": verdict,
           "wall_s": round(time.perf_counter() - t_start, 1)}
    try:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"json: {os.path.abspath(a.json)}")
    except OSError as exc:
        print(f"json not written: {exc}")
    return rc


if __name__ == "__main__":
    sys.exit(run())
