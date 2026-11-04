#!/usr/bin/env python3
"""depth_ab: long-context A/B in minutes. Every arm resumes one document from its --kv-disk checkpoint at depth D, then prefills a short tail and decodes, on the
served lane, flags switched with POST /admin/flags. No full prefill per arm: a 1M cell costs the restore plus two chunks.

A cell: switch the flags (drops the memory prefix store), warm request D + 64 tokens (disk restore, its time is printed), then the measured request D + --tail
tokens with --decode greedy tokens: prefill tok/s = new tokens / ttft, decode tok/s = the server's window, largest active and KV of /health while it runs, md5 of the
reply. kv_disk_ro=1 keeps the store as it is, so all arms read the same checkpoint. base2 prices the drift. Levers that change the KV layout show their prefill and
decode cost here, not their memory: a restored pool is not rebuilt, so read memory from one full run of the final candidate.

    tools/depth_ab.py --build                                     # once: writes the missing checkpoints (a real prefill from the deepest one on disk)
    tools/depth_ab.py --arm 'pfc:{"pf_idx_cluster":1}' --arm 'blocks:{"iscore_mma":1,"iscore_blocks":1}' --depths 524288,1040384
"""
import argparse
import hashlib
import http.client
import json
import os
import statistics
import sys
import threading
import time

CORPUS = os.path.expanduser("~/kyojin-work/hermes-test/corpus1m.json")
CKPT_SUFFIX = ".kv.safetensors"


def digest(ids, n):
    import array
    return hashlib.blake2b(array.array("i", ids[:n]).tobytes(), digest_size=16).hexdigest()


def on_disk(root, ns, ids, b):
    """In the lane's own namespace: a chain built under another refusal-hook spec or pack does not resume."""
    return os.path.isfile(os.path.join(root, ns, f"{b:09d}-{digest(ids, b)}{CKPT_SUFFIX}"))


def pin(root, ns, ids, b):
    """Protect the chain ending at b from LRU eviction (kvdisk pins.json): a 1M checkpoint costs 10 minutes to rebuild, serving traffic must never remove it."""
    dg = digest(ids, b)
    if os.path.isfile(os.path.join(root, ns, f"{b:09d}-{dg}{CKPT_SUFFIX}")):
        path = os.path.join(root, ns, "pins.json")
        try:
            pins = json.load(open(path))
        except (OSError, ValueError):
            pins = []
        if [b, dg] not in pins:
            tmp = path + ".new"
            with open(tmp, "w") as f:
                json.dump(pins + [[b, dg]], f)
            os.replace(tmp, path)
        print(f"pinned {b} in {ns}", flush=True)


def req(port, method, path, body=None, timeout=2400):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, (json.loads(data) if data else {})


def set_flags(port, values):
    end = time.time() + 300
    while time.time() < end:
        st, d = req(port, "POST", "/admin/flags", values, 60)
        if st == 200:
            return
        if st != 409:
            raise SystemExit(f"/admin/flags {values}: {st} {str(d)[:200]}")
        time.sleep(1)
    raise SystemExit("/admin/flags stayed busy")


class ArmFail(Exception):
    """One arm's request failed (engine error): record it and go on with the next arm."""


def complete(port, ids, max_tokens):
    t0 = time.time()
    st, d = req(port, "POST", "/v1/completions", {"model": "x", "prompt": list(ids), "max_tokens": max_tokens, "temperature": 0, "seed": 7, "stream": False, "ignore_eos": True})
    if st != 200:
        raise ArmFail(f"completion {st}: {str(d)[:200]}")
    u, t = d.get("usage") or {}, d.get("timings") or {}
    n = int(u.get("completion_tokens") or 0)
    ttft, total = float(t.get("ttft_s") or 0), float(t.get("total_s") or 0)
    return {"wall": time.time() - t0, "prompt": int(u.get("prompt_tokens") or 0), "cached": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0), "n": n, "ttft": ttft,
            "dec": (n - 1) / max(total - ttft, 1e-9) if n > 1 else None, "md5": hashlib.md5(((d.get("choices") or [{}])[0].get("text") or "").encode()).hexdigest()[:8]}


class Poll(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port, self.stop, self.active, self.kv = port, False, 0.0, 0.0

    def run(self):
        while not self.stop:
            try:
                _, h = req(self.port, "GET", "/health", None, 5)
                self.active, self.kv = max(self.active, float(h.get("active_gb") or 0)), max(self.kv, float(h.get("kv_gb") or 0))
            except OSError:
                pass
            time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8292)
    ap.add_argument("--corpus", default=CORPUS, help="JSON list of token ids (the document)")
    ap.add_argument("--kv-disk", default=os.path.expanduser("~/kyojin-work/ds41-lane/kvdisk_chk0"))
    ap.add_argument("--depths", default="131072,524288,1040384")
    ap.add_argument("--tail", type=int, default=16384)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--arm", action="append", default=[], help='name:{"flag": value}')
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=30)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ids = json.load(open(a.corpus))
    ids = ids["ids"] if isinstance(ids, dict) else ids
    depths = [int(x) for x in a.depths.split(",")]
    tail_end = max(depths) + a.tail
    if len(ids) < tail_end:
        ids = (ids * (tail_end // len(ids) + 1))[:tail_end]
    ns = req(a.port, "GET", "/v1/stats")[1]["disk_ns"]
    for D in depths:
        print(f"depth {D} on disk {on_disk(a.kv_disk, ns, ids, D)} in {ns}", flush=True)
    t0 = time.time()
    flags_now = req(a.port, "GET", "/admin/flags")[1]["flags"]
    if "kv_disk_ro" not in flags_now:
        raise SystemExit("this tree has no kv_disk_ro")
    if a.build:
        set_flags(a.port, {"kv_disk_ro": 0})
        for D in sorted(depths):
            if not on_disk(a.kv_disk, ns, ids, D):
                r = complete(a.port, ids[:D + 16], 1)
                print(f"built {D}: cached {r['cached']} in {r['wall']:.0f} s ({(r['prompt'] - r['cached']) / max(r['ttft'], 1e-9):.0f} tok/s)", flush=True)
        pin(a.kv_disk, ns, ids, max(depths))
    arms = []
    for spec in a.arm:
        name, _, js = spec.partition(":")
        arms.append((name, json.loads(js or "{}")))
    if not arms:
        return 0
    arms = [("base", {})] + arms + [("base2", {})]
    keys = sorted({k for _, f in arms for k in f})
    missing = [k for k in keys if k not in flags_now]
    if missing:
        raise SystemExit(f"flags not in this tree: {missing}")
    baseline = {k: flags_now[k]["value"] for k in keys}
    res = []
    try:
        for D in depths:
            base = []
            for name, f in arms:
                if (time.time() - t0) / 60 > a.max_minutes:
                    raise SystemExit("max minutes")
                set_flags(a.port, {**baseline, **f, "kv_disk_ro": 1})
                p = Poll(a.port)
                try:
                    w = complete(a.port, ids[:D + 64], 1)
                    if w["cached"] < D:
                        raise SystemExit(f"RESUME MISS at {D}: cached {w['cached']}. Run --build.")
                    p.start()
                    m = complete(a.port, ids[:D + a.tail], a.decode)
                except ArmFail as e:
                    print(f"{D:8d} {name:12s} FAILED {e}", flush=True)
                    res.append(dict(arm=name, depth=D, error=str(e)))
                    if a.json:
                        json.dump(res, open(a.json, "w"), indent=1)
                    continue
                finally:
                    p.stop = True
                new = m["prompt"] - m["cached"]
                pf = new / max(m["ttft"], 1e-9)
                c = dict(arm=name, depth=D, restore=round(w["ttft"], 1), cached=m["cached"], new=new, pf=round(pf), dec=round(m["dec"] or 0, 1), md5=m["md5"], active=round(p.active, 1), kv=round(p.kv, 2))
                res.append(c)
                if name.startswith("base"):
                    base.append(pf)
                ref = statistics.mean(base) if base else pf
                print(f"{D:8d} {name:12s} restore {c['restore']:6.1f} s  new {new:6d}  prefill {c['pf']:5d} tok/s ({100 * (pf / ref - 1):+5.1f} %)  decode {c['dec']:6.1f}  md5 {c['md5']}  active {c['active']:6.1f}  kv {c['kv']:5.2f}", flush=True)
                if a.json:
                    json.dump(res, open(a.json, "w"), indent=1)
    finally:
        set_flags(a.port, {**baseline, "kv_disk_ro": 0})
    print(f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
