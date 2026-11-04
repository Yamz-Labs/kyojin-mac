"""flatdec3 A/B: served decode rate at 128K / 512K / 1M, levers off and on, one prefilled document, greedy answers compared.

What it measures. Decode tok/s (T=0, `--decode` greedy tokens) at three depths of ONE fixed document, for every arm of
`--arm name:{"flag": val}`. The rate is the server's own decode window, (n - 1) / (total_s - ttft_s), so the resume to the first token is out of the
number; the wall rate and the resume time are printed next to it because a lever that only moves the prefix load is not a decode lever.

Why one document and no prefill per cell. The document is built ONCE (`--doc-tokens`, default = the deepest depth) through the served lane, so its
`--kv-disk` store holds one checkpoint every `--checkpoint` tokens. Every cell then sends a strict prefix of that document (plus a fixed question),
which the lane reads back from its own checkpoint: `cached_tokens` must equal the depth and `prompt_tokens - cached_tokens` the question, or the cell
is refused and the run stops - a full prefill per cell at 1M would cost more than the whole measurement and would change what is measured.

Arms and order. Cells are interleaved per repetition: base, the levers, and base again as the last cell of every depth. The two identical base cells
price the drift inside one hold (the flatdec curve read 0-12 % between them at 512K), and every delta is quoted against their mean, so a lever has to
clear that spread to count. `--arm name:..:rebuild` is accepted for a lever that changes the KV layout, but nothing here does: these are decode-side
scores, they read the same checkpoints.

Bounded. `--max-minutes` (default 90) stops between cells (never inside one) and writes what it has; `--min-free-gib` (default 12) stops before a depth;
the JSON is rewritten after every cell, so a kill keeps the shallow depths. `--dry-run` prints the plan and the document digests and sends nothing.

    python tools/flatdec3_ab.py --out scratch/flatdec3_ab.json
    python tools/flatdec3_ab.py --no-build --reps 5 --out scratch/flatdec3_ab_n5.json
    python tools/flatdec3_ab.py --depths 131072,1048576 --decode 32 --reps 2 --dry-run

Arms it runs by default: base {}, iscore_mma {iscore_mma: 1}, iscore_blocks {iscore_blocks: 1}, both {iscore_mma: 1, iscore_blocks: 1}, and the
control base2 {}. Flags are switched with POST /admin/flags on the idle lane (no restart); the pre-run values are restored at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import statistics as S
import subprocess
import sys
import time

CKPT_SUFFIX = ".kv.safetensors"
WORDS = ("harbor senate grain river signal kernel forest bridge market copper winter ledger orbit valley "
         "canopy pillar meadow granite anchor lattice compass orchard quarry saddle textile tunnel").split()
DEFAULT_ARMS = [("base", {}), ("mma", {"iscore_mma": 1}), ("blocks", {"iscore_blocks": 1}),
                ("both", {"iscore_mma": 1, "iscore_blocks": 1})]


# ---------------------------------------------------------------------------------------------------------------- pure helpers
def doc_ids(n: int, seed: int, tokenize) -> list:
    """Deterministic token ids of length n: seeded sections of generic words, each tokenized on its own (no special tokens)."""
    rnd, out = random.Random(int(seed)), []
    while len(out) < n:
        text = f"Section {rnd.randint(0, 10 ** 6)}. " + " ".join(rnd.choice(WORDS) for _ in range(60)) + ".\n"
        out.extend(tokenize(text))
    return out[:n]


def digest(ids, n: int) -> str:
    """blake2b-16 hex of the first n ids as int32 bytes: kyojin_mlx.kvdisk.digest, without importing mlx."""
    import numpy as np
    return hashlib.blake2b(np.asarray(ids[:n], dtype=np.int32).tobytes(), digest_size=16).hexdigest()


def depth_plan(depths, checkpoint: int, ctx: int, reserve: int) -> list:
    """[(label, D)] on the checkpoint grid, clamped so that a prompt of D + reserve tokens stays below the lane's context."""
    top = (ctx - reserve) // checkpoint * checkpoint
    out, seen = [], set()
    for d in depths:
        lab = f"{d // 1000000}M" if d >= 1000000 else (f"{d // 1024}K" if d % 1024 == 0 else str(d))
        dd = min(int(d), top) // checkpoint * checkpoint
        if dd < checkpoint or dd in seen:            # below one grid step there is no checkpoint to resume from                                                  # two requested depths clamped onto the same grid point: measure once
            continue
        seen.add(dd)
        out.append((lab, dd))
    return out


def ckpt_file(root: str, ids, b: int):
    """Path of the checkpoint file of the first b tokens under any namespace of the `--kv-disk` root, or None."""
    if not root or b <= 0:
        return None
    name = f"{b:09d}-{digest(ids, b)}{CKPT_SUFFIX}"
    try:
        for ns in os.listdir(root):
            p = os.path.join(root, ns, name)
            if os.path.isfile(p):
                return p
    except FileNotFoundError:
        return None
    return None


def free_gib() -> float:
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout) / 2 ** 30
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"free percentage:\s*(\d+)", out)
        if m:
            return total * int(m.group(1)) / 100.0
    except Exception:                                                   # noqa: BLE001
        pass
    return float("nan")


def med(v):
    v = [x for x in v if x is not None]
    return float(S.median(v)) if v else float("nan")


def spread(v):
    """max - min over the values, as a percentage of their median: the drift of one arm inside a hold."""
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return None
    m = med(v)
    return round(100.0 * (max(v) - min(v)) / m, 1) if m else None


def se(v):
    v = [x for x in v if x is not None]
    return round(S.stdev(v) / len(v) ** 0.5, 2) if len(v) > 1 else None


# ---------------------------------------------------------------------------------------------------------------- lane client
class Lane:
    def __init__(self, host="127.0.0.1", port=8292, timeout=2400):
        self.host, self.port, self.timeout = host, int(port), timeout

    def _req(self, method, path, body=None):
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            c.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            if r.status != 200:
                raise RuntimeError(f"{method} {path} -> {r.status}: {data[:300].decode('utf-8', 'replace')}")
            return json.loads(data)
        finally:
            c.close()

    def health(self):
        return self._req("GET", "/health")

    def up(self, wait=900.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                self._req("GET", "/health")
                return True
            except Exception:                                           # noqa: BLE001  (not up yet / restarting)
                time.sleep(3.0)
        return False

    def stats(self):
        return self._req("GET", "/v1/stats")

    def flags(self):
        return self._req("GET", "/admin/flags")["flags"]

    def set_flags(self, values: dict) -> None:
        if not values:
            return
        for _ in range(600):
            try:
                self._req("POST", "/admin/flags", dict(values))
                return
            except RuntimeError as e:
                if "409" not in str(e):
                    raise
                time.sleep(1.0)
        raise RuntimeError("flags: the lane stayed busy for 10 minutes")

    def idle(self, wait=600.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                h = self.health()
            except Exception:                                           # noqa: BLE001
                time.sleep(2.0)
                continue
            if not (h.get("waiting") or h.get("prefilling") or h.get("decoding")):
                return True
            time.sleep(1.0)
        return False

    def complete(self, ids, max_tokens: int, seed: int = 7) -> dict:
        body = {"model": "x", "prompt": list(ids), "max_tokens": int(max_tokens), "temperature": 0, "seed": int(seed),
                "stream": False, "ignore_eos": True}
        t0 = time.time()
        d = self._req("POST", "/v1/completions", body)
        wall = time.time() - t0
        u, t = d.get("usage") or {}, d.get("timings") or {}
        n = int(u.get("completion_tokens") or 0)
        gen = max(float(t.get("total_s") or 0) - float(t.get("ttft_s") or 0), 1e-9)
        text = (d.get("choices") or [{}])[0].get("text") or ""
        return {"wall_s": round(wall, 3), "prompt": int(u.get("prompt_tokens") or 0),
                "cached": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0), "n": n,
                "ttft_s": round(float(t.get("ttft_s") or 0.0), 3), "total_s": round(float(t.get("total_s") or 0.0), 3),
                "rate": round((n - 1) / gen, 3) if n > 1 else None, "wall_rate": round((n - 1) / max(wall - float(t.get("ttft_s") or 0), 1e-9), 3)
                if n > 1 else None,
                "md5": hashlib.md5(text.encode()).hexdigest()[:10], "head": text[:120]}


# ---------------------------------------------------------------------------------------------------------------- the measurement
class Run:
    def __init__(self, lane: Lane, ids, questions: dict, checkpoint: int, decode: int, reps: int, disk: str | None,
                 min_free: float, max_minutes: float, out: str):
        self.lane, self.ids, self.q = lane, ids, questions
        self.ckpt, self.decode, self.reps = int(checkpoint), int(decode), int(reps)
        self.disk, self.min_free, self.max_minutes, self.out = disk, float(min_free), float(max_minutes), out
        self.t0 = time.time()
        self.cells: list = []

    def out_of_time(self) -> bool:
        return (time.time() - self.t0) / 60.0 > self.max_minutes

    def cell(self, name: str, flags: dict, label: str, D: int, cls: str, rep: int) -> dict:
        """One decode cell: prompt = the document's first D tokens + the question, greedy."""
        self.lane.set_flags(flags)
        r = self.lane.complete(self.ids[:D] + self.q[cls], self.decode)
        row = {"arm": name, "depth": label, "D": D, "cls": cls, "rep": rep, "flags": flags, "n": r["n"],
               "rate": r["rate"], "wall_rate": r["wall_rate"], "ttft_s": r["ttft_s"], "wall_s": r["wall_s"],
               "cached": r["cached"], "new": r["prompt"] - r["cached"], "expect_new": len(self.q[cls]),
               "md5": r["md5"], "head": r["head"], "ckpt_file": bool(ckpt_file(self.disk, self.ids, r["cached"])) if r["cached"] else False}
        row["resumed"] = r["cached"] == D and row["new"] == row["expect_new"]
        self.cells.append(row)
        print(f"    {name:8s} {label:4s} {cls:5s} rep{rep}: {r['rate']} tok/s (wall {r['wall_rate']}) resume {r['ttft_s']:6.2f} s "
              f"cached {r['cached']} +{row['new']} {'OK' if row['resumed'] else 'RESUME-MISS'} {r['md5']}", flush=True)
        return row

    def build(self) -> dict:
        fg = free_gib()
        print(f"[{time.strftime('%H:%M:%S')}] building the document ({len(self.ids)} tokens, free {fg:.1f} GiB)", flush=True)
        if not (fg >= self.min_free):
            raise SystemExit(f"STOP: free memory {fg:.1f} GiB < {self.min_free} GiB before the build")
        s0, t0 = self.lane.stats(), time.time()
        r = self.lane.complete(self.ids, 1)
        wall = time.time() - t0
        s1 = self.lane.stats()
        dt = float(s1.get("timing", {}).get("t_pf_chunk", 0)) - float(s0.get("timing", {}).get("t_pf_chunk", 0))
        toks = int(s1.get("prefill_tokens", 0)) - int(s0.get("prefill_tokens", 0))
        return {"tokens": toks, "wall_s": round(wall, 1), "chunk_s": round(dt, 1), "rate": round(toks / dt, 1) if dt else None,
                "rate_wall": round(toks / wall, 1) if wall else None, "cached": r["cached"]}

    def depth(self, label: str, D: int, arms: list, deck: list) -> dict:
        fg = free_gib()
        print(f"[{time.strftime('%H:%M:%S')}] depth {label} ({D} tokens): free {fg:.1f} GiB", flush=True)
        if not (fg >= self.min_free):
            raise SystemExit(f"STOP: free memory {fg:.1f} GiB < {self.min_free} GiB at depth {label}")
        names = [n for n, _, _ in arms]
        seq = [(n, f) for n, f, s in arms if not s] + [(n, f) for n, f, s in arms if s]
        cells = []
        for rep in range(self.reps):
            for name, flags in seq:
                for cls in self.q:
                    if self.out_of_time():
                        print(f"    --max-minutes reached: stopping before {name}/{label}/{cls} rep{rep}", flush=True)
                        return {"depth": D, "label": label, "cells": cells, "stopped": "max-minutes"}
                    cells.append(self.cell(name, flags, label, D, cls, rep))
        ctl = [(c["arm"], c["cls"], c["rep"]) for c in cells]
        bad = [c for c in cells if not c["resumed"]]
        return {"depth": D, "label": label, "cells": cells, "resume_misses": bad,
                "control_cells": len(ctl)}

    # -- per depth: rates, the drift of the two identical base cells, deltas against their mean, exactness across arms
    def summary(self, depths: list, arms: list) -> dict:
        rep = {"depths": {}, "base": arms[0][0], "control": "base2"}
        print("\n flatdec3: decode tok/s (T=0, greedy), median of reps; deltas against the mean of the two base cells")
        print(f" {'depth':>5s} {'class':>6s} " + " ".join(f"{n:>16s}" for n in [x[0] for x in arms] + ["base2"]))
        for d in depths:
            lab = d["label"]
            r = {"rates": {}, "spread_pct": {}, "delta_pct": {}, "exact": {}, "mismatch": [], "resume_misses": len(d.get("resume_misses") or [])}
            for cls in self.q:
                base = [c["rate"] for c in d["cells"] if c["arm"] == "base" and c["cls"] == cls]
                ctl = [c["rate"] for c in d["cells"] if c["arm"] == "base2" and c["cls"] == cls]
                ctl_ref = med(base + ctl)
                r["rates"][cls] = {n: med([c["rate"] for c in d["cells"] if c["arm"] == n and c["cls"] == cls]) for n, _, _ in arms}
                r["rates"][cls]["base2"] = med(ctl)
                r["spread_pct"][cls] = spread(base + ctl)
                r["delta_pct"][cls] = {n: (round(100 * (r["rates"][cls][n] / ctl_ref - 1), 1)
                                           if r["rates"][cls][n] == r["rates"][cls][n] and ctl_ref == ctl_ref and ctl_ref else None)
                                       for n, _, _ in arms}
                ref = [(c["md5"], c["n"]) for c in d["cells"] if c["arm"] == "base" and c["cls"] == cls]
                for c in [x for x in d["cells"] if x["cls"] == cls]:
                    if ref and (c["md5"], c["n"]) not in ref:
                        r["mismatch"].append({"cls": cls, "arm": c["arm"], "rep": c["rep"], "md5": c["md5"], "base": ref[0][0]})
                r["exact"][cls] = not r["mismatch"]
                print(f" {lab:>5s} {cls:>6s} " + " ".join(
                    f"{(r['rates'][cls][n] if r['rates'][cls][n] == r['rates'][cls][n] else 0):16.1f}"
                    for n in [x[0] for x in arms] + ["base2"]))
                print(f" {'':5s} {'':6s} " + "  ".join(
                    f"{n} {r['delta_pct'][cls][n]:+.1f} %" for n, _, _ in arms if r["delta_pct"][cls].get(n) is not None)
                      + f"   (base spread {r['spread_pct'][cls]} %)")
            rep["depths"][lab] = r
        labs = [d["label"] for d in depths]
        if len(labs) > 1:
            rep["flat"] = {n: {cls: round(rep["depths"][labs[-1]]["rates"][cls].get(n, float("nan"))
                                          / rep["depths"][labs[0]]["rates"][cls].get(n, float("nan")), 3)
                               if rep["depths"][labs[0]]["rates"][cls].get(n) else None
                               for cls in self.q}
                           for n, _, _ in arms}
            print("\n flatness (deepest / shallowest rate), 1.0 = flat with depth:")
            for n in [x[0] for x in arms]:
                print(f"   {n:8s} " + "  ".join(f"{cls}: {v:.3f}x" for cls, v in rep["flat"][n].items() if v))
        return rep


# ---------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("KYOJIN_MODEL", os.path.expanduser("~/ds41/model")))
    ap.add_argument("--port", type=int, default=8292)
    ap.add_argument("--depths", default="131072,524288,1040384")
    ap.add_argument("--checkpoint", type=int, default=8192, help="the lane's --kv-disk-block (checkpoint grid)")
    ap.add_argument("--ctx", type=int, default=None, help="request context of the served lane (default: the pack's max_position_embeddings)")
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--doc-tokens", type=int, default=0, help="document length (default: the deepest depth)")
    ap.add_argument("--seed", type=int, default=11031)
    ap.add_argument("--kv-disk", default=os.path.expanduser("~/kyojin-work/ds41-lane/kvdisk_chk0"))
    ap.add_argument("--min-free-gib", type=float, default=12.0)
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--arm", action="append", default=[], help='name:{"flag": value}[:rebuild] (default: base, iscore_mma, iscore_blocks, both)')
    ap.add_argument("--no-build", action="store_true", help="the document is already checkpointed on disk (nothing is prefilled up front)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the document digests, send nothing")
    ap.add_argument("--out", default="scratch/flatdec3_ab.json")
    a = ap.parse_args()

    from transformers import PreTrainedTokenizerFast
    tk = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.model, "tokenizer.json"))
    enc = lambda s: tk.encode(s, add_special_tokens=False)          # noqa: E731
    cfgp = os.path.join(a.model, "config.json")
    cfg = json.load(open(cfgp))
    ctx = int(a.ctx or (cfg.get("text_config") or {}).get("max_position_embeddings") or cfg.get("max_position_embeddings") or 32768)
    q = {"prose": enc("\nWrite a long, detailed paragraph about how the ideas in this document could be used by a small town.\n"),
         "code": enc("\nWrite a complete Python function that extracts the unique keys from this kind of record and returns them sorted, "
                     "with docstrings and unit tests.\n")}
    reserve = max(len(v) for v in q.values()) + a.decode + 8
    plan = depth_plan([int(x) for x in a.depths.split(",")], a.checkpoint, ctx, reserve)
    doc_tokens = int(a.doc_tokens or max(D for _, D in plan) + reserve)
    arms = []
    for spec in (a.arm or [f"{n}:{json.dumps(f)}" for n, f in DEFAULT_ARMS]):
        salted = spec.endswith(":rebuild")
        spec = spec[:-8] if salted else spec
        name, _, js = spec.partition(":")
        arms.append((name, json.loads(js or "{}"), salted))
    if len(arms) > 1:
        arms = arms + [("base2", {}, False)]                        # the control: the same flags as base, measured last

    print(f"flatdec3_ab: ctx {ctx}, depths {plan}, document {doc_tokens} tokens, checkpoint {a.checkpoint}, decode {a.decode} x {a.reps} reps, "
          f"max {a.max_minutes:.0f} min, arms {[x[0] for x in arms]}")
    ids = doc_ids(doc_tokens, a.seed, enc)
    print(f"  doc len {len(ids)} digest@doc {digest(ids, len(ids))} digest@128K {digest(ids, min(131072, len(ids)))}")
    if a.dry_run:
        print("  plan only: nothing sent. Missing checkpoints are expected before the first build.")
        for lab, D in plan:
            print(f"   {lab:4s} D {D:8d}  cells {len(arms) * len(q) * a.reps:3d}  prompt {D + reserve} tokens  "
                  f"checkpoint on disk {'yes' if ckpt_file(a.kv_disk, ids, D) else 'no'}")
        return 0

    lane = Lane(port=a.port)
    if not lane.up(wait=900):
        raise SystemExit("the lane does not answer /health")
    if not lane.idle(wait=900):
        raise SystemExit("the lane is busy: no measurement")
    before = lane.flags()
    run = Run(lane, ids, q, a.checkpoint, a.decode, a.reps, a.kv_disk, a.min_free_gib, a.max_minutes, a.out)
    partial = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "ctx": ctx, "doc_tokens": doc_tokens, "checkpoint": a.checkpoint,
               "decode": a.decode, "reps": a.reps, "arms": {n: f for n, f, _ in arms}, "build": None, "depths": [],
               "cells": run.cells}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def dump():
        partial["seconds"] = round(time.time() - run.t0, 1)
        json.dump(partial, open(a.out, "w"), indent=1)

    try:
        if not a.no_build:
            partial["build"] = run.build()
            dump()
        for lab, D in plan:
            d = run.depth(lab, D, arms, q)
            partial["depths"].append(d)
            dump()                                                  # every depth is on disk: a later stop keeps the shallow ones
            miss = d.get("resume_misses") or []
            if miss:
                m = miss[0]
                raise SystemExit(f"RESUME MISS {m['arm']} at depth {lab}: cached {m['cached']} of {m['D']}, {m['new']} new tokens "
                                 f"(expected {m['expect_new']}). No full prefill per cell: stopping. Check --checkpoint against the lane's "
                                 f"--kv-disk-block and that the document is the one already checkpointed (same --seed).")
        partial["summary"] = run.summary(partial["depths"], arms)
    finally:
        lane.set_flags({k: v["value"] for k, v in before.items() if v.get("runtime")})
        dump()
    print(f"\n{round(time.time() - run.t0)} s, written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
