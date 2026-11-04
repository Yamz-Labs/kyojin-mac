#!/usr/bin/env python3
"""CED prefill and disk KV checkpoints through the real server engine, on a DS4.1 pack or slice (day-1 / Air check).

usage: serve_kv_check.py PACK [--ced-mid N] [--tokens 2600] [--chunk 512] [--disk-block 1024] [--new 16] [--mtp-depth 2] [--dir DIR]

Greedy requests through the Engine (server.open_kv_disk builds the same KvDisk as `--kv-disk`), prefix memory off so that only the disk can answer:
  ref     P1 and P2 = P1 + 3 tokens, CED on, no disk: the tokens a cold server gives
  write   P1 with a fresh disk directory: tokens == ref, one checkpoint per disk block written
  reload  a NEW engine and a NEW KvDisk on the same directory (a restarted server) serves P2: tokens == ref, the longest checkpoint (the largest multiple
          of the block below len(P2)) is a hit, only the tokens after it are prefilled, time to first token against the cold one
  exact   the same requests with CED off: another namespace, so no hit, and tokens == the exact reference (CED against exact agreement is printed)
Slices too shallow for the even CED split need --ced-mid (tools/ced_gate.py shows the overrides). With --mtp-depth the draft ring is part of every
checkpoint (the slice's own draft head accepts nothing, but its ring is saved, reloaded and must not change a token). Exit 1 on any mismatch."""
import argparse
import os
import shutil
import tempfile
import threading
import time

import mlx.core as mx
import numpy as np

from kyojin_mlx import ced
from kyojin_mlx.serve import server as SV
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve.engine import Engine

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pack")
ap.add_argument("--ced-mid", type=int)
ap.add_argument("--tokens", type=int, default=2600)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--disk-block", type=int, default=1024)
ap.add_argument("--new", type=int, default=16)
ap.add_argument("--mtp-depth", type=int, default=2)
ap.add_argument("--dir")
a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
tmp = a.dir or tempfile.mkdtemp(prefix="serve_kv_check_")
be, cfg = load_backend(a.pack, mtp=a.mtp_depth > 0, ced=True, ced_mid=a.ced_mid, log=print)
lm = be.lm
V = cfg.get("text_config", cfg)["vocab_size"]
rng = np.random.default_rng(7)
P1 = rng.integers(16, V - 16, a.tokens).tolist()
P2 = P1 + [17, 18, 19]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}", flush=True)


def serve(prompt, use_disk, kv_dir=None):
    """One request on a fresh engine on this thread; returns (tokens, seconds to first token, status, engine)."""
    ns = argparse.Namespace(kv_disk=kv_dir if use_disk else None, kv_disk_block=a.disk_block, kv_disk_gib=0, prefill_chunk=a.chunk)
    disk = SV.open_kv_disk(ns, be, log=lambda *x: None)
    eng = Engine(be, sessions=1, ctx=a.tokens + 256, chunk=a.chunk, block=a.chunk, prefix_bytes=0, mtp_depth=a.mtp_depth, disk=disk,
                 log=lambda *x: print("   [engine]", *x, flush=True))
    res = {}

    def driver():
        t0 = time.perf_counter()
        h = eng.submit(prompt, max_new=a.new)
        toks = []
        while True:
            ev = h.events.get()
            if ev[0] == "tok":
                if not toks:
                    res["ttft"] = time.perf_counter() - t0
                toks.append(ev[1])
            else:
                break
        res["toks"], res["st"] = toks, eng.status()
        eng.stop()
    th = threading.Thread(target=driver)
    th.start()
    eng.run()
    th.join()
    return res["toks"], res["ttft"], res["st"], eng


def agree(x, y):
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))


try:
    st = ced.state(lm)
    check("CED armed", st is not None and st.on, f"(mid {st.lay.mid}, tail {st.tail})" if st else "(layout not supported: pass --ced-mid for a slice)")
    ns = be.disk_namespace()
    check("disk namespace", ns is not None, str(ns and ns["ced"]))
    print(f"ref: {a.tokens} tokens, chunk {a.chunk}, disk block {a.disk_block}, mtp-depth {a.mtp_depth}", flush=True)
    r1, t1, _, _ = serve(P1, False)
    r2, t2, _, _ = serve(P2, False)
    print(f"  cold ttft: P1 {t1:.2f} s, P2 {t2:.2f} s", flush=True)
    w, tw, stw, _ = serve(P1, True, tmp)
    nfiles = (a.tokens - 1) // a.disk_block
    check("write: tokens == ref", w == r1)
    check("write: one checkpoint per disk block", stw["disk_files"] == nfiles and stw["disk_saved"] == nfiles, f"({stw['disk_files']} files, {stw['disk_bytes'] / 2 ** 20:.1f} MiB)")
    r, tr, str_, eng = serve(P2, True, tmp)
    top = ((len(P2) - 1) // a.disk_block) * a.disk_block
    check("reload: tokens == ref", r == r2, "" if r == r2 else f"(first difference at token {agree(r, r2)})")
    check("reload: hit at the longest checkpoint", str_["disk_hit_tokens"] == top and eng.stats["prefix_hit_tokens"] == top, f"({str_['disk_hit_tokens']} of {len(P2)} tokens)")
    ring = eng.prefix.last_draft
    check("reload: drafter ring restored from the checkpoint", a.mtp_depth == 0 or (ring is not None and any(x is not None for x in ring)))
    check("reload: only the rest is prefilled", eng.stats["prefill_tokens"] == len(P2) - top, f"({eng.stats['prefill_tokens']} tokens)")
    print(f"  ttft: reload {tr:.2f} s against cold {t2:.2f} s ({eng.prefix.disk.describe()})", flush=True)
    ced.set_on(lm, False)
    e1, _, _, _ = serve(P1, False)
    x, _, stx, engx = serve(P1, True, tmp)
    check("exact: other namespace, no hit", stx["disk_hits"] == 0 and engx.prefix.disk.dir != eng.prefix.disk.dir)
    check("exact: tokens == exact reference", x == e1)
    print(f"  CED against exact: {agree(r1, e1)} of {a.new} tokens agree before the first difference", flush=True)
    ced.set_on(lm, True)
finally:
    if not a.dir:
        shutil.rmtree(tmp, ignore_errors=True)
print("VERDICT " + ("PASS" if ok else "FAIL"), flush=True)
raise SystemExit(0 if ok else 1)
