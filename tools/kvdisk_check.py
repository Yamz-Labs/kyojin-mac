#!/usr/bin/env python3
"""Disk KV checkpoints on real weights and at real size (kyojin_mlx/kvdisk.py).

  --pack DIR   round trip on a real pack or slice: prefill, checkpoint at every block, reload in a fresh cache, then compare EVERY cache array bit
               for bit (shape, dtype, bytes) with the state the prefill had at that boundary, and the logits of the next chunk plus --decode greedy
               decode steps with those of a cache that never went to disk. Exit 1 on any difference. With --ced the prefill runs the CED path
               (--mid / --text-overrides as in tools/ced_gate.py for shallow slices).
  --synth DIR  timing of a real-size DS4.1 cache: the 40-layer layout of DIR/config.json (models/ds_l10 has it), --tokens of random values in the
               shapes and dtype the model writes, --blocks checkpoint files of --block tokens: MB per file, save ms, load ms of 1 file and of the
               whole chain (the cost of a prefix hit), against the time to recompute those tokens at --prefill-tok-s.

usage: kvdisk_check.py [--pack DIR] [--synth DIR] [--dir TMPDIR] [--tokens 2048] [--block 1024] [--chunk 512] [--decode 8] [--ced] [--mid N]
                       [--text-overrides JSON] [--blocks 4] [--prefill-tok-s 2000]"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from kyojin_mlx import ced as C, kvdisk as K  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pack")
ap.add_argument("--synth")
ap.add_argument("--dir", help="where the checkpoints go (default: a temporary directory, removed at the end)")
ap.add_argument("--tokens", type=int, default=2048)
ap.add_argument("--block", type=int, default=1024)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--decode", type=int, default=8)
ap.add_argument("--ced", action="store_true")
ap.add_argument("--mid", type=int)
ap.add_argument("--text-overrides", default="")
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--prefill-tok-s", type=float, default=2000.0)
ap.add_argument("--seed", type=int, default=0)


def snapshot(cache):
    return [mx.array(a) for a in cache[0].state]


def same_arrays(a, b):
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x.shape != y.shape or x.dtype != y.dtype or not bool(mx.array_equal(x, y).item())]
    return len(a) == len(b) and not bad, bad


def real_check(a, tmp):
    import ced_gate as G
    overrides = json.loads(a.text_overrides) if a.text_overrides else None
    model, spec, cfg, tables = G.load(a.pack, overrides)
    lm = model.language_model
    tc = cfg.get("text_config", cfg)
    if a.ced:
        C.enable(model, spec, mid=a.mid)
    sig = C.signature(lm) if a.ced else "exact"
    vocab = tc["vocab_size"]
    ids = [int(x) for x in np.random.default_rng(a.seed).integers(3, vocab, a.tokens)]
    kv = K.KvDisk(tmp, {"arch": spec.name, "pack": K.pack_fingerprint(a.pack), "chunk": a.chunk, "ced": sig}, lm.make_cache, block=a.block)
    cache, snaps, s = lm.make_cache(), {}, 0
    t0 = time.time()
    while s < a.tokens:
        e = min(a.tokens, (s // a.chunk + 1) * a.chunk)
        out = lm(mx.array([ids[s:e]]), cache=cache)
        mx.eval(out.logits, *cache[0].state)
        s = e
        if s % a.block == 0:
            snaps[s] = snapshot(cache)
            kv.save(ids, s, cache)
    e0 = cache[0]
    eng = "none" if e0.engram is None else tuple(e0.engram.shape)
    live = sum(1 for slot in e0._slots for x in slot if x is not None)
    print(f"prefill {a.tokens} tokens ({sig}) {time.time() - t0:.1f} s; {len(kv.index)} files {kv.bytes() / 2 ** 20:.1f} MiB; {live} live per-layer arrays, "
          f"Engram history {eng}", flush=True)
    ok = True
    n_state = len(snaps[min(snaps)]) - len(type(cache[0]).HANDOFF)
    for b in sorted(snaps):
        if b >= a.tokens:
            continue
        sub = ids[:b + 1]                              # a request that shares b tokens and has one more: the longest proper prefix is b
        got_b, c2, _ = kv.lookup(sub)
        if got_b != b:
            print(f"FAIL lookup at {b}: got {got_b}")
            ok = False
            continue
        same, bad = same_arrays(snapshot(c2)[:n_state], snaps[b][:n_state])
        print(f"  block {b}: {n_state} arrays {'identical' if same else 'DIFFER at ' + str(bad[:6])}", flush=True)
        ok &= same
        # logits of the next chunk + decode: reload vs a cache that never left memory
        tail = ids[b:min(a.tokens, b + a.chunk)]
        c_ref = lm.make_cache()
        s2 = 0
        while s2 < b:
            e = min(b, (s2 // a.chunk + 1) * a.chunk)
            lm(mx.array([ids[s2:e]]), cache=c_ref)
            mx.eval(*c_ref[0].state)
            s2 = e
        lg_a = lm(mx.array([tail]), cache=c_ref).logits[:, -1].astype(mx.float32)
        lg_b = lm(mx.array([tail]), cache=c2).logits[:, -1].astype(mx.float32)
        mx.eval(lg_a, lg_b)
        eq = bool(mx.array_equal(lg_a, lg_b).item())
        tok = int(mx.argmax(lg_a).item())
        dec_ok = True
        for _ in range(a.decode):
            x = lm(mx.array([[tok]]), cache=c_ref).logits[:, -1].astype(mx.float32)
            y = lm(mx.array([[tok]]), cache=c2).logits[:, -1].astype(mx.float32)
            mx.eval(x, y)
            dec_ok &= bool(mx.array_equal(x, y).item())
            tok = int(mx.argmax(x).item())
        print(f"  block {b}: next-chunk logits {'identical' if eq else 'DIFFER'}, {a.decode} decode steps {'identical' if dec_ok else 'DIFFER'}", flush=True)
        ok &= eq and dec_ok
    print(kv.describe(), flush=True)
    if tables is not None:
        tables.close()
    return ok


def synth_cache(cfg, T):
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
    n = cfg["num_hidden_layers"]
    ratios, kv_src = cfg["compress_ratios"], set(cfg["kv_source_layer_ids"])
    hd, ihd, win = cfg["head_dim"], cfg["index_head_dim"], cfg["sliding_window"]
    c = DeepseekV41Cache(n)
    rng = np.random.default_rng(1)
    rnd = lambda shape, dt=mx.float16: mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dt)
    for l in range(n):
        c.window[l] = rnd((1, win, hd))
        if l in kv_src:
            r = ratios[l]
            c.compress[l] = rnd((1, T // r, hd))
            c.keys[l] = rnd((1, T // r, ihd))
        c._compress_buffers[l], c._key_buffers[l] = c.compress[l], c.keys[l]
    if cfg.get("engram_layer_ids"):
        c.engram = mx.array(rng.integers(0, 1 << 20, (1, T)).astype(np.int64))
    c.offset = T
    mx.eval(*c.state)
    return [c]


def synth_check(a, tmp):
    cfg = json.load(open(os.path.join(a.synth, "config.json")))
    cfg = cfg.get("text_config", cfg)
    n = cfg["num_hidden_layers"]
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
    kv = K.KvDisk(tmp, {"arch": "dsv4-synth", "layers": n}, lambda: [DeepseekV41Cache(n)], block=a.block)
    ids = [int(x) for x in np.random.default_rng(2).integers(3, 100000, a.block * a.blocks + 1)]
    print(f"synthetic cache: {n} layers, ratios {sorted(set(cfg['compress_ratios']))}, kv sources {cfg['kv_source_layer_ids']}, window {cfg['sliding_window']}, "
          f"block {a.block} tokens", flush=True)
    sizes, saves = [], []
    for k in range(1, a.blocks + 1):
        cache = synth_cache(cfg, k * a.block)
        t0 = time.perf_counter()
        sz = kv.save(ids, k * a.block, cache)
        saves.append(time.perf_counter() - t0)
        sizes.append(sz)
    print(f"  files (MiB): {[round(s / 2 ** 20, 1) for s in sizes]}   save ms: {[round(1000 * t) for t in saves]}", flush=True)
    for k in (1, a.blocks):
        sub = ids[:k * a.block + 1]
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            b, c2, _ = kv.lookup(sub)
            best = min(best, time.perf_counter() - t0)
        assert b == k * a.block
        recompute = k * a.block / a.prefill_tok_s
        print(f"  load of a {k * a.block}-token prefix ({k} file{'s' if k > 1 else ''}): {1000 * best:.0f} ms (page cache warm); recompute at "
              f"{a.prefill_tok_s:.0f} tok/s = {recompute:.1f} s -> {recompute / best:.0f}x", flush=True)
    # same-size files from a cold read: drop the OS page cache is not possible without root, so report the warm number and the file size
    print(f"  per 8K block: {sizes[-1] / 2 ** 20:.1f} MiB on disk ({sizes[-1] / a.block * 8192 / 2 ** 20:.1f} MiB scaled to 8192 tokens)", flush=True)
    print(kv.describe(), flush=True)
    return True


def main():
    a = ap.parse_args()
    tmp = a.dir or tempfile.mkdtemp(prefix="kvdisk_check_")
    ok = True
    try:
        if a.pack:
            ok &= real_check(a, tmp)
        if a.synth:
            ok &= synth_check(a, tmp)
    finally:
        if not a.dir:
            shutil.rmtree(tmp, ignore_errors=True)
    print("VERDICT " + ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


main()
