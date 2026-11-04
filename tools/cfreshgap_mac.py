#!/usr/bin/env python3
"""cfreshgap on Metal: the per-chunk host syncs of a fresh prefill, before / after, on the tiny random DS4.1 model (tests/mini_ds41.py). Needs mlx + mlx-vlm, no pack, no server.
Bounded inside the tool: mx.set_memory_limit(3.5 GiB), a peak check after every chunk (stops above it), --tokens <= 4096.

    python3 tools/cfreshgap_mac.py [--tokens 4096] [--chunk 256] [--reps 3]

Arms (module switches, same as the env flags): base | pool (KYOJIN_PF_POOL_LAZY=1, existing, never A/B'd on the fresh-327K lane) | keys (KYOJIN_PF_KEYS_LAZY=1, new) | both.
Per arm: a chunked prefill through the real lm (fresh cache), mx.eval of the chunk's logits and the cache state each chunk (what prefill_chunk does).
  * EXIT 1 if the logits of any chunk, or the dense pool / key rows at the end, differ from base (bit for bit);
  * deferred checks taken (fp4pool.HITS lazy) = host syncs removed, and settles;
  * median / min / max wall ms of the chunks in the first and the last quarter of the prompt, per arm, over --reps repeats. A tiny model: read it as 'no slowdown' and as the
    SIGN of the per-chunk cost; it is not the served number. The served check is tools/depth_ab.py at fresh 327K with the flag env set (see the report).
Nothing here was run by the author: the sandbox that wrote it has no Metal."""
import argparse
import gc
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "tests"), os.path.join(HERE, "..")]
import mlx.core as mx  # noqa: E402

LIMIT = int(3.5 * 2 ** 30)
ARMS = {"base": (False, False), "pool": (True, False), "keys": (False, True), "both": (True, True)}


def guard():
    if mx.get_peak_memory() > LIMIT:
        sys.exit("peak above 3.5 GiB: stopped")


def bits(a):
    return a.view({2: mx.uint16, 4: mx.uint32}[a.dtype.size]) if a.dtype.size in (2, 4) else a


def run(lm, ids, chunk, arm):
    from kyojin_mlx import fp4pool as FP
    FP.LAZY_POOL, FP.LAZY_KEYS = ARMS[arm]
    for k in FP.HITS:
        FP.HITS[k] = 0
    gc.collect()
    cache = FP.wrap_cache(lm.make_cache())
    outs, ms = [], []
    for s in range(0, len(ids), chunk):
        t0 = time.perf_counter()
        lg = lm(mx.array([ids[s:s + chunk]]), cache=cache).logits[:, -1]
        mx.eval(lg, *cache[0].state)
        ms.append((time.perf_counter() - t0) * 1e3)
        outs.append(lg)
        guard()
    dense = [mx.array(x) for sl in (cache[0].compress, cache[0].keys) for x in sl if x is not None]
    mx.eval(dense)
    return outs, dense, ms, dict(FP.HITS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if not 0 < a.tokens <= 4096 or a.chunk <= 0 or a.reps <= 0:
        sys.exit("--tokens must be in 1..4096")
    mx.set_memory_limit(LIMIT)
    import mini_ds41 as M
    lm, _ = M.make(sliding_window=16)
    ids = M.tokens(a.tokens)
    run(lm, ids[:a.chunk * 2], a.chunk, "base")                                # compile / warm
    ref, ok = None, True
    print("arm   lazy checks  settles  early-q ms/chunk (med min max)  late-q ms/chunk (med min max)   exact")
    for arm in ARMS:
        early, late, same = [], [], True
        for r in range(a.reps):
            outs, dense, ms, h = run(lm, ids, a.chunk, arm)
            q = max(1, len(ms) // 4)
            early += ms[1:q + 1]; late += ms[-q:]
            if ref is None:
                ref = (outs, dense)
            else:
                same &= (all(bool(mx.array_equal(bits(x), bits(y)).item()) for x, y in zip(ref[0], outs))
                         and len(ref[1]) == len(dense)
                         and all(x.shape == y.shape and bool(mx.array_equal(bits(x), bits(y)).item()) for x, y in zip(ref[1], dense)))
        ok &= same
        f = lambda v: "%.2f %.2f %.2f" % (statistics.median(v), min(v), max(v))
        print("%-5s %11d %8d  %-32s %-32s %s" % (arm, h["lazy"], h["settle"], f(early), f(late), "yes" if same else "NO"))
    print("peak %.2f GiB" % (mx.get_peak_memory() / 2 ** 30))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
