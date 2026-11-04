#!/usr/bin/env python3
"""ckvseg on Metal: KYOJIN_KV_HANDOFF_FREE 0 against 1 (and KYOJIN_KV_SEG_ONCE 0 against 1 on the prefix store) on the tiny random DS4.1 model (tests/mini_ds41.py).
Needs mlx + mlx-vlm, no pack, no server. Bounded inside the tool: mx.set_memory_limit(3.5 GiB) and a peak check after every step (stops above it), --tokens <= 8192, --turns <= 8.

    python3 tools/ckvseg_mac.py [--tokens 2048] [--chunk 256] [--steps 16] [--turns 6]

Per arm (flag 0, flag 1): a chunked prefill through the real lm, then --steps single decode steps (direct lm calls; the batched served core is not exercised here), then
  * logits equality of every prefill chunk and every decode row between the two arms. EXIT 1 if any differs;
  * bytes: what the engine bills (arrays_of), per role from kvledger.summary (packed, state, handoff, mirror, slack), and MLX active memory kept by the cache (gc + eval before the read),
    each per token;
  * time: median wall ms of a prefill chunk and of a decode step (a tiny model: read it as 'no slowdown', never as a gain; the tool prints the spread so you can see the noise).
Part C: a nested prefix store (kvseg segments, one snapshot per turn): store bytes after each entry, the sum of the entry sizes, the live session's `kv_bytes` with kv_seg_once 0 and 1.
Nothing here is a claim about the real DS4.1 pack: per-token figures are the tiny model's (fp32 here); the formula for the real sizes is in docs/reports/ckvseg_REPORT.md."""
import argparse
import gc
import os
import statistics
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "tests"), os.path.join(HERE, "..")]
import mlx.core as mx  # noqa: E402

GIB = 2 ** 30
LIMIT = int(3.5 * GIB)


def guard():
    if mx.get_peak_memory() > LIMIT:
        sys.exit("peak above 3.5 GiB: stopped")


def mem():
    gc.collect(); mx.eval()
    return mx.get_active_memory()


def arm(lm, ids, more, chunk, steps, flag):
    """One arm: returns (logit rows list, stats dict)."""
    from kyojin_mlx import fp4pool as FP, kvledger as KL, rows as R
    from kyojin_mlx.serve.backends import arrays_of
    os.environ["KYOJIN_KV_HANDOFF_FREE"] = str(flag)
    base = mem()
    cache = FP.wrap_cache(lm.make_cache())
    outs, t_chunk = [], []
    for s in range(0, len(ids), chunk):
        t0 = time.perf_counter()
        lg = lm(mx.array([ids[s:s + chunk]]), cache=cache).logits
        mx.eval(lg, *cache[0].state)
        if KL.enabled():                                                      # what VlmBackend.prefill_chunk does after the evaluation
            KL.free_handoff(cache)
        t_chunk.append((time.perf_counter() - t0) * 1e3)
        outs.append(lg[:, -1])
        guard()
    n = len(ids)
    kept = mem() - base
    summ = KL.summary(cache, n)
    bill = sum(a.nbytes for a in arrays_of(cache))
    t_step = []
    for t in more[:steps]:                                                    # direct decode, one session
        t0 = time.perf_counter()
        lg = lm(mx.array([[t]]), cache=cache).logits
        mx.eval(lg)
        t_step.append((time.perf_counter() - t0) * 1e3)
        outs.append(lg[:, -1])
        guard()
    st = dict(bill=bill, kept=kept, summary=summ, chunk_ms=t_chunk, step_ms=t_step)
    del cache
    return outs, st


def spread(v):
    v = sorted(v)
    return "median %.2f ms (min %.2f, max %.2f, n=%d)" % (statistics.median(v), v[0], v[-1], len(v))


def part_c(lm, ids, turns, per, seg):
    from kyojin_mlx import fp4pool as FP, rows as R
    from kyojin_mlx.serve.backends import Backend, arrays_of
    from kyojin_mlx.serve.engine import Engine, PrefixCache
    FP.SEG = seg

    class B(Backend):
        def fork(self, cache):
            return R.fork(cache)

        def snapshot(self, cache):
            return R.snapshot(cache)

    cache = FP.wrap_cache(lm.make_cache())
    pc = PrefixCache(B(), per, 1 << 40)
    sizes, s = [], 0
    for k in range(turns):
        while s < per * (k + 1):
            mx.eval(lm(mx.array([ids[s:s + 64]]), cache=cache).logits, *cache[0].state)
            s += 64
        pc.store((s, bytes([k])), cache)
        sizes.append(pc.bytes)
        guard()
    each = [sum(a.nbytes for a in arrays_of(e[0])) for e in pc.entries.values()]
    sess = types.SimpleNamespace(cache=cache, kv_bytes=Backend().nbytes(cache), share_seed=None)
    eng = types.SimpleNamespace(prefix=pc, bc_bytes=0, bc=None, prefilling=[], ready=[sess], members=[], waiting=[])
    res = {}
    for f in ("0", "1"):
        os.environ["KYOJIN_KV_SEG_ONCE"] = f
        res[f] = Engine.kv_bytes(eng)
    FP.SEG = 0
    print("\nC. nested prefix store (kvseg %d rows, one entry per %d tokens)" % (seg, per))
    print("   store bytes after each entry:", sizes)
    print("   steps (marginal bytes per entry):", [b - a for a, b in zip(sizes, sizes[1:])])
    print("   sum of entry sizes %d, store %d (%.2fx of the sum)" % (sum(each), pc.bytes, pc.bytes / sum(each)))
    print("   Engine.kv_bytes: kv_seg_once 0 -> %d, 1 -> %d (overlap removed %d)" % (res["0"], res["1"], res["0"] - res["1"]))
    return res["0"] >= res["1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--turns", type=int, default=6)
    a = ap.parse_args()
    if not 256 <= a.tokens <= 8192 or not 1 <= a.turns <= 8 or not 1 <= a.steps <= 64:
        sys.exit("--tokens in [256, 8192], --turns in [1, 8], --steps in [1, 64]")
    mx.set_memory_limit(LIMIT)
    import mini_ds41 as M
    from kyojin_mlx import fp4pool as FP
    print("device:", mx.default_device())
    lm, _ = M.make(sliding_window=16)
    ids, more = M.tokens(a.tokens), M.tokens(2 * a.steps, seed=7)
    res = {}
    for flag in (0, 1, 0, 1):                                                 # 0 1 0 1: a drift between the two reads of one flag shows
        res.setdefault(flag, []).append(arm(lm, ids, more, a.chunk, a.steps, flag))
    bad = 0
    for i in range(len(res[0])):
        o0, o1 = res[0][i][0], res[1][i][0]
        for j, (x, y) in enumerate(zip(o0, o1)):
            if not bool(mx.array_equal(x, y).item()):
                bad += 1
                print("LOGITS DIFFER at row %d (pair %d): max abs %.3g" % (j, i, float(mx.abs(x - y).max().item())))
    print("\nA/B logits rows compared: %d pairs x %d rows, differing: %d" % (len(res[0]), len(res[0][0][0]), bad))
    for flag in (0, 1):
        st = res[flag][-1][1]
        s = st["summary"]
        print("\nflag %d: bill %d B (%.1f B/token), MLX active kept by the cache %d B (%.1f B/token)" % (flag, st["bill"], st["bill"] / a.tokens, st["kept"], st["kept"] / a.tokens))
        print("   per token by role:", s["per_token"])
        print("   prefill chunk", spread([x for r in res[flag] for x in r[1]["chunk_ms"][1:]]))
        print("   decode step  ", spread([x for r in res[flag] for x in r[1]["step_ms"]]))
    b0, b1 = res[0][-1][1]["bill"], res[1][-1][1]["bill"]
    k0, k1 = res[0][-1][1]["kept"], res[1][-1][1]["kept"]
    print("\nbill %d -> %d B (%.1f%%); kept %d -> %d B (%.1f%%). The bill falls by the handoff; 'kept' falls only by what the handoff alone pinned" % (b0, b1, 100 * (b1 - b0) / b0, k0, k1, 100 * (k1 - k0) / max(k0, 1)))
    ok = part_c(lm, M.tokens(a.turns * 192, seed=3), a.turns, 192, 64)
    if bad:
        print("\nEXIT 1: logits differ between flag 0 and flag 1")
        sys.exit(1)
    if not ok:
        print("\nEXIT 1: kv_seg_once raised the bill")
        sys.exit(1)
    print("\nOK: logits equal flag 0 against 1")


if __name__ == "__main__":
    main()
