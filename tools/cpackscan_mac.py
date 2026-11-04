"""Mac check for KYOJIN_PF_IDX_CPACK (cpackscan): flag 0 against flag 1, bounded under 3.5 GiB, no model weights, no lane. Exit 1 if the logits (or the selected sets) differ,
exit 2 if the packed path never ran (a number from a path that did not engage is not a number), exit 0 otherwise. Minutes at most.

    python tools/cpackscan_mac.py                              # both parts, Metal when available
    python tools/cpackscan_mac.py --kv-dir ~/kv_disk_dir --layer 2   # + the REAL index keys of that layer (kv-disk checkpoint chain, mmap) in part A
    python tools/cpackscan_mac.py --check-only                 # CPU logic only (numpy scoring, small sizes), what the sandbox could run

Part A  selection, isolated (one plain layer's worth of work): index keys on the fp4 grid in a KeySlots layer, 128-dim fp16 queries, 32 heads. Arms: flag 0 = the shipped chain on
        the dense mirror (pf_flat.plain_topk when it applies, else the fscore chain), flag 1 = transient decode of the codes + cpackscan.try_plain. Compared: the selected index sets,
        bit for bit. Printed: survivor fraction by group size (128 = the pf_idx_bound bound, 32, 16), fallback blocks, ms per arm (Metal only), retained key bytes (dense mirror vs
        packed codes), peak memory. Key families: runs (clustered, best case), gauss (isotropic, worst case, ledger gbound2), and with --kv-dir the real keys (queries are the keys
        themselves, a proxy: kv-disk checkpoints hold no index queries).
Part B  logits, the wiring: the mini DS4.1 model (tests/mini_ds41.py with 32 x 128 index heads) prefilled in chunks with the hook of ds_pf_idx._impl_indexer, flag 0 then flag 1:
        the last-row logits must be equal bit for bit, the stored key codes the same, the path counters above 0, and the plain layer's key mirror absent after the flag-1 prefill.
        Printed: wall time per arm (Metal), key mirror bytes retained, peak memory. Random weights prune nothing, so the bound path is forced through (maxpct 100): this part proves
        wiring and exactness, it says nothing about the survivor fraction of the real model. Without Metal the model stays fp32 and the Metal rotary kernel is replaced by an fp4
        round trip (the same in both arms)."""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx

from kyojin_mlx import cpackscan as CP
from kyojin_mlx import depthflat as DF
from kyojin_mlx import ds_pf_idx as PI
from kyojin_mlx import fp4pack
from kyojin_mlx import idx_topk as IT
from kyojin_mlx import pf_bound as PB
from kyojin_mlx import pf_flat as PF
from kyojin_mlx import pfroof2 as PR
from kyojin_mlx.fp4pool import KeySlots

NH, D = 32, 128
GIB = 2 ** 30
CAP = 3.5 * GIB
GPU = mx.metal.is_available() and mx.default_device() == mx.gpu
FAILS, NOT_ENGAGED, MIRROR_FAILS = [], [], []
RESULT = {"gpu": GPU, "part_a": [], "part_b": {}}


def gate(name, ok, extra=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def peak():
    return mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()


def check_cap(tag):
    p = peak()
    print(f"    peak memory {p / GIB:.2f} GiB ({tag})", flush=True)
    if p > CAP:
        print(f"ABORT: peak {p / GIB:.2f} GiB is over the {CAP / GIB:.1f} GiB bound", flush=True)
        sys.exit(3)
    return p


def reset_peak():
    (mx.reset_peak_memory if hasattr(mx, "reset_peak_memory") else mx.metal.reset_peak_memory)()


def fp4(x):
    c, s = fp4pack.pack(mx.array(np.asarray(x, dtype=np.float32)), 32, False)
    return fp4pack.unpack(c, s, 32, False, mx.float16)


def med(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = fn()
        mx.eval(r)
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


# ----------------------------------------------------------------------------------------------------------------- part A
def family(name, Tc, rng, sb):
    if name == "gauss":
        return rng.standard_normal((Tc, D)).astype(np.float32)
    cen = rng.standard_normal((4096, D)).astype(np.float32)                       # runs of sb adjacent positions around one of 4096 centres
    return np.repeat(cen[rng.integers(0, 4096, size=-(-Tc // sb))], sb, axis=0)[:Tc] + 0.004 * rng.standard_normal((Tc, D)).astype(np.float32)


def part_a_case(tag, kn, rows, topk, reps, real_q=None):
    Tc = kn.shape[0]
    rng = np.random.default_rng(1)
    keys = fp4(kn)[None]
    q = (fp4(kn[rng.integers(0, Tc, size=rows * NH)].reshape(rows, NH, D) * 0.5) if real_q else fp4(rng.standard_normal((rows, NH, D)) * 0.5))[None]
    w = mx.array((rng.standard_normal((1, rows, NH)) * 0.05).astype(np.float16))
    start = Tc - rows
    slots = KeySlots([None])
    slots.defer_dense = True
    slots.append_rows(0, keys)
    slots.defer_dense = False
    e = slots.raw(0)
    print(f"  [{tag}] Tc {Tc}, rows {rows}, topk {topk}", flush=True)

    def stock_arm():                                                                # flag 0: the shipped chain on the dense keys
        if PF.METAL_OPT and PF.bmax_ok(q, keys, w, Tc, topk, start, 1, False):
            return PF.plain_topk(q, keys, w, start, 1, topk, rows)
        return IT.scan(lambda t0, n: PI.scores_range(q, keys, w, start, 1, t0, n) if PF.METAL_OPT else mx.array(
            PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(keys[0]), np.array(w[0, t0:t0 + n]), start, 1, t0)), rows, IT.qblock(rows), lambda t0, s: PI.topk_block(s, topk))

    ref = np.array(stock_arm())
    rec = {"tag": tag, "Tc": Tc, "rows": rows, "topk": topk, "packed_bytes_per_key": 68, "dense_bytes_per_key": 256}
    for g in (128, 32, 16):
        CP.GROUP = g
        CP._state.clear()
        for h in CP.HITS:
            CP.HITS[h] = 0
        dec = CP.decode(slots, e, Tc)
        gate(f"decode == dense keys (G{g})", bool(mx.array_equal(dec, keys)))
        got = CP.try_plain(q, dec, w, start, 1, topk, rows, key=f"a{g}")
        served = got is not None
        if served:
            gate(f"selected set == stock (G{g})", np.array_equal(np.array(got), ref))
        print(f"      G{g:<3} {'served' if served else 'MISS (first block over the cap: normal chain runs)'}  survivor fraction {CP.survivor_fraction():.3f}  "
              f"fallback blocks {CP.HITS['fallback_blocks']}", flush=True)
        rec[f"G{g}"] = {"served": served, "survivor_fraction": CP.survivor_fraction(), "fallback_blocks": CP.HITS["fallback_blocks"]}
        if served and g == 32:
            RESULT_ENGAGED.append(tag)
    CP.GROUP = 32
    if GPU:
        CP._state.clear()
        dec = CP.decode(slots, e, Tc)
        def flag1():
            CP._state.clear()                                                       # every rep pays the bound: a skip window would time the stock path only
            d = CP.decode(slots, e, Tc)
            r = CP.try_plain(q, d, w, start, 1, topk, rows, key="t")
            return r if r is not None else stock_arm()

        t0 = med(stock_arm, reps)
        t1 = med(flag1, reps)
        print(f"      ms  flag 0 {t0:.1f}   flag 1 (decode + bound + rescoring, or the normal chain on a miss) {t1:.1f}   x{t1 / t0:.2f}   [x rows 8192/{rows} = per chunk]", flush=True)
        rec.update({"ms_flag0": t0, "ms_flag1": t1})
    else:
        print("      ms: no Metal, not timed", flush=True)
    rec["retained_key_bytes"] = {"dense_mirror": Tc * 256, "packed_codes": Tc * 68}
    RESULT["part_a"].append(rec)
    check_cap(tag)


RESULT_ENGAGED: list = []


def part_a(a):
    rng = np.random.default_rng(0)
    CP.ENABLED, CP.MIN_KEYS = True, 0
    PB.MAXPCT = a.maxpct
    PF.BMAX_MIN = 0
    rows = a.rows or (256 if GPU else 24)
    print("part A: selection, flag 0 (shipped chain) against flag 1 (packed bound)", flush=True)
    for Tc in [int(t) for t in a.tc.split(",")]:
        for fam in ("runs", "gauss"):
            reset_peak()
            part_a_case(f"{fam}", family(fam, Tc, rng, PB.sb_size()), rows, a.topk, a.reps)
    if a.kv_dir:
        import pf_bound_probe as PP
        real = PP.real_keys(a.kv_dir, a.layer, max(int(t) for t in a.tc.split(",")))
        reset_peak()
        part_a_case(f"real layer {a.layer}", np.array(real, dtype=np.float32), rows, a.topk, a.reps, real_q=True)
    else:
        print("  no real keys given (--kv-dir): runs / gauss only bracket the answer; the survivor fraction on REAL keys decides whether this lever pays at all", flush=True)


# ----------------------------------------------------------------------------------------------------------------- part B
def part_b(a):
    print("part B: logits, mini DS4.1 model through the indexer hook, flag 0 then flag 1", flush=True)
    import mini_ds41 as M
    from mlx.utils import tree_map
    lm, cfg = M.make(seed=3, index_n_heads=32, index_head_dim=128, index_topk=16, candidate_topk_blocks=24, candidate_block_size=8, max_position_embeddings=1 << 17)
    f16 = lambda p: p.astype(mx.float16) if p.dtype == mx.float32 else p
    if GPU:
        lm.update(tree_map(f16, lm.parameters()))
    else:
        for layer in lm.layers:
            ix = getattr(layer.attn, "indexer", None)
            if ix is not None:
                ix.update(tree_map(f16, ix.parameters()))
    mx.eval(lm.parameters())
    mod = __import__(type(lm.layers[0].attn).__module__, fromlist=["Indexer"])
    orig_call, orig_rot = mod.Indexer.__call__, PI.fq4_rot
    patched = PI._impl_indexer(orig_call)
    if GPU:
        mod.Indexer.__call__ = patched
    else:
        cast = lambda t: None if t is None else t.astype(mx.float16)
        mod.Indexer.__call__ = lambda self, x, qr, latent, sp, off, cache: patched(self, cast(x), cast(qr), cast(latent), sp, off, cache)
        PI.fq4_rot = lambda q, cos, sin, rd: fp4(np.array(q.astype(mx.float32))).reshape(q.shape)
    PI.ENABLED, PF.BMAX, PF.BMAX_MIN = True, True, 0
    PB.MAXPCT = 100
    CP.MIN_KEYS = a.model_min
    n_tok, chunk = a.tokens, a.chunk
    ids = M.tokens(n_tok, seed=5)

    def prefill(flag):
        from kyojin_mlx import fp4pool
        CP.ENABLED = flag
        CP._state.clear()
        for h in CP.HITS:
            CP.HITS[h] = 0
        cache = fp4pool.wrap_cache(lm.make_cache())
        reset_peak()
        out, t = None, time.perf_counter()
        for i, s in enumerate(range(0, n_tok, chunk)):
            PI.ENABLED = GPU or i > 0                                               # CPU stub: chunk 0 has no past and runs the vendor chain
            out = lm(mx.array([ids[s:s + chunk]]), cache=cache).logits
            mx.eval(out)
        PI.ENABLED = True
        wall = (time.perf_counter() - t) * 1e3
        return cache[0], out[:, -1].astype(mx.float32), wall, peak()

    try:
        c0, l0, w0, p0 = prefill(False)
        c1, l1, w1, p1 = prefill(True)
    finally:
        mod.Indexer.__call__, PI.fq4_rot = orig_call, orig_rot
    hits = dict(CP.HITS)
    eq = bool(mx.array_equal(l0, l1))
    gate("last-row logits equal, flag 0 vs flag 1", eq, f"(max abs diff {float(mx.abs(l0 - l1).max()):.3e})")
    same_codes = all(c0.keys.raw(i) is None or (bool(mx.array_equal(c0.keys.raw(i).codes[:, :c0.keys.raw(i).n], c1.keys.raw(i).codes[:, :c1.keys.raw(i).n]))
                                                 and bool(mx.array_equal(c0.keys.raw(i).scales[:, :c0.keys.raw(i).n], c1.keys.raw(i).scales[:, :c1.keys.raw(i).n])))
                     for i in range(len(c0.keys)))
    gate("stored key codes + scales equal", same_codes)
    served = hits["chunks"] + hits["missed_chunks"]
    if served == 0 or hits["decoded_rows"] == 0:
        NOT_ENGAGED.append("part B")
    print(f"    counters {hits}", flush=True)
    m0, m1 = CP.mirror_bytes(c0), CP.mirror_bytes(c1)
    plain_mirror = [i for i in range(len(c1.keys)) if c1.keys.raw(i) is not None and getattr(c1.keys.raw(i), "mirror", None) is not None]
    print(f"    [{'PASS' if m1 < m0 else 'FAIL'}] key mirror bytes retained: flag 1 < flag 0 ({m0} -> {m1} bytes; layers still holding a mirror under flag 1: {plain_mirror}, the candidate source keeps its own)", flush=True)
    if not m1 < m0:
        MIRROR_FAILS.append("mirror bytes not lower")
    print(f"    wall {w0:.0f} ms (flag 0) vs {w1:.0f} ms (flag 1){'' if GPU else '   [CPU stub: not a speed number]'}, peak {max(p0, p1) / GIB:.2f} GiB", flush=True)
    RESULT["part_b"] = {"logits_equal": eq, "codes_equal": same_codes, "counters": hits, "mirror_bytes": [m0, m1], "wall_ms": [w0, w1], "peak_gib": max(p0, p1) / GIB,
                        "df_hits": dict(DF.HITS), "tokens": n_tok, "chunk": chunk}
    check_cap("part B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc", default="131072,262144" if GPU else "4096,8192")
    ap.add_argument("--rows", type=int, default=0, help="query rows of part A (default 256 on Metal, 24 without)")
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--maxpct", type=int, default=25, help="pf_idx_bound_maxpct for part A (the shipped value)")
    ap.add_argument("--tokens", type=int, default=16384 if GPU else 1536)
    ap.add_argument("--chunk", type=int, default=4096 if GPU else 512)
    ap.add_argument("--model-min", type=int, default=2048 if GPU else 256, help="pf_idx_cpack_min for part B (keys)")
    ap.add_argument("--kv-dir")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.check_only:
        a.tc, a.tokens, a.chunk, a.model_min, a.reps, a.topk = "8192", 1536, 512, 256, 1, 64
    print(f"GPU {GPU}  Metal fused rescoring {'yes' if GPU and PI.NX.nax_available() else 'no'}  mlx {mx.__version__}", flush=True)
    if GPU and not PI.NX.nax_available():
        print("NOTE: no NAX/matmul2d here: pf_flat.fscore_ok is False, cpackscan.applies is False, the packed path cannot engage on this Mac.", flush=True)
    part_a(a)
    if not a.skip_b:
        part_b(a)
    if not RESULT_ENGAGED:
        NOT_ENGAGED.append("part A (no case served by the bound at G32)")
    RESULT["fails"], RESULT["not_engaged"] = FAILS, NOT_ENGAGED
    print("RESULT " + json.dumps(RESULT, default=str), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(RESULT, f, indent=1, default=str)
    if FAILS:
        print("FAIL: " + ", ".join(FAILS), flush=True)
        sys.exit(1)
    if MIRROR_FAILS:
        print("MIRROR: " + ", ".join(MIRROR_FAILS) + " (logits were equal)", flush=True)
        sys.exit(4)
    if NOT_ENGAGED:
        print("NOT ENGAGED: " + "; ".join(NOT_ENGAGED) + " -- no number above says anything about the packed path", flush=True)
        sys.exit(2)
    print("all gates pass", flush=True)


if __name__ == "__main__":
    main()
