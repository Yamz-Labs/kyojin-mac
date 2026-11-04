"""Mac probe for KYOJIN_PF_IDX_BOUND (pf_bound): does the exact superblock bound prune anything on index keys that look like the model's, and is it exact on the GPU?
Bounded: a few hundred query rows, Tc up to 262144, peak memory printed and asserted < 3.5 GiB, minutes at most. No model, no lane.

    python tools/pf_bound_probe.py                                   # synthetic key families, Tc 65536 and 262144
    python tools/pf_bound_probe.py --kv-dir ~/kv_disk_dir --layer 2  # + the REAL index keys of layer 2 from an existing kv-disk checkpoint chain (read by mmap)
    python tools/pf_bound_probe.py --check-only

What it prints, per key family x Tc x superblock size SB:
  * clustering, query independent: median over superblocks of (radius around the mean) / (median key norm), and box width / key L1 norm. Near 1 or above: the bound
    cannot prune, whatever the queries; well below 1: keys inside a superblock are alike.
  * survivor fraction: superblocks whose bound reaches the exact threshold (mean over rows), the fallback blocks, and PASS / FAIL for exactness against the stock fscore chain
    (same indices, bit for bit).
  * ms per query block, bound path against the stock chain, and the 8192-row-chunk figure (x rows scale). Timing needs Metal; without it the logic runs on the numpy reference
    at small sizes and the timing is skipped.
Queries: Gaussian, or with --kv-dir the real keys themselves as query rows (a proxy: kv-disk checkpoints hold no index queries; a real query dump would replace it).
The verdict that matters for lane time: the survivor fraction on REAL keys. If it is above ~70 % the lever never pays and stays at 0."""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlx.core as mx

from kyojin_mlx import ds_pf_idx as PI
from kyojin_mlx import idx_topk as IT
from kyojin_mlx import pf_bound as PB
from kyojin_mlx import pf_flat as PF
from kyojin_mlx import pfroof2 as PR

NH, D, TOPK = 32, 128, 512
GIB = 2 ** 30
FAILS = []
GPU = mx.metal.is_available() and mx.default_device() == mx.gpu and getattr(PI.NX, "nax_available", lambda: False)()


def gate(name, ok, extra=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


# ----------------------------------------------------------------------------------------------------------------- key families
def smooth(n, width, rng):
    x = rng.standard_normal((n + width, D)).astype(np.float32)
    c = np.cumsum(x, axis=0)
    return (c[width:] - c[:-width]) / np.sqrt(width)


def family(name, Tc, sb, rng):
    if name == "gauss":                                   # isotropic: the worst case, nothing to prune
        return rng.standard_normal((Tc, D)).astype(np.float32)
    if name == "runs":                                    # runs of sb adjacent positions around one of 4096 centres (best case for a box bound)
        cen = rng.standard_normal((4096, D)).astype(np.float32)
        return np.repeat(cen[rng.integers(0, 4096, size=-(-Tc // sb))], sb, axis=0)[:Tc] + 0.05 * rng.standard_normal((Tc, D)).astype(np.float32)
    if name == "smooth":                                  # keys drift slowly with the position (moving average of 4 sb)
        return smooth(Tc, 4 * sb, rng)
    if name == "lowrank":                                 # rank 8 + noise: anisotropic keys
        return (rng.standard_normal((Tc, 8)) @ rng.standard_normal((8, D)) / np.sqrt(8) + 0.1 * rng.standard_normal((Tc, D))).astype(np.float32)
    raise ValueError(name)


def to_half(x):
    return np.clip(x, -60000, 60000).astype(np.float16)


# ----------------------------------------------------------------------------------------------------------------- real keys from a kv-disk chain
def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def real_keys(kv_dir, layer, max_rows):
    """Index keys [rows, 128] fp16 of layer `layer` from the longest checkpoint chain under kv_dir (safetensors files, tensor g.keys.<layer>, one delta per file), by mmap."""
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.expanduser(kv_dir)) for f in fs if f.endswith(".kv.safetensors")]
    meta = {}
    for p in files:
        h, off = _header(p)
        head = json.loads(h["__metadata__"]["kyojin_kv"])
        meta[head["digest"]] = (p, head, h, off)
    if not meta:
        raise SystemExit(f"no checkpoint under {kv_dir}")
    tip = max(meta.values(), key=lambda m: m[1]["b"])
    chain = []
    cur = tip
    while cur is not None:
        chain.append(cur)
        cur = meta.get(cur[1].get("parent"))
    parts, got = [], 0
    for p, head, h, off in reversed(chain):
        t = h.get(f"g.keys.{layer}")
        if t is None:
            continue
        a, b = t["data_offsets"]
        assert t["dtype"] == "F16" and t["shape"][-1] == D, t
        arr = np.memmap(p, dtype=np.float16, mode="r", offset=off + a, shape=(t["shape"][-2], D))
        parts.append(np.asarray(arr[:max(0, max_rows - got)]))
        got += parts[-1].shape[0]
        if got >= max_rows:
            break
    if not parts:
        raise SystemExit(f"layer {layer} holds no index keys in that chain (not an index-key owner?)")
    k = np.concatenate(parts, axis=0)
    print(f"real keys: layer {layer}, {k.shape[0]} rows from {len(chain)} checkpoint(s), tip at {tip[1]['b']} tokens")
    return k.astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------- measures
def stats(k, sb):
    n = (k.shape[0] // sb) * sb
    blk = k[:n].reshape(-1, sb, D)
    cm = blk.mean(1)
    r = np.sqrt(((blk - cm[:, None]) ** 2).sum(-1).max(1))
    norm = np.median(np.linalg.norm(k, axis=1))
    box = (blk.max(1) - blk.min(1)).sum(-1) / np.maximum(np.abs(blk).sum(-1).mean(1), 1e-9)
    return float(np.median(r) / norm), float(np.median(box))


def med(fn, reps):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def run_case(tag, keys_np, rows, sb, rng, reps, real_q):
    Tc = keys_np.shape[0]
    k = mx.array(to_half(keys_np)[None])
    if real_q:
        qn = keys_np[rng.integers(0, Tc, size=(rows, NH))]
    else:
        qn = rng.standard_normal((rows, NH, D)).astype(np.float32)
    q = mx.array(to_half(qn)[None])
    w = mx.array((rng.standard_normal((1, rows, NH)) * 0.05).astype(np.float16))
    ratio, start = 1, Tc - rows
    PB.SB, PB.MIN_KEYS, PB.ENABLED = sb, 0, True
    PB._state.clear()
    for h in PB.HITS:
        PB.HITS[h] = 0
    rs, rb = stats(keys_np, sb)
    if GPU:
        stock = lambda t0, n: PI.topk_block(PI.scores_range(q, k, w, start, ratio, t0, n), TOPK)
    else:
        stock = lambda t0, n: IT.topk(mx.array(PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(k[0]), np.array(w[0, t0:t0 + n]), start, ratio, t0)), TOPK)
    PF.GSCORE = GPU                                       # the flag set it needs on the GPU (pf_idx_gscore=1): the same kernel the lane would run
    # direct soundness gate: the bound is never below the fscore (kernel) score of any covered key, on the finite entries
    summ = PB.summaries(mx.array(to_half(keys_np)[None]), sb)
    ub = PB.bounds(q, w, summ, start, ratio, 0, rows)
    if GPU:
        sc = PI.scores_range(q, k, w, start, ratio, 0, rows)
    else:
        sc = mx.array(PR.ref_scores(np.array(q[0]), np.array(k[0]), np.array(w[0]), start, ratio, 0))
    nsb = summ["nsb"]
    sc = mx.pad(sc, [(0, 0), (0, nsb * sb - Tc)], constant_values=-float("inf")).reshape(rows, nsb, sb).max(-1)
    fin = mx.isfinite(sc)
    slack = mx.where(fin, ub - sc, mx.array(float("inf")))
    mn = float(mx.min(slack).item())
    gate(f"{tag} Tc={Tc} SB={sb}: bound >= fscore block max on every finite entry", mn >= 0.0, f"(min slack {mn:.3g})")
    del sc, ub, slack, fin
    ref = IT.scan(lambda t0, n: (t0, n), rows, IT.qblock(rows), lambda t0, tn: stock(tn[0], tn[1]))
    PB._state.clear()
    got = PB.try_plain(q, k, w, start, ratio, TOPK, rows, key="probe", stock=stock)
    frac = PB.survivor_fraction()
    print(f"  {tag:8s} Tc={Tc:7d} SB={sb:4d}  radius/norm {rs:5.2f}  box/L1 {rb:5.2f}  survivors {100 * frac:5.1f} %  fallback blocks {PB.HITS['fallback_blocks']}/{PB.HITS['blocks']}", flush=True)
    gate(f"{tag} Tc={Tc} SB={sb}: indices == stock chain", got is not None and bool(mx.array_equal(got, ref).item()))
    if GPU and got is not None:
        PB._state.clear()
        t_new = med(lambda: PB.try_plain(q, k, w, start, ratio, TOPK, rows, key="probe", stock=stock), reps)
        PB._state.clear()
        t_old = med(lambda: IT.scan(lambda t0, n: (t0, n), rows, IT.qblock(rows), lambda t0, tn: stock(tn[0], tn[1])), reps)
        print(f"           ms for {rows} rows: stock {t_old:8.1f}  bound {t_new:8.1f}  ratio {t_new / t_old:5.2f}   [8192 rows: {t_old * 8192 / rows:8.0f} vs {t_new * 8192 / rows:8.0f}]", flush=True)
    peak = mx.get_peak_memory() / GIB
    gate(f"peak {peak:.2f} GiB < 3.5", peak < 3.5)
    mx.reset_peak_memory()
    del k, q, w, ref, got
    mx.clear_cache()
    return frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc", default="65536,262144")
    ap.add_argument("--sb", default="32,64,128")
    ap.add_argument("--rows", type=int, default=0, help="query rows (default 256 on Metal, 24 without)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--kv-dir")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    rows = a.rows or (256 if GPU else 24)
    tcs = [int(x) for x in a.tc.split(",")]
    if not GPU:
        print("no Metal GPU / NAX: numpy reference scoring, small sizes, no timing")
        tcs = [min(t, 8192) for t in tcs]
    rng = np.random.default_rng(0)
    t_all = time.perf_counter()
    fams = ["gauss", "runs", "smooth", "lowrank"]
    fr = {}
    for Tc in tcs:
        for sb in [int(x) for x in a.sb.split(",")]:
            for f in fams:
                fr[(f, Tc, sb)] = run_case(f, family(f, Tc, sb, rng), rows, sb, rng, 0 if a.check_only else a.reps, False)
    if a.kv_dir:
        real = real_keys(a.kv_dir, a.layer, max(tcs))
        for sb in [int(x) for x in a.sb.split(",")]:
            fr[("REAL", real.shape[0], sb)] = run_case("REAL", real, min(rows, 256), sb, rng, 0 if a.check_only else a.reps, True)
    print("\nsurvivor fraction (percent) by family: " + "; ".join(f"{k[0]}/{k[1]}/{k[2]}: {100 * v:.1f}" for k, v in fr.items()))
    real_fr = [v for k, v in fr.items() if k[0] == "REAL"]
    if real_fr:
        print(f"REAL keys: best survivor fraction {100 * min(real_fr):.1f} % -> " + ("worth a lane A/B" if min(real_fr) < 0.5 else "the bound prunes too little: keep KYOJIN_PF_IDX_BOUND=0"))
    else:
        print("no real keys given (--kv-dir): the synthetic families only bracket the answer (gauss = worst case, runs = best case)")
    print(f"total {time.perf_counter() - t_all:.1f} s; {'ALL GATES PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
