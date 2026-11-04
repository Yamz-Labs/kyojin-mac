#!/usr/bin/env python3
"""Per-token GPU tables from a /v1/stats dump taken with KYOJIN_GPU_SPAN=1 (and, for the kernel table, KYOJIN_GPU_KERN=1 MLX_MAX_OPS_PER_BUFFER=1).

  python tools/slope_tables.py STATS.json [--floor-us 4.7] [--peak 1030]

Reads the lagged `gpuL_*` numbers (a step's stamps have all landed one tick later; the unlagged gpu_* miss the step in flight).
 span / busy / idle  per token: first buffer start to last buffer end, union of buffer spans, holes between them (never a sum of buffers).
 kernel rows         with one dispatch per buffer every stamp is that kernel's own start-to-end plus a per-buffer floor (--floor-us, read from a
                     chained trivial kernel in the same mode); `net` = raw - floor. `bytes` = weight bytes of the call from the kernel name shape
                     (rows x cols x bytes/weight x experts); GB/s = bytes / net; gap = net - bytes / peak."""
import argparse, json, re

BPW = {"q8_0": 34 / 32, "q4_k": 144 / 256, "q5_k": 176 / 256, "q6_k": 210 / 256}


def nbytes(name):
    m = re.search(r"fg_(gemv8|gemvk|pairp|pairk)_(q8_0|q4_k|q5_k|q6_k)_n2(u|gu)?__(\d+)_(\d+)_(\d+)(?:_(\d+))?(?:_(\d+))?", name)
    if not m:
        return None
    kind, t, var = m.group(1), m.group(2), m.group(3)
    a, b = int(m.group(5)), int(m.group(6))
    if kind.startswith("pair"):          # gate + up of the routed experts: shape rows_kin_nout_2_topk
        return 2 * int(m.group(8)) * a * b * BPW[t] if m.group(8) else None
    if var == "gu":                      # routed down: topk experts x nout x kin
        return 8 * a * b * BPW[t]
    if var == "u":
        return None
    return a * b * BPW[t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--floor-us", type=float, default=4.7)
    ap.add_argument("--peak", type=float, default=1030.0)
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()
    s = json.load(open(a.stats))
    s = s.get("stats", s)
    n = s["gpuL_n_r1"]
    g = lambda k: s.get(f"gpuL_{k}_r1", 0.0) / n
    print(f"ticks {n}  per token: span {g('span'):.3f} ms  busy {g('busy'):.3f}  idle {g('idle'):.3f}  buffers {g('buf'):.1f}")
    rows = []
    for k, v in s.items():
        m = re.match(r"gpuL_k_(.+)_r1$", k)
        if m:
            c = s.get(f"gpuL_kc_{m.group(1)}_r1", 0) / n
            if c >= 0.5:
                rows.append((v / n, c, m.group(1)))
    rows.sort(reverse=True)
    print(f"{'ms/tok':>7} {'n/tok':>6} {'raw us':>7} {'net us':>7} {'MB':>6} {'GB/s':>6} {'gap us':>7}  kernel")
    for ms, c, nm in rows[:a.top]:
        raw = ms / c * 1e3
        net = max(raw - a.floor_us, 0.1)
        b = nbytes(nm)
        mb = f"{b / 1e6:6.1f}" if b else "     -"
        bw = f"{b / net / 1e3:6.0f}" if b else "     -"
        gap = f"{net - b / a.peak / 1e3:7.1f}" if b else "      -"
        print(f"{ms:7.3f} {c:6.1f} {raw:7.1f} {net:7.1f} {mb} {bw} {gap}  {nm[:70]}")


main()
