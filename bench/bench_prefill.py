"""S3 MoE prefill on a real stacked layer: time and peak memory vs the decode-path chunks of 4 rows and vs the per-expert (unfused) path,
plus row-by-row error of S3 against the decode path.
   python bench/bench_prefill.py --dir models/ds_l3 --arch ds [--rows 64,512,2048] [--group-gib 2] [--unf-rows 16] [--check 64] [--json out.json]
Routing is uniform random (worst case for S3: nearly every expert is active from 512 tokens; real routing is skewed). Activations: fp16 N(0, 2^2).
unfused = MoeUnfused (one rot + GEMV + finalize per expert and projection, host id read); its tok/s does not depend on T, so it is measured
on --unf-rows rows routed inside a pool of --pool experts (it keeps one weight copy per touched expert)."""
import argparse, json, statistics, time

import numpy as np
import mlx.core as mx

from kyojin_mlx import moe as M, moe_prefill as MP
from bench_moe import ARCH, load, routing


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y = fn()
        mx.eval(y)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), y


def peak_gib():
    return mx.get_peak_memory() / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--arch", choices=list(ARCH), required=True)
    ap.add_argument("--rows", default="64,512,2048"); ap.add_argument("--group-gib", type=float, default=2.0)
    ap.add_argument("--unf-rows", type=int, default=16); ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--check", type=int, default=64); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-dec", action="store_true"); ap.add_argument("--json")
    a = ap.parse_args()
    A = ARCH[a.arch]
    g, u, d = load(a.dir, A["proj"])
    E, topk, H = g.experts, A["topk"], g.kin
    dec = M.MoeGather(g, u, d, topk, wpre=A["wpre"], fq=A["fq"])
    pf = MP.MoePrefill(dec, group_gib=a.group_gib)
    res = {"dir": a.dir, "arch": a.arch, "experts": E, "topk": topk, "hidden": H, "group_gib": a.group_gib, "group_experts": pf.group_size(),
           "kcounts": {int(k): int((np.array(g.k) == k).sum()) for k in g.ks}, "rows": []}
    rng = np.random.default_rng(1)
    print(f"{a.dir}: E={E} topk={topk} hidden={H} inter={g.nout} K counts {res['kcounts']} | group {pf.group_size()} experts ({a.group_gib} GiB)")

    # ---- exactness: S3 (f16 and f32 plan) vs the decode path, row by row
    nchk = a.check
    ids, w = routing(nchk, E, topk, 7)
    x = mx.array((rng.standard_normal((nchk, H)) * 2).astype(np.float16))
    ref = np.concatenate([np.array(dec(x[i:i + 4], ids[i:i + 4], w[i:i + 4])) for i in range(0, nchk, 4)])
    for prec in ("f16", "f32"):
        got = np.array(MP.MoePrefill(dec, group_gib=a.group_gib, prec=prec)(x, ids, w))
        per = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
        e = dict(prec=prec, rows=nchk, rel_l2=float(np.linalg.norm(got - ref) / np.linalg.norm(ref)), row_rel_max=float(per.max()),
                 row_rel_median=float(np.median(per)), cos=float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref))))
        res.setdefault("exact", []).append(e)
        print(f"exact vs decode path, {nchk} rows, S3 {prec}: rel L2 {e['rel_l2']:.2e}, per-row median {e['row_rel_median']:.2e} max {e['row_rel_max']:.2e}")

    # ---- speed
    for T in [int(v) for v in a.rows.split(",")]:
        ids, w = routing(T, E, topk, 11 + T)
        x = mx.array((rng.standard_normal((T, H)) * 2).astype(np.float16))
        mx.eval(x, ids, w)
        row = {"T": T}
        pf(x, ids, w); mx.eval()
        mx.reset_peak_memory(); base = mx.get_active_memory() / 2 ** 30
        t, _ = timed(lambda: pf(x, ids, w), a.reps)
        row.update(s3_s=t, s3_tok_s=T / t, s3_peak_gib=peak_gib(), s3_extra_gib=peak_gib() - base, **pf.last)
        if not a.no_dec:
            step = 4
            t, _ = timed(lambda: mx.concatenate([dec(x[i:i + step], ids[i:i + step], w[i:i + step]) for i in range(0, T, step)]), 1)
            row.update(dec4_s=t, dec4_tok_s=T / t)
        res["rows"].append(row)
        print(f"T={T:5d}  S3 {row['s3_s']*1e3:8.1f} ms = {row['s3_tok_s']:8.1f} tok/s (groups {row['groups']}, active {row['active']}, peak extra {row['s3_extra_gib']:.2f} GiB)"
              + (f" | decode x4 {row['dec4_s']*1e3:8.1f} ms = {row['dec4_tok_s']:7.1f} tok/s | S3/dec x{row['dec4_s']/row['s3_s']:.1f}" if not a.no_dec else ""), flush=True)

    # ---- unfused baseline (pool-restricted routing)
    if a.unf_rows:
        unf = M.MoeUnfused(g, u, d, topk, wpre=A["wpre"], fq=A["fq"])
        n = a.unf_rows
        ids, w = routing(n, E, topk, 99, pool=a.pool)
        x = mx.array((rng.standard_normal((n, H)) * 2).astype(np.float16))
        unf(x, ids, w)
        t, _ = timed(lambda: unf(x, ids, w), 1)
        res["unfused"] = dict(rows=n, s=t, tok_s=n / t)
        print(f"unfused per-expert path: {n} rows in {t*1e3:.0f} ms = {n/t:.1f} tok/s (per-layer, size independent)")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
