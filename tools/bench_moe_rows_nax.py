#!/usr/bin/env python3
"""D4 routed experts: one DS4.1 MoE layer at 1..32 rows (decode / MTP verify / N-user step): the GEMV gather path (one decode per routed pair),
the union GEMV (one decode per distinct expert, <= 8 rows per expert), and the fused trellis-decode + NAX path (MoePrefill with every expert fused).
Real layer-3 experts (a pool of --esub; virtual experts alias the pool through the offset table, so a layer with 150 distinct experts fits in memory).
Routing is synthetic (no router trace exists on the Air): independent top-k over --experts with Zipf --skew, optionally --overlap = fraction of a
row's experts copied from the previous row (a verify chain). Chain of CHAIN calls with different routings in one mx.eval, min of REPS.
   PYTHONPATH=$PWD:$PWD/bench:$PWD/tools python tools/bench_moe_rows_nax.py [--rows 1,2,4,8,16,32] [--overlap 0,0.5] [--skew 0,1]"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx
from safetensors import safe_open
from kyojin_mlx import moe as M, moe_prefill as MP, metal_nax as NX
from nax_common import mini_stack

CAP_GB = 2.8


def virtual(st: M.Stack, V: int) -> M.Stack:
    """V experts that alias the pool of st (expert e -> e % pool): same words, V-long offset / k / suh / svh tables (kernels read off[e], kk[e] only)."""
    pool = st.experts
    phys = np.arange(V) % pool
    off = np.array(st.offset)[:pool][phys]
    off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
    return M.Stack(st.words, mx.array(off), mx.array(np.array(st.k)[phys]), mx.array(np.array(st.suh)[phys]), mx.array(np.array(st.svh)[phys]), st.kt, st.nt, st.cb)


def route(R, topk, E, skew, overlap, rng):
    p = 1.0 / np.arange(1, E + 1) ** skew if skew else np.ones(E)
    p = p[rng.permutation(E)]; p /= p.sum()
    ids = np.zeros((R, topk), np.int64)
    ids[0] = rng.choice(E, topk, replace=False, p=p)
    for r in range(1, R):
        keep = rng.random(topk) < overlap
        row = list(ids[r - 1][keep])
        while len(row) < topk:
            e = int(rng.choice(E, p=p))
            if e not in row:
                row.append(e)
        ids[r] = row
    return ids


def compact(ids):
    """Distinct experts -> 0..d-1 (random order); returns (mapped ids, d)."""
    u, inv = np.unique(ids, return_inverse=True)
    return inv.reshape(ids.shape).astype(np.int32), len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/kyojin-mlx/models/ds_l3"))
    ap.add_argument("--esub", type=int, default=48); ap.add_argument("--experts", type=int, default=384); ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--rows", default="1,2,4,8,16,32"); ap.add_argument("--overlap", default="0"); ap.add_argument("--skew", default="0")
    ap.add_argument("--reps", type=int, default=15); ap.add_argument("--chain", type=int, default=6)
    ap.add_argument("--arms", default="gather,union,nax"); ap.add_argument("--json", default="")
    a = ap.parse_args()
    mx.set_cache_limit(256 << 20)
    rng = np.random.default_rng(1)
    kv = None
    f = sorted(__import__("glob").glob(os.path.join(a.dir, "stacked.*.safetensors")))[0]
    with safe_open(f, "np") as s:
        k = [x for x in s.keys() if x.endswith(".experts.w1.k")][0]
        kv = s.get_tensor(k)
    k2 = [i for i in range(len(kv)) if kv[i] == 2][:a.esub // 2]; k3 = [i for i in range(len(kv)) if kv[i] == 3][:a.esub - len(k2)]
    exps = sorted(k2 + k3)
    pool = [mini_stack(a.dir, p, exps) for p in ("w1", "w3", "w2")]
    mx.eval(*[s.words for s in pool])
    print(f"pool {len(exps)} real experts, K {pool[0].ks}, bytes {sum(s.words.size for s in pool)*4/1e6:.0f} MB", flush=True)
    rows_l = [int(v) for v in a.rows.split(",")]
    out = []
    Vmax = 200
    g, u, d = (virtual(s, Vmax) for s in pool)
    dec = M.MoeGather(g, u, d, a.topk, wpre=True, fq=False, pin_rows=1)
    dec_u = M.MoeGather(g, u, d, a.topk, wpre=True, fq=False, pin_rows=1, union=True)
    pf = MP.MoePrefill(dec, fused=True, fused_max=10 ** 9)
    xs_all = None
    for skew in [float(v) for v in a.skew.split(",")]:
        for ov in [float(v) for v in a.overlap.split(",")]:
            print(f"== skew {skew} overlap {ov}", flush=True)
            for R in rows_l:
                batches = []
                for c in range(a.chain):
                    ids_full = route(R, a.topk, a.experts, skew, ov, rng)
                    ids, dist = compact(ids_full)
                    if dist > Vmax:
                        raise SystemExit(f"{dist} distinct experts > {Vmax}")
                    perm = rng.permutation(Vmax)[:dist]                       # spread over the virtual table (physical pool is cycled)
                    ids = perm[ids].astype(np.int32)
                    wts = rng.uniform(0.1, 1.0, (R, a.topk)).astype(np.float32); wts = wts / wts.sum(-1, keepdims=True) * 1.5
                    x = rng.standard_normal((R, g.kin)).astype(np.float16) * 2
                    cnt = np.bincount(ids.reshape(-1))[np.bincount(ids.reshape(-1)) > 0]
                    batches.append((mx.array(x), mx.array(ids), mx.array(wts), dist, cnt))
                mx.eval(*[v for b in batches for v in b[:3]])
                dist_m = np.mean([b[3] for b in batches]); maxc = np.mean([b[4].max() for b in batches])
                step = max(1, min(a.topk and 64 // a.topk, R))

                def gather(b, moe):
                    x, ids, wts = b[:3]
                    if R <= step:
                        return moe(x, ids, wts)
                    return mx.concatenate([moe(x[i:i + step], ids[i:i + step], wts[i:i + step]) for i in range(0, R, step)], axis=0)
                arms = {"gather": lambda b: gather(b, dec), "union": lambda b: gather(b, dec_u), "nax": lambda b: pf(b[0], b[1], b[2])}
                res = {"skew": skew, "overlap": ov, "rows": R, "pairs": R * a.topk, "distinct": float(dist_m), "max_rows_per_expert": float(maxc)}
                ys = {}
                names = a.arms.split(",")
                best = {n: 1e9 for n in names}
                for n in names:
                    mx.eval(*[arms[n](b) for b in batches[:2]])
                for _ in range(a.reps):                                     # interleaved arms: a busy Air hits all of them alike
                    for n in names:
                        t0 = time.perf_counter()
                        mx.eval(*[arms[n](b) for b in batches])
                        best[n] = min(best[n], (time.perf_counter() - t0) / a.chain)
                for n in names:
                    res[n + "_ms"] = best[n] * 1e3
                    ys[n] = np.array(arms[n](batches[0]))
                # host plan cost of the nax arm (np.array(ids), sort, uploads): not needed on a device-side plan
                t0 = time.perf_counter()
                for b in batches:
                    pf._plan(np.array(b[1]))
                res["plan_ms"] = (time.perf_counter() - t0) / a.chain * 1e3
                ref = ys.get("gather")
                rel = {n: float(np.linalg.norm(y - ref) / np.linalg.norm(ref)) for n, y in ys.items() if ref is not None and n != "gather"}
                res["rel"] = rel
                msg = " ".join(f"{n} {res[n + '_ms']:7.3f}" for n in a.arms.split(","))
                print(f"  R={R:2d} pairs {R*a.topk:3d} distinct {dist_m:5.1f} max rows/expert {maxc:3.1f} | {msg} ms | nax host plan {res['plan_ms']:.3f} | rel {rel}", flush=True)
                out.append(res)
                pk = mx.get_peak_memory() / 1e9
                if pk > CAP_GB:
                    print(f"peak {pk:.2f} GB over cap, abort"); sys.exit(3)
    print(f"peak {mx.get_peak_memory()/1e9:.2f} GB")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


main()
