"""Full-width MoE layer on real stacked packs: exactness (fused vs per-expert vs numpy fp64) and decode time (fused vs unfused).
   python bench/bench_moe.py --dir models/glm_l10 --arch glm [--rows 1,2,3,4] [--np-rows 1,2] [--chain 8] [--json out.json]
Timing models a decode: a chain of `chain` layers (router matmul -> top-k -> MoE -> rms norm) feeding each other.
  fused   = ids stay on the GPU, five kernel launches per layer, one eval for the whole chain (async)
  unfused = per-expert Mul1Gemv path, router ids read on the host every layer (one sync per layer), ~16 launches per expert
The router is restricted to a pool of experts so the unfused path keeps a bounded set of per-expert copies."""
import argparse, glob, json, os, statistics, time

import numpy as np
import mlx.core as mx

from kyojin_mlx import moe as M, ref_numpy as R

ARCH = {"ds": dict(proj=("w1", "w3", "w2"), topk=6, wpre=True, fq=True),
        "glm": dict(proj=("gate_proj", "up_proj", "down_proj"), topk=8, wpre=False, fq=False)}


def load(dirpath, proj):
    f = sorted(glob.glob(os.path.join(dirpath, "stacked.*.safetensors")))[0]
    d = mx.load(f)
    keys = [k for k in d if k.endswith(f".experts.{proj[0]}.trellis")]
    assert len(keys) == 1, keys
    pre = keys[0][:-len(f".{proj[0]}.trellis")]
    stacks = [M.Stack.from_arrays(d, f"{pre}.{p}") for p in proj]
    mx.eval(*[a for s in stacks for a in (s.words, s.offset, s.k, s.suh, s.svh)])
    return stacks


def synth_stack(kin, nout, ks, cb, seed):
    """Tiny random valid ragged stack (GPU work ~0): exposes the launch / encode / sync overhead of each path."""
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    words, off = [], [0]
    for K in ks:
        w = np.ascontiguousarray(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16)).view("<u4").reshape(-1)
        words.append(w); off.append(off[-1] + len(w))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], (len(ks), n)) * rng.uniform(0.5, 1.5, (len(ks), n))).astype(np.float16))
    return M.Stack(mx.array(np.concatenate(words)), mx.array(np.array(off, "<u4")), mx.array(np.array(ks, "u1")), f16(kin), f16(nout), kt, nt, cb)


def overhead(arch, rows, chain, reps, n):
    """Per-layer time of both paths with ~no GPU work (kin = nout = 256): launch + encode + host-sync floor."""
    A = ARCH[arch]
    E, topk, H, I = 32, A["topk"], 256, 256
    ks = [2, 3] * (E // 2)
    g, u, d = synth_stack(H, I, ks, "mul1", 1), synth_stack(H, I, ks, "mul1", 2), synth_stack(I, H, ks, "mul1", 3)
    rng = np.random.default_rng(5)
    router = mx.array((rng.standard_normal((E, H)) / np.sqrt(H)).astype(np.float32))
    x0 = mx.array(rng.standard_normal((rows, H)).astype(np.float16))
    fused = M.MoeGather(g, u, d, topk, wpre=A["wpre"], fq=False)
    unf = M.MoeUnfused(g, u, d, topk, wpre=A["wpre"], fq=False)
    rf = chain_runner(lambda x, i, w: fused(x, i, w), router, x0, rows, chain, E, topk, E, False)
    ru = chain_runner(lambda x, i, w: unf(x, i, w), router, x0, rows, chain, E, topk, E, True)
    rf(2); ru(2)
    return med(rf, reps, n) / chain * 1e3, med(ru, reps, n) / chain * 1e3


def routing(rows, E, topk, seed, pool=None):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(pool or E, topk, replace=False) for _ in range(rows)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (rows, topk)).astype(np.float32)
    return mx.array(ids), mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))


def np_ref(stacks, x, ids, w, limit, wpre, cb):
    g, u, d = stacks
    out = np.zeros((x.shape[0], d.nout), np.float64)
    ids = np.array(ids)
    w = np.array(w)
    for r in range(x.shape[0]):
        xr = np.array(x[r:r + 1]).astype(np.float64)
        for j in range(ids.shape[1]):
            e = int(ids[r, j])
            lin = lambda s, v: R.gemv(v, np.array(s.expert_trellis(e)), np.array(s.suh[e]), np.array(s.svh[e]), dtype=np.float64, cb=cb)
            gt, ut = np.minimum(lin(g, xr), limit), np.clip(lin(u, xr), -limit, limit)
            a = gt / (1 + np.exp(-gt)) * ut
            if wpre:
                a = a * w[r, j]
            y = lin(d, a)[0]
            out[r] += y if wpre else y * w[r, j]
    return out


def cmp(a, b):
    a, b = np.array(a, np.float64), np.array(b, np.float64)
    return dict(rel_max=float(np.abs(a - b).max() / np.abs(b).max()), cos=float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))))


def bitdiff(a, b):
    return int((np.array(a, np.float32).view(np.uint32) != np.array(b, np.float32).view(np.uint32)).sum())


def med(fn, reps, n):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(n)
        ts.append((time.perf_counter() - t0) / n)
    return statistics.median(ts)


def chain_runner(layer_fn, router, x0, rows, L, E, topk, pool, sync_each):
    """L layers: scores = x @ router^T (restricted to the pool) -> top-k -> layer_fn(x, ids, w) -> rms norm."""
    ones = mx.ones((x0.shape[1],), mx.float16)
    mask = mx.array(np.where(np.arange(E) < pool, 0.0, -1e4).astype(np.float32))

    def run(n):
        for _ in range(n):
            x = x0
            for _l in range(L):
                sc = (x.astype(mx.float32) @ router.T) + mask
                ids = mx.argpartition(-sc, topk - 1, axis=-1)[:, :topk].astype(mx.int32)
                w = mx.softmax(mx.take_along_axis(sc, ids, axis=-1), axis=-1) * 1.5
                y = layer_fn(x, ids, w)
                x = mx.fast.rms_norm(y.astype(mx.float16), ones, 1e-6)
                if sync_each:
                    mx.eval(x)
            mx.eval(x)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--arch", choices=list(ARCH), required=True)
    ap.add_argument("--rows", default="1,2,3,4"); ap.add_argument("--np-rows", default="1,2")
    ap.add_argument("--chain", type=int, default=8); ap.add_argument("--reps", type=int, default=5); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pool", type=int, default=32); ap.add_argument("--tgk", default="auto", help="auto or gu,dn e.g. 2,4")
    ap.add_argument("--overhead", action="store_true", help="only the launch/encode/sync floor with tiny synthetic stacks")
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--no-time", action="store_true"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    A = ARCH[a.arch]
    if a.overhead:
        for rows in [int(v) for v in a.rows.split(",")]:
            f0, u0 = overhead(a.arch, rows, a.chain, a.reps, a.n)
            print(f"overhead floor {a.arch} rows {rows}: fused {f0:.3f} ms/layer, unfused {u0:.3f} ms/layer")
        return
    g, u, d = stacks = load(a.dir, A["proj"])
    E, topk, H = g.experts, A["topk"], g.kin
    cb = g.cb
    ks = {int(k): int((np.array(g.k) == k).sum()) for k in g.ks}
    print(f"# {a.dir}: experts {E} (K counts {ks}), hidden {H}, inter {g.nout}, topk {topk}, wpre {A['wpre']}, fq {A['fq']}, {cb}", flush=True)
    res = dict(dir=a.dir, arch=a.arch, experts=E, kcounts=ks, exact=[], timing=[])
    tg = None if a.tgk == "auto" else [int(v) for v in a.tgk.split(",")]
    mk = lambda cls, **kw: cls(g, u, d, topk, wpre=A["wpre"], tgk_gu=tg and tg[0], tgk_dn=tg and tg[1], **kw)
    # ---------------- exactness
    for rows in [int(v) for v in a.rows.split(",")]:
        ids, w = routing(rows, E, topk, 100 + rows)
        x = mx.array(np.random.default_rng(7 + rows).standard_normal((rows, H)).astype(np.float16))
        fused = mk(M.MoeGather, fq=A["fq"])
        unf = mk(M.MoeUnfused, fq=A["fq"])
        t0 = time.perf_counter()
        yf = np.array(fused(x, ids, w))
        yu = np.array(unf(x, ids, w))
        ent = dict(rows=rows, fused_vs_unfused_bitdiff=bitdiff(yf, yu), words=yf.size, tgk=[fused._params(rows)["tgu"], fused._params(rows)["tdn"]],
                   fused_vs_unfused=cmp(yf, yu))
        if str(rows) in a.np_rows.split(","):
            # fp64 reference: no fake quant (it is a deliberate fp8 rounding), compare the fused path with fq off
            yf0 = np.array(mk(M.MoeGather, fq=False)(x, ids, w))
            ref = np_ref(stacks, x, ids, w, 10.0, A["wpre"], cb)
            ent["fused_vs_numpy_fp64"] = cmp(yf0, ref)
            if A["fq"]:
                ent["fq_vs_numpy_fp64"] = cmp(yf, ref)
        res["exact"].append(ent)
        print(f"rows {rows}: fused vs unfused: {ent['fused_vs_unfused_bitdiff']}/{yf.size} words differ, rel {ent['fused_vs_unfused']['rel_max']:.2e}"
              + (f" | vs numpy fp64: rel {ent['fused_vs_numpy_fp64']['rel_max']:.2e} cos {ent['fused_vs_numpy_fp64']['cos']:.10f}" if "fused_vs_numpy_fp64" in ent else "")
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    if a.no_time:
        return
    # ---------------- timing
    rng = np.random.default_rng(3)
    router = mx.array((rng.standard_normal((E, H)) / np.sqrt(H)).astype(np.float32))
    for rows in [int(v) for v in a.rows.split(",")]:
        x0 = mx.array(rng.standard_normal((rows, H)).astype(np.float16))
        mx.eval(x0, router)
        ent = dict(rows=rows)
        cfgs = [("auto", None)]
        if a.sweep:
            cfgs = [(f"{p},{q}", (p, q)) for p in (1, 2, 4) for q in (1, 2, 4)]
        best = None
        for label, t in cfgs:
            fused = M.MoeGather(g, u, d, topk, wpre=A["wpre"], fq=A["fq"], tgk_gu=(t[0] if t else tg and tg[0]), tgk_dn=(t[1] if t else tg and tg[1]))
            run = chain_runner(lambda x, i, w: fused(x, i, w), router, x0, rows, a.chain, E, topk, a.pool, False)
            run(2)
            tf = med(run, a.reps, a.n) / a.chain
            if a.sweep:
                print(f"  rows {rows} tgk {label}: fused {tf * 1e3:.3f} ms/layer", flush=True)
            if best is None or tf < best[0]:
                best = (tf, label, fused._params(rows)["tgu"], fused._params(rows)["tdn"])
        ent["fused_ms"], ent["fused_tgk"] = best[0] * 1e3, [best[2], best[3]]
        unf = M.MoeUnfused(g, u, d, topk, wpre=A["wpre"], fq=A["fq"], tgk_gu=best[2], tgk_dn=best[3])
        run_u = chain_runner(lambda x, i, w: unf(x, i, w), router, x0, rows, a.chain, E, topk, a.pool, True)
        run_u(2)
        ent["unfused_ms"] = med(run_u, a.reps, a.n) / a.chain * 1e3
        ent["unfused_cached_gemv"] = len(unf._lin)
        ent["speedup"] = ent["unfused_ms"] / ent["fused_ms"]
        # weight traffic and rate (K from the touched experts is not known here: use mean K)
        meank = float(np.mean(np.array(g.k).astype(np.float64)))
        wts = rows * topk * 3 * g.kin * g.nout            # weights touched per layer (rows * topk experts, 3 projections), upper bound
        ent["gweights_per_s_fused"] = wts / (ent["fused_ms"] * 1e-3) / 1e9
        res["timing"].append(ent)
        print(f"rows {rows}: fused {ent['fused_ms']:.3f} ms/layer (tgk {best[2]},{best[3]}), unfused {ent['unfused_ms']:.3f} ms/layer "
              f"-> x{ent['speedup']:.1f}; {ent['gweights_per_s_fused']:.0f} Gw/s on distinct pairs, {len(unf._lin)} cached GEMVs", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
