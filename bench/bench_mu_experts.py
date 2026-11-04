"""Routed experts for R rows (R = N users x (D+1)): plain per-pair gather vs old union vs grouped (one decode per distinct expert), real DS4.1 slice layer.
   python bench/bench_mu_experts.py --dir models/ds2m --route route.npz [--rows 4,8,12,16,32] [--dplus1 4] [--sources real,uni384,ov384] [--rounds 7] [--json out.json]
Routing sources: `real` = router outputs captured from decode of independent users (tools/mu_route.py; the slice has 32 experts, so sharing is far above the
384-expert model), `uni384` / `ov384` = a 384-expert view of the same weights (the 32 experts tiled, same decode work per pair) with independent uniform
routing / each row copying half of the previous row's experts (verify chain). Interleaved rounds over the paths, ms per layer (chain of calls per eval), min and p25."""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from bench_moe import ARCH, load
from kyojin_mlx import moe as M

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--route", required=True); ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--rows", default="4,8,12,16,32"); ap.add_argument("--dplus1", type=int, default=4); ap.add_argument("--sources", default="real,uni384,ov384")
ap.add_argument("--rounds", type=int, default=7); ap.add_argument("--draws", type=int, default=4); ap.add_argument("--chain", type=int, default=6)
ap.add_argument("--paths", default="plain,union,grouped"); ap.add_argument("--json"); ap.add_argument("--cache-mb", type=int, default=256)
a = ap.parse_args()
mx.set_cache_limit(a.cache_mb * 2 ** 20)
A = ARCH["ds"]
g, u, d = load(a.dir, A["proj"])
for st in (g, u, d):                                   # the safetensors arrays are file mappings: copy them so no timed call page-faults
    st.words = st.words + mx.zeros((1,), st.words.dtype)
    mx.eval(st.words)
TILE = 12


def tiled(st):
    return M.Stack(st.words, mx.concatenate([st.offset[:-1]] * TILE + [st.offset[-1:]]), mx.tile(st.k, (TILE,)), mx.tile(st.suh, (TILE, 1)),
                   mx.tile(st.svh, (TILE, 1)), st.kt, st.nt, st.cb)


g3, u3, d3 = tiled(g), tiled(u), tiled(d)
topk, H = A["topk"], g.kin
mk = lambda stk, **kw: M.MoeGather(*stk, topk, wpre=A["wpre"], fq=A["fq"], pin_rows=1, **kw)
paths = a.paths.split(",")


def build(stk):
    m = {"plain": mk(stk, grouped=False, hotk=0), "union": mk(stk, union=True, grouped=False, hotk=0), "grouped": mk(stk, grouped=True, hotk=0)}
    for p in paths:
        if p.startswith("nax"):                                  # naxK: groups of >= K rows through the NAX GEMM
            m[p] = mk(stk, grouped=True, hotk=int(p[3:]))
    return m


mods = {"real": build((g, u, d)), "big": build((g3, u3, d3))}
z = np.load(a.route)
RX, RI, RW = z[f"L{a.layer}_x"], z[f"L{a.layer}_ids"], z[f"L{a.layer}_w"]
nu, ns = RI.shape[:2]
rng = np.random.default_rng(23)
ones = mx.ones((H,), mx.float16)


def draw(src, R):
    D1 = a.dplus1 if R % a.dplus1 == 0 else 1
    N = R // D1
    if src == "real":
        us = rng.choice(nu, N, replace=False)
        sl = [(u_, int(rng.integers(0, ns - D1))) for u_ in us]
        ids = np.concatenate([RI[u_, s:s + D1] for u_, s in sl]); w = np.concatenate([RW[u_, s:s + D1] for u_, s in sl])
        x = np.concatenate([RX[u_, s:s + D1] for u_, s in sl])
        return ids, w, x
    E = g3.experts
    ids = []
    for _ in range(N):
        prev = None
        for _ in range(D1):
            if src == "uni384" or prev is None:
                row = rng.choice(E, topk, replace=False)
            else:
                row = prev.copy()
                for j in range(topk):
                    if rng.random() > 0.5:
                        while True:
                            c = int(rng.integers(E))
                            if c not in row:
                                row[j] = c; break
            ids.append(row); prev = row
    w = rng.uniform(0.1, 1.0, (R, topk)).astype(np.float32); w = w / w.sum(-1, keepdims=True) * 1.5
    x = RX[rng.integers(0, nu, R), rng.integers(0, ns, R)]
    return np.stack(ids), w, x


def t_layer(fn, x0, ids, w):
    t0 = time.perf_counter()
    x = x0
    for _ in range(a.chain):
        x = mx.fast.rms_norm(fn(x, ids, w).astype(mx.float16), ones, 1e-6)
    mx.eval(x)
    return (time.perf_counter() - t0) / a.chain * 1e3


def stats(v):
    v = np.sort(np.array(v)); return float(v[0]), float(v[max(0, len(v) // 4 - 0)])


res = {}
print(f"# {a.dir} layer {a.layer}: topk {topk}, hidden {H}, inter {g.nout}, D+1 {a.dplus1}, chain {a.chain}, rounds {a.rounds} x draws {a.draws}", flush=True)
for src in a.sources.split(","):
    mm = mods["real" if src == "real" else "big"]
    for R in [int(v) for v in a.rows.split(",")]:
        mins = {p: [] for p in paths}; p25s = {p: [] for p in paths}; uniq = []; exact = True; errs = {}
        for _ in range(a.draws):
            ids, w, x = draw(src, R)
            uniq.append(len(set(ids.reshape(-1).tolist())))
            ids_m, w_m, x_m = mx.array(ids.astype(np.int32)), mx.array(w.astype(np.float32)), mx.array(x.astype(np.float16))
            if "union" in paths and R > 8:
                pass
            ref = np.array(mm["plain"](x_m, ids_m, w_m)).view(np.uint32)
            if "grouped" in paths:
                exact &= bool(np.array_equal(ref, np.array(mm["grouped"](x_m, ids_m, w_m)).view(np.uint32)))
            for p in paths:
                if p.startswith("nax"):
                    r64, o64 = np.array(mm["plain"](x_m, ids_m, w_m), np.float64), np.array(mm[p](x_m, ids_m, w_m), np.float64)
                    errs.setdefault(p, []).append(float(np.linalg.norm(r64 - o64) / np.linalg.norm(r64)))
            for p in paths:
                t_layer(mm[p], x_m, ids_m, w_m)                                      # warm (compile, params)
            sm = {p: [] for p in paths}
            for _ in range(a.rounds):
                for p in paths:
                    sm[p].append(t_layer(mm[p], x_m, ids_m, w_m))
            for p in paths:
                mn, q = stats(sm[p]); mins[p].append(mn); p25s[p].append(q)
        ent = {p: (float(np.mean(mins[p])), float(np.mean(p25s[p]))) for p in paths}                   # per draw min / p25 over the rounds, mean over draws
        res[f"{src}/R{R}"] = dict(unique=float(np.mean(uniq)), pairs=R * topk, exact=exact, ms=ent)
        print(f"{src:7s} R={R:2d} pairs {R * topk:3d} distinct {np.mean(uniq):5.1f} | " + " | ".join(f"{p} {ent[p][0]:6.3f}/{ent[p][1]:6.3f}" for p in paths)
              + f" ms (min/p25)" + (f" | grouped x{ent['plain'][0] / ent['grouped'][0]:.2f} vs plain, x{ent['union'][0] / ent['grouped'][0]:.2f} vs union" if "grouped" in paths and "union" in paths else "") + "".join(f" | {p} x{ent['plain'][0] / ent[p][0]:.2f}" for p in paths if p.startswith("nax")) + f" | bitwise {exact}" + "".join(f" | {p} relL2 {np.max(v):.1e}" for p, v in errs.items()), flush=True)
print(f"peak {mx.get_peak_memory() / 2**30:.2f} GiB")
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
