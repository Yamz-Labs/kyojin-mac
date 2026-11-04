"""Kernel autotune of a pack on THIS device (qualify --stage autotune, MLX). CLI, run with the venv python (PYTHONPATH = the engine checkout + tools):
  autotune_mlx.py tune PACK OUT.json [--rounds 5] [--quick] [--only INxOUTxK ...] [--add INxOUTxK ...]    time the candidates of every dense linear shape of the pack, write
                                                                                        <PACK>/tuning/<device>.json (kyojin_mlx/tuning.py) + the full table to OUT.json
  autotune_mlx.py shapes PACK                                                           the shapes found in the pack header (no GPU)
Shapes come from the safetensors headers (every 3-D `*.trellis` tensor = one dense EXL3 linear [kt, nt, 16K]); weights are random trellis words of that shape,
so a pack of any size is tuned in a few hundred MB: the Ultra pack is never loaded. Stacked tensors (MoE experts, grouped wo_a, 4-D) are not tuned here.
Candidates (all through the real Exl3Linear code path):
  decode (1 row)   tgk, the k-splits of the GEMV: powers of two up to kt / 8 (default = kernels.pick_tgk: the device fill, tools/gemv_fill.py --host)
  prefill rows R   fused trellis-decode + NAX matmul with each of the 3 NAX tiles of metal_nax._BLOCK_CFG, against decode + MLX matmul; fused_max = the largest R
                   where a fused variant wins, nax_cfg[R] = the fastest tile
Method: short chains of dependent calls, interleaved candidates, a short idle gap before each timing (burst clock), `min` over rounds. A candidate replaces the default only
when its min is at least MIN_GAIN faster in two successive head-to-heads (below that is noise: the default stays), and only when its output matches the default's to REL_TOL."""
import argparse
import glob
import gc
import json
import random
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

MIN_GAIN = 0.05
REL_TOL = 5e-3
ROWS_GRID = (5, 8, 12, 16, 32, 128, 256, 512, 1024, 2048)
IDLE_S = 0.1
BIG_WEIGHT_BYTES = 64 * 2 ** 20       # packed weight above this: decode candidates limited to the default's neighbours
BIG_DENSE_BYTES = 384 * 2 ** 20       # fp16 weight above this: the decode + matmul variant is not built (1 GiB for a 248K head), fused only


def shapes_of(pack) -> dict:
    """{(in, out, K): count} of the 3-D `*.trellis` tensors in the pack's safetensors headers."""
    out: dict = {}
    for f in sorted(glob.glob(str(Path(pack) / "model*.safetensors"))):
        with open(f, "rb") as b:
            n = struct.unpack("<Q", b.read(8))[0]
            h = json.loads(b.read(n))
        for k, v in h.items():
            if k.endswith(".trellis") and len(v["shape"]) == 3:
                kt, nt, w = v["shape"]
                out[(kt * 16, nt * 16, w // 16)] = out.get((kt * 16, nt * 16, w // 16), 0) + 1
    return out


def codebook_of(pack) -> str:
    return json.load(open(Path(pack) / "config.json"))["quantization_config"].get("codebook", "mul1")


def make_linear(kin, nout, K, cb, tune=None, seed=0):
    from kyojin_mlx.exl3_linear import Exl3Linear
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-2 ** 15, 2 ** 15, size=(kt, nt, 16 * K), dtype=np.int16))
    suh = mx.array(rng.choice([-1.0, 1.0], kin).astype(np.float16))
    svh = mx.array((rng.standard_normal(nout) * 0.05).astype(np.float16))
    lin = Exl3Linear(tr, suh, svh, cb=cb, tune=tune or {})
    mx.eval(lin._words)
    return lin


def chain(lin, x, n):
    """n dependent calls (the output feeds the next input: sliced or tiled to the input width)."""
    kin, nout = lin.in_features, lin.out_features
    y = x
    for _ in range(n):
        y = lin(y)
        y = y[..., :kin] if nout >= kin else mx.concatenate([y] * (-(-kin // nout)), axis=-1)[..., :kin]
    return y


def bench(cands: dict, rounds: int, n_chain: int, idle: float = IDLE_S) -> dict:
    """cands: name -> thunk(n) returning an mx.array. Interleaved rounds in a shuffled order; per name: per-call ms min and p25."""
    for t in cands.values():
        mx.eval(t(2))
    ts = {k: [] for k in cands}
    names = list(cands)
    for _ in range(rounds):
        random.shuffle(names)
        for k in names:
            time.sleep(idle)
            t0 = time.perf_counter()
            mx.eval(cands[k](n_chain))
            ts[k].append((time.perf_counter() - t0) / n_chain * 1e3)
    return {k: dict(min=min(v), p25=float(np.percentile(v, 25))) for k, v in ts.items()}


def choose(res: dict, dflt, cands: dict, rounds: int, n_chain: int):
    """The fastest candidate if it beats the default by MIN_GAIN, confirmed by a second head-to-head of the pair with twice the rounds (the Air's GPU is shared and
    its clock moves: a first-round win that does not repeat is noise and the default stays)."""
    best = min(res, key=lambda k: res[k]["min"])
    if best == dflt or res[best]["min"] >= res[dflt]["min"] * (1 - MIN_GAIN):
        return dflt, res
    again = bench({best: cands[best], dflt: cands[dflt]}, rounds * 2, n_chain)
    res = {**res, **{k: dict(v, first=res[k]["min"]) for k, v in again.items()}}
    return (best if again[best]["min"] < again[dflt]["min"] * (1 - MIN_GAIN) else dflt), res


def rel_diff(a, b) -> float:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-9))


def tune_decode(kin, nout, K, cb, rounds) -> dict:
    from kyojin_mlx import kernels as KN
    kt = kin // 16
    dflt = KN.pick_tgk(kt, nout // 128)
    cand = [t for t in (1, 2, 4, 8, 16, 32) if t * 8 <= kt or t == dflt]
    if kin * nout * K // 8 > BIG_WEIGHT_BYTES:                       # wide head: only the neighbours of the default (memory: one instance per candidate)
        cand = [t for t in cand if dflt // 2 <= t <= dflt * 2]
    x = mx.array(np.random.default_rng(1).standard_normal((1, kin)).astype(np.float16))
    lins = {t: make_linear(kin, nout, K, cb, {"tgk": t}) for t in cand}
    ref = lins[dflt](x)
    ok = {t: rel_diff(l(x), ref) for t, l in lins.items()}
    cands = {t: (lambda n, l=l: chain(l, x, n)) for t, l in lins.items() if ok[t] <= REL_TOL}
    pick, res = choose(bench(cands, rounds, 24), dflt, cands, rounds, 24)
    out = dict(default=dflt, pick=pick, ms={str(t): round(r["min"], 4) for t, r in res.items()}, rel_diff={str(t): float(f"{v:.2g}") for t, v in ok.items()})
    del lins
    gc.collect(); mx.clear_cache()
    return out


def tune_prefill(kin, nout, K, cb, rounds) -> dict:
    from kyojin_mlx import metal_nax as NX
    from kyojin_mlx import exl3_linear as EL, tuning as TN
    lin = make_linear(kin, nout, K, cb)
    cfgs = {b: c for b, c in NX._BLOCK_CFG.items()}
    rows_out = {}
    mma0, ks0 = EL.MMA_ROWS, EL.NAX_KS
    EL.MMA_ROWS = 0                                               # the tile sweep times the fused NAX path itself: no MMA GEMV, no k-split in its way
    EL.NAX_KS = 0
    for R in ROWS_GRID:
        if R * kin * 2 > 2 ** 27 or R * nout * 2 > 2 ** 28:        # 128 MiB inputs / 256 MiB outputs per call: keeps the peak well under 4 GiB (the Air's rule)
            continue
        x = mx.array(np.random.default_rng(2).standard_normal((1, R, kin)).astype(np.float16))
        dflt_cfg = NX.pick_cfg([R])
        variants = {**({} if kin * nout * 2 > BIG_DENSE_BYTES else {"dense": None}), **{f"fused{b}": c for b, c in cfgs.items()}}

        def run(name):
            def f(n):
                lin.fused_max = 0 if name == "dense" else 10 ** 9
                lin.tune = {} if name == "dense" else {"nax_cfg": {str(R): list(variants[name])}}
                lin._nax_rows = {}
                return chain(lin, x, n)
            return f
        ref = None
        diffs = {}
        for name in variants:
            y = run(name)(1)
            mx.eval(y)
            ref = y if ref is None else ref                   # first variant = the reference of the output check (dense when it exists)
            diffs[name] = rel_diff(y, ref)
        # the default behaviour at R: what the constants choose
        lin.fused_max, lin.tune, lin._nax_rows = EL.DENSE_FUSED_MAX, {}, {}
        dflt_name = "dense" if not lin._fused_ok(R) and "dense" in variants else next(n for n, c in variants.items() if c == dflt_cfg)
        cands = {n: run(n) for n in variants if diffs[n] <= REL_TOL}
        nc = 2 if R >= 512 else 6
        pick, res = choose(bench(cands, rounds, nc), dflt_name, cands, rounds, nc)
        rows_out[str(R)] = dict(default=dflt_name, pick=pick, ms={n: round(r["min"], 4) for n, r in res.items()}, rel_diff={n: float(f"{v:.2g}") for n, v in diffs.items()})
        if 5 <= R <= 16 and pick.startswith("fused") and nout % 128 == 0:       # the verify band: k-split of the fused matmul (1 = none), against the constants' choice
            EL.NAX_KS = -1
            lin.fused_max, lin.tune, lin._nax_rows, lin._nax_ks = 10 ** 9, {}, {}, {}
            kd = lin._ks_for(R)
            EL.NAX_KS = 0
            kc = {f"ks{k}": k for k in (1, 2, 4, 8)}

            def run_ks(k):
                def f(n):
                    lin.fused_max = 10 ** 9
                    lin.tune = {"nax_cfg": {str(R): list(cfgs[int(pick[5:])])}, "nax_ks": {str(R): k}}
                    lin._nax_rows, lin._nax_ks = {}, {}
                    return chain(lin, x, n)
                return f
            kref = None
            kdiff = {}
            for n, k in kc.items():
                y = run_ks(k)(1)
                mx.eval(y)
                kref = y if kref is None else kref
                kdiff[n] = rel_diff(y, kref)
            kcands = {n: run_ks(k) for n, k in kc.items() if kdiff[n] <= REL_TOL and (k == 1 or lin._kt % (k * 4) == 0)}
            kpick, kres = choose(bench(kcands, rounds, 6), f"ks{kd}", kcands, rounds, 6)
            rows_out[str(R)]["ks"] = dict(default=f"ks{kd}", pick=kpick, ms={n: round(r["min"], 4) for n, r in kres.items()}, rel_diff={n: float(f"{v:.2g}") for n, v in kdiff.items()})
            # MMA GEMV against the best fused matmul at this row count: the per-shape mma_max
            kbest = int(kpick[2:])
            cfg_b = list(cfgs[int(pick[5:])])

            def run_mma(on):
                def f(n):
                    lin.fused_max = 10 ** 9
                    lin.tune = {"nax_cfg": {str(R): cfg_b}, "nax_ks": {str(R): kbest}, "mma_max": 16}
                    lin._nax_rows, lin._nax_ks = {}, {}
                    EL.MMA_ROWS = 5 if on else 0
                    try:
                        return chain(lin, x, n)
                    finally:
                        EL.MMA_ROWS = 0
                return f
            mdiff = rel_diff(run_mma(True)(1), run_mma(False)(1))
            mdflt = "mma" if R <= TN.chip()["mma_max"] else "nax"
            mc = {"mma": run_mma(True), "nax": run_mma(False)} if mdiff <= REL_TOL else {"nax": run_mma(False)}
            mpick, mres = choose(bench(mc, rounds, 6), mdflt if mdflt in mc else "nax", mc, rounds, 6)
            rows_out[str(R)]["mma"] = dict(default=mdflt, pick=mpick, ms={n: round(r["min"], 4) for n, r in mres.items()}, rel_diff=float(f"{mdiff:.2g}"))
        del x
        mx.clear_cache()
    EL.MMA_ROWS, EL.NAX_KS = mma0, ks0
    return rows_out


def entry_from(dec: dict, pf: dict) -> dict:
    from kyojin_mlx import metal_nax as NX
    e = {}
    if dec["pick"] != dec["default"]:
        e["tgk"] = dec["pick"]
    fused_rows = [int(r) for r, v in pf.items() if v["pick"].startswith("fused")]
    dense_rows = [int(r) for r, v in pf.items() if v["pick"] == "dense"]
    if pf:
        # crossover: the largest tested R below the first R where decode + matmul wins (rows are monotone in practice)
        first_dense = min(dense_rows) if dense_rows else None
        fm = max([r for r in fused_rows if first_dense is None or r < first_dense], default=0)
        from kyojin_mlx import exl3_linear as EL
        top = max(int(r) for r in pf)
        if first_dense is not None or top >= EL.DENSE_FUSED_MAX:      # a crossover was seen, or fused wins up to the constant: the table states the value it measured
            e["fused_max"] = fm                                       # (a truncated grid, e.g. the wide head, that never saw a crossover keeps the constant)
        cfg = {str(r): list(NX._BLOCK_CFG[int(pf[str(r)]["pick"][5:])]) for r in fused_rows if first_dense is None or r < first_dense}
        if cfg:
            e["nax_cfg"] = cfg
        kb = {r: v["ks"] for r, v in pf.items() if "ks" in v}
        if any(v["pick"] != v["default"] for v in kb.values()):         # a row count takes the entry of the next tuned one: state the whole band once one differs
            e["nax_ks"] = {r: int(v["pick"][2:]) for r, v in kb.items()}
        mm = {int(r): v["mma"] for r, v in pf.items() if "mma" in v}
        if mm and any(v["pick"] != v["default"] for v in mm.values()):           # rows between two tested ones follow the next tested row (as nax_ks)
            win = 4                                                              # = GEMV_MAX_ROWS: the MMA band is empty
            for r in sorted(mm):
                if mm[r]["pick"] != "mma":
                    break
                win = r
            e["mma_max"] = win
    return e


def changes_of(e: dict) -> list:
    """What the entry changes against the constants: tgk (when set), fused_max (when not DENSE_FUSED_MAX), the NAX tiles that are not pick_cfg's."""
    from kyojin_mlx import exl3_linear as EL, metal_nax as NX
    out = []
    if "tgk" in e:
        out.append("tgk")
    if e.get("fused_max", EL.DENSE_FUSED_MAX) != EL.DENSE_FUSED_MAX:
        out.append(f"fused_max={e['fused_max']}")
    out += [f"cfg@{r}" for r, c in e.get("nax_cfg", {}).items() if tuple(c) != NX.pick_cfg([int(r)])]
    if "mma_max" in e:
        out.append(f"mma_max={e['mma_max']}")
    if "nax_ks" in e:
        out.append("ks@" + ",".join(f"{r}={k}" for r, k in e["nax_ks"].items()))
    return out


def tune_gdn(pack, rounds: int) -> dict:
    """Gated-delta recurrence of Qwen3.5 / Flash-Next packs: stock mlx_lm step kernel vs mx.fast.gated_delta_update (chunked, NAX) at the pack's head dims.
    Returns {"min_t": N} (N = smallest tested row count from which the fast path is >= MIN_GAIN faster at every larger one) or {} (keep stock)."""
    cfg = json.loads((Path(pack) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    Hk, Hv = tc.get("linear_num_key_heads"), tc.get("linear_num_value_heads")
    if not Hk or tc.get("linear_key_head_dim") != 128 or tc.get("linear_value_head_dim") != 128:
        return {}
    from mlx_lm.models.gated_delta import gated_delta_kernel
    wins = {}
    for T in (16, 64, 256, 512, 2048, 4096):
        mx.random.seed(T)
        q, k = (mx.random.normal((1, T, Hk, 128)).astype(mx.bfloat16) * 0.088 for _ in range(2))
        v = mx.random.normal((1, T, Hv, 128)).astype(mx.bfloat16)
        g = mx.exp(-mx.random.uniform(0.001, 0.2, (1, T, Hv))); b = mx.sigmoid(mx.random.normal((1, T, Hv)))
        s0 = mx.random.normal((1, Hv, 128, 128)) * 0.1
        mx.eval(q, k, v, g, b, s0)
        old = lambda: gated_delta_kernel(q, k, v, g, b, s0)
        new = lambda: mx.fast.gated_delta_update(q, k, v, g, b, s0)
        n = max(1, 2048 // T); best = {"old": 1e9, "new": 1e9}
        mx.eval(old()); mx.eval(new())
        for _ in range(rounds):
            for name, f in (("old", old), ("new", new)):
                time.sleep(0.1); t = time.perf_counter()
                for _ in range(n):
                    mx.eval(f())
                best[name] = min(best[name], (time.perf_counter() - t) / n)
        wins[T] = best["new"] <= best["old"] * (1 - MIN_GAIN)
        print(f"gdn T={T}: stock {best['old']*1e3:.2f} ms  fast {best['new']*1e3:.2f} ms  {'fast' if wins[T] else 'stock'}", flush=True)
    ts = sorted(wins)
    ok = [t for t in ts if all(wins[u] for u in ts if u >= t)]
    return {"min_t": ok[0]} if ok else {}


SEAM_GRID = (256, 512, 1024, 2048)
SEAM_TOL = 0.01          # a variant counts as "not slower" within this noise margin and as "faster" beyond it


def seam_thresholds(times: dict) -> dict:
    """times {rows: {"off": s, "mlp": s, "layer": s, "split": s}} (prefill chunk seconds with the fused seams off, the fused MLP middle, + layer seams, + layer seams in the split form)
    -> the "prefill" table entry (keys not listed keep their defaults, see tuning.py). Rows between grid points take the rule of the grid point above them."""
    rows = sorted(times)
    t = lambda r, v: times[r][v]
    out: dict = {}
    ok_mlp = [r for r in rows if all(t(u, "mlp") <= t(u, "off") * (1 + SEAM_TOL) for u in rows if u >= r)]
    if not ok_mlp:
        return {"mlp_pf": 0, "layer_pf": 0}
    if ok_mlp[0] > rows[0]:
        out["mlp_min_rows"] = ok_mlp[0]
    good = [r for r in rows if r >= ok_mlp[0] and all(t(u, "layer") <= t(u, "mlp") * (1 + SEAM_TOL) for u in rows if ok_mlp[0] <= u <= r)]
    if not good:
        out["layer_pf"] = 0
        return out
    out["layer_max_rows"] = good[-1]
    better = [r for r in rows if r > good[-1] and all(t(u, "split") <= t(u, "mlp") * (1 - SEAM_TOL) for u in rows if good[-1] < u <= r)]
    if better:
        out["layer_split_max_rows"] = better[-1]
    return out


def tune_seams(pack, rounds: int) -> dict:
    """Whole-model prefill chunk of the pack with the fused seams off / MLP middle / + layer seams / + split seam 1, at SEAM_GRID rows -> seam_thresholds. {} when the pack has no dense Qwen3.5 layers
    (every variant is then the same code: nothing differs by more than the noise)."""
    from kyojin_mlx import layer_pf as LP, mlp_pf as MP
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(str(pack))
    def setv(v):
        MP.ENABLED = v != "off"
        LP.ENABLED = v in ("layer", "split")
        LP.MAX_ROWS, LP.SPLIT_MAX_ROWS = (1 << 20, 0) if v == "layer" else ((0, 1 << 20) if v == "split" else (0, 0))
    times = {}
    try:
        for R in SEAM_GRID:
            ids = [(i * 7919) % 5000 + 100 for i in range(R)]
            best = {v: 1e9 for v in ("off", "mlp", "layer", "split")}
            for v in best:
                setv(v); mx.eval(be.prefill_chunk(be.make_cache(), ids))
            for _ in range(rounds):
                for v in best:
                    setv(v); c = be.make_cache(); mx.synchronize(); time.sleep(0.1); t0 = time.perf_counter()
                    mx.eval(be.prefill_chunk(c, ids)); best[v] = min(best[v], time.perf_counter() - t0)
            times[R] = best
            print(f"seams R={R}: " + "  ".join(f"{v} {best[v]*1e3:.1f} ms" for v in best), flush=True)
    finally:
        MP.ENABLED = LP.ENABLED = LP.MAX_ROWS = LP.SPLIT_MAX_ROWS = None
    if all(max(abs(b / r["off"] - 1) for b in r.values()) <= SEAM_TOL for r in times.values()):
        return {}
    return seam_thresholds(times)


def cmd_tune(a):
    from kyojin_mlx import exl3_linear as EL, tuning as TN
    cb = codebook_of(a.pack)
    shapes = shapes_of(a.pack)
    for sp in a.add or []:                      # shapes not in the pack: a model that is not here yet (day-0 what-if), same codebook
        kin, nout, K = (int(v) for v in sp.split("x"))
        shapes[(kin, nout, K)] = 0
    if a.only:
        want = {tuple(int(v) for v in s.split("x")) for s in a.only}
        shapes = {k: v for k, v in shapes.items() if k in want}
    t_all = time.time()
    table, full = {}, {}
    for (kin, nout, K), cnt in sorted(shapes.items()):
        t0 = time.time()
        dec = tune_decode(kin, nout, K, cb, a.rounds)
        pf = {} if a.quick else tune_prefill(kin, nout, K, cb, a.rounds)
        e = entry_from(dec, pf) if pf else ({"tgk": dec["pick"]} if dec["pick"] != dec["default"] else {})
        k = TN.key(kin, nout, K)
        table[k] = e
        ch = changes_of(e)
        full[k] = dict(count=cnt, decode=dec, prefill=pf, entry=e, changes=ch, wall_s=round(time.time() - t0, 1))
        print(f"{k} x{cnt}: decode tgk {dec['default']}->{dec['pick']} ({dec['ms'].get(str(dec['default']))} -> {dec['ms'].get(str(dec['pick']))} ms); "
              f"prefill fused_max {e.get('fused_max', '-')}; vs constants: {', '.join(ch) or 'nothing'}; {time.time()-t0:.0f} s", flush=True)
    if not a.quick:
        gd = tune_gdn(a.pack, a.rounds)
        if gd:
            table["gdn"] = gd
        print(f"gdn: {gd or 'stock'}")
        sm = tune_seams(a.pack, a.rounds)
        if sm:
            table["prefill"] = sm
        print(f"prefill seams: {sm or 'defaults'}")
    meta = dict(pack=str(a.pack), codebook=cb, rounds=a.rounds, min_gain=MIN_GAIN, rel_tol=REL_TOL, rows_grid=list(ROWS_GRID), wall_s=round(time.time() - t_all, 1),
                peak_gib=round(mx.get_peak_memory() / 2 ** 30, 2), at=time.strftime("%F %T"))
    f = TN.write(a.pack, table, meta)
    Path(a.out).write_text(json.dumps(dict(file=str(f), device=TN.device_id(), kernel_version=TN.kernel_version(), **meta, shapes=full), indent=1))
    print(f"wrote {f} ({len(table)} shapes, {sum(1 for v in full.values() if v['changes'])} differ from the constants, {meta['wall_s']} s, peak {meta['peak_gib']} GiB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("pack"); t.add_argument("out")
    t.add_argument("--rounds", type=int, default=5); t.add_argument("--quick", action="store_true", help="decode k-splits only")
    t.add_argument("--only", nargs="*", help="shapes INxOUTxK to tune")
    t.add_argument("--add", nargs="*", help="extra shapes INxOUTxK (not in the pack: what-if for a model that is not converted yet)")
    s = sub.add_parser("shapes"); s.add_argument("pack")
    a = ap.parse_args()
    if a.cmd == "shapes":
        for k, v in sorted(shapes_of(a.pack).items()):
            print(k, v)
        return
    cmd_tune(a)


if __name__ == "__main__":
    main()
