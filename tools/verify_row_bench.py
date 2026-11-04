#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify-cost levers on synthetic weights of the pack's real geometry (run on the Mac; nothing here was timed by the author, there is no GPU in the sandbox
it was written in).

Arms, each read at call time and measured A B B A per row count (one variable at a time):
  moe_sgred   KYOJIN_MOE_SGRED 0 / 1   one routed MoE layer (gate | up -> swiglu -> down) at 1..6 rows, grouped path, distinct or overlapping routing
  fuse_rows   KYOJIN_FUSE_ROWS 0 / 6   the dense linears of a layer at 1..6 rows: wo_b (single), wq_a + wkv (same-input group), sh_down (single), wo_a (8 groups)
Timing: slope of chains of n and 4n calls on fresh inputs in one eval (no per-eval floor), min over --reps; us per call, then the per-row slope
(t(r) - t(1)) / (r - 1) and, labelled as a projection, x --layers layers. The projection is the sum of these layer pieces only: attention, indexer, glue
and lm_head are not in it, it is not a served number.
Numerics (--no-num to skip): output of each arm against an f32 numpy reference, rows 5 and 6 by default; a synthetic head (hidden -> --vocab f32 random)
turns the layer output into logits: top-1 agreement with the reference rows and relative logit error, `today` and `lever` side by side.
Memory: expert banks + dense weights are counted before anything is allocated and the run stops above --cap-gib (3.5).

  python tools/verify_row_bench.py [--rows 1,2,3,4,5,6] [--experts 64] [--route distinct|overlap] [--reps 5] [--layers 40] [--no-num] [--no-time]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import ref_numpy as R  # noqa: E402

# real geometry of the pack (tools/dense_gemv_bench.py SH, config.json): hidden 5120, moe inter 2304, topk 6, q_lora 1280, kv 512, o 8 groups x (4096 -> 1024)
HIDDEN, INTER, TOPK = 5120, 2304, 6
DENSE = {"wo_b": (8192, 5120, 5), "wq_a": (5120, 1280, 5), "wkv": (5120, 512, 5), "sh_down": (2304, 5120, 5)}
WO_A = (8, 4096, 1024, 5)                                   # groups, kin, nout, K
KCYCLE = (2, 3, 3, 4)                                       # bit width per expert (mean 3)


# ------------------------------------------------------------------------------------------------ pure numpy (no mlx): synthetic weights + f32 reference
def expert_arrays(proj, e, kin, nout, K):
    """Deterministic random valid linear of (projection, expert): the reference regenerates it instead of keeping the bank on the host."""
    rng = np.random.default_rng(1_000_003 * (proj + 1) + e)
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    return tr, suh, svh


def bank_bytes(experts, hidden=HIDDEN, inter=INTER):
    n = 0
    for e in range(experts):
        K = KCYCLE[e % len(KCYCLE)]
        n += 3 * (hidden * inter // 16 * K * 2) + 2 * (hidden + inter) * 2 * 3
    return n


def dense_bytes():
    n = sum(k * o // 16 * K * 2 for k, o, K in DENSE.values())
    g, k, o, K = WO_A
    return n + g * k * o // 16 * K * 2


def lin_ref(X, tr, suh, svh, wr=None):
    """f32 reference of one linear on rows X [r, k]: same math as ref_numpy.gemv(dtype=float32), with the decoded Wr passed in to decode once per expert."""
    wr = R.decode_tiles(tr, "mul1").astype(np.float32) if wr is None else wr
    xh = R.fwht128(X.astype(np.float32) * suh.astype(np.float32)[None, :], axis=1) * np.float32(R.INV_SQRT_HAD)
    inner = R.fwht128(xh @ wr, axis=1) * np.float32(R.INV_SQRT_HAD)
    return inner * svh.astype(np.float32)[None, :]


def moe_ref(x, ids, w, hidden, inter, limit=10.0, wpre=True, expert_fn=expert_arrays, kof=lambda e: KCYCLE[e % len(KCYCLE)]):
    """Routed MoE layer in f32 (swiglu with the clamp of the layer, routing weight before the down projection when wpre). One decode per distinct expert."""
    rows, topk = ids.shape
    out = np.zeros((rows, hidden), np.float32)
    for e in sorted({int(v) for v in ids.reshape(-1)}):
        sel = [(r, j) for r in range(rows) for j in range(topk) if int(ids[r, j]) == e]
        X = x[[r for r, _ in sel]].astype(np.float32)
        K = kof(e)
        g = lin_ref(X, *expert_fn(0, e, hidden, inter, K))
        u = lin_ref(X, *expert_fn(1, e, hidden, inter, K))
        gt, ut = np.minimum(g, limit), np.clip(u, -limit, limit)
        a = gt / (1.0 + np.exp(-gt)) * ut
        ww = np.array([w[r, j] for r, j in sel], np.float32)[:, None]
        if wpre:
            a = a * ww
        y = lin_ref(a.astype(np.float32), *expert_fn(2, e, inter, hidden, K))
        if not wpre:
            y = y * ww
        for i, (r, _) in enumerate(sel):
            out[r] += y[i]
    return out


def route(rows, experts, mode, seed):
    """distinct: every row its own topk experts (the worst case for sharing); overlap: rows share half of their experts (what a verify block of one text does)."""
    rng = np.random.default_rng(seed)
    base = rng.choice(experts, TOPK, replace=False)
    ids = []
    for r in range(rows):
        if mode == "overlap":
            keep = base[: TOPK // 2]
            rest = rng.choice([e for e in range(experts) if e not in keep], TOPK - len(keep), replace=False)
            ids.append(np.concatenate([keep, rest]))
        else:
            ids.append(rng.choice(experts, TOPK, replace=False))
    w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
    return np.stack(ids).astype(np.int32), (w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32)


def head_metrics(y, ref, head):
    """(relative output error, relative logit error, top-1 matches of the rows) with a synthetic f32 head [hidden, vocab]."""
    y, ref = y.astype(np.float32), ref.astype(np.float32)
    ly, lr = y @ head, ref @ head
    rel_y = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
    rel_l = float(np.linalg.norm(ly - lr) / np.linalg.norm(lr))
    top = int((ly.argmax(-1) == lr.argmax(-1)).sum())
    return rel_y, rel_l, top, y.shape[0]


# ------------------------------------------------------------------------------------------------ mlx side
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,3,4,5,6")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--route", default="distinct,overlap")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--num-rows", default="6", help="rows checked against the f32 reference (each distinct expert is decoded on the host, ~11 s each: ~5 min per routing at 6 rows); 5,6 for both")
    ap.add_argument("--no-num", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--cap-gib", type=float, default=3.5)
    a = ap.parse_args()
    rows_l = [int(v) for v in a.rows.split(",")]
    need = bank_bytes(a.experts) + dense_bytes()
    print(f"weights: {a.experts} experts {bank_bytes(a.experts)/2**30:.2f} GiB + dense {dense_bytes()/2**30:.2f} GiB = {need/2**30:.2f} GiB (cap {a.cap_gib})")
    if need > a.cap_gib * 2**30:
        raise SystemExit(f"over the {a.cap_gib} GiB cap: lower --experts")

    import mlx.core as mx
    if not mx.metal.is_available():
        raise SystemExit("verify_row_bench: needs a Metal GPU (the numpy reference is importable on its own)")
    from kyojin_mlx import moe as M, kernels as KN, exl3_linear as EL, fused_group as FG, grouped as GR

    try:
        dev = mx.metal.device_info().get("device_name", "?")
    except Exception:                                                        # deprecated in newer MLX
        dev = "?"
    print("device:", dev, "| flags at start: SGRED", M.SGRED, "FUSE_ROWS", KN.FUSE_ROWS)

    # ---- build the routed layer
    def stack(proj, kin, nout):
        words, off, ks = [], [0], []
        sh, sv = [], []
        for e in range(a.experts):
            K = KCYCLE[e % len(KCYCLE)]
            tr, su, sv_ = expert_arrays(proj, e, kin, nout, K)
            w = np.ascontiguousarray(tr).view("<u4").reshape(-1)
            words.append(w); off.append(off[-1] + len(w)); ks.append(K); sh.append(su); sv.append(sv_)
        st = M.Stack(mx.array(np.concatenate(words)), mx.array(np.array(off, "<u4")), mx.array(np.array(ks, "u1")), mx.array(np.stack(sh)), mx.array(np.stack(sv)),
                     kin // 16, nout // 16, "mul1")
        mx.eval(st.words, st.offset, st.k, st.suh, st.svh)
        return st
    g, u, d = stack(0, HIDDEN, INTER), stack(1, HIDDEN, INTER), stack(2, INTER, HIDDEN)
    moe = M.MoeGather(g, u, d, TOPK, wpre=True, fq=False, grouped=True, hotk=0, pin_rows=1)     # pin_rows=1 as rows.enable sets it on the served model
    EL.ROWS_NAX, EL.ROW_SPLIT_MAX = False, 4 * KN.MAX_ROWS                    # the served verify rows (rows.enable, KYOJIN_VERIFY_NAX=0): 5 / 6 rows = the GEMV many-rows path, not NAX / MMA

    # ---- dense pieces
    rng = np.random.default_rng(5)

    def lin(name):
        kin, nout, K = DENSE[name]
        tr, su, sv = R.random_linear(kin, nout, K, seed=sorted(DENSE).index(name) + 11)
        m = EL.Exl3Linear(mx.array(tr), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16)
        mx.eval(m.parameters())
        return m, (tr, su, sv)
    wo_b, wo_b_np = lin("wo_b")
    wq_a, wq_a_np = lin("wq_a")
    wkv, wkv_np = lin("wkv")
    sh_down, sh_down_np = lin("sh_down")
    grp = FG.Exl3Group([wq_a, wkv])
    G_, kin_o, nout_o, K_o = WO_A
    tro = rng.integers(-32768, 32768, (G_, kin_o // 16, nout_o // 16, 16 * K_o)).astype(np.int16)
    suo = (rng.choice([-1.0, 1.0], (G_, kin_o)) * rng.uniform(.5, 1.5, (G_, kin_o))).astype(np.float16)
    svo = (rng.choice([-1.0, 1.0], (G_, nout_o)) * rng.uniform(.5, 1.5, (G_, nout_o))).astype(np.float16)
    wo_a = GR.GroupedGemv(mx.array(tro), mx.array(suo), mx.array(svo))
    mx.eval(wo_a.words)

    def xin(rows, k, seed, dt=np.float16):
        return mx.array((np.random.default_rng(seed).standard_normal((rows, k)) * 0.7).astype(dt))

    # ---- arms
    def set_moe(on):
        M.SGRED = bool(on)

    def set_fr(rows):
        KN.FUSE_ROWS = rows

    def call_moe(rows, mode):
        ids, w = route(rows, a.experts, mode, 100 + rows)
        ins = [(xin(rows, HIDDEN, 10 + i), mx.array(ids), mx.array(w)) for i in range(3)]
        return (lambda x, i, w: moe(x, i, w)), ins

    def call_dense(name, rows):
        k_in = {"wo_b": 8192, "wq_a": 5120, "wkv": 5120, "sh_down": 2304}
        if name == "wo_b":
            return (lambda x: wo_b(x)), [(xin(rows, 8192, 20 + i),) for i in range(3)]
        if name == "sh_down":
            return (lambda x: sh_down(x)), [(xin(rows, 2304, 30 + i),) for i in range(3)]
        if name == "s5":
            # today: rows above 4 leave the group (FusedMember falls back to each linear); lever: the group's one-launch kernel when it fits
            def f(x):
                if rows <= KN.fuse_rows() and grp.can_fuse(rows):
                    return grp.run(x)
                return [wq_a(x), wkv(x)]
            return f, [(xin(rows, 5120, 40 + i),) for i in range(3)]
        if name == "wo_a":
            return (lambda x: wo_a(x)), [(mx.array((np.random.default_rng(50 + i).standard_normal((1, G_, rows, kin_o)) * .3).astype(np.float16)),) for i in range(3)]
        raise KeyError(name)

    def timeit(f, ins):
        n = len(ins)

        def run(c):
            outs = [f(*ins[i % n]) for i in range(c)]
            flat = [o for out in outs for o in (out if isinstance(out, (list, tuple)) else [out])]
            t0 = time.perf_counter(); mx.eval(*flat); return time.perf_counter() - t0
        T = {n: [], 4 * n: []}
        for r in range(a.reps + 1):
            for c in T:
                dt = run(c)
                if r:
                    T[c].append(dt)
        return (min(T[4 * n]) - min(T[n])) / (3 * n) * 1e6

    if not a.no_time:
        print("\n== timing: us per call (slope of n and 4n chained calls, min of %d), arms interleaved ==" % a.reps)
        table = {}
        for mode in a.route.split(","):
            print(f"\nMoE layer, routing {mode}   rows:" + "".join(f"{r:>9}" for r in rows_l))
            best = {0: [], 1: []}
            for r in rows_l:
                f, ins = call_moe(r, mode)
                cur = {0: [], 1: []}
                for on in (0, 1, 1, 0):                                      # A B B A: drift cancels
                    set_moe(on)
                    mx.eval(f(*ins[0]))
                    cur[on].append(timeit(f, ins))
                for on in (0, 1):
                    best[on].append(min(cur[on]))
            for label, on in (("today        ", 0), ("moe_sgred=1  ", 1)):
                table[("moe", mode, on)] = best[on]
                print(label + "".join(f"{v:9.0f}" for v in best[on]))
            set_moe(0)
            t0, t1 = table[("moe", mode, 0)], table[("moe", mode, 1)]
            if len(rows_l) > 1:
                s0 = (t0[-1] - t0[0]) / max(1, rows_l[-1] - rows_l[0]); s1 = (t1[-1] - t1[0]) / max(1, rows_l[-1] - rows_l[0])
                print(f"per extra row: today {s0:.1f} us  moe_sgred {s1:.1f} us  (x{a.layers} layers: {s0*a.layers/1000:.2f} -> {s1*a.layers/1000:.2f} ms, MoE layer only)")
        for name, label in (("wo_b", "wo_b single"), ("s5", "wq_a+wkv group"), ("sh_down", "sh_down single"), ("wo_a", "wo_a 8 groups")):
            print(f"\n{label:<16} rows:" + "".join(f"{r:>9}" for r in rows_l))
            res = {}
            cur = {0: {r: [] for r in rows_l}, 6: {r: [] for r in rows_l}}
            for r in rows_l:
                for fr in (0, 6, 6, 0):                                      # A B B A
                    set_fr(fr)
                    f, ins = call_dense(name, r)
                    o = f(*ins[0]); mx.eval(*(o if isinstance(o, (list, tuple)) else [o]))
                    cur[fr][r].append(timeit(f, ins))
            for lab, fr in (("today        ", 0), ("fuse_rows=6  ", 6)):
                res[fr] = [min(cur[fr][r]) for r in rows_l]
                print(lab + "".join(f"{v:9.0f}" for v in res[fr]))
            set_fr(0)
            if len(rows_l) > 1:
                s0 = (res[0][-1] - res[0][0]) / max(1, rows_l[-1] - rows_l[0]); s1 = (res[6][-1] - res[6][0]) / max(1, rows_l[-1] - rows_l[0])
                print(f"per extra row: today {s0:.1f} us  fuse_rows {s1:.1f} us")

    if not a.no_num:
        print("\n== numerics against the f32 reference (synthetic head %d x %d; top-1 = rows whose argmax logit equals the reference's) ==" % (HIDDEN, a.vocab))
        head = (np.random.default_rng(1).standard_normal((HIDDEN, a.vocab)) / np.sqrt(HIDDEN)).astype(np.float32)
        for mode in a.route.split(","):
            for r in [int(v) for v in a.num_rows.split(",")]:
                ids, w = route(r, a.experts, mode, 100 + r)
                x = xin(r, HIDDEN, 10)
                ref = moe_ref(np.array(x), ids, w, HIDDEN, INTER, limit=moe.limit, wpre=True)
                for lab, on in (("today", 0), ("moe_sgred", 1)):
                    set_moe(on)
                    y = np.array(moe(x, mx.array(ids), mx.array(w)))
                    ry, rl, top, n = head_metrics(y, ref, head)
                    print(f"MoE {mode:<8} rows {r} {lab:<10} rel out {ry:.3e}  rel logits {rl:.3e}  top-1 {top}/{n}")
                set_moe(0)
        dh = {"wo_b": (wo_b, wo_b_np, 8192), "sh_down": (sh_down, sh_down_np, 2304)}
        for r in [int(v) for v in a.num_rows.split(",")]:
            for name, (m, (tr, su, sv), kin) in dh.items():
                x = xin(r, kin, 60 + r)
                ref = R.gemv(np.array(x).astype(np.float32), tr, su, sv, dtype=np.float32)
                hd = (np.random.default_rng(2).standard_normal((ref.shape[1], a.vocab)) / np.sqrt(ref.shape[1])).astype(np.float32)
                for lab, fr in (("today", 0), ("fuse_rows", 6)):
                    set_fr(fr)
                    y = np.array(m(x))
                    ry, rl, top, n = head_metrics(y, ref, hd)
                    print(f"dense {name:<8} rows {r} {lab:<10} rel out {ry:.3e}  rel logits {rl:.3e}  top-1 {top}/{n}")
                set_fr(0)
            x = xin(r, 5120, 70 + r)
            for lab, fr in (("today", 0), ("fuse_rows", 6)):
                set_fr(fr)
                f, _ = call_dense("s5", r)
                outs = f(x)
                for nm, (tr, su, sv), yo in (("wq_a", wq_a_np, outs[0]), ("wkv", wkv_np, outs[1])):
                    ref = R.gemv(np.array(x).astype(np.float32), tr, su, sv, dtype=np.float32)
                    hd = (np.random.default_rng(3).standard_normal((ref.shape[1], a.vocab)) / np.sqrt(ref.shape[1])).astype(np.float32)
                    ry, rl, top, n = head_metrics(np.array(yo), ref, hd)
                    print(f"dense s5:{nm:<5} rows {r} {lab:<10} rel out {ry:.3e}  rel logits {rl:.3e}  top-1 {top}/{n}")
            set_fr(0)
    print("\nflags restored: SGRED", M.SGRED, "FUSE_ROWS", KN.FUSE_ROWS)


if __name__ == "__main__":
    main()
