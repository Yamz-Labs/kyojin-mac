"""Hy4 decode micro (no model load): the attention core and the router of one decode step (43 / 42 layers chained), stock vs fused, interleaved ABAB.
  attend: 12 non-windowed kpool layers (nh 32, row width 832, keep mask) + 31 windowed layers (nh 64, window 512, row width 576); the stock arm is
          MlaAttention.attend with the mask _call_fixed builds, the fused arm is mla_attend_fused (both after the same k_b absorb, before the same v_b absorb).
  route:  42 MoE routers (3072 -> 512, K 16, sigmoid, bias, norm, scale 3.998): GenericMoE.route with the flag off / on.
Prints 'MICRO <what> <T> <ctx> <arm>: <ms per step median> (min .. max)' and 'MICRO diff ...' (max abs difference of the outputs)."""
import os, statistics, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.getcwd())
from kyojin_mlx import foreign_skeleton_mla as MLA, foreign_moe_generic as G, mla_attend_fused as MAF

KV, RD, NOPE, V = 512, 64, 192, 256


def dep(x, y):
    r = mx.depends(x, [y])
    return r[0] if isinstance(r, (list, tuple)) else r

rng = np.random.default_rng(0)
REPS = int(os.environ.get("N7_REPS", "7"))


def layer_types():
    return [("kp", 32, 0, 832)] * 12 + [("swa", 64, 512, 576)] * 31


def make_layers():
    out = []
    for kind, nh, win, D in layer_types():
        a = MLA.MlaAttention(3072, nh, 1024, KV, NOPE, RD, V, sinks=True)
        a.k_b = mx.array((rng.standard_normal((nh, KV, NOPE)) * 0.05).astype(np.float16))
        a.v_b = mx.array((rng.standard_normal((nh, V, KV)) * 0.05).astype(np.float16))
        a.sinks = mx.array(rng.standard_normal((nh,)).astype(np.float32))
        out.append((a, kind, nh, win, D))
    mx.eval([x[0].k_b for x in out] + [x[0].v_b for x in out])
    return out


def run_attend(layers, T, C, off, bufs, qs, fused):
    o = None
    for (a, kind, nh, win, D), buf, (qn, qp), keep in zip(layers, bufs, qs, KEEPS[(T, C, off)]):
        if o is not None:
            qn = dep(qn, o)
        offa = mx.array([off], dtype=mx.int32)
        if fused:
            qa = MLA._amm(qn, a.k_b)
            o = MLA._amm(MAF.attend(qa, qp, buf, offa, keep, a.sinks, a.scale, win, KV, RD), a.v_b)
        else:
            Sw = C if kind == "kp" else min(win + T - 1, C)
            base = mx.zeros((1,), dtype=mx.int32) if kind == "kp" else mx.maximum(offa + T - Sw, 0)
            kv = buf if kind == "kp" else mx.slice(buf, base, (2,), (1, 1, Sw, D))
            kpos = base + mx.arange(Sw)
            qpos = offa.reshape(-1) + mx.arange(T)
            vis = kpos[None, :] <= qpos[:, None]
            if win:
                vis = vis & (qpos[:, None] - kpos[None, :] < win)
            o = a.attend(qn, qp, kv[..., :KV], kv[..., KV:KV + RD], vis[None, None], keep if kind == "kp" else None, slice_ok=False)
    return o


KEEPS = {}


def attend_micro(T, C, off):
    layers = make_layers()
    bufs, qs, keeps = [], [], []
    for (a, kind, nh, win, D) in layers:
        b = np.zeros((1, 1, C, D), np.float16)
        b[0, 0, :off + T] = (rng.standard_normal((off + T, D)) * 0.4).astype(np.float16)
        bufs.append(mx.array(b))
        qs.append((mx.array((rng.standard_normal((1, nh, T, NOPE)) * 0.3).astype(np.float16)), mx.array((rng.standard_normal((1, nh, T, RD)) * 0.5).astype(np.float16))))
        if kind == "kp":
            k = rng.random((1, 1, T, C)) < 0.5
            k[..., :1] = True
            keeps.append(mx.array(k))
        else:
            keeps.append(None)
    KEEPS[(T, C, off)] = keeps
    mx.eval(bufs + [q for p in qs for q in p] + [k for k in keeps if k is not None])
    res = {}
    for arm in ("stock", "fused", "stock", "fused"):
        ts = []
        for r in range(REPS + 1):
            t = time.perf_counter()
            o = run_attend(layers, T, C, off, bufs, qs, arm == "fused")
            mx.eval(o)
            if r:
                ts.append((time.perf_counter() - t) * 1000)
        res.setdefault(arm, []).append(statistics.median(ts))
        res.setdefault(arm + "_all", []).extend(ts)
    for arm in ("stock", "fused"):
        v = res[arm + "_all"]
        print(f"MICRO attend T={T} ctx={off} C={C} {arm}: {statistics.median(v):.3f} ms/step (min {min(v):.3f} max {max(v):.3f}; two blocks {res[arm][0]:.3f} {res[arm][1]:.3f})", flush=True)
    a, b = run_attend(layers, T, C, off, bufs, qs, False), run_attend(layers, T, C, off, bufs, qs, True)
    mx.eval(a, b)
    print(f"MICRO attend T={T} ctx={off} diff last-layer out: max abs {float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max()):.5f} (max |stock| {float(mx.abs(a).max()):.4f})", flush=True)


def route_micro(T):
    spec = G.MoeSpec(n_expert=512, top_k=16, gating="sigmoid", select_bias=True, norm_weights=True, weight_scale=3.998)
    blks = []
    for _ in range(42):
        blk = G.GenericMoE.__new__(G.GenericMoE)
        G.nn.Module.__init__(blk)
        blk.spec_ = spec
        blk.router_w = mx.array((rng.standard_normal((512, 3072)) * 0.02).astype(np.float32))
        blk.router_b = None
        blk.sel_b = mx.array((rng.standard_normal((512,)) * 0.1).astype(np.float32))
        blks.append(blk)
    xs = [mx.array((rng.standard_normal((T, 3072)) * 0.5).astype(np.float16)) for _ in range(42)]
    mx.eval([b.router_w for b in blks] + xs)

    def step(on):
        G.ROUTE_RS = on
        acc = None
        for blk, x in zip(blks, xs):
            if acc is not None:
                x = dep(x, acc)
            i, w = blk.route(x)
            acc = w
        return acc
    res = {False: [], True: []}
    for arm in (False, True, False, True):
        for r in range(REPS + 1):
            t = time.perf_counter()
            mx.eval(step(arm))
            if r:
                res[arm].append((time.perf_counter() - t) * 1000)
    G.ROUTE_RS = False
    for arm, nm in ((False, "stock"), (True, "fused")):
        v = res[arm]
        print(f"MICRO route T={T} {nm}: {statistics.median(v):.3f} ms/step of 42 layers (min {min(v):.3f} max {max(v):.3f})", flush=True)


if __name__ == "__main__":
    for T in (1, 6):
        route_micro(T)
    for T, C, off in ((1, 1024, 830), (6, 1024, 830), (1, 4096, 4000), (6, 4096, 4000)):
        attend_micro(T, C, off)
