#!/usr/bin/env python3
"""DS4.1 per-op table on one RAW layer of the HF pack (yamz-labs/DeepSeek-V4.1-Flash-EXL3-Yamz, one tensor per expert), loaded through
kyojin_mlx.patch.load_pack (load-time restack, no conversion). Real shapes, all 384 (trunk) / 128 (mtp) experts in memory.
  decode  : routed MoE (MoeGather, 1 row, topk 6) | shared expert | attention linears (wq_a, wq_b, wkv, wo_b, indexer wq_b) | grouped wo_a (8 slices)
  prefill : routed MoE through MoePrefill (fused trellis decode + NAX when available) at --T tokens, uniform routing (every expert active)
Each op: time per call = slope of chains of --chain and 4x --chain calls with fresh inputs in one eval ((t(4n) - t(n)) / 3n, min of --reps): the fixed
eval / sync cost of one chain is not charged to the op. Dense linears rotate over --pool weight copies (one decode step reads each weight once, not from the
SLC). ms_chain = the old per-op method (one chain of --chain calls / chain, same weights), kept for comparison. Bytes read, byte roof at --bw GB/s.
Exactness: --ref DIR (a converted stacked pack of the same layer, e.g. models/ds_l10) -> MoeGather / MoePrefill outputs of the restacked raw
layer vs the stacked layer must be bit-identical; MoeGather vs MoeUnfused (per-expert reference GEMVs) and MoePrefill vs MoeGather: max rel error.
usage: ds41_raw_slice.py RAW_DIR [--layer 10 | --mtp 0] [--ref DIR] [--T 512,2048] [--bw 131.9] [--json OUT]
Fetch a layer: python -c "from huggingface_hub import hf_hub_download as d; d('yamz-labs/DeepSeek-V4.1-Flash-EXL3-Yamz','layers.10.safetensors',local_dir='RAW_DIR')"."""
import argparse, json, os, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import patch as P, moe as M, moe_prefill as MP, metal_nax as NX  # noqa: E402
from kyojin_mlx.exl3_linear import Exl3Linear  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("raw")
ap.add_argument("--layer", type=int, default=10)
ap.add_argument("--mtp", type=int, default=None, help="an mtp.<N> stage instead of a trunk layer")
ap.add_argument("--ref", default=None)
ap.add_argument("--T", default="512,2048")
ap.add_argument("--bw", type=float, default=131.9, help="byte roof GB/s (Air M5 probe, ROOF96.md)")
ap.add_argument("--chain", type=int, default=8)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--pool", type=int, default=8, help="weight copies per dense decode linear")
ap.add_argument("--json", default="")
a = ap.parse_args()
mx.set_cache_limit(256 << 20)
pre = f"mtp.{a.mtp}" if a.mtp is not None else f"layers.{a.layer}"
TOPK = 3 if a.mtp is not None else 6
rng = np.random.default_rng(0)


def load(d):
    t0 = time.time()
    t = P.load_pack(d, prefix=pre + ".")
    return t, time.time() - t0


def stacks(t):
    st = {r: M.Stack.from_arrays(t, f"{pre}.ffn.experts.{w}", "mul1") for r, w in (("gate", "w1"), ("up", "w3"), ("down", "w2"))}
    mx.eval(*[x for s in st.values() for x in (s.words, s.offset, s.k, s.suh, s.svh)])
    return st


def gather(st):
    return M.MoeGather(st["gate"], st["up"], st["down"], TOPK, limit=10.0, wpre=True, fq=True)


def timeit(fns, inputs):
    """slope and old-method per-call time in ms: fns = a pool of callables (call i uses fns[i % len]), inputs = --chain input tuples"""
    n = len(inputs)
    def run(m, pool):
        outs = [(fns[i % len(fns)] if pool else fns[0])(*inputs[i % n]) for i in range(m)]
        t0 = time.perf_counter(); mx.eval(*outs); return time.perf_counter() - t0
    T = {v: [] for v in ((n, True), (4 * n, True), (n, False))}
    for r in range(a.reps + 1):
        for v in T:
            dt = run(*v)
            if r:
                T[v].append(dt)
    slope = (min(T[(4 * n, True)]) - min(T[(n, True)])) / (3 * n)
    return slope * 1e3, min(T[(n, False)]) / n * 1e3


def expert_bytes(st, e):
    return sum(int(np.array(s.offset[e + 1]) - np.array(s.offset[e])) * 4 + s.suh.shape[1] * 2 + s.svh.shape[1] * 2 for s in st.values())


rows, res = [], dict(pack=a.raw, prefix=pre, bw=a.bw)
t, lt = load(a.raw)
st = stacks(t)
E, H = st["gate"].experts, st["gate"].suh.shape[1]
res["load_s"], res["experts"], res["hidden"] = round(lt, 2), E, H
eb = np.array([expert_bytes(st, e) for e in range(E)])
moe = gather(st)


def row(op, ms, nbytes, note=""):
    ms, ms_chain = ms if isinstance(ms, tuple) else (ms, None)
    roof = nbytes / (a.bw * 1e9) * 1e3
    rows.append(dict(op=op, ms=round(ms, 4), ms_chain=None if ms_chain is None else round(ms_chain, 4), mb=round(nbytes / 1e6, 2), roof_ms=round(roof, 4), pct_roof=round(100 * roof / ms, 1), note=note))


# ---- decode: routed MoE, 1 row
dec_in = []
for _ in range(a.chain):
    ids = rng.choice(E, TOPK, replace=False)
    dec_in.append((mx.array(rng.standard_normal((1, H)).astype(np.float16)), mx.array(ids[None].astype(np.int32)),
                   mx.array(rng.random((1, TOPK)).astype(np.float32))))
mx.eval(*[x for i in dec_in for x in i])
ms = timeit([lambda x, i, w: moe(x, i, w)], dec_in)
row("moe_decode_routed", ms, float(np.mean([eb[np.array(i[1]).ravel()].sum() for i in dec_in])), f"MoeGather 1 row, topk {TOPK}, real per-expert bits")

# ---- decode: dense linears of the layer (1 row)
for name in ("ffn.shared_experts.w1", "ffn.shared_experts.w3", "ffn.shared_experts.w2", "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "attn.indexer.wq_b", "main_proj"):
    p = f"{pre}.{name}"
    if f"{p}.trellis" not in t:
        continue
    lins = [Exl3Linear(t[f"{p}.trellis"] + 0 * c, t[f"{p}.suh"], t[f"{p}.svh"], cb="mul1") for c in range(a.pool)]
    lin = lins[0]
    mx.eval([l.parameters() for l in lins])
    xs = [(mx.array(rng.standard_normal((1, lin.in_features)).astype(np.float16)),) for _ in range(a.chain)]
    nb = t[f"{p}.trellis"].nbytes + t[f"{p}.suh"].nbytes + t[f"{p}.svh"].nbytes
    row(f"lin_{name}", timeit(lins, xs), nb, f"{lin.out_features}x{lin.in_features} K={t[f'{p}.trellis'].shape[2] // 16}")
p = f"{pre}.attn.wo_a.slices"
if f"{p}.trellis" in t:
    mls = [P.Exl3MultiLinear(t[f"{p}.trellis"] + 0 * c, t[f"{p}.suh"], t[f"{p}.svh"], "mul1", fq_in=True) for c in range(a.pool)]
    ml = mls[0]
    mx.eval([m.parameters() for m in mls])
    xs = [(mx.array(rng.standard_normal((1, ml.num_groups, 1, ml.in_features)).astype(np.float16)),) for _ in range(a.chain)]
    nb = sum(t[f"{p}.{k}"].nbytes for k in ("trellis", "suh", "svh"))
    row("wo_a_grouped", timeit(mls, xs), nb, f"{ml.num_groups} groups {ml.out_features}x{ml.in_features}, fp8 input fold")

# ---- prefill: routed MoE over T tokens (all experts active)
pf = MP.MoePrefill(moe)
res["nax"] = bool(NX.nax_available())
pf_in = {}
for T in [int(x) for x in a.T.split(",") if x]:
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(T)]).astype(np.int32)
    x = mx.array((rng.standard_normal((T, H)) * 0.5).astype(np.float16)); i = mx.array(ids); w = mx.array(rng.random((T, TOPK)).astype(np.float32))
    mx.eval(x, i, w)
    pf_in[T] = (x, i, w)
    best = 1e9
    for r in range(4):
        t0 = time.perf_counter(); mx.eval(pf(x, i, w)); dt = time.perf_counter() - t0
        if r:
            best = min(best, dt)
    act = len(np.unique(ids))
    row(f"moe_prefill_T{T}", best * 1e3, float(eb[np.unique(ids)].sum()), f"MoePrefill fused={pf.fused}, {act} experts active, {T / best:.0f} tok/s per layer")

# ---- exactness
ex = {}
x, i, w = dec_in[0]
y_g = moe(x, i, w)
y_u = M.MoeUnfused(st["gate"], st["up"], st["down"], TOPK, limit=10.0, wpre=True, fq=True)(x, i, w)
mx.eval(y_g, y_u)
ex["gather_vs_unfused_maxrel"] = float(mx.max(mx.abs(y_g - y_u)) / (mx.max(mx.abs(y_u)) + 1e-30))
T0 = min(pf_in)
xp, ip, wp = (v[:16] for v in pf_in[T0])
y_pf, y_gr = pf(xp, ip, wp), mx.concatenate([moe(xp[r:r + 1], ip[r:r + 1], wp[r:r + 1]) for r in range(16)])
mx.eval(y_pf, y_gr)
ex["prefill_vs_gather_maxrel_16rows"] = float(mx.max(mx.abs(y_pf - y_gr)) / (mx.max(mx.abs(y_gr)) + 1e-30))
if a.ref:
    tr, _ = load(a.ref)
    sr = stacks(tr)
    moe_r = gather(sr)
    y_r = moe_r(x, i, w)
    xb, ib, wb = pf_in[T0]
    y_pfr, y_pfa = MP.MoePrefill(moe_r)(xb, ib, wb), pf(xb, ib, wb)
    mx.eval(y_r, y_pfr, y_pfa)
    ex["decode_raw_vs_stacked_bitident"] = bool(mx.array_equal(y_g, y_r).item())
    ex[f"prefill_T{T0}_raw_vs_stacked_bitident"] = bool(mx.array_equal(y_pfa, y_pfr).item())
res["exact"], res["ops"], res["peak_gib"] = ex, rows, round(mx.get_peak_memory() / 2 ** 30, 2)
print(f"{pre}: {E} experts, hidden {H}, load {res['load_s']} s, nax {res['nax']}, peak {res['peak_gib']} GiB, roof {a.bw} GB/s")
print(f"{'op':34s} {'ms':>9s} {'MB':>8s} {'roof ms':>9s} {'% roof':>7s} {'ms_chain':>9s}  note")
for r in rows:
    print(f"{r['op']:34s} {r['ms']:9.4f} {r['mb']:8.2f} {r['roof_ms']:9.4f} {r['pct_roof']:7.1f} {'' if r.get('ms_chain') is None else format(r['ms_chain'], '9.4f'):>9s}  {r['note']}")
print("exact:", json.dumps(ex))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
