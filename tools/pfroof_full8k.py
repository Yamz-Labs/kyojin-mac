#!/usr/bin/env python3
"""pfroof step 2: full 8K-token MoE prefill pass (one DS4.1 layer, 384 virtual
experts, real dims) with skewed routing (mixture: 96 hot experts take 75 pct
of the pairs; ~16 min rows, several hundred max, mean 128). SPLIT_GU on vs
off: bit-identity + timing (interleaved, medians)."""
import os, sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from bench_moe import synth_stack  # noqa: E402
from kyojin_mlx import moe as M, moe_prefill as MP  # noqa: E402

mx.set_cache_limit(256 << 20)
try:
    import mlx_vlm.models.deepseek_v41.fakequant  # noqa
    FQ = True
except ImportError:
    FQ = False
print(f"fq={FQ}", flush=True)

POOL, E, TOPK, H, I, K, T = 48, 384, 6, 5120, 2304, 3, 8192
g0 = synth_stack(H, I, [K] * POOL, "mul1", 1)
u0 = synth_stack(H, I, [K] * POOL, "mul1", 2)
d0 = synth_stack(I, H, [K] * POOL, "mul1", 3)


def virtual(st, V):
    pool = st.experts
    phys = np.arange(V) % pool
    off = np.array(st.offset)[:pool][phys]
    off = np.concatenate([off, off[-1:] + 1]).astype("<u4")
    return type(st)(st.words, mx.array(off), mx.array(np.array(st.k)[phys]),
                    mx.array(np.array(st.suh)[phys]), mx.array(np.array(st.svh)[phys]),
                    st.kt, st.nt, st.cb)


g, u, d = (virtual(s, E) for s in (g0, u0, d0))
dec = M.MoeGather(g, u, d, TOPK, wpre=True, fq=FQ)

# skewed routing: 96 hot experts take 75% of pairs
rng = np.random.default_rng(7)
hot = rng.choice(E, 96, replace=False)
p = np.full(E, 0.25 / (E - 96))
p[hot] = 0.75 / 96
ids = np.stack([rng.choice(E, TOPK, replace=False, p=p) for _ in range(T)]).astype(np.int32)
w = rng.uniform(0.1, 1.0, (T, TOPK)).astype(np.float32)
w = (w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32)
cnt = np.bincount(ids.reshape(-1), minlength=E)
print(f"routing: active {np.count_nonzero(cnt)}/{E} mean {cnt.mean():.0f} "
      f"min {cnt[cnt > 0].min()} max {cnt.max()} p50 {np.median(cnt):.0f} "
      f"top8 {sorted(cnt, reverse=True)[:8]}", flush=True)
x = mx.array((rng.standard_normal((T, H)) * 2).astype(np.float16))
idsm, wm = mx.array(ids), mx.array(w)
mx.eval(x, idsm, wm)

ts = {False: [], True: []}
ys = {}
ratios = []
for r in range(7):
    got = {}
    seq = (False, True) if rng.random() < 0.5 else (True, False)
    for split in seq:
        MP.SPLIT_GU = split
        pf = MP.MoePrefill(dec, chunk=8192)
        y = pf(x, idsm, wm); mx.eval(y)
        mx.synchronize(); t0 = time.perf_counter()
        y = pf(x, idsm, wm); mx.eval(y)
        got[split] = (time.perf_counter() - t0, np.array(y))
    for split in (False, True):
        ts[split].append(got[split][0])
        ys[split] = got[split][1]
    ratios.append(got[True][0] / got[False][0])
fl = T * TOPK * (2 * H * I + 2 * I * H)  # gate+up+down flop per pair
for split in (False, True):
    v = ts[split]
    print(f"split={int(split)}: med {np.median(v) * 1e3:.0f}ms "
          f"min {min(v) * 1e3:.0f}ms pairs/s {T * TOPK / np.median(v) / 1e6:.2f}M "
          f"gate+up+down {fl / np.median(v) / 1e12:.1f}TF", flush=True)
y0, y1 = ys[False], ys[True]
print(f"paired split/comb ratios: {[f'{x:.3f}' for x in ratios]} "
      f"med {np.median(ratios):.3f}", flush=True)
print(f"bit-identical split vs combined: {np.array_equal(y0, y1)} "
      f"rel {np.linalg.norm(y1 - y0) / np.linalg.norm(y0):.2e}", flush=True)
# per-bucket TF projection: gate+up kernel TF at production rows (from bucket table)
print(f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
