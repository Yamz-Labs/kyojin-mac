"""Non-MoE time of a DS4.1 layer at prefill, stock attention path vs the dsapf path, one process, interleaved, routed experts stubbed out.
  python tools/dsapf_report.py PACK T1,T2,... [reps=3] [--skip-stock-above N]
Layer 0 of the pack is a full layer (compressor + indexer), layer 1 a reuse layer; the printed per-layer numbers feed the whole-model estimate
40 * (MoE layer + non-MoE layer) (MoE layer from tools/prof_moe_pf.py)."""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, ds_pf_attn as A, ds_pf_idx as I
pack = sys.argv[1]; Ts = [int(t) for t in sys.argv[2].split(",")]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
skip_above = int(sys.argv[sys.argv.index("--skip-stock-above") + 1]) if "--skip-stock-above" in sys.argv else 1 << 30
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model
class Stub:                                    # routed experts out: what is left is attention, hyper-connections, router, shared expert
    def __call__(self, x, inds, weights=None, *a, **k):
        return mx.zeros(x.shape[:-1] + (x.shape[-1],), dtype=x.dtype) if inds.ndim == x.ndim else mx.zeros_like(x)
for layer in lm.layers:
    f = layer.ffn
    if hasattr(f, "switch_mlp"):
        f.switch_mlp = Stub()
times = {}
def layer_hooks():
    for i, layer in enumerate(lm.layers):
        cls = layer.__class__
        def mk(i, cls):
            def __call__(self, *a, **k):
                t0 = time.perf_counter(); out = cls.__call__(self, *a, **k); mx.eval(out[0]); times[i] = time.perf_counter() - t0; return out
            return __call__
        layer.__class__ = type(cls.__name__, (cls,), {"__call__": mk(i, cls)})
layer_hooks()
for T in Ts:
    ids = mx.array([np.random.default_rng(0).integers(100, 20000, T).tolist()])
    res = {}
    for mode in ("stock", "new"):
        if mode == "stock" and T > skip_above:
            continue
        res[mode] = []
    for r in range(reps + 1):
        for mode in res:
            A.ENABLED = I.ENABLED = (mode == "new")
            t0 = time.perf_counter(); mx.eval(lm(ids, cache=lm.make_cache()).logits); tot = time.perf_counter() - t0
            if r: res[mode].append((tot, dict(times)))
    for mode, v in res.items():
        b = min(v, key=lambda x: x[0])
        per = {i: min(x[1][i] for x in v) for i in b[1]}
        print(f"T={T:5d} {mode:5s} 2-layer forward {1e3 * b[0]:8.1f} ms   layer0 (full) {1e3 * per[0]:7.1f}   layer1 (reuse) {1e3 * per[1]:7.1f}   -> {1e3 * per[1] / T * 1000:.1f} us/token reuse layer")
