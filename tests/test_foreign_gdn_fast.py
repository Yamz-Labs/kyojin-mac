"""GPU micro: the fused gated-delta decode kernel on the qwen4_exp GatedDeltaNet (and the stock Qwen3.5 class) vs the module's own __call__, B = 1 / 4, S = 1 / 3, state carry.
Run: PYTHONPATH=. python tests/test_foreign_gdn_fast.py   (needs the Metal GPU; prints one RESULT line per case)"""
import sys
from types import SimpleNamespace as NS

import mlx.core as mx

from kyojin_mlx import foreign_gdn_fast as F


def run(kind, dt, B, S, steps=6, tol=None, dtb32=False):
    from mlx_vlm.models.cache import ArraysCache
    cfg = NS(hidden_size=512, linear_num_value_heads=16, linear_num_key_heads=8, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
             rms_norm_eps=1e-6, output_gate_type="sigmoid", hidden_act="silu")
    if kind == "qw4":
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet as C
    else:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet as C
    mx.random.seed(3)
    m = C(cfg)
    for k, v in list(m.parameters().items()):
        pass
    from mlx.utils import tree_map
    m.update(tree_map(lambda p: (mx.random.normal(p.shape) * 0.05).astype(dt) if p.ndim > 1 else p.astype(dt), m.parameters()))
    m.A_log = mx.log(mx.random.uniform(low=0.5, high=8, shape=(16,)))
    m.dt_bias = (mx.random.normal((16,)) * 0.5).astype(mx.float32 if dtb32 else dt)      # dtb32: GGUF stores ssm_dt f32
    m.norm.weight = (1 + 0.1 * mx.random.normal((128,))).astype(dt)
    mx.eval(m.parameters())
    m.eval()
    mode = F.qualifies(m)
    F.STATS["fast"] = 0
    stock = C.__call__
    m.__dict__["_kfg_mode"] = mode
    C.__call__ = F._wrap(stock)
    try:
        c1, c2 = ArraysCache(2), ArraysCache(2)
        worst, eq = 0.0, True
        for i in range(steps):
            x = (mx.random.normal((B, S, 512)) * 2).astype(dt)
            F.ACTIVE = False
            y1 = m(x, None, c1)
            F.ACTIVE = True
            y2 = m(x, None, c2)
            mx.eval(y1, y2, c1[0], c1[1], c2[0], c2[1])
            for a, b in ((y1, y2), (c1[0], c2[0]), (c1[1], c2[1])):
                d = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
                worst = max(worst, d)
                eq &= bool(mx.array_equal(a, b).item())
        print(f"RESULT {kind} {dt} dtb32={dtb32} B={B} S={S} mode={mode} fast={F.STATS['fast']} bit_equal={eq} max_abs_diff={worst:.3g}")
        return mode is not None and F.STATS["fast"] == steps and (eq or worst < (tol or 0.1))
    finally:
        C.__call__ = stock


if __name__ == "__main__":
    ok = True
    for kind in ("qw4", "stock"):
        for dt in (mx.bfloat16, mx.float16):
            for B, S in ((1, 1), (4, 1), (1, 3)):
                ok &= run(kind, dt, B, S)
            ok &= run(kind, dt, 1, 1, dtb32=True)
            ok &= run(kind, dt, 4, 1, dtb32=True)
    print("ALL OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
