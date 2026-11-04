"""mhc_expand / mhc_pre prompt rows == stock bit for bit. usage: test_mhc_pf.py"""
import numpy as np, mlx.core as mx, importlib
from kyojin_mlx import fuse as F
L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
rng = np.random.default_rng(0)
ok = True
for S in (2, 7, 64, 300, 2048):
    D = 5120
    x = mx.array(rng.standard_normal((1, S, D)).astype(np.float16)); res = mx.array(rng.standard_normal((1, S, 4, D)).astype(np.float16))
    post = mx.array(rng.uniform(0, 2, (1, S, 4)).astype(np.float32)); comb = mx.array(rng.uniform(0, 1, (1, S, 4, 4)).astype(np.float32)); pre = mx.array(rng.uniform(0, 1, (1, S, 4)).astype(np.float32))
    stock_e = L.hc_expand(x, res, post, comb); stock_p = L.DeepseekV41Block.hc_pre(res, pre)
    ne = F._impl_mhc_expand(L.hc_expand)(x, res, post, comb); npre = F._impl_mhc_pre(L.DeepseekV41Block.hc_pre)(res, pre)
    mx.eval(stock_e, stock_p, ne, npre)
    e_eq, p_eq = bool(mx.array_equal(stock_e, ne).item()), bool(mx.array_equal(stock_p, npre).item())
    print(f"S={S}: expand bit-equal {e_eq} (max|d| {mx.abs(stock_e.astype(mx.float32)-ne.astype(mx.float32)).max().item():.2e}), pre bit-equal {p_eq}")
    ok &= e_eq and p_eq
print("ALL EQUAL" if ok else "DIFF")
