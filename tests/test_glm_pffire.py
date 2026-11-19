"""BP-PFFIRE: the prompt mHC mix on the served fp16 weight, and the prompt indexer selection (pf_idx) against the stock chain. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import fired as FD, glm_pf_glue as G, glm_pf_idx as PI   # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _stock_mix(x, fn, eps):
    y = x.astype(mx.float32)
    z = mx.fast.rms_norm(y, None, eps)                       # the stock HyperConnection: rms_norm of the fp32 streams, then z @ fn.T
    return z @ fn.T


@pytest.mark.parametrize("S,K", [(16, 4096), (100, 4096), (1031, 16384)])
def test_mix_fp16_fn_vs_stock(S, K):
    rng = np.random.default_rng(S + K)
    x = mx.array((rng.standard_normal((S, K)) * 3).astype(np.float16))
    fn16 = mx.array((rng.standard_normal((24, K)) * 0.02).astype(np.float16))
    assert G.mix_why(x, fn16) is None                         # the served loader's dtype is accepted now
    got = G.mix(x, fn16, 1e-6)
    ref = _stock_mix(x, fn16, 1e-6)
    mx.eval(got, ref)
    g, r = np.array(got, dtype=np.float64), np.array(ref, dtype=np.float64)
    xn, fnn = np.array(x).astype(np.float64), np.array(fn16).astype(np.float64)
    f64 = (xn / np.sqrt((xn ** 2).mean(1, keepdims=True) + 1e-6)) @ fnn.T
    rel = lambda a, b: np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean())
    print(f"S={S} K={K}: kernel vs f64 {rel(g, f64):.3e}, stock vs f64 {rel(r, f64):.3e}, kernel vs stock {rel(g, r):.3e}")
    assert rel(g, f64) < 1e-5                                   # the kernel is fp32-class (fp16 weights are exact in fp32)
    assert rel(g, f64) <= rel(r, f64) * 1.01 + 1e-7             # and no worse than the stock chain (whatever precision MLX's fp32 matmul really has on this GPU)


def test_mix_fp32_fn_unchanged():
    rng = np.random.default_rng(5)
    x = mx.array((rng.standard_normal((64, 4096)) * 3).astype(np.float16))
    fn = mx.array((rng.standard_normal((24, 4096)) * 0.02).astype(np.float32))
    assert G.mix_why(x, fn) is None
    assert G.mix_why(x, fn.astype(mx.bfloat16)) == "fn_dtype=bfloat16"
    got, ref = G.mix(x, fn, 1e-6), _stock_mix(x, fn, 1e-6)
    mx.eval(got, ref)
    g, r = np.array(got, dtype=np.float64), np.array(ref, dtype=np.float64)
    xn, fnn = np.array(x).astype(np.float64), np.array(fn).astype(np.float64)
    f64 = (xn / np.sqrt((xn ** 2).mean(1, keepdims=True) + 1e-6)) @ fnn.T
    assert np.sqrt(((g - f64) ** 2).mean() / (f64 ** 2).mean()) < 1e-5


def _stock_select():
    from mlx_vlm.models.glm5_next import language as L
    return L._exact_pool_select


@pytest.mark.parametrize("T,pc,k", [(128, 700, 512), (300, 2100, 512)])
def test_pf_idx_set_vs_stock(T, pc, k, monkeypatch):
    monkeypatch.setattr(PI, "ENABLED", True)
    rng = np.random.default_rng(T + pc)
    q = mx.array(rng.standard_normal((1, T, 32, 128)).astype(np.float16))
    keys = mx.array(rng.standard_normal((1, pc, 128)).astype(np.float16))
    w = mx.array(np.abs(rng.standard_normal((1, T, 32))).astype(np.float32))
    kv = pc * 4 + T
    pool_ends = mx.arange(pc, dtype=mx.int32) * 4 + 3
    pool_valid = mx.ones((1, pc), dtype=mx.bool_)
    qpos = (kv - T) + mx.arange(T)
    assert PI.why_not(q, keys, w, k) is None
    st, sv = _stock_select()(q, keys, w, pool_ends[None], pool_valid, qpos, k, 0.0883883, 512)
    got, gv = PI.select(q, keys, w, pool_ends[None], pool_valid, qpos, k, 0.0883883)
    mx.eval(st, sv, got, gv)
    st, sv, got, gv = np.array(st)[0], np.array(sv)[0], np.array(got)[0], np.array(gv)[0]
    same = 0
    for t in range(T):
        a, b = set(st[t][sv[t]].tolist()), set(got[t][gv[t]].tolist())
        same += a == b
        assert len(a ^ b) <= 2                                  # fp32 accumulation order can only flip pools that tie within ~1e-6
    assert same >= int(0.98 * T)


def test_pf_idx_q_none_is_the_few_pool_case():
    """q is None exactly when select_k == pool_count (every pool selected, nothing to score): the stock chain returns arange, no kernel is needed."""
    k = mx.zeros((1, 100, 128), dtype=mx.float16)
    assert PI.why_not(None, k, None, 100) in ("flag_off", "q_none")
