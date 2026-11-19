"""G5d GLM-5.3 KDA prompt path: chunk algebra (numpy), the NAX scan and the fused prework / norm-gate kernels against float64 references, flag defaults, install no-op. Metal tests need a Mac."""
import os, sys
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from kyojin_mlx import flags, kda_pf   # noqa: E402
from kda_chunk_ref import seq_ref, chunk_ref   # noqa: E402


def test_flag_registered_off_by_default():
    assert "glm_kda_pf" in flags.FLAGS and flags.FLAGS["glm_kda_pf"].default == "0"


def test_install_is_noop_when_off_or_other_model(monkeypatch):
    class M:
        language_model = None

    assert kda_pf.install(M()) == 0


@pytest.mark.parametrize("T", [16, 37, 100])
def test_chunk_algebra_f64(T):
    rng = np.random.default_rng(T)
    D = 128
    q = rng.standard_normal((T, D)) / 11.3
    k = rng.standard_normal((T, D)); k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((T, D)); beta = 1 / (1 + np.exp(-rng.standard_normal(T)))
    lg = -5.0 / (1 + np.exp(-rng.standard_normal((T, D)) * 3))
    S0 = rng.standard_normal((D, D)) * 0.1
    y0, S1 = seq_ref(q, k, v, lg, beta, S0)
    y, S = chunk_ref(q, k, v, lg, beta, S0, 16, np.float64)
    assert np.abs(y - y0).max() < 1e-10 * np.abs(y0).max() and np.abs(S - S1).max() < 1e-10 * np.abs(S1).max()


def _rel(x, r):
    x = np.asarray(x, np.float64); r = np.asarray(r, np.float64)
    return float(np.sqrt(((x - r) ** 2).mean() / (r ** 2).mean()))


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("T,mode", [(16, "real"), (69, "real"), (300, "worst"), (131, "slow")])
def test_kda_nax_vs_f64(T, mode):
    from kyojin_mlx.kda_nax import kda_nax
    H, D, bf = 2, 128, mx.bfloat16
    mx.random.seed(T)
    q = (mx.random.normal((1, T, H, D)) * 0.01).astype(bf); k = (mx.random.normal((1, T, H, D)) * 0.09).astype(bf); v = mx.random.normal((1, T, H, D)).astype(bf)
    a = (mx.random.normal((1, T, H, D)) * 2).astype(bf); b = mx.random.normal((1, T, H)).astype(bf)
    A_log = mx.log(mx.random.uniform(1, 16, (H,))); dtb = mx.random.normal((H, D)) * 0.5
    if mode == "worst":
        a = mx.full((1, T, H, D), 30.0).astype(bf); dtb = mx.zeros((H, D)); A_log = mx.zeros((H,))
    elif mode == "slow":
        a = mx.full((1, T, H, D), -30.0).astype(bf)
    S0 = mx.random.normal((1, H, D, D)) * 0.05
    beta = mx.sigmoid(b)
    y, S = kda_nax(q, k, v, a, beta, A_log, dtb, -5.0, S0)
    mx.eval(y, S)
    f = lambda t: np.array(t.astype(mx.float32)).astype(np.float64)
    for h in range(H):
        x = np.exp(f(A_log)[h]) * (f(a)[0, :, h] + f(dtb)[h])
        lg = -5.0 / (1 + np.exp(-x))
        y64, S64 = seq_ref(f(q)[0, :, h], f(k)[0, :, h], f(v)[0, :, h], lg, f(beta)[0, :, h], f(S0)[0, h])
        assert not np.isnan(f(y)).any()
        assert _rel(f(y)[0, :, h], y64) < 8e-3 and _rel(f(S)[0, h], S64) < 5e-5      # G5e: 4e-6 with the hi/lo split MMAs (1e-3 without)


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("S", [64, 100, 257])
def test_prework_and_normgate_vs_f64(S):
    H, D = 4, 128
    C = 3 * H * D
    rng = np.random.default_rng(S)
    qkv = mx.array((rng.standard_normal((1, S, C)) * 2).astype(np.float32)).astype(mx.float16)
    cs = mx.array((rng.standard_normal((1, 3, C)) * 2).astype(np.float32)).astype(mx.float16)
    w = mx.array((rng.standard_normal((C, 4, 1)) * 0.5).astype(np.float32)).astype(mx.float16)
    q, k, v, nc = kda_pf.prework(qkv, cs, w, H, D)
    mx.eval(q, k, v, nc)
    x = np.concatenate([np.array(cs.astype(mx.float32))[0], np.array(qkv.astype(mx.float32))[0]], 0).astype(np.float64)
    wn = np.array(w.astype(mx.float32))[:, :, 0].astype(np.float64)
    conv = sum(x[t:t + S] * wn[:, t] for t in range(4))
    act = conv / (1 + np.exp(-conv))
    a4 = act.reshape(S, 3 * H, D)
    l2 = lambda z: z / np.sqrt((z ** 2).sum(-1, keepdims=True) + 1e-6)
    assert _rel(np.array(q.astype(mx.float32))[0], l2(a4[:, :H]) * D ** -0.5) < 2e-3
    assert _rel(np.array(k.astype(mx.float32))[0], l2(a4[:, H:2 * H])) < 2e-3
    assert _rel(np.array(v.astype(mx.float32))[0], a4[:, 2 * H:]) < 2e-3
    assert np.array_equal(np.array(nc.astype(mx.float32)), np.array(qkv.astype(mx.float32))[:, S - 3:])
    y = mx.array(rng.standard_normal((1, S, H, D)).astype(np.float32)).astype(mx.float16)
    g = mx.array(rng.standard_normal((1, S, H, D)).astype(np.float32)).astype(mx.float16)
    nw = mx.array((1 + 0.1 * rng.standard_normal(D)).astype(np.float32)).astype(mx.float16)
    out = kda_pf.norm_gate(y, g, nw, 1e-6, H, D)
    yn, gn, nwn = (np.array(t.astype(mx.float32)).astype(np.float64) for t in (y, g, nw))
    ref = yn / np.sqrt((yn ** 2).mean(-1, keepdims=True) + 1e-6) * nwn / (1 + np.exp(-gn))
    mx.eval(out)
    assert _rel(np.array(out.astype(mx.float32))[0], ref.reshape(S, -1)[...]) < 3e-3


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_nax_mma_truncates_fp32_operands():
    """The root cause of G5e: the NAX unit keeps 10 mantissa bits of an fp32 operand (truncation toward zero), the accumulator is exact; hi / lo recovers fp32."""
    from kyojin_mlx import gdn_nax as GN
    src = """
  auto _n = thread_position_in_grid.z;
  mlx::steel::NAXTile<float, 1, 1> A_t, B_t, C_t;
  A_t.load(a, 16); B_t.load(b, 16);
  MM16x16x16(C_t, 0, A_t, false, 0, B_t, false, 0);
  C_t.store(c, 16);
"""
    k = mx.fast.metal_kernel(name="g5e_mma_trunc", input_names=["a", "b"], output_names=["c"], header=GN.header() + "\n", source=GN._norefs(src))
    A = np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32)
    I = np.eye(16, dtype=np.float32)
    run = lambda x: np.array(k(inputs=[mx.array(x), mx.array(I)], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(16, 16)], output_dtypes=[mx.float32])[0])
    hi = (A.view(np.uint32) & 0xffffe000).view(np.float32)
    assert np.array_equal(run(A), hi)                                  # truncated, not rounded
    assert np.array_equal(run(hi), hi)                                 # a 10-bit operand passes exactly
    assert np.abs(run(A - hi) + hi - A).max() < 1e-6 * np.abs(A).max()


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("dt", ["bf16", "fp16"])
def test_prework_and_normgate_round_like_stock(dt):
    """G5e: the fused kernels reproduce the stock rounding sites (silu in T, rms_norm T(x * inv) then * scale in T, o_norm w * T(x * inv), sigmoid in T): a different rounding
    changed 30-40% of the bf16 elements by one ulp and moved the greedy output of the full model."""
    import mlx.nn as nn
    dtype = {"bf16": mx.bfloat16, "fp16": mx.float16}[dt]
    H, D, S = 4, 128, 150
    C = 3 * H * D
    rng = np.random.default_rng(5)
    qkv = mx.array((rng.standard_normal((1, S, C)) * 2).astype(np.float32)).astype(dtype)
    w = mx.array((rng.standard_normal((C, 4, 1)) * 0.5).astype(np.float32)).astype(dtype)
    conv = nn.Conv1d(C, C, kernel_size=4, bias=False, groups=C, padding=0)
    conv.weight = w
    cs = mx.zeros((1, 3, C), dtype=dtype)
    out = nn.silu(conv(mx.concatenate([cs, qkv], axis=1)))
    qs, ks, vs = (t.reshape(1, S, H, D) for t in mx.split(out, 3, axis=-1))
    sc, eps = D ** -0.5, 1e-6 / D
    qs = sc ** 2 * mx.fast.rms_norm(qs, None, eps)
    ks = sc * mx.fast.rms_norm(ks, None, eps)
    q, k, v, _ = kda_pf.prework(qkv, cs, w, H, D)
    for a, b in ((q, qs), (k, ks), (v, vs)):
        a = np.array(a.astype(mx.float32)); b = np.array(b.astype(mx.float32))
        assert (a != b).mean() < 1e-4                                    # bit-identical today (simd_sum like the stock rms_norm); was 30-40% of the elements
    y = mx.array(rng.standard_normal((1, S, H, D)).astype(np.float32)).astype(dtype)
    g = mx.array(rng.standard_normal((1, S, H, D)).astype(np.float32)).astype(dtype)
    nw = mx.array((1 + 0.1 * rng.standard_normal(D)).astype(np.float32)).astype(dtype)
    norm = nn.RMSNorm(D, eps=1e-6); norm.weight = nw
    ref = np.array((norm(y) * mx.sigmoid(g)).reshape(1, S, -1).astype(mx.float32))
    got = np.array(kda_pf.norm_gate(y, g, nw, 1e-6, H, D).astype(mx.float32))
    assert (got != ref).mean() < 1e-4
