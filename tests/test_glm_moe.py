"""G2c fused GLM decode MoE: the select kernel (sigmoid + bias, top-k, normalise, scale) and the combine kernel against a numpy reference."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import glm_moe as GM  # noqa: E402


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("T", [1, 3, 8])
@pytest.mark.parametrize("norm", [True, False])
def test_select_matches_reference(T, norm):
    rng = np.random.default_rng(T)
    E, K, scale = 288, 8, 2.5
    lg = rng.normal(size=(T, E)).astype(np.float32) * 2
    bias = rng.normal(size=(E,)).astype(np.float32) * 0.1
    inds, wts = GM.select(mx.array(lg), mx.array(bias), T, E, K, norm, scale)
    inds, wts = np.asarray(inds).reshape(T, K), np.asarray(wts).reshape(T, K)
    s = 1 / (1 + np.exp(-lg))
    for t in range(T):
        ref = np.argsort(-(s[t] + bias), kind="stable")[:K]
        assert sorted(inds[t]) == sorted(ref)
        w = {int(i): float(s[t, i]) for i in ref}
        tot = sum(w.values()) + 1e-20
        for i, wv in zip(inds[t], wts[t]):
            exp = (w[int(i)] / tot if norm else w[int(i)]) * scale
            assert abs(wv - exp) < 1e-5 * max(1.0, abs(exp))


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_select_ties_lowest_index():
    E, K = 288, 8
    lg = np.zeros((1, E), np.float32)
    inds, _ = GM.select(mx.array(lg), mx.zeros((E,)), 1, E, K, True, 1.0)
    assert list(np.asarray(inds)) == list(range(K))


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("has_shared", [True, False])
def test_combine_matches_reference(has_shared):
    rng = np.random.default_rng(5)
    T, K, H = 3, 8, 256
    y = rng.normal(size=(T * K, H)).astype(np.float16)
    w = rng.uniform(0.0, 1.0, size=(T * K,)).astype(np.float32)
    sh = rng.normal(size=(T, H)).astype(np.float16) if has_shared else None
    out = np.asarray(GM.combine(mx.array(y), mx.array(w), mx.array(sh) if has_shared else None, T, K, H)).astype(np.float32)
    prod = (y.astype(np.float32).reshape(T, K, H) * w.astype(np.float16).astype(np.float32).reshape(T, K, 1)).astype(np.float16).astype(np.float32)
    ref = prod.sum(1).astype(np.float16).astype(np.float32)
    if has_shared:
        ref = (ref + sh.astype(np.float32)).astype(np.float16).astype(np.float32)
    assert np.abs(out - ref).max() <= 2e-3 * max(1.0, np.abs(ref).max())


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("T", [17, 300])
def test_prompt_combine_and_select_match_the_stock_block(T):
    """G5h flag glm_pf_moe: select + combine on prompt rows against the stock `_expert_select` + `SwitchGLU._combine` (same fp16 roundings, bit identical combine)."""
    from mlx_vlm.models.glm5_next import language as L
    from mlx_vlm.models.switch_layers import SwitchGLU
    E, K, H, scale = 288, 8, 512, 2.5
    mx.random.seed(T)
    lg = mx.random.normal((T, E)) * 2
    bias = mx.random.normal((E,)) * 0.1
    inds, wts = GM.select(lg, bias, T, E, K, True, scale)
    si, sw = L._expert_select(lg, bias, K, 1, 1, scale, True)
    a = np.sort(np.asarray(inds).reshape(T, K), axis=-1)
    b = np.sort(np.asarray(si).reshape(T, K), axis=-1)
    assert (a == b).all()
    y = (mx.random.normal((T, K, H)) * 0.5).astype(mx.float16)
    sh = (mx.random.normal((T, H)) * 0.5).astype(mx.float16)
    ref = SwitchGLU._combine(y, sw, sh)                                  # stock weights, stock slot order
    out = GM.combine(y.reshape(T * K, H), sw.reshape(-1), sh, T, K, H)   # same slot order as the stock selection
    o, r = np.asarray(out.astype(mx.float32)), np.asarray(ref.astype(mx.float32))
    prod = (np.asarray(y.astype(mx.float32)) * np.asarray(sw.astype(mx.float16).astype(mx.float32))[..., None]).astype(np.float16).astype(np.float64)
    f = prod.sum(1) + np.asarray(sh).astype(np.float64)                                          # f64 sum of the fp16 products: the kernel (f32 accumulate, one rounding) is the closer one
    assert np.abs(o - f).max() <= np.abs(r - f).max() + 1e-6
    assert np.abs(o - f).max() <= 2.0 ** -8                                                      # one fp16 ulp at |value| < 4 is 2^-9


def test_pf_why_allows_fp32_x_from_mhc_collapse(monkeypatch):
    """G5h: `hyper_connection.HyperConnection.__call__` collapses the mHC streams in float32 for every prompt chunk (T > DECODE_BLOCK_SIZE=8), so real
    traffic always hands the MoE block fp32 x, and the real NVFP4 pack stores the router (gate) weight as fp16, not fp32. The PF path's own ops (select on
    the router logits, combine on switch_mlp / shared_experts outputs) never read x itself (`tiled_linear`, `shared_experts` and `ForeignSwitchGLU.__call__`
    all cast x internally, and `tiled_linear` promotes a fp16-weight matmul of a fp32 input to fp32 output regardless), so fp32 x and a fp16/bf16 router
    weight must stay eligible; only a dtype the downstream calls cannot cast from (complex64) is rejected."""
    from kyojin_mlx import foreign_linear as FL

    monkeypatch.setenv("KYOJIN_GLM_PF_MOE", "1")

    class _StubSwitchGLU(FL.ForeignSwitchGLU):
        def __init__(self):                                   # pylint: disable=super-init-not-called
            pass                                               # _pf_why only does isinstance(); no DevPack needed

    def _blk(weight_dtype):
        class Gate:
            top_k, n_group, topk_group = 8, 1, 1
            weight = mx.zeros((288, 512), weight_dtype)

        class Blk:
            sharding_group = None
            gate = Gate()
            switch_mlp = _StubSwitchGLU()
        return Blk()

    for x_dtype, w_dtype, eligible in ((mx.float16, mx.float32, True), (mx.float32, mx.float16, True), (mx.float32, mx.bfloat16, True), (mx.complex64, mx.float32, False)):
        x = mx.zeros((1, 32, 512), x_dtype)
        why = GM._pf_why(_blk(w_dtype), x)
        assert (why is None) == eligible, (x_dtype, w_dtype, why)


def _fake_metal_kernel(names):
    """Stand-in for mx.fast.metal_kernel: records the compiled name, returns zero arrays of the requested shape/dtype (no Metal needed)."""
    def fake(name, input_names, output_names, header, source, **_):
        names.append(name)
        def run(inputs, template=None, grid=None, threadgroup=None, output_shapes=None, output_dtypes=None, **_):
            return [mx.zeros(tuple(s), d) for s, d in zip(output_shapes, output_dtypes)]
        return run
    return fake


def _pf_blk(shared_dtype, y_dtype, calls, T=32, E=64, K=4, H=512):
    """Stub MoE block for `_pf_call`: `shared_experts` and `switch_mlp` return the given dtypes, `_stock_call` records the fallback."""
    from kyojin_mlx import foreign_linear as FL

    class _StubSwitchGLU(FL.ForeignSwitchGLU):
        def __init__(self, dtype):  # pylint: disable=super-init-not-called
            self.dtype = dtype

        def __call__(self, x, inds):
            return mx.zeros((inds.size, H), self.dtype)                   # one row per slot: T * K rows, reshape(T * K, H) downstream

    class Gate:
        top_k, n_group, topk_group = K, 1, 1
        weight = mx.zeros((E, H), mx.float32)
        e_score_correction_bias = mx.zeros((E,), mx.float32)
        norm_topk_prob, routed_scaling_factor = True, 2.5

    class Blk:
        sharding_group = None
        gate = Gate()
        switch_mlp = _StubSwitchGLU(y_dtype)

        def shared_experts(self, x):
            return mx.zeros((1, T, H), shared_dtype)

        def _stock_call(self, x, *a, residual=None, **kw):
            calls.append(residual)
            return "stock"

    return Blk()


def test_pf_call_falls_back_when_combine_inputs_are_not_fp16(monkeypatch):
    """The fused prefill combine reads y and sh as half4: a shared expert (or expert block) returning fp32 must take the stock route, not have its bytes reinterpreted."""
    from kyojin_mlx import archs as A

    monkeypatch.setenv("KYOJIN_GLM_PF_MOE", "1")
    monkeypatch.setattr(mx.fast, "metal_kernel", _fake_metal_kernel([]))
    monkeypatch.setattr(A, "lang", lambda spec: type("L", (), {"tiled_linear": staticmethod(lambda fn, v: fn(v))})())
    sel_before = set(GM._SEL)
    try:
        x = mx.zeros((1, 32, 512), mx.float32)                     # mHC collapse hands prompt chunks fp32
        for shared_dtype, y_dtype, stock in ((mx.float32, mx.float16, True), (mx.float16, mx.float32, True), (mx.float16, mx.float16, False)):
            calls = []
            out = GM._pf_call(_pf_blk(shared_dtype, y_dtype, calls), x)
            assert (out == "stock") == stock, (shared_dtype, y_dtype, out)
            assert (len(calls) == 1) == stock, (shared_dtype, y_dtype, calls)
    finally:
        for key in set(GM._SEL) - sel_before:
            del GM._SEL[key]


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_select_all_nan_logits_stay_in_range():
    """A row of all-NaN logits used to select index 0xFFFFFFFF (NaN never passes `>` against the -INFINITY best), an out-of-range gather index."""
    E, K = 288, 8
    lg = np.full((2, E), np.nan, np.float32)
    inds, _ = GM.select(mx.array(lg), mx.zeros((E,), mx.float32), 2, E, K, True, 2.5)
    a = np.asarray(inds).astype(np.int64)
    assert ((a >= 0) & (a < E)).all(), a


def test_select_two_scales_two_kernels(monkeypatch):
    """SCALE is baked into the kernel source: the mlx kernel name (its compile cache key) must carry it, or the second scale reuses the first kernel."""
    names = []
    monkeypatch.setattr(mx.fast, "metal_kernel", _fake_metal_kernel(names))
    sel_before = set(GM._SEL)
    try:
        E, K = 96, 2
        lg, bias = mx.zeros((1, E), mx.float32), mx.zeros((E,), mx.float32)
        GM.select(lg, bias, 1, E, K, False, 2.5)
        GM.select(lg, bias, 1, E, K, False, 1.0)
        assert len(names) == 2 and len(set(names)) == 2, names
    finally:
        for key in set(GM._SEL) - sel_before:
            del GM._SEL[key]
