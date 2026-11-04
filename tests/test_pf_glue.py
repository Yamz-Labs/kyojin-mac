"""Prompt-row glue kernels (kyojin_mlx.pf_glue) vs the stock op chains, bit for bit (values compared as float32: signed zeros aside). Needs a Mac (Metal) and mlx-vlm 0.7.10 on the path."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
from kyojin_mlx import pf_glue as PG                                      # noqa: E402
from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE            # noqa: E402
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq   # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

H, D, RD, G = 64, 512, 64, 8
YARN = {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536}
ROPES = {"compressed": lambda: DeepseekV4RoPE(RD, 160000, YARN, 1048576), "plain": lambda: DeepseekV4RoPE(RD, 10000, None, 1048576)}


def f32(a):
    return np.array(a.astype(mx.float32))


@pytest.mark.parametrize("kind", ["compressed", "plain"])
@pytest.mark.parametrize("S,start", [(64, 0), (200, 0), (257, 4096)])
def test_q_rope_and_o_rope_inv(kind, S, start):
    rope = ROPES[kind]()
    rng = np.random.default_rng(S)
    qb = mx.array((rng.standard_normal((1, S, H * D)) * 2).astype(np.float16))
    ref = rope(qb.reshape(1, S, H, D).transpose(0, 2, 1, 3), start)
    assert np.array_equal(f32(PG.q_rope(qb, rope, start, H, D)), f32(ref))
    o = mx.array((rng.standard_normal((1, H, S, D)) * 2).astype(np.float16))
    r = rope(o, start, inverse=True).reshape(1, G, -1, S, D).transpose(0, 1, 3, 2, 4).flatten(-2)
    assert np.array_equal(f32(PG.o_rope_inv(o, rope, start, G)), f32(r))


@pytest.mark.parametrize("S", [64, 333])
def test_fq_t(S):
    rng = np.random.default_rng(S)
    x = mx.array((rng.standard_normal((1, G, S, 1024)) * rng.choice([0.01, 1.0, 30.0], (1, G, S, 1))).astype(np.float16))
    ref = fq(x.transpose(0, 2, 1, 3).flatten(-2))
    assert np.array_equal(f32(PG.fq_t(x)), f32(ref))


@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("S", [64, 129])
def test_expand_and_combine(dt, S):
    from mlx_vlm.models.deepseek_v41.language import hc_expand
    rng = np.random.default_rng(S)
    x = mx.array(rng.standard_normal((1, S, 5120)).astype(np.float32) * 3).astype(dt)
    res = mx.array(rng.standard_normal((1, S, 4, 5120)).astype(np.float32) * 3).astype(dt)
    post = mx.array(rng.uniform(0, 2, (1, S, 4)).astype(np.float32)); comb = mx.array(rng.uniform(0, 1, (1, S, 4, 4)).astype(np.float32))
    assert np.array_equal(f32(PG.expand(x, res, post, comb)), f32(hc_expand(x, res, post, comb)))
    y = mx.array(rng.standard_normal((1, S, 1, 5120)).astype(np.float32) * 3)
    ref = (y.astype(mx.float32).sum(-2) + x.astype(mx.float32)).astype(dt)
    assert np.array_equal(f32(PG.combine(y, x)), f32(ref))


@pytest.mark.parametrize("S", [64, 300])
def test_pre_norm_fq(S):
    import mlx.nn as nn
    rng = np.random.default_rng(S)
    D = 5120
    h = mx.array((rng.standard_normal((1, S, 4, D)) * rng.choice([0.05, 1.0, 40.0], (1, S, 4, 1))).astype(np.float16))
    pre = mx.array(rng.uniform(0, 1.2, (1, S, 4)).astype(np.float32))
    norm = nn.RMSNorm(D, eps=1e-6); norm.weight = mx.array((rng.standard_normal(D) * 0.5 + 1).astype(np.float16))
    ref = norm(((pre[..., None] * h.astype(mx.float32)).sum(axis=2)).astype(mx.float16))
    x, xq = PG.pre_norm_fq(h, pre, norm.weight, 1e-6)
    assert np.array_equal(f32(x), f32(ref))
    assert np.array_equal(f32(xq), f32(fq(ref)))


def test_wo_a_prompt():
    import importlib, os
    from kyojin_mlx import patch as P, exl3_linear as EL
    pack = os.path.expanduser("~/kyojin-mlx/models/ds2")
    if not os.path.isdir(pack):
        pytest.skip("no ds2 pack")
    model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
    a = model.language_model.layers[1].attn
    rope = a.rope
    S = EL.DENSE_FUSED_MAX + 200
    o = mx.array((np.random.default_rng(5).standard_normal((1, 64, S, 512)) * 2).astype(np.float16))
    assert PG.wo_a_ok(a.wo_a, o)
    ref = a.wo_b(fq(a.wo_a(fq(rope(o, 0, inverse=True).reshape(1, 8, -1, S, 512).transpose(0, 1, 3, 2, 4).flatten(-2))).transpose(0, 2, 1, 3).flatten(-2)))
    got = a.wo_b(PG.wo_a_prompt(a.wo_a, o, rope, 0))
    assert np.array_equal(f32(got), f32(ref))


def test_wq_b_rope():
    import os
    from kyojin_mlx import patch as P, exl3_linear as EL
    pack = os.path.expanduser("~/kyojin-mlx/models/ds2")
    if not os.path.isdir(pack):
        pytest.skip("no ds2 pack")
    model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
    a = model.language_model.layers[1].attn
    S = EL.DENSE_FUSED_MAX + 123
    x = mx.array((np.random.default_rng(6).standard_normal((1, S, a.wq_b.in_features)) * 2).astype(np.float16))
    assert PG.wq_b_ok(a.wq_b, x, 64, 512)
    ref = a.rope(a.wq_b(x).reshape(1, S, 64, 512).transpose(0, 2, 1, 3), 7)
    assert np.array_equal(f32(PG.wq_b_rope(a.wq_b, x, a.rope, 7, 64, 512)), f32(ref))


def test_wo_a_prompt_pieces():
    import os
    from kyojin_mlx import patch as P, exl3_linear as EL
    pack = os.path.expanduser("~/kyojin-mlx/models/ds2")
    if not os.path.isdir(pack):
        pytest.skip("no ds2 pack")
    model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
    a = model.language_model.layers[1].attn
    S = EL.DENSE_FUSED_MAX + 100                     # launch groups of 256 rows: 4 full groups + 1 short one
    o = mx.array((np.random.default_rng(8).standard_normal((1, 64, S, 512)) * 2).astype(np.float16))
    pieces = [o[0, :, i:i + 256] + mx.zeros((1, 1, 1), dtype=mx.float16) for i in range(0, S, 256)]
    assert PG.wo_a_ok(a.wo_a, pieces)
    assert np.array_equal(f32(PG.wo_a_prompt(a.wo_a, pieces, a.rope, 3)), f32(PG.wo_a_prompt(a.wo_a, o, a.rope, 3)))


@pytest.fixture
def nofq():
    import importlib
    m = importlib.import_module("mlx_vlm.models.deepseek_v41.fakequant")
    m.DISABLE = True
    yield
    m.DISABLE = False


def test_kernels_with_fake_quant_disabled(nofq):
    """The accuracy gate's --no-fq (fakequant.DISABLE): the fused kernels pass the values through, like the stock functions."""
    import os
    import mlx.nn as nn
    from kyojin_mlx import patch as P, exl3_linear as EL
    rng = np.random.default_rng(11)
    x = mx.array(rng.standard_normal((1, 8, 70, 1024)).astype(np.float16))
    assert np.array_equal(f32(PG.fq_t(x)), f32(x.transpose(0, 2, 1, 3).flatten(-2)))
    D, S = 5120, 70
    h = mx.array((rng.standard_normal((1, S, 4, D)) * 3).astype(np.float16)); pre = mx.array(rng.uniform(0, 1.2, (1, S, 4)).astype(np.float32))
    norm = nn.RMSNorm(D, eps=1e-6); norm.weight = mx.array((rng.standard_normal(D) * 0.5 + 1).astype(np.float16))
    ref = norm(((pre[..., None] * h.astype(mx.float32)).sum(axis=2)).astype(mx.float16))
    xn, xq = PG.pre_norm_fq(h, pre, norm.weight, 1e-6)
    assert np.array_equal(f32(xn), f32(ref)) and np.array_equal(f32(xq), f32(ref))
    pack = os.path.expanduser("~/kyojin-mlx/models/ds2")
    if os.path.isdir(pack):
        model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
        a = model.language_model.layers[1].attn
        S = EL.DENSE_FUSED_MAX + 50
        o = mx.array((rng.standard_normal((1, 64, S, 512)) * 2).astype(np.float16))
        ref = a.wo_b(a.wo_a(a.rope(o, 0, inverse=True).reshape(1, 8, -1, S, 512).transpose(0, 1, 3, 2, 4).flatten(-2)).transpose(0, 2, 1, 3).flatten(-2))
        assert np.array_equal(f32(a.wo_b(PG.wo_a_prompt(a.wo_a, o, a.rope, 0))), f32(ref))


def test_unwrap_passive_rows_wrappers():
    import mlx.nn as nn
    from kyojin_mlx import rows as RW
    lin = nn.Linear(8, 8)
    h, s = RW.Hoist(lin), RW.Sink(lin)
    assert PG.unwrap(h) is lin and PG.unwrap(s) is lin and PG.unwrap(lin) is lin
    h.st.pre, s.st.grab = mx.zeros((1, 8)), []
    assert PG.unwrap(h) is h and PG.unwrap(s) is s
