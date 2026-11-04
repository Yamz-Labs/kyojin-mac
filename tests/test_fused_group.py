"""S5 grouped same-input linears (kyojin_mlx.fused_group) vs the separate Exl3Linear path: bit-identical for 1-4 rows, equal and ragged members."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)
import mlx.nn as nn

from kyojin_mlx import fused_group as FG
from kyojin_mlx.exl3_linear import Exl3Linear


def make(kin, nout, K, seed, cb):
    rng = np.random.default_rng(seed)
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    return Exl3Linear(mx.array(tr), mx.array(suh), mx.array(svh), cb=cb)


def bits(a):
    return np.array(a).view(np.uint16)


CASES = [("gate_up", 512, [1024, 1024], 2, "mcg"), ("qkv", 512, [1024, 256, 256], 3, "mcg"), ("ragged_tgk", 4096, [1024, 16384], 2, "mul1"),
         ("four", 1024, [512, 512, 512, 512], 4, "mul1"), ("qkvz", 2048, [4096, 1024, 1024, 2048], 2, "mcg")]


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def grp(request):
    _, kin, outs, K, cb = request.param
    mem = [make(kin, n, K, 11 * i + K, cb) for i, n in enumerate(outs)]
    return mem, FG.Exl3Group(mem)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("dt", [np.float16, np.float32])
def test_bit_exact(grp, rows, dt):
    mem, g = grp
    x = mx.array((np.random.default_rng(rows).standard_normal((rows, mem[0].in_features)) * 2).astype(dt))
    got = g.run(x)
    for m, y in zip(mem, got):
        want = m(x)
        assert y.dtype == want.dtype == mx.float16 and y.shape == want.shape
        assert np.array_equal(bits(y), bits(want))


def test_ragged_case_really_ragged():
    c = [c for c in CASES if c[0] == "ragged_tgk"][0]
    _, kin, outs, K, cb = c
    g = FG.Exl3Group([make(kin, n, K, 5 + i, cb) for i, n in enumerate(outs)])
    assert len(set(g.tgk)) > 1, g.tgk


def test_zero_rows_keep_signed_zero(grp):
    mem, g = grp
    x = np.random.default_rng(3).standard_normal((2, mem[0].in_features)).astype(np.float16)
    x[1] = 0
    x = mx.array(x)
    for m, y in zip(mem, g.run(x)):
        assert np.array_equal(bits(y), bits(m(x)))


class Block(nn.Module):
    """gate / up / down MLP + q / k / v on the same input, the way a decoder layer calls them."""

    def __init__(self, kin=512, hid=1024):
        super().__init__()
        self.gate_proj = make(kin, hid, 2, 1, "mcg")
        self.up_proj = make(kin, hid, 2, 2, "mcg")
        self.down_proj = make(hid, kin, 2, 3, "mcg")
        self.q = make(kin, 1024, 3, 4, "mcg")
        self.k = make(kin, 256, 3, 5, "mcg")
        self.v = make(kin, 256, 3, 6, "mcg")
        self.single = make(kin, 512, 2, 7, "mcg")

    def __call__(self, x):
        h = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return h, self.q(x), self.k(x), self.v(x), self.single(h)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Block(), Block()]

    def __call__(self, x):
        outs = []
        for l in self.layers:
            h, q, k, v, s = l(x)
            outs += [h, q, k, v, s]
            x = x + h
        return outs


def test_install_and_kill_switch():
    model = Toy()
    x = mx.array(np.random.default_rng(1).standard_normal((1, 1, 512)).astype(np.float16))
    base = [np.array(o) for o in model(x)]
    info = FG.install(model, forward=lambda: model(x))
    assert info["groups"] == 4 and info["members"] == 10 and info["launches_saved"] == 2 * (3 + 6), info
    assert all(isinstance(l.gate_proj, FG.FusedMember) and isinstance(l.q, FG.FusedMember) for l in model.layers)
    assert not isinstance(model.layers[0].down_proj, FG.FusedMember) and not isinstance(model.layers[0].single, FG.FusedMember)
    on = [np.array(o) for o in model(x)]
    assert all(np.array_equal(a.view(np.uint16), b.view(np.uint16)) for a, b in zip(base, on))
    FG.ENABLED = False
    try:
        off = [np.array(o) for o in model(x)]
    finally:
        FG.ENABLED = True
    assert all(np.array_equal(a.view(np.uint16), b.view(np.uint16)) for a, b in zip(base, off))


@pytest.mark.parametrize("shape", [(1, 3, 512), (2, 1, 512), (1, 8, 512)])
def test_shapes_and_prefill_fallback(shape):
    model = Toy()
    x = mx.array(np.random.default_rng(2).standard_normal(shape).astype(np.float16))
    base = [np.array(o) for o in model(x)]
    FG.install(model, forward=lambda: model(mx.array(np.zeros((1, 1, 512), np.float16))))
    got = [np.array(o) for o in model(x)]
    assert all(a.shape == b.shape and np.array_equal(a.view(np.uint16), b.view(np.uint16)) for a, b in zip(base, got))


def test_input_dtype_float32_and_second_token_recomputes():
    model = Toy()
    FG.install(model, forward=lambda: model(mx.array(np.zeros((1, 1, 512), np.float16))))
    for seed in (3, 4):
        x = mx.array(np.random.default_rng(seed).standard_normal((1, 1, 512)).astype(np.float32))
        got = [np.array(o) for o in model(x)]
        FG.ENABLED = False
        try:
            want = [np.array(o) for o in model(x)]
        finally:
            FG.ENABLED = True
        assert all(np.array_equal(a.view(np.uint16), b.view(np.uint16)) for a, b in zip(want, got))


# ---------------------------------------------------------------------------------------------- real DS4.1 slice (ds2: 2 real layers)
import os

SHRUNK = os.environ.get("KYOJIN_SHRUNK")


@pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set")
def test_ds41_slice_s5_logits_bit_identical():
    pytest.importorskip("mlx_vlm")
    from kyojin_mlx import patch as P
    pack = os.path.join(SHRUNK, "ds2")
    if not os.path.isdir(pack):
        pytest.skip(f"{pack} missing")
    model, spec, cfg = P.build_model(pack)
    P.patch_model(model, pack, spec, cfg)
    lm = model.language_model
    vocab = cfg["text_config"]["vocab_size"]
    seq = np.random.default_rng(1).integers(3, vocab, 40).tolist()
    info = FG.install(model)
    assert info["groups"] >= 5 and info["fake_quant_memo"] and info["window_math_shared"], info
    names = [n for g in info["names"] for n in g]
    assert any(n.endswith("attn.wq_a") for n in names) and any(n.endswith("attn.wkv") for n in names)
    caches = {v: lm.make_cache() for v in (False, True)}
    for toks in (seq[:5], seq[5:6], seq[6:8], seq[8:9], seq[9:13], seq[13:14]):                # prefill (5 rows: dense fallback), decode, 2 and 4 rows
        outs = {}
        for en in (False, True):
            FG.ENABLED = en
            outs[en] = lm(mx.array([toks]), cache=caches[en]).logits
            mx.eval(outs[en])
        FG.ENABLED = True
        assert bool(mx.array_equal(outs[False], outs[True]).item()), len(toks)
