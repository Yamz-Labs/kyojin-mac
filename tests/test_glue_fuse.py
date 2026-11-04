"""Glue fuse (kyojin_mlx.glue_fuse): the collapse + RMSNorm launch and the block body are bit-identical to the stock op sequence.

mix_norm merges `mhc_pre(x, pre)` and the RMSNorm that consumes it, and runs the mHC sinkhorn in the same threadgroup. The norm's element layout
and reduction are the ones of `mx.fast.rms_norm`, so the output must match the stock pair bit for bit, for one and several rows (a row may not
depend on how many rows share the launch). The block body replaces the four glue calls of a DS4.1 block with the fused ones and keeps every
other call, so a whole block must reproduce the stock body's bits. Needs a Mac (Metal).
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn                                                        # noqa: E402

from kyojin_mlx import fuse as F                                           # noqa: E402
from kyojin_mlx import glue_fuse as GF                                     # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture(autouse=True)
def _on():
    """The kernels run whatever the switch says; the block body only takes the call when it is on."""
    was = GF.ENABLED
    GF.ENABLED = True
    yield
    GF.ENABLED = was


def same(a, b):
    a, b = np.array(a), np.array(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("D", [128, 256, 512, 4096, 5120])
@pytest.mark.parametrize("B", [1, 2, 5])
def test_mix_norm_bit_identical(D, B):
    """mix_norm == (fuse.mhc_mix_pre, then mx.fast.rms_norm of its collapsed row), on the same sinkhorn the block uses."""
    rng = np.random.default_rng(D * 7 + B)
    for name, xs in (("normal", rng.standard_normal((B, 1, 4, D)) * 0.02),
                     ("tiny", rng.standard_normal((B, 1, 4, D)) * 1e-6),
                     ("wide", rng.standard_normal((B, 1, 4, D)) * 30),
                     ("zeros", np.zeros((B, 1, 4, D))),
                     ("sparse", (rng.standard_normal((B, 1, 4, D)) * (rng.random((B, 1, 4, D)) < 0.1)))):
        x = mx.array(xs.astype(np.float16))
        pre = mx.array(rng.standard_normal((B, 1, 4)).astype(np.float32) * (3 if name == "wide" else 1))
        w = mx.array((rng.standard_normal(D) * 0.5).astype(np.float16))
        mixes = mx.array(rng.standard_normal((B, 1, 24)).astype(np.float32) * (40 if name == "wide" else 3))
        scale = mx.array([1.0, 0.5, 2.0], dtype=mx.float32)
        base = mx.array(rng.standard_normal(24).astype(np.float32))
        eps, hc_eps, iters = 1e-20, 1e-6, 20
        y, rpre, rpost, rcomb = F.mhc_mix_pre(mixes, scale, base, x, pre, iters, hc_eps)
        ref = mx.fast.rms_norm(y, w, eps)
        got, gpre, gpost, gcomb = GF.mix_norm(mixes, x, pre, w, eps, scale, base, iters, hc_eps)
        mx.eval(got, gpre, gpost, gcomb)
        assert same(got, ref), f"{name} D={D} B={B}: norm"
        assert same(gpre, rpre), f"{name} D={D} B={B}: pre"
        assert same(gpost, rpost), f"{name} D={D} B={B}: post"
        assert same(gcomb, rcomb), f"{name} D={D} B={B}: comb"


def test_mix_norm_row_independence():
    """A row keeps its bits whatever the number of rows in the launch (multi-user decode, verify rows)."""
    rng = np.random.default_rng(5)
    D = 5120
    x = mx.array((rng.standard_normal((3, 1, 4, D)) * 0.02).astype(np.float16))
    pre = mx.array(rng.standard_normal((3, 1, 4)).astype(np.float32))
    w = mx.array((rng.standard_normal(D) * 0.5).astype(np.float16))
    mixes = mx.array(rng.standard_normal((3, 1, 24)).astype(np.float32) * 3)
    scale, base = mx.array([1.0, 1.0, 1.0], dtype=mx.float32), mx.array(rng.standard_normal(24).astype(np.float32))
    out3, pre3, post3, comb3 = GF.mix_norm(mixes, x, pre, w, 1e-20, scale, base, 20, 1e-6)
    for r in range(3):
        one, p1, po1, c1 = GF.mix_norm(mixes[r:r + 1], x[r:r + 1], pre[r:r + 1], w, 1e-20, scale, base, 20, 1e-6)
        mx.eval(one, p1, po1, c1)
        assert same(one, out3[r:r + 1])
        assert same(p1, pre3[r:r + 1]) and same(po1, post3[r:r + 1]) and same(c1, comb3[r:r + 1])


class _Stub:
    def __init__(self, k, b):
        self.k, self.b = k, b

    def __call__(self, x, *a, **k):
        return (x * self.k + self.b).astype(x.dtype)


def _block(rng, hc_mixes, D=5120):
    """A DS4.1-shaped block: the mHC mix matrices, the two norms, and deterministic stubs for the two sublayers. `hc_mixes` is the reference
    coefficient chain the body under test uses (the model's own, or fuse.mhc_mix_pre + its sinkhorn)."""
    class Small(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.layer_idx, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps = 0, 4, 20, 1e-6
            self.norm_eps = 1e-20
            self.attn_norm = nn.RMSNorm(D, eps=self.norm_eps)
            self.ffn_norm = nn.RMSNorm(D, eps=self.norm_eps)
            mix = (2 + self.hc_mult) * self.hc_mult
            self.hc_attn_fn = mx.array(rng.standard_normal((mix, self.hc_mult * D)).astype(np.float32) * 0.01)
            self.hc_ffn_fn = mx.array(rng.standard_normal((mix, self.hc_mult * D)).astype(np.float32) * 0.01)
            self.hc_attn_base = mx.array(rng.standard_normal(mix).astype(np.float32))
            self.hc_ffn_base = mx.array(rng.standard_normal(mix).astype(np.float32))
            self.hc_attn_scale = mx.array([1.0, 0.5, 2.0], dtype=mx.float32)
            self.hc_ffn_scale = mx.array([1.0, 0.5, 2.0], dtype=mx.float32)
            self.attn_norm.weight = mx.array((rng.standard_normal(D) * 0.5).astype(np.float16))
            self.ffn_norm.weight = mx.array((rng.standard_normal(D) * 0.5).astype(np.float16))
            self.attn, self.ffn = _Stub(0.5, 0.25), _Stub(-0.75, 0.125)

    Small.hc_pre = staticmethod(lambda x, pre: F.mhc_pre(x, pre))
    Small.hc_mixes = lambda self, x, fn, scale, base: hc_mixes(self, x, fn, scale, base)
    return Small()


def test_block_bit_identical():
    """glue_fuse.block against the stock block body on the same input: same bits, for one and two rows."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
    from mlx_vlm.models.deepseek_v41.language import hc_expand, hc_mix_coeffs
    rng = np.random.default_rng(11)
    blk = _block(rng, lambda self, x, fn, scale, base: hc_mix_coeffs(x, fn, scale, base, self.hc_mult, self.hc_sinkhorn_iters,
                                                                     self.hc_eps, self.norm_eps))
    for B in (1, 2):
        h = mx.array((rng.standard_normal((B, 1, 4, 5120)) * 0.02).astype(np.float16))
        pre = mx.array(rng.standard_normal((B, 1, 4)).astype(np.float32))
        assert GF.block_ok(blk, h, pre, None, None)
        residual = h
        ap, apo, ac = blk.hc_mixes(h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
        x = blk.attn_norm(blk.hc_pre(h, pre))
        x = blk.attn(x)
        x = hc_expand(x, residual, apo, ac)
        residual = x
        fp, fpo, fc = blk.hc_mixes(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
        x = blk.ffn_norm(blk.hc_pre(x, ap))
        x = blk.ffn(x)
        ref, ref_fp = hc_expand(x, residual, fpo, fc), fp
        got, got_fp = GF.block(blk, h, pre, None, None)
        mx.eval(got, ref, got_fp, ref_fp)
        assert same(got, ref), f"B={B}: block output"
        assert same(got_fp, ref_fp), f"B={B}: ffn_pre"


def test_block_ok_rejects():
    """The fused body only takes the cases it reproduces: fp16 one-row mHC blocks with plain RMSNorm weights."""
    rng = np.random.default_rng(13)
    blk = _block(rng, lambda *a: None)
    h = mx.array((rng.standard_normal((1, 1, 4, 5120)) * 0.02).astype(np.float16))
    pre = mx.array(rng.standard_normal((1, 1, 4)).astype(np.float32))
    try:
        assert GF.block_ok(blk, h, pre, None, None)
        assert not GF.block_ok(blk, h.astype(mx.float32), pre.astype(mx.float32), None, None)         # fp32 stream
        assert not GF.block_ok(blk, mx.zeros((2, 1, 4, 5120), dtype=mx.float16), mx.zeros((1, 1, 4), dtype=mx.float32), None, None)
        assert not GF.block_ok(blk, h, pre, mx.zeros((2,), dtype=mx.bool_), None)                     # mask of another row count
        assert GF.block_ok(blk, h, pre, mx.zeros((1,), dtype=mx.bool_), None)
        GF.ENABLED = False
        assert not GF.block_ok(blk, h, pre, None, None)                                               # switch off
    finally:
        GF.ENABLED = True


def _stacked_site_block(rng):
    """A block whose `hc_mixes` is the rows site (rows.mix_vector + sinkhorn), then wrapped by a later per-instance subclass that only overrides
    __call__ (CED / speculative decode do this): `hc_mixes` is then inherited, not in type(blk).__dict__."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
    from mlx_vlm.models.deepseek_v41.language import hc_mix_coeffs
    from kyojin_mlx import rows as RW
    blk = _block(rng, lambda self, x, fn, scale, base: hc_mix_coeffs(x, fn, scale, base, self.hc_mult, self.hc_sinkhorn_iters,
                                                                     self.hc_eps, self.norm_eps))
    RW._override(blk, "hc_mixes", RW._hc_block_mixes)
    cls = type(blk)
    blk.__class__ = type(cls.__name__, (cls,), {"__call__": lambda self, *a: None, "__module__": cls.__module__})
    return blk


def test_site_found_along_mro():
    rng = np.random.default_rng(17)
    blk = _stacked_site_block(rng)
    assert "hc_mixes" not in type(blk).__dict__
    assert GF.site_installed(blk)
    plain = _block(rng, lambda *a: None)
    assert not GF.site_installed(plain)


@pytest.mark.parametrize("B", [1, 2])
def test_mix_norm_matches_site_block(B):
    """With the rows site installed (also under a stacked subclass), the fused collapse + norm + sinkhorn must give the bits of the unfused chain
    blk.hc_mixes -> hc_pre -> attn_norm: the mix vector has to come from the same dot (rowdot), not from `@`."""
    rng = np.random.default_rng(19 + B)
    blk = _stacked_site_block(rng)
    h = mx.array((rng.standard_normal((B, 1, 4, 5120)) * 0.5).astype(np.float16))
    pre = mx.array(rng.standard_normal((B, 1, 4)).astype(np.float32))
    rp, rpo, rc = blk.hc_mixes(h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
    ref = blk.attn_norm(blk.hc_pre(h, pre))
    got, gp, gpo, gc = GF.mix_norm(GF.mixes_of(blk, h, blk.hc_attn_fn), h, pre, blk.attn_norm.weight, blk.attn_norm.eps, blk.hc_attn_scale,
                                   blk.hc_attn_base, blk.hc_sinkhorn_iters, blk.hc_eps)
    mx.eval(ref, got, rp, rpo, rc, gp, gpo, gc)
    assert same(got, ref) and same(gp, rp) and same(gpo, rpo) and same(gc, rc)


@pytest.mark.parametrize("B", [1, 2])
def test_fold_under_stacked_subclass_bit_identical(B):
    """fuse_layer's `fold` (rows.mix_vector + fuse.mhc_mix_pre, one launch for sinkhorn + collapse) is taken for a block whose rows site sits under a
    stacked per-instance subclass, and gives the bits of the unfused hc_mixes + hc_pre."""
    from kyojin_mlx import rows as RW
    rng = np.random.default_rng(23 + B)
    blk = _stacked_site_block(rng)
    assert GF.site_installed(blk) and blk.hc_mult == 4
    for sc in (0.02, 1.0, 30.0):
        h = mx.array((rng.standard_normal((B, 1, 4, 5120)) * sc).astype(np.float16))
        pre = mx.array(rng.standard_normal((B, 1, 4)).astype(np.float32))
        rp, rpo, rc = blk.hc_mixes(h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
        ry = blk.hc_pre(h, pre)
        mixes = RW.mix_vector(blk, h, blk.hc_attn_fn)
        y, p, po, c = F.mhc_mix_pre(mixes, blk.hc_attn_scale, blk.hc_attn_base, h, pre, blk.hc_sinkhorn_iters, blk.hc_eps)
        mx.eval(ry, rp, rpo, rc, y, p, po, c)
        assert same(y, ry) and same(p, rp) and same(po, rpo) and same(c, rc)


def test_fold_predicate_sees_stacked_site():
    from kyojin_mlx import fuse_layer as FLY
    rng = np.random.default_rng(29)
    blk = _stacked_site_block(rng)
    assert FLY.fold_ok(blk) == F.MIXPRE
    assert not FLY.fold_ok(_block(rng, lambda *a: None))
