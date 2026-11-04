"""The hidden expand kernels saturate at the fp16 range instead of writing inf, and stay bit-identical below it."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import fuse, pf_glue, stagek

D, S = 1024, 3


def _inputs(scale, dtype=np.float16):
    rng = np.random.default_rng(3)
    x = mx.array(np.clip(rng.standard_normal((S, D)) * scale, -60000, 60000).astype(dtype))
    r = mx.array(np.clip(rng.standard_normal((S, 4, D)) * scale, -60000, 60000).astype(dtype))
    post = mx.array(rng.uniform(0.5, 2.0, (S, 4)).astype(np.float32))
    comb = mx.array(rng.uniform(0.0, 0.5, (S, 4, 4)).astype(np.float32))
    return x, r, post, comb


def _ref(x, r, post, comb):
    mixed = mx.einsum("baj,bad->bjd", comb, r.astype(mx.float32))
    return (post[:, :, None] * x.astype(mx.float32)[:, None] + mixed)


def _fuse(x, r, post, comb):
    return fuse.mhc_expand(x.reshape(S, 1, D), r.reshape(S, 1, 4, D), post.reshape(S, 1, 4), comb.reshape(S, 1, 4, 4))[0].reshape(S, 4, D)


def _pf(x, r, post, comb):
    return pf_glue.expand(x[None], r[None], post[None], comb[None])[0]


@pytest.mark.parametrize("fn", [_fuse, _pf])
def test_identical_below_limit(fn):
    a = _inputs(3.0)
    X, R, P, C = [np.array(v).astype(np.float64) for v in a]
    ref, y = np.einsum("baj,bad->bjd", C, R) + P[:, :, None] * X[:, None], np.array(fn(*a))   # exact in fp64: the kernel is within one fp16 step
    assert np.isfinite(y).all() and (np.abs(y - ref) <= np.spacing(np.abs(ref).astype(np.float16)).astype(np.float64)).all()


@pytest.mark.parametrize("fn", [_fuse, _pf])
def test_saturates_on_overflow(fn):
    x, r, post, comb = _inputs(40000.0)
    x = mx.where(mx.arange(D)[None] == 5, mx.array(np.inf, dtype=mx.float16), x)  # an inf input is rescued too
    y = np.array(fn(x, r, post, comb))
    ref = np.array(_ref(x, r, post, comb))
    assert np.isfinite(y).all() and np.abs(y).max() == 65504.0
    big = np.abs(ref) > 65520
    assert big.any() and np.array_equal(y[big], np.sign(ref[big]) * 65504.0)


def test_bf16_not_clamped():
    x, r, post, comb = _inputs(1.0e5, np.float32)
    x, r = x.astype(mx.bfloat16), r.astype(mx.bfloat16)
    y = _fuse(x, r, post, comb)
    mx.eval(y)
    assert float(mx.abs(y.astype(mx.float32)).max()) > 65504


def test_stage_expand_saturates():
    rng = np.random.default_rng(5)
    y = mx.array(np.clip(rng.standard_normal((S, 1, D)) * 40000, -60000, 60000).astype(np.float16))
    res = mx.array(np.clip(rng.standard_normal((S, 1, 4, D)) * 40000, -60000, 60000).astype(np.float16))
    z = mx.array((rng.standard_normal((S, stagek.NMIX)) * 0.3).astype(np.float32))
    scale = mx.array(np.array([0.7, 0.9, 1.1], np.float32))
    base = mx.array((rng.standard_normal(stagek.NMIX) * 0.3).astype(np.float32))
    out, out32 = stagek.expand_stage(y, res, z, scale, base, 20, 1e-6)
    mx.eval(out, out32)
    o = np.array(out)
    assert np.isfinite(o).all() and np.abs(o).max() <= 65504.0


def test_sat_helper():
    a = mx.array(np.array([np.inf, -np.inf, 1.5], np.float16))
    assert np.array(fuse.sat(a)).tolist() == [65504.0, -65504.0, 1.5]


def test_collapse_saturates():
    rng = np.random.default_rng(9)
    x = mx.array(np.clip(rng.standard_normal((S, 1, 4, D)) * 60000, -60000, 60000).astype(np.float16))
    pre = mx.array(np.full((S, 1, 4), 0.9, np.float32))                         # the weights sum to 3.6: the fp32 sum leaves the fp16 range
    y = np.array(fuse.mhc_pre(x, pre))
    assert np.isfinite(y).all() and np.abs(y).max() == 65504.0


# ---- cmerge: source-level guards, CPU only (no Metal needed) ------------------------------------------------------------------------------------------------------
# The eight merged lever branches add no kernel that writes or collapses a hidden stream. These counts are the clamp sites of the writers as of perf/integ (8336080, 2143f3f);
# a merge or a later edit that drops one fails here even on a machine without a GPU. tools/cmerge_srchash.py prints the same module hashes for a tree-to-tree comparison.
import os as _os
import re as _re

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_CLAMP = _re.compile(r"clamp\([^;]*?-65504\.0f, 65504\.0f\)")
# module -> (Metal clamp sites, python-level saturations: `mx.clip(.., 65504)` / `sat(...)` calls with the limit)
_SITES = {"fuse": (3, 1), "pf_glue": (2, 0), "suglue": (3, 0), "stagek": (1, 1), "glue_fuse": (1, 0), "gluefuse": (1, 0)}


@pytest.mark.parametrize("mod", sorted(_SITES))
def test_stream_writers_keep_their_clamp(mod):
    src = open(_os.path.join(_ROOT, "kyojin_mlx", mod + ".py")).read()
    metal, py = _SITES[mod]
    assert len(_CLAMP.findall(src)) >= metal, f"{mod}: a Metal clamp site is gone (perf/integ has {metal})"
    assert len(_re.findall(r"mx\.clip\([^\n]*65504", src)) + len(_re.findall(r"\bsat\(", src)) - (1 if mod == "fuse" else 0) >= py


def test_python_stream_clips_stay():
    for mod, needle in (("ds_pf_mhc", "fuse.sat"), ("ablit", "mx.clip(")):
        assert needle in open(_os.path.join(_ROOT, "kyojin_mlx", mod + ".py")).read(), mod


@pytest.mark.parametrize("name", ["decfilter.py", "pf_bound.py"])
def test_new_lever_modules_write_no_stream(name):
    """cflatdec (dec_filter) and cflatpf (pf_idx_bound) select index keys: no Metal source of their own, no hidden-stream writer or collapse."""
    src = open(_os.path.join(_ROOT, "kyojin_mlx", name)).read()
    assert "metal_kernel" not in src and "kernel void" not in src
    for word in ("mhc_expand", "hc_pre", "hc_post", "expand_stage", "collapse"):
        assert word not in src, f"{name} mentions {word}"


def test_expert_gemm_sources_write_no_stream():
    """The NAX expert kernels (unroll, rowend, persist) and the gate / up epilogue write fp16 expert products (`Y`, the SwiGLU activation), never a stream: the stream is written by
    the expand kernels above, which clamp. No source the new levers build carries a stream store."""
    from kyojin_mlx import metal_nax as NX
    for args in ((True, "full", 2, True, "", False, True, False, 0, True, True), (True, "full", 1, True, "", False, False, False, 0, False, False)):
        src = NX.coop_source((3,), "mul1r2", 32, 2, 32, 64, *args)
        assert "65504" not in src and "hidden" not in src
