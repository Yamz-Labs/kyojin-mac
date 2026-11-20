# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_FOREIGN_CASTFREE (flags.foreign_castfree): a one-byte stored linear (fp8 / q8_0 / fp8b) on the gemv8 route reads a float32 / bfloat16 x and writes
a float32 y inside the GEMV (foreign_metal.gemv8_source xin / yout), instead of x.astype(float16) -> gemv8 -> y.astype(out_dtype) (2 AsType launches per call).

Expected numeric difference vs stock: NONE (bit-identical). The staging rounds x to half exactly as the cast did (round to nearest even), the products and the
order of the sums are the stock kernel's, and y is the stock fp16-rounded value widened to float32 (as astype(float32) did).

CPU part (runs anywhere, no kernel is launched: the graphs are built lazily and read with mx.export_to_dot, never evaluated):
  * the stock kernel text is byte-identical (defaults, and against the base branch's file when git has it),
  * the variant text has exactly the staging / output replacements,
  * the route predicate DevPack.cf_ok(R) == "apply() of R rows ends in _gemv8",
  * the lazy graph of a float32-out ForeignLinear on a float32 x: 2 AsType with the flag off, 0 with it on, the same custom kernel count.
Metal part (KYOJIN_CASTFREE_METAL=1 on a Metal machine; the c80 cell runs it): flag on == flag off, np.array_equal on the raw bits, every format x input dtype x
output dtype x rows 1 / 2 / 8 x k (full and partial last 512-chunk), with fp16-overflowing and subnormal inputs, plus ForeignLinear end to end with a bias.
"""
import os
import re
import subprocess
import tempfile

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_linear as FL, foreign_metal as FM, foreign_pack as FP

METAL = os.environ.get("KYOJIN_CASTFREE_METAL") == "1" and mx.metal.is_available()
needs_metal = pytest.mark.skipif(not METAL, reason="Metal run only with KYOJIN_CASTFREE_METAL=1 (the c80 cell)")
FORMATS = ("fp8", "q8_0", "fp8b")
N = 256
BASE = os.environ.get("KYOJIN_CASTFREE_BASE", "hy4-a")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pack(fmt, n, k, rng):
    q = rng.integers(0, 256, (n, k), dtype=np.uint8)
    if fmt == "q8_0":
        return FP.Pack("q8_0", n, k, q.view(np.int8).view(np.uint8), rng.uniform(1e-3, 3e-3, (n, k // 32)).astype(np.float16), np.ones(n, np.float32))
    q[(q & 127) == 127] = 0x38                                     # no NaN codes
    if fmt == "fp8":
        return FP.Pack("fp8", n, k, q, None, rng.uniform(5e-5, 2e-4, n).astype(np.float32))
    g = rng.uniform(64.0, 127.0, (-(-n // 128), -(-k // 128))).astype(np.float32)
    return FP.Pack("fp8b", n, k, q, g[np.arange(n) // 128][:, (np.arange(k // 32) * 32) // 128].astype(np.float16), np.float32(2.0 ** -20))


class _flag:
    def __init__(self, on): self.on = on
    def __enter__(self): self.old, FL.CASTFREE = FL.CASTFREE, self.on
    def __exit__(self, *a): FL.CASTFREE = self.old


def _census(y):
    fd, path = tempfile.mkstemp(suffix=".dot"); os.close(fd)
    try:
        mx.export_to_dot(path, y)
        labels = re.findall(r'\[label ?= ?"([^"]*)", shape=rectangle\]', open(path).read())
    finally:
        os.unlink(path)
    return labels


# ------------------------------------------------------------------ CPU
def test_stock_text_unchanged():
    for fmt in FORMATS:
        for nbr in (1, 2, 4):
            for g in (False, True):
                for u in (False, True):
                    s = FM.gemv8_source(fmt, nbr, g, u)
                    assert s == FM.gemv8_source(fmt, nbr, g, u, "h", "h")
                    assert "g8_pack" not in s and "device float* YP" not in s
    try:
        old = subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:kyojin_mlx/foreign_metal.py"], capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip(f"base {BASE} not readable with git")
    ns = {"__name__": "fm_stock", "__package__": "kyojin_mlx"}
    exec(compile(old, "fm_stock", "exec"), ns)
    for fmt in FORMATS:
        for nbr in (1, 2, 4):
            for g in (False, True):
                for u in (False, True):
                    assert FM.gemv8_source(fmt, nbr, g, u) == ns["gemv8_source"](fmt, nbr, g, u), (fmt, nbr, g, u)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("xin,yout", [("f", "f"), ("f", "h"), ("b", "f"), ("b", "h"), ("h", "f")])
def test_variant_text(fmt, xin, yout):
    s, h = FM.gemv8_source(fmt, 2, False, False, xin, yout), FM.gemv8_source(fmt, 2)
    assert s.count("g8_pack_f(") == (xin == "f") and s.count("g8_pack_b(") == (xin == "b")
    assert ("device float* YP = y;" in s) == (yout == "f") and ("as_type<ushort>(half(" in s) == (yout == "f")
    assert ("device const half* XP" in s) == (xin == "h")
    # everything else (decode, products, reduction) is the stock text
    strip = lambda t: [l for l in t.splitlines() if "XP" not in l and "YP" not in l]
    assert strip(s) == strip(h)
    with pytest.raises(AssertionError):
        FM.gemv8_source(fmt, 2, True, False, xin, yout)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("k", [576, 1536, 3072])
def test_route_matches_apply(fmt, k, monkeypatch):
    """cf_ok(R) is True exactly when apply() of R rows reaches _gemv8 (apply is run lazily on CPU with _gemv8 stubbed; nothing is evaluated)."""
    with mx.stream(mx.cpu):
        d = FL.DevPack(_pack(fmt, N, k, np.random.default_rng(k)))
        hit = []
        monkeypatch.setattr(FL.DevPack, "_gemv8", lambda self, x2, nbr=None: (hit.append(1), mx.zeros((x2.shape[0], self.n), mx.float16))[1])
        for R in (1, 2, 4, 8, 9, 16):
            hit.clear()
            try:
                d.apply(mx.zeros((R, k), mx.float16))
            except Exception:
                pass                                          # a non-gemv8 route may refuse a CPU stream: it is not gemv8 either way
            assert d.cf_ok(R) == bool(hit), (fmt, k, R, d.cf_ok(R), hit)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("xd,od,casts_off", [(mx.float32, mx.float32, 2), (mx.bfloat16, mx.float32, 2), (mx.float16, mx.float32, 1), (mx.float32, mx.float16, 1)])
def test_graph_casts_removed(fmt, xd, od, casts_off):
    with mx.stream(mx.cpu):
        lin = FL.ForeignLinear(_pack(fmt, N, 1536, np.random.default_rng(3)), out_dtype=od)
        x = mx.zeros((1, 1, 1536), xd)
        with _flag(False):
            off = _census(lin(x))
        with _flag(True):
            on = _census(lin(x))
    assert off.count("AsType") == casts_off and on.count("AsType") == 0, (off, on)
    assert len(off) - len(on) == casts_off
    assert sum("Custom" in l or "kyojin" in l.lower() for l in on) == sum("Custom" in l or "kyojin" in l.lower() for l in off) >= 1, (off, on)


def test_flag_declared_default_off():
    from kyojin_mlx import flags as F
    f = F.FLAGS["foreign_castfree"]
    assert f.env == "KYOJIN_FOREIGN_CASTFREE" and f.default == "0" and f.runtime and f.attrs[0][:2] == ("foreign_linear", "CASTFREE")


# ------------------------------------------------------------------ Metal (bit-equality)
def _x(rng, R, k, dt, overflow=False):
    a = rng.normal(0, 1, (R, k)).astype(np.float32)
    a[:, ::97] *= 30.0                                                 # a wider range, the outputs stay well inside fp16
    a[:, 5::89] *= 1e-6                                                # fp16 subnormals
    a[:, 7] = 65000.0                                                  # one column near the fp16 top (f32 -> 64992, bf16 65024 -> 65024): finite in every dtype
    if overflow:
        a[0, 3] = 1e5                                                  # beyond fp16: the cast saturates to inf, the staging must too (row 0 all inf / nan)
    return mx.array(a).astype(dt)


def _bits(y):
    a = np.array(y)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint16)


@needs_metal
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("k", [576, 1536, 3072, 4096])
def test_metal_bit_equal(fmt, k):
    rng = np.random.default_rng(k + len(fmt))
    d = FL.DevPack(_pack(fmt, N, k, rng))
    for R in (1, 2, 8):
        if not d.cf_ok(R):
            continue
        for xd in (mx.float32, mx.bfloat16, mx.float16):
            for ov in (False, True):
                x = _x(rng, R, k, xd, ov)
                for od in (mx.float32, mx.float16):
                    ref = d._gemv8(x.astype(mx.float16)).astype(od)
                    y = d.gemv8_cf(x, od)
                    mx.eval(ref, y)
                    assert y.dtype == od and np.array_equal(_bits(y), _bits(ref)), (fmt, k, R, xd, od, ov, int((_bits(y) != _bits(ref)).sum()))
                    if not ov:
                        assert np.isfinite(np.array(y.astype(mx.float32))).mean() > 0.5        # the normal case really compares finite values


@needs_metal
@pytest.mark.parametrize("fmt", FORMATS)
def test_metal_linear_end_to_end(fmt):
    rng = np.random.default_rng(11)
    k = 3072
    bias = mx.array(rng.normal(0, 1, N).astype(np.float32))
    for od, xd in ((mx.float32, mx.float32), (mx.float32, mx.bfloat16), (mx.float16, mx.float32)):
        lin = FL.ForeignLinear(_pack(fmt, N, k, rng), bias=bias, out_dtype=od)
        x = _x(rng, 1, k, xd).reshape(1, 1, k)
        with _flag(False):
            a = lin(x); mx.eval(a)
        with _flag(True):
            b = lin(x); mx.eval(b)
        assert a.dtype == b.dtype == od and np.array_equal(_bits(a), _bits(b)), (fmt, od, xd)
