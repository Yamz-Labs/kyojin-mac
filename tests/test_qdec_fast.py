# SPDX-License-Identifier: Apache-2.0
"""Flag qdec_fast (KYOJIN_QDEC_FAST): the Q3_K gather / pair / glu GEMV decodes and multiplies in one pass (foreign_metal dot_q3k) instead of decoding 32 floats first.

CPU part (numpy, runs anywhere): the unpack + fold math of dot_q3k, written op for op as the Metal text (uint32 shifts / masks, the half2 magic 0x6400, the x sums, the scale / offset fold),
equals the float64 dec_q3k product for random units (the only difference is float rounding); the generated source of the flag off is the old one (no dot_q3k call, w[] decoded), the flag on has no w[]
and calls dot_q3k; other formats are untouched; the flag wiring and the kernel cache keys.
Metal part (KYOJIN_METAL_TESTS=1, Metal machine only): the pair / gather / glu kernels with the flag on equal the flag off and the float64 reference over 1..8 rows."""
import os
import re

import numpy as np
import pytest

from kyojin_mlx import flags, foreign_metal as FM

METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1"
needs_metal = pytest.mark.skipif(not METAL, reason="KYOJIN_METAL_TESTS=1 (Metal machine only)")
U32 = np.uint32


def _half_pair(u):
    """as_type<half2>(u) - half2(1024): the two 16-bit lanes of u as float16 (exact), minus 1024 in float16."""
    lo = (u & U32(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (u >> U32(16)).astype(np.uint16).view(np.float16)
    return (lo - np.float16(1024)).astype(np.float32), (hi - np.float16(1024)).astype(np.float32)


def dot_q3k_emul(qx, qy, hx, s0, s1, xv):
    """dot_q3k of foreign_metal.HDRK3, op for op (uint32 arithmetic; float32 products and sums in the Metal order: 4 accumulators)."""
    qx, qy, hx = U32(qx), U32(qy), U32(hx)
    ha = U32(((hx & U32(0xFF)) | ((hx & U32(0xFF00)) << U32(8))) << U32(2))
    hb = U32((((hx >> U32(16)) & U32(0xFF)) | ((hx >> U32(8)) & U32(0x00FF0000))) << U32(2))
    xv = xv.astype(np.float32)
    sx0, sx1 = np.float32(0), np.float32(0)
    for e in range(16):
        sx0 = np.float32(sx0 + xv[e])
        sx1 = np.float32(sx1 + xv[16 + e])
    a0 = a1 = b0 = b1 = np.float32(0)
    for e in range(8):
        ux = U32(((qx >> U32(2 * e)) & U32(0x00030003)) | ((ha >> U32(e)) & U32(0x00040004)) | U32(0x64006400))
        uy = U32(((qy >> U32(2 * e)) & U32(0x00030003)) | ((hb >> U32(e)) & U32(0x00040004)) | U32(0x64006400))
        x0l, x0h = _half_pair(ux)
        x1l, x1h = _half_pair(uy)
        a0 = np.float32(a0 + x0l * xv[e])
        a1 = np.float32(a1 + x0h * xv[8 + e])
        b0 = np.float32(b0 + x1l * xv[16 + e])
        b1 = np.float32(b1 + x1h * xv[24 + e])
    s0, s1 = np.float32(s0), np.float32(s1)
    return float(s0 * np.float32((a0 + a1) - np.float32(4) * sx0) + s1 * np.float32((b0 + b1) - np.float32(4) * sx1))


def dec_q3k_ref(qx, qy, hx, s0, s1, xv):
    """float64: w[e] = s0 * (2-bit + 4 * hi bit - 4), w[16 + e] = s1 * (...), dot with xv (dec_q3k + the 32-term sum of gemvk_source)."""
    w = np.zeros(32)
    for e in range(16):
        w[e] = s0 * (((qx >> (2 * e)) & 3) + (((hx >> e) & 1) << 2) - 4)
        w[16 + e] = s1 * (((qy >> (2 * e)) & 3) + (((hx >> (16 + e)) & 1) << 2) - 4)
    return float(np.dot(w, xv.astype(np.float64)))


def test_unpack_fold_matches_decode():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(400):
        qx, qy, hx = (int(v) for v in rng.integers(0, 2 ** 32, 3, dtype=np.uint64))
        d = float(np.float16(rng.uniform(1e-3, 3e-2)))
        s0, s1 = np.float32(d) * np.float32(int(rng.integers(-32, 32))), np.float32(d) * np.float32(int(rng.integers(-32, 32)))
        xv = (rng.standard_normal(32) * 0.7).astype(np.float16).astype(np.float32)
        ref = dec_q3k_ref(qx, qy, hx, float(s0), float(s1), xv)
        got = dot_q3k_emul(qx, qy, hx, s0, s1, xv)
        scale = max(1e-6, float(np.abs(xv).sum()) * 4 * (abs(float(s0)) + abs(float(s1))) / 2)
        worst = max(worst, abs(got - ref) / scale)
    assert worst < 5e-6, worst


def test_code_extremes():
    """All codes 0 (w = -4 s), all 7 (w = +3 s), x of one sign: the fold has no cancellation trouble."""
    xv = np.linspace(0.1, 3.0, 32).astype(np.float16).astype(np.float32)
    for q, h in ((0, 0), (0xFFFFFFFF, 0xFFFFFFFF), (0xAAAAAAAA, 0x0F0F0F0F), (0x55555555, 0xF0F0F0F0)):
        ref = dec_q3k_ref(q, q, h, 0.01, -0.02, xv)
        got = dot_q3k_emul(q, q, h, 0.01, -0.02, xv)
        assert abs(got - ref) <= 1e-5 * max(1.0, abs(ref)), (q, h, got, ref)


def test_half_magic_exact():
    """1024 + code is exact in half for every code 0..7 in both lanes, and the subtraction gives the code back."""
    for a in range(8):
        for b in range(8):
            u = U32(0x64006400 | a | (b << 16))
            lo, hi = _half_pair(u)
            assert (lo, hi) == (a, b)


def _norm(s):
    return re.sub(r"\s+", " ", s)


def test_source_off_and_on():
    old = os.environ.pop("KYOJIN_QDEC_FAST", None)
    try:
        assert not FM.qdec_fast("q3_k")
        off = FM.gemvk_source("q3_k", 2, True, False, False)
        assert "float w[NBR * 32];" in off and "dec_q3k(cu[j]" in off and "dot_q3k(" not in off
        os.environ["KYOJIN_QDEC_FAST"] = "1"
        assert FM.qdec_fast("q3_k") and not FM.qdec_fast("q4_k") and not FM.qdec_fast("q5_k")
        on = FM.gemvk_source("q3_k", 2, True, False, False)
        assert "float w[" not in on and "dec_q3k(cu[j]" not in on and "dot_q3k(cu[j], ul, xv, sx0, sx1)" in on and "sx0 +=" in on
        assert _norm(on) != _norm(off)
        for fmt in FM.FORMATSK:
            if fmt != "q3_k":
                assert FM.gemvk_source(fmt, 2, True, False, False) == FM._gemvk_source(fmt, 2, True, False, False, False)
        for glu, part in ((False, False), (True, False), (False, True)):          # glu staging and the partial-chunk edits still apply on the fast text
            t = FM.gemvk_source("q3_k", 2, True, glu, part)
            assert "dot_q3k(" in t and ("glu8(" in t) == glu and ("min(ul, UPR - 1u)" in t) == part
        assert "dot_q3k" in FM.hdrk("q3_k") and "dot_q3k" not in FM.hdrk("q5_k")
    finally:
        if old is None:
            os.environ.pop("KYOJIN_QDEC_FAST", None)
        else:
            os.environ["KYOJIN_QDEC_FAST"] = old


def test_off_text_is_the_old_text():
    """The flag off must give the text of the pre-flag source: the decode statement and the 32-term sum, with the same spacing."""
    off = FM.gemvk_source("q3_k", 2, True, False, False) if not FM.qdec_fast("q3_k") else None
    if off is None:
        pytest.skip("KYOJIN_QDEC_FAST is set in the environment")
    assert "UNR for (uint j = 0; j < NBR; ++j) dec_q3k(cu[j], ul, &w[j * 32u]);" in off
    assert ("UNR for (uint j = 0; j < NBR; ++j) { float s = 0.0f;\n          UNR for (uint e = 0; e < 32; ++e) s += w[j * 32u + e] * xv[e];\n"
            "          acc[j * R + r] += s; }") in off


def test_flag_wiring():
    f = flags.FLAGS["qdec_fast"]
    assert f.env == "KYOJIN_QDEC_FAST" and f.default == "0" and f.runtime
    old = os.environ.get(f.env)
    try:
        flags.set("qdec_fast", 1)
        assert FM.qdec_fast("q3_k") is True
        flags.set("qdec_fast", 0)
        assert FM.qdec_fast("q3_k") is False
    finally:
        if old is None:
            os.environ.pop(f.env, None)
        else:
            os.environ[f.env] = old


def test_kernel_cache_keys():
    pytest.importorskip("mlx.core")
    from kyojin_mlx import foreign_linear as FL
    src = open(FL.__file__).read()
    assert "(\"qf\",) if fast" in src and "_gemvk_kern_c(fmt, nbr, gather, glu, act, partial, fast)" in src and "fast = FM.qdec_fast(fmt)" in src
    assert '("kp" if part else "k") + ("q" if FM.qdec_fast(self.fmt) else "")' in src
    mu = open(os.path.join(os.path.dirname(FL.__file__), "moe_union.py")).read()
    assert '(("qf",) if qf else ())' in mu and "FM.qdec_fast(fmt)" in mu
    assert src.count('FM.PATHS["qdec_fast"] += 1') == 1 and mu.count('FM.PATHS["qdec_fast"] += 1') == 2      # fires-check counters


def test_fires_counter_declared():
    assert FM.PATHS == {"qdec_fast": FM.PATHS["qdec_fast"]} and isinstance(FM.PATHS["qdec_fast"], int)


# ------------------------------------------------------------------------------------------------------------------------------- Metal
def _q3_stack(E, n, k, rng):
    from kyojin_mlx import foreign_gguf as GF, foreign_linear as FL
    packs = []
    for _ in range(E):
        raw = rng.integers(0, 256, (n, k // 256 * 110), dtype=np.uint8)
        for b in range(k // 256):
            raw[:, b * 110 + 108:b * 110 + 110] = np.array([0x00, 0x1c], np.uint8)        # f16 0.0039
        packs.append(GF.to_pack("Q3_K", raw, k))
    return FL.DevPack(packs), packs


@needs_metal
@pytest.mark.parametrize("S", [1, 2, 4, 8, 16, 32])
def test_metal_pair_on_vs_off_vs_ref(S, monkeypatch):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL, foreign_pack as FP
    rng = np.random.default_rng(100 + S)
    E, n, k = 16, 512, 1024
    g, gp = _q3_stack(E, n, k, rng)
    u, up = _q3_stack(E, n, k, rng)
    assert g.pair_ok(u)
    x = mx.array((rng.standard_normal((S, k)) * 0.5).astype(np.float16))
    e32 = mx.array(rng.integers(0, E, S).astype(np.uint32))
    tok = mx.arange(S).astype(mx.uint32)
    monkeypatch.delenv("KYOJIN_QDEC_FAST", raising=False)
    y_off = g.gather_pair_raw(u, x, tok, e32)
    monkeypatch.setenv("KYOJIN_QDEC_FAST", "1")
    y_on = g.gather_pair_raw(u, x, tok, e32)
    y1_on = g.gather(x, tok, e32)                                                          # the single gather kernel (cache key of _gemvk_kern)
    mx.eval(y_off, y_on, y1_on)
    a, b = np.array(y_off.astype(mx.float32)), np.array(y_on.astype(mx.float32))
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.abs(a).max() > 0
    assert np.abs(a - b).max() <= 2e-3 * np.abs(a).max()
    assert np.abs(np.array(y1_on.astype(mx.float32)) - b[:S]).max() <= 2e-3 * np.abs(a).max()
    xn, ei = np.array(x).astype(np.float64), np.array(e32)
    for pk, off in ((gp, 0), (up, S)):
        for s in range(S):
            ref = FP.dequant(pk[ei[s]]).astype(np.float64) @ xn[s]
            assert np.abs(b[off + s] - ref).max() <= 3e-3 * max(1e-3, np.abs(ref).max())
    monkeypatch.delenv("KYOJIN_QDEC_FAST", raising=False)


@needs_metal
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_metal_switchglu_on_vs_off(rows, monkeypatch):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    rng = np.random.default_rng(7 + rows)
    E, K, H, F = 16, 8, 512, 512
    g, _ = _q3_stack(E, F, H, rng)
    u, _ = _q3_stack(E, F, H, rng)
    d, _ = _q3_stack(E, H, F, rng)
    sw = FL.ForeignSwitchGLU(g, u, d)
    x = mx.array(rng.standard_normal((rows, 1, H)).astype(np.float32) * 0.5).astype(mx.float16)
    inds = mx.array(np.stack([rng.choice(E, K, replace=False) for _ in range(rows)])[:, None, :].astype(np.uint32))
    monkeypatch.setattr(FL, "GATEUP_PAIR", True)
    monkeypatch.delenv("KYOJIN_QDEC_FAST", raising=False)
    ref = sw(x, inds)
    monkeypatch.setenv("KYOJIN_QDEC_FAST", "1")
    out = sw(x, inds)
    mx.eval(ref, out)
    r, o = np.array(ref.astype(mx.float32)), np.array(out.astype(mx.float32))
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(o))
    assert np.abs(r - o).max() <= 2e-2 * max(1e-3, float(np.abs(r).max()))
