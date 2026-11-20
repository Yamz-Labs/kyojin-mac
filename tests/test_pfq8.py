"""BP5-Q8: int8-activation routed-expert GEMM (foreign_pfq8). Host side (pack_i8 / quant_act) runs on the CPU; gmm_i8 / mlp need the Metal GPU."""
import os, sys
import numpy as np
import pytest
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def _e4(b):
    b = b.astype(np.int32) & 127
    e, m = b >> 3, (b & 7).astype(np.float32)
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8) * 2.0 ** (e - 7.0))
    return np.where(b == 127, 0.0, v).astype(np.float32)


def ref_weight(dp):
    """Independent numpy decode of a stacked nvfp4 pack: float32 [E, n, k]."""
    qs = np.array(dp.qs)
    E, n, kw = qs.shape
    nib = (qs[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15
    x2 = KV[nib.reshape(E, n, kw * 8)]
    sc = _e4(np.array(dp.meta).view(np.uint8).reshape(E, n, -1))
    return x2 * 0.5 * np.repeat(sc, 16, axis=-1) * np.array(dp.gs).reshape(E, 1, 1)


def test_pack_i8_host(monkeypatch):
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_pfq8 as Q8
    with mx.stream(mx.cpu):
        dp = rand_pack(3, 16, 256, 1)
        wi, ws = Q8.pack_i8(dp)
        assert wi.dtype == mx.int8 and ws.dtype == mx.float16 and wi.shape == (3, 16, 256) and ws.shape == (3, 16)
        w = ref_weight(dp)
        deq = np.array(Q8.dequant_i8(wi, ws))
        rowmax = np.abs(w).max(-1, keepdims=True)
        assert (np.abs(deq - w) <= rowmax / 127.0 * 0.5001 + rowmax * 6e-4).all()          # half an int8 step per row + fp16 rounding of the scale (2^-11 x 127 steps)
        assert np.sqrt(((deq - w) ** 2).mean()) <= 0.01 * np.sqrt((w ** 2).mean())
        assert Q8.pack_i8(dp)[0] is wi                                               # cached on the pack


def test_quant_act_host():
    import kyojin_mlx.foreign_pfq8 as Q8
    with mx.stream(mx.cpu):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((40, 512)) * rng.uniform(0.01, 3, (40, 1))).astype(np.float16)
        x[3] = 0
        c, s = Q8.quant_act(mx.array(x))
        c, s = np.array(c), np.array(s)
        assert c.dtype == np.int8 and np.abs(c).max() <= 127
        assert np.allclose(s[np.arange(40) != 3], np.abs(x.astype(np.float32)).max(-1)[np.arange(40) != 3] / 127.0, rtol=1e-6)
        deq = c.astype(np.float32) * s[:, None]
        assert np.abs(deq - x.astype(np.float32)).max(-1).max() <= (s * 0.5001).max()
        assert not np.any(c[3])


pytestmark_gpu = pytest.mark.skipif(mx.default_device() != mx.gpu or not mx.metal.is_available(), reason="needs the Metal GPU")


@pytestmark_gpu
@pytest.mark.parametrize("E,H,I,T,K", [(12, 1024, 512, 700, 4), (8, 512, 256, 130, 2)])
def test_mlp_matches_fp16(monkeypatch, E, H, I, T, K):
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_linear as FL, kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_pfq8 as Q8
    gate, up, down = rand_pack(E, I, H, 1), rand_pack(E, I, H, 2), rand_pack(E, H, I, 3)
    glu = FL.ForeignSwitchGLU(gate, up, down, limit=10.0)
    rng = np.random.default_rng(0)
    x2 = mx.array((rng.standard_normal((T, H)) * 0.005).astype(np.float16))
    e = np.stack([rng.permutation(E)[:K] for _ in range(T)]).reshape(-1).astype(np.uint32)
    e[:150] = 3                                                   # one big run, ragged tiles
    e32 = mx.array(e)
    monkeypatch.setenv("KYOJIN_KQ_GMM_GLM", "1")
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8_MIN", "1")
    ref = KQ.mlp(glu, x2, e32, K, 32)
    y = Q8.mlp(glu, x2, e32, K)
    r, n = np.array(ref).astype(np.float32), np.array(y).astype(np.float32)
    assert not np.isnan(n).any()
    rel = np.sqrt(((r - n) ** 2).mean()) / np.sqrt((r ** 2).mean())
    assert rel <= 0.03, rel                                       # int8 acts (2^-7) + int8 weights (per-row scale): documented bound, see docs/PFQ8.md


def ref_codes(dp):
    """numpy model of the in-kernel int8 weight code: Q = floor(64 * e4 * (127 / 6) / emax_row + 0.5), code = rint(doubled e2m1 * Q / 128) (round half even), plus (emax [E, n])."""
    qs = np.array(dp.qs)
    E, n, kw = qs.shape
    nib = (qs[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15
    x2 = KV[nib.reshape(E, n, kw * 8)]
    e4 = _e4(np.array(dp.meta).view(np.uint8).reshape(E, n, -1))
    em = e4.max(-1)
    em = np.where(em > 0, em, 1.0).astype(np.float32)
    fm = (np.float32(127.0) / np.float32(6.0)) / em
    q = np.floor((e4 * fm[..., None]).astype(np.float32) * np.float32(64.0) + np.float32(0.5))
    q = np.repeat(q, 16, axis=-1)
    return np.clip(np.rint(x2 * q / 128.0), -127, 127), em


def test_swar_decode_rounds_half_even():
    """The kernel's integer decode (lane = T[byte] * Q + R2, then lane += (lane >> 7) & 1, code = bits 7..14) against np.rint(v Q / 128), for every Q the decode can see and every stored byte,
    with a ties census: the v1 floor((v Q + 64) / 128) rounded every tie UP (0.7-0.9 % of the elements, +0.004 code of mean bias)."""
    import kyojin_mlx.foreign_pfq8 as Q8
    lut = np.array([(Q8.KVALUES_FP4[n & 15] + 12) for n in range(16)], np.int64)
    byte = np.arange(256)
    lo, hi = lut[byte & 15], lut[byte >> 4]
    T = (lo | (hi << 16)).astype(np.uint64)
    vlo, vhi = Q8.KVALUES_FP4[byte & 15], Q8.KVALUES_FP4[byte >> 4]
    nt = 0
    for Q in range(0, 1356):                                        # Q = round(64 * e4 * 127 / (6 emax)) <= 1355
        R2 = np.uint64((63 + 32768 - 12 * Q) * 0x00010001)
        z = (T * np.uint64(Q) + R2) & np.uint64(0xFFFFFFFF)
        z = (z + ((z >> np.uint64(7)) & np.uint64(0x00010001))) & np.uint64(0xFFFFFFFF)
        for v, sh in ((vlo, 0), (vhi, 16)):
            lane = (z >> np.uint64(sh)) & np.uint64(0xFFFF)
            assert lane.max() < 65536
            code = ((lane >> np.uint64(7)) & np.uint64(0xFF)).astype(np.int64)
            code = np.where(code >= 128, code - 256, code)
            want = np.rint(v * Q / 128.0).astype(np.int64)
            assert (code == want).all(), (Q, np.nonzero(code != want)[0][:4])
            nt += int(((v * Q + 64) % 128 == 0).sum())
    assert nt > 0                                                       # the grid does contain ties (else this test proves nothing)


def test_ok_tile_and_why_not_reasons(monkeypatch):
    from types import SimpleNamespace as NS
    import kyojin_mlx.foreign_pfq8 as Q8
    p = lambda **k: NS(**{**dict(fmt="nvfp4", E=8, rowgs=False, k=4096, n=2048), **k})
    assert Q8.ok_tile(p())
    assert not Q8.ok_tile(p(k=384)) and not Q8.ok_tile(p(k=640)) and not Q8.ok_tile(p(n=100)) and not Q8.ok_tile(p(E=0)) and not Q8.ok_tile(p(rowgs=True)) and not Q8.ok_tile(p(fmt="q4"))
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8", "1")
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8_MIN", "100")
    glu = lambda g=None, u=None, d=None, limit=10.0: NS(g=g or p(), u=u or p(), d=d or p(k=2048, n=4096), limit=limit)
    assert Q8.why_not(glu(), 1000) == "?"                               # eligible by shape: mlp() would run
    assert Q8.why_not(glu(), 10) == "slots<min"
    assert Q8.why_not(glu(limit=None), 1000) == "no_swiglu_limit"
    assert Q8.why_not(glu(d=p(k=2176, n=4096)), 1000) == "down:k%256"       # 2176 = 17 * 128: passes the BK check, not the quant kernels' k % 256
    assert Q8.why_not(glu(u=p(n=100)), 1000) == "up:n%64"
    assert Q8.why_not(glu(g=p(rowgs=True)), 1000) == "gate:rowgs"
    assert Q8.why_not(glu(g=p(E=0)), 1000) == "gate:not_experts"
    assert Q8.why_not(glu(g=p(fmt="q4")), 1000) == "gate:fmt=q4"
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8", "0")
    assert Q8.why_not(glu(), 1000) == "flag_off"


def test_flag_registered():
    from kyojin_mlx import flags as FL
    f = FL.FLAGS["glm_pf_q8"]
    assert f.env == "KYOJIN_GLM_PF_Q8" and f.default == "0" and f.runtime


def test_row_emax_is_cached_on_the_resident_pack():
    """flag glmtile: the prefill path reads a fresh untiled() copy per call; the cache must live on the resident pack (one build, no per-call mx.eval)."""
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_pfq8 as Q8
    with mx.stream(mx.cpu):
        root = rand_pack(3, 16, 256, 1)
        def copy():
            v = object.__new__(type(root)); v.__dict__.update(root.__dict__); v._parent = root
            return v
        c1, c2 = copy(), copy()
        em1 = Q8.row_emax(c1)
        assert Q8.row_emax(c2) is em1 and Q8.row_fm(c2) is Q8.row_fm(c1) and Q8.row_emax(root) is em1
        assert not hasattr(c1, "_pfq8_em") or c1.__dict__.get("_pfq8_em") is None or c1._pfq8_em is root._pfq8_em
        np.testing.assert_allclose(np.array(Q8.row_fm(root)) * np.array(em1), 127.0 / 6.0, rtol=1e-6)


def _check_blocks(codes, sc, x, qb):
    T, K = x.shape
    sc = np.asarray(sc)
    assert sc.shape == (T, K // qb)
    ref = np.abs(x.astype(np.float32)).reshape(T, K // qb, qb).max(-1)
    nz = ref > 0
    assert np.allclose(sc[nz], ref[nz] / 127.0, rtol=1e-6) and (sc[~nz] == 1.0).all()
    deq = codes.astype(np.float32).reshape(T, K // qb, qb) * sc[..., None]
    assert np.abs(deq - x.astype(np.float32).reshape(T, K // qb, qb)).max() <= sc.max() * 0.5001


def test_quant_act_blocks_host():
    import kyojin_mlx.foreign_pfq8 as Q8
    with mx.stream(mx.cpu):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((20, 1024)) * 0.1).astype(np.float16)
        x[5, 700] = 40.0                                                # one outlier: only ITS block's scale grows
        x[7] = 0
        for qb in (128, 256):
            c, s = Q8.quant_act(mx.array(x), qb)
            _check_blocks(np.array(c), np.array(s), x, qb)
        c, s = Q8.quant_act(mx.array(x), 128)
        s = np.array(s)
        assert s[5, 700 // 128] > 100 * s[5, 0] and np.isclose(s[5, 0], np.abs(x[5, :128].astype(np.float32)).max() / 127.0, rtol=1e-6)


pytestmark_gpu = pytest.mark.skipif(mx.default_device() != mx.gpu or not mx.metal.is_available(), reason="needs the Metal GPU")


@pytestmark_gpu
@pytest.mark.parametrize("E,H,I,T,K", [(12, 1024, 512, 700, 4), (8, 512, 256, 130, 2)])
def test_mlp_matches_fp16(monkeypatch, E, H, I, T, K):
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_linear as FL, kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_pfq8 as Q8
    gate, up, down = rand_pack(E, I, H, 1), rand_pack(E, I, H, 2), rand_pack(E, H, I, 3)
    glu = FL.ForeignSwitchGLU(gate, up, down, limit=10.0)
    rng = np.random.default_rng(0)
    x2 = mx.array((rng.standard_normal((T, H)) * 0.005).astype(np.float16))
    e = np.stack([rng.permutation(E)[:K] for _ in range(T)]).reshape(-1).astype(np.uint32)
    e[:150] = 3                                                   # one big run, ragged tiles
    e32 = mx.array(e)
    monkeypatch.setenv("KYOJIN_KQ_GMM_GLM", "1")
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8_MIN", "1")
    ref = KQ.mlp(glu, x2, e32, K, 32)
    y = Q8.mlp(glu, x2, e32, K)
    r, n = np.array(ref).astype(np.float32), np.array(y).astype(np.float32)
    assert not np.isnan(n).any()
    rel = np.sqrt(((r - n) ** 2).mean()) / np.sqrt((r ** 2).mean())
    assert rel <= 0.03, rel                                       # int8 acts (2^-7) + int8 weights (per-row scale): documented bound


@pytestmark_gpu
def test_mlp_down_fp16_variant(monkeypatch):
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_linear as FL, kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_pfq8 as Q8
    E, H, I, T, K = 8, 512, 256, 400, 2
    glu = FL.ForeignSwitchGLU(rand_pack(E, I, H, 1), rand_pack(E, I, H, 2), rand_pack(E, H, I, 3), limit=10.0)
    rng = np.random.default_rng(0)
    x2 = mx.array((rng.standard_normal((T, H)) * 0.005).astype(np.float16))
    e32 = mx.array(np.stack([rng.permutation(E)[:K] for _ in range(T)]).reshape(-1).astype(np.uint32))
    monkeypatch.setenv("KYOJIN_KQ_GMM_GLM", "1")
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8_MIN", "1")
    ref = np.array(KQ.mlp(glu, x2, e32, K, 32)).astype(np.float32)
    r8 = np.array(Q8.mlp(glu, x2, e32, K)).astype(np.float32)
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8_DOWN", "fp16")
    rf = np.array(Q8.mlp(glu, x2, e32, K)).astype(np.float32)
    rel = lambda a: np.sqrt(((a - ref) ** 2).mean() / (ref ** 2).mean())
    assert rel(rf) <= 0.03 and rel(rf) <= rel(r8) * 1.05, (rel(rf), rel(r8))


def _exact_ref(xc, xs, wc, em, gs, off, qb):
    S, Kd = xc.shape
    E, N = em.shape
    ref = np.zeros((S, N), np.float64)
    for ex in range(E):
        r = slice(off[ex], off[ex + 1])
        for b in range(Kd // qb):
            part = xc[r, b * qb:(b + 1) * qb].astype(np.float64) @ wc[ex][:, b * qb:(b + 1) * qb].T.astype(np.float64)
            ref[r] += part * xs[r, b, None]
        ref[r] *= em[ex][None, :] * gs[ex] * (6.0 / 127.0)
    return ref


@pytestmark_gpu
@pytest.mark.parametrize("scat", [False, True])
@pytest.mark.parametrize("E,N,Kd,S", [(5, 128, 256, 300), (5, 128, 640, 300), (3, 64, 2048, 257), (4, 128, 4096, 520)])
def test_gmm_exact_model(scat, E, N, Kd, S):
    """gmm_i8 against a numpy model of the SAME int8 arithmetic with one activation scale per (row, K block): isolates kernel bugs from the quantization error.
    K = 640 (5 K steps, odd), 2048 / 4096 (16 / 32 steps): weight buffer reuse, scale buffer reuse, accumulation across many blocks."""
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_pfq8 as Q8
    dp = rand_pack(E, N, Kd, 5)
    rng = np.random.default_rng(1)
    e = np.sort(rng.integers(0, E, S)).astype(np.uint32)
    e[:70] = 1
    e = np.sort(e)
    order, tb, off = KQ.csort(mx.array(e), E, Q8.BM)
    xc = rng.integers(-127, 128, (S, Kd)).astype(np.int8)
    xs = (2.0 ** rng.integers(-12, -4, (S, Kd // Q8.QB))).astype(np.float32)          # scales differ per block: a wrong block index shows at once
    ordn = np.array(order)
    xk = xc.reshape(S, -1, 4)[..., [0, 2, 1, 3]].reshape(S, -1)
    y = np.array(Q8.gmm_i8(mx.array(xk), mx.array(xs), dp, tb, off, S, orow=order if scat else None)).astype(np.float32)
    wc, em = ref_codes(dp)
    ref = _exact_ref(xc, xs, wc, em, np.array(dp.gs), np.array(off), Q8.QB)
    if scat:
        out = np.zeros_like(ref)
        out[ordn] = ref
        ref = out
    assert np.abs(y - ref).max() <= 2e-3 * np.abs(ref).max() + 1e-4, np.abs(y - ref).max() / np.abs(ref).max()


@pytestmark_gpu
@pytest.mark.parametrize("qb", [128, 256])
def test_quant_rows_matches_quant_act(qb):
    import kyojin_mlx.foreign_pfq8 as Q8
    rng = np.random.default_rng(4)
    x = rng.standard_normal((53, 1024)).astype(np.float16) * 0.1
    x[3] = 0
    x[9, 333] = 30.0
    c, s = Q8.quant_rows(mx.array(x), qb)
    rc, rs = Q8.quant_act(mx.array(x), qb)
    assert np.allclose(np.array(s), np.array(rs), rtol=1e-6)
    assert np.abs(np.array(c).astype(np.int32) - np.array(rc).astype(np.int32)).max() <= 1
    _check_blocks(np.array(c), np.array(s), x, qb)


@pytestmark_gpu
@pytest.mark.parametrize("qb", [128, 256])
def test_silu_quant_matches_two_step(qb):
    import kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_pfq8 as Q8
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((37, 1024)) * 4).astype(np.float16)
    b = (rng.standard_normal((37, 1024)) * 4).astype(np.float16)
    b[2, 100] = 60.0
    a, b = mx.array(a), mx.array(b)
    for lim in (None, 10.0):
        c, s = Q8.silu_quant(a, b, lim, qb)
        rc, rs = Q8.quant_act(KQ.silu_perm(a, b, mx.arange(1024, dtype=mx.int32), lim), qb)
        rc = Q8.perm4(rc)
        assert np.allclose(np.array(s), np.array(rs), rtol=1e-5)
        assert np.abs(np.array(c).astype(np.int32) - np.array(rc).astype(np.int32)).max() <= 1


@pytestmark_gpu
def test_block_scales_beat_row_scale_on_outlier_rows(monkeypatch):
    """The v1 failure: ONE scale per row, so an outlier channel (sink token / SwiGLU spike) costs every other element of the row its resolution. Per K block, the error of a
    row with amax / rms = 50 stays near the Gaussian row's."""
    import kyojin_mlx.foreign_pfq8 as Q8
    rng = np.random.default_rng(6)
    x = rng.standard_normal((64, 4096)).astype(np.float32)
    x[:, 1000] = 50.0 * x.std(1)
    xm = mx.array(x.astype(np.float16))
    c1, s1 = Q8.quant_act(xm)
    c2, s2 = Q8.quant_rows(xm, 128)
    e1 = np.sqrt(((np.array(c1).astype(np.float32) * np.array(s1)[:, None] - x) ** 2).sum(1) / (x ** 2).sum(1)).mean()
    e2 = np.sqrt(((np.array(c2).astype(np.float32).reshape(64, 32, 128) * np.array(s2)[..., None] - x.reshape(64, 32, 128)) ** 2).reshape(64, -1).sum(1) / (x ** 2).sum(1)).mean()
    assert e2 < 0.45 * e1, (e1, e2)


@pytestmark_gpu
@pytest.mark.parametrize("scat", [False, True])
@pytest.mark.parametrize("E,N,Kd,S", [(5, 128, 256, 300), (3, 64, 2048, 257), (4, 128, 4096, 520)])
def test_gmm_exact_model_row_scale(scat, E, N, Kd, S):
    """qb == K (flag pf_gemm_i8_row): ONE scale per row, the int32 partial runs over the whole K (16 / 32 steps): exact numpy model of the same arithmetic."""
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_pfq8 as Q8
    dp = rand_pack(E, N, Kd, 5)
    rng = np.random.default_rng(1)
    e = np.sort(rng.integers(0, E, S)).astype(np.uint32)
    e[:70] = 1
    e = np.sort(e)
    order, tb, off = KQ.csort(mx.array(e), E, Q8.BM)
    xc = rng.integers(-127, 128, (S, Kd)).astype(np.int8)
    xs = (2.0 ** rng.integers(-12, -4, (S, 1))).astype(np.float32)
    ordn = np.array(order)
    xk = xc.reshape(S, -1, 4)[..., [0, 2, 1, 3]].reshape(S, -1)
    y = np.array(Q8.gmm_i8(mx.array(xk), mx.array(xs), dp, tb, off, S, orow=order if scat else None, qb=Kd)).astype(np.float32)
    wc, em = ref_codes(dp)
    ref = _exact_ref(xc, xs, wc, em, np.array(dp.gs), np.array(off), Kd)
    if scat:
        out = np.zeros_like(ref)
        out[ordn] = ref
        ref = out
    assert np.abs(y - ref).max() <= 2e-3 * np.abs(ref).max() + 1e-4, np.abs(y - ref).max() / np.abs(ref).max()


@pytestmark_gpu
def test_row_quant_kernels_match_reference():
    import kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_pfq8 as Q8
    rng = np.random.default_rng(4)
    x = rng.standard_normal((53, 1024)).astype(np.float16) * 0.1
    x[3] = 0
    x[9, 333] = 30.0
    c, s = Q8.quant_rows(mx.array(x), 1024)
    rc, rs = Q8.quant_act(mx.array(x))
    assert tuple(s.shape) == (53, 1) and np.allclose(np.array(s)[:, 0], np.array(rs), rtol=1e-6)
    assert np.abs(np.array(c).astype(np.int32) - np.array(rc).astype(np.int32)).max() <= 1
    a = mx.array((rng.standard_normal((37, 1024)) * 4).astype(np.float16))
    b = mx.array((rng.standard_normal((37, 1024)) * 4).astype(np.float16))
    c, s = Q8.silu_quant(a, b, 10.0, 1024)
    rc, rs = Q8.quant_act(KQ.silu_perm(a, b, mx.arange(1024, dtype=mx.int32), 10.0))
    assert tuple(s.shape) == (37, 1) and np.allclose(np.array(s)[:, 0], np.array(rs), rtol=1e-5)
    assert np.abs(np.array(c).astype(np.int32) - np.array(Q8.perm4(rc)).astype(np.int32)).max() <= 1


@pytestmark_gpu
def test_mlp_row_flag_matches_fp16(monkeypatch):
    from g2b_bench import rand_pack
    import kyojin_mlx.foreign_linear as FL, kyojin_mlx.foreign_kqgmm as KQ, kyojin_mlx.foreign_pfq8 as Q8
    E, H, I, T, K = 12, 1024, 512, 700, 4
    glu = FL.ForeignSwitchGLU(rand_pack(E, I, H, 1), rand_pack(E, I, H, 2), rand_pack(E, H, I, 3), limit=10.0)
    rng = np.random.default_rng(0)
    x2 = mx.array((rng.standard_normal((T, H)) * 0.005).astype(np.float16))
    e32 = mx.array(np.stack([rng.permutation(E)[:K] for _ in range(T)]).reshape(-1).astype(np.uint32))
    monkeypatch.setenv("KYOJIN_KQ_GMM_GLM", "1")
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8_MIN", "1")
    monkeypatch.delenv("KYOJIN_PF_GEMM_I8_ROW", raising=False)
    assert not Q8.row_scale()
    ref = np.array(KQ.mlp(glu, x2, e32, K, 32)).astype(np.float32)
    yb = np.array(Q8.mlp(glu, x2, e32, K)).astype(np.float32)
    monkeypatch.setenv("KYOJIN_PF_GEMM_I8_ROW", "1")
    assert Q8.row_scale()
    yr = np.array(Q8.mlp(glu, x2, e32, K)).astype(np.float32)
    assert not np.isnan(yr).any() and not np.allclose(yr, yb)
    rel = lambda a: np.sqrt(((a - ref) ** 2).mean() / (ref ** 2).mean())
    assert rel(yr) <= 0.04, (rel(yr), rel(yb))
