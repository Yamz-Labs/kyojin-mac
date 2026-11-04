"""TurboQuant KV (kyojin_mlx/tq.py): numpy codec reference vs the Metal writer / dequant, the engine's plane layout, and the cache's attention
(fused decode kernel and chunked path) against an oracle over the dequantized pool with exact fp16 sink / window rows. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import kvwin as KW, tq as TQ                              # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
HQ, HK = 8, 2

# codes (arange(32) * 7 + 3 + bits) % 2^bits packed by exllamav3 ext_fallbacks._pack_cache_codes (the HIP / CUDA pool layout)
GOLD = {5: [3377654644, 2233378608, 1104831484, 227946424, 2863311530],
        6: [3971577154, 3398914848, 2826252558, 2253594604, 2981212593, 2981212593]}


@pytest.fixture(autouse=True)
def _hook():
    KW.install()


@pytest.mark.parametrize("bits", [5, 6])
def test_pack_matches_engine_layout(bits):
    codes = (np.arange(32) * 7 + 3 + bits) % (1 << bits)
    w = TQ.pack_ref(codes.reshape(1, 32), bits)[0]
    assert w.tolist() == GOLD[bits]
    assert (TQ.unpack_ref(w[None], bits)[0] == codes).all()


def test_rotation_orthonormal():
    for d in (64, 128, 256):
        m = TQ.rotation(d)
        assert np.allclose(m @ m.T, np.eye(d), atol=1e-5)


@pytest.mark.parametrize("d,bits", [(128, 5), (128, 6), (256, 5), (256, 6)])
def test_codec_norm_correction(d, bits):
    x = np.random.default_rng(1).standard_normal((512, d)).astype(np.float32) * 3
    y = x @ TQ.rotation(d).T
    w, s, _ = TQ.encode_ref(y, bits)
    r = TQ.decode_ref(w, s, bits, d).astype(np.float32)
    assert np.abs(np.linalg.norm(r, axis=-1) / np.linalg.norm(y, axis=-1) - 1).max() < 2e-3
    assert np.linalg.norm(r @ TQ.rotation(d) - x) / np.linalg.norm(x) < {5: 0.055, 6: 0.03}[bits]


@pytest.mark.parametrize("d,bits", [(64, 6), (128, 5), (256, 5), (256, 6)])
def test_metal_writer_vs_reference(d, bits):
    r = np.random.default_rng(2)
    y = (r.standard_normal((3000, d)) * (1 + 4 * r.random((3000, 1)))).astype(np.float32)
    w_ref, s_ref, codes = TQ.encode_ref(y, bits)
    w, s = TQ.tq_write(mx.array(y), bits)
    w, s = np.array(w), np.array(s)
    cmp_codes = TQ.unpack_ref(w.reshape(-1, d // 32, bits), bits).reshape(-1, d)
    bad = cmp_codes != codes
    nrm = np.linalg.norm(y, axis=-1, keepdims=True)
    mid = TQ.midpoints(d, bits)
    t = y / nrm
    dist = np.abs(t[bad][:, None] - mid[None]).min(-1) if bad.any() else np.zeros(0)
    assert bad.mean() < 2e-5, bad.mean()                                  # only fp32 summation-order flips of a value sitting on a midpoint
    assert (dist < 1e-6).all()
    ok = ~bad.any(-1)
    assert (w[ok] == w_ref[ok]).all()                                     # bit-exact words wherever every code agrees
    assert np.abs(s.astype(np.float32)[ok] / s_ref.astype(np.float32)[ok] - 1).max() < 2e-3


@pytest.mark.parametrize("d,bits", [(128, 6), (256, 5), (256, 6)])
def test_metal_dequant_bit_exact(d, bits):
    y = np.random.default_rng(3).standard_normal((500, d)).astype(np.float32)
    w, s, _ = TQ.encode_ref(y, bits)
    ref = TQ.decode_ref(w, s, bits, d)
    got = np.array(TQ.tq_dequant(mx.array(w), mx.array(s), bits, d))
    assert (got.view(np.uint16) == ref.view(np.uint16)).all()


# ------------------------------------------------------------------------------------------------ cache attention
def data(L, d, seed=0):
    r = np.random.default_rng(seed)
    f = lambda *s: mx.array((r.standard_normal(s) * (1 + r.random(s) * 2)).astype(np.float16))
    return f(1, HQ, L, d), f(1, HK, L, d), f(1, HK, L, d)


def run(cache, q, k, v, chunks, d=256):
    outs, s = [], 0
    for c in chunks:
        kv = cache.update_and_fetch(k[:, :, s:s + c], v[:, :, s:s + c])
        outs.append(KW._base.scaled_dot_product_attention(q[:, :, s:s + c], kv[0], kv[1], cache=cache, scale=d ** -0.5, mask=None))
        s += c
    return mx.concatenate(outs, axis=2)


def oracle(q, k, v, kb, vb, S, W, d, lo=None):
    """Per row p: rotated q against [fp16 rotated keys for the sinks and [lo[p], p] (default lo[p] = p - W + 1), dequantized numpy-reference pool elsewhere]."""
    L = k.shape[2]
    M = TQ.rotation(d)
    kr = np.array(k).astype(np.float32) @ M.T
    vr = np.array(v).astype(np.float32) @ M.T
    qr = np.array(q).astype(np.float32) @ M.T
    kr16, vr16 = kr.astype(np.float16).astype(np.float32), vr.astype(np.float16).astype(np.float32)
    dk = np.stack([TQ.decode_ref(*TQ.encode_ref(kr[0, h], kb)[:2], kb, d) for h in range(HK)])[None].astype(np.float32)
    dv = np.stack([TQ.decode_ref(*TQ.encode_ref(vr[0, h], vb)[:2], vb, d) for h in range(HK)])[None].astype(np.float32)
    out = np.zeros((1, HQ, L, d), np.float32)
    for p in range(L):
        keep = (np.arange(L) < S) | ((np.arange(L) > p - W) if lo is None else (np.arange(L) >= lo[p]))
        kk = np.where(keep[None, None, :, None], kr16, dk)[:, :, :p + 1]
        vv = np.where(keep[None, None, :, None], vr16, dv)[:, :, :p + 1]
        kk, vv = np.repeat(kk, HQ // HK, axis=1), np.repeat(vv, HQ // HK, axis=1)
        s = (qr[:, :, p:p + 1] @ kk.swapaxes(-1, -2)) * d ** -0.5
        s = np.exp(s - s.max(-1, keepdims=True))
        out[:, :, p:p + 1] = (s / s.sum(-1, keepdims=True)) @ vv
    return out @ M


def err(a, b):
    return float(np.abs(np.array(a).astype(np.float32) - b).max() / (np.abs(b).max() + 1e-9))


@pytest.mark.parametrize("d,bits,S,W", [(256, 6, 4, 32), (256, 5, 4, 32), (128, 6, 0, 0), (128, 5, 3, 16)])
@pytest.mark.parametrize("fused", [True, False])
def test_attention_vs_oracle(d, bits, S, W, fused, monkeypatch):
    monkeypatch.setenv("KYOJIN_TQ_DECODE", "1" if fused else "0")
    monkeypatch.setattr(TQ, "PF", 2)                                    # decode-faithful chunk path (the oracle's per-row window)
    L = 150
    q, k, v = data(L, d)
    ref = oracle(q, k, v, bits, bits, S, W, d)
    for chunks in ([100, 30, 1, 1, 18], [L], [100] + [1] * 50):
        L2 = sum(chunks)
        got = run(TQ.TQKVCache(bits, bits, S, W), q, k, v, chunks, d)
        assert err(got, ref[:, :, :L2]) < 4e-3, (chunks, err(got, ref[:, :, :L2]))


@pytest.mark.parametrize("d,bits,S,W", [(256, 6, 4, 32), (256, 5, 4, 32), (128, 6, 0, 0), (128, 5, 3, 16)])
@pytest.mark.parametrize("pf", [1, 2])
def test_prefill_chunk_vs_oracle(d, bits, S, W, pf, monkeypatch):
    """KYOJIN_TQ_PF=1: rows of a chunk of >= PF_MIN rows read the whole ring + chunk at fp16 (keys from max(S, start - ring) up); PF=2 and short
    chunks keep the per-row window. First chunk (prev < sink) included."""
    monkeypatch.setattr(TQ, "PF", pf)
    L = 150
    q, k, v = data(L, d, 7)
    cache = TQ.TQKVCache(bits, bits, S, W)
    for chunks in ([100, 30, 1, 1, 18], [L], [40, 3, 60, 47]):
        lo, s0 = np.zeros(L, np.int64), 0
        for c in chunks:
            ring = min(s0, W + cache.margin) if W > 0 else 0
            for p in range(s0, s0 + c):
                lo[p] = (max(S, s0 - ring) if W > 0 else p + 1) if pf == 1 and c >= TQ.PF_MIN else p - W + 1
            s0 += c
        ref = oracle(q, k, v, bits, bits, S, W, d, lo)
        got = run(TQ.TQKVCache(bits, bits, S, W), q, k, v, chunks, d)
        assert err(got, ref) < 4e-3, (chunks, err(got, ref))


def test_fused_decode_matches_generic():
    d = 256
    q, k, v = data(700, d, 5)
    outs = []
    for fused in ("1", "0"):
        import os
        os.environ["KYOJIN_TQ_DECODE"] = fused
        outs.append(run(TQ.TQKVCache(6, 6, 4, 64), q, k, v, [600] + [1] * 100, d)[:, :, 600:])
    os.environ["KYOJIN_TQ_DECODE"] = "1"
    assert err(outs[0], np.array(outs[1]).astype(np.float32)) < 5e-3                 # the generic path softmaxes in fp16


def test_trim_fork_state_roundtrip():
    d = 256
    q, k, v = data(300, d, 7)
    c = TQ.TQKVCache(6, 6, 4, 64)
    run(c, q, k, v, [250, 1, 1, 1, 1])
    f = c.fork()
    assert c.trim(3) == 3 and c.offset == 251
    g = TQ.TQKVCache(6, 6, 4, 64)
    g.state = [x for x in c.state]
    a = run(c, q[:, :, 251:252], k[:, :, 251:252], v[:, :, 251:252], [1], d)
    b = run(g, q[:, :, 251:252], k[:, :, 251:252], v[:, :, 251:252], [1], d)
    assert (np.array(a) == np.array(b)).all()
    assert f.offset == 254


def test_verify_after_rollback_within_margin_matches_straight_cache():
    """Rollback the margin covers keeps a shorter fp16 ring (no pool refill); the next verify chunk reads exactly what a cache that never saw the rolled-back rows reads."""
    d = 256
    q, k, v = data(300, d, 8)
    a, b = TQ.TQKVCache(6, 6, 4, 64), TQ.TQKVCache(6, 6, 4, 64)
    run(a, q, k, v, [250], d)
    run(b, q, k, v, [250, 1, 1, 1, 1], d)
    assert b.trim(4) == 4 and b.rk.shape[2] == 64 + TQ.MARGIN - 4
    ra = run(a, q[:, :, 250:255], k[:, :, 250:255], v[:, :, 250:255], [5], d)
    rb = run(b, q[:, :, 250:255], k[:, :, 250:255], v[:, :, 250:255], [5], d)
    assert (np.array(ra) == np.array(rb)).all()


# ------------------------------------------------------------------------------------------------ tiled K layout (LAYOUT 2)
@pytest.mark.parametrize("chunks", [[100, 30, 1, 1, 18], [31, 1, 33, 64, 1, 1, 200], [1] * 70])
def test_tiled_k_pool_matches_row_writer(chunks):
    """Any chunking (unaligned starts, single-token decode writes, growth past a step) leaves the same K words / scales as quantizing all rows at once,
    and the tiled dequant of any row range equals the row-layout dequant."""
    d, bits = 256, 6
    L = sum(chunks)
    _, k, v = data(L, d, 11)
    c = TQ.TQKVCache(bits, bits, 4, 32)
    s = 0
    for n in chunks:
        c.update_and_fetch(k[:, :, s:s + n], v[:, :, s:s + n])
        s += n
    kr = TQ.rotate(k, d).astype(mx.float32)
    w, sc = TQ.tq_write(kr.reshape(-1, d), bits)
    st = c.state
    assert (np.array(st[0]) == np.array(w).reshape(1, HK, L, -1)).all() and (np.array(st[1]) == np.array(sc).reshape(1, HK, L)).all()
    assert c.kq[0].shape[-1] == 32 and c.kq[0].shape[2] * 32 == c.kq[1].shape[-1] == c.vq[0].shape[2]
    for lo, hi in ((0, L), (5, 37), (33, 34), (L - 40, L)):
        ref = TQ.tq_dequant(w.reshape(HK, L, -1)[:, lo:hi].reshape(-1, w.shape[-1]), sc.reshape(HK, L)[:, lo:hi].reshape(-1), bits, d).reshape(1, HK, hi - lo, d)
        assert (np.array(c._dq_k(lo, hi, d, mx.float16)).view(np.uint16) == np.array(ref).view(np.uint16)).all(), (lo, hi)


@pytest.mark.parametrize("hq,hk,d", [(6, 2, 128), (12, 2, 64), (2, 2, 256), (16, 1, 128)])
def test_fused_decode_gqa_shapes(hq, hk, d):
    r = np.random.default_rng(4)
    f = lambda h, n: mx.array(r.standard_normal((1, h, n, d)).astype(np.float16))
    k, v, q = f(hk, 400), f(hk, 400), f(hq, 400)
    outs = []
    for fused in ("1", "0"):
        import os
        os.environ["KYOJIN_TQ_DECODE"] = fused
        outs.append(run(TQ.TQKVCache(6, 6, 4, 48), q, k, v, [333] + [1] * 40, d)[:, :, 333:])
    os.environ["KYOJIN_TQ_DECODE"] = "1"
    assert err(outs[0], np.array(outs[1]).astype(np.float32)) < 5e-3


def test_fused_decode_batched():
    """B > 1 decode goes through the fused kernel (one grid over B * H) and matches the generic path and the B = 1 results row by row."""
    import os
    d, hq, hk = 128, 8, 2
    r = np.random.default_rng(6)
    f = lambda h, n: mx.array(r.standard_normal((3, h, n, d)).astype(np.float16))
    q, k, v = f(hq, 300), f(hk, 300), f(hk, 300)
    ch = [250] + [1] * 20
    outs = {}
    for fused in ("1", "0"):
        os.environ["KYOJIN_TQ_DECODE"] = fused
        outs[fused] = run(TQ.TQKVCache(6, 6, 4, 32), q, k, v, ch, d)[:, :, 250:]
    os.environ["KYOJIN_TQ_DECODE"] = "1"
    assert err(outs["1"], np.array(outs["0"]).astype(np.float32)) < 5e-3
    for i in range(3):
        one = run(TQ.TQKVCache(6, 6, 4, 32), q[i:i + 1], k[i:i + 1], v[i:i + 1], ch, d)[:, :, 250:]
        assert err(outs["1"][i:i + 1], np.array(one).astype(np.float32)) < 2e-3, i


@pytest.mark.parametrize("nsplit", [4, 5, 8, 32])
def test_every_pool_key_is_covered(nsplit, monkeypatch):
    """Pool lengths around multiples of the 32-key tile with a sink of 3 (tile-misaligned start) and few splits: the last key must not drop out."""
    monkeypatch.setattr(TQ, "NSPLIT", nsplit)
    monkeypatch.setattr(TQ, "PF", 2)                                    # the chunk reference must be decode-faithful
    d = 128
    q, k, v = data(260, d, 9)
    for n in range(100, 140, 3):
        got = run(TQ.TQKVCache(6, 6, 3, 16), q, k, v, [n - 1, 1], d)[:, :, n - 1:]
        ref = run(TQ.TQKVCache(6, 6, 3, 16), q, k, v, [n], d)[:, :, n - 1:]
        assert err(got, np.array(ref).astype(np.float32)) < 5e-3, (nsplit, n)


@pytest.mark.parametrize("d,bits,S,W", [(256, 6, 4, 32), (128, 5, 3, 16), (128, 6, 4, 24)])
@pytest.mark.parametrize("kern", ["plain", "rows", "nax"])
def test_fused_verify_rows_vs_oracle(d, bits, S, W, kern, monkeypatch):
    """Chunks of 2..8 rows (MTP verify) through the fused kernel: per-row causal mask, per-row fp16 window and pool precision at the window boundary.
    kern: the pool part in the plain kernel (short pools), the one-pass verify-rows kernel, or the NAX verify-rows kernel (matrix units, M5+)."""
    if kern == "nax":
        from kyojin_mlx.sdpa_dec import nax_ok
        if not nax_ok():
            pytest.skip("no matrix units")
    monkeypatch.setattr(TQ, "ROWS_MIN", 2048 if kern == "plain" else 0)
    monkeypatch.setattr(TQ, "NAX", int(kern == "nax"))
    monkeypatch.setenv("KYOJIN_TQ_DECODE", "1")
    monkeypatch.setattr(TQ, "PF", 2)                                    # decode-faithful first chunk (the oracle's per-row window)
    L = 150
    q, k, v = data(L, d, 12)
    ref = oracle(q, k, v, bits, bits, S, W, d)
    for chunks in ([100, 4, 1, 7, 2, 8, 3, 5], [60, 2, 2, 2, 8, 8, 8, 8], [90, 6, 6, 6, 6, 6, 6, 6, 6, 6]):
        L2 = sum(chunks)
        c = TQ.TQKVCache(bits, bits, S, W)
        got = run(c, q, k, v, chunks, d)
        assert err(got, ref[:, :, :L2]) < 4e-3, (chunks, err(got, ref[:, :, :L2]))
    assert c._verify_ok(1, 4)


def test_chunk_pool_write_matches_single_rows():
    """Verify-sized chunks (in-place strided tile columns, crossing tile edges) and a long prefill chunk (rows-view patch) write the same pool as one row at a time."""
    d = 128
    x = mx.random.normal((1, HK, 200, d)).astype(mx.float16)
    a, b = TQ.TQKVCache(6, 6, 4, 64), TQ.TQKVCache(6, 6, 4, 64)
    a.update_and_fetch(x[:, :, :100], x[:, :, :100])
    for j in range(100, 200, 7):
        a.update_and_fetch(x[:, :, j:j + 7], x[:, :, j:j + 7])
    for j in range(200):
        b.update_and_fetch(x[:, :, j:j + 1], x[:, :, j:j + 1])
    for pa, pb in ((a.kq, b.kq), (a.vq, b.vq)):
        assert mx.array_equal(pa[0][:, :, :200 // 32 if pa is a.kq else 200], pb[0][:, :, :200 // 32 if pb is b.kq else 200]).item()
        assert mx.array_equal(pa[1][:, :, :200], pb[1][:, :, :200]).item()
