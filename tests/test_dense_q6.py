"""BP3-Q6 (flag glmdenseq, KYOJIN_GLM_DENSE_Q): symmetric 6-bit g32 copy of the fp8 / fp8b dense linears, read by the 1..4 row GEMV."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import foreign_linear as FL, glm_dense6 as G6, foreign_pack as FP

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU")


def _pack(n, k, rng):
    q = rng.integers(0, 256, (n, k), dtype=np.uint8)
    q[(q & 127) == 127] = 0x38
    g = rng.uniform(64.0, 127.0, (-(-n // 128), -(-k // 128))).astype(np.float32)
    return FP.Pack("fp8b", n, k, q, g[np.arange(n) // 128][:, (np.arange(k // 32) * 32) // 128].astype(np.float16), np.float32(2.0 ** -20))


def _close(y, ref):
    y, ref = np.array(y, np.float32), np.array(ref, np.float32)
    assert np.isfinite(y).all()
    assert np.abs(y - ref).max() <= 2e-3 * np.abs(ref).max() + 1e-3, np.abs(y - ref).max()


def _ref(d):
    wq, sc = (np.array(a) for a in d.g6)
    return mx.array(G6.dequant(wq, sc, d.n, d.k))


def test_pack_roundtrip_and_device_quantizer():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((70, 1024)) * 0.02).astype(np.float32)
    w[1, :32] = 0.0
    words, sc, deq = G6.quantize(w)
    assert np.array_equal(G6.dequant(words, sc, 70, 1024), deq)
    assert np.sqrt(((w - deq) ** 2).sum() / (w ** 2).sum()) < 0.025
    wm, sm = G6.quantize_mx(mx.array(w))
    assert np.array_equal(np.array(wm), words) and np.array_equal(np.array(sm), sc)
    assert words.shape == (70, 1024 // 512 * 96) and sc.shape == (70, 32)


@needs_gpu
@pytest.mark.parametrize("n,k", [(512, 4096), (1536, 4096), (16384, 1536), (4096, 2048), (4096, 12288), (4104, 2560), (154880 // 8, 4096)])
def test_q6_gemv(n, k, monkeypatch):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(FL, "DENSE_Q", True)
    rng = np.random.default_rng(n + k)
    d = FL.DevPack(_pack(n, k, rng))
    assert d.g6 is not None and d.g3_ok(1)
    w = _ref(d)
    for R in (1, 2, 3, 4):
        x = mx.array(rng.normal(0, 1, (R, k)).astype(np.float16))
        _close(d.apply(x), x.astype(mx.float32) @ w.T)
    # close to the fp8 original too (6-bit noise ~ 2 % per weight, ~ 0.3 % of the output norm after the k-sum)
    x = mx.array(rng.normal(0, 1, (1, k)).astype(np.float16))
    full = x.astype(mx.float32) @ d.dequant().astype(mx.float32).T
    y = d.apply(x).astype(mx.float32)
    assert float(mx.linalg.norm(y - full) / mx.linalg.norm(full)) < 0.04


@needs_gpu
def test_q6_flag_off_and_rows(monkeypatch):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    rng = np.random.default_rng(1)
    monkeypatch.setattr(FL, "DENSE_Q", False)
    assert FL.DevPack(_pack(256, 1024, rng)).g6 is None
    monkeypatch.setattr(FL, "DENSE_Q", True)
    d = FL.DevPack(_pack(256, 1024, rng))
    assert d.g6 is not None
    assert FL.DevPack(_pack(256, 768, rng)).g6 is None                        # k % 512
    x = mx.array(rng.normal(0, 1, (5, 1024)).astype(np.float16))              # R > 4: the fp8 bytes (no 6-bit rounding)
    _close(d.apply(x), x.astype(mx.float32) @ d.dequant().astype(mx.float32).T)
    monkeypatch.setattr(FL, "DENSE_GEMV", False)
    assert FL.DevPack(_pack(256, 1024, rng)).g6 is None                       # needs the G3 flag


@needs_gpu
def test_q6_grouped_rows_nbytes(monkeypatch):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(FL, "DENSE_Q", True)
    rng = np.random.default_rng(2)
    ds = [FL.DevPack(_pack(n, 2048, rng)) for n in (1024, 512, 512)]
    g = FL.DevPack.grouped(ds)
    assert g.g6[0].shape[0] == 2048 and g.g6[1].shape[0] == 2048
    x = mx.array(rng.normal(0, 1, (2, 2048)).astype(np.float16))
    y = g.apply(x)
    _close(y, x.astype(mx.float32) @ _ref(g).T)
    _close(y[:, :1024], ds[0].apply(x))
    r = g.rows(700)
    _close(r.apply(x), y[:, :700])
    assert g.nbytes() > ds[0].nbytes()


@needs_gpu
@pytest.mark.parametrize("dt", [mx.float32, mx.bfloat16])
def test_q6_staged_cast(dt, monkeypatch):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(FL, "DENSE_Q", True)
    rng = np.random.default_rng(3)
    d = FL.DevPack(_pack(1024, 4096, rng))
    for R in (1, 2, 4):
        x = mx.array(rng.normal(0, 1, (R, 4096)).astype(np.float32)).astype(dt)
        y = d._gemv8g3(x)
        ref = d._gemv8g3(x.astype(mx.float16))
        assert np.array_equal(np.array(y.astype(mx.float32)), np.array(ref.astype(mx.float32)))


@needs_gpu
@pytest.mark.parametrize("k,limit", [(2048, 7.0), (12288, 3.5)])
def test_q6_glu(k, limit, monkeypatch):
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(FL, "DENSE_Q", True)
    rng = np.random.default_rng(k)
    d = FL.DevPack(_pack(4096, k, rng))
    w = _ref(d)
    for R in (1, 3):
        y = mx.array((rng.normal(0, 3, (R, 2 * k))).astype(np.float16))
        gate, up = y[:, :k].astype(mx.float32), y[:, k:].astype(mx.float32)
        h = (mx.minimum(gate, limit) * mx.sigmoid(mx.minimum(gate, limit))) * mx.clip(up, -limit, limit)
        _close(d._gemv8g3(y, limit=limit), h @ w.T)


@needs_gpu
def test_q6_all_rows_gate_mode(monkeypatch):
    """KYOJIN_GLM_DENSE_Q_ALL (quality gate): every row count, prefill sizes included, reads the 6-bit weights; dequant_mx = the reference decode."""
    monkeypatch.setattr(FL, "DENSE_GEMV", True)
    monkeypatch.setattr(FL, "DENSE_Q", True)
    rng = np.random.default_rng(5)
    d = FL.DevPack(_pack(1024, 2048, rng))
    w = _ref(d)
    assert np.array_equal(np.array(G6.dequant_mx(d.g6[0], d.g6[1], d.n, d.k).astype(mx.float32)), np.array(w.astype(mx.float16).astype(mx.float32)))
    monkeypatch.setattr(FL, "DENSE_Q_ALL", True)
    for R in (1, 5, 40, 300):
        x = mx.array(rng.normal(0, 1, (R, 2048)).astype(np.float16))
        _close(d.apply(x), x.astype(mx.float32) @ w.T)
    assert np.array_equal(np.array(d.dequant().astype(mx.float32)), np.array(w.astype(mx.float16).astype(mx.float32)))
