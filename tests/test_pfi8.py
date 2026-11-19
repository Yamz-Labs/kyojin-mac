"""foreign_pfi8: dense int8 prefill GEMM. quant_weight / why_not / flags run on the CPU; gemm_i8 and the DevPack hooks need the Metal GPU."""
import os, sys
import numpy as np
import pytest
import mlx.core as mx


def test_quant_weight_roundtrip():
    from kyojin_mlx import foreign_pfi8 as P
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((64, 512)) * 0.05).astype(np.float16)
    w[3] = 0
    q, s = P.quant_weight(mx.array(w))
    q, s = np.array(q), np.array(s)
    assert q.dtype == np.int8 and np.abs(q).max() <= 127 and (q[3] == 0).all()
    rec = q.astype(np.float32) * s[:, None]
    err = np.abs(rec - w.astype(np.float32)).max(axis=1)
    assert (err <= s * 0.5 + 1e-7).all()
    assert (np.abs(q).max(axis=1)[np.arange(64) != 3] == 127).all()


def test_flags_registered_live():
    from kyojin_mlx import flags
    for name, env in (("pf_gemm_i8", "KYOJIN_PF_GEMM_I8"), ("pf_gemm_i8_dense", "KYOJIN_PF_GEMM_I8_DENSE")):
        f = flags.FLAGS[name]
        assert f.env == env and f.default == "0" and f.runtime


def test_alias_and_why_not(monkeypatch):
    from kyojin_mlx import foreign_pfi8 as P, foreign_pfq8 as Q8

    class D:
        E = None; n = 4096; k = 4096; fmt = "fp8"
    monkeypatch.delenv("KYOJIN_PF_GEMM_I8_DENSE", raising=False)
    assert P.why_not(D, 8192) == "flag_off"
    monkeypatch.setenv("KYOJIN_PF_GEMM_I8_DENSE", "1")
    assert P.why_not(D, 8192) == "" and P.why_not(D, 100) == "rows<min"
    D2 = type("D2", (), dict(E=None, n=4100, k=4096, fmt="fp8"))
    assert P.why_not(D2, 8192) == "shape"
    assert P.why_not(type("D3", (), dict(E=None, n=4096, k=2048, fmt="fp8")), 8192) == "small"
    assert P.why_not(type("D4", (), dict(E=None, n=2048, k=4096, fmt="fp8")), 8192) == "small"
    assert P.why_not(type("D5", (), dict(E=None, n=24576, k=4096, fmt="fp8")), 8192) == ""
    monkeypatch.delenv("KYOJIN_PF_GEMM_I8", raising=False)
    monkeypatch.delenv("KYOJIN_GLM_PF_Q8", raising=False)
    assert not Q8.on()
    monkeypatch.setenv("KYOJIN_GLM_PF_Q8", "1")
    assert Q8.on()
    monkeypatch.delenv("KYOJIN_GLM_PF_Q8")
    monkeypatch.setenv("KYOJIN_PF_GEMM_I8", "1")
    assert Q8.on()


@pytest.mark.parametrize("S,n,k", [(300, 128, 256), (8192 + 77, 192, 1024), (5, 64, 512)])
def test_gemm_i8_exact_model(S, n, k):
    from kyojin_mlx import foreign_pfi8 as P
    rng = np.random.default_rng(S)
    x = rng.integers(-127, 128, (S, k)).astype(np.int8)
    w = rng.integers(-127, 128, (n, k)).astype(np.int8)
    xs = (rng.random((S, k // 128)) + 0.5).astype(np.float32)
    ws = (rng.random(n) * 1e-5 + 1e-6).astype(np.float32)
    y = np.array(P.gemm_i8(mx.array(x), mx.array(xs), mx.array(w), mx.array(ws))).astype(np.float32)
    ref = np.zeros((S, n), np.float64)
    for b in range(k // 128):
        ref += (x[:, b * 128:(b + 1) * 128].astype(np.int64) @ w[:, b * 128:(b + 1) * 128].astype(np.int64).T) * xs[:, b:b + 1]
    ref *= ws[None, :]
    assert np.allclose(y, ref, rtol=2e-3, atol=2e-3 * np.abs(ref).max())


def test_hook_rows(monkeypatch):
    from kyojin_mlx import foreign_pfi8 as P
    monkeypatch.delenv("KYOJIN_PF_GEMM_I8_MIN", raising=False)
    assert P.hook_rows() == 1024
    monkeypatch.setenv("KYOJIN_PF_GEMM_I8_MIN", "64")
    assert P.hook_rows() == 64


def test_weights_on_the_fly_matches_quant_weight_and_is_not_cached():
    from kyojin_mlx import foreign_pfi8 as P
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((192, 1024)) * 0.02).astype(np.float16)
    w[5] = 0
    w[7, 100] = 3.0

    class D:
        E = None; fmt = "dense"; n = 192; k = 1024
    D.w = mx.array(w)
    q, s = P.weights(D)
    rq, rs = P.quant_weight(D.w)
    assert tuple(s.shape) == (192,) and np.allclose(np.array(s), np.array(rs), rtol=1e-6)
    assert np.abs(np.array(q).astype(np.int32) - np.array(rq).astype(np.int32)).max() <= 1
    assert not hasattr(D, "_pfi8")
