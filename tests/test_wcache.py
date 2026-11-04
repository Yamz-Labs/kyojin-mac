"""Prompt-scoped weight cache (kyojin_mlx.wcache): same bits as no cache, decoded once per linear per scope, freed at the end, off above the cap."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

import mlx.nn as nn
from kyojin_mlx import exl3_linear as EL, kernels as KN, wcache as WC


def _lin(kin, nout, K, seed):
    rng = np.random.default_rng(seed)
    kt, nt = kin // 16, nout // 16
    tr = mx.array(rng.integers(-32768, 32768, (kt, nt, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    return EL.Exl3Linear(tr, f16(kin), f16(nout), cb="mul1")


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = _lin(256, 384, 3, 1), _lin(384, 256, 2, 2)


def _x(rows, kin, seed):
    return mx.array(np.random.default_rng(seed).standard_normal((1, rows, kin)).astype(np.float16))


@pytest.fixture(autouse=True)
def unfused(monkeypatch):
    monkeypatch.setattr(EL, "DENSE_FUSED", "0")                 # rows above the GEMV chunk: decode + matmul (the path with a fp16 weight)
    monkeypatch.setattr(EL, "DENSE_FUSED_MAX", 32)


def test_scope_is_bit_identical_and_decodes_once(monkeypatch):
    m = Toy()
    xa, xb = _x(70, 256, 3), _x(70, 384, 4)
    ref = [np.array(m.a(xa)), np.array(m.b(xb))]
    monkeypatch.setenv("KYOJIN_PF_WCACHE_GB", "1")
    calls = []
    orig = KN.decode_wr
    monkeypatch.setattr(KN, "decode_wr", lambda *a, **k: calls.append(1) or orig(*a, **k))
    with WC.prefill_scope(m, 70, 3):
        got = [[np.array(m.a(xa)), np.array(m.b(xb))] for _ in range(3)]
        assert WC._state is not None
    assert WC._state is None and len(calls) == 2, calls          # one decode per linear for three chunks
    assert WC.last["on"] and WC.last["hits"] == 4 and WC.last["bytes"] == (256 * 384 * 2) * 2
    for g in got:
        assert all(np.array_equal(a, b) for a, b in zip(g, ref))
    calls.clear()
    m.a(xa), m.a(xa)
    assert len(calls) == 2                                       # outside a scope every call decodes


@pytest.mark.parametrize("gb,chunks,rows", [("0", 3, 70), ("1e-9", 3, 70), ("1", 1, 70)])
def test_scope_off(gb, chunks, rows, monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_WCACHE_GB", gb)
    m = Toy()
    with WC.prefill_scope(m, rows, chunks):
        assert WC._state is None
    assert not WC.last["on"]


def test_scope_off_for_fused_rows(monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_WCACHE_GB", "1")
    with WC.prefill_scope(Toy(), 16, 4):                        # rows the fused decode + NAX path takes: no fp16 weight to cache
        assert WC._state is None


def test_need_bytes_skips_the_head():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Toy()]
            self.head = _lin(256, 512, 2, 5)
    assert WC.need_bytes(M()) == (256 * 384 + 384 * 256) * 2
