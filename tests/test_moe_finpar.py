"""KYOJIN_MOE_FINPAR: the routed combine launch with the pairs side by side. CPU checks: the generated source computes the same per-pair term as the serial
launch and adds the terms in the same order (so the output is bit-identical by construction), the launch shape, the flag. The Metal bit-identity and the
timing are tools/moe_k_bench.py's finpar arm and the test below when a GPU is present."""
import re

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL, metal_src as S, moe as M  # noqa: E402

METAL = mx.metal.is_available()


@pytest.fixture(autouse=True)
def restore():
    f = M.FINPAR
    yield
    M.FINPAR = f


def _term(src):
    """The per-pair statements (everything between the partial loads and the store), whitespace-normalised."""
    a = src.index("float4 acc = float4(0.0f);")
    b = src.index("y4 = float4(v0, v1, v2, v3)")
    return re.sub(r"\s+", " ", src[a:b])


@pytest.mark.parametrize("wpost", [False, True])
def test_pair_term_is_the_serial_one(wpost):
    ser, par = S.final_gather_source(wpost), S.final_par_source(wpost, 6)
    assert _term(ser) == _term(par)                                   # partial loads, butterfly, sign vector: the same statements
    assert ("y4 = y4 * wts[p];" in par) == wpost == ("y4 = y4 * wts[p];" in ser)
    tail = lambda s: re.sub(r"\s+", " ", s[s.index("y4 = float4(v0, v1, v2, v3)"):s.index(";", s.index("y4 = float4(v0, v1, v2, v3)")) + 1])
    assert tail(ser) == tail(par)


def test_pairs_are_added_in_order_by_simdgroup_zero():
    par = S.final_par_source(True, 7)
    assert "threadgroup float4 ys[7][32];" in par
    assert "ys[j][lane] = y4;" in par and "threadgroup_barrier(mem_flags::mem_threadgroup);" in par
    body = par[par.index("if (j == 0)"):]
    assert "for (uint q = 0; q < TOPK; ++q) out = out + ys[q][lane];" in body
    assert par.index("ys[j][lane] = y4;") < par.index("threadgroup_barrier") < par.index("if (j == 0)")
    assert "return;" not in par                                       # every simdgroup reaches the barrier and the shuffles


def test_bf16_slot_only_in_the_shared_variant():
    assert "0x7FFFu" not in S.final_par_source(False, 7)
    sh = S.final_par_source(False, 7, True)
    assert sh.count("0x7FFFu") == 1 and sh.index("0x7FFFu") < sh.index("ys[j][lane] = y4;")
    assert "j == TOPK - 1" in sh


def test_launch_shape():
    for blocks, rows, nj in ((40, 1, 6), (18, 4, 7)):
        grid, tg = S.final_par_grid(blocks, rows, nj)
        assert tg == (32 * nj, 1, 1) and grid[0] // tg[0] == blocks and grid[1] == rows


def test_numpy_order_equivalence():
    """Adding the pair terms through a [nj] table in order j gives the serial running sum bit for bit."""
    rng = np.random.default_rng(1)
    ys = rng.standard_normal((7, 128)).astype(np.float32) * 10 ** rng.uniform(-3, 3, (7, 1)).astype(np.float32)
    serial = np.zeros(128, np.float32)
    for j in range(7):
        serial = serial + ys[j]
    table = np.zeros(128, np.float32)
    for q in range(7):
        table = table + ys[q]
    assert np.array_equal(serial, table)


def test_flag():
    f = FL.FLAGS["moe_finpar"]
    assert f.default == "0" and f.runtime and f.attrs[0][:2] == ("moe", "FINPAR")
    assert 'KYOJIN_MOE_FINPAR", "0"' in open(M.__file__).read()


@pytest.mark.skipif(not METAL, reason="needs Metal")
def test_metal_bit_identical_to_the_serial_launch():
    rng = np.random.default_rng(5)
    for topk, wpost, bf in ((6, False, False), (7, False, True), (8, True, False)):
        n, rows, tgk = 128 * 3, 2, 2
        part = mx.array(rng.standard_normal((rows * topk, tgk, n)).astype(np.float32))
        ids = mx.array(rng.integers(0, 4, rows * topk).astype(np.int32))
        wts = mx.array(rng.uniform(0.1, 1, rows * topk).astype(np.float32))
        svh = mx.array(rng.standard_normal((4, n)).astype(np.float16))
        params = mx.array([tgk, n, topk], dtype=mx.uint32)
        ser_src = S.final_gather_sh_source(wpost) if bf else S.final_gather_source(wpost)
        ks = mx.fast.metal_kernel(name="t_ser", input_names=["part", "ids", "wts", "svh", "params"], output_names=["y"], source=ser_src)
        kp = mx.fast.metal_kernel(name="t_par", input_names=["part", "ids", "wts", "svh", "params"], output_names=["y"], source=S.final_par_source(wpost, topk, bf))
        g1, t1 = S.had_grid(n // 128, rows)
        g2, t2 = S.final_par_grid(n // 128, rows, topk)
        a = ks(inputs=[part, ids, wts, svh, params], grid=g1, threadgroup=t1, output_shapes=[(rows, n)], output_dtypes=[mx.float32])[0]
        b = kp(inputs=[part, ids, wts, svh, params], grid=g2, threadgroup=t2, output_shapes=[(rows, n)], output_dtypes=[mx.float32])[0]
        assert np.array_equal(np.array(a), np.array(b))
