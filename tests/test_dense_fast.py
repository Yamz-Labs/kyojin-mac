"""KYOJIN_DENSE_FAST: the fused decode linear's finalize reads the TGK partials with one plain float4 load per k-split instead of four scalar relaxed
atomic loads (metal_src.tail_reads). Same values, same k order of the adds: the output must be bit-identical to the three-launch path and to the atomic
variant, for the single linear, the same-input group and the grouped wo_a, at 1-4 rows, both input dtypes, with and without the folded fp8 fake quant.
Also checks the flag plumbing (env -> kernels.TAIL_READS) and that the default path still generates the old text."""
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import fused_group as FG
from kyojin_mlx import kernels as KN
from kyojin_mlx import metal_src as S


def trellis(kin, nout, K, seed):
    rng = np.random.default_rng(seed)
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    return tr, suh, svh


def bits(a):
    a = np.array(a)
    return a.view(np.uint16) if a.dtype == np.float16 else a.view(np.uint32)


def x_of(rows, kin, dt, seed=0):
    return mx.array((np.random.default_rng(seed).standard_normal((rows, kin)) * 3).astype(dt))


# small (the shapes the flag targets) and real DS4.1 shapes and bitrates: wkv 5120x512, wq_a 5120x1280, shared gate 5120x2304, down 2304x5120,
# wq_b 1280x4096 (indexer), a 64-column boundary shape, and an odd n / k for the grid edges
SHAPES = [(5120, 512, 5), (5120, 1280, 5), (5120, 2304, 5), (2304, 5120, 5), (1280, 4096, 5), (384, 1024, 3), (256, 256, 2), (8192, 5120, 5)]


@pytest.mark.parametrize("kin,nout,K", SHAPES, ids=[f"{a}x{b}K{c}" for a, b, c in SHAPES])
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_single_bit_identical(monkeypatch, kin, nout, K, rows):
    """fast finalize reads == atomic finalize reads == the three-launch path (rot, GEMV, finalize)."""
    tr, su, sv = trellis(kin, nout, K, kin + K + rows)
    g = KN.Mul1Gemv(tr, su, sv, rows=rows, out_dtype=mx.float16, cb="mul1")
    assert g.can_fuse(rows)
    for dt, fq in ((np.float16, False), (np.float16, True), (np.float32, False)):
        x = x_of(rows, kin, dt, rows)
        want = g(x, fq8=fq)                                     # three launches: the reference arithmetic
        monkeypatch.setattr(KN, "TAIL_READS", S.TAIL_ATOM)
        for _ in range(6):                                      # the counter sets rotate: every call sees other counters
            assert np.array_equal(bits(g.fused(x, fq8=fq)), bits(want)), (dt, fq, "atom")
        monkeypatch.setattr(KN, "TAIL_READS", S.TAIL_VEC)
        for _ in range(6):
            assert np.array_equal(bits(g.fused(x, fq8=fq)), bits(want)), (dt, fq, "vec")


@pytest.mark.parametrize("kin,nout,K", [(1280, 4096, 5), (5120, 2304, 5), (2304, 5120, 5)])
@pytest.mark.parametrize("rows", [1, 4])
def test_single_rot2_bit_identical(monkeypatch, kin, nout, K, rows):
    """The two-launch form (rot kernel, then GEMV + finalize tail) with the fast reads: the tail is the same substitution."""
    monkeypatch.setattr(KN, "ROT2_MIN_BLOCKS", 1)
    monkeypatch.setattr(KN, "TAIL_READS", S.TAIL_VEC)
    tr, su, sv = trellis(kin, nout, K, kin + rows)
    g = KN.Mul1Gemv(tr, su, sv, rows=rows, out_dtype=mx.float32, cb="mul1")
    for dt, fq in ((np.float16, False), (np.float32, False)):
        x = x_of(rows, kin, dt, rows)
        monkeypatch.setattr(KN, "TAIL_READS", S.TAIL_ATOM)
        want = g.fused(x, fq8=fq)
        monkeypatch.setattr(KN, "TAIL_READS", S.TAIL_VEC)
        for _ in range(6):
            assert np.array_equal(bits(g.fused(x, fq8=fq)), bits(want))


GROUPS = [("wq_a_wkv", 5120, [1280, 512], 5), ("gate_up", 5120, [2304, 2304], 5), ("qkv", 512, [1024, 256, 256], 3)]
GROUP_IDS = [g[0] for g in GROUPS]


@pytest.mark.parametrize("name,kin,outs,K", GROUPS, ids=GROUP_IDS)
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_group_bit_identical(monkeypatch, name, kin, outs, K, rows):
    """The S5 same-input group (the served wq_a + wkv / gate + up launch) with the fast reads."""
    mem = [EL.Exl3Linear(*(mx.array(a) for a in trellis(kin, n, K, 11 * i + K)), cb="mul1") for i, n in enumerate(outs)]
    g = FG.Exl3Group(mem)
    assert g.can_fuse(rows)
    for dt, fq in ((np.float16, False), (np.float16, True), (np.float32, False)):
        x = x_of(rows, kin, dt, 7)
        want = g.fin(g.gemv(g.rot(x, fq), rows), rows)
        for reads in (S.TAIL_ATOM, S.TAIL_VEC):
            monkeypatch.setattr(KN, "TAIL_READS", reads)
            for _ in range(6):
                for a, b in zip(g.fused(x, fq), want):
                    assert np.array_equal(bits(a), bits(b)), (dt, fq, reads)


def test_independent_calls_in_flight(monkeypatch):
    """Up to FUSED_SETS independent calls in one graph (they can overlap on the GPU) keep their bits with the fast reads too, round after round: the plain
    loads rely on the device fence of the arrival block, so the visibility is what this repeats."""
    tr, su, sv = trellis(5120, 512, 5, 1)
    g = KN.Mul1Gemv(tr, su, sv, rows=1, out_dtype=mx.float16)
    xs = [x_of(1, 5120, np.float16, s) for s in range(KN.Mul1Gemv.FUSED_SETS)]
    monkeypatch.setattr(KN, "TAIL_READS", S.TAIL_ATOM)
    want = [g.fused(x) for x in xs]
    mx.eval(*want)
    monkeypatch.setattr(KN, "TAIL_READS", S.TAIL_VEC)
    for _ in range(40):
        got = [g.fused(x) for x in xs]
        mx.eval(*got)
        for a, b in zip(got, want):
            assert np.array_equal(bits(a), bits(b))


def test_default_source_unchanged(monkeypatch):
    """TAIL_ATOM (the default) generates the same kernel text as before the lever: no pf4 in the tail, the atomic loads still there."""
    monkeypatch.setattr(KN, "TAIL_READS", S.TAIL_ATOM)
    atom = S.fused_source(5, 1, "mul1", True, False, 640)
    vec = S.fused_source(5, 1, "mul1", True, False, 640, reads=S.TAIL_VEC)
    assert "pf4" not in atom and "pf4" in vec
    assert "atomic_load_explicit(pb" in atom and "atomic_load_explicit(pb" not in vec
    assert atom.count("atomic_fetch_add_explicit") == 1 and vec.count("atomic_fetch_add_explicit") == 1

    def without_reads(s, kind):
        if kind == "atom":
            i = s.index("#define LDP(j)")
            j = s.index("#undef LDP", i) + len("#undef LDP")
        else:
            i = s.index("const device float4* pf4")
            j = s.index("acc += pf4[ulong(kb) * ks4];", i) + len("acc += pf4[ulong(kb) * ks4];")
        return " ".join((s[:i] + s[j:]).split())
    # the read block is the only difference between the two texts
    assert without_reads(atom, "atom") == without_reads(vec, "vec")


def test_flag_env():
    """KYOJIN_DENSE_FAST=1 selects the fast reads and the whole decode call stays bit-identical under it; the default is on (flags table)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = r"""
from kyojin_mlx import kernels as KN, metal_src as S
print(KN.TAIL_READS == S.TAIL_VEC)
"""
    check = r"""
import numpy as np, mlx.core as mx
from kyojin_mlx import kernels as KN, metal_src as S
import sys; sys.path.insert(0, "tests")
import dense_db as db
assert KN.TAIL_READS == S.TAIL_VEC, KN.TAIL_READS
ok = True
for name in ("wkv", "wq_a", "sh.down"):
    kin, nout, K = db.BY[name]
    lin = db.synth(kin, nout, K)
    for R in (1, 4):
        x = mx.random.normal((R, kin)).astype(mx.float16)
        g = lin._gemv(R)
        want = g(x)                       # three launches
        got = lin(x)                      # the module call: one fused launch with the fast reads
        mx.eval(want, got)
        a, b = np.array(want), np.array(got)
        ok = ok and np.array_equal(a.view(np.uint16), b.view(np.uint16))
print(ok)
"""
    env = dict(os.environ, KYOJIN_DENSE_FAST="0")                    # off pinned: the flag table default is on
    env["PYTHONPATH"] = here
    env["KYOJIN_HALF_ROWS"] = "0"                                    # the bitwise check is on the fp32 row FMAs
    off = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=here)
    assert off.stdout.strip() == "False", off.stderr[-400:]
    env["KYOJIN_DENSE_FAST"] = "1"
    on = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=here)
    assert on.stdout.strip() == "True", on.stderr[-400:]
    full = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, env=env, cwd=here)
    assert full.stdout.strip() == "True", (full.stdout, full.stderr[-600:])
