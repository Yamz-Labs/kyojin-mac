"""Chunk-tree top-k (kyojin_mlx.topk_tree, flag topk_tree).

Without Metal: the numpy model of the whole algorithm (topk_tree_model) against a brute-force sort and mask, the plan constants, the kernel source assembly
(mlx and idxfuse stubbed, `_LEVEL` read from idxfuse.py), and the flag row. With Metal: the kernel against idxfast._run (7-launch radix chain), same indices,
same order, same -1 tail, same mask on t < lens."""
import ast
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

from kyojin_mlx import flags as FL
from kyojin_mlx import topk_tree_model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _small(monkeypatch, maxc=4096, minchunk=2048):
    monkeypatch.setattr(M, "MAXC", maxc)
    monkeypatch.setattr(M, "MIN_CHUNK", minchunk)


def _rows(rng, kind, Tm):
    s = rng.normal(size=(1, Tm)) * 3
    if kind == "ties":
        s = np.round(s)
    elif kind == "same":
        s = np.zeros((1, Tm))
    elif kind == "int":
        s = rng.integers(-3, 3, size=(1, Tm)).astype(float)
    elif kind == "inf":
        s[:, rng.random(Tm) < 0.9] = -np.inf
    s = s.astype(np.float16)
    if kind in ("ties", "int"):
        s[0, ::7] = np.float16(-0.0)
        s[0, 3::11] = np.float16(0.0)
    return s


@pytest.mark.parametrize("kind", ["n", "ties", "same", "int", "inf"])
@pytest.mark.parametrize("Tm,L", [(6000, 6000), (6000, 4000), (9000, 9000), (9000, 100), (9000, 0), (8192, 8192), (9000, 2049), (9000, 2048)])
def test_model_topk_equals_brute_force(monkeypatch, kind, Tm, L):
    _small(monkeypatch)
    rng = np.random.default_rng(Tm + L)
    k = 64
    for _ in range(3):
        s = _rows(rng, kind, Tm)
        ln = np.array([L])
        assert np.array_equal(M.topk(s, ln, k), M.ref_topk(s, ln, k))


def test_model_rows_with_different_lens_and_k_above_chunk(monkeypatch):
    _small(monkeypatch)
    rng = np.random.default_rng(1)
    s = np.concatenate([_rows(rng, "ties", 7000) for _ in range(4)])
    ln = np.array([7000, 2500, 31, 0])
    for k in (16, 200, 1500):
        assert np.array_equal(M.topk(s, ln, k), M.ref_topk(s, ln, k))


@pytest.mark.parametrize("kind", ["n", "ties", "inf", "same"])
@pytest.mark.parametrize("L", [9000, 8999, 8001, 100, 9, 8, 1])
@pytest.mark.parametrize("kb", [2, 5, 50, 2000])
def test_model_mask_equals_brute_force(kind, L, kb):
    rng = np.random.default_rng(L + kb)
    s = _rows(rng, kind, 9000)
    assert np.array_equal(M.mask_row(s[0], L, kb, 8), M.ref_mask_row(s[0], L, kb, 8))
    assert np.array_equal(M.mask_row(s[0], L, kb, 4), M.ref_mask_row(s[0], L, kb, 4))


def test_key16_is_monotone_for_fp16_and_bf16_patterns():
    f = np.sort(np.random.default_rng(0).normal(size=4000).astype(np.float32) * 30)
    h = f.astype(np.float16)
    assert np.all(np.diff(M.key16(h.view(np.uint16)).astype(np.int64)) >= 0)
    bits = (f.view(np.uint32) >> 16).astype(np.uint16)                    # truncated bf16 patterns of the sorted floats
    assert np.all(np.diff(M.key16(bits).astype(np.int64)) >= 0)
    assert M.key16(np.array([0x8000], np.uint16))[0] == M.key16(np.array([0], np.uint16))[0]


def test_plan_constants_do_not_move_inside_a_class():
    """The served width moves every token: a kernel constant that follows it would build a pipeline (~200 ms) per token."""
    k = 512
    base = M.plan(16384, k)
    for tm in (16385, 32768, 100000, 131072, 200003, 262144):
        cs, nc, r1, r2, ncm, rb = M.plan(tm, k)
        assert (cs, r1, r2, ncm, rb) == (base[0], base[2], base[3], base[4], base[5]), tm
        assert nc * k <= r2 * M.NT and nc <= ncm
    assert M.plan(262144, k)[0] == 4096 and M.plan(1 << 20, k)[0] == 16384 and M.plan(1 << 21, k)[0] == 32768
    with pytest.raises(ValueError):
        M.plan(1 << 22, k)                                                  # more than 15 local bits: the caller keeps the radix select


def test_plan_candidate_slots_fit_the_registers_of_the_last_chunk():
    for k in (128, 256, 512, 1024):
        for tm in (16384, 65536, 262144, 524288):
            cs, nc, r1, r2, ncm, rb = M.plan(tm, k)
            assert cs % M.NT == 0 and cs <= M.MAX_CHUNK and nc * k <= r2 * M.NT and r2 <= 33
            assert (cs // 8 * nc) <= rb * M.NT


# ----------------------------------------------------------------------------------------------------------------- kernel source, no mlx
def _stubbed_module(monkeypatch):
    src = open(os.path.join(ROOT, "kyojin_mlx", "idxfuse.py")).read()
    consts = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in ("_LEVEL", "_HEADER"):
            v = node.value
            while isinstance(v, ast.BinOp):
                v = v.right
            consts[node.targets[0].id] = ast.literal_eval(v)
    mxs = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    for n in ("float16", "bfloat16", "float32", "int32", "uint32", "bool_", "array"):
        setattr(core, n, n)
    mxs.core = core
    fuse = types.ModuleType("kyojin_mlx.idxfuse")
    fuse._LEVEL, fuse._HEADER = consts["_LEVEL"], consts["_HEADER"]
    fuse.C = types.SimpleNamespace(_u32=lambda t: t)
    fuse._counter = lambda n: n
    monkeypatch.setitem(sys.modules, "mlx", mxs)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "kyojin_mlx.idxfuse", fuse)
    spec = importlib.util.spec_from_file_location("kyojin_mlx.topk_tree", os.path.join(ROOT, "kyojin_mlx", "topk_tree.py"))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "kyojin_mlx.topk_tree", mod)
    spec.loader.exec_module(mod)
    return mod, core


def _src(mod, k, kb, bs=8, tm=262144):
    cs, nc, r1, r2, ncm, rb = M.plan(tm, k, bs)
    return mod._source(k, kb, bs, cs, r1, r2, ncm, rb), (cs, r1, r2, ncm, rb)


@pytest.mark.parametrize("k,kb", [(512, 0), (512, 2048), (256, 0), (1024, 64)])
def test_source_has_no_placeholder_left_and_balanced_braces(monkeypatch, k, kb):
    mod, _ = _stubbed_module(monkeypatch)
    s, (cs, r1, r2, ncm, rb) = _src(mod, k, kb)
    assert "@" not in s
    assert s.count("{") == s.count("}") and s.count("(") == s.count(")") and s.count("[") == s.count("]")
    assert f"uint kk[{r1}]" in s and f"uint kk[{r2}]" in s and f"coff[{ncm} + 1]" in s
    assert ("bk_a" in s) == (kb > 0) and ("mout[" in s) == (kb > 0)
    assert (f"uint kb[{rb}]" in s) == (kb > 0)
    assert s.count("atomic_thread_fence") == 2 and "lastg" in s


def test_source_levels_use_the_wanted_counts(monkeypatch):
    mod, _ = _stubbed_module(monkeypatch)
    s, _ = _src(mod, 512, 2048)
    assert "(Lc) <= (512)" in s and "(nv) <= (512)" in s and "(nd) <= (2047)" in s


def test_applies_gates(monkeypatch):
    mod, core = _stubbed_module(monkeypatch)
    f16 = core.float16
    assert mod.applies(1, 262144, 512, f16) and mod.applies(7, 32768, 512, core.bfloat16)
    assert not mod.applies(1, 8192, 512, f16)                              # below TOPK_TREE_MIN
    assert not mod.applies(64, 262144, 512, f16)                           # prefill tile
    assert not mod.applies(1, 262144, 512, core.float32)
    assert not mod.applies(1, 262144, 64, f16) and not mod.applies(1, 262144, 4096, f16)
    assert mod.applies(1, 262144, 512, f16, 2048, 8)
    assert not mod.applies(1, 1 << 20, 512, f16, 2048, 8)                  # block words of the whole row past RB_MAX
    assert mod.applies(1, 1 << 20, 512, f16)                               # the plain variant has no such bound
    assert not mod.applies(1, 262144, 512, f16, 1, 8) and not mod.applies(1, 262144, 512, f16, 2048, 3)
    monkeypatch.setattr(mod, "MIN_KEYS", 0)
    assert mod.applies(1, 4096, 512, f16)


# ----------------------------------------------------------------------------------------------------------------- flag rows
def test_flag_rows_are_default_off_and_aligned_with_their_modules():
    f = FL.FLAGS["topk_tree"]
    assert f.env == "KYOJIN_TOPK_TREE" and f.default == "0" and f.runtime and f.attrs == (("idxfast", "TOPK_TREE", FL._on),)
    g = FL.FLAGS["topk_tree_min"]
    assert g.env == "KYOJIN_TOPK_TREE_MIN" and g.kind == "int" and g.attrs[0][:2] == ("topk_tree", "MIN_KEYS")
    fast = open(os.path.join(ROOT, "kyojin_mlx", "idxfast.py")).read()
    tree = open(os.path.join(ROOT, "kyojin_mlx", "topk_tree.py")).read()
    assert 'os.environ.get("KYOJIN_TOPK_TREE", "0")' in fast
    assert f'os.environ.get("KYOJIN_TOPK_TREE_MIN", "{g.default}")' in tree


# ----------------------------------------------------------------------------------------------------------------- Metal
@pytest.mark.parametrize("dt", ["float16", "bfloat16"])
@pytest.mark.parametrize("Tm,N", [(20000, 1), (66000, 3), (262144, 1), (300000, 7)])
def test_kernel_equals_radix_chain(dt, Tm, N):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    from kyojin_mlx import idxfast as IX
    from kyojin_mlx import topk_tree as TT
    rng = np.random.default_rng(Tm)
    dtype = getattr(mx, dt)
    for kind in ("n", "ties", "inf", "same"):
        raw = np.concatenate([_rows(rng, kind, Tm) for _ in range(N)]).astype(np.float32)
        s = mx.array(raw).astype(dtype)
        lens = mx.array([Tm, Tm - 37, 5000, 0, 1, 513, Tm][:N], dtype=mx.int32)
        for k, kb, bs in ((512, 0, 8), (512, 2048, 8), (128, 0, 8), (512, 64, 4)):
            if not TT.applies(N, Tm, k, dtype, kb, bs):                     # the only legal refusal: the block words of the row do not fit RB_MAX per thread
                assert kb and M.plan(Tm, k, bs)[5] > TT.RB_MAX and -(-Tm // bs) > TT.RB_MAX * TT.NT, (Tm, k, kb, bs)
                continue
            idx, m = TT.topk_tree(s, lens, k, kb, bs)
            ref_i, ref_m = IX._run(s, lens, k, kb if kb else 2, bs, kb > 0)
            mx.eval(idx, ref_i)
            assert np.array_equal(np.array(idx), np.array(ref_i)), (kind, k, kb)
            if kb:
                a, b, ln = np.array(m), np.array(ref_m), np.array(lens)
                for u in range(N):
                    assert np.array_equal(a[u, :ln[u]], b[u, :ln[u]]), (kind, u)


def test_gate_refuses_the_block_variant_only_when_the_block_words_do_not_fit():
    """300000 keys, k 512, 64 blocks of 4: the chunk is 8192 keys, the row has 75000 block words and RB_MAX x 1024 = 65536 slots. The Metal kernel test used to assert
    `applies` for it (2 failures, fp16 / bf16); a refusal is the gate working, the caller keeps the radix select."""
    mx = pytest.importorskip("mlx.core")
    from kyojin_mlx import topk_tree as TT
    assert M.plan(300000, 512, 4)[5] == 128 > TT.RB_MAX and -(-300000 // 4) > TT.RB_MAX * TT.NT
    assert not TT.applies(7, 300000, 512, mx.float16, 64, 4)
    assert TT.applies(7, 300000, 512, mx.float16, 0, 4)                       # no mask: no block words
    assert TT.applies(7, 300000, 512, mx.bfloat16, 2048, 8)                   # 37500 block words fit
    for Tm in (20000, 66000, 262144):
        for k, kb, bs in ((512, 0, 8), (512, 2048, 8), (128, 0, 8), (512, 64, 4)):
            assert TT.applies(7, Tm, k, mx.float16, kb, bs), (Tm, k, kb, bs)
