"""cidxfuse: decode indexer scores + chunk-tree top-k + candidate mask in one launch (kyojin_mlx.cidxfuse, flag cidxfuse, default 0).

Without Metal (and without mlx): the kernel text assembly (topk_tree's text with the score stage spliced in, mlx / idxfuse stubbed), the packed-key rewrite of
idxfuse.pack_source on it, the gate arithmetic, the flag row and the dispatch site. With Metal (Mac only): the fused kernel against
topk_tree(idxfuse.scores_fast(...)) and against the stock chain index_scores(rt) -> index_topk_rt -> candidate_mask_rt, bit for bit."""
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
KM = os.path.join(ROOT, "kyojin_mlx")


def _load(monkeypatch):
    """cidxfuse.py and topk_tree.py executed against stubs; the kernel texts and pack_source are the real ones read from idxfuse.py."""
    src = open(os.path.join(KM, "idxfuse.py")).read()
    consts, funcs = {}, []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in ("_LEVEL", "_HEADER", "_PK_HELPER", "_PK_LOAD"):
            v = node.value
            while isinstance(v, ast.BinOp):
                v = v.right
            consts[node.targets[0].id] = ast.literal_eval(v)
        if isinstance(node, ast.FunctionDef) and node.name == "pack_source":
            funcs.append(node)
    ns = {"_PK_LOAD": consts["_PK_LOAD"]}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "pack_source", "exec"), ns)
    core = types.ModuleType("mlx.core")
    for n in ("float16", "bfloat16", "float32", "int32", "uint32", "bool_", "array"):
        setattr(core, n, n)
    mxs = types.ModuleType("mlx")
    mxs.core = core
    fuse = types.ModuleType("kyojin_mlx.idxfuse")
    fuse._LEVEL, fuse._HEADER, fuse._PK_HELPER = consts["_LEVEL"], consts["_HEADER"], consts["_PK_HELPER"]
    fuse.pack_source = ns["pack_source"]
    fuse.KPS = 4
    fuse.C = types.SimpleNamespace(_u32=lambda t: t)
    fuse._counter = lambda n: n
    fuse.scores_supported = lambda nh, di, dt: 0 < nh <= 32 and di % 8 == 0 and dt in (core.float16, core.bfloat16, core.float32) and nh * di * 2 <= 24 * 1024
    fp = types.ModuleType("kyojin_mlx.fp4pool")
    fired = types.ModuleType("kyojin_mlx.fired")
    fired.hit = lambda *a, **k: None
    for name, mod in (("mlx", mxs), ("mlx.core", core), ("kyojin_mlx.idxfuse", fuse), ("kyojin_mlx.fp4pool", fp), ("kyojin_mlx.fired", fired)):
        monkeypatch.setitem(sys.modules, name, mod)
    fuse.fired = fired
    out = []
    for name in ("topk_tree", "cidxfuse"):
        spec = importlib.util.spec_from_file_location("kyojin_mlx." + name, os.path.join(KM, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "kyojin_mlx." + name, mod)
        spec.loader.exec_module(mod)
        monkeypatch.setattr(sys.modules["kyojin_mlx"], name, mod, raising=False)
        out.append(mod)
    return out[0], out[1], core


def _src(cf, k, kb, bs=8, tm=262144):
    cs, nc, r1, r2, ncm, rb = M.plan(tm, k, bs)
    return cf._source(k, kb, bs, cs, r1, r2, ncm, rb), cs


@pytest.mark.parametrize("k,kb", [(512, 0), (512, 2048), (256, 0), (1024, 64)])
def test_source_splices_the_score_stage_and_drops_the_buffer(monkeypatch, k, kb):
    _, cf, _ = _load(monkeypatch)
    s, cs = _src(cf, k, kb)
    assert "@" not in s and s.count("{") == s.count("}") and s.count("(") == s.count(")") and s.count("[") == s.count("]")
    assert "scores + ulong" not in s and "sc[" not in s.replace("scs[", "")             # no read of a score buffer left
    assert f"threadgroup T scs[{cs}]" in s and "scs[i]" in s and (("scs[t - c0]" in s) == (kb > 0))
    assert s.count("fma(") == 1 and s.count("simd_shuffle") == 1 and s.count("atomic_thread_fence") == 2
    assert s.index("threadgroup T scs") < s.index("uint kk[") and s.count("threadgroup_barrier(mem_flags::mem_threadgroup);\n    {\n        const uint Lc") <= 1
    assert "cand_a" in s and "cmask" in s and "const uint Tc = ip[4]" in s and "const uint Ck = ip[3]" in s


def test_score_stage_is_the_scores_fast_chain(monkeypatch):
    """Same arithmetic text as idxfuse._SCORES: FMA chain from 0 in channel order, T() cast, relu, x w rounded to T, head sum in order 0..NH-1 from 0."""
    _, cf, _ = _load(monkeypatch)
    t = open(os.path.join(KM, "idxfuse.py")).read()
    for line in ("acc[j] = fma(qv, kf[j][i], acc[j]);", "const T tile = T(acc[j]);", "const T rel = tile > T(0) ? tile : T(0);",
                 "const float pr = float(T(float(rel) * wv));", "for (uint h = 0; h < NH; ++h) total += simd_shuffle(pr, ushort(h));",
                 "for (uint dc = 0; dc < DI; dc += 8) {", "kf[j][i] = ok[j] ? float(kp[j * DI + dc + i]) : 0.0f;"):
        assert t.count(line) >= 2 and cf._SCORE_STAGE.count(line) == 1, line


def test_packed_key_rewrite_applies_to_the_fused_text(monkeypatch):
    _, cf, _ = _load(monkeypatch)
    s, _ = _src(cf, 512, 2048)
    p = sys.modules["kyojin_mlx.idxfuse"].pack_source(s)
    assert "pk_key<T>(kc, ksc, j, dc + i, DI)" in p and "const device uchar* kc = kcodes + (ulong(sess[u]) * Ck + t0) * (DI / 2);" in p and "keys +" not in p


def test_applies_gates(monkeypatch):
    tt, cf, core = _load(monkeypatch)
    f16 = core.float16
    assert cf.applies(1, 245760, 512, f16, 32, 128) and cf.applies(6, 245760, 512, core.bfloat16, 32, 128, 2048, 8)
    assert cf.applies(1, 524288, 512, f16, 32, 128)                         # chunk 8192, the edge
    assert not cf.applies(1, 524289, 512, f16, 32, 128)                     # chunk 16384: threadgroup memory
    assert not cf.applies(1, 1 << 20, 512, f16, 32, 128)
    assert not cf.applies(1, 8192, 512, f16, 32, 128) and not cf.applies(32, 245760, 512, f16, 32, 128)
    assert not cf.applies(1, 245760, 512, core.float32, 32, 128) and not cf.applies(1, 245760, 512, f16, 33, 128) and not cf.applies(1, 245760, 512, f16, 32, 100)
    for tm in (16384, 100000, 245760, 524288):
        cs, nc, r1, r2, ncm, rb = M.plan(tm, 512)
        assert cf.smem_bytes(32, 128, cs, ncm) <= cf.TG_BYTES


def test_flag_row_dispatch_site_and_default_off():
    f = FL.FLAGS["cidxfuse"]
    assert f.env == "KYOJIN_CIDXFUSE" and f.default == "0" and f.runtime and f.attrs == (("dsv4_core", "CIDXFUSE", FL._on), ("cidxfuse", "ENABLED", FL._on))
    core = open(os.path.join(KM, "dsv4_core.py")).read()
    mod = open(os.path.join(KM, "cidxfuse.py")).read()
    assert 'CIDXFUSE = os.environ.get("KYOJIN_CIDXFUSE", "0")' in core and 'os.environ.get("KYOJIN_CIDXFUSE", "0")' in mod
    assert core.index("elif cif is not None:") < core.index("elif rt and fuse:")        # after the exact filters, before the stock chain
    assert "CIDXFUSE and dsel is None and rt and not fuse" in core and "fres is None and fblk is None and fbs is None and f8 is None" in core


# ----------------------------------------------------------------------------------------------------------------- Metal (Mac)
def _mx():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    return mx


def _case(mx, dt, N, Tm, kind, seed):
    rng = np.random.default_rng(seed)
    dtype = getattr(mx, dt)
    q = rng.normal(size=(N, 32, 128)).astype(np.float32)
    keys = rng.normal(size=(N, Tm, 128)).astype(np.float32)
    if kind == "ties":                                                        # coarse values: many equal scores, +-0 mixed
        q, keys = np.round(q), np.round(keys)
        keys[:, ::5] = 0.0
    elif kind == "dup":                                                       # repeated key rows: exact score ties across chunks
        keys[:, 1::2] = keys[:, ::2][:, : keys[:, 1::2].shape[1]]
    w = rng.normal(size=(N, 32)).astype(np.float32)
    return mx.array(q).astype(dtype), mx.array(keys).astype(dtype), mx.array(w).astype(dtype)


@pytest.mark.parametrize("dt", ["float16", "bfloat16"])
@pytest.mark.parametrize("kind", ["n", "ties", "dup"])
@pytest.mark.parametrize("N,Tm,kb,usecand", [(1, 20000, 0, False), (3, 66000, 0, True), (2, 66000, 2048, False), (6, 131072, 0, False), (1, 262144, 2048, False)])
def test_fused_equals_tree_over_scores_fast_and_the_stock_chain(dt, kind, N, Tm, kb, usecand):
    mx = _mx()
    from kyojin_mlx import cidxfuse as CF
    from kyojin_mlx import dsv4_core as C
    from kyojin_mlx import idxfast as IX
    from kyojin_mlx import idxfuse as IXF
    from kyojin_mlx import topk_tree as TT
    q, keys, w = _case(mx, dt, N, Tm, kind, Tm + N)
    k, bs = 512, 8
    if not CF.applies(N, Tm, k, q.dtype, 32, 128, kb, bs):
        pytest.skip("gate")
    lens = mx.array([Tm, Tm - 37, 5000, 0, 1, 513][:N], dtype=mx.int32)
    sess = mx.arange(N, dtype=mx.int32)
    cand = None
    if usecand:
        cand = mx.array(np.random.default_rng(3).random((N, Tm)) < 0.5)
    idx, m = CF.fused(q, keys, w, lens, Tm, k, cand, sess, kb, bs)
    sc = IXF.scores_fast(q, keys, w, lens, Tm, cand, sess)
    ti, tm = TT.topk_tree(sc, lens, k, kb, bs) if TT.applies(N, Tm, k, q.dtype, kb, bs) else IX._run(sc, lens, k, kb, bs, kb > 0)
    ref = C.index_scores(q, keys, w, lens, Tm, cand, sess, rt=True)
    ri = C.index_topk_rt(ref, lens, k)
    mx.eval(idx, m if m is not None else idx, ti, ri)
    assert np.array_equal(np.array(idx), np.array(ti)) and np.array_equal(np.array(idx), np.array(ri))
    if kb:
        rm = C.candidate_mask_rt(ref, lens, kb, bs)
        L = np.array(lens)
        a, b, c = np.array(m), np.array(tm), np.array(rm)
        for u in range(N):
            assert np.array_equal(a[u, : L[u]], b[u, : L[u]]) and np.array_equal(a[u, : L[u]], c[u, : L[u]])
    else:
        assert m is None


def test_repeated_calls_and_changing_lens_reuse_one_kernel():
    mx = _mx()
    from kyojin_mlx import cidxfuse as CF
    from kyojin_mlx import idxfuse as IXF
    from kyojin_mlx import topk_tree as TT
    q, keys, w = _case(mx, "float16", 2, 66000, "n", 9)
    for step in range(6):
        lens = mx.array([66000 - 1000 * step, 40000 + step], dtype=mx.int32)
        idx, _ = CF.fused(q, keys, w, lens, 66000, 512)
        ti, _ = TT.topk_tree(IXF.scores_fast(q, keys, w, lens, 66000), lens, 512)
        assert np.array_equal(np.array(idx), np.array(ti))
    assert len([n for n in CF._k if n.startswith("kyojin_cidxfuse_512_0_8")]) == 1
