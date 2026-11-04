"""pf_moe_rowmap / pf_moe_epi: the prefill MoE without its two copies.

1. `KYOJIN_PF_MOE_ROWMAP`: the gate / up GEMM reads the token rows of x through the slot -> token map and applies the
   projection's input transform (suh, H128 / sqrt(128), k permutation) in the tile load, instead of `_rot` writing a
   2P x K fp16 copy.
2. `KYOJIN_PF_MOE_EPI`: gate + up + SwiGLU + routing weight in one kernel, so the [2P, inter] product is never written
   and re-read.

Synthetic only: random valid trellis banks (tests/test_moe.make_stack), never the real model, a few hundred MB at most.
Each lever is judged two ways: bit-identical to the current path (same tiles, same per-element arithmetic) and against
an independent numpy f32 reference (decoded weights, H128, SwiGLU, fp32 matmuls), with the current path's error printed
next to it.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import flags as FL, metal_nax as NX, metal_nax_gu as GU, moe as M, moe_prefill as MP
from test_moe import make_stack, E, TOPK, KS


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return cb, g, u, d


def route(rows, seed):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(rows)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
    return mx.array(ids), mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))


def xs(rows, kin, seed):
    return mx.array(np.random.default_rng(seed).standard_normal((rows, kin)).astype(np.float16) * np.float16(2.0))


def h128(v: np.ndarray) -> np.ndarray:
    """The kernels' H128 (in-place butterfly over the last axis of 128, / sqrt(128))."""
    idx = np.arange(v.shape[-1])
    v = v.astype(np.float32)
    for hs in (1, 2, 4, 8, 16, 32, 64):
        a = v.copy()
        b = v[..., idx ^ hs]
        v = np.where((idx & hs) != 0, b - a, a + b)
    return v * np.float32(1.0 / np.sqrt(128.0))


def ref_f32(moe, x, ids, wts):
    """Independent f32 reference: decode the fp16 weights, H128 / SwiGLU / routing weight / fp32 matmuls in numpy."""
    ids_np, wts_np, x_np = np.array(ids), np.array(wts), np.array(x, np.float32)
    T, topk = ids_np.shape
    flat = ids_np.reshape(-1)
    eids = mx.array(np.unique(flat).astype(np.int32))
    epos = {int(e): i for i, e in enumerate(np.array(eids))}
    g, u, d = moe.g, moe.u, moe.d
    wg = np.array(MP.decode_group((g,), eids, moe.cb), np.float32)
    wu = np.array(MP.decode_group((u,), eids, moe.cb), np.float32)
    wd = np.array(MP.decode_group((d,), eids, moe.cb), np.float32)
    out = np.zeros((T, d.nout), np.float32)
    for r in range(T):
        for j in range(topk):
            p = r * topk + j
            e = int(flat[p])
            i = epos[e]
            # one Hadamard on each side of the inner weight (rot and mid/final in the kernels)
            gv = h128(wg[i] @ h128(x_np[r] * np.array(g.suh[e], np.float32))) * np.array(g.svh[e], np.float32)
            uv = h128(wu[i] @ h128(x_np[r] * np.array(u.suh[e], np.float32))) * np.array(u.svh[e], np.float32)
            g4 = np.minimum(gv, moe.limit)
            with np.errstate(over="ignore"):
                a = (g4 / (1.0 + np.exp(-moe.alpha * g4))) * (np.clip(uv, -moe.limit, moe.limit) + moe.beta)
            if moe.wpre:
                a = a * np.float32(wts_np[r, j])
            dv = h128(wd[i] @ h128(a * np.array(d.suh[e], np.float32))) * np.array(d.svh[e], np.float32)
            y = dv
            if not moe.wpre:
                y = y * np.float32(wts_np[r, j])
            out[r] += y
    return out


def err(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


@pytest.fixture
def flags_off(monkeypatch):
    """Both levers off for the reference arm, restored after the test."""
    for name, attr in (("pf_moe_rowmap", "MOE_ROWMAP"), ("pf_moe_epi", "MOE_EPI")):
        monkeypatch.setattr(MP, attr, False)


def test_rowmap_gemm_bit_identical(layer):
    """The row-map gate/up GEMM equals rot(perm) + the device-A GEMM bit for bit, for every row count and bitrate mix."""
    cb, g, u, d = layer
    moe = M.MoeGather(g, u, d, TOPK)
    pf = MP.MoePrefill(moe)
    for rows in (7, 70, 200):
        ids, _ = route(rows, 200 + rows)
        x = xs(rows, g.kin, 210 + rows)
        order, proc, cnt, cum, inv, nhot = pf._plan(np.array(ids))
        flat_e = np.array(ids).reshape(-1)[order].astype(np.int32)
        tok = (order // TOPK).astype(np.uint32)
        n = int(cum[-1])
        nc = MP.cap_rows(n)
        eids = mx.array(proc.astype(np.int32))
        cl = mx.array(cum.astype(np.uint32))
        rowoff = mx.concatenate([cl[:-1], cl + n])
        ids_s, tok_s = mx.array(flat_e), mx.array(tok)
        mr, cfg = int(cnt.max()), NX.pick_cfg(cnt)
        xh = pf._rot(x, tok_s, ids_s, n, True, True, True, nc)
        ref = NX.expert_gemm((g, u), eids, rowoff, xh, 2 * nc, mr, cfg, ks=tuple(sorted(set(g.ks) | set(u.ks))))
        got = NX.expert_gemm((g, u), eids, rowoff, None, 2 * nc, mr, cfg, ks=tuple(sorted(set(g.ks) | set(u.ks))),
                             rowmap=(x, mx.concatenate([tok_s, tok_s]), g.suh, u.suh))
        mx.eval(ref, got)
        assert np.array_equal(np.array(ref).view(np.uint16), np.array(got).view(np.uint16)), (cb, rows)


def _class(moe, ids, want=None):
    """Host plan pieces of one bitrate class (the prefill pass unit); want: only experts whose gate/up bitrate matches."""
    pf = MP.MoePrefill(moe)
    order, proc, cnt, cum, inv, nhot = pf._plan(np.array(ids))
    kc = pf._kclass()
    keep = np.array([e in set(proc.tolist()) and (want is None or (moe.g.kh[e] == moe.u.kh[e] == want)) for e in range(len(kc))])
    proc = np.array([e for e in proc if keep[e]])
    assert len(proc) >= 2
    flat = np.array(ids).reshape(-1)
    sel = np.zeros(len(flat), np.bool_)
    for e in proc:
        sel |= flat == e
    order_c = order[sel[order]]
    flat_e = flat[order_c].astype(np.int32)
    tok = (order_c // TOPK).astype(np.uint32)
    counts = np.bincount(flat_e, minlength=moe.g.experts)[proc]
    cum_c = np.concatenate([[0], np.cumsum(counts)])
    return pf, proc, counts, cum_c, flat_e, tok, order_c


def test_epi_bit_identical(layer):
    """gate + up + SwiGLU in one kernel equals the GEMM pair + mid, with and without the row map."""
    cb, g, u, d = layer
    moe = M.MoeGather(g, u, d, TOPK)
    pf = MP.MoePrefill(moe)
    for rows in (23, 90):
        ids, wts = route(rows, 300 + rows)
        x = xs(rows, g.kin, 310 + rows)
        pf, proc, cnt, cum, flat_e, tok, order_c = _class(moe, ids, want=2)
        n, nc = int(cum[-1]), MP.cap_rows(int(cum[-1]))
        mr, cfg = int(cnt.max()), NX.pick_cfg(cnt)
        eids = mx.array(proc.astype(np.int32))
        cl = mx.array(cum.astype(np.uint32))
        rowoff = mx.concatenate([cl[:-1], cl + n])
        ids_s, tok_s = mx.array(flat_e), mx.array(tok)
        wts_s = mx.array(np.array(wts).reshape(-1)[order_c])
        xh = pf._rot(x, tok_s, ids_s, n, True, True, True, nc)
        inner = NX.expert_gemm((g, u), eids, rowoff, xh, 2 * nc, mr, cfg, ks=(2,))
        ref = pf._mid(inner, ids_s, wts_s, n, True, True, nc)
        for rmap in (False, True):
            got = GU.fused_gu_swiglu(g, u, d, eids, proc, cl, None if rmap else xh, n, mr, cfg, wts_s, moe,
                                     rowmap=(x, tok_s, g.suh, u.suh) if rmap else None)
            mx.eval(ref, got)
            a, b = np.array(ref)[:n], np.array(got)
            assert np.array_equal(a.view(np.uint16), b.view(np.uint16)), (cb, rows, rmap)


@pytest.mark.parametrize("wf", [(False, False), (True, True)])
@pytest.mark.parametrize("arm", [("pf_moe_rowmap",), ("pf_moe_epi",), ("pf_moe_rowmap", "pf_moe_epi")])
def test_pipeline_bit_identical(layer, arm, wf, flags_off):
    """The whole prefill chunk through MoePrefill: same bits as the current path for every flag combination (wpre + fq too)."""
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK, wpre=wf[0], fq=wf[1])
    ids, wts = route(150, 400)
    x = xs(150, g.kin, 401)
    for attr in ("MOE_ROWMAP", "MOE_EPI"):
        setattr(MP, attr, False)
    ref = np.array(MP.MoePrefill(dec)(x, ids, wts))
    for name in arm:
        setattr(MP, FL.FLAGS[name].attrs[0][1], True)
    got = np.array(MP.MoePrefill(dec)(x, ids, wts))
    assert np.array_equal(ref.view(np.uint32), got.view(np.uint32)), (cb, arm)


@pytest.mark.parametrize("arm", [(), ("pf_moe_rowmap",), ("pf_moe_epi",), ("pf_moe_rowmap", "pf_moe_epi")])
def test_error_vs_f32_reference(layer, arm, flags_off, capsys):
    """Max error vs an independent f32 reference (numpy, decoded fp16 weights): reported next to the current path's."""
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK)
    ids, wts = route(16, 500)
    x = xs(16, g.kin, 501)
    for name in ("pf_moe_rowmap", "pf_moe_epi"):
        setattr(MP, FL.FLAGS[name].attrs[0][1], False)
    ref = ref_f32(dec, x, ids, wts)
    cur = np.array(MP.MoePrefill(dec)(x, ids, wts))
    for name in arm:
        setattr(MP, FL.FLAGS[name].attrs[0][1], True)
    got = np.array(MP.MoePrefill(dec)(x, ids, wts))
    e_cur, e_new = err(cur, ref), err(got, ref)
    with capsys.disabled():
        print(f"{cb} {arm or ('current',)}: max error vs f32 ref current {e_cur:.2e}, arm {e_new:.2e}")
    assert e_new < 1e-2 and e_new < max(2 * e_cur, 1e-6), (cb, arm, e_cur, e_new)


def test_flags_runtime_switch(layer, monkeypatch):
    """POST /admin/flags path: set_many rewrites the module literal the prefill reads per call, default stays 0."""
    assert FL.FLAGS["pf_moe_rowmap"].default == "0" and FL.FLAGS["pf_moe_epi"].default == "0"
    monkeypatch.setenv("KYOJIN_PF_MOE_ROWMAP", "0")
    monkeypatch.setenv("KYOJIN_PF_MOE_EPI", "0")
    monkeypatch.setattr(MP, "MOE_ROWMAP", False)
    monkeypatch.setattr(MP, "MOE_EPI", False)
    FL.set_many({"pf_moe_rowmap": "1", "pf_moe_epi": "1"})
    assert MP.MOE_ROWMAP and MP.MOE_EPI
    # and the pipeline output is unchanged by the switch (bit-identical path)
    cb, g, u, d = layer
    dec = M.MoeGather(g, u, d, TOPK)
    ids, wts = route(40, 600)
    x = xs(40, g.kin, 601)
    on = np.array(MP.MoePrefill(dec)(x, ids, wts))
    FL.set_many({"pf_moe_rowmap": "0", "pf_moe_epi": "0"})
    mp_off = np.array(MP.MoePrefill(dec)(x, ids, wts))
    assert np.array_equal(on.view(np.uint32), mp_off.view(np.uint32))
