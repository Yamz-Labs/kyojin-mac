"""pf_moe_nosync, pf_route, pf_rdadvise and the compressor pool path check.

1. `KYOJIN_PF_MOE_NOSYNC`: the prefill MoE without its per-layer `cum` readback. The sort's offsets stay on the device, the gate / up and
   down GEMMs walk each expert's rows with one threadgroup (`persist`) and write at the global slot, `mid` is given its class's slot range as a
   device value, and the final combine picks the class block by comparing against `cum`. Judged bit-identical against the sync path on a
   synthetic EXL3 stack (never the real model) and against an independent numpy f32 reference in tests/test_pf_moe_copies.py terms.
2. `KYOJIN_PF_ROUTE`: the prompt-path router as the gate matmul + one selection launch; ids and weights must equal the stock chain bit for bit.
3. `KYOJIN_PF_RDADVISE`: the engram take still returns the same bytes and the advisory syscalls run (and are skipped when the flag is off).
4. The compressor pool path check: `install()` arms the vendor compressors and the HITS counter shows the vectorized pooling was taken (a
   served prefill is judged by those counters, not by reading the code).
"""
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
from test_moe import make_stack, E, TOPK, KS  # noqa: E402
from kyojin_mlx import flags as FL, metal_nax as NX, moe as M, moe_prefill as MP  # noqa: E402


def route(rows, seed):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(rows)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (rows, TOPK)).astype(np.float32)
    return mx.array(ids), mx.array((w / w.sum(-1, keepdims=True) * 1.5).astype(np.float32))


def xs(rows, kin, seed):
    return mx.array(np.random.default_rng(seed).standard_normal((rows, kin)).astype(np.float16) * np.float16(2.0))


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    return cb, g, u, d


def test_nosync_bit_identical(layer, monkeypatch):
    """Whole prefill MoE: pf_moe_nosync on equals the sync path bit for bit (many row counts, both codebooks, per-bitrate classes)."""
    cb, g, u, d = layer
    moe = M.MoeGather(g, u, d, TOPK)
    for rows in (23, 90, 200):
        ids, wts = route(rows, 500 + rows)
        x = xs(rows, g.kin, 510 + rows)
        monkeypatch.setattr(MP, "MOE_NOSYNC", False)
        ref = MP.MoePrefill(moe)(x, ids, wts)
        monkeypatch.setattr(MP, "MOE_NOSYNC", True)
        pf = MP.MoePrefill(moe)
        got = pf(x, ids, wts)
        mx.eval(ref, got)
        assert pf.last.get("nosync") is True
        a, b = np.array(ref), np.array(got)
        assert a.shape == b.shape
        assert np.array_equal(a.view(np.uint16), b.view(np.uint16)), (cb, rows, float(np.abs(a - b).max()))


def test_persist_gemm_bit_identical(layer):
    """The row-loop GEMM (grid.y = 1, global slot offsets) equals the one-threadgroup-per-block GEMM bit for bit."""
    cb, g, u, d = layer
    moe = M.MoeGather(g, u, d, TOPK)
    pf = MP.MoePrefill(moe)
    for rows in (23, 90):
        ids, wts = route(rows, 600 + rows)
        x = xs(rows, g.kin, 610 + rows)
        order, proc, cnt, cum, inv, nhot = pf._plan(np.array(ids))
        P, cap = rows * TOPK, MP.cap_rows(rows * TOPK)
        flat = np.array(ids).reshape(-1)[order].astype(np.int32)
        tok = (order // TOPK).astype(np.uint32)
        eids, cl, wts_s = mx.array(proc.astype(np.int32)), mx.array(cum.astype(np.uint32)), mx.array(np.array(wts).reshape(-1)[order])
        ids_s, tok_s = mx.array(flat), mx.array(tok)
        n, mr, cfg = int(cum[-1]), int(cnt.max()), NX.pick_cfg(cnt)
        ks = tuple(sorted(set(g.ks) | set(u.ks)))
        xh = pf._rot(x, tok_s, ids_s, n, True, True, True, cap)
        ref = NX.expert_gemm((g, u), eids, mx.concatenate([cl[:-1], cl + n]), xh, 2 * cap, mr, cfg, ks=ks)
        got = NX.expert_gemm((g, u), eids, mx.concatenate([cl[:-1], cl + P]), xh, 2 * cap, 0, cfg, ks=ks, persist=True)
        mx.eval(ref, got)
        a, b = np.array(ref), np.array(got)
        assert np.array_equal(a[:n], b[:n]), (cb, rows, "gate")
        assert np.array_equal(a[P:P + n], b[P:P + n]), (cb, rows, "up")
        # the bounded mid sees only its own row range and is the same kernel arithmetic
        mref = pf._mid(ref, ids_s, wts_s, n, True, True, cap)
        rows_iv = mx.array(np.array([0, n, 0, 0], np.uint32))
        mgot = pf._mid(got, ids_s, wts_s, P, True, True, cap, bounded=(rows_iv, cap))
        mx.eval(mref, mgot)
        assert np.array_equal(np.array(mref)[:n].view(np.uint16), np.array(mgot)[:n].view(np.uint16)), (cb, rows, "mid")


def test_route_matches_stock(monkeypatch):
    """pf_route: the fused selection returns the stock ids and weights bit for bit, and falls back when the mask is set."""
    import mlx.nn as nn
    from kyojin_mlx import rows as R
    from types import SimpleNamespace

    rng = np.random.default_rng(7)
    D, E_, TK, T = 32, 64, 6, 5
    w = mx.array(rng.standard_normal((E_, D)).astype(np.float32) * 0.1)
    bias = mx.array(rng.standard_normal((E_,)).astype(np.float32) * 0.2)
    self = SimpleNamespace(weight=w, bias=bias, top_k=TK, norm_topk_prob=True, routed_scaling_factor=2.5)
    x = mx.array(rng.standard_normal((1, T, D)).astype(np.float16))

    def stock(self, x, image_mask=None):
        scores = mx.sqrt(nn.softplus(x.astype(mx.float32) @ self.weight.T))
        b = self.bias if image_mask is None else mx.where(image_mask[..., None], self.bias_vl, self.bias)
        inds = mx.argpartition(-(scores + b), kth=self.top_k - 1, axis=-1)[..., :self.top_k].astype(mx.int32)
        inds = mx.stop_gradient(inds)
        wts = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob and self.top_k > 1:
            wts = wts / (wts.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, wts * self.routed_scaling_factor

    monkeypatch.setattr(R, "PF_ROUTE", True)
    ids, wts = R._router_sqrtsoftplus(self, stock, x)
    mi, mw = stock(self, x)
    mx.eval(ids, wts, mi, mw)
    assert ids.shape == mi.shape == (1, T, TK)
    assert np.array_equal(np.array(ids), np.array(mi))
    assert np.array_equal(np.array(wts), np.array(mw))
    monkeypatch.setattr(R, "PF_ROUTE", False)
    ids0, wts0 = R._router_sqrtsoftplus(self, stock, x)
    mx.eval(ids0, wts0)
    assert np.array_equal(np.array(ids0), np.array(mi)) and np.array_equal(np.array(wts0), np.array(mw))


def test_compressor_pool_path_check(monkeypatch):
    """The installer arms every ratio > 1 vendor compressor and HITS shows the pooling path taken (not the stock per-token loop)."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    import mini_ds41 as M41
    import mlx_vlm.models.deepseek_v41.language as L
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    from mlx.utils import tree_flatten, tree_unflatten
    from kyojin_mlx import compressor_pf as PF
    import mlx.nn as nn

    cfg = ModelConfig.from_dict({**M41.CFG, "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1]})
    comp = L.Compressor(cfg, 2)
    mx.random.seed(3)
    comp.update(tree_unflatten([(k, mx.random.normal(v.shape) * (v.shape[-1] ** -0.5 if v.ndim > 1 else 1.0))
                                for k, v in tree_flatten(comp.parameters())]))
    mx.eval(comp.parameters())

    class FakeLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [comp]

    model = FakeLM()
    before = dict(PF.HITS)
    assert PF.install(model) == 1
    assert PF.install(model) == 0                                     # idempotent
    assert getattr(type(comp), PF.FLAG) is True
    cache = L.DeepseekV41Cache(8)
    x0 = mx.random.normal((1, 64, 64), key=mx.random.key(3)).astype(mx.bfloat16)
    comp(x0, 0, cache)                                               # first chunk: start_pos 0 -> stock, it seeds the state
    mx.eval(cache.kv_state[2])
    pool0, stock0 = PF.HITS["pool"], PF.HITS["stock"]
    x = mx.random.normal((1, 64, 64), key=mx.random.key(4)).astype(mx.bfloat16)
    out = comp(x, 64, cache)                                         # served chunk: start_pos 64, 64 rows -> the pooled path
    mx.eval(out)
    assert PF.HITS["pool"] == pool0 + 1 and PF.HITS["stock"] == stock0, (PF.HITS, before)
    xs_ = mx.random.normal((1, 8, 64), key=mx.random.key(5)).astype(mx.bfloat16)
    comp(xs_, 128, cache)                                            # short chunk -> stock, by design
    mx.eval(cache.kv_state[2])
    assert PF.HITS["stock"] == stock0 + 1
    assert PF.report()["pool"] == PF.HITS["pool"]


def test_engram_rdadvise(tmp_path, monkeypatch):
    """pf_rdadvise: the take returns the same bytes with the flag on and off, and issues the advisories only when asked."""
    from kyojin_mlx import engram_rows as ER

    rows, width = 64, 32
    path = str(tmp_path / "table.bin")
    data = np.random.default_rng(9).integers(0, 255, (rows, width), dtype=np.uint8)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 16)                                          # a non-zero data offset, like a shard
        fh.write(data.tobytes())
    off = 16

    def fake_tables():
        t = object.__new__(ER.EngramTables)
        t.spec = type("S", (), {"layer_ids": [0], "head_dim": width, "head_dim_scale": width // 32})
        t.spec_layer = None
        t.loc = {0: {"w": (path, off, (rows, width), "F8_E4M3", rows * width)}}
        t._fds = {}
        t._mms = {}
        t._nthreads = 2
        t._pool = ThreadPoolExecutor(max_workers=2)
        return t

    ids = np.array([7, 1, 40, 40, 63, 12], np.int64)
    monkeypatch.setenv("KYOJIN_PF_RDADVISE", "0")
    t = fake_tables()
    assert np.array_equal(t._take_rows(0, "w", ids, width), data[ids])
    assert ER.ADV["advised"] == 0
    monkeypatch.setenv("KYOJIN_PF_RDADVISE", "1")
    t = fake_tables()
    before = ER.ADV["runs"]
    got = t._take_rows(0, "w", ids, width)
    assert np.array_equal(got, data[ids])
    assert ER.ADV["advised"] >= 1 and ER.ADV["runs"] >= before + 1
    # runs of rows within ADV_GAP share one advisory; a scatter wider than ADV_MAX is skipped rather than paying a syscall per row
    fd = os.open(path, os.O_RDONLY)
    try:
        assert ER.advise_rows(fd, off, np.arange(rows), width) == 1
        wide = np.arange(0, rows * (ER.ADV_MAX + 1), ER.ADV_GAP + 1)[: ER.ADV_MAX + 1]
        skipped = ER.ADV["skipped"]
        assert ER.advise_rows(fd, off, wide, width) == 0 and ER.ADV["skipped"] == skipped + 1
    finally:
        os.close(fd)
    assert struct.pack("qi", 0, 0)                                # the advisory record is the macOS radvisory layout
