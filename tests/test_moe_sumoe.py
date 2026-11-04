"""KYOJIN_PORT_SUMOE: the shared expert rides the routed MoE launches (moe_sumoe). Routed part and shared slot are bit-identical to the plain gather
path (pinned k-split); against the stock dense shared chain (bf16 gate / up / swiglu) the output agrees to bf16 rounding, KL and top-1 on a random
head inside the owner's rounding-level gate (mean KL <= 0.002, top-1 >= 99 %). Needs a Mac (Metal)."""
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import moe as M  # noqa: E402
from kyojin_mlx import moe_sumoe as SU  # noqa: E402
from test_moe import E, KS, TOPK, _route, _x, make_stack  # noqa: E402
from test_moe_union import route  # noqa: E402

KSH = 3
TG = dict(tgk_gu=2, tgk_dn=2)


@pytest.fixture(scope="module", params=["mul1", "mcg"])
def layer(request):
    cb = request.param
    kin, ninter = 256, 384
    g, _ = make_stack(kin, ninter, KS, cb, 1)
    u, _ = make_stack(kin, ninter, KS, cb, 2)
    d, _ = make_stack(ninter, kin, KS, cb, 3)
    sg, gp = make_stack(kin, ninter, [KSH], cb, 11)
    su, up = make_stack(kin, ninter, [KSH], cb, 12)
    sd, dp = make_stack(ninter, kin, [KSH], cb, 13)
    parts = [gp[0], up[0], dp[0]]
    ext = SU.with_shared(g, u, d, [(mx.array(p[0]), mx.array(p[1]), mx.array(p[2])) for p in parts])
    return cb, ext, parts


def _bf16(a):
    return a.astype(mx.bfloat16).astype(mx.float32)


def _ref(ext, rows, ids, w, x, fq8, sfq=False, **kw):
    """Routed sum (plain gather over the extended stacks) + the shared expert as a one-pair gather, rounded to bf16, added last."""
    g, u, d = ext
    routed = M.MoeGather(g, u, d, TOPK, grouped=False, **TG, **kw)
    kw1 = dict(kw, fq=kw.get("fq", False) and sfq)
    shared = M.MoeGather(g, u, d, 1, grouped=False, **TG, **kw1)
    yr = routed(x, ids, w, fq8)
    ys = shared(x, mx.full((rows, 1), E, dtype=mx.int32), mx.ones((rows, 1), dtype=mx.float32), fq8)
    return yr + _bf16(ys)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["distinct", "identical", "partial"])
@pytest.mark.parametrize("fq", [False, True])
@pytest.mark.parametrize("sfq", [False, True])
def test_matches_gather_plus_shared(layer, rows, mode, fq, sfq):
    cb, ext, _ = layer
    ids, w = route(rows, mode, 7)
    x = _x(rows, ext[0].kin, 8)
    kw = dict(limit=10.0, wpre=True, fq=fq)
    sm = SU.MoeSharedFused(*ext, TOPK, shared_fq=sfq, **TG, **kw)
    got, want = sm(x, ids, w, fq8=fq), _ref(ext, rows, ids, w, x, fq, sfq, **kw)
    mx.eval(got, want)
    assert np.array_equal(np.array(got).view(np.uint32), np.array(want).view(np.uint32)), (rows, mode, fq)


def test_launch_count(layer):
    """Five kernels a layer: rot, gemv gate+up, mid, gemv down, final."""
    cb, ext, _ = layer
    sm = SU.MoeSharedFused(*ext, TOPK, **TG, limit=10.0, wpre=True)
    ids, w = _route(2, 4)
    x = _x(2, ext[0].kin, 5)
    mx.eval(sm(x, ids, w))
    keys = [k for k in sm._kn]
    assert sum(1 for k in keys if k[0] in ("rotsh", "midsh", "finsh", "gemv")) == 5, keys


def _dense_shared(ext, parts, cb, x, fq8):
    from kyojin_mlx.exl3_linear import Exl3Linear
    mk = lambda p: Exl3Linear(mx.array(p[0]), mx.array(p[1]), mx.array(p[2]), cb=cb, out_dtype=mx.bfloat16)
    lg, lu, ld = (mk(p) for p in parts)
    xb = x.astype(mx.bfloat16)
    gt, up = lg(xb), lu(xb)
    gt, up = mx.minimum(gt, 10.0), mx.clip(up, -10.0, 10.0)
    return ld((gt * mx.sigmoid(gt)) * up).astype(mx.float32)


@pytest.mark.parametrize("rows", [1, 4])
def test_close_to_stock_dense_shared_chain(layer, rows):
    cb, ext, parts = layer
    ids, w = route(rows, "distinct", 21)
    x = _x(rows, ext[0].kin, 22)
    kw = dict(limit=10.0, wpre=True, fq=False)
    sm = SU.MoeSharedFused(*ext, TOPK, **TG, **kw)
    got = sm(x, ids, w)
    routed = M.MoeGather(*ext, TOPK, grouped=False, **TG, **kw)(x, ids, w)
    stock = routed + _dense_shared(ext, parts, cb, x, False)
    mx.eval(got, stock)
    a, b = np.array(got), np.array(stock)
    rel = np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
    assert rel < 2e-2, rel
    rng = np.random.default_rng(0)
    head = rng.standard_normal((a.shape[1], 512)).astype(np.float32) / np.sqrt(a.shape[1]) * 8.0
    la, lb = a @ head, b @ head
    pa, pb = np.exp(la - la.max(-1, keepdims=True)), np.exp(lb - lb.max(-1, keepdims=True))
    pa, pb = pa / pa.sum(-1, keepdims=True), pb / pb.sum(-1, keepdims=True)
    kl = (pb * (np.log(pb + 1e-12) - np.log(pa + 1e-12))).sum(-1).mean()
    assert kl <= 0.002 and (la.argmax(-1) == lb.argmax(-1)).all(), kl


# ---- end to end: build_moe on a tiny MoE block, flag ON vs OFF ----------------------------------------------------------------------------

import mlx.nn as nn  # noqa: E402

from kyojin_mlx import patch as P  # noqa: E402
from kyojin_mlx.archs import MoeSpec  # noqa: E402
from kyojin_mlx.exl3_linear import Exl3Linear  # noqa: E402

LIMIT = 10.0


class _Act(nn.Module):
    limit = LIMIT


class _Old(nn.Module):
    """What build_moe reads from the replaced switch module."""
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Module()
        self.gate_proj.weight = mx.zeros((E, 1, 1))
        self.activation = _Act()


class _Gate(nn.Module):
    def __init__(self, kin, seed):
        super().__init__()
        self.top_k = TOPK
        self.w = mx.array(np.random.default_rng(seed).standard_normal((E, kin)).astype(np.float32))

    def __call__(self, x):
        sc = mx.sigmoid(x.astype(mx.float32) @ self.w.T)
        ids = mx.argpartition(-sc, kth=TOPK - 1, axis=-1)[..., :TOPK].astype(mx.int32)
        w = mx.take_along_axis(sc, ids, axis=-1)
        return ids, w / w.sum(axis=-1, keepdims=True)


class _Shared(nn.Module):
    def __init__(self, parts, cb):
        super().__init__()
        mk = lambda p: Exl3Linear(mx.array(p[0]), mx.array(p[1]), mx.array(p[2]), cb=cb)
        self.gate_proj, self.up_proj, self.down_proj = (mk(p) for p in parts)
        self.swiglu_limit = LIMIT

    def __call__(self, x):
        g, u = self.gate_proj(x).astype(mx.float32), self.up_proj(x).astype(mx.float32)
        a = nn.silu(mx.minimum(g, LIMIT)) * mx.clip(u, -LIMIT, LIMIT)
        return self.down_proj(a.astype(x.dtype))


class _Block(nn.Module):
    """combine: shared first, routed call adds it (mlx-vlm MoE); slot: routed slot sum, then the shared add (DeepseekV41MoE)."""
    def __init__(self, parts, cb, slot, seed=5):
        super().__init__()
        self.slot = slot
        self.gate = _Gate(parts[0][0].shape[0] * 16, seed)
        self.switch_mlp = _Old()
        self.shared_experts = _Shared(parts, cb)

    def __call__(self, x):
        ids, w = self.gate(x)
        if self.slot:
            y = self.switch_mlp(x, ids, w).astype(mx.float32).sum(-2)
            return (y + self.shared_experts(x).astype(mx.float32)).astype(x.dtype)
        shared = self.shared_experts(x)
        return self.switch_mlp(x, ids, w, shared)


def _kl_top1(a, b, seed=0):
    head = mx.array(np.random.default_rng(seed).standard_normal((a.shape[-1], 512)).astype(np.float32) / np.sqrt(a.shape[-1]))
    la, lb = (z.astype(mx.float32) @ head * 8 for z in (a, b))
    pa, pb = mx.softmax(la, -1), mx.softmax(lb, -1)
    kl = mx.mean(mx.sum(pa * (mx.log(pa + 1e-20) - mx.log(pb + 1e-20)), -1)).item()
    return kl, float(mx.mean((mx.argmax(la, -1) == mx.argmax(lb, -1)).astype(mx.float32)).item())


@pytest.mark.parametrize("slot", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4, 9])
def test_build_moe_flag_on_matches_off(layer, monkeypatch, slot, rows):
    monkeypatch.setattr(P, "MOE_MAX_ROWS", int(os.environ.get("KYOJIN_MOE_ROWS", "4")))  # rows.enable (fuse tests earlier in the suite) raises it: this test covers the decode band
    monkeypatch.setattr(P, "PF_CROSS_NAX", 2.5)        # 12 toy experts, top-6: the Ultra crossover (1.75 rows per expert) would send 4 rows to the S3 prefill path
    cb, ext, parts = layer
    kin = ext[0].kin
    g, u, d = (M.Stack(s.words, s.offset[:E + 1], s.k[:E], s.suh[:E], s.svh[:E], s.kt, s.nt, s.cb) for s in ext)     # routed-only stacks
    mspec = MoeSpec(roles={}, convention="slot" if slot else "combine", weight_before_down=slot)
    out = {}
    for on in (False, True):
        monkeypatch.setattr(SU, "SUMOE", on)
        blk = _Block(parts, cb, slot)
        rep = {}
        new = P.build_moe(blk.switch_mlp, blk, dict(gate=g, up=u, down=d), mspec, True, rep)
        assert (new.msh is not None) == on and rep.get("sumoe_layers", 0) == int(on)
        blk.switch_mlp = new
        if on:
            calls = []
            inner = new.msh.__class__.__call__
            monkeypatch.setattr(new.msh.__class__, "__call__", lambda self, *a, **k: (calls.append(1), inner(self, *a, **k))[1])
        x = _x(rows, kin, 21).astype(mx.float16).reshape(1, rows, kin)
        y = blk(x)
        mx.eval(y)
        out[on] = y
        if on:
            assert bool(calls) == (rows <= P.MOE_MAX_ROWS), (rows, calls)          # decode rows run the fused launches, wider chunks the stock path
    a, b = out[True].reshape(rows, -1), out[False].reshape(rows, -1)
    err = float(mx.linalg.norm((a - b).astype(mx.float32)) / mx.linalg.norm(b.astype(mx.float32)))
    kl, top1 = _kl_top1(a, b)
    assert err < 2e-2 and kl <= 0.002 and top1 >= 0.99, (err, kl, top1)


def test_fastdec_close_at_r1(layer):
    """KYOJIN_MOE_FASTDEC reaches the sumoe gather launch at R1 (cheaper mul1 decode, not bit-identical): rel L2 and KL inside the gate."""
    cb, ext, _ = layer
    if cb != "mul1":
        pytest.skip("mul1fd is a mul1 decode")
    ids, w = route(1, "distinct", 7)
    x = _x(1, ext[0].kin, 8)
    ref = SU.MoeSharedFused(*ext, TOPK, **TG, limit=10.0, wpre=True)
    fd = SU.MoeSharedFused(*ext, TOPK, **TG, limit=10.0, wpre=True)
    fd.fastdec = True
    a, b = fd(x, ids, w), ref(x, ids, w)
    mx.eval(a, b)
    assert float(mx.linalg.norm(a - b) / mx.linalg.norm(b)) < 2e-3
    kl, top1 = _kl_top1(a, b)
    assert kl <= 0.002 and top1 >= 0.99


def _unique_bytes(arrays):
    seen, total = set(), 0
    for a in arrays:
        if id(a) not in seen:
            seen.add(id(a))
            total += a.nbytes
    return total


def test_wired_layer_releases_routed_stacks_from_pack(layer, monkeypatch):
    """patch_model's stacks loop (KYOJIN_PORT_SUMOE=1): once the extended stacks are wired, the routed-only tensors leave the pack
    dict, so the load never holds two copies of the MoE weights (the 172 GiB pack does not fit twice)."""
    monkeypatch.setattr(SU, "SUMOE", True)
    cb, ext, parts = layer
    kin = ext[0].kin
    g, u, d = (M.Stack(s.words, s.offset[:E + 1], s.k[:E], s.suh[:E], s.svh[:E], s.kt, s.nt, s.cb) for s in ext)
    t = {}
    for n, s in (("gate", g), ("up", u), ("down", d)):
        t.update({f"moe.{n}.{suf}": a for suf, a in (("trellis", s.words), ("offset", s.offset), ("k", s.k), ("suh", s.suh), ("svh", s.svh))})
    t["moe.gate.shape"] = mx.array([s.kt, s.nt], dtype=mx.uint32)
    t["other.tensor"] = mx.zeros((4,), dtype=mx.float16)
    blk = _Block(parts, cb, True)
    new = P.build_moe(blk.switch_mlp, blk, dict(gate=g, up=u, down=d), MoeSpec(roles={}, convention="slot", weight_before_down=True), True, {})
    assert new.msh is not None
    ext_arrays = [a for s in (new.moe.g, new.moe.u, new.moe.d) for a in (s.words, s.offset, s.k, s.suh, s.svh)]
    before = _unique_bytes(list(t.values()) + ext_arrays)
    orig = _unique_bytes(list(t.values()))
    assert before > 1.5 * orig                     # the pack pins the originals, the extended stacks are copies
    P._release_superseded_stacks(t, {"gate": "moe.gate", "up": "moe.up", "down": "moe.down"})
    assert t["other.tensor"].shape == (4,)         # unrelated entries stay
    after = _unique_bytes(list(t.values()) + ext_arrays)
    mx.eval(new.moe.g.words)                       # the extended stacks still work after the release
    assert after < 1.2 * orig
    assert new.moe.g.experts == E + 1
