"""decrows: the attention front-end kernel equals split + q/k rms_norm + rope + layout bit for bit (rows 1..8, offsets, dtypes); the add + norm seam at 1..8 rows."""
import itertools
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in ("..",)]
import mlx.core as mx
import mlx.nn as nn
import pytest

from kyojin_mlx import decfuse as DF
from kyojin_mlx import decrows as DR

HQ, HK, D, DRP, THETA = 16, 4, 256, 64, 10000000


def stock(y, wq, wk, eps, B, L, offset):
    qo = y[..., :HQ * 2 * D].reshape(B, L, HQ, 2 * D)
    q, gate = mx.split(qo, 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    k = y[..., HQ * 2 * D:HQ * 2 * D + HK * D]
    v = y[..., HQ * 2 * D + HK * D:]
    q = mx.fast.rms_norm(q, wq, eps).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(B, L, HK, D), wk, eps).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, HK, D).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, DRP, traditional=False, base=THETA, scale=1.0, offset=offset)
    k = mx.fast.rope(k, DRP, traditional=False, base=THETA, scale=1.0, offset=offset)
    return q, k, v, gate


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("B,L", [(1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (1, 5), (1, 8)])
@pytest.mark.parametrize("offset", [0, 1, 37, 4093, 131071])
def test_qk_prep_bit_equal(dt, B, L, offset):
    mx.random.seed(B * 100 + L + offset)
    NT = HQ * 2 * D + 2 * HK * D
    y = (mx.random.normal((B, L, NT)) * 2).astype(dt)
    wq = (mx.random.normal((D,)) * 0.3 + 1).astype(dt)
    wk = (mx.random.normal((D,)) * 0.3 + 1).astype(dt)
    ref = stock(y, wq, wk, 1e-6, B, L, offset)
    got = DR.qk_prep(y, wq, wk, 1e-6, HQ, HK, D, DRP, THETA, 1.0, offset, B, L)
    mx.eval(ref, got)
    for name, r, g in zip("qkvg", ref, got):
        assert r.shape == g.shape, name
        assert mx.array_equal(r, g), (name, int(mx.sum(r != g).item()))


@pytest.mark.parametrize("rows", range(1, 9))
def test_add_rms_rows(rows):
    mx.random.seed(rows)
    H = 2560
    x = mx.random.normal((rows, 1, H)).astype(mx.bfloat16)
    r = mx.random.normal((rows, 1, H)).astype(mx.bfloat16)
    nm = nn.RMSNorm(H, eps=1e-6)
    nm.weight = (mx.random.normal((H,)) * 0.2 + 1).astype(mx.bfloat16)
    h, n = DF.add_rms(x, r, nm)
    h0 = x + r
    n0 = nm(h0)
    mx.eval(h, n, h0, n0)
    assert mx.array_equal(h, h0) and mx.array_equal(n, n0)


def stock_router(logits, k, norm):
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("E,k", [(16, 2), (64, 6), (128, 8), (256, 8), (512, 10)])
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("norm", [True, False])
def test_router_topk_bit_equal(dt, E, k, rows, norm):
    mx.random.seed(E * 10 + rows)
    lg = (mx.random.normal((rows, 1, E)) * 3).astype(dt)
    i0, s0 = stock_router(lg, k, norm)
    i1, s1 = DR.router_topk(lg, k, norm)
    mx.eval(i0, s0, i1, s1)
    assert i0.shape == i1.shape and s0.shape == s1.shape
    assert mx.array_equal(i0.astype(mx.uint32), i1), "indices (order included)"
    assert mx.array_equal(s0, s1), int(mx.sum(s0 != s1).item())


@pytest.mark.parametrize("E,k,norm", [(16, 2, True), (64, 6, True), (64, 6, False)])
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_sparse_moe_block_bit_equal(monkeypatch, E, k, norm, rows):
    from types import SimpleNamespace
    from mlx_lm.models import qwen3_next as Q3N
    monkeypatch.setattr(DR, "ENABLED", True)
    DR.patch_moe()
    args = SimpleNamespace(hidden_size=256, moe_intermediate_size=128, shared_expert_intermediate_size=128, norm_topk_prob=norm, num_experts=E, num_experts_per_tok=k)
    mx.random.seed(E + rows)
    blk = Q3N.Qwen3NextSparseMoeBlock(args)
    blk.set_dtype(mx.bfloat16)
    x = mx.random.normal((rows, 1, 256)).astype(mx.bfloat16)
    mx.eval(blk.parameters(), x)
    monkeypatch.setattr(DR, "ENABLED", False)
    y0 = blk(x)
    monkeypatch.setattr(DR, "ENABLED", True)
    y1 = blk(x)
    mx.eval(y0, y1)
    assert mx.array_equal(y0, y1)


def test_router_topk_ties():
    mx.random.seed(5)
    lg = (mx.random.normal((8, 1, 256)) * 0.02).astype(mx.bfloat16)           # near-uniform probabilities: many exact ties in bf16
    i0, s0 = stock_router(lg, 8, True)
    i1, s1 = DR.router_topk(lg, 8, True)
    mx.eval(i0, s0, i1, s1)
    assert mx.array_equal(i0.astype(mx.uint32), i1) and mx.array_equal(s0, s1)
