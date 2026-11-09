# SPDX-License-Identifier: Apache-2.0
"""attn_core_fuse (dsv4_core.ATTN_CORE_FUSE): the kv chain also writes the stacked window buffer the commit keeps. Core outputs and every cache
buffer must equal the old path bit for bit, on all layer kinds (local, kv source, reuse, reindex), over several decode steps, ring wrapped."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decstage_fixture as FX  # noqa: E402

from kyojin_mlx import archs, dsv4_core as DC, rows as RW  # noqa: E402

CFG = dict(FX.REAL, num_hidden_layers=6, compress_ratios=[0, 2, 2, 1, 1, 1], kv_source_layer_ids=[1, 3], index_source_layer_ids=[1, 3, 4],
           candidate_source_layer_id=3, index_topk=8, candidate_topk_blocks=8, candidate_block_size=8, sliding_window=64, max_position_embeddings=8192)


@pytest.fixture(scope="module")
def model():
    m, _ = FX.build(**CFG)
    RW.enable(m, archs.arch_of({"model_type": "deepseek_v41"}))
    return m


def _inputs(m, seed=3):
    rng = np.random.default_rng(seed)
    c = m.language_model.layers[0].attn.config
    f = lambda *s, sc=1.0: mx.array((rng.standard_normal(s) * sc).astype(np.float16))
    return [(f(1, 1, c.hidden_size), (f(1, 1, c.q_lora_rank), f(1, 1, c.num_attention_heads, c.head_dim, sc=0.5), f(1, 1, c.head_dim, sc=2.0)))
            for _ in m.language_model.layers]


def _run(m, ctx, fuse, steps=4):
    DC.ATTN_CORE_FUSE = fuse
    ids = FX.toks(ctx)
    c1 = m.make_cache()
    with RW.HL.last_row():
        mx.eval(m.language_model(mx.array(ids)[None], cache=c1).logits)
    cache = RW.batch([c1])
    outs = []
    for t in range(steps):
        ins = _inputs(m, 10 + t)
        o = [DC.core(m.language_model.layers[l].attn, x, cache[0], pre=pre, stage=True) for l, (x, pre) in enumerate(ins)]
        mx.eval(o)
        outs += [np.array(x.astype(mx.float32)) for x in o]
        for r in cache[0].rows:
            r.offset += 1
        cache[0]._dirty = True
    st = DC.stack_rows(cache[0])
    arrs = []
    for name in ("win", "cmp", "kvs", "scs", "keys", "cnt", "kcnt"):
        for a in getattr(st, name):
            if a is not None and hasattr(a, "shape"):
                arrs.append(np.array(a.astype(mx.float32)) if a.dtype != mx.uint32 else np.array(a))
    return outs, arrs


@pytest.mark.parametrize("ctx", [37, 130, 131])
def test_core_and_cache_bit_equal(model, ctx):
    old = DC.ATTN_CORE_FUSE
    try:
        a_out, a_st = _run(model, ctx, False)
        b_out, b_st = _run(model, ctx, True)
    finally:
        DC.ATTN_CORE_FUSE = old
    assert len(a_st) == len(b_st) > 0
    for x, y in zip(a_out, b_out):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    for x, y in zip(a_st, b_st):
        assert np.array_equal(x, y, equal_nan=True)


def test_one_launch_less(model):
    """The scatter of the commit is gone from the layer's graph (3 kernels instead of 3 + a scatter)."""
    old = DC.ATTN_CORE_FUSE
    try:
        counts = {}
        for fuse in (False, True):
            DC.ATTN_CORE_FUSE = fuse
            c1 = model.make_cache()
            with RW.HL.last_row():
                mx.eval(model.language_model(mx.array(FX.toks(40))[None], cache=c1).logits)
            cache = RW.batch([c1])
            ins = _inputs(model)
            for _ in range(2):
                mx.eval([DC.core(model.language_model.layers[l].attn, x, cache[0], pre=pre, stage=True) for l, (x, pre) in enumerate(ins)])
            x, pre = ins[0]
            counts[fuse] = FX.census(DC.core(model.language_model.layers[0].attn, x, cache[0], pre=pre, stage=True))["dispatches"]
    finally:
        DC.ATTN_CORE_FUSE = old
    assert counts[True] == counts[False] - 1, counts
