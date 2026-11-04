"""Compiled chained verify (fuse_layer._chain_block) and the pieces it rests on: the B-row mHC mix kernel equals B single-row launches bit for bit; the compiled chain gives
the logits of the eager chain to the bit, over many rounds at advancing positions, with no retrace after the warm-up and no fallback; the rope/position tables of a chain are
cached per position. Real-slice cases need KYOJIN_SHRUNK=<dir with ds2/, optionally ds2e/> and Metal."""
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
SHRUNK = os.environ.get("KYOJIN_SHRUNK")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_mix_pre_rows_equal_single_row_launches(dtype):
    from kyojin_mlx import fuse as F
    rng = np.random.default_rng(11)
    B, D = 5, 1024
    mixes = mx.array(rng.standard_normal((B, 1, 24)).astype(np.float32))
    scale = mx.array(rng.standard_normal(3).astype(np.float32) * 0.3)
    base = mx.array(rng.standard_normal(24).astype(np.float32) * 0.3)
    x = mx.array(rng.standard_normal((B, 1, 4, D)).astype(np.float32)).astype(dtype)
    pre = mx.array(rng.random((B, 1, 4)).astype(np.float32))
    whole = F.mhc_mix_pre(mixes, scale, base, x, pre, 20, 1e-6)
    for b in range(B):
        one = F.mhc_mix_pre(mixes[b:b + 1], scale, base, x[b:b + 1], pre[b:b + 1], 20, 1e-6)
        for w, o in zip(whole, one):
            assert np.array_equal(np.array(w[b:b + 1].astype(mx.float32)), np.array(o.astype(mx.float32)))


def test_rope_and_position_tables_are_cached_per_position():
    from kyojin_mlx import fuse_attn as FA
    a = FA.pos_array(7)
    assert FA.pos_array(9) is not a and FA.pos_array(7) is a                       # a second position does not evict the first
    for p in range(FA._KEEP * 2):
        FA.pos_array(1000 + p)
    assert len(FA._pos) <= FA._KEEP


def load(name):
    from kyojin_mlx import patch as P, rows as R, spec as S
    pack = os.path.join(SHRUNK or "", name)
    if not os.path.isdir(pack):
        pytest.skip(f"{pack} missing or KYOJIN_SHRUNK not set")
    model, spec, cfg = P.build_model(pack)
    P.patch_model(model, pack, spec, cfg)
    R.enable(model, spec)
    S.install(model, spec)
    return model, cfg


@pytest.mark.parametrize("name", ["ds2", "ds2e"])
def test_compiled_chain_bit_identical_flat_and_no_fallback(name, monkeypatch):
    from kyojin_mlx import fuse_attn as FA, fuse_layer as FL, rows as R, spec as S
    monkeypatch.setattr(FA, "MMA", False)                                          # tolerance levers (fp16 multi-query MMA attention, split P@V): the chain and the eager
    monkeypatch.setattr(FA, "SPLIT_PV", False)                                     # reference group the rows differently, so the bits differ by design (day1_gates bitwise mode)
    model, cfg = load(name)
    V = cfg["text_config"]["vocab_size"]
    rng = np.random.default_rng(5)
    cache, lg = R.prefill(model, rng.integers(16, V - 16, 150).tolist())
    tok = int(mx.argmax(lg[0]).item())
    snap, fb_on = None, 0
    for r in range(48):                                                            # depths 0..4 drafts, accepted count cycling: positions advance every round
        n = 1 + r % 5
        toks = [tok] + rng.integers(16, V - 16, n - 1).tolist()
        runs = {}
        for chain_on in (True, False):
            FL.CHAIN = chain_on
            fb = FL.STATS["fallback"]
            try:
                out, _, ch = S.verify(model, cache, toks)
                mx.eval(out)
            finally:
                FL.CHAIN = True
            fb_on += (FL.STATS["fallback"] - fb) * chain_on                         # the eager reference runs fall back by design
            runs[chain_on] = (np.array(out), ch)
        assert np.array_equal(runs[True][0], runs[False][0]), (r, n)
        acc = r % n
        cache = S.commit(runs[True][1], acc)
        tok = int(np.argmax(runs[True][0][acc]))
        if r == 14:                                                                # every depth was seen with the chain on: all regions traced
            snap = dict(FL.STATS)
    assert fb_on == 0, fb_on
    assert snap is not None and FL.STATS["trace_chain"] == snap["trace_chain"], (snap, FL.STATS)
    assert FL.STATS["call_chain"] > snap["call_chain"] > 0
