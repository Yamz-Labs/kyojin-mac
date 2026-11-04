"""flag ttft_floor: the two pieces of large-chunk machinery a short prefill chunk does not need.

1. bound_layers: a chunk of at most ttft_floor_rows rows takes a bound every ttft_floor_bound layers instead of every 4. Same arithmetic, so the logits
   and the cache state are bit for bit the same; the bound count drops.
2. compressor_pf: a prompt chunk of `ratio`..63 rows pools its groups in one step instead of the stock per-token loop (8 launches per token per ratio-2
   layer). The pooled values and the state are the loop's, bit for bit, for every alignment and length of that band; the stock loop stays for chunks
   that close no group (fewer than `ratio` rows).

The flag is default 0: with it off every counter below reads exactly today's path.
"""
import os
import sys

import mlx.core as mx
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import archs as A, bound_layers as BL, ced as C, compressor_pf as PF, flags as FL  # noqa: E402
from test_compressor_pf import make_pair, twin_caches  # noqa: E402

ROWS_BAND = (2, 3, 5, 7, 17, 53, 63)


@pytest.fixture
def floor(monkeypatch):
    """ttft_floor on, the way flags.set('ttft_floor', 1) writes it, and off again afterwards."""
    def set_on(on=True, rows=256, every=16):
        monkeypatch.setattr(BL, "TTFT_FLOOR", on)
        monkeypatch.setattr(PF, "TTFT_FLOOR", on)
        monkeypatch.setattr(BL, "TTFT_ROWS", rows)
        monkeypatch.setattr(BL, "TTFT_BOUND_EVERY", every)
    return set_on


# ------------------------------------------------------------------------------------------------------------------ the flag itself
def test_default_is_today_and_runtime_switchable(monkeypatch):
    for name in ("ttft_floor", "ttft_floor_rows", "ttft_floor_bound"):
        assert name in FL.FLAGS and FL.FLAGS[name].default == ("0" if name == "ttft_floor" else FL.FLAGS[name].default)
    assert FL.FLAGS["ttft_floor"].default == "0" and FL.FLAGS["ttft_floor"].runtime
    assert BL.TTFT_ROWS == 256 and BL.TTFT_BOUND_EVERY == 16 and BL.EVERY == 4
    env = {k: os.environ.get(k) for k in ("KYOJIN_TTFT_FLOOR", "KYOJIN_TTFT_FLOOR_ROWS", "KYOJIN_TTFT_FLOOR_BOUND")}
    try:
        FL.set("ttft_floor", 1)
        assert BL.TTFT_FLOOR and PF.TTFT_FLOOR
        FL.set("ttft_floor_bound", 8)
        assert BL.TTFT_BOUND_EVERY == 8 and int(os.environ["KYOJIN_TTFT_FLOOR_BOUND"]) == 8
        FL.set("ttft_floor_rows", 128)
        assert BL.TTFT_ROWS == 128
        FL.set("ttft_floor", 0)
        assert not BL.TTFT_FLOOR and not PF.TTFT_FLOOR
        with pytest.raises(ValueError):
            FL.set("ttft_floor", "maybe")
    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        BL.TTFT_FLOOR, PF.TTFT_FLOOR = env["KYOJIN_TTFT_FLOOR"] == "1", env["KYOJIN_TTFT_FLOOR"] == "1"
        BL.TTFT_ROWS, BL.TTFT_BOUND_EVERY = 256, 16


# ------------------------------------------------------------------------------------------------------------------ the compressor band
@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("n", ROWS_BAND)
@pytest.mark.parametrize("start", [2000, 2001, 2002, 2003])
def test_pooled_short_chunk_equals_the_stock_loop(floor, ratio, n, start):
    if n < ratio:
        pytest.skip("a chunk that closes no group stays on the stock loop")
    L, cfg, ref, pooled = make_pair(ratio)
    c1, c2 = twin_caches(L, ref, start, 3)
    x = mx.random.normal((1, n, 64), key=mx.random.key(11)).astype(mx.bfloat16)
    a, b = ref(x, start, c1), PF._pool_call(pooled, lambda m, xx, ss, cc: type(ref).__call__(m, xx, ss, cc), x, start, c2)
    assert a is not None and b is not None and bool(mx.array_equal(a, b).item())
    assert bool(mx.array_equal(c1.kv_state[2], c2.kv_state[2]).item())
    assert bool(mx.array_equal(c1.score_state[2], c2.score_state[2]).item())


def test_band_below_ratio_stays_stock(floor):
    """A chunk shorter than the ratio cannot write a whole state: the flag must leave it to the stock loop (decode, a 1-token round)."""
    floor(True)
    L, cfg, ref, pooled = make_pair(2)
    c1, c2 = twin_caches(L, ref, 2001, 3)
    x = mx.random.normal((1, 1, 64), key=mx.random.key(5)).astype(mx.bfloat16)
    PF.HITS.update(pool=0, stock=0)
    a = PF._pool_call(pooled, lambda m, xx, ss, cc: type(ref).__call__(m, xx, ss, cc), x, 2001, c2)
    assert PF.HITS["stock"] == 1 and PF.HITS["pool"] == 0
    b = ref(x, 2001, c1)
    assert (a is None and b is None) or bool(mx.array_equal(a, b).item())
    assert bool(mx.array_equal(c1.kv_state[2], c2.kv_state[2]).item())


def test_off_keeps_the_loop_in_the_band(floor):
    floor(False)
    L, cfg, ref, pooled = make_pair(2)
    c1, c2 = twin_caches(L, ref, 2000, 3)
    x = mx.random.normal((1, 53, 64), key=mx.random.key(7)).astype(mx.bfloat16)
    PF.HITS.update(pool=0, stock=0)
    PF._pool_call(pooled, lambda m, xx, ss, cc: type(ref).__call__(m, xx, ss, cc), x, 2000, c2)
    assert PF.HITS["stock"] == 1 and PF.HITS["pool"] == 0


def test_counted_ops_drop_in_the_band(floor):
    """The stock loop rewrites each token's slot with concatenates (two per token, per state); the pooled call uses a handful. Count them, plus the
    Metal kernel launches, with the flag off and on."""
    L, cfg, ref, pooled = make_pair(2)
    calls = {"cat": 0, "kern": 0}
    cat, mk = mx.concatenate, mx.fast.metal_kernel

    def counting_cat(*a, **k):
        calls["cat"] += 1
        return cat(*a, **k)

    def counting_mk(*a, **k):
        fn = mk(*a, **k)

        def call(*aa, **kk):
            calls["kern"] += 1
            return fn(*aa, **kk)
        return call

    mx.concatenate, mx.fast.metal_kernel = counting_cat, counting_mk
    try:
        out = {}
        for on in (False, True):
            floor(on)
            c1, c2 = twin_caches(L, ref, 2000, 3)
            x = mx.random.normal((1, 53, 64), key=mx.random.key(9)).astype(mx.bfloat16)
            calls["cat"] = calls["kern"] = 0
            PF.HITS.update(pool=0, stock=0)
            o = PF._pool_call(pooled, lambda m, xx, ss, cc: type(ref).__call__(m, xx, ss, cc), x, 2000, c2)
            mx.eval(o, c2.kv_state[2], c2.score_state[2])
            out[on] = (calls["cat"], calls["kern"], PF.HITS["pool"])
    finally:
        mx.concatenate, mx.fast.metal_kernel = cat, mk
    assert out[False][2] == 0 and out[True][2] == 1                       # the path moved
    assert out[False][0] >= 2 * 53                                         # the loop: the slot rewrites of 53 tokens
    assert out[True][0] <= 8                                               # the pooled call: the open group and its state
    assert out[True][1] <= out[False][1]


# ------------------------------------------------------------------------------------------------------------------ the layer bound
def _chunk_logits(lm, cache, ids, s, n):
    lg = lm(mx.array([ids[s:s + n]]), cache=cache).logits
    mx.eval(lg, *cache[0].state)
    return lg


@pytest.fixture
def bounded(floor):
    """Two models of the same weights with the layer bound installed (no CED: every layer of a chunk runs its full rows, so the bound cadence is
    visible in one number for the whole chunk)."""
    out = []
    for _ in range(2):
        lm, cfg = M.make()
        BL.install(lm)
        out.append(lm)
    try:
        yield out
    finally:
        BL.uninstall()


def test_bound_stride_and_bit_identical_chunks(bounded, floor):
    """One prompt: 2000 rows, then a 3-row and a 200-row chunk (a tool round), flag off and on."""
    ids = M.tokens(2203)
    lm0, lm1 = bounded
    c0, c1 = lm0.make_cache(), lm1.make_cache()
    assert bool(mx.array_equal(_chunk_logits(lm0, c0, ids, 0, 2000), _chunk_logits(lm1, c1, ids, 0, 2000)).item())
    BL.reset()
    floor(True)
    lg0 = _chunk_logits(lm0, c0, ids, 2000, 3)
    b0 = BL.stats()["bounds"]
    BL.reset()
    lg1 = _chunk_logits(lm1, c1, ids, 2000, 3)
    b1 = BL.stats()["bounds"]
    assert b0 == b1 == 0                                                   # 8 layers, and a 3-row chunk is under MIN_ROWS anyway
    assert bool(mx.array_equal(lg0, lg1).item())
    floor(False)                                                           # the default: every 4 layers
    BL.reset()
    lg0 = _chunk_logits(lm0, c0, ids, 2003, 200)
    off = BL.stats()["bounds"]
    BL.reset()
    floor(True)                                                            # every 16 layers: fewer evals, the same bits
    lg1 = _chunk_logits(lm1, c1, ids, 2003, 200)
    on = BL.stats()["bounds"]
    assert off == 2 and on == 0, (off, on)
    assert bool(mx.array_equal(lg0, lg1).item())
    for slot in ("window", "compress", "keys", "kv_state", "score_state"):
        a, b = getattr(c0[0], slot), getattr(c1[0], slot)
        for x, y in zip(a, b):
            if x is not None and y is not None:
                assert bool(mx.array_equal(x, y).item()), slot


def test_bound_stride_leaves_a_long_chunk_alone(bounded, floor):
    """A chunk above ttft_floor_rows keeps the large-chunk stride (the transients of a 8192-row chunk are why the bound exists)."""
    ids = M.tokens(1400)
    lm, _ = bounded
    cache = lm.make_cache()
    _chunk_logits(lm, cache, ids, 0, 200)
    floor(True, rows=256, every=16)
    BL.reset()
    _chunk_logits(lm, cache, ids, 200, 1000)                               # 1000 rows: over the flag's band
    assert BL.stats()["bounds"] == 2                                        # 8 layers, every 4 -> 2, as today
