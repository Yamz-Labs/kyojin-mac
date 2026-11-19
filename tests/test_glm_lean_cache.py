"""BP-CACHE: `kyojin_mlx.glm_lean_cache` (flag `KYOJIN_GLM_LEAN_CACHE`) against the stock
`mlx_vlm.models.cache.BatchPoolingCache.update_and_fetch` (the one concretely proven, unconditional
per-session Python loop BP-IDX found outside G10c's `accumulate_fast` / `pool_compress_b`): the visible
cache state (`pooled[i, :pool_lengths[i]]` + `_pool_lengths`) must be bit-identical to stock after any
sequence of calls, including speculative start / commit / abort / partial accept, B=1 and B=4, and a pool
boundary row (remainder 1 of a kpool=4 ratio)."""
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_vlm.models.cache import BatchPoolingCache as Stock    # noqa: E402

os.environ.setdefault("KYOJIN_GLM_LEAN_CACHE", "1")
from kyojin_mlx import glm_lean_cache as LC    # noqa: E402

LC.install()


def _lean_cache(ratio, left_padding):
    """A BatchPoolingCache whose `update_and_fetch` is the lean path (flag forced on via LC.ENABLED)."""
    c = Stock(ratio, left_padding)
    LC.ENABLED = True
    return c


def _stock_cache(ratio, left_padding):
    c = Stock(ratio, left_padding)
    return c


def _run_sequence(cache, pushes, D=8, dtype=mx.float16, seed=0):
    """pushes: list of (B, n_new_cols) -> feed `accumulate_windows` + `_compress_pools` stock logic to
    produce a realistic `px` for `update_and_fetch`, then call it. Returns the list of returned `pooled`
    arrays (post-call) for inspection."""
    mx.random.seed(seed)
    outs = []
    for kv_cols, gate_cols in pushes:
        kv = mx.random.normal((cache.batch_size, kv_cols, D)).astype(dtype)
        gate = mx.random.normal((cache.batch_size, gate_cols, D)).astype(dtype)
        ready_k, ready_gate, _ = cache.accumulate_windows(kv, gate, 0)
        ape = mx.zeros((cache.ratio, D), dtype=mx.float32)
        if ready_k.shape[1] == 0:
            new_pool = mx.zeros((cache.batch_size, 0, D), dtype=dtype)
        else:
            # same arithmetic as Glm5NextIndexer._compress_pools, done generically for N windows
            kp = int(cache.ratio)
            nwin = ready_k.shape[1] // kp
            rk = mx.unflatten(ready_k, 1, (nwin, kp))
            rg = mx.unflatten(ready_gate, 1, (nwin, kp))
            logits = rg.astype(mx.float32) + ape[None, None]
            probs = mx.softmax(logits, axis=2, precise=True).astype(dtype)
            new_pool = (probs * rk).sum(axis=2)
        out = cache.update_and_fetch(new_pool)
        mx.eval(out)
        outs.append(out)
    return outs


def _visible_equal(stock_c, lean_c, outs_stock, outs_lean):
    assert stock_c._pool_lengths == lean_c._pool_lengths, (stock_c._pool_lengths, lean_c._pool_lengths)
    for a, b in zip(outs_stock, outs_lean):
        if a is None and b is None:
            continue
        for i, pl in enumerate(stock_c._pool_lengths):
            if pl == 0:
                continue
            av = a[i, :pl]
            bv = b[i, :pl]
            assert mx.array_equal(av, bv), (i, pl)


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("ratio", [4])
def test_pool_insert_matches_stock_plain(B, ratio):
    left_padding = [0] * B
    stock_c = _stock_cache(ratio, left_padding)
    lean_c = _lean_cache(ratio, left_padding)
    LC.ENABLED = False
    try:
        pushes = [(2, 2)] * 10 + [(5, 5)] * 3 + [(1, 1)] * 6
        outs_stock = _run_sequence(stock_c, pushes, seed=1)
    finally:
        pass
    LC.ENABLED = True
    outs_lean = _run_sequence(lean_c, pushes, seed=1)
    LC.ENABLED = False
    _visible_equal(stock_c, lean_c, outs_stock, outs_lean)


def test_pool_insert_matches_stock_b4_ragged():
    """Sessions complete windows at genuinely different times (B=4, different left_padding so each
    session's `valid_lengths` -- and so `new_counts` -- diverges per call even though every push feeds
    the whole batch the same column count): the exact scenario where stock's per-row loop and some rows
    getting new_counts=0 while others get 1+ in the SAME call matters."""
    B, ratio = 4, 4
    left_padding = [0, 1, 2, 3]
    stock_c = _stock_cache(ratio, left_padding)
    lean_c = _lean_cache(ratio, left_padding)

    def push(cache, seed):
        mx.random.seed(seed)
        outs = []
        for cols in (1, 1, 1, 1, 2, 1, 3, 1, 1):
            kv = mx.random.normal((B, cols, 8)).astype(mx.float16)
            gate = mx.random.normal((B, cols, 8)).astype(mx.float16)
            ready_k, ready_gate, _ = cache.accumulate_windows(kv, gate, 0)
            if ready_k.shape[1] == 0:
                new_pool = mx.zeros((B, 0, 8), dtype=mx.float16)
            else:
                kp = cache.ratio
                nwin = ready_k.shape[1] // kp
                rk = mx.unflatten(ready_k, 1, (nwin, kp))
                rg = mx.unflatten(ready_gate, 1, (nwin, kp))
                probs = mx.softmax(rg.astype(mx.float32), axis=2, precise=True).astype(mx.float16)
                new_pool = (probs * rk).sum(axis=2)
            out = cache.update_and_fetch(new_pool)
            mx.eval(out)
            outs.append(out)
        return outs

    LC.ENABLED = False
    outs_stock = push(stock_c, 7)
    LC.ENABLED = True
    outs_lean = push(lean_c, 7)
    LC.ENABLED = False
    _visible_equal(stock_c, lean_c, outs_stock, outs_lean)


def test_pool_insert_boundary_row_remainder_one():
    """Pool boundary row: kpool=4, push exactly kpool*N + 1 tokens (remainder 1) across B=4, then a few
    more one-token pushes to cross several more boundaries (the case the brief calls out by name)."""
    B, ratio = 4, 4
    left_padding = [0] * B
    stock_c = _stock_cache(ratio, left_padding)
    lean_c = _lean_cache(ratio, left_padding)

    def push(cache, seed):
        mx.random.seed(seed)
        outs = []
        total = ratio * 2040 + 1
        chunks = [ratio] * (total // ratio) + ([total % ratio] if total % ratio else [])
        for cols in chunks[:12] + [1] * 5:
            kv = mx.random.normal((B, cols, 8)).astype(mx.bfloat16)
            gate = mx.random.normal((B, cols, 8)).astype(mx.bfloat16)
            ready_k, ready_gate, _ = cache.accumulate_windows(kv, gate, 0)
            if ready_k.shape[1] == 0:
                new_pool = mx.zeros((B, 0, 8), dtype=mx.bfloat16)
            else:
                kp = cache.ratio
                nwin = ready_k.shape[1] // kp
                rk = mx.unflatten(ready_k, 1, (nwin, kp))
                rg = mx.unflatten(ready_gate, 1, (nwin, kp))
                probs = mx.softmax(rg.astype(mx.float32), axis=2, precise=True).astype(mx.bfloat16)
                new_pool = (probs * rk).sum(axis=2)
            out = cache.update_and_fetch(new_pool)
            mx.eval(out)
            outs.append(out)
        return outs

    LC.ENABLED = False
    outs_stock = push(stock_c, 11)
    LC.ENABLED = True
    outs_lean = push(lean_c, 11)
    LC.ENABLED = False
    _visible_equal(stock_c, lean_c, outs_stock, outs_lean)


@pytest.mark.parametrize("B", [1, 4])
def test_pool_insert_speculative_commit_and_abort(B):
    """start_speculation -> commit (full accept) and -> abort (reject) must leave `pooled` / `_pool_lengths`
    identical to stock in both outcomes, lean path on."""
    ratio = 4
    left_padding = [0] * B

    def run(flag_on, commit):
        LC.ENABLED = flag_on
        c = Stock(ratio, left_padding)
        mx.random.seed(21)
        for cols in (3, 2, 4, 1):
            kv = mx.random.normal((B, cols, 8)).astype(mx.float16)
            gate = mx.random.normal((B, cols, 8)).astype(mx.float16)
            ready_k, ready_gate, _ = c.accumulate_windows(kv, gate, 0)
            new_pool = (mx.zeros((B, 0, 8), mx.float16) if ready_k.shape[1] == 0
                        else ready_k.reshape(B, -1, c.ratio, 8).mean(axis=2).astype(mx.float16))
            mx.eval(c.update_and_fetch(new_pool))
        c.prepare(lengths=[1000] * B)
        gen = c.start_speculation(4)
        for cols in (1, 1, 1, 1):
            kv = mx.random.normal((B, cols, 8)).astype(mx.float16)
            gate = mx.random.normal((B, cols, 8)).astype(mx.float16)
            ready_k, ready_gate, _ = c.accumulate_windows(kv, gate, 0)
            new_pool = (mx.zeros((B, 0, 8), mx.float16) if ready_k.shape[1] == 0
                        else ready_k.reshape(B, -1, c.ratio, 8).mean(axis=2).astype(mx.float16))
            mx.eval(c.update_and_fetch(new_pool))
        if commit:
            c.commit_speculation([4] * B, gen)
        else:
            c.abort_speculation(gen)
        return c

    c_stock = run(False, True)
    c_lean = run(True, True)
    LC.ENABLED = False
    assert c_stock._pool_lengths == c_lean._pool_lengths
    for i, pl in enumerate(c_stock._pool_lengths):
        if pl:
            assert mx.array_equal(c_stock.pooled[i, :pl], c_lean.pooled[i, :pl])

    c_stock = run(False, False)
    c_lean = run(True, False)
    LC.ENABLED = False
    assert c_stock._pool_lengths == c_lean._pool_lengths
    for i, pl in enumerate(c_stock._pool_lengths):
        if pl:
            assert mx.array_equal(c_stock.pooled[i, :pl], c_lean.pooled[i, :pl])


def test_flag_off_is_the_exact_stock_method():
    """KYOJIN_GLM_LEAN_CACHE=0 (or LC.ENABLED False at call time) must dispatch to the unmodified stock
    bound method, not a re-implementation that happens to agree on these tests."""
    LC.ENABLED = False
    c = Stock(4, [0, 0])
    mx.random.seed(3)
    kv = mx.random.normal((2, 4, 8)).astype(mx.float16)
    gate = mx.random.normal((2, 4, 8)).astype(mx.float16)
    ready_k, ready_gate, _ = c.accumulate_windows(kv, gate, 0)
    new_pool = ready_k.reshape(2, 1, 4, 8).mean(axis=2).astype(mx.float16)
    before = LC.STATS["pool_insert_stock"]
    mx.eval(c.update_and_fetch(new_pool))
    assert LC.STATS["pool_insert_stock"] == before + 1
