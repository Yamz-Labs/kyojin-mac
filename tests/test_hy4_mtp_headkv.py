# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_MTP_HEAD_KV (default off): the MTP draft head's KV is a fixed_kv.FixedKVCache (fixed-state buffer, device offset) laid out like the trunk's layer of the same kind, so the head
attention can take the trunk's kept-key gather path. CPU only, no Metal (tiny synthetic Hy4 model of test_hy4_mtp.py; run with nice -n 19, one file at a time).

  flags      default off; the head cache type
  logits     head output on the fixed cache == on the concatenating KVCache over a prompt chunk, single rows, rollbacks, a verify-like block, many buffer growths: drafted ids, head KV rows, pending pair
  rounds     the reference round driver: tokens and drafts per round equal for stock / HEAD_KV
  snapshot   Hy4MtpBackend._copy_draft (prefix snapshot / restore) forks a fixed head cache: independent, copy on write, the draft after a restore equals the original's
  counters   mtp_head_fixed_kv fires as the flag says; mtp_head_gather fires on the kept-key gather path; both reach the `[flags 60s]` totals
  noleak     the head block is not a step-compile layer and a finished session's head KV is not kept alive by any module state"""
import gc
import math
import os
import sys
import weakref

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kyojin_mlx import fired as FD  # noqa: E402
from kyojin_mlx import hy4_mtp as M  # noqa: E402
from kyojin_mlx.foreign_skeleton_mla import MlaAttention as MLA  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402
from test_hy4_mtp import _PATH, V, _fresh, _prompts, built  # noqa: E402,F401  (the synthetic Hy4 model + head of the MTP tests)

FLAGS = ("KYOJIN_HY4_MTP_HEAD_KV", "KYOJIN_HY4_FIXED_PKCACHE", "KYOJIN_HY4_FIXED_GATHER", "KYOJIN_HY4_FIXED_GATHER_DEC")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in FLAGS:
        monkeypatch.delenv(k, raising=False)
    orig = M.Mtp._fixed_cache

    def small(self):
        c = orig(self)
        c.step = 8 * c.align // math.gcd(8, c.align)             # a head buffer grows every 8 rows: many growths inside the scripts
        return c
    monkeypatch.setattr(M.Mtp, "_fixed_cache", small)


def _arm(monkeypatch, h, pk="0"):
    monkeypatch.setenv("KYOJIN_HY4_MTP_HEAD_KV", h)
    monkeypatch.setenv("KYOJIN_HY4_FIXED_PKCACHE", pk)


def _kv(st):
    c = st.cache
    if getattr(c, "fixed_kv", False):
        return np.array(c.cache[0][:, :, :c.offset, :])
    return np.array(c.keys[..., :c.offset, :])


def _primed(model, mtp, ids):
    cache = _fresh(model)
    st = mtp.new_state()
    lg, h = mtp.forward(cache, ids)
    mtp.feed(st, ids, h)
    mtp.absorb(st)
    return cache, st, int(mx.argmax(lg[-1]).item())


def _script(D, seed=5):
    """(rows, rollback before the call): a prompt in two chunks, single rows, scratch chains trimmed away, a verify-like block, rollbacks inside a pooling group."""
    rng = np.random.default_rng(seed)
    plan = [(9, 0), (7, 0), (1, 0), (1, 0), (1, 0), (1, 3), (1, 0), (5, 0), (1, 2), (1, 0), (4, 0), (1, 1), (1, 0), (1, 0), (6, 3), (1, 0), (12, 0), (1, 0), (1, 5), (3, 0)]
    return [(mx.array(rng.integers(0, V, n), dtype=mx.int32)[None], mx.array(rng.standard_normal((1, n, D)).astype(np.float32)), tr) for n, tr in plan]


def _gather_env(monkeypatch):
    """The lane's gather trio: kpool gather from 1 slot here (the tiny head has few keys; the lane uses 16384 / 8192) + the pooled-key cache."""
    monkeypatch.setenv("KYOJIN_HY4_FIXED_GATHER", "1")
    monkeypatch.setenv("KYOJIN_HY4_FIXED_GATHER_DEC", "1")
    monkeypatch.setenv("KYOJIN_HY4_FIXED_PKCACHE", "1")


def test_flags_default_off(monkeypatch):
    assert not M.head_kv_fixed()
    monkeypatch.setenv("KYOJIN_HY4_MTP_HEAD_KV", "0")
    assert not M.head_kv_fixed()
    monkeypatch.setenv("KYOJIN_HY4_MTP_HEAD_KV", "1")
    assert M.head_kv_fixed()


@pytest.mark.parametrize("h,fixed", [("0", False), ("1", True)])
def test_head_cache_type(built, monkeypatch, h, fixed):
    _, path, model, head, mtp = built
    _arm(monkeypatch, h)
    st = mtp.new_state()
    assert getattr(st.cache, "fixed_kv", False) is fixed
    if not fixed:
        assert isinstance(st.cache, KVCache)


@pytest.mark.parametrize("pk", ["0", "1"])
def test_head_logits_fixed_cache_equal_concat(built, monkeypatch, pk):
    """The head block on the fixed cache == on the concatenating KVCache, step by step, across rollbacks and growths."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    monkeypatch.setenv("KYOJIN_HY4_FIXED_PKCACHE", pk)
    script = _script(mtp.dim)
    ref_c, new_c = KVCache(), mtp._fixed_cache()
    assert getattr(new_c, "fixed_kv", False)
    n0 = MLA.PATHS.get("mtp_head_fixed_kv", 0)
    for i, (toks, h, tr) in enumerate(script):
        if tr:
            ref_c.trim(tr)
            new_c.trim(tr)
        a = head(model, toks, h, ref_c)
        b = head(model, toks, h, new_c)
        mx.eval(a, b)
        assert np.allclose(np.array(a), np.array(b), rtol=1e-4, atol=1e-5), f"step {i}: max abs {float(mx.max(mx.abs(a - b)).item()):.3g}"
        assert ref_c.offset == new_c.offset
    assert MLA.PATHS.get("mtp_head_fixed_kv", 0) - n0 == len(script)              # fires: one per head call on the fixed cache, none for the KVCache calls
    assert new_c.cache[0].shape[2] > 8 * 2                                       # the buffer grew several times


@pytest.mark.parametrize("depth", [1, 3, 5])
@pytest.mark.parametrize("pi", [0, 1])
@pytest.mark.parametrize("pk", ["0", "1"])
def test_draft_ids_and_head_state_equal(built, monkeypatch, depth, pi, pk):
    """stock vs HEAD_KV: drafted ids, head offset and head KV rows (the pair of `pending` kept, chain rows trimmed). The HEAD_KV arm runs twice, the stock path again last."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    ids = [int(t) for t in _prompts()[pi]]
    runs = []
    for h in ("0", "1", "1", "0"):
        _arm(monkeypatch, h, pk)
        _, st, pending = _primed(model, mtp, ids)
        assert getattr(st.cache, "fixed_kv", False) == (h == "1")
        d = mtp.draft(st, pending, depth)
        runs.append((d, st.cache.offset, _kv(st)))
    for r in runs[1:]:
        assert r[0] == runs[0][0] and len(r[0]) == depth
        assert r[1] == runs[0][1]
        assert np.allclose(r[2], runs[0][2], rtol=1e-4, atol=1e-5)


def test_counters_fire_as_the_flag_says(built, monkeypatch):
    _, path, model, head, mtp = built
    _PATH[0] = path
    ids = [int(t) for t in _prompts()[0]]
    for h in ("0", "1"):
        _arm(monkeypatch, h)
        b = MLA.PATHS.get("mtp_head_fixed_kv", 0)
        _, st, pending = _primed(model, mtp, ids)
        mtp.draft(st, pending, 3)
        assert (MLA.PATHS.get("mtp_head_fixed_kv", 0) > b) == (h == "1"), h


def test_rounds_and_tokens_equal(built, monkeypatch):
    """The reference round driver (draft, verify, accept, commit) over 40 tokens: drafts per round and generated ids equal for stock / HEAD_KV."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    ids = _prompts()[0]
    out = {}
    for h in ("0", "1"):
        _arm(monkeypatch, h, "1")
        seen, orig = [], mtp.draft

        def spy(st, pending, depth, _o=orig, _s=seen):
            r = _o(st, pending, depth)
            _s.append(r)
            return r
        mtp.draft = spy
        try:
            cache = _fresh(model)
            st = mtp.new_state()
            toks = M.generate(mtp, cache, st, ids, 40, 3, exact=False)
        finally:
            mtp.draft = orig
        out[h] = (toks, seen)
    assert out["1"][0] == out["0"][0]
    assert out["1"][1] == out["0"][1]


def test_prefix_snapshot_restore_forks_the_head_cache(built, monkeypatch):
    """Hy4MtpBackend._copy_draft (prefix snapshot / pf_share seed / draft restore) on a fixed head cache: independent copy, the draft after the restore equals the original's."""
    from kyojin_mlx.serve.hy4mtp import Hy4MtpBackend
    _, path, model, head, mtp = built
    _PATH[0] = path
    _arm(monkeypatch, "1")
    ids = [int(t) for t in _prompts()[1]]
    _, st, pending = _primed(model, mtp, ids)
    snap = Hy4MtpBackend._copy_draft(st)
    assert getattr(snap.cache, "fixed_kv", False) and snap.cache is not st.cache
    assert snap.cache.offset == st.cache.offset and snap.toks == st.toks
    kv0 = _kv(snap).copy()
    d_orig = mtp.draft(st, pending, 3)                                           # the original continues (writes rows past the snapshot, trims them back)
    again = Hy4MtpBackend._copy_draft(snap)
    d_snap = mtp.draft(again, pending, 3)
    assert d_snap == d_orig
    assert np.allclose(_kv(snap), kv0)                                           # the snapshot itself never moved
    assert snap.cache.offset == len(ids)


def test_copy_draft_stock_cache_unchanged(built, monkeypatch):
    from kyojin_mlx.serve.hy4mtp import Hy4MtpBackend
    _, path, model, head, mtp = built
    _PATH[0] = path
    _arm(monkeypatch, "0")
    _, st, pending = _primed(model, mtp, [int(t) for t in _prompts()[0]])
    snap = Hy4MtpBackend._copy_draft(st)
    assert isinstance(snap.cache, KVCache) and snap.cache.offset == st.cache.offset


def test_fires_totals_carry_the_head_counters(monkeypatch):
    monkeypatch.setitem(MLA.PATHS, "mtp_head_fixed_kv", 11)
    monkeypatch.setitem(MLA.PATHS, "mtp_head_gather", 4)
    t = FD.totals()
    assert t["mtp_head_fixed_kv"] == 11 and t["mtp_head_gather"] == 4
    assert "mtp_head_fixed_kv=11 mtp_head_gather=4" in FD.format_line(60, "x 1/0", t)


def test_head_block_is_not_a_step_compile_layer_and_no_kv_is_pinned(built, monkeypatch):
    """The head block keeps its own class (no step-compile program, so no program template can hold the first request's KV); a finished session's head cache is freed."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    _arm(monkeypatch, "1", "1")
    assert not getattr(head.block.__class__, "_step_compiled", False)
    ids = [int(t) for t in _prompts()[0]]
    refs = []
    for _ in range(3):
        _, st, pending = _primed(model, mtp, ids)
        mtp.draft(st, pending, 3)
        refs.append(weakref.ref(st.cache))
        del st
    gc.collect()
    assert all(r() is None for r in refs)


def test_head_logits_on_the_gather_path_equal_concat(built, monkeypatch):
    """Head on the fixed cache with the gather attend (kept keys only) == head on the concatenating KVCache (masked attend), across rollbacks and growths; mtp_head_gather fires."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    _gather_env(monkeypatch)
    ref_c, new_c = KVCache(), mtp._fixed_cache()
    g0 = MLA.PATHS.get("mtp_head_gather", 0)
    for i, (toks, h, tr) in enumerate(_script(mtp.dim)):
        if tr:
            ref_c.trim(tr)
            new_c.trim(tr)
        a = head(model, toks, h, ref_c)
        b = head(model, toks, h, new_c)
        mx.eval(a, b)
        assert np.allclose(np.array(a), np.array(b), rtol=1e-4, atol=1e-5), f"step {i}: max abs {float(mx.max(mx.abs(a - b)).item()):.3g}"
    assert MLA.PATHS.get("mtp_head_gather", 0) > g0, "the gather attend never ran inside the head block"


@pytest.mark.parametrize("pi", [0, 1])
def test_draft_ids_equal_on_the_gather_path(built, monkeypatch, pi):
    """Lane flags (gather + PKCACHE): drafted ids and head KV of stock / HEAD_KV equal; the gather counter fires for the HEAD_KV arm only."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    _gather_env(monkeypatch)
    ids = [int(t) for t in _prompts()[pi]]
    runs = []
    for h in ("0", "1", "1", "0"):
        monkeypatch.setenv("KYOJIN_HY4_MTP_HEAD_KV", h)
        g0 = MLA.PATHS.get("mtp_head_gather", 0)
        _, st, pending = _primed(model, mtp, ids)
        d = mtp.draft(st, pending, 3)
        assert (MLA.PATHS.get("mtp_head_gather", 0) > g0) == (h == "1"), h
        runs.append((d, st.cache.offset, _kv(st)))
    for r in runs[1:]:
        assert r[0] == runs[0][0] and r[1] == runs[0][1]
        assert np.allclose(r[2], runs[0][2], rtol=1e-4, atol=1e-5)
