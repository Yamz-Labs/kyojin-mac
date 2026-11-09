"""Measurement probes (kyojin_mlx.taps): spec_ptap, router_tap, prefix miss reasons. They only read values: tokens, routing and cache hits are the same on or off."""
import json
import time
import types

import mlx.core as mx
import numpy as np

from kyojin_mlx import rows as R
from kyojin_mlx import specsam, taps
from kyojin_mlx.kvdisk import prefix_digests
from kyojin_mlx.serve.engine import PrefixCache
from kyojin_mlx.serve.sampling import Sampling
from test_serve_spec import FakeSpec, run


# ------------------------------------------------------------------------------------------------------------------ spec_ptap
def test_ptap_values_match_the_sampling_distribution():
    rng = np.random.default_rng(0)
    lg = mx.array(rng.normal(size=(4, 50)).astype(np.float32) * 2)
    smp = Sampling(0.7, 0.9, 20, 0.0, 1)
    res = types.SimpleNamespace(starts=[0], logits=lg)
    drafts = [3, 7, 11]
    out = taps.ptap_collect([types.SimpleNamespace(sampling=smp)], [[1] + drafts], res, {})
    p, h = out[0]
    assert len(p) == len(h) == 3
    for j, d in enumerate(drafts):
        lp = np.array(specsam.logp(lg[j], smp), dtype=np.float64)
        pr = np.exp(lp)
        assert abs(pr.sum() - 1) < 1e-5
        assert abs(p[j] - pr[d]) < 1e-6
        assert abs(h[j] - (-(pr[pr > 0] * lp[pr > 0]).sum())) < 1e-5


def test_ptap_skips_greedy_sessions():
    res = types.SimpleNamespace(starts=[0], logits=mx.zeros((3, 8)))
    assert taps.ptap_collect([types.SimpleNamespace(sampling=Sampling())], [[1, 2, 3]], res, {}) == {}


def _reqs():
    return {"a": ([1, 2, 3, 4, 5], dict(max_new=40, sampling=Sampling(1.0, 0.95, 0, 0.0, 7)))}


def test_ptap_log_and_tokens_unchanged(monkeypatch, tmp_path):
    outp = tmp_path / "p.jsonl"
    monkeypatch.setenv("KYOJIN_SPEC_PTAP", "0")
    base, _, _ = run(_reqs(), FakeSpec(err=3), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert not outp.exists()
    monkeypatch.setenv("KYOJIN_SPEC_PTAP", "1")
    monkeypatch.setenv("KYOJIN_SPEC_PTAP_OUT", str(outp))
    got, st, _ = run(_reqs(), FakeSpec(err=3), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert got["a"][0] == base["a"][0] and got["a"][1] == base["a"][1]               # bit-identical tokens
    lines = [json.loads(x) for x in outp.read_text().splitlines()]
    assert lines and len(lines) <= st["spec_rounds"]
    assert sum(ln["accepted"] for ln in lines) == st["spec_accepted"]
    for ln in lines:
        assert ln["src"] in ("mtp", "lookup") and ln["temperature"] == 1.0 and ln["top_p"] == 0.95
        assert len(ln["p_draft"]) == len(ln["entropy"]) == len(ln["match"]) == ln["n"]
        assert all(0.0 <= x <= 1.0 + 1e-6 for x in ln["p_draft"]) and all(e >= 0 for e in ln["entropy"])
        lead = 0
        while lead < ln["n"] and ln["match"][lead]:
            lead += 1
        assert ln["accepted"] <= lead


def test_ptap_off_without_path(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_PTAP", "1")
    monkeypatch.delenv("KYOJIN_SPEC_PTAP_OUT", raising=False)
    assert not taps.ptap_on()


# ------------------------------------------------------------------------------------------------------------------ router_tap
def test_router_tap_aggregates_shares():
    taps.router_reset()
    w = np.array([[0.1, 0.6, 0.3], [0.5, 0.45, 0.05], [0.2, 0.2, 0.6]], dtype=np.float32)
    wt = mx.array(w)
    taps.router_tap(object(), wt[:, None])             # [rows, 1, k] as the decode router returns
    taps.router_flush(force=True)
    assert np.array_equal(np.array(wt), w)             # input untouched
    snap = taps.router_snapshot()["layers"]["0"]
    sh = -np.sort(-w, axis=1)
    sh = sh / sh.sum(1, keepdims=True)
    assert snap["rows"] == 3
    assert np.allclose(snap["sum"], sh.sum(0), atol=1e-5) and np.allclose(snap["sumsq"], (sh ** 2).sum(0), atol=1e-5)
    for t, name in zip(taps.THRESHOLDS, ("lt02", "lt05", "lt10", "lt15")):
        assert snap[name] == [float(x) for x in (sh < t).sum(0)]
    taps.router_reset()


def _router_module():
    rng = np.random.default_rng(3)
    return types.SimpleNamespace(weight=mx.array(rng.normal(size=(64, 32)).astype(np.float32)), bias=mx.zeros((64,)), top_k=6, norm_topk_prob=True,
                                 routed_scaling_factor=1.5)


def test_router_site_output_is_identical_with_the_tap(monkeypatch):
    m = _router_module()
    x = mx.array(np.random.default_rng(4).normal(size=(5, 1, 32)).astype(np.float32))
    taps.router_reset()
    monkeypatch.setattr(taps, "ROUTER", False)
    i0, w0 = R._router_sqrtsoftplus(m, None, x)
    assert taps.router_snapshot() is None
    monkeypatch.setattr(taps, "ROUTER", True)
    i1, w1 = R._router_sqrtsoftplus(m, None, x)
    assert np.array_equal(np.array(i0), np.array(i1)) and np.array_equal(np.array(w0), np.array(w1))
    taps.router_flush(force=True)
    snap = taps.router_snapshot()["layers"]["0"]
    assert snap["rows"] == 5 and abs(sum(snap["sum"]) - 5.0) < 1e-4            # the shares of a row add up to 1
    taps.router_reset()


def test_router_prompt_rows_are_not_tapped(monkeypatch):
    m = _router_module()
    taps.router_reset()
    monkeypatch.setattr(taps, "ROUTER", True)
    R._router_sqrtsoftplus(m, lambda self, x, image_mask=None: (mx.zeros((1, 3, 6), dtype=mx.int32), mx.ones((1, 3, 6))), mx.zeros((1, 3, 32)))
    assert taps.router_snapshot() is None


# ------------------------------------------------------------------------------------------------------------------ prefix reasons
class _B:
    def snapshot(self, c): return c
    def restore(self, s): return s
    def fork(self, c): return c
    def nbytes(self, s): return 100
    def shared_parts(self, s): return {}
    def draft_snapshot(self, d): return d


def _cache(budget=10_000):
    logs = []
    return PrefixCache(_B(), 4, budget, log=logs.append), logs


def _store(pc, ids, upto):
    for n, dg in prefix_digests(ids, list(range(pc.block, upto + 1, pc.block))):
        pc.store((n, dg), {"n": n})


def test_prefix_reasons():
    pc, logs = _cache()
    a = list(range(100, 140))
    assert pc.lookup(a)[0] == 0
    assert "prompt=40 hit=0 tier=none reason=restart" in logs[-1]
    pc.lookup(list(range(500, 520)))
    assert "reason=new" in logs[-1]
    _store(pc, a, 36)
    assert pc.lookup(a)[0] == 36 and "hit=36 tier=mem reason=hit" in logs[-1]
    pc.lookup(a[:21] + [999] + a[22:])                              # differs at token 21: the shared prefix is 20 tokens
    assert "hit=20 tier=mem reason=hit" in logs[-1]
    pc.entries.clear(); pc.bytes = 0
    pc.lookup(a)                                                    # miss: records a as the last prompt of its family
    pc.lookup(a[:30] + [998] * 10)                                  # differs from it at token 30, inside block (28, 32]
    assert "reason=diverged D_in=(28,32] entry_len=40" in logs[-1]
    pc.lookup(a[:36] + [7, 8, 9, 10, 11])                           # extends the last prompt's family member, no entry holds it
    pc.lookup(a[:36] + [7, 8, 9, 10, 11, 12])                       # the previous prompt is a prefix of this one
    assert "reason=unstored" in logs[-1]
    assert pc.reasons["restart"] == 1 and pc.reasons["new"] == 1 and pc.reasons["unstored"] == 1


def test_prefix_evicted_reason():
    pc, logs = _cache(budget=250)                                   # room for two entries
    pc._first = False
    a = list(range(100, 140))
    _store(pc, a, 12)                                               # keys 4, 8, 12: the key at 4 was dropped
    assert len(pc.entries) == 2 and len(pc.evicted) == 1
    pc.lookup(a[:6])                                                # its longest key (4) is the evicted one
    assert "hit=0 tier=none reason=evicted evicted_len=4" in logs[-1]
    pc.lookup(a[:3])
    assert "reason=short" in logs[-1]


def test_prefix_hit_path_is_cheap():
    pc, logs = _cache()
    ids = list(range(2000))
    _store(pc, ids, 1600)
    pc.lookup(ids)
    t0 = time.perf_counter()
    for _ in range(50):
        r = pc.lookup(ids)
    per_ms = (time.perf_counter() - t0) / 50 * 1e3
    assert r[0] == 1600
    t0 = time.perf_counter()
    for _ in range(50):
        pc._log_lookup(ids, 1600, "mem")
    log_ms = (time.perf_counter() - t0) / 50 * 1e3
    print(f"\nlookup {per_ms:.3f} ms, reason line {log_ms:.3f} ms (2000 tokens, block 4)")
    assert log_ms < 1.0


def test_router_snapshot_mixed_topk():
    """Routers of different top-k (main 6, draft head 3) must not break the snapshot (it feeds /health and /v1/stats)."""
    import types
    import mlx.core as mx
    from kyojin_mlx import taps as T
    T.router_reset()
    a, b = types.SimpleNamespace(), types.SimpleNamespace()
    T.router_tap(a, mx.array([[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]]))
    T.router_tap(b, mx.array([[0.6, 0.3, 0.1]]))
    T.router_flush(force=True)
    s = T.router_snapshot()
    assert s["all_k"] == 6 and s["all"]["rows"] == 1 and len(s["other_k_layers"]) == 1
    T.router_reset()
