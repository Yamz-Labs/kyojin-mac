"""Device-side MoE slot sort (kyojin_mlx.moe_sort) vs the numpy host plan: same expert ranges,
consistent slot maps, on several (T, top-k, E) including experts with 0 rows."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import moe_sort as MS


def ref_plan(ids_np, E, topk):
    flat = ids_np.reshape(-1)
    counts = np.bincount(flat, minlength=E)
    rank = np.zeros(E, np.int64)
    rank[np.nonzero(counts)[0]] = np.arange((counts > 0).sum())
    order = np.argsort(rank[flat], kind="stable")
    cum = np.concatenate([[0], np.cumsum(counts)])
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return counts, cum, order, inv


@pytest.mark.parametrize("T,topk,E,seed", [(9, 6, 12, 0), (64, 6, 12, 1), (300, 6, 64, 2),
                                          (1024, 6, 384, 3), (1, 1, 8, 4), (17, 8, 12, 5),
                                          (512, 6, 384, 6)])
def test_sort_matches_host_plan(T, topk, E, seed):
    rng = np.random.default_rng(seed)
    pool = rng.choice(E, size=max(topk, E * 3 // 4), replace=False)  # some experts get 0 rows
    ids_np = np.stack([rng.choice(pool, topk, replace=False) for _ in range(T)]).astype(np.int32)
    w = rng.uniform(0.1, 1.0, (T, topk)).astype(np.float32)
    ids, wts = mx.array(ids_np), mx.array(w)
    mx.eval(ids, wts)
    tok, ids_s, wts_s, inv, cum = MS.plan_device(ids, wts, E, topk)
    mx.eval(tok, ids_s, wts_s, inv, cum)
    counts, cum_ref, order, inv_ref = ref_plan(ids_np, E, topk)
    assert np.array_equal(np.array(cum), cum_ref), "expert offsets differ"
    assert np.array(counts).sum() == T * topk
    tok_h, ids_h, wts_h, inv_h = (np.array(tok), np.array(ids_s), np.array(wts_s), np.array(inv))
    flat = ids_np.reshape(-1)
    wflat = w.reshape(-1)
    # same multiset per expert segment (order inside a segment may differ)
    for e in range(E):
        s0, s1 = cum_ref[e], cum_ref[e + 1]
        if s1 == s0:
            continue
        seg_tok = np.sort(tok_h[s0:s1])
        want_tok = np.sort(np.nonzero(flat == e)[0] // topk)
        assert np.array_equal(seg_tok, want_tok), (e, T, topk)
        assert np.array_equal(np.sort(ids_h[s0:s1]), np.full(s1 - s0, e, dtype=np.int32)), (e, T, topk)
        seg = np.stack([tok_h[s0:s1], np.array([flat[t * topk + j] for t in tok_h[s0:s1] for j in range(topk)
                                               if flat[t * topk + j] == e])], axis=1) if False else None
        # weights follow their pairs
        for s in range(s0, s1):
            t = int(tok_h[s])
            cands = [wflat[t * topk + j] for j in range(topk) if flat[t * topk + j] == e]
            assert any(abs(float(wts_h[s]) - float(c)) < 1e-9 for c in cands), (e, s)
    # inv consistency: inv[p] points at a slot whose (tok, wts) match pair p
    for p in [0, T * topk // 2, T * topk - 1]:
        s = int(inv_h[p])
        assert int(tok_h[s]) == p // topk
        assert abs(float(wts_h[s]) - float(wflat[p])) < 1e-9


def test_sort_zero_row_experts_explicit():
    E, topk, T = 8, 2, 6
    ids_np = np.array([[0, 1]] * T, dtype=np.int32)  # experts 2..7 get 0 rows
    wts = mx.array(np.ones((T, topk), np.float32))
    tok, ids_s, wts_s, inv, cum = MS.plan_device(mx.array(ids_np), wts, E, topk)
    mx.eval(tok, ids_s, wts_s, inv, cum)
    assert np.array(cum).tolist() == [0, 6, 12, 12, 12, 12, 12, 12, 12]
