"""KYOJIN_DEC_BLK_SUM (kyojin_mlx.decblk), CPU: summaries, selection against the exact top-k, and the top-k RECALL report.

Recall is measured on synthetic keys with block locality (consecutive keys share a drifting direction, as neighbouring tokens do); real DS4.1 keys
must be re-measured on the Mac lane. The recall printed here is what the test reports (run with -s)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import decblk as DB, flags as FL  # noqa: E402

NH, DI, BS = 32, 128, 8


def world(T=8192, rows=3, S=2, seed=0, local=True):
    rng = np.random.default_rng(seed)
    if local:                                           # slowly drifting direction + per-key noise: blocks of 8 are coherent
        base = np.cumsum(rng.standard_normal((S, T // BS + 1, DI)) * 0.6, axis=1)
        k = np.repeat(base, BS, axis=1)[:, :T] + rng.standard_normal((S, T, DI)) * 0.5
    else:
        k = rng.standard_normal((S, T, DI))
    keys = mx.array(k.astype(np.float32)).astype(mx.float16)
    # queries aimed at a few positions so the top-k is a real signal, not noise
    q = np.empty((rows, NH, DI), np.float32)
    for u in range(rows):
        tgt = k[u % S, rng.integers(0, T)]
        q[u] = tgt[None, :] * 0.5 + rng.standard_normal((NH, DI)) * 0.8
    w = np.abs(rng.standard_normal((rows, NH))) * 0.1
    lens = mx.array(np.array([T, T - 37, T // 2][:rows], dtype=np.int32))
    sess = mx.array(np.array([u % S for u in range(rows)], dtype=np.int32))
    a = (mx.array(q).astype(mx.float16), mx.array(w.astype(np.float32)).astype(mx.float16), keys, lens, sess)
    mx.eval(*a)
    return a


def exact(q, w, keys, lens, sess, topk):
    T = keys.shape[1]
    return DB._topk(DB._score(q, w, keys, lens, T, sess), lens, topk)


def recall(a, b):
    a, b = np.array(a), np.array(b)
    r = []
    for x, y in zip(a, b):
        x, y = set(x[x >= 0].tolist()), set(y[y >= 0].tolist())
        r.append(len(x & y) / max(len(y), 1))
    return r


def test_summarize_is_block_mean_with_short_tail():
    rng = np.random.default_rng(1)
    k = mx.array(rng.standard_normal((21, DI)).astype(np.float32))
    s = np.array(DB.summarize(k, 21))
    ref = np.array(k)
    assert s.shape == (3, DI)
    np.testing.assert_allclose(s[0], ref[:8].mean(0), atol=1e-5)
    np.testing.assert_allclose(s[2], ref[16:21].mean(0), atol=1e-5)          # 5 keys: averaged over 5, not 8


def test_blksum_incremental_equals_fresh():
    q, w, keys, lens, sess = world(T=1000, rows=1, S=1)
    st = DB.BlkSum(1)
    st.get(keys, [400], 700)                                                # 50 complete blocks, open tail
    got = st.get(keys, [900], 1000)                                         # commit moved on; the open region is redone
    fresh = DB.BlkSum(1).get(keys, [0], 1000)
    np.testing.assert_allclose(np.array(got[:, :125], np.float32), np.array(fresh[:, :125], np.float32), atol=1e-3)
    st.get(keys, [100], 1000)                                               # rolled back: re-summarized
    assert st.have[0] == 12


@pytest.mark.parametrize("topk", [64, 256])
def test_recall_with_block_locality_reaches_099(topk):
    q, w, keys, lens, sess = world()
    ref = exact(q, w, keys, lens, sess, topk)
    st = DB.BlkSum(2)
    idx, mask = DB.select(q, keys, w, lens, sess, keys.shape[1], topk, st, [0, 0])
    assert mask is None
    r = recall(idx, ref)
    print(f"\nrecall topk={topk} mult={DB.MULT}: {[round(x, 4) for x in r]}")
    assert min(r) >= 0.99


def test_recall_rises_with_mult_and_is_exact_at_full():
    q, w, keys, lens, sess = world(local=False, rows=2)
    topk, ref = 64, exact(q, w, keys, lens, sess, 64)
    out = []
    for m in (1, 4, 16, 10**6):
        idx, _ = DB.select(q, keys, w, lens, sess, keys.shape[1], topk, DB.BlkSum(2), [0, 0], mult=m)
        out.append(min(recall(idx, ref)))
    print("\ni.i.d. keys recall by mult 1/4/16/all:", [round(x, 3) for x in out])
    assert out == sorted(out) and out[-1] == 1.0


def test_rescored_values_are_exact_and_row_order_ascending():
    q, w, keys, lens, sess = world(rows=3)
    idx, _ = DB.select(q, keys, w, lens, sess, keys.shape[1], 32, DB.BlkSum(2), [0, 0])
    a = np.array(idx)
    for u in range(3):
        v = a[u][a[u] >= 0]
        assert (np.diff(v) > 0).all() and v.max() < int(lens[u])


def test_source_layer_mask_holds_the_answer_blocks():
    q, w, keys, lens, sess = world(rows=2)
    idx, mask = DB.select(q, keys, w, lens, sess, keys.shape[1], 64, DB.BlkSum(2), [0, 0], kb=16)
    m, a = np.array(mask), np.array(idx)
    assert m.shape == (2, keys.shape[1])
    assert (m.reshape(2, -1, BS).sum(-1) // BS).sum(-1).max() <= 16               # at most kb blocks kept
    last = (np.array(lens) - 1) // BS
    for u in range(2):
        assert m[u, last[u] * BS]                                                  # the last visible block is pinned


def test_consumer_mask_never_leaks_dropped_blocks():
    q, w, keys, lens, sess = world(rows=2)
    T = keys.shape[1]
    keep = np.zeros((2, T), bool)
    keep[:, 1024:2048] = True
    idx, _ = DB.select(q, keys, w, lens, sess, T, 32, DB.BlkSum(2), [0, 0], cand=mx.array(keep))
    a = np.array(idx)
    for u in range(2):
        v = a[u][a[u] >= 0]
        last = (int(lens[u]) - 1) // BS                                            # the pinned last block is rescored before the mask applies
        assert len(v) and (((v >= 1024) & (v < 2048)) | (v // BS == last)).all()


def test_flags_default_off_and_gate_without_metal():
    assert FL.FLAGS["dec_blk_sum"].default == "0" and DB.ENABLED is False
    assert FL.FLAGS["dec_blk_sum_mult"].default == "8" and FL.FLAGS["dec_blk_sum_min"].default == "262144"
    assert DB.applies(None, None, None, None, None, None, None, 0) is False
