"""dsa_decode.scores_rows (KYOJIN_ISCORE_MMA): the indexer scores of every row of a batched pass in one launch, against the solo scoring
kernel of the same row (exact, same arithmetic) and against the one-thread-per-key chain it replaces at depth (rounding-level, same -inf)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41.language")

from kyojin_mlx import dsa_decode as D                                            # noqa: E402
from kyojin_mlx import dsv4_core as C                                             # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
NH, DI = 32, 128


def world(n_rows=3, cap=1024, sessions=2, dtype=mx.float16, seed=3, mask=None):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_rows, NH, DI)).astype(np.float32)
    w = rng.standard_normal((n_rows, NH)).astype(np.float32)
    keys = rng.standard_normal((sessions, cap, DI)).astype(np.float32)
    base = [150, cap, cap // 2, 3 * cap // 4, cap]
    lens = np.array([base[i % len(base)] for i in range(n_rows)], dtype=np.int32)
    sess = np.array([i % sessions for i in range(n_rows)], dtype=np.int32)
    cand = None
    if mask == "half":
        m = np.zeros((n_rows, cap), dtype=bool)
        m[:, : cap // 2] = True
        cand = m
    elif mask == "blocks":                       # 8-key blocks, like select_candidate_blocks
        blocks = rng.random((n_rows, cap // 8)) < 0.5
        cand = np.repeat(blocks, 8, axis=1)
    a = (mx.array(q).astype(dtype), mx.array(w).astype(dtype), mx.array(keys).astype(dtype),
         mx.array(lens), mx.array(sess), None if cand is None else mx.array(cand))
    mx.eval(*[x for x in a if x is not None])
    return a


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("mask", [None, "half", "blocks"])
def test_rows_equal_the_solo_scores(dtype, mask):
    """Row u of the batched launch holds exactly scores(q[u], w[u], keys[sess[u]], lens[u], cand[u]) with t_max = lens[u]."""
    q, w, keys, lens, sess, cand = world(dtype=dtype, mask=mask)
    t_max = int(lens.max())
    got = D.scores_rows(q, w, keys, lens, t_max, cand, sess)
    mx.eval(got)
    got = np.array(got.astype(mx.float32))
    for u in range(q.shape[0]):
        row_cand = None if cand is None else cand[u]
        ref = D.scores(q[u], w[u], keys[int(sess[u])], lens[u: u + 1], int(lens[u]), row_cand)
        mx.eval(ref)
        ref = np.array(ref.astype(mx.float32))[: int(lens[u])]                     # scores() pads its output to whole 512-key threadgroups
        np.testing.assert_array_equal(got[u, : int(lens[u])], ref)


def test_rows_against_the_thread_per_key_chain():
    """The values the batched core served before: same finite keys, same -inf mask, scores within the fp16 rounding of the head sum order."""
    q, w, keys, lens, sess, cand = world(n_rows=4, cap=2048, mask="blocks")
    t_max = 2048
    a = np.array(D.scores_rows(q, w, keys, lens, t_max, cand, sess).astype(mx.float32))
    b = np.array(C.index_scores(q, keys, w, lens, t_max, cand, sess, rt=True).astype(mx.float32))
    for u in range(q.shape[0]):
        n_u = int(lens[u])                                        # columns past lens are not written by either kernel (rt width)
        inf_a, inf_b = ~np.isfinite(a[u, :n_u]), ~np.isfinite(b[u, :n_u])
        np.testing.assert_array_equal(inf_a, inf_b)                # the same keys are masked out
        ok = np.isfinite(a[u, :n_u])
        diff = np.abs(a[u, :n_u][ok].astype(np.float32) - b[u, :n_u][ok].astype(np.float32))
        scale = max(1.0, float(np.abs(b[u, :n_u][ok]).max()))
        assert diff.max() <= 2e-3 * scale, (u, diff.max(), scale)
        # the ordering that the top-k depends on is stable: no key pair swaps by more than the noise
        order_a, order_b = np.argsort(-a[u, :n_u][ok]), np.argsort(-b[u, :n_u][ok])
        assert (order_a[:64] == order_b[:64]).mean() >= 0.98


def test_rows_geometry_gate_and_shapes():
    assert D.rows_supported(NH, DI, mx.float16) and D.rows_supported(NH, DI, mx.bfloat16)
    assert not D.rows_supported(16, DI, mx.float16) and not D.rows_supported(NH, 64, mx.float16)
    assert not D.rows_supported(NH, DI, mx.float32)
    q, w, keys, lens, sess, cand = world(n_rows=2, cap=512, mask=None)
    out = D.scores_rows(q, w, keys, lens, 512, cand, sess)
    assert out.shape == (2, 512) and out.dtype == q.dtype


def test_rows_single_session_default():
    """sess=None reads key buffer u; one row reproduces the solo kernel with the identity session map."""
    q, w, keys, lens, sess, _ = world(n_rows=1, cap=256, sessions=1, seed=9)
    lens = mx.array([256], dtype=mx.int32)                        # every key valid, like the solo kernel's n
    out = D.scores_rows(q, w, keys, lens, 256)
    ref = D.scores(q[0], w[0], keys[0], lens[:1], 256)
    mx.eval(out, ref)
    np.testing.assert_array_equal(np.array(out.astype(mx.float32))[0][:256], np.array(ref.astype(mx.float32))[:256])
