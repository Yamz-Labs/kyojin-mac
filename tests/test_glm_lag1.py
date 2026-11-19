"""Flag glm_lag1 (KYOJIN_GLM_LAG1): the lazy GLM draft and the verify rows built from the device draft equal the host rows of the eager path."""
import mlx.core as mx

from kyojin_mlx.serve import backends as B


def _be(capture):
    be = B.GlmBackend.__new__(B.GlmBackend)
    be.gated_reset = lambda bc: None
    be._verify_rows = lambda bc, rows, lens, tl: capture.append((rows.tolist(), lens, tl)) or "pass"
    return be


def test_lazy_drafts_pad_and_tolist():
    jobs = [(None, 5, 0, 2, None), (None, 6, 0, 0, None), (None, 7, 0, 1, None)]
    lh = B.GlmLazyDrafts(jobs, {0: mx.array([11, 12], dtype=mx.int32), 2: mx.array([21], dtype=mx.int32)})
    assert lh.depths == [2, 0, 1] and lh.counts == [2, 0, 1] and lh.slot == {0: 0, 2: 1}
    assert lh.toks.tolist() == [[11, 12], [21, 0]]
    assert lh.tolist() == [[11, 12], [], [21]] and lh.keep is None


def test_verify_lazy_rows_equal_host_rows():
    cap = []
    be = _be(cap)
    jobs = [(None, 5, 0, 2, None), (None, 6, 0, 0, None), (None, 7, 0, 1, None)]
    lh = B.GlmLazyDrafts(jobs, {0: mx.array([11, 12], dtype=mx.int32), 2: mx.array([21], dtype=mx.int32)})
    assert be.spec_verify_lazy(None, lh, [5, 6, 7]) == "pass"
    rows, lens, tl = cap[0]
    host = [[5, 11, 12], [6], [7, 21]]
    K = 3
    assert rows == [t + [t[-1]] * (K - len(t)) for t in host]      # the padding of spec_verify_bc
    assert lens == [3, 1, 2] and tl is None


def test_glm_pass_commit_uses_device_rows():
    got = []

    class Tx:
        def commit(self, n):
            got.append(("tx", n))

    class D:
        def append(self, h, toks, p0):
            got.append((h.shape, toks.tolist(), p0))

    rows = mx.array([[5, 11, 12], [6, 6, 6]], dtype=mx.int32)
    ps = B.GlmPass(Tx(), None, [0, 3], mx.zeros((2, 3, 4)), None, rows=rows)
    ps.commit([1, 0], [D(), D()], [10, 20])
    assert got[0] == ("tx", [2, 1]) and got[1] == ((2, 4), [5, 11], 10) and got[2] == ((1, 4), [6], 20)
