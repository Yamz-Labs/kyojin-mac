import mlx.core as mx

from kyojin_mlx.glm_nextn import GlmDraft


def _h(n, v=0.0):
    return mx.full((n, 4), v, dtype=mx.float16)


def test_append_contiguous_and_gap():
    d = GlmDraft()
    d.append(_h(3), mx.array([5, 6, 7]), 10)
    assert (d.pos, d.m) == (10, 3)
    d.append(_h(2), mx.array([8, 9]), 13)                     # next positions: concatenated
    assert (d.pos, d.m, d.htok.tolist()) == (10, 5, [5, 6, 7, 8, 9])
    d.cache = object()
    d.append(_h(1), mx.array([3]), 40)                        # a gap restarts the head's cache
    assert (d.pos, d.m, d.cache) == (40, 1, None)


def test_pending_after_consumption():
    d = GlmDraft()
    d.append(_h(2), mx.array([1, 2]), 0)
    d.pos, d.hid, d.htok = 2, None, None                      # what NextN.feed leaves after feeding both pairs
    d.append(_h(1), mx.array([9]), 2)
    assert (d.pos, d.m) == (2, 1)
