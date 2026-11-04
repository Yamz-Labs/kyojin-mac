"""spec_rs draws line up with the lazy draft's live-job slots: a session with no draft this round (depth 0) in front of a drafting one must not
shift the drafting session's tokens (LazyDrafts.slot indexes the live jobs only; spec_verify_lazy and tolist read lh.toks[lh.slot[i]])."""
import types

import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx.serve import backends as BK, engine as EN        # noqa: E402
from kyojin_mlx.serve.sampling import Sampling                    # noqa: E402

V, BLOCK = 16, 4


def members(n):
    return [types.SimpleNamespace(sampling=Sampling(temperature=1.0), key=None, seed=i) for i in range(n)]


def lazy(depths, live_logits):
    jobs = [(None, None, 0, d, None) for d in depths]
    live = [i for i, d in enumerate(depths) if d > 0]
    toks = mx.zeros((len(live), BLOCK), dtype=mx.int32)
    lh = BK.LazyDrafts(None, jobs, live, toks, mx.zeros((len(live), BLOCK)), lg=live_logits)
    return lh


@pytest.mark.parametrize("depths", [[0, 3], [0, 3, 0, 2], [2, 0, 3], [3, 3]])
def test_rs_tokens_follow_the_slots(depths):
    ms = members(len(depths))
    n_live = sum(1 for d in depths if d > 0)
    # live job j puts all its mass on token 3 + j: after the draw, row j of the stacked array is that token whatever the keys
    lg = mx.stack([mx.repeat(mx.where(mx.arange(V) == 3 + j, 50.0, -50.0)[None], BLOCK, axis=0) for j in range(n_live)])
    lh = lazy(depths, lg)
    fake = types.SimpleNamespace(_next_key=lambda s: EN.Engine._next_key(fake, s))
    rows, q, stacked = EN.Engine._rs_draws(fake, lh, ms)
    assert tuple(stacked.shape) == (n_live, BLOCK)
    lh.set_toks(stacked)
    out = lh.tolist()
    for i, d in enumerate(depths):
        if d > 0:
            assert out[i] == [3 + lh.slot[i]] * d, (i, out)
        else:
            assert out[i] == []
        assert (i in rows) == (d > 0)
