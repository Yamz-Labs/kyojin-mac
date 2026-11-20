"""draft_nbytes / draft_arrays of the generic_foreign MTP backend: the head cache state carries non-array entries (index_block_ratio int,
None index keys); a prefix store / save_disk of a session with a draft must size it without raising (c44 FP4 served crash 17/11)."""
import mlx.core as mx
from kyojin_mlx.serve import fnmtp


def _draft():
    d = fnmtp.DraftState()
    d.mc.update_and_fetch(mx.zeros((1, 2, 3, 8)), mx.zeros((1, 2, 3, 8)))
    d.mc.update_indexer(mx.zeros((1, 3, 4)), mx.arange(3).reshape(1, 3))
    d.mc.index_block_ratio = 4                       # an int in the cache state (set by the block indexer)
    d.pend = [mx.zeros((1, 1, 16))]
    return d


def test_draft_nbytes_state_with_ints():
    be = object.__new__(fnmtp.FnMtpBackend) if hasattr(fnmtp, "FnMtpBackend") else None
    cls = next(c for c in vars(fnmtp).values() if isinstance(c, type) and hasattr(c, "draft_nbytes") and c.__module__ == fnmtp.__name__)
    be = object.__new__(cls)
    d = _draft()
    n = cls.draft_nbytes(be, d)
    assert n == 2 * 3 * 2 * 8 * 4 + 3 * 4 * 4 + 3 * 4 + 16 * 4, n
    assert all(isinstance(a, mx.array) for a in cls.draft_arrays(be, d))
