"""Batched drafting: one drafter pass over the rows of all sessions proposes the same tokens as one pass per session.
Needs a Qwen3.5 mlx pack with an mtp/ sidecar (KYOJIN_Q35MLX, default ~/rivals/models/Qwen3.5-4B-mlx4)."""
import os

import pytest

mx = pytest.importorskip("mlx.core")
PACK = os.environ.get("KYOJIN_Q35MLX", os.path.expanduser("~/rivals/models/Qwen3.5-4B-mlx4"))
pytestmark = [pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal"),
              pytest.mark.skipif(not os.path.isdir(os.path.join(PACK, "mtp")), reason="needs a Qwen3.5 mlx pack with an mtp/ sidecar")]


def test_batched_draft_equals_per_session_draft():
    from kyojin_mlx.serve import q35mlx as QM
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(PACK)
    ids = list(range(1000, 1000 + 300))
    sufs = [list(range(2000 + 10 * i, 2008 + 10 * i + i % 3)) for i in range(5)]
    base = be.make_cache(); d0 = be.new_draft()
    for s in range(0, len(ids), 256):
        be.prefill_chunk(base, ids[s:s + 256], draft=d0)
    H = d0.pend[-1].shape[-1]
    ks = [1 + i % 4 for i in range(len(sufs))]
    hids = [mx.random.normal((1, k, H)).astype(d0.pend[-1].dtype) for k in ks]
    depths = [1 + i % 3 for i in range(len(sufs))]
    out = {}
    for mode in (False, True):
        QM._BATCH_DRAFT = mode
        jobs = []
        for sf, k, h, dd in zip(sufs, ks, hids, depths):
            c = be.fork(base); d = be.draft_restore(be.draft_snapshot(d0))
            be.prefill_chunk(c, sf[:-1], draft=d)
            d.pend = [h]                                          # k pending trunk rows, as after a partly accepted verify
            jobs.append((d, sf[-1], d.mv + k, dd, (ids, sf + [7] * k)))
        out[mode] = be.draft_many(jobs)
    QM._BATCH_DRAFT = True
    assert out[True] == out[False]
    assert [len(x) for x in out[True]] == depths
