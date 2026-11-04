"""creopen item 2 levers that need no GPU: pf_moe_bail_split (a bailed pass reruns as <= fused_max-row device-plan passes), pf_moe_rm_rows (row map gate by
rows per expert), pf_nosync_y (threadgroups per expert of the persist GEMM). Kernel timing is checked on the lane (tools/bench_pf_moe.py)."""
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL, metal_nax as NX, moe_prefill as MP


def test_defaults_off_and_wired():
    for name in ("pf_moe_bail_split", "pf_nosync_y", "pf_moe_rm_rows"):
        assert FL.FLAGS[name].default == "0"
    assert MP.BAIL_SPLIT is False and MP.RM_ROWS == 0 and NX.NOSYNC_Y == 0
    FL.set_many({"pf_moe_bail_split": "1", "pf_nosync_y": "32", "pf_moe_rm_rows": "96"})
    try:
        assert MP.BAIL_SPLIT is True and NX.NOSYNC_Y == 32 and MP.RM_ROWS == 96
    finally:
        FL.set_many({"pf_moe_bail_split": "0", "pf_nosync_y": "0", "pf_moe_rm_rows": "0"})
    assert MP.BAIL_SPLIT is False and NX.NOSYNC_Y == 0 and MP.RM_ROWS == 0


def test_rm_allowed_gate():
    assert MP.rm_allowed(10 ** 6, 3, 0)                       # 0 = every class
    assert MP.rm_allowed(96 * 40, 40, 96) and not MP.rm_allowed(96 * 40 + 1, 40, 96)
    assert MP.rm_allowed(0, 0, 5) and not MP.rm_allowed(6, 0, 5)       # no active expert: the mean is over max(1, active)


def test_persist_source_default_unchanged_and_y_applies():
    args = ((3,), "mul1", 32, 2, 32, 64, True, "full", 2, True, "", False, True, False)
    base = NX.coop_source(*args)
    assert NX.coop_source(*args, 0) == base                   # 0 = PERSIST_Y: the default source is untouched
    assert NX.coop_source(*args, NX.PERSIST_Y) == base
    wide = NX.coop_source(*args, 32)
    assert wide != base and f"mg += {32 * 64})" in wide and f"mg += {NX.PERSIST_Y * 64})" in base


def test_bail_split_reruns_in_fused_max_passes(monkeypatch):
    pf = MP.MoePrefill(types.SimpleNamespace(), chunk=8192, fused=False)
    pf.fused, pf.h = True, True
    seen = []

    def glu(x, ids, wts):
        seen.append(x.shape[0])
        return None if x.shape[0] > MP.FUSED_MAX else x[:, :4] * 2.0

    monkeypatch.setattr(pf, "_chunk_glu", glu)
    monkeypatch.setattr(MP, "BAIL_SPLIT", True)
    T = 2 * MP.FUSED_MAX + 100
    x = mx.array(np.arange(T * 4, dtype=np.float32).reshape(T, 4))
    y = pf._chunk(x, mx.zeros((T, 6), dtype=mx.int32), mx.ones((T, 6)))
    assert seen == [T, MP.FUSED_MAX, MP.FUSED_MAX, 100]
    assert np.array_equal(np.array(y), np.array(x) * 2.0)


def test_bail_split_off_keeps_host_plan(monkeypatch):
    pf = MP.MoePrefill(types.SimpleNamespace(), chunk=8192, fused=False)
    pf.fused, pf.h = True, True
    monkeypatch.setattr(pf, "_chunk_glu", lambda x, ids, wts: None)
    monkeypatch.setattr(MP, "BAIL_SPLIT", False)
    n = MP.FUSED_MAX + 1
    with pytest.raises(Exception):                             # falls to the host plan, which needs a real MoE (here a bare namespace)
        pf._chunk(mx.zeros((n, 4)), mx.zeros((n, 6), dtype=mx.int32), mx.ones((n, 6)))
