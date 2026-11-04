"""pf_tail_grp: one grouped routed-MoE call for a prefill row band wider than a decode chunk (the CED decoder tail: 128 rows x top-6 = 768 pairs).

1. `test_wide_plan`: the wide plan kernel (moe.metal_src._GROUP_PLAN_BIG, up to 1024 pairs on one threadgroup) gives the same lead / cnt_ / mem
   as the 256-pair kernel at NP <= 256, and the plan a numpy group-by computes at 768 pairs (the CED tail's pair count).
2. `test_tail_grp_values`: with the lever on, the whole band runs as ONE grouped call instead of ceil(rows / 4) four-row calls. The per-row
   kernel bodies are the same (the hardcoded class rows of GROUP_CLASSES); only the k-split (tgk) of the wider call differs, so the values agree
   with the base arm and with the per-expert reference inside the fp32 partial-sum class (max rel difference printed).
3. The flag's default and its runtime attribute.
"""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))
from test_moe import make_stack, E, TOPK, KS, _route, _x  # noqa: E402
from kyojin_mlx import fired as FD, flags as FL, metal_src as S, moe as M, patch as P  # noqa: E402


def plan_np(ids, rows):
    """numpy group plan: lead [sum(plan_caps)], cnt_ [NP], mem [NP * ROWS] (the per-class layout of _GROUP_PLAN)."""
    ids = [int(v) for v in np.array(ids).reshape(-1)]
    npairs = len(ids)
    groups = {}
    for p, e in enumerate(ids):
        groups.setdefault(e, []).append(p)
    lows = [lo for lo, _ in S.group_classes()]
    caps = S.plan_caps(npairs)
    lead = np.full(sum(caps), 0xFFFFFFFF, dtype=np.uint32)
    cnt_ = np.zeros(npairs, dtype=np.uint32)
    mem = np.zeros(npairs * rows, dtype=np.uint32)
    per_class = {i: [] for i in range(len(lows))}
    for e, ps in groups.items():
        c = max(i for i, lo in enumerate(lows) if len(ps) >= lo)
        per_class[c].append((min(ps), len(ps), ps))
    base = 0
    for c, cap in enumerate(caps):
        for slot, (lead_p, cnt, ps) in enumerate(sorted(per_class[c])):
            lead[base + slot] = lead_p
            cnt_[lead_p] = cnt
            mem[lead_p * rows:lead_p * rows + cnt] = ps
        base += cap
    return lead, cnt_, mem


def run_plan(npairs, rows, big, seed=3, ids=None):
    ids = _route_pair_ids(npairs, seed) if ids is None else ids
    k = M._kernel(("gplan_test", npairs, big), name=f"kyojin_gplan_test_{npairs}" + ("_big" if big else ""), input_names=["ids", "params"],
                  output_names=["lead", "cnt_", "mem"], source=S.group_plan_source(0, False, big))
    shapes = [(sum(S.plan_caps(npairs)),), (npairs,), (npairs * rows,)]
    return k(inputs=[ids, mx.array([npairs, rows], dtype=mx.uint32)], grid=(256, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=shapes, output_dtypes=[mx.uint32] * 3)


def _route_pair_ids(npairs, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, E, npairs).astype(np.int32))


@pytest.mark.parametrize("npairs", [24, 96, 256])
def test_wide_plan_matches_256(npairs):
    """NP <= 256: the wide plan and the 256-pair kernel write the same lead / cnt_ / mem (cnt_ only for leaders: the other rows are not written)."""
    for rows in (4, 128):
        a = run_plan(npairs, rows, False, seed=npairs + rows)
        b = run_plan(npairs, rows, True, seed=npairs + rows)
        lead_a, lead_b = np.array(a[0]), np.array(b[0])
        assert np.array_equal(lead_a, lead_b)
        lead = lead_a[lead_a != 0xFFFFFFFF]                                           # leader pair ids (the only cnt_ / mem rows written)
        assert lead.size > 0
        cnt_a, cnt_b = np.array(a[1]), np.array(b[1])
        assert np.array_equal(cnt_a[lead], cnt_b[lead])
        m = np.arange(rows)[None, :] < cnt_a[lead][:, None]                            # members only: past cnt neither kernel writes
        assert np.array_equal(np.array(a[2]).reshape(npairs, rows)[lead][m], np.array(b[2]).reshape(npairs, rows)[lead][m])


@pytest.mark.parametrize("npairs", [128, 768, 1024])
def test_wide_plan_768(npairs):
    """The CED tail's 768 pairs (and the 1024 cap): the wide plan equals the numpy group-by."""
    rows = 128
    lead, cnt, mem = run_plan(npairs, rows, True, seed=99)
    lead_n, cnt_n, mem_n = plan_np(_route_pair_ids(npairs, 99), rows)
    up = cnt_n > 0
    assert np.array_equal(np.array(lead), lead_n)
    assert np.array_equal(np.array(cnt)[up], cnt_n[up])
    lead = np.array(lead)[np.array(lead) != 0xFFFFFFFF]
    m = np.arange(rows)[None, :] < cnt_n[lead][:, None]                                 # members only: past cnt the kernel writes nothing
    assert np.array_equal(np.array(mem).reshape(npairs, rows)[lead][m], mem_n.reshape(npairs, rows)[lead][m])


def _layer(wpre=False, fq=False, cb="mul1"):
    g, _ = make_stack(256, 384, KS, cb, 1)
    u, _ = make_stack(256, 384, KS, cb, 2)
    d, _ = make_stack(384, 256, KS, cb, 3)
    return M.MoeGather(g, u, d, TOPK, wpre=wpre, fq=fq)


@pytest.mark.parametrize("wpre,fq", [(False, False), (True, True)])
@pytest.mark.parametrize("rows", [12, 40, 128])
@pytest.mark.parametrize("half_rows", [0, 4])
def test_tail_grp_values(monkeypatch, rows, wpre, fq, half_rows):
    """The one-call band and the four-row chunks agree with the per-expert path in the same fp32 class (max rel difference printed)."""
    monkeypatch.setattr(S, "HALF_ROWS", half_rows)                                   # 4 = the served packed-fp16 row class
    monkeypatch.setattr(P, "PF_MIN_ROWS", str(1 << 20))                              # stay in the decode band (E is 12 experts here, its floor is 3 rows)
    monkeypatch.setattr(FD, "ON", True)
    moe = _layer(wpre, fq)
    ids, wts = _route(rows, 400 + rows)
    x = _x(rows, 256, 500 + rows)
    ref = M.MoeUnfused(moe.g, moe.u, moe.d, TOPK, wpre=wpre, fq=fq)(x, ids, wts)     # per-expert Mul1Gemv, fp32
    monkeypatch.setattr(M, "TAIL_GRP", False)
    base = P._moe_rows(moe, x, ids, wts)
    half_self = 0.0
    if half_rows:                                                                   # the class of the path it replaces: its own fp16-row deviation
        monkeypatch.setattr(S, "HALF_ROWS", 0)
        fp32 = P._moe_rows(moe, x, ids, wts)
        mx.eval(fp32)
        half_self = float(np.abs(np.array(base) - np.array(fp32)).max())
        monkeypatch.setattr(S, "HALF_ROWS", half_rows)
    monkeypatch.setattr(M, "TAIL_GRP", True)
    FD.reset()
    arm = P._moe_rows(moe, x, ids, wts)
    assert FD.HITS.get("pf_tail_grp") == 1, "the one-call band did not run"
    mx.eval(base, arm, ref)
    b, a, r = np.array(base), np.array(arm), np.array(ref)
    scale = max(np.abs(r).max(), 1e-6)
    print(f"rows {rows} half {half_rows}: base vs ref {np.abs(b - r).max() / scale:.2e}, arm vs ref {np.abs(a - r).max() / scale:.2e}, "
          f"arm vs base {np.abs(a - b).max() / scale:.2e}, the half-row class of base {half_self / scale:.2e}")
    assert np.abs(a - r).max() <= 2 * np.abs(b - r).max() + 1e-6                 # no worse than the path it replaces
    assert np.abs(a - b).max() <= max(2.5 * half_self, 1e-6)                     # inside the existing fp16-row class (measured 1.0-1.7x)


def test_flag_default():
    f = FL.FLAGS["pf_tail_grp"]
    assert f.default == "0" and f.env == "KYOJIN_PF_TAIL_GRP" and f.runtime
    assert f.attrs == (("moe", "TAIL_GRP", FL._on),)
