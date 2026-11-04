"""cmoe8k: CPU-side counts of the prefill MoE path (launches, syncs, allocations, threadgroups, weight decodes) on a small synthetic shape that keeps the
served ratios (rows per expert ~64 on average, 1/6 of the experts hot, ~54 % of the pairs in experts of <= 64 rows), and the pf_nosync_rowend fix.

The REAL MoePrefill code runs with `mx.fast.metal_kernel` replaced by a recorder (tools/pf_moe_count.py), so nothing here times a kernel or checks a kernel's
bits: those need Metal (tools/bench_pf_moe_served.py). What is checked is what the host asks the GPU to do."""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import pf_moe_count as C  # noqa: E402
from kyojin_mlx import flags as FL, metal_nax as NX, moe_prefill as MP  # noqa: E402

E, BASE, HIDDEN, INTER = 64, 704, 512, 512          # BASE = rows of a default MoE pass at this scale (the served 4096 pass has the same rows per expert)
ARMS = ("base4096", "chunk8192", "nosync4096", "nosync_rowend4096", "nosync8192", "nosync_rowend8192", "rowmap", "epi", "rowmap+epi")


@pytest.fixture(scope="module")
def res():
    moe = C.make_moe(E=E, hidden=HIDDEN, inter=INTER)
    pop = C.served_popularity(E, C.TOPK, BASE)
    return {a: C.run_arm(a, rows=2 * BASE, layers=3, moe=moe, pop=pop, base_pass=BASE) for a in ARMS}


def test_epi_cap_flag_and_fresh_allocations():
    f = FL.FLAGS["pf_moe_epi_cap"]
    assert f.default == "0" and f.env == "KYOJIN_PF_EPI_CAP" and MP.EPI_CAP is False
    FL.set_many({"pf_moe_epi_cap": "1"})
    try:
        assert MP.EPI_CAP is True
    finally:
        FL.set_many({"pf_moe_epi_cap": "0"})
    moe = C.make_moe(E=E, hidden=HIDDEN, inter=INTER)
    pop = C.served_popularity(E, C.TOPK, BASE)
    r = {a: C.run_arm(a, rows=BASE, layers=40, moe=moe, pop=pop, base_pass=BASE) for a in ("base4096", "epi", "epi_cap")}
    assert r["epi"]["cache"]["tail_bytes"] > 5 * max(r["epi_cap"]["cache"]["tail_bytes"], 1 << 20)      # exact-size output: a miss per layer even after the cache is warm
    assert r["epi_cap"]["cache"]["tail_bytes"] <= r["base4096"]["cache"]["tail_bytes"] + (64 << 20)
    assert r["epi_cap"]["launches"] == r["epi"]["launches"] and r["epi_cap"]["gemm"]["row_blocks"] == r["epi"]["gemm"]["row_blocks"]


def test_flag_off_by_default_and_wired():
    f = FL.FLAGS["pf_nosync_rowend"]
    assert f.default == "0" and f.env == "KYOJIN_PF_NOSYNC_ROWEND" and f.runtime
    assert MP.NOSYNC_ROWEND is False
    FL.set_many({"pf_nosync_rowend": "1"})
    try:
        assert MP.NOSYNC_ROWEND is True
    finally:
        FL.set_many({"pf_nosync_rowend": "0"})
    assert MP.NOSYNC_ROWEND is False


def test_rowend_source_differs_only_on_the_row_range_line():
    args = ((3,), "mul1", 32, 2, 32, 64, True, "full", 2, True, "", False, True, False, 0)
    old, new = NX.coop_source(*args), NX.coop_source(*args, rowend=True)
    assert NX.coop_source(*args, rowend=False) == old                     # default source untouched
    a, b = old.splitlines(), new.splitlines()
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 1 and "rowoff[z + 1]" in diff[0][0] and "rowoff[zo + 1]" in diff[0][1] and "z >= GL" in diff[0][1]
    with pytest.raises(AssertionError):
        NX.coop_source((3,), "mul1", 32, 2, 32, 64, True, "full", 1, True, "", False, True, False, 0, True)       # one projection: rowend is gate / up only


def test_launch_and_sync_counts(res):
    b, c = res["base4096"], res["chunk8192"]
    assert c["launches"] * 2 == b["launches"]                  # one pass per layer instead of two: every launch of a pass halves
    assert c["syncs"] < b["syncs"] and c["bailed"] == 0 and b["bailed"] == 0
    assert res["nosync4096"]["syncs"] < b["syncs"]             # no cum readback per pass
    assert res["nosync8192"]["syncs"] <= 1                     # no eval at all in a single pass (the one is the first-call rank table)
    assert res["nosync8192"]["launches"] < res["nosync4096"]["launches"]


def test_chunk8192_counts_favour_the_merge_except_concurrency(res):
    b, c = res["base4096"]["gemm"], res["chunk8192"]["gemm"]
    assert c["row_blocks"] <= b["row_blocks"] and c["decoded_bytes"] <= b["decoded_bytes"]      # fewer weight decodes, never more (ceil(a + b) <= ceil(a) + ceil(b))
    assert abs(c["dead_tg"] - b["dead_tg"]) <= 0.05 * b["dead_tg"]                               # dead threadgroups: grid.y follows the busiest expert, which doubles while passes halve
    assert c["max_blocks"] > b["max_blocks"]                                                      # one hot expert's blocks run side by side: the one thing that grows
    assert res["chunk8192"]["max_buffer"] >= 1.9 * res["base4096"]["max_buffer"]


def test_nosync_glued_row_range_costs_decodes_and_a_serial_chain(res):
    b = res["base4096"]["gemm"]
    for arm in ("nosync4096", "nosync8192"):
        g = res[arm]["gemm"]
        assert g["row_blocks"] > 1.2 * b["row_blocks"]         # the last gate expert of each class walks garbage rows
        assert g["chain_max"] >= 4 * res[arm.replace("nosync", "nosync_rowend")]["gemm"]["chain_max"] or g["chain_max"] >= 7
    for arm in ("nosync4096", "nosync8192"):
        fixed = res[arm.replace("nosync", "nosync_rowend")]
        base = res["base4096" if arm.endswith("4096") else "chunk8192"]["gemm"]
        assert fixed["gemm"]["row_blocks"] == base["row_blocks"]                    # same row blocks as the sync path
        assert fixed["gemm"]["decoded_bytes"] == base["decoded_bytes"]
        assert fixed["launches"] == res[arm]["launches"] and abs(fixed["alloc_bytes"] - res[arm]["alloc_bytes"]) < 1024      # same launches, same buffers: only the row range moves
        assert fixed["gemm"]["chain_sum"] < 0.5 * res[arm]["gemm"]["chain_sum"]


def test_nosync_still_pays_class_sized_copies(res):
    assert res["nosync4096"]["alloc_bytes"] > 1.5 * res["base4096"]["alloc_bytes"]      # P-sized buffers per class (not fixed by rowend)
    assert res["nosync4096"]["concat_bytes"] > 1.5 * res["base4096"]["concat_bytes"]     # `part` = classes x cap rows


def test_rowmap_and_epi_counts(res):
    b = res["base4096"]
    assert res["rowmap"]["by"].get("rot", 0) == 0 < b["by"]["rot"]
    assert res["rowmap"]["gemm"]["xform_x"] > 1                    # the input transform repeats once per N-group threadgroup
    assert res["epi"]["by"].get("gemm_gu_epi", 0) > 0 and res["epi"]["by"].get("mid", 0) < b["by"]["mid"]
    assert res["epi"]["gemm"]["decoded_bytes"] == b["gemm"]["decoded_bytes"]            # same blocks as pick_cfg
    assert res["rowmap+epi"]["gemm"]["decoded_bytes"] > 1.2 * b["gemm"]["decoded_bytes"]      # the row map forces 32-row blocks in the epilogue kernel


def test_rowend_rowoff_values_and_last_gate_range():
    """The rowoff the nosync path hands the persist GEMM: glued form ends the last gate expert at class start + P, rowend form at its class end."""
    moe = C.make_moe(E=E, hidden=HIDDEN, inter=INTER)
    pop = C.served_popularity(E, C.TOPK, BASE)
    last = {}
    for rowend in (False, True):
        rec = C.Rec()
        saved = (NX.site_on, NX.nax_available, C.MS.plan_device, C.mx.fast.metal_kernel, MP.MOE_NOSYNC, MP.NOSYNC_ROWEND)
        try:
            NX.site_on, NX.nax_available, C.MS.plan_device = (lambda s: True), (lambda: True), C.plan_twin
            C.mx.fast.metal_kernel = rec.kernel_factory
            MP.MOE_NOSYNC, MP.NOSYNC_ROWEND = True, rowend
            for m in (MP, NX, C.GU, C.MS):
                m._kc.clear()
            ids = C.route(pop, BASE, C.TOPK, 3, 0, BASE)
            MP.MoePrefill(moe, chunk=BASE)(mx.zeros((BASE, HIDDEN), dtype=mx.float16), mx.array(ids), mx.zeros((BASE, C.TOPK)))
            gus = [L for L in rec.launches if C.classify(L["name"]) == "gemm" and "_p2_" in L["name"]]
        finally:
            NX.site_on, NX.nax_available, C.MS.plan_device, C.mx.fast.metal_kernel, MP.MOE_NOSYNC, MP.NOSYNC_ROWEND = saved
        P = BASE * C.TOPK
        assert len(gus) >= 2
        for L in gus:
            ro, GL = np.array(L["ins"]["rowoff"]).astype(np.int64), len(np.array(L["ins"]["eids"]))
            if rowend:
                assert len(ro) == 2 * GL + 2 and ro[GL] - ro[0] == ro[2 * GL + 1] - ro[GL + 1] and ro[GL + 1] == ro[0] + P
                last_len = ro[GL] - ro[GL - 1]
            else:
                assert len(ro) == 2 * GL + 1 and ro[GL] == ro[0] + P
                last_len = ro[GL] - ro[GL - 1]
            last.setdefault(rowend, []).append((last_len, ro[0], ro[GL - 1], GL))
        if not rowend:
            assert max(n for n, *_ in last[False]) > 2 * P // 3 - P // 4        # the glued range runs the last gate expert over most of the other classes' slots
    assert max(n for n, *_ in last[True]) < max(n for n, *_ in last[False]) / 4


def test_cache_sim_reuse_rule():
    mib = 1 << 20
    r = C.cache_sim([[10 * mib, 20 * mib], [10 * mib, 20 * mib + 3 * C.MLX_PAGE], [10 * mib - C.MLX_PAGE]])
    assert r["first_n"] == 2 and r["steady_hit_n"] == 2          # the same size and a size within 2 pages are reused; 20 MiB + 3 pages is a miss
    assert r["steady_n"] == 1


def test_served_bench_script_bounds_and_aliasing(capsys, monkeypatch):
    """tools/bench_pf_moe_served.py: the memory bound is computed before anything is built, the weights alias onto a few physical banks, nosync at 8192 rows is refused."""
    import bench_pf_moe_served as B
    assert B.weight_bytes(16, (2, 3, 3, 4)) < 0.3 * 2 ** 30 and B.weight_bytes(64, (2, 3, 3, 4)) < 1.0 * 2 ** 30
    s = B.make_stack(64, 32, (2, 3, 3, 4), 12, 4, "mul1", 1)
    off, kk = np.array(s.offset), np.array(s.k)
    assert len(off) == 13 and (off[:4] == off[4:8]).all() and (off[:4] == off[8:12]).all() and (kk[:4] == kk[8:12]).all()     # logical e reads bank e % 4
    assert list(kk[:4]) == [2, 3, 3, 4] and int(s.words.size) == sum(8 * k * 4 * 2 for k in (2, 3, 3, 4))
    monkeypatch.setattr(sys, "argv", ["x", "--arm", "nosync8192", "--estimate-only"])
    assert B.main() == 2 and '"refused": true' in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x", "--arm", "base4096", "--rows", "4096", "--estimate-only"])
    assert B.main() == 0 and "memory bound" in capsys.readouterr().out


def test_served_bench_script_dry_run_end_to_end(capsys, monkeypatch):
    """The whole Mac script (stacks, routing sets, timing loop, digest, FD hits) on the CPU against recording kernels: catches Python errors before the Mac does."""
    import json
    import bench_pf_moe_served as B
    monkeypatch.setattr(sys, "argv", ["x", "--arm", "nosync_rowend4096", "--rows", "4096", "--layers", "1", "--sets", "1", "--reps", "1", "--dry-run"])
    assert B.main() == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["dry_run"] and out["fd"].get("pf_nosync_rowend", 0) >= 1 and out["fd"].get("pf_moe_nosync", 0) >= 1 and not out["over_budget"]
    assert MP.MOE_NOSYNC is False and MP.NOSYNC_ROWEND is False                                            # module flags restored


def test_nosync_glued_range_is_exact_with_one_bitrate_class():
    """One class: the class start is 0 and its end is P, so the glued range [gate | up] closes the last gate expert correctly: nosync == base == rowend in row blocks.
    The excess of the glued form is (classes - 1) * P rows per pass and layer; the ledger's +1.25 % (pfsmall) and -21 % (orch5) cannot be told apart from it
    on the CPU (both trees carry the glued form, git show 281d5f5)."""
    moe = C.make_moe(E=E, ks=(3,), hidden=HIDDEN, inter=INTER)
    pop = C.served_popularity(E, C.TOPK, BASE)
    r = {a: C.run_arm(a, rows=BASE, layers=2, moe=moe, pop=pop, base_pass=BASE) for a in ("base4096", "nosync4096", "nosync_rowend4096")}
    assert r["nosync4096"]["gemm"]["row_blocks"] == r["base4096"]["gemm"]["row_blocks"] == r["nosync_rowend4096"]["gemm"]["row_blocks"]
    assert r["nosync4096"]["gemm"]["chain_sum"] == r["nosync_rowend4096"]["gemm"]["chain_sum"]
    two = C.make_moe(E=E, ks=(2, 3), hidden=HIDDEN, inter=INTER)
    r2 = {a: C.run_arm(a, rows=BASE, layers=2, moe=two, pop=pop, base_pass=BASE) for a in ("base4096", "nosync4096")}
    assert r2["nosync4096"]["gemm"]["row_blocks"] > r2["base4096"]["gemm"]["row_blocks"]           # two classes already pay
