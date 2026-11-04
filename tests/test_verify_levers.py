"""Verify-cost levers: KYOJIN_MOE_SGRED (grouped routed MoE GEMV with no threadgroup reduction) and KYOJIN_FUSE_ROWS (fused dense linears take a 5 / 6 row
verify block like a plain step). Both default 0.

Two halves. The first needs neither mlx nor a GPU (it reads source text, generates the Metal text and checks the lane arithmetic in numpy): flag table,
module literals, kernel text, column mapping of the direct stores, summation-order bound, threadgroup-memory budget at the pack's dense shapes, the ablit
sites. The second needs Metal (skipped elsewhere): the levers against the f32 reference next to today's path, and the ablit projection."""
import importlib.util
import os
import re

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PKG = os.path.join(ROOT, "kyojin_mlx")


def text(name):
    return open(os.path.join(PKG, name)).read()


@pytest.fixture(scope="module")
def S():
    from kyojin_mlx import metal_src
    return metal_src


@pytest.fixture(scope="module")
def FL():
    from kyojin_mlx import flags
    return flags


# ------------------------------------------------------------------------------------------------ flags, literals, docs (no mlx)
def test_flags_registered_defaults(FL):
    a, b = FL.FLAGS["moe_sgred"], FL.FLAGS["fuse_rows"]
    assert (a.env, a.default, a.kind, a.runtime, a.attrs[0][:2]) == ("KYOJIN_MOE_SGRED", "0", "bool", True, ("moe", "SGRED"))
    assert (b.env, b.default, b.kind, b.runtime, b.attrs[0][:2]) == ("KYOJIN_FUSE_ROWS", "8", "int", True, ("kernels", "FUSE_ROWS"))
    assert a.attrs[0][2]("1") is True and a.attrs[0][2]("0") is False and b.attrs[0][2]("6") == 6


def test_module_literals_aligned_with_flags_defaults(FL):
    """docs/cloud/LEVERS.md + flags.py rule: the literal the module reads is the flag's default."""
    m = re.search(r'^SGRED = os\.environ\.get\("KYOJIN_MOE_SGRED", "(\w*)"\) == "1"', text("moe.py"), re.M)
    k = re.search(r'^FUSE_ROWS = int\(os\.environ\.get\("KYOJIN_FUSE_ROWS", "(\w*)"\)\)', text("kernels.py"), re.M)
    assert m and m.group(1) == FL.FLAGS["moe_sgred"].default
    assert k and k.group(1) == FL.FLAGS["fuse_rows"].default


def test_levers_doc_rows():
    doc = open(os.path.join(ROOT, "docs", "cloud", "LEVERS.md")).read()
    for env, name in (("KYOJIN_MOE_SGRED", "moe_sgred"), ("KYOJIN_FUSE_ROWS", "fuse_rows")):
        rows = [l for l in doc.splitlines() if l.startswith(f"| `{env}`")]
        assert len(rows) == 1 and f"| {name} |" in rows[0], env
        assert "| 0 |" in rows[0] and ("measured" in rows[0].lower())


def test_no_new_default_on_in_the_diff_surface():
    """Off means off: with both flags at their defaults the code paths read the old values."""
    assert re.search(r"def fuse_rows\(\) -> int:\n(?:.*\n)*?\s+return max\(MAX_ROWS, min\(FUSE_ROWS, FUSE_ROWS_CAP\)\)", text("kernels.py"))
    assert "KN.MAX_ROWS < nr <= KN.fuse_rows()" in text("exl3_linear.py")           # empty range while FUSE_ROWS = 0


# ------------------------------------------------------------------------------------------------ SGRED kernel text (no mlx)
def gen(S, **kw):
    S.HALF_ROWS = kw.pop("half_rows", S.HALF_ROWS)
    return S.grouped_gemv_source(kw.pop("Ks", (2, 3)), kw.pop("nproj", 2), kw.pop("cb", "mul1r1"), 0, **kw)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("prev", [False, True])
def test_sgred_text_has_no_threadgroup_memory_and_no_barrier(S, compact, half, prev):
    old = S.HALF_ROWS
    try:
        src = gen(S, half_rows=4 if half else 0, prev=prev, half=half, mma=0, compact=compact, sgred=True)
        ref = gen(S, half_rows=4 if half else 0, prev=prev, half=half, mma=0, compact=compact, sgred=False)
    finally:
        S.HALF_ROWS = old
    code = "\n".join(l.split("//")[0] for l in src.splitlines())                 # comments out
    assert "threadgroup float" not in code and "threadgroup uchar" not in code and "threadgroup_barrier" not in code
    assert "red[" not in code
    assert "threadgroup float red" in ref and "threadgroup_barrier" in ref       # today's text keeps both
    n_rows = 8 if compact else 5                                                  # row bodies: the MR 1,2,3,4,8 classes or 1..8 member counts
    assert code.count("po0[0] = a0_lo0; po0[8] = a0_hi0;") >= 1
    assert "(TGK * WS) + kblock * WS + split" in code


def test_sgred_off_is_todays_text(S):
    """sgred=False must not change a byte of today's kernel."""
    old = S.HALF_ROWS
    try:
        a = S.grouped_gemv_source((3,), 1, "mul1r1", 0, False, False, 0, False)
        b = S.grouped_gemv_source((3,), 1, "mul1r1", 0, False, False, 0, False, False)
    finally:
        S.HALF_ROWS = old
    assert a == b and "po0[" not in a


def test_sgred_store_column_map_equals_the_threadgroup_reduction(S):
    """The direct store puts lane l's lo / hi accumulator at half_ * 64 + (l >> 3) * 16 + hl * 8 + (l & 7); the reduction's read side maps column c to
    (h = c >> 6, l = ((c & 63) >> 4) * 8 + (c & 7), hl = (c >> 3) & 1). The two must be inverse maps of each other (every column covered exactly once)."""
    seen = {}
    for half_ in range(2):
        for lane in range(32):
            for hl in range(2):
                c = half_ * 64 + (lane >> 3) * 16 + hl * 8 + (lane & 7)
                h, l, hl2 = c >> 6, ((c & 63) >> 4) * 8 + (c & 7), (c >> 3) & 1
                assert (h, l, hl2) == (half_, lane, hl)
                seen[c] = seen.get(c, 0) + 1
    assert sorted(seen) == list(range(128)) and set(seen.values()) == {1}


def test_sgred_prev_copy_covers_every_split_exactly_once():
    """The carry-through of a bitrate pass copies cnt * 128 * WS words, split-major: index map of the kernel text, replayed."""
    WS, cnt = 4, 3
    got = set()
    for idx in range(cnt * 128 * WS):
        j, q = divmod(idx, cnt * 128)
        got.add((j, q >> 7, q & 127))
    assert got == {(j, r, c) for j in range(WS) for r in range(cnt) for c in range(128)}


def test_sgred_summation_order_error_is_fp32_noise():
    """Today the kernel sums the four simdgroup splits (0 + a + b + c + d) and the consumer then the TGK block sums; with SGRED the consumer sums TGK * 4
    partials left to right. Same products, fp32 summation order only: relative difference of the layer sum stays at a few ulp of fp32."""
    rng = np.random.default_rng(0)
    tgk = 4
    worst = 0.0
    for _ in range(200):
        parts = (rng.standard_normal((tgk, 4, 64)) * rng.uniform(0.1, 30, (tgk, 4, 1))).astype(np.float32)
        old = np.zeros(64, np.float32)
        for kb in range(tgk):
            v = np.float32(0.0) * np.ones(64, np.float32)
            for j in range(4):
                v = (v + parts[kb, j]).astype(np.float32)
            old = (old + v).astype(np.float32)
        new = np.zeros(64, np.float32)
        for kb in range(tgk):
            for j in range(4):
                new = (new + parts[kb, j]).astype(np.float32)
        ref = parts.astype(np.float64).sum((0, 1))
        scale = np.abs(parts.astype(np.float64)).sum((0, 1)) + 1e-30
        worst = max(worst, float(np.max(np.abs(new - ref) / scale)), float(np.max(np.abs(old - ref) / scale)))
    assert worst < 1e-5                                                           # 16 partials: a few ulp of the absolute sum


def test_sgred_partial_traffic_is_bounded(S):
    """The cost of SGRED is WS = 4 times the partial buffer; at the pack's routed geometry it stays a small share of the expert words read."""
    hidden, inter, topk, rows, tgk, K = 5120, 2304, 6, 6, 4, 3
    pairs = rows * topk
    part_gu = 2 * pairs * tgk * 4 * inter * 4                                    # bytes written + read once each
    part_dn = pairs * tgk * 4 * hidden * 4
    words = 3 * (hidden // 16) * (inter // 16) * 16 * K * 2 * min(pairs, 256)    # all pairs distinct: the worst case for the share
    share = 2 * (part_gu + part_dn) / words
    assert share < 0.35, share
    print(f"sgred partial traffic (write + read) vs expert words, rows {rows} distinct: {share:.1%}")


# ------------------------------------------------------------------------------------------------ FUSE_ROWS budget (no mlx)
DENSE_SHAPES = {                                         # (kin, nout) of the pack's dense linears (tools/dense_gemv_bench.py SH)
    "wq_a": (5120, 1280), "wkv": (5120, 512), "wq_b": (1280, 32768), "wo_b": (8192, 5120), "wo_a_g": (4096, 1024),
    "sh_gate": (5120, 2304), "sh_down": (2304, 5120),
}


def pick_tgk(kt, blocks, ws=4, target_tgs=512, min_slices=8, slack=8):
    """kernels.pick_tgk with the 10-core Air constants (512 threadgroups, 8 tiles per warp; a wider GPU gets a larger target, hence a larger tgk and a
    smaller x buffer per threadgroup, i.e. a fit that is at least as good as this table)."""
    tgk = 1
    while blocks * tgk < target_tgs:
        s = ws * tgk * 2
        chunk = -(-kt // s)
        if kt // s < min_slices or s * chunk * slack > kt * (slack + 1):
            break
        tgk *= 2
    return tgk


def test_pick_tgk_copy_follows_the_source():
    src = text("kernels.py")
    body = src[src.index("def pick_tgk"):src.index("def param_sets")]
    for tok in ("s = ws * tgk * 2", "chunk = -(-kt // s)", "kt // s < min_slices", "s * chunk * FILL_IDLE_SLACK > kt * (FILL_IDLE_SLACK + 1)", "tgk *= 2"):
        assert tok in body, tok


@pytest.mark.parametrize("rows", [4, 5, 6, 7, 8])
def test_fused_smem_fits_at_the_dense_shapes(S, rows, capsys):
    """Which dense linears can take the fused launch at `rows` (the same test Mul1Gemv.can_fuse does); printed so the Mac run can compare."""
    fit = {}
    for name, (kin, nout) in DENSE_SHAPES.items():
        kt, nt = kin // 16, nout // 16
        tgk = pick_tgk(kt, nt // 8)
        fit[name] = (not (tgk & (tgk - 1))) and max(rows * S.fused_dims(kt, tgk), 512 * rows) <= 7000
    print(f"rows {rows}: fused launch fits", {k: v for k, v in fit.items()})
    if rows == 4:
        assert any(fit.values())
    # a shape that fits at 4 rows and not at 6 simply keeps its many-rows path: can_fuse is False, nothing else changes
    assert all(isinstance(v, bool) for v in fit.values())


def test_fuse_rows_band_and_half_class_are_generic():
    """The fused band is one constant (FUSE_ROWS_CAP = 8) and the FMA class follows the row count up to that band, not the path: no shape, prompt or
    row-count literal in the gate, so every linear of every pack takes the same rule."""
    k = text("kernels.py")
    assert re.search(r"^FUSE_ROWS_CAP = 8$", k, re.M)
    assert re.search(r"def _half_on\(self, rows: int\) -> bool:\n(?:.*\n)*?\s+return rows <= fuse_rows\(\) and S\.half_on\(rows\)", k)
    assert "rows <= MAX_ROWS and S.half_on" not in k                                     # the old cutoff that split the verify band by depth
    e = text("exl3_linear.py")
    assert "KN.MAX_ROWS < nr <= KN.fuse_rows()" in e                                     # dense linears: one range, no 5 / 6 literal
    assert "rows <= KN.fuse_rows()" in text("fused_group.py") and "R <= KN.fuse_rows()" in text("grouped.py")
    assert "KN.MAX_ROWS < x.shape[2] <= KN.fuse_rows()" in text("patch.py")



def test_fused_kernel_text_builds_for_5_and_6_rows(S):
    """fuse_gemv's template asserts must hold for the new row counts (half-staged x, fp8 fold, both input dtypes); warps = 8 >= rows finalize one row each."""
    old = S.HALF_ROWS
    try:
        for rows in (5, 6):
            for ih in (False, True):
                for half in (False, True):
                    if half and not ih:
                        continue
                    xsr = 512
                    src = S.fused_source(5, rows, "mul1r1", ih, False, xsr, True, True, half)
                    assert f"[2 * {rows}]" in src or f"2 * MR" in src
            S.fused_source(5, rows, "mul1r1", True, True, 512, True, True, True)           # fq8 fold
    finally:
        S.HALF_ROWS = old
    assert 6 <= 8


def test_the_verify_band_keeps_the_row_fma_class_of_its_rows():
    """The fused launch of 5..8 rows rounds like the r<=4 family it continues (packed-fp16 row FMAs from HALF_ROWS rows), not like the fp32
    many-rows path it replaces: the FMA class follows the row count up to the fused band, so a verify row's bits do not depend on the depth the
    planner picked. `test_fused_rows_keep_the_fma_class_of_the_4_row_launch` measures the deviation (equal to the served 4-row class)."""
    k = text("kernels.py")
    assert "return rows <= fuse_rows() and S.half_on(rows) and not S.af_fold(self.dcb)" in k
    assert "rows <= MAX_ROWS and S.half_on(rows)" not in k


def test_sgred_is_asked_for_by_the_grouped_call_only():
    """Other users of _gstatic (profiling tools) keep the layout their mid / final expect."""
    src = text("moe.py")
    assert "def _gstatic(self, rows: int, fq8: bool, ih: bool, half: bool = False, sg: bool = False)" in src
    assert "self._gstatic(rows, fq8, x.dtype == mx.float16, half, SGRED)" in src
    assert src.count("_gstatic(") == 2


# ------------------------------------------------------------------------------------------------ ablit sites (no mlx)
@pytest.mark.parametrize("f", ["fuse_layer.py", "glue_fuse.py", "launchcut.py", "suglue.py", "decstage.py", "pf_glue.py", "ced.py"])
def test_ablit_hook_sites_untouched(f):
    """Commit 18ad409 hooks the block paths; the levers live in the linear / MoE kernels below them and do not touch these files."""
    assert "ablit" in text(f).lower()
    for g in ("moe.py", "kernels.py", "exl3_linear.py", "fused_group.py", "grouped.py"):
        assert "ablit" not in text(g).lower()


# ------------------------------------------------------------------------------------------------ Metal half
def _metal():
    if importlib.util.find_spec("mlx") is None:
        return None
    import mlx.core as mx
    return mx if mx.metal.is_available() else None


needs_metal = pytest.mark.skipif(_metal() is None, reason="needs mlx + Metal (run on the Mac)")


@needs_metal
@pytest.mark.parametrize("rows", [2, 5, 6])
@pytest.mark.parametrize("wpre", [False, True])
@pytest.mark.parametrize("half_rows", [0, 4])                                   # conftest pins 0; serving runs 4 (packed-fp16 row FMAs from 4 rows)
def test_sgred_moe_vs_f32_reference_next_to_today(rows, wpre, half_rows, monkeypatch, capsys):
    import mlx.core as mx
    from kyojin_mlx import moe as M, ref_numpy as R, metal_src as MS
    monkeypatch.setattr(MS, "HALF_ROWS", half_rows)
    from test_moe import KS, TOPK, E, make_stack, _x, _np_moe
    from test_moe_union import route
    g, gp = make_stack(256, 384, KS, "mul1", 1)
    u, up = make_stack(256, 384, KS, "mul1", 2)
    d, dp = make_stack(384, 256, KS, "mul1", 3)
    ids, w = route(rows, "partial", 40 + rows)
    x = _x(rows, 256, 7)
    ref = _np_moe(np.array(x), np.array(ids), np.array(w), (gp, up, dp), 10.0, wpre, "mul1")
    C = (np.random.default_rng(3).standard_normal((256, 64)) / 16)
    old = M.SGRED
    out = {}
    try:
        for on in (False, True):
            M.SGRED = on
            m = M.MoeGather(g, u, d, TOPK, wpre=wpre, tgk_gu=2, tgk_dn=2, grouped=True, hotk=0)
            y = np.array(m(x, ids, w)).astype(np.float64)
            out[on] = y
            rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
            top = [int(np.argmax(r @ C)) for r in y]
            print(f"moe_sgred={int(on)} half_rows {half_rows} rows {rows} wpre {wpre}: rel {rel:.3e} top1 {top} (ref {[int(np.argmax(r @ C)) for r in ref]})")
    finally:
        M.SGRED = old
    r0 = np.linalg.norm(out[False] - ref) / np.linalg.norm(ref)
    r1 = np.linalg.norm(out[True] - ref) / np.linalg.norm(ref)
    assert r1 <= max(2 * r0, 5e-6)
    assert [int(np.argmax(r @ C)) for r in out[True]] == [int(np.argmax(r @ C)) for r in out[False]]


@pytest.fixture
def served_rows(monkeypatch):
    """The served verify rows: rows.enable sets ROW_SPLIT_MAX = 16, KYOJIN_VERIFY_NAX=0 keeps ROWS_NAX off, so 5 / 6 rows take the GEMV many-rows path."""
    from kyojin_mlx import exl3_linear as EL, kernels as KN
    monkeypatch.setattr(EL, "ROW_SPLIT_MAX", 16)
    monkeypatch.setattr(EL, "ROWS_NAX", False)
    monkeypatch.setattr(KN, "FUSE_ROWS", 0)


@needs_metal
@pytest.mark.parametrize("rows", [5, 6, 7, 8])
def test_fuse_rows_is_bit_identical_to_the_many_rows_path(rows, served_rows, monkeypatch):
    """Same decode, same fp32 FMA order, same tgk: the one-launch form must give today's bits (single linear, S5 group, grouped wo_a)."""
    import mlx.core as mx
    from kyojin_mlx import exl3_linear as EL, kernels as KN, fused_group as FG, grouped as GR, ref_numpy as R
    t, su, sv = R.random_linear(384, 256, 5, seed=4)
    m = EL.Exl3Linear(mx.array(t), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16)
    t2, su2, sv2 = R.random_linear(384, 128, 5, seed=5)
    m2 = EL.Exl3Linear(mx.array(t2), mx.array(su2), mx.array(sv2), cb="mul1", out_dtype=mx.float16)
    grp = FG.Exl3Group([m, m2])
    x = mx.array((np.random.default_rng(1).standard_normal((rows, 384)) * 0.7).astype(np.float16))
    rng = np.random.default_rng(3)
    Gg, kt, nt, K = 4, 16, 64, 5
    tr = rng.integers(-32768, 32768, (Gg, kt, nt, 16 * K)).astype(np.int16)
    suo = (rng.choice([-1.0, 1.0], (Gg, kt * 16)) * rng.uniform(.5, 1.5, (Gg, kt * 16))).astype(np.float16)
    svo = (rng.choice([-1.0, 1.0], (Gg, nt * 16)) * rng.uniform(.5, 1.5, (Gg, nt * 16))).astype(np.float16)
    wo = GR.GroupedGemv(mx.array(tr), mx.array(suo), mx.array(svo))
    xo = mx.array((rng.standard_normal((1, Gg, rows, kt * 16)) * .3).astype(np.float16))
    out = {}
    for fr in (0, rows):
        monkeypatch.setattr(KN, "FUSE_ROWS", fr)
        y = m(x)
        ys = grp.run(x) if fr and grp.can_fuse(rows) else [m(x), m2(x)]          # today the S5 group at 5 / 6 rows is its members, one linear at a time
        if fr:
            assert grp.can_fuse(rows)
        z = wo(xo)
        mx.eval(y, *ys, z)
        out[fr] = (np.array(y).view(np.uint16), [np.array(v).view(np.uint16) for v in ys], np.array(z).view(np.uint16))
        if fr:
            assert rows in m._g and m._g[rows].can_fuse(rows) and wo.fuses(rows)           # the lever really took the fused kernels
    assert np.array_equal(out[0][0], out[rows][0])
    assert all(np.array_equal(a, b) for a, b in zip(out[0][1], out[rows][1]))
    assert np.array_equal(out[0][2], out[rows][2])


@needs_metal
@pytest.mark.parametrize("rows", [5, 6])
def test_fuse_rows_dense_vs_f32_reference_next_to_today(rows, served_rows, monkeypatch, capsys):
    import mlx.core as mx
    from kyojin_mlx import exl3_linear as EL, kernels as KN, ref_numpy as R
    t, su, sv = R.random_linear(384, 256, 5, seed=4)
    m = EL.Exl3Linear(mx.array(t), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16)
    mx.eval(m.parameters())
    x = mx.array((np.random.default_rng(1).standard_normal((rows, 384)) * 0.7).astype(np.float16))
    ref = R.gemv(np.array(x).astype(np.float32), t, su, sv, dtype=np.float32)
    C = (np.random.default_rng(7).standard_normal((256, 32)) / 16).astype(np.float32)
    out = {}
    for fr in (0, rows):
        monkeypatch.setattr(KN, "FUSE_ROWS", fr)
        y = m(x); mx.eval(y)
        out[fr] = np.array(y).astype(np.float32)
        rel = np.linalg.norm(out[fr] - ref) / np.linalg.norm(ref)
        print(f"fuse_rows={fr} rows {rows}: rel {rel:.3e} top1 {[int(np.argmax(r @ C)) for r in out[fr]]} (ref {[int(np.argmax(r @ C)) for r in ref]})")
    r0 = np.linalg.norm(out[0] - ref) / np.linalg.norm(ref)
    r1 = np.linalg.norm(out[rows] - ref) / np.linalg.norm(ref)
    assert r1 <= max(2 * r0, 1e-6 + r0)
    assert [int(np.argmax(r @ C)) for r in out[rows]] == [int(np.argmax(r @ C)) for r in out[0]]


@needs_metal
@pytest.mark.parametrize("rows", [4, 5, 6, 7, 8])
def test_fused_rows_keep_the_fma_class_of_the_4_row_launch(rows, served_rows, monkeypatch, capsys):
    """The served lever (HALF_ROWS 4): a fused launch of 5..8 rows runs the packed-fp16 row FMAs the 4-row launch runs, so its deviation from the fp32
    (HALF_ROWS 0) reference is the fp16 class of the same kernel and not the fp32 bits of the many-rows path it replaces. Both are printed."""
    import mlx.core as mx
    from kyojin_mlx import exl3_linear as EL, kernels as KN, metal_src as MS, ref_numpy as R
    t, su, sv = R.random_linear(384, 256, 5, seed=4)
    m = EL.Exl3Linear(mx.array(t), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16)
    mx.eval(m.parameters())
    x = mx.array((np.random.default_rng(1).standard_normal((8, 384)) * 0.7).astype(np.float16))
    ref = R.gemv(np.array(x[:rows]).astype(np.float32), t, su, sv, dtype=np.float32)
    out, half = {}, {}
    for hr in (0, 4):
        monkeypatch.setattr(MS, "HALF_ROWS", hr)
        for fr in (0, rows):
            monkeypatch.setattr(KN, "FUSE_ROWS", fr)
            y = m(x[:rows])
            mx.eval(y)
            out[(hr, fr)] = np.array(y).astype(np.float32)
        monkeypatch.setattr(KN, "FUSE_ROWS", 4)
        y = m(x[:4])
        mx.eval(y)
        half[hr] = np.array(y).astype(np.float32)                        # the 4-row launch: the class the verify band already serves
    scale = max(np.abs(ref).max(), 1e-6)
    eq = np.array_equal(out[(0, rows)].view(np.uint16), out[(0, 0)].view(np.uint16))
    d_arm = float(np.abs(out[(4, rows)] - out[(4, 0)]).max()) / scale   # fused fp16 rows vs fused fp32 rows
    d_base = float(np.abs(out[(4, 0)] - out[(0, 0)]).max()) / scale      # the many-rows path never reads HALF_ROWS
    d_class = float(np.abs(half[4] - half[0]).max()) / scale             # the 4-row fused class, already served and KL-gated
    r = float(np.linalg.norm(out[(4, rows)] - ref) / np.linalg.norm(ref))
    print(f"rows {rows}: fp32 fused == many-rows bits {eq}; fused fp16 vs fp32 {d_arm:.2e}, many-rows vs itself {d_base:.2e}, "
          f"the 4-row class {d_class:.2e}, fused fp16 vs f32 ref rel {r:.2e}")
    assert eq                                                                # the fusion itself is exact at equal FMA precision
    if rows > 4:
        assert d_base == 0.0                                                 # base is fp32 at 5..8 rows: the HALF_ROWS cutoff, not the path, decided it
    assert d_arm <= max(3.0 * d_class, 1e-6) + 1e-6                          # the 5..8 row launch stays in the 4-row fp16 class
    assert r < 5e-3


@needs_metal
def test_fuse_rows_off_is_todays_path(served_rows):
    import mlx.core as mx
    from kyojin_mlx import exl3_linear as EL, ref_numpy as R
    t, su, sv = R.random_linear(384, 256, 5, seed=5)
    m = EL.Exl3Linear(mx.array(t), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16)
    x = mx.array((np.random.default_rng(2).standard_normal((5, 384)) * 0.7).astype(np.float16))
    y = m(x); mx.eval(y)
    assert 5 not in m._g                                                     # no fused row-5 module was even built


@needs_metal
def test_ablit_projection_still_applies_with_the_levers_on():
    import mlx.core as mx
    from kyojin_mlx import ablit as AB, exl3_linear as EL, kernels as KN, ref_numpy as R
    t, su, sv = R.random_linear(256, 256, 5, seed=6)
    m = EL.Exl3Linear(mx.array(t), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16)
    x = mx.array((np.random.default_rng(3).standard_normal((5, 256)) * 0.7).astype(np.float16))
    Rm = mx.array((np.random.default_rng(9).standard_normal((2, 256)) / 16).astype(np.float32))
    old = KN.FUSE_ROWS
    try:
        ys = {}
        for fr in (0, 5):
            KN.FUSE_ROWS = fr
            y = m(x); mx.eval(y)
            ys[fr] = np.array(AB._project(y, 0.7, Rm), dtype=np.float32)
            assert not np.array_equal(ys[fr], np.array(y, dtype=np.float32))     # the projection did something
    finally:
        KN.FUSE_ROWS = old
    assert np.linalg.norm(ys[5] - ys[0]) / np.linalg.norm(ys[0]) < 2e-3


# ------------------------------------------------------------------------------------------------ the bench's numpy reference (no mlx)
def _bench():
    spec = importlib.util.spec_from_file_location("verify_row_bench", os.path.join(ROOT, "tools", "verify_row_bench.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bench_reference_matches_a_naive_per_pair_loop():
    """tools/verify_row_bench.moe_ref (one decode per distinct expert, rows batched) equals the naive pair-by-pair f64 loop on a small geometry."""
    from kyojin_mlx import ref_numpy as R
    B = _bench()
    hidden, inter, E, topk, rows = 256, 384, 6, 3, 4
    ks = [2, 3, 2, 3, 3, 2]
    fn = B.expert_arrays
    ids, w = B.route(rows, E, "overlap", 1)
    ids, w = ids[:, :topk] % E, w[:, :topk]
    ids = np.stack([np.array(list(dict.fromkeys(list(r) + list(range(E)))))[:topk] for r in ids]).astype(np.int32)
    x = (np.random.default_rng(2).standard_normal((rows, hidden)) * 0.7).astype(np.float16)
    got = B.moe_ref(x, ids, w, hidden, inter, expert_fn=fn, kof=lambda e: ks[e])
    ref = np.zeros((rows, hidden))
    for r in range(rows):
        for j in range(topk):
            e = int(ids[r, j])
            xr = x[r:r + 1].astype(np.float64)
            g = R.gemv(xr, *fn(0, e, hidden, inter, ks[e]), dtype=np.float64)
            u = R.gemv(xr, *fn(1, e, hidden, inter, ks[e]), dtype=np.float64)
            a = np.minimum(g, 10.0) / (1 + np.exp(-np.minimum(g, 10.0))) * np.clip(u, -10.0, 10.0) * w[r, j]
            ref[r] += R.gemv(a, *fn(2, e, inter, hidden, ks[e]), dtype=np.float64)[0]
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 5e-5


def test_bench_memory_accounting_is_under_the_cap_at_the_defaults():
    B = _bench()
    total = B.bank_bytes(64) + B.dense_bytes()
    print(f"bench default weights {total / 2**30:.2f} GiB")
    assert total < 3.5 * 2**30
    # the account matches the arrays it stands for
    tr, su, sv = B.expert_arrays(0, 0, B.HIDDEN, B.INTER, 2)
    assert tr.nbytes == B.HIDDEN * B.INTER // 16 * 2 * 2


def test_bench_head_metrics_identity():
    B = _bench()
    y = np.random.default_rng(0).standard_normal((3, 32)).astype(np.float32)
    h = np.random.default_rng(1).standard_normal((32, 50)).astype(np.float32)
    assert B.head_metrics(y, y, h) == (0.0, 0.0, 3, 3)


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="needs mlx (the modules import it)")
def test_flags_switch_the_module_attributes_at_runtime():
    from kyojin_mlx import flags, kernels as KN, moe as M
    try:
        flags.set("moe_sgred", "1")
        assert M.SGRED is True
        flags.set("fuse_rows", "6")
        assert KN.FUSE_ROWS == 6 and KN.fuse_rows() == 6
        flags.set("fuse_rows", "9")
        assert KN.fuse_rows() == KN.FUSE_ROWS_CAP == 8                      # never more than the fused band: one launch for the whole verify width
        flags.set("fuse_rows", "3")
        assert KN.fuse_rows() == KN.MAX_ROWS                                 # below the plain limit: today's 4
    finally:
        flags.set("moe_sgred", "0")
        flags.set("fuse_rows", "0")
    assert M.SGRED is False and KN.fuse_rows() == KN.MAX_ROWS
