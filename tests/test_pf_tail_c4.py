"""pf_tail_c4, CPU-only (no Metal kernel runs here; the Metal kernels are timed by tools/ccedtail_bench.py on the Mac).

1. The gate (`patch.tail_band_takes`) and the path table (`patch.pf_min_rows`): for every (experts, top-k, GPU cores, NAX) which path serves a band of `rows`,
   how many launches per layer, and that the new flag never takes a call at or above the S3 floor and never a decode / verify row count.
2. The arithmetic contract of grouping, bit for bit on small synthetic tensors: a numpy model of the routed pair body (the per-pair dot product in a fixed
   k order) run chunk by chunk (the c4 path) and grouped by expert over the whole band (one call) gives the same bits when every pair takes the fp32 body
   (HALF_ROWS=0). It is a MODEL of the dispatcher's contract (class by member count from metal_src.group_classes, half body by metal_src.half_on), not proof of
   what the Metal kernels do. It covers full chunks only: the remainder chunk of rows % step != 0 rows runs other kernels (1 row: the per-pair gather, k-split 4 / 2;
   2-3 rows: the per-pair split-K, rows_ugrouped), `test_k_split_of_chunks_and_wide_call` pins the k-splits on the real geometry.
3. Where the contract breaks: with packed-fp16 rows (HALF_ROWS=4, the served default) the body of a pair depends on how many rows of the CALL share its
   expert, so grouping the band changes the body of some pairs. The test pins that, so nobody labels the lever exact at HALF_ROWS=4.
"""
import numpy as np
import pytest

from kyojin_mlx import flags as FL, metal_src as S, moe as M, patch as P
from kyojin_mlx.pld import MAX_DRAFT


# ---- 1. gate and path table ------------------------------------------------------------------------------------------------------------------

class FakeMoe:
    def __init__(self, experts, topk, grouped=True, hotk=0):
        self.g = type("G", (), {"experts": experts})()
        self.topk, self.grouped, self.hotk = topk, grouped, hotk


def path(rows, moe, *, step=4, c4=True, grp=False, pairs=1024):
    """Which path `patch._moe_rows` takes, mirroring its order of checks."""
    if rows <= step:
        return "decode", 1
    if grp and moe.grouped and not moe.hotk and rows * moe.topk <= pairs:
        return "pf_tail_grp", 1
    floor = P.pf_min_rows(moe)
    if c4 and P.tail_band_takes(rows, moe.topk, step, floor, bool(moe.grouped), bool(moe.hotk), pairs):
        return "pf_tail_c4", 1
    if rows >= floor:
        return "S3", 1
    return "chunks", -(-rows // step)


@pytest.fixture
def cfg(monkeypatch):
    def set_(nax, cores_ge32, cross=None, row_split=16):
        from kyojin_mlx import moe_prefill as MP
        monkeypatch.setattr(MP, "fused_default", lambda: nax)
        monkeypatch.setattr(P, "PF_MIN_ROWS", cross)
        monkeypatch.setattr(P, "PF_CROSS_NAX", 1.75 if cores_ge32 else 2.5)
        monkeypatch.setattr(P.EL, "ROW_SPLIT_MAX", row_split)
    return set_


TABLE = [  # experts, top-k, nax, >=32 cores, floor
    (384, 6, True, True, 112), (384, 6, True, False, 160), (384, 6, False, True, 896), (384, 6, False, False, 896),
    (256, 8, True, True, 56), (128, 8, False, True, 224), (64, 6, True, True, 18),
]


@pytest.mark.parametrize("E,k,nax,big,floor", TABLE)
def test_floor_table(cfg, E, k, nax, big, floor):
    cfg(nax, big)
    assert P.pf_min_rows(FakeMoe(E, k)) == max(floor, 17)       # never below the multi-user row split + 1


@pytest.mark.parametrize("E,k,nax,big,floor", TABLE)
@pytest.mark.parametrize("rows", [1, 4, 5, 8, 9, 17, 37, 56, 111, 112, 113, 128, 160, 170, 171, 896])
def test_new_flag_engages_only_below_the_floor(cfg, E, k, nax, big, floor, rows):
    cfg(nax, big)
    moe = FakeMoe(E, k)
    fl = P.pf_min_rows(moe)
    p, launches = path(rows, moe)
    base, base_launches = path(rows, moe, c4=False)
    if p == "pf_tail_c4":
        assert 8 < rows < fl and rows * k <= 1024 and base == "chunks" and launches == 1 < base_launches
    else:
        assert p == base                                          # flag on changes nothing outside the band
    if rows >= fl:
        assert p == "S3"
    if rows <= max(4, 16, MAX_DRAFT + 1):
        assert p == base                                          # decode, verify, multi-user rows


@pytest.mark.parametrize("split", [16, 24, 31])
def test_verify_and_multiuser_rows_are_never_taken(cfg, split):
    """enable_mtp raises ROW_SPLIT_MAX to mtp_rows (24 served): every row count up to it keeps the chunk path, flag on or off."""
    cfg(True, True, row_split=split)
    moe = FakeMoe(384, 6)
    for rows in range(1, split + 1):
        assert path(rows, moe) == path(rows, moe, c4=False)
    assert path(split + 1, moe) == ("pf_tail_c4", 1)


def test_served_box_band(cfg):
    """DS4.1-class pack (384 experts, top-6) on a >= 32-core NAX GPU, ROW_SPLIT_MAX 24: floor 112. The 128-row CED tail is S3 already (not taken); the lever
    DOES take prompt calls of 25..111 rows (a short suffix after a prefix hit, a small last chunk) when the flag is on."""
    cfg(True, True, row_split=24)
    moe = FakeMoe(384, 6)
    assert [r for r in range(1, 300) if path(r, moe)[0] == "pf_tail_c4"] == list(range(25, 112))
    assert path(128, moe, c4=True) == ("S3", 1) == path(128, moe, c4=False)


def test_engages_when_s3_is_off_or_the_floor_is_above_the_band(cfg):
    cfg(False, True)                                              # no NAX path: floor 896
    assert path(128, FakeMoe(384, 6)) == ("pf_tail_c4", 1) and path(128, FakeMoe(384, 6), c4=False) == ("chunks", 32)
    cfg(True, False)                                              # NAX under 32 cores: 2.5 * 384 / 6 = 160
    assert path(128, FakeMoe(384, 6)) == ("pf_tail_c4", 1)
    assert path(37, FakeMoe(384, 6)) == ("pf_tail_c4", 1) and path(37, FakeMoe(384, 6), c4=False) == ("chunks", 10)


def test_nax_grouped_mode_has_hotk_and_is_refused(cfg):
    """KYOJIN_ROWS_GROUPED=nax sets hotk = 4: the chunks run the hot path (_call_hot), the gate refuses the model."""
    cfg(False, True)
    assert path(128, FakeMoe(384, 6, hotk=4))[0] == "chunks"


def test_not_grouped_or_hotk_model_never_engages(cfg):
    cfg(False, True)
    assert path(128, FakeMoe(384, 6, grouped=False))[0] == "chunks"
    assert path(128, FakeMoe(384, 6, hotk=4))[0] == "chunks"


def test_multiuser_step_is_not_four(cfg):
    """rows.enable sets the chunk step to MAX_PAIRS // topk (42 at top-6): the chunk path is then 4 launches for 128 rows, and rows <= 42 never reach the band."""
    cfg(False, True)
    moe = FakeMoe(384, 6)
    step = M.MAX_PAIRS // 6
    assert path(128, moe, step=step, c4=False) == ("chunks", 4)
    assert path(42, moe, step=step)[0] == "decode"
    assert path(43, moe, step=step) == ("pf_tail_c4", 1)


def test_flag_default_and_runtime_attr(monkeypatch):
    f = FL.FLAGS["pf_tail_c4"]
    assert f.default == "0" and f.env == "KYOJIN_PF_TAIL_C4" and f.runtime and f.kind == "bool"
    assert M.TAIL_C4 is False
    old = M.TAIL_C4
    try:
        FL.set("pf_tail_c4", "1")
        assert M.TAIL_C4 is True
        FL.set("pf_tail_c4", "0")
        assert M.TAIL_C4 is False
    finally:
        M.TAIL_C4 = old


# ---- 2/3. the arithmetic contract of grouping, numpy ------------------------------------------------------------------------------------------

def route(rows, E, k, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(E, k, replace=False) for _ in range(rows)]).astype(np.int64)


def body_of_pairs(ids, step, half_rows):
    """Per pair (flattened row-major): True = packed-fp16 body. A call of `r` rows is half-staged when half_on(r); a pair whose group in that call has
    `m` members runs in the class of the highest lowest-k <= m, whose rows-per-pass decides the fp16 FMA (metal_src.group_classes, half_on)."""
    S_half = S.HALF_ROWS
    S.HALF_ROWS = half_rows
    try:
        classes = S.group_classes()
        out = np.zeros(ids.size, dtype=bool)
        rows = ids.shape[0]
        for lo in range(0, rows, step):
            blk = ids[lo:lo + step]
            call_half = S.half_on(blk.shape[0])
            flat = blk.reshape(-1)
            for e in np.unique(flat):
                sel = np.nonzero(flat == e)[0]
                cls = max((c for c in classes if len(sel) >= c[0]), key=lambda c: c[0])
                out[lo * ids.shape[1] + sel] = call_half and S.half_on(cls[1])
        return out
    finally:
        S.HALF_ROWS = S_half


def model_pair(x, w, half):
    """One pair's dot product in a fixed k order: fp32 sequential FMAs, or the fp16 body (x and w in fp16, 8-tile partial sums in fp16, flushed to fp32)."""
    if not half:
        acc = np.float32(0)
        for a, b in zip(x.astype(np.float32), w.astype(np.float32)):
            acc = np.float32(acc + np.float32(a * b))
        return acc
    acc = np.float32(0)
    xh, wh = x.astype(np.float16), w.astype(np.float16)
    for t in range(0, len(x), 8):
        p = np.float16(0)
        for a, b in zip(xh[t:t + 8], wh[t:t + 8]):
            p = np.float16(p + np.float16(a * b))
        acc = np.float32(acc + np.float32(p))
    return acc


def run(ids, X, W, bodies, order):
    """Out[row, slot] = pair dot, visiting pairs in `order` (c4: chunk by chunk, expert by expert; grouped: expert by expert over the band)."""
    out = np.zeros(ids.shape, dtype=np.float32)
    k = ids.shape[1]
    for p in order:
        r, s = divmod(int(p), k)
        out[r, s] = model_pair(X[r], W[ids[r, s]], bodies[p])
    return out


def orders(ids, step):
    k = ids.shape[1]
    flat = ids.reshape(-1)
    c4 = []
    for lo in range(0, ids.shape[0], step):
        idx = np.arange(lo * k, min(ids.shape[0], lo + step) * k)
        c4 += list(idx[np.argsort(flat[idx], kind="stable")])
    grp = list(np.argsort(flat, kind="stable"))
    return c4, grp


@pytest.mark.parametrize("rows", [8, 36, 128, 168])
def test_grouped_equals_chunks_bit_for_bit_fp32_rows(rows):
    """Model of the contract, full chunks only (rows a multiple of the step)."""
    E, k, kin = 24, 6, 32
    rng = np.random.default_rng(rows)
    ids = route(rows, E, k, seed=rows)
    X = rng.standard_normal((rows, kin)).astype(np.float16)
    W = rng.standard_normal((E, kin)).astype(np.float16)
    bodies = body_of_pairs(ids, 4, 0)
    assert not bodies.any()
    c4, grp = orders(ids, 4)
    a, b = run(ids, X, W, bodies, c4), run(ids, X, W, body_of_pairs(ids, rows, 0), grp)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("rows", [5, 37, 128, 170])
def test_half_rows_body_depends_on_the_call(rows):
    """HALF_ROWS=4: pairs whose body flips between the 4-row chunks and the one wide call exist; the numbers (CPU arithmetic on random top-6 over 384 experts)
    are printed by the report test below. Here: the flip set is not empty for the wide call and the values of those pairs differ."""
    E, k, kin = 384, 6, 64
    rng = np.random.default_rng(rows + 1)
    ids = route(rows, E, k, seed=rows + 1)
    c4_body = body_of_pairs(ids, 4, 4)
    wide_body = body_of_pairs(ids, rows, 4)
    flip = c4_body != wide_body
    if rows >= 37:
        assert flip.any()
    X = (rng.standard_normal((rows, kin)) * 2).astype(np.float16)
    W = rng.standard_normal((E, kin)).astype(np.float16)
    c4, grp = orders(ids, 4)
    a, b = run(ids, X, W, c4_body, c4), run(ids, X, W, wide_body, grp)
    diff = (a.view(np.uint32) != b.view(np.uint32)).reshape(-1)
    assert not diff[~flip].any()                                   # pairs with the same body are bit-equal whatever the grouping
    if flip.any():
        assert diff[flip].any()                                    # a flipped body changes bits at the fp16 class


def test_flip_fraction_report(capsys):
    """Numbers for the report: fraction of pairs whose body differs, 384 experts top-6 uniform routing (a bound: real routing is more skewed)."""
    ids = route(128, 384, 6, seed=0)
    f = float((body_of_pairs(ids, 4, 4) != body_of_pairs(ids, 128, 4)).mean())
    with capsys.disabled():
        print(f"\npf_tail_c4 CPU arithmetic: 128 rows x top-6 over 384 experts, uniform routing: {100 * f:.1f} % of pairs change body at HALF_ROWS=4; 0 % at HALF_ROWS=0")
    assert 0.0 < f < 1.0


def test_k_split_of_chunks_and_wide_call():
    """Real DS4.1 geometry (H 5120, I 2304, top-6), 8 experts: _params launches no kernel, so the k-splits the two paths pick are readable on CPU.
    Full chunks (>= 3 rows) and the wide call all pick tgk (1, 1), so a band of a multiple of the step has equal k-splits. The remainder chunk does not:
    1 row picks (4, 2) on the per-pair gather, 2 rows (2, 1) on the per-pair split-K (rows_ugrouped) - fp32 partial order differs from the wide call."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    from moe_fuse_bench import make_stack
    ks = [3, 2, 3, 2, 2, 3, 2, 2]
    g, u, d = make_stack(5120, 2304, ks, 1), make_stack(5120, 2304, ks, 2), make_stack(2304, 5120, ks, 3)
    m = M.MoeGather(g, u, d, 6, limit=10.0, wpre=True, fq=False, grouped=True, hotk=0)
    kt = {r: (m._params(r)["tgu"], m._params(r)["tdn"]) for r in (1, 2, 3, 4, 17, 37, 42, 128, 170)}
    assert {kt[r] for r in (3, 4, 17, 37, 42, 128, 170)} == {(1, 1)}
    assert kt[1] == (4, 2) and kt[2] == (2, 1)
