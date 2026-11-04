"""The device-aware threadgroup fill of the 1-row decode GEMV (kernels.pick_tgk / moe.pick_tgk / tuning.fill_tgs).

The fill only chooses the number of k-splits (threadgroups) of a launch: the partials are still summed in k order, so a linear's output rounds at 1e-7
relative against the old k-split. These tests pin the heuristic (pure host arithmetic: they run on any machine) and, on a Mac, that a filled linear
still gives the reference result and still takes the one-launch fused path.
"""
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import kernels as KN, metal_src as S, moe as M, tuning as TN

OLD = (512, 8)          # target threadgroups / k-tiles per warp before the fill (KYOJIN_FILL=0)
SHAPES = [("wq_a", 5120, 1280), ("wkv", 5120, 512), ("wq_b", 1280, 32768), ("wo_b", 8192, 5120), ("wo_a_g", 4096, 1024),
          ("sh_gate", 5120, 2304), ("sh_down", 2304, 5120), ("idx_q", 1280, 4096), ("heads", 32768, 8192)]


def _old_pick(kt, blocks, ws=4):
    tgk = 1
    while blocks * tgk < OLD[0] and kt // (ws * tgk * 2) >= OLD[1]:
        tgk *= 2
    return tgk


@pytest.fixture
def cores(monkeypatch):
    """The device the heuristic sees: the core count (and a chip() that does not depend on the MLX version)."""
    def set_(n):
        TN._chip = None
        monkeypatch.setattr(TN, "gpu_cores", lambda: n)
        monkeypatch.setattr(TN, "chip", lambda: dict(mma_max=8, nax_ks_tgs=128, nax_ks_few=32, nax_ks_weights=16_000_000, cores=n))
        return n
    return set_


def test_fill_tgs_is_per_core(cores):
    """8 threadgroups per GPU core, never below the 512 the Air measured; the tile floor falls with the core count (8 on the Air)."""
    for n, tgs, tiles in ((10, 512, 8), (20, 512, 4), (40, 512, 2), (64, 512, 2), (80, 640, 2)):
        cores(n)
        assert TN.fill_tgs() == max(512, n * TN.FILL_PER_CORE) == tgs
        assert TN.fill_min_tiles() == tiles


def test_fill_off_gives_the_old_constants(monkeypatch):
    """KYOJIN_FILL=0 (TN.FILL False): the helpers hand the pick_tgk loop the 512 threadgroups / 8 tiles it was measured with on the Air."""
    monkeypatch.setattr(TN, "FILL", False)
    assert TN.fill_tgs() == OLD[0] and TN.fill_min_tiles() == OLD[1]


@pytest.mark.parametrize("n", [10, 64])
def test_pick_tgk_properties(cores, n):
    """Power of two, at least one k-split, at most kt / 4 (each of the 4 warps of a threadgroup keeps a 16-element k tile), and the loop's stop:
    either the threadgroup target is reached or the next split would leave a warp fewer than `min_tiles` tiles."""
    cores(n)
    target, mint = TN.fill_tgs(), TN.fill_min_tiles()
    for name, kin, nout in SHAPES:
        kt, blocks = kin // 16, nout // 128
        tgk = KN.pick_tgk(kt, blocks)
        assert tgk >= 1 and tgk & (tgk - 1) == 0, (name, tgk)
        assert tgk * 4 <= kt, (name, tgk, kt)
        assert blocks * tgk >= target or kt // (4 * tgk * 2) < mint or kt % (4 * tgk * 2), (name, tgk, kt)


def test_fill_never_lowers_the_threadgroups(cores):
    """On every shape the fill picks at least the old k-split (a bigger GPU never gets fewer threadgroups than the Air's constants gave it)."""
    for n in (10, 64):
        cores(n)
        for name, kin, nout in SHAPES:
            kt, blocks = kin // 16, nout // 128
            assert KN.pick_tgk(kt, blocks) >= _old_pick(kt, blocks), (name, n)


def test_fill_raises_the_ds41_small_linears(cores):
    """The small DS4.1 projections (the ones a decode step chains) fill a 64-core GPU 2-8x; the ones that are already full keep their split, and no
    k-split is ever left without tiles (kt is a multiple of ws * tgk)."""
    cores(64)
    want = {"wq_a": (8, 16), "wkv": (8, 16), "wo_a_g": (8, 32), "sh_gate": (8, 16), "sh_down": (4, 8), "idx_q": (2, 4),
            "wq_b": (2, 2), "wo_b": (16, 16)}
    for name, kin, nout in SHAPES:
        kt, blocks = kin // 16, nout // 128
        old, new = _old_pick(kt, blocks), KN.pick_tgk(kt, blocks)
        assert new >= old, (name, old, new)
        assert -(-kt // (4 * new)) >= min(2, kt // (4 * new)), (name, new, kt)      # tiles per warp
        if name in want:
            assert (old, new) == want[name], (name, old, new, blocks)


def test_air_is_bit_identical_on_every_shape(cores):
    """The Air's 10 cores reproduce the old loop for every (kt, column blocks) a linear can have (kt a multiple of 8 up to k=65536, 1..4096 blocks):
    the fill only ever adds threadgroups on a GPU wider than the one the constants were measured on."""
    cores(10)
    for kt in range(8, 4097, 8):
        for blocks in (1, 2, 3, 4, 5, 8, 10, 18, 24, 32, 40, 64, 128, 256, 1024, 4096):
            assert KN.pick_tgk(kt, blocks) == _old_pick(kt, blocks), (kt, blocks)


def test_air_picks_the_measured_constants(cores):
    """On the Air's 10 cores the fill is a no-op: every shape and the routed MoE keep the tgk the constants were measured with (so the Air's bits and
    timings do not move)."""
    cores(10)
    for name, kin, nout in SHAPES:
        kt, blocks = kin // 16, nout // 128
        assert KN.pick_tgk(kt, blocks) == _old_pick(kt, blocks), name
    assert TN.fill_tgs() == 512 and TN.fill_min_tiles() == 8


def test_moe_is_already_filled(cores):
    """The routed MoE is not part of the fill: its k-split already targets 480 threadgroups (7.5 per core at 64 cores, 13 at the DS4.1 expert shapes),
    so it keeps its own pick and its 8-tile floor."""
    cores(64)
    for kt, blocks, pairs in ((320, 18, 12), (144, 40, 6)):
        tgk = M.pick_tgk(kt, blocks, pairs)
        assert tgk == M.pick_tgk(kt, blocks, pairs, 480)
        assert blocks * pairs * tgk >= 480, (kt, blocks, pairs)


def test_linear_takes_the_filled_split(cores):
    """An EXL3 linear built at a DS4.1 shape picks the fill's k-split and keeps the one-launch fused path (the launch count does not move)."""
    import numpy as np
    from kyojin_mlx import exl3_linear as EL
    cores(64)
    rng = np.random.default_rng(0)
    kin, nout, K = 5120, 512, 4
    t = mx.array(rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16))
    f16 = lambda n: mx.array((rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)).astype(np.float16))
    lin = EL.Exl3Linear(t, f16(kin), f16(nout), cb="mul1")
    g = lin._g[1]
    assert g.tgk == KN.pick_tgk(kin // 16, nout // 16 // 8) == 16
    assert g.can_fuse(1) and g.blocks * g.tgk == 64


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_filled_tgk_matches_the_reference(cores):
    """The same linear at the old and the filled k-split: both on the fp32 reference within 1e-4, the filled split matches the old one to 1e-3, and the
    one-launch fused kernel (rot + GEMV + finalize, which reads the TGK partials in k order) is the same bits as the three-launch path at that split."""
    import numpy as np
    from kyojin_mlx import ref_numpy as R
    cores(64)
    rng = np.random.default_rng(3)
    kin, nout, K = 4096, 1024, 4
    t = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    su = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    sv = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    x = rng.standard_normal((1, kin)).astype(np.float32)
    ref = R.gemv(x, t, su, sv)
    old_t, fill_t = _old_pick(kin // 16, nout // 128), KN.pick_tgk(kin // 16, nout // 128)
    assert (old_t, fill_t) == (8, 32), (old_t, fill_t)
    ys = {}
    for tgk in (old_t, fill_t):
        lin = KN.Mul1Gemv(t, su, sv, rows=1, tgk=tgk)
        y = np.array(lin(mx.array(x)))
        assert float(np.abs(y - ref).max() / np.abs(ref).max()) < 1e-4, tgk
        assert lin.can_fuse(1) == (max(S.fused_dims(lin.kt, tgk), 512) <= KN.Mul1Gemv.FUSED_SMEM_FLOATS) and lin.can_fuse(1)
        nz = int((np.array(lin.fused(mx.array(x))).view(np.uint16) != y.view(np.uint16)).sum())     # fused == three launches, bit for bit
        assert nz == 0, (tgk, nz)
        ys[tgk] = y
    a, b = (ys[k] for k in sorted(ys))
    assert float(np.abs(a - b).max() / np.abs(a).max()) < 1e-3


def test_grouped_multilinear_divides_the_fill(cores, monkeypatch):
    """The wo_a groups run in ONE launch: the k-split fills the device divided by G, so the launch reaches the target (512 on 64 cores) instead of
    overshooting it 8x, and the per-group slices keep the grouped kernel's tgk (the two paths stay bit-identical)."""
    import numpy as np
    from kyojin_mlx import patch as P
    cores(64)
    monkeypatch.setattr(P, "GROUPED_WOA", True)
    rng = np.random.default_rng(1)
    G, kin, nout, K = 8, 4096, 1024, 4
    t = mx.array(rng.integers(-32768, 32768, (G, kin // 16, nout // 16, 16 * K), dtype=np.int16))
    f16 = lambda *s: mx.array((rng.choice([-1.0, 1.0], s) * rng.uniform(0.5, 1.5, s)).astype(np.float16))
    ml = P.Exl3MultiLinear(t, f16(G, kin), f16(G, nout), cb="mul1")
    assert ml.grouped is not None and ml.grouped.tgk == ml.groups[0]._g[1].tgk == 8
    assert ml.num_groups * (nout // 128) * ml.grouped.tgk == 512
