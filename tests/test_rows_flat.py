"""More than 4 rows in ONE GEMV launch (row groups on the grid, metal_src.gemv_rg_source): a row keeps the bits it has in a 1-row or a 4-row launch,
for a plain EXL3 linear, the grouped wo_a and every row count up to the last group being partial."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import kernels as KN
from kyojin_mlx.exl3_linear import Exl3Linear
from kyojin_mlx.patch import Exl3MultiLinear

ROWS = [5, 6, 7, 8, 9, 12, 13, 16, 17, 31, 33, 40]


@pytest.fixture(autouse=True)
def _solo_bits(monkeypatch):
    """These tests assert that a row keeps its solo bits: pin the NAX rows switch (ON by default, a rounding class) off."""
    from kyojin_mlx import exl3_linear as EL
    monkeypatch.setattr(EL, "ROWS_NAX", False)
    monkeypatch.setenv("KYOJIN_ROWS_NAX", "0")                      # and in the subprocesses of the checks


def make(kin, nout, K, seed, cb, G=None):
    rng = np.random.default_rng(seed)
    sh = (kin // 16, nout // 16, 16 * K) if G is None else (G, kin // 16, nout // 16, 16 * K)
    suh_sh, svh_sh = ((kin,), (nout,)) if G is None else ((G, kin), (G, nout))
    tr = rng.integers(-32768, 32768, sh, dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], suh_sh) * rng.uniform(0.5, 1.5, suh_sh)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], svh_sh) * rng.uniform(0.5, 1.5, svh_sh)).astype(np.float16)
    return mx.array(tr), mx.array(suh), mx.array(svh)


def bits(a):
    return np.array(a).view(np.uint16)


CASES = [("mul1", 5, 512, 256), ("mul1", 3, 1024, 384), ("mcg", 2, 512, 1024), ("mcg", 4, 256, 128)]


@pytest.fixture(scope="module", params=CASES, ids=lambda c: f"{c[0]}-K{c[1]}-{c[2]}x{c[3]}")
def lin(request):
    cb, K, kin, nout = request.param
    old = EL.ROW_SPLIT_MAX
    EL.ROW_SPLIT_MAX = 64
    yield Exl3Linear(*make(kin, nout, K, 5 + K, cb), cb=cb)
    EL.ROW_SPLIT_MAX = old


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dt", [np.float16, np.float32])
def test_linear_rows_equal_their_solo_bits(lin, rows, dt):
    x = mx.array((np.random.default_rng(rows).standard_normal((rows, lin.in_features)) * 2).astype(dt))
    got = lin(x)
    solo = mx.concatenate([lin(x[i:i + 1]) for i in range(rows)], axis=0)
    chunks = mx.concatenate([lin._gemv(min(4, rows - i))(x[i:i + 4]) for i in range(0, rows, 4)], axis=0)       # the former 4-row launches
    assert got.shape == (rows, lin.out_features)
    assert np.array_equal(bits(got), bits(solo)) and np.array_equal(bits(got), bits(chunks))


def test_linear_three_launches_whatever_the_rows(monkeypatch):
    """rot + one row-group GEMV + finalize: 3 custom kernel calls from 5 to 40 rows (the former 4-row launches made 3 per group of rows)."""
    import collections
    orig = mx.fast.metal_kernel
    cnt = collections.Counter()

    def counting(*a, **k):
        ker = orig(*a, **k)

        def call(*ca, **ck):
            cnt[k.get("name", "?")] += 1
            return ker(*ca, **ck)
        return call
    monkeypatch.setattr(mx.fast, "metal_kernel", counting)
    monkeypatch.setattr(KN, "_cache", {})
    monkeypatch.setattr(KN, "_rot_cache", {})
    monkeypatch.setattr(EL, "ROW_SPLIT_MAX", 64)
    lin = Exl3Linear(*make(512, 256, 3, 9, "mcg"), cb="mcg")
    for rows in (5, 17, 40):
        cnt.clear()
        lin(mx.zeros((rows, lin.in_features), dtype=mx.float16))
        assert sum(cnt.values()) == 3 and any("gemvrg" in k for k in cnt), (rows, dict(cnt))


@pytest.fixture(scope="module", params=[("mul1", 5, 8), ("mcg", 2, 8), ("mul1", 3, 4)], ids=lambda p: f"{p[0]}-K{p[1]}-G{p[2]}")
def multi(request):
    cb, K, G = request.param
    old = EL.ROW_SPLIT_MAX
    EL.ROW_SPLIT_MAX = 64
    yield Exl3MultiLinear(*make(512, 256, K, 7 + K, cb, G), cb)
    EL.ROW_SPLIT_MAX = old


@pytest.mark.parametrize("rows", ROWS)
def test_grouped_rows_equal_four_row_launches(multi, rows):
    ml = multi
    x = mx.array((np.random.default_rng(rows + 100).standard_normal((1, ml.num_groups, rows, ml.in_features)) * 2).astype(np.float16))
    got = ml.grouped(x)
    want = mx.concatenate([ml.grouped(x[:, :, i:i + 4]) for i in range(0, rows, 4)], axis=2)
    assert got.shape == (1, ml.num_groups, rows, ml.out_features)
    assert np.array_equal(bits(got), bits(want))


@pytest.mark.parametrize("users", [5, 9, 16, 33])
def test_users_as_rows_dispatch(multi, users):
    """The decode step of N users (x [N, G, 1, in]) takes the grouped kernel in one pass and equals the per-user solo call."""
    ml = multi
    x = mx.array((np.random.default_rng(users).standard_normal((users, ml.num_groups, 1, ml.in_features)) * 2).astype(np.float16))
    got = ml(x)
    solo = mx.concatenate([ml(x[i:i + 1]) for i in range(users)], axis=0)
    assert np.array_equal(bits(got), bits(solo))


@pytest.mark.parametrize("rows", [65, 300, 1100, 2100])
def test_grouped_prompt_fq_fold_bit_identical(rows):
    """Prompt rows of an fq_in grouped linear (wo_a): the fp8 round trip inside the rot (fused NAX path up to DENSE_FUSED_MAX rows, per-group dense
    path above) gives the bits of the separate fake quant."""
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
    ml = Exl3MultiLinear(*make(512, 256, 3, 10, "mul1", 8), "mul1", fq_in=True)
    x = mx.array((np.random.default_rng(rows).standard_normal((1, 8, rows, 512)) * 3).astype(np.float16))
    got = ml(x)
    ml.fq_in = False
    want = ml(fake_quant_fp8_ue8m0(x))
    assert np.array_equal(bits(got), bits(want))


@pytest.mark.parametrize("rows", [3, 65, 300, 2100])
def test_linear_fq8_arg_bit_identical(rows):
    """Exl3Linear(x, fq8=True) == Exl3Linear(fake_quant(x)) on every row path."""
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
    lin = Exl3Linear(*make(512, 256, 3, 11, "mul1"), cb="mul1")
    x = mx.array((np.random.default_rng(rows).standard_normal((rows, 512)) * 3).astype(np.float16))
    assert np.array_equal(bits(lin(x, fq8=True)), bits(lin(fake_quant_fp8_ue8m0(x))))


@pytest.mark.parametrize("rows", [65, 300, 2100])
def test_linear_noted_fq_prompt_fold_bit_identical(rows, monkeypatch):
    """A prompt-row input that fqfuse knows as fq(src) runs the round trip inside the rot (fq node unread) with the bits of the fq output."""
    from kyojin_mlx import fqfuse as FQF
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
    monkeypatch.setattr(FQF, "_pairs", [])
    lin = Exl3Linear(*make(512, 256, 3, 12, "mul1"), cb="mul1")
    src = mx.array((np.random.default_rng(rows).standard_normal((1, rows, 512)) * 3).astype(np.float16))
    want = lin(fake_quant_fp8_ue8m0(src))
    y = fake_quant_fp8_ue8m0(src)
    FQF.note(src, y)
    seen = []
    orig = lin._dense
    monkeypatch.setattr(lin, "_dense", lambda *a, **k: (seen.append(k.get("fq8")), orig(*a, **k))[1])
    got = lin(y)
    assert seen == [True] and np.array_equal(bits(got), bits(want))
