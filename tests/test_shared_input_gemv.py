"""Flag shared_input_gemv (foreign_linear.DenseLinear / DenseGroupMember / group_dense, the 16-bit GEMV). CPU parts run anywhere; the Metal half needs KYOJIN_METAL_TESTS=1 on a Mac."""
import os
import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from kyojin_mlx import foreign_linear as FL

METAL = os.environ.get("KYOJIN_METAL_TESTS") == "1"
if not METAL:
    mx.set_default_device(mx.cpu)                                  # CPU parts never touch the GPU


@pytest.fixture(autouse=True)
def _flag():
    old = FL.SHARED_GEMV
    FL.SHARED_GEMV = True
    yield
    FL.SHARED_GEMV = old


def _w(n, k, dt, seed):
    a = np.random.default_rng(seed).standard_normal((n, k)).astype(np.float32) * 0.1
    return mx.array(a).astype(dt)


def test_dense_linear_is_the_fp32_matrix_product_and_half_the_bytes():
    for dt in (mx.bfloat16, mx.float16):
        w = _w(24, 40, dt, 1)
        x = mx.array(np.random.default_rng(2).standard_normal((3, 40)).astype(np.float32))
        lin = FL.DenseLinear(w)
        ref = np.array(x, np.float64) @ np.array(w.astype(mx.float32), np.float64).T      # fp64 reference: on Metal the GEMV only reorders the f32 sum and mlx's fp32 GPU matmul is only ~1e-3 accurate, so no bit-equality
        got = np.array(lin(x), np.float64)
        assert np.abs(got - ref).max() <= (1e-5 if not METAL else 1e-3) * max(np.abs(ref).max(), 1.0)
        assert lin.dev.full.nbytes * 2 == w.astype(mx.float32).nbytes
        assert lin(x.reshape(1, 3, 40)).shape == (1, 3, 24)


class _Toy(nn.Module):
    """q / k / v read one input, o reads another, `twice` is called twice (never grouped)."""

    def __init__(self, k=64, dt=mx.bfloat16):
        super().__init__()
        self.emb = nn.Embedding(10, k)
        self.q, self.k_, self.v = (FL.DenseLinear(_w(n, k, dt, i)) for i, n in enumerate((32, 16, 16)))
        self.o = FL.DenseLinear(_w(8, 32, dt, 9))
        self.twice = FL.DenseLinear(_w(8, k, dt, 10))

    def __call__(self, ids):
        h = self.emb(ids)
        a, b, c = self.q(h), self.k_(h), self.v(h)
        return self.o(a) + b.sum() + c.sum() + self.twice(h).sum() + self.twice(h * 2).sum()


def _fake_gpu(monkeypatch, calls):
    monkeypatch.setattr(FL, "bf_gemv_ok", lambda w, R: FL.SHARED_GEMV and 1 <= R <= FL.BF_ROWS)

    def ref(x2, w):
        calls.append((tuple(x2.shape), tuple(w.shape)))
        return x2.astype(mx.float32) @ w.astype(mx.float32).T
    monkeypatch.setattr(FL, "bf_gemv", ref)


def test_find_and_group_same_input_one_pass_and_same_values(monkeypatch):
    calls = []
    m = _Toy()
    ids = mx.array([[3]])
    want = m(ids)
    r = FL.group_dense(m, log=lambda *a: None)
    assert r["groups"] == 1 and r["members"] == 3 and r["launches_saved"] == 2     # q, k, v; `o` has its own input, `twice` is called twice
    assert type(m.q).__name__ == "DenseGroupMember" and type(m.o).__name__ == "DenseLinear" and type(m.twice).__name__ == "DenseLinear"
    _fake_gpu(monkeypatch, calls)
    mx.eval(m(ids))
    got = m(ids)
    assert mx.allclose(got, want, atol=1e-5)
    g = [c for c in calls if c[1][0] == 64]
    assert len(g) == 2 and all(c[1] == (64, 64) for c in g)                         # one concatenated pass per call (two calls), not three
    h = m.emb(ids)
    solo = [FL.DenseLinear(_w(n, 64, mx.bfloat16, i))(h) for i, n in enumerate((32, 16, 16))]
    for mod, s in zip((m.q, m.k_, m.v), solo):
        assert mx.allclose(mod(h), s, atol=1e-6)                                            # each member's slice is its own product
    assert mx.allclose(m.q(h), solo[0], atol=1e-6)                                          # a second ask with the same input recomputes (the slice was taken once), no stale value


def test_flag_off_and_prompt_rows_run_the_members_alone(monkeypatch):
    calls = []
    m = _Toy()
    FL.group_dense(m, log=lambda *a: None)
    _fake_gpu(monkeypatch, calls)
    h = mx.array(np.random.default_rng(5).standard_normal((1, 64)).astype(np.float32))
    on = m.q(h)
    assert len(calls) == 1
    FL.SHARED_GEMV = False
    off = m.q(h)
    assert len(calls) == 1 and mx.allclose(on, off, atol=1e-6)                              # off: the member's own float32 matmul (the same values)
    FL.SHARED_GEMV = True
    big = mx.array(np.random.default_rng(6).standard_normal((20, 64)).astype(np.float32))
    n0 = len(calls)
    y = m.k_(big)
    assert len(calls) == n0 and y.shape == (20, 16)                                 # 20 rows > BF_ROWS: not the GEMV


def test_trace_failure_leaves_the_model(monkeypatch):
    class Bad(_Toy):
        def __call__(self, ids):
            raise RuntimeError("needs more than tokens")
    m = Bad()
    assert FL.group_dense(m, log=lambda *a: None)["groups"] == 0 and type(m.q).__name__ == "DenseLinear"


@pytest.mark.skipif(not METAL, reason="KYOJIN_METAL_TESTS=1 (Metal)")
@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("R", [1, 2, 4, 8])
def test_metal_gemv_matches_fp32_matrix(dt, R):
    mx.set_default_device(mx.gpu)
    for n, k in ((4098, 3072), (120, 1024), (3072, 16384)):
        w = _w(n, k, dt, n)
        x = mx.array(np.random.default_rng(R).standard_normal((R, k)).astype(np.float32))
        got = FL.bf_gemv(x, w)
        mx.eval(got)
        ref = np.array(x, np.float64) @ np.array(w.astype(mx.float32), np.float64).T      # float64 reference: the GPU fp32 matmul may round its products (tf32-like)
        err = float(np.abs(np.array(got, np.float64) - ref).max()); scale = float(np.abs(ref).max())
        assert err <= 1e-5 * max(scale, 1.0), (n, k, err, scale)                    # same products, only the sum's order differs
        assert (np.array(got).argmax(-1) == ref.argmax(-1)).all()


@pytest.mark.skipif(not METAL, reason="KYOJIN_METAL_TESTS=1 (Metal)")
def test_metal_group_slices_and_head_shape():
    mx.set_default_device(mx.gpu)
    m = _Toy(k=256)
    ids = mx.array([[3]])
    FL.SHARED_GEMV = False
    want = m(ids)
    mx.eval(want)
    FL.SHARED_GEMV = True
    FL.group_dense(m, log=lambda *a: None)
    got = m(ids)
    mx.eval(got)
    assert float(mx.abs(got - want).max()) < 2e-3 * max(float(mx.abs(want).max()), 1.0)
    head = FL.DenseLinear(_w(120832, 3072, mx.bfloat16, 3))
    h = mx.array(np.random.default_rng(1).standard_normal((1, 3072)).astype(np.float32))
    y = head(h)
    ref = np.array(h, np.float64) @ np.array(head.dev.w.astype(mx.float32), np.float64).T
    mx.eval(y)
    assert int(mx.argmax(y)) == int(ref.argmax()) and head.dev.full.nbytes == 120832 * 3072 * 2


class _ToyQ8(nn.Module):
    def __init__(self, k=512, ns=(16, 32)):
        super().__init__()
        from kyojin_mlx import foreign_gguf as GF
        self.emb = nn.Embedding(10, k)
        rng = np.random.default_rng(3)
        self.a, self.b = (FL.ForeignLinear(GF.to_pack("Q8_0", rng.integers(0, 255, (n, k // 32 * 34), dtype=np.uint8), k)) for n in ns)

    def __call__(self, ids):
        h = self.emb(ids)
        return self.a(h).sum() + self.b(h).sum()


def test_stored_group_is_switchable_and_the_member_alone_keeps_its_rows():
    m = _ToyQ8()
    orig = [np.array(m.a.dev.qs), np.array(m.b.dev.qs)]
    r = FL.group_same_input(m, log=lambda *a: None, toggle=True)
    assert r["groups"] == 1 and r["members"] == 2
    for mod, q in zip((m.a, m.b), orig):
        assert mod._tog is True
        FL.SHARED_GEMV = False
        lin = None
        try:
            mod._alone                                       # the lazily built ForeignLinear of the member's rows (the build is the CPU-safe half; its call is a Metal kernel)
            lin = FL.ForeignLinear.__new__(FL.ForeignLinear)
            nn.Module.__init__(lin)
            lin.dev = mod._g.dev.rows(list(range(*mod._ab)))
        finally:
            FL.SHARED_GEMV = True
        assert np.array_equal(np.array(lin.dev.qs), q) and lin.dev.n == mod.out_features and bool(mx.all(lin.dev.gs == mod._g.dev.gs[mod._ab[0]:mod._ab[1]]))
