"""pfroof2: causal-width prefill indexer. CPU tests mirror the score kernel in numpy and check the selection logic against the stock full-width chain;
the Metal tests (marked) run the real kernels and skip cleanly without a GPU."""
import os
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import idx_topk as IT          # noqa: E402
from kyojin_mlx import pfroof2 as PR           # noqa: E402

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture(autouse=True)
def cpu_device():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def make(T, Tc, NH=4, seed=0, quant=False):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((T, NH, 128)).astype(np.float16)
    keys = rng.standard_normal((Tc, 128)).astype(np.float16)
    w = (rng.standard_normal((T, NH)) * 0.05).astype(np.float16)
    if quant:                                                         # few distinct values: many exact ties, also among the -inf columns
        q = np.round(q * 2) / 2
        keys = np.round(keys * 2) / 2
        w = np.round(w * 64) / 64
        q, keys, w = q.astype(np.float16), keys.astype(np.float16), w.astype(np.float16)
    return q, keys, w


def stock_topk(s, k):
    return np.array(IT.topk(mx.array(s), k))


def restricted_topk(q, keys, w, start, ratio, qb, k, cand):
    """The pfroof2 block loop with the numpy scorer in place of the Metal one."""
    T, Tc = q.shape[0], keys.shape[0]
    out = []
    for t0, n in IT.plan(T, qb):
        W = PR.causal_width(start, t0, n, ratio, Tc, k)
        s = PR.ref_scores(q[t0:t0 + n], keys[:W], w[t0:t0 + n], start, ratio, t0=t0 + 0)
        s = np.where(cand[t0:t0 + n, :W], s, -np.inf) if cand is not None else s
        out.append(np.array(PR.select_topk(mx.array(s), k)))
    return np.concatenate(out, axis=0)


def full_topk(q, keys, w, start, ratio, qb, k, cand):
    """Stock shape: all Tc columns per block."""
    T = q.shape[0]
    out = []
    for t0, n in IT.plan(T, qb):
        s = PR.ref_scores(q[t0:t0 + n], keys, w[t0:t0 + n], start, ratio, t0=t0)
        s = np.where(cand[t0:t0 + n], s, -np.inf) if cand is not None else s
        out.append(stock_topk(s, k))
    return np.concatenate(out, axis=0)


@pytest.mark.parametrize("T", [33, 100, 257])
@pytest.mark.parametrize("start", [0, 40, 300])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("k", [8, 64])
@pytest.mark.parametrize("qb", [0, 16, 64])
@pytest.mark.parametrize("quant", [False, True], ids=["dense", "ties"])
@pytest.mark.parametrize("use_cand", [False, True], ids=["plain", "cand"])
def test_causal_width_equals_full_width(T, start, ratio, k, qb, quant, use_cand):
    Tc = (start + T) // ratio
    if Tc < k:
        pytest.skip("topk is min(index_topk, Tc) in the model")
    q, keys, w = make(T, Tc, seed=T + start + ratio, quant=quant)
    cand = np.random.default_rng(T).random((T, Tc)) < 0.6 if use_cand else None
    a = restricted_topk(q, keys, w, start, ratio, IT.qblock(T, qb), k, cand)
    b = full_topk(q, keys, w, start, ratio, IT.qblock(T, qb), k, cand)
    assert a.shape == b.shape == (T, k)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("n,W", [(5, 16), (32, 64), (7, 128)])
def test_select_topk_with_k_equal_width_is_arange(n, W):
    """First block of a fresh prompt: W == topk. Rows are mostly -inf with ties (and a few all -inf rows); stock gives 0 .. W-1."""
    rng = np.random.default_rng(W)
    s = np.where(rng.random((n, W)) < 0.15, np.round(rng.standard_normal((n, W)) * 2) / 2, -np.inf).astype(np.float32)
    s[0] = -np.inf
    got = np.array(PR.select_topk(mx.array(s), W))
    assert got.dtype == np.int32 and got.shape == (n, W)
    assert np.array_equal(got, stock_topk(s, W))
    assert np.array_equal(got, np.broadcast_to(np.arange(W), (n, W)))


def test_scores_of_the_kept_columns_are_the_stock_values():
    q, keys, w = make(40, 90, seed=3)
    full = PR.ref_scores(q, keys, w, 20, 2)
    W = PR.causal_width(20, 0, 40, 2, 90, 8)
    assert np.array_equal(PR.ref_scores(q, keys[:W], w, 20, 2), full[:, :W])
    assert np.all(full[:, W:] == -np.inf)                             # nothing finite is dropped


def test_ref_scores_match_a_plain_formula():
    q, keys, w = make(16, 24, NH=3, seed=9)
    got = PR.ref_scores(q, keys, w, 8, 1)
    prod = (q.astype(np.float32) @ keys.astype(np.float32).T.reshape(1, 128, 24)[0]) if False else np.einsum("thd,cd->thc", q.astype(np.float32), keys.astype(np.float32))
    prod = prod.astype(np.float16).astype(np.float32)
    ref = (np.maximum(prod, 0) * w[:, :, None].astype(np.float32)).sum(1)
    clen = (8 + np.arange(16) + 1)[:, None]
    ref = np.where(np.arange(24)[None] < clen, ref, -np.inf)
    fin = np.isfinite(ref)
    assert np.array_equal(np.isfinite(got), fin)
    assert np.allclose(got[fin], ref[fin], rtol=1e-5, atol=1e-6)


def test_width_bounds():
    assert PR.causal_width(0, 0, 512, 1, 8192, 512) == 512
    assert PR.causal_width(0, 7680, 512, 1, 8192, 512) == 8192
    assert PR.causal_width(0, 0, 32, 1, 8192, 512) == 512             # never below topk
    assert PR.causal_width(0, 0, 33, 2, 40, 8) == 40                  # capped by Tc
    for start in (0, 1000, 8191):
        for t0 in range(0, 4096, 512):
            W = PR.causal_width(start, t0, 512, 2, (start + 4096) // 2, 512)
            assert W >= min((start + t0 + 512) // 2, (start + 4096) // 2) and W <= (start + 4096) // 2


def test_flag_reads_the_environment_at_call_time(monkeypatch):
    monkeypatch.delenv(PR.ENV, raising=False)
    assert not PR.enabled()
    monkeypatch.setenv(PR.ENV, "1")
    assert PR.enabled()
    monkeypatch.setenv(PR.ENV, "0")
    assert not PR.enabled()
    monkeypatch.setenv(PR.ENV_TOPK, "0")
    assert not PR.topk_mt_enabled()


def test_flags_registered_default_off():
    from kyojin_mlx import flags as FL
    assert FL.FLAGS["pfroof2"].env == PR.ENV and FL.FLAGS["pfroof2"].default == "0"
    assert FL.FLAGS["pfroof2_topk"].default == "0"                  # the flag table must not override the module default (argsort + sort)
    assert FL.FLAGS["pfroof2_topk"].default == os.environ.get(PR.ENV_TOPK, "0")


@pytest.mark.parametrize("Tc", [8192, 32768, 65536])
def test_workspace_plateaus_with_chunk_size(Tc):
    """Transient bytes of an indexer block depend on the block size and the key width, never on the chunk rows: 8192, 16384, 32768 rows give one figure."""
    ws = {T: PR.indexer_workspace_bytes(T, Tc, 512, 512, start=Tc - T if Tc >= T else 0) for T in (8192, 16384, 32768) if Tc >= T}
    base = PR.indexer_workspace_bytes(8192, Tc, 512, 512, start=Tc - 8192)
    assert all(v == base for v in ws.values()), ws
    stock = PR.indexer_workspace_bytes(32768, Tc, 512, 512, start=Tc - 32768 if Tc >= 32768 else 0, restricted=False)
    assert PR.indexer_workspace_bytes(32768, Tc, 512, 512, start=Tc - 32768 if Tc >= 32768 else 0) <= stock
    assert base <= (1 << 26) * 2 + 12 * 512 * Tc                       # half product capped by max_elems, plus three [512, Tc] passes


def test_workspace_never_exceeds_the_stock_shape():
    for T in (512, 4096, 8192):
        for start in (0, 4096):
            a = PR.indexer_workspace_bytes(T, start + T, 512, 512, start=start)
            b = PR.indexer_workspace_bytes(T, start + T, 512, 512, start=start, restricted=False)
            assert a <= b


def test_budget_model_is_consistent():
    rows = PR.budget()
    assert len(rows) >= 6
    for r in rows:
        assert r["roof_s"] > 0 and r["time_s"] > 0 and r["anchor"]
    stock, new = rows[-2], rows[-1]
    assert new["time_s"] < stock["time_s"] and new["roof_s"] < stock["roof_s"]
    assert rows[0]["roof_s"] == pytest.approx(495e6 * 8192 * 40 / 90e12)


# ------------------------------------------------------------------------------------------ Metal (the lead runs these on the Ultra)
@needs_metal
@pytest.mark.parametrize("start,ratio,topk,qb", [(0, 1, 16, 64), (0, 2, 16, 64), (300, 2, 16, 64), (0, 1, 64, 64)])
def test_metal_indexer_flag_on_equals_off(start, ratio, topk, qb, monkeypatch):
    mx.set_default_device(mx.gpu)
    pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
    import test_idx_topk as TI
    from kyojin_mlx import idxfast as IX
    L, I, SimpleNamespace = TI.L, TI.I, TI.SimpleNamespace
    monkeypatch.setattr(TI, "TOPK", topk)
    T = 150
    idx, cfg = TI.make_indexer(ratio, -1)
    fused = I._impl_indexer(L.Indexer.__call__)
    rng = np.random.default_rng(7)
    n = (start + T) // ratio
    x = mx.array(rng.standard_normal((1, T, cfg.hidden_size)).astype(np.float32)).astype(mx.float16)
    qr = mx.array(rng.standard_normal((1, T, cfg.q_lora_rank)).astype(np.float32)).astype(mx.float16)
    keys = mx.array(rng.standard_normal((1, 400, cfg.index_head_dim)).astype(np.float32)).astype(mx.float16)
    calls = {"mt": 0, "widths": []}
    real_mt, real_sr = IX.topk_mt, I.scores_range
    monkeypatch.setattr(IX, "topk_mt", lambda *a, **k: (calls.__setitem__("mt", calls["mt"] + 1), real_mt(*a, **k))[1])
    monkeypatch.setattr(I, "scores_range", lambda q, kk, *a, **k: (calls["widths"].append(kk.shape[1]), real_sr(q, kk, *a, **k))[1])
    res, info = {}, {}
    for name, on, tk in (("off", "0", "1"), ("on_radix", "1", "1"), ("on_sort", "1", "0")):
        monkeypatch.setenv(PR.ENV, on)
        monkeypatch.setenv(PR.ENV_TOPK, tk)
        monkeypatch.setenv(IT.QBLOCK_ENV, str(qb))
        calls["mt"], calls["widths"] = 0, []
        r = fused(idx, x, qr, None, start, 0, SimpleNamespace(index_k=keys, candidates=None))
        mx.eval(r)
        res[name], info[name] = np.array(r), (calls["mt"], list(calls["widths"]))
    assert res["off"].shape == (1, T, min(topk, n))
    assert np.array_equal(res["off"], res["on_radix"])
    assert np.array_equal(res["off"], res["on_sort"])
    if not I.TOPK_MT:                                                # E4 (KYOJIN_PF_IDX_TOPK_MT) puts the radix select on every path
        assert info["off"][0] == 0
    assert info["on_sort"][0] == 0                                  # the argsort path never does
    assert info["on_radix"][0] >= 1, info                            # the new path really ran
    assert min(info["on_radix"][1]) < n and info["off"][1] == [n] * len(info["off"][1])


# ------------------------------------------------------------------------------------------ integration on the CPU (Metal kernels replaced by numpy mirrors)
COL_MIN = 64


def _cpu_indexer_env(monkeypatch):
    pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
    import test_idx_topk as TI
    I = TI.I
    seen = []

    def scores_range(q, keys, w, start, ratio, t0, n, max_elems=1 << 26, n0=0):
        seen.append(keys.shape[1])
        return mx.array(PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(keys[0]), np.array(w[0, t0:t0 + n]), start, ratio, t0=t0, n0=n0))

    monkeypatch.setattr(I, "scores_range", scores_range)
    monkeypatch.setenv(I.CAND_ENV, "0")                          # E1 (cand_topk) bypasses scores_range and needs a block-granular mask: tests/test_pf_idx_cand.py
    monkeypatch.setattr(I, "fq4_rot", lambda q, cos, sin, rd: q)
    return TI, I, seen


@pytest.mark.parametrize("T,start,ratio,cand_layer,qb", [(100, 0, 1, -1, 16), (100, 40, 2, -1, 32), (129, 0, 2, 0, 64), (100, 40, 1, 0, 16),
                                                           (64, 0, 2, 1, 16), (129, 40, 1, -1, 512)])
def test_indexer_integration_flag_on_equals_off(T, start, ratio, cand_layer, qb, monkeypatch):
    TI, I, seen = _cpu_indexer_env(monkeypatch)
    L, mk = TI.L, TI.make_indexer
    idx, cfg = mk(ratio, cand_layer)
    fused = I._impl_indexer(L.Indexer.__call__)
    rng = np.random.default_rng(T + start + ratio)
    n = (start + T) // ratio
    x = mx.array(rng.standard_normal((1, T, cfg.hidden_size)).astype(np.float32)).astype(mx.float16)
    qr = mx.array(rng.standard_normal((1, T, cfg.q_lora_rank)).astype(np.float32)).astype(mx.float16)
    keys = mx.array(rng.standard_normal((1, 400, cfg.index_head_dim)).astype(np.float32)).astype(mx.float16)
    cand = mx.array(rng.random((1, T, n)) < 0.7) if cand_layer == 0 else None
    monkeypatch.setenv(TI.IT.QBLOCK_ENV, str(qb))
    res, widths = {}, {}
    for name in ("0", "1"):
        monkeypatch.setenv(PR.ENV, name)
        seen.clear()
        cache = TI.SimpleNamespace(index_k=keys, candidates=cand)
        r = fused(idx, x, qr, None, start, 0, cache)
        mx.eval(r)
        res[name] = (np.array(r), None if cache.candidates is None else np.array(cache.candidates))
        widths[name] = list(seen)
    assert np.array_equal(res["0"][0], res["1"][0])
    if cand_layer == 1:
        assert np.array_equal(res["0"][1], res["1"][1])
        assert widths["0"] == widths["1"]                           # candidate source: stock width kept
    elif cand_layer != 1 and n > COL_MIN and qb < T:
        assert max(widths["1"]) <= n and min(widths["1"]) < n         # the narrow blocks really are narrower
