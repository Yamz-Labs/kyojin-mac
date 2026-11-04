"""pf_cluster (KYOJIN_PF_IDX_CLUSTER): the candidate block set of a query block against the exact full-width top-k, on the CPU with the numpy score
mirror. Two things are asserted: the shape of the set (every probe row returns its exact set, the union is a superset of the probes' top-k, the output keeps
the stock contract) and the recall of the rows between the probes on keys shaped like the pack's (fp4 ue8m0 codes, 32-value power-of-two blocks) and on
clustered keys. The Metal kernels behind the rescoring are checked by tools/bench_pf_flat.py and test_pf_flat_metal.py; here the same glue runs on numpy.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL                 # noqa: E402
from kyojin_mlx import idx_topk as IT              # noqa: E402
from kyojin_mlx import pf_cluster as PC            # noqa: E402
from kyojin_mlx import pf_flat as PF               # noqa: E402
from kyojin_mlx import pfroof2 as PR               # noqa: E402

NH, D, BS = 32, 128, 8
CODES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)      # the fp4 ue8m0 grid of the index keys


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(PF, "METAL_OPT", False)
    monkeypatch.setattr(PC, "ENABLED", True)
    monkeypatch.setattr(PC, "MIN_KEYS", 0)
    monkeypatch.setattr(PC, "PROBES", 16)
    monkeypatch.setattr(PC, "MULT", 4)
    monkeypatch.setattr(PC, "CAP", 8192)
    monkeypatch.setattr(PC, "MAXPCT", 100)
    monkeypatch.setattr(PC, "RECALL", False)
    monkeypatch.setattr(IT, "QBLOCK_ENV", "KYOJIN_IDX_QBLOCK")
    PC.HITS.update({k: 0 for k in PC.HITS})


def fp4_keys(rng, Tc, centroids=24, spread=0.35):
    """Keys shaped like the pack's index keys: every value a code of the fp4 grid times a power-of-two scale per 32-value block, drawn around `centroids`
    learned-looking centres (so a few regions of the past dominate), which is what makes a candidate set carry over from one row to the next."""
    c = rng.standard_normal((centroids, D)).astype(np.float32)
    own = rng.integers(0, centroids, Tc)
    k = c[own] + spread * rng.standard_normal((Tc, D)).astype(np.float32)
    k = np.array(mx.array(k).astype(mx.float16), dtype=np.float32)
    blk = k.reshape(Tc, D // 32, 32)
    amax = np.abs(blk).max(-1, keepdims=True) + 1e-9
    scale = np.exp2(np.ceil(np.log2(amax)) - 2.0)                     # a power of two, as ue8m0
    q = np.clip(blk / scale, -6, 6)
    idx = np.abs(np.abs(q)[..., None] - np.abs(CODES)).argmin(-1)     # the nearest code
    val = np.sign(q) * np.abs(CODES)[idx]
    return mx.array((val * scale).reshape(1, Tc, D).astype(np.float16))


def queries(rng, T, templates=6, spread=0.4):
    """Query rows in `templates` groups (a prompt chunk whose neighbours ask nearly the same thing), plus the head weights."""
    c = rng.standard_normal((templates, NH, D)).astype(np.float32)
    own = np.arange(T) * templates // T
    q = c[own] + spread * rng.standard_normal((T, NH, D)).astype(np.float32)
    w = (rng.standard_normal((T, NH)) * 0.05).astype(np.float16)
    return mx.array(q[None].astype(np.float16)), mx.array(w[None])


def exact(q, k, w, start, ratio, T):
    return mx.array(PR.ref_scores(np.array(q[0]), np.array(k[0]), np.array(w[0]), start, ratio))


def stock(s, topk):
    return np.array(IT.topk(s, topk))


def recall(got, ref):
    """Mean fraction of the exact top-k keys the lever returned, over the rows."""
    return float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(got, ref)]))


def run(q, k, w, start, ratio, topk, T, qblock="0"):
    import os
    os.environ[IT.QBLOCK_ENV] = qblock
    try:
        return np.array(PC.plain_topk(q, k, w, start, ratio, topk, T))
    finally:
        os.environ.pop(IT.QBLOCK_ENV, None)


# ----------------------------------------------------------------------------------------------------------------------------- gates and shape
def test_applies_gates(monkeypatch):
    rng = np.random.default_rng(0)
    q, w = queries(rng, 16)
    k = fp4_keys(rng, 4096)
    assert PC.applies(q, k, w, 4096, 32, 4 * 4096, 4, True)
    assert not PC.applies(q, k, w, 4096, 32, 4 * 4096, 4, False)                  # the candidate source and its consumers keep their own paths
    assert not PC.applies(q, k, w, 4096, 32, 4 * 8, 4, True)                       # first row sees fewer than topk keys
    assert not PC.applies(q, k, w, 64, 32, 4 * 4096, 4, True)                     # too few keys
    monkeypatch.setattr(PC, "ENABLED", False)
    assert not PC.applies(q, k, w, 4096, 32, 4 * 4096, 4, True)
    monkeypatch.setattr(PC, "ENABLED", True)
    monkeypatch.setattr(PC, "MIN_KEYS", 8192)
    assert not PC.applies(q, k, w, 4096, 32, 4 * 4096, 4, True)
    monkeypatch.setattr(PC, "MIN_KEYS", 0)
    monkeypatch.setattr(PC, "CAP", 1)
    assert not PC.applies(q, k, w, 4096, 32, 4 * 4096, 4, True)                   # a cap below topk / 8 cannot hold a top-k


def test_probe_rows_include_both_ends():
    assert PC.probe_rows(512, 8) == [0, 64, 128, 192, 256, 320, 384, 448, 511]
    assert PC.probe_rows(3, 8) == [0, 1, 2]                                        # a block no longer than the probe count: every row is a probe
    rows = PC.probe_rows(7, 4)
    assert rows == sorted(set(rows)) and rows[0] == 0 and rows[-1] == 6


def test_output_contract_and_probe_exactness():
    rng = np.random.default_rng(1)
    T, Tc, topk, ratio = 24, 4096, 24, 4
    q, w = queries(rng, T)
    k = fp4_keys(rng, Tc)
    start = ratio * (Tc - T)
    ref = stock(exact(q, k, w, start, ratio, T), topk)
    got = run(q, k, w, start, ratio, topk, T)
    assert got.shape == (T, topk) and got.dtype == np.int32
    assert np.all(np.diff(got, axis=1) >= 0) and got.max() < Tc                # ascending absolute positions, all inside the past
    rows = PC.probe_rows(T, PC.PROBES)
    assert np.array_equal(got[rows], ref[rows])                               # a probe row returns exactly the stock set
    assert (PC.HITS["blocks"] > 0) and (PC.HITS["fallback"] == 0)
    assert PC.HITS["union_blocks"] >= PC.HITS["blocks"] * topk // BS


def test_all_rows_identical_is_exact():
    """Every row of the block asks the same question: the probes' set covers the whole block and the answer is the stock one."""
    rng = np.random.default_rng(2)
    T, Tc, topk, ratio = 16, 4096, 24, 2
    c = rng.standard_normal((1, NH, D)).astype(np.float32)
    q = mx.array(np.repeat(c, T, axis=0)[None].astype(np.float16))
    w = mx.array(np.full((1, T, NH), 0.05, dtype=np.float16))
    k = fp4_keys(rng, Tc)
    start = ratio * (Tc - T)
    ref = stock(exact(q, k, w, start, ratio, T), topk)
    got = run(q, k, w, start, ratio, topk, T, qblock="6")
    assert np.array_equal(got, ref)


def test_cap_falls_back_to_the_stock_block():
    rng = np.random.default_rng(3)
    T, Tc, topk, ratio = 12, 4096, 16, 4
    q, w = queries(rng, T)
    k = fp4_keys(rng, Tc)
    start = ratio * (Tc - T)
    ref = stock(exact(q, k, w, start, ratio, T), topk)
    PC.CAP = 16                                                              # restored by the fixture
    got = run(q, k, w, start, ratio, topk, T, qblock="5")
    assert np.array_equal(got, ref)                                           # a set this narrow cannot hold topk keys: stock, value for value
    assert PC.HITS["fallback"] > 0 and PC.HITS["blocks"] == 0


def test_wide_union_falls_back_and_the_chunk_stops_probing(monkeypatch):
    """A union that covers more than `maxpct` of the keys cannot beat the stock pass: that block runs stock and, after two of them, the rest of the chunk runs
    stock without paying probes."""
    rng = np.random.default_rng(8)
    T, Tc, topk, ratio = 24, 4096, 24, 4
    q, w = queries(rng, T)
    k = fp4_keys(rng, Tc)
    start = ratio * (Tc - T)
    ref = stock(exact(q, k, w, start, ratio, T), topk)
    monkeypatch.setattr(PC, "CAP", 8192)
    monkeypatch.setattr(PC, "MAXPCT", 5)
    got = run(q, k, w, start, ratio, topk, T, qblock="8")
    assert np.array_equal(got, ref)                                           # exact: every block took the stock path
    assert PC.HITS["blocks"] == 0 and PC.HITS["fallback"] == 3
    assert PC.HITS["probe_keys"] <= PC.SKIP_BLOCKS * 8 * PC.MULT * topk        # probes paid for at most SKIP_BLOCKS blocks


# ----------------------------------------------------------------------------------------------------------------------------- recall
def test_recall_fp4_keys_default_knobs():
    """The shipped defaults (16 probes, mult 4) on keys shaped like the pack's: the rows between the probes mostly agree with them, and the cost is capped:
    a union never carries more than probes x mult x topk keys."""
    rng = np.random.default_rng(4)
    T, Tc, topk, ratio = 48, 8192, 32, 4
    q, w = queries(rng, T)
    k = fp4_keys(rng, Tc)
    start = ratio * (Tc - T)
    ref = stock(exact(q, k, w, start, ratio, T), topk)
    got = run(q, k, w, start, ratio, topk, T, qblock="48")
    rec = recall(got, ref)
    assert rec >= 0.85, f"recall {rec:.3f}"                                    # measured 0.93 on this seed
    assert PC.HITS["union_keys"] <= PC.HITS["blocks"] * PC.PROBES * PC.MULT * topk * BS      # <= probes x mult x topk 8-key blocks per block


def test_recall_rises_with_the_knobs(monkeypatch):
    """Recall is the knob: more probes and a wider per-probe set cost key scors but buy back the missed keys."""
    rng = np.random.default_rng(5)
    T, Tc, topk, ratio = 64, 16384, 32, 4
    q, w = queries(rng, T, templates=12)
    k = fp4_keys(rng, Tc, centroids=48)
    start = ratio * (Tc - T)
    ref = stock(exact(q, k, w, start, ratio, T), topk)
    base = recall(run(q, k, w, start, ratio, topk, T, qblock="64"), ref)
    monkeypatch.setattr(PC, "PROBES", 32)
    monkeypatch.setattr(PC, "MULT", 16)
    wide = recall(run(q, k, w, start, ratio, topk, T, qblock="64"), ref)
    assert wide > base, (base, wide)                                          # measured 0.75 -> 1.00 on this seed
    assert wide >= 0.95, f"recall {wide:.3f}"


def test_recall_on_clustered_and_isotropic_keys():
    """Two key shapes the ledger names: clustered (a bound prunes something) and isotropic (nothing prunes). The union path's recall depends only on how
    much the rows of a block agree, which is the quantity the probe count buys."""
    rng = np.random.default_rng(6)
    T, Tc, topk, ratio = 32, 8192, 32, 4
    q, w = queries(rng, T, templates=1, spread=0.05)                            # one template: the whole block asks the same thing
    start = ratio * (Tc - T)
    k = fp4_keys(rng, Tc)
    clustered = recall(run(q, k, w, start, ratio, topk, T, qblock="32"), stock(exact(q, k, w, start, ratio, T), topk))
    assert clustered >= 0.90, clustered                                       # measured 0.96 on this seed
    iso = mx.array(rng.standard_normal((1, Tc, D)).astype(np.float16))         # isotropic: neighbouring rows agree less
    iso_rec = recall(run(q, iso, w, start, ratio, topk, T, qblock="32"), stock(exact(q, iso, w, start, ratio, T), topk))
    assert iso_rec > 0.5                                                      # still far from random (32 of 8192 keys)
    assert clustered >= iso_rec


def test_recall_counter_samples_non_probe_rows(monkeypatch):
    rng = np.random.default_rng(7)
    T, Tc, topk, ratio = 48, 4096, 24, 4
    q, w = queries(rng, T)
    k = fp4_keys(rng, Tc)
    start = ratio * (Tc - T)
    monkeypatch.setattr(PC, "RECALL", True)
    run(q, k, w, start, ratio, topk, T)
    assert PC.HITS["recall_total"] > 0
    assert 0.0 <= PC.recall_pct() <= 100.0


# ----------------------------------------------------------------------------------------------------------------------------- dispatch
def test_indexer_dispatch_serves_a_plain_layer(monkeypatch):
    """`ds_pf_idx._impl_indexer` really takes the union path for a plain layer (and leaves every row's answer inside the stock contract): the stock arm runs
    the numpy-mirrored score chain, the lever arm the same rows through pf_cluster."""
    pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
    import test_idx_topk as TI
    I = TI.I

    def scores_range(q, keys, w, start, ratio, t0, n, max_elems=1 << 26, n0=0, kq=None):
        return mx.array(PR.ref_scores(np.array(q[0, t0:t0 + n]), np.array(keys[0]), np.array(w[0, t0:t0 + n]), start, ratio, t0=t0, n0=n0))

    monkeypatch.setattr(I, "scores_range", scores_range)
    monkeypatch.setattr(I, "fq4_rot", lambda q, cos, sin, rd: q)
    monkeypatch.setattr(I, "ENABLED", True)
    idx, cfg = TI.make_indexer(2, -1)                                          # a plain layer: no candidate source, no consumers
    fused = I._impl_indexer(TI.L.Indexer.__call__)
    rng = np.random.default_rng(9)
    T, start = 96, 400
    n = (start + T) // 2
    x = mx.array(rng.standard_normal((1, T, cfg.hidden_size)).astype(np.float32)).astype(mx.float16)
    qr = mx.array(rng.standard_normal((1, T, cfg.q_lora_rank)).astype(np.float32)).astype(mx.float16)
    keys = fp4_keys(rng, n, centroids=8)
    monkeypatch.setenv(IT.QBLOCK_ENV, "32")
    got = {}
    for name, on in (("off", False), ("on", True)):
        monkeypatch.setattr(PC, "ENABLED", on)
        cache = TI.SimpleNamespace(index_k=keys, candidates=None)
        r = fused(idx, x, qr, None, start, 0, cache)
        mx.eval(r)
        got[name] = np.array(r)[0]
    topk = min(TI.TOPK, n)
    assert got["off"].shape == got["on"].shape == (T, topk)
    assert np.all(np.diff(got["on"], axis=1) >= 0)                              # ascending absolute key positions
    assert PC.HITS["calls"] > 0 and PC.HITS["blocks"] > 0                       # the lever path ran, not the fallback
    assert recall(got["on"], got["off"]) >= 0.75, recall(got["on"], got["off"])


# ----------------------------------------------------------------------------------------------------------------------------- flags
def test_flag_table_matches_module_literals():
    for name, attr, default in (("pf_idx_cluster", "ENABLED", "0"), ("pf_idx_cluster_min", "MIN_KEYS", "65536"),
                                ("pf_idx_cluster_probes", "PROBES", "16"), ("pf_idx_cluster_mult", "MULT", "4"),
                                ("pf_idx_cluster_cap", "CAP", "8192"), ("pf_idx_cluster_maxpct", "MAXPCT", "25"),
                                ("pf_idx_cluster_recall", "RECALL", "0")):
        f = FL.FLAGS[name]
        assert f.default == default and (f.attrs[0][0], f.attrs[0][1]) == ("pf_cluster", attr)
    assert FL.FLAGS["pf_idx_cluster"].default == "0"                            # approximate: off until the owner asks for it
