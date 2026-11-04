"""Certified decode indexer filter (kyojin_mlx.decfilter, lever dec_filter), CPU only.

The exact kernel (`dsv4_core._ISCORE`) and the stock select are emulated in numpy with the kernel's own arithmetic (fp32 FMA chain over the dim, tile rounded to the
dtype, relu, product rounded to the dtype, head sum in head order in fp32, result rounded to the dtype; value descending / lower index first / ascending / -1 tail).
The filter runs for real (mlx on the CPU device: the fp16 matmul, the block max, the gather, the certificate) with those two functions plugged in. Checks:
 - whenever the certificate holds, the selected keys equal the stock selection over ALL keys, bit for bit (and the certificate holds on the plain cases);
 - the per-key bound |a - exact fp32 total| <= delta on adversarial data;
 - the incremental key norms equal a from-scratch pass and never store an open-pass row."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import decfilter as DF  # noqa: E402

NH, DI, K, B = 32, 128, 32, 16


def exact_total(q, keys, w, order="kernel"):
    """q [NH, DI] f16, keys [W, DI] f16, w [NH] f16 -> (fp32 total before the final rounding [W], stored score f16 [W]) with the kernel's arithmetic.
    order "rev" / "chunk": the dim summed from the other end / in 8-wide partial sums, the heads in reverse: a different but legal fp32 order (what a
    matmul does), used to check the bound against real reordering differences."""
    q64, k64 = q.astype(np.float64), keys.astype(np.float64)
    acc = np.zeros((NH, keys.shape[0]), dtype=np.float32)
    dims = range(DI - 1, -1, -1) if order == "rev" else range(DI)
    if order == "chunk":
        for d0 in range(0, DI, 8):
            part = np.zeros_like(acc)
            for d in range(d0, d0 + 8):
                part = (q64[:, d, None] * k64[None, :, d] + part).astype(np.float32)
            acc = (acc + part).astype(np.float32)
    else:
        for d in dims:
            acc = (q64[:, d, None] * k64[None, :, d] + acc).astype(np.float32)       # fma: exact product, one rounding of the sum
    tile = acc.astype(q.dtype)
    rel = np.maximum(tile, np.zeros((), q.dtype))
    pr = (rel.astype(np.float32) * w.astype(np.float32)[:, None]).astype(q.dtype)
    total = np.zeros(keys.shape[0], dtype=np.float32)
    for h in (range(NH - 1, -1, -1) if order != "kernel" else range(NH)):
        total = (total + pr[h].astype(np.float32)).astype(np.float32)
    return total, total.astype(q.dtype)


def exact_fn(q, gk, w, lens):
    q, gk, w = np.array(q), np.array(gk), np.array(w)
    out = np.stack([exact_total(q[n], gk[n], w[n])[1] for n in range(q.shape[0])])
    return mx.array(out)


def topk_np(scores, lens, k):
    s, lens = np.array(scores), np.array(lens)
    out = np.full((s.shape[0], k), -1, dtype=np.int32)
    for n in range(s.shape[0]):
        L = int(lens[n])
        v = s[n, :L].astype(np.float64)
        order = np.lexsort((np.arange(L), -v))[:k]
        out[n, :min(k, L)] = np.sort(order)
    return mx.array(out)


def make(rng, N, T, kind="n", scale=1.0, wscale=0.05, signed=False):
    q = (rng.standard_normal((N, NH, DI)) * scale).astype(np.float16)
    keys = (rng.standard_normal((T, DI)) * scale).astype(np.float16)
    w = (rng.standard_normal((N, NH)) * wscale).astype(np.float16)
    if not signed:
        w = np.abs(w)
    if kind == "quant":
        q, keys = (np.round(q * 2) / 2).astype(np.float16), (np.round(keys * 2) / 2).astype(np.float16)
        w = (np.round(w * 64) / 64).astype(np.float16)
    if kind == "cluster":                                    # keys sit in a few tight groups around centres; one centre aligned with a query
        c = rng.standard_normal((40, DI)).astype(np.float32) * scale
        keys = (c[rng.integers(0, 40, size=T)] + rng.standard_normal((T, DI)) * 0.15 * scale).astype(np.float16)
    return q, keys, w


def run(q, keys, w, sess, lens, k=K, budget=None, tile=None):
    S = max(sess) + 1
    N, T0 = q.shape[0], keys.shape[0]
    T = -(-max(lens) // B) * B
    kk = np.zeros((S, T0, DI), dtype=np.float16)
    for s in range(S):
        kk[s] = np.roll(keys, 97 * s, axis=0)               # a different key set per session
    kmx = mx.array(kk)
    norms = mx.stack([DF.row_norms(kmx[s, :T]) for s in range(S)])
    idx, ok = DF.select(mx.array(q), kmx, mx.array(w), sess, lens, T, k, exact_fn, topk_np, norms, block=B, budget=budget, tile=tile)
    ref = np.full((N, k), -1, dtype=np.int32)
    for u in range(N):
        sc = exact_total(q[u], kk[sess[u], :lens[u]], w[u])[1]
        ref[u] = np.array(topk_np(mx.array(sc[None]), [lens[u]], k))[0]
    return np.array(idx), np.array(ok), ref


@pytest.mark.parametrize("kind", ["n", "cluster"])
def test_certified_equals_stock_plain(kind):
    rng = np.random.default_rng(1)
    q, keys, w = make(rng, 3, 4096, kind)
    idx, ok, ref = run(q, keys, w, [0, 0, 1], [4093, 4094, 3800], budget=24)
    assert ok.all(), ok
    assert np.array_equal(idx, ref)


@pytest.mark.parametrize("kind,scale,wscale,signed", [("n", 1.0, 0.05, True), ("n", 30.0, 0.01, True), ("n", 1e-3, 1.0, False), ("quant", 1.0, 0.05, False),
                                                      ("cluster", 1.0, 0.05, True), ("n", 4.0, 0.002, True)])
def test_whenever_certified_equal_to_stock(kind, scale, wscale, signed):
    rng = np.random.default_rng(7)
    q, keys, w = make(rng, 3, 3072, kind, scale, wscale, signed)
    idx, ok, ref = run(q, keys, w, [0, 0, 1], [3070, 3071, 2000], budget=16)
    for u in range(3):
        if ok[u]:
            assert np.array_equal(idx[u], ref[u]), (kind, u)


def test_short_rows_take_everything():
    rng = np.random.default_rng(3)
    q, keys, w = make(rng, 2, 1024)
    idx, ok, ref = run(q, keys, w, [0, 1], [20, 31], budget=40)      # fewer valid keys than k, a partial last block
    assert ok.all() and np.array_equal(idx, ref)
    assert (idx[:, 31:] == -1).all() and (idx[0, 20:] == -1).all()


def test_tiny_budget_fails_the_certificate_not_the_answer():
    rng = np.random.default_rng(5)
    q, keys, w = make(rng, 2, 4096)
    idx, ok, ref = run(q, keys, w, [0, 1], [4096, 4000], budget=1)   # one block per k: cannot hold the answer
    for u in range(2):
        assert (not ok[u]) or np.array_equal(idx[u], ref[u])
    assert not ok.all()


def test_tiles_do_not_change_the_result():
    rng = np.random.default_rng(11)
    q, keys, w = make(rng, 2, 4096)
    a = run(q, keys, w, [0, 0], [4090, 4095], budget=24, tile=4096)
    b = run(q, keys, w, [0, 0], [4090, 4095], budget=24, tile=1024)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("kind,scale,wscale,signed", [("n", 1.0, 0.05, True), ("n", 30.0, 0.01, True), ("n", 1e-3, 1.0, True), ("quant", 1.0, 0.05, True),
                                                      ("cluster", 3.0, 0.05, True)])
def test_bound_dominates_the_exact_total(kind, scale, wscale, signed):
    rng = np.random.default_rng(13)
    q, keys, w = make(rng, 1, 2048, kind, scale, wscale, signed)
    knorm = DF.row_norms(mx.array(keys))
    a, d = DF.approx_session(mx.array(q[0:1]), mx.array(keys), mx.array(w[0:1]), knorm)
    a, d = np.array(a)[0], np.array(d)[0]
    worst = 0.0
    for order in ("kernel", "rev", "chunk"):
        u, _ = exact_total(q[0], keys, w[0], order)
        worst = max(worst, float((np.abs(a - u) / d).max()))
    assert worst < 0.5, worst
    u, _ = exact_total(q[0], keys, w[0], "rev")
    assert np.abs(a - u).max() > 0 or kind == "quant"                 # not vacuous: the reordered emulation really differs from the matmul


def test_bound_bf16_dtype_constants():
    assert DF.eps_of(mx.bfloat16) == 2.0 ** -8 and DF.eps_of(mx.float16) == 2.0 ** -11
    rng = np.random.default_rng(17)
    q, keys, w = make(rng, 1, 512)
    qb, kb, wb = mx.array(q).astype(mx.bfloat16), mx.array(keys).astype(mx.bfloat16), mx.array(w).astype(mx.bfloat16)
    a, d = DF.approx_session(qb, kb, wb, DF.row_norms(kb))
    assert np.isfinite(np.array(a)).all() and (np.array(d) > 0).all()


def test_norms_incremental_equal_scratch_and_open_rows_not_stored():
    rng = np.random.default_rng(19)
    keys = mx.array(rng.standard_normal((2, 600, DI)).astype(np.float16))
    n = DF.Norms(2)
    first = n.get(keys, [100, 300], 512)
    assert n.have == [100, 300]
    assert np.array_equal(np.array(first), np.array(DF.row_norms(keys[:, :512])))
    changed = np.array(keys)
    changed[0, 150:200] = 1.0                                        # rows past the committed count (an open pass) are rewritten by the next pass
    keys2 = mx.array(changed)
    second = n.get(keys2, [140, 320], 512)
    assert n.have == [140, 320]
    assert np.array_equal(np.array(second), np.array(DF.row_norms(keys2[:, :512])))
    n.get(keys2, [10, 320], 512)                                     # the committed count went down (cache rebuilt): restart, no stale rows
    assert n.have[0] == 10


def test_flag_row_default_off():
    from kyojin_mlx import flags as FL
    assert FL.FLAGS["dec_filter"].default == "0" and FL.FLAGS["dec_filter_min"].kind == "int"
    assert DF.ENABLED is False


# ---------------------------------------------------------------------------------------------------------------- the candidate source (mask)
def stock_mask(scores, lens, kb, bs, cap):
    """select_candidate_blocks as the kernel defines it (tests/test_pf_flat.stock_select_blocks): block maxima over keys < lens, the block of the last key first
    whatever its maximum, the kb best blocks (ties to the lower block), a block kept only when its maximum is not -inf. bool [N, cap], valid for t < lens."""
    out = np.zeros((scores.shape[0], cap), dtype=bool)
    for n in range(scores.shape[0]):
        L = int(lens[n])
        nbk = -(-L // bs)
        sp = np.full(nbk * bs, -np.inf)
        sp[:L] = scores[n, :L].astype(np.float64)
        bm = sp.reshape(nbk, bs).max(-1)
        key = bm.copy()
        key[(L - 1) // bs] = np.inf
        order = np.argsort(-key, kind="stable")[:min(kb, nbk)]
        for b in order:
            if key[b] > -np.inf:
                out[n, b * bs:(b + 1) * bs] = True
    return out


def run_source(q, keys, w, sess, lens, k, kb, bs, budget, cap):
    S = max(sess) + 1
    N, T0 = q.shape[0], keys.shape[0]
    T = -(-max(lens) // bs) * bs
    kk = np.zeros((S, cap, DI), dtype=np.float16)
    for s in range(S):
        kk[s, :T0] = np.roll(keys, 97 * s, axis=0)
    kmx = mx.array(kk)
    norms = mx.stack([DF.row_norms(kmx[s, :T]) for s in range(S)])
    idx, mask, ok = DF.select(mx.array(q), kmx, mx.array(w), sess, lens, T, k, exact_fn, topk_np, norms, block=bs, budget=budget, kb=kb, cap=cap)
    ref_i = np.full((N, k), -1, dtype=np.int32)
    ref_s = np.full((N, cap), -np.inf)
    for u in range(N):
        sc = exact_total(q[u], kk[sess[u], :lens[u]], w[u])[1]
        ref_s[u, :lens[u]] = sc
        ref_i[u] = np.array(topk_np(mx.array(sc[None]), [lens[u]], k))[0]
    return np.array(idx), np.array(mask), np.array(ok), ref_i, stock_mask(ref_s, lens, kb, bs, cap)


@pytest.mark.parametrize("kind", ["n", "cluster"])
def test_source_mask_and_idx_equal_stock(kind):
    rng = np.random.default_rng(21)
    q, keys, w = make(rng, 3, 4096, kind)
    lens = [4093, 4094, 3801]                                  # partial last blocks, rows of one session at consecutive positions, a shorter second session
    idx, mask, ok, ri, rm = run_source(q, keys, w, [0, 0, 1], lens, K, 24, 8, 3, 4352)
    assert ok.all(), ok
    assert np.array_equal(idx, ri)
    for u in range(3):
        assert np.array_equal(mask[u, :lens[u]], rm[u, :lens[u]])


@pytest.mark.parametrize("lens", [[20, 31], [300, 5], [200, 513]])
def test_source_short_rows_fewer_blocks_than_kb(lens):
    rng = np.random.default_rng(23)
    q, keys, w = make(rng, 2, 1024)
    idx, mask, ok, ri, rm = run_source(q, keys, w, [0, 1], lens, K, 24, 8, 40, 1024)
    for u in range(2):
        if ok[u]:
            assert np.array_equal(idx[u], ri[u])
            assert np.array_equal(mask[u, :lens[u]], rm[u, :lens[u]])
    assert ok.all()                                            # everything is gathered: nothing left out, the certificate is trivial


def test_source_whenever_certified_equal_to_stock_ties():
    rng = np.random.default_rng(25)
    q, keys, w = make(rng, 3, 3072, "quant")
    lens = [3070, 3071, 2000]
    idx, mask, ok, ri, rm = run_source(q, keys, w, [0, 0, 1], lens, K, 24, 8, 2, 3328)
    for u in range(3):
        if ok[u]:
            assert np.array_equal(idx[u], ri[u]) and np.array_equal(mask[u, :lens[u]], rm[u, :lens[u]])


def test_source_tiny_budget_never_certifies_a_wrong_answer():
    rng = np.random.default_rng(27)
    q, keys, w = make(rng, 2, 4096)
    idx, mask, ok, ri, rm = run_source(q, keys, w, [0, 1], [4096, 4000], K, 24, 8, 1, 4096)
    assert not ok.all()
    for u in range(2):
        if ok[u]:
            assert np.array_equal(idx[u], ri[u]) and np.array_equal(mask[u, :4000], rm[u, :4000])


# ---------------------------------------------------------------------------------------------------------------- the stateful wiring (layer, Norms across passes)
from types import SimpleNamespace  # noqa: E402


def _stub_idxr(source, k=K, kb=24, bs=8):
    return SimpleNamespace(index_topk=k, is_candidate_source=source, candidate_topk_blocks=kb, candidate_block_size=bs, uses_candidates=False, index_head_dim=DI, n_heads=NH)


def _drive(monkeypatch, source, steps=6, sess_rows=((3, 2),)):
    """Simulated verify passes on a stacked key buffer [S, cap, DI]. Each pass writes the keys of its rows (positions past the committed count), asks `DF.layer` for the
    selection of every row, compares with the stock emulation over ALL keys, then commits only the accepted rows; the next pass overwrites the rejected ones with different keys."""
    monkeypatch.setattr(DF, "default_exact", exact_fn)
    monkeypatch.setattr(DF, "default_topk", topk_np)
    monkeypatch.setattr(DF, "MIN_KEYS", 64)
    monkeypatch.setattr(DF, "BUDGET", 3)
    monkeypatch.setattr(DF, "BLOCK", B)
    rng = np.random.default_rng(31)
    S, cap = len(sess_rows), 1536
    keys = np.zeros((S, cap, DI), dtype=np.float16)
    committed = [1000 + 37 * s for s in range(S)]
    for s in range(S):
        keys[s, :committed[s]] = rng.standard_normal((committed[s], DI)).astype(np.float16)
    idxr = _stub_idxr(source)
    st = SimpleNamespace(kcnt={7: list(committed)})
    served = checked = 0
    for step in range(steps):
        sess, lens, rows_q = [], [], []
        for s, nrows in enumerate(sess_rows):
            keys[s, committed[s]:committed[s] + nrows + 2] = rng.standard_normal((nrows + 2, DI)).astype(np.float16)       # this pass's keys (rejected ones are rewritten next time)
            for r in range(nrows):
                sess.append(s)
                lens.append(committed[s] + r + 1)
        N = len(sess)
        q = rng.standard_normal((N, NH, DI)).astype(np.float16)
        w = np.abs(rng.standard_normal((N, NH))).astype(np.float16) * np.float16(0.05)
        plan = SimpleNamespace(sess=sess)
        em = SimpleNamespace(tmax=max(lens), lens=lens)
        res = DF.layer(st, 7, idxr, mx.array(q), mx.array(keys), mx.array(w), plan, em, cap)
        ref_i, ref_m = [], []
        for u in range(N):
            sc = exact_total(q[u], keys[sess[u], :lens[u]], w[u])[1]
            ref_i.append(np.array(topk_np(mx.array(sc[None]), [lens[u]], K))[0])
            if source:
                full = np.full((1, cap), -np.inf)
                full[0, :lens[u]] = sc
                ref_m.append(stock_mask(full, [lens[u]], 24, 8, cap)[0])
        if res is not None:
            served += 1
            idx, mask = np.array(res[0]), res[1]
            for u in range(N):
                assert np.array_equal(idx[u], ref_i[u]), (step, u)
                if source:
                    assert np.array_equal(np.array(mask)[u, :lens[u]], ref_m[u][:lens[u]]), (step, u)
                checked += 1
        for s, nrows in enumerate(sess_rows):                    # commit the accepted prefix only
            committed[s] += 1 + (step % nrows)
        st.kcnt[7] = list(committed)
    return served, checked, st


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("sess_rows", [((3,),), ((3,), (2,))])
def test_layer_over_simulated_passes_equals_stock(monkeypatch, source, sess_rows):
    sess_rows = tuple(s[0] for s in sess_rows)
    served, checked, st = _drive(monkeypatch, source, sess_rows=sess_rows)
    assert served >= 5 and checked >= 5 * sum(sess_rows)        # the certificate holds on the plain synthetic keys at nearly every pass; every served pass equals the stock selection
    nrm = st.__dict__["fnorm"][7]
    assert nrm.have == st.kcnt[7][:len(nrm.have)] or all(h <= c for h, c in zip(nrm.have, st.kcnt[7]))


def test_can_sync_is_off_inside_a_gated_step_and_in_replay(monkeypatch):
    from kyojin_mlx import dsv4_core as DC, engram_mlx as EM
    assert DF.can_sync()
    monkeypatch.setattr(EM, "DEFER_DEPTH", 1)
    assert not DF.can_sync()
    monkeypatch.setattr(EM, "DEFER_DEPTH", 0)
    monkeypatch.setattr(DC, "_CANON_POOLS", True)
    assert not DF.can_sync()
    monkeypatch.setattr(DC, "_CANON_POOLS", False)
    monkeypatch.setattr(DF, "ENABLED", True)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    q = mx.zeros((1, NH, DI), dtype=mx.float16)
    keys = mx.zeros((1, 400000, DI), dtype=mx.float16)
    assert DF.applies(_stub_idxr(False), keys, q, 300000, 1)
    monkeypatch.setattr(EM, "DEFER_DEPTH", 2)
    assert not DF.applies(_stub_idxr(False), keys, q, 300000, 1)
    cons = _stub_idxr(False)
    cons.uses_candidates = True
    monkeypatch.setattr(EM, "DEFER_DEPTH", 0)
    assert not DF.applies(cons, keys, q, 300000, 1)


def test_deferred_context_counts_depth():
    from kyojin_mlx import engram_mlx as EM
    import inspect
    src = inspect.getsource(EM.EngramStage.deferred)
    assert "DEFER_DEPTH += 1" in src and "DEFER_DEPTH -= 1" in src
