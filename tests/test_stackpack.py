"""stackpack: the packed stack of the served batched decode (flag KYOJIN_STACKPACK, default 0). CPU only: the batched core itself is Metal only, so this file
proves the parts that run here (stack, _put, commit-back, the gather and the remap under the kernels' index rule, the packed key load expression, retained bytes)
and tools/cstackpack_mac.py proves the logits of real batched steps."""
import os
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import dsv4_core as DC, fp4pack, idxfuse as IXF, longctx as LC, stackpack as SP  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def eq(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def grid(shape, block, e4m3, seed=0, dtype=mx.float16):
    """Random rows ON the fp4 grid of (block, e4m3): what the producers (fake_quant_fp4_e4m3, rot_fq4) hand the cache."""
    x = mx.random.normal(shape, key=mx.random.key(seed)) * 3.0
    c, s = fp4pack.pack(x.astype(dtype), block, e4m3)
    return fp4pack.unpack(c, s, block, e4m3, dtype)


# ------------------------------------------------------------------------------------------------------------------------ registry
def test_flag_registered_default_off_and_switchable():
    from kyojin_mlx import flags
    f = flags.FLAGS["stackpack"] if hasattr(flags, "FLAGS") else None
    assert f is not None and f.env == "KYOJIN_STACKPACK" and f.default == "0" and f.runtime
    assert SP.ENABLED is False or os.environ.get("KYOJIN_STACKPACK", "0") not in ("0", "")


def test_levers_row_present():
    txt = open(os.path.join(REPO, "docs", "cloud", "LEVERS.md")).read()
    assert "`KYOJIN_STACKPACK` | stackpack |" in txt


# ------------------------------------------------------------------------------------------------------------------------ _put
def plan_em(pos, counts, r):
    return DC.Plan(pos, counts).emit(r)


@pytest.mark.parametrize("canon", [False, True])
@pytest.mark.parametrize("block,e4m3,D", [(16, True, 64), (32, False, 64)])
def test_packed_put_equals_pack_of_dense_put(monkeypatch, canon, block, e4m3, D):
    """Two emitting passes, the second one growing the buffer past its capacity: the packed stack unpacks to exactly the dense buffer, row for row."""
    monkeypatch.setattr(DC, "_CANON_POOLS", canon)
    step, S, r = 8, 3, 4
    dense, packed = [None], [None]
    pos_a, pos_b = [3, 7, 11], [3 + 4 * 9, 7 + 4 * 9, 11 + 4 * 9]
    for k, (pos, seed) in enumerate(((pos_a, 1), (pos_b, 2))):
        em = plan_em(pos, [1, 1, 1], r)
        vals = grid((em.n, D), block, e4m3, seed)
        DC._put(dense, 0, S, (D,), mx.float16, em, vals, step)
        DC._put(packed, 0, S, (D,), mx.float16, em, vals, step, (block, e4m3))
        assert SP.is_packed(packed[0]) and dense[0].shape == packed[0].shape
        # rows the passes wrote are equal; the zero rows are too
        assert eq(packed[0].dense(), dense[0])
    assert dense[0].shape[1] > step                                      # the growth path ran


def test_dense_put_untouched_when_not_packed():
    em = plan_em([3, 7], [1, 1], 4)
    lst = [None]
    DC._put(lst, 0, 2, (64,), mx.float16, em, grid((em.n, 64), 16, True), 8)
    assert isinstance(lst[0], mx.array)


# ------------------------------------------------------------------------------------------------------------------------ gather + remap
def kernel_read(pool_rows, sess, idx, cap):
    """The rows a kernel of dsv4_core / fuse_attn reads: for each row and each list entry, pool[sess[r], idx] if 0 <= idx < cap, else nothing (None)."""
    pool_rows = np.array(pool_rows.astype(mx.float32))
    out = []
    for r in range(idx.shape[0]):
        out.append([None if (e < 0 or e >= cap) else pool_rows[sess[r], e] for e in idx[r].tolist()])
    return out


@pytest.mark.parametrize("counts", [[1, 1, 1], [3, 1, 2], [2, 2]])
def test_gather_slab_reads_the_same_rows_in_the_same_order(counts):
    S, cap, D, K = len(counts), 40, 64, 6
    ps_codes = fp4pack.pack(grid((S, cap, D), 16, True, 5), 16, True)
    ps = SP.PackedStack(ps_codes[0], ps_codes[1], mx.float16, 16, True)
    dense = ps.dense()
    n = sum(counts)
    sess = [s for s, c in enumerate(counts) for _ in range(c)]
    first = [sum(counts[:s]) for s in range(S)]
    rows_in = [i - first[sess[i]] for i in range(n)]
    rng = np.random.default_rng(3)
    idx = rng.integers(-1, cap, size=(n, K)).astype(np.int32)
    idx[0, 0], idx[0, 1], idx[-1, -1] = cap, cap + 17, -1                 # outside the buffer: the kernels' own bound
    pool, idx2 = SP.gather_slab(ps, mx.array(idx), mx.array(sess, dtype=mx.int32), mx.array(rows_in, dtype=mx.int32), S, max(counts))
    assert pool.shape[0] == S and pool.shape[1] == max(counts) * K
    want = kernel_read(dense, sess, idx, cap)
    got = kernel_read(pool, sess, np.array(idx2), pool.shape[1])
    for w_row, g_row in zip(want, got):
        assert len(w_row) == len(g_row)
        for w, g in zip(w_row, g_row):
            assert (w is None) == (g is None) and (w is None or np.array_equal(w, g))


# ------------------------------------------------------------------------------------------------------------------------ packed key load
def emulate_pk_key(codes, scales, d, dtype):
    """pk_key of idxfuse._PK_HELPER in numpy, for the element d of one key: nibble, LUT, sign, as_type<float>(s << 23), one fp32 product, one cast."""
    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    b = int(codes[d >> 1])
    c = (b >> 4) if (d & 1) else (b & 15)
    mag = lut[c & 7]
    v = -mag if (c & 8) else mag
    sc = np.array([int(scales[d >> 5]) << 23], np.uint32).view(np.float32)[0]
    return np.float32(dtype(np.float32(v * sc)))


@pytest.mark.parametrize("dtype,npd", [(mx.float16, np.float16), (mx.float32, np.float32)])
def test_packed_key_expression_equals_unpack(dtype, npd):
    DI = 128
    x = grid((64, DI), 32, False, 9, dtype)
    # exercise the extremes of the exponent byte as well
    x = mx.concatenate([x, x * 2.0 ** -9, x * 2.0 ** 8])
    c, s = fp4pack.pack(x, 32, False)
    ref = np.array(fp4pack.unpack(c, s, 32, False, dtype).astype(mx.float32))
    cn, sn = np.array(c), np.array(s)
    for j in range(0, cn.shape[0], 5):
        got = np.array([emulate_pk_key(cn[j], sn[j], d, npd) for d in range(DI)], np.float32)
        assert np.array_equal(got.view(np.uint32), ref[j].view(np.uint32))


def test_packed_key_kernel_text_keeps_the_dense_form():
    dense_scores, dense_fused = IXF._SCORES, IXF._source(512, 0, 8, False, 2)
    for src in (dense_scores, dense_fused):
        assert src.count("float(kp[j * DI + dc + i])") == 1 and "kcodes" not in src          # flag 0: the old text, untouched
        pk = IXF.pack_source(src)
        assert pk.count("pk_key<T>(kc, ksc, j, dc + i, DI)") == 1 and pk.count("kcodes") == 1 and "kp[" not in pk
        assert pk.count("kscales") == 1
    assert "float(kp[j * DI + dc + i])" in IXF._SCORES


# ------------------------------------------------------------------------------------------------------------------------ the stack on the tiny model
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import fp4pool as FP, rows as R  # noqa: E402
from kyojin_mlx.serve.backends import arrays_of  # noqa: E402


def build(n, lens, flag):
    SP.ENABLED = flag
    lm, _ = M.make(sliding_window=16)
    users = []
    for i in range(n):
        c = FP.wrap_cache(lm.make_cache())
        mx.eval(lm(mx.array([M.tokens(lens[i], seed=1 + i)]), cache=c).logits, *c[0].state)
        users.append(c)
    return users


def dense_of_rows(st):
    out = []
    for r in st.rows:
        out.append([None if x is None else mx.array(x) for k in ("compress", "keys", "window") for x in getattr(r, k)])
    mx.eval(*[x for b in out for x in b if x is not None])
    return out


@pytest.fixture(autouse=True)
def _restore_flag():
    old = SP.ENABLED
    yield
    SP.ENABLED = old


@pytest.mark.parametrize("lens", [[320, 320, 320], [300, 450, 129], [480, 7, 260]])
def test_stack_then_writeback_is_bit_identical_ragged(lens):
    """Ragged lengths (one session with almost no pooled rows): the packed stack, written back, gives every layer bit for bit; same cap, same counts."""
    ref_users = build(len(lens), lens, False)
    st0 = DC.stack_rows(R.batch(ref_users)[0])
    caps0 = [None if a is None else a.shape for a in st0.cmp + st0.keys]
    cnt0 = (st0.cnt, st0.kcnt)
    st0.sync_rows()
    want = dense_of_rows(st0)
    users = build(len(lens), lens, True)
    st1 = DC.stack_rows(R.batch(users)[0])
    assert st1.packed and st1._dirty
    assert any(SP.is_packed(a) for a in st1.cmp + st1.keys)
    caps1 = [None if a is None else a.shape for a in st1.cmp + st1.keys]
    assert caps0 == caps1 and cnt0 == (st1.cnt, st1.kcnt)
    mx.eval(*st1.stacked_arrays())
    st1.sync_rows()
    got = dense_of_rows(st1)
    for w, g in zip(want, got):
        assert len(w) == len(g)
        for x, y in zip(w, g):
            assert (x is None) == (y is None) and (x is None or eq(x, y))


def test_one_session_keeps_the_current_path():
    SP.HITS.update({k: 0 for k in SP.HITS})
    users = build(1, [320], True)
    st = DC.stack_rows(R.batch(users)[0])
    assert not st.packed and not st._dirty and SP.HITS["stack"] == 0
    assert not any(SP.is_packed(a) for a in st.cmp + st.keys)
    assert all(isinstance(a, mx.array) for a in st.cmp + st.keys if a is not None)


def test_flag_off_builds_dense_stack():
    users = build(2, [320, 320], False)
    st = DC.stack_rows(R.batch(users)[0])
    assert not st.packed and not any(SP.is_packed(a) for a in st.cmp + st.keys) and not st._dirty


def test_dense_layer_falls_back_for_that_layer_only():
    users = build(2, [320, 320], True)
    # session 1 holds layer l dense (an off-grid layer: PackedSlots keeps it as a plain array): the layer stays dense, the others pack
    bc = R.batch(users)
    rows = bc[0].rows
    l = next(i for i in range(len(rows[0].compress)) if rows[1].compress.raw(i) is not None)
    rows[1].compress.set_raw(l, rows[1].compress[l])
    st = DC.stack_rows(bc[0])
    assert isinstance(st.cmp[l], mx.array)
    assert any(SP.is_packed(a) for i, a in enumerate(st.cmp) if i != l and a is not None)


def test_retained_bytes_drop_and_mirror_is_not_built():
    from kyojin_mlx import depthflat as DF
    n, lens = 3, [900, 900, 900]

    def held(flag):
        users = build(n, lens, flag)
        bc = R.batch(users)
        mx.eval()
        b0 = DF.HITS["build"]
        st = DC.stack_rows(bc[0])
        mx.eval(*st.stacked_arrays())
        return sum(a.nbytes for a in arrays_of(bc)), sum(a.nbytes for a in st.stacked_arrays()), DF.HITS["build"] - b0
    c0, s0, m0 = held(False)
    c1, s1, m1 = held(True)
    print(f"\ncounted {c0} -> {c1} ({c1 / c0:.3f}x); stacked arrays {s0} -> {s1} ({s1 / s0:.3f}x); mirrors built by the stack {m0} -> {m1}")
    assert c1 < c0 and s1 < s0
    assert m1 == 0


def test_session_entry_is_a_session_entry_without_unpack():
    ps_codes = fp4pack.pack(grid((2, 24, 64), 16, True, 4), 16, True)
    ps = SP.PackedStack(ps_codes[0], ps_codes[1], mx.float16, 16, True)
    e = SP.session_entry(ps, 1, 10)
    assert e.n == 10 and e.codes.shape == (1, 10, 32) and e.dtype == mx.float16
    assert eq(fp4pack.unpack(e.codes, e.scales, 16, True, mx.float16), ps.dense()[1:2, :10])
