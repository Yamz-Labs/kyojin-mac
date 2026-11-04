"""cdecpack, hooked at the read: the served decode reads the packed pool and index keys, gathers the selected rows, and keeps no dense mirror.

CPU: the read (`view`), the write (`put` against the dense scatter of `_put`), the stacked state (`stack_rows` / `commit_layer` / `sync_rows` round trip), the
ledger (no mirror, packed bytes only) and the compact slab against the dense rows. Metal: the served batched core (`dsv4_core.core`) of a compressing layer with
the flag 0 against 1, and `dsv4_core.attend` on the compact slab against the dense pool, both bit for bit."""
import os
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import cdecpack as CD, depthflat as DF, dsv4_core as DC, fp4pack, fp4pool, stackpack as SP  # noqa: E402
from kyojin_mlx.fp4pool import KeySlots, PoolSlots  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(autouse=True)
def _no_slot_stack(monkeypatch):
    monkeypatch.setenv("KYOJIN_MU_KEEP", "0")                             # this lever's one-session path; a slot stack (mu_keep) keeps the dense form


def eq(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def grid(shape, block, e4m3, seed=0, dtype=mx.float16):
    """Random rows ON the fp4 grid of (block, e4m3): what the producers (fake_quant_fp4_e4m3, rot_fq4) hand the cache."""
    x = mx.random.normal(shape, key=mx.random.key(seed)) * 3.0
    c, s = fp4pack.pack(x.astype(dtype), block, e4m3)
    return fp4pack.unpack(c, s, block, e4m3, dtype)


def packed(cls, n, D, block, e4m3, seed=0):
    """Slots holding one layer of `n` rows exactly as the cache does after a prefill append, without the dense read an append takes."""
    s = cls([None])
    s.set_raw(0, fp4pool._Pk(*fp4pack.pack(grid((1, n, D), block, e4m3, seed), block, e4m3), n, mx.float16))
    return s


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(DF, "ENABLED", True)
    monkeypatch.setattr(DF, "MIRROR_BYTES", 1 << 30)
    monkeypatch.setattr(fp4pool, "MEMO_BYTES", 0)
    for k in DF.HITS:
        DF.HITS[k] = 0
    for k in CD.HITS:
        CD.HITS[k] = 0
    yield


# ------------------------------------------------------------------------------------------------------------------------ registry
def test_flag_registered_default_off_and_documented():
    from kyojin_mlx import flags
    f = flags.FLAGS["cdecpack"]
    assert f.env == "KYOJIN_CDECPACK" and f.default == "0" and f.runtime
    txt = open(os.path.join(REPO, "docs", "cloud", "LEVERS.md")).read()
    assert "`KYOJIN_CDECPACK` | cdecpack |" in txt


# ------------------------------------------------------------------------------------------------------------------------ the read
@pytest.mark.parametrize("cls,D,block,e4", [(PoolSlots, 64, 16, True), (KeySlots, 128, 32, False)])
def test_view_is_the_codes_of_the_session_with_no_mirror(cls, D, block, e4):
    s = packed(cls, 40, D, block, e4)
    e = s.raw(0)
    v = CD.view(s, 0)
    assert isinstance(v, fp4pool.PackedView) and v.block == block and v.e4m3 == e4
    assert (v.codes is e.codes or eq(v.codes, e.codes)) and v.n == 40 and v.stride == e.codes.shape[1]
    assert e.mirror is None and e.memo is None and DF.HITS["build"] == 0
    assert eq(v.dense(), fp4pack.unpack(e.codes[:, :40], e.scales[:, :40], block, e4, mx.float16))
    assert CD.HITS["read"] == 1 and CD.HITS["drop"] == 0 and v[:1, :17].n == 17 and v[:1, :17].block == block


def test_view_drops_the_mirror_the_prefill_read_built():
    """The dense read during prefill (append_rows -> _dec) leaves the mirror resident; the first decode read drops it, and no later read rebuilds it."""
    s = PoolSlots([None])
    s.append_rows(0, grid((1, 40, 64), 16, True))                          # the prefill append: builds the mirror
    assert s.raw(0).mirror is not None and DF.HITS["build"] == 1
    v = CD.view(s, 0)
    assert s.raw(0).mirror is None and s.raw(0).memo is None and CD.HITS["drop"] == 1
    assert eq(CD.compact(v, mx.array([[0, 39]], dtype=mx.int32))[0][0][1],
              fp4pack.unpack(s.raw(0).codes[0][39], s.raw(0).scales[0][39], 16, True, mx.float16))
    assert DF.HITS["build"] == 1 and s.raw(0).mirror is None                # the gather does not put the mirror back


def test_view_refuses_a_layer_without_a_packed_form():
    s = PoolSlots([None])
    s.set_raw(0, mx.zeros((1, 4, 40), dtype=mx.float16))                 # 40 % 16 != 0: the layer stays dense
    assert CD.view(s, 0) is None and CD.HITS["skip"] == 1


# ------------------------------------------------------------------------------------------------------------------------ the write
@pytest.mark.parametrize("cls,D,block,e4", [(PoolSlots, 64, 16, True), (KeySlots, 128, 32, False)])
def test_put_equals_the_dense_scatter_and_grows_in_place(cls, D, block, e4):
    """Two emitting passes of a one-session decode: the packed scatter into the session's own codes unpacks to exactly the dense buffer `_put` builds."""
    s = packed(cls, 3, D, block, e4)
    v = CD.view(s, 0)
    dense, held = [v.dense()], [v]
    r, step = 2, 8
    for k, (pos, counts) in enumerate((([5, 6], [2]), ([7, 8, 9, 10], [4]), ([11, 12], [2]))):
        em = DC.Plan(pos, counts).emit(r)
        vals = grid((em.n, D), block, e4, seed=k + 1)
        DC._put(dense, 0, 1, (D,), mx.float16, em, vals, step)
        DC._put(held, 0, 1, (D,), mx.float16, em, vals, step)
        assert CD.is_view(held[0]) and CD.HITS["put"] == k + 1
        n = int(max(em.widx)) + 1
        assert held[0].n == n and held[0].stride >= n
        got = fp4pack.unpack(held[0].codes[:, :n], held[0].scales[:, :n], block, e4, mx.float16)
        assert eq(got, dense[0][:, :n])                                  # same rows, same bits
        assert eq(s.raw(0).codes[:, :n], held[0].codes[:, :n])            # written through to the session's own entry
    assert s.raw(0).mirror is None and DF.HITS["build"] == 1
    assert s.raw(0).n == 3                                                # the valid count is the commit's, not the write's


def test_settle_keeps_the_committed_count_and_the_written_rows():
    s = packed(PoolSlots, 2, 64, 16, True)
    v = CD.view(s, 0)
    em = DC.Plan([4, 5, 6, 7], [4]).emit(2)                               # a 4-row pass: 2 groups close
    vals = grid((em.n, 64), 16, True, seed=5)
    DC._put([v], 0, 1, (64,), mx.float16, em, vals, 8)
    assert CD.HITS["put"] == 1
    v2 = CD.settle(v, [2])                                                # a commit that keeps 2 rows (one group accepted)
    assert v2.n == 2 and s.raw(0).n == 2 and CD.HITS["settle"] == 1
    assert eq(v2.codes[:, :2], v.codes[:, :2])                            # the accepted rows are untouched
    assert fp4pool.stride(v2) == v2.codes.shape[1]


# ------------------------------------------------------------------------------------------------------------------------ the stacked state
class Row:
    step = 8

    def reset_handoff(self):
        pass


class RowCache:
    """The per-session cache surface `stack_rows` / `sync_rows` / `commit_layer` read (window, packed pool and keys, compressor state)."""

    def __init__(self, layers, D=64, KD=128):
        self.rows = [Row()]
        r = self.rows[0]
        r.window = [None] * layers
        r.compress, r.keys = PoolSlots([None] * layers), KeySlots([None] * layers)
        r._compress_buffers, r._key_buffers = [None] * layers, [None] * layers
        r.kv_state, r.score_state = [None] * layers, [None] * layers
        r.offset = 0
        r.engram = None


def _stacked_bytes(st):
    return sum(a.nbytes for a in SP.arrays_of(st.cmp[0])) + sum(a.nbytes for a in SP.arrays_of(st.keys[0]))


def test_stack_rows_one_session_holds_the_codes_and_no_mirror(monkeypatch):
    monkeypatch.setattr(CD, "ENABLED", True)
    c = RowCache(1)
    c.rows[0].compress, c.rows[0].keys = packed(PoolSlots, 300, 64, 16, True), packed(KeySlots, 300, 128, 32, False)
    st = DC.stack_rows(c)
    assert CD.is_view(st.cmp[0]) and CD.is_view(st.keys[0])
    assert st.cnt == [[300]] and st.kcnt == [[300]]
    assert c.rows[0].compress.raw(0).mirror is None and c.rows[0].keys.raw(0).mirror is None and DF.HITS["build"] == 0
    assert _stacked_bytes(st) == 300 * (32 + 4 + 64 + 4)                 # 36 B pool row + 68 B key row packed, against 128 + 256 dense
    walk = fp4pool.retained_walk([c, st])                                # the view aliases the session's own entry: counted once, no mirror
    assert walk["mirror"] == 0 and walk["memo"] == 0 and walk["counted"] == 300 * (32 + 4 + 64 + 4)
    assert st.packed is False and CD.HITS["read"] == 2
    # a step that emits one pool row and one key row writes into the session's own codes and syncs nothing back
    em = DC.Plan([300], None).emit(2)
    DC._put(st.cmp, 0, 1, (64,), mx.float16, em, grid((1, 64), 16, True, seed=9), 8, None)
    DC._put(st.keys, 0, 1, (128,), mx.float16, em, grid((1, 128), 32, False, seed=9), 8, None)
    st.pend[0] = SimpleNamespace(win=None, kvs=None, scs=None, cmp=st.cmp[0], keys=st.keys[0], r=2, cnt=None, kcnt=None, kvr=None, kvrow=None, scrow=None)
    st.commit_layer(0, None, [300])
    assert st.cnt == [[150]] and st.cmp[0].n == 150
    st._dirty = True
    st.sync_rows()
    e, k = c.rows[0].compress.raw(0), c.rows[0].keys.raw(0)
    assert e.n == 150 and k.n == 150 and e.mirror is None and k.mirror is None
    assert eq(e.codes[:, :150], st.cmp[0].codes[:, :150]) and eq(k.codes[:, :150], st.keys[0].codes[:, :150])


def test_stacked_state_round_trip_equals_the_dense_path(monkeypatch):
    """The same three decode steps through the flag 0 stack (dense buffers, write-back) and the flag 1 stack (the session's codes): same rows, same bits."""
    def run(on):
        monkeypatch.setattr(CD, "ENABLED", on)
        for k in DF.HITS:
            DF.HITS[k] = 0
        c = RowCache(1)
        c.rows[0].compress, c.rows[0].keys = packed(PoolSlots, 5, 64, 16, True), packed(KeySlots, 5, 128, 32, False)
        for pos, vals in ((6, grid((1, 64), 16, True, seed=20)), (8, grid((1, 64), 16, True, seed=21)), (10, grid((1, 64), 16, True, seed=22))):
            st = DC.stack_rows(c)
            em = DC.Plan([pos], None).emit(2)
            DC._put(st.cmp, 0, 1, (64,), mx.float16, em, vals, 8, None)
            st.pend[0] = SimpleNamespace(win=None, kvs=None, scs=None, cmp=st.cmp[0], keys=None, r=2, cnt=None, kcnt=None, kvr=None, kvrow=None, scrow=None)
            st.commit_layer(0, None, [pos])
            st._dirty = True
            st.sync_rows()
            st.unstack()
        e = c.rows[0].compress.raw(0)
        return np.array(e.codes[:, :e.n]), np.array(e.scales[:, :e.n]), DF.HITS["build"]

    off = run(False)
    on = run(True)
    assert np.array_equal(off[0], on[0]) and np.array_equal(off[1], on[1])       # bit for bit, same packed rows
    assert off[2] > 0                                                           # flag 0 reads through the depthflat mirror
    assert on[2] == 0                                                           # flag 1 never builds one


# ------------------------------------------------------------------------------------------------------------------------ the attention input
def test_compact_slab_equals_the_dense_rows_it_replaces():
    s = packed(PoolSlots, 700, 64, 16, True)
    v = CD.view(s, 0)
    dense = np.array(fp4pack.unpack(v.codes[:, :v.n], v.scales[:, :v.n], 16, True, mx.float16)[0])
    idx = mx.array([[3, 699, 0, -1, 250, 700, 9999, 3], [5, 6, 7, 8, -1, -1, 120, 640]], dtype=mx.int32)
    rows, remap = CD.compact(v, idx)
    r, i, m = np.array(rows[0]), np.array(idx), np.array(remap)
    for a in range(2):
        for b in range(8):
            if 0 <= i[a, b] < 700:
                assert m[a, b] >= 0 and np.array_equal(r[m[a, b]], dense[i[a, b]])
            else:
                assert m[a, b] == -1
    assert CD.HITS["compact"] == 1 and CD.HITS["rows"] == 16
    assert s.raw(0).mirror is None and DF.HITS["build"] == 0


# ------------------------------------------------------------------------------------------------------------------------ Metal: the served core
D, W, H, QL, G, OL = 512, 128, 4, 256, 2, 64
metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


class _WoA:
    def __init__(self, rng):
        self.w = mx.array((rng.standard_normal((G, (H // G) * D, OL)) * 0.02).astype(np.float16))

    def __call__(self, x):
        return mx.stack([x[:, g] @ self.w[g] for g in range(G)], axis=1)


def make_attn(seed=0, ratio=2):
    """The DeepseekV41Attention surface `dsv4_core.core` reads: dense linears, the real rope, a real Compressor and Indexer on a one-layer config."""
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    from mlx_vlm.models.deepseek_v41.language import Compressor, DeepseekV4RoPE, Indexer
    rng = np.random.default_rng(seed)
    cfg = ModelConfig(hidden_size=D, head_dim=D, q_lora_rank=QL, num_attention_heads=H, o_groups=G, num_hidden_layers=1,
                      compress_ratios=[ratio], kv_source_layer_ids=[0], index_source_layer_ids=[0], index_n_heads=32, index_head_dim=128,
                      index_topk=8, candidate_source_layer_id=-1, qk_rope_head_dim=64, rope_theta=10000.0)
    a = SimpleNamespace(layer_idx=0, n_heads=H, head_dim=D, window_size=W, config=cfg, compress_ratio=ratio, is_kv_source=True,
                        is_index_source=True, o_groups=G, scale=D ** -0.5, attn_sink=mx.zeros((H,), dtype=mx.float32),
                        rope=DeepseekV4RoPE(64, 10000, None), compressor=Compressor(cfg, 0), indexer=Indexer(cfg, 0),
                        _latent_theta=cfg.compress_rope_theta, _latent_yarn=(1, 32, 1, 65536))
    a.wq_a, a.wq_b = nn.Linear(D, QL, bias=False), nn.Linear(QL, H * D, bias=False)
    a.wkv, a.wo_b = nn.Linear(D, D, bias=False), nn.Linear(G * OL, D, bias=False)
    a.q_norm, a.kv_norm = nn.RMSNorm(QL), nn.RMSNorm(D)
    a.wo_a = _WoA(rng)
    for name in ("wq_a", "wq_b", "wkv", "wo_b"):
        getattr(a, name).weight = mx.array((rng.standard_normal(getattr(a, name).weight.shape) * 0.05).astype(np.float16))
    for name in ("q_norm", "kv_norm"):
        getattr(a, name).weight = mx.array((rng.standard_normal(getattr(a, name).weight.shape) * 0.05).astype(np.float16))
    for mod, scale in ((a.compressor.wkv, 0.05), (a.compressor.wgate, 0.05), (a.indexer.wq_b, 0.05), (a.indexer.weights_proj, 0.05), (a.indexer.wk, 0.05)):
        mod.weight = mx.array((rng.standard_normal(mod.weight.shape) * scale).astype(np.float16))
    return a


def _sess(n_pool, D=D, KD=128, seed=0):
    """One session cache: the pool / keys / compressor state `core` reads, all on the fp4 grid."""
    c = RowCache(1, D, KD)
    r = c.rows[0]
    if n_pool:
        r.compress.set_raw(0, fp4pool._Pk(*fp4pack.pack(grid((1, n_pool, D), 16, True, seed), 16, True), n_pool, mx.float16))
        r.keys.set_raw(0, fp4pool._Pk(*fp4pack.pack(grid((1, n_pool, KD), 32, False, seed), 32, False), n_pool, mx.float16))
        r.kv_state[0] = grid((1, 2, D), 16, True, seed=seed + 1).astype(mx.float32)
        r.score_state[0] = grid((1, 2, D), 16, True, seed=seed + 2).astype(mx.float32)
    return c


def run_core(monkeypatch, on, c, x, pos, counts):
    monkeypatch.setattr(CD, "ENABLED", on)
    c.rows[0].offset = pos[0]                                           # the plan reads the session's position
    for k in CD.HITS:
        CD.HITS[k] = 0
    for k in DF.HITS:
        DF.HITS[k] = 0
    attn = make_attn()
    if counts is not None:
        st = DC.stack_rows(c)
        st.begin(counts)
    o = DC.core(attn, x, c)
    mx.eval(o)
    st = DC.stack_rows(c)
    if counts is not None:
        st.commit([counts[0] - 1], DC.plan_of(c))
        mx.eval(*[a for l in range(1) for a in (st.cmp[l].codes if CD.is_view(st.cmp[l]) else st.cmp[l])])
    return np.array(o), CD.HITS["compact"], st


@metal
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("pos,counts", [(10, None), (11, None), (40, [2]), (42, [4])])
def test_served_core_flag0_against_flag1_bit_identical(monkeypatch, pos, counts, fuse):
    """The batched core of a compressing (kv source) layer: the dense pool of flag 0 against the packed view + compact slab of flag 1."""
    monkeypatch.setattr(DC, "ROWS_CORE_FUSE", fuse)                     # the fused row core of a pass, and the batched attention otherwise
    rng = np.random.default_rng(3)
    n = 1 if counts is None else counts[0]
    x = mx.array((rng.standard_normal((n, 1, D)) * 0.1).astype(np.float16))
    n_pool = pos // 2 + 1                                               # the session already holds the pooled rows of its positions
    ref = run_core(monkeypatch, False, _sess(n_pool, seed=4), x, [pos + i for i in range(n)], counts)
    got = run_core(monkeypatch, True, _sess(n_pool, seed=4), x, [pos + i for i in range(n)], counts)
    assert np.array_equal(ref[0], got[0]), int((ref[0] != got[0]).sum())
    assert got[1] > 0                                                   # cdecpack.compact reached the served core
    if pos > 16:                                                        # idx_short: below index_topk=8 pooled keys there is no score call at all
        assert CD.HITS["key_packed"] > 0                                 # the score kernels loaded the key codes
    assert ref[2].cmp[0].dtype == got[2].cmp[0].dtype
    assert CD.is_view(got[2].cmp[0]) and CD.is_view(got[2].keys[0])
    assert got[2].rows[0].compress.raw(0).mirror is None


@metal
def test_served_core_key_fallback_unpacks_for_the_call_and_keeps_no_mirror(monkeypatch):
    """The score chain without a packed-key kernel (fuse / fast / score_fast off): `dense_of` unpacks that layer for the call only, and the output is the same."""
    monkeypatch.setattr(DC, "IDX_FUSE", False)
    monkeypatch.setattr(DC, "IDX_FAST", False)
    monkeypatch.setattr(DC, "IDX_SCORE_FAST", False)
    rng = np.random.default_rng(5)
    x = mx.array((rng.standard_normal((1, 1, D)) * 0.1).astype(np.float16))
    ref = run_core(monkeypatch, False, _sess(21, seed=4), x, [41], None)
    got = run_core(monkeypatch, True, _sess(21, seed=4), x, [41], None)
    assert np.array_equal(ref[0], got[0]), int((ref[0] != got[0]).sum())
    assert CD.HITS["key_dense"] > 0 and CD.HITS["key_packed"] == 0 and CD.HITS["compact"] > 0
    assert got[2].rows[0].keys.raw(0).mirror is None and DF.HITS["build"] == 0


@metal
def test_attend_on_the_compact_slab_equals_the_dense_pool():
    """The batched attention kernel reads a row by index with the test `0 <= idx < N`: on the compact slab with the remapped index it reads the same bits."""
    rng = np.random.default_rng(11)
    N, Cp, K = 4, 40, 8
    q = mx.array((rng.standard_normal((N, H, D)) * 0.1).astype(np.float16))
    win = mx.array((rng.standard_normal((N, W, D)) * 0.1).astype(np.float16))
    pool = mx.array((rng.standard_normal((1, Cp, D)) * 0.1).astype(np.float16))
    idx = mx.array(np.sort(rng.choice(Cp, (N, K), replace=False), axis=1).astype(np.int32))
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    pos = mx.array([5, 130, 300, 127], dtype=mx.int32)
    s = PoolSlots([None])
    s.set_raw(0, fp4pool._Pk(*fp4pack.pack(pool, 16, True), Cp, mx.float16))
    v = CD.view(s, 0)
    dense = fp4pack.unpack(v.codes[:, :v.n], v.scales[:, :v.n], 16, True, mx.float16)
    sess = mx.zeros((N,), dtype=mx.uint32)                                  # one session: every row reads the same pool buffer
    ref = DC.attend(q, win, dense, idx, pos, sink, 0.044, K, sess=sess)
    rows, remap = CD.compact(v, idx)
    got = DC.attend(q, win, rows, remap, pos, sink, 0.044, K, sess=sess)
    mx.eval(ref, got)
    assert np.array_equal(np.array(ref), np.array(got)), int((np.array(ref) != np.array(got)).sum())
    assert CD.HITS["compact"] == 1 and s.raw(0).mirror is None and DF.HITS["build"] == 0
