"""mirrorq: the dense depthflat mirror is not retained by a one-session decode.

CPU: the pool view (the session's own codes, the mirror dropped), the key stack unpacked from the codes against `_stack_ragged` bit for bit, the decoded slab
against `fp4pack.unpack` and against `cdecpack.compact`, the append (codes grow, no read-back, no mirror rebuilt), the rollback (a rejected count hides the
rows, a re-decode sees the committed ones), the byte bound (nothing dense of the layer survives but the slab of the named rows) and flag 0 unchanged.
Metal: the fused decode kernel against `fp4pack.unpack` element for element, and the served batched core at flag 0 against flag 1 bit for bit.
"""
import os
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from kyojin_mlx import cdecpack as CD, depthflat as DF, dsv4_core as DC, fp4pack, fp4pool, mirrorq as MQ  # noqa: E402
from kyojin_mlx.fp4pool import KeySlots, PoolSlots  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(autouse=True)
def _no_slot_stack(monkeypatch):
    monkeypatch.setenv("KYOJIN_MU_KEEP", "0")                             # this lever's one-session path; a slot stack (mu_keep) keeps the dense form


def eq(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def bits(a, b):
    return np.array(a).view(np.uint16).tolist() == np.array(b).view(np.uint16).tolist()


def grid(shape, block, e4m3, seed=0, dtype=mx.float16):
    """Random rows ON the fp4 grid of (block, e4m3): what the producers hand the cache."""
    x = mx.random.normal(shape, key=mx.random.key(seed)) * 3.0
    c, s = fp4pack.pack(x.astype(dtype), block, e4m3)
    return fp4pack.unpack(c, s, block, e4m3, dtype)


def packed(cls, n, D, block, e4m3, seed=0):
    """Slots holding one layer of `n` rows exactly as the cache does after a prefill append, without the dense read an append takes."""
    s = cls([None])
    s.set_raw(0, fp4pool._Pk(*fp4pack.pack(grid((1, n, D), block, e4m3, seed), block, e4m3), n, mx.float16))
    return s


def warm(s, l=0):
    """The state a served session is in when the decode starts: the prefill read built the dense mirror."""
    mx.eval(s[l])
    assert s.raw(l).mirror is not None
    return s


class Row:
    step = 8

    def reset_handoff(self):
        pass


class RowCache:
    """The per-session cache surface `stack_rows` / `sync_rows` / `commit_layer` read."""

    def __init__(self, layers, D=64, KD=128):
        self.rows = [Row()]
        r = self.rows[0]
        r.window = [None] * layers
        r.compress, r.keys = PoolSlots([None] * layers), KeySlots([None] * layers)
        r._compress_buffers, r._key_buffers = [None] * layers, [None] * layers
        r.kv_state, r.score_state = [None] * layers, [None] * layers
        r.offset = 0
        r.engram = None


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(MQ, "ENABLED", True)
    monkeypatch.setattr(CD, "ENABLED", False)
    monkeypatch.setattr(DF, "ENABLED", True)
    monkeypatch.setattr(DF, "MIRROR_BYTES", 1 << 30)
    monkeypatch.setattr(fp4pool, "MEMO_BYTES", 0)
    for k in DF.HITS:
        DF.HITS[k] = 0
    for k in CD.HITS:
        CD.HITS[k] = 0
    for k in MQ.HITS:
        MQ.HITS[k] = 0
    yield


# ------------------------------------------------------------------------------------------------------------------------ registry
def test_flag_registered_default_off_and_documented():
    from kyojin_mlx import flags
    f = flags.FLAGS["mirrorq"]
    assert f.env == "KYOJIN_MIRRORQ" and f.default == "0" and f.runtime
    assert ("mirrorq", "ENABLED") in [(a[0], a[1]) for a in f.attrs]
    txt = open(os.path.join(REPO, "docs", "cloud", "LEVERS.md")).read()
    assert "`KYOJIN_MIRRORQ` | mirrorq |" in txt


# ------------------------------------------------------------------------------------------------------------------------ the pool: the codes, no mirror
def test_pool_view_is_the_session_codes_and_drops_the_mirror():
    s = warm(packed(PoolSlots, 300, 64, 16, True))
    nb = s.raw(0).mirror.nbytes
    v = MQ.pool_view(s, 0)
    assert isinstance(v, fp4pool.PackedView) and v.n == 300 and v.block == 16 and v.e4m3 is True
    assert v.codes is s.raw(0).codes or eq(v.codes, s.raw(0).codes)
    assert s.raw(0).mirror is None and s.raw(0).memo is None
    assert MQ.HITS["free"] == 1 and MQ.HITS["drop_bytes"] == nb and MQ.HITS["pool_view"] == 1
    assert eq(v.dense(), fp4pack.unpack(s.raw(0).codes[:, :300], s.raw(0).scales[:, :300], 16, True, mx.float16))
    assert MQ.pool_view(PoolSlots([None]), 0) is None and MQ.HITS["skip"] == 1      # an empty layer: no packed form


def test_pool_view_refuses_a_layer_above_the_mirror_budget(monkeypatch):
    s = warm(packed(PoolSlots, 64, 64, 16, True))
    monkeypatch.setattr(DF, "MIRROR_BYTES", 8)                                     # depthflat keeps the stock path there, and so does this
    assert MQ.pool_view(s, 0) is None and s.raw(0).mirror is not None and MQ.HITS["skip"] == 1


# ------------------------------------------------------------------------------------------------------------------------ the slab: only the named rows
def test_decode_rows_equals_the_dense_rows_it_replaces():
    s = packed(PoolSlots, 700, 64, 16, True)
    v = MQ.pool_view(s, 0)
    dense = fp4pack.unpack(v.codes[:, :v.n], v.scales[:, :v.n], 16, True, mx.float16)
    idx = mx.array([[3, 699, 0, -1, 250, 700, 9999, 3], [5, 6, 7, 8, -1, -1, 120, 640]], dtype=mx.int32)
    rows, remap = MQ.decode_rows(v, idx)
    assert rows.shape == (1, idx.size, 64) and remap.shape == idx.shape
    r, i, m = np.array(rows[0]), np.array(idx), np.array(remap)
    for a in range(i.shape[0]):
        for b in range(i.shape[1]):
            if 0 <= i[a, b] < 700:
                assert m[a, b] >= 0 and bits(r[m[a, b]], dense[0, i[a, b]])     # the same bits, not merely a close value
            else:
                assert m[a, b] == -1
    assert MQ.HITS["decode"] == 1 and MQ.HITS["decode_rows"] == idx.size
    assert MQ.slab_bytes(idx, 64) == idx.size * 64 * 2
    assert s.raw(0).mirror is None and DF.HITS["build"] == 0


def test_decode_rows_equals_cdecpack_compact():
    s = packed(PoolSlots, 900, 64, 16, True)
    v = MQ.pool_view(s, 0)
    idx = mx.array(np.random.default_rng(7).integers(-3, 903, (3, 9)).astype(np.int32))
    rows, remap = MQ.decode_rows(v, idx)
    ref, refmap = CD.compact(v, idx)
    assert eq(rows, ref) and eq(remap, refmap)                                    # same slab, same remap, bit for bit


def test_decode_rows_chunks_a_wide_slab(monkeypatch):
    monkeypatch.setattr(MQ, "SLAB_ROWS", 5)                                       # a slab wider than SLAB_ROWS: decoded in steps
    s = packed(PoolSlots, 40, 64, 16, True)
    v = MQ.pool_view(s, 0)
    idx = mx.arange(40, dtype=mx.int32)[None]
    rows, _ = MQ.decode_rows(v, idx)
    assert eq(rows, fp4pack.unpack(v.codes, v.scales, 16, True, mx.float16)[:, :40])


# ------------------------------------------------------------------------------------------------------------------------ the key stack: the same rows, no mirror
def test_stack_dense_equals_the_ragged_dense_stack(monkeypatch):
    a, b = warm(packed(KeySlots, 300, 128, 32, False, seed=1)), warm(packed(KeySlots, 210, 128, 32, False, seed=2))
    ref, refcnt = DC._stack_ragged([a[0], b[0]], 8)                               # the flag 0 stack, before the mirrors are dropped
    got, cnt = MQ.stack_dense([a, b], 0, 8)
    assert cnt == refcnt == [300, 210]
    assert got.shape == ref.shape and eq(got, ref)
    assert a.raw(0).mirror is None and b.raw(0).mirror is None and MQ.HITS["keys_stack"] == 1


def test_stack_dense_chunks_the_unpack(monkeypatch):
    monkeypatch.setattr(MQ, "CHUNK", 7)                                           # rows are independent: the chunk split must not change a bit
    a = packed(KeySlots, 50, 128, 32, False)
    ref = fp4pack.unpack(a.raw(0).codes[:, :50], a.raw(0).scales[:, :50], 32, False, mx.float16)
    got, cnt = MQ.stack_dense([a], 0, 8)
    assert cnt == [50] and eq(got[0:1, :50], ref)


def test_stack_dense_declines_a_layer_it_cannot_read(monkeypatch):
    dense = PoolSlots([None])
    dense.set_raw(0, mx.zeros((1, 4, 64), dtype=mx.float16))                       # off the fp4 grid: the layer stays dense
    assert MQ.stack_dense([dense], 0, 8) is None and MQ.HITS["stack_skip"] == 1


# ------------------------------------------------------------------------------------------------------------------------ the stack: the decode entry point
def test_stack_rows_one_session_reads_the_codes_and_keeps_no_mirror():
    c = RowCache(1)
    c.rows[0].compress, c.rows[0].keys = warm(packed(PoolSlots, 300, 64, 16, True)), warm(packed(KeySlots, 300, 128, 32, False))
    assert DF.HITS["build"] == 2                                                   # the prefill read built both mirrors
    st = DC.stack_rows(c)
    assert MQ.is_view(st.cmp[0]) and st.cnt == [[300]]                             # the pool: the session's own codes, no copy
    assert st.keys[0].shape == (1, -(-301 // 8) * 8, 128) and st.kcnt == [[300]]   # the keys: the dense stack the core reads
    assert c.rows[0].compress.raw(0).mirror is None and c.rows[0].keys.raw(0).mirror is None
    assert st.mq_keys == {0} and DF.HITS["build"] == 2                             # nothing rebuilt a mirror
    assert MQ.HITS["keys_copy"] == 1                                               # the prefill's key mirror was copied, then dropped
    walk = fp4pool.retained_walk([c, st])
    assert walk["mirror"] == 0 and walk["memo"] == 0
    assert walk["counted"] == 300 * (64 // 2 + 64 // 16) + 300 * (128 // 2 + 128 // 32)     # both session layers stay packed


def test_stack_rows_flag0_is_the_dense_path(monkeypatch):
    monkeypatch.setattr(MQ, "ENABLED", False)
    c = RowCache(1)
    c.rows[0].compress, c.rows[0].keys = packed(PoolSlots, 40, 64, 16, True), packed(KeySlots, 40, 128, 32, False)
    st = DC.stack_rows(c)
    assert not MQ.is_view(st.cmp[0]) and st.cmp[0].shape[2] == 64 and getattr(st, "mq_keys", None) is None
    assert c.rows[0].compress.raw(0).mirror is not None                            # flag 0: the mirror is the dense form, and stays


def test_append_writes_the_codes_and_never_rebuilds_a_mirror():
    c = RowCache(1)
    c.rows[0].compress, c.rows[0].keys = warm(packed(PoolSlots, 300, 64, 16, True)), warm(packed(KeySlots, 300, 128, 32, False))
    st = DC.stack_rows(c)
    em = DC.Plan([300], None).emit(2)
    pool_new, key_new = grid((1, 64), 16, True, seed=9), grid((1, 128), 32, False, seed=9)
    DC._put(st.cmp, 0, 1, (64,), mx.float16, em, pool_new, 8, None)
    DC._put(st.keys, 0, 1, (128,), mx.float16, em, key_new, 8, None)
    assert MQ.is_view(st.cmp[0])                                                   # packed straight into the session's own codes
    st.pend[0] = SimpleNamespace(win=None, kvs=None, scs=None, cmp=st.cmp[0], keys=st.keys[0], r=2, cnt=None, kcnt=None, kvr=None, kvrow=None, scrow=None)
    st.commit_layer(0, None, [300])
    assert st.cnt == [[150]] and st.cmp[0].n == 150 and st.kcnt == [[150]]
    st._dirty = True
    st.sync_rows()
    assert MQ.HITS["no_readback"] == 1 and DF.HITS["build"] == 2                   # no read-back, so no mirror rebuilt
    e, k = c.rows[0].compress.raw(0), c.rows[0].keys.raw(0)
    assert e.n == 150 and k.n == 150
    assert e.mirror is None and k.mirror is None
    assert eq(st.cmp[0].codes[:, :150], e.codes[:, :150])                          # the same codes, written once
    want = fp4pack.unpack(k.codes[:, 149:150], k.scales[:, 149:150], 32, False, mx.float16)
    assert eq(st.keys[0][:, 149:150], want)                                        # the row the pass wrote is the row the stack reads


def test_rollback_hides_the_rows_written_past_the_commit():
    """A verify pass writes the pooled rows of four draft positions and keeps one: the row the rejected part wrote past the commit stays unread, and the
    slab decode of the next step sees only the committed rows."""
    c = RowCache(1)
    c.rows[0].compress, c.rows[0].keys = warm(packed(PoolSlots, 2, 64, 16, True)), warm(packed(KeySlots, 2, 128, 32, False))
    c.rows[0].offset = 3                                                           # positions 3..6: (3+1)//2 = 2 pooled rows already held
    st = DC.stack_rows(c)
    plan = DC.Plan([3, 4, 5, 6], None)
    em = plan.emit(2)
    assert em.rows == [0, 2] and em.widx == [1, 2] and em.need == 3
    vals = grid((2, 64), 16, True, seed=11)
    DC._put(st.cmp, 0, 1, (64,), mx.float16, em, vals, 8, None)
    st.pend[0] = SimpleNamespace(win=None, kvs=None, scs=None, cmp=st.cmp[0], keys=None, r=2, cnt=None, kcnt=None, kvr=None, kvrow=None, scrow=None)
    st.commit_layer(0, None, [3])                                                  # only position 3 of the four is kept
    assert st.cnt == [[2]] and st.cmp[0].n == 2
    v = st.cmp[0]
    idx = mx.array([[0, 1, 2, 3], [-1, 4, 5, 99]], dtype=mx.int32)
    slab, remap = MQ.decode_rows(v, idx)
    r, m = np.array(slab[0]), np.array(remap)
    assert m[1].tolist() == [-1, -1, -1, -1]                                       # past the committed count: no row, exactly as a dense out-of-range read
    assert m[0, 2] == -1 and m[0, 3] == -1
    assert bits(r[m[0, 1]], fp4pack.unpack(v.codes[:, 1:2], v.scales[:, 1:2], 16, True, mx.float16)[0, 0])
    assert c.rows[0].compress.raw(0).mirror is None


# ------------------------------------------------------------------------------------------------------------------------ Metal
metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@metal
@pytest.mark.parametrize("D,block,e4", [(512, 16, True), (128, 32, False)])
def test_kernel_equals_unpack(D, block, e4):
    """The fused decode kernel against fp4pack.unpack: every element, bit for bit (sign, -0.0 and the fp16 rounding of the product included)."""
    assert MQ.ENABLED
    cls = PoolSlots if e4 else KeySlots
    s = packed(cls, 200, D, block, e4, seed=5)
    v = MQ.pool_view(s, 0)
    ref = fp4pack.unpack(v.codes, v.scales, block, e4, mx.float16)
    idx = mx.arange(200, dtype=mx.int32)[None]
    MQ.SLAB_ROWS = 64                                                              # several launches
    try:
        rows, remap = MQ.decode_rows(v, idx)
    finally:
        pass
    assert MQ.HITS["decode_kernel"] == 1
    got = np.array(rows[0])
    assert np.array_equal(got.view(np.uint16), np.array(ref[0]).view(np.uint16)), int((got.view(np.uint16) != np.array(ref[0]).view(np.uint16)).sum())


@metal
def test_kernel_scale_decode_covers_every_byte():
    """The in-kernel e4m3 / ue8m0 decode against `fp4pack.unpack` for EVERY scale byte and EVERY code nibble: the one place this module does not call
    `fp4pack.unpack` itself. 0x7F / 0xFF are the e4m3 NaN payloads, which the packer cannot emit (asserted below), so they are left out."""
    from kyojin_mlx import fp4pack as FP
    for cls, D, block, e4, rows in ((PoolSlots, 512, 16, True, 8), (KeySlots, 128, 32, False, 64)):
        nblk, n = D // block, rows
        codes = np.zeros((1, n, D // 2), dtype=np.uint8)
        for i in range(n):
            codes[0, i] = np.arange(D // 2) & 0xFF                                   # every byte value 0..255 over a row
        scales = np.zeros((1, n, nblk), dtype=np.uint8)
        scales[0] = (np.arange(n * nblk) % 256).reshape(n, nblk)                     # every scale byte, 0..255
        s = cls([None])
        s.set_raw(0, fp4pool._Pk(mx.array(codes), mx.array(scales), n, mx.float16))
        v = MQ.pool_view(s, 0)
        ref = np.array(FP.unpack(v.codes, v.scales, block, e4, mx.float16)[0])
        got = np.array(MQ.decode_rows(v, mx.arange(n, dtype=mx.int32)[None])[0][0])
        live = np.repeat(~np.isin(np.array(scales[0]), (0x7F, 0xFF)), block, axis=1)
        bad = (got.view(np.uint16) != ref.view(np.uint16)) & live
        assert not bad.any(), (int(bad.sum()), np.argwhere(bad)[:4])
    # the packer's own range never lands on those two bytes
    x = mx.array((np.random.default_rng(0).standard_normal((256, 512)) * 12).astype(np.float16))
    c, sc = FP.pack(x[None], 16, True)
    assert not np.isin(np.array(sc), (0x7F, 0xFF)).any()
    c, sc = FP.pack(mx.array((np.random.default_rng(1).standard_normal((256, 128)) * 12).astype(np.float16))[None], 32, False)
    assert not np.isin(np.array(sc), (0x7F, 0xFF)).any()


@metal
def test_kernel_matches_the_mlx_decode_on_a_random_index_set(monkeypatch):
    s = packed(PoolSlots, 300, 512, 16, True, seed=8)
    v = MQ.pool_view(s, 0)
    idx = mx.array(np.random.default_rng(3).integers(-2, 302, (2, 24)).astype(np.int32))
    got, remap = MQ.decode_rows(v, idx)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    ref, refmap = MQ.decode_rows(v, idx)
    assert eq(got, ref) and eq(remap, refmap) and MQ.HITS["decode_kernel"] == 1 and MQ.HITS["decode_fallback"] == 1


# ------------------------------------------------------------------------------------------------------------------------ the served core
D, W, H, QL, G, OL = 512, 128, 4, 256, 2, 64


class _WoA:
    def __init__(self, rng):
        self.w = mx.array((rng.standard_normal((G, (H // G) * D, OL)) * 0.02).astype(np.float16))

    def __call__(self, x):
        return mx.stack([x[:, g] @ self.w[g] for g in range(G)], axis=1)


def make_attn(seed=0, ratio=2):
    """The DeepseekV41Attention surface `dsv4_core.core` reads: dense linears, the real rope, a real Compressor and Indexer on a one-layer config."""
    import mlx.nn as nn
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


def sess(n_pool, seed=4):
    """One session cache: the pool / keys / compressor state `core` reads, all on the fp4 grid."""
    c = RowCache(1, D, 128)
    r = c.rows[0]
    if n_pool:
        r.compress.set_raw(0, fp4pool._Pk(*fp4pack.pack(grid((1, n_pool, D), 16, True, seed), 16, True), n_pool, mx.float16))
        r.keys.set_raw(0, fp4pool._Pk(*fp4pack.pack(grid((1, n_pool, 128), 32, False, seed), 32, False), n_pool, mx.float16))
        r.kv_state[0] = grid((1, 2, D), 16, True, seed=seed + 1).astype(mx.float32)
        r.score_state[0] = grid((1, 2, D), 16, True, seed=seed + 2).astype(mx.float32)
    return c


def run_steps(monkeypatch, on, steps, seed=3, drop_attn=False):
    """`steps` = [(pos, counts or None, n_rows)] over one session cache, the core call after the stack: the flag 0 path against the flag 1 path.
    dcache is rebuilt per run (same seed), a commit between steps writes the codes back, so step 2 reads what step 1 wrote."""
    monkeypatch.setattr(MQ, "ENABLED", on)
    rng = np.random.default_rng(seed)
    c = None
    outs = []
    for pos, counts, n in steps:
        if c is None:
            c = sess(pos // 2 + 1)
        c.rows[0].offset = pos
        for k in DF.HITS:
            DF.HITS[k] = 0
        attn = make_attn()
        st = DC.stack_rows(c)
        if counts is not None:
            st.begin(counts)
        x = mx.array((rng.standard_normal((n, 1, D)) * 0.1).astype(np.float16))
        o = DC.core(attn, x, c)
        mx.eval(o)
        if counts is not None:
            st.commit([counts[0] - 1], DC.plan_of(c))
            mx.eval(*st.stacked_arrays())
        outs.append(np.array(o))
    return outs, c, st


@metal
def test_served_core_flag0_against_flag1_bit_identical(monkeypatch):
    """The batched core of a compressing (kv source) layer: the dense pool and keys of flag 0 against the codes and the decoded slab of flag 1."""
    for counts, pos, n in ((None, 10, 1), (None, 11, 1), ([2], 40, 2), ([4], 42, 4)):
        ref, rc, rst = run_steps(monkeypatch, False, [(pos, counts, n)])
        got, gc, gst = run_steps(monkeypatch, True, [(pos, counts, n)])
        assert np.array_equal(ref[0], got[0]), int((ref[0] != got[0]).sum())
        assert MQ.is_view(gst.cmp[0]) and MQ.HITS["decode"] > 0                    # the slab decode reached the served core
        assert gst.rows[0].compress.raw(0).mirror is None and gst.rows[0].keys.raw(0).mirror is None
        assert DF.HITS["build"] == 0                                               # flag 1 never built one
        assert rst.rows[0].compress.raw(0).mirror is not None                      # flag 0 kept it


@metal
def test_two_decode_steps_read_what_the_first_wrote(monkeypatch):
    """Step 2 reads the pool row and the key row step 1 appended: the codes of step 1, not a stale dense copy. Same bits, both flags."""
    steps = [(40, [2], 2), (42, [4], 4)]
    ref, rc, _ = run_steps(monkeypatch, False, steps)
    got, gc, gst = run_steps(monkeypatch, True, steps)
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])
    assert int((ref[1] != got[1]).sum()) == 0
    st_ref, st_got = DC.stack_rows(rc), DC.stack_rows(gc)                          # both flags leave the same state behind the two steps
    nk = min(st_ref.kcnt[0][0], st_got.kcnt[0][0])
    nc = min(st_ref.cnt[0][0], st_got.cnt[0][0])
    assert nk > 1 and nc > 1
    assert eq(st_ref.keys[0][:, :nk], st_got.keys[0][:, :nk])                       # the dense key stack the core reads
    pref = lambda p: DC.CD.dense_of(p)[:, :nc]                                     # the pool table the attention reads (dense at flag 0, the codes at flag 1)
    assert eq(pref(st_ref.cmp[0]), pref(st_got.cmp[0]))
    assert MQ.is_view(st_got.cmp[0]) and st_got.cmp[0].n == nc


