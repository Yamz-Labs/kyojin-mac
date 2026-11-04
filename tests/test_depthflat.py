"""KYOJIN_DEPTHFLAT: dense mirror of the fp4-packed pool / index keys. CPU only (no Metal kernel in this lever); the reference is the stock whole-layer unpack."""
import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import depthflat as DF, fp4pack, fp4pool
from kyojin_mlx.fp4pool import PoolSlots, KeySlots


@pytest.fixture(autouse=True)
def cpu_and_flag(monkeypatch):
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(DF, "ENABLED", True)
    monkeypatch.setattr(DF, "MIRROR_BYTES", 1 << 30)
    for k in DF.HITS:
        DF.HITS[k] = 0
    yield
    mx.set_default_device(dev)


def grid(n, D=512, seed=0, block=16, e4=True):
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32)).astype(mx.float16)
    c, s = fp4pack.pack(x, block, e4)
    return fp4pack.unpack(c, s, block, e4, mx.float16)


def stock(slots, i):
    """The whole-layer unpack of the stock path (flag off), the reference."""
    DF.ENABLED = False
    try:
        e = slots.raw(i)
        return fp4pack.unpack(e.codes[:, :e.n], e.scales[:, :e.n], slots.BLOCK, slots.E4M3, e.dtype)
    finally:
        DF.ENABLED = True


class Spy:
    """Counts the rows every fp4pack.unpack call touches."""
    def __init__(self, monkeypatch):
        self.rows = 0
        self.calls = 0
        orig = fp4pack.unpack

        def counted(codes, scales, *a, **k):
            self.rows += codes.shape[1]
            self.calls += 1
            return orig(codes, scales, *a, **k)
        monkeypatch.setattr(fp4pack, "unpack", counted)


@pytest.mark.parametrize("cls,D,block,e4", [(PoolSlots, 512, 16, True), (KeySlots, 128, 32, False)])
def test_r1_one_row_per_step_bit_identical_and_o_new_rows(cls, D, block, e4, monkeypatch):
    s = cls([None])
    parts = [grid(900, D, 1, block, e4)] + [grid(1, D, 10 + j, block, e4) for j in range(40)]
    spy = Spy(monkeypatch)
    s.append_rows(0, parts[0])                                   # verify-pack of 900 rows + one mirror build of 900 rows
    dense = parts[0]
    for p in parts[1:]:
        s.append_rows(0, p)                                      # verify-pack of 1 row + a 1-row mirror tail
        out = s[0]                                               # mirror hit
        dense = mx.concatenate([dense, p], axis=1)
        assert mx.array_equal(out, dense)
    assert spy.rows == 2 * 900 + 2 * 40                          # stock: 40 whole-layer unpacks of 900+ rows more
    assert mx.array_equal(out, stock(s, 0))
    assert DF.HITS["build"] == 1 and DF.HITS["tail"] == 40 and DF.HITS["hit"] == 40


def test_r4_four_rows_per_pass_and_truncated_view(monkeypatch):
    s = PoolSlots([None])
    s.append_rows(0, grid(600, seed=2))
    dense = s[0]
    for j in range(10):
        p = grid(4, seed=20 + j)
        s.append_rows(0, p)
        dense = mx.concatenate([dense, p], axis=1)
        e = s.raw(0)
        v = fp4pool.PackedView(e.codes, e.scales, e.n, e.dtype, (s, e))
        assert mx.array_equal(v.dense(), dense)
        cut = v[:1, : e.n - 3]                                   # a verify that commits fewer rows reads a shorter prefix
        assert cut.n == e.n - 3 and mx.array_equal(cut.dense(), dense[:, : e.n - 3])
    assert DF.HITS["view"] == 20


def test_view_older_than_owner_n_is_still_exact():
    s = PoolSlots([None])
    s.append_rows(0, grid(300, seed=3))
    e = s.raw(0)
    old = fp4pool.PackedView(e.codes, e.scales, e.n, e.dtype, (s, e))
    ref = s[0]
    s.append_rows(0, grid(50, seed=4))                           # the owner moved on (and grew past the 256-row step)
    assert mx.array_equal(old.dense(), ref)
    assert s[0].shape[1] == 350


def test_growth_across_capacity_and_mirror_survives_realloc():
    s = KeySlots([None])
    dense = None
    for j, m in enumerate([100, 100, 100, 100, 700, 1, 1, 300]):
        p = grid(m, 128, 30 + j, 32, False)
        out = s.append_rows(0, p)
        dense = p if dense is None else mx.concatenate([dense, p], axis=1)
        assert mx.array_equal(out, dense)
        assert mx.array_equal(s[0], dense)
    e = s.raw(0)
    assert e.mirror.shape[1] >= e.n and e.mn == e.n
    assert mx.array_equal(s[0], stock(s, 0))
    assert DF.HITS["build"] == 1                                 # the mirror followed every realloc of the code buffer


def test_cow_copy_and_fork_never_leak_between_caches():
    s = PoolSlots([None])
    s.append_rows(0, grid(400, seed=5))
    base = s[0]
    c = s.copy()
    s.append_rows(0, grid(5, seed=6))
    c.append_rows(0, grid(7, seed=7))
    assert s[0].shape[1] == 405 and c[0].shape[1] == 407
    assert mx.array_equal(s[0][:, :400], base) and mx.array_equal(c[0][:, :400], base)
    assert mx.array_equal(s[0], stock(s, 0)) and mx.array_equal(c[0], stock(c, 0))
    assert not mx.array_equal(s[0][:, 400:405], c[0][:, 400:405])


def test_off_grid_write_goes_dense_with_mirror_present():
    s = PoolSlots([None])
    g = grid(40, seed=8)
    s.append_rows(0, g)
    _ = s[0]
    raw = mx.array(np.random.default_rng(9).standard_normal((1, 3, 512)).astype(np.float32)).astype(mx.float16)
    out = s.append_rows(0, raw)
    assert mx.array_equal(out, mx.concatenate([g, raw], axis=1))
    s[0] = g[:, :10]                                             # setitem: a fresh entry, no stale mirror
    assert mx.array_equal(s[0], g[:, :10])


def test_budget_keeps_stock_path(monkeypatch):
    monkeypatch.setattr(DF, "MIRROR_BYTES", 1024)
    s = PoolSlots([None])
    g = grid(50, seed=11)
    s.append_rows(0, g)
    assert mx.array_equal(s[0], g) and DF.HITS["skip"] >= 1 and s.raw(0).mirror is None


def test_flag_off_is_the_stock_path(monkeypatch):
    monkeypatch.setattr(DF, "ENABLED", False)
    s = PoolSlots([None])
    g = grid(30, seed=12)
    s.append_rows(0, g)
    assert mx.array_equal(s[0], g) and s.raw(0).mirror is None and DF.HITS["build"] == 0


def test_capacity_slack():
    from kyojin_mlx import longctx as LC
    assert LC.capacity(1000, 256) == 1280 and LC.capacity(1_000_000, 256) == 1_125_120
    assert LC.capacity(256, 256) == 512


def test_vendor_cache_roundtrip_with_mirror():
    pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
    P = fp4pool.make_packed_class(DeepseekV41Cache)
    d, p = DeepseekV41Cache(2), P(2)
    for c in (d, p):
        for j in range(30):
            c.append_compressed(0, grid(1, seed=40 + j)); c.append_index_keys(1, grid(1, D=128, seed=70 + j, block=32, e4=False))
    assert mx.array_equal(p.compress[0], d.compress[0]) and mx.array_equal(p.keys[1], d.keys[1])
    assert all(mx.array_equal(a, b) for a, b in zip(p.state, d.state))


@pytest.mark.parametrize("cls,D,block,e4", [(PoolSlots, 512, 16, True), (KeySlots, 128, 32, False)])
def test_speculative_rollback_by_snapshot_restore(cls, D, block, e4):
    """Rejected draft rows: the engine never lowers n, it snapshots (copy / spec.copy_layer = cow) and restores. The restored layer must read the
    new rows, not the rejected drafts, and the drafting side must stay correct too."""
    s = cls([None])
    base = grid(300, D, 1, block, e4)
    s.append_rows(0, base)
    assert float(mx.abs(s[0] - base).max()) == 0.0                                # mirror built at n = 300
    snap = s.copy()                                                              # snapshot before the draft
    snap2 = cls([None]); snap2.set_raw(0, cls.cow(s.raw(0)))                     # copy_layer-style snapshot
    draft = grid(3, D, 2, block, e4)
    s.append_rows(0, draft)
    d = s[0]                                                                     # the draft side reads (mirror -> 303)
    assert d.shape[1] == 303 and float(mx.abs(d - stock(s, 0)).max()) == 0.0
    new = grid(3, D, 3, block, e4)                                               # rejected: restore the snapshot, append other rows at the same positions
    for r in (snap, snap2):
        r.append_rows(0, new)
        out = r[0]
        assert out.shape[1] == 303
        assert float(mx.abs(out - stock(r, 0)).max()) == 0.0
        assert float(mx.abs(out[:, 300:] - new).max()) == 0.0
        assert float(mx.abs(out[:, :300] - base).max()) == 0.0
    assert float(mx.abs(s[0][:, 300:] - draft).max()) == 0.0                     # the drafting side still has its own rows


def test_restore_over_a_live_slot_with_set_raw():
    """Rollback in place: set_raw(l, snapshot) over the layer that already drafted. The restored entry keeps its own (older) mirror length."""
    s = KeySlots([None])
    s.append_rows(0, grid(200, 128, 1, 32, False))
    _ = s[0]
    snap = KeySlots.cow(s.raw(0))
    for j in range(4):                                                           # several drafted steps, each read
        s.append_rows(0, grid(2, 128, 20 + j, 32, False))
        _ = s[0]
    s.set_raw(0, snap)
    new = grid(5, 128, 99, 32, False)
    s.append_rows(0, new)
    out = s[0]
    assert out.shape[1] == 205 and float(mx.abs(out - stock(s, 0)).max()) == 0.0
    assert float(mx.abs(out[:, 200:] - new).max()) == 0.0


def test_cow_held_across_steps_stays_exact_and_independent():
    """A snapshot held while the live layer decodes many steps (the worst case for buffer copies on Metal) stays exact on both sides."""
    s = KeySlots([None])
    s.append_rows(0, grid(500, 128, 1, 32, False))
    _ = s[0]
    held = KeySlots.cow(s.raw(0))
    for j in range(30):
        s.append_rows(0, grid(1, 128, 40 + j, 32, False))
        out = s[0]
    assert out.shape[1] == 530 and float(mx.abs(out - stock(s, 0)).max()) == 0.0
    h = KeySlots([None]); h.set_raw(0, held)
    assert h[0].shape[1] == 500 and float(mx.abs(h[0] - stock(h, 0)).max()) == 0.0


def test_snapshot_keeps_only_the_stored_kv_and_continues_identically(monkeypatch):
    monkeypatch.setattr(fp4pool, "SEG", 0)                             # the plain (aliasing) snapshot this test is about; kvseg now defaults to 8192
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Cache
    from kyojin_mlx import rows as R
    from kyojin_mlx.fp4pool import make_packed_class
    c = make_packed_class(DeepseekV41Cache)(2)
    a, b, d = grid(300, seed=31), grid(40, seed=32), grid(12, seed=33)
    c.append_compressed(0, a)
    c.compress_kv = c.compress[0]                                       # mirror built, handoff published
    snap = R.snapshot([c])[0]
    e = snap.compress.raw(0)
    assert e.mirror is None and e.mn == 0 and e.n == 300
    assert snap.compress_kv is None and c.compress_kv is not None and c.compress.raw(0).mirror is not None
    c.append_compressed(0, b)                                           # the live session goes on: the snapshot does not move
    assert mx.array_equal(snap.compress[0], a)
    hit = R.fork([snap])[0]                                             # a hit grows its own copy, the mirror comes back on the first read
    hit.append_compressed(0, d)
    assert mx.array_equal(hit.compress[0], mx.concatenate([a, d], axis=1))
    assert mx.array_equal(snap.compress[0], a)
    assert mx.array_equal(c.compress[0], mx.concatenate([a, b], axis=1))
