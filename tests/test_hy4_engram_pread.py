"""Hy4 Engram rows read from the file by deduped parallel preads (engram_gguf.RowReader) against the memory-map gather (engram_gguf.read_rows). CPU only, synthetic GGUF.
Exact = the same floats: rows with repeats and any order, both stored layouts, the cold (pread) and the hot (map) path, the per-layer lazy mapping, the next-chunk hint
(also a stale one), dead token ids, a rollback and a session swap between the hint and the read."""
import os
import sys

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kyojin_mlx import engram_gguf as EG  # noqa: E402
from kyojin_mlx import engram_rows as ER  # noqa: E402
from kyojin_mlx import foreign_gguf as GF  # noqa: E402
from kyojin_mlx import foreign_skeleton as SK  # noqa: E402
from kyojin_mlx import foreign_synth as SY  # noqa: E402


def _synth(tmp_path, plain=False, nl=2, nh=3, vocab=60, cvs=23):
    ng, hd = 3, 64
    primes, c = [], 211
    while len(primes) < nl * (ng - 1) * nh:
        c += 1
        if EG._is_prime(c):
            primes.append(c)
    offs = np.concatenate([[0], np.cumsum(primes)[:-1]]).tolist()
    rows = int(sum(primes))
    rng = np.random.default_rng(5)
    v = rng.integers(0, 256, (rows, hd)).astype(np.uint8)
    v[(v & 127) == 127] = 0x38
    if plain:
        blk, tn = v, "F8_E4M3"
    else:
        sc = rng.integers(118, 128, (rows, hd // 32)).astype(np.uint8)
        blk, tn = np.concatenate([sc[:, :, None], v.reshape(rows, -1, 32)], -1).reshape(rows, -1), "F8_E4M3"
    mult = (rng.integers(0, 1 << 20, nl * ng) * 2 + 1).tolist()
    a = "synth"
    kv = {"general.architecture": a, f"{a}.ple.layers": [1, 3, 5, 7][:nl], f"{a}.ple.ngram_size": ng, f"{a}.ple.heads_per_ngram": nh,
          f"{a}.embedding_length_per_layer_input": hd, f"{a}.ple.layer_multipliers": mult, f"{a}.ple.head_vocab_sizes": primes,
          f"{a}.ple.head_offsets": offs, f"{a}.engram.token_map": (np.arange(vocab) % cvs).tolist(), f"{a}.engram.pad_token_id": 2,
          f"{a}.engram.compressed_vocab_size": cvs, f"{a}.engram.vocab_size": 211}
    p = str(tmp_path / ("p.gguf" if plain else "b.gguf"))
    SY.write_gguf(p, kv, [(EG.TABLE, (hd, rows), tn, blk)])
    return p, vocab


class _Rr:
    """rowread stand-in: the real native reads, `hot` forced (0 = every row cold -> preads, big = hot -> the map)."""

    def __init__(self, real, hot):
        self.read, self._real, self._hot = real.read, real, hot

    def hot(self, m, r, n, w, k):
        return self._hot if self._hot >= 0 else self._real.hot(m, r, n, w, k)


def _force(monkeypatch, hot):
    real = ER._rowread()
    if real is None:
        pytest.skip("no C compiler for the native row reader")
    monkeypatch.setattr(ER, "_rowread", lambda: _Rr(real, hot))


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("hot", [0, 10 ** 6])
def test_fetch_equals_read_rows(tmp_path, monkeypatch, plain, hot):
    _force(monkeypatch, hot)
    p, vocab = _synth(tmp_path, plain)
    g = GF.GgufFile(p)
    spec = EG.EngramGgufSpec.from_gguf(g)
    assert spec.table_type == ("F8_E4M3" if plain else "F8_E4M3_B32")
    rd = EG.RowReader(g, spec)
    rng = np.random.default_rng(1)
    for shape in ((1,), (7,), (5, 4), (40, 2, 6), (0,)):
        r = rng.integers(0, spec.table_rows, shape)
        if r.size:
            r.reshape(-1)[::3] = r.reshape(-1)[0]                           # repeats, unsorted
        assert np.array_equal(rd.fetch(r), EG.read_rows(g, spec, r))
    last = np.array([0, spec.table_rows - 1, spec.table_rows - 1, 0])
    assert np.array_equal(rd.fetch(last), EG.read_rows(g, spec, last))
    with pytest.raises(AssertionError):
        rd.fetch(np.array([spec.table_rows]))
    rd.close()


def test_hash_rows_tail_equals_whole_history(tmp_path):
    p, vocab = _synth(tmp_path)
    spec = EG.EngramGgufSpec.from_gguf(GF.GgufFile(p))
    rng = np.random.default_rng(2)
    ids = rng.integers(-3, vocab + 5, 90)                                    # dead ids on both sides
    whole = EG.GgufRowMap(spec)
    want = whole.rows_valid(ids)
    # frozen reference: the window loop over the whole history, as before the refactor
    cids, dead = EG.compressed_ids(spec, ids)
    pos = np.arange(len(ids))
    toks = np.stack([np.where(pos - j < 0, spec.pad_id, cids[np.maximum(pos - j, 0)]) for j in range(spec.ngram_size)], -1)
    prod = toks.astype(np.uint64)[:, None, :] * spec.multipliers.astype(np.uint64)[None]
    roll, out = prod[..., 0], []
    for n in range(1, spec.ngram_size):
        roll = np.bitwise_xor(roll, prod[..., n])
        out.append((roll[..., None] % spec.primes[:, n - 1].astype(np.uint64)[None]).astype(np.int64))
    ref = np.concatenate(out, -1) + spec.offsets[None]
    assert np.array_equal(want[0], ref) and np.array_equal(want[1], ~dead)
    for a in (0, 1, 2, 3, 17, 60):                                           # from a tail only
        for b in (a + 1, a + 9, 90):
            got = EG.hash_rows(spec, cids[max(0, a - 2):a], cids[a:b], start=a)
            assert np.array_equal(got, ref[a:b])
    m = EG.GgufRowMap(spec)                                                  # chunked rows_valid = one call
    parts = [m.rows_valid(ids[i:j])[0] for i, j in ((0, 5), (5, 6), (6, 50), (50, 90))]
    assert np.array_equal(np.concatenate(parts), ref)


def _ids(n, vocab, seed=0):
    return np.random.default_rng(seed).integers(-2, vocab + 4, n).tolist()


@pytest.mark.parametrize("hot", [0, 10 ** 6])
def test_source_chunks_lazy_and_ahead_equal_map(tmp_path, monkeypatch, hot):
    _force(monkeypatch, hot)
    monkeypatch.setattr(EG, "LAZY_MIN", 8)
    p, vocab = _synth(tmp_path)
    g = GF.GgufFile(p)
    ids = _ids(120, vocab)
    chunks = [(0, 40), (40, 90), (90, 120)]
    monkeypatch.setattr(EG, "PREAD", False)
    ref_src = SK.engram_source_of(g)
    ref = [ref_src(ids[a:b], a) for a, b in chunks]
    monkeypatch.setattr(EG, "PREAD", True)
    for lazy, ahead in ((False, False), (True, False), (True, True)):
        monkeypatch.setattr(EG, "LAZY", lazy)
        monkeypatch.setattr(EG, "AHEAD", ahead)
        src = SK.engram_source_of(g)
        h0 = EG.STATS["ahead_hit"]
        for k, (a, b) in enumerate(chunks):
            if k + 1 < len(chunks):
                src.hint(ids, *chunks[k + 1])                                # the engine's hint comes BEFORE the chunk that runs
            got = src(ids[a:b], a)
            assert list(got) == list(ref[k]) or sorted(got) == sorted(ref[k])
            for lid, (rows, valid) in ref[k].items():
                g_rows, g_valid = got.get(lid)
                assert np.array_equal(g_rows, rows) and np.array_equal(g_valid, valid) and g_rows.dtype == rows.dtype
        if ahead:
            assert EG.STATS["ahead_hit"] > h0                                # the hint landed
        src.reader.close()


def test_stale_hint_and_swaps_stay_exact(tmp_path, monkeypatch):
    _force(monkeypatch, 0)
    monkeypatch.setattr(EG, "LAZY_MIN", 8)
    p, vocab = _synth(tmp_path)
    g = GF.GgufFile(p)
    ids, other = _ids(100, vocab, 1), _ids(100, vocab, 2)
    monkeypatch.setattr(EG, "PREAD", False)
    ref = SK.engram_source_of(g)
    want = {k: ref(ids[a:b], a) for k, (a, b) in enumerate([(0, 30), (30, 60)])}
    ref.truncate(40)                                                         # rollback to 40, replay 40..60
    want_re = ref(ids[40:60], 40)
    monkeypatch.setattr(EG, "PREAD", True)
    src = SK.engram_source_of(g)
    src.hint(other, 30, 60)                                                  # a hint for other tokens: pieces that only miss
    got0 = src(ids[0:30], 0)
    src.hint(ids, 30, 60)
    got1 = src(ids[30:60], 30)
    h = src.get_history()                                                    # 60 positions
    for k, got in ((0, got0), (1, got1)):
        for lid, (rows, valid) in want[k].items():
            assert np.array_equal(got.get(lid)[0], rows)
    src.hint(ids, 40, 60)                                                    # hint made, then a rollback: the rows of the replay come from the same ids
    src.truncate(40)
    got_re = src(ids[40:60], 40)
    for lid, (rows, valid) in want_re.items():
        assert np.array_equal(got_re.get(lid)[0], rows)
    src.set_history(h)                                                       # a session swap to the saved history (30 + 30 positions)
    got_sw = src(ids[60:90], 60)
    ref.set_history(h)
    want_sw = ref(ids[60:90], 60)
    for lid, (rows, valid) in want_sw.items():
        assert np.array_equal(got_sw.get(lid)[0], rows)
    src.reader.close()


def test_hint_below_lazy_min_and_off_are_noops(tmp_path, monkeypatch):
    p, vocab = _synth(tmp_path)
    src = SK.engram_source_of(GF.GgufFile(p))
    ids = _ids(300, vocab)
    monkeypatch.setattr(EG, "PREAD", True)
    monkeypatch.setattr(EG, "AHEAD", True)
    assert src.hint(ids, 0, 10) is False                                     # fewer than LAZY_MIN positions
    monkeypatch.setattr(EG, "AHEAD", False)
    assert src.hint(ids, 0, 280) is False
    monkeypatch.setattr(EG, "AHEAD", True)
    assert src.hint(ids, 0, 280) is True
    src.reader.close()


def test_lazy_rows_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(EG, "LAZY_MIN", 4)
    monkeypatch.setattr(EG, "PREAD", True)
    monkeypatch.setattr(EG, "LAZY", True)
    p, vocab = _synth(tmp_path)
    src = SK.engram_source_of(GF.GgufFile(p))
    out = src(_ids(12, vocab), 0)
    assert isinstance(out, EG.LazyRows) and len(out) == 2 and set(out) == {1, 3} and 1 in out and 9 not in out
    assert out.get(9) is None and out.get(9, "x") == "x"
    assert [k for k, _ in out.items()] == [1, 3] and len(out.values()) == 2
    rows, valid = out[1]
    assert rows.shape == (12, 6, 64) and valid.shape == (12,) and out.pending(1) is False
    src.reader.close()


def test_flags_default_off():
    """The Engram pread / lazy / ahead path is opt-in on this branch (KYOJIN_HY4_ENGRAM_PREAD / _LAZY / _AHEAD = 1); unset = the memory-map gather of before."""
    import os
    from kyojin_mlx import flags
    for n in ("PREAD", "LAZY", "AHEAD"):
        assert flags.FLAGS["hy4_engram_" + n.lower()].default == "0"
    if not any(k.startswith("KYOJIN_HY4_ENGRAM_") for k in os.environ):
        assert not (EG.PREAD or EG.LAZY or EG.AHEAD)
