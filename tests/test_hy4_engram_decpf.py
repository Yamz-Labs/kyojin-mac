"""Hy4 Engram rows of speculative decode rounds: prefetch while the draft head runs (KYOJIN_HY4_ENGRAM_DECPF), the per-row hot / cold split read and the decode-phase
counters (KYOJIN_HY4_ENGRAM_DEC_LOG). CPU only, synthetic GGUF; a stub reader sleeps to model the SSD latency. Exact = the same floats as the memory-map gather."""
import ctypes
import os
import sys
import time

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_hy4_engram_pread import _force, _ids, _synth  # noqa: E402
from test_hy4_mtp import built, _prompts  # noqa: E402,F401  (the synthetic Hy4 model with a draft head)

from kyojin_mlx import engram_gguf as EG  # noqa: E402
from kyojin_mlx import engram_rows as ER  # noqa: E402
from kyojin_mlx import foreign_gguf as GF  # noqa: E402
from kyojin_mlx import foreign_skeleton as SK  # noqa: E402


def _slow(monkeypatch, delay):
    """Every row read (`_read_unique`) takes `delay` s on top of the real work: the cold SSD."""
    real = EG.RowReader._read_unique

    def slow(self, u, split, cold):
        time.sleep(delay)
        return real(self, u, split, cold)
    monkeypatch.setattr(EG.RowReader, "_read_unique", slow)


def _equal(got, ref):
    assert sorted(got) == sorted(ref)
    for lid, (rows, valid) in ref.items():
        g_rows, g_valid = got.get(lid)
        assert np.array_equal(g_rows, rows) and np.array_equal(g_valid, valid) and g_rows.dtype == rows.dtype


def _prefill(src, ids, n):
    src(ids[:n], 0)


@pytest.mark.parametrize("hot", [0, 10 ** 6])
def test_rounds_with_prefetch_equal_map(tmp_path, monkeypatch, hot):
    """Rounds of [pending, *drafts] with a rollback after each: same floats as the map, with the prefetch complete, partial, stale (other draft ids) or absent."""
    _force(monkeypatch, hot)
    _slow(monkeypatch, 0.002)
    p, vocab = _synth(tmp_path)
    g = GF.GgufFile(p)
    ids = _ids(160, vocab, seed=3)
    monkeypatch.setattr(EG, "PREAD", False)
    ref = SK.engram_source_of(g)
    monkeypatch.setattr(EG, "PREAD", True)
    monkeypatch.setattr(EG, "DECPF", True)
    src = SK.engram_source_of(g)
    _equal(src(ids[:40], 0), ref(ids[:40], 0))                              # the prompt (no round pieces yet)
    pos = 40
    rng = np.random.default_rng(7)
    for r in range(30):
        k = int(rng.integers(1, 6))                                         # 1 + k rows
        row = ids[pos:pos + 1 + k]
        mode = r % 5
        if mode == 0:                                                       # pending + every draft, one by one (as the ids become known)
            assert src.dec_begin(row[0])
            for t in row[1:]:
                src.dec_push([t])
        elif mode == 1:                                                     # pending, then all drafts at once
            src.dec_begin(row[0])
            src.dec_push(row[1:])
        elif mode == 2:                                                     # stale: the verify asks other drafts than the prefetch saw
            src.dec_begin(row[0])
            src.dec_push([(t + 1) % vocab for t in row[1:]])
        elif mode == 3:                                                     # only the pending token
            src.dec_begin(row[0])
        if mode == 4 or r % 7 == 0:
            time.sleep(0.02)                                                # let the reads land before the verify
        got = src(row, pos)
        _equal(got, ref(row, pos))
        for lid in got:
            got.get(lid)
        m = int(rng.integers(0, k + 1))                                     # accept m drafts: keep pending + m, roll the rest back
        src.truncate(pos + 1 + m)
        ref.truncate(pos + 1 + m)
        pos += 1 + m
    src.reader.close()


def test_stale_round_other_offset_and_session_swap(tmp_path, monkeypatch):
    """A round begun at one position, then the history changes (another session / rollback): the pieces are keyed by the verify offset, so they only miss."""
    _force(monkeypatch, 0)
    p, vocab = _synth(tmp_path)
    g = GF.GgufFile(p)
    ids = _ids(100, vocab, seed=4)
    monkeypatch.setattr(EG, "PREAD", False)
    ref = SK.engram_source_of(g)
    monkeypatch.setattr(EG, "PREAD", True)
    monkeypatch.setattr(EG, "DECPF", True)
    src = SK.engram_source_of(g)
    src(ids[:30], 0)
    ref(ids[:30], 0)
    src.dec_begin(ids[30])
    src.dec_push(ids[31:33])
    time.sleep(0.05)
    src.truncate(25)                                                        # the session rolled back: the verify starts at 25, not 30
    ref.truncate(25)
    _equal(src(ids[25:29], 25), ref(ids[25:29], 25))
    h = src.get_history()
    src.set_history(h[:20])                                                 # another history of 20 positions
    ref.set_history(h[:20])
    src.dec_begin(ids[20])
    src.dec_push(ids[21:24])
    _equal(src(ids[20:24], 20), ref(ids[20:24], 20))
    src.reader.close()


def test_prefetch_hides_the_read_latency(tmp_path, monkeypatch):
    """With every read costing 40 ms (stub), a round whose rows were prefetched while the 'draft' ran waits ~0 for them; without it the verify pays the read."""
    _force(monkeypatch, 0)
    _slow(monkeypatch, 0.04)
    p, vocab = _synth(tmp_path)
    g = GF.GgufFile(p)
    ids = _ids(80, vocab, seed=5)
    monkeypatch.setattr(EG, "PREAD", True)
    monkeypatch.setattr(EG, "DECPF", True)
    src = SK.engram_source_of(g)
    src(ids[:30], 0)

    def verify(row, pos):
        t = time.perf_counter()
        out = src(row, pos)
        for lid in out:
            out.get(lid)
        return time.perf_counter() - t
    cold = verify(ids[30:34], 30)                                           # no prefetch: >= one 40 ms read on the request path
    src.truncate(30)
    src.dec_begin(ids[30])
    src.dec_push(ids[31:34])
    time.sleep(0.3)                                                         # the draft chain: the reads run meanwhile
    warm = verify(ids[30:34], 30)
    assert cold >= 0.04
    assert warm < cold / 3, (cold, warm)
    src.reader.close()


def test_split_read_cold_rows_only(tmp_path, monkeypatch):
    """`split`: the cached rows come from the map, only the cold rows go to preads (the all-or-nothing sample would fault the cold minority serially)."""
    real = ER._rowread()
    if real is None:
        pytest.skip("no C compiler for the native row reader")
    p, vocab = _synth(tmp_path)
    g = GF.GgufFile(p)
    spec = EG.EngramGgufSpec.from_gguf(g)
    seen = []

    class Rr:
        def __init__(self, mask):
            self.read, self.hot = self._read, (real.hot if mask else (lambda m, r, n, w, k: 0))
            if mask:
                self.hotmask = self._hotmask

        def _read(self, fd, base, rows, n, width, dst, threads):
            seen.append(n)
            return real.read(fd, base, rows, n, width, dst, threads)

        def _hotmask(self, m, r, n, w, out):
            a = (ctypes.c_uint8 * n).from_address(out)
            for i in range(n):
                a[i] = 1 if i % 3 == 0 else 0                               # every third row cached
    rd = EG.RowReader(g, spec)
    u = np.unique(np.random.default_rng(2).integers(0, spec.table_rows, 90))
    monkeypatch.setattr(ER, "_rowread", lambda: Rr(True))
    assert np.array_equal(rd.read_unique(u, "sync", split=True), rd.raw[u])
    assert seen == [int(u.size - (u.size + 2) // 3)]                        # only the 2/3 cold rows were pread
    seen.clear()
    monkeypatch.setattr(ER, "_rowread", lambda: Rr(False))                  # a library without rr_hotmask: one mincore call per row (all cold here), same bytes
    assert np.array_equal(rd.read_unique(u, "sync", split=True), rd.raw[u])
    assert seen == [u.size]
    rd.close()


def test_dec_counters_both_paths_and_phase(tmp_path, monkeypatch, capsys):
    """The decode counters run on the map path (PREAD=0) AND the pread path, skip prompt-phase calls and print one line per request."""
    _force(monkeypatch, 0)
    p, vocab = _synth(tmp_path)
    g = GF.GgufFile(p)
    ids = _ids(60, vocab, seed=6)
    monkeypatch.setattr(EG, "DEC_LOG", True)
    for pread in (False, True):
        monkeypatch.setattr(EG, "PREAD", pread)
        monkeypatch.setattr(EG, "DECPF", pread)
        for k in EG._DKEYS:
            EG.DEC[k] = EG.DEC_RUN[k] = 0
        src = SK.engram_source_of(g)
        with EG.phase("pf"):
            src(ids[:30], 0)                                                # prompt: not counted (and its tail < LAZY_MIN would otherwise look like decode)
        assert EG.DEC["calls"] == 0
        for pos in (30, 32):                                                # two rounds of 4 rows, 2 kept each
            if pread:
                src.dec_begin(ids[pos])
                src.dec_push(ids[pos + 1:pos + 4])
                time.sleep(0.05)
            out = src(ids[pos:pos + 4], pos)
            for lid in out:
                out.get(lid)
            src.truncate(pos + 2)
        assert EG.DEC["calls"] == 2 and EG.DEC["pos"] == 8 and EG.DEC["rows"] == 8 * 2 * 6
        assert EG.DEC["unique"] > 0 and EG.DEC["cold"] > 0 and EG.DEC["src_s"] > 0
        if pread:
            assert EG.DEC["pf_pieces"] == 4 and EG.DEC["pf_hit"] > 0
        else:
            assert EG.DEC["map_calls"] == 2 and EG.DEC["miss"] == EG.DEC["unique"]
        capsys.readouterr()
        with EG.phase("pf"):                                                # the next request starts: the line of this one
            pass
        line = capsys.readouterr().out
        assert "hy4 engram dec request: calls 2" in line and "exposed" in line
        assert EG.DEC_RUN["calls"] == 0
        src.reader.close()


def test_flag_default_off():
    from kyojin_mlx import flags
    assert flags.FLAGS["hy4_engram_decpf"].default == "0"
    if not any(k.startswith("KYOJIN_HY4_ENGRAM_") for k in os.environ):
        assert not (EG.DECPF or EG.DEC_LOG)


def test_native_hotmask_matches_hot(tmp_path):
    real = ER._rowread()
    if real is None:
        pytest.skip("no C compiler for the native row reader")
    assert real.hotmask is not None, "librowread was built before rr_hotmask"
    p, vocab = _synth(tmp_path)
    g = GF.GgufFile(p)
    spec = EG.EngramGgufSpec.from_gguf(g)
    rd = EG.RowReader(g, spec)
    u = np.unique(np.random.default_rng(9).integers(0, spec.table_rows, 70))
    mask = rd._hotmask(real, np.ascontiguousarray(u, np.int64))
    one = [real.hot(rd.raw.ctypes.data, np.ascontiguousarray(u[i:i + 1], np.int64).ctypes.data, 1, rd.rb, 1) for i in range(u.size)]
    assert mask.tolist() == [bool(x) for x in one]
    assert np.array_equal(rd.read_unique(u, "sync", split=True), rd.raw[u])        # the real reader, no stub
    rd.close()


@pytest.mark.parametrize("hot", [0, 10 ** 6])
def test_mtp_generate_with_decpf_equals_plain(built, monkeypatch, hot):
    """The real draft / verify loop on the synthetic Hy4 model: the greedy ids with the prefetch (pread + DECPF, slow reads) equal the plain decode on the map."""
    from kyojin_mlx import hy4_mtp as M
    from mlx_lm.models.cache import KVCache
    _, path, model, head, mtp = built
    _force(monkeypatch, hot)
    _slow(monkeypatch, 0.001)

    def fresh():
        model.engram_source = SK.engram_source_of(GF.GgufFile(path))
        return [KVCache() for _ in model.layers]
    ids = _prompts()[1]
    monkeypatch.setattr(EG, "PREAD", False)
    monkeypatch.setattr(EG, "DECPF", False)
    off = M.generate(mtp, fresh(), None, ids, 48, 3, spec=False)
    monkeypatch.setattr(EG, "PREAD", True)
    monkeypatch.setattr(EG, "DECPF", True)
    cache, stats = fresh(), []
    on = M.generate(mtp, cache, mtp.new_state(), ids, 48, 3, spec=True, stats=stats)
    assert on == off and stats
    model.engram_source.reader.close()
