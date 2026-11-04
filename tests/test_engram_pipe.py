"""Engram without a host sync in the decode step (kyojin_mlx.engram_mlx: DeviceRowEmbedding, StagedRowEmbedding + EngramStage, pipelined_greedy).
Synthetic tables for the embeds (bit for bit against the pread embed); a ds2 pack carrying an Engram layer (KYOJIN_SHRUNK=<dir with ds2/>, as in
tests/test_engram_rows.py) for the model level: tokens of the pipelined loop equal spec.plain_greedy, the hash ids one step ahead equal the host
RowMap, and no command buffer is committed while the graph of a step is built (needs tools/cbcount/libcbcount.dylib, see tools/engram_pipe.py)."""
import ctypes
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from kyojin_mlx import engram_rows as ER
from test_engram_rows import make_shards, spec_with_rows

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import engram_mlx as EM

SHRUNK = os.environ.get("KYOJIN_SHRUNK")


def tables_in(tmp_path):
    sp, rows = spec_with_rows()
    make_shards(tmp_path, sp, rows)
    t = ER.EngramTables(str(tmp_path), sp)
    assert t.validate() == []
    return t, rows


def test_read_rows_into_matches_read_rows(tmp_path):
    t, rows = tables_in(tmp_path)
    ask = np.array([5, 0, rows[1] - 1, 5, 77, 3, 3, 900, 12, 4001])                     # unsorted, repeats, first and last row
    for pos in (0, 1):
        r = ask % rows[pos] if pos == 0 else ask
        w, s = t.read_rows(pos, r)
        wb, sb = np.zeros_like(w), np.zeros_like(s)
        assert t.read_rows_into([(pos, r, wb, sb)]) == len(r)
        assert np.array_equal(w, wb) and np.array_equal(s, sb)
    w0, s0 = t.read_rows(0, ask % rows[0])
    w1, s1 = t.read_rows(1, ask)
    bufs = [np.zeros((len(ask), 256), np.uint8), np.zeros((len(ask), 8), np.uint8), np.zeros((len(ask), 256), np.uint8), np.zeros((len(ask), 8), np.uint8)]
    t.read_rows_into([(0, ask % rows[0], bufs[0], bufs[1]), (1, ask, bufs[2], bufs[3])])               # two layers in one batch
    assert all(np.array_equal(x, y) for x, y in zip(bufs, (w0, s0, w1, s1)))
    assert t.read_rows_into([]) == 0
    t.COLD_S = -1.0                                                                     # every pread counts as a miss: the rest of the batch goes to the pool
    bufs2 = [np.zeros_like(b) for b in bufs]
    t.read_rows_into([(0, ask % rows[0], bufs2[0], bufs2[1]), (1, ask, bufs2[2], bufs2[3])])
    assert all(np.array_equal(x, y) for x, y in zip(bufs2, (w0, s0, w1, s1)))
    t.COLD_S = 1e9                                                                      # never: all in the calling thread
    bufs3 = [np.zeros_like(b) for b in bufs]
    t.read_rows_into([(0, ask % rows[0], bufs3[0], bufs3[1]), (1, ask, bufs3[2], bufs3[3])])
    assert all(np.array_equal(x, y) for x, y in zip(bufs3, (w0, s0, w1, s1)))
    with pytest.raises(AssertionError):
        t.read_rows_into([(0, [rows[0]], np.zeros((1, 256), np.uint8), np.zeros((1, 8), np.uint8))])
    t.close()


def test_device_embedding_is_bit_exact_over_chunks(tmp_path):
    t, rows = tables_in(tmp_path)
    ref = EM.ParallelRowEmbedding(t, 0)
    rng = np.random.default_rng(3)
    cr = rows[0] // 4 + 3                                                               # 4 chunks, the last one shorter
    idx = rng.integers(0, rows[0], (2, 3, 24))
    idx[0, 0, :6] = [0, cr - 1, cr, 2 * cr - 1, 2 * cr, rows[0] - 1]                    # every chunk boundary
    want = np.array(ref(mx.array(idx)).astype(mx.float32))
    for chunk in (cr, None):                                                            # forced small chunks, then the device default (one chunk)
        dev = EM.DeviceRowEmbedding.from_tables(t, 0, chunk, block_rows=1000)
        assert len(dev._w) == (4 if chunk else 1)
        got = dev(mx.array(idx))
        assert got.shape == (2, 3, 24, 256) and got.dtype == mx.bfloat16
        assert np.array_equal(np.array(got.astype(mx.float32)), want)
    t.close()


def test_tables_fit_in_memory_policy(tmp_path):
    t, rows = tables_in(tmp_path)
    need = sum(t.loc[lid][k][4] for lid in t.spec.layer_ids for k in ("w", "s"))
    assert EM.tables_fit_in_memory(t, 0, wired_bytes=need + (8 << 30))
    assert not EM.tables_fit_in_memory(t, 1, wired_bytes=need + (8 << 30))              # one byte of model too much
    t.close()


def test_staged_embedding_fills_after_the_graph_is_built(tmp_path):
    t, rows = tables_in(tmp_path)
    stage = EM.EngramStage(t)
    emb = EM.StagedRowEmbedding(stage, 1)
    ref = EM.ParallelRowEmbedding(t, 1)
    idx = np.random.default_rng(4).integers(0, rows[1], (3, 1, 24))
    want = np.array(ref(mx.array(idx)).astype(mx.float32))
    out = np.array(emb(mx.array(idx)).astype(mx.float32))                               # not deferred: the synchronous pread path
    assert np.array_equal(out, want)
    stage.poison = True
    with stage.deferred():
        lazy = emb(mx.array(idx))                                                       # slots registered, nothing read
        assert len(stage.pending) == 1
        early = np.array(lazy.astype(mx.float32))                                       # evaluated before the fill: the poison reaches the output
    assert np.isnan(early).all()
    stage.pending = []
    stage.poison = False
    with stage.deferred():
        lazy = emb(mx.array(idx))
    host = np.zeros((3, 1, 2, 24), np.int64)
    host[:, :, 1, :] = idx
    assert stage.fill(host) == idx.size
    assert np.array_equal(np.array(lazy.astype(mx.float32)), want)
    assert emb._slots[idx.size][0] is not emb._slots[idx.size][1]                       # a ring of slot sets: a step in flight is never overwritten
    t.close()


# ----------------------------------------------------------------------------------------------------------------- model level
def engram_pack(tmp_path):
    src = os.path.join(SHRUNK, "ds2")
    if not os.path.isdir(src):
        pytest.skip(f"{src} missing")
    out = str(tmp_path / "pack")
    subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "make_engram_pack.py"), src, out], check=True, capture_output=True)
    return out


@pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set")
def test_pipelined_greedy_equals_plain_greedy(tmp_path):
    pytest.importorskip("mlx_vlm")
    from kyojin_mlx import patch as P, rows as R, spec as S
    out = engram_pack(tmp_path)
    model, spec, cfg = P.build_model(out)
    P.patch_model(model, out, spec, cfg)
    R.enable(model, spec)
    lm = model.language_model
    es = ER.EngramSpec.from_config(cfg)
    tables = ER.EngramTables(out, es)
    ids = np.random.default_rng(0).integers(16, cfg["text_config"]["vocab_size"] - 16, 70).tolist()
    ref = S.plain_greedy(model, ids, 12)                                                # stock loop, memmap embed, a host read per token
    originals = [lm.layers[l].engram.embed for l in es.layer_ids]
    variants = {"par": [EM.ParallelRowEmbedding(tables, p) for p in range(len(es.layer_ids))]}
    stage = EM.install_staged_gather(model, tables)
    variants["staged"] = stage.embeds
    variants["dev"] = EM.install_device_gather(model, tables, chunk_rows=4000)
    for name, mods in variants.items():
        for early in (False, True):
            if name != "staged" and early:
                continue
            for l, m in zip(es.layer_ids, mods):
                lm.layers[l].engram.embed = m
            stage.early = early
            cache, lg = R.prefill(model, ids)
            tok = mx.argmax(lg, axis=-1)
            mx.eval(tok)
            got = [int(tok.item())] + list(EM.pipelined_greedy(model, cache, tok, 11, stage=stage if name == "staged" else None))
            assert got == ref, (name, early)
    # the hash ids built one step ahead are the host RowMap's
    tm = json.load(open(os.path.join(out, "engram_token_map.json")))
    tm = tm.get("token_map", tm) if isinstance(tm, dict) and "token_map" in tm else ([tm[str(i)] for i in range(len(tm))] if isinstance(tm, dict) else tm)
    rm = ER.RowMap(es, tm)
    got_ids = []
    orig = EM.hashes_ahead

    def rec(m, e, y):
        r = orig(m, e, y)
        got_ids.append(np.asarray(r)[0, 0])
        return r
    EM.hashes_ahead = rec
    try:
        for l, m in zip(es.layer_ids, variants["staged"]):
            lm.layers[l].engram.embed = m
        cache, lg = R.prefill(model, ids)
        tok = mx.argmax(lg, axis=-1)
        toks = list(EM.pipelined_greedy(model, cache, tok, 6, stage=stage))
    finally:
        EM.hashes_ahead = orig
    host = rm(np.array(ids + [int(tok.item())] + toks))
    assert np.array_equal(np.stack(got_ids), host[len(ids):len(ids) + len(got_ids)])
    # the opt-in entry point: auto picks the resident table when it fits, the staged path otherwise
    for wired, want in ((None, "device"), (1 << 20, "staged")):
        mode, stg = EM.install_engram_pipeline(model, tables, 0, "auto", wired_bytes=wired)
        assert mode == want and (stg is None) == (want == "device")
        cache, lg = R.prefill(model, ids)
        tok = mx.argmax(lg, axis=-1)
        mx.eval(tok)
        assert [int(tok.item())] + list(EM.pipelined_greedy(model, cache, tok, 11, stage=stg)) == ref, mode
    tables.close()


CBCOUNT = os.path.join(HERE, "..", "tools", "cbcount", "libcbcount.dylib")


@pytest.mark.skipif(not SHRUNK or not os.path.exists(CBCOUNT), reason="KYOJIN_SHRUNK not set or tools/cbcount/libcbcount.dylib not built")
def test_no_command_buffer_is_committed_while_the_graph_is_built(tmp_path):
    pytest.importorskip("mlx_vlm")
    from kyojin_mlx import patch as P, rows as R
    lib = ctypes.CDLL(CBCOUNT)
    lib.cb_install()
    lib.cb_commits.restype = ctypes.c_long
    out = engram_pack(tmp_path)
    model, spec, cfg = P.build_model(out)
    P.patch_model(model, out, spec, cfg)
    R.enable(model, spec)
    lm = model.language_model
    es = ER.EngramSpec.from_config(cfg)
    tables = ER.EngramTables(out, es)
    ids = np.random.default_rng(0).integers(16, cfg["text_config"]["vocab_size"] - 16, 70).tolist()
    stage = EM.install_staged_gather(model, tables)
    stage.early = False
    dev = EM.install_device_gather(model, tables)
    par = [EM.ParallelRowEmbedding(tables, p) for p in range(len(es.layer_ids))]
    for name, mods, stg in (("par", par, None), ("staged", stage.embeds, stage), ("dev", dev, None)):
        for l, m in zip(es.layer_ids, mods):
            lm.layers[l].engram.embed = m
        cache, lg = R.prefill(model, ids)
        tok = mx.argmax(lg, axis=-1)
        mx.eval(tok)
        built = []
        marks = {}

        def probe(tag):
            if tag == "start":
                mx.synchronize()
            marks[tag] = lib.cb_commits()
            if tag == "built":
                built.append(marks["built"] - marks["start"])
        if name == "par":                                                               # the stock embed reads the indices inside the build
            mx.synchronize()
            c0 = lib.cb_commits()
            R.step(model, cache, tok)
            assert lib.cb_commits() - c0 >= 1
            continue
        list(EM.pipelined_greedy(model, cache, tok, 6, stage=stg, probe=probe))
        assert built and all(b == 0 for b in built[1:]), (name, built)
    tables.close()


CGATE = os.path.join(HERE, "..", "tools", "commitgate", "libcommitgate.dylib")


@pytest.mark.skipif(not os.path.exists(CGATE), reason="tools/commitgate/libcommitgate.dylib not built")
def test_commit_gate_holds_a_step_until_release():
    lib = EM._commit_gate()
    x = mx.random.normal((256, 256))
    mx.eval(x)
    c0 = lib.cg_seen()
    mx.eval((x @ x).sum())                                                              # one step: count its command buffers
    time.sleep(0.05)
    k = lib.cg_seen() - c0
    assert k >= 1
    y = (x @ x + 1).sum()
    lib.cg_hold(k)
    mx.async_eval(y)
    got = []
    th = threading.Thread(target=lambda: got.append(float(y.item())))
    th.start()
    th.join(0.5)
    assert th.is_alive() and not got, "the step ran although its commits are held"
    assert lib.cg_release() >= 1
    th.join(10)
    assert got and not th.is_alive()
    assert lib.cg_release() == 0 and lib.cg_seen() - c0 == 2 * k                      # nothing left held, no stray credit: the next step is not held
    z = (x @ x + 2).sum()
    mx.eval(z)


GATE_SCRIPT = """
import sys, numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P, rows as R, spec as S, engram_mlx as EM, engram_rows as ER
out = sys.argv[1]
model, spec, cfg = P.build_model(out)
P.patch_model(model, out, spec, cfg)
R.enable(model, spec)
es = ER.EngramSpec.from_config(cfg)
tables = ER.EngramTables(out, es)
ids = np.random.default_rng(0).integers(16, cfg["text_config"]["vocab_size"] - 16, 70).tolist()
ref = S.plain_greedy(model, ids, 12)
stage = EM.install_staged_gather(model, tables)
stage.early = False
stage.gate = True
for run in range(3):                                                                    # run 0 calibrates the buffers per step; later runs resync and hold
    cache, lg = R.prefill(model, ids)
    tok = mx.argmax(lg, axis=-1)
    mx.eval(tok)
    got = [int(tok.item())] + list(EM.pipelined_greedy(model, cache, tok, 11, stage=stage))
    assert got == ref, (run, got, ref)
assert stage.cg_k is not None, "the gate never armed"
print("GATE OK", stage.cg_k)
"""


@pytest.mark.skipif(not SHRUNK or not os.path.exists(CGATE), reason="KYOJIN_SHRUNK not set or tools/commitgate/libcommitgate.dylib not built")
def test_commit_gate_pipelined_greedy_equals_plain_greedy(tmp_path):
    pytest.importorskip("mlx_vlm")
    out = engram_pack(tmp_path)
    env = dict(os.environ, MLX_MAX_OPS_PER_BUFFER="100000", MLX_MAX_MB_PER_BUFFER="1000000", PYTHONPATH=os.pathsep.join([os.path.join(HERE, ".."), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", GATE_SCRIPT, out], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "GATE OK" in r.stdout, r.stdout[-500:] + r.stderr[-1500:]
