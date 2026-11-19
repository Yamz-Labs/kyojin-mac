"""Mapped row table (flag KYOJIN_MAPPED_ROW_TABLE): qwen4exp's per-layer n-gram table stays in the mapped file, only the rows of a step are decoded. CPU, synthetic."""
import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_gguf as GF  # noqa: E402
from kyojin_mlx import foreign_linear as FL  # noqa: E402
from kyojin_mlx import foreign_names as FN  # noqa: E402
from kyojin_mlx import foreign_qwen4exp_gguf as Q  # noqa: E402
from kyojin_mlx import foreign_synth as SY  # noqa: E402
from kyojin_mlx.foreign_mapped_rows import MappedRows  # noqa: E402

ROWS, K = 64, 160                                   # the real table: 160-wide IQ4_NL rows (5 blocks of 32, 90 bytes a row)


@pytest.fixture(autouse=True)
def _rows(monkeypatch):
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")
    Q.register()
    yield
    monkeypatch.undo()
    Q.register()


def _file(tmp_path, tname="IQ4_NL", rows=ROWS, k=K):
    blocks = SY.random_blocks(tname, rows, k, np.random.default_rng(3))
    path = str(tmp_path / "m.gguf")
    SY.write_gguf(path, {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1}, [("per_layer_token_embd.weight", (k, rows), tname, blocks)])
    return path, blocks


def test_rows_equal_dequantised_rows(tmp_path):
    path, blocks = _file(tmp_path)
    g = GF.GgufFile(path)
    raw = g.raw("per_layer_token_embd.weight")
    assert raw.shape == (ROWS, 90) and np.array_equal(raw, blocks)
    full = GF.dequant("IQ4_NL", blocks, K).reshape(ROWS, K).astype(np.float16)
    m = MappedRows(raw, "IQ4_NL", K)
    ids = np.array([[[3, 3, 63, 0], [7, 3, 12, 12]]])                    # (B, T, heads), repeated rows
    out = m(mx.array(ids))
    assert out.shape == (1, 2, 4, K) and out.dtype == mx.float16
    assert np.array_equal(np.asarray(out), full[ids])
    assert m.stats["rows"] == 8 and m.stats["distinct"] == 5 and m.stats["bytes"] == 5 * 90        # only the distinct rows were read
    with pytest.raises(IndexError):
        m(mx.array([ROWS]))
    g.close()


def test_type_without_row_decode_refused():
    with pytest.raises(GF.GgufTypeError):
        MappedRows(np.zeros((2, 4), np.uint8), "NOT_A_TYPE", 32)


def test_switch_off_raises_on_reads_mapped(tmp_path, monkeypatch):
    path, _ = _file(tmp_path)
    monkeypatch.delenv("KYOJIN_MAPPED_ROW_TABLE", raising=False)
    assert "per_layer_token_embd.weight" in FN.ARCHS["qwen4exp"].unserved
    with pytest.raises(GF.GgufError, match="no loader path"):
        FL._read_gguf(path, "qwen4exp")
    monkeypatch.setenv("KYOJIN_MAPPED_ROW_TABLE", "1")
    src = FL._read_gguf(path, "qwen4exp")
    raw, tn, width, suffix = src.mapped["per_layer_token_embd.weight"]
    assert tn == "IQ4_NL" and width == K and raw.shape == (ROWS, 90) and suffix == ".ple.ple_embedding.ngram_embedding"
    assert not src.packs and not src.embed and not src.dense                # nothing was decoded or copied


def test_class_lookup_takes_the_mapped_table(tmp_path):
    """the mlx-vlm class's own n-gram hash feeds the mapped table: same output as a reference gather of the dequantised rows"""
    mod = pytest.importorskip("mlx_vlm.models.qwen4_exp")
    import test_qwen4exp_gguf as T
    kv = T._kv()
    kv.update({"qwen4exp.embedding_length": 256, "qwen4exp.embedding_length_per_layer_input": 32, "qwen4exp.ple.heads_per_ngram": 4})
    m = mod.Model(mod.ModelConfig.from_dict(Q.config(T._G(kv), "qwen4exp")))
    e = m.language_model.model.layers[1].ple.ple_embedding
    sizes = [7, 11, 13, 17, 19, 23, 29, 31]                                 # small heads so a small table covers every id
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
    e.ngram_heads_vocab_sizes, e.ngram_heads_offsets = mx.array(sizes, dtype=mx.int64), mx.array(offs, dtype=mx.int64)
    rows = sum(sizes)
    path, blocks = _file(tmp_path, rows=rows, k=32)
    raw = GF.GgufFile(path).raw("per_layer_token_embd.weight")
    ref = GF.dequant("IQ4_NL", blocks, 32).reshape(rows, 32).astype(np.float16)
    paths = [p for p, _ in m.named_modules() if p.endswith(".ple.ple_embedding.ngram_embedding")]
    assert paths == ["language_model.model.layers.1.ple.ple_embedding.ngram_embedding"]
    seen = []

    def spy(ids):
        seen.append(np.asarray(ids))
        return mx.array(ref[np.asarray(ids)])
    e.ngram_embedding = spy
    toks = mx.array([[5, 9, 2, 9, 40]], dtype=mx.int64)
    want = e(toks, None)
    e.ngram_embedding = MappedRows(raw, "IQ4_NL", 32)
    got = e(toks, None)
    assert got.shape == want.shape == (1, 5, 8 * 32) and np.array_equal(np.asarray(got), np.asarray(want)) and seen[0].shape == (1, 5, 8)


def _day0(tmp_path):
    import sys
    sys.path.insert(0, "tools")
    import day0_inspect as D
    kv = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 1}
    t = [("token_embd.weight", (32, 8), "F32", np.zeros((8, 32), np.float32)), ("per_layer_token_embd.weight", (16, 8), "F32", np.zeros((8, 16), np.float32))]
    p = tmp_path / "q.gguf"
    SY.write_gguf(p, kv, t)
    return D, D.inspect(str(p), use_census=False)


def test_day0_off_refuses_on_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.delenv("KYOJIN_MAPPED_ROW_TABLE", raising=False)
    D, r = _day0(tmp_path)
    assert r["verdict"] == "MODEL CODE NEEDED" and "Arch.unserved" in r["why"] and r["mapped_tables"] == []
    monkeypatch.setenv("KYOJIN_MAPPED_ROW_TABLE", "1")
    D, r = _day0(tmp_path)
    assert r["verdict"] == "UNVERIFIED" and r["verdict"] != "READY" and r["mapped_tables"] == ["per_layer_token_embd.weight"] and "no Metal proof" in r["why"]
    assert D.serve_blocker("qwen4exp", ["per_layer_token_embd.weight"]) is None or "no model class" in D.serve_blocker("qwen4exp", ["per_layer_token_embd.weight"])


def test_serve_blocker_makes_the_loaders_class_call(monkeypatch):
    import sys
    sys.path.insert(0, "tools")
    import day0_inspect as D
    from kyojin_mlx import foreign_vlm_classes as VC
    seen = []
    monkeypatch.setattr(VC, "resolve", lambda cands, config=None, families=True: seen.append((list(cands), dict(config), families)) or (None, None, "", "family"))
    A = FN.ARCHS["qwen4exp"]
    assert D.serve_blocker("qwen4exp", []) is None
    assert seen == [([A.model_type], dict(model_type=A.model_type, architectures=list(A.architectures)), True)]       # gguf_config's two keys, as load_foreign passes them


def test_row_batch_equals_per_layer(tmp_path):
    """KYOJIN_MAPPED_ROW_BATCH: all layers' rows in one host call, bit-equal to the per-layer path, over a multi-step decode with a cache window."""
    mod = pytest.importorskip("mlx_vlm.models.qwen4_exp")
    import test_qwen4exp_gguf as T
    from kyojin_mlx.foreign_mapped_rows import install_batch
    kv = T._kv()
    kv.update({"qwen4exp.embedding_length": 256, "qwen4exp.embedding_length_per_layer_input": 32, "qwen4exp.ple.heads_per_ngram": 4})
    m = mod.Model(mod.ModelConfig.from_dict(Q.config(T._G(kv), "qwen4exp")))
    import copy
    sizes = [7, 11, 13, 17, 19, 23, 29, 31]
    rows = sum(sizes) * 3
    path, blocks = _file(tmp_path, rows=rows, k=32)
    raw = GF.GgufFile(path).raw("per_layer_token_embd.weight")
    L = m.language_model.model.layers
    L[1].ple.ple_embedding.ngram_embedding = MappedRows(raw, "IQ4_NL", 32)          # the 128-shard class table is replaced before the layer is copied
    L[0].ple, L[2].ple = copy.deepcopy(L[1].ple), copy.deepcopy(L[1].ple)
    es = [(p, e) for p, e in m.named_modules() if p.endswith(".ple.ple_embedding")]
    assert len(es) == 3
    for i, (_p, e) in enumerate(es):
        offs = (np.concatenate([[0], np.cumsum(sizes)[:-1]]) + i * sum(sizes)).tolist()
        e.ngram_heads_vocab_sizes, e.ngram_heads_offsets = mx.array(sizes, dtype=mx.int64), mx.array(offs, dtype=mx.int64)

    class C(list):
        def update_window(self, i, hist, n):
            self[i] = hist[:, -n:]
    steps = [mx.array([[5, 9, 2]], dtype=mx.int64)] + [mx.array([[t]], dtype=mx.int64) for t in (40, 7, 7, 3)]

    def run():
        caches = [C([None] * 4) for _ in es]
        out = []
        for ids in steps:
            out.append([np.asarray(e(ids, c)) for (_p, e), c in zip(es, caches)])
        return out
    want = run()
    b = install_batch(m)
    got = run()
    assert b.stats["prefetch"] == len(steps) and b.stats["layer_calls"] == len(steps) * len(es)
    for w, g in zip(want, got):
        for a, c in zip(w, g):
            assert np.array_equal(a, c)


def test_row_prefetch_is_value_neutral(tmp_path, monkeypatch):
    """KYOJIN_MAPPED_ROW_PREFETCH: the background pre-touch only reads; the rows come back bit-equal with and without it."""
    from kyojin_mlx import foreign_mapped_rows as MR
    raw = np.random.default_rng(5).integers(0, 255, (ROWS, 90), dtype=np.uint8)
    ids = mx.array(np.array([[3, 7, 7, 60]]))
    monkeypatch.delenv("KYOJIN_MAPPED_ROW_PREFETCH", raising=False)
    a = np.asarray(MR.MappedRows(raw, "IQ4_NL", K)(ids))
    monkeypatch.setenv("KYOJIN_MAPPED_ROW_PREFETCH", "1")
    t = MR.MappedRows(raw, "IQ4_NL", K)
    t._pf.thr.join()
    assert np.array_equal(a, np.asarray(t(ids))) and t.prefetch_stats["bytes"] == raw.size
    assert MR.pretouch(raw, 0.0)["bytes"] == 0
