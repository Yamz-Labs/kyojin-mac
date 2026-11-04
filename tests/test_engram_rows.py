"""DS4.1 Engram on the Mac (kyojin_mlx.engram_rows). Synthetic shards for the readers and checks; mlx-vlm for the cross-checks.
Real data: KYOJIN_ENGRAM_SLICE=<npz from tools/engram_cut.py> (+ the config / tokenizer, see tools/engram_check.py) runs the box comparison;
KYOJIN_SHRUNK=<dir with ds2/> runs the loader test (patch_model on a shrunk pack made to carry an Engram layer)."""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from kyojin_mlx import engram_rows as ER
import make_engram_pack as MK

SPEC = ER.EngramSpec(layer_ids=(1, 3), num_embeddings=(), max_ngram_size=4, vocab_size=1000, n_heads=8, head_dim=256, pad_token_id=2, compressed_vocab_size=300)


def spec_with_rows():
    rows = tuple(sum(p for per in layer for ng in per for p in [ng]) for layer in SPEC.primes())
    return ER.EngramSpec(SPEC.layer_ids, rows, 4, 1000, 8, 256, 2, 300), rows


def token_map(n_tokens=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([np.arange(300), rng.integers(0, 300, n_tokens - 300)])


def make_shards(tmp_path, spec, rows, seed=0, drop_scale_dtype=False, truncate=False, bad_shape=False):
    rng = np.random.default_rng(seed)
    for n, (lid, r) in enumerate(zip(spec.layer_ids, rows)):
        tens = [(f"layers.{lid}.engram.embed.weight", "F8_E4M3", (r + (1 if bad_shape else 0), 256), MK.fp8_bytes(rng, (r + (1 if bad_shape else 0), 256)).tobytes()),
                (f"layers.{lid}.engram.embed.scale", "F8_E4M3" if drop_scale_dtype else "F8_E8M0", (r, 8), rng.integers(118, 126, (r, 8)).astype(np.uint8).tobytes())]
        MK.write_st(str(tmp_path / f"model-{n:05d}.safetensors"), tens)
    if truncate:
        p = str(tmp_path / "model-00001.safetensors")
        with open(p, "r+b") as fh:
            fh.truncate(os.path.getsize(p) - 100)


def test_primes_distinct_and_above_vocab():
    sp, _ = spec_with_rows()
    flat = [p for layer in sp.primes() for per in layer for p in per]
    assert len(flat) == len(set(flat)) == 2 * 3 * 8 and min(flat) >= sp.vocab_size
    assert sp.n_cols == 24


def test_from_config_none_without_engram():
    assert ER.EngramSpec.from_config({"text_config": {"engram_layer_ids": []}}) is None
    assert ER.EngramSpec.from_config({}) is None


def test_dequant_matches_mlx_mxfp8():
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(1)
    w, s = MK.fp8_bytes(rng, (40, 256), 0, 14), rng.integers(100, 140, (40, 8)).astype(np.uint8)
    got = ER.dequant_rows(w, s)
    want = np.array(mx.dequantize(mx.array(w).view(mx.uint32), mx.array(s), group_size=32, bits=8, mode="mxfp8", dtype=mx.float32))
    assert np.array_equal(got, want)


def test_rowmap_matches_mlx_vlm_hash_state():
    mx = pytest.importorskip("mlx.core")
    eng = pytest.importorskip("mlx_vlm.models.deepseek_v41.engram")
    sp, rows = spec_with_rows()
    tm = token_map()
    vcfg = SimpleNamespace(engram_layer_ids=list(sp.layer_ids), engram_max_ngram_size=4, engram_n_heads=8, engram_vocab_size=sp.vocab_size,
                           engram_num_embeddings=list(rows), engram_head_dim=256, engram_pad_token_id=2, engram_compressed_vocab_size=300)
    hasher = eng.NgramHashState(vcfg, eng.EngramLayout.from_config(vcfg), token_map=[int(x) for x in tm])
    ids = np.random.default_rng(3).integers(0, 2000, 57)
    want = np.array(hasher(mx.array(ids[None]), 0, SimpleNamespace(engram=None)))[0]
    rm = ER.RowMap(sp, tm)
    assert np.array_equal(rm(ids), want)
    rm2, parts = ER.RowMap(sp, tm), []                                   # prefill chunks then single decode steps
    for a, b in ((0, 20), (20, 21), (21, 40), (40, 41), (41, 57)):
        parts.append(rm2(ids[a:b], a))
    assert np.array_equal(np.concatenate(parts), want)
    assert want.shape == (57, 2, 24) and all(int(want[:, p].max()) < rows[p] for p in range(2))


def test_rowmap_start_pos_checked():
    sp, _ = spec_with_rows()
    rm = ER.RowMap(sp, token_map())
    rm(np.arange(5))
    with pytest.raises(AssertionError):
        rm(np.arange(3), 4)
    rm.reset()
    assert rm(np.arange(3), 0).shape == (3, 2, 24)


def test_tables_read_rows_and_prefetch(tmp_path):
    sp, rows = spec_with_rows()
    make_shards(tmp_path, sp, rows)
    t = ER.EngramTables(str(tmp_path), sp)
    assert t.validate() == []
    ask = np.array([5, 0, rows[0] - 1, 5, 77])
    w, s = t.read_rows(0, ask)
    h, base = ER.read_header(str(tmp_path / "model-00000.safetensors"))
    raw = np.fromfile(str(tmp_path / "model-00000.safetensors"), dtype=np.uint8)
    wo, so = h["layers.1.engram.embed.weight"]["data_offsets"][0] + base, h["layers.1.engram.embed.scale"]["data_offsets"][0] + base
    for j, r in enumerate(ask):
        assert np.array_equal(w[j], raw[wo + r * 256: wo + (r + 1) * 256]) and np.array_equal(s[j], raw[so + r * 8: so + (r + 1) * 8])
    hashes = np.stack([np.random.default_rng(p).integers(0, rows[p], (30, 24)) for p in range(2)], axis=1)
    assert t.prefetch(hashes).result() == sum(np.unique(hashes[:, p]).size for p in range(2))
    with pytest.raises(AssertionError):
        t.read_rows(0, [rows[0]])
    t.close()


def test_memmap_rows_match_preads(tmp_path):
    sp, rows = spec_with_rows()
    make_shards(tmp_path, sp, rows)
    t = ER.EngramTables(str(tmp_path), sp)
    rng = np.random.default_rng(3)
    for ask in (np.unique(rng.integers(0, rows[0], 400)), rng.integers(0, rows[0], 400)):   # sorted unique (the embeds' case) and any order with repeats
        ref = t.read_rows(0, ask)
        t.BIG = 1
        got = t.read_rows(0, ask)
        t.BIG = ER.EngramTables.BIG
        assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    t.close()


def test_rows_ahead_same_embedding(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from kyojin_mlx import engram_mlx as EM
    sp, rows = spec_with_rows()
    make_shards(tmp_path, sp, rows)
    t = ER.EngramTables(str(tmp_path), sp)
    h = mx.array(np.stack([np.random.default_rng(p).integers(0, rows[p], (1, 80, 24)) for p in range(2)], axis=2))   # [1, L, layers, cols]
    embs = [EM.ParallelRowEmbedding(t, p) for p in range(2)]
    ref = [e(h[:, :, p, :]) for p, e in enumerate(embs)]
    t.ahead = EM.EngramAhead(t)
    t.ahead.start(h)
    assert t.ahead.started == 1 and set(t.ahead.jobs) == {0, 1}
    got = [e(h[:, :, p, :]) for p, e in enumerate(embs)]
    assert t.ahead.jobs == {}
    assert all(mx.array_equal(a, b).item() for a, b in zip(ref, got))
    t.ahead.token_map = token_map()
    t.ahead.warm(list(range(300, 500)), 100, 200)
    t.ahead.start(h)                                                     # the queued warm runs behind the chunk's reads
    [e(h[:, :, p, :]) for p, e in enumerate(embs)]
    t._pre.submit(lambda: None).result()
    assert t.ahead.warmed == 1 and t.ahead._warm is None
    t.ahead.start(h[:, :8])                                              # a decode-sized call reads ahead too
    assert set(t.ahead.jobs) == {0, 1} and t.ahead.started == 3
    t.ahead.START_ROWS = 64                                              # the prefill-only rule
    t.ahead.start(h[:, :8])
    assert t.ahead.jobs == {} and t.ahead.started == 3
    t.close()


@pytest.mark.parametrize("kw,needle", [(dict(drop_scale_dtype=True), "dtype"), (dict(bad_shape=True), "shape"), (dict(truncate=True), "beyond the file size")])
def test_validate_catches_problems(tmp_path, kw, needle):
    sp, rows = spec_with_rows()
    make_shards(tmp_path, sp, rows, **kw)
    bad = ER.EngramTables(str(tmp_path), sp).validate()
    assert bad and any(needle in b for b in bad), bad


def test_validate_missing_table(tmp_path):
    sp, rows = spec_with_rows()
    make_shards(tmp_path, sp, rows)
    os.remove(tmp_path / "model-00001.safetensors")
    assert any("not found" in b for b in ER.EngramTables(str(tmp_path), sp).validate())


def test_merged_index_pack_wins_and_conflicts(tmp_path):
    sp, rows = spec_with_rows()
    make_shards(tmp_path, sp, rows)
    # an official Engram shard also carries q / k / wkv of its layer: the pack's copy wins, the shard contributes the embeds only
    rng = np.random.default_rng(0)
    for n, lid in enumerate(sp.layer_ids):
        p = tmp_path / f"model-{n:05d}.safetensors"
        h, base = ER.read_header(str(p))
        body = open(p, "rb").read()[base:]
        extra = [(f"layers.{lid}.engram.q_weight", "BF16", (4, 8), rng.bytes(64))]
        tens = [(k, v["dtype"], v["shape"], body[v["data_offsets"][0]:v["data_offsets"][1]]) for k, v in h.items()] + extra
        MK.write_st(str(p), tens)
    MK.write_st(str(tmp_path / "layers.1.safetensors"), [("layers.1.engram.q_weight", "BF16", (4, 8), rng.bytes(64)), ("layers.1.ffn.x", "BF16", (2,), rng.bytes(4))])
    w = ER.merged_index(str(tmp_path), sp)
    assert w["layers.1.engram.q_weight"] == "layers.1.safetensors" and w["layers.1.engram.embed.weight"] == "model-00000.safetensors"
    assert "layers.3.engram.q_weight" not in w                                   # only in an Engram shard: dropped, the pack has its own
    path = ER.write_merged_index(str(tmp_path), w)
    assert os.path.exists(path)
    MK.write_st(str(tmp_path / "layers.2.safetensors"), [("layers.1.ffn.x", "BF16", (2,), rng.bytes(4))])
    with pytest.raises(ValueError, match="layers.1.ffn.x"):
        ER.merged_index(str(tmp_path), sp)
    os.remove(tmp_path / "model-00001.safetensors")
    os.remove(tmp_path / "layers.2.safetensors")
    with pytest.raises(ValueError, match="is in no shard"):
        ER.merged_index(str(tmp_path), sp)


def test_link_engram_shards(tmp_path):
    sp, rows = spec_with_rows()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    make_shards(src, sp, rows)
    MK.write_st(str(src / "other.safetensors"), [("layers.0.x", "BF16", (2,), b"\0\0\0\0")])
    names = ER.link_engram_shards(str(dst), str(src))
    assert names == ["model-00000.safetensors", "model-00001.safetensors"] and ER.link_engram_shards(str(dst), str(src)) == names
    assert not (dst / "other.safetensors").exists()


# ----------------------------------------------------------------------------------------------------------------- loader (needs a shrunk pack)
SHRUNK = os.environ.get("KYOJIN_SHRUNK")


@pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set")
def test_patch_pack_with_engram_layer(tmp_path):
    """ds2 + a small Engram layer: patch_model installs OffloadedEngramEmbedding (no table in the parameter tree), the forward reads only the
    hashed rows and equals a forward whose embed is a numpy gather of the same rows."""
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from mlx.utils import tree_flatten
    from kyojin_mlx import patch as P
    src = os.path.join(SHRUNK, "ds2")
    if not os.path.isdir(src):
        pytest.skip(f"{src} missing")
    out = str(tmp_path / "pack")
    subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "make_engram_pack.py"), src, out], check=True, capture_output=True)
    model, spec, cfg = P.build_model(out)
    P.patch_model(model, out, spec, cfg)
    lm = model.language_model
    emb = lm.layers[1].engram.embed
    assert type(emb).__name__ == "OffloadedEngramEmbedding"
    assert not [k for k, _ in tree_flatten(lm.layers[1].engram.parameters()) if "embed" in k]
    ids = mx.array([[5, 17, 300, 2000, 17, 5, 1234, 99]])
    got = lm(ids, cache=lm.make_cache()).logits
    mx.eval(got)
    sp = ER.EngramSpec.from_config(cfg)
    tables = ER.EngramTables(out, sp)
    assert tables.validate() == []

    class NumpyEmbed:
        def __call__(self, idx):
            i = np.array(idx)
            w, s = tables.read_rows(0, i.reshape(-1))
            return mx.array(ER.dequant_rows(w, s).reshape(*i.shape, 256)).astype(mx.bfloat16)
    lm.layers[1].engram.embed = NumpyEmbed()
    want = lm(ids, cache=lm.make_cache()).logits
    mx.eval(want)
    assert np.array_equal(np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32)))
    from kyojin_mlx import engram_mlx as EM
    lm.layers[1].engram.embed = emb                                               # back to the mlx-vlm memmap embed
    again = lm(ids, cache=lm.make_cache()).logits
    mx.eval(again)
    assert np.array_equal(np.array(again.astype(mx.float32)), np.array(want.astype(mx.float32)))
    mods = EM.install_parallel_gather(model, tables)                              # parallel pread gather: same logits as the memmap gather
    par = lm(ids, cache=lm.make_cache()).logits
    mx.eval(par)
    assert mods[0].reads > 0 and np.array_equal(np.array(par.astype(mx.float32)), np.array(want.astype(mx.float32)))
    tables.close()


# ----------------------------------------------------------------------------------------------------------------- real slice from the box
SLICE = os.environ.get("KYOJIN_ENGRAM_SLICE")


@pytest.mark.skipif(not SLICE, reason="KYOJIN_ENGRAM_SLICE not set")
def test_real_slice_against_box():
    d = os.path.dirname(SLICE)
    cmd = [sys.executable, os.path.join(HERE, "..", "tools", "engram_check.py"), SLICE, os.path.join(d, "config.json"), os.path.join(d, "tokenizer.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]


# ----------------------------------------------------------------------------------------------------------------- real-size table, never evaluated
@pytest.mark.skipif(not (SLICE and SHRUNK), reason="needs KYOJIN_ENGRAM_SLICE (its sparse/ dir from tools/engram_check.py) and KYOJIN_SHRUNK")
def test_real_size_table_is_never_evaluated(tmp_path):
    """ds2 + the real 95 GiB layer-1 table (sparse file, real header): load must not touch the table. Fails before any eval when a 384M-row array is in the tree."""
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    import resource
    from mlx.utils import tree_flatten
    from kyojin_mlx import patch as P, engram_mlx as EM
    shard = os.path.join(os.path.dirname(SLICE), "sparse", "model-00047-of-00048.safetensors")
    src = os.path.join(SHRUNK, "ds2")
    if not (os.path.exists(shard) and os.path.isdir(src)):
        pytest.skip("sparse shard or ds2 missing")
    import gc
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()                  # the peak is per process: earlier tests of a full run must not count
    out = str(tmp_path / "pack")
    subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "make_engram_pack.py"), src, out, "--real-shard", shard], check=True, capture_output=True)
    model, spec, cfg = P.build_model(out)
    P.patch_model(model, out, spec, cfg, materialize=False)
    lm = model.language_model
    assert type(lm.layers[1].engram.embed).__name__ == "OffloadedEngramEmbedding"
    assert not [k for k, v in tree_flatten(lm.parameters()) if v.shape and v.shape[0] > 10 ** 7], "a table-sized array is in the parameter tree"
    es = ER.EngramSpec.from_config(cfg)
    tables = ER.EngramTables(out, es)
    assert tables.validate() == []
    ids = mx.array([[5, 17, 300, 2000, 17, 5, 1234, 99]])
    got = lm(ids, cache=lm.make_cache()).logits
    mx.eval(got)
    assert bool(mx.all(mx.isfinite(got.astype(mx.float32))).item())
    EM.install_parallel_gather(model, tables)
    again = lm(ids, cache=lm.make_cache()).logits
    mx.eval(again)
    assert np.array_equal(np.array(got.astype(mx.float32)), np.array(again.astype(mx.float32)))
    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30                  # bytes on macOS
    print(f"peak mlx {mx.get_peak_memory() / 2 ** 30:.2f} GiB, rss {rss_gib:.2f} GiB")
    assert mx.get_peak_memory() < 4 * 2 ** 30 and rss_gib < 6
    tables.close()
