"""N4 Engram from a GGUF (kyojin_mlx.engram_gguf). CPU only.
Synthetic: keys -> spec checks, row hash against a plain loop, B32 table rows, forward against a numpy copy.
DS4.1 (skipped when absent): the UD-Q4_K_XL GGUF (range-read header copies; the table rows of the 4 windows and blk.1 / blk.14 engram_gate + engram_kv
fetched in place by HTTP range) against the SERVED DS4.1 pack /path/to/ds41/model (safetensors) and its engram path (engram_rows.RowMap of
mlx_vlm deepseek_v41 Engram), read only: same token ids -> same row ids -> same rows -> same gate."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from kyojin_mlx import engram_gguf as EG  # noqa: E402
from kyojin_mlx import foreign_gguf as GF  # noqa: E402
from kyojin_mlx import foreign_synth as SY  # noqa: E402

DS41 = os.path.expanduser("/path/to/data")
PACK = "/path/to/ds41/model"
SERVED_ROWS = os.path.expanduser("/path/to/data")
FETCHED = os.path.expanduser("/path/to/data")
WINDOWS = [[671, 4787, 13769, 46012, 54994, 1060, 270, 41638], [3465, 10150, 3913, 3395, 361, 1354, 313, 855],        # 4 x 8 tokens of real text
           [5019, 20297, 65274, 1432, 532, 588, 630, 619], [1124, 223, 939, 24, 14, 270, 54186, 17808]]                # (window 3 continues window 2)


# ------------------------------------------------------------------------------------------------------------------------------- synthetic
def _primes_above(v, n):
    out, c = [], v
    while len(out) < n:
        c += 1
        if EG._is_prime(c):
            out.append(c)
    return out


def _synth(tmp_path, bad_offsets=False, drop=()):
    nl, ng, nh, hd, hc, dim, vocab, cvs = 2, 3, 2, 64, 2, 32, 50, 20
    primes = _primes_above(97, nl * (ng - 1) * nh)
    offs = np.concatenate([[0], np.cumsum(primes)[:-1]]).tolist()
    if bad_offsets:
        offs[3] += 1
    rows = int(sum(primes))
    rng = np.random.default_rng(3)
    v = rng.integers(0, 256, (rows, hd)).astype(np.uint8)
    v[(v & 127) == 127] = 0x38
    sc = rng.integers(118, 128, (rows, hd // 32)).astype(np.uint8)
    blk = np.concatenate([sc[:, :, None], v.reshape(rows, -1, 32)], -1).reshape(rows, -1)
    tmap = (np.arange(vocab) % cvs).tolist()
    mult = (rng.integers(0, 1 << 20, nl * ng) * 2 + 1).tolist()
    gate = rng.standard_normal((hc, dim)).astype(np.float32)
    kvw = (rng.standard_normal((dim * (hc + 1), (ng - 1) * nh * hd)) * 0.05).astype(np.float32)
    a = "synth"
    kv = {"general.architecture": a, f"{a}.ple.layers": [1, 3], f"{a}.ple.ngram_size": ng, f"{a}.ple.heads_per_ngram": nh,
          f"{a}.embedding_length_per_layer_input": hd, f"{a}.ple.layer_multipliers": mult, f"{a}.ple.head_vocab_sizes": primes,
          f"{a}.ple.head_offsets": offs, f"{a}.engram.token_map": tmap, f"{a}.engram.pad_token_id": 2, f"{a}.engram.compressed_vocab_size": cvs,
          f"{a}.engram.vocab_size": 97}
    for key in drop:
        del kv[f"{a}.{key}"]
    p = str(tmp_path / "e.gguf")
    SY.write_gguf(p, kv, [(EG.TABLE, (hd, rows), "F8_E4M3", blk), ("blk.1.engram_gate.weight", (dim, hc), "F32", gate.view(np.uint8)),
                          ("blk.1.engram_kv.weight", kvw.shape[::-1], "F32", kvw.view(np.uint8))])
    return p, dict(v=v, sc=sc, tmap=np.array(tmap), mult=np.array(mult).reshape(nl, ng), primes=np.array(primes).reshape(nl, ng - 1, nh),
                   offs=np.array(offs).reshape(nl, -1), gate=gate, kvw=kvw, hc=hc, dim=dim)


def test_synthetic_spec_rows_and_forward(tmp_path):
    p, d = _synth(tmp_path)
    g = GF.GgufFile(p)
    spec = EG.EngramGgufSpec.from_gguf(g)
    assert spec.table_type == "F8_E4M3_B32" and spec.n_cols == 4 and spec.layer_rows(0) + spec.layer_rows(1) == spec.table_rows
    ids = [5, 7, 2, 44, 13, 5]
    m = EG.GgufRowMap(spec)
    got = np.concatenate([m(ids[:4]), m(ids[4:])])                         # two calls: the history carries the window
    c = d["tmap"][np.array(ids)]
    for pos in range(len(ids)):
        win = [c[pos - j] if pos - j >= 0 else d["tmap"][2] for j in range(3)]
        for li in range(2):
            for n in range(1, 3):
                hsh = 0
                for j in range(n + 1):
                    hsh ^= int(win[j]) * int(d["mult"][li, j])
                for h in range(2):
                    assert got[pos, li, (n - 1) * 2 + h] == hsh % int(d["primes"][li, n - 1, h]) + int(d["offs"][li, (n - 1) * 2 + h])
    r = EG.read_rows(g, spec, got[:, 0])
    want = (GF.F8_E4M3_TAB[d["v"][got[:, 0]]].reshape(len(ids), 4, -1, 32) * np.exp2(d["sc"][got[:, 0]].astype(np.float32) - 127)[..., None]).reshape(r.shape)
    assert np.array_equal(r, want)
    loc = EG.table_loc(g, spec)
    assert loc[1][1] - loc[0][1] == spec.layer_rows(0) * 66 and loc[0][4] == 66
    gate, kvw = EG.layer_weights(g, 1)
    assert np.array_equal(gate, d["gate"]) and np.array_equal(kvw, d["kvw"])
    x = mx.array(np.random.default_rng(4).standard_normal((len(ids), d["hc"], d["dim"])).astype(np.float32))
    score, out = EG.engram_forward(x, r, gate, kvw, eps=1e-6, act_fp8=False)
    xn, e = np.array(x), r.reshape(len(ids), -1)
    kv = e @ kvw.T
    key, val = kv[:, :d["hc"] * d["dim"]].reshape(len(ids), d["hc"], -1), kv[:, d["hc"] * d["dim"]:]
    dot = (xn * gate * key).sum(-1) / np.sqrt((xn ** 2).mean(-1) + 1e-6) / np.sqrt((key ** 2).mean(-1) + 1e-6) * d["dim"] ** -0.5
    sc = np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), 1e-6))
    assert np.allclose(np.array(score), sc, rtol=1e-4, atol=1e-5)
    assert np.allclose(np.array(out), xn + (1 / (1 + np.exp(-sc)))[..., None] * val[:, None], rtol=1e-4, atol=1e-4)


def test_synthetic_bad_offsets_fail_loudly(tmp_path):
    p, _ = _synth(tmp_path, bad_offsets=True)
    with pytest.raises(GF.GgufError, match="head_offsets"):
        EG.EngramGgufSpec.from_gguf(GF.GgufFile(p))


def test_predicate_keys_absent_same_spec(tmp_path):
    """engram.vocab_size / compressed_vocab_size are predicates (DS4.1 writes them, Hy4-flash not): without them the spec is the same, the
    compressed vocab taken from token_map; with a wrong one the file fails loudly."""
    full = EG.EngramGgufSpec.from_gguf(GF.GgufFile(_synth(tmp_path)[0]))
    (tmp_path / "b").mkdir()
    bare = EG.EngramGgufSpec.from_gguf(GF.GgufFile(_synth(tmp_path / "b", drop=("engram.vocab_size", "engram.compressed_vocab_size"))[0]))
    for f in ("layer_ids", "ngram_size", "n_heads", "head_dim", "pad_id", "table_rows", "table_type"):
        assert getattr(full, f) == getattr(bare, f), f
    for f in ("multipliers", "primes", "offsets", "token_map"):
        assert np.array_equal(getattr(full, f), getattr(bare, f)), f
    p, _ = _synth(tmp_path / "b", drop=("engram.compressed_vocab_size",))
    g = GF.GgufFile(p)
    g.kv["synth.engram.compressed_vocab_size"] = 21
    with pytest.raises(GF.GgufError, match="compressed_vocab_size"):
        EG.EngramGgufSpec.from_gguf(g)
    g.kv.pop("synth.engram.compressed_vocab_size")
    g.kv["synth.engram.vocab_size"] = 1 << 20
    with pytest.raises(GF.GgufError, match="primes"):
        EG.EngramGgufSpec.from_gguf(g)


def test_row_hash_uint64_past_int64():
    """Hy4-flash's multipliers (up to 8.5e13) times its compressed ids (up to 120831) pass int64 max: the hash is uint64 (llama.cpp engram_hash,
    M:5211-5216). Rows = a Python-int reference mod 2^64, on products above 2^63."""
    nl, ng, nh, cv = 1, 3, 2, 120832
    mult = np.array([[84779352922315, 16745155128523, 61234567890123]], np.int64)
    primes = np.array(_primes_above(10312007, (ng - 1) * nh), np.int64).reshape(nl, ng - 1, nh)
    offs = np.concatenate([[0], np.cumsum(primes.reshape(-1))[:-1]]).reshape(nl, -1)
    tmap = np.arange(cv, dtype=np.int64)
    spec = EG.EngramGgufSpec((1,), ng, nh, 96, mult, primes, offs, tmap, 7, int(primes.sum()), "F8_E4M3")
    ids = [120831, 120830, 99999, 5, 120000, 7]
    got = EG.GgufRowMap(spec)(ids)
    M = (1 << 64) - 1
    assert max(int(t) * int(mult.max()) for t in ids) > np.iinfo(np.int64).max
    for i in range(len(ids)):
        win = [ids[i - j] if i - j >= 0 else 7 for j in range(ng)]
        rolling = (win[0] * int(mult[0, 0])) & M
        for n in range(1, ng):
            rolling ^= (win[n] * int(mult[0, n])) & M
            for h in range(nh):
                assert got[i, 0, (n - 1) * nh + h] == rolling % int(primes[0, n - 1, h]) + int(offs[0, (n - 1) * nh + h])


# ------------------------------------------------------------------------------------------------------------------------------- DS4.1
def _have_ds41():
    if not (os.path.exists(DS41) and os.path.isdir(PACK) and os.path.exists(SERVED_ROWS) and os.path.exists(FETCHED)):
        return False
    have = {(e[0].split("-")[-3], e[2]) for e in json.load(open(FETCHED))}
    return ("00008", 264) in have and ("00009", 81920) in have


needs_ds41 = pytest.mark.skipif(not _have_ds41(), reason="DS4.1 GGUF header copies + fetched rows, served pack or served engram_rows.py absent")


def _served_rows_module():
    sp = importlib.util.spec_from_file_location("served_engram_rows", SERVED_ROWS)
    m = importlib.util.module_from_spec(sp)
    sys.modules["served_engram_rows"] = m
    sp.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ds():
    ER = _served_rows_module()
    g = GF.GgufFile(DS41)
    spec = EG.EngramGgufSpec.from_gguf(g)
    cfg = json.load(open(os.path.join(PACK, "config.json")))
    smap = ER.RowMap(ER.EngramSpec.from_config(cfg), json.load(open(os.path.join(PACK, "engram_token_map.json"))))
    gmap = EG.GgufRowMap(spec)
    srows, grows = [], []
    for i, w in enumerate(WINDOWS):
        if i != 3:
            smap.reset()
            gmap.reset()
        srows.append(smap(w))
        grows.append(gmap(w))
    return dict(ER=ER, g=g, spec=spec, cfg=cfg, srows=np.stack(srows), grows=np.stack(grows))


@needs_ds41
def test_ds41_indices_equal_served(ds):
    spec, g = ds["spec"], ds["g"]
    assert spec.layer_ids == (1, 14) and spec.table_type == "F8_E4M3_B32" and spec.n_cols == 24 and spec.head_dim == 256
    ss = ds["ER"].EngramSpec.from_config(ds["cfg"])
    assert (spec.layer_rows(0), spec.layer_rows(1)) == ss.num_embeddings                          # 384006168 + 384016682 = 768022850 rows
    assert np.array_equal(spec.token_map, np.array(json.load(open(os.path.join(PACK, "engram_token_map.json")))))
    assert np.array_equal(spec.primes, np.array(ss.primes())) and np.array_equal(spec.multipliers, ss.multipliers())
    base = np.array([spec.layer_base(0), spec.layer_base(1)])[None, None, :, None]
    assert ds["grows"].shape == (4, 8, 2, 24) and np.array_equal(ds["grows"], ds["srows"] + base)


@needs_ds41
def test_ds41_rows_equal_served(ds):
    ER, g, spec = ds["ER"], ds["g"], ds["spec"]
    for li, shard in enumerate(("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")):
        f = os.path.join(PACK, shard)
        h, base = ER.read_header(f)
        lid = spec.layer_ids[li]
        r = ds["srows"][:, :, li].reshape(-1)
        wv = h[f"layers.{lid}.engram.embed.weight"]
        ws = h[f"layers.{lid}.engram.embed.scale"]
        with open(f, "rb") as fh:
            v = np.stack([np.frombuffer(os.pread(fh.fileno(), 256, base + wv["data_offsets"][0] + int(i) * 256), np.uint8) for i in r])
            s = np.stack([np.frombuffer(os.pread(fh.fileno(), 8, base + ws["data_offsets"][0] + int(i) * 8), np.uint8) for i in r])
        served = ER.dequant_rows(v, s)                                                             # the served table: e4m3 x e8m0 per 32, float32
        got = EG.read_rows(g, spec, ds["grows"][:, :, li].reshape(-1))
        assert (np.abs(served).sum(1) > 0).all()                                                   # real rows, not the sparse copy's zeros
        assert np.array_equal(got, served)                                                         # bit-equal float32


def _served_layer(ER, lid):
    """Served Engram weights of layer `lid` (layers.<lid>.safetensors, raw bytes by header): q_weight, k_weight bf16 -> f32; wkv F8_E4M3 x F8_E8M0
    per 32 x 32 block -> f32."""
    f = os.path.join(PACK, f"layers.{lid}.safetensors")
    h, base = ER.read_header(f)

    def raw(name, dt):
        v = h[f"layers.{lid}.engram.{name}"]
        return np.fromfile(f, dt, count=int(np.prod(v["shape"])), offset=base + v["data_offsets"][0]).reshape(v["shape"])
    q = mx.array((raw("q_weight", np.uint16).astype(np.uint32) << 16).view(np.float32))
    k = mx.array((raw("k_weight", np.uint16).astype(np.uint32) << 16).view(np.float32))
    w = mx.from_fp8(mx.array(raw("wkv.weight", np.uint8)), dtype=mx.float32)
    s = mx.power(2.0, mx.array(raw("wkv.scale", np.uint8)).astype(mx.float32) - 127.0)
    n, kk = w.shape
    w = (w.reshape(n // 32, 32, kk // 32, 32) * s.reshape(n // 32, 1, kk // 32, 1)).reshape(n, kk)
    return q, k, w


@needs_ds41
@pytest.mark.parametrize("li", [0, 1])
def test_ds41_gate_equal_served(ds, li):
    import mlx_vlm.models.deepseek_v41.engram as VE
    from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
    g, spec = ds["g"], ds["spec"]
    lid = spec.layer_ids[li]
    t = ds["cfg"].get("text_config", ds["cfg"])
    hc, dim, eps = int(t["hc_mult"]), int(t["hidden_size"]), float(t["rms_norm_eps"])
    q, k, sw = _served_layer(ds["ER"], lid)
    gate, kvw = EG.layer_weights(g, lid)
    assert np.array_equal(gate, np.array(q * k))                                                  # engram_gate = q_weight * k_weight, exact in f32
    rel = float(mx.max(mx.abs(mx.array(kvw) - sw)) / mx.max(mx.abs(sw)))
    assert rel < 1 / 254                                                                           # Q8_0 of the fp8 weight: half a step of the 32-block's absmax / 127
    rows = EG.read_rows(g, spec, ds["grows"][:, :, li].reshape(32, 24))                           # the 32 positions of the 4 windows
    x = (mx.random.normal((32, hc, dim), key=mx.random.key(lid)) * 0.5).astype(mx.bfloat16)        # synthetic residual stream (no model run): bf16
    # served reference: mlx_vlm Engram.__call__ body, served weights
    emb = mx.array(rows).astype(x.dtype).reshape(32, -1)
    kv = fake_quant_fp8_ue8m0(emb) @ sw.T.astype(x.dtype)
    key, value = mx.split(kv, [hc * dim], axis=-1)
    s_ref = VE._engram_gate_score(x, key.reshape(32, hc, dim), q, k, mx.array(dim ** -0.5, dtype=mx.float32), eps, 1e-6)
    o_ref = VE._engram_apply_gate(x[None], value[None], mx.sigmoid(s_ref)[None])[0]
    # N4 on the served weights (same math: equal within bf16 rounding of kv), then on the GGUF weights (Q8_0 kv)
    s_n4, o_n4 = EG.engram_forward(x, rows, np.array(q * k), np.array(sw), eps=eps, kv_dtype=x.dtype)
    s_gg, o_gg = EG.engram_forward(x, rows, gate, kvw, eps=eps, kv_dtype=x.dtype)                 # measured: sigmoid gate within 7e-3 (blk.1) / 7e-4 (blk.14), out 7e-3 of max
    gr, gn, gg = (np.array(mx.sigmoid(a)) for a in (s_ref, s_n4, s_gg))
    orf = np.array(o_ref.astype(mx.float32))
    scale = np.abs(orf).max()
    print(f"layer {lid}: out |d|/max n4-served {np.abs(np.array(o_n4.astype(mx.float32)) - orf).max() / scale:.2e}, gguf-served {np.abs(np.array(o_gg.astype(mx.float32)) - orf).max() / scale:.2e}; gate |d| n4-served {np.abs(gn - gr).max():.2e}, gguf-served {np.abs(gg - gr).max():.2e}; kv weight rel {rel:.2e}")
    assert np.abs(gn - gr).max() < 2e-3 and np.abs(np.array(o_n4.astype(mx.float32)) - orf).max() < 2e-2 * scale
    assert np.abs(gg - gr).max() < 1e-2 and np.abs(np.array(o_gg.astype(mx.float32)) - orf).max() < 2e-2 * scale
