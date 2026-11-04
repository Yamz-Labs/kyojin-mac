"""KVWIN: quantized KV cache + fp16 sink/window overlay. Oracle = per-row attention over dequantized pool keys with the sink and
window rows replaced by the exact fp16 keys. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import kvwin as KW                                       # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
HQ, HK, D = 8, 2, 64
D_BIG = 256                                                              # Qwen3.5 head dim: 5 / 6 bit packs 40 / 48 words per row


@pytest.fixture(autouse=True)
def _hook():
    KW.install()


def data(L, seed=0):
    r = np.random.default_rng(seed)
    f = lambda *s: mx.array((r.standard_normal(s) * (1 + r.random(s) * 2)).astype(np.float16))
    return f(1, HQ, L, D), f(1, HK, L, D), f(1, HK, L, D)


def run(cache, q, k, v, chunks):
    outs, s = [], 0
    for c in chunks:
        kv = cache.update_and_fetch(k[:, :, s:s + c], v[:, :, s:s + c])
        outs.append(KW._base.scaled_dot_product_attention(q[:, :, s:s + c], kv[0], kv[1], cache=cache, scale=D ** -0.5, mask=None))
        s += c
    return mx.concatenate(outs, axis=2)


def oracle(q, k, v, kb, vb, gs, S, W):
    L = k.shape[2]
    dk = mx.dequantize(*mx.quantize(k, group_size=gs, bits=kb), group_size=gs, bits=kb)
    dv = mx.dequantize(*mx.quantize(v, group_size=gs, bits=vb), group_size=gs, bits=vb)
    rows = []
    for p in range(L):
        keep = (mx.arange(L) < S) | (mx.arange(L) > p - W)
        kk = mx.where(keep[None, None, :, None], k, dk)[:, :, :p + 1]
        vv = mx.where(keep[None, None, :, None], v, dv)[:, :, :p + 1]
        rows.append(mx.fast.scaled_dot_product_attention(q[:, :, p:p + 1].astype(mx.float32), kk.astype(mx.float32), vv.astype(mx.float32),
                                                         scale=D ** -0.5, mask=None))
    return mx.concatenate(rows, axis=2)


def err(a, b):
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


@pytest.mark.parametrize("bits,gs", [(4, 64), (8, 64), (6, 32), (5, 32)])
def test_no_overlay_equals_mlx_lm_quantized_cache(bits, gs):
    from mlx_lm.models.cache import QuantizedKVCache
    q, k, v = data(150)
    chunks = [100, 30, 1, 1, 18]
    a = run(KW.WinKVCache(bits, bits, gs, 0, 0, f32=False), q, k, v, chunks)
    up = QuantizedKVCache(group_size=gs, bits=bits)
    outs, s = [], 0
    for c in chunks:
        kv = up.update_and_fetch(k[:, :, s:s + c], v[:, :, s:s + c])
        m = "causal" if c > 1 else None
        outs.append(KW._orig_qsdpa(q[:, :, s:s + c], kv[0], kv[1], scale=D ** -0.5, mask=m, group_size=gs, bits=bits))
        s += c
    b = mx.concatenate(outs, axis=2)
    assert np.array_equal(np.array(a).view(np.uint16), np.array(b).view(np.uint16))


@pytest.mark.parametrize("S,W", [(4, 16), (0, 16), (4, 0), (2, 33)])
@pytest.mark.parametrize("chunks", [[120], [1] * 120, [50, 7, 1, 62], [64, 56]])
def test_matches_oracle_and_is_chunk_independent(S, W, chunks):
    q, k, v = data(120, 1)
    got = run(KW.WinKVCache(4, 4, 64, S, W), q, k, v, chunks)
    ref = oracle(q, k, v, 4, 4, 64, S, W)
    assert err(got, ref) < 4e-2, err(got, ref)


def test_overlay_is_exact_when_it_covers_everything():
    q, k, v = data(60, 2)
    got = run(KW.WinKVCache(4, 4, 64, 4, 64), q, k, v, [60])
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")
    assert err(got, ref) < 2e-2


def test_overlay_reduces_error():
    q, k, v = data(300, 3)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")
    e0 = err(run(KW.WinKVCache(4, 4, 64, 0, 0), q, k, v, [300]), ref)
    e1 = err(run(KW.WinKVCache(4, 4, 64, 4, 64), q, k, v, [300]), ref)
    assert e1 < e0


def test_trim_within_margin_is_exact_and_deep_trim_is_close():
    q, k, v = data(100, 4)
    base = KW.WinKVCache(4, 4, 64, 4, 16)
    run(base, q, k, v, [95])
    ref = run(base, q[:, :, 95:], k[:, :, 95:], v[:, :, 95:], [5])
    for n, tol in ((5, 0.0), (40, 3e-2)):
        c = KW.WinKVCache(4, 4, 64, 4, 16)
        run(c, q, k, v, [95])
        assert c.trim(n) == n and c.offset == 95 - n
        run(c, q[:, :, 95 - n:95], k[:, :, 95 - n:95], v[:, :, 95 - n:95], [n])
        got = run(c, q[:, :, 95:], k[:, :, 95:], v[:, :, 95:], [5])
        assert err(got, ref) <= tol, (n, err(got, ref))


def test_trim_within_margin_does_not_refill_from_the_pool():
    """A rollback the margin covers keeps fp16 ring rows only: a lazy refill would hold the old pool, so the next pool write would copy it whole."""
    q, k, v = data(100, 6)
    c = KW.WinKVCache(4, 4, 64, 4, 16, 8)
    run(c, q, k, v, [95])
    assert c.rk.shape[-2] == 24 and c.trim(8) == 8 and c.rk.shape[-2] == 16
    assert c.trim(4) == 4 and c.rk.shape[-2] == 16                     # below the window: refilled from the pool


def test_fork_is_independent():
    q, k, v = data(80, 5)
    a = KW.WinKVCache(4, 4, 64, 4, 16)
    run(a, q, k, v, [60])
    b = a.fork()
    ra = run(a, q[:, :, 60:], k[:, :, 60:], v[:, :, 60:], [20])
    run(b, q[:, :, 60:70], k[:, :, 70:80], v[:, :, 70:80], [10])         # b diverges
    rb = run(KW.WinKVCache(4, 4, 64, 4, 16), q, k, v, [60, 20])[:, :, 60:]
    assert a.offset == 80 and b.offset == 70 and err(ra, rb) == 0.0


def test_switch_and_bytes(monkeypatch):
    monkeypatch.delenv(KW.ENV, raising=False)
    assert KW.config() is None
    monkeypatch.setenv(KW.ENV, "0")
    assert KW.config() is None
    monkeypatch.setenv(KW.ENV, "5:5:32:4:128")
    assert KW.config() == dict(kbits=5, vbits=5, group_size=32, sink=4, window=128, f32=False)
    monkeypatch.setenv(KW.ENV, "4:4")
    with pytest.raises(ValueError):
        KW.config()
    assert KW.bytes_per_token(256, 4, 4, 64, 0, 0, 4096) == 2 * (128 + 16)
    assert KW.bytes_per_token(256, 4, 4, 64, 4, 128, 16384) < KW.bytes_per_token(256, 8, 8, 64, 0, 0, 16384)


@pytest.mark.parametrize("bits", [5, 6])
def test_odd_bit_widths_pack_correctly_at_head_dim_256(bits):
    r = np.random.default_rng(7)
    k = mx.array(r.standard_normal((1, 2, 40, D_BIG)).astype(np.float16))
    c = KW.WinKVCache(bits, bits, 32, 0, 0)
    c.update_and_fetch(k, k)
    ref = mx.quantize(k, group_size=32, bits=bits)
    assert all(np.array_equal(np.array(a[..., :40, :]), np.array(b)) for a, b in zip(c.kq, ref))


def test_default_is_off_and_model_path(monkeypatch):
    """KYOJIN_KVWIN unset -> plain mlx-lm caches. With KYOJIN_KVWIN_MODEL=<Qwen3.5 mlx-lm dir>: switch on -> WinKVCache on the full
    attention layers only, greedy tokens of a 2-user batched decode == solo decode, and a forked prefix cache continues identically."""
    import os
    from mlx_lm.models.cache import KVCache
    from kyojin_mlx import multiuser as MU
    monkeypatch.delenv(KW.ENV, raising=False)
    mdir = os.environ.get("KYOJIN_KVWIN_MODEL")
    if not mdir:
        pytest.skip("KYOJIN_KVWIN_MODEL not set")
    from mlx_lm import load
    model, _ = load(mdir)
    MU.install()                                                         # per-user SDPA for RowsKV (the batched decode path)
    assert not any(isinstance(c, KW.WinKVCache) for c in MU.make_prompt_cache(model))
    monkeypatch.setenv(KW.ENV, "4:4:64:4:32")
    ids = [list(range(100, 160)), list(range(300, 345))]
    users = [MU.prefill(model, i)[0] for i in ids]
    assert sum(isinstance(c, KW.WinKVCache) for c in users[0]) == 6 and not any(type(c) is KVCache for c in users[0])
    firsts = [MU.prefill(model, i)[1] for i in ids]
    first = mx.array([int(mx.argmax(f, axis=-1).item()) for f in firsts])
    solo = []
    for u, f in zip([MU.fork_cache(u) for u in users], first):
        solo.append(MU.decode_batched(model, MU.batch_caches([u]), mx.array([int(f)]), 12)[0])
    both = MU.decode_batched(model, MU.batch_caches(users), first, 12)
    assert both == solo
